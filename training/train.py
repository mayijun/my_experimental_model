"""训练入口：数据源装配 + 训练循环 + 可选 checkpoint 落盘。

用法（CPU 冒烟）：
    .venv/bin/python -m training.train --preset tiny --steps 150
用法（缩放）：
    .venv/bin/python -m training.train --preset base --dim 512 --depth 12 --steps 2000
"""

import argparse
import dataclasses
import os
import random

import numpy as np
import torch

from data_prepare.hf_datasets import (
    ensure_special_tokens,
    load_hf_tokenizer,
    real_batch_fn,
    real_batch_iterator,
)
from data_prepare.synthetic import build_dataset, synthetic_batch_fn
from evaluation.metrics import evaluate, evaluate_real
from model import PRESETS, MultimodalJEPA, build_config
from model.representation import update_target
from training.losses import forward_losses, maybe_autocast


def train(model, batch_fn, pad_id, mask_id, device, steps, lr):
    """通用训练循环：从 batch_fn 拉 (img, txt)，跑「编码→对齐/推理→解码」。"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        img, txt = batch_fn()
        with maybe_autocast(device):
            loss, parts = forward_losses(model, img, txt, pad_id, mask_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_target(model, model.cfg.ema_decay)

        if step == 1 or step % 25 == 0 or step == steps:
            detail = " ".join(f"{k}={v.item():.3f}" for k, v in parts.items())
            print(f"step {step:4d} | loss {loss.item():.4f} | {detail}")
    return model


def save_checkpoint(path, model, cfg, vocab_size, itos=None):
    """落盘模型权重与重建模型/词表所需的最小元信息（供 inference 加载）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": dataclasses.asdict(cfg),
            "vocab_size": vocab_size,
            "vocab_itos": itos,
        },
        path,
    )
    print(f"checkpoint 已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="small")
    # 模型规模（覆盖 preset 默认值）
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    # 训练
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dataset-size", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="")  # 例如 runs/mjepa.pt
    # 数据源：synthetic（内置合成图文对，离线） / real（HF datasets 真实图文对）
    parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--dataset-name", type=str, default="lambdalabs/pokemon-blip-captions")
    parser.add_argument("--dataset-config", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--data-dir", type=str, default="")  # 本地 imagefolder，优先于 dataset-name。
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg = build_config(
        args.preset,
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
    )
    assert cfg.dim % cfg.heads == 0, f"dim ({cfg.dim}) 必须能被 heads ({cfg.heads}) 整除"
    assert cfg.image_size % cfg.patch_size == 0, "image_size 必须能被 patch_size 整除"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | preset: {args.preset} | dim={cfg.dim} depth={cfg.depth} "
          f"heads={cfg.heads} image={cfg.image_size} patch={cfg.patch_size} "
          f"latent_len={cfg.text_len}")

    # —— 数据源 / 词表 ——
    if args.data == "real":
        if args.hf_endpoint:
            os.environ["HF_ENDPOINT"] = args.hf_endpoint
        tokenizer = load_hf_tokenizer(args.tokenizer_name)
        pad_id, mask_id, vocab_size = ensure_special_tokens(tokenizer)
        gen = real_batch_iterator(args, tokenizer, pad_id, cfg.text_len, cfg.image_size)
        batch_fn = real_batch_fn(gen, args.batch_size, device)
        source = args.data_dir or args.dataset_name
        print(f"数据: 真实图文对（{source}） | tokenizer: {args.tokenizer_name} | "
              f"vocab={vocab_size} pad={pad_id} mask={mask_id}")
    else:
        images, caption_ids, captions, tokenizer = build_dataset(
            args.dataset_size, cfg.image_size, cfg.text_len, seed=args.seed
        )
        pad_id, mask_id = tokenizer.pad_id, tokenizer.mask_id
        vocab_size = tokenizer.vocab_size
        batch_fn = synthetic_batch_fn(images, caption_ids, args.batch_size, device)

    model = MultimodalJEPA(cfg, vocab_size).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数: {total/1e6:.2f}M 总 | {trainable/1e6:.2f}M 可训练 | vocab={vocab_size}")

    train(model, batch_fn, pad_id, mask_id, device, args.steps, args.lr)

    if args.save:
        save_checkpoint(
            args.save, model, cfg, vocab_size,
            itos=getattr(tokenizer, "itos", None),
        )

    if args.data == "real":
        evaluate_real(model, gen, tokenizer, pad_id, device, args.eval_samples)
    else:
        evaluate(model, images, caption_ids, captions, tokenizer, device, args.eval_samples)


if __name__ == "__main__":
    main()
