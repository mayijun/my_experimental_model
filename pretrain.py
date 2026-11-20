"""用 HuggingFace Accelerate + datasets 在公开语料上预训练 Flow Matching + JEPA。

能力：
- datasets 流式加载公开语料（默认 wikitext-103），字符级分词；
- Accelerate 统一管理设备、bf16 混合精度、梯度累积与多卡；
- 每 save-every 步保存 checkpoint，可 --resume 续训；
- 打印吞吐（step/s、token/s），用于估预算。

用法（单卡 A100）：
    python pretrain.py --stage1-steps 2000 --stage2-steps 5000
用法（多卡，先 accelerate config）：
    accelerate launch pretrain.py --stage1-steps 2000 --stage2-steps 5000
本地 JSONL（每行一个 {"text": "..."}）快速冒烟：
    python pretrain.py --dataset-files /tmp/corpus.jsonl --dim 32 --n-heads 4 \
        --encoder-layers 1 --flow-layers 2 --stage1-steps 3 --stage2-steps 3

依赖：pip install accelerate datasets
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset

from flow_matching_jepa import (
    FlowJEPA,
    make_corrupted_context,
    reconstruction_loss,
    variance_loss,
    update_target,
)


# ------------------------- 数据管线 -------------------------


def build_char_tokenizer(examples, sample_size=5000, text_field="text"):
    """从流式语料采样收集唯一字符，构建字符级词表。"""
    chars = set()
    seen = 0
    for example in examples:
        chars.update(example[text_field])
        seen += 1
        if seen >= sample_size:
            break
    chars = sorted(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos


def encode(text, stoi):
    """字符级编码，未知字符跳过，保证每个位置都是合法 ID。"""
    return [stoi[c] for c in text if c in stoi]


class WindowDataset(IterableDataset):
    """把流式文本串接成固定长度 seq_len 的 token 窗口。"""

    def __init__(self, examples, stoi, seq_len, text_field="text"):
        self.examples = examples
        self.stoi = stoi
        self.seq_len = seq_len
        self.text_field = text_field

    def __iter__(self):
        buffer = []
        for example in self.examples:
            buffer.extend(encode(example[self.text_field], self.stoi))
            while len(buffer) >= self.seq_len:
                yield {"input_ids": buffer[: self.seq_len]}
                buffer = buffer[self.seq_len :]


def collate(batch):
    """把 dict 列表拼成 [B, L] 的 token 张量。"""
    return torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)


def next_batch(data_iter, dataloader):
    """无限取批；流式数据耗尽时重建迭代器（这里不会发生，除非语料很小）。"""
    try:
        return next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter)


# ------------------------- 参数分组 -------------------------


def stage1_parameters(model):
    """阶段一训练：在线编码器 + JEPA 预测器 + 解码器 + 可训练量化器。"""
    return (
        list(model.online.parameters())
        + list(model.jepa_predictor.parameters())
        + list(model.decoder.parameters())
        + [p for p in model.quantizer.parameters() if p.requires_grad]
    )


def stage2_parameters(model):
    """阶段二训练：Flow 预测器 + 解码器（编码器冻结）。"""
    return list(model.flow.parameters()) + list(model.decoder.parameters())


def stage1_step(model, clean, mask_id):
    """返回 (总损失, 各项损失)。"""
    context_ids = make_corrupted_context(clean, mask_id)
    context = model.online(context_ids)
    online_clean = model.online(clean)
    with torch.no_grad():
        target = model.target(clean)

    prediction = model.jepa_predictor(context)
    jepa_loss = (1 - F.cosine_similarity(prediction, target, dim=-1)).mean()
    z_q, quant_loss, _ = model.quantize(online_clean)
    recon_loss = reconstruction_loss(model.decoder, z_q, clean)
    z_pred_q, _, _ = model.quantize(prediction)
    masked_recon = reconstruction_loss(model.decoder, z_pred_q, clean)
    var_loss = variance_loss(online_clean)
    loss = jepa_loss + recon_loss + 0.5 * masked_recon + 0.1 * var_loss + quant_loss
    return loss, (jepa_loss, recon_loss, masked_recon, var_loss, quant_loss)


def stage2_step(model, clean, mask_id):
    """返回 (总损失, (flow_loss, recon_loss))。"""
    context_ids = make_corrupted_context(clean, mask_id, min_ratio=0.0)
    with torch.no_grad():
        context = model.online(context_ids)
        target = model.target(clean)
        target_q, _, _ = model.quantize(target)

    noise = torch.randn_like(target)
    t = torch.rand(clean.size(0), device=clean.device)
    z_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target
    velocity_target = target - noise
    velocity = model.flow(z_t, t, context)
    flow_loss = (velocity - velocity_target).pow(2).mean()
    recon_loss = reconstruction_loss(model.decoder, target_q, clean)
    return flow_loss + recon_loss, (flow_loss, recon_loss)


# ------------------------- 主流程 -------------------------


def load_examples(args):
    """返回流式 examples 迭代器，并按需构建词表。"""
    from datasets import load_dataset

    if args.dataset_files:
        examples = load_dataset("json", data_files=args.dataset_files, split="train", streaming=True)
    else:
        examples = load_dataset(args.dataset_name, args.dataset_config, split=args.split, streaming=True)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 数据
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-files", type=str, default="")  # 本地 JSONL 路径，优先于 name/config。
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--vocab-sample", type=int, default=5000)
    # 模型
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--flow-layers", type=int, default=12)
    parser.add_argument("--quantizer", type=str, default="fsq")
    # 训练
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--stage1-steps", type=int, default=2000)
    parser.add_argument("--stage2-steps", type=int, default=5000)
    # checkpoint / 日志
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    mixed_precision = "bf16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(
        mixed_precision=mixed_precision, gradient_accumulation_steps=args.grad_accum
    )

    # —— 词表（续训时从 config.json 恢复，保证词表与模型一致）——
    if args.resume:
        with open(os.path.join(args.checkpoint_dir, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        stoi = {c: i for i, c in enumerate(config["vocab"])}
        itos = {i: c for i, c in enumerate(config["vocab"])}
        vocab_size = len(stoi) + 1
        mask_id = len(stoi)
        args.seq_len = config["seq_len"]
        args.dim = config["dim"]
        args.n_heads = config["n_heads"]
        args.encoder_layers = config["encoder_layers"]
        args.flow_layers = config["flow_layers"]
        args.quantizer = config["quantizer"]
        args.lr = config["lr"]
    else:
        raw = load_examples(args)
        stoi, itos = build_char_tokenizer(raw, args.vocab_sample, args.text_field)
        vocab_size = len(stoi) + 1
        mask_id = len(stoi)
        config = {
            "vocab": sorted(stoi.keys()),
            "seq_len": args.seq_len,
            "dim": args.dim,
            "n_heads": args.n_heads,
            "encoder_layers": args.encoder_layers,
            "flow_layers": args.flow_layers,
            "quantizer": args.quantizer,
            "lr": args.lr,
        }

    # —— 模型与优化器 ——
    assert args.dim % args.n_heads == 0, f"dim ({args.dim}) 必须能被 n_heads ({args.n_heads}) 整除"
    model = FlowJEPA(
        vocab_size=vocab_size,
        max_len=args.seq_len,
        dim=args.dim,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        flow_layers=args.flow_layers,
        quantizer=args.quantizer,
    )
    optimizer1 = torch.optim.AdamW(stage1_parameters(model), lr=args.lr)
    optimizer2 = torch.optim.AdamW(stage2_parameters(model), lr=1e-3)

    model, optimizer1, optimizer2 = accelerator.prepare(model, optimizer1, optimizer2)

    if accelerator.is_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(os.path.join(args.checkpoint_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"参数: {total / 1e6:.1f}M 总 | {trainable / 1e6:.1f}M 可训练 | "
              f"vocab={vocab_size} seq={args.seq_len} dim={args.dim} enc={args.encoder_layers} flow={args.flow_layers}")

    resume_stage, resume_step = 1, 0
    if args.resume:
        accelerator.load_state(args.checkpoint_dir)
        with open(os.path.join(args.checkpoint_dir, "trainer_state.json"), encoding="utf-8") as f:
            state = json.load(f)
        resume_stage, resume_step = state["stage"], state["step"]
        if accelerator.is_main_process:
            print(f"已恢复：stage={resume_stage} step={resume_step}")

    # —— 数据加载器（重新取流式迭代器，避免词表采样时的消耗）——
    raw = load_examples(args)
    dataset = WindowDataset(raw, stoi, args.seq_len, args.text_field)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    dataloader = accelerator.prepare(dataloader)
    data_iter = iter(dataloader)

    tokens_per_step = args.batch_size * args.seq_len

    def run_loop(stage, num_steps, optimizer, step_fn, start_step=0):
        model.train()
        if stage == 2:
            model.quantizer.eval()  # 阶段二冻结码本，避免 EMA 让目标分布漂移。
        t0 = time.time()
        for step in range(1, num_steps + 1):
            if step <= start_step:
                continue
            clean = next_batch(data_iter, dataloader).to(accelerator.device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, parts = step_fn(model, clean, mask_id)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            if stage == 1:
                update_target(model)
            if step % args.log_every == 0 and accelerator.is_main_process:
                dt = time.time() - t0
                sps = args.log_every / dt
                tps = sps * tokens_per_step
                print(f"[stage {stage}] step {step:5d}/{num_steps} | loss {loss.item():.4f} | "
                      f"{sps:.1f} step/s | {tps/1e3:.1f}k token/s")
                t0 = time.time()
            if step % args.save_every == 0 and accelerator.is_main_process:
                accelerator.save_state(args.checkpoint_dir)
                with open(os.path.join(args.checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                    json.dump({"stage": stage, "step": step}, f, ensure_ascii=False)
        if accelerator.is_main_process:
            accelerator.save_state(args.checkpoint_dir)
            with open(os.path.join(args.checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump({"stage": stage, "step": num_steps}, f, ensure_ascii=False)

    # —— 阶段编排（支持断点续训）——
    def freeze_encoder():
        for parameter in model.online.parameters():
            parameter.requires_grad_(False)

    if resume_stage == 1:
        if accelerator.is_main_process:
            print("阶段一：JEPA 表征 + 量化/解码")
        run_loop(1, args.stage1_steps, optimizer1, stage1_step, start_step=resume_step)
        freeze_encoder()
        if accelerator.is_main_process:
            print("阶段二：冻结编码器，训练 Flow Matching")
        run_loop(2, args.stage2_steps, optimizer2, stage2_step, start_step=0)
    elif resume_stage == 2:
        freeze_encoder()
        if accelerator.is_main_process:
            print("阶段二：冻结编码器，训练 Flow Matching")
        run_loop(2, args.stage2_steps, optimizer2, stage2_step, start_step=resume_step)
    else:
        if accelerator.is_main_process:
            print("阶段一：JEPA 表征 + 量化/解码")
        run_loop(1, args.stage1_steps, optimizer1, stage1_step, start_step=0)
        freeze_encoder()
        if accelerator.is_main_process:
            print("阶段二：冻结编码器，训练 Flow Matching")
        run_loop(2, args.stage2_steps, optimizer2, stage2_step, start_step=0)

    if accelerator.is_main_process:
        print("预训练完成。")


if __name__ == "__main__":
    main()
