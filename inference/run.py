"""推理：加载 checkpoint，在 Z 内做「文本 → 图像 / 图像 → 文本」的往返。

用法：
    .venv/bin/python -m inference.run --ckpt runs/mjepa.pt --text "redcircleleft"
    .venv/bin/python -m inference.run --ckpt runs/mjepa.pt --image-npy out.npy
"""

import argparse

import numpy as np
import torch

from data_prepare.synthetic import CharTokenizer
from model import MJEPAConfig, MultimodalJEPA


@torch.no_grad()
def encode_image(model, image):
    """图像 [B,C,H,W] → 潜 token Z [B,L,d]。"""
    _, tokens = model.image_online(image)
    return tokens


@torch.no_grad()
def encode_text(model, ids):
    """字符 id [B,L] → 潜 token Z [B,L,d]。"""
    _, tokens = model.text_online(ids)
    return tokens


@torch.no_grad()
def text_to_image(model, ids):
    """文本 id → 共享 Z → 图像 [B,C,H,W]。"""
    return model.image_decoder(encode_text(model, ids))


@torch.no_grad()
def image_to_text(model, image):
    """图像 → 共享 Z → 文本 id [B,L]。"""
    return model.text_decoder(encode_image(model, image)).argmax(-1)


def load_checkpoint(path, device):
    """从 training.train --save 落盘的 checkpoint 恢复 (model, cfg, tokenizer)。"""
    blob = torch.load(path, map_location=device)
    cfg = MJEPAConfig(**blob["cfg"])
    model = MultimodalJEPA(cfg, blob["vocab_size"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg, _restore_tokenizer(blob)


def _restore_tokenizer(blob):
    """按 tokenizer_meta 重建词表；兼容旧 checkpoint 的 vocab_itos 字段。"""
    meta = blob.get("tokenizer_meta")
    if meta and meta.get("type") == "llada":
        from data_prepare.hf_datasets import LLaDATokenizer

        return LLaDATokenizer(meta["model"])
    if meta and meta.get("type") == "char" and meta.get("itos"):
        return CharTokenizer.from_itos(meta["itos"])
    itos = blob.get("vocab_itos")
    return CharTokenizer.from_itos(itos) if itos else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--text", type=str, default="")          # 文本 → 图像
    parser.add_argument("--image-npy", type=str, default="")     # 图像 → 文本
    parser.add_argument("--out", type=str, default="")           # 生成的图像存成 .npy
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tokenizer = load_checkpoint(args.ckpt, device)
    print(f"device: {device} | dim={cfg.dim} depth={cfg.depth} latent_len={cfg.text_len}")

    if args.text:
        if tokenizer is None:
            raise ValueError("该 checkpoint 未保存字符词表，无法把文本编码成 id。")
        ids = torch.tensor([tokenizer.encode(args.text, cfg.text_len)], device=device)
        image = text_to_image(model, ids)[0].cpu().numpy()
        print(f"文本 → 图像: {args.text!r} -> shape {image.shape}, "
              f"range [{image.min():.2f}, {image.max():.2f}]")
        if args.out:
            np.save(args.out, image)
            print(f"已保存: {args.out}")

    if args.image_npy:
        if tokenizer is None:
            raise ValueError("该 checkpoint 未保存字符词表，无法解码出文本。")
        arr = np.load(args.image_npy).astype(np.float32)
        image = torch.from_numpy(arr).unsqueeze(0).to(device)
        ids = image_to_text(model, image)
        print(f"图像 → 文本: {tokenizer.decode(ids[0].tolist())!r}")


if __name__ == "__main__":
    main()
