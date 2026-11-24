"""真实图文数据（HuggingFace datasets 流式 + HF tokenizer + PIL）。

依赖在函数内部惰性导入，故本模块可以安全地被包级 __init__ 引用而不强制安装 datasets。
"""

import random

import numpy as np
import torch


def load_hf_tokenizer(name):
    """惰性加载 HF tokenizer。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def ensure_special_tokens(tokenizer):
    """确保存在 pad / mask token，返回 (pad_id, mask_id, vocab_size)。

    不同 tokenizer 对特殊 token 的支持不同（如 gpt2 无 pad/mask），这里统一补齐，
    使 JEPA 掩码与文本 padding 对任意 HF tokenizer 都可用。
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    return tokenizer.pad_token_id, tokenizer.mask_token_id, len(tokenizer)


def encode_text(tokenizer, text, length, pad_id):
    """把一段文本 tokenize 成固定 length 的 id 序列（截断 + pad）。"""
    ids = tokenizer.encode(text, add_special_tokens=False)[:length]
    return ids + [pad_id] * (length - len(ids))


def image_transform(pil_img, size, train=True):
    """PIL 图像 → [C,H,W] float32 ∈ [0,1]，统一 resize 到 size（不依赖 torchvision）。"""
    img = pil_img.convert("RGB") if hasattr(pil_img, "convert") else pil_img
    img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    if train and random.random() < 0.5:  # 训练时随机水平翻转做增强。
        arr = arr[:, :, ::-1].copy()
    return torch.from_numpy(arr)


def real_batch_iterator(args, tokenizer, pad_id, text_len, image_size):
    """返回无限重复的 (image_tensor, text_ids) 生成器，来源是 HF datasets 流式。

    支持两类来源：
    - 远程 HF 数据集：--dataset-name / --dataset-config / --split；
    - 本地 imagefolder：--data-dir（含 metadata.jsonl 的图片目录），供离线快速验证。
    """
    from datasets import load_dataset

    if args.data_dir:
        ds = load_dataset("imagefolder", data_dir=args.data_dir, split="train", streaming=True)
    else:
        ds = load_dataset(
            args.dataset_name, args.dataset_config or None, split=args.split, streaming=True
        )

    def gen():
        while True:
            for ex in ds:
                img = ex.get(args.image_key)
                text = ex.get(args.text_key)
                if img is None or not text:
                    continue
                yield image_transform(img, image_size), encode_text(tokenizer, text, text_len, pad_id)

    return gen()


def real_batch_fn(gen, batch_size, device):
    """真实数据：从流式生成器里取一批，拼成 (img, txt) 张量。"""
    def fn():
        imgs, ids = [], []
        for _ in range(batch_size):
            try:
                im, tids = next(gen)
            except StopIteration:
                break
            imgs.append(im)
            ids.append(tids)
        return torch.stack(imgs).to(device), torch.tensor(ids, dtype=torch.long).to(device)

    return fn
