"""中档（0.3b / 1b / 3b / 7b）训练与 eval 数据准备：生成 manifest + 可选缓存下载 + 打印训练命令。

默认针对中型档（3b/7b：image_size=512 → text_len=4096，LLaDA tokenizer）。数据清单采用
「高质量图文对」作为跨模态对齐（未经过算子变换的 Z）与双向生成（经过算子变换的 Z）的主数据；
纯文本语料的重建/生成走 data_prepare.prepare，这里只负责图文对。

产物：
- <out>/medium_manifest.json：train/eval 数据集清单（加权混合）+ 形状参数，供
  `training.train --data-config <out>/medium_manifest.json` 消费。
- （--download 时）把各数据集的 parquet 分片并行缓存到 $JEPA_CACHE（复用 download.py 的
  HTTP Range + 断点续传），避免训练时网络成为瓶颈。

用法：
    export HF_ENDPOINT=https://hf-mirror.com
    uv run python -m data_prepare.medium --preset 3b --out data
    uv run python -m data_prepare.medium --preset 7b --download --max-shards 2
"""

import argparse
import json
import os

from model.config import PRESETS, build_config

# 中档默认图文对清单。字段名（image_key / text_key）以各数据集实际 schema 为准；
# 训练前可用 `datasets.load_dataset(name, split="train").features` 核对。caption 为 list 的
# 数据集（flickr30k / coco）由 hf_datasets._first_text 自动取首条。
#
# 关键：这里只选「图片内嵌」的数据集（parquet 里直接存 PIL 图，开箱即用）。Recap-DataComp-1B /
# COYO-700M / WIT 在 HF 上是「URL + caption」形式，图片需用 img2dataset 按 URL 另下载，
# 且全量是几十 TB（大档规模），不适合中档 3b/7b——大档再换它们。
#
#   datacomp_medium   128M 对，图片内嵌（jpg/txt），~1-2TB，中档主训（3b/7b 合适）
#   datacomp_small    12.8M 对，图片内嵌（jpg/txt），~100-200GB，中档起步/验证
#   大档（图片需另下载，几十 TB）：Recap-DataComp-1B（LLaMA-3 重标注）/ DFN / COYO-700M / WIT
MEDIUM_TRAIN = [
    {
        "name": "mlfoundations/datacomp_medium",
        "config": "",
        "split": "train",
        "image_key": "jpg",
        "text_key": "txt",
        "weight": 1.0,
    },
]

# 中档 eval 图文对：标准图文检索留出集，caption 为 list（代码自动取首条）。
MEDIUM_EVAL = [
    {
        "name": "nlphuji/flickr30k",
        "config": "",
        "split": "test",
        "image_key": "image",
        "text_key": "caption",
        "weight": 1.0,
    },
    {
        "name": "HuggingFaceM4/COCO",
        "config": "",
        "split": "validation",
        "image_key": "image",
        "text_key": "sentences_raw",
        "weight": 1.0,
    },
]

# 冒烟/管线自检用的小型种子数据集（字段 image/text 单值，train.py 默认同款）。
QUICK_TRAIN = [
    {"name": "lambdalabs/pokemon-blip-captions", "config": "", "split": "train",
     "image_key": "image", "text_key": "text", "weight": 1.0},
]
QUICK_EVAL = QUICK_TRAIN


def build_manifest(preset, quick=False):
    """按 preset 生成 manifest dict：形状参数（image_size / text_len）+ train/eval 清单。"""
    cfg = build_config(preset)
    return {
        "preset": preset,
        "image_size": cfg.image_size,
        "text_len": cfg.text_len,
        "tokenizer": "llada",
        "llada_model": "GSAI-ML/LLaDA-8B-Instruct",
        "train": QUICK_TRAIN if quick else MEDIUM_TRAIN,
        "eval": QUICK_EVAL if quick else MEDIUM_EVAL,
    }


def cache_dataset(spec, max_shards):
    """把单个数据集 spec 的 parquet 分片并行缓存到 $JEPA_CACHE（复用 download.py 的下载逻辑）。"""
    from data_prepare import download as dl

    ns, name = spec["name"].split("/", 1)
    cfg = spec.get("config") or "default"
    split = spec.get("split", "train")
    urls = dl._urls(ns, name, cfg, split)
    if max_shards:
        urls = urls[:max_shards]
    print(f"  {spec['name']}/{split}: {len(urls)} 个分片", flush=True)
    total = 0
    for u in urls:
        _, size, status = dl._fetch(u)
        total += size
        print(f"    [{status}] {dl._key(u)}: {size / 1e6:.1f} MB", flush=True)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=list(PRESETS), default="3b")
    p.add_argument("--out", type=str, default="data")
    p.add_argument("--quick", action="store_true",
                   help="用小型种子数据集（pokemon）自检管线，替代中档真实清单")
    p.add_argument("--download", action="store_true", help="并行缓存 parquet 分片到 $JEPA_CACHE")
    p.add_argument("--max-shards", type=int, default=0, help="每个数据集最多缓存的分片数（0=全部）")
    args = p.parse_args()

    if args.download:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("JEPA_CACHE", "/root/autodl-tmp/cache")

    manifest = build_manifest(args.preset, quick=args.quick)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "medium_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"manifest 已生成: {out_path}")
    print(f"  preset={args.preset} image_size={manifest['image_size']} "
          f"text_len={manifest['text_len']} tokenizer={manifest['tokenizer']}")
    print(f"  train={len(manifest['train'])} 个数据集 | eval={len(manifest['eval'])} 个数据集")

    if args.download:
        print("缓存下载（train）...")
        for spec in manifest["train"]:
            cache_dataset(spec, args.max_shards)
        print("缓存下载（eval）...")
        for spec in manifest["eval"]:
            cache_dataset(spec, args.max_shards)

    print("\n训练命令（GPU，中档）:")
    print(f"  export HF_ENDPOINT=https://hf-mirror.com")
    print(f"  uv run python -m training.train --preset {args.preset} --data real "
          f"--tokenizer llada --backbone hybrid --data-config {out_path}")
    print("  （--data-config 会按 manifest 加权混合 train 数据集，并用独立 eval 集评测）")
    print("\ncaption 质量优先（下载图片 → 离线 imagefolder）:")
    print("  uv run python -m data_prepare.img2folder --out data/coyo --max-samples 2000000 \\")
    print("      --num-proc 64 --score-key image_scores --min-score 0.3")
    print("  uv run python -m training.train --preset 3b --data real --data-dir data/coyo \\")
    print("      --tokenizer llada --backbone hybrid")


if __name__ == "__main__":
    main()
