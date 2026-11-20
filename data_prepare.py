"""下载并准备「中英混合」字符级预训练语料，输出 JSONL 供 pretrain.py 使用。

数据来源（走 HuggingFace 镜像 hf-mirror.com，需先 export HF_ENDPOINT）：
- 英文：Salesforce/wikitext（wikitext-103-raw-v1 或 wikitext-2-raw-v1），自带 train/validation/test 三切分；
- 中文：wikimedia/wikipedia（20231101.zh），无官方切分，这里自行留出不重叠切片。

输出（每行一个 {"text": "..."}）：
- <out>/train.jsonl   中英按块交错，供第一阶段/第二阶段训练；
- <out>/eval.jsonl    留出集（英文 validation+test + 中文留出切片），供评测；
- <out>/meta.json     统计信息。

用法：
    export HF_ENDPOINT=https://hf-mirror.com
    # 完整规模
    python data_prepare.py --out-dir /root/autodl-tmp/data \
        --en-config wikitext-103-raw-v1 --en-train 400000 --zh-train 400000
    # 冒烟（小规模，wikitext-2 秒下）
    python data_prepare.py --out-dir /root/autodl-tmp/data \
        --en-config wikitext-2-raw-v1 --en-train 2000 --zh-train 2000 \
        --en-eval 200 --zh-eval 200
"""
import argparse
import json
import os

from datasets import load_dataset


def _nonempty(examples, text_field="text"):
    for ex in examples:
        t = ex.get(text_field)
        if isinstance(t, str) and t.strip():
            yield t


def _take(examples, limit):
    out = []
    for t in examples:
        out.append(t)
        if len(out) >= limit:
            break
    return out


def load_wikitext(config, split, limit):
    """英文 wikitext（namespaced 镜像），按 split 取前 limit 条非空文本。"""
    ds = load_dataset("Salesforce/wikitext", config, split=split, streaming=True)
    return _take(_nonempty(ds), limit)


def load_wikipedia(cfg, limit):
    """中文维基百科，取前 limit 条非空文章。"""
    ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
    return _take(_nonempty(ds), limit)


def write_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def interleave(en, zh, block=100):
    """按块交错中英文本，保证词表采样时两种语言都覆盖到。"""
    out = []
    i = j = 0
    while i < len(en) or j < len(zh):
        out.extend(en[i : i + block])
        i += block
        out.extend(zh[j : j + block])
        j += block
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="/root/autodl-tmp/data")
    p.add_argument("--en-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--zh-config", type=str, default="20231101.zh")
    p.add_argument("--en-train", type=int, default=400000)
    p.add_argument("--zh-train", type=int, default=400000)
    p.add_argument("--en-eval", type=int, default=60000)
    p.add_argument("--zh-eval", type=int, default=20000)
    p.add_argument("--block", type=int, default=100)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"英文训练集 (Salesforce/wikitext {args.en_config}, train x {args.en_train}) ...")
    en_train = load_wikitext(args.en_config, "train", args.en_train)
    print(f"  英文训练 {len(en_train)} 条")

    print(f"中文训练集 (wikimedia/wikipedia {args.zh_config}, x {args.zh_train}) ...")
    zh_train = load_wikipedia(args.zh_config, args.zh_train)
    print(f"  中文训练 {len(zh_train)} 条")

    print("交错并写出 train.jsonl ...")
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), interleave(en_train, zh_train, args.block))

    print("英文留出集 (wikitext validation + test) ...")
    en_eval = load_wikitext(args.en_config, "validation", args.en_eval)
    en_eval += load_wikitext(args.en_config, "test", max(0, args.en_eval - len(en_eval)))

    print(f"中文留出集 (wikipedia {args.zh_config}, 留出 {args.zh_eval}) ...")
    zh_eval = load_wikipedia(args.zh_config, args.zh_train + args.zh_eval)[args.zh_train :]
    if not zh_eval:  # 兜底：极端小规模下留出切片为空则重取一小段
        zh_eval = load_wikipedia(args.zh_config, args.zh_eval)

    print("写出 eval.jsonl ...")
    write_jsonl(os.path.join(args.out_dir, "eval.jsonl"), interleave(en_eval, zh_eval, args.block))

    meta = {
        "en_train": len(en_train),
        "zh_train": len(zh_train),
        "train_total": len(en_train) + len(zh_train),
        "en_eval": len(en_eval),
        "zh_eval": len(zh_eval),
        "eval_total": len(en_eval) + len(zh_eval),
        "en_source": f"Salesforce/wikitext {args.en_config}",
        "zh_source": f"wikimedia/wikipedia {args.zh_config}",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"完成：train={meta['train_total']} eval={meta['eval_total']} -> {args.out_dir}")


if __name__ == "__main__":
    main()
