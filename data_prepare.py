"""下载并准备「中英混合」字符级预训练语料，输出 JSONL 供 pretrain.py 使用。

数据来源（走 HuggingFace 镜像 hf-mirror.com，需先 export HF_ENDPOINT）：
- 英文：Salesforce/wikitext（默认 wikitext-103-raw-v1）；
- 中文：wikimedia/wikipedia（默认 20231101.zh）。

下载策略（绕开 hf-mirror 对 wikitext 的 paths-info 503）：
- 英文 wikitext 分片较小（~157MB），整片下载后本地读取；
- 中文 wikipedia 分片很大（~587MB），用 HTTP Range 只读所需行组。

按「字符预算」截取（--en-train-chars / --zh-train-chars / ...），使「1 个 epoch」的训练时长可控。
两种语言的文本都切成 ~512 字符的小段再交错，保证流式词表采样与训练都能覆盖双语。

输出（每行一个 {"text": "..."}）：
- <out>/train.jsonl  中英交错，供两阶段训练；
- <out>/eval.jsonl   留出集（英文 validation/test + 中文留出切片）；
- <out>/meta.json    统计信息。
"""
import argparse
import json
import os
import shutil
import tempfile

import pyarrow.parquet as pq
import requests

HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
CACHE_DIR = os.environ.get("JEPA_CACHE", "/root/autodl-tmp/cache")


def _parquet_urls(namespace, name, config, split):
    """查询 parquet 分片清单，并把域名改写为镜像。"""
    url = f"{HF_MIRROR}/api/datasets/{namespace}/{name}/parquet/{config}/{split}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return [u.replace("https://huggingface.co", HF_MIRROR) for u in r.json()]


# ------------------------- 英文：整片下载 -------------------------


def _cache_key(url):
    """由 parquet URL 生成稳定缓存名：config_split_shard.parquet。"""
    parts = url.split("/parquet/")
    return parts[1].replace("/", "_") if len(parts) == 2 else url.split("/")[-1]


def _download(url, dest):
    """下载 parquet 分片（跟随镜像 302 重定向），先查/写持久缓存再复制。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, _cache_key(url))
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        shutil.copyfile(cached, dest)
        print(f"      缓存命中 {_cache_key(url)}", flush=True)
        return
    with requests.get(url, stream=True, timeout=(60, 1800)) as r:
        r.raise_for_status()
        with open(cached, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    shutil.copyfile(cached, dest)


def load_wikitext(config, split, char_budget):
    """英文 wikitext：整片下载后本地读取，累计到 char_budget 字符即停。"""
    urls = _parquet_urls("Salesforce", "wikitext", config, split)
    print(f"    wikitext {config}/{split}: {len(urls)} 个 parquet 分片")
    texts = []
    total = 0
    tmp = tempfile.mkdtemp(prefix="jepa_en_")
    for i, u in enumerate(urls):
        dest = os.path.join(tmp, f"{i}.parquet")
        print(f"      下载分片 {i} ({u.split('/')[-1]}) ...", flush=True)
        _download(u, dest)
        table = pq.read_table(dest, columns=["text"])
        os.remove(dest)
        for x in table["text"].to_pylist():
            if isinstance(x, str) and x.strip():
                texts.append(x)
                total += len(x)
                if total >= char_budget:
                    return texts
    return texts


# ------------------------- 中文：整片下载（并行缓存） -------------------------


def load_wikipedia_split(cfg, train_chars, eval_chars):
    """中文维基百科：逐分片下载（命中缓存）后本地逐行组读取，累计到预算即停。"""
    urls = _parquet_urls("wikimedia", "wikipedia", cfg, "train")
    print(f"    wikipedia {cfg}/train: {len(urls)} 个 parquet 分片")
    tmp = tempfile.mkdtemp(prefix="jepa_zh_")
    dest = os.path.join(tmp, "zh.parquet")
    train, eval_ = [], []
    t_total = e_total = 0
    for u in urls:
        print(f"      下载分片 {u.split('/')[-1]} ...", flush=True)
        _download(u, dest)
        pf = pq.ParquetFile(dest)
        for i in range(pf.num_row_groups):
            col = pf.read_row_group(i, columns=["text"])["text"].to_pylist()
            for x in col:
                if not (isinstance(x, str) and x.strip()):
                    continue
                if t_total < train_chars:
                    train.append(x)
                    t_total += len(x)
                elif e_total < eval_chars:
                    eval_.append(x)
                    e_total += len(x)
                else:
                    return train, eval_
        os.remove(dest)
    return train, eval_


# ------------------------- 切分 / 交错 / 写出 -------------------------


def _chunk(text, size=512):
    """把长文本切成 ~size 字符的小段，用于中英细粒度交错。"""
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def chunk_all(texts, size=512):
    out = []
    for t in texts:
        out.extend(c for c in _chunk(t, size) if c.strip())
    return out


def write_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def interleave(en, zh, block=50):
    """按块交错中英文本（切段后粒度已较细），保证两阶段与词表都覆盖双语。"""
    out = []
    i = j = 0
    while i < len(en) or j < len(zh):
        out.extend(en[i : i + block])
        i += block
        out.extend(zh[j : j + block])
        j += block
    return out


def _chars(texts):
    return sum(len(t) for t in texts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="/root/autodl-tmp/data")
    p.add_argument("--en-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--zh-config", type=str, default="20231101.zh")
    p.add_argument("--en-train-chars", type=int, default=50_000_000)
    p.add_argument("--zh-train-chars", type=int, default=50_000_000)
    p.add_argument("--en-eval-chars", type=int, default=2_000_000)
    p.add_argument("--zh-eval-chars", type=int, default=2_000_000)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--block", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"英文训练集 (Salesforce/wikitext {args.en_config}, train, {args.en_train_chars} 字符) ...")
    en_train = load_wikitext(args.en_config, "train", args.en_train_chars)
    print(f"  英文训练 {len(en_train)} 条 / {_chars(en_train)} 字符")

    print(f"中文训练+留出 (wikimedia/wikipedia {args.zh_config}, {args.zh_train_chars}+{args.zh_eval_chars} 字符) ...")
    zh_train, zh_eval = load_wikipedia_split(args.zh_config, args.zh_train_chars, args.zh_eval_chars)
    print(f"  中文训练 {len(zh_train)} 条 / {_chars(zh_train)} 字符 | 留出 {len(zh_eval)} 条 / {_chars(zh_eval)} 字符")

    print("英文留出集 (wikitext validation + test) ...")
    en_eval = load_wikitext(args.en_config, "validation", args.en_eval_chars)
    if _chars(en_eval) < args.en_eval_chars:
        en_eval += load_wikitext(args.en_config, "test", args.en_eval_chars - _chars(en_eval))
    print(f"  英文留出 {len(en_eval)} 条 / {_chars(en_eval)} 字符")

    print(f"切成 ~{args.chunk} 字符小段并交错 ...")
    en_train = chunk_all(en_train, args.chunk)
    zh_train = chunk_all(zh_train, args.chunk)
    en_eval = chunk_all(en_eval, args.chunk)
    zh_eval = chunk_all(zh_eval, args.chunk)

    print("写出 train.jsonl / eval.jsonl ...")
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), interleave(en_train, zh_train, args.block))
    write_jsonl(os.path.join(args.out_dir, "eval.jsonl"), interleave(en_eval, zh_eval, args.block))

    meta = {
        "en_train_chars": _chars(en_train),
        "zh_train_chars": _chars(zh_train),
        "en_eval_chars": _chars(en_eval),
        "zh_eval_chars": _chars(zh_eval),
        "train_examples": len(en_train) + len(zh_train),
        "eval_examples": len(en_eval) + len(zh_eval),
        "en_source": f"Salesforce/wikitext {args.en_config}",
        "zh_source": f"wikimedia/wikipedia {args.zh_config}",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"完成：train={meta['train_examples']} 段 / {_chars(en_train) + _chars(zh_train)} 字符, "
          f"eval={meta['eval_examples']} 段 -> {args.out_dir}")


if __name__ == "__main__":
    main()
