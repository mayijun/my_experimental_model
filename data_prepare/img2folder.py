"""把「URL + caption」型图文对数据集下载成 HF imagefolder（图片 + metadata.jsonl），
供 `training.train --data-dir` 离线训练。

背景：Recap-DataComp-1B / COYO-700M / WIT 等大规模图文对在 HF 上是「URL + caption」形式，
图片 bytes 并不内嵌，`real_batch_iterator` 直接拿到的只是 URL。本脚本并发下载图片、按质量分
筛选、生成 imagefolder，caption 质量好于 DataComp 原始 alt-text，图片内嵌开箱即用。

默认数据源 COYO-700M（747M 对，含 CLIP/美学评分可筛高质量子集）。中档 3b/7b 建议下
20M–100M 子集（图片约 ~200GB–1TB），用 `--max-samples` 控制规模；纯 CPU 即可跑。

用法：
    export HF_ENDPOINT=https://hf-mirror.com
    uv run python -m data_prepare.img2folder --out data/coyo \
        --max-samples 2000000 --num-proc 64 --score-key image_scores --min-score 0.3
    uv run python -m training.train --preset 3b --data real --data-dir data/coyo \
        --tokenizer llada --backbone hybrid

字段名以各数据集 schema 为准（`datasets.load_dataset(name).features` 可核对）：
    COYO-700M     url / text / image_scores / clip_similarity_vitl14
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _num(v):
    return v if v is not None else 0.0


def collect_items(ds, url_key, text_key, score_key, min_score, max_samples):
    """流式读数据集，筛掉低质量样本，收集 (url, text) 直到 max_samples 个。"""
    items = []
    for ex in ds:
        url = ex.get(url_key)
        text = ex.get(text_key)
        if not url or not text:
            continue
        if score_key:
            score = _num(ex.get(score_key))
            if score < min_score:
                continue
        items.append((url, text))
        if len(items) >= max_samples:
            break
    return items


def download_one(job, out_dir):
    idx, url = job
    try:
        r = requests.get(url, timeout=15, stream=True)
        if r.status_code != 200:
            return None
        data = r.content
        if len(data) < 1024:  # 过小，不是有效图片
            return None
        fname = f"{idx:09d}.jpg"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(data)
        return fname
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=str, default="kakaobrain/coyo-700m")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--out", type=str, default="data/coyo")
    p.add_argument("--max-samples", type=int, default=2_000_000)
    p.add_argument("--num-proc", type=int, default=64)
    p.add_argument("--url-key", type=str, default="url")
    p.add_argument("--text-key", type=str, default="text")
    p.add_argument("--score-key", type=str, default="image_scores", help="质量分字段，空串=不过滤")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=15)
    args = p.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    print(f"读取 {args.source}/{args.split}（流式，收集 {args.max_samples} 条高质量样本）...", flush=True)
    ds = load_dataset(args.source, split=args.split, streaming=True)
    items = collect_items(ds, args.url_key, args.text_key,
                          args.score_key, args.min_score, args.max_samples)
    print(f"已收集 {len(items)} 条，开始并发下载到 {args.out}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    jobs = [(i, url) for i, (url, _) in enumerate(items)]
    ok = 0
    meta = []
    with ThreadPoolExecutor(max_workers=args.num_proc) as pool:
        futures = {pool.submit(download_one, j, args.out): j for j in jobs}
        for fut in as_completed(futures):
            fname = fut.result()
            if fname is None:
                continue
            idx = futures[fut][0]
            meta.append({"file_name": fname, "text": items[idx][1]})
            ok += 1
            if ok % 5000 == 0:
                print(f"  已成功 {ok}/{len(items)}", flush=True)

    meta_path = os.path.join(args.out, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"完成：成功 {ok}/{len(items)}（{ok / len(items) * 100:.1f}%）→ {args.out}")
    print(f"metadata: {meta_path}")
    print(f"\n训练命令：")
    print(f"  uv run python -m training.train --preset 3b --data real --data-dir {args.out} "
          f"--tokenizer llada --backbone hybrid")


if __name__ == "__main__":
    main()
