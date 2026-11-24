"""并行预下载「全量」中英预训练语料的分片到缓存目录，并实测每个分片精确字节数。

背景：hf-mirror 单连接只有 ~50KB/s，但支持 HTTP Range；用 8 段并行下载可到 ~2.2MB/s。
下载到缓存后，prepare.py 的 _download 会命中缓存、跳过重复下载。

鲁棒性：
- 每个分片切成 8 段并行下载，每段先写到 <file>.partN 临时文件，全部成功后再拼装成最终文件；
- 每段带重试（连接中断自动重下），避免 ChunkedEncodingError 导致半成品；
- 因此最终文件「存在且非空」即可信为完整（半成品只会留下 .part 临时文件，不会污染缓存）。

缓存命名与 data_prepare._cache_key 完全一致：
    <config>_<split>_<shard>.parquet  （例如 wikitext-103-raw-v1_train_0.parquet）

用法：
    export HF_ENDPOINT=https://hf-mirror.com
    python -m data_prepare.download
"""
import concurrent.futures
import glob
import os
import shutil
import time

import requests

HF = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
CACHE = os.environ.get("JEPA_CACHE", "/root/autodl-tmp/cache")
CHUNKS = 16  # 每个分片分 16 段并行下载（每段约 ~16MB，规避镜像单连接 ~15MB 断流）
RETRIES = 10  # 每段最多重试次数（断点续传，失败后从已下载字节处继续）

DATASETS = [
    ("Salesforce", "wikitext", "wikitext-103-raw-v1", "train"),
    ("Salesforce", "wikitext", "wikitext-103-raw-v1", "validation"),
    ("Salesforce", "wikitext", "wikitext-103-raw-v1", "test"),
    ("wikimedia", "wikipedia", "20231101.zh", "train"),
]


def _urls(ns, name, cfg, split):
    r = requests.get(f"{HF}/api/datasets/{ns}/{name}/parquet/{cfg}/{split}", timeout=120)
    r.raise_for_status()
    return [u.replace("https://huggingface.co", HF) for u in r.json()]


def _key(u):
    parts = u.split("/parquet/")
    return parts[1].replace("/", "_") if len(parts) == 2 else u.split("/")[-1]


def _size(u):
    r = requests.head(u, allow_redirects=True, timeout=120)
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def _download_range(u, start, end, part):
    expected = end - start + 1
    last_err = None
    for attempt in range(RETRIES):
        written = os.path.getsize(part) if os.path.exists(part) else 0
        cur = start + written
        if cur > end:
            return
        try:
            r = requests.get(
                u,
                headers={"Range": f"bytes={cur}-{end}"},
                stream=True,
                timeout=(60, 600),
            )
            if r.status_code != 206:
                raise RuntimeError(f"非 206 响应: {r.status_code}")
            with open(part, "ab") as f:  # 追加写，断点续传
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    written += len(chunk)
            if written >= expected:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} 下载失败: {last_err}")


def _fetch(u):
    k = _key(u)
    dest = os.path.join(CACHE, k)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return k, os.path.getsize(dest), "缓存命中"

    # 清理可能残留的半成品临时文件
    for stale in glob.glob(dest + ".part*"):
        os.remove(stale)

    total = _size(u)
    seg = (total + CHUNKS - 1) // CHUNKS
    ranges = [(i * seg, min((i + 1) * seg, total) - 1) for i in range(CHUNKS)]
    parts = [f"{dest}.part{i}" for i in range(CHUNKS)]

    with concurrent.futures.ThreadPoolExecutor(CHUNKS) as ex:
        futures = [
            ex.submit(_download_range, u, s, e, p)
            for (s, e), p in zip(ranges, parts)
        ]
        for fut in futures:
            fut.result()  # 任一失败会抛出，届时 .part 残留、dest 未生成

    with open(dest, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    for p in parts:
        os.remove(p)

    return k, os.path.getsize(dest), "下载完成"


def main():
    tasks = []
    for ns, name, cfg, split in DATASETS:
        for u in _urls(ns, name, cfg, split):
            tasks.append(u)
    print(f"共 {len(tasks)} 个分片待下载到 {CACHE}")

    grand = 0
    for u in tasks:
        k, size, status = _fetch(u)
        grand += size
        print(f"  [{status}] {k}: {size / 1e6:.1f} MB", flush=True)
    print(f"=== 全部完成，缓存合计 {grand / 1e9:.2f} GB ===")


if __name__ == "__main__":
    main()
