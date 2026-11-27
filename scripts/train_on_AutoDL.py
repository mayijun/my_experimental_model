#!/usr/bin/env python3
"""AutoDL 单卡 3b / L=1024 全流程：环境自举 → ImageNet-1K → 训练 → 评测。

AutoDL 专属行为全部集中在本文件：
- 数据集按「公开数据(/root/autodl-pub) → 共享数据(网盘/数据盘转存目录) → 公网下载」
  三级顺序定位；已解压目录只读使用，只有 tar/zip 时才解压；
- 用 uv 安装与显卡匹配的 torch（A800 → cu126，RTX PRO 6000/Blackwell → cu130）；
- HF 镜像、数据盘缓存、bf16 checkpoint（约 13.7GB）；
- TensorBoard 事件默认写入 `/root/tf-logs`（AutoDL 文档），AutoPanel 直接可视化。
其余全部复用现有代码：model、training.train、data_prepare.hf_datasets、
inference.run、evaluation.metrics。

运行：
    python scripts/train_on_AutoDL.py
    python scripts/train_on_AutoDL.py --steps 20000 --batch-size 2
    python scripts/train_on_AutoDL.py --imagenet-root /root/autodl-pub/ImageNet/ILSVRC2012
    python scripts/train_on_AutoDL.py --shared-root /root/autodl-nas/datasets
    python scripts/train_on_AutoDL.py --download-urls https://.../ILSVRC2012_img_train.tar

数据查找顺序：公开数据 `/root/autodl-pub` → 共享数据（`--shared-root` /
`/root/autodl-nas`、`/root/autodl-fs`、`/root/autodl-tmp` 等转存位置）→ 公网下载
（`--download-urls` 指定的 tar/zip 直链；未指定时走 HuggingFace ILSVRC/imagenet-1k，
该数据集是 gated，需要先接受许可并设置 HF_TOKEN）。

注意：ImageNet-1K 训练集约 1.28M 张、解压后约 150GB。若公开/共享数据已是解压目录，
脚本直接只读使用，不占数据盘；若只有 ILSVRC2012_img_train.tar，需要先把数据盘扩到
≥160GB，否则脚本会明确报错而不会写满磁盘。
"""

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
if str(REPO) not in sys.path:  # 允许 `python scripts/train_on_AutoDL.py` 直接导入仓库内模块
    sys.path.insert(0, str(REPO))
VENV_PY = REPO / ".venv" / "bin" / "python"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
CANDIDATE_NAMES = ("ImageNet", "imagenet", "ImageNet-1K", "imagenet-1k", "imagenet1k",
                   "ILSVRC2012", "ImageNet2012", "ImageNet1K")
CLASS_FILES = ("LOC_synset_mapping.txt", "imagenet_classes.txt", "classnames.txt",
               "classes.txt", "synset_words.txt")
CLASS_URLS = (
    "https://cdn.jsdelivr.net/gh/pytorch/hub@master/imagenet_classes.txt",
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
)
TEMPLATES = [
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a picture of a {}.",
]

# 数据源优先级：AutoDL「公开数据」→「共享数据」→ 公网下载。
PUBLIC_DATA_ROOTS = ("/root/autodl-pub",)   # 控制台「公开数据」，只读挂载
SHARE_DATA_ROOTS = (                        # 「共享数据」转存到网盘/数据盘后的常见落点
    "/root/autodl-nas",
    "/root/autodl-fs",
    "/root/autodl-tmp/shared_data",
    "/root/autodl-tmp/shareData",
    "/root/autodl-tmp/share_data",
)
IMAGENET_HF = "ILSVRC/imagenet-1k"          # 公网兜底：gated 数据集，需要 HF_TOKEN


# ---------------------------------------------------------------- bootstrap
def _in_venv():
    try:
        return Path(sys.executable).resolve() == VENV_PY.resolve()
    except OSError:
        return False


def _venv_ok():
    if not VENV_PY.exists():
        return False
    probe = subprocess.run(
        [str(VENV_PY), "-c", "import torch, datasets, transformers, PIL"],
        capture_output=True,
    )
    return probe.returncode == 0


def _uv_bin():
    uv = shutil.which("uv")
    if uv:
        return uv
    cand = Path(sys.executable).with_name("uv")
    if cand.exists():
        return str(cand)
    print("[bootstrap] 安装 uv ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "uv",
         "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
        check=True,
    )
    return str(Path(sys.executable).with_name("uv"))


def _cuda_extra():
    try:
        cap = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        return "cu130" if int(cap.split(".")[0]) >= 12 else "cu126"
    except Exception:  # noqa: BLE001
        return "cu126"


def bootstrap():
    uv = _uv_bin()
    env = dict(os.environ)
    env.setdefault("UV_DEFAULT_INDEX", "https://pypi.tuna.tsinghua.edu.cn/simple")
    extra = _cuda_extra()
    print(f"[bootstrap] uv sync（torch extra={extra}）...")
    subprocess.run(
        [uv, "sync", "--extra", extra, "--extra", "data", "--extra", "hf", "--extra", "train"],
        cwd=REPO, env=env, check=True,
    )
    print("[bootstrap] 安装 bitsandbytes（失败则训练自动回退 adafactor）...")
    subprocess.run([uv, "pip", "install", "bitsandbytes>=0.43"], cwd=REPO, env=env, check=False)
    subprocess.run([uv, "cache", "clean"], cwd=REPO, env=env, check=False)


def reexec_in_venv():
    os.execv(str(VENV_PY), [str(VENV_PY), str(SCRIPT), *sys.argv[1:]])


# ---------------------------------------------------------------- ImageNet
def _looks_like_class_dir(path):
    try:
        subs = [e.name for e in os.scandir(path) if e.is_dir()]
    except OSError:
        return False
    if len(subs) < 10:
        return False
    wnids = sum(bool(re.fullmatch(r"n\d{8}", s)) for s in subs)
    # ImageNet-100 等子集的类别目录也可能是数字序号（000 ~ 099）
    numeric = sum(bool(re.fullmatch(r"\d+", s)) for s in subs)
    return max(wnids, numeric) >= max(5, len(subs) // 2)


def _layout(path):
    """返回 (train_dir, val_dir, archive)，archive 表示尚未解压的文件。"""
    p = Path(path)
    if not p.is_dir():
        return None, None, None
    if (p / "train").is_dir() and _looks_like_class_dir(p / "train"):
        for name in ("val", "validation"):
            val = p / name
            if val.is_dir() and _looks_like_class_dir(val):
                return p / "train", val, None
        return p / "train", None, None
    if _looks_like_class_dir(p):
        return p, None, None
    for name in ("ILSVRC2012_img_train.tar", "imagenet.tar", "ImageNet.tar"):
        if (p / name).exists():
            return None, None, p / name
    try:
        entries = sorted(p.iterdir())
    except OSError:
        return None, None, None
    for f in entries:
        low = f.name.lower()
        if f.is_file() and ("imagenet" in low or "ilsvrc" in low) and low.endswith(
                (".tar", ".zip", ".tar.gz")):
            return None, None, f
    zips = [f for f in entries if f.suffix.lower() == ".zip" or f.name.endswith(".tar.gz")]
    if zips:
        return None, None, zips[0]
    return None, None, None


def _scan_for_imagenet(base):
    """在单个根目录下寻找可用的 ImageNet-1K（类别目录或 train/val 目录、tar/zip 压缩包）。"""
    base = Path(base)
    if not base.is_dir():
        return None
    try:
        children = list(base.iterdir())
    except OSError:
        return None
    cands = [base] + [base / n for n in CANDIDATE_NAMES]
    for child in children:
        if not child.is_dir():
            continue
        low = child.name.lower()
        if "imagenet" in low or "ilsvrc" in low:
            cands.append(child)
            try:  # 「数据集名/版本」两级目录，如 ImageNet/ILSVRC2012
                cands.extend(c for c in child.iterdir() if c.is_dir())
            except OSError:
                pass
        elif any((child / n).exists() for n in CANDIDATE_NAMES):
            cands.append(child)
            cands.extend(child / n for n in CANDIDATE_NAMES if (child / n).exists())
    seen = set()
    for cand in cands:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            train, _, archive = _layout(cand)
        except OSError:
            continue
        if train or archive:
            return cand
    return None


def _find_imagenet(root, share_roots=()):
    """按 公开数据 → 共享数据 → 本地 的顺序定位 ImageNet-1K。

    返回 (路径, 来源标签)；都找不到时返回 (None, None)，由调用方走公网下载兜底。
    """
    env = os.environ.get("AUTODL_IMAGENET", "")
    if env and Path(env).exists():
        return Path(env), "AUTODL_IMAGENET"
    tiers = (
        ("公开数据", PUBLIC_DATA_ROOTS),
        ("共享数据", tuple(share_roots) + SHARE_DATA_ROOTS),
        ("本地", (str(root), str(Path(root) / "data" / "imagenet"), "/root/autodl-tmp")),
    )
    for label, roots in tiers:
        seen = set()
        for base in roots:
            key = str(Path(base))
            if key in seen:
                continue
            seen.add(key)
            found = _scan_for_imagenet(base)
            if found:
                return found, label
    return None, None


def _find_flickr(root, share_roots=()):
    """寻找 Flickr30k（含 flickr30k_train.parquet 的目录），用于图文对跨模态续训。

    返回目录 Path 或 None。扫描顺序：环境变量 AUTODL_FLICKR → 共享数据 →
    公开数据 /root/autodl-pub → 数据盘 /root/autodl-tmp/autodl-pub → 工作目录。
    """
    env = os.environ.get("AUTODL_FLICKR", "")
    if env and Path(env).exists():
        return Path(env)
    roots = (
        tuple(share_roots) + SHARE_DATA_ROOTS
        + ("/root/autodl-pub", "/root/autodl-tmp/autodl-pub",
           str(root), str(Path(root) / "data"))
    )
    seen = set()
    for base in roots:
        key = str(Path(base))
        if key in seen or not Path(base).is_dir():
            continue
        seen.add(key)
        p = Path(base)
        cands = [p, p / "Flickr30k", p / "flickr30k", p / "Flicker30k"]
        for c in cands:
            if (c / "flickr30k_train.parquet").exists():
                return c
        try:
            for child in p.iterdir():
                if child.is_dir() and (child / "flickr30k_train.parquet").exists():
                    return child
        except OSError:
            pass
    return None


def _find_coco(root, share_roots=()):
    """寻找 COCO Captions（含 train-*.parquet 的目录），566k 条真实图文对。

    扫描顺序：环境变量 AUTODL_COCO → 工作目录 /root/datasets → 共享数据 → 公开数据。
    """
    env = os.environ.get("AUTODL_COCO", "")
    if env and Path(env).exists():
        return Path(env)
    roots = (
        "/root/datasets", str(root), str(Path(root) / "data"),
    ) + tuple(share_roots) + SHARE_DATA_ROOTS + ("/root/autodl-pub", "/root/autodl-tmp/autodl-pub")
    seen = set()
    for base in roots:
        key = str(Path(base))
        if key in seen or not Path(base).is_dir():
            continue
        seen.add(key)
        p = Path(base)
        cands = [p, p / "COCO_captions", p / "coco_captions", p / "COCO"]
        for c in cands:
            if list(c.glob("train-*.parquet")):
                return c
        try:
            for child in p.iterdir():
                if child.is_dir() and list(child.glob("train-*.parquet")):
                    return child
        except OSError:
            pass
    return None


def _need_space(path, need, what="解压"):
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    free = shutil.disk_usage(p).free
    if free < need:
        raise SystemExit(
            f"[ImageNet] {what}需要约 {need / 1e9:.0f}GB，数据盘仅剩 {free / 1e9:.0f}GB。\n"
            "请在 AutoDL 扩容数据盘到 ≥160GB，或改用已解压的公开/共享数据目录：\n"
            "  python scripts/train_on_AutoDL.py --imagenet-root <含 train/n0xxxxxxxx 的目录>\n"
            "公开数据与共享数据中已解压的目录会只读使用，不会复制。"
        )


def _extract_train_tar(tar_path, out_dir):
    """解 ILSVRC2012_img_train.tar：外层 1000 个 <wnid>.tar，内层是该类 JPEG。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / ".inner.tar"
    with tarfile.open(tar_path) as outer:
        members = [m for m in outer if m.isfile() and m.name.endswith(".tar")]
        print(f"[ImageNet] 解压 {tar_path.name}：{len(members)} 个类别 tar → {out}")
        for i, m in enumerate(members, 1):
            wnid = Path(m.name).stem
            target = out / wnid
            if target.is_dir() and any(target.iterdir()):
                continue
            with outer.extractfile(m) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tmp) as inner:
                try:
                    inner.extractall(target, filter="data")
                except TypeError:
                    inner.extractall(target)
            tmp.unlink()
            if i % 50 == 0 or i == len(members):
                print(f"[ImageNet]   {i}/{len(members)} 类")
    return out


def _extract_archive(archive, out_dir):
    archive = Path(archive)
    out = Path(out_dir)
    _need_space(out, int(archive.stat().st_size * 1.1) + 2_000_000_000)
    if archive.name.endswith(".tar"):
        return _extract_train_tar(archive, out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[ImageNet] 解压 {archive.name} → {out}（可能需要较长时间）")
    if archive.name.endswith(".zip"):
        if shutil.which("unzip"):
            subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(out)], check=True)
        else:
            with zipfile.ZipFile(archive) as z:
                z.extractall(out)
    else:
        with tarfile.open(archive) as tf:
            try:
                tf.extractall(out, filter="data")
            except TypeError:
                tf.extractall(out)
    return out


def _find_classes_file(*roots):
    for root in roots:
        if not root:
            continue
        for base in (Path(root), Path(root).parent, REPO / "data"):
            for name in CLASS_FILES:
                p = base / name
                if p.exists():
                    return str(p)
    return None


def _download_classes(dest):
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in CLASS_URLS:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                lines = [ln.decode("utf-8").strip() for ln in r.readlines() if ln.strip()]
            if len(lines) >= 999:
                dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
                print(f"[ImageNet] 已下载类别名: {dest}")
                return str(dest)
        except Exception as e:  # noqa: BLE001
            print(f"[ImageNet] 类别名下载失败 {url}: {e}")
    print("[ImageNet] 无法获取类别名，回退用 wnid 作为文本（语义较弱）")
    return None


def _download_archive(url, dest_dir, retries=4):
    """下载单个压缩包到 dest_dir（支持断点续传，已完整则跳过）。"""
    import urllib.error
    import urllib.request

    name = url.split("?")[0].rstrip("/").split("/")[-1] or "imagenet.tar"
    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        pos = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"} if pos else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                status = getattr(r, "status", 200)
                mode = "ab" if pos and status == 206 else "wb"
                if mode == "wb":
                    pos = 0
                total = pos + int(r.headers.get("Content-Length") or 0)
                print(f"[download] {name}：{pos / 1e9:.2f}GB / 约 {total / 1e9:.2f}GB")
                mark = pos // (1 << 30)
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        pos += len(chunk)
                        if pos // (1 << 30) > mark:
                            mark = pos // (1 << 30)
                            print(f"[download]   {pos / 1e9:.2f}GB", flush=True)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 416:  # Range 越界，说明本地文件已经完整
                return dest
            print(f"[download] {name} 失败（第 {attempt + 1}/{retries} 次）: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[download] {name} 失败（第 {attempt + 1}/{retries} 次）: {e}")
        time.sleep(2 * (attempt + 1))
    raise SystemExit(
        f"[ImageNet] 公网下载失败: {url}\n"
        "ImageNet 官方 tar 需要登录 image-net.org；可改用 --download-urls 指定可直连镜像，\n"
        "或不传 --download-urls，让脚本走 HuggingFace ILSVRC/imagenet-1k（需 HF_TOKEN）。"
    )


def _download_imagenet_hf(dest, limit=0):
    """公网兜底：从 HuggingFace 下载 ILSVRC/imagenet-1k 并导出为 ImageFolder。

    ILSVRC/imagenet-1k 是 gated 数据集：需先在 HF 页面接受许可并设置 HF_TOKEN。
    导出后 train/、val/ 下均为 n0xxxxxxxx 类别目录，可直接被 `_layout` 识别。
    """
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except ImportError as e:  # noqa: BLE001
        raise SystemExit(f"[ImageNet] 缺少 datasets 依赖，无法公网下载: {e}") from e

    print(f"[ImageNet] 开始公网下载 {IMAGENET_HF}（gated，需 HF_TOKEN）...")
    total = 0
    for split, sub in (("train", "train"), ("validation", "val")):
        try:
            ds = load_dataset(IMAGENET_HF, split=split, streaming=True)
            labels = ds.features["label"].names
            ds = ds.cast_column("image", HFImage(decode=False))  # 保留原始 JPEG 字节
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                "[ImageNet] 公网下载失败：ILSVRC/imagenet-1k 是 gated 数据集。\n"
                "  1) 在 https://huggingface.co/datasets/ILSVRC/imagenet-1k 接受许可；\n"
                "  2) export HF_TOKEN=hf_xxx（HF_ENDPOINT 默认已指向 hf-mirror.com）；\n"
                "  3) 或改用 --download-urls 指定可直连的 ImageNet tar/zip 镜像。\n"
                f"原始错误: {e}"
            ) from e
        out = Path(dest) / sub
        for i, ex in enumerate(ds):
            if limit and i >= limit:
                break
            raw, label = ex.get("image"), ex.get("label")
            data = raw.get("bytes") if isinstance(raw, dict) else None
            if not data or label is None:
                continue
            ext = (Path(raw.get("path") or "").suffix or ".jpg") if isinstance(raw, dict) else ".jpg"
            cls_dir = out / labels[label]
            cls_dir.mkdir(parents=True, exist_ok=True)
            (cls_dir / f"{i:08d}{ext}").write_bytes(data)
            total += 1
            if total % 10000 == 0:
                print(f"[ImageNet]   已导出 {total} 张（{split} {i + 1}）", flush=True)
    print(f"[ImageNet] 公网下载完成：{dest}（{total} 张）")


def _download_dataset(root, urls=(), limit=0):
    """公网下载兜底：优先用显式 tar/zip 直链，否则走 HuggingFace ILSVRC/imagenet-1k。"""
    out = Path(root) / "data" / "imagenet"
    out.mkdir(parents=True, exist_ok=True)
    if urls:
        print(f"[ImageNet] 公开/共享数据均未找到，从公网下载 {len(urls)} 个压缩包 ...")
        for url in urls:
            _download_archive(url, out)
        return out
    need = max(2_000_000_000, limit * 400_000) if limit else 160_000_000_000
    _need_space(out, need, what="下载")
    _download_imagenet_hf(out, limit)
    return out


def prepare_imagenet(root, imagenet_root=None, share_roots=(), download_urls=(), download_limit=0):
    """按「公开数据 → 共享数据 → 公网下载」定位 ImageNet-1K。

    返回 (train_dir, val_dir, classes_file)。
    """
    if imagenet_root:
        found, source = Path(imagenet_root), "指定路径"
    else:
        found, source = _find_imagenet(root, share_roots)
    if found is None:
        found, source = _download_dataset(root, download_urls, download_limit), "公网下载"
    print(f"[ImageNet] 数据来源: {source} -> {found}")
    train, val, archive = _layout(found)
    if archive and not train:
        out = _extract_archive(archive, Path(root) / "data" / "imagenet")
        train, val, _ = _layout(out)
        found = out
        if not train:
            raise SystemExit(f"[ImageNet] 解压后仍未找到 train/n0xxxxxxxx 目录: {out}")
    if not train:
        raise SystemExit(f"[ImageNet] {found} 既不是 ImageFolder 结构，也没有可解压的 tar/zip。")

    classes_file = _find_classes_file(found, train)
    if not classes_file:
        classes_file = _download_classes(Path(root) / "data" / "imagenet_classes.txt")
    print(f"[ImageNet] train={train} val={val} classes_file={classes_file}")
    return str(train), (str(val) if val else None), classes_file


def build_manifest(train_dir, val_dir, classes_file, holdout, manifest_path,
                   flickr_root=None, flickr_weight=1.0, imagenet_weight=1.0,
                   coco_root=None, coco_weight=1.0):
    """构建中档 manifest：ImageNet-1K（类名模板）+ Flickr30k + COCO（真实图文对）加权混合。

    Flickr30k / COCO 的 parquet 列：image（HF Image 懒加载）/ caption（Flickr 为 list[str]
    取第 0 条、COCO 为 str）；二者提供真实自然语言描述，是 run#1（仅 ImageNet 类名模板）
    缺失的跨模态对齐信号来源。
    """
    train_specs, eval_specs = [], []
    if train_dir:
        train_specs.append({
            "type": "imagefolder", "name": "autodl-imagenet1k-train",
            "root": train_dir, "classes_file": classes_file,
            "split": "train" if not val_dir else "all",
            "holdout": 0 if val_dir else holdout,
            "templates": TEMPLATES, "weight": float(imagenet_weight),
        })
        eval_specs.append({
            "type": "imagefolder", "name": "autodl-imagenet1k-eval",
            "root": val_dir or train_dir, "classes_file": classes_file,
            "split": "all" if val_dir else "holdout",
            "holdout": 0 if val_dir else holdout,
            "templates": TEMPLATES[:1], "weight": float(imagenet_weight),
        })
    if flickr_root:
        flickr_root = Path(flickr_root)
        train_specs.append({
            "name": "flickr30k-train",
            "data_files": [str(flickr_root / "flickr30k_train.parquet")],
            "image_key": "image", "text_key": "caption", "split": "train",
            "weight": float(flickr_weight),
        })
        eval_specs.append({
            "name": "flickr30k-val",
            "data_files": [str(flickr_root / "flickr30k_val.parquet")],
            "image_key": "image", "text_key": "caption", "weight": float(flickr_weight),
        })
    if coco_root:
        coco_root = Path(coco_root)
        coco_files = sorted(str(p) for p in coco_root.glob("*.parquet"))
        if coco_files:
            train_specs.append({
                "name": "coco-train",
                "data_files": coco_files,
                "image_key": "image", "text_key": "caption", "split": "train",
                "weight": float(coco_weight),
            })
    if not train_specs:
        raise SystemExit("manifest 为空：未找到 ImageNet / Flickr30k / COCO。")
    manifest = {
        "preset": "3b", "image_size": 256, "text_len": 1024,
        "tokenizer": "llada", "llada_model": "GSAI-ML/LLaDA-8B-Instruct",
        "train": train_specs, "eval": eval_specs,
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    coco_added = bool(coco_root and list(Path(coco_root).glob("*.parquet")))
    parts = []
    if train_dir:
        parts.append(f"imagenet(w={imagenet_weight})")
    if flickr_root:
        parts.append(f"flickr30k(w={flickr_weight})")
    if coco_added:
        parts.append(f"coco(w={coco_weight})")
    print(f"[manifest] {manifest_path}：{' + '.join(parts)}（holdout={0 if val_dir else holdout}/类）")
    return manifest_path


# ---------------------------------------------------------------- train/eval
def _default_logdir(root):
    """TensorBoard 事件目录。

    AutoDL 文档（https://www.autodl.com/docs/tensorboard/）：事件写到 `/root/tf-logs`
    后，AutoPanel 内置的 TensorBoard 会直接读取。故在 AutoDL 上默认用它，
    非 AutoDL 环境退回工作目录下的 tensorboard/。
    """
    tf_logs = Path("/root/tf-logs")
    if tf_logs.is_dir() or Path("/root/autodl-tmp").is_dir():
        return tf_logs / "jepa"
    return Path(root) / "tensorboard"


def _bnb_ok():
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes as bnb

        p = torch.nn.Parameter(torch.zeros(8, device="cuda"))
        p.grad = torch.zeros_like(p)
        bnb.optim.AdamW8bit([p]).step()
        return True
    except Exception:  # noqa: BLE001
        return False


def run_train(args, manifest_path, ckpt, logdir):
    from training import train as train_mod

    optim = "adamw8bit" if _bnb_ok() else "adafactor"
    argv = [
        "train_on_AutoDL",
        "--preset", "3b",
        "--image-size", "256", "--patch-size", "8",
        "--data", "real", "--tokenizer", "llada", "--data-config", str(manifest_path),
        "--backbone", "hybrid", "--share-trunk", "1", "--grad-checkpoint", "1",
        "--embed-factor", "128", "--cross-mask", "1", "--psi", "1",
        "--flow", "1", "--flow-adaln", "1", "--flow-steps", "20",
        "--optim", optim, "--batch-size", str(args.batch_size), "--lr", str(args.lr),
        "--steps", str(args.steps), "--flow-stage-steps", str(args.flow_steps),
        "--save", str(ckpt), "--save-dtype", "bf16",
        "--eval-samples", str(args.eval_samples), "--seed", str(args.seed),
        "--logdir", str(logdir),
    ]
    if args.rho:
        # token 层 ρ 压缩（§2.6 / §3.4.2 先写后压）：--rho 开启，课程 S1 冻结 ρ 记忆、S2 解冻。
        # rho_eps_z / rho_rate_weight 未显式指定时不传 → 由 model/config.py 默认值接管（单一事实源）；
        # 显式传入则覆盖 config 默认，保留调参能力。
        argv += [
            "--rho-compress",
            "--rho-head-dim", str(args.rho_head_dim),
            "--rho-beta", str(args.rho_beta),
            "--rho-eps", str(args.rho_eps),
            "--rho-eps-keep", str(args.rho_eps_keep),
            "--rho-b-a", str(args.rho_b_a),
            "--rho-b-c", str(args.rho_b_c),
            "--rho-b-min", str(args.rho_b_min),
            "--rho-b-max", str(args.rho_b_max),
        ]
        if args.rho_eps_z is not None:
            argv += ["--rho-eps-z", str(args.rho_eps_z)]
        if args.rho_rate_weight is not None:
            argv += ["--rho-rate-weight", str(args.rho_rate_weight)]
        if args.rho_eps_keep_modal is not None:
            argv += ["--rho-eps-keep-modal", str(args.rho_eps_keep_modal)]
    if args.resume:
        argv += ["--resume", str(args.resume)]
    if args.save_every:
        argv += ["--save-every", str(args.save_every)]
    if args.grad_accum > 1:
        argv += ["--grad-accum", str(args.grad_accum)]
    print(f"[train] optim={optim} batch={args.batch_size} steps={args.steps}+{args.flow_steps}"
          + (f" | resume={args.resume}" if args.resume else "")
          + (f" | save_every={args.save_every}" if args.save_every else ""))
    old = sys.argv
    sys.argv = argv
    try:
        train_mod.main()
    finally:
        sys.argv = old


def run_eval(args, manifest_path, ckpt):
    import torch

    from data_prepare.hf_datasets import mixture_iterator
    from evaluation.metrics import evaluate_real
    from inference.run import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tokenizer = load_checkpoint(str(ckpt), device)
    if tokenizer is None:
        raise SystemExit("checkpoint 未保存 tokenizer，无法解码。")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    gen = mixture_iterator(
        manifest["eval"], tokenizer, tokenizer.pad_id, cfg.text_len, cfg.image_size,
        seed=args.seed, shuffle_buffer=0,
    )
    print(f"\n[eval] ImageNet-1K holdout | L={cfg.text_len} d={cfg.dim} n={args.eval_samples}")
    evaluate_real(model, gen, tokenizer, tokenizer.pad_id, device, num=args.eval_samples)


# ---------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=str, default="",
                   help="AutoDL 数据盘工作目录（默认 /root/autodl-tmp/jepa_autodl）")
    p.add_argument("--imagenet-root", type=str, default="",
                   help="显式指定 ImageNet-1K 目录，优先于自动搜索（默认按 公开数据 → 共享数据 顺序查找）")
    p.add_argument("--shared-root", action="append", default=[], metavar="DIR",
                   help="共享数据目录（可多次指定；默认扫描 /root/autodl-nas、/root/autodl-fs、"
                        "/root/autodl-tmp 等常见转存位置）")
    p.add_argument("--download-urls", type=str, default="",
                   help="公开数据与共享数据都没有时的公网 tar/zip 直链（逗号分隔）；"
                        "留空则从 HuggingFace ILSVRC/imagenet-1k 下载（gated，需 HF_TOKEN）")
    p.add_argument("--download-limit", type=int, default=0,
                   help="公网下载每个 split 的图片上限（仅调试用，0=全量）")
    p.add_argument("--holdout", type=int, default=20, help="无 val 目录时每类切出的评测张数")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--flow-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--logdir", type=str, default="",
                   help="TensorBoard 事件目录（默认 AutoDL 的 /root/tf-logs/jepa）")
    p.add_argument("--skip-bootstrap", action="store_true")
    p.add_argument("--force-setup", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    # —— resume 续训：从 checkpoint 恢复 cfg/权重/优化器/步数，继续训练 ——
    p.add_argument("--resume", type=str, default="",
                   help="从该 checkpoint 续训（恢复权重/优化器/步数）；steps 会自动 bump 到 ckpt 步 + --resume-extra-steps")
    p.add_argument("--resume-extra-steps", type=int, default=10000,
                   help="resume 时若 --steps 不大于 ckpt 步数，自动追加的训练步数（默认 10000）")
    p.add_argument("--save-every", type=int, default=0,
                   help="每 N 步原子落盘（0=仅结束时）；长跑防中断丢进度")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="梯度累积步数（>1 等效增大有效 batch，稳定 VICReg 方差/协方差等 batch 统计量）")
    # —— Flickr30k（真实图文对，跨模态对齐信号）——
    p.add_argument("--flickr-root", type=str, default="",
                   help="显式指定 Flickr30k 目录（含 flickr30k_train.parquet）；默认按 公开数据→数据盘 自动发现")
    p.add_argument("--no-flickr", action="store_true", help="manifest 不加入 Flickr30k")
    p.add_argument("--no-imagenet", action="store_true",
                   help="manifest 不加入 ImageNet（纯 Flickr 续训时用，避免再跑已训过的 ImageNet）")
    p.add_argument("--flickr-weight", type=float, default=1.0, help="Flickr30k 在混合 manifest 中的采样权重")
    p.add_argument("--imagenet-weight", type=float, default=1.0, help="ImageNet 在混合 manifest 中的采样权重")
    # —— COCO Captions（真实图文对，566k 条）——
    p.add_argument("--coco-root", type=str, default="",
                   help="显式指定 COCO Captions 目录（含 train-*.parquet）；默认自动发现 /root/datasets/COCO_captions 等")
    p.add_argument("--no-coco", action="store_true", help="manifest 不加入 COCO")
    p.add_argument("--coco-weight", type=float, default=1.0, help="COCO 在混合 manifest 中的采样权重")
    # —— token 层 ρ 压缩（§2.6 / §3.4.2 先写后压）：--rho 开启；课程 S1 冻结 ρ 记忆、S2 解冻 + 码率 warmup ——
    p.add_argument("--rho", action="store_true", help="开启 token 层 ρ 压缩（先写后压课程）")
    p.add_argument("--rho-head-dim", type=int, default=64, help="ρ 记忆 M 的 k/v 维度（冻结，O(D²) 显存）")
    p.add_argument("--rho-beta", type=float, default=0.1, help="delta rule 写入率")
    p.add_argument("--rho-eps", type=float, default=0.0, help="绝对阈值（0 禁用，改相对阈值）")
    p.add_argument("--rho-eps-z", type=float, default=None,
                   help="相对阈值 z-score：ε_ρ=批内均值+z·σ（不指定则用 config 默认 0.5）")
    p.add_argument("--rho-eps-keep", type=float, default=0.02, help="ε 强制采样比例（防自我强化退化）")
    p.add_argument("--rho-eps-keep-modal", type=float, default=None,
                   help="§8.6 式 57 模态层保留下界 ε_m（不指定则用 config 默认 0 关闭；>0 防整模态被裁）")
    p.add_argument("--rho-b-a", type=float, default=1.0, help="b_t surprisal 系数 a")
    p.add_argument("--rho-b-c", type=float, default=0.5, help="b_t 条件熵代理系数 c")
    p.add_argument("--rho-b-min", type=float, default=0.0, help="b_t clamp 下界")
    p.add_argument("--rho-b-max", type=float, default=4.0, help="b_t clamp 上界")
    p.add_argument("--rho-rate-weight", type=float, default=None,
                   help="码率正则权重 λ（R+λD，不指定则用 config 默认 0.1）")
    return p.parse_args()


def main():
    args = parse_args()

    # —— AutoDL 环境：镜像/缓存全部放数据盘 ——
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    root = Path(args.root or (
        "/root/autodl-tmp/jepa_autodl"
        if Path("/root/autodl-tmp").is_dir() else "./autodl_work"
    )).resolve()
    os.environ.setdefault("HF_HOME", str(root / "hf_home"))
    os.environ.setdefault("JEPA_CACHE", str(root / "data" / "cache"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not args.skip_bootstrap and not _in_venv():
        if _venv_ok() and not args.force_setup:
            reexec_in_venv()
        bootstrap()
        reexec_in_venv()

    import torch

    if not torch.cuda.is_available():
        print("[warn] 当前没有可用 CUDA，将在 CPU 上运行（仅适合小规模冒烟）")
    else:
        print(f"[gpu] {torch.cuda.get_device_name(0)} "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}GiB")

    # —— resume：自动把 --steps bump 到 ckpt 步 + 增量，避免 steps<=start_step 时训练空转 ——
    if args.resume:
        resume_blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_step = int(resume_blob.get("step", 0))
        del resume_blob
        if args.steps <= ckpt_step:
            args.steps = ckpt_step + args.resume_extra_steps
        print(f"[resume] 从 {args.resume} 续训：ckpt_step={ckpt_step} → 目标 steps={args.steps}")

    # —— 1. 数据装配：ImageNet-1K（公开数据 → 共享数据 → 公网下载）+ Flickr30k（自动发现）——
    train_dir = val_dir = classes_file = None
    if not args.no_imagenet:
        train_dir, val_dir, classes_file = prepare_imagenet(
            root, args.imagenet_root or None,
            share_roots=args.shared_root,
            download_urls=[u.strip() for u in args.download_urls.split(",") if u.strip()],
            download_limit=args.download_limit,
        )
    flickr_root = None
    if not args.no_flickr:
        flickr_root = args.flickr_root or _find_flickr(root, args.shared_root)
        if flickr_root:
            print(f"[Flickr30k] 数据来源: {flickr_root}")
        else:
            print("[Flickr30k] 未找到（--no-flickr 可关闭此提示）；仅用 ImageNet")
    coco_root = None
    if not args.no_coco:
        coco_root = args.coco_root or _find_coco(root, args.shared_root)
        if coco_root:
            print(f"[COCO] 数据来源: {coco_root}")
        else:
            print("[COCO] 未找到（--no-coco 可关闭此提示）")
    manifest_path = Path(args.manifest) if args.manifest else root / "data" / "autodl_manifest.json"
    build_manifest(train_dir, val_dir, classes_file, args.holdout, manifest_path,
                   flickr_root=flickr_root, flickr_weight=args.flickr_weight,
                   imagenet_weight=args.imagenet_weight,
                   coco_root=coco_root, coco_weight=args.coco_weight)

    ckpt = Path(args.ckpt) if args.ckpt else root / "runs" / "jepa_3b_l1024.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # AutoDL 默认把 TensorBoard 事件写到 /root/tf-logs，AutoPanel 直接可视化。
    logdir = Path(args.logdir) if args.logdir else _default_logdir(root)
    logdir.mkdir(parents=True, exist_ok=True)

    # —— 2. 训练（复用 training.train）——
    if not args.skip_train:
        run_train(args, manifest_path, ckpt, logdir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("[train] 跳过")

    # —— 3. 评测（复用 evaluation.metrics.evaluate_real）——
    if not args.skip_eval:
        if not ckpt.exists():
            raise SystemExit(f"checkpoint 不存在: {ckpt}")
        run_eval(args, manifest_path, ckpt)
    else:
        print("[eval] 跳过")

    print(f"\n完成。checkpoint: {ckpt}")
    print(f"TensorBoard 事件: {logdir}（AutoDL 控制台 AutoPanel → TensorBoard）")


if __name__ == "__main__":
    main()
