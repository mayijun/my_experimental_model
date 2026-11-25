#!/usr/bin/env python3
"""AutoDL 单卡 3b / L=1024 全流程：环境自举 → ImageNet-1K → 训练 → 评测。

AutoDL 专属行为全部集中在本文件：
- 自动定位 `/root/autodl-pub` 下的 ImageNet-1K（已解压优先，只有 tar/zip 时才解压）；
- 用 uv 安装与显卡匹配的 torch（A800 → cu126，RTX PRO 6000/Blackwell → cu130）；
- HF 镜像、数据盘缓存、bf16 checkpoint（约 13.7GB）；
- TensorBoard 事件默认写入 `/root/tf-logs`（AutoDL 文档），AutoPanel 直接可视化。
其余全部复用现有代码：model、training.train、data_prepare.hf_datasets、
inference.run、evaluation.metrics。

运行：
    python scripts/train_on_AutoDL.py
    python scripts/train_on_AutoDL.py --steps 20000 --batch-size 2
    python scripts/train_on_AutoDL.py --imagenet-root /root/autodl-pub/ImageNet/ILSVRC2012

注意：ImageNet-1K 训练集约 1.28M 张、解压后约 150GB。若公开数据已是解压目录，
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
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
if str(REPO) not in sys.path:  # 允许 `python scripts/train_on_AutoDL.py` 直接导入仓库内模块
    sys.path.insert(0, str(REPO))
VENV_PY = REPO / ".venv" / "bin" / "python"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
CANDIDATE_NAMES = ("ImageNet", "imagenet", "ImageNet-1K", "imagenet1k",
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
    return wnids >= max(5, len(subs) // 2)


def _layout(path):
    """返回 (train_dir, val_dir, archive)，archive 表示尚未解压的文件。"""
    p = Path(path)
    if (p / "train").is_dir() and _looks_like_class_dir(p / "train"):
        val = p / "val"
        return p / "train", (val if _looks_like_class_dir(val) else None), None
    if _looks_like_class_dir(p):
        return p, None, None
    for name in ("ILSVRC2012_img_train.tar", "imagenet.tar", "ImageNet.tar"):
        if (p / name).exists():
            return None, None, p / name
    zips = sorted(list(p.glob("*.zip")) + list(p.glob("*.tar.gz")))
    if zips:
        return None, None, zips[0]
    return None, None, None


def _find_imagenet(root):
    env = os.environ.get("AUTODL_IMAGENET", "")
    cands = [env] + [str(Path(base) / n) for base in ("/root/autodl-pub", "/root/autodl-tmp")
                     for n in CANDIDATE_NAMES]
    pub = Path("/root/autodl-pub")
    if pub.is_dir():
        for child in pub.iterdir():
            if not child.is_dir():
                continue
            if "imagenet" in child.name.lower() or "ilsvrc" in child.name.lower():
                cands.append(str(child))
                cands.extend(str(g) for g in child.iterdir() if g.is_dir())
            elif (child / "ILSVRC2012_img_train.tar").exists():
                cands.append(str(child))
    for cand in cands:
        if cand and Path(cand).exists():
            train, val, archive = _layout(cand)
            if train or archive:
                return Path(cand)
    return None


def _need_space(path, need):
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    free = shutil.disk_usage(p).free
    if free < need:
        raise SystemExit(
            f"[ImageNet] 解压需要约 {need / 1e9:.0f}GB，数据盘仅剩 {free / 1e9:.0f}GB。\n"
            "请在 AutoDL 扩容数据盘到 ≥160GB，或改用已解压的公开数据目录：\n"
            "  python scripts/train_on_AutoDL.py --imagenet-root <含 train/n0xxxxxxxx 的目录>\n"
            "若公开数据已是解压目录，脚本会只读使用，不会复制。"
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


def prepare_imagenet(root, imagenet_root=None):
    """定位或解压 ImageNet-1K，返回 (train_dir, val_dir, classes_file)。"""
    found = Path(imagenet_root) if imagenet_root else _find_imagenet(root)
    if not found:
        raise SystemExit(
            "未找到 ImageNet-1K。请先在 AutoDL 控制台「公开数据」搜索 ImageNet，\n"
            "复制其路径后用 --imagenet-root 指定；或等待公开数据挂载到 /root/autodl-pub。"
        )
    print(f"[ImageNet] 数据集目录: {found}")
    train, val, archive = _layout(found)
    if archive and not train:
        out = _extract_archive(archive, Path(root) / "data" / "imagenet")
        train, val, _ = _layout(out)
        if not train:
            raise SystemExit(f"[ImageNet] 解压后仍未找到 train/n0xxxxxxxx 目录: {out}")
    if not train:
        raise SystemExit(f"[ImageNet] {found} 既不是 ImageFolder 结构，也没有可解压的 tar/zip。")

    classes_file = _find_classes_file(found, train)
    if not classes_file:
        classes_file = _download_classes(Path(root) / "data" / "imagenet_classes.txt")
    print(f"[ImageNet] train={train} val={val} classes_file={classes_file}")
    return str(train), (str(val) if val else None), classes_file


def build_manifest(train_dir, val_dir, classes_file, holdout, manifest_path):
    train_spec = {
        "type": "imagefolder", "name": "autodl-imagenet1k-train",
        "root": train_dir, "classes_file": classes_file,
        "split": "train" if not val_dir else "all",
        "holdout": 0 if val_dir else holdout,
        "templates": TEMPLATES, "weight": 1.0,
    }
    eval_spec = {
        "type": "imagefolder", "name": "autodl-imagenet1k-eval",
        "root": val_dir or train_dir, "classes_file": classes_file,
        "split": "all" if val_dir else "holdout",
        "holdout": 0 if val_dir else holdout,
        "templates": TEMPLATES[:1], "weight": 1.0,
    }
    manifest = {
        "preset": "3b", "image_size": 256, "text_len": 1024,
        "tokenizer": "llada", "llada_model": "GSAI-ML/LLaDA-8B-Instruct",
        "train": [train_spec], "eval": [eval_spec],
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ImageNet] manifest: {manifest_path}（holdout={0 if val_dir else holdout}/类）")
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
    print(f"[train] optim={optim} batch={args.batch_size} steps={args.steps}+{args.flow_steps}")
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
                   help="ImageNet-1K 目录（含 train/n0xxxxxxxx），默认自动搜索 /root/autodl-pub")
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

    # —— 1. ImageNet-1K ——
    train_dir, val_dir, classes_file = prepare_imagenet(root, args.imagenet_root or None)
    manifest_path = Path(args.manifest) if args.manifest else root / "data" / "autodl_manifest.json"
    build_manifest(train_dir, val_dir, classes_file, args.holdout, manifest_path)

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
