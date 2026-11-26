"""训练入口：数据源装配 + 训练循环 + 可选 checkpoint 落盘。

用法（CPU 冒烟）：
    .venv/bin/python -m training.train --preset tiny --steps 150 --tokenizer char --backbone transformer
用法（中型，默认）：
    .venv/bin/python -m training.train --preset 0.3b --tokenizer llada --backbone hybrid
用法（缩放）：
    .venv/bin/python -m training.train --preset base --dim 512 --depth 12 --steps 2000
"""

import argparse
import dataclasses
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data_prepare.hf_datasets import (
    LLaDATokenizer,
    ensure_special_tokens,
    load_hf_tokenizer,
    mixture_iterator,
    real_batch_fn,
    real_batch_iterator,
)
from data_prepare.synthetic import CharTokenizer, build_dataset, synthetic_batch_fn
from evaluation.metrics import evaluate, evaluate_real
from model import MJEPAConfig, PRESETS, MultimodalJEPA, build_config
from model.representation import update_target
from training.losses import maybe_autocast


def setup_distributed():
    """按 torchrun 环境变量初始化分布式；未提供时退回单进程（rank=0, world_size=1）。

    返回 (rank, world_size, device)。多卡时按 LOCAL_RANK 绑定 CUDA 设备，
    `torchrun --nproc_per_node=N -m training.train ...` 即可启动。
    """
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, world_size, device


def _unwrap(model):
    """DDP 包装下取回裸模型。访问 cfg / 子模块 / state_dict / update_target 时用，
    只有前向（model(...)）必须走 DDP 包装以同步梯度。"""
    return model.module if isinstance(model, DDP) else model


def build_param_groups(model, lr):
    """PAC 分层 lr（§3.4「分层 lr（软锚定）」）：锚编码器低 lr、新编码器/新解码器高 lr（~3×）、Z/共享层基准 lr。

    解码器是模态专属解释器（§4.2(a)），新模态接入时其解码器应与新编码器同档高 lr，
    对应 §3.4 S1「训新前端+M+头」里的解码头；锚解码器仍归 shared（S2 起 ×0.1 防漂移）。
    """
    cfg = model.cfg
    anchor_enc = model.image_online if cfg.anchor_modality == "image" else model.text_online
    new_enc = model.text_online if cfg.anchor_modality == "image" else model.image_online
    new_dec = model.text_decoder if cfg.anchor_modality == "image" else model.image_decoder
    shared_ids = set()
    if model.shared_online is not None:
        shared_ids = {id(p) for p in model.shared_online.parameters()}
    anchor_params = [p for p in anchor_enc.parameters() if id(p) not in shared_ids]
    new_params = [p for p in new_enc.parameters() if id(p) not in shared_ids]
    new_dec_params = list(new_dec.parameters())
    anchor_ids = {id(p) for p in anchor_params}
    new_ids = {id(p) for p in new_params}
    new_dec_ids = {id(p) for p in new_dec_params}
    rest = [
        p for p in model.parameters()
        if id(p) not in anchor_ids and id(p) not in new_ids and id(p) not in new_dec_ids
    ]
    return [
        {"name": "anchor", "params": anchor_params, "lr": lr, "base_lr": lr},
        {"name": "new", "params": new_params, "lr": lr * cfg.pac_new_lr_scale,
         "base_lr": lr * cfg.pac_new_lr_scale},
        {"name": "new_decoder", "params": new_dec_params, "lr": lr * cfg.pac_new_lr_scale,
         "base_lr": lr * cfg.pac_new_lr_scale},
        {"name": "shared", "params": rest, "lr": lr, "base_lr": lr},
    ]


def pac_anchor_prob(step, total_steps, cfg):
    """PAC 锚模态采样概率（§3.4 式 21）：p_anchor(t)=p_inf+(p_0-p_inf)·σ(-(t-t0)/τ)。"""
    t = step / max(1, total_steps)
    p0, pinf, tau, t0 = cfg.pac_anchor_p0, cfg.pac_anchor_pinf, cfg.pac_tau, cfg.pac_t0
    return pinf + (p0 - pinf) * float(torch.sigmoid(torch.tensor(-(t - t0) / tau)))


def apply_pac_schedule(optimizer, step, total_steps, cfg):
    """PAC 阶段 lr 缩放（§3.4 表）：S2（50–90%）起 Z/共享层 lr ×pac_z_lr_scale。"""
    if not cfg.pac_enabled:
        return
    t = step / max(1, total_steps)
    z_scale = cfg.pac_z_lr_scale if 0.5 <= t < 0.9 else 1.0
    for g in optimizer.param_groups:
        name = g.get("name", "")
        g["lr"] = g["base_lr"] * (z_scale if name == "shared" else 1.0)


def build_optimizer(params, lr, optim):
    """按 §3.4 选优化器（省显存开关）：

    adamw8bit（默认，bitsandbytes，省 ~50%，CUDA-only，未装时自动降级 adamw）
    / adafactor（torch 内置，省 ~50%，T5/PaLM 验证）/ adamw（基线）。
    """
    if optim == "adafactor":
        return torch.optim.Adafactor(params, lr=lr, weight_decay=0.0)
    if optim == "adamw8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr)
        except ImportError:
            print("[warn] bitsandbytes 未安装（CUDA-only），回退到 AdamW。装：uv sync --extra train-gpu")
        except RuntimeError as e:
            print(f"[warn] AdamW8bit 初始化失败（{e}），回退到 AdamW。")
    return torch.optim.AdamW(params, lr=lr)


def make_optimizer(model, lr, optim):
    """重建优化器（含 PAC 参数分组），供 resume 恢复 optimizer 状态与 train() 复用。"""
    raw = _unwrap(model)
    params = build_param_groups(raw, lr) if raw.cfg.pac_enabled else raw.parameters()
    return build_optimizer(params, lr, optim)


def _capture_rng():
    """捕获 python/torch/numpy 三套随机状态，供 resume 精确续训。

    注意：HF datasets 的流式 shuffle buffer 内部状态无法序列化，故数据游标不能靠 rng 恢复，
    只能做到「同一 seed 下从头 shuffle」；真正的「不重复」由数据台账去重保证（见 data_prepare/ledger.py）。
    """
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }


def _restore_rng(state):
    """恢复 _capture_rng 捕获的三套随机状态（缺 key 时跳过，兼容旧 checkpoint）。"""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "random" in state:
        random.setstate(state["random"])


def _restore_tokenizer(blob):
    """从 checkpoint 的 tokenizer_meta 重建词表（resume 用，兼容 llada / char 两种）。"""
    meta = blob.get("tokenizer_meta") or {}
    if meta.get("type") == "llada":
        return LLaDATokenizer(meta["model"])
    if meta.get("type") == "char":
        return CharTokenizer.from_itos(meta["itos"])
    return None


def _spec_source(spec):
    """从 manifest spec 推断 source_type 与 source_uri（台账溯源用）。"""
    if spec.get("type") == "imagefolder":
        return "imagefolder", spec.get("root")
    if spec.get("data_files"):
        return "parquet", str(spec.get("data_files"))
    return "hf", spec.get("name")


def register_datasets(ledger, args, data_cfg):
    """训练前把本次要用的公开数据集登记进台账，返回 {dataset_id: spec} 供训练后更新进度。

    来源优先 manifest（--data-config），其次单数据集（--dataset-name / --data-dir）；
    synthetic 数据无公开来源，返回空。
    """
    if data_cfg and data_cfg.get("train"):
        specs = data_cfg["train"]
    elif args.data == "real":
        if args.data_dir:
            specs = [{"name": f"imagefolder:{args.data_dir}", "type": "imagefolder",
                      "root": args.data_dir, "config": "", "split": args.split}]
        else:
            specs = [{"name": args.dataset_name, "config": args.dataset_config or "",
                      "split": args.split, "type": "hf"}]
    else:
        return {}
    out = {}
    for spec in specs:
        source_type, source_uri = _spec_source(spec)
        dsid = ledger.mark_dataset(
            name=spec.get("name") or "", config=spec.get("config", ""),
            split=spec.get("split", "train"),
            source_type=source_type, source_uri=source_uri,
        )
        out[dsid] = spec
    return out


def make_writer(logdir, rank=0):
    """rank0 且 logdir 非空时创建 TensorBoard SummaryWriter；未安装 tensorboard 时返回 None。

    训练默认开启事件记录（`--logdir` 指定目录，空字符串关闭）；AutoDL 默认写入
    `/root/tf-logs`，AutoPanel 内置的 TensorBoard 会直接读取该目录。
    """
    if rank != 0 or not logdir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[warn] 未安装 tensorboard，跳过事件记录。装：uv sync --extra train")
        return None
    writer = SummaryWriter(log_dir=logdir)
    print(f"TensorBoard: {logdir}（tensorboard --logdir {logdir}）")
    return writer


def train(model, batch_fn, pad_id, mask_id, device, steps, lr, optim="adamw", rank=0,
          writer=None, optimizer=None, start_step=0):
    """通用训练循环：从 batch_fn 拉 (img, txt)，跑「编码→对齐/推理→解码」。

    resume 时传入已恢复状态的 optimizer 与 start_step，从 start_step+1 继续；
    steps 为「绝对目标总步数」，resume 后应 > start_step，否则本函数跳过主训练。
    """
    model.train()
    raw = _unwrap(model)
    cfg = raw.cfg
    if optimizer is None:
        optimizer = make_optimizer(model, lr, optim)
    ema_state = {}

    def _log_scalar(tag, val, step):
        # 原始值 + EMA 平滑值双写：原始曲线保留诊断尖峰，_ema 曲线压平小 batch 噪声便于读趋势
        prev = ema_state.get(tag, val)
        cur = 0.95 * prev + 0.05 * val
        ema_state[tag] = cur
        writer.add_scalar(tag, val, step)
        writer.add_scalar(tag + "_ema", cur, step)

    for step in range(start_step + 1, steps + 1):
        img, txt = batch_fn()
        with maybe_autocast(device):
            # 走 model(...)（而非直接调 forward_losses），使 DDP 能同步梯度。
            loss, parts = model(img, txt, pad_id, mask_id, step=step, total_steps=steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        apply_pac_schedule(optimizer, step, steps, cfg)
        update_target(raw, cfg.ema_decay)

        if writer is not None:
            _log_scalar("train/loss", loss.item(), step)
            for k, v in parts.items():
                _log_scalar(f"train/{k}", v.item(), step)
            cur_lr = optimizer.param_groups[0].get("lr")
            if cur_lr is not None:
                writer.add_scalar("train/lr", cur_lr, step)
            if cfg.pac_enabled:
                writer.add_scalar("train/p_anchor", pac_anchor_prob(step, steps, cfg), step)

        if rank == 0 and (step == 1 or step % 25 == 0 or step == steps):
            detail = " ".join(f"{k}={v.item():.3f}" for k, v in parts.items())
            extra = f" | p_anchor={pac_anchor_prob(step, steps, cfg):.3f}" if cfg.pac_enabled else ""
            print(f"step {step:4d} | loss {loss.item():.4f} | {detail}{extra}")
    return model


def train_flow_stage(model, batch_fn, device, steps, lr, optim="adamw", rank=0, writer=None):
    """Flow 阶段（§4.2(e)）：冻结 E_ξ/E_θ 与解码器，只训速度场 v_ψ，不 update_target。

    p_1（冻结目标 latent 分布）静止是 Flow Matching 运输映射 well-posed 的前提；
    与 V-JEPA 2「冻结世界模型、单独训视觉解码器」同构。主训练完成后调用。
    多卡下走 model(..., mode="flow")：encoder/decoder 参数不参与本前向，靠 DDP 的
    find_unused_parameters=True 跳过，只同步 velocity 梯度（该参数会强制 use_orig_params）。
    """
    raw = _unwrap(model)
    cfg = raw.cfg
    if raw.velocity is None:
        raise ValueError("Flow 阶段需要 flow_enabled=True（模型未启用速度场 v_ψ）。")
    model.train()
    optimizer = build_optimizer(
        [p for p in raw.velocity.parameters() if p.requires_grad], lr, optim
    )
    ema_state = {}

    def _log_scalar(tag, val, step):
        prev = ema_state.get(tag, val)
        cur = 0.95 * prev + 0.05 * val
        ema_state[tag] = cur
        writer.add_scalar(tag, val, step)
        writer.add_scalar(tag + "_ema", cur, step)

    for step in range(1, steps + 1):
        img, txt = batch_fn()
        with maybe_autocast(device):
            loss, parts = model(img, txt, mode="flow")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if writer is not None:
            _log_scalar("flow/loss", loss.item(), step)
            for k, v in parts.items():
                _log_scalar(f"flow/{k}", v.item(), step)
        if rank == 0 and (step == 1 or step % 25 == 0 or step == steps):
            print(f"flow-stage {step:4d} | flow {parts['flow'].item():.4f} "
                  f"| flow_cond {parts['flow_cond'].item():.4f}")
    return model


def _cast_floating(obj, dtype):
    """递归把 state_dict 中的浮点张量转成 dtype 并移回 CPU（非浮点张量与 python 标量原样保留）。"""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", dtype) if torch.is_floating_point(obj) else obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cast_floating(v, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cast_floating(v, dtype) for v in obj)
    return obj


def save_checkpoint(path, model, cfg, vocab_size, tokenizer_meta=None, save_dtype="fp32",
                    optimizer=None, step=0, rng_state=None):
    """落盘模型权重 + 续训所需完整状态（供 inference 加载与 training --resume 续训）。

    tokenizer_meta：字符级为 {"type": "char", "itos": [...]}，
    LLaDA 为 {"type": "llada", "model": "GSAI-ML/LLaDA-8B-Instruct"}。
    save_dtype="bf16" 时权重转 bf16 再落盘（3b 档约 13.7GB，fp32 约 27.4GB）；
    inference 的 load_state_dict 会自动把 bf16 拷贝回 fp32 参数。
    optimizer / step / rng_state 为续训状态（--resume 恢复）；inference 忽略这些键。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # 统一落盘 dtype：save_dtype="bf16" 时，model 权重与 optimizer moments 降到 bf16；
    # step 保持 fp32（Adafactor bias correction 依赖整数步数精度）；rng 的 uint8 非浮点原样保留。
    state = model.state_dict_for_save()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    if save_dtype == "bf16":
        # 逐张量转到 CPU 再降精度，避免在 GPU 上额外驻留一份权重。
        state = _cast_floating(state, torch.bfloat16)
        if optimizer_state is not None:
            # moments（row_var/col_var）降 bf16，load_state_dict 会按 param dtype 自动 cast 回 fp32；
            # step 保持 fp32，避免步数 >256 后 bf16 无法精确表示整数导致的精度损失。
            optimizer_state = {
                "state": {
                    pid: {
                        name: (val if name == "step" else _cast_floating(val, torch.bfloat16))
                        for name, val in pstate.items()
                    }
                    for pid, pstate in optimizer_state["state"].items()
                },
                "param_groups": optimizer_state["param_groups"],
            }
    torch.save(
        {
            "model": state,
            "cfg": dataclasses.asdict(cfg),
            "vocab_size": vocab_size,
            "tokenizer_meta": tokenizer_meta,
            "step": step,
            "optimizer": optimizer_state,
            "rng_state": rng_state,
        },
        path,
    )
    print(f"checkpoint 已保存: {path}（dtype={save_dtype}，step={step}）")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="0.3b")
    # 模型规模（覆盖 preset 默认值）
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    # 主干选型（§2.2.5）
    parser.add_argument("--backbone", choices=["transformer", "mamba", "hybrid"], default=None)
    parser.add_argument("--attn-layers", type=int, default=None)
    # 图像前端（§2.2.7）：patch（单层卷积）/ siglip2（LLaDA-V 前端：SigLIP-2 + MLP）
    parser.add_argument("--image-frontend", choices=["patch", "siglip2"], default=None)
    parser.add_argument("--siglip2-model", type=str, default=None)
    parser.add_argument("--freeze-vision", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    # token 层 ρ 残差压缩 + 比特门控 b_t（§2.6，默认关闭，开启后按相对阈值丢冗余）
    parser.add_argument("--rho-compress", action="store_true", default=None)
    parser.add_argument("--rho-head-dim", type=int, default=None)
    parser.add_argument("--rho-beta", type=float, default=None)
    parser.add_argument("--rho-eps", type=float, default=None)       # 绝对阈值（0 禁用，改相对阈值 --rho-eps-z）
    parser.add_argument("--rho-eps-z", type=float, default=None)     # 相对阈值 z-score：ε_ρ = 批内均值 + z·σ
    parser.add_argument("--rho-eps-keep", type=float, default=None)  # ε 强制采样比例（防自我强化退化）
    parser.add_argument("--rho-b-a", type=float, default=None)       # b_t surprisal 系数 a
    parser.add_argument("--rho-b-c", type=float, default=None)       # b_t 条件熵代理系数 c
    parser.add_argument("--rho-b-min", type=float, default=None)     # b_t clamp 下界
    parser.add_argument("--rho-b-max", type=float, default=None)     # b_t clamp 上界
    parser.add_argument("--rho-rate-weight", type=float, default=None)  # 码率正则权重 λ（R+λD）
    # 训练
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--optim", choices=["adamw", "adafactor", "adamw8bit"], default="adamw8bit")
    # 对齐方法（§3.4）：infonce / byol（默认）/ dino
    parser.add_argument("--alignment-method", choices=["infonce", "byol", "dino"], default=None)
    # 交叉掩码（§3.2 式 18）与 PAC（§3.4 式 21）
    parser.add_argument("--cross-mask", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--pac", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--anchor-modality", choices=["image", "text"], default=None)
    # PSCO 状态依赖收缩算子（§3.7.3）
    parser.add_argument("--psi", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--psi-transient-steps", type=int, default=None)
    parser.add_argument("--psi-scan-steps", type=int, default=None)
    parser.add_argument("--psi-delta", type=float, default=None)
    parser.add_argument("--psi-gamma", type=float, default=None)
    parser.add_argument("--psi-residual", choices=["none", "mlp"], default=None)
    parser.add_argument("--psi-readout-weight", type=float, default=None)
    parser.add_argument("--psi-var-weight", type=float, default=None)
    parser.add_argument("--psi-stop-weight", type=float, default=None)
    parser.add_argument("--psi-consist-weight", type=float, default=None)
    parser.add_argument("--flow", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--flow-depth", type=int, default=None)
    parser.add_argument("--flow-heads", type=int, default=None)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--flow-weight", type=float, default=None)
    parser.add_argument("--flow-cond", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--flow-cond-weight", type=float, default=None)
    parser.add_argument("--flow-adaln", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--share-trunk", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--embed-factor", type=int, default=None)  # 文本词嵌入低秩分解维 F（0=关闭）
    parser.add_argument("--grad-checkpoint", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)  # 梯度检查点（省激活显存）
    parser.add_argument("--flow-stage-steps", type=int, default=0)  # >0 时主训练后进入 Flow 阶段（§4.2(e)）
    parser.add_argument("--dataset-size", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="")  # 例如 runs/mjepa.pt
    parser.add_argument("--save-dtype", choices=["fp32", "bf16"], default="fp32")  # 落盘精度（bf16 省一半磁盘）
    parser.add_argument("--resume", type=str, default="")  # 从 checkpoint 续训：恢复 cfg/权重/优化器/步数/rng
    parser.add_argument("--ledger", type=str, default="")  # 数据台账 SQLite 路径（空=关闭；记录已训练数据集）
    # TensorBoard：默认开启，事件写入 --logdir（空字符串关闭）；AutoDL 用 /root/tf-logs
    parser.add_argument("--logdir", type=str, default="runs/tensorboard")
    # 数据源：synthetic（内置合成图文对，离线） / real（HF datasets 真实图文对）
    parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    # 文本 tokenizer：llada（现成词表，内置 mask id，见 math.md §2.2.7） / char（字符级，离线冒烟）
    parser.add_argument("--tokenizer", choices=["char", "llada"], default="llada")
    parser.add_argument("--llada-model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset-name", type=str, default="lambdalabs/pokemon-blip-captions")
    parser.add_argument("--dataset-config", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--data-dir", type=str, default="")  # 本地 imagefolder，优先于 dataset-name。
    parser.add_argument("--data-config", type=str, default="")  # 中档 manifest（data_prepare.medium 生成）：多数据集加权混合 + 独立 eval 集。
    parser.add_argument("--shuffle-buffer", type=int, default=10000)  # 流式 buffer shuffle 大小（0=关闭；越大越接近全局随机，越耗内存）。
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    args = parser.parse_args()

    # —— 中档 manifest：多数据集加权混合 + 独立 eval 集（data_prepare.medium 生成）——
    data_cfg = None
    if args.data_config:
        with open(args.data_config, encoding="utf-8") as f:
            data_cfg = json.load(f)

    # —— resume：加载 checkpoint（cfg/词表/步数/优化器/rng 都从这里恢复）——
    resume_blob = None
    start_step = 0
    if args.resume:
        resume_blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_step = int(resume_blob.get("step", 0))

    # —— 分布式初始化（torchrun）：rank/world_size/device；未用 torchrun 时退回单进程 ——
    rank, world_size, device = setup_distributed()

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    if resume_blob is not None and resume_blob.get("rng_state"):
        _restore_rng(resume_blob["rng_state"])

    image_size = args.image_size or (data_cfg or {}).get("image_size")
    cfg = build_config(
        args.preset,
        image_size=image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        backbone=args.backbone,
        attn_layers=args.attn_layers,
        image_frontend=args.image_frontend,
        siglip2_model=args.siglip2_model,
        freeze_vision=args.freeze_vision,
        rho_compress=args.rho_compress,
        rho_head_dim=args.rho_head_dim,
        rho_beta=args.rho_beta,
        rho_eps=args.rho_eps,
        rho_eps_z=args.rho_eps_z,
        rho_eps_keep=args.rho_eps_keep,
        rho_b_a=args.rho_b_a,
        rho_b_c=args.rho_b_c,
        rho_b_min=args.rho_b_min,
        rho_b_max=args.rho_b_max,
        rho_rate_weight=args.rho_rate_weight,
        alignment_method=args.alignment_method,
        cross_mask=args.cross_mask,
        pac_enabled=args.pac,
        anchor_modality=args.anchor_modality,
        psi_enabled=args.psi,
        psi_transient_steps=args.psi_transient_steps,
        psi_scan_steps=args.psi_scan_steps,
        psi_delta=args.psi_delta,
        psi_gamma=args.psi_gamma,
        psi_residual=args.psi_residual,
        psi_readout_weight=args.psi_readout_weight,
        psi_var_weight=args.psi_var_weight,
        psi_stop_weight=args.psi_stop_weight,
        psi_consist_weight=args.psi_consist_weight,
        flow_enabled=args.flow,
        flow_depth=args.flow_depth,
        flow_heads=args.flow_heads,
        flow_steps=args.flow_steps,
        flow_weight=args.flow_weight,
        flow_cond_enabled=args.flow_cond,
        flow_cond_weight=args.flow_cond_weight,
        flow_adaln=args.flow_adaln,
        share_trunk=args.share_trunk,
        embed_factor=args.embed_factor,
        grad_checkpoint=args.grad_checkpoint,
    )
    if resume_blob is not None:
        # resume：cfg 以 checkpoint 为准，忽略本次 CLI 覆盖项（保证权重形状一致）。
        cfg = MJEPAConfig(**resume_blob["cfg"])
    assert cfg.width % cfg.heads == 0, f"主干宽度 ({cfg.width}) 必须能被 heads ({cfg.heads}) 整除"
    assert cfg.image_size % cfg.vision_patch == 0, "image_size 必须能被前端 patch 边长整除"

    if rank == 0:
        print(f"device: {device} | world_size: {world_size} | preset: {args.preset} "
              f"| d={cfg.dim} d_hidden={cfg.width} "
          f"depth={cfg.depth} heads={cfg.heads} backbone={cfg.backbone} "
          f"frontend={cfg.image_frontend} image={cfg.image_size} patch={cfg.patch_size} "
          f"latent_len={cfg.text_len} rho_compress={cfg.rho_compress} "
          f"align={cfg.alignment_method} cross_mask={cfg.cross_mask} pac={cfg.pac_enabled} "
          f"psi={cfg.psi_enabled}")

    # —— 词表（tokenizer）选择 ——
    tokenizer_meta = None
    if resume_blob is not None and resume_blob.get("tokenizer_meta"):
        # resume：优先从 checkpoint 恢复词表，保证 vocab 一致；vocab_size 稍后统一用 checkpoint 值。
        tokenizer_meta = resume_blob["tokenizer_meta"]
        tokenizer = _restore_tokenizer(resume_blob)
        pad_id, mask_id = tokenizer.pad_id, tokenizer.mask_id
    elif args.tokenizer == "llada":
        tokenizer = LLaDATokenizer(args.llada_model)
        pad_id, mask_id, vocab_size = tokenizer.pad_id, tokenizer.mask_id, tokenizer.vocab_size
        tokenizer_meta = {"type": "llada", "model": args.llada_model}
    elif args.data == "real":
        if args.hf_endpoint:
            os.environ["HF_ENDPOINT"] = args.hf_endpoint
        tokenizer = load_hf_tokenizer(args.tokenizer_name)
        pad_id, mask_id, vocab_size = ensure_special_tokens(tokenizer)
    else:
        tokenizer = None  # 由 build_dataset 内部构造 CharTokenizer

    # —— 数据台账（可选）：登记本次公开数据集，训练后记录进度/完成 ——
    ledger = None
    registered = {}
    if args.ledger:
        from data_prepare.ledger import Ledger

        ledger = Ledger(args.ledger)
        registered = register_datasets(ledger, args, data_cfg)
        if rank == 0:
            print(f"数据台账: {args.ledger}（登记 {len(registered)} 个数据集）")

    # —— 数据装配 ——
    eval_gen = None
    if args.data == "real":
        tok_name = args.llada_model if args.tokenizer == "llada" else args.tokenizer_name
        if data_cfg and data_cfg.get("train"):
            # 中档 manifest：train 多数据集加权混合 + 独立 eval 集。
            train_specs = data_cfg["train"]
            eval_specs = data_cfg.get("eval") or train_specs
            gen = mixture_iterator(
                train_specs, tokenizer, pad_id, cfg.text_len, cfg.image_size,
                seed=args.seed + rank, shuffle_buffer=args.shuffle_buffer,
                ledger=ledger,
            )
            eval_gen = mixture_iterator(
                eval_specs, tokenizer, pad_id, cfg.text_len, cfg.image_size,
                seed=args.seed + rank, shuffle_buffer=args.shuffle_buffer,
            )
            source = f"{len(train_specs)} 个数据集（manifest）"
        else:
            gen = real_batch_iterator(
                args, tokenizer, pad_id, cfg.text_len, cfg.image_size,
                shuffle_buffer=args.shuffle_buffer, seed=args.seed + rank,
            )
            eval_gen = gen
            source = args.data_dir or args.dataset_name
        batch_fn = real_batch_fn(gen, args.batch_size, device)
        print(f"数据: 真实图文对（{source}） | tokenizer: {tok_name} | "
              f"vocab={vocab_size} pad={pad_id} mask={mask_id}")
    else:
        images, caption_ids, captions, tokenizer = build_dataset(
            args.dataset_size, cfg.image_size, cfg.text_len, seed=args.seed + rank, tokenizer=tokenizer
        )
        pad_id, mask_id = tokenizer.pad_id, tokenizer.mask_id
        vocab_size = tokenizer.vocab_size
        if tokenizer_meta is None:
            tokenizer_meta = {"type": "char", "itos": getattr(tokenizer, "itos", None)}
        batch_fn = synthetic_batch_fn(images, caption_ids, args.batch_size, device)

    if resume_blob is not None:
        vocab_size = resume_blob["vocab_size"]  # 词表大小以 checkpoint 为准
    model = MultimodalJEPA(cfg, vocab_size).to(device)
    if resume_blob is not None:
        raw = _unwrap(model)
        raw.load_state_dict(resume_blob["model"], strict=False)
        raw.init_target_from_online()
    if world_size > 1:
        # find_unused_parameters=True：默认 BYOL 下 proj 等分支本步不参与 loss，
        # 需遍历图跳过未用参数，否则 DDP 报「未参与反向」错。代价是一次图遍历。
        # CPU（gloo）下不传 device_ids；CUDA 下绑定本地卡。
        ddp_kwargs = {"device_ids": [device]} if torch.cuda.is_available() else {}
        model = DDP(model, find_unused_parameters=True, **ddp_kwargs)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"参数: {total/1e6:.2f}M 总 | {trainable/1e6:.2f}M 可训练 | vocab={vocab_size}")

    writer = make_writer(args.logdir, rank)

    # —— 优化器：统一在 main 创建（PAC 参数分组）；resume 时恢复 optimizer 状态（moment 等）——
    optimizer = make_optimizer(_unwrap(model), args.lr, args.optim)
    if resume_blob is not None and resume_blob.get("optimizer") is not None:
        optimizer.load_state_dict(resume_blob["optimizer"])

    train(model, batch_fn, pad_id, mask_id, device, args.steps, args.lr, optim=args.optim,
          rank=rank, writer=writer, optimizer=optimizer, start_step=start_step)

    if args.flow_stage_steps > 0:
        if rank == 0:
            print(f"\n=== Flow 阶段（§4.2(e)）：冻结 E_ξ/E_θ，只训 v_ψ，{args.flow_stage_steps} 步 ===")
        train_flow_stage(model, batch_fn, device, args.flow_stage_steps, args.lr, optim=args.optim,
                         rank=rank, writer=writer)

    if rank == 0 and args.save:
        # DDP 下 state_dict 键带 "module." 前缀，落盘前解包回裸模型，供 inference 直接加载。
        save_checkpoint(args.save, _unwrap(model), cfg, vocab_size,
                        tokenizer_meta=tokenizer_meta, save_dtype=args.save_dtype,
                        optimizer=optimizer, step=max(start_step, args.steps),
                        rng_state=_capture_rng())

    # —— 数据台账收尾：记录本次训练会话 + 累加数据集进度（仅 rank0）——
    if rank == 0 and ledger is not None:
        trained_steps = max(0, args.steps - start_step)
        ledger.record_run(
            run_id=f"{args.preset}-{int(time.time())}", preset=args.preset,
            ckpt_path=args.save, start_step=start_step,
            end_step=max(start_step, args.steps), steps=args.steps,
            config=dataclasses.asdict(cfg),
        )
        for dsid in registered:
            ledger.update_dataset_progress(dsid, steps=trained_steps)
        ledger.close()

    if rank == 0:
        if args.data == "real":
            i2t, t2i = evaluate_real(_unwrap(model), eval_gen, tokenizer, pad_id, device,
                                     args.eval_samples)
        else:
            i2t, t2i = evaluate(_unwrap(model), images, caption_ids, captions, tokenizer, device,
                                args.eval_samples)
        if writer is not None:
            writer.add_scalar("eval/i2t_top1", i2t, args.steps)
            writer.add_scalar("eval/t2i_top1", t2i, args.steps)

    if writer is not None:
        writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
