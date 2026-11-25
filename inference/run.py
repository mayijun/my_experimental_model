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
from model.flow import euler_sample, make_projection


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


def sample_logits(logits, temperature=1.0):
    """温度采样：temperature<=0 退化为 argmax；否则 softmax(logits/T) + multinomial（§4.2(f)）。"""
    if temperature <= 0:
        return logits.argmax(-1)
    V = logits.size(-1)
    probs = torch.softmax(logits.reshape(-1, V) / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).reshape(logits.shape[:-1])


@torch.no_grad()
def generate_text(model, n_samples, n_steps, temperature=1.0, solver="euler"):
    """无条件生成文本：噪声 →（软投影）采样 → 文本采样（§4.1.4 + §4.1.7 + §4.2(f)）。"""
    if model.velocity is None:
        raise ValueError("该模型未启用 Flow（flow_enabled=False），无法生成。")
    device = next(model.parameters()).device
    z_0 = torch.randn(n_samples, model.cfg.text_len, model.dim, device=device)
    proj = make_projection(model.text_decoder.head.weight)
    z_1 = euler_sample(model.velocity, z_0, n_steps, projection=proj, solver=solver)
    return sample_logits(model.text_decoder(z_1), temperature)  # [n, L]


@torch.no_grad()
def generate_image(model, n_samples, n_steps, solver="euler"):
    """无条件生成图像：噪声 →（软投影）采样 → 图像（§4.1.4 + §4.1.7）。"""
    if model.velocity is None:
        raise ValueError("该模型未启用 Flow（flow_enabled=False），无法生成。")
    device = next(model.parameters()).device
    z_0 = torch.randn(n_samples, model.cfg.text_len, model.dim, device=device)
    proj = make_projection(model.image_decoder.head.weight)
    z_1 = euler_sample(model.velocity, z_0, n_steps, projection=proj, solver=solver)
    return model.image_decoder(z_1)  # [n, C, H, W]


def load_checkpoint(path, device):
    """从 training.train --save 落盘的 checkpoint 恢复 (model, cfg, tokenizer)。

    推理只走在线编码器 + 解码器 + 速度场，故加载后：
      1. 删除 EMA 目标编码器（share_trunk 下 3.06B），释放约一半参数显存；
      2. CUDA 下把权重转 bf16，进一步减半显存并利用 Tensor Core 提速。
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = MJEPAConfig(**blob["cfg"])
    model = MultimodalJEPA(cfg, blob["vocab_size"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    for attr in ("image_target", "text_target", "shared_target"):
        if getattr(model, attr, None) is not None:
            delattr(model, attr)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
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


def compile_for_inference(model, enabled):
    """CUDA 下用 torch.compile 优化推理主干。

    - velocity：Flow 采样循环里反复调用，用 mode="reduce-overhead"（内部走 CUDA Graph
      缓存静态图）消除每步的 kernel launch 开销；
    - 两个 decoder：单次调用，用默认 mode 做算子融合。
    含 graph break 的模块（如 SSM 的并行扫描）自动回落 eager，不影响正确性。
    """
    if not (enabled and hasattr(torch, "compile")):
        return
    if model.velocity is not None:
        model.velocity = torch.compile(model.velocity, mode="reduce-overhead")
    model.image_decoder = torch.compile(model.image_decoder)
    model.text_decoder = torch.compile(model.text_decoder)


def solver_hint(n_steps, solver):
    """按采样步数给 ODE 求解器建议（§4.1.4 的工程权衡）。

    二阶方法（heun/midpoint）每步评估两次速度场、成本 ×2，仅在低步数下换来的
    误差收益值得；步数一多，一阶 Euler 的截断误差已足够小，二阶只白耗算力。
    """
    if n_steps <= 8:
        rec, why = "heun", "步数少时二阶方法同步数误差明显更小"
    elif n_steps <= 20:
        rec, why = "euler", "中等步数下一阶精度已够，二阶 ×2 成本不划算"
    else:
        rec, why = "euler", "步数足够多时一阶精度已充分，勿用二阶浪费算力"
    return rec, why


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--text", type=str, default="")          # 文本 → 图像
    parser.add_argument("--image-npy", type=str, default="")     # 图像 → 文本
    parser.add_argument("--out", type=str, default="")           # 生成的图像存成 .npy
    parser.add_argument("--generate-text", type=int, default=0, metavar="N")    # 无条件生成 N 条文本
    parser.add_argument("--generate-image", type=int, default=0, metavar="N")   # 无条件生成 N 张图像
    parser.add_argument("--flow-steps", type=int, default=0)     # 采样步数；0 用 cfg.flow_steps
    parser.add_argument("--solver", type=str, default="euler", choices=("euler", "midpoint", "heun"))  # ODE 积分器（§4.1.4）
    parser.add_argument("--temperature", type=float, default=1.0)  # 文本生成温度；<=0 退化 argmax
    parser.add_argument("--compile", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)  # None=auto：CUDA 自动开 torch.compile
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tokenizer = load_checkpoint(args.ckpt, device)
    use_compile = args.compile if args.compile is not None else (device.type == "cuda")
    compile_for_inference(model, use_compile)
    print(f"device: {device} | dim={cfg.dim} depth={cfg.depth} latent_len={cfg.text_len} "
          f"| compile={use_compile}")

    gen_steps = args.flow_steps or cfg.flow_steps
    if args.generate_text or args.generate_image:
        rec, why = solver_hint(gen_steps, args.solver)
        if rec != args.solver:
            print(f"[提示] flow_steps={gen_steps} → 推荐 --solver {rec}（{why}）；当前为 {args.solver}。")

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

    if args.generate_text:
        if tokenizer is None:
            raise ValueError("该 checkpoint 未保存字符词表，无法解码出文本。")
        ids = generate_text(model, args.generate_text, gen_steps, args.temperature, solver=args.solver)
        for i, row in enumerate(ids):
            print(f"生成文本[{i}]: {tokenizer.decode(row.tolist())!r}")

    if args.generate_image:
        images = generate_image(model, args.generate_image, gen_steps, solver=args.solver)
        for i, img in enumerate(images):
            arr = img.cpu().numpy()
            print(f"生成图像[{i}]: shape {arr.shape}, range [{arr.min():.2f}, {arr.max():.2f}]")
            if args.out:
                np.save(f"{args.out}_{i}.npy", arr)
                print(f"已保存: {args.out}_{i}.npy")


if __name__ == "__main__":
    main()
