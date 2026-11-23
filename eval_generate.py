"""加载预训练 checkpoint，在留出集上评测 Flow Matching + JEPA 的生成能力。

评测内容：
1. 重建指标（在 eval.jsonl 上）：字符级困惑度、top-1/top-5 准确率
   —— 衡量「文本 -> 潜变量瓶颈 -> 还原字符」的信息保留能力；
2. 条件生成：给定中/英文前缀，用 Flow 采样 + 量化 + 解码生成续写，展示样例；
3. 多样性：生成文本的一元/二元字符熵率；
4. 自洽性 / 重编码失真：生成潜变量 Z 与「重编码 E(T)」的 MSE/余弦，及解码自洽率。

用法（需与 pretrain.py 同一目录，且已生成 checkpoints/）：
    python eval_generate.py --checkpoint-dir checkpoints \
        --eval-file /root/autodl-tmp/data/eval.jsonl \
        --inference-steps 30 --solver heun --num-samples 32
"""
import argparse
import json
import math
import sys
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import flow_matching_jepa as fm  # noqa: E402


# ------------------------- 词表 -------------------------


def load_vocab(config):
    vocab = config["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}
    mask_id = len(vocab)
    return stoi, itos, mask_id


def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]


# ------------------------- 重建指标 -------------------------


@torch.no_grad()
def eval_reconstruction(model, eval_texts, stoi, seq_len, mask_id, device):
    """在留出文本上统计字符级困惑度与 top-k 准确率。"""
    model.eval()
    total_ce = 0.0
    total_n = 0
    top1 = 0
    top5 = 0
    buffer = []
    for t in eval_texts:
        buffer.extend(encode(t, stoi))
        while len(buffer) >= seq_len:
            ids = torch.tensor(buffer[:seq_len], dtype=torch.long, device=device).unsqueeze(0)
            buffer = buffer[seq_len:]
            z = model.online(ids)
            z_q, _, _ = model.quantize(z)
            logits = model.decoder(z_q)  # [1,L,V]
            logits[..., mask_id] = -torch.inf
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
            total_ce += ce.item() * ids.numel()
            total_n += ids.numel()
            pred = logits.argmax(dim=-1)
            top1 += (pred == ids).sum().item()
            top5 += (logits.topk(5, dim=-1).indices == ids.unsqueeze(-1)).any(dim=-1).sum().item()
    if total_n == 0:
        return None
    return {
        "ppl": math.exp(total_ce / total_n),
        "top1": top1 / total_n,
        "top5": top5 / total_n,
    }


# ------------------------- 条件生成 -------------------------


@torch.no_grad()
def generate(model, prompt, stoi, itos, mask_id, seq_len, device, steps, solver, readout, temperature):
    model.eval()
    ids = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    for i, c in enumerate(prompt[:seq_len]):
        if c in stoi:
            ids[0, i] = stoi[c]
    context = model.online(ids)
    z = fm.flow_sample(model, context, steps, device, solver)
    tokens = fm.decode_latents(model, z, mask_id, readout, temperature)[0].tolist()
    prefix = min(len(prompt), seq_len)
    tokens[:prefix] = ids[0, :prefix].tolist()
    text = "".join(itos[i] for i in tokens)
    return text, z, tokens


def entropy_rate(texts):
    flat = [c for t in texts for c in t]
    total = len(flat)
    if total == 0:
        return 0.0, 0.0
    uni = Counter(flat)
    h1 = -sum((c / total) * math.log2(c / total) for c in uni.values())
    pairs = Counter()
    for t in texts:
        for a, b in zip(t, t[1:]):
            pairs[(a, b)] += 1
    succ = {}
    for (a, b), c in pairs.items():
        succ.setdefault(a, Counter())[b] = c
    h2 = 0.0
    for a, n in uni.items():
        s = succ.get(a)
        if not s:
            continue
        tot = sum(s.values())
        h2 += (n / total) * (-sum((v / tot) * math.log2(v / tot) for v in s.values()))
    return h1, h2


# ------------------------- 主流程 -------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--eval-file", type=str, default="/root/autodl-tmp/data/eval.jsonl")
    p.add_argument("--inference-steps", type=int, default=30)
    p.add_argument("--solver", choices=["euler", "heun"], default="heun")
    p.add_argument("--readout", choices=["argmax", "sample"], default="argmax")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--max-eval-chars", type=int, default=200000)
    p.add_argument(
        "--prompts",
        type=str,
        default="人工智能的未来,机器学习需要,The quick brown fox,In recent years",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{args.checkpoint_dir}/config.json", encoding="utf-8") as f:
        config = json.load(f)
    stoi, itos, mask_id = load_vocab(config)
    seq_len = config["seq_len"]

    model = fm.FlowJEPA(
        vocab_size=len(config["vocab"]) + 1,
        max_len=seq_len,
        dim=config["dim"],
        n_heads=config["n_heads"],
        encoder_layers=config["encoder_layers"],
        flow_layers=config["flow_layers"],
        decoder_layers=config.get("decoder_layers", 2),
        jepa_layers=config.get("jepa_layers", 4),
        quantizer=config["quantizer"],
    ).to(device)

    state = torch.load(f"{args.checkpoint_dir}/model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"已加载 checkpoint：dim={config['dim']} enc={config['encoder_layers']} "
          f"flow={config['flow_layers']} quantizer={config['quantizer']} seq={seq_len} "
          f"vocab={len(config['vocab'])}")

    # 1) 重建指标
    eval_texts = []
    with open(args.eval_file, encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            eval_texts.append(json.loads(line)["text"])
    recon = eval_reconstruction(model, eval_texts, stoi, seq_len, mask_id, device)
    print("\n=== 1) 留出集重建指标（文本 -> 潜变量 -> 字符） ===")
    if recon:
        print(f"  困惑度 ppl = {recon['ppl']:.2f} | top-1 = {recon['top1']*100:.2f}% | "
              f"top-5 = {recon['top5']*100:.2f}%")

    # 2) 条件生成样例
    prompts = [s for s in args.prompts.split(",") if s.strip()]
    gen_texts = []
    zs = []
    print("\n=== 2) 条件生成样例 ===")
    with torch.no_grad():
        for prompt in prompts:
            text, z, tokens = generate(
                model, prompt, stoi, itos, mask_id, seq_len, device,
                args.inference_steps, args.solver, args.readout, args.temperature,
            )
            gen_texts.append(text)
            zs.append(z)
            print(f"  提示「{prompt}」\n    -> {text}")

    # 3) 多样性
    h1, h2 = entropy_rate(gen_texts)
    log2v = math.log2(len(config["vocab"]))
    print("\n=== 3) 生成文本多样性（字符熵率） ===")
    print(f"  一元熵 H1 = {h1:.4f} bits/char | 二元熵 H2 = {h2:.4f} bits/char | "
          f"上界 log2|V| = {log2v:.4f}")

    # 4) 自洽性 / 重编码失真
    z = torch.cat(zs, dim=0)  # [M,L,D]
    ids = torch.tensor([[stoi[c] for c in t if c in stoi] for t in gen_texts],
                       dtype=torch.long, device=device)
    ids = ids[:, :seq_len]
    if ids.size(1) < seq_len:
        pad = torch.full((ids.size(0), seq_len - ids.size(1)), mask_id, dtype=torch.long, device=device)
        ids = torch.cat([ids, pad], dim=1)
    with torch.no_grad():
        z_hat = model.online(ids)
        mse = F.mse_loss(z_hat, z).item()
        cos = F.cosine_similarity(z_hat, z, dim=-1).mean().item()
        recon_ids = model.decoder(z_hat).argmax(dim=-1)
        valid = ids != mask_id
        self_cons = (recon_ids[valid] == ids[valid]).float().mean().item()
    print("\n=== 4) 重编码失真与自洽性 ===")
    print(f"  MSE(Z, E(T)) = {mse:.6f} | cosine = {cos:.4f} | 解码自洽率 = {self_cons*100:.2f}%")

    print("\n评测完成。")


if __name__ == "__main__":
    main()
