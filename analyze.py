"""分析脚本：验证 math.md §4 的推论 1、2。

推论 1：生成文本的熵率满足  H(T) <= L * log2(|V|)。
推论 2：重编码失真  D_T = E || Z - E_theta(T) ||^2 > 0，
        即文本 token 无法无损还原连续潜变量 Z。

用法：
    .venv/bin/python analyze.py --num-samples 256 --stage1-steps 200 --stage2-steps 500
"""

import argparse
import math
import sys
from collections import Counter

import torch
import torch.nn.functional as F

# 让脚本能从任意工作目录导入同目录的模型文件。
sys.path.insert(0, "/home/simon-ma/JEPA")
import flow_matching_jepa as fm  # noqa: E402


# ---------------------------------------------------------------------------
# 数据与词表（与 flow_matching_jepa.main 完全一致）
# ---------------------------------------------------------------------------
def build_data():
    chars = sorted(set(fm.CORPUS))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    mask_id = len(chars)  # 输出侧会被屏蔽，不计入有效词表
    data = torch.tensor([stoi[ch] for ch in fm.CORPUS], dtype=torch.long)
    return chars, stoi, itos, mask_id, data


# ---------------------------------------------------------------------------
# 批量生成：同时返回文本、连续潜变量 Z、token id
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_batch(model, prompts, stoi, itos, mask_id, device, inference_steps):
    model.eval()
    n = len(prompts)
    length = fm.SEQ_LEN
    context_ids = torch.full((n, length), mask_id, dtype=torch.long, device=device)
    for i, prompt in enumerate(prompts):
        for j, ch in enumerate(prompt[:length]):
            context_ids[i, j] = stoi[ch]
    context = model.online(context_ids)

    z = torch.randn(n, length, fm.DIM, device=device)  # 连续潜变量 Z 的起点
    dt = 1.0 / inference_steps
    for step in range(inference_steps):
        t = torch.full((n,), step * dt, device=device)
        z = z + dt * model.flow(z, t, context)  # Euler 积分，终点即生成的 Z

    logits = model.decoder(z)
    logits[..., mask_id] = -torch.inf  # [MASK] 不允许出现在输出
    ids = logits.argmax(dim=-1)  # [n, L]

    # 保留给定前缀（与 fm.generate 一致）
    for i, prompt in enumerate(prompts):
        for j, ch in enumerate(prompt[:length]):
            ids[i, j] = stoi[ch]

    texts = ["".join(itos[idx] for idx in row.tolist()) for row in ids]
    return texts, z, ids


# ---------------------------------------------------------------------------
# 熵率估计：一元熵 + 二元条件熵（后者是更好的熵率估计）
# ---------------------------------------------------------------------------
def entropy_rate(texts):
    flat = [ch for t in texts for ch in t]
    total = len(flat)
    if total == 0:
        return 0.0, 0.0

    unigram = Counter(flat)
    h_unigram = -sum((c / total) * math.log2(c / total) for c in unigram.values())

    # H(X_i | X_{i-1}) = sum_a p(a) * H(X_i | X_{i-1}=a)
    # p(a) 用一元边际估计；条件分布用相邻对统计。
    pair_count = Counter()
    for t in texts:
        for a, b in zip(t, t[1:]):
            pair_count[(a, b)] += 1
    succ_by_prev = {}
    for (a, b), c in pair_count.items():
        succ_by_prev.setdefault(a, Counter())[b] = c

    h_bigram = 0.0
    for a, n_a in unigram.items():
        succ = succ_by_prev.get(a)
        if not succ:
            continue
        s = sum(succ.values())
        h_cond = -sum((v / s) * math.log2(v / s) for v in succ.values())
        h_bigram += (n_a / total) * h_cond  # 权重 p(a)

    return h_unigram, h_bigram


# ---------------------------------------------------------------------------
# 重编码失真：把生成文本 T 重新编码，与产生它的连续潜变量 Z 对比
# ---------------------------------------------------------------------------
@torch.no_grad()
def reencode_distortion(model, z, ids):
    z_hat = model.online(ids)  # \hat Z = E_theta(T)，形状 [n, L, d]
    mse = F.mse_loss(z_hat, z).item()
    cos = F.cosine_similarity(z_hat, z, dim=-1).mean().item()
    # token 自洽率：重编码后再解码，是否还能还原出同一个 token
    recon_ids = model.decoder(z_hat).argmax(dim=-1)
    self_consistency = (recon_ids == ids).float().mean().item()
    return mse, cos, self_consistency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-steps", type=int, default=200)
    parser.add_argument("--stage2-steps", type=int, default=500)
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=256, help="生成样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chars, stoi, itos, mask_id, data = build_data()
    vocab_out = len(chars)  # 输出有效词表（不含 [MASK]）

    model = fm.FlowJEPA(vocab_size=len(chars) + 1).to(device)
    print(f"device: {device} | 输出词表 |V|={vocab_out} | L={fm.SEQ_LEN} | d={fm.DIM}")
    fm.train(model, data, mask_id, device, args.stage1_steps, args.stage2_steps)

    # ---- 生成样本（空前缀 => 全部位置都由 flow 生成）----
    prompts = [""] * args.num_samples
    texts, z, ids = generate_batch(
        model, prompts, stoi, itos, mask_id, device, args.inference_steps
    )

    # 展示几个带前缀的样本，直观判断质量
    print("\n=== 生成样本示例（带前缀）===")
    demo_prompts = ["人工智能", "机器学习", "联合"]
    demo_texts, _, _ = generate_batch(
        model, demo_prompts, stoi, itos, mask_id, device, args.inference_steps
    )
    for p, t in zip(demo_prompts, demo_texts):
        print(f"  提示「{p}」-> {t}")

    # =====================================================================
    # 推论 1：生成文本的熵率
    # =====================================================================
    h_uni, h_bi = entropy_rate(texts)
    seq_entropy_est = fm.SEQ_LEN * h_bi  # 整条序列的估计熵
    upper_per_char = math.log2(vocab_out)
    upper_total = fm.SEQ_LEN * upper_per_char

    # 参考：训练语料的熵率
    corpus_texts = [fm.CORPUS[i : i + fm.SEQ_LEN] for i in range(0, len(fm.CORPUS) - fm.SEQ_LEN, fm.SEQ_LEN)]
    c_uni, c_bi = entropy_rate(corpus_texts)

    print("\n=== 推论 1：生成文本熵率  H(T) <= L * log2|V| ===")
    print(f"  一元熵        H_1  = {h_uni:.4f} bits/char")
    print(f"  二元条件熵    H_2  = {h_bi:.4f} bits/char  （熵率估计）")
    print(f"  估计序列熵    L*H_2 = {seq_entropy_est:.2f} bits")
    print(f"  理论上界      L*log2|V| = {fm.SEQ_LEN}*{upper_per_char:.4f} = {upper_total:.2f} bits")
    print(f"  语料参考      H_2(corpus) = {c_bi:.4f} bits/char")
    ok1 = seq_entropy_est <= upper_total + 1e-6
    print(f"  推论 1 验证：{'PASS ✓' if ok1 else 'FAIL ✗'}  （估计序列熵应不超过上界）")

    # =====================================================================
    # 推论 2：重编码失真 > 0
    # =====================================================================
    mse, cos, acc = reencode_distortion(model, z, ids)
    print("\n=== 推论 2：重编码失真  D_T = E || Z - E_theta(T) ||^2 > 0 ===")
    print(f"  MSE(Z, E_theta(T))         = {mse:.6f}")
    print(f"  cosine(Z, E_theta(T))      = {cos:.4f}  （1 = 完全还原）")
    print(f"  token 自洽率 argmax D(E(T))= {acc*100:.2f}%")
    ok2 = mse > 1e-8
    print(f"  推论 2 验证：{'PASS ✓' if ok2 else 'FAIL ✗'}  （MSE 应严格大于 0，即有损）")

    print("\n=== 小结 ===")
    print(f"  熵率上界比值  H_2 / log2|V| = {h_bi/upper_per_char*100:.1f}%"
          f"  （越接近 100% 说明生成越接近均匀随机）")
    if cos < 0.999:
        print(f"  余弦相似度 {cos:.4f} < 1，证实文本无法无损还原连续潜变量（有损压缩）。")


if __name__ == "__main__":
    main()
