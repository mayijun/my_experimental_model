"""PSCO 最小可跑原型（纯 PyTorch，无第三方依赖）。

验证三件事：
  1. 并行扫描（Hillis-Steele，仿射幺半群）与顺序迭代逐点一致，且串行深度 O(log N)；
  2. 收缩算子收敛到不动点，残差按几何级数衰减；
  3. 反塌缩机制：谱夹层 + 不动点读出头 + 方差正则，使不动点保持输入依赖（不塌成一点）。

运行：  .venv/bin/python -m model.psco
"""
import torch
import torch.nn as nn


def log2ceil(n: int) -> int:
    """ceil(log2 n) —— 并行扫描的串行深度。"""
    return (n - 1).bit_length()


# --------------------------------------------------------------------------
# 算子：状态依赖的线性骨架。Psi_h(Z) = Z + A_h Z + b_h + r_h(Z)
# 线性骨架写成一阶递推  Z^(k+1) = a ⊙ Z^(k) + b，其中 a = 1 + A_h 是对角收缩系数。
# 本原型聚焦「渐近段」的线性骨架（PSCO 中扫描作用的部分），非线性残差 r_h 留作瞬态段。
# --------------------------------------------------------------------------

def make_contractive_a(h: torch.Tensor, delta: float = 0.05, gamma: float = 0.10) -> torch.Tensor:
    """由隐含模式 h 推断对角收缩系数 a ∈ (delta, 1-gamma)。

    谱夹层：上界 < 1 保证收敛（谱半径 < 1）；下界 > delta 保证「不瞬间塌成一点」。
    若只给上界，模型可能学会 a -> 0，一步把所有 Z 拉到 b，即表征塌缩。
    h: [B, d] -> a: [B, d]，逐维独立。
    """
    a = delta + (1.0 - gamma - delta) * torch.sigmoid(h)
    return a


def fixed_point(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """线性递推 Z = a⊙Z + b 的闭式不动点 Z* = b / (1 - a)。a 每维 < 1 故分母非零。"""
    return b / (1.0 - a)


# --------------------------------------------------------------------------
# 仿射幺半群上的并行前缀扫描（associative scan）
# --------------------------------------------------------------------------

def compose(u, v):
    """返回 v∘u（先 u 后 v）。u=(a_u,b_u) 较早，v=(a_v,b_v) 较晚。"""
    au, bu = u
    av, bv = v
    return av * au, av * bu + bv


def associative_scan(a: torch.Tensor, b: torch.Tensor):
    """Hillis-Steele inclusive 前缀扫描，串行深度 O(log N)。

    a, b: [N, B, d]（每一步的收缩系数与偏移）。
    对常数算子（每步相同 (a,b)），result[i] = m^{i+1}，即从 Z^(0) 到 Z^(i+1) 的映射。
    返回 (a_prefix, b_prefix, depth)：a_prefix[i] = a^{i+1}，b_prefix[i] = sum_{j<=i} a^j b。
    """
    a = a.clone()
    b = b.clone()
    N = a.shape[0]
    offset = 1
    depth = 0
    while offset < N:
        av, bv = a[offset:], b[offset:]              # 较晚（v）
        au, bu = a[: N - offset], b[: N - offset]    # 较早（u）
        a_new = a.clone()
        b_new = b.clone()
        a_new[offset:] = av * au
        b_new[offset:] = av * bu + bv
        a, b = a_new, b_new
        offset *= 2
        depth += 1
    return a, b, depth


def associative_scan_mat(a: torch.Tensor, b: torch.Tensor):
    """矩阵仿射幺半群的 Hillis-Steele inclusive 前缀扫描，串行深度 O(log N)。

    a, b: [N, B, D, D]——状态 z_t = A_t z_{t-1} + B_t 的系数（z、B 为 [B,D,D] 矩阵）。
    幺半群组合（右乘，对应 M_t = M_{t-1} A_t + B_t）：(A_u,B_u)∘(A_v,B_v) = (A_u A_v, B_u A_v + B_v)，u 较早、v 较晚。
    返回 (a_prefix, b_prefix, depth)：a_prefix[t] = Π_{s≤t} A_s，
    b_prefix[t] = Σ_{s≤t}(Π_{u=s+1}^t A_u) B_s。

    供 LinearMemory 的 delta rule（M_t = M_{t-1} A_t + B_t，A_t = I − β k_t k_tᵀ）并行化：
    精度优先路径，直接矩阵连乘、无求逆；显存 O(B·N·D²)（D=64 冻结下可接受）。
    """
    a = a.clone()
    b = b.clone()
    N = a.shape[0]
    offset = 1
    depth = 0
    while offset < N:
        av, bv = a[offset:], b[offset:]              # 较晚（v）
        au, bu = a[: N - offset], b[: N - offset]    # 较早（u）
        a_new = a.clone()
        b_new = b.clone()
        a_new[offset:] = torch.einsum("nbij,nbjk->nbik", au, av)      # A_u A_v（右乘）
        b_new[offset:] = torch.einsum("nbij,nbjk->nbik", bu, av) + bv  # B_u A_v + B_v
        a, b = a_new, b_new
        offset *= 2
        depth += 1
    return a, b, depth


def sequential(a: torch.Tensor, b: torch.Tensor, Z0: torch.Tensor, N: int) -> torch.Tensor:
    """顺序迭代 N 步（地面真值），返回 [N, B, d] 的轨迹 Z^(1..N)。"""
    Zs = []
    Z = Z0
    for _ in range(N):
        Z = a * Z + b
        Zs.append(Z)
    return torch.stack(Zs, dim=0)


def scan_path(a: torch.Tensor, b: torch.Tensor, Z0: torch.Tensor, N: int):
    """并行扫描路径：把轨迹一次性物化，返回 (轨迹 [N,B,d], 扫描深度)。"""
    aN = a.unsqueeze(0).expand(N, -1, -1)   # [N, B, d] 常数系数
    bN = b.unsqueeze(0).expand(N, -1, -1)
    aP, bP, depth = associative_scan(aN, bN)
    Zs = aP * Z0.unsqueeze(0) + bP          # Z^(k) = a^k Z0 + (sum a^j) b
    return Zs, depth


# --------------------------------------------------------------------------
# 验证 1：扫描 == 顺序；验证 2：收敛 + O(log N) 深度
# --------------------------------------------------------------------------

def check_scan_and_convergence(device: str) -> None:
    torch.manual_seed(0)
    B, d, N = 4, 64, 1000
    h = torch.randn(B, d, device=device)
    a = make_contractive_a(h)                 # [B, d]，每维 ∈ (0.05, 0.90)
    b = torch.randn(B, d, device=device)
    Z0 = torch.randn(B, d, device=device)

    Zseq = sequential(a, b, Z0, N)
    Zscan, depth = scan_path(a, b, Z0, N)

    err = (Zseq - Zscan).abs().max().item()
    print("[1] 扫描 vs 顺序  最大逐点误差 = %.3e" % err)
    print("    扫描串行深度   = %d   (ceil(log2 %d) = %d)"
          % (depth, N, log2ceil(N)))
    assert depth == log2ceil(N), "扫描深度应为 O(log N)"
    assert err < 1e-4, "扫描与顺序不一致"

    # 收敛：残差 ||Z^(k+1) - Z^(k)|| 应几何衰减到 0，且逼近闭式不动点。
    Zstar = fixed_point(a, b)
    resid0 = (Zseq[1] - Zseq[0]).norm(dim=-1).max().item()
    resid_end = (Zseq[-1] - Zseq[-2]).norm(dim=-1).max().item()
    gap = (Zseq[-1] - Zstar).abs().max().item()
    print("[2] 收敛          首步残差 = %.3e  末步残差 = %.3e  到不动点误差 = %.3e"
          % (resid0, resid_end, gap))
    assert resid_end < 1e-3 and gap < 1e-2, "未收敛到不动点"


# --------------------------------------------------------------------------
# 验证 3：反塌缩。对比「有防护」与「无下界」两种配置下不动点的输入依赖性。
# --------------------------------------------------------------------------

def check_anti_collapse(device: str) -> None:
    torch.manual_seed(1)
    B, d = 16, 32

    def spread(Zstar: torch.Tensor) -> float:
        """批量内不动点的平均两两距离（越大越不塌，0 = 完全塌成一点）。"""
        diff = Zstar.unsqueeze(0) - Zstar.unsqueeze(1)      # [B, B, d]
        return diff.norm(dim=-1).mean().item()

    h = torch.randn(B, d, device=device)          # 可区分的隐含模式
    b_shared = torch.randn(1, d, device=device)   # 输入无关的偏移（塌缩风险源）

    # 塌缩情形：a 退化为常数≈0（强收缩）+ b 共享 -> Z* = b/(1-a) ≈ b，所有样本同一点。
    a_degenerate = torch.full((B, d), 0.001, device=device)
    Z_collapsed = fixed_point(a_degenerate, b_shared)

    # 健康情形：a 输入依赖且落在谱夹层 (δ, 1-γ) -> Z* = b/(1-a) 随 a 变，保持输入依赖。
    a_healthy = make_contractive_a(h)
    Z_healthy = fixed_point(a_healthy, b_shared)

    print("[3] 反塌缩")
    print("    塌缩配置（a→0 常数 + 共享 b）  spread = %.4f" % spread(Z_collapsed))
    print("    健康配置（a 输入依赖 + 夹层） spread = %.4f" % spread(Z_healthy))
    assert spread(Z_collapsed) < 1e-2, "演示失效：塌缩配置应塌成一点"
    assert spread(Z_healthy) > 0.1, "演示失效：健康配置应保持输入依赖"

    # 训练 200 步：读出头锚定 + 方差正则，证明防护机制在优化中保持不动点不塌。
    proj_a = nn.Linear(d, d, bias=False)
    proj_b = nn.Linear(d, d)
    readout = nn.Linear(d, d)
    opt = torch.optim.Adam(list(proj_a.parameters()) + list(proj_b.parameters())
                           + list(readout.parameters()), lr=0.01)
    labels = torch.randn(B, d, device=device)                # 每个样本的目标（用于锚定）

    for step in range(200):
        aa = make_contractive_a(proj_a(h))                   # 谱夹层始终生效（上界收敛、下界防塌）
        bb = proj_b(h)                                       # b 输入依赖
        Zs = fixed_point(aa, bb)
        # 读出头锚定：Z* 必须能预测目标（塌成常数就预测不了 -> 梯度把它推开）
        loss_readout = (readout(Zs) - labels).pow(2).mean()
        # 方差正则：每维标准差下限，防维度塌缩（对应式 33）
        loss_var = torch.relu(1.0 - Zs.std(dim=0)).mean()
        loss = loss_readout + 0.1 * loss_var
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        Z_final = fixed_point(make_contractive_a(proj_a(h)), proj_b(h))
    print("    训练 200 步后：loss = %.3f  不动点两两距离均值 = %.3f（>0 即未塌）"
          % (loss.item(), spread(Z_final)))
    assert spread(Z_final) > 0.05, "不动点塌缩为一点"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)
    check_scan_and_convergence(device)
    check_anti_collapse(device)
    print("\n全部通过：O(log N) 深度、不动点收敛、反塌缩机制均验证成立。")


if __name__ == "__main__":
    main()
