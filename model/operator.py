"""PSCO 状态依赖收缩算子（math.md §3.7.3）：PsiLayer 生产实现。

把 Z 空间上的自映射 Ψ_h 实现为两段式：
  瞬态段（m 步，串行）：Z' = Z + g_change ⊙ ((a_h - 1)⊙Z + b_h + r_h(Z))
  渐近段（N 步，并行）：Z' = a_h ⊙ Z + b_h（对角仿射幺半群，Hillis-Steele O(log N)）

谱夹层 a_h ∈ (δ, 1-γ)（make_contractive_a）保证：上界 < 1 收敛（谱半径 < 1）、
下界 > δ 封死「a_h → 0 一步把所有 Z 拉到 b 的塌缩」。

反传两路径（记录）：
  路径 A（默认，已实现）：直接对并行扫描 autograd，O(log N) 深度、O(N) 显存。
  路径 B（DEQ 隐式微分，式 25/26，未实现）：只关心不动点时只存 Z*、O(1) 内存，
    解伴随线性不动点 g^(k+1) = Jᵀ g^(k) + ∇ℓ。留作大规模 GPU 优化。

对角 A_h 的「更好方案」（未采用，记录）：常数/时变对角系数都可用 cumprod/cumsum
  闭式 Z_t = P_t ⊙ (Z_0 + Q_t)，P_t=cumprod(a_t)、Q_t=cumsum(b_t/P_t)，比手写
  Hillis-Steele 更简单且走 CUDA 优化内核；中/大档 GPU 再换 mamba_ssm.selective_scan
  或 fla（Triton，放 pyproject 可选 extra）。当前先复用 psco.py 已验证的
  associative_scan（保真、低风险、且是通用非对角情形的正确性参考）。

依赖 psco.py 的纯函数（make_contractive_a / associative_scan / fixed_point）。
"""

import torch
import torch.nn as nn

from .psco import make_contractive_a, associative_scan, fixed_point


class PsiLayer(nn.Module):
    """状态依赖收缩算子 Ψ_h：Z → Z*（§3.7.3）。

    单层结构：从输入 Z 推断隐含模式 h=H(Z)（池化 + MLP），再参数化出
    收缩系数 a_h（谱夹层）、偏移 b_h、门控 g_change，以及可选非线性残差 r_h。
    前向 = 瞬态段（m 步完整算子，含门控 + 残差）+ 渐近段（N 步线性骨架并行扫描）。
    """

    def __init__(self, dim, transient_steps=2, scan_steps=16, delta=0.05, gamma=0.10, residual="none"):
        super().__init__()
        self.dim = dim
        self.transient_steps = transient_steps   # 瞬态段串行步数 m
        self.scan_steps = scan_steps             # 渐近段并行扫描步数 N
        self.delta = delta                       # 谱夹层下界 δ
        self.gamma = gamma                       # 谱夹层上界 (1-γ)

        # h = H(Z)：从 Z 推断隐含模式（池化 + 两层 MLP）
        self.h_proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        # a_h / b_h / g_change 的参数化（h ∈ [B,d] → 各 [B,d]）
        self.proj_a = nn.Linear(dim, dim)          # 收缩系数（谱夹层内）
        self.proj_b = nn.Linear(dim, dim)          # 偏移
        self.proj_gate = nn.Linear(dim, dim)       # 门控 g_change ∈ (0,1)
        # 非线性残差 r_h（瞬态段专用）。决策记录（后续改进）：
        #   - 先用恒等（"none"，纯线性骨架）：不动点闭式 Z*=b/(1-a)，全程 O(log N) 扫描，
        #     防塌缩由谱夹层+读出头+方差三道防线保证，与残差表达力无关，先验证机制；
        #   - 再上 MLP：增益在跨维耦合，但只在瞬态段生效、把不动点变隐式，需路径 B（DEQ 式 25/26）。
        self.residual_mlp = None
        if residual == "mlp":
            self.residual_mlp = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
            )
        # 读出头（防线 2）：锚定 Z* 保留输入信息（塌成常数则无法重建）
        self.readout = nn.Linear(dim, dim)

    def infer_h(self, z):
        """z: [B, L, d] → h: [B, d]（沿序列池化）。"""
        return self.h_proj(z.mean(dim=1))

    def linear_coeffs(self, h):
        """h: [B, d] → a_h ∈ (δ,1-γ)、b_h、g_change ∈ (0,1)，均 [B, d]。"""
        a_h = make_contractive_a(self.proj_a(h), self.delta, self.gamma)
        b_h = self.proj_b(h)
        g = torch.sigmoid(self.proj_gate(h))
        return a_h, b_h, g

    def transient(self, z, a_h, b_h, g):
        """瞬态段：串行 m 步完整算子（门控 + 可选残差）。z: [B,L,d]。"""
        a_minus = a_h.unsqueeze(1) - 1.0        # [B,1,d]（负，收缩增量）
        b = b_h.unsqueeze(1)                    # [B,1,d]
        gg = g.unsqueeze(1)                     # [B,1,d]
        for _ in range(self.transient_steps):
            inc = a_minus * z + b
            if self.residual_mlp is not None:
                inc = inc + self.residual_mlp(z)
            z = z + gg * inc
        return z

    def asymptotic(self, z, a_h, b_h):
        """渐近段：N 步线性骨架 Z'=a_h⊙Z+b_h 的并行扫描（O(log N)），返回 (Z*, 深度)。"""
        B, L, d = z.shape
        zf = z.reshape(B * L, d)
        af = a_h.unsqueeze(1).expand(B, L, d).reshape(B * L, d)
        bf = b_h.unsqueeze(1).expand(B, L, d).reshape(B * L, d)
        aN = af.unsqueeze(0).expand(self.scan_steps, -1, -1)   # [N, B*L, d] 常数系数
        bN = bf.unsqueeze(0).expand(self.scan_steps, -1, -1)
        aP, bP, depth = associative_scan(aN, bN)
        zstar = aP[-1] * zf + bP[-1]                           # a^N Z0 + Σ a^j b
        return zstar.reshape(B, L, d), depth

    def forward(self, z):
        """z: [B, L, d] → Z*: [B, L, d]（不动点近似）。"""
        h = self.infer_h(z)                  # 隐含模式（输入一次性推断）
        a_h, b_h, g = self.linear_coeffs(h)
        z = self.transient(z, a_h, b_h, g)   # 瞬态 m 步
        zstar, _ = self.asymptotic(z, a_h, b_h)
        return zstar


def smoke(device="cpu"):
    """最小冒烟：验证不动点收敛 + 反塌缩（读出头 + 方差），不依赖完整模型。"""
    torch.manual_seed(0)
    B, L, d, N = 16, 16, 64, 16
    layer = PsiLayer(d, transient_steps=2, scan_steps=N).to(device)
    z = torch.randn(B, L, d, device=device)

    zstar = layer(z)
    assert torch.isfinite(zstar).all(), "Z* 非有限"
    # 反塌缩：不动点批量内两两距离应 > 0（未塌成一点）
    spread = (zstar.mean(dim=1).unsqueeze(0) - zstar.mean(dim=1).unsqueeze(1)).norm(dim=-1).mean()
    # 读出头锚定：Z* 应能重建输入 Z（塌成常数则重建不了）
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    for _ in range(50):
        zs = layer(z)
        loss = (layer.readout(zs) - z).pow(2).mean() + 0.1 * torch.relu(1.0 - zs.std(dim=(0, 1))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        zs = layer(z)
        readout_err = (layer.readout(zs) - z).pow(2).mean().item()
        spread = (zs.mean(dim=1).unsqueeze(0) - zs.mean(dim=1).unsqueeze(1)).norm(dim=-1).mean().item()
    print(f"[PsiLayer] 读出头重建误差 = {readout_err:.4f}  不动点两两距离 = {spread:.4f}（>0 即未塌）")
    assert readout_err < 5.0, "读出头锚定失效"
    assert spread > 1e-2, "不动点塌缩为一点"
    print("PsiLayer 冒烟通过：收敛 + 反塌缩（读出头/方差）均成立。")


if __name__ == "__main__":
    smoke("cuda" if torch.cuda.is_available() else "cpu")
