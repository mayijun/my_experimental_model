"""Flow Matching（§4.1）最小实现：潜空间速度场 v_ψ + 欧氏 OT 直线基线 + Euler 采样。

§4.1.5/§4.1.6 已论证潜空间流形 M 泛型弯曲，真「最短流」是 M 上的测地线（RFM），
欧氏直线 z_t=(1-t)z_0+t z_1 只是局部一阶近似、全局会离开 M（off-manifold）。
本文件按 §4.1.6 末的务实结论先落地欧氏 FM 基线（回归恒定速度 z_1-z_0）；
测地整流 / RFM（需 exp/log 测地映射与拉回度量 G）留作后续阶段。VelocityField
的时间条件主干不随度量切换改变，升级时仅替换损失与采样入口。
"""

import math

import torch
import torch.nn as nn

from .encoder import TransformerBlock


class SinusoidalTimeEmbedding(nn.Module):
    """时间 t∈[0,1] 的正弦嵌入 → R^{dim}（扩散/流匹配标准做法）。"""

    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(1, half)
        )
        self.register_buffer("freqs", freqs)
        self.dim = dim

    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=self.freqs.device)
        t = t.float()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = t.unsqueeze(1) * self.freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, 2*half]
        if emb.size(-1) < self.dim:
            emb = nn.functional.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


class VelocityField(nn.Module):
    """速度场 v_ψ(z_t, t)（§4.1）：潜空间点 z_t 与时间 t → 速度 [B, L, d]。

    欧氏 OT 直线基线回归恒定速度 z_1-z_0（§4.1.2）。位置混合复用 TransformerBlock；
    时间条件用正弦嵌入加性注入每个位置（保持最小，AdaLN 留作后续可选）。
    """

    def __init__(self, dim, depth, heads, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, z_t, t):
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, dim]
        h = z_t + t_emb
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


def flow_loss(velocity, z_1):
    """无条件 Flow Matching 损失（§4.1.2，欧氏 OT 直线基线）。

    z_0~N(0,I)，t~U[0,1]，z_t=(1-t)z_0+t z_1，回归 v_ψ(z_t,t)→z_1-z_0。
    z_1 应为冻结目标潜变量（§4.2(e)：Flow 阶段 E_ξ 冻结，只训 v_ψ）。
    """
    b, _, _ = z_1.shape
    device = z_1.device
    z_0 = torch.randn_like(z_1)
    t = torch.rand(b, device=device)
    t_exp = t.view(b, 1, 1)
    z_t = (1.0 - t_exp) * z_0 + t_exp * z_1
    v_pred = velocity(z_t, t)
    target = z_1 - z_0
    return (v_pred - target).pow(2).mean()


@torch.no_grad()
def euler_sample(velocity, z_0, n_steps):
    """Euler 采样（§4.1.4）：z_{t+Δt}=z_t+v_ψ(z_t,t)Δt。"""
    device = z_0.device
    b = z_0.size(0)
    z = z_0
    dt = 1.0 / max(1, n_steps)
    for i in range(n_steps):
        t = torch.full((b,), i * dt, device=device)
        z = z + velocity(z, t) * dt
    return z


def smoke():
    """VelocityField 冒烟：验证前向有限、flow_loss 下降、Euler 采样有限。"""
    torch.manual_seed(0)
    dim, depth, heads, L = 64, 2, 4, 16
    vel = VelocityField(dim, depth, heads)
    z_1 = torch.randn(4, L, dim)
    print(f"参数: {sum(p.numel() for p in vel.parameters())/1e3:.2f}K")
    losses = []
    opt = torch.optim.AdamW(vel.parameters(), lr=1e-3)
    for _ in range(20):
        opt.zero_grad()
        loss = flow_loss(vel, z_1)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"flow_loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    z_0 = torch.randn(4, L, dim)
    z_1_hat = euler_sample(vel, z_0, n_steps=8)
    print(f"采样有限: {torch.isfinite(z_1_hat).all().item()} | z 范数 {z_1_hat.norm():.2f}")


if __name__ == "__main__":
    smoke()
