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

from .encoder import SelfAttention, TransformerBlock, _ckpt


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


class AdaLNZero(nn.Module):
    """AdaLN-Zero 调制（DiT）：条件向量 c → 6 组 (scale, shift, gate)，末层零初始化。

    零初始化使 DiT 块初始退化为恒等残差，深网也能稳训（对比加法注入）。
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, c):
        return self.net(c).chunk(6, dim=-1)  # 6×[B, d]


class DiTBlock(nn.Module):
    """DiT 块（AdaLN-Zero）：时间/条件逐 block 逐维度调制 scale/shift，门控残差。

    与加法注入不同，条件信号能精细控制每一层、每一维的语义；残差门控零初始化。
    """

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.adaln = AdaLNZero(dim)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = (m.unsqueeze(1) for m in self.adaln(c))  # 各 [B, 1, d]
        h = self.attn(self.norm1(x) * (1.0 + s1) + b1)
        x = x + g1 * h
        h = self.mlp(self.norm2(x) * (1.0 + s2) + b2)
        x = x + g2 * h
        return x


class VelocityField(nn.Module):
    """速度场 v_ψ(z_t, t)（§4.1）：潜空间点 z_t 与时间 t → 速度 [B, L, d]。

    欧氏 OT 直线基线回归恒定速度 z_1-z_0（§4.1.2）。主干可选两种时间条件：
      - use_adaln=False：加法注入（最小实现，冒烟档）；
      - use_adaln=True：DiT 的 AdaLN-Zero 条件（规模化档，逐 block 调制）。
    """

    def __init__(self, dim, depth, heads, mlp_ratio=4.0, use_adaln=False, grad_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.use_adaln = use_adaln
        self.grad_checkpoint = grad_checkpoint
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.cond_embed = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))  # 条件注入（§4.1.8）
        block = DiTBlock if use_adaln else TransformerBlock
        self.blocks = nn.ModuleList([block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, z_t, t, c=None):
        t_emb = self.time_embed(t)  # [B, dim]
        if c is not None:
            # 条件 c [B,L,d] → 全局池化 → [B,d]，注入时间嵌入（最小条件注入，§4.1.8）
            t_emb = t_emb + self.cond_embed(c.mean(dim=1))
        if self.use_adaln:
            h = z_t
            for blk in self.blocks:
                h = _ckpt(blk, h, t_emb, enabled=self.grad_checkpoint, training=self.training)  # AdaLN-Zero 条件 [B, d]
            return self.head(self.norm(h))
        h = z_t + t_emb.unsqueeze(1)  # 加法注入 [B, 1, d]
        for blk in self.blocks:
            h = _ckpt(blk, h, enabled=self.grad_checkpoint, training=self.training)
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


def conditional_flow_loss(velocity, c, z_1):
    """条件 Flow Matching 损失（§4.1.2 条件变体，§4.1.8）。

    c = 条件潜变量（源模态，冻结），z_1 = 目标潜变量（冻结），z_0~N(0,I)（§4.4 熵源）。
    z_t=(1-t)z_0+t z_1，回归 v_ψ(z_t,t,c)→z_1-z_0。c 经全局池化注入速度场。
    """
    b, _, _ = z_1.shape
    device = z_1.device
    z_0 = torch.randn_like(z_1)
    t = torch.rand(b, device=device)
    t_exp = t.view(b, 1, 1)
    z_t = (1.0 - t_exp) * z_0 + t_exp * z_1
    v_pred = velocity(z_t, t, c=c)
    target = z_1 - z_0
    return (v_pred - target).pow(2).mean()


def projection_components(readout_weight, eps=1e-6):
    """读出层权重 W=J_D [V,d] → 右奇异向量 V_r [d,r]，正交投影 P_z=V_r V_r^T（§4.1.7 软投影）。

    P_z 投影到 im J_D^T（token 区分方向），核方向（token 内自由度）被投影掉。
    当 V>=d 满列秩时 r=d、P_z=I（退化为欧氏）。返回 V_r [d,r]；
    应用 P_z v = V_r(V_r^T v)，O(L·d·r)。冻结解码器时可预计算一次、摊销到每步。
    """
    W = readout_weight.detach().float()
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)  # Vh [min(V,d), d]
    r = int((S > eps * S.max()).sum().item())
    return Vh[:r].T.contiguous()  # [d, r]


def make_projection(readout_weight, eps=1e-6):
    """把读出层权重转成软投影算子 v ↦ P_z v（[B,L,d] → [B,L,d]）。"""
    V_r = projection_components(readout_weight, eps)
    P = V_r @ V_r.t()  # [d,d] 对称幂等
    return lambda v: v @ P


@torch.no_grad()
def euler_sample(velocity, z_0, n_steps, projection=None):
    """Euler 采样（§4.1.4），可选软投影 retraction（§4.1.7）：z_{t+Δt}=z_t+P_z v_ψ(z_t,t)Δt。"""
    device = z_0.device
    b = z_0.size(0)
    z = z_0
    dt = 1.0 / max(1, n_steps)
    for i in range(n_steps):
        t = torch.full((b,), i * dt, device=device)
        v = velocity(z, t)
        if projection is not None:
            v = projection(v)
        z = z + v * dt
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

    # 软投影 retraction（§4.1.7）：验证 P_z 幂等/对称/秩，及投影后采样有限。
    W = torch.randn(30, dim) * 0.1  # 模拟 V<d 的 char 读出层（V=30 < d=64）
    V_r = projection_components(W)
    P = V_r @ V_r.t()
    print(f"软投影: r={V_r.size(1)} | 幂等 {(P @ P - P).abs().max():.2e} | 对称 {(P - P.t()).abs().max():.2e}")
    z_1_hat_p = euler_sample(vel, z_0, n_steps=8, projection=make_projection(W))
    print(f"投影采样有限: {torch.isfinite(z_1_hat_p).all().item()} | z 范数 {z_1_hat_p.norm():.2f}")

    # 条件流（§4.1.8）：c 条件注入，验证损失下降。
    c = torch.randn(4, L, dim)
    cond = []
    opt2 = torch.optim.AdamW(vel.parameters(), lr=1e-3)
    for _ in range(20):
        opt2.zero_grad()
        loss = conditional_flow_loss(vel, c, z_1)
        loss.backward()
        opt2.step()
        cond.append(loss.item())
    print(f"条件 flow_loss: {cond[0]:.3f} -> {cond[-1]:.3f}")

    # DiT 主干（AdaLN-Zero）：验证前向/损失下降/采样有限。
    vel_adaln = VelocityField(dim, depth, heads, use_adaln=True)
    print(f"DiT(AdaLN-Zero) 参数: {sum(p.numel() for p in vel_adaln.parameters())/1e3:.2f}K")
    opt3 = torch.optim.AdamW(vel_adaln.parameters(), lr=1e-3)
    a = [flow_loss(vel_adaln, z_1)]
    for _ in range(19):
        opt3.zero_grad()
        loss = flow_loss(vel_adaln, z_1)
        loss.backward()
        opt3.step()
        a.append(loss.item())
    z_1_hat_d = euler_sample(vel_adaln, z_0, n_steps=8)
    print(f"DiT flow_loss: {a[0]:.3f} -> {a[-1]:.3f} | 采样有限 {torch.isfinite(z_1_hat_d).all().item()}")


if __name__ == "__main__":
    smoke()
