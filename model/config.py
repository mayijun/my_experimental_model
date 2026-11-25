"""模型超参：规模可调控、可扩展。

修改任意字段即可在「冒烟验证」与「大规模训练」间伸缩；PRESETS 只是常用起点。
"""

from dataclasses import dataclass


@dataclass
class MJEPAConfig:
    """多模态 JEPA 的超参。"""

    image_size: int = 32       # 输入图像边长（H == W）。
    patch_size: int = 8        # 每个 patch 的边长。
    image_channels: int = 3    # RGB。
    dim: int = 128             # 潜空间维度 d（§2.2.7 的语义瓶颈，编码器输出 Z∈R^{L×d}）。
    depth: int = 4             # 编码器 & 解码器的层数。
    heads: int = 8             # 注意力头数（需整除主干宽度 hidden_dim）。
    mlp_ratio: float = 4.0     # 前馈隐层宽度 = 主干宽度 * mlp_ratio。
    hidden_dim: int = 0        # 主干宽度 d_hidden；0 表示与 d 绑定（冒烟档，§2.2.7）。
    backbone: str = "hybrid"   # 主干选型：transformer / mamba / hybrid（§2.2.5）。
    attn_layers: int = 2       # hybrid 下前 attn_layers 层为注意力，其余为 SSM。
    share_trunk: bool = False   # 图文编码器共享同一条主干（§2.8 共享 E；省 ~44% 静态显存）
    embed_factor: int = 0       # 文本词嵌入低秩分解维 F（ALBERT 式 V×H → V×F + F×H；0=关闭，§2.2.7）
    # 图像前端：patch（单层卷积，冒烟）/ siglip2（LLaDA-V 前端：SigLIP-2 + MLP 投影 → 连续 Z）
    image_frontend: str = "patch"
    siglip2_model: str = "google/siglip2-base-patch16-224"
    freeze_vision: bool = True  # 冻结 SigLIP-2 视觉编码器（LLaDA-V 默认冻结，只训 MLP 投影 + 主干）。
    # token 层 ρ 残差压缩（§2.6 创新编码）：按冗余度 ρ=‖v−Mk‖² 丢冗余、mask 填充保 L。
    # 每个模态一个私有记忆 M（接口预留多模态），冒烟默认关闭。
    rho_compress: bool = False  # 是否启用 token 层 ρ 压缩。
    rho_head_dim: int = 64      # 记忆 M 的 k/v 维度（冻结，不随 hidden_dim 缩放，§2.6）。
    rho_beta: float = 0.1       # delta rule 写入率。
    rho_eps: float = 0.0        # 绝对阈值（默认 0 禁用，改相对阈值 rho_eps_z）。
    rho_eps_z: float = 0.0      # 相对阈值 z-score：ε_ρ = 批内均值 + z·σ（ρ 绝对量级随初始化/head_dim 变，须相对标定）。
    rho_eps_keep: float = 0.02  # ε 强制采样比例（防「丢弃→学不到」自我强化退化）。
    # 比特门控 b_t = clamp(a·log(1/p(ρ)) + c·H(x|rest), b_min, b_max)（§2.6 式 665）
    rho_b_a: float = 1.0        # surprisal 系数（ρ 的分布罕见度）。
    rho_b_c: float = 0.5        # 条件熵代理系数（ρ 作 H 上界代理，§2.6「测差值别估熵」）。
    rho_b_min: float = 0.0      # b_t clamp 下界。
    rho_b_max: float = 4.0      # b_t clamp 上界。
    rho_rate_weight: float = 0.0  # 码率正则权重（§2.6 式 665）：R=mean(b_t)，与重建失真构成 R+λD。
    ema_decay: float = 0.99    # 目标编码器的 EMA 系数 μ。
    mask_ratio: float = 0.5    # JEPA 掩码比例。
    temperature: float = 0.07  # InfoNCE 温度。
    # 损失权重
    jepa_weight: float = 1.0
    recon_weight: float = 1.0
    cross_weight: float = 0.5   # 硬对齐（双向重建）权重。
    align_weight: float = 0.1   # InfoNCE 软对齐权重。
    var_weight: float = 0.1     # VICReg 方差正则（防塌缩）。

    # —— 对齐方法（§3.4）：跨模态软对齐信号源，默认 BYOL（无负样本、无 batch 依赖）——
    alignment_method: str = "byol"   # infonce（软对齐兜底）/ byol（默认）/ dino
    byol_proj_dim: int = 0           # BYOL/DINO 投影输出维；0 表示 = dim
    byol_hidden_dim: int = 0         # 投影/预测头隐层维；0 表示 = 4*dim
    byol_ema_decay: float = 0.99     # BYOL 目标投影头 EMA
    dino_teacher_temp: float = 0.04  # DINO 教师温度（锐化，越低越尖）
    dino_student_temp: float = 0.1   # DINO 学生温度
    dino_center_momentum: float = 0.9  # DINO centering 动量

    # —— 交叉掩码（§3.2 式 18）：本模态剩余 + 其他模态联合预测被掩部分，防模态坍缩 ——
    cross_mask: bool = True          # 启用跨模态交叉掩码
    cross_mask_weight: float = 1.0   # 交叉掩码损失权重
    cross_mask_ratio_start: float = 0.2  # ρ_mask(t) 起始掩码比
    cross_mask_ratio_end: float = 0.8    # ρ_mask(t) 终点掩码比

    # —— 渐进锚定课程 PAC（§3.4 式 21）：锚模态先定 Z，其余向锚对齐 ——
    pac_enabled: bool = False        # 启用 PAC（冒烟档默认关，无多模态收益）
    anchor_modality: str = "image"   # 锚模态：image / text
    pac_anchor_p0: float = 1.0       # 锚概率初值
    pac_anchor_pinf: float = 0.5     # 锚概率终值（1/K，K=2）
    pac_tau: float = 0.1             # sigmoid 温度（归一化步数后）
    pac_t0: float = 0.2              # 锚概率开始下降的归一化时刻
    pac_new_lr_scale: float = 3.0    # 新模态 lr 倍数（锚低 lr、新高 lr）
    pac_z_lr_scale: float = 0.1      # S2 起 Z/共享层 lr 缩放

    # —— PSCO 状态依赖收缩算子（§3.7.3）：Z 空间自映射 Ψ_h，谱夹层防塌缩 ——
    psi_enabled: bool = True         # 启用 PsiLayer（§3.7.3 算子层，防塌缩第三层，已接入默认训练）
    psi_transient_steps: int = 2     # 瞬态段串行步数 m（含门控 + 残差）
    psi_scan_steps: int = 16         # 渐近段并行扫描步数 N
    psi_delta: float = 0.05          # 谱夹层下界 δ
    psi_gamma: float = 0.10          # 谱夹层上界 (1-γ)
    psi_residual: str = "none"       # 非线性残差 r_h：none（线性骨架）/ mlp
    psi_readout_weight: float = 0.1  # 读出头锚定损失权重（防线 2）
    psi_var_weight: float = 0.1      # 方差正则权重（防线 3）
    psi_stop_weight: float = 0.1     # §3.7 L_stop 不动点/停止正则：Ψ_h(Z_clean)≈Z_clean（读端退化为恒等）
    psi_consist_weight: float = 0.1  # §3.7 L_consist 路径无关一致性：Ψ_h(Z_ctx)≈Z_tgt（改端变换到目标）
    psi_gen_weight: float = 0.1      # §3.9 生成 = D(Ψ(Z))：解码算子不动点 Z*=Ψ(Z_ctx) 重建目标（接通「算子→不动点→解码→生成」）

    # —— Flow Matching（§4.1）：潜空间速度场 v_ψ，欧氏 OT 直线基线（测地整流/RFM 后续阶段）——
    flow_enabled: bool = True        # 启用速度场（§4.1.2 无条件生成，z_0=噪声 → z_1=冻结目标 Z）
    flow_depth: int = 2              # v_ψ 主干层数（位置混合）
    flow_heads: int = 4              # v_ψ 注意力头数（需整除 d）
    flow_steps: int = 8              # Euler 采样步数（§4.1.4）
    flow_weight: float = 0.1         # 流匹配损失权重（§4.1.2）
    flow_cond_enabled: bool = True   # 条件流（§4.1.8）：text→image / image→text 跨模态，z_0=噪声、c=源模态 Z
    flow_cond_weight: float = 0.1    # 条件流损失权重
    flow_adaln: bool = False         # 速度场主干：True=DiT AdaLN-Zero 条件（规模化档），False=加法注入（冒烟档）

    @property
    def vision_patch(self) -> int:
        """图像前端的 patch 边长：siglip2 用 16（SigLIP-2），patch 前端用 patch_size。"""
        return 16 if self.image_frontend == "siglip2" else self.patch_size

    @property
    def n_patches(self) -> int:
        return (self.image_size // self.vision_patch) ** 2

    @property
    def width(self) -> int:
        """主干宽度 d_hidden；冒烟档（hidden_dim=0）与潜空间 d 绑定。"""
        return self.hidden_dim or self.dim

    @property
    def text_len(self) -> int:
        # 硬对齐要求两个模态共享同一个 Z 形状 [B, L, d]，故文本长度 == patch 数。
        return self.n_patches


PRESETS = {
    # 冒烟三档：d 与主干宽度绑定（hidden_dim=0），CPU 可跑（§2.2.7）。
    "tiny": dict(image_size=32, patch_size=8, dim=64, depth=2, heads=4),
    "small": dict(image_size=32, patch_size=8, dim=128, depth=4, heads=8),
    "base": dict(image_size=64, patch_size=8, dim=256, depth=6, heads=8),
    # 中型：主干宽、潜空间窄，d = d_hidden / 8，词表用 LLaDA（§2.2.7 规模表）。
    "0.3b": dict(image_size=256, patch_size=8, hidden_dim=1024, dim=128, depth=24, heads=16, share_trunk=True),
    "1b": dict(image_size=256, patch_size=8, hidden_dim=2048, dim=256, depth=24, heads=16, share_trunk=True),
    "3b": dict(image_size=512, patch_size=8, hidden_dim=3072, dim=384, depth=32, heads=32, share_trunk=True),
    "7b": dict(image_size=512, patch_size=8, hidden_dim=4096, dim=512, depth=32, heads=32, share_trunk=True),
}


def build_config(preset="0.3b", **overrides):
    cfg = MJEPAConfig(**PRESETS.get(preset, {}))
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
