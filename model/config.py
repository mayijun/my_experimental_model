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
    dim: int = 128             # 潜空间 / 模型宽度 d。
    depth: int = 4             # 编码器 & 解码器的 Transformer 层数。
    heads: int = 8             # 注意力头数（需整除 dim）。
    mlp_ratio: float = 4.0     # 前馈隐层宽度 = dim * mlp_ratio。
    ema_decay: float = 0.99    # 目标编码器的 EMA 系数 μ。
    mask_ratio: float = 0.5    # JEPA 掩码比例。
    temperature: float = 0.07  # InfoNCE 温度。
    # 损失权重
    jepa_weight: float = 1.0
    recon_weight: float = 1.0
    cross_weight: float = 0.5   # 硬对齐（双向重建）权重。
    align_weight: float = 0.1   # InfoNCE 软对齐权重。
    var_weight: float = 0.1     # VICReg 方差正则（防塌缩）。

    @property
    def n_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def text_len(self) -> int:
        # 硬对齐要求两个模态共享同一个 Z 形状 [B, L, d]，故文本长度 == patch 数。
        return self.n_patches


PRESETS = {
    "tiny": dict(image_size=32, patch_size=8, dim=64, depth=2, heads=4),
    "small": dict(image_size=32, patch_size=8, dim=128, depth=4, heads=8),
    "base": dict(image_size=64, patch_size=8, dim=256, depth=6, heads=8),
}


def build_config(preset="small", **overrides):
    cfg = MJEPAConfig(**PRESETS.get(preset, {}))
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
