"""训练：损失/目标定义与训练循环。"""

from .losses import (
    cosine_loss,
    forward_losses,
    info_nce,
    mask_patches,
    mask_text,
    maybe_autocast,
    variance_loss,
)

__all__ = [
    "mask_patches",
    "mask_text",
    "cosine_loss",
    "variance_loss",
    "info_nce",
    "maybe_autocast",
    "forward_losses",
]
