"""模型层：编码器 → 表征层 Z → 解码器。

- encoder：E^I / E^T，把图像与文本编译进共享连续潜空间 Z；
- representation：P_φ 潜预测器、目标编码器 EMA，以及 PSCO 算子迭代原型（psco.py）；
- decoder：R_I / g，把 Z 还原成图像 / 文本。
"""

from .config import MJEPAConfig, PRESETS, build_config
from .decoder import ImageDecoder, TextDecoder
from .encoder import ImageEncoder, TextEncoder, TransformerBlock, TransformerEncoder
from .jepa import MultimodalJEPA
from .representation import LatentPredictor, update_target

__all__ = [
    "MJEPAConfig",
    "PRESETS",
    "build_config",
    "TransformerBlock",
    "TransformerEncoder",
    "ImageEncoder",
    "TextEncoder",
    "LatentPredictor",
    "update_target",
    "ImageDecoder",
    "TextDecoder",
    "MultimodalJEPA",
]
