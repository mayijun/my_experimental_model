"""数据准备：语料下载、格式整理、合成/真实图文对与词表。"""

from .synthetic import (
    COLORS,
    POSITIONS,
    SHAPES,
    CharTokenizer,
    build_dataset,
    draw_image,
    make_caption,
    synthetic_batch_fn,
)

__all__ = [
    "COLORS",
    "SHAPES",
    "POSITIONS",
    "draw_image",
    "make_caption",
    "CharTokenizer",
    "build_dataset",
    "synthetic_batch_fn",
]
