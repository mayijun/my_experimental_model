"""合成图文对数据 + 字符词表（离线可用，不依赖 torchvision / PIL / datasets）。

图像是「颜色 × 形状 × 位置」三个可解释因子合成的彩色形状，文字是它们的紧凑描述，
因此天然适合验证跨模态硬对齐（文本 token → 图像、图像 token → 文本）。
"""

import random

import numpy as np
import torch


COLORS = [("red", (1.0, 0.0, 0.0)), ("green", (0.0, 1.0, 0.0)),
          ("blue", (0.0, 0.0, 1.0)), ("gold", (1.0, 0.8, 0.0)),
          ("cyan", (0.0, 1.0, 1.0)), ("pink", (1.0, 0.4, 0.7))]
SHAPES = ["circle", "square", "cross", "tri"]
POSITIONS = [("left", 0.3), ("mid", 0.5), ("right", 0.7)]


def draw_image(color_name, shape, pos_name, size):
    """用 numpy 画一个彩色形状，返回 float32 图像 [3,H,W] ∈ [0,1]。"""
    h = w = size
    img = np.zeros((3, h, w), dtype=np.float32)
    rgb = dict(COLORS)[color_name]
    cx = int(dict(POSITIONS)[pos_name] * w)
    cy = h // 2
    r = max(2, size // 6)
    yy, xx = np.mgrid[0:h, 0:w]
    if shape == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    elif shape == "square":
        mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
    elif shape == "cross":
        mask = (np.abs(xx - cx) <= r // 3) | (np.abs(yy - cy) <= r // 3)
    else:  # tri：顶点在上、底边在下的等腰三角形。
        top, bot = cy - r, cy + r
        half = r * (yy - top) / (bot - top)
        mask = (yy >= top) & (yy <= bot) & (np.abs(xx - cx) <= half)
    for c in range(3):
        img[c, mask] = rgb[c]
    # 弱背景，让位置信息（cx）可观测。
    background = np.random.uniform(0.05, 0.2, size=(h, w)).astype(np.float32)
    img = np.where(mask, img, background[None, :, :])
    return torch.from_numpy(img)


def make_caption(color, shape, pos, text_len):
    """生成与图像配对的文字。短潜长度用紧凑形式，长潜长度用自然语言。"""
    if text_len < 20:
        s = f"{color}{shape}{pos}"
    else:
        s = f"a {color} {shape} at the {pos}"
    return s


class CharTokenizer:
    """字符级词表。保留 0=PAD、1=MASK 两个特殊 token。"""

    def __init__(self, corpus):
        chars = sorted(set(corpus))
        self.itos = ["<pad>", "<mask>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = 0
        self.mask_id = 1
        self.vocab_size = len(self.itos)

    @classmethod
    def from_itos(cls, itos):
        """从已保存的 itos 恢复词表（推理侧加载 checkpoint 用）。"""
        obj = cls("")
        obj.itos = list(itos)
        obj.stoi = {c: i for i, c in enumerate(obj.itos)}
        obj.vocab_size = len(obj.itos)
        return obj

    def encode(self, text, length):
        ids = [self.stoi[c] for c in text[:length] if c in self.stoi]
        return ids + [self.pad_id] * (length - len(ids))

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i not in (self.pad_id, self.mask_id))


def build_dataset(n, image_size, text_len, seed=0):
    """生成 n 个 (图像, 文字) 配对的合成数据集，返回 (images, caption_ids, captions, tokenizer)。"""
    rng = random.Random(seed)
    images, ids_list, captions = [], [], []
    color_names = [c for c, _ in COLORS]
    shape_names = SHAPES
    pos_names = [p for p, _ in POSITIONS]
    for _ in range(n):
        color = rng.choice(color_names)
        shape = rng.choice(shape_names)
        pos = rng.choice(pos_names)
        images.append(draw_image(color, shape, pos, image_size))
        captions.append(make_caption(color, shape, pos, text_len))
    # 词表从全部文字构建。
    tokenizer = CharTokenizer("".join(captions))
    ids_list = [tokenizer.encode(s, text_len) for s in captions]
    images = torch.stack(images)
    caption_ids = torch.tensor(ids_list, dtype=torch.long)
    return images, caption_ids, captions, tokenizer


def synthetic_batch_fn(images, caption_ids, batch_size, device):
    """合成数据：从预生成张量里随机采样一批 (img, txt)。"""
    n = images.size(0)

    def fn():
        idx = torch.randint(0, n, (batch_size,))
        return images[idx].to(device), caption_ids[idx].to(device)

    return fn
