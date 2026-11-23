"""多模态 JEPA：图像 + 文本共享潜空间 Z 的「编码 → 对齐/推理 → 解码」模型。

对应 math.md 的三阶段（不含自反馈修正，§5 不实现）：

1. 编码（§2、§3.2）：图像 ViT 编码器 E^I 与文本 Transformer 编码器 E^T 把各自模态
   编译进同一个连续潜空间 Z = [B, L, d]（L 个潜 token）。
2. 对齐 / 推理（§3）：
   - 硬对齐（§3.2 式 17）：图像/文本共用同一坐标系，靠「共享解码器 + 双向重建」保证
     图像 token 能经文本解码器 g 还原文本、文本 token 能经图像解释器 R_I 还原图像；
   - 软对齐对照（§3.4）：InfoNCE 拉近图文 [CLS] 投影；
   - 推理（§2.2 JEPA）：在线编码器读掩码上下文，预测器 P_φ 在 Z 内预测目标编码器
     （EMA、stop-gradient）的干净潜变量。
3. 解码（§3.2）：图像解释器 R_I（Z → 像素）与文本解释器 g（Z → 词表 logits）。

模型规模完全由 MJEPAConfig 控制（width/深度/头数/patch/分辨率），预设 tiny/small/base，
可从 CPU 冒烟验证一路放大到 GPU 大规模训练。本文件只用 torch + numpy，不依赖
torchvision / transformers / PIL。

用法（CPU 冒烟）：
    .venv/bin/python multimodal_jepa.py --preset tiny --steps 150
用法（缩放）：
    .venv/bin/python multimodal_jepa.py --preset base --dim 512 --depth 12 --steps 2000
"""

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


# ---------------------------------------------------------------------------
# 配置：模型规模可调控、可扩展
# ---------------------------------------------------------------------------


@dataclass
class MJEPAConfig:
    """多模态 JEPA 的超参。修改任意字段即可在「冒烟验证」与「大规模训练」间伸缩。"""

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


# ---------------------------------------------------------------------------
# 基础 Transformer 模块
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-LN 自注意力块：残差 + 多头注意力 + 残差 + MLP。"""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """堆叠若干 TransformerBlock，末尾 LayerNorm。"""

    def __init__(self, dim, heads, depth, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 阶段一 · 编码：图像 / 文本 → 共享潜空间 Z
# ---------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """E^I：把图像 [B,C,H,W] 编译成 patch token 序列 + [CLS]。

    掩码时把被掩 patch 替换为可学习 mask_token，供 JEPA 推理使用。
    """

    def __init__(self, image_size, patch_size, channels, dim, heads, depth, mlp_ratio=4.0):
        super().__init__()
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.encoder = TransformerEncoder(dim, heads, depth, mlp_ratio)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):
        batch = x.size(0)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B,N,d]
        if mask is not None:  # mask: [B,N] bool，True 处替换为 mask_token。
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        cls = self.cls_token.expand(batch, -1, -1)
        h = torch.cat([cls, tokens], dim=1) + self.pos_embed
        h = self.encoder(h)
        return h[:, 0], h[:, 1:]  # (cls [B,d], tokens [B,N,d])


class TextEncoder(nn.Module):
    """E^T：把字符 id 序列 [B,L] 编译成 token 序列 + [CLS]。

    掩码由「把部分字符替换为 mask_id」实现，与 BERT 一致。
    """

    def __init__(self, vocab_size, max_len, dim, heads, depth, mlp_ratio=4.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, dim))
        self.encoder = TransformerEncoder(dim, heads, depth, mlp_ratio)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, ids):
        batch = ids.size(0)
        x = self.embedding(ids)
        cls = self.cls_token.expand(batch, -1, -1)
        h = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.size(1) + 1]
        h = self.encoder(h)
        return h[:, 0], h[:, 1:]


class LatentPredictor(nn.Module):
    """P_φ：在 Z 内把掩码上下文的潜变量预测成干净目标潜变量（逐位置共享 MLP）。"""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, context):
        return self.net(context)  # [B,L,d] -> [B,L,d]


# ---------------------------------------------------------------------------
# 阶段三 · 解码：共享 Z → 图像 / 文本
# ---------------------------------------------------------------------------


class ImageDecoder(nn.Module):
    """R_I：把潜 token [B,L,d] 还原成图像 [B,C,H,W]（共享：图像 token 与文本 token 都走它）。"""

    def __init__(self, image_size, patch_size, channels, dim, heads, depth, mlp_ratio=4.0):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.n_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        self.decoder = TransformerEncoder(dim, heads, depth, mlp_ratio)
        self.head = nn.Linear(dim, patch_size * patch_size * channels)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z):
        h = self.decoder(z + self.pos_embed[:, : z.size(1)])
        patches = self.head(h)  # [B,L,p*p*C]
        batch = z.size(0)
        p = self.patch_size
        c = self.channels
        hp = self.image_size // p
        # 把 patch 折叠回图像：像素混洗的逆过程。
        return patches.reshape(batch, hp, hp, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(
            batch, c, self.image_size, self.image_size
        )


class TextDecoder(nn.Module):
    """g：把潜 token [B,L,d] 还原成字符 logits [B,L,V]（共享：文本 token 与图像 token 都走它）。"""

    def __init__(self, vocab_size, max_len, dim, heads, depth, mlp_ratio=4.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.decoder = TransformerEncoder(dim, heads, depth, mlp_ratio)
        self.head = nn.Linear(dim, vocab_size)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z):
        h = self.decoder(z + self.pos_embed[:, : z.size(1)])
        return self.head(h)  # [B,L,V]


# ---------------------------------------------------------------------------
# 组装：MultimodalJEPA
# ---------------------------------------------------------------------------


class MultimodalJEPA(nn.Module):
    """图像 + 文本共享 Z 的完整模型：双在线/目标编码器 + 预测器 + 共享双解码器。"""

    def __init__(self, cfg: MJEPAConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        length = cfg.text_len

        # —— 编码（在线，接收梯度）——
        self.image_online = ImageEncoder(
            cfg.image_size, cfg.patch_size, cfg.image_channels, cfg.dim,
            cfg.heads, cfg.depth, cfg.mlp_ratio,
        )
        self.text_online = TextEncoder(
            vocab_size, length, cfg.dim, cfg.heads, cfg.depth, cfg.mlp_ratio,
        )

        # —— 编码（目标，EMA、stop-gradient）——
        self.image_target = ImageEncoder(
            cfg.image_size, cfg.patch_size, cfg.image_channels, cfg.dim,
            cfg.heads, cfg.depth, cfg.mlp_ratio,
        )
        self.text_target = TextEncoder(
            vocab_size, length, cfg.dim, cfg.heads, cfg.depth, cfg.mlp_ratio,
        )
        self.image_target.load_state_dict(self.image_online.state_dict())
        self.text_target.load_state_dict(self.text_online.state_dict())
        for module in (self.image_target, self.text_target):
            for p in module.parameters():
                p.requires_grad_(False)

        # —— 推理：JEPA 潜预测器（模态各一）——
        self.image_predictor = LatentPredictor(cfg.dim, cfg.mlp_ratio)
        self.text_predictor = LatentPredictor(cfg.dim, cfg.mlp_ratio)

        # —— 解码（共享，双向可还原）——
        self.image_decoder = ImageDecoder(
            cfg.image_size, cfg.patch_size, cfg.image_channels, cfg.dim,
            cfg.heads, cfg.depth, cfg.mlp_ratio,
        )
        self.text_decoder = TextDecoder(
            vocab_size, length, cfg.dim, cfg.heads, cfg.depth, cfg.mlp_ratio,
        )

        # —— 软对齐投影头（InfoNCE 用）——
        self.proj = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim))


# ---------------------------------------------------------------------------
# 目标编码器 EMA 更新
# ---------------------------------------------------------------------------


@torch.no_grad()
def update_target(model, decay):
    """ξ ← μξ + (1−μ)θ，让 JEPA 目标缓慢移动（stop-gradient 防表征塌缩）。"""
    pairs = [
        (model.image_online, model.image_target),
        (model.text_online, model.text_target),
    ]
    for online, target in pairs:
        for o, t in zip(online.parameters(), target.parameters()):
            t.mul_(decay).add_(o, alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# 掩码 / 损失工具
# ---------------------------------------------------------------------------


def mask_patches(batch, n_patches, ratio, device):
    """随机遮住一部分 patch，返回 [B,N] 的 bool 掩码。"""
    return torch.rand(batch, n_patches, device=device) < ratio


def mask_text(ids, mask_id, ratio, device):
    """随机把一部分字符替换为 mask_id。"""
    out = ids.clone()
    out[torch.rand(ids.shape, device=device) < ratio] = mask_id
    return out


def cosine_loss(a, b):
    """潜空间余弦距离，JEPA 预测目标。"""
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def variance_loss(z):
    """VICReg 方差约束：惩罚标准差 < 1 的维度，抑制特征维度塌缩。"""
    std = z.reshape(-1, z.size(-1)).std(dim=0, unbiased=False)
    return F.relu(1.0 - std).mean()


def info_nce(zi, zt, temperature):
    """对称 InfoNCE：图像 [CLS] 与文本 [CLS] 的软对齐。"""
    zi = F.normalize(zi, dim=-1)
    zt = F.normalize(zt, dim=-1)
    logits = zi @ zt.t() / temperature  # [B,B]
    labels = torch.arange(zi.size(0), device=zi.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def maybe_autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------------------------------------------------------------------------
# 前向 + 损失
# ---------------------------------------------------------------------------


def forward_losses(model, img, txt, pad_id, mask_id):
    """跑通「编码 → 对齐/推理 → 解码」，返回 (总损失, 分项损失 dict)。"""
    cfg = model.cfg
    batch = img.size(0)
    device = img.device

    # —— 编码（在线，干净 + 掩码）——
    img_cls, img_tokens = model.image_online(img)
    txt_cls, txt_tokens = model.text_online(txt)

    img_mask = mask_patches(batch, cfg.n_patches, cfg.mask_ratio, device)
    txt_masked = mask_text(txt, mask_id, cfg.mask_ratio, device)
    _, img_ctx = model.image_online(img, mask=img_mask)
    _, txt_ctx = model.text_online(txt_masked)

    # —— 目标（EMA、无梯度）——
    with torch.no_grad():
        _, img_tgt = model.image_target(img)
        _, txt_tgt = model.text_target(txt)

    # —— 推理：JEPA 潜预测 ——
    img_pred = model.image_predictor(img_ctx)
    txt_pred = model.text_predictor(txt_ctx)

    # —— 解码：自重建 ——
    img_self = model.image_decoder(img_tokens)
    txt_self = model.text_decoder(txt_tokens)

    # —— 解码：硬对齐（双向重建，穿过同一个 Z）——
    img_cross = model.image_decoder(txt_tokens)   # 文本 token → 图像
    txt_cross = model.text_decoder(img_tokens)    # 图像 token → 文本

    # —— 软对齐：InfoNCE ——
    zi = model.proj(img_cls)
    zt = model.proj(txt_cls)

    jepa = cosine_loss(img_pred, img_tgt) + cosine_loss(txt_pred, txt_tgt)
    recon_img_self = F.mse_loss(img_self, img)
    recon_txt_self = F.cross_entropy(
        txt_self.reshape(-1, txt_self.size(-1)), txt.reshape(-1), ignore_index=pad_id
    )
    recon_img_cross = F.mse_loss(img_cross, img)
    recon_txt_cross = F.cross_entropy(
        txt_cross.reshape(-1, txt_cross.size(-1)), txt.reshape(-1), ignore_index=pad_id
    )
    align = info_nce(zi, zt, cfg.temperature)
    var = variance_loss(img_tokens) + variance_loss(txt_tokens)

    total = (
        cfg.jepa_weight * jepa
        + cfg.recon_weight * (recon_img_self + recon_txt_self)
        + cfg.cross_weight * (recon_img_cross + recon_txt_cross)
        + cfg.align_weight * align
        + cfg.var_weight * var
    )
    parts = {
        "jepa": jepa, "recon_img": recon_img_self, "recon_txt": recon_txt_self,
        "cross_img": recon_img_cross, "cross_txt": recon_txt_cross,
        "align": align, "var": var,
    }
    return total, parts


# ---------------------------------------------------------------------------
# 合成图文对数据 + 字符词表（无 torchvision/PIL）
# ---------------------------------------------------------------------------

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

    def encode(self, text, length):
        ids = [self.stoi[c] for c in text[:length]]
        return ids + [self.pad_id] * (length - len(ids))

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i not in (self.pad_id, self.mask_id))


def build_dataset(n, image_size, text_len, seed=0):
    """生成 n 个 (图像, 文字) 配对的合成数据集，返回 (images, caption_ids, captions)。"""
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


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def train(model, images, caption_ids, pad_id, mask_id, device, steps, batch_size, lr):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n = images.size(0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,))
        img = images[idx].to(device)
        txt = caption_ids[idx].to(device)
        with maybe_autocast(device):
            loss, parts = forward_losses(model, img, txt, pad_id, mask_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_target(model, model.cfg.ema_decay)

        if step == 1 or step % 25 == 0 or step == steps:
            detail = " ".join(f"{k}={v.item():.3f}" for k, v in parts.items())
            print(f"step {step:4d} | loss {loss.item():.4f} | {detail}")
    return model


# ---------------------------------------------------------------------------
# 评测：跨模态检索 + 双向重建
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, images, caption_ids, captions, tokenizer, device, num=16):
    model.eval()
    idx = torch.randperm(images.size(0))[:num]
    img = images[idx].to(device)
    txt = caption_ids[idx].to(device)

    # —— 检索：图像 [CLS] ↔ 文本 [CLS] 余弦最近邻 ——
    img_cls, img_tokens = model.image_online(img)
    txt_cls, txt_tokens = model.text_online(txt)
    zi = F.normalize(model.proj(img_cls), dim=-1)
    zt = F.normalize(model.proj(txt_cls), dim=-1)
    sim = zi @ zt.t()  # [num,num]
    i2t = (sim.argmax(dim=1) == torch.arange(num, device=device)).float().mean().item()
    t2i = (sim.argmax(dim=0) == torch.arange(num, device=device)).float().mean().item()

    # —— 双向重建 ——
    img_self = model.image_decoder(img_tokens)
    txt_self = model.text_decoder(txt_tokens).argmax(-1)
    img_cross = model.image_decoder(txt_tokens)
    txt_cross = model.text_decoder(img_tokens).argmax(-1)

    print(f"\n=== 跨模态检索（{num} 对）===")
    print(f"  图像→文本 top-1 = {i2t*100:.1f}% | 文本→图像 top-1 = {t2i*100:.1f}%")
    print(f"\n=== 重建误差（越低越好）===")
    print(f"  图像自重建 MSE = {F.mse_loss(img_self, img).item():.4f} | "
          f"文本→图像 MSE = {F.mse_loss(img_cross, img).item():.4f}")
    print(f"\n=== 样例（真值 / 文本自重建 / 图像→文本）===")
    for i in range(min(4, num)):
        truth = captions[idx[i].item()]
        self_dec = tokenizer.decode(txt_self[i].tolist())
        cross_dec = tokenizer.decode(txt_cross[i].tolist())
        print(f"  真值    : {truth}")
        print(f"  自重建  : {self_dec}")
        print(f"  图像→文本: {cross_dec}")
    return i2t, t2i


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="small")
    # 模型规模（覆盖 preset 默认值）
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    # 训练
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dataset-size", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg = build_config(
        args.preset,
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
    )
    assert cfg.dim % cfg.heads == 0, f"dim ({cfg.dim}) 必须能被 heads ({cfg.heads}) 整除"
    assert cfg.image_size % cfg.patch_size == 0, "image_size 必须能被 patch_size 整除"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | preset: {args.preset} | dim={cfg.dim} depth={cfg.depth} "
          f"heads={cfg.heads} image={cfg.image_size} patch={cfg.patch_size} "
          f"latent_len={cfg.text_len}")

    # 数据 + 词表
    images, caption_ids, captions, tokenizer = build_dataset(
        args.dataset_size, cfg.image_size, cfg.text_len, seed=args.seed
    )
    pad_id, mask_id = tokenizer.pad_id, tokenizer.mask_id

    model = MultimodalJEPA(cfg, tokenizer.vocab_size).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数: {total/1e6:.2f}M 总 | {trainable/1e6:.2f}M 可训练 | vocab={tokenizer.vocab_size}")

    train(model, images, caption_ids, pad_id, mask_id, device,
          args.steps, args.batch_size, args.lr)

    evaluate(model, images, caption_ids, captions, tokenizer, device, args.eval_samples)


if __name__ == "__main__":
    main()
