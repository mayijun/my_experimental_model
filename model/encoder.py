"""编码器：图像 / 文本 → 共享潜空间 Z（math.md §2）。

E^I 与 E^T 各自把模态编译进同一个连续潜空间 Z = [B, L, d]，并支持掩码输入
（图像换 mask_token、文本换 mask_id），供 JEPA 潜预测使用。
"""

import torch
import torch.nn as nn


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
