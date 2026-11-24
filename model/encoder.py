"""编码器：图像 / 文本 → 共享潜空间 Z（math.md §2）。

E^I 与 E^T 各自把模态编译进同一个连续潜空间 Z = [B, L, d]，并支持掩码输入
（图像换 mask_token、文本换 mask_id），供 JEPA 潜预测使用。
"""

import torch
import torch.nn as nn

from .psco import associative_scan


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


class SSMBlock(nn.Module):
    """对角 SSM 块：z_{t+1} = a_t ⊙ z_t + b_t，输出连续（§2.6 契约）。

    用 psco.associative_scan 做并行前缀扫描（O(log L) 深度）；
    a_t 经谱夹层钳在 (δ, 1-γ)：上界 <1 保证收敛、下界 >δ 防瞬间塌缩（§3.7.3）。
    这是 Mamba/S4 对角状态空间（ZOH 离散化）的简化实现，满足「Z 全程连续、不量化」。
    """

    def __init__(self, dim, mlp_ratio=4.0, delta=0.05, gamma=0.10):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)  # -> (a_raw, b)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self._scan(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def _scan(self, h):
        ab = self.in_proj(h)  # [B, L, 2d]
        a_raw, b = ab.chunk(2, dim=-1)
        a = self.delta + (1.0 - self.gamma - self.delta) * torch.sigmoid(a_raw)
        # 并行前缀扫描：psco.associative_scan 沿第 0 维（序列）扫描 [L, B, d]。
        a_seq = a.transpose(0, 1).contiguous()
        b_seq = b.transpose(0, 1).contiguous()
        _, b_pref, _ = associative_scan(a_seq, b_seq)
        return b_pref.transpose(0, 1)  # 零初始状态：输出即累计状态轨迹


class MambaEncoder(nn.Module):
    """全 SSM 主干：堆叠若干 SSMBlock。"""

    def __init__(self, dim, depth, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([SSMBlock(dim, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class HybridEncoder(nn.Module):
    """混合主干（§2.2.5 候选 2）：前 attn_layers 层注意力 + 其余 SSM。

    前段注意力抓局部/位置敏感模式，后段 SSM 做长程压缩，兼顾表达力与线性复杂度。
    """

    def __init__(self, dim, heads, depth, mlp_ratio=4.0, attn_layers=2):
        super().__init__()
        attn_layers = max(0, min(attn_layers, depth))
        blocks = [TransformerBlock(dim, heads, mlp_ratio) for _ in range(attn_layers)]
        blocks += [SSMBlock(dim, mlp_ratio) for _ in range(attn_layers, depth)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def build_backbone(backbone, dim, heads, depth, mlp_ratio=4.0, attn_layers=2):
    """按 §2.2.5 的主干选型返回序列编码器（transformer / mamba / hybrid）。"""
    if backbone == "mamba":
        return MambaEncoder(dim, depth, mlp_ratio)
    if backbone == "hybrid":
        return HybridEncoder(dim, heads, depth, mlp_ratio, attn_layers)
    return TransformerEncoder(dim, heads, depth, mlp_ratio)


class ImageEncoder(nn.Module):
    """E^I：把图像 [B,C,H,W] 编译成 patch token 序列 + [CLS]。

    掩码时把被掩 patch 替换为可学习 mask_token，供 JEPA 推理使用。
    主干在 hidden_dim（d_hidden）内混合，再经 head 投影到潜空间 d（§2.2.7 解耦）。
    """

    def __init__(self, image_size, patch_size, channels, dim, heads, depth,
                 mlp_ratio=4.0, hidden_dim=0, backbone="hybrid", attn_layers=2):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.encoder = build_backbone(backbone, hidden_dim, heads, depth, mlp_ratio, attn_layers)
        self.head = nn.Linear(hidden_dim, dim) if hidden_dim != dim else nn.Identity()
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):
        batch = x.size(0)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B,N,d_hidden]
        if mask is not None:  # mask: [B,N] bool，True 处替换为 mask_token。
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        cls = self.cls_token.expand(batch, -1, -1)
        h = torch.cat([cls, tokens], dim=1) + self.pos_embed
        h = self.head(self.encoder(h))  # [B, 1+L, d]
        return h[:, 0], h[:, 1:]  # (cls [B,d], tokens [B,N,d])


class TextEncoder(nn.Module):
    """E^T：把 token id 序列 [B,L] 编译成 token 序列 + [CLS]。

    掩码由「把部分 token 替换为 mask_id」实现，与 BERT 一致。
    主干在 hidden_dim 内混合，再投影到潜空间 d。
    """

    def __init__(self, vocab_size, max_len, dim, heads, depth,
                 mlp_ratio=4.0, hidden_dim=0, backbone="hybrid", attn_layers=2):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, hidden_dim))
        self.encoder = build_backbone(backbone, hidden_dim, heads, depth, mlp_ratio, attn_layers)
        self.head = nn.Linear(hidden_dim, dim) if hidden_dim != dim else nn.Identity()
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, ids):
        batch = ids.size(0)
        x = self.embedding(ids)
        cls = self.cls_token.expand(batch, -1, -1)
        h = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.size(1) + 1]
        h = self.head(self.encoder(h))
        return h[:, 0], h[:, 1:]
