"""解码器：共享 Z → 图像 / 文本（math.md §4）。

两个解码器都接收同一形状的潜 token [B,L,d]：图像 token 与文本 token 都走它们，
这正是「硬对齐 / 双向重建」能把两个模态焊在同一坐标系的原因。
"""

import torch
import torch.nn as nn

from .encoder import build_backbone


class ImageDecoder(nn.Module):
    """R_I：把潜 token [B,L,d] 还原成图像 [B,C,H,W]（共享：图像 token 与文本 token 都走它）。

    主干与编码器同构（transformer / mamba / hybrid，§4.2(g) 对称架构），默认 hybrid。
    """

    def __init__(self, image_size, patch_size, channels, dim, heads, depth,
                 mlp_ratio=4.0, backbone="hybrid", attn_layers=2, grad_checkpoint=False):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.n_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        self.decoder = build_backbone(backbone, dim, heads, depth, mlp_ratio, attn_layers, grad_checkpoint)
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
    """g：把潜 token [B,L,d] 还原成字符 logits [B,L,V]（共享：文本 token 与图像 token 都走它）。

    主干与编码器同构（transformer / mamba / hybrid，§4.2(g) 对称架构），默认 hybrid。
    """

    def __init__(self, vocab_size, max_len, dim, heads, depth,
                 mlp_ratio=4.0, backbone="hybrid", attn_layers=2, grad_checkpoint=False):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.decoder = build_backbone(backbone, dim, heads, depth, mlp_ratio, attn_layers, grad_checkpoint)
        self.head = nn.Linear(dim, vocab_size)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z):
        h = self.decoder(z + self.pos_embed[:, : z.size(1)])
        return self.head(h)  # [B,L,V]
