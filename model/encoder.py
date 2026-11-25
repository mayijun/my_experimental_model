"""编码器：图像 / 文本 → 共享潜空间 Z（math.md §2）。

E^I 与 E^T 各自把模态编译进同一个连续潜空间 Z = [B, L, d]，并支持掩码输入
（图像换 mask_token、文本换 mask_id），供 JEPA 潜预测使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .psco import associative_scan, associative_scan_mat


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


class LinearMemory(nn.Module):
    """delta rule 线性记忆（§2.6 创新编码的记忆 M）。

    更新：M_t = M_{t-1} + β (v_t − M_{t-1} k_t) k_tᵀ；
    冗余度 ρ_t = ‖v_t − M_{t-1} k_t‖²（写入前用旧记忆 M_{t-1} 预测 v_t）。
    ρ≈0 表示该 token 已被记忆吸收、可丢；ρ 大表示不可预测、必须留。

    D（head_dim）是冻结的单头记忆宽度，不随主干 hidden_dim 缩放（§2.6 GPU 并行）；
    大型模型扩展靠加 head（多份独立 M），而非放大 D——放大 D 是 O(D²) 显存平方增长。

    更新用矩阵仿射并行扫描（O(log L) 深度，精度优先路径）：M_t = M_{t-1} A_t + B_t，
    复用 psco.associative_scan_mat。显存 O(B·L·D²)，D=64 冻结下可接受；
    显存优先路径（DeltaNet chunked / WY 低秩，O(B·L·D)）仅记录不实现，D 需放大时再启用。
    """

    def __init__(self, dim, head_dim=64, beta=0.1):
        super().__init__()
        self.k_proj = nn.Linear(dim, head_dim, bias=False)
        self.v_proj = nn.Linear(dim, head_dim, bias=False)
        self.beta = beta

    def forward(self, x):
        # x: [B, L, dim] -> rho: [B, L]
        k = F.normalize(self.k_proj(x), dim=-1)  # 归一化 k，保证 β‖k‖² 落稳定区间（β<2 收敛）
        v = self.v_proj(x)
        B, L, D = k.shape

        # delta rule 矩阵仿射：M_t = M_{t-1} A_t + B_t，
        #   A_t = I − β k_t k_tᵀ（对称满秩、对所有行共享），B_t = β v_t k_tᵀ（秩 1）。
        kk = k.unsqueeze(-1) @ k.unsqueeze(-2)                      # [B, L, D, D]
        A = torch.eye(D, device=x.device, dtype=x.dtype) - self.beta * kk
        Bm = self.beta * v.unsqueeze(-1) @ k.unsqueeze(-2)          # [B, L, D, D]

        # 矩阵仿射并行扫描（O(log L) 深度）：M_prefix[t] = M_{t+1}（含第 t+1 步写入）。
        A_seq = A.transpose(0, 1).contiguous()                      # [L, B, D, D]
        B_seq = Bm.transpose(0, 1).contiguous()
        _, M_prefix, _ = associative_scan_mat(A_seq, B_seq)

        # ρ_t = ‖v_t − M_{t-1} k_t‖²，M_{t-1} = M_prefix 左移一位（首位补零 = M_0 = 0）。
        M_prev = torch.cat([torch.zeros_like(M_prefix[:1]), M_prefix[:-1]], dim=0)
        y = torch.einsum("lbij,lbj->lbi", M_prev, k.transpose(0, 1))  # M_{t-1} k_t, [L, B, D]
        rho = (v.transpose(0, 1) - y).pow(2).sum(-1).transpose(0, 1)  # [B, L]
        return rho


class RhoCompressor(nn.Module):
    """token 层 ρ 残差压缩（§2.6）：按冗余度 ρ 丢冗余、mask 填充保 L。

    keep_t = (ρ_t > ε_ρ) ∨ sink(t) ∨ force(t)：
      - ρ>ε_ρ：不可预测，必须留；
      - sink(t)：位置 0 硬编码不丢（softmax 泄洪口）；
      - force(t)：ε 强制采样，随机保留一小部分低 ρ token 防自我强化退化。
    被丢位置用 mask_token 填充，序列长度不变（保 L），硬对齐不破。

    每个模态一个实例（私有 M）；多模态接入即新增一个实例，接口天然扩展。
    """

    def __init__(self, dim, head_dim=64, beta=0.1, eps_rho=0.1, eps_keep=0.02):
        super().__init__()
        self.memory = LinearMemory(dim, head_dim, beta)
        self.eps_rho = eps_rho
        self.eps_keep = eps_keep

    def forward(self, x, mask_token):
        # x: [B, L, dim]; mask_token: [1, 1, dim] 或 [1, dim]
        # 返回 (x_tilde, keep [B,L] bool, rho [B,L])
        rho = self.memory(x)               # [B, L]
        keep = rho > self.eps_rho          # 冗余度大 = 不可预测 = 保留
        keep[:, 0] = True                  # sink 保护：位置 0 不丢
        if self.training and self.eps_keep > 0:
            # ε 强制采样：随机保留一小部分被丢的低 ρ token（§2.6 补丁 2）
            drop = ~keep
            force = drop & (torch.rand_like(rho) < self.eps_keep)
            keep = keep | force
        x_tilde = torch.where(keep.unsqueeze(-1), x, mask_token)
        return x_tilde, keep, rho


class SigLIP2Frontend(nn.Module):
    """LLaDA-V 视觉前端：冻结的 SigLIP-2 视觉编码器 + 可训练 MLP 投影（§2.2.7）。

    LLaDA-V 用 SigLIP-2 提取连续 patch 特征（不量化），MLP 连接器投影到语言嵌入空间。
    这里把投影目标设为本框架的主干宽度 d_hidden，输出 [B, n_patches, d_hidden]，
    满足 §2.6「Z 全程连续、有损只在 token 层」的契约。

    运行依赖（仅 siglip2 前端需要，放 hf extra）：transformers（已有）+ 首次加载需
    联网下载 SigLIP-2 权重；图像预处理用 F.interpolate + SigLIP 归一化，无需 pillow。
    """

    def __init__(self, model_name, hidden_dim, image_size, freeze=True):
        super().__init__()
        from transformers import Siglip2VisionModel

        self.vision = Siglip2VisionModel.from_pretrained(model_name)
        self.freeze = freeze
        if freeze:
            self.vision.eval()
            for p in self.vision.parameters():
                p.requires_grad_(False)
        vision_dim = self.vision.config.hidden_size
        self.image_size = image_size
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 4),
            nn.GELU(),
            nn.Linear(vision_dim * 4, hidden_dim),
        )

    def forward(self, x):
        # x: [B, C, H, W]，值域 [0,1]；resize 到 image_size 并按 SigLIP 归一化 ((x-0.5)/0.5)。
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        if self.freeze:
            with torch.no_grad():
                feat = self.vision(pixel_values=x).last_hidden_state  # [B, n_patches, vision_dim]
        else:
            feat = self.vision(pixel_values=x).last_hidden_state
        return self.proj(feat)  # [B, n_patches, hidden_dim]


class ImageEncoder(nn.Module):
    """E^I：把图像 [B,C,H,W] 编译成 patch token 序列 + [CLS]。

    掩码时把被掩 patch 替换为可学习 mask_token，供 JEPA 推理使用。
    主干在 hidden_dim（d_hidden）内混合，再经 head 投影到潜空间 d（§2.2.7 解耦）。
    """

    def __init__(self, image_size, patch_size, channels, dim, heads, depth,
                 mlp_ratio=4.0, hidden_dim=0, backbone="hybrid", attn_layers=2,
                 image_frontend="patch", siglip2_model="google/siglip2-base-patch16-224",
                 freeze_vision=True, rho_compress=False, rho_head_dim=64,
                 rho_beta=0.1, rho_eps=0.1, rho_eps_keep=0.02):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.frontend = image_frontend
        if image_frontend == "siglip2":
            # LLaDA-V 前端：冻结 SigLIP-2 + MLP 投影 → [B, n_patches, d_hidden]。
            self.patch_embed = SigLIP2Frontend(siglip2_model, hidden_dim, image_size, freeze_vision)
            self.n_patches = (image_size // 16) ** 2
        else:
            self.patch_embed = nn.Conv2d(channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
            self.n_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.encoder = build_backbone(backbone, hidden_dim, heads, depth, mlp_ratio, attn_layers)
        self.head = nn.Linear(hidden_dim, dim) if hidden_dim != dim else nn.Identity()
        # token 层 ρ 压缩（§2.6）：私有记忆 M，图像模态一份。
        self.rho_compressor = (
            RhoCompressor(hidden_dim, rho_head_dim, rho_beta, rho_eps, rho_eps_keep)
            if rho_compress else None
        )
        self.last_keep = None  # 供测试/可视化读取最近一次的 ρ 判据
        self.last_rho = None
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):
        batch = x.size(0)
        if self.frontend == "siglip2":
            tokens = self.patch_embed(x)  # [B,N,d_hidden]
        else:
            tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B,N,d_hidden]
        if self.rho_compressor is not None:  # 先写后压：干净 tokens 上算 ρ 丢冗余，mask 填充保 L
            tokens, self.last_keep, self.last_rho = self.rho_compressor(tokens, self.mask_token)
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
                 mlp_ratio=4.0, hidden_dim=0, backbone="hybrid", attn_layers=2,
                 rho_compress=False, rho_head_dim=64, rho_beta=0.1,
                 rho_eps=0.1, rho_eps_keep=0.02):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, hidden_dim))
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # ρ 压缩的 mask 填充
        self.encoder = build_backbone(backbone, hidden_dim, heads, depth, mlp_ratio, attn_layers)
        self.head = nn.Linear(hidden_dim, dim) if hidden_dim != dim else nn.Identity()
        # token 层 ρ 压缩（§2.6）：私有记忆 M，文本模态一份。
        self.rho_compressor = (
            RhoCompressor(hidden_dim, rho_head_dim, rho_beta, rho_eps, rho_eps_keep)
            if rho_compress else None
        )
        self.last_keep = None
        self.last_rho = None
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_embed, std=0.02)

    def forward(self, ids):
        batch = ids.size(0)
        x = self.embedding(ids)
        if self.rho_compressor is not None:  # 先写后压：embedding 上算 ρ 丢冗余，mask 填充保 L
            x, self.last_keep, self.last_rho = self.rho_compressor(x, self.mask_embed)
        cls = self.cls_token.expand(batch, -1, -1)
        h = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.size(1) + 1]
        h = self.head(self.encoder(h))
        return h[:, 0], h[:, 1:]
