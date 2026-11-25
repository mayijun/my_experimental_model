"""模型组装：双在线/目标编码器 + 表征层（预测器） + 共享双解码器。"""

import torch
import torch.nn as nn

from .config import MJEPAConfig
from .decoder import ImageDecoder, TextDecoder
from .encoder import ImageEncoder, TextEncoder
from .alignment import build_alignment_head
from .representation import CrossMaskPredictor, LatentPredictor
from .operator import PsiLayer
from .flow import VelocityField


def _decoder_heads(dim, head_dim=64):
    """解码器在潜空间 d 内做注意力，返回能整除 d 的注意力头数。"""
    heads = max(1, dim // head_dim)
    while heads > 1 and dim % heads != 0:
        heads -= 1
    return heads


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
            hidden_dim=cfg.hidden_dim, backbone=cfg.backbone, attn_layers=cfg.attn_layers,
            image_frontend=cfg.image_frontend, siglip2_model=cfg.siglip2_model,
            freeze_vision=cfg.freeze_vision, rho_compress=cfg.rho_compress,
            rho_head_dim=cfg.rho_head_dim, rho_beta=cfg.rho_beta,
            rho_eps=cfg.rho_eps, rho_eps_z=cfg.rho_eps_z, rho_eps_keep=cfg.rho_eps_keep,
            rho_b_a=cfg.rho_b_a, rho_b_c=cfg.rho_b_c, rho_b_min=cfg.rho_b_min, rho_b_max=cfg.rho_b_max,
        )
        self.text_online = TextEncoder(
            vocab_size, length, cfg.dim, cfg.heads, cfg.depth, cfg.mlp_ratio,
            hidden_dim=cfg.hidden_dim, backbone=cfg.backbone, attn_layers=cfg.attn_layers,
            rho_compress=cfg.rho_compress, rho_head_dim=cfg.rho_head_dim,
            rho_beta=cfg.rho_beta, rho_eps=cfg.rho_eps, rho_eps_z=cfg.rho_eps_z, rho_eps_keep=cfg.rho_eps_keep,
            rho_b_a=cfg.rho_b_a, rho_b_c=cfg.rho_b_c, rho_b_min=cfg.rho_b_min, rho_b_max=cfg.rho_b_max,
        )

        # —— 编码（目标，EMA、stop-gradient）——
        self.image_target = ImageEncoder(
            cfg.image_size, cfg.patch_size, cfg.image_channels, cfg.dim,
            cfg.heads, cfg.depth, cfg.mlp_ratio,
            hidden_dim=cfg.hidden_dim, backbone=cfg.backbone, attn_layers=cfg.attn_layers,
            image_frontend=cfg.image_frontend, siglip2_model=cfg.siglip2_model,
            freeze_vision=cfg.freeze_vision, rho_compress=cfg.rho_compress,
            rho_head_dim=cfg.rho_head_dim, rho_beta=cfg.rho_beta,
            rho_eps=cfg.rho_eps, rho_eps_z=cfg.rho_eps_z, rho_eps_keep=cfg.rho_eps_keep,
            rho_b_a=cfg.rho_b_a, rho_b_c=cfg.rho_b_c, rho_b_min=cfg.rho_b_min, rho_b_max=cfg.rho_b_max,
        )
        self.text_target = TextEncoder(
            vocab_size, length, cfg.dim, cfg.heads, cfg.depth, cfg.mlp_ratio,
            hidden_dim=cfg.hidden_dim, backbone=cfg.backbone, attn_layers=cfg.attn_layers,
            rho_compress=cfg.rho_compress, rho_head_dim=cfg.rho_head_dim,
            rho_beta=cfg.rho_beta, rho_eps=cfg.rho_eps, rho_eps_z=cfg.rho_eps_z, rho_eps_keep=cfg.rho_eps_keep,
            rho_b_a=cfg.rho_b_a, rho_b_c=cfg.rho_b_c, rho_b_min=cfg.rho_b_min, rho_b_max=cfg.rho_b_max,
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
        dec_heads = _decoder_heads(cfg.dim)
        self.image_decoder = ImageDecoder(
            cfg.image_size, cfg.patch_size, cfg.image_channels, cfg.dim,
            dec_heads, cfg.depth, cfg.mlp_ratio,
        )
        self.text_decoder = TextDecoder(
            vocab_size, length, cfg.dim, dec_heads, cfg.depth, cfg.mlp_ratio,
        )

        # —— 软对齐投影头（§3.4）：InfoNCE 用 proj；BYOL/DINO 用 align ——
        self.proj = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim))
        self.align = build_alignment_head(cfg)

        # —— 交叉掩码预测器（§3.2 式 18）——
        self.cross_mask_predictor = (
            CrossMaskPredictor(cfg.dim, _decoder_heads(cfg.dim)) if cfg.cross_mask else None
        )

        # —— PSCO 状态依赖收缩算子（§3.7.3）：共享 Z 上的自映射，防塌缩 ——
        self.psi_layer = (
            PsiLayer(cfg.dim, cfg.psi_transient_steps, cfg.psi_scan_steps,
                     cfg.psi_delta, cfg.psi_gamma, cfg.psi_residual)
            if cfg.psi_enabled else None
        )

        # —— Flow Matching 速度场（§4.1）：潜空间 v_ψ，欧氏 OT 直线基线 ——
        self.velocity = (
            VelocityField(cfg.dim, cfg.flow_depth, cfg.flow_heads, cfg.mlp_ratio)
            if cfg.flow_enabled else None
        )
