"""模型组装：双在线/目标编码器 + 表征层（预测器） + 共享双解码器。"""

import torch
import torch.nn as nn

from .config import MJEPAConfig
from .decoder import ImageDecoder, TextDecoder
from .encoder import ImageEncoder, TextEncoder, build_backbone
from .alignment import build_alignment_head
from .representation import CrossMaskPredictor, LatentPredictor
from .operator import PsiLayer
from .flow import VelocityField, conditional_flow_loss, flow_loss


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

        # —— 共享主干（§2.8 共享 E）：图文复用同一条 trunk，薄接口每模态一份 ——
        if cfg.share_trunk:
            trunk_kwargs = dict(
                backbone=cfg.backbone, dim=(cfg.hidden_dim or cfg.dim), heads=cfg.heads,
                depth=cfg.depth, mlp_ratio=cfg.mlp_ratio, attn_layers=cfg.attn_layers,
                grad_checkpoint=cfg.grad_checkpoint,
            )
            self.shared_online = build_backbone(**trunk_kwargs)
            self.shared_target = build_backbone(**trunk_kwargs)
        else:
            self.shared_online = None
            self.shared_target = None

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
            grad_checkpoint=cfg.grad_checkpoint, shared_encoder=self.shared_online,
        )
        self.text_online = TextEncoder(
            vocab_size, length, cfg.dim, cfg.heads, cfg.depth, cfg.mlp_ratio,
            hidden_dim=cfg.hidden_dim, backbone=cfg.backbone, attn_layers=cfg.attn_layers,
            rho_compress=cfg.rho_compress, rho_head_dim=cfg.rho_head_dim,
            rho_beta=cfg.rho_beta, rho_eps=cfg.rho_eps, rho_eps_z=cfg.rho_eps_z, rho_eps_keep=cfg.rho_eps_keep,
            rho_b_a=cfg.rho_b_a, rho_b_c=cfg.rho_b_c, rho_b_min=cfg.rho_b_min, rho_b_max=cfg.rho_b_max,
            embed_factor=cfg.embed_factor,
            grad_checkpoint=cfg.grad_checkpoint, shared_encoder=self.shared_online,
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
            grad_checkpoint=cfg.grad_checkpoint, shared_encoder=self.shared_target,
        )
        self.text_target = TextEncoder(
            vocab_size, length, cfg.dim, cfg.heads, cfg.depth, cfg.mlp_ratio,
            hidden_dim=cfg.hidden_dim, backbone=cfg.backbone, attn_layers=cfg.attn_layers,
            rho_compress=cfg.rho_compress, rho_head_dim=cfg.rho_head_dim,
            rho_beta=cfg.rho_beta, rho_eps=cfg.rho_eps, rho_eps_z=cfg.rho_eps_z, rho_eps_keep=cfg.rho_eps_keep,
            rho_b_a=cfg.rho_b_a, rho_b_c=cfg.rho_b_c, rho_b_min=cfg.rho_b_min, rho_b_max=cfg.rho_b_max,
            embed_factor=cfg.embed_factor,
            grad_checkpoint=cfg.grad_checkpoint, shared_encoder=self.shared_target,
        )
        self.image_target.load_state_dict(self.image_online.state_dict())
        self.text_target.load_state_dict(self.text_online.state_dict())
        # 目标编码器（含 share_trunk 时的共享目标主干）全部冻结：只被 EMA 更新、仅在
        # no_grad 下前向。冻结 shared_target 还避免 DDP 因「requires_grad 但无梯度」报错。
        target_modules = [self.image_target, self.text_target]
        if self.shared_target is not None:
            target_modules.append(self.shared_target)
        for module in target_modules:
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
            backbone=cfg.backbone, attn_layers=cfg.attn_layers, grad_checkpoint=cfg.grad_checkpoint,
        )
        self.text_decoder = TextDecoder(
            vocab_size, length, cfg.dim, dec_heads, cfg.depth, cfg.mlp_ratio,
            backbone=cfg.backbone, attn_layers=cfg.attn_layers, grad_checkpoint=cfg.grad_checkpoint,
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
            VelocityField(cfg.dim, cfg.flow_depth, cfg.flow_heads, cfg.mlp_ratio,
                          use_adaln=cfg.flow_adaln, grad_checkpoint=cfg.grad_checkpoint)
            if cfg.flow_enabled else None
        )

    def forward(self, img, txt, pad_id=None, mask_id=None, step=None, total_steps=None, mode="pretrain"):
        """训练前向。mode="pretrain"：编码→对齐/推理→解码（§2-§5）；
        mode="flow"：只训速度场 v_ψ（§4.2(e)），冻结 E_ξ/E_θ，仅 velocity 产生梯度。

        包成 model(...) 使 loss 出自 model.forward，从而能被
        torch.nn.parallel.DistributedDataParallel 正确包裹并同步梯度（直接调用
        image_online/text_decoder 等子模块会绕过 DDP 梯度同步）。
        """
        if mode == "flow":
            return self._forward_flow(img, txt)
        from training.losses import forward_losses

        return forward_losses(self, img, txt, pad_id, mask_id, step=step, total_steps=total_steps)

    def _forward_flow(self, img, txt):
        """Flow 阶段前向（§4.1.2/§4.2(e)）：冻结目标编码器采样 z_1，回归 v_ψ。"""
        if self.velocity is None:
            raise ValueError("Flow 阶段需要 flow_enabled=True（模型未启用速度场 v_ψ）。")
        with torch.no_grad():
            _, img_tgt = self.image_target(img)
            _, txt_tgt = self.text_target(txt)
        flow = flow_loss(self.velocity, img_tgt) + flow_loss(self.velocity, txt_tgt)
        flow_cond = torch.zeros((), device=img_tgt.device)
        if self.cfg.flow_cond_enabled:
            flow_cond = (
                conditional_flow_loss(self.velocity, txt_tgt, img_tgt)
                + conditional_flow_loss(self.velocity, img_tgt, txt_tgt)
            )
        loss = self.cfg.flow_weight * flow + self.cfg.flow_cond_weight * flow_cond
        # 与 pretrain 分支一致返回三元组 (loss, parts, stats)；flow 阶段暂无额外诊断统计。
        return loss, {"flow": flow, "flow_cond": flow_cond}, {}

    def state_dict_for_save(self):
        """去重后的落盘 state_dict（§2.8 共享 E）。

        share_trunk 下 `image_online.encoder` / `text_online.encoder` 与 `shared_online`
        是同一主干对象，`state_dict()` 会把主干参数序列化三份；目标编码器
        （shared_target/image_target/text_target）是 EMA 副本，落盘也属重复。
        这里只保留唯一在线参数：共享主干一份 + 各模态薄接口 + 解码器/速度场/对齐/预测器。
        目标编码器 resume 时由 init_target_from_online() 从在线重新初始化。
        """
        full = self.state_dict()
        skip = ("shared_target.", "image_target.", "text_target.")
        if self.cfg.share_trunk:
            skip += ("image_online.encoder.", "text_online.encoder.")
        return {k: v for k, v in full.items() if not k.startswith(skip)}

    def init_target_from_online(self):
        """目标编码器从在线编码器重新初始化（EMA 冷启动），resume 时恢复被省略的目标副本。

        image_target.load_state_dict(image_online.state_dict()) 会把在线共享主干
        一并复制进 shared_target（经 image_target.encoder 引用），故无需单独处理 shared_target。
        """
        self.image_target.load_state_dict(self.image_online.state_dict())
        self.text_target.load_state_dict(self.text_online.state_dict())
