"""训练目标与损失：编码 → 对齐/推理 → 解码的前向与分项损失（math.md §2.2、§3.2–3.4）。"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model.flow import flow_loss


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


def rate_loss(b):
    """码率正则（§2.6 式 665）：b_t 的均值 = 每 token 平均码率预算 R。

    最小化 R 即「用更少比特装下同样信息」，与重建失真构成率失真权衡 R+λD；
    梯度经 b_t → ρ → 记忆投影 (k_proj, v_proj) 回传，使记忆学得更会吸收冗余（ρ↓）。
    b 为 None（未启用 ρ 压缩）时返回 None，由调用方置零。
    """
    if b is None:
        return None
    return b.mean()


def info_nce(zi, zt, temperature):
    """对称 InfoNCE：图像 [CLS] 与文本 [CLS] 的软对齐。"""
    zi = F.normalize(zi, dim=-1)
    zt = F.normalize(zt, dim=-1)
    logits = zi @ zt.t() / temperature  # [B,B]
    labels = torch.arange(zi.size(0), device=zi.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def byol_loss(online_img, online_txt, target_img, target_txt):
    """BYOL 跨模态自蒸馏损失（§3.4）：在线预测拟合 EMA 目标，无负样本、无 batch 依赖。"""
    return cosine_loss(online_img, target_txt) + cosine_loss(online_txt, target_img)


def dino_loss(s_img, s_txt, t_img, t_txt, student_temp, teacher_temp):
    """DINO 自蒸馏损失（§3.4）：学生软标签拟合教师软标签（锐化），跨模态对称。"""
    p_i = F.softmax(s_img / student_temp, dim=-1)
    p_t = F.softmax(s_txt / student_temp, dim=-1)
    q_t = F.softmax(t_txt / teacher_temp, dim=-1)  # 教师（已 stop-gradient）
    q_i = F.softmax(t_img / teacher_temp, dim=-1)
    return (
        -(q_t * torch.log(p_i + 1e-8)).sum(-1)
        - (q_i * torch.log(p_t + 1e-8)).sum(-1)
    ).mean()


def alignment_loss(model, img_cls, txt_cls):
    """按 cfg.alignment_method 分派软对齐损失（§3.4），默认 BYOL。"""
    cfg = model.cfg
    if cfg.alignment_method == "infonce":
        return info_nce(model.proj(img_cls), model.proj(txt_cls), cfg.temperature)
    if cfg.alignment_method == "byol":
        return byol_loss(
            model.align(img_cls), model.align(txt_cls),
            model.align.target(img_cls), model.align.target(txt_cls),
        )
    if cfg.alignment_method == "dino":
        return dino_loss(
            model.align(img_cls), model.align(txt_cls),
            model.align.teacher_forward(img_cls), model.align.teacher_forward(txt_cls),
            cfg.dino_student_temp, cfg.dino_teacher_temp,
        )
    return torch.zeros((), device=img_cls.device)


def cross_mask_ratio(step, total_steps, start, end):
    """交叉掩码递进（§3.4）：ρ_mask(t)=min(end, start+(end-start)·t/T)。"""
    if step is None or total_steps is None:
        return 0.5 * (start + end)
    t = step / max(1, total_steps)
    return min(end, start + (end - start) * t)


def psi_losses(model, img_z, txt_z, img_ctx, txt_ctx, img_tgt, txt_tgt):
    """PSCO 算子（§3.7.3 + §3.7）损失：防塌缩三道 + 推理两目标。

    - readout（防线 2）：读出头锚定 Z* 重建输入 Z（塌成常数则重建不了）；
    - var（防线 3）：方差正则，防维度塌缩；
    - stop（§3.7 L_stop）：不动点/停止正则，Ψ_h(Z_clean)≈Z_clean，读端 g_change→0 退化为恒等；
    - consist（§3.7 L_consist）：路径无关一致性，Ψ_h(Z_ctx)≈Z_tgt，改端把上下文变换到目标。

    返回 (readout, var, stop, consist)。
    """
    img_star = model.psi_layer(img_z)
    txt_star = model.psi_layer(txt_z)
    readout = (
        (model.psi_layer.readout(img_star) - img_z).pow(2).mean()
        + (model.psi_layer.readout(txt_star) - txt_z).pow(2).mean()
    )
    var = variance_loss(img_star) + variance_loss(txt_star)
    # §3.7 L_stop：干净 Z 已是不动点（读端恒等，改分量 g_change→0）
    stop = (img_star - img_z).pow(2).mean() + (txt_star - txt_z).pow(2).mean()
    # §3.7 L_consist：算子把掩码上下文变换到干净目标（改端，改分量 g_change→1）
    img_ctx_star = model.psi_layer(img_ctx)
    txt_ctx_star = model.psi_layer(txt_ctx)
    consist = (img_ctx_star - img_tgt).pow(2).mean() + (txt_ctx_star - txt_tgt).pow(2).mean()
    return readout, var, stop, consist


def maybe_autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_losses(model, img, txt, pad_id, mask_id, step=None, total_steps=None):
    """跑通「编码 → 对齐/推理 → 解码」，返回 (总损失, 分项损失 dict)。

    step/total_steps 供交叉掩码递进 ρ_mask(t) 与 PAC 锚概率调度使用；None 时取中值。
    """
    cfg = model.cfg
    batch = img.size(0)
    device = img.device

    # —— 编码（在线，干净 + 掩码）——
    img_cls, img_tokens = model.image_online(img)
    txt_cls, txt_tokens = model.text_online(txt)
    # 码率信号 b_t（§2.6 式 665）：取自在线编码器干净输入，表示编码完整输入所需码率。
    # 必须在掩码 forward（下面）覆盖 last_b 之前捕获。
    img_b = model.image_online.last_b
    txt_b = model.text_online.last_b

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

    jepa = cosine_loss(img_pred, img_tgt) + cosine_loss(txt_pred, txt_tgt)
    recon_img_self = F.mse_loss(img_self, img)
    recon_txt_self = F.cross_entropy(
        txt_self.reshape(-1, txt_self.size(-1)), txt.reshape(-1), ignore_index=pad_id
    )
    recon_img_cross = F.mse_loss(img_cross, img)
    recon_txt_cross = F.cross_entropy(
        txt_cross.reshape(-1, txt_cross.size(-1)), txt.reshape(-1), ignore_index=pad_id
    )
    align = alignment_loss(model, img_cls, txt_cls)
    var = variance_loss(img_tokens) + variance_loss(txt_tokens)
    rate = torch.zeros((), device=device)
    if cfg.rho_compress and img_b is not None and txt_b is not None:
        rate = 0.5 * (img_b.mean() + txt_b.mean())

    # —— 交叉掩码（§3.2 式 18）：本模态剩余 + 其他模态联合预测被掩部分，防模态坍缩 ——
    cross_pred = torch.zeros((), device=device)
    if cfg.cross_mask and model.cross_mask_predictor is not None:
        ratio = cross_mask_ratio(
            step, total_steps, cfg.cross_mask_ratio_start, cfg.cross_mask_ratio_end
        )
        cm_img_mask = mask_patches(batch, cfg.n_patches, ratio, device)
        cm_txt_masked = mask_text(txt, mask_id, ratio, device)
        _, cm_img_ctx = model.image_online(img, mask=cm_img_mask)
        _, cm_txt_ctx = model.text_online(cm_txt_masked)
        cm_pred_img, cm_pred_txt = model.cross_mask_predictor(cm_img_ctx, cm_txt_ctx)
        cross_pred = cosine_loss(cm_pred_img, img_tgt) + cosine_loss(cm_pred_txt, txt_tgt)

    # —— PSCO 算子（§3.7.3 + §3.7）：防塌缩三道 + 推理两目标（L_stop / L_consist）——
    psi_readout = torch.zeros((), device=device)
    psi_var = torch.zeros((), device=device)
    psi_stop = torch.zeros((), device=device)
    psi_consist = torch.zeros((), device=device)
    if cfg.psi_enabled and model.psi_layer is not None:
        psi_readout, psi_var, psi_stop, psi_consist = psi_losses(
            model, img_tokens, txt_tokens, img_ctx, txt_ctx, img_tgt, txt_tgt
        )

    # —— Flow Matching（§4.1.2）：无条件欧氏 OT 直线，z_0=噪声，z_1=冻结目标 Z ——
    flow = torch.zeros((), device=device)
    if cfg.flow_enabled and model.velocity is not None:
        flow = flow_loss(model.velocity, img_tgt) + flow_loss(model.velocity, txt_tgt)

    total = (
        cfg.jepa_weight * jepa
        + cfg.recon_weight * (recon_img_self + recon_txt_self)
        + cfg.cross_weight * (recon_img_cross + recon_txt_cross)
        + cfg.align_weight * align
        + cfg.var_weight * var
        + cfg.rho_rate_weight * rate
        + cfg.cross_mask_weight * cross_pred
        + cfg.psi_readout_weight * psi_readout
        + cfg.psi_var_weight * psi_var
        + cfg.psi_stop_weight * psi_stop
        + cfg.psi_consist_weight * psi_consist
        + cfg.flow_weight * flow
    )
    parts = {
        "jepa": jepa, "recon_img": recon_img_self, "recon_txt": recon_txt_self,
        "cross_img": recon_img_cross, "cross_txt": recon_txt_cross,
        "align": align, "var": var, "rate": rate, "cross_pred": cross_pred,
        "psi_readout": psi_readout, "psi_var": psi_var,
        "psi_stop": psi_stop, "psi_consist": psi_consist,
        "flow": flow,
    }
    return total, parts
