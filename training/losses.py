"""训练目标与损失：编码 → 对齐/推理 → 解码的前向与分项损失（math.md §2.2、§3.2–3.4）。"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F


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


def maybe_autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_losses(model, img, txt, pad_id, mask_id):
    """跑通「编码 → 对齐/推理 → 解码」，返回 (总损失, 分项损失 dict)。"""
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
    rate = torch.zeros((), device=device)
    if cfg.rho_compress and img_b is not None and txt_b is not None:
        rate = 0.5 * (img_b.mean() + txt_b.mean())

    total = (
        cfg.jepa_weight * jepa
        + cfg.recon_weight * (recon_img_self + recon_txt_self)
        + cfg.cross_weight * (recon_img_cross + recon_txt_cross)
        + cfg.align_weight * align
        + cfg.var_weight * var
        + cfg.rho_rate_weight * rate
    )
    parts = {
        "jepa": jepa, "recon_img": recon_img_self, "recon_txt": recon_txt_self,
        "cross_img": recon_img_cross, "cross_txt": recon_txt_cross,
        "align": align, "var": var, "rate": rate,
    }
    return total, parts
