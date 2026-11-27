"""训练目标与损失：编码 → 对齐/推理 → 解码的前向与分项损失（math.md §2.2、§3.2–3.4）。"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model.flow import conditional_flow_loss, flow_loss


def mask_patches(batch, n_patches, ratio, device):
    """随机遮住一部分 patch，返回 [B,N] 的 bool 掩码。"""
    return torch.rand(batch, n_patches, device=device) < ratio


def block_mask_patches(batch, n_patches, ratio, device, num_blocks=4):
    """I-JEPA 式多块掩码（§2.2.1）：在 2D patch 网格上采样 num_blocks 个随机矩形块，
    每块面积约 ratio/num_blocks 的 n_patches；返回 [B,N] bool（True=掩）。

    逐位置 Bernoulli 掩码会给「被掩 patch」留下相邻可见 patch（可抄捷径、学不到全局结构），
    连续块掩码的块内部两侧皆掩、无抄捷径（Lean: BlockMask.block_interior_no_shortcut）。
    """
    side = int(round(n_patches ** 0.5))
    mask = torch.zeros(batch, n_patches, dtype=torch.bool, device=device)
    block_area = max(1, int(ratio * n_patches / max(1, num_blocks)))
    for b in range(batch):
        m2d = mask[b].view(side, side)
        for _ in range(num_blocks):
            ar = float(0.5 + 1.5 * torch.rand(1, device=device).item())  # 宽高比 0.5~2.0
            bh = max(1, min(side, int(round((block_area * ar) ** 0.5))))
            bw = max(1, min(side, int(round(block_area / bh))))
            r = torch.randint(0, side - bh + 1, (1,), device=device).item()
            c = torch.randint(0, side - bw + 1, (1,), device=device).item()
            m2d[r:r + bh, c:c + bw] = True
    return mask


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


def covariance_loss(z):
    """VICReg 协方差约束（§2.2.4(c)）：惩罚维度间协方差，防维度冗余/低秩塌缩。z: [N,d]。

    N 可为 batch（对齐投影空间，CLS 每样本一行）或 batch×位置（token 序列 Z 摊平），
    二者语义一致：把 d 维潜空间的维度间去相关，迫使信息沿 d 个维度铺开而非挤进低秩子空间。
    """
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.t() @ z) / max(z.size(0) - 1, 1)
    d = cov.size(0)
    off = cov - torch.diag(cov.diag())
    return (off ** 2).sum() / d


def gauss_penalty(z):
    """分布正则（UniJEPA/LeWorldModel 高斯正则的逐维投影，§2.2.4(b) 补强）：mean_j (σ_j²-1)²。

    双端正则：σ<1（塌缩）与 σ>1（发散）都被惩罚，零集（σ≥0）恰为 {σ=1}；严格强于
    relu(1-σ) 的单边 hinge——后者在 σ≥1 处饱和为 0、梯度消失，无法把已塌缩的 σ 拉回
    （Lean: DistributionReg.gauss_strictly_stronger / varPenalty_saturated）。
    """
    std = z.reshape(-1, z.size(-1)).std(dim=0, unbiased=False)
    return ((std ** 2 - 1.0) ** 2).mean()


def gram_anchor_loss(z_student, z_teacher):
    """DINOv3 Gram-anchoring（§3.4）：锚定 dense 特征 Gram 矩阵到冻结 Gram teacher。

    L_Gram = ‖X_S·X_Sᵀ − X_G·X_Gᵀ‖²_F，X 为 L2 归一化的 dense 特征（[B,N,d] → [B,N,N] Gram）。
    z_teacher 来自冻结锚点（no_grad）；塌缩时学生 Gram 退化为全 1（秩 1），与非塌缩 teacher
    的 Gram 差异 > 0，故把学生从塌缩推开。
    """
    xs = F.normalize(z_student, dim=-1)
    xg = F.normalize(z_teacher, dim=-1)
    gs = torch.bmm(xs, xs.transpose(1, 2))
    gg = torch.bmm(xg, xg.transpose(1, 2))
    return F.mse_loss(gs, gg)


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
    if cfg.alignment_method in ("dino", "dinov3"):
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


def pac_anchor_prob(step, total_steps, cfg):
    """PAC 锚模态采样概率（§3.4 式 21）：p_anchor(t)=p_inf+(p_0-p_inf)·σ(-(t-t0)/τ)。

    单一事实源（train.py 日志与 losses.py 课程门控共用）：t<t0 时锚概率≈p0
    （先由锚模态定 Z），t→1 时降到 pinf=1/K（两模态均衡）。
    """
    t = step / max(1, total_steps)
    p0, pinf, tau, t0 = cfg.pac_anchor_p0, cfg.pac_anchor_pinf, cfg.pac_tau, cfg.pac_t0
    return pinf + (p0 - pinf) * float(torch.sigmoid(torch.tensor(-(t - t0) / tau)))


def curriculum_stage(step, total_steps, cfg):
    """§3.4.2 阶段标签：S1 写/建 Z → S2 压/ρ → S3 推理/Ψ 改端。

    S4（生成/Flow）是主训练结束后独立的 train_flow_stage，不在此主预训阶段内。
    curriculum_enabled=False 时返回 "off"（旁路，等价旧行为）；缺步数信息时按后期处理。
    """
    if not cfg.curriculum_enabled:
        return "off"
    if step is None or total_steps is None:
        return "S3"
    t = step / max(1, total_steps)
    if t < cfg.curriculum_s1_end:
        return "S1"
    if t < cfg.curriculum_s3_start:
        return "S2"
    return "S3"


def curriculum_gates(step, total_steps, cfg):
    """§3.4.2 平稳性门控：返回各损失项的门控系数（curriculum_enabled=False 时全 1.0）。

    数学根基：Curriculum.lean 定理 1（目标漂移 ⟹ 最优解非 Cauchy 不收敛）
    + 定理 5（阶段适定性链 S1→S2→S3→S4）：在上游表征 ε-平稳前，把依赖它的下游目标门控为 0。
      - flow：flow_in_pretrain=False 时主预训不计 flow（Flow 仅在冻结 E 的 S4 训练）；
      - psi_change（L_consist/L_gen）：t<s3_start 时为 0（Z_tgt 未平稳，改端无稳定最优解）；
      - psi_read（readout/stop）：S1/S2 低权重 curriculum_psi_read_scale（学拟合 Z 的头，薄接口先接入）；
      - psi_reg（var/h_var/h_cov）：S1/S2 权重 curriculum_psi_reg_scale（纯正则、固定阈值、目标不漂移，无需薄接入）；
      - rate：t<rho_start 时码率损失权重 0（前端/主干未平稳时 ρ 记忆不适定，warmup）；
      - anchor/new：PAC p_anchor 缩放锚/新模态损失权重（损失权重门控近似单模态投喂，见 README 问题 14）。
    """
    if not cfg.curriculum_enabled:
        return dict(flow=1.0, psi_read=1.0, psi_reg=1.0, psi_change=1.0, rate=1.0, anchor=1.0, new=1.0)
    if step is None or total_steps is None:
        t = 1.0  # 评估/缺步数：按已收敛处理，不门控
    else:
        t = step / max(1, total_steps)
    flow_gate = 1.0 if cfg.flow_in_pretrain else 0.0
    before_s3 = t < cfg.curriculum_s3_start
    psi_read = cfg.curriculum_psi_read_scale if before_s3 else 1.0
    psi_reg = cfg.curriculum_psi_reg_scale if before_s3 else 1.0
    psi_change = 0.0 if before_s3 else cfg.curriculum_psi_change_scale
    rate_gate = 0.0 if t < cfg.curriculum_rho_start else 1.0
    if cfg.pac_enabled:
        if step is None or total_steps is None:
            p = cfg.pac_anchor_pinf
        else:
            p = pac_anchor_prob(step, total_steps, cfg)
        # 期望梯度：锚模态项 ×p、新模态项 ×(1-p)；p→1 时新模态梯度消失（先定 Z），p→0.5 时均衡。
        anchor_gate, new_gate = p, 1.0 - p
    else:
        anchor_gate = new_gate = 1.0
    return dict(flow=flow_gate, psi_read=psi_read, psi_reg=psi_reg, psi_change=psi_change,
                rate=rate_gate, anchor=anchor_gate, new=new_gate)


def psi_losses(model, img_z, txt_z, img_ctx, txt_ctx, img_tgt, txt_tgt):
    """PSCO 算子（§3.7.3 + §3.7）损失：防塌缩五道 + 推理两目标。

    - readout（防线 2）：读出头锚定 Z* 重建输入 Z（塌成常数则重建不了）；
    - var（防线 3）：方差正则，防维度塌缩；
    - h_var/h_cov（防线 4）：h=H(Z) 的信息量约束（VICReg 方差 + 协方差），防 h 塌缩为常数
      使算子退化为常值映射；
    - nontriv（防线 5）：轨迹非平凡性，瞬态段首步位移下界，防算子首步跳不动点；
    - stop（§3.7 L_stop）：不动点/停止正则，Ψ_h(Z_clean)≈Z_clean，读端 g_change→0 退化为恒等；
    - consist（§3.7 L_consist）：路径无关一致性，Ψ_h(Z_ctx)≈Z_tgt，改端把上下文变换到目标。

    返回按模态拆分的 dict；供 forward_losses 做 PAC 模态门控与课程阶段门控
    （读端薄接口 readout/stop 属 psi_read 门控，纯正则 var/h_var/h_cov 属 psi_reg 门控，改端 consist/gen/nontriv 属 psi_change 门控）。
    """
    out = {}
    img_star = model.psi_layer(img_z)
    txt_star = model.psi_layer(txt_z)
    out["readout_i"] = (model.psi_layer.readout(img_star) - img_z).pow(2).mean()
    out["readout_t"] = (model.psi_layer.readout(txt_star) - txt_z).pow(2).mean()
    out["var_i"] = variance_loss(img_star)
    out["var_t"] = variance_loss(txt_star)
    # §3.7 L_stop：干净 Z 已是不动点（读端恒等，改分量 g_change→0）
    out["stop_i"] = (img_star - img_z).pow(2).mean()
    out["stop_t"] = (txt_star - txt_z).pow(2).mean()
    # §3.7.3 防线 4：h 信息量约束（VICReg 方差 + 协方差，防 h 塌缩为常数 → 算子退化为常值映射）
    h_i = model.psi_layer.infer_h(img_z)
    h_t = model.psi_layer.infer_h(txt_z)
    out["h_var_i"] = variance_loss(h_i)
    out["h_var_t"] = variance_loss(h_t)
    out["h_cov_i"] = covariance_loss(h_i)
    out["h_cov_t"] = covariance_loss(h_t)
    # §3.7 L_consist：算子把掩码上下文变换到干净目标（改端，改分量 g_change→1）
    img_ctx_star = model.psi_layer(img_ctx)
    txt_ctx_star = model.psi_layer(txt_ctx)
    out["consist_i"] = (img_ctx_star - img_tgt).pow(2).mean()
    out["consist_t"] = (txt_ctx_star - txt_tgt).pow(2).mean()
    # §3.7.3 防线 5：轨迹非平凡性（改端瞬态段首步位移下界，防算子首步跳不动点）
    disp_i = model.psi_layer.first_step_disp(img_ctx)
    disp_t = model.psi_layer.first_step_disp(txt_ctx)
    out["nontriv_i"] = F.relu(model.cfg.psi_nontriv_eps - disp_i)
    out["nontriv_t"] = F.relu(model.cfg.psi_nontriv_eps - disp_t)
    out["img_ctx_star"] = img_ctx_star
    out["txt_ctx_star"] = txt_ctx_star
    return out


def maybe_autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_losses(model, img, txt, pad_id, mask_id, step=None, total_steps=None):
    """跑通「编码 → 对齐/推理 → 解码」，返回 (总损失, 分项损失 dict, 诊断统计 dict)。

    诊断统计（第三项）不参与反传，仅为 TensorBoard 观测服务：ρ 保留率、潜空间 std/均值、
    跨模态余弦、交叉掩码比例、文本重建准确率，用于验证 math.md 核心命题是否成立。
    未启用的模块对应键不出现（而非写 0），避免曲线噪声。

    step/total_steps 供交叉掩码递进 ρ_mask(t) 与 PAC 锚概率调度使用；None 时取中值。
    """
    cfg = model.cfg
    batch = img.size(0)
    device = img.device

    # —— 课程门控（§3.4.2）：上游 ε-平稳前下游目标权重置 0；curriculum_enabled=False 时全 1 ——
    gates = curriculum_gates(step, total_steps, cfg)
    anchor_is_img = cfg.anchor_modality == "image"
    # §3.4.2 先写后压：S1（t<curriculum_rho_start）时 ρ 压缩休眠（不丢 token、不产生码率信号），
    # S2 起激活（丢 token + 码率 warmup）。curriculum_enabled=False 时 gates["rate"]=1 → 恒激活（旧行为）。
    rho_active = gates["rate"] > 0.0

    # —— 编码（在线，干净 + 掩码）——
    img_cls, img_tokens = model.image_online(img, rho_active=rho_active)
    txt_cls, txt_tokens = model.text_online(txt, rho_active=rho_active)
    # 码率信号 b_t（§2.6 式 665）：取自在线编码器干净输入，表示编码完整输入所需码率。
    # 必须在掩码 forward（下面）覆盖 last_b 之前捕获。
    img_b = model.image_online.last_b
    txt_b = model.text_online.last_b

    img_mask = (
        block_mask_patches(batch, cfg.n_patches, cfg.mask_ratio, device, cfg.block_mask_num)
        if cfg.block_mask else mask_patches(batch, cfg.n_patches, cfg.mask_ratio, device)
    )
    txt_masked = mask_text(txt, mask_id, cfg.mask_ratio, device)
    _, img_ctx = model.image_online(img, mask=img_mask, rho_active=rho_active)
    _, txt_ctx = model.text_online(txt_masked, rho_active=rho_active)

    # —— 目标（EMA、无梯度）——
    with torch.no_grad():
        _, img_tgt = model.image_target(img, rho_active=rho_active)
        _, txt_tgt = model.text_target(txt, rho_active=rho_active)
        pool_img_tgt = img_tgt.mean(dim=1)
        pool_txt_tgt = txt_tgt.mean(dim=1)
        # 冻结锚点 f₀（§2.2.4(d)）：cls_readout 与 Gram-anchoring 的不漂移目标
        if getattr(model, "anchor_image", None) is not None:
            _, img_anchor = model.anchor_image(img, rho_active=False)
            _, txt_anchor = model.anchor_text(txt, rho_active=False)
        else:
            img_anchor, txt_anchor = img_tgt, txt_tgt
        pool_img_anchor = img_anchor.mean(dim=1)
        pool_txt_anchor = txt_anchor.mean(dim=1)

    def mscale(img_term, txt_term):
        """按 PAC p_anchor 缩放锚/新模态损失（损失权重门控近似单模态投喂，README 问题 14）。"""
        a = gates["anchor"] if anchor_is_img else gates["new"]
        n = gates["new"] if anchor_is_img else gates["anchor"]
        return a * img_term + n * txt_term

    # —— 推理：JEPA 潜预测（按模态拆分，供 mscale 门控）——
    img_pred = model.image_predictor(img_ctx)
    txt_pred = model.text_predictor(txt_ctx)
    jepa_i = cosine_loss(img_pred, img_tgt)
    jepa_t = cosine_loss(txt_pred, txt_tgt)

    # —— 解码：自重建 ——
    img_self = model.image_decoder(img_tokens)
    txt_self = model.text_decoder(txt_tokens)
    recon_img_self = F.mse_loss(img_self, img)
    recon_txt_self = F.cross_entropy(
        txt_self.reshape(-1, txt_self.size(-1)), txt.reshape(-1), ignore_index=pad_id
    )

    # —— 解码：硬对齐（双向重建，穿过同一个 Z）；按目标模态归属 ——
    img_cross = model.image_decoder(txt_tokens)   # 文本 token → 图像
    txt_cross = model.text_decoder(img_tokens)    # 图像 token → 文本
    recon_img_cross = F.mse_loss(img_cross, img)
    recon_txt_cross = F.cross_entropy(
        txt_cross.reshape(-1, txt_cross.size(-1)), txt.reshape(-1), ignore_index=pad_id
    )

    # 跨模态软对齐：锚/新共同参与（正是「新模态向锚对齐」的信号），不按 p_anchor 门控
    align = alignment_loss(model, img_cls, txt_cls)
    var_i = variance_loss(img_tokens)
    var_t = variance_loss(txt_tokens)
    # —— token 序列 Z 的协方差去相关（§2.2.4(c) 补足）：原协方差只作用在对齐投影空间（CLS），
    #    而 token 序列（重建 + JEPA 的核心潜变量）只受方差约束、无去相关，仍可低秩塌缩。
    #    方差管「每个维度有足够方差」，协方差管「维度间不冗余」，二者合起来才是完整 VICReg。
    cov_i = covariance_loss(img_tokens.reshape(-1, img_tokens.size(-1)))
    cov_t = covariance_loss(txt_tokens.reshape(-1, txt_tokens.size(-1)))
    # —— 对齐投影空间反塌缩（VICReg 方差 + 协方差）：补足 BYOL 余弦 + LayerNorm 无防塌缩的缺口。
    #    math.md §2.2.4 ⚠️ 已指出「余弦 + EMA 不能排除塌缩」；CLS 只受对齐损失监督，故需在此显式正则。
    align_var = torch.zeros((), device=device)
    align_cov = torch.zeros((), device=device)
    align_gauss = torch.zeros((), device=device)
    if model.align is not None or cfg.alignment_method == "infonce":
        align_rep_i = model.embed(img_cls)
        align_rep_t = model.embed(txt_cls)
        align_var = 0.5 * (variance_loss(align_rep_i) + variance_loss(align_rep_t))
        align_cov = 0.5 * (covariance_loss(align_rep_i) + covariance_loss(align_rep_t))
        align_gauss = 0.5 * (gauss_penalty(align_rep_i) + gauss_penalty(align_rep_t))
    # —— CLS 读出头锚定（§3.7.3 防线 2）：CLS 预测冻结锚点 f₀ 的池化 token（不漂移），防 CLS 退化塌缩 ——
    cls_readout_i = F.mse_loss(model.cls_readout(img_cls), pool_img_anchor)
    cls_readout_t = F.mse_loss(model.cls_readout(txt_cls), pool_txt_anchor)
    # —— DINOv3 Gram-anchoring（§3.4）：dense 特征 Gram 锚到冻结 Gram teacher（anchor f₀）——
    gram_i = gram_t = torch.zeros((), device=device)
    if cfg.alignment_method == "dinov3" and getattr(model, "anchor_image", None) is not None:
        gram_i = gram_anchor_loss(img_tokens, img_anchor)
        gram_t = gram_anchor_loss(txt_tokens, txt_anchor)
    rate = torch.zeros((), device=device)
    if cfg.rho_compress and img_b is not None and txt_b is not None:
        rate = 0.5 * (img_b.mean() + txt_b.mean())

    # —— 交叉掩码（§3.2 式 18）：本模态剩余 + 其他模态联合预测被掩部分，防模态坍缩 ——
    cross_pred_i = torch.zeros((), device=device)
    cross_pred_t = torch.zeros((), device=device)
    cm_ratio = None  # 实际使用的交叉掩码比例（供诊断日志，验证递进调度）
    if cfg.cross_mask and model.cross_mask_predictor is not None:
        cm_ratio = cross_mask_ratio(
            step, total_steps, cfg.cross_mask_ratio_start, cfg.cross_mask_ratio_end
        )
        cm_img_mask = mask_patches(batch, cfg.n_patches, cm_ratio, device)
        cm_txt_masked = mask_text(txt, mask_id, cm_ratio, device)
        _, cm_img_ctx = model.image_online(img, mask=cm_img_mask, rho_active=rho_active)
        _, cm_txt_ctx = model.text_online(cm_txt_masked, rho_active=rho_active)
        cm_pred_img, cm_pred_txt = model.cross_mask_predictor(cm_img_ctx, cm_txt_ctx)
        cross_pred_i = cosine_loss(cm_pred_img, img_tgt)
        cross_pred_t = cosine_loss(cm_pred_txt, txt_tgt)

    # —— PSCO 算子（§3.7.3 + §3.7）：防塌缩五道 + 推理两目标（L_stop / L_consist）——
    psi_readout_i = psi_readout_t = torch.zeros((), device=device)
    psi_var_i = psi_var_t = torch.zeros((), device=device)
    psi_stop_i = psi_stop_t = torch.zeros((), device=device)
    psi_consist_i = psi_consist_t = torch.zeros((), device=device)
    psi_gen_i = psi_gen_t = torch.zeros((), device=device)
    psi_h_var_i = psi_h_var_t = torch.zeros((), device=device)
    psi_h_cov_i = psi_h_cov_t = torch.zeros((), device=device)
    psi_nontriv_i = psi_nontriv_t = torch.zeros((), device=device)
    if cfg.psi_enabled and model.psi_layer is not None:
        p = psi_losses(model, img_tokens, txt_tokens, img_ctx, txt_ctx, img_tgt, txt_tgt)
        psi_readout_i, psi_readout_t = p["readout_i"], p["readout_t"]
        psi_var_i, psi_var_t = p["var_i"], p["var_t"]
        psi_stop_i, psi_stop_t = p["stop_i"], p["stop_t"]
        psi_consist_i, psi_consist_t = p["consist_i"], p["consist_t"]
        psi_h_var_i, psi_h_var_t = p["h_var_i"], p["h_var_t"]
        psi_h_cov_i, psi_h_cov_t = p["h_cov_i"], p["h_cov_t"]
        psi_nontriv_i, psi_nontriv_t = p["nontriv_i"], p["nontriv_t"]
        img_ctx_star, txt_ctx_star = p["img_ctx_star"], p["txt_ctx_star"]
        # §3.9 生成 = D(Ψ(Z))：把算子不动点 Z*=Ψ(Z_ctx)（掩码上下文的完成）解码回目标
        img_gen = model.image_decoder(img_ctx_star)
        txt_gen = model.text_decoder(txt_ctx_star)
        psi_gen_i = F.mse_loss(img_gen, img)
        psi_gen_t = F.cross_entropy(
            txt_gen.reshape(-1, txt_gen.size(-1)), txt.reshape(-1), ignore_index=pad_id
        )

    # —— Flow Matching（§4.1.2 + §4.1.8）：curriculum 门控（flow_in_pretrain=False 时主预训不计）——
    flow_i = flow_t = torch.zeros((), device=device)
    flow_cond_i = flow_cond_t = torch.zeros((), device=device)
    if cfg.flow_enabled and model.velocity is not None and gates["flow"] > 0:
        flow_i = flow_loss(model.velocity, img_tgt)
        flow_t = flow_loss(model.velocity, txt_tgt)
        if cfg.flow_cond_enabled:
            # 条件流（§4.1.8）：text→image（目标=图像）+ image→text（目标=文本），按目标模态归属
            flow_cond_i = conditional_flow_loss(model.velocity, txt_tgt, img_tgt)
            flow_cond_t = conditional_flow_loss(model.velocity, img_tgt, txt_tgt)

    # —— 组装总损失：PAC 模态门控（mscale）× 课程阶段门控（gates）——
    total = (
        cfg.jepa_weight * mscale(jepa_i, jepa_t)
        + cfg.recon_weight * mscale(recon_img_self, recon_txt_self)
        + cfg.cross_weight * mscale(recon_img_cross, recon_txt_cross)
        + cfg.align_weight * align
        + cfg.var_weight * mscale(var_i, var_t)
        + cfg.cov_weight * mscale(cov_i, cov_t)
        + cfg.align_var_weight * align_var
        + cfg.align_cov_weight * align_cov
        + cfg.align_gauss_weight * align_gauss
        + cfg.gram_anchor_weight * mscale(gram_i, gram_t)
        + cfg.cls_readout_weight * mscale(cls_readout_i, cls_readout_t)
        + cfg.rho_rate_weight * rate * gates["rate"]
        + cfg.cross_mask_weight * mscale(cross_pred_i, cross_pred_t)
        + cfg.psi_readout_weight * mscale(psi_readout_i, psi_readout_t) * gates["psi_read"]
        + cfg.psi_stop_weight * mscale(psi_stop_i, psi_stop_t) * gates["psi_read"]
        + cfg.psi_var_weight * mscale(psi_var_i, psi_var_t) * gates["psi_reg"]
        + cfg.psi_h_var_weight * mscale(psi_h_var_i, psi_h_var_t) * gates["psi_reg"]
        + cfg.psi_h_cov_weight * mscale(psi_h_cov_i, psi_h_cov_t) * gates["psi_reg"]
        + cfg.psi_consist_weight * mscale(psi_consist_i, psi_consist_t) * gates["psi_change"]
        + cfg.psi_nontriv_weight * mscale(psi_nontriv_i, psi_nontriv_t) * gates["psi_change"]
        + cfg.psi_gen_weight * mscale(psi_gen_i, psi_gen_t) * gates["psi_change"]
        + cfg.flow_weight * mscale(flow_i, flow_t)
        + cfg.flow_cond_weight * mscale(flow_cond_i, flow_cond_t)
    )
    parts = {
        "jepa": jepa_i + jepa_t, "recon_img": recon_img_self, "recon_txt": recon_txt_self,
        "cross_img": recon_img_cross, "cross_txt": recon_txt_cross,
        "align": align, "var": var_i + var_t, "cov": cov_i + cov_t, "rate": rate,
        "align_var": align_var, "align_cov": align_cov,
        "align_gauss": align_gauss,
        "gram_anchor": gram_i + gram_t,
        "cls_readout": cls_readout_i + cls_readout_t,
        "cross_pred": cross_pred_i + cross_pred_t,
        "psi_readout": psi_readout_i + psi_readout_t,
        "psi_var": psi_var_i + psi_var_t,
        "psi_h_var": (psi_h_var_i + psi_h_var_t) * gates["psi_reg"],
        "psi_h_cov": (psi_h_cov_i + psi_h_cov_t) * gates["psi_reg"],
        "psi_stop": psi_stop_i + psi_stop_t,
        # 改端（consist/nontriv/gen）记入阶段门控后的值：S1/S2 为 0，S3 起非 0（课程可观测）
        "psi_consist": (psi_consist_i + psi_consist_t) * gates["psi_change"],
        "psi_nontriv": (psi_nontriv_i + psi_nontriv_t) * gates["psi_change"],
        "psi_gen": (psi_gen_i + psi_gen_t) * gates["psi_change"],
        "flow": (flow_i + flow_t) * gates["flow"],
        "flow_cond": (flow_cond_i + flow_cond_t) * gates["flow"],
    }

    # —— 诊断统计（不参与反传，仅 TensorBoard 观测；验证 math.md 核心命题是否成立）——
    stats = {}
    # 1) ρ 压缩实际保留率：验证「压缩率随记忆容量自动平衡」是否发生。
    #    rate 下降但 keep_ratio 不变 ⇒ 码率正则在空转（记忆没学会吸收冗余）。
    if cfg.rho_compress:
        ik = getattr(model.image_online, "last_keep", None)
        tk = getattr(model.text_online, "last_keep", None)
        if ik is not None:
            stats["keep_ratio_img"] = ik.float().mean().detach()
        if tk is not None:
            stats["keep_ratio_txt"] = tk.float().mean().detach()
    # 2) 潜空间健康度：var loss 只惩罚 std<1，这里记录实际 std 与均值幅度，
    #    第一时间发现维度塌缩（std→0）或均值漂移（|mean| 塌到 0），不等 loss 发散。
    stats["z_std_img"] = img_tokens.detach().std(dim=(0, 1)).mean()
    stats["z_std_txt"] = txt_tokens.detach().std(dim=(0, 1)).mean()
    stats["z_mean_img"] = img_tokens.detach().abs().mean()
    stats["z_mean_txt"] = txt_tokens.detach().abs().mean()
    # 3) 跨模态对齐强度（非 loss）：proj 空间配对余弦均值，直观判断软对齐是否收敛。
    with torch.no_grad():
        ai = model.embed(img_cls.detach())
        at = model.embed(txt_cls.detach())
        zi = F.normalize(ai, dim=-1)
        zt = F.normalize(at, dim=-1)
        stats["cross_cos_sim"] = (zi * zt).sum(-1).mean()
        # 对齐投影空间实际标准差：直接观测 CLS 投影是否塌缩（align_var≈0.98 ⟹ std≈0.02）。
        stats["align_std"] = 0.5 * (ai.std(dim=0).mean() + at.std(dim=0).mean())
    # 4) 交叉掩码实际比例：递进调度 ρ_mask(t)，诊断 cross_pred 曲线变化时需对照。
    if cm_ratio is not None:
        stats["cross_mask_ratio"] = torch.tensor(cm_ratio, device=device)
    # 5) 文本自重建 token 准确率（排除 pad）：recon_txt 是 CE，曲线看不出实际解码正确率。
    with torch.no_grad():
        pred = txt_self.detach().argmax(-1)
        valid = txt != pad_id
        stats["txt_acc"] = ((pred == txt) & valid).sum().float() / valid.sum().clamp_min(1).float()

    return total, parts, stats
