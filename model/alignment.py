"""对齐头（math.md §3.4）：BYOL / DINO 的投影-预测结构，替换 InfoNCE 软对齐兜底。

- BYOL（默认）：在线投影 g_θ + 预测 q_θ 拟合目标投影 g_ξ（EMA、stop-gradient），
  无负样本、无 batch 依赖；跨模态版 = img 在线预测 txt 目标、反之亦然。
- DINO：教师 EMA + centering + 温度锐化，学生软标签交叉熵。

两者都满足 §3.4「软对齐作对照、硬对齐（双向重建）为主」的定位：只提供 [CLS] 聚合
token 的弱对齐信号，不承担防塌缩主责（主责在重建 + VICReg + 交叉掩码）。
"""

import copy

import torch
import torch.nn as nn


def _mlp(in_dim, hidden, out_dim, final_norm=True):
    """投影/预测 MLP。用 LayerNorm 替代 BYOL 原文的 BatchNorm（小 batch 更稳）。"""
    layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
    if final_norm:
        layers += [nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim)]
    else:
        layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class BYOLHead(nn.Module):
    """BYOL 跨模态自蒸馏头（§3.4）。

    forward(z) -> q_θ(g_θ(z))（在线，含梯度）
    target(z)  -> g_ξ(z)（目标，stop-gradient，EMA 跟踪 g_θ）
    损失：cosine(forward(img_cls), target(txt_cls)) + cosine(forward(txt_cls), target(img_cls))
    """

    def __init__(self, dim, proj_dim=None, hidden_dim=None, ema_decay=0.99):
        super().__init__()
        proj_dim = proj_dim or dim
        hidden_dim = hidden_dim or 4 * dim
        self.ema_decay = ema_decay
        self.projector = _mlp(dim, hidden_dim, proj_dim)
        self.predictor = _mlp(proj_dim, hidden_dim, proj_dim)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

    def forward(self, z):
        return self.predictor(self.projector(z))

    def representation(self, z):
        """在线投影器输出 g_θ(z)：检索 / 反塌缩正则共用的稳定表征（不含 predictor）。"""
        return self.projector(z)

    @torch.no_grad()
    def target(self, z):
        return self.target_projector(z)

    @torch.no_grad()
    def update_target(self, decay=None):
        decay = self.ema_decay if decay is None else decay
        for o, t in zip(self.projector.parameters(), self.target_projector.parameters()):
            t.mul_(decay).add_(o, alpha=1.0 - decay)


class DINOHead(nn.Module):
    """DINO 自蒸馏头（§3.4）：教师 EMA + centering + 温度锐化。

    forward(z) -> 学生 logits（含梯度）
    teacher(z) -> 教师 logits（EMA、centering、stop-gradient）
    损失：CE(softmax(s/τ_s), softmax(t/τ_t)) 跨模态对称，τ_s > τ_t（教师更尖）。
    """

    def __init__(self, dim, proj_dim=None, hidden_dim=None, ema_decay=0.99, center_momentum=0.9):
        super().__init__()
        proj_dim = proj_dim or dim
        hidden_dim = hidden_dim or 4 * dim
        self.ema_decay = ema_decay
        self.center_momentum = center_momentum
        self.student = _mlp(dim, hidden_dim, proj_dim, final_norm=False)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.register_buffer("center", torch.zeros(proj_dim))

    def forward(self, z):
        return self.student(z)

    def representation(self, z):
        """学生投影器输出 s_θ(z)：检索 / 反塌缩正则共用的稳定表征。"""
        return self.student(z)

    @torch.no_grad()
    def teacher_forward(self, z):
        t = self.teacher(z)
        self.center = self.center_momentum * self.center + (1.0 - self.center_momentum) * t.mean(dim=0)
        return t - self.center

    @torch.no_grad()
    def update_target(self, decay=None):
        decay = self.ema_decay if decay is None else decay
        for o, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.mul_(decay).add_(o, alpha=1.0 - decay)


def build_alignment_head(cfg):
    """按 cfg.alignment_method 返回对齐头；infonce 返回 None（沿用 model.proj）。

    dinov3 复用 DINOHead（centering + 锐化，DINOv3 的头与 DINO 同构）；其 Gram-anchoring
    是独立的 dense 正则（见 losses.gram_anchor_loss），不在此头内实现。
    """
    if cfg.alignment_method == "byol":
        return BYOLHead(cfg.dim, cfg.byol_proj_dim, cfg.byol_hidden_dim, cfg.byol_ema_decay)
    if cfg.alignment_method in ("dino", "dinov3"):
        return DINOHead(cfg.dim, cfg.byol_proj_dim, cfg.byol_hidden_dim,
                        cfg.byol_ema_decay, cfg.dino_center_momentum)
    return None
