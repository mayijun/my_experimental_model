"""表征层 Z：JEPA 潜预测器与目标编码器 EMA 更新（math.md §2.2、§3.5）。

- LatentPredictor（P_φ）：在 Z 内把掩码上下文的潜变量预测成干净目标潜变量；
- update_target：目标编码器 EMA 跟踪在线编码器，提供 stop-gradient 目标。

PSCO（§3.7.3）的算子迭代层也挂在表征层，见 model/psco.py。
"""

import torch
import torch.nn as nn

from .encoder import TransformerBlock


class LatentPredictor(nn.Module):
    """P_φ：在 Z 内把掩码上下文的潜变量预测成干净目标潜变量（逐位置共享 MLP）。"""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, context):
        return self.net(context)  # [B,L,d] -> [B,L,d]


class CrossMaskPredictor(nn.Module):
    """跨模态掩码预测器（§3.2 式 18）：对 [img_ctx; txt_ctx] 做共享自注意力，
    使图像 token 能补文本被掩缺失、反之亦然，从而防「单模态坍缩」（命题 6）。"""

    def __init__(self, dim, heads, depth=1, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)]
        )

    def forward(self, img_ctx, txt_ctx):
        L = img_ctx.size(1)
        x = torch.cat([img_ctx, txt_ctx], dim=1)  # [B, 2L, d]
        for block in self.blocks:
            x = block(x)
        return x[:, :L], x[:, L:]


def _ema_params(online, target, decay):
    """对两个同构模块逐参数 EMA：target ← μ·target + (1−μ)·online。"""
    for o, t in zip(online.parameters(), target.parameters()):
        t.mul_(decay).add_(o, alpha=1.0 - decay)


def _ema_thin_params(online, target, decay):
    """EMA 薄接口参数（跳过共享主干 encoder.*，主干单独 EMA 一次）。"""
    for (no, o), (nt, t) in zip(online.named_parameters(), target.named_parameters()):
        if no.startswith("encoder."):
            continue
        t.mul_(decay).add_(o, alpha=1.0 - decay)


@torch.no_grad()
def update_target(model, decay):
    """ξ ← μξ + (1−μ)θ，让 JEPA 目标缓慢移动（stop-gradient 防表征塌缩）。

    share_trunk=True 时：共享主干只 EMA 一次（shared_online→shared_target），
    避免因图文都引用同一 trunk 而被 EMA 两次（decay 会变成 μ²）；薄接口各自 EMA。
    """
    if model.shared_online is not None:
        _ema_params(model.shared_online, model.shared_target, decay)
        for online, target in [(model.image_online, model.image_target),
                               (model.text_online, model.text_target)]:
            _ema_thin_params(online, target, decay)
    else:
        for online, target in [(model.image_online, model.image_target),
                               (model.text_online, model.text_target)]:
            _ema_params(online, target, decay)
    # 对齐头（BYOL/DINO）的目标投影 EMA
    align = getattr(model, "align", None)
    if align is not None and hasattr(align, "update_target"):
        align.update_target(decay)
