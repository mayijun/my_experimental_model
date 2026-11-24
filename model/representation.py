"""表征层 Z：JEPA 潜预测器与目标编码器 EMA 更新（math.md §2.2、§3.5）。

- LatentPredictor（P_φ）：在 Z 内把掩码上下文的潜变量预测成干净目标潜变量；
- update_target：目标编码器 EMA 跟踪在线编码器，提供 stop-gradient 目标。

PSCO（§3.7.3）的算子迭代层也挂在表征层，见 model/psco.py。
"""

import torch
import torch.nn as nn


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


@torch.no_grad()
def update_target(model, decay):
    """ξ ← μξ + (1−μ)θ，让 JEPA 目标缓慢移动（stop-gradient 防表征塌缩）。"""
    pairs = [
        (model.image_online, model.image_target),
        (model.text_online, model.text_target),
    ]
    for online, target in pairs:
        for o, t in zip(online.parameters(), target.parameters()):
            t.mul_(decay).add_(o, alpha=1.0 - decay)
