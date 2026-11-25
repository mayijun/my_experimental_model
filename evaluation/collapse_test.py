"""单模态坍缩实验（math.md §3.2 命题 6）：验证交叉掩码对模态坍缩的惩罚。

构造：把某一模态编码器的输出上下文强制替换为常数（模拟「输出与输入无关」的坍缩），
比较交叉掩码损失是否显著上升。

命题 6 断言：若某模态坍缩为常数，它负责补的跨模态缺失就重建失败，交叉掩码损失爆炸，
梯度会把该模态拽回非平凡。本实验用「正常 vs 坍缩」的损失对比来量化这一判据。

用法：
    .venv/bin/python -m evaluation.collapse_test
"""

import argparse

import torch

from data_prepare.synthetic import build_dataset, synthetic_batch_fn
from model import PRESETS, MultimodalJEPA, build_config
from training.losses import cosine_loss, mask_patches, mask_text


@torch.no_grad()
def cross_mask_loss(model, img, txt, mask_id, ratio=0.5, collapse_text=False):
    """计算交叉掩码损失；collapse_text=True 时把文本上下文替换为常数（模拟坍缩）。"""
    batch = img.size(0)
    device = img.device
    img_mask = mask_patches(batch, model.cfg.n_patches, ratio, device)
    txt_masked = mask_text(txt, mask_id, ratio, device)
    _, img_ctx = model.image_online(img, mask=img_mask)
    _, txt_ctx = model.text_online(txt_masked)
    if collapse_text:
        txt_ctx = torch.zeros_like(txt_ctx)  # 文本编码器坍缩为常数 0
    _, img_tgt = model.image_target(img)
    _, txt_tgt = model.text_target(txt)
    pred_img, pred_txt = model.cross_mask_predictor(img_ctx, txt_ctx)
    return (cosine_loss(pred_img, img_tgt) + cosine_loss(pred_txt, txt_tgt)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="tiny")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config(args.preset, cross_mask=True, alignment_method="byol")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, caption_ids, _, tokenizer = build_dataset(
        args.dataset_size, cfg.image_size, cfg.text_len, seed=args.seed
    )
    mask_id = tokenizer.mask_id
    batch_fn = synthetic_batch_fn(images, caption_ids, args.batch_size, device)
    img, txt = batch_fn()

    model = MultimodalJEPA(cfg, tokenizer.vocab_size).to(device)
    model.eval()

    assert model.cross_mask_predictor is not None, "交叉掩码未启用（cross_mask=True 应为真）"

    normal = cross_mask_loss(model, img, txt, mask_id, collapse_text=False)
    collapsed = cross_mask_loss(model, img, txt, mask_id, collapse_text=True)

    ratio = collapsed / (normal + 1e-8)
    print(f"交叉掩码损失 | 正常文本: {normal:.4f} | 坍缩文本(常数): {collapsed:.4f} "
          f"| 坍缩/正常 = {ratio:.2f}x")

    if collapsed > normal:
        print("PASS：文本坍缩使交叉掩码损失上升，交叉掩码能惩罚单模态坍缩（命题 6）。")
    else:
        print("FAIL：交叉掩码未惩罚坍缩，需检查跨模态交互是否建立。")


if __name__ == "__main__":
    main()
