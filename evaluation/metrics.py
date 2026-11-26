"""评测：跨模态检索与双向重建（math.md §3.2 硬对齐、§3.4 软对齐）。"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, images, caption_ids, captions, tokenizer, device, num=16):
    model.eval()
    idx = torch.randperm(images.size(0))[:num]
    img = images[idx].to(device)
    txt = caption_ids[idx].to(device)

    # —— 检索：图像 [CLS] ↔ 文本 [CLS] 余弦最近邻 ——
    img_cls, img_tokens = model.image_online(img)
    txt_cls, txt_tokens = model.text_online(txt)
    zi = F.normalize(model.proj(img_cls), dim=-1)
    zt = F.normalize(model.proj(txt_cls), dim=-1)
    sim = zi @ zt.t()  # [num,num]
    i2t = (sim.argmax(dim=1) == torch.arange(num, device=device)).float().mean().item()
    t2i = (sim.argmax(dim=0) == torch.arange(num, device=device)).float().mean().item()

    # —— 双向重建 ——
    img_self = model.image_decoder(img_tokens)
    txt_self = model.text_decoder(txt_tokens).argmax(-1)
    img_cross = model.image_decoder(txt_tokens)
    txt_cross = model.text_decoder(img_tokens).argmax(-1)

    print(f"\n=== 跨模态检索（{num} 对）===")
    print(f"  图像→文本 top-1 = {i2t*100:.1f}% | 文本→图像 top-1 = {t2i*100:.1f}%")
    print(f"\n=== 重建误差（越低越好）===")
    print(f"  图像自重建 MSE = {F.mse_loss(img_self, img).item():.4f} | "
          f"文本→图像 MSE = {F.mse_loss(img_cross, img).item():.4f}")
    print(f"\n=== 样例（真值 / 文本自重建 / 图像→文本）===")
    for i in range(min(4, num)):
        truth = captions[idx[i].item()]
        self_dec = tokenizer.decode(txt_self[i].tolist())
        cross_dec = tokenizer.decode(txt_cross[i].tolist())
        print(f"  真值    : {truth}")
        print(f"  自重建  : {self_dec}")
        print(f"  图像→文本: {cross_dec}")
    return i2t, t2i


@torch.no_grad()
def evaluate_real(model, gen, tokenizer, pad_id, device, num=16):
    """真实数据的跨模态检索 + 图像→文本解码样例。"""
    model.eval()
    imgs, ids = [], []
    for _ in range(num):
        try:
            im, tids = next(gen)
        except StopIteration:
            break
        imgs.append(im)
        ids.append(tids)
    if not imgs:
        return 0.0, 0.0
    # load_checkpoint 在 CUDA 下会把权重转 bf16，输入需对齐到同一 dtype
    dtype = next(model.parameters()).dtype
    img = torch.stack(imgs).to(device=device, dtype=dtype)
    txt = torch.tensor(ids, dtype=torch.long).to(device)

    img_cls, img_tokens = model.image_online(img)
    txt_cls, _ = model.text_online(txt)
    zi = F.normalize(model.proj(img_cls), dim=-1)
    zt = F.normalize(model.proj(txt_cls), dim=-1)
    sim = zi @ zt.t()
    labels = torch.arange(len(imgs), device=device)
    i2t = (sim.argmax(dim=1) == labels).float().mean().item()
    t2i = (sim.argmax(dim=0) == labels).float().mean().item()

    txt_cross = model.text_decoder(img_tokens).argmax(-1)
    print(f"\n=== 真实数据跨模态检索（{len(imgs)} 对）===")
    print(f"  图像→文本 top-1 = {i2t*100:.1f}% | 文本→图像 top-1 = {t2i*100:.1f}%")
    print("=== 图像→文本解码样例（真值 / 预测）===")
    for i in range(min(3, len(imgs))):
        truth = tokenizer.decode(ids[i], skip_special_tokens=True)
        pred = tokenizer.decode(txt_cross[i].tolist(), skip_special_tokens=True)
        print(f"  真值: {truth}\n  预测: {pred}")
    return i2t, t2i
