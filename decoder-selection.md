# 解码器技术选型（备选方案）

> **范围**：解码器 $D\in\{R_I,g\}$，把共享潜变量 $Z\in\mathbb R^{L\times d}$ 渲回各模态（图像 $R_I$ / 文本 $g$）。与 [backbone-selection.md](backbone-selection.md)（编码主干选型）平行，本文只做「选型 + 备选方案记录」，不改代码。
>
> 契约基线（不可违背）：[math.md](math.md) §2.6「$Z$ 全程连续、不量化」、§4.2(c)「确定性编码器下 ELBO 不成立（不加 VAE KL）」、§4.2(a)「读出头线性打分、不 weight tying」、三档同构可伸缩。

## 1. 现状

| 解码器 | 结构 | 位置 |
|---|---|---|
| `ImageDecoder`（$R_I$） | 位置混合主干（transformer/mamba/hybrid，经 `build_backbone`）+ `Linear(d, p·p·C)` + 像素混洗逆折叠 | [decoder.py](file:///home/simon-ma/JEPA/model/decoder.py#L13-L40) |
| `TextDecoder`（$g$） | 位置混合主干 + `Linear(d, V)` 线性打分（Voronoi 超平面量化，argmax） | [decoder.py](file:///home/simon-ma/JEPA/model/decoder.py#L44-L59) |

当前图像解码是「MAE 式 ViT 解码器 + 单层线性逐 patch 回归像素」，无逐级上采样；文本解码是「并行非自回归 transformer + 线性读出头」，已支持温度采样（`sample_logits`）。

## 2. 备选方案（从零训练，均为 Z 自有潜空间，无 latent 契约错配）

| 方案 | 角色 | 关键点 | 依赖 |
|---|---|---|---|
| **VAE 式卷积解码器** | 潜 token → 图像像素（连续→连续） | reshape `[B,L,d]→[B,H/p,W/p,d]` 后转置卷积/PixelShuffle 逐级上采样；**只取 `p(x|z)` 重建，不加 KL**（§4.2(c)） | 零新依赖，CPU 冒烟可跑 |
| **DiT 式潜空间生成** | 潜空间内生成（时间条件 transformer） | 升级现有 `VelocityField`（[flow.py](file:///home/simon-ma/JEPA/model/flow.py#L43-L74)）到 AdaLN-Zero 条件、加深度；仍输出潜 token，需再经像素解码器落图 | 零新依赖（bf16/compile 可选） |
| **ViT/MAE 式解码器** | 潜 token → 逐 patch 像素 | 现状已具备，可加宽/加深、独立 `decoder_depth/heads` | 零新依赖 |

## 3. 现成模块备选

| 方案 | 来源 | 契约代价 |
|---|---|---|
| **SD VAE 解码器**（`AutoencoderKL`） | `diffusers`/HF | 期望 VAE latent `(B,4,H/8,W/8)`，须把 Z 投影/reshape 对齐并改图像损失为 latent 损失；加 `diffusers` extra |
| Chameleon / Show-o / Emu 解码头 | VLM 家族 | 绑定离散图像 token 体系，与「Z 连续」契约冲突 |

## 4. 建议落地顺序

1. 补 VAE 式卷积解码器（零依赖、冒烟可跑、替换 `Linear+fold`），提升图像重建质量；
2. 升级 `VelocityField` 到真 DiT（AdaLN-Zero），作为独立的生成增强阶段；
3. ViT/MAE 式加深加宽、独立配 `decoder_depth/heads`，作为规模档的算力/质量权衡。
