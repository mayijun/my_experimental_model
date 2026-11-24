# Flow Matching + JEPA

> **状态：已归档（2026-11-24）**
> 本项目最初的两阶段「JEPA + 条件 Flow Matching」字符级生成器已作为**废案**归档，实现代码已删除。
> 思路、数学要点、实现结构与全部实测结论见 [废案.md](废案.md)；不受影响的数学推导仍保留在 [math.md](math.md)。

一句话概括核心思想：

> 潜空间是对世界规律的统一描述。图像、视频是「连续对连续」的高保真描述；**文字是同一潜空间的有损压缩——一份关于语义的摘要，而不是逐点重建。**

---

## 项目要回答的问题

1. **为什么文字天生有损？** 潜变量 $Z$ 连续、文本 $T$ 取值于有限词表 $\mathcal{V}^L$，由 $H(T) \le L\log_2|\mathcal{V}|$ 硬性封顶；而连续潜空间的量化熵随分辨率趋于无穷。于是 $H(Z \mid T) > 0$ 永远成立。
2. **为什么是「摘要」而不是随机丢弃？** 文本是在固定码率下最大化语义互信息 $I(T;Y)$ 的**确定性信息瓶颈（DIB）**变量：语义在、措辞丢。
3. **怎么实现和验证？** 曾用「JEPA 阶段 + Flow 阶段」两阶段模型落地，但生成质量未达可用标准，已归档。

---

## 归档结论（详见 [废案.md](废案.md)）

- 编码器在掩码位置上几乎不携带逐位置信息，$p(z\mid \text{mask})$ 近似退化；
- 逐位置解码器对潜变量误差零容忍（余弦 0.997 → 99.6% top-1，余弦 0.77 → 3.8%）；
- Flow 目标退化为均值回归，低维语义流与「预测器 + 语义流」组合均无增益；
- 全量 1 epoch 改造版重建 ppl 从 2.56 崩到 16226（低 $t$ 的 `sampled_recon` 是病态目标）。

---

## 当前保留的文件

| 文件 | 作用 |
|---|---|
| [`math.md`](math.md) | **数学基础**：符号表（§0）、架构总览（§1）、编码（§2，JEPA + 有损性命题）、对齐与推理（§3）、解码（§4，Flow Matching + 并行解码）、自反馈修正（§5）、可检验推论（§6） |
| [`废案.md`](废案.md) | 已归档方案的核心思路、数学要点、实现结构、实测数据与废弃原因 |
| [`multimodal_jepa.py`](multimodal_jepa.py) | 多模态 JEPA（图像 + 文本共享潜空间）独立实现，不依赖已删除的文本通道代码 |
| [`data_prepare.py`](data_prepare.py) | 下载并准备中英混合字符级语料（wikitext + 中文维基），输出 JSONL |
| [`download_full.py`](download_full.py) | 并行预下载全量语料分片，支持 HTTP Range 与断点续传 |

---

## 快速开始（多模态模型）

```bash
# 用项目自带 venv 的 Python（含 torch）
.venv/bin/python multimodal_jepa.py --preset tiny --steps 150
```

放大到 GPU：

```bash
.venv/bin/python multimodal_jepa.py --preset base --dim 512 --depth 12 --steps 2000
```

数据准备（如需重建语料）：

```bash
python data_prepare.py --out-dir data --en-train 400000 --zh-train 400000
```

---

## 一句话总结

`math.md` 讲「为什么 + 推导」，`废案.md` 讲「曾经怎么做、为什么失败」，`multimodal_jepa.py` 是仍在维护的实现，`data_prepare.py` / `download_full.py` 负责语料。
