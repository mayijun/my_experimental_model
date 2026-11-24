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

## 代码结构

```
data_prepare/    数据：语料下载/整理、合成与真实图文对、字符词表
  prepare.py      中英混合字符级语料 → JSONL
  download.py     全量语料分片并行预下载（HTTP Range + 断点续传）
  synthetic.py    合成图文对 + CharTokenizer
  hf_datasets.py  HF datasets 流式图文对 + HF tokenizer
model/           模型：编码器 → 表征层 Z → 解码器
  config.py       MJEPAConfig / PRESETS / build_config
  encoder.py      E^I、E^T（Transformer 主干）
  representation.py  P_φ 潜预测器、目标编码器 EMA
  decoder.py      R_I、g（共享双解码器）
  jepa.py         MultimodalJEPA 组装
  psco.py         PSCO 并行扫描收缩算子原型
training/        训练：损失/目标 + 训练循环 + checkpoint
  losses.py       掩码、余弦/方差/InfoNCE、forward_losses
  train.py        入口（数据装配、训练、评测）
inference/       推理：加载 checkpoint，Z 内正向/反向映射
  run.py          文本→图像、图像→文本
evaluation/      评测：跨模态检索 + 双向重建
  metrics.py      evaluate / evaluate_real
```

| 文件 | 作用 |
|---|---|
| [`math.md`](math.md) | **数学基础**：符号表（§0）、架构总览（§1）、编码（§2，JEPA + 有损性命题）、对齐与推理（§3）、解码（§4，Flow Matching + 并行解码）、自反馈修正（§5）、可检验推论（§6） |
| [`entropy_and_coding.md`](entropy_and_coding.md) | §2.6 的展开：熵论、可约损失与 token 层有损压缩 |
| [`废案.md`](废案.md) | 已归档方案的核心思路、数学要点、实现结构、实测数据与废弃原因 |

---

## 环境管理（uv）

依赖与版本锁在 [`pyproject.toml`](pyproject.toml) + [`uv.lock`](uv.lock)，环境是 `.venv`（Python 由 [`.python-version`](.python-version) 固定）。

```bash
# CPU 机器
uv sync --extra cpu --extra data --extra hf --extra train
# GPU 机器（CUDA 12.6，驱动 >= 525.60）
uv sync --extra cu126 --extra data --extra hf --extra train
# GPU 机器（CUDA 13.0，驱动 >= 580，Blackwell 等新卡）
uv sync --extra cu130 --extra data --extra hf --extra train

uv run python -m training.train --preset tiny --steps 150   # 自动同步后运行
uv add <包名>            # 新增依赖并更新锁
uv add --optional data <包名>
uv lock --upgrade        # 升级并重新锁定
```

可选依赖分组：

| 分组 | 内容 | 用途 |
|---|---|---|
| `cpu` / `cu126` / `cu130` | torch | **三选一**，由 `[tool.uv] conflicts` 保证互斥 |
| `data` | requests、pyarrow | 语料下载整理 |
| `hf` | datasets、transformers、pillow | 真实图文对 |
| `train` | accelerate | 多卡 / bf16 |

三个 torch 分组互斥，因此**不能用 `uv sync --all-extras`**（会同时启用 cpu 与 cu 而报冲突），必须显式列出所需分组。torch 锁在同一个版本 2.14.1（[`pyproject.toml`](pyproject.toml) 中 `torch==2.14.1`），CPU / GPU 环境只差 wheel 构建标签，行为可比；换 CUDA 主线（如 `cu118`、`cu128`）时改 `[project.optional-dependencies]` 与 `[[tool.uv.index]]` 的地址，再 `uv lock` 即可。

---

## 快速开始（多模态模型）

```bash
# 首次：按锁文件建环境（CPU；GPU 换成 --extra cu126 / --extra cu130）
uv sync --extra cpu --extra data --extra hf --extra train

# CPU 冒烟
uv run python -m training.train --preset tiny --steps 150
```

放大到 GPU，并落盘 checkpoint（GPU 机器先按上面的 cu126 / cu130 建环境）：

```bash
uv run python -m training.train --preset base --dim 512 --depth 12 --steps 2000 --save runs/mjepa.pt
```

推理（需要上一步的 checkpoint）：

```bash
uv run python -m inference.run --ckpt runs/mjepa.pt --text "redcircleleft" --out out.npy
uv run python -m inference.run --ckpt runs/mjepa.pt --image-npy out.npy
```

数据准备（如需重建语料）：

```bash
uv run python -m data_prepare.prepare --out-dir data --en-train 400000 --zh-train 400000
uv run python -m data_prepare.download   # 可选：先并行预下载全量分片
```

---

## 一句话总结

`math.md` 讲「为什么 + 推导」，`废案.md` 讲「曾经怎么做、为什么失败」，`model/` 是模型本体（编码器 / 表征层 / 解码器），`training/` + `inference/` + `evaluation/` 是训练、推理与评测，`data_prepare/` 负责语料。
