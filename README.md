# 多模态 JEPA：共享潜空间 Z

> 潜空间是对世界规律的统一描述。图像、视频是「连续对连续」的高保真描述；**文字是同一潜空间的有损压缩——一份关于语义的摘要，而不是逐点重建。**

当前实现：图像 + 文本两个模态，各自编码进同一个连续潜空间 $Z=\mathbb R^{L\times d}$，在 $Z$ 内做 JEPA 潜预测、跨模态对齐与双向解码。全程不量化 $Z$，有损只发生在 token 层的 ρ 压缩。

---

## 项目要回答的问题

1. **为什么文字天生有损？** 潜变量 $Z$ 连续、文本 $T$ 取值于有限词表 $\mathcal{V}^L$，由 $H(T) \le L\log_2|\mathcal{V}|$ 硬性封顶；而连续潜空间的量化熵随分辨率趋于无穷。于是 $H(Z \mid T) > 0$ 永远成立。
2. **为什么是「摘要」而不是随机丢弃？** 文本是在固定码率下最大化语义互信息 $I(T;Y)$ 的**确定性信息瓶颈（DIB）**变量：语义在、措辞丢。
3. **怎么实现和验证？** 用「JEPA 掩码潜预测 + 双向重建 + 跨模态对齐」把两个模态焊进同一坐标系，并以跨模态检索、双向重建与模态坍缩实验量化验证。

---

## 代码结构

```
data_prepare/    数据：语料 / 合成图文对 / 真实图文对 / tokenizer
  prepare.py      中英混合字符级语料 → JSONL
  download.py     全量语料分片并行预下载（HTTP Range + 断点续传）
  synthetic.py    离线合成图文对 + CharTokenizer
  hf_datasets.py  HF datasets 流式图文对 + HF / LLaDA tokenizer
model/           模型：编码器 → 共享潜空间 Z → 解码器
  config.py       MJEPAConfig / PRESETS / build_config
  encoder.py      E^I、E^T；Transformer / SSM / hybrid 主干；SigLIP-2 前端；ρ 压缩
  representation.py  P_φ 潜预测器、交叉掩码预测器、目标编码器 EMA
  alignment.py    BYOL / DINO 对齐头
  decoder.py      R_I、g（共享双解码器）
  jepa.py         MultimodalJEPA 组装
  psco.py         PSCO 并行扫描与收缩算子原型（可独立运行）
training/        训练：损失 + 训练循环 + checkpoint
  losses.py       掩码、余弦/方差/InfoNCE/BYOL/DINO、交叉掩码、forward_losses
  train.py        入口（数据装配、训练、评测）
inference/       推理：加载 checkpoint，Z 内正向/反向映射
  run.py          文本→图像、图像→文本
evaluation/      评测
  metrics.py      跨模态检索 + 双向重建
  collapse_test.py  交叉掩码对模态坍缩的惩罚实验（§3.2 命题 6）
```

| 文档 | 内容 |
|---|---|
| [`math.md`](math.md) | **数学基础**：符号表（§0）、架构总览（§1）、编码（§2，JEPA + 有损性命题 + ρ 压缩 + 规模解耦）、对齐与推理（§3）、解码（§4，规划）、自反馈修正（§5）、可检验推论（§6） |
| [`backbone-selection.md`](backbone-selection.md) | 编码主干选型（transformer / mamba / hybrid）与三档规模代价 |
| [`lean-math/`](lean-math/README.md) | math.md 核心信息论与算子命题的 Lean 4 形式化验证 |

---

## 多模态架构：私有记忆 M + 共享主干 + 共享潜空间 Z

每个模态走同一条「token 层 ρ 有损压缩 → 共享主干 → 共享潜空间 Z」管线（当前已接入图像、文本）：

```
图像 x ─► M_图像(私有) ─► ρ ─► 丢/留 ─► x̃ ─► E(共享主干) ─┐
文本 x ─► M_文本(私有) ─► ρ ─► 丢/留 ─► x̃ ─► E(共享主干) ─┼─► Z(跨模态共享潜空间)
（新模态）─► M_新(私有) ─► ρ ─► 丢/留 ─► x̃ ─► E(共享主干) ─┘
```

- **私有 M**：记忆/ρ 判据（`ρ=‖v−Mk‖²`）只判「观测冗余」（模态内），每个模态一份，新模态接入 = 新增一份，纯增量；
- **共享 E**：主干编码器全模态复用（薄接口 + 厚共享主干，math.md §2.8）；
- **共享 Z**：跨模态统一发生在连续潜空间（对齐/推理共用坐标系），不在 token 层。

实现见 [encoder.py](model/encoder.py) 的 `LinearMemory` / `RhoCompressor`，配置见 `MJEPAConfig.rho_*`；推导见 math.md §2.6（ρ 压缩）与 §2.8（分层统一）。

---

## 训练阶段（渐进锚定课程 PAC）

多模态模型按 4 阶段渐进训练（math.md §3.4），每条主线对应一个硬约束：

| 阶段 | 做什么 | 冻/训 | 目的 |
|---|---|---|---|
| S0 锚预训练 | 锚模态单模态 JEPA/重建 | 训 M + E + 解码器 | 建 Z 雏形、让 M 学会 ρ 判据 |
| S1 增量对齐 | 逐个模态接入 | 冻 Z/E，训新前端+M+头 | 新模态挤进 Z，防灾难性遗忘 |
| S2 均衡软化 | 全模态联合 | 全放开，Z 低 lr | 均衡、纳新、防漂移 |
| S3 收尾 | 微调 + 下游 | 全模态微调 | 收敛 |

私有 M 需「先写后压」warmup：每模态接入先关 `rho_compress` 让主干学特征，再开让 M 学 ρ 判据。冒烟档默认关闭 PAC（`pac_enabled=False`），用 `--pac 1` 打开。

### 生成阶段（主训练 → Flow 阶段 → 自反馈，math.md §4.2(e)/§5）

在 PAC 课程之外，「世界模型 → 生成解码器 → 闭环」这条主线再拆两个独立阶段，与 V-JEPA 2「冻结世界模型、单独训视觉解码器」同构：

| 阶段 | 做什么 | 冻/训 | 入口 |
|---|---|---|---|
| 主训练（联合） | 编码→对齐→推理→解码 + PSCO 算子 + Flow 联合 | 全训练（含 v_ψ） | `--steps` |
| Flow 阶段 | p_1 静止，只训速度场 v_ψ（§4.2(e)） | 冻 E_ξ/E_θ + 解码器，只训 v_ψ，不 `update_target` | `--flow-stage-steps N` |
| 自反馈（§5，规划） | 生成→误差→状态修正闭环（式 49） | — | 未实现 |

速度场主干可配置：加法注入（冒烟档，默认）或 DiT 的 AdaLN-Zero（规模化档），用 `--flow-adaln` 切换。Flow 阶段冻结 E_ξ 保证 p_1 静止，是 Flow Matching 运输映射 well-posed 的前提。

---

## 主干与规模

- `--backbone transformer | mamba | hybrid`：短序列用注意力（SDPA，GPU 自动分派 FlashAttention），长序列切 SSM 或混合；`mamba` / `hybrid` 当前用内置对角 SSM（CPU 可跑），GPU 版 `mamba-ssm` 的接入方案见 [backbone-selection.md](backbone-selection.md)。
- `PRESETS`：冒烟三档 `tiny` / `small` / `base`（CPU 可跑），中型 `0.3b` / `1b`，大规模 `3b` / `7b`；同一套模型定义与训练入口，只靠配置伸缩。
- 图像前端 `--image-frontend patch | siglip2`：冒烟用单层卷积 patch，真实数据可用 SigLIP-2 + MLP（`hf` extra，冻结视觉塔）。
- token 层 ρ 压缩（`--rho-compress`）默认关闭，开启后按相对阈值丢冗余并可选码率正则。

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

uv run python -m training.train --preset tiny --steps 150 --tokenizer char --backbone transformer   # 自动同步后运行
uv add <包名>            # 新增依赖并更新锁
uv add --optional data <包名>
uv lock --upgrade        # 升级并重新锁定
```

可选依赖分组：

| 分组 | 内容 | 用途 |
|---|---|---|
| `cpu` / `cu126` / `cu130` | torch | **三选一**，由 `[tool.uv] conflicts` 保证互斥 |
| `data` | requests、pyarrow | 语料下载整理 |
| `hf` | datasets、transformers、pillow | 真实图文对、SigLIP-2、LLaDA tokenizer |
| `train` | accelerate | 多卡 / bf16 |

三个 torch 分组互斥，因此**不能用 `uv sync --all-extras`**（会同时启用 cpu 与 cu 而报冲突），必须显式列出所需分组。torch 锁在同一个版本 2.14.1（[`pyproject.toml`](pyproject.toml) 中 `torch==2.14.1`），CPU / GPU 环境只差 wheel 构建标签，行为可比；换 CUDA 主线（如 `cu118`、`cu128`）时改 `[project.optional-dependencies]` 与 `[[tool.uv.index]]` 的地址，再 `uv lock` 即可。

---

## 快速开始

```bash
# 首次：按锁文件建环境（CPU；GPU 换成 --extra cu126 / --extra cu130）
uv sync --extra cpu --extra data --extra hf --extra train

# CPU 冒烟：合成图文对 + 字符级 tokenizer
uv run python -m training.train --preset tiny --steps 150 --tokenizer char --backbone transformer

# 真实图文对 + LLaDA 词表 + 混合主干（需要 GPU 与 hf extra）
uv run python -m training.train --preset 0.3b --data real --tokenizer llada --backbone hybrid
```

放大规模并落盘 checkpoint：

```bash
uv run python -m training.train --preset base --dim 512 --depth 12 --steps 2000 --save runs/mjepa.pt
```

推理（需要上一步的 checkpoint）：

```bash
uv run python -m inference.run --ckpt runs/mjepa.pt --text "redcircleleft" --out out.npy
uv run python -m inference.run --ckpt runs/mjepa.pt --image-npy out.npy
```

验证与实验：

```bash
uv run python -m model.psco                 # PSCO 并行扫描 / 收敛 / 反塌缩自检
uv run python -m evaluation.collapse_test   # 模态坍缩判据：正常 vs 坍缩的交叉掩码损失
```

数据准备（如需重建语料）：

```bash
uv run python -m data_prepare.prepare --out-dir data --en-train-chars 4000000 --zh-train-chars 4000000
uv run python -m data_prepare.download      # 可选：先并行预下载全量分片
```

---

## 一句话总结

`math.md` 讲「为什么 + 推导」，`backbone-selection.md` 讲编码主干怎么选，`lean-math/` 把核心命题机器验证，`model/` 是模型本体（编码器 / 表征层 Z / 解码器），`training/` + `inference/` + `evaluation/` 是训练、推理与评测，`data_prepare/` 负责语料。
