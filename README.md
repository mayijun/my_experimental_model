# Flow Matching + JEPA

用「联合嵌入预测（JEPA）+ 条件 Flow Matching」在**连续潜空间里生成离散文本**的模型，以及支撑它的数学论证。

一句话概括核心思想：

> 潜空间是对世界规律的统一描述。图像、视频是「连续对连续」的高保真描述；**文字是同一潜空间的有损压缩——一份关于语义的摘要，而不是逐点重建。**

完整的架构、数学推导、命题与逐式代码对齐见 [math.md](math.md)。

---

## 项目要回答的问题

1. **为什么文字天生有损？** 潜变量 $Z$ 是连续的，文本 $T$ 取值于有限词表 $\mathcal{V}^L$，由 $H(T) \le L\log_2|\mathcal{V}|$ 硬性封顶；而连续潜空间的量化熵随分辨率趋于无穷。于是 $H(Z \mid T) > 0$ 永远成立——给定一段文字，潜空间里总有它说不清的细节。
2. **为什么是「摘要」而不是随机丢弃？** 文本不是任意量化，而是在固定码率下最大化语义互信息 $I(T;Y)$ 的**确定性信息瓶颈（DIB）**变量：语义在、措辞丢。
3. **怎么实现和验证？** 用「JEPA 阶段 + Flow 阶段」两阶段模型把这条文本通道落地，并在脚本上直接跑出可检验的推论。

---

## 模型架构（概览）

两阶段潜变量生成器（JEPA 阶段 → Flow 阶段）：

- **JEPA 阶段 · 表征学习**：双编码器（在线 $E_\theta$ + EMA 目标 $E_\xi$）在掩码预测下学连续潜表征，配合离散量化瓶颈（FSQ / VQ / RVQ），再经带位置混合的 Transformer 解码器还原字符；防塌缩靠重建损失 + VICReg 方差正则。
- **Flow 阶段 · 潜空间生成（条件 Flow Matching）**：冻结编码器，用线性（OT）插值路径 $z_t=(1-t)z_0+t z_1$ 训练速度场 $v_\psi(z_t,t,c)$，推理时 Euler/Heun 积分把噪声「流」成潜变量再量化解码。

关键设计取舍：**串行深度是 GPU 并行的判据**——自回归解码 $O(L)$ vs Flow $O(N)$（与序列长度解耦），SD3 / Flux / Sora / π0 是其大规模背书。

完整的架构总览、公式–模块对照与串行深度对比见 [math.md](math.md) §1、§4.3。

---

## 目录结构

| 文件 | 作用 |
|---|---|
| [`math.md`](math.md) | **架构 + 数学基础**：按「编码→对齐与推理→解码→自反馈修正」四阶段组织——符号表（§0）、架构总览（§1）、编码（§2，JEPA + 有损性命题）、对齐与推理（§3）、解码（§4，Flow Matching + 并行解码）、自反馈修正（§5）、可检验推论（§6），公式与代码逐式对齐 |
| [`flow_matching_jepa.py`](flow_matching_jepa.py) | 自包含的**教学版**模型：双编码器、量化器、Flow、MaskGIT、后验采样，单文件可跑 |
| [`pretrain.py`](pretrain.py) | **正式预训练**脚本：HuggingFace Accelerate + datasets，多卡 / bf16 / 断点续训 |
| [`data_prepare.py`](data_prepare.py) | 下载并准备中英混合字符级语料（wikitext + 中文维基），输出 JSONL |
| [`eval_generate.py`](eval_generate.py) | 加载 checkpoint 评测：重建困惑度、条件生成、多样性、重编码失真 |
| [`analyze.py`](analyze.py) | 验证 math.md §6 的两条推论（熵率上界、重编码失真 > 0） |
| [`.trae/plan_mosaic_text_unified_lossy_channel.md`](.trae/plan_mosaic_text_unified_lossy_channel.md) | 扩展路线 P1–P8（一致性、马赛克、CFG、算子空间、跨模态对齐、互补压缩、闭环观测器），对应 math.md §2.6、§3、§4 |
| [`run_pipeline.sh`](run_pipeline.sh) | 数据 → 训练 → 评测一条龙（`MODE=smoke` 冒烟 / `MODE=full` 完整） |
| [`run_pretrain_a100.sh`](run_pretrain_a100.sh) | A100 上跑 0.3B 规模预训练的现成配置 |

---

## 快速开始

### 1. 冒烟测试（几分钟跑通端到端）

教学版单文件模型，内置小语料，CPU 就能跑：

```bash
# 用项目自带 venv 的 Python（含 torch）
.venv/bin/python flow_matching_jepa.py --stage1-steps 200 --stage2-steps 500
```

常用开关：

```bash
.venv/bin/python flow_matching_jepa.py \
    --quantizer fsq \        # none | fsq | vq | rvq
    --decode flow \          # flow | maskgit
    --solver heun \          # euler | heun
    --posterior-samples 4    # T→Z 后验采样轨迹数，展示「文字背后的模糊性」
```

### 2. 验证核心推论

```bash
.venv/bin/python analyze.py --num-samples 256
```

会打印两条结论是否 PASS：

- **推论 1**：生成文本的熵率 $\hat H(T) \le L\log_2|\mathcal{V}|$；
- **推论 2**：重编码失真 $D_T = \mathbb{E}\|Z - E_\theta(T)\|^2 > 0$，即文本无法无损还原连续潜变量。

### 3. 正式预训练（GPU）

先准备中英混合语料，再训练，最后评测：

```bash
MODE=smoke bash run_pipeline.sh     # 小规模端到端
MODE=full  bash run_pipeline.sh     # 0.3B 完整预训练
```

或在 A100 上直接跑：

```bash
bash run_pretrain_a100.sh
```

依赖：`torch` + `accelerate` + `datasets`（`pip install accelerate datasets`）。

---

## 一句话总结各个文件的关系

`math.md` 讲「为什么 + 架构」，`flow_matching_jepa.py` 讲「最小可用实现」，`pretrain.py` + `data_prepare.py` + `eval_generate.py` 把它放大到真实语料与 GPU，`analyze.py` 把纸面上的命题变成可跑的 PASS/FAIL。
