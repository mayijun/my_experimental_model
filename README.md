# Flow Matching + JEPA

用「联合嵌入预测（JEPA）+ 条件 Flow Matching」在**连续潜空间里生成离散文本**的模型，以及支撑它的数学论证。

一句话概括核心思想：

> 潜空间是对世界规律的统一描述。图像、视频是「连续对连续」的高保真描述；**文字是同一潜空间的有损压缩——一份关于语义的摘要，而不是逐点重建。**

---

## 项目要回答的问题

1. **为什么文字天生有损？** 潜变量 $Z$ 是连续的，文本 $T$ 取值于有限词表 $\mathcal{V}^L$，由 $H(T) \le L\log_2|\mathcal{V}|$ 硬性封顶；而连续潜空间的量化熵随分辨率趋于无穷。于是 $H(Z \mid T) > 0$ 永远成立——给定一段文字，潜空间里总有它说不清的细节。
2. **为什么是「摘要」而不是随机丢弃？** 文本不是任意量化，而是在固定码率下最大化语义互信息 $I(T;Y)$ 的**确定性信息瓶颈（DIB）**变量：语义在、措辞丢。
3. **怎么实现和验证？** 用两阶段模型把这条文本通道落地，并在脚本上直接跑出可检验的推论。

完整的推导、命题与逐式代码对齐见 [math.md](math.md)。

---

## 扩展方向（规划中）

`math.md` §3.7–3.15 把「统一潜空间」从单一文本通道推广到多模态、闭环、训练策略、文字时间性与统一预测框架，对应 [plan](.trae/plan_mosaic_text_unified_lossy_channel.md) 的 P1–P8（尚未实现）：

- **马赛克同构（P2）**：视觉通道 $R_I$ 的最小实例，与文本通道同为多对一映射，但损失机制不同——连续降采样（分辨率）vs 离散码本（基数）。
- **一致性约束（P1）**：后验采样 $\hat Z$ 重新量化必须回到 $T$，不漂出纤维 $\mathcal F_T$。
- **描述 vs 指令（P5）**：引入算子空间 $\Omega$，把文字区分为状态投影 $T_d=g(Z)$ 与状态变换 $Z'=\Psi(Z,\omega)$。
- **跨模态对齐（P6）**：编译器/解释器框架 + 穿过 $Z$ 的双向重建硬对齐 + 跨模态交叉掩码防模态坍缩。
- **互补压缩（P7）**：视频/语音通道，$I(V;I,T)>\max(I(V;I),I(V;T))$，图片与文字互补编码视频。
- **闭环观测器（P8）**：预测-校正的卡尔曼滤波视角，创新序列白噪声作为模型充分性判据。
- **训练策略（§3.13）**：硬对齐的落地方法——增量通道冻结（防遗忘 vs stop-gradient 防塌缩）、软对齐对照（五投影机制）、渐进锚定课程（PAC）投喂、锚预训练的蒸馏预热。
- **文字时间性与表征空间（§3.14）**：CoT 是算子轨迹而非状态快照；图像/视频只能锚定已观测状态，未来/变化/规律/反事实靠 JEPA 的表征空间（预测而非重构）容纳。
- **统一预测框架（§3.15）**：重建是预测的退化特例——同一预测目标 $P_\phi(c,\omega)$ 采样不同「上下文-指令-目标」三元组即可覆盖重建/补全/未来/指令/反事实。

---

## 模型架构

两阶段潜变量生成器：

- **阶段一 · 表征学习（JEPA）**
  双编码器（在线 $E_\theta$ + EMA 目标 $E_\xi$）在掩码预测下学习连续潜表征，配合一个离散量化瓶颈（FSQ / VQ / RVQ）把潜变量压成离散码，再经线性解码器还原字符。防塌缩靠重建损失 + VICReg 方差正则。

- **阶段二 · 潜空间生成（条件 Flow Matching）**
  冻结编码器，用线性（OT）插值路径 $z_t = (1-t)z_0 + t z_1$ 训练速度场 $v_\psi(z_t,t,c)$，推理时用 Euler/Heun 积分把噪声「流」成潜变量，再量化解码成文本。生成可并行：Flow 一步到位，或 MaskGIT 式迭代并行填充。

关键设计取舍：**串行深度**是 GPU 并行的判据。自回归解码是 $O(L)$（随序列线性增长），而 Flow 是 $O(N)$（积分步数，与序列长度解耦）。这就是选择条件 Flow Matching 的核心原因，SD3 / Flux / Sora / π0 是其大规模背书。

---

## 目录结构

| 文件 | 作用 |
|---|---|
| [`flow_matching_jepa.py`](flow_matching_jepa.py) | 自包含的**教学版**模型：双编码器、量化器、Flow、MaskGIT、后验采样，单文件可跑 |
| [`pretrain.py`](pretrain.py) | **正式预训练**脚本：HuggingFace Accelerate + datasets，多卡 / bf16 / 断点续训 |
| [`data_prepare.py`](data_prepare.py) | 下载并准备中英混合字符级语料（wikitext + 中文维基），输出 JSONL |
| [`eval_generate.py`](eval_generate.py) | 加载 checkpoint 评测：重建困惑度、条件生成、多样性、重编码失真 |
| [`analyze.py`](analyze.py) | 验证 math.md §4 的两条推论（熵率上界、重编码失真 > 0） |
| [`math.md`](math.md) | 数学基础：JEPA / Flow Matching / 统一潜空间命题（§3.1–3.6）+ 多模态、闭环、训练策略、文字时间性与统一预测框架扩展（§3.7–3.15），公式与代码逐式对齐 |
| [`.trae/plan_mosaic_text_unified_lossy_channel.md`](.trae/plan_mosaic_text_unified_lossy_channel.md) | 扩展路线 P1–P8：一致性、马赛克、CFG、算子空间、跨模态对齐、互补压缩、闭环观测器 |
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

`math.md` 讲「为什么」，`flow_matching_jepa.py` 讲「最小可用实现」，`pretrain.py` + `data_prepare.py` + `eval_generate.py` 把它放大到真实语料与 GPU，`analyze.py` 把纸面上的命题变成可跑的 PASS/FAIL。
