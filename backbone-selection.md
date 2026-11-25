# Z 空间编码主干技术选型（§2.2.5 成熟化方案）

> **范围与形态**：本文只做「选型」，不改代码。针对 [math.md](file:///home/simon-ma/JEPA/math.md) §2.2.5 的**编码器序列主干**（`transformer / mamba / hybrid` 三者）给出成熟模块选型、三档（CPU 冒烟 / GPU 中型 / scaling）参数量·显存·算力代价、以及「如何通过配置切换」。
>
> 契约基线（选型不可违背）：[math.md](file:///home/simon-ma/JEPA/math.md) §2.6「$Z$ 全程连续、不量化」、§2.2.4 / §3.7.3「反塌缩 + 对角线 SSM 谱夹层」、§2.2.7「$d{:}d_{\text{hidden}}=1{:}8$ 解耦」、§4.5「通路 / 关系分离」。任何候选都只是 $E$ 的内部实现细节，对外接口 $x\mapsto Z\in\mathbb R^{L\times d}$ 不变（§2.2.5 原文结论）。

---

## 1. 结论总表

| 主干 | 成熟模块（GPU） | CPU 冒烟回退 | 是否新依赖 | 契约风险点 |
|---|---|---|---|---|
| `transformer` | `F.scaled_dot_product_attention`（已在用，GPU 自动分派 FlashAttention-2 / Mem-Efficient）+ `torch.compile` | 同一 SDPA 原生回退 | 无（SDPA 是 torch 内置） | 无；短序列默认 |
| `mamba` | `mamba-ssm`（`Mamba2`，选择性扫描 CUDA kernel） | 手写对角 SSM（[encoder.py](file:///home/simon-ma/JEPA/model/encoder.py) 的 `SSMBlock`）+ [psco.py](file:///home/simon-ma/JEPA/model/psco.py) 的 `associative_scan` | 是，放 `mamba` extra | 谱夹层需经 ZOH/Δ 重映射，不能直接套 `sigmoid(a_raw)` 钳法 |
| `hybrid` | SDPA（前 `attn_layers` 层）+ `mamba-ssm`（其余层） | SDPA + 手写对角 SSM | 是（仅 `mamba` 部分） | 同上（仅 SSM 段） |

**一句话结论**：短序列（冒烟 tiny/small/base，$L=16/64$）默认 `transformer`（SDPA 已成熟，无需再引入第三方）；长序列（$L\gtrsim 10^3$，0.3B–7B 及更大）切 `hybrid` 或 `mamba`，SSM 段用 `mamba-ssm`（GPU）、CPU 冒烟保留手写对角 SSM 作回退。**不引入** HF `transformers` 作为 from-scratch 主干。

---

## 2. 选型约束：主干必须满足的 math.md 契约

选型判断依据以下五条契约（[math.md](file:///home/simon-ma/JEPA/math.md) §2.2.5 / §2.6 / §2.2.7 / §3.7.3）：

- **C1（$Z$ 连续）**：主干输出必须是连续潜向量 $Z\in\mathbb R^{L\times d}$，主干内不得有任何 VQ / FSQ / 离散化。有损只发生在 token 层（§2.6）。
- **C2（反塌缩）**：重建损失（主力）+ VICReg 方差正则（[losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `variance_loss`）是表征层防塌缩的两道硬约束（§2.2.4）。
- **C3（对角线 SSM 谱夹层）**：SSM 族的收缩系数 $\tilde a_h\in(\delta,\,1-\gamma)$，上界 $<1$ 保收敛、下界 $>\delta$ 防「一步塌成一点」（§3.7.3）。这是**自研** SSM 的硬契约，成熟 Mamba 需经 ZOH/Δ 离散化间接保证（见 §4.3）。
- **C4（$d{:}d_{\text{hidden}}=1{:}8$）**：大规模下主干宽 $d_{\text{hidden}}$、潜空间窄 $d$（$r=8$，可调 $[4,16]$）；冒烟三档绑定（`dim` 既是主干宽度又是潜维度）（§2.2.7）。
- **C5（通路 / 关系分离）**：前端是纯通路（逐位置 $O(L)$），关系由主干内的成对交互承载（§4.5）。主干选型即「关系建模器」的选型。

---

## 3. 候选一：Transformer（位置混合注意力）

### 3.1 现状

[encoder.py](file:///home/simon-ma/JEPA/model/encoder.py#L14-L72) 的 `SelfAttention` / `TransformerBlock` / `TransformerEncoder` 是「薄封装」：

- `SelfAttention` 用 `F.scaled_dot_product_attention`（[encoder.py](file:///home/simon-ma/JEPA/model/encoder.py#L37-L37)）——这**就是**成熟内核：GPU 上自动分派 FlashAttention-2 / Memory-Efficient Attention，CPU 回退原生实现，注意力公式、输出形状、参数量级均不变。
- `TransformerBlock` 是 Pre-LN 残差 + 多头注意力 + 4× FFN，与业界主干同构。

### 3.2 成熟模块选型

| 备选 | 是否采用 | 理由 |
|---|---|---|
| `F.scaled_dot_product_attention`（当前） | ✅ 保留 | 已是 torch 内置的成熟 attention 内核，无需再包一层 |
| `flash-attn`（`flash_attn` 显式 FlashAttention-2/3） | ⚪ 可选 | 仅当需要 FA-3 / 手动控制 kernel 时；SDPA 默认已覆盖 FA-2，多数场景无收益 |
| `torch.compile` | ✅ 建议 | 融合逐元素 op、生成 Triton kernel，无新依赖（torch 自带），第 1 档纯 CPU 同样可用 |
| HF `transformers`（`BertModel`/`GPT2Model` 等） | ❌ 不用 | 面向**预训练权重**设计，自带预训练头 / 位置假设；本框架是 from-scratch 掩码潜预测 + 硬对齐，复用不到其权重，反而引入调用约定耦合 |

### 3.3 结论

Transformer 主干的「成熟化」**不是换库**，而是：① 保持 SDPA 内核（已成熟）；② 加 `torch.compile` 可选开关；③ 明确 HF 只用于**前端/词表**（SigLIP-2 视觉前端、LLaDA tokenizer 已在用），不用于主干。短序列（$L<d_{\text{hidden}}$）下 FFN 项 $O(Ld^2)$ 主导，attention 的 $O(L^2d)$ 未成瓶颈，Transformer 常数因子最优、实现最成熟，是冒烟档与中型档的默认。

---

## 4. 候选二：Mamba / SSM（选择性扫描）

### 4.1 现状：手写「对角线性递推」≠ 真 Mamba

[encoder.py](file:///home/simon-ma/JEPA/model/encoder.py#L75-L106) 的 `SSMBlock` 是**简化对角线性递推**：

$$z_{t+1}=a_t\odot z_t+b_t,\qquad a_t=\delta+(1-\gamma-\delta)\,\sigma(a_{\text{raw}}),$$

用 [psco.py](file:///home/simon-ma/JEPA/model/psco.py#L52-L74) 的 `associative_scan` 做并行前缀扫描（$O(\log L)$ 深度）。它**不是** Mamba 的选择性扫描——没有输入依赖的 $B,C,\Delta$，没有 ZOH 离散化、没有 `causal-conv1d`。它是满足 C1/C3 契约的**自研原型**，用于验证「谱夹层 + 反塌缩」的正确性，而非性能方案。

### 4.2 成熟模块选型：`mamba-ssm`

| 备选 | 是否采用 | 理由 |
|---|---|---|
| `mamba-ssm`（`Mamba` / `Mamba2`） | ✅ GPU 采用 | 选择性扫描官方实现，$O(L)$ 时间、$O(1)$ 状态，训练用关联扫描并行、$O(\log L)$ 深度——正是 §4.3 判据要求的原语 |
| `causal-conv1d` | ✅ 随 `mamba-ssm` | `mamba-ssm` 依赖，提供关联扫描 CUDA kernel |
| 手写对角 SSM（当前 `SSMBlock`） | ✅ CPU 回退 | 冒烟档无 GPU 时保留，满足 C1/C3，但不具选择性 |

### 4.3 谱夹层契约与真 Mamba 的差异（必须标注）

这是本框架「仅当成熟模块无法满足 math.md 契约时才自研」的关键点：

- **手写 `SSMBlock`** 直接参数化 $\tilde a=\delta+(1-\gamma-\delta)\sigma(a_{\text{raw}})$，逐维钳进 $(\delta,1-\gamma)$，谱夹层**逐字成立**（对应 [psco.py](file:///home/simon-ma/JEPA/model/psco.py#L25-L33) 的 `make_contractive_a`）。
- **真 Mamba** 的 $A$ 是**输入依赖**（selective），谱半径不直接受控；收缩需经 ZOH 离散化 $\bar A=\exp(\Delta A)$ 与 $\Delta$ 的上下界重映射来**间接**保证 $\rho(\bar A)<1$。不能把 `sigmoid(a_raw)` 的钳法直接套到 Mamba 上。

**结论**：`mamba-ssm` 满足 C1（输出连续）、C5（线性复杂度），但 C3 的谱夹层需在接入层额外用 $\Delta$ 重参数化 + 谱正则 $\mathcal L_{\text{spec}}=\operatorname{relu}(\|\bar A\|_2-(1-\gamma))^2$ 兜底（§3.7.3 式）。这属于「自研补充」，需在代码注释中标明与 §3.7.3 的对应关系——与手写 `SSMBlock` 的 `delta/gamma` 钳法语义一致、实现位置不同。

### 4.4 CPU 回退的扫描实现三档（§3.7.3 已有结论）

[math.md](file:///home/simon-ma/JEPA/math.md#L1181-L1187) 已给出并行扫描的三档落地，作为 `mamba-ssm` 的 CPU 回退路径：

1. **纯 PyTorch log-depth 扫描**（当前 `psco.associative_scan`，零新依赖 + `torch.compile` 融合）——冒烟档（$d\le256$、短序列）够用；
2. **chunked scan + `torch.compile`**——长序列下的工程折中；
3. **Triton / `mamba-ssm` 内核**——$L\gtrsim10^3$ 才需要（对应 §4.2(g) 的 Mamba 收益区）。

### 4.5 结论

SSM 主干的成熟化 = **GPU 用 `mamba-ssm`（选择性扫描）+ CPU 保留手写对角 SSM 作回退**，谱夹层契约经 Δ 重参数化 + 谱正则兜底。短序列不启用（§2.2.5 明说 $L=16$ 时 Mamba 常数因子略亏、无收益）。

---

## 5. 候选三：Hybrid（注意力 + SSM 交错）

### 5.1 选型

[encoder.py](file:///home/simon-ma/JEPA/model/encoder.py#L123-L149) 的 `HybridEncoder` / `build_backbone` 已实现「前 `attn_layers` 层 SDPA 注意力 + 其余层 SSM」。成熟化即上述两者的组合：

- 注意力段：`F.scaled_dot_product_attention`（成熟，无新依赖）；
- SSM 段：`mamba-ssm`（GPU）/ 手写对角 SSM（CPU 回退）。

### 5.2 结论

`hybrid` 是 0.3B–7B 中型档的**推荐默认**：前段注意力抓局部 / 位置敏感模式，后段 SSM 做长程压缩，兼顾表达力与线性复杂度（§2.2.5 候选 2）。配置上只需 `backbone="hybrid"` + `attn_layers`，与现有 `PRESETS` / CLI 完全同构，无需改调用约定。

---

## 6. 三档可扩展性：参数量 / 显存 / 算力

规模表取自 §2.2.7（$d_{\text{hidden}}$ / 层数 $n$ / $L$ / $d$）：

| 规模 | $d_{\text{hidden}}$ | $n$ | $L$ | $d$（潜） | 主干 Transformer 参数 | 主干 Mamba 参数（≈0.55×） |
|---|---|---|---|---|---|---|
| tiny | 64 | 2 | 16 | 64 | ~98K | ~54K |
| small | 128 | 4 | 16 | 128 | ~0.79M | ~0.43M |
| base | 256 | 6 | 64 | 256 | ~4.7M | ~2.6M |
| 0.3b | 1024 | 24 | 1024 | 128 | ~302M | ~166M |
| 1b | 2048 | 24 | 1024 | 256 | ~1.21B | ~0.66B |
| 3b | 3072 | 32 | 4096 | 384 | ~3.62B | ~2.0B |
| 7b | 4096 | 32 | 4096 | 512 | ~6.44B | ~3.5B |

> 参数公式：Transformer 主干 $P\approx12\,n\,d_{\text{hidden}}^2$（§2.2.7，FFN $4d$）；Mamba（$d_{\text{inner}}=2d$）每层约 $6$–$8\,d_{\text{hidden}}^2$，故 ≈0.5–0.6×。上表为**单条序列主干**（`build_backbone` 单实例）；完整 `MultimodalJEPA` 含在线+目标双编码器 × 双模态 + 双解码器 + embedding，总量约为上表的 3–4 倍（与 `PRESETS` 的 0.3B/7B 命名同量级）。

### 6.1 时间复杂度（单层前向）

| 主干 | mixer 项 | FFN 项 | 主导区间 |
|---|---|---|---|
| Transformer | $O(L^2\,d)$ | $O(L\,d^2)$ | $L>d$ 时 attention 主导 |
| Mamba | $O(L\,d)$ | $O(L\,d^2)$ | FFN 恒主导 |

**交叉点判据**：$L\lesssim d_{\text{hidden}}$ 时（冒烟 tiny/small/base：$L=16/64 \ll d$）FFN 主导，Mamba 无收益；$L\gtrsim d_{\text{hidden}}$（3b/7b：$L=4096>d_{\text{hidden}}$）attention 的 $L^2$ 项主导，Mamba 的 $O(L)$ 扫描兑现收益（§4.2(g) 的 $L\gtrsim10^3$ 一致）。

### 6.2 显存（训练激活，bf16）

| 主干 | 激活显存 | 说明 |
|---|---|---|
| Transformer（朴素 attention） | $O(B\,L^2\,\text{heads})$ | $L$ 大时不可行 |
| Transformer（FlashAttention） | $O(B\,L\,d)$ | SDPA 自动分派，消掉 $L^2$ 矩阵 |
| Mamba | $O(B\,L\,d)$ | 扫描状态 $O(B\,L\,d)$，无 $L^2$ 项 |

关键：显存优势主要来自 FlashAttention（Transformer 也能拿到）而非 Mamba 本身；Mamba 的独特优势在**长序列下的 KV 缓存**——Transformer 每步重算增长中的上下文（$O(L)$ per token），Mamba 每步只更新 $O(1)$ 隐藏状态（§4.2(g)）。

### 6.3 串行深度（§4.3 判据，非 FLOPs）

| 主干 | 单层前向串行深度 | 总 |
|---|---|---|
| Transformer | $O(1)$（并行 attention） | $O(n)$ |
| Mamba | $O(\log L)$（关联扫描） | $O(n\log L)$ |
| Hybrid | 前段 $O(1)$ + 后段 $O(\log L)$ | $O(n_{\text{attn}}+n_{\text{ssm}}\log L)$ |

三者对 GPU 都无串行瓶颈（§2.2.6 同一判据），差异在常数因子与显存。

---

## 7. 依赖与 extra 划分

按「GPU-only 组件放可选 extra、保证第 1 档纯 CPU 可跑」的规则：

| 组件 | 归属 | 现状 |
|---|---|---|
| `F.scaled_dot_product_attention` / `torch.compile` | torch 内置 | 无需新依赖 |
| `mamba-ssm` + `causal-conv1d` | 新增 `mamba` extra（GPU-only） | 待加 |
| HF `transformers` / `datasets` / `pillow` | 已有 `hf` extra（SigLIP-2 前端、LLaDA tokenizer） | 已存在 |
| 手写 `SSMBlock` + `psco` 扫描 | 生产路径常驻（纯 PyTorch） | 已存在 |

建议在 [pyproject.toml](file:///home/simon-ma/JEPA/pyproject.toml) 增加：

```toml
[project.optional-dependencies]
# GPU-only：选择性扫描主干（§2.2.5 候选 2）
mamba = ["mamba-ssm>=2.2"]
```

第 1 档 CPU 冒烟**不装** `mamba` extra，`backbone="mamba"` 时自动落到手写对角 SSM 回退；第 2/3 档 GPU 环境装 `mamba` extra 后走 `mamba-ssm`。

---

## 8. 配置切换方式

现有 [config.py](file:///home/simon-ma/JEPA/model/config.py) 已具备同构切换，无需改调用约定：

- `MJEPAConfig.backbone ∈ {transformer, mamba, hybrid}`（[config.py](file:///home/simon-ma/JEPA/model/config.py#L21-L21)）；
- `MJEPAConfig.attn_layers`（hybrid 下前 N 层为注意力，[config.py](file:///home/simon-ma/JEPA/model/config.py#L22-L22)）；
- `PRESETS` 同构只调超参（[config.py](file:///home/simon-ma/JEPA/model/config.py#L71-L81)）；
- CLI：`--backbone` / `--attn-layers`（[train.py](file:///home/simon-ma/JEPA/training/train.py#L82-L83)）。

**成熟化落地需补的两处**（本文不实现）：① [encoder.py](file:///home/simon-ma/JEPA/model/encoder.py#L143-L149) 的 `build_backbone` 在 `backbone="mamba"` 时增加 `mamba-ssm` 分支（GPU 可用则用 `Mamba2`，否则回退 `SSMBlock`）；② 在 `Mamba2` 外层加 Δ 重参数化 + 谱正则，兑现 §4.3 的 C3 契约。

---

## 9. 切换判据（§2.2.5 实测信号）

出现下列信号再切 Mamba/hybrid，否则维持 Transformer（§2.2.5 原文）：

1. 训练显存随 $L$ **二次增长**（attention 矩阵成为瓶颈）；
2. 单步耗时随 $L$ **超线性**增长；
3. 目标序列长度要扩到**文档 / 视频帧级**（$L\gtrsim10^3$）。

信号出现前，Transformer（SDPA）是「实现成熟、常数因子优、短序列最快」的默认。

---

## 10. 建议落地顺序（供后续实现参考，本文不执行）

1. 加 `mamba` extra（[pyproject.toml](file:///home/simon-ma/JEPA/pyproject.toml)）。
2. `build_backbone` 增加 `mamba-ssm` 分支 + CPU 回退 + Δ 谱正则。
3. 冒烟档跑 `--preset tiny --backbone transformer/mamba/hybrid` 验证三路 loss 均下降、$Z$ 不塌缩（`variance_loss` + 重建）。
4. GPU 档 `--preset 0.3b --backbone hybrid` 验证 FlashAttention / `mamba-ssm` 显存与单步耗时。
