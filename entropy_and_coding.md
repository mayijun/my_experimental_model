# 熵论与编码算法：从选择性有损压缩到创新编码

> 本文是 [`math.md`](file:///home/simon-ma/JEPA/math.md) §2.6「不可压缩内容的生成元化压缩」的完整展开，自成一篇章节。目标回答一个问题：**离散 token 的编码器应该用什么样的有损压缩算法**。答案是"预测残差的变速率量化"（创新编码），其判据是可约损失而非原始熵。本文从信息论边界推到这个结论，再给出 GPU 并行实现。

---

## 0. 一句话结论

- **可压缩内容**：丢 $H(Z\mid Y)$、留 $I(Z;Y)$（信息瓶颈）。
- **不可压缩内容**：它本身就是 $I(Z;Y)$ 的种子、丢无可丢，正确操作是识别成生成元、存算子按需生成轨道。
- **熵**是**码率**信号（决定给多少比特），**不是显著性**信号（决定值不值得留）。
- **取舍判据**是**可约损失**（预测误差），操作核心是"测差值、别估熵"。
- **离散 token 的编码算法** = token 层按冗余度 $\rho$ 丢/留、$Z$ 保留完整精华，GPU 上逐元素决策 + 并行扫描记忆。

---

## 1. 定位：编码器压缩、解码器解压缩

编码器与解码器是一对**有损对**：

$$X \xrightarrow{\ E\ } Z \xrightarrow{\ D\ } \hat X,\qquad \hat X = D(E(X)) \ne X.$$

编码器 $E$ 把输入压进低维潜变量 $Z$（丢信息），解码器 $D$ 只能恢复"能恢复的那部分"。有损不是缺陷而是**设计目标**——理想是只丢"与任务无关的细节"，保留"主要脉络"。

把 $Z$ 的信息拆两半：

- **主要脉络** $I(Z;Y)$：$Z$ 与语义因子 $Y$ 的互信息；
- **无关细节** $H(Z\mid Y)$：给定 $Y$ 后 $Z$ 的残余熵。

理想编码器即信息瓶颈：

$$\max_E I(Z;Y)\quad\text{s.t.}\quad I(X;Z)\le R,$$

或 $\mathcal L_{IB}=I(X;Z)-\beta I(Z;Y)$。码率预算 $R$ 有限 → 把有限"处理能力"全花在 $I(Z;Y)$ 上、挤掉 $H(Z\mid Y)$。解码器于是只能恢复"骨架"，重建误差主要落在 $Y$ 无关方向。

---

## 2. 不可压缩内容的边界：生成元化压缩

### 2.1 问题：$H(Z\mid Y)\approx 0$ 时无可丢

定理/公理是被人类预先压到只剩 $I(Z;Y)$ 的内容。此时 $H(Z\mid Y)\approx 0$，再套 $I(X;Z)\le R$ 会把 $I(Z;Y)$ 一起切掉。必须换思路：**不可压缩的只是"陈述文本"，不是"内容"**。一个公理的可预测性不在它自身句子里，而在它作为**生成元**产生的轨道里（它推出整个理论）。

### 2.2 种子与轨道

把内容拆成"种子" $S$ 与"轨道" $O$（推论）：

$$H(\text{内容})\approx H(S)+H(O\mid S),\qquad H(O\mid S)\to 0.$$

轨道可预测（给定种子，推论被决定）。于是压缩 = **存储 $S$、按需生成 $O$**，码率从 $H(\text{内容})$ 降到 $H(S)$。这是"规律折叠成转移算子"的压缩版：定理 = 算子空间 $\Omega$ 的一个生成元，轨道由 $\Psi_h$ 重新生成、从不存储。

---

## 3. 熵论

### 3.1 熵是码率信号，不是显著性信号

记忆/表征容量有限（$C$ 比特）时，正确目标是

$$\max I(M;\text{未来})-\lambda|M|,$$

即 rate–distortion / information bottleneck。surprisal 和熵都只是这个目标的**廉价局部代理**。**纯熵门控主要命中"本来就难的文本"，而非"需要记住的事实"**——熵与词频、词性、领域强相关，不是好的可记忆性信号。

因此两个角色要分开：

| 信号 | 角色 | 公式 |
|---|---|---|
| 熵 $H[p]$ / surprisal | **码率分配**（给多少比特） | $H[p]=-\sum_v p\log p$，$s_t=-\log p(x_t\mid x_{<t})$ |
| 可约损失 $\Delta$ | **取舍/写入判据**（值不值得占容量） | $\Delta(z)=\mathrm{KL}(p^*\Vert p_{\theta,M})-\mathrm{KL}(p^*\Vert p_{\theta,M\oplus z})$ |

**attention sink 反例**：第一个 token 的注意力熵极低，却极重要。所以"低熵 → 可丢"是错的；熵不能单独做去留判据。

### 3.2 可约损失：正确的取舍判据

对单 token 的交叉熵损失做分解：

$$\ell_t = \ell^*_t + r_t,\qquad
\ell^*_t = -\log p^*(x_t\mid c),\qquad
r_t = \log\frac{p^*(x_t\mid c)}{p_\theta(x_t\mid c,M)}.$$

- $\ell^*_t$ 是 Bayes 最优损失（**不可约**），任何模型都压不下去；
- $r_t$ 是**对数似然比**（**可约**），单位 nat——"你比最优预测差了多少"。

取期望即 KL 分解：

$$\mathbb E[\ell] = \underbrace{H(p^*(\cdot\mid c))}_{\text{不可约}}\ +\ \underbrace{\mathrm{KL}(p^*(\cdot\mid c)\Vert p_\theta(\cdot\mid c,M))}_{\text{可约}}.$$

**"不可约"是相对的**：

$$H(x_t\mid c) = H(x_t\mid c,M) + I(x_t;M\mid c).$$

第二项正是记忆能吃掉的量。所以"可约损失"应定义为**相对于当前记忆/表征容量**能消掉的那部分，不是绝对噪声。

**操作核心：测差值、别估熵。** 定义写入 $z$ 的增益：

$$\Delta(z) = \mathbb E_{c}\ \mathbb E_{x\sim p^*(\cdot\mid c)}\big[\ell(x\mid c,M)-\ell(x\mid c,M\oplus z)\big] = \mathrm{KL}(p^*\Vert p_{\theta,M})-\mathrm{KL}(p^*\Vert p_{\theta,M\oplus z}).$$

**不可约项在减法里自动抵消，不需要知道 $p^*$**。可约损失是一个**相对量**：同一验证集上、写入前后的平均 loss 之差。

### 3.3 五种可计算估计方法

核心要求：**验证集必须是"未来/他处"的**，不能是刚写进去的东西本身。

| 方法 | 公式/做法 | 成本 | 适用 |
|---|---|---|---|
| (1) 重放集差分（金标准） | 维护 buffer $B$，写前后各算一次 loss 差 | 2×|B| 前向 | 离线/批量 |
| (2) 前视窗口（最便宜） | 写完后在接下来 $W$ 个 token 上算 loss 差 | 无需 buffer | 在线 |
| (3) 双模型差分 | 带/不带记忆两次前向的 loss 差 | (1) 的特例 | 无 buffer 对照 |
| (4) 熵分解上界 | $H_{\text{tot}}-H_{\text{alea}}=\mathrm{MI}$（dropout 采样） | 单前向 | 粗筛 |
| (5) 经验条件熵查表 | 后缀数组统计下一 token 经验分布熵 | 高 | 离线校准 |

其中 (4) 的 $\mathrm{MI}=H[\frac1K\sum_k p_k]-\frac1K\sum_k H[p_k]$ 给出"更多数据能消掉多少"的上界，不完全等于"更多容量能消掉多少"。

### 3.4 比特预算

- 统一单位：$\Delta$ 用 nat，除以 $\ln 2$ 得 bit；
- 理想校准下 $\Delta(\text{bits})\approx I(M;x_{\text{future}}\mid c)$，与目标函数对齐；
- 每比特收益 = $\Delta/\text{bits}(z)$，按此贪心/背包分配；
- **判据：$\Delta(z)-\lambda\cdot\text{bits}(z)>0$ 才写。**

---

## 4. 编码算法

### 4.1 变速率神经压缩（连续域）

Ballé 式管线 = 带熵模型的 VAE：

```
x ──► g_a(分析) ──► y ──► Q(量化) ──► ŷ ──► 算术编码 ──► 比特流
                         │                        ▲
                         └──► 超先验 h_a ──► ẑ ──(也编码)─┘
```

损失是率失真：

$$\mathcal L = \underbrace{-\log p(\hat y)}_{\text{码率 }R}\ +\ \lambda\underbrace{\|x-\hat x\|^2}_{\text{失真 }D}.$$

"自适应"两层：

- **内容自适应（隐式）**：熵模型 $-\log p(\hat y)$ 对复杂样本天然更大 → 自动分更多比特；
- **显式变速率**（Choi et al. 2019）：把 $\lambda$ 做成 per-sample 输入，学 $g_a(x,\lambda),g_s(\hat y,\lambda)$，同一模型手动拨码率。

落地件：Ballé et al. 2018（scale hyperprior）、Minnen et al. 2018（autoregressive priors）、Choi et al. 2019（conditional autoencoder）。

### 4.2 离散 token 的两种形态

token 已是离散、无连续潜变量可量化，Ballé 式管线不直接作用于原始 token。区分两种形态：

1. **无损 token 压缩 = 语言模型 + 算术编码**。熵模型即 $p(x_i\mid x_{<i})$（categorical/softmax），码率 $R=-\sum_i\log p(x_i\mid x_{<i})$；"变速率"自动发生——可预测 token 花少比特、意外 token 花多比特。依赖建模从"超先验捕空间依赖"退化为"自回归上下文捕序列依赖"。
2. **有损 token 压缩 = token 层的 ρ 残差压缩**。有损发生在**输入 token 层**，不是潜空间——$Z$ 是编码器已压好的精华、不该再损失。判据是冗余度 $\rho=\|v-M^\top k\|^2$（见 §4.3）：

$$x \xrightarrow{\text{按 }\rho\text{ 丢/留}} \tilde x \xrightarrow{E} Z\ \text{（精华，完整）}\xrightarrow{g} \hat x.$$

token 通道走无损（LM 熵模型），token 层走有损（ρ 残差压缩）。

### 4.3 创新编码（修正）：token 层残差压缩，$Z$ 是完整精华

有损压缩发生在 token 层，不是潜空间——$Z$ 是编码器已压好的精华 $I(Z;Y)$，不该再损失；该丢的是 token 层的冗余：

```
token x ──► [写进记忆 M] ──► 残差 ρ = ‖v − Mᵀk‖² ──► 按 ρ 决定丢/留 ──► x̃ ──► E ──► Z(精华，完整)
```

判据是**冗余度 $\rho=\|v-M^\top k\|^2$**（给定记忆后还重建不了多少），不是注意力——注意力与重要性**反相关**（attention sink、高频虚词注意力高却无内容，实体词注意力低却是关键）。$\rho\approx0$ 说明已被记忆吸收、可丢；$\rho$ 大说明不可预测、必须留。

**先写后压**：token 先写进记忆，再算 $\rho$ 当失真，按 $\rho$ 决定留不留、留几比特：

$$b_t=\mathrm{clamp}\big(a\log_1p(\rho_t)+c\,H(x_t\mid\text{rest}),\ b_{\min},b_{\max}\big),\qquad \mathrm{keep}_t=(\rho_t>\epsilon_\rho)\lor(H_{\text{cond},t}>\epsilon_H)\lor\mathrm{is\_sink}(t).$$

$\rho$ 是 delta rule 的中间量，零额外前向。**压缩率内生自平衡**：$M$ 有空位 → 全吸收 → $\rho\approx0$ → 激进压缩；$M$ 满 → 旧信息被 decay 挤出 → 新 token $\rho$ 上升 → 自动保留更多——压缩器和记忆互为溢出区，稳态自动收敛。

**两个补丁**：(1) sink 保护——位置 0 和开头几个 token 硬编码不丢（softmax 泄洪口）；(2) ε 强制采样——每批保留 2–5% 低 $\rho$ 低熵 token，防"丢弃→学不到→loss 降不下→继续丢"的自我强化退化。

**失真度量**：用下游 loss 增量 $\Delta L_{\text{downstream}}=\mathbb E[\ell(x|\text{压缩后},M)-\ell(x|\text{原始},M)]$，不用重建 MSE。被丢 token 由解码器从骨架 $Z$ 随机填充——被丢的本来就是冗余，填充不改变大意。

### 4.4 GPU 并行：逐元素决策 + 并行扫描记忆

判据 $\rho_t$ 是逐 token 标量，决策全是逐元素运算、无变长循环：

1. 并行写记忆 + 并行算 $\rho_t=\|v_t-M_{t-1}k_t\|^2$（记忆更新用线性注意力的**并行扫描**，串行深度 $O(\log L)$）；
2. 并行算 $b_t$（逐元素 `clamp`）与 `keep_t`（逐元素布尔掩码）；
3. 并行用 `keep_t` 掩码 gather 得到保留 token $\tilde x$；
4. $\tilde x$ 进编码器 $E$（Transformer 并行）得 $Z$。

**串行深度 = 记忆扫描 $O(\log L)$ + 决策 $O(1)$ + 编码器 $O(1)$**，优于 Flow 的 $O(N)$、远好于自回归 $O(L)$；$\rho$、$b$、`keep` 全 $O(1)$ 逐元素。唯一串行件仍是"编成比特流"的算术编码，本框架训练用可微代理、推理留掩码索引，两者都不碰。

---

## 5. 自推方法论三步

1. **不可压缩性检测**：熵模型读 $\hat H(\hat z)$，超阈值判为"种子"（定理）而非"描述"，决定走 IB 还是走生成元路线；
2. **存储生成元而非轨道**：种子存入 $\Omega$ 作为生成元，轨道由 $\Psi_h$ 按需生成，码率封顶在 $H(S)$；
3. **自适应码率兜底**：半压缩内容（有冗余但不多）回落到变速率熵模型，纯种子则不再尝试压下去。

---

## 6. 与本框架的映射

| 本文概念 | 框架已有件 | 对应 |
|---|---|---|
| token 残差 $\rho=\|v-M^\top k\|^2$ | delta rule 中间量 | 冗余度/可约损失的单步代理 |
| token 层丢/留掩码 | 记忆 $M$ + 掩码 gather | 按 $\rho$ 决定丢留与比特 |
| "测差值" | §5 自反馈、§3.7 $\mathcal L_{\text{consist}}$ | 不可约项抵消的同一原理 |
| 下游 loss 增量 | §5 自反馈闭环 | 失真度量 |
| 生成元/轨道 | 算子空间 $\Omega$、$\Psi_h$ | 存算子、按需生成轨道 |

现成机制（DeltaNet 的 $v-M^\top k$、Titans 的 surprisal 门控 + momentum/decay、Compressive Transformer 的 rate–distortion 压缩）都是"可约损失单步估计 + 预算 + 衰减"的组合，与本文三步方法论同构。

---

## 7. 收束

> 可压缩的内容：丢 $H(Z\mid Y)$，留 $I(Z;Y)$（IB）。
> 不可压缩的内容：识别成生成元，存储算子、按需生成轨道。
> 熵决定码率，可约损失决定去留；操作核心是"测差值、别估熵"。
> 离散 token 编码器 = 先预测、量化创新、码率按预测误差自适应；GPU 上"算满再掩码"，串行深度 $O(D_{\max})$。
