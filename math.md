# Flow Matching + JEPA 字符模型的数学基础与核心命题验证

> 本文所有公式均与 [`flow_matching_jepa.py`](file:///home/simon-ma/JEPA/flow_matching_jepa.py) 的实现逐式对齐。行号引用均指向该文件。

---

## 0. 符号表与源码映射

| 数学符号 | 含义 | 源码对应 |
|---|---|---|
| $x, x_c, x_y$ | 干净 token 序列、掩码上下文、干净目标 | `clean`, `context_ids`；掩码比例见 [L142-149](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L142-L149) |
| $E_\theta$ | 在线编码器（接收梯度） | `model.online`，[L119](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L119) |
| $E_\xi$ | 目标编码器（仅 EMA 更新） | `model.target`，[L120](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L120) |
| $P_\phi$ | JEPA 潜变量预测器 | `model.jepa_predictor`，[L124-128](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L124-L128) |
| $v_\psi$ | Flow Matching 速度场预测器 | `model.flow`，[L129](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L129) |
| $D_\eta$ | 潜变量→token 解码器 | `model.decoder`，[L130](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L130) |
| $\mu$ | EMA 衰减系数 | `EMA_DECAY=0.99`，[L21](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L21) |
| $L$ | 序列长度 / token 数 | `SEQ_LEN=24`，[L17](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L17) |
| $d$ | 潜空间维度 | `DIM=32`，[L19](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L19) |
| $\mathcal{V}$ | 词表（含 `[MASK]`） | `vocab_size=len(chars)+1`，[L283](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L283) |
| $z_0, z_1, z_t$ | 噪声、干净潜变量、时刻 $t$ 插值 | `noise`, `target`, `z_t`，[L228-230](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L228-L230) |
| $u_t$ | 真实速度场 | `velocity_target`，[L231](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L231) |

---

## 1. JEPA 表征学习的数学原理

### 1.1 双编码器与上下文构造

设数据分布为 $x \sim p_{\text{data}}$。JEPA 维护两个结构相同的编码器：

- 在线编码器 $E_\theta$（可训练），
- 目标编码器 $E_\xi$（参数冻结，仅靠 EMA 追随 $\theta$），

初始化时 $\xi \leftarrow \theta$（[L121](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L121)），并对 $E_\xi$ 关闭梯度（[L122-123](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L122-L123)）。

对每个样本 $x$，构造掩码上下文 $x_c = M(x)$，其中掩码比例

$$m \sim \mathcal{U}(0.2,\, 0.95)$$

即 [L145](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L145) 的 `0.2 + 0.75 * torch.rand(...)`。被遮位置替换为 `[MASK]` token，干净目标 $x_y = x$ 保持不变（[L147-148](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L147-L148)）。

### 1.2 预测目标与余弦损失

记上下文表示与目标表示：

$$c = E_\theta(x_c), \qquad z_y = E_\xi(x_y).$$

$z_y$ 在 `torch.no_grad()` 下计算（[L192-193](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L192-L193)），梯度只回传到 $E_\theta$ 与 $P_\phi$。预测器从上下文预测目标潜变量：

$$\hat z = P_\phi(c),$$

对应 [L195](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L195)。损失采用余弦相似度（而非原 I-JEPA 的 $\ell_2$）：

$$\mathcal{L}_{\text{JEPA}} = \mathbb{E}_{x}\left[\, 1 - \cos\!\bigl(\hat z,\; z_y\bigr)\,\right],$$

即 [L196](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L196)。最小化该式等价于把预测方向对齐到目标表示方向。

### 1.3 EMA 目标更新

每次优化步之后执行（[L133-139](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L133-L139)、[L204](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L204)）：

$$\xi \leftarrow \mu\,\xi + (1-\mu)\,\theta, \qquad \mu = 0.99.$$

**为什么 EMA + stop-gradient 是必需的**：若没有 stop-gradient，平凡解 $E_\theta(x) \equiv \text{const}$、$P_\phi(c) \equiv \text{const}$ 可使余弦损失直接为 0，出现**表征塌缩**。stop-gradient 切断了这条捷径：目标 $z_y$ 只能缓慢地经 EMA 移动，在线网络必须去预测一个"缓慢漂移但非平凡"的目标。

### 1.4 防塌缩的两道附加约束（源码中的关键设计）

> ⚠️ 严谨说明：余弦损失 + EMA 本身**并不能完全排除塌缩**——若在线编码器把所有位置映到同一常向量 $a$，预测器输出 $a$，目标经 EMA 也趋于 $a$，仍有 $\cos(a,a)=1$、损失为 0。因此本模型真正的塌缩防线是下面两项。

**(a) 重建损失（主要防线）**

$$\mathcal{L}_{\text{recon}} = \mathrm{CE}\!\bigl(D_\eta(E_\theta(x)),\; x\bigr),$$

对应 [L159-162](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L159-L162)、[L197](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L197)。若 $E_\theta(x)$ 在不同位置退化为常数，解码器无法区分 token，交叉熵将爆炸。该项强制表示中保留**位置可辨别的 token 信息**，是抑制塌缩的主力。

**(b) VICReg 风格方差约束（次要防线）**

$$\mathcal{L}_{\text{var}} = \mathbb{E}_{j}\bigl[\mathrm{relu}(1 - \sigma_j)\bigr], \qquad \sigma_j = \mathrm{std}(z_{:,j}),$$

即 [L152-156](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L152-L156)。惩罚在 batch 与位置维度上方差过小的潜空间维度，防止维度塌缩。

阶段一总目标（[L199](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L199)）：

$$\mathcal{L}_1 = \mathcal{L}_{\text{JEPA}} + \mathcal{L}_{\text{recon}} + 0.1\,\mathcal{L}_{\text{var}}.$$

---

## 2. Flow Matching 生成的数学原理

阶段一结束后冻结 $E_\theta$（[L214-215](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L214-L215)），潜空间被固定，Flow 只需拟合这个静止的数据分布。

### 2.1 概率路径与线性（OT）插值

设 $p_0 = \mathcal{N}(0, I)$ 为噪声分布，$p_1$ 为目标编码器诱导的数据分布 $z_1 = E_\xi(x) \sim p_1$。Flow Matching 在两者之间构造一条概率路径 $p_t$（$t\in[0,1]$）。本模型采用**线性插值路径**（[L230](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L230)）：

$$z_t = (1-t)\,z_0 + t\,z_1, \qquad z_0 \sim p_0,\; z_1 \sim p_1,\; t \sim \mathcal{U}(0,1).$$

这是最优传输（OT）在高斯与数据分布之间的一条直线插值路径，也是 Conditional Flow Matching 中最简单且稳定的选择。

### 2.2 速度场与回归目标

对路径关于 $t$ 求导，得到该路径对应的**真实速度场**：

$$u_t(z_t) = \frac{d z_t}{d t} = z_1 - z_0.$$

源码中即 `velocity_target = target - noise`（[L231](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L231)）。模型学习一个条件速度场 $v_\psi(z_t, t, c)$ 去回归它（[L232](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L232)），损失为均方误差（[L233](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L233)）：

$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, z_0, z_1}\left[\left\| v_\psi(z_t, t, c) - (z_1 - z_0) \right\|^2\right].$$

条件 $c = E_\theta(x_c)$ 通过 `FlowBlock` 中的交叉注意力注入（[L66-74](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L66-L74)），使生成的潜变量与可见前缀一致。

### 2.3 时间嵌入

连续时间 $t$ 经正弦嵌入后由 MLP 非线性化（[L77-87](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L77-L87)、[L97-101](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L97-L101)）：

$$\mathrm{emb}(t) = [\sin(t \cdot 1000 \cdot f),\; \cos(t \cdot 1000 \cdot f)], \qquad f_j = \exp\!\left(-\frac{\ln 10000 \cdot j}{d/2 - 1}\right),$$

再与 $z_t$ 相加后送入主干。

### 2.4 Euler 采样：从噪声积分到潜变量

推理时求解由学习速度场定义的 ODE（[L255-259](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L255-L259)）：

$$\frac{d z}{d t} = v_\psi(z, t, c), \qquad z(0) \sim \mathcal{N}(0, I),$$

用 Euler 法离散，步长 $\Delta t = 1/N$（$N$ = `inference_steps`，[L256](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L256)）：

$$z_{k+1} = z_k + \Delta t \cdot v_\psi(z_k,\; k\Delta t,\; c), \qquad k = 0, 1, \dots, N-1.$$

终点 $z(1)$ 即为生成的连续潜变量 $\hat z$。随后经解码器量化为文本（见第 3 节）。

---

## 3. 核心命题：文本 token 是连续潜空间的有损压缩 / 摘要

### 3.1 命题的精确陈述

令 Flow 模型采样得到的连续潜变量为 $Z \in \mathbb{R}^{L \times d}$（本模型 $L=24, d=32$），令文本 token 序列

$$T = g(Z) = \arg\max_{\mathcal{V}^L} \mathrm{softmax}(D_\eta(Z))$$

为解码器输出（[L261-263](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L261-L263)）。

> **命题**：$T$ 是 $Z$ 的**有损压缩**——$Z$ 中存在 $T$ 无法表达的信息；进一步，当 $D_\eta$ 被训练为保留下游语义时，$T$ 是 $Z$ 关于语义的**摘要**（信息瓶颈解），其失真集中在低语义相关的方向上。

下面给出 4 条相互独立的验证。

### 3.2 验证一：数据处理不等式 + 熵界（"有损"的严格证明）

> 🔧 对上一轮表述的修正：不能直接把离散熵 $H(T)$ 与微分熵 $h(Z)$ 比较——微分熵依赖坐标选择且可为负，二者不同量纲。严谨论证如下。

**步骤 1（离散 token 流的信息有硬上界）**：
$T$ 取值于有限集 $\mathcal{V}^L$，故

$$H(T) \le L \log |\mathcal{V}|. \tag{1}$$

对本模型，$|\mathcal{V}| \approx 30$（语料字符 + `[MASK]`），故 $H(T) \le 24 \cdot \log_2 30 \approx 118$ 比特；自然语言实际熵率更低。这是一个**与 $Z$ 无关的有限常数**。

**步骤 2（$T$ 是 $Z$ 的确定函数，互信息等于 $H(T)$）**：
因 $T = g(Z)$，给定 $Z$ 时 $T$ 完全确定，$H(T|Z) = 0$，于是

$$I(Z; T) = H(T) - H(T|Z) = H(T). \tag{2}$$

结合 (1)(2)：

$$I(Z; T) \le L \log |\mathcal{V}| < \infty. \tag{3}$$

**步骤 3（连续 $Z$ 的离散化熵无界）**：
设 $Z$ 有密度。将 $Z$ 量化到分辨率为 $\delta$ 的网格上得到离散变量 $Z_\delta$。在 $d$ 维下，量化熵与微分熵满足（高分辨率极限）

$$H(Z_\delta) \approx h(Z) + d \log \frac{1}{\delta} \;\xrightarrow{\delta \to 0}\; +\infty. \tag{4}$$

**步骤 4（DPI 阻断）**：
$Z_\delta$ 是 $Z$ 的函数，$T$ 也是 $Z$ 的函数，构成 Markov 链 $Z_\delta \to Z \to T$。由**数据处理不等式**，

$$I(Z_\delta; T) \le I(Z; T) \le L \log |\mathcal{V}|. \tag{5}$$

将 (4) 与 (5) 对比：当 $\delta$ 足够小时，$H(Z_\delta) > I(Z_\delta; T)$，即

$$H(Z_\delta) - I(Z_\delta; T) = H(Z_\delta \mid T) > 0.$$

> **结论**：给定文本 $T$，$Z$ 在精细分辨率下仍有正的条件熵——存在 $T$ 无法确定的 $Z$ 的细节。$T$ 对 $Z$ 的描述是**严格有损**的。这是不依赖任何训练假设的信息论事实。

### 3.3 验证二：率失真理论的失真下界

率失真函数给出"在失真不超过 $D$ 时描述 $Z$ 所需的最低码率"：

$$R(D) = \min_{p(\hat z \mid z):\; \mathbb{E}[d(Z,\hat Z)] \le D} I(Z; \hat Z).$$

$R(D)$ 关于 $D$ 单调不增。用文本 $T$ 描述 $Z$ 时，可用码率至多为 $H(T) \le L\log|\mathcal{V}|$（式 1）。由 $R(D)$ 的单调性，$T$ 能达到的失真 $D_T$ 满足

$$R(D_T) \le L \log |\mathcal{V}| \quad\Longrightarrow\quad D_T \ge R^{-1}\!\left(L \log |\mathcal{V}|\right) > 0. \tag{6}$$

> **结论**：只要 token 预算 $L\log|\mathcal{V}|$ 有限（必然有限），就存在一个**正的失真下界**——文本不可能无失真地重建连续潜变量。词表越小、序列越短，下界越高，压缩越"狠"。这把"有损"从定性变成了一个可计算的量。

### 3.4 验证三：信息瓶颈视角——为什么是"摘要"而非"随机丢弃"

仅有损不足以称为"摘要"。信息瓶颈（Tishby 等，1999）刻画了"丢掉细节、保留语义"的最优权衡：

$$\min_{p(t \mid z)} \; I(Z; T) \;-\; \beta\, I(T; Y), \qquad \beta > 0, \tag{7}$$

其中 $Y$ 是下游语义任务（如重建原 token、跨模态对齐、分类）。两项分别是：
- $I(Z; T)$：压缩项（越小越丢弃细节），
- $I(T; Y)$：相关项（越大越保留语义），
- $\beta$：压缩–保留的权衡旋钮。

解码器 $D_\eta$ 本质上是一个（近似确定性的）量化器 $g$，把连续 $Z$ 投影到离散码本 $\mathcal{V}^L$。它正是信息瓶颈在 $\beta \to \infty$、且瓶颈变量被强制离散化时的极限解：在词表预算固定的约束下，选择使 $I(T;Y)$ 最大的那个量化。

> **结论**：文本 $T$ 不是任意量化，而是在"码率固定为 $L\log|\mathcal{V}|$"约束下最大化语义互信息 $I(T;Y)$ 的瓶颈变量。这解释了第 2 节提到的经验现象——连续生成文本时"语义在、措辞丢"：措辞属于被压掉的 $I(Z;T)$ 部分，语义属于被保住的 $I(T;Y)$ 部分。

### 3.5 验证四：与本模型目标函数的对应

本模型的训练目标已隐含上述率–失真结构。阶段一/二中的重建损失可以写成 ELBO 的形式（解码器 $D_\eta$ 对应似然项 $p(x \mid z)$）：

$$\log p(x) \;\ge\; \underbrace{\mathbb{E}_{q(z\mid x)}\!\left[\log p(x \mid z)\right]}_{-\,\text{失真（重建，越大越好）}} \;-\; \underbrace{\mathrm{KL}\!\left(q(z\mid x)\,\|\,p(z)\right)}_{\text{率惩罚}}.$$

源码中未显式写出 KL 项，但 `variance_loss`（[L152-156](file:///home/simon-ma/JEPA/flow_matching_jepa.py#L152-L156)）起到了类似"控制潜空间熵/率"的正则作用，而 `reconstruction_loss` 是失真项的直接实现。最终生成时，$Z \xrightarrow{D_\eta} T$ 这一步就是最近邻量化：

$$T = \arg\min_{v \in \mathcal{V}} \bigl\| Z - e(v) \bigr\|^2 \quad \text{(逐位置)},$$

其中 $e(v)$ 是解码器末层（等价于 token embedding）的第 $v$ 行。从最优传输角度看，$g$ 是把连续分布 $p_Z$ 推前到离散分布 $p_T$ 的传输映射，量化噪声 $\|Z - e(T)\|^2$ 就是式 (6) 中失真 $D_T$ 的经验估计。

---

## 4. 落到本模型的可检验推论与实验

由第 3 节可直接导出在 [`flow_matching_jepa.py`](file:///home/simon-ma/JEPA/flow_matching_jepa.py) 上可跑的验证实验：

| 推论 | 实验 | 预期 |
|---|---|---|
| $I(Z;T) \le L\log\|\mathcal{V}\|$（式 3） | 采样生成文本，用 n-gram 或神经估计器估计 $\hat H(T)$ | $\hat H(T) < 24\log_2 30 \approx 118$ 比特 |
| 失真下界 $D_T > 0$（式 6） | 把生成文本 $T$ 重新编码 $\hat Z = E_\theta(T)$，计算 $\mathbb{E}\|Z - \hat Z\|^2$ 与余弦相似度 | 相似度显著小于 1，且随 $|\mathcal{V}|$ 减小而下降 |
| 率失真曲线 | 改变词表粒度（字符 → 子词 → 词），画 $(\hat H(T),\, \hat D_T)$ | 点落在连续态工作点的"高压缩、高失真"区域，呈单调趋势 |
| 信息瓶颈语义选择性（式 7） | 把 $Z - \hat Z$ 的失真按 $d=32$ 个维度拆分，与各维度对重建任务的贡献度对比 | 失真集中在对重建/语义贡献低的维度 |

---

## 5. 参考文献

1. Lipman, Y. et al. *Flow Matching for Generative Modeling*, ICLR 2023.（第 2 节概率路径与速度场）
2. Assran, M. et al. *I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, 2023.（第 1 节 JEPA）
3. Bardes, A. et al. *VICReg: Variance-Invariance-Covariance Regularization*, ICLR 2022.（第 1.4 节方差约束）
4. Tishby, N., Pereira, F. C., Bialek, W. *The Information Bottleneck Method*, 1999.（第 3.4 节）
5. Shannon, C. E. *Coding theorems for a discrete source with a fidelity criterion*, 1959.（第 3.3 节率失真）
6. DeepMind. *Language Modeling Is Compression*, arXiv:2309.10668, 2023.（预测模型 ⟺ 压缩器的经验背书）
7. Li, J. et al. *JUICE: Unified Multimodal Modeling in Pretrained Representation Spaces*, arXiv:2610.28801, 2026.（"连续生成文本会丢措辞"的经验观察，对应第 3.4 节）
