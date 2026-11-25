# Lean 形式化：math.md 信息论论证的推导与检查

本目录用 Lean 4 + mathlib 形式化了 [math.md](../math.md) 中 §3.2–§3.4 的核心信息论命题，
并据此检查文档中的数值与表述。所有定理均通过 `lake build` 编译验证（无 `sorry`/`axiom`）。

## 文件结构

| 文件 | 内容 |
|------|------|
| `LeanMath/Entropy.lean` | Shannon 熵定义、非负性、**最大熵定理**（式 (1)） |
| `LeanMath/InfoTheory.lean` | 互信息/条件熵定义、链式法则（式 (15)）、确定性信道 **I(Z;T)=H(T)**（式 (3)） |
| `LeanMath/Numeric.lean` | 位计数常量的 `norm_num` 精确校验与 `Float` 近似 |
| `LeanMath/Lossy.lean` | 命题 1/2/10 的**离散核心**：鸽笼原理下的有损性（`card β < card α ⟹ 非单射`）、碰撞 ⟹ 解码器不可逆、非单射 ⟹ 反馈创新非零 |
| `LeanMath/Affine.lean` | §3.7.3 PSCO 仿射幺半群：复合律 (31) 的**表示同态**、结合律 + 单位元、线性递推闭式（式 1129 的升幂形式） |
| `LeanMath/Contractive.lean` | §3.7.1 收缩算子：复合仍是收缩（Banach 根基）、Neumann 级数代数身份 `(1-a)Σaⁱ = 1-aᵏ` |
| `LeanMath/FixedPoint.lean` | （纯 Lean，无 mathlib）§3.7「结论状态 = 变换链不动点」：轨迹稳定 ⟹ 不动点、读模式退化为恒等、pred 迭代收敛到 0 |
| `LeanMath/Semigroup.lean` | （纯 Lean，无 mathlib）§3.9 式 (29)–(30) 算子幺半群 + 命题 11 二分：`Ψ=id ⟺ 重建`、`Ψ≠id ⟺ 生成 ⟺ 移动了某点` |

## 形式化结果 ↔ math.md 命题对照

### 1. 最大熵定理（式 (1)）

```lean
lemma entropy_le_log_card [Nonempty α] (p : α → ℝ) (hp₀ : ∀ a, 0 ≤ p a)
    (hsum : ∑ a, p a = 1) :
    entropy p ≤ Real.log (Fintype.card α)
```

即 $H(T) \le \log|\mathcal{V}^L| = L\log|\mathcal{V}|$。证明用 Jensen 不等式
（`Real.concaveOn_negMulLog` 是 `-x\log x$ 的凹性），把分布与均匀分布比较。这是 math.md
论证的**根基**：文本码率有一个与潜变量 $Z$ 无关的有限上界。

### 2. 确定性信道下 I(Z;T) = H(T)（式 (3)）

```lean
lemma condEntropy_zero_of_deterministic (g : α → β) (p : α → ℝ) :
    condEntropy (diag g p) = 0

lemma mutualInformation_eq_entropy_of_deterministic (g : α → β) (p : α → ℝ) :
    mutualInformation (diag g p) = entropy (map g p)
```

核心引理是**确定性标记不增加熵**：

```lean
lemma entropy_diag (g : α → β) (p : α → ℝ) : entropy (diag g p) = entropy p
```

即 $H(Z, g(Z)) = H(Z)$，从而 $H(T\mid Z)=0$、$I(Z;T)=H(T)$。这直接支撑"压缩来自码本
基数而非随机噪声"（式 (14) 的 DIB 论点）。

### 3. 语义/措辞码率分解（式 (15)）

```lean
lemma entropy_marginalX_eq_mutualInformation_add_condEntropyYX (p : α × β → ℝ) :
    entropy (marginalX p) = mutualInformation p + condEntropyYX p
```

即 $H(T) = I(T;Y) + H(T\mid Y)$，其中 $I(T;Y)$ 是语义码率、$H(T\mid Y)$ 是措辞码率。

### 4. 有损性的离散核心（命题 1 / 2 / 10）

命题 1（有损性）、命题 2（失真地板）、命题 10（软反馈创新恒非零）的**连续版本**依赖微分熵
与率失真测度论（mathlib 暂缺），但其共同核心是同一个**离散鸽笼事实**，已在
`Lossy.lean` 机械化：

```lean
theorem exists_collision_of_card_lt (g : α → β) (h : Fintype.card β < Fintype.card α) :
    ∃ a₁ a₂ : α, a₁ ≠ a₂ ∧ g a₁ = g a₂

theorem decoder_cannot_invert (g : α → β) (d : β → α)
    (hcol : ∃ a₁ a₂, a₁ ≠ a₂ ∧ g a₁ = g a₂) :
    ∃ a, d (g a) ≠ a

theorem non_injective_implies_feedback_nonzero (g : α → β) (E : β → α)
    (hcol : ∃ a₁ a₂, a₁ ≠ a₂ ∧ g a₁ = g a₂) :
    ∃ a, E (g a) ≠ a
```

即：码本严格小于源集 ⟹ 必有碰撞 ⟹ 任意解码/重编码 `d`/`E` 都无法同时还原碰撞点 ⟹ 软反馈
创新 $e = \hat Z - E(g Z)$ 恒非零。**正确方向是单向鸽笼**；反方向（非单射 ⟹ 码本更小）为
假（`Bool` 上 `const false` 是反例，见「问题 5」）。

### 5. PSCO 仿射幺半群与线性递推闭式（§3.7.3）

`Affine.lean` 把 §3.7.3 的「算子可以横向量化并行扫描」落到数学根基：仿射对
$(M,u)$ 的复合律 $(\mathrm{lin}, \mathrm{trans})$ 满足

```lean
theorem comp_apply (a b : Aff K V) (x : V) : (a.comp b) x = a (b x)   -- 表示同态
theorem comp_assoc (a b c : Aff K V) : a.comp (b.comp c) = (a.comp b).comp c
theorem comp_id / id_comp : 单位元 (id, 0)
theorem iterate_eq_closedForm (M) (b) (k) (z) :
    iterate M b k z = (powL M k) z + ∑ i ∈ range k, (powL M i) b
```

`comp_apply`（复合律 = 函数复合）与 `comp_assoc`（结合律）正是 Blelloch 上下两趟扫描
能以**任意括号顺序**算出同一前缀的保证；`iterate_eq_closedForm` 是文档式 (1129)
降幂闭式的升幂等价形式。`Contractive.lean` 进一步补上「收缩映射的复合仍是收缩」与
Neumann 级数代数身份，作为 §3.7.1 DEQ 式 (34) 的代数核心。

### 6. 算子幺半群与命题 11 二分（§3.9）

纯 Lean（不依赖 mathlib）的 `LeanMath/Semigroup.lean` 把命题 11 的「重建 vs 生成」二分
写成可判定式：

```lean
def isReconstruction (f : Op Z) : Prop := f = idOp
def isGeneration    (f : Op Z) : Prop := f ≠ idOp

theorem generation_iff_not_reconstruction (f) : isGeneration f ↔ ¬ isReconstruction f
theorem generation_iff_moves_some_point (f) : isGeneration f ↔ ∃ z, f z ≠ z
```

即「生成 ⟺ 非恒等 ⟺ 移动了某个点」，与 §3.9 式 (29)「重建 = D(id(Z))、生成 = D(Ψ(Z,ω))、
Ψ≠id」严格对应；`LeanMath/FixedPoint.lean` 则证明「结论状态 = 变换链的不动点」及
读模式（改分量为零）退化为恒等的两个基本断言。

## 数值校验结果

`lake build` 的 `#eval` 输出与 `norm_num` 精确结果：

| 量 | math.md 值 | Lean 计算结果 | 判定 |
|----|-----------|--------------|------|
| $24\log_2 77$ | 150.40 bit（式 (2)） | `150.402877` | ✅ |
| $Ld = 24\cdot 32$ | 768 | `norm_num` | ✅ |
| $\|\mathcal{V}\|^L = 77^{24}$ | — | `1887023914800623499934622000276222741687622241` | ✅ |
| $768\log_2 100$ | ≈ 5102 bit（式 (4)） | `5102.481554` | ✅ |
| $\frac12\log_2(2\pi e)$ | ≈ 2.047 bit（式 (26)） | `2.047096` | ✅ |
| $768\cdot 2.047$ | ≈ 1572 bit（式 (26)） | `1572.169409` | ✅ |

## 发现的问题

### 问题 1（数值笔误）：`150.85` 应为 `150.40`

math.md 第 258 行：

> 例如 $\delta = 10^{-2}$ 时 $Ld\log_2 100 \approx 5102$ bit，远超 (2) 的 **150.85** bit。

但式 (2)（第 244 行）与式 (3)（第 251 行）均为 **150.40** bit，且
$24\log_2 77 = 150.402877\ldots$。`150.85` 是笔误，应改为 `150.40`。

### 问题 2（量化表述不够严格）：命题 3.1 的"任意精细分辨率"

命题 3.1 与式 (6) 写 $H(Z_\delta \mid T) > 0$，其严格成立依赖 $\delta$ 足够小：

- 对**固定**的 $\delta$，若 $\delta$ 较粗（量化网格很粗），$T$ 可能完全确定 $Z_\delta$，
  此时 $H(Z_\delta\mid T)=0$，式 (6) 的最后一步 $>0$ 不成立。
- 正确表述应为「存在 $\delta_0$，使所有 $\delta<\delta_0$ 有 $H(Z_\delta\mid T)>0$」，
  或写成极限 $\lim_{\delta\to 0}H(Z_\delta\mid T)=+\infty$。论证依据是式 (4)：
  $H(Z_\delta)\to\infty$ 而 $I(Z_\delta;T)\le L\log_2|\mathcal{V}|$ 有界。

### 问题 3（离散/连续熵混用）：式 (4) 是渐近式

式 (4) 的 $H(Z_\delta) \approx h(Z) + Ld\log_2(1/\delta)$ 把离散量化熵 $H$ 与微分熵 $h$
并列，只有高分辨率极限下成立。严格写应是

$$H(Z_\delta) - Ld\log_2\frac{1}{\delta} \;\xrightarrow{\delta\to 0}\; h(Z).$$

对连续 $Z$，$h(Z\mid T)$ 可为负，所以"正条件熵"的直觉必须落在**量化后**的
$H(Z_\delta\mid T)$ 上，而非 $h(Z\mid T)$。这是文档中 $H$ 与 $h$ 的记号切换点，建议显式标注。

### 问题 4（未形式化的部分，作为已知标准结果标注）

以下内容在本形式化中**未机械化**，但已核实为教科书标准结果，特此说明而非留空：

- 式 (5) 数据处理不等式 $I(Z_\delta;T)\le I(Z;T)$：需 KL 散度的单调性/对数求和不等式。
  注意：式 (6) 的结论其实只用到更弱的 $I(Z_\delta;T)\le H(T)$（由 $I(X;Y)=H(Y)-H(Y\mid X)\le H(Y)$），
  而该弱式等价于条件熵非负，可由 $-x\log x$ 的凹性（超加性）直接得到——本目录 `Entropy.lean`
  已具备这一凹性，可进一步补证。
- 式 (8)–(12) 率失真理论、式 (7) MMSE 下界：连续源/测度论与微分熵的范畴，mathlib 暂无
  相关库，未展开。

### 问题 5（命题 1 的「任意精细分辨率」过强，且双向措辞需收紧）

math.md 第 219 行命题 1 断言「对**任意**有限字母表的确定性 $g$，在**任意**精细分辨率
$\delta$ 下 $H(Z_\delta\mid T)>0$」。形式化显示这个断言**过强**，需要两处收紧：

1. **「任意精细分辨率」不成立**：有损性本质是**计数条件**（码本 $|\mathcal V|^L$ 不足以区分
   源集 $|\operatorname{supp}Z_\delta|$），而非对任意 $\delta$ 的无条件结论。当 $\delta$ 尚不
   精细（$|\operatorname{supp}Z_\delta|\le|\mathcal V|^L$）时，$g$ 可以是单射，此时
   $H(Z_\delta\mid T)=0$。正确的量化说法是「存在 $\delta_0$，使所有 $\delta<\delta_0$ 有
   $H(Z_\delta\mid T)>0$」（即本文「问题 2」的同一结论，此处从计数侧独立给出）。
2. **「有损性 ⟺ 码本更小」的双向措辞需收紧为单向**：形式化证明的非平凡方向只有
   `card β < card α ⟹ g 非单射`（鸽笼原理）。反方向**不成立**——`Lossy.lean` 给出反例
   `Function.const false : Bool → Bool`：非单射但 $\operatorname{card}\beta=
   \operatorname{card}\alpha$。故文档若把命题 1 写成「有损当且仅当码本更小」，需改为
   「码本严格更小 ⟹ 有损」这一单向结论，或补充「源集有限且码本更小」这一前提。

### 问题 6（§3.7.3 谱夹层强于 §3.7.1 收缩条件，二者不可混用）

§3.7.1 的收敛条件是**谱半径** $\rho(J)<1$；§3.7.3 的谱夹层是**逐维**
$\delta<\tilde a_{h,i}<1-\gamma$（$\delta>0$）。二者在**一般矩阵**下不等价：

- $\rho(A)<1$ 允许特征值落在 $(-1,0)$（负特征值），这对应**振荡收敛**（轨迹来回摆动而非
  单调收缩）；谱夹层 $\delta>0$ 的下界把这类负特征值**排除**了。
- 谱夹层只对**对角矩阵** $\tilde A_h=\operatorname{diag}(a_h)$ 成立（此时特征值 = 对角元），
  且它蕴含 $\rho(\tilde A_h)<1-\gamma$；但 $\rho<1$ 本身**不蕴含**逐维下界 $\delta>0$。

math.md 第 1210 行「逐维 $\delta<\tilde a_{h,i}<1-\gamma$（等价于 $\rho(\tilde A_h)<1-\gamma$
且 $\min_i\tilde a_{h,i}>\delta$）」的「等价于」**在对角情形下成立**（因为对角时
$\rho=\max_i|\tilde a_{h,i}|$ 且全部为正 ⟹ $\rho=\max_i\tilde a_{h,i}$），但应显式标注
「对角矩阵」前提：对非对角/非对称 $\tilde A_h$，$\rho<1$ 与逐维夹层是两个**不同强度**的
条件，谱夹层更严（额外排除负特征值与过小收缩）。建议在 §3.7.3 明确：谱夹层是「收缩条件 +
禁止一步塌缩（$\delta$ 下界）」的**并集约束**，而非 $\rho<1$ 的等价重写。

### 问题 7（命题 7 / 8 / 9 是设计/诊断断言，非可形式化定理）

命题 7（描述/推理角色由网络自推断）、命题 8（创新序列白噪声 ⟺ 模型充分）、命题 9（推理的
表征空间内禀性）属于**架构设计约束 / 经验诊断判据**，而非可由纯数学推导证明的定理：

- 命题 7 断言 $h$ 落在「读/改」两端决定角色，依赖**网络实际学到的表征**，无法脱离训练
  分布与优化动力学作纯符号证明；
- 命题 8 的「创新白噪声 ⟺ 充分模型」是统计诊断（需对 $e_t$ 的自相关/偏置作假设检验），
  其「⟹」方向成立（充分模型 ⟹ 白噪声）可证，但「⟸」方向是经验启发；
- 命题 9 是**可解释性立场**（token 化是接口而非推理环节），不是数学命题。

本目录对这些命题**未机械化**（区别于命题 1/2/10/11 的离散核心），如实标注为设计断言。
若文档需要，可把命题 8 的「⟹」方向（充分模型 ⟹ 创新白噪声）作为一条可证命题补进
`Lossy.lean` 的统计扩展，但这已超出当前 mathlib 的信息论能力范围。

形式化确立的两条不变式——(a) 码率上界 $H(T)\le L\log_2|\mathcal{V}|$；(b) 分解
$H(T)=I(T;Y)+H(T\mid Y)$——指向以下效率方向：

1. **直线路径 + 一步/少步蒸馏（降低串行深度）**
   OT/直线路径下速度场 $u_t = z_1-z_0$ 与 $t$ 无关；若训练 $v_\psi$ 逼近常数速度，
   Euler 离散**一步即精确**。N 步残差来自 $\partial_t v_\psi\ne 0$。因此「reflow →
   consistency/LCM/DMD」的方向可把串行步数从 ~50 压到 1–4，是式 (3) 之外的纯计算收益。

2. **压缩语义瓶颈 $Y$、缩小潜空间维度 $d$（降低率失真成本）**
   式 (11) 的视觉码率 $\frac{Ld}{2}\log_2\frac{\sigma^2}{D}$ 与维度 $d$ 成正比。若先学一个
   低维语义因子 $Y$（即式 (15) 中的 $Y$），只在 $Y$ 上做流匹配，$d$ 从 768 降到语义流形
   本征维，代价按比例下降；措辞部分交给离散头，不再走连续流。

3. **并行离散读出（MaskGIT 式），串行深度 $L\to O(\log L)$**
   $T=g(Z)$ 逐 token argmax 需要 $L=24$ 个串行步。信道只需 $\le|\mathcal{V}|^L$ 个码字，
   与生成顺序无关；改用掩码并行预测可把串行步数降为 $O(\log L)$，而码率上界不变。

4. **按语义重要度分配码率（降低有效 $L$）**
   式 (10) 的失真地板 $D^\star$ 由 $L\log_2|\mathcal{V}|$ 决定。对无语义贡献的 token
   （标点、语气、冗余指代）用更短目标或共享码本，把码率集中到 $I(T;Y)$，可在不损语义
   的前提下压低 $D^\star$。

5. **单一连续-离散联合流（消除两阶段量化间隙）**
   当前「流 → argmax」两阶段在式 (16) 的线性打分处引入量化失配。改为在 logits 空间上做
   单条流、把离散瓶颈并入流的末端，可减少一次重建失真、并进一步减少串行步。

6. **线性注意力 / Mamba 替代 $O(L^2)$ 自注意力**（§3.6(g) 已提及，纯计算效率，不改变信息论结论）。

## 复现

```bash
cd lean-math && lake build
```

（Lean 4.36.0 + mathlib，见 `lean-toolchain`。全部 9 个模块——含纯 Lean 的
`FixedPoint` / `Semigroup`——均由 `lake build` 一次编译验证，无 `sorry`/`axiom`。）
