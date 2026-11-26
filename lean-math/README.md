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
| `LeanMath/Retraction.lean` | §4.1.7 黎曼流匹配 GPU 友好近似：retraction 公理与一阶一致、欧氏 retraction 退化特例、投影 retraction 幂等核心、低秩度量 `rank(JᵀJ) ≤ min(\|𝒱\|, d)` |
| `LeanMath/AssociativeMemory.lean` | §2.9 CARO 替代结构（门控结合扫描态）的代数核心：更新是仿射（复用 `PSCO.Aff` 幺半群 ⇒ 并行扫描）、内容可寻址读出恒等式、残差⇔被历史重建、不动点/误差/收敛、反塌缩 |

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

### 7. §4.1.7 黎曼流匹配的 GPU 友好近似（retraction-based）

`Retraction.lean` 把 §4.1.7 的数学核心机械化（把切空间/环境空间统一抽象为同一赋范向量
空间 `E`，即环境坐标下的展开）：

```lean
structure Retraction (E) where
  base : E
  toFun : E → E
  map_zero : toFun 0 = base                       -- R_z(0) = z
  fderiv_id : HasFDerivAt toFun (.id ℝ E) 0        -- dR_z(0) = id

theorem Retraction.first_order_consistency (R : Retraction E) :
    (fun v => R v - R.base - v) =o[𝓝 0] fun v => v   -- R(v) = z + v + o(‖v‖)

theorem rank_pullbackMetric_le_min (J : Matrix m n ℝ) :
    (Jᵀ * J).rank ≤ min (Fintype.card m) (Fintype.card n)  -- rank(G) ≤ min(|𝒱|, d)
```

四条结论：

1. **retraction 公理**：`R_z(0)=z` 与 `dR_z(0)=id` 作为 `Retraction` 结构的两字段，与 §4.1.7
   的 `R_z(0)=z`、`(d/dt)R_z(tv)|_{t=0}=v` 对应；
2. **一阶一致**（`first_order_consistency`）：Fréchet 可微 + `dR(0)=id` 直接给出
   `R(v)=z+v+o(‖v‖)`——这是 §4.1.7 `exp_z(v)=R_z(v)+O(‖v‖²)` 去掉二阶余项的一阶部分；
3. **欧氏 FM 是退化特例**（`euclideanRetraction`）：`R_z(v)=z+v` 满足两条公理；且
   `projectionRetraction_id_eq_euclidean` 证明投影 retraction 在 `π=id` 时退化为它；
4. **低秩度量**（`rank_pullbackMetric_*`）：`rank(JᵀJ)=rank(J)≤min(|𝒱|,d)`（引用 mathlib
   `Matrix.rank_transpose_mul_self` 与 `rank_le_card_height/width`），是 §4.1.7 梯度 3 与
   §4.2(a)「可区分维度至多 min(|𝒱|,d)」的精确线性代数形式；
5. **线性投影 retraction**（`linearProjectionRetraction_fderiv` / `linearProjection_idOnImage`
   / `linearProjectionRetraction_eq_euclidean_of_id`）：软投影 `R_z(v)=z+P_z v` 的导数恰为
   `P_z`（非 `id`），且幂等投影在像空间上是恒等——合起来证明它是「切空间 `im J_Dᵀ` 上的
   一阶精确 retraction」，`P_z=id` 时退化为欧氏 retraction。这是「问题 9」修正建议
   （软投影 vs 硬 argmax 两分）的已验证核心。

投影 retraction 的代数核心（`IsProjection` 幂等 + `projectionRetraction_lands_on_manifold`）
证明 `π(R_z v)=R_z v`——即一步之后恒在不动点集（流形）上，是 §4.1.6/4.1.7「路径始终在
M 上、全程可解码」的代数保证。真流形的 exp/log、测地线 ODE 与「投影 retraction 是
retraction」的 `fderiv_id` 公理（需 π 为光滑投影）不在本文件展开，见「问题 8」「问题 9」。

### 8. CARO 替代结构：门控结合扫描（§2.9）

`LeanMath/AssociativeMemory.lean`（命名空间 `AssocMem`）形式化了 §2.9 提出的 CARO
（Context-addressed Associative-scan Reasoning Operator）——把「按 ρ 丢/留 → 落入 Z → 在 Z
中对齐并推理」折叠为**单一仿射结合扫描态**。在任意域 `K`、任意 `K`-模 `V`、任意双线性型
`B : V →ₗ (V →ₗ K)` 下机械化以下断言（全部无 `sorry` / 无新增公理，`lake build` 通过）：

| 断言 | 定理 | math.md |
|------|------|---------|
| 状态更新 `S ↦ a•S + g•outer(k,v)` 是 `PSCO.Aff` 仿射映射 ⇒ 复用 `Affine.lean` 结合幺半群 ⇒ 前缀扫描合法 | `update_is_affine` | §2.9.1 / §3.7.3 |
| 读出恒等式 `S_T·q = Σ_{i<T} (Π_{j=i+1}^{T-1} a_j)·g_i·B(k_i,q)·v_i`（内容可寻址长程检索，线性状态实现） | `stateFrom_readout` | (CARO-1) |
| 推论：预测值 `S_t·k_t` = 历史按 `B(k_i,k_t)` 加权和 | `prediction_is_content_sum` | (CARO-1) |
| 残差 `ρ = ‖v − S_{t-1}·k_t‖`；`ρ=0 ⟺ v_t 被历史精确重建 ⟺ 冗余` | `residual_zero_iff_reconstructed` | (CARO-2) |
| 不动点 `S*=(1−a)⁻¹·b` 验证 + 误差恒等式 `S⁽ⁿ⁾−S* = aⁿ·(S⁰−S*)` + `aⁿ→0`（`a∈[0,1)`） | `fixedPoint_is_fixed` / `iterAff_error` / `real_pow_tendsto_zero` | (CARO-3)(CARO-4) |
| 反塌缩：`b₁≠b₂ ⇒ 不动点不同`（输入依赖，不坍缩到单一状态） | `no_collapse` | §2.9.5 契约 2 |

关键：`stateFrom_readout` 的归纳证明显示，**常数大小的线性状态 `S`** 即可实现对整个历史的
**精确**内容寻址聚合——这是 CARO 在 `O(n)` 时间内维持 Transformer 内容可寻址性、同时
超越 Mamba（递推无查询内容选择）的代数根基。详见 math.md §2.9。未形式化的部分（复杂度、
softmax 核近似、「超越 Mamba」经验断言）见「问题 10」。

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

### 问题 8（§4.1.7 的「O(‖v‖²)」需 C² 光滑性，两条公理只给出一阶 o(‖v‖)）

math.md §4.1.7 写 `exp_z(v) = R_z(v) + O(‖v‖²)` 与「每步偏差 O(‖Δt·v‖²)」。但
`Retraction.lean` 形式化的两条公理（`R(0)=z`、`dR(0)=id`，仅要求 Fréchet 可微）**只蕴含**
一阶一致 `R(v)=z+v+o(‖v‖)`（`first_order_consistency`），不蕴含二阶界。要从
`o(‖v‖)` 升级到 `O(‖v‖²)`，需要 retraction 至少 `C²`（或 `C¹` 且导数 Lipschitz），使
Taylor 余项被二次项控制——这是标准事实，但**不在两条公理之内**。

**修正建议（面向 GPU 友好近似）：把「每步偏差 O(‖Δt·v‖²)」拆成两类正交误差、分两档
retraction 表述，而不是合并成一句：**

1. **截断误差（随 Δt→0 消失）**：对**光滑** retraction（含线性投影 `R_z(v)=z+P_z v`），
   Taylor 余项给出 `R_z(v)=z+v+O(‖v‖²)`，故每步 off-manifold 偏差 `O(Δt²)`、可被 reflow
   吸收。前提是 retraction 为 `C²`——两条公理只给 `o(‖v‖)`。
2. **量化误差（不随 Δt→0 消失）**：**硬 argmax** 投影 `E(g(z+v))` 在 Voronoi 胞内是常数
   （`dR=0`）、跨胞界是 O(1) 跳变，既非 C² 也不一阶一致，其「每步偏差」是 **O(1) 量化跳变**，
   只能靠增大 `|𝒱|`/`d` 或软量化（FSQ/温度退火，§4.2(f)）压小，**不能**靠减小 Δt 或 reflow 消除。

建议 §4.1.7 结论句改为：「光滑 retraction（线性投影）把 off-manifold 偏差压到每步 O(Δt²)、
可被 reflow 吸收；硬 argmax 读出引入的是 O(1) 量化地板（码本分辨率），二者是正交的两类
误差，不应合并成同一句『每步 O(Δt²)』」。

### 问题 9（§4.1.7「投影 retraction」是 retraction 需光滑子流形前提，本目录只机械化代数核心）

§4.1.7 断言 `R_z(v)=E(g(z+v))` 是 retraction。`Retraction.lean` 已机械化其**代数核心**：
幂等 `π∘π=π`（`IsProjection`）、基点 `R_z(0)=z`（当 `π z = z`）、像落在不动点集
`π(R_z v)=R_z v`（`projectionRetraction_lands_on_manifold`）。但「`dR_z(0)=id`」这条
`fderiv_id` 公理要求 `π=E∘g` 是**光滑**投影到子流形 `M=Fix(π)`、且 `dπ(z)` 在切空间
`T_zM` 上为恒等——这属于子流形/横截性范畴，本目录**未机械化**，且对离散 argmax 量化器
`g`（分段常值、不可微）该光滑前提天然不成立。更严重的是：硬 argmax 的 `E(g(z+v))` 是
**分段常值**（胞内 `dR=0`、胞界跳变），连一阶一致都不满足（导数在胞内恒为 0，不是 id）。

**修正建议（面向 GPU 友好近似）：把「自然 retraction = 投影（decode–reencode）」改为两层，**
并已用 `linearProjectionRetraction_fderiv` / `linearProjection_idOnImage` /
`linearProjectionRetraction_eq_euclidean_of_id` 三条定理机械化：

1. **连续流的 retraction（软、一阶精确、GPU 友好）**：`R_z(v)=z+P_z v`，其中
   `P_z=J_D^+J_D` 是拉回度量 `G=J_D^TJ_D` 的像空间（token 区分方向 = `T_zM`）正交投影。
   它是**线性（C^∞）、一阶精确（无 o 项）、每步 O(1)、秩 ≤ min(|𝒱|,d)**。
   `linearProjectionRetraction_fderiv` 证明其导数恰为 `P_z`；`linearProjection_idOnImage`
   证明 `P_z` 在 `im J_Dᵀ` 上恒等——二者合起来即「在切空间 `T_zM=im J_Dᵀ` 上是一阶一致的
   retraction」，且它是 §4.1.7 梯度 3「低秩度量利用」的精确形式。
2. **硬 decode–reencode 的职责**：`E(g(z+v))` 是**离散读出/锚定**（snap-to-codebook），保证
   落在可解码点上，但它是分段常值（胞内 `dR=0`、胞界跳变），**不是** retraction；应定位为
   「最终生成 token 的离散化」（§4.2(f) 的 argmax/FSQ），而非连续流的每步映射。

建议 §4.1.7 把「投影 retraction」明确为「**切空间正交投影（软）** `R_z(v)=z+P_z v`」，并把
`E(g(z+v))` 改述为「离散读出/锚定」，避免把分段常值的硬量化当作一阶一致的 retraction。
这样「欧氏 FM + 软投影」才是最小 GPU 友好的 RFM 近似：连续流每步用 `z←z+P_z v`（O(1)、
秩 r≤min(|𝒱|,d)），只在最后读出一步做硬 argmax。

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

### 问题 10（§2.9 CARO：复杂度 / GPU / softmax 近似 / 「超越 Mamba」均为设计断言，未形式化）

`AssociativeMemory.lean` 机械化的是 CARO 的**代数恒等式与收敛性**（任意域 `K`、模 `V`、
双线性型 `B`），但以下断言**未在 Lean 中形式化**，属设计分析，需实测或额外理论：

1. **复杂度 `O(·)`**：§2.9.3 的五维复杂度表（训练计算/显存、推理单步/总耗/显存）是渐进
   分析；复杂度不是 mathlib 已机械化的对象，常数因子、kernel 融合、真实墙钟时间需实测。
2. **「超越 Mamba」的经验断言**：Mamba 长程 needle 检索弱、CARO 内容寻址更强，是基于机制
   的判断，未做基准实测；在特定数据/超参下可能不成立。
3. **softmax 匹配质量（kernel caveat）**：(CARO-1) 对任意双线性 `B` 精确成立，但 `B` 等价于
   线性注意力核；要达到 softmax-attention 质量需正核近似（`exp(k·q)` 等），这是工程近似，
   近似误差未形式化。
4. **GPU 落地**：CARO 的并行扫描复用 §2.6/§3.7.3 的同原语，但实际 kernel 实现（Triton/pscan）、
   bf16/AMP、`torch.compile` 融合均未在本目录验证。

形式化与未形式化的边界在 math.md §2.9.6 与 §1.0 对照表中如实标注，不把设计断言伪装成定理。

## 复现

```bash
cd lean-math && lake build
```

（Lean 4.36.0 + mathlib，见 `lean-toolchain`。全部 10 个模块——含纯 Lean 的
`FixedPoint` / `Semigroup`——均由 `lake build` 一次编译验证，无 `sorry`/`axiom`。）
