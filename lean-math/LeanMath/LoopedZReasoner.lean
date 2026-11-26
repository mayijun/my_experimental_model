/-
§3.10 替代结构「Looped Z-Reasoner」的数学核心：
把 Z 内的 N 步算子复合 Ψ^(N)（§3.5 / §3.7）与解码端 N 步 Euler 流（§4.1）
折叠为 **单一权重共享算子在 Z 上迭代到不动点** 的内部回环：

    展开基线：   Z^(0) = Z_in,  Z^(k+1) = Ψ_{θ_k, h_k}(Z^(k)),  Z_out = r(Z^(N))
    回环推理器： Z^(0) = Z_in,  Z^(k+1) = Ψ_θ(Z^(k); h^(k)),  h^(k) = H(Z^(k))
                 Z_out = r(Z^(T))           （或收敛到不动点 Z* 后读出 r(Z*)）

与 §2.10 Looped Encoder 的关键区别：
  - §2.10 循环「输入 → Z」的编码器块（样本内权重固定）；
  - 本节循环「Z → Z」的推理算子 Ψ_h（h 跨步自推断 = 状态依赖）。
  - 状态依赖使 Z 回环多两个独有代价：① 收缩约束（膨胀算子不动点不可达，
    `expanding_diverges`）；② h 未收敛时无单算子不动点，DEQ 失效（已由
    `FixedPoint.stabilized_is_fixed_point` 刻画，本节引用）。

线性化骨架（固定模式 h 收敛处）即仿射迭代 Z ↦ M·Z + b，与 §3.7.3 PSCO / §2.9 CARO /
§2.10 MR-Loop 是同一个仿射不动点迭代（`Affine.lean` 幺半群），收敛性复用已验证结论。

本文件机械化以下断言（均无 sorry / 无新增公理）：

1. **表达力三层包含**（纯函数代数）：
   全绑定回环 f^∘T  ⊆  状态依赖回环 sdl Ψ h T（各步算子 ∈ im Ψ）  ⊆  全独立前馈。
   `tied_is_sdl` / `sdl_realized_by_feedforward`。
2. **参数容量计数**：基线 N·p_Ψ vs 回环 p_Ψ + p_h；p_h ≤ (N-1)·p_Ψ 时省参
   （`param_loop_le_free` / `param_loop_lt_free`）——h 网远小于算子网时显著省。
3. **反算子塌缩（模式分离）**：固定线性部分 M 下，偏移 b 单射决定不动点：
   b₁ ≠ b₂ ⇒ fp₁ ≠ fp₂（`fixedPoint_diff_b_diff_fp`，任意线性 M 成立，无需收缩）。
   即不同推理指令给出不同结论，Ψ_h 不坍缩到单一答案。
4. **Neumann 级数极限**（式 (34) BPTT vs DEQ）：|a| < 1 时
   Σ_{i<k} a^i → 1/(1-a)（`geometric_series_limit`，拼 `Contractive.geom_sum_one_sub`）。
5. **收缩约束代价**：K > 1 时膨胀算子 x ↦ K·x 从 x₀ ≠ 0 迭代 ‖K^n·x₀‖ → ∞
   （`expanding_diverges`）——回环只能求吸引不动点，能表达的算法类受限。
6. **串行深度对比**（本节独有算力收益）：并行扫描深度 `scanDepth` 满足
   `scanDepth(2T) = scanDepth(T) + 1`（对数增长），展开 `unrolledDepth` 线性增长；
   `scanDepth ≤ unrolledDepth` 且 `scanDepth = O(unrolledDepth)`。
7. **DEQ 极限**：‖M‖ < 1 时从任意初值 Z_in 迭代按范数 → 不动点 fp
   （`reasoner_tendsTo_fixedPoint`，推广 §2.10 `mrLoop_tendsTo_fixedPoint` 到任意初值）。

未形式化（设计分析，如实标注，见 math.md §3.10 与 README 问题 12）：
常数因子 / 墙钟 / 访存 / kernel 融合、非线性 Ψ_h 表达力与「同参更深」泛化、
DEQ 数值稳定性、状态依赖段 m 的最优取值——这些依赖实测，非 Lean 可机械化对象。
-/

import Mathlib
import LeanMath.Affine
import LeanMath.AssociativeMemory
import LeanMath.Contractive
import LeanMath.FixedPoint
import LeanMath.LoopedEncoder

open scoped BigOperators
open Asymptotics Filter

namespace LoopedZReasoner

universe u v

/-! ## 1. 表达力三层包含：全绑定 ⊆ 状态依赖回环 ⊆ 全独立前馈 -/

variable {Z : Type u} {H : Type v}

/-- 前馈网：层列表的复合。`feedforward [f₁, …, f_N] = f₁ ∘ ⋯ ∘ f_N`（先 f_N 后 f₁）。 -/
def feedforward (fs : List (Z → Z)) : Z → Z :=
  List.foldr (fun f g => f ∘ g) id fs

/-- 状态依赖回环：从 z₀ 起依次作用 Ψ(h 0), Ψ(h 1), …, Ψ(h (T-1))。
    对应 math.md (LZR-2)：Z^(k+1) = Ψ_{h^(k)}(Z^(k))，h^(k) = H(Z^(k)) 跨步自推断。 -/
def sdl (Ψ : H → Z → Z) (h : Nat → H) : Nat → Z → Z
  | 0, z => z
  | k + 1, z => Ψ (h k) (sdl Ψ h k z)

/-- 右包含：任意状态依赖回环都能被某个层列表实现（层 i = Ψ(h i)，全在 im Ψ 内）。
    故状态依赖回环函数族 ⊆ 全独立前馈函数族。 -/
theorem sdl_realized_by_feedforward (Ψ : H → Z → Z) (h : Nat → H) (T : Nat) :
    ∃ (fs : List (Z → Z)), fs.length = T ∧ feedforward fs = sdl Ψ h T := by
  induction T with
  | zero =>
    refine ⟨[], by simp, ?_⟩
    funext z
    simp [feedforward, sdl]
  | succ T ih =>
    rcases ih with ⟨fs, hlen, hfs⟩
    refine ⟨(Ψ (h T)) :: fs, by simp [hlen], ?_⟩
    funext z
    have hstep : feedforward ((Ψ (h T)) :: fs) z = Ψ (h T) (sdl Ψ h T z) := by
      calc
        feedforward ((Ψ (h T)) :: fs) z
            = (Ψ (h T) ∘ feedforward fs) z := by rfl
        _ = Ψ (h T) (feedforward fs z) := rfl
        _ = Ψ (h T) (sdl Ψ h T z) := by rw [hfs]
    rw [hstep]
    rfl

/-- 左包含：全绑定回环 f^∘T（所有层取同一个 f）是状态依赖回环的特例——取 H = Unit、
    h 为常值。故全绑定函数族 ⊆ 状态依赖回环函数族。 -/
theorem tied_is_sdl (f : Z → Z) (T : Nat) :
    (fun z : Z => f^[T] z) =
      sdl (fun (_ : Unit) => f) (fun (_ : Nat) => ()) T := by
  funext z
  induction T with
  | zero => simp [sdl]
  | succ T ih =>
    calc
      f^[T + 1] z
          = f (f^[T] z) := by simp [Function.iterate_succ_apply']
      _ = f (sdl (fun (_ : Unit) => f) (fun (_ : Nat) => ()) T z) := by rw [ih]
      _ = sdl (fun (_ : Unit) => f) (fun (_ : Nat) => ()) (T + 1) z := by
        simp [sdl]

/-! ## 2. 参数容量计数 -/

/-- 展开基线：N 步各一套算子参数 ⇒ N·p_Ψ。 -/
def param_free (N pΨ : Nat) : Nat := N * pΨ

/-- 回环：单一算子 p_Ψ + h 网 p_h（与迭代次数 T 无关）。 -/
def param_loop (pΨ ph : Nat) : Nat := pΨ + ph

/-- 当 h 网参数不超过省下的 (N-1)·p_Ψ 时，回环总参数 ≤ 展开基线。 -/
theorem param_loop_le_free (N pΨ ph : Nat) (hN : 1 ≤ N)
    (h : ph ≤ (N - 1) * pΨ) :
    param_loop pΨ ph ≤ param_free N pΨ := by
  rcases Nat.exists_eq_succ_of_ne_zero (show N ≠ 0 from by omega) with ⟨m, rfl⟩
  have hpred : m + 1 - 1 = m := by omega
  have h' : ph ≤ m * pΨ := by simpa [hpred] using h
  have hsum : pΨ + ph ≤ pΨ + m * pΨ := by linarith
  have hN' : pΨ + m * pΨ = (m + 1) * pΨ := by ring
  rw [hN'] at hsum
  simpa [param_loop, param_free] using hsum

/-- 当 N ≥ 2 且 p_h 严格小于 (N-1)·p_Ψ 时，回环严格省参（容量代价的核心收益）。 -/
theorem param_loop_lt_free (N pΨ ph : Nat) (hN : 1 < N)
    (h : ph < (N - 1) * pΨ) :
    param_loop pΨ ph < param_free N pΨ := by
  rcases Nat.exists_eq_succ_of_ne_zero (show N ≠ 0 from by omega) with ⟨m, rfl⟩
  have hpred : m + 1 - 1 = m := by omega
  have h' : ph < m * pΨ := by simpa [hpred] using h
  have hsum : pΨ + ph < pΨ + m * pΨ := by linarith
  have hN' : pΨ + m * pΨ = (m + 1) * pΨ := by ring
  rw [hN'] at hsum
  simpa [param_loop, param_free] using hsum

/-! ## 3. 反算子塌缩：偏移 b 单射决定不动点（模式分离） -/

variable {K : Type*} [Field K]
variable {V : Type u} [AddCommGroup V] [Module K V]

/-- 对固定线性部分 M，不动点方程 fp = M·fp + b 中 b 由 fp 唯一决定：
    fp₁ = fp₂ ⇒ b₁ = b₂。其逆否命题即模式分离：b₁ ≠ b₂ ⇒ fp₁ ≠ fp₂。
    含义：不同推理指令（不同 b_h）给出不同结论不动点，Ψ_h 不坍缩到单一答案。
    注意：此结论对任意线性 M 成立（无需收缩）——即使 M 不收缩、不动点迭代不收敛，
    不动点本身仍单射依赖于 b。 -/
theorem fixedPoint_diff_b_diff_fp (M : V →ₗ[K] V) (b₁ b₂ fp₁ fp₂ : V)
    (h1 : fp₁ = M fp₁ + b₁) (h2 : fp₂ = M fp₂ + b₂)
    (hne : b₁ ≠ b₂) : fp₁ ≠ fp₂ := by
  intro heq
  have hb1 : b₁ = fp₁ - M fp₁ := by
    have hstep : fp₁ - M fp₁ = M fp₁ + b₁ - M fp₁ := congrArg (fun x => x - M fp₁) h1
    have : fp₁ - M fp₁ = b₁ := by rw [hstep]; abel
    exact this.symm
  have hb2 : b₂ = fp₂ - M fp₂ := by
    have hstep : fp₂ - M fp₂ = M fp₂ + b₂ - M fp₂ := congrArg (fun x => x - M fp₂) h2
    have : fp₂ - M fp₂ = b₂ := by rw [hstep]; abel
    exact this.symm
  have hb : b₁ = b₂ := by
    rw [hb1, hb2, heq]
  exact hne hb

/-! ## 4. Neumann 级数极限（式 (34)：BPTT 截断 vs DEQ 完整逆） -/

/-- 实幂 a^n → 0 当 |a| < 1（允许 a 为负）。拼 `AssocMem.real_pow_tendsto_zero`
    （其要求 0 ≤ a < 1）于 |a|，再用 |a^n| = |a|^n 夹逼。 -/
lemma pow_abs_tendsto_zero (a : ℝ) (h : |a| < 1) :
    Filter.Tendsto (fun n : Nat => a ^ n) Filter.atTop (nhds (0 : ℝ)) := by
  have hnonneg : 0 ≤ |a| := abs_nonneg a
  have h1 : Filter.Tendsto (fun n : Nat => |a| ^ n) Filter.atTop (nhds (0 : ℝ)) :=
    AssocMem.real_pow_tendsto_zero hnonneg h
  have h2 : ∀ n : Nat, ‖‖a ^ n‖‖ ≤ |a| ^ n := by
    intro n
    rw [Real.norm_of_nonneg (show 0 ≤ ‖a ^ n‖ from norm_nonneg (a ^ n))]
    rw [Real.norm_eq_abs, abs_pow]
  have h3 : Filter.Tendsto (fun n : Nat => ‖a ^ n‖) Filter.atTop (nhds (0 : ℝ)) :=
    squeeze_zero_norm h2 h1
  exact tendsto_zero_iff_norm_tendsto_zero.2 h3

/-- 几何级数极限：|a| < 1 ⇒ Σ_{i<k} a^i → 1/(1-a)。
    代数核心是 `Contractive.geom_sum_one_sub`：(1-a)·Σ_{i<k} a^i = 1 - a^k；
    再由 a^k → 0（`pow_abs_tendsto_zero`）取极限。
    对应 math.md (LZR-5) / §4.1.5 式 (34)：L 步 BPTT 截断 Neumann 级数
    Σ_{k<L} J^k 在 ρ(J)<1 时收敛到完整逆 (I-J)^{-1}（标量版，矩阵同构）。 -/
theorem geometric_series_limit (a : ℝ) (ha : |a| < 1) :
    Filter.Tendsto (fun k : Nat => (∑ i ∈ Finset.range k, a ^ i : ℝ))
      Filter.atTop (nhds (1 / (1 - a))) := by
  have ha1 : a ≠ 1 := by
    intro h
    rw [h] at ha
    norm_num at ha
  have hne : (1 - a : ℝ) ≠ 0 := by
    intro h2
    have : a = 1 := by linarith
    exact ha1 this
  have hsum : ∀ k : Nat, (1 - a) * (∑ i ∈ Finset.range k, a ^ i) = 1 - a ^ k :=
    fun k => Contractive.geom_sum_one_sub a k
  have heq : ∀ k : Nat, (∑ i ∈ Finset.range k, a ^ i) = (1 - a ^ k) / (1 - a) := by
    intro k
    apply (eq_div_iff hne).mpr
    have : (∑ i ∈ Finset.range k, a ^ i) * (1 - a) = 1 - a ^ k := by
      have hmul : (1 - a) * (∑ i ∈ Finset.range k, a ^ i) = 1 - a ^ k := hsum k
      rw [mul_comm] at hmul
      exact hmul
    exact this
  have hlim_a : Filter.Tendsto (fun k : Nat => a ^ k) Filter.atTop (nhds (0 : ℝ)) :=
    pow_abs_tendsto_zero a ha
  have hconst : Filter.Tendsto (fun (_ : Nat) => (1 : ℝ)) Filter.atTop (nhds (1 : ℝ)) :=
    tendsto_const_nhds
  have hlim1 : Filter.Tendsto (fun k : Nat => (1 : ℝ) - a ^ k) Filter.atTop (nhds (1 : ℝ)) := by
    simpa [sub_zero] using hconst.sub hlim_a
  have hlim2 : Filter.Tendsto (fun k : Nat => (1 - a ^ k) / (1 - a))
      Filter.atTop (nhds (1 / (1 - a))) :=
    hlim1.div tendsto_const_nhds hne
  simpa [heq] using hlim2

/-! ## 5. 收缩约束代价：膨胀算子不动点迭代发散 -/

/-- 膨胀代价：K > 1 时，线性算子 x ↦ K·x 的不动点是 0，但从任意 x₀ ≠ 0 迭代，
    ‖K^n·x₀‖ → ∞（不收敛到不动点 0）。
    含义：回环求不动点要求算子收缩（ρ(J) < 1）；膨胀型算子的不动点**迭代不可达**，
    故回环只能表达「收缩复合」类算法——这是为参数共享 + O(log N) 深度 + O(1) 显存
    付出的表达力代价（展开基线无此约束，每步可膨胀）。 -/
theorem expanding_diverges (K : ℝ) (hK : 1 < K) (x0 : ℝ) (hx : x0 ≠ 0) :
    Filter.Tendsto (fun n : Nat => ‖K ^ n * x0‖) Filter.atTop Filter.atTop := by
  have hKpos : 0 < K := by linarith
  have h₁ : ∀ n : Nat, ‖K ^ n * x0‖ = K ^ n * |x0| := by
    intro n
    rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_pos hKpos]
  have habs : 0 < |x0| := abs_pos.mpr hx
  have h2 : Filter.Tendsto (fun n : Nat => K ^ n) Filter.atTop Filter.atTop :=
    tendsto_pow_atTop_atTop_of_one_lt hK
  have h3' : Filter.Tendsto (fun n : Nat => |x0| * K ^ n) Filter.atTop Filter.atTop :=
    h2.const_mul_atTop habs
  have h3 : Filter.Tendsto (fun n : Nat => K ^ n * |x0|) Filter.atTop Filter.atTop := by
    convert h3' using 1
    funext n
    ring
  have heq : (fun n : Nat => ‖K ^ n * x0‖) = fun n : Nat => K ^ n * |x0| := by
    funext n
    exact h₁ n
  rw [heq]
  exact h3

/-! ## 6. 串行深度对比：并行扫描 O(log N) vs 展开 O(N)（本节独有算力收益） -/

/-- 展开基线的串行深度 = 步数 N（每步串行依赖前一步）。 -/
def unrolledDepth (n : Nat) : Nat := n

/-- 并行扫描（Hillis-Steele / Blelloch）的串行深度：每轮把 offset 翻倍，
    故深度 ≈ ⌈log₂ N⌉。用 `Nat.strongRec` 强递归定义，使「翻倍加一」性质可证：
    n ≥ 2 时 scanDepth(n) = scanDepth(n/2) + 1，n < 2 时为 0。 -/
def scanDepth : Nat → Nat :=
  Nat.strongRec fun n ih =>
    if h : 2 ≤ n then
      ih (n / 2) (Nat.div_lt_self (by omega) Nat.one_lt_two) + 1
    else
      0

/-- n < 2 时 scanDepth(n) = 0。 -/
lemma scanDepth_base (n : Nat) (h : n < 2) : scanDepth n = 0 := by
  rw [scanDepth, Nat.strongRec_eq]
  have h' : ¬(2 ≤ n) := by omega
  simp [h']

/-- 对 n ≥ 2，扫描深度的递推式：scanDepth(n) = scanDepth(n/2) + 1。 -/
lemma scanDepth_eq (n : Nat) (hn : 2 ≤ n) : scanDepth n = scanDepth (n / 2) + 1 := by
  rw [scanDepth, Nat.strongRec_eq]
  simp [hn]
  rfl

/-- 扫描深度对数增长：输入翻倍时深度只 +1（对比展开的 ×2）。 -/
theorem scanDepth_double (m : Nat) (hm : 1 ≤ m) :
    scanDepth (2 * m) = scanDepth m + 1 := by
  have h2 : 2 ≤ 2 * m := by omega
  rw [scanDepth_eq (2 * m) h2]
  have hdiv : (2 * m) / 2 = m := Nat.mul_div_cancel_left m (by norm_num)
  rw [hdiv]

/-- 展开深度线性增长：输入翻倍时深度 ×2。 -/
theorem unrolled_double (m : Nat) :
    unrolledDepth (2 * m) = 2 * unrolledDepth m := by
  rfl

/-- 扫描深度不超过展开深度：log₂ N ≤ N。 -/
theorem scanDepth_le_unrolled (n : Nat) : scanDepth n ≤ n := by
  induction n using Nat.strong_induction_on with
  | ind n ih =>
    by_cases h : 2 ≤ n
    · rw [scanDepth_eq n h]
      have hsmall : n / 2 < n := Nat.div_lt_self (by omega) Nat.one_lt_two
      have hrec : scanDepth (n / 2) ≤ n / 2 := ih (n / 2) hsmall
      have hle : n / 2 + 1 ≤ n := by omega
      linarith
    · rw [scanDepth_base n (by omega)]
      omega

/-- 扫描深度 = O(展开深度)，即 log₂ N = O(N)（常数 1）。
    与 §2.10 的 big-O 形式化风格一致；严格的「log N = o(N)」是其强化（设计跟进）。 -/
theorem scanDepth_isBigO_unrolled :
    (fun n : Nat => (scanDepth n : ℝ)) =O[Filter.atTop] (fun n : Nat => (n : ℝ)) := by
  rw [isBigO_iff_isBigOWith]
  refine ⟨1, ?_⟩
  rw [isBigOWith_iff, eventually_atTop]
  refine ⟨0, fun n _ => ?_⟩
  have hle : scanDepth n ≤ n := scanDepth_le_unrolled n
  have hpos : 0 ≤ (n : ℝ) := by positivity
  have hpos2 : 0 ≤ (scanDepth n : ℝ) := by positivity
  rw [Real.norm_of_nonneg hpos2, Real.norm_of_nonneg hpos]
  have hcast : (scanDepth n : ℝ) ≤ (n : ℝ) := by exact_mod_cast hle
  simpa [mul_one] using hcast

/-! ## 7. DEQ 极限：从任意初值迭代到不动点（推广 §2.10 到任意初值） -/

/-- DEQ 极限：‖M‖ < 1 且 fp 是仿射迭代 fp = M·fp + b 的不动点时，从任意初值 z0
    出发的迭代 PSCO.iterate M b t z0 按范数收敛到 fp。
    推广 §2.10 `mrLoop_tendsTo_fixedPoint`（其初值固定为 0、b = M·e）到任意初值 z0——
    Z 回环的初值是 Z_in（编码器输出或上一阶段状态），不必为 0。
    拼 `LoopedEncoder.affine_iter_error`（误差恒等式）+ `LoopedEncoder.gradient_decay`
    （算子范数幂次界）+ `AssocMem.real_pow_tendsto_zero`（K^t → 0），经 squeeze_zero。 -/
theorem reasoner_tendsTo_fixedPoint {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (M : E →L[ℝ] E) (b z0 fp : E)
    (hfp : fp = M fp + b) (hK : ‖M‖ < 1) :
    Filter.Tendsto (fun t : Nat => PSCO.iterate (M.toLinearMap) b t z0)
      Filter.atTop (nhds fp) := by
  have hnonneg : 0 ≤ ‖M‖ := norm_nonneg _
  have htz : Filter.Tendsto (fun t : Nat => ‖M‖ ^ t) Filter.atTop (nhds (0 : ℝ)) :=
    AssocMem.real_pow_tendsto_zero hnonneg hK
  have htz2 : Filter.Tendsto (fun t : Nat => ‖M‖ ^ t * ‖z0 - fp‖)
      Filter.atTop (nhds (0 : ℝ)) := by
    have h := htz.const_mul ‖z0 - fp‖
    simpa [mul_comm] using h
  have hle : ∀ t : Nat, ‖PSCO.iterate (M.toLinearMap) b t z0 - fp‖ ≤ ‖M‖ ^ t * ‖z0 - fp‖ := by
    intro t
    have h1 : PSCO.iterate (M.toLinearMap) b t z0 - fp
        = PSCO.powL (M.toLinearMap) t (z0 - fp) :=
      LoopedEncoder.affine_iter_error (M.toLinearMap) b t z0 fp hfp
    rw [h1, LoopedEncoder.powL_toLinearMap M t (z0 - fp)]
    simpa using LoopedEncoder.gradient_decay M t (z0 - fp)
  have hle2 : ∀ t : Nat, ‖‖PSCO.iterate (M.toLinearMap) b t z0 - fp‖‖ ≤ ‖M‖ ^ t * ‖z0 - fp‖ := by
    intro t
    rw [Real.norm_of_nonneg (norm_nonneg (PSCO.iterate (M.toLinearMap) b t z0 - fp))]
    exact hle t
  have hnorm : Filter.Tendsto (fun t : Nat => ‖PSCO.iterate (M.toLinearMap) b t z0 - fp‖)
      Filter.atTop (nhds (0 : ℝ)) := squeeze_zero_norm hle2 htz2
  have hg : Filter.Tendsto (fun t : Nat => PSCO.iterate (M.toLinearMap) b t z0 - fp)
      Filter.atTop (nhds (0 : E)) := tendsto_zero_iff_norm_tendsto_zero.2 hnorm
  have hfp' : Filter.Tendsto (fun _ : Nat => fp) Filter.atTop (nhds fp) := tendsto_const_nhds
  simpa [sub_add_cancel] using hg.add hfp'

end LoopedZReasoner
