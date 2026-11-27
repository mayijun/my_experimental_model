/-
时间分层 = 收缩关联扫描上的跨步前缀选择 + 谱上的慢/快分离（借鉴 H-JEPA 的数学核心）。

对应 math.md 新增章节「时间分层（H-JEPA 借鉴）」，回答两个问题：

  Q1 「可预测性 → 保留/丢弃」补充判据是否数学有利：是。其可判定的代数核心是
     「收缩谱上的慢/快分离」——慢特征（大特征值）在长 horizon 下持久性更大
     （persistence_mono），快特征（a ∈ [0,1)）衰减到 0（fast_decay），快分量相对
     慢分量随 horizon 趋于可忽略（fast_slow_ratio）。故「留慢丢快」是谱夹层下界的
     horizon 化，而非启发式。

  Q2 框架是否已隐含时间分层：未以「独立潜空间堆叠」隐含，但同一收缩关联扫描的
     「跨步前缀」即是一层——分层 = 前缀索引选择，零额外算子；跨步复合满足
     M^(a+b)=M^a∘M^b（powL_add），仍仿射、仍收缩，故每层不动点收敛由基层直接继承
     （iterate_still_contracting）。

本文件机械化（无 sorry / 无新增公理，纯代数/实数）：

1. 慢/快谱分离：持久性 a^k 随 a 单调；快特征衰减到 0；快慢比 (a/b)^k → 0。
2. 跨步 = 算子幂（加法性）：powL M (a+b) = powL M a ∘ powL M b。
3. 跨步保收缩：LipschitzWith K f（K<1）的 n 次迭代仍是严格收缩（常数 K^n<1）。
-/

import Mathlib
import LeanMath.Affine

open scoped BigOperators
open scoped NNReal

namespace TemporalHierarchy

universe u

/-! ## 1. 慢/快谱分离（「可预测性 → 保留/丢弃」判据的代数核心） -/

/-- 持久性单调：0 ≤ a ≤ b ⟹ a^k ≤ b^k。
    对角收缩 M=diag(ã) 的分量 i 在 horizon k 的持久性为 ã_i^k；
    特征值越大（越「慢」）持久性越大，即越值得在高层保留。 -/
theorem persistence_mono (a b : ℝ) (ha0 : 0 ≤ a) (hab : a ≤ b) (k : ℕ) :
    a ^ k ≤ b ^ k := by
  exact pow_le_pow_left₀ ha0 hab k

/-- 快特征衰减到 0：a ∈ [0,1) ⟹ a^k → 0。
    收缩谱上任何分量最终都衰减（收敛到不动点），快特征（小 a）衰减更快。 -/
theorem fast_decay (a : ℝ) (ha0 : 0 ≤ a) (ha1 : a < 1) :
    Filter.Tendsto (fun k : ℕ => a ^ k) Filter.atTop (nhds (0 : ℝ)) := by
  exact tendsto_pow_atTop_nhds_zero_of_lt_one ha0 ha1

/-- 快慢比趋于 0：0 ≤ a < b < 1 ⟹ (a/b)^k → 0。
    即快分量（a）相对慢分量（b）随 horizon 趋于可忽略——「留慢丢快」的精确数学含义：
    长时域下慢分量主导，快分量可安全丢弃。 -/
theorem fast_slow_ratio (a b : ℝ) (ha0 : 0 ≤ a) (hab : a < b) (hb1 : b < 1) :
    Filter.Tendsto (fun k : ℕ => (a / b) ^ k) Filter.atTop (nhds (0 : ℝ)) := by
  have hb0 : 0 < b := lt_of_le_of_lt ha0 hab
  have hrat0 : 0 ≤ a / b := div_nonneg ha0 (le_of_lt hb0)
  have hrat1 : a / b < 1 := (div_lt_one hb0).mpr hab
  exact tendsto_pow_atTop_nhds_zero_of_lt_one hrat0 hrat1

/-! ## 2. 跨步 = 算子幂（时间分层 = 跨步前缀选择） -/

/-- 幂的加法性：M^(a+b) = M^a ∘ M^b。
    这是「把 a 步打包成一步、再走 b 步」的代数前提——打包后仍可自由组合。
    即：第 ℓ 层算子（走 a 步）与第 ℓ+1 层算子（走 b 步）复合 = 走 a+b 步。 -/
theorem powL_add {K : Type*} [Field K] {V : Type u} [AddCommGroup V] [Module K V]
    (M : V →ₗ[K] V) (a b : ℕ) :
    PSCO.powL M (a + b) = (PSCO.powL M a).comp (PSCO.powL M b) := by
  induction a with
  | zero =>
      rw [Nat.zero_add]
      change PSCO.powL M b = LinearMap.id.comp (PSCO.powL M b)
      rw [LinearMap.id_comp]
  | succ a ih =>
      rw [Nat.succ_add]
      simp only [PSCO.powL]
      rw [ih]
      rfl

/-! ## 3. 跨步保收缩（每层不动点收敛由基层继承） -/

/-- n 次迭代的 Lipschitz 常数是 K^n。 -/
theorem lipschitz_iterate {α : Type u} [PseudoEMetricSpace α]
    {K : ℝ≥0} {f : α → α} (hf : LipschitzWith K f) (n : ℕ) :
    LipschitzWith (K ^ n) (f^[n]) := by
  induction n with
  | zero =>
      intro x y
      simp
  | succ n ih =>
      rw [Function.iterate_succ]
      have hpow : K ^ n * K = K ^ (n + 1) := by rw [pow_succ]
      simpa [hpow] using ih.comp hf

/-- 跨步保收缩：严格收缩 f（常数 K<1）的任意正次迭代仍是严格收缩（常数 K^n<1）。
    即每一层（跨步打包）的不动点收敛保证（Banach 不动点定理）由基层直接继承，
    无需为每层重证——这正是「时间分层不破坏框架收缩契约」的代数根基。 -/
theorem iterate_still_contracting {α : Type u} [PseudoEMetricSpace α]
    {K : ℝ≥0} {f : α → α} (hf : LipschitzWith K f) (hK : K < 1) :
    ∀ n : ℕ, 0 < n → ∃ K' : ℝ≥0, K' < 1 ∧ LipschitzWith K' (f^[n]) := by
  intro n hn
  refine ⟨K ^ n, ?_, lipschitz_iterate hf n⟩
  have hpow : K ^ n < 1 := pow_lt_one₀ (show (0 : ℝ≥0) ≤ K from by positivity) hK (Nat.ne_of_gt hn)
  exact hpow

/-! ## 4. 时间分层的不动点不变性（更深入的推导） -/

variable {K : Type*} [Field K]
variable {V : Type u} [AddCommGroup V] [Module K V]

/-- 几何部分和：`geomSum M s = Σ_{i<s} M^i`（线性算子的几何级数前 s 项）。 -/
def geomSum (M : V →ₗ[K] V) (s : ℕ) : V →ₗ[K] V :=
  ∑ i ∈ Finset.range s, PSCO.powL M i

/-- 伸缩求和引理（向量版本）：`Σ_{i<s} M^i z - Σ_{i<s} M^{i+1} z = z - M^s z`。 -/
lemma geom_telescope_apply (M : V →ₗ[K] V) (s : ℕ) (z : V) :
    (∑ i ∈ Finset.range s, (PSCO.powL M i) z)
      - (∑ i ∈ Finset.range s, (PSCO.powL M (i + 1)) z)
    = z - (PSCO.powL M s) z := by
  induction s with
  | zero => simp [PSCO.powL]
  | succ s ih =>
      rw [Finset.sum_range_succ, Finset.sum_range_succ]
      have hre : ((∑ i ∈ Finset.range s, (PSCO.powL M i) z) + (PSCO.powL M s) z)
          - ((∑ i ∈ Finset.range s, (PSCO.powL M (i + 1)) z) + (PSCO.powL M (s + 1)) z)
        = ((∑ i ∈ Finset.range s, (PSCO.powL M i) z) - (∑ i ∈ Finset.range s, (PSCO.powL M (i + 1)) z))
          + ((PSCO.powL M s) z - (PSCO.powL M (s + 1)) z) := by abel
      rw [hre, ih]
      abel

/-- 伸缩恒等式（几何级数）：`(id - M) ∘ (Σ_{i<s} M^i) = id - M^s`。
    这是「跨步层的不动点 = 基层的不动点」的代数根基：粗算子 `(M^s, Σ M^i b)` 的
    不动点方程 `(id - M^s) z = Σ M^i b` 与基层 `(id - M) z = b` 由本式互推。 -/
theorem geom_series_telescope (M : V →ₗ[K] V) (s : ℕ) :
    (LinearMap.id - M).comp (geomSum M s) = LinearMap.id - PSCO.powL M s := by
  ext z
  simp [geomSum]
  rw [← geom_telescope_apply M s z]
  rfl

/-- 不动点迭代稳定：`M z + b = z`（z 是基层不动点）⟹ 任意 k 步迭代都停在 z。 -/
lemma fixed_point_iterate (M : V →ₗ[K] V) (b : V) (z : V) (hz : M z + b = z) :
    ∀ k : ℕ, PSCO.iterate M b k z = z := by
  intro k
  induction k with
  | zero => simp [PSCO.iterate]
  | succ k ih => simp [PSCO.iterate, ih, hz]

/-- **不动点的跨步不变性**（时间分层的核心）：基层不动点 z（满足 `M z + b = z`）
    也是任意 stride s 的粗算子 `(M^s, Σ_{i<s} M^i b)` 的不动点。即「抽象结论（不动点）
    不随时间分辨率改变」——这是 §3.11 谱时间分层「每层结论一致」的代数保证。 -/
theorem fixedPoint_stride_invariant (M : V →ₗ[K] V) (b : V) (z : V) (hz : M z + b = z) (s : ℕ) :
    (PSCO.powL M s) z + (∑ i ∈ Finset.range s, (PSCO.powL M i) b) = z := by
  have h := fixed_point_iterate M b z hz s
  rw [PSCO.iterate_eq_closedForm] at h
  simpa [PSCO.closedForm] using h

/-- **速率-分辨率权衡**：`K ∈ [0,1]` 时 `K^s ≤ K`（`s ≥ 1`）。
    即 stride 越大（层越「粗」），收缩常数越小，Banach 迭代收敛越快——但时间分辨率
    越粗（每步覆盖越长 horizon）。这是「越粗越快收敛」的定量形式。 -/
theorem rate_mono_stride {K : ℝ≥0} (hK1 : K ≤ 1) {s : ℕ} (hs : 1 ≤ s) :
    K ^ s ≤ K := by
  simpa using (pow_le_pow_right_of_le_one' (a := K) hK1 hs)

/-! ## 5. 最优 stride 的闭式（速率-分辨率权衡的定量形式） -/

/-- **跨步幂的乘法性**：`(K^s)^n = K^(s·n)`。
    即 n 次 stride-s 粗迭代的残余界是 `K^(s·n)`，**只依赖总 horizon `s·n`**，
    与如何把 `s·n` 分解成 (stride, 迭代次数) 无关。这是「平权权衡」的代数根基：
    任何 stride 选择在相同总 horizon 下给出相同残余界。 -/
theorem pow_mul_stride {K : ℝ≥0} (s n : ℕ) :
    (K ^ s) ^ n = K ^ (s * n) := by
  exact (pow_mul K s n).symm

/-- **残余随总 horizon 单调递减**：`K ∈ [0,1]` 且 `H ≤ H'` ⟹ `K^{H'} ≤ K^H`。
    即覆盖越远（总 horizon 越大），残余越小。 -/
theorem residual_horizon_mono {K : ℝ≥0} (hK1 : K ≤ 1) {H H' : ℕ} (hHH' : H ≤ H') :
    K ^ H' ≤ K ^ H := by
  simpa using (pow_le_pow_right_of_le_one' (a := K) hK1 hHH')

/-- **迭代次数随 stride 单调递减（非严格）**：`s ≤ s'` ⟹ `K^(s'·n) ≤ K^(s·n)`。
    即 stride 越大（层越粗），相同迭代次数 n 覆盖的残余越小；等价地，达到同一残余
    所需的粗迭代次数不增加——「越粗越快」的定量形式，为「选最大可行 stride」提供依据。 -/
theorem coarse_iterations_mono {K : ℝ≥0} (hK1 : K ≤ 1) {s s' n : ℕ} (hss' : s ≤ s') :
    K ^ (s' * n) ≤ K ^ (s * n) := by
  exact residual_horizon_mono hK1 (Nat.mul_le_mul_right n hss')

/-! ## 6. Z 空间的隐式时间轴（序列输入注入，区别于 §3.11 的时间分层） -/

/-- 序列输入注入的线性递推：`Z^(t+1) = M Z^(t) + x_t`。
    每步注入**不同**输入 x_t（区别于 §2.10 MR-Loop 的固定输入 e）。
    时间 = 输入序列下标 t，折叠进算子幂与卷积，而非 Z 的显式维度——
    这是「原方案单 Z 隐式承载时间轴」的机制，使有时间的视频与无时间的文字同维、可对齐。 -/
def iterateSeq (M : V →ₗ[K] V) (x : Nat → V) : Nat → V → V
  | 0, z => z
  | t+1, z => M (iterateSeq M x t z) + x t

/-- 序列卷积部分和：`Σ_{i<t} M^{t-1-i} x_i`（升幂卷积，离散时间线性系统的输入-状态项）。 -/
def seqConv (M : V →ₗ[K] V) (x : Nat → V) (t : Nat) : V :=
  ∑ i ∈ Finset.range t, (PSCO.powL M (t - 1 - i)) (x i)

/-- 幂的单步左复合：`M ∘ M^{t-1-i} = M^{t-i}`（对 `i < t`）。 -/
lemma powL_comp_succ_sub (M : V →ₗ[K] V) {t i : ℕ} (hi : i < t) :
    M.comp (PSCO.powL M (t - 1 - i)) = PSCO.powL M (t - i) := by
  have h : t - 1 - i + 1 = t - i := by omega
  rw [← h]
  rfl

/-- **序列递推的闭式（升幂卷积）**：`Z^(t) = M^t Z^(0) + Σ_{i<t} M^{t-1-i} x_i`。
    这是「隐式时间」的数学核心：状态 Z^(t) 携带**完整输入历史** (x_0,…,x_{t-1})，
    时间轴折叠进算子 M 的幂与卷积，而非 Z 的显式维度——故 Z 与无时间的文字模态
    同维、可对齐。对比 §2.10 MR-Loop（固定输入 e、收敛到输入依赖但**时间不变**的不动点），
    序列注入（输入随 t 变化）才是「时间」而非「深度」。 -/
theorem iterateSeq_eq_closedForm (M : V →ₗ[K] V) (x : Nat → V) (t : Nat) (z : V) :
    iterateSeq M x t z = (PSCO.powL M t) z + seqConv M x t := by
  induction t generalizing z with
  | zero => simp [iterateSeq, seqConv, PSCO.powL]
  | succ t ih =>
      rw [iterateSeq, ih]
      unfold seqConv
      rw [map_add, map_sum]
      have hshift : (∑ i ∈ Finset.range t, M ((PSCO.powL M (t - 1 - i)) (x i)))
                  = (∑ i ∈ Finset.range t, (PSCO.powL M (t - i)) (x i)) := by
        apply Finset.sum_congr rfl
        intro i hi
        have hit : i < t := Finset.mem_range.mp hi
        rw [← LinearMap.comp_apply, powL_comp_succ_sub M hit]
      rw [hshift]
      rw [show x t = (PSCO.powL M (t - t)) (x t) by simp [PSCO.powL]]
      rw [add_assoc]
      rw [← Finset.sum_range_succ]
      rfl

/-! ## 7. 融合方案：单尺度隐式时间 + 慢特征保护（谱正则 / 记忆门控）

对应 math.md §3.13「多时间尺度 vs 单尺度隐式时间 + ρ压缩」的融合方案。

融合的取舍：
  - 保留 §3.12 的骨架：单 Z 隐式时间（序列输入注入闭式）+ ρ压缩在 token 层；
  - 引入 §3.11 的「留慢丢快」思想，但降级为**谱正则 / 记忆门控**（保护慢特征），
    **不引入显式多层**（避开独立层参数陷阱与平权冗余 `pow_mul_stride`）；
  - 舍弃 §3.11 的显式时间分层（多尺度不提升精度，只调度分辨率/记忆）。

慢特征保护的数学形式：引入**静态线性门控算子** P（谱值 ∈ [0,1]，LipschitzWith 1），
被保护算子 M̃ = P ∘ M。P 静态线性 ⟹ M̃ 仍线性 ⟹ PSCO 仿射幺半群 + 并行扫描 + 闭式卷积
全部保留（`iterateSeq_eq_closedForm` 对任意线性 M 通用）；P 在慢方向接近恒等（保留）、
快方向接近 0（丢弃）⟹ 谱抬升 ⟹ 长程记忆下界抬升。

本文件机械化（无 sorry / 无新增公理）：

7.1 保护保持严格收缩：M̃ = P ∘ M 的 Lipschitz 常数 ≤ K（不放大）。
7.2 保护强度单调：遗忘率 α 越小（保护越强）⟹ 被保护谱值 1-α(1-a) 越大。
7.3 长程记忆下界抬升：谱抬升 γ_slow < γ ⟹ (1-γ_slow)^k > (1-γ)^k（严格）。
7.4 慢/快记忆比 → ∞：a_fast < a_slow ⟹ (a_slow/a_fast)^k → atTop（长程慢方向主导）。 -/

/-- **保护保持严格收缩**：门控 P（LipschitzWith 1，如软投影/遗忘门，谱值 ∈ [0,1]）
    与收缩 M（LipschitzWith K，K<1）复合后仍是严格收缩，且常数仍 ≤ K。
    即「保护」不放大收缩常数 ⟹ 融合不破坏 §3.7.1 的 Banach 收敛契约，
    也不破坏 §3.7.3 的 PSCO 并行扫描（M̃ 仍可组合、可扫描）。 -/
theorem protect_still_contracting {α : Type u} [PseudoEMetricSpace α]
    {K : ℝ≥0} {M P : α → α}
    (hM : LipschitzWith K M) (hP : LipschitzWith 1 P) (hK : K < 1) :
    LipschitzWith K (P ∘ M) := by
  simpa using (hP.comp hM)

/-- **保护强度单调**：遗忘率 α 越小（保护越强）⟹ 被保护谱值 ã̃ = 1 - α(1-ã) 越大。
    即门控系数 α ∈ [0,1] 连续调节「遗忘多少」：α=1 无保护（ã̃=ã）、α→0 完全保护（ã̃→1）。 -/
theorem protection_spectral_mono (a : ℝ) (ha1 : a ≤ 1) {α β : ℝ} (hαβ : α ≤ β) :
    1 - β * (1 - a) ≤ 1 - α * (1 - a) := by
  have h1a : 0 ≤ 1 - a := by linarith
  have hαβ' : α * (1 - a) ≤ β * (1 - a) := mul_le_mul_of_nonneg_right hαβ h1a
  linarith

/-- **长程记忆下界抬升**：把均匀谱夹层上界 1-γ 抬到慢方向 1-γ_slow（γ_slow < γ）后，
    慢方向单位向量的记忆保持率从 (1-γ)^k 抬升到 (1-γ_slow)^k（严格大于）。
    即「保护慢特征」把长程记忆下界从均匀上界抬升——这是融合方案补上 §3.12 缺点的核心。 -/
theorem memory_lower_bound_lifted (γ_slow γ : ℝ)
    (hγs : 0 ≤ γ_slow) (hγslt : γ_slow < γ) (hγ1 : γ < 1) (k : ℕ) (hk : k ≠ 0) :
    (1 - γ) ^ k < (1 - γ_slow) ^ k := by
  have h_base_lt : 1 - γ < 1 - γ_slow := by linarith
  have h_base_nonneg : 0 ≤ 1 - γ := by linarith
  exact pow_lt_pow_left₀ h_base_lt h_base_nonneg hk

/-- **慢/快记忆比 → ∞**：慢方向特征值 a_slow > 快方向 a_fast ⟹ (a_slow/a_fast)^k → atTop。
    这是 `fast_slow_ratio`（快/慢比 → 0）的对偶：保护慢特征后，慢方向相对快方向的
    长程记忆**无限主导**（比值趋于无穷），故「留慢丢快」在长时域下几乎无损。 -/
theorem slow_fast_memory_ratio_tendsto_atTop (a_slow a_fast : ℝ)
    (h0 : 0 ≤ a_fast) (hlt : a_fast < a_slow) :
    Filter.Tendsto (fun k : ℕ => (a_slow / a_fast) ^ k) Filter.atTop Filter.atTop := by
  have hpos : 0 < a_fast := lt_of_le_of_lt h0 hlt
  have hgt1 : 1 < a_slow / a_fast := (one_lt_div hpos).mpr hlt
  exact tendsto_pow_atTop_atTop_of_one_lt hgt1

end TemporalHierarchy
