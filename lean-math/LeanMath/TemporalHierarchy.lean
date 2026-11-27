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

/-! ## 7. 融合方案：慢特征保护 + 快特征自反馈（§3.13）

以单 Z 隐式时间（§3.12 / 本文件 §6）为骨架，把多时间尺度（§3.11）的谱分离降级为
「谱正则（慢特征保护）+ 反馈门控（快特征自反馈）」两个轻量机制，融合二者优缺：

  - 慢特征（时间无关 / 弱相关，特征值 a_s 接近 1）：**保护** = 把特征值抬升到
    a_s' ∈ [a_s, 1)，仍收缩（protected_slow_still_contracting），但留存被抬升
    （slow_retention_lifted）、慢快留存差距被拉大（slow_fast_gap_widened）——
    即「长程记忆下界抬升」。
  - 快特征（时间相关 / 短期状态，特征值 a_f 接近 0）：**自反馈** = 门控在快子空间的修正
    F = C ∘ Π_f；其修正随 horizon 以 a_f^k → 0 衰减（feedback_correction_decays，短期局部化），
    且对慢子空间零扰动（feedback_spares_slow）——即「对长程无关特征影响变小（此处为 0）」。
-/

/-- 慢特征保护抬升留存（长程记忆下界逐点抬升）：把慢特征值从 a_s 抬升到 a_s'
    （a_s ≤ a_s'）后，任意 horizon k 的留存 a_s'^k ≥ a_s^k。
    保护 = 减缓慢维收缩、留存更多历史（对比 §3.11 的 `persistence_mono` 快/慢单调性）。 -/
theorem slow_retention_lifted (a_s a_s' : ℝ) (has0 : 0 ≤ a_s) (hss' : a_s ≤ a_s') (k : ℕ) :
    a_s ^ k ≤ a_s' ^ k := by
  exact pow_le_pow_left₀ has0 hss' k

/-- 保护后仍收缩：受保护慢特征值 a_s' ∈ [0,1) 时其 k 次幂（k>0）仍 < 1，
    即保护（抬升慢维特征值）不破坏框架的收缩契约（谱夹层 δ < ã_i < 1-γ，保护后仍 ρ<1）。 -/
theorem protected_slow_still_contracting (a_s' : ℝ) (ha0 : 0 ≤ a_s') (ha1 : a_s' < 1)
    (k : ℕ) (hk : k ≠ 0) :
    a_s' ^ k < 1 := by
  exact pow_lt_one₀ ha0 ha1 hk

/-- 保护拉大慢快留存差距：保护（a_s' ≥ a_s）后，慢留存与快留存之差被放大，
    即「慢（保留）vs 快（衰减）」的分离被强化——快相对慢更快可忽略
    （这是 §3.11 `fast_slow_ratio` 的「保护强化」版本，用差值而非比值，避开除法）。 -/
theorem slow_fast_gap_widened (a_f a_s a_s' : ℝ)
    (haf0 : 0 ≤ a_f) (hfs : a_f ≤ a_s) (hss' : a_s ≤ a_s') (k : ℕ) :
    a_s ^ k - a_f ^ k ≤ a_s' ^ k - a_f ^ k := by
  have hret : a_s ^ k ≤ a_s' ^ k := slow_retention_lifted a_s a_s' (le_trans haf0 hfs) hss' k
  linarith

/-- 自反馈修正随 horizon 衰减（短期局部化）：快特征值 a_f ∈ [0,1) 的反馈修正
    a_f^k → 0，即自反馈只修正近期（短期）误差，对远期影响趋于 0
    （与 §3.11 `fast_decay` 同构，语义上归属「快特征自反馈」）。 -/
theorem feedback_correction_decays (a_f : ℝ) (haf0 : 0 ≤ a_f) (haf1 : a_f < 1) :
    Filter.Tendsto (fun k : ℕ => a_f ^ k) Filter.atTop (nhds (0 : ℝ)) := by
  exact tendsto_pow_atTop_nhds_zero_of_lt_one haf0 haf1

/-- 门控自反馈（F = C ∘ Π_f，只在快子空间作用）：修正算子 C 先经快投影 Π_f 门控，
    故反馈只作用于快（时间相关）子空间，慢（时间无关）子空间不受影响。 -/
def gatedFeedback (C Pf : V →ₗ[K] V) : V →ₗ[K] V :=
  C.comp Pf

/-- 门控自反馈不扰动慢特征：若 Π_f 杀死慢子空间（∀z, Pf (Ps z) = 0），
    则门控反馈 F = C ∘ Pf 在慢子空间上恒为 0——
    自反馈对「长程无关（时间不变）特征」零扰动（即「影响变小」，此处为 0）。
    这是「自反馈修正短期、放过长期」的代数保证：慢特征被保护，快特征被修正。 -/
theorem feedback_spares_slow (Ps Pf C : V →ₗ[K] V)
    (h_kill : ∀ z : V, Pf (Ps z) = 0) (z : V) :
    gatedFeedback C Pf (Ps z) = 0 := by
  rw [gatedFeedback, LinearMap.comp_apply, h_kill, map_zero]

end TemporalHierarchy

/-! # ZHierarchy：全 Z 逐维软门控（机制二的推广，§3.14）

把 §3.13 融合方案「机制二」的二值门控（Π_f 快投影，权重 0/1）推广为**逐维软门控**
G = diag(p)，p_i ∈ [0,1]：每个维度 i 独立拥有自己的门控权重 p_i（= 0 放过 = 慢特征保护；
= 1 全修正 = 快特征；中间值连续调节）。即「所有 Z 维度各自分层」——机制二从「快/慢二分」
推广到「全谱逐维」。

GPU 友善与契约相容的数学根基（本节全部机械化）：
  - diagGate_pointwise：第 i 维输出只依赖第 i 维输入（逐点独立、embarrassingly parallel）。
  - diagGate_comp：门控复合 = 逐维权重相乘（可合并为单次逐元素缩放，并入 PSCO 扫描幺半群元）。
  - soft_gate_preserves_contraction：0≤a<1、0≤p≤1 ⟹ p·a∈[0,1)（全 Z 分层仍收缩 ρ<1）。
  - gate_preserves_upper_bound / gate_preserves_lower_bound：谱夹层相容性（见 §SpectralSandwich）。
  - diagGate_zero_dim / diagGate_mono：门控逐维放过 / 连续调节（推广 feedback_spares_slow）。

namespace ZHierarchy

section DiagonalGate

variable {K : Type*} [Field K]
variable {ι : Type*}

/-- 对角门控：diag(p) 对向量 z 逐维缩放，(diag(p) z) i = p i * z i。
    这是「机制二推广到所有域」的数学形式：每个维度 i 独立拥有门控权重 p i。 -/
def diagGate (p : ι → K) : (ι → K) →ₗ[K] (ι → K) where
  toFun := fun z i => p i * z i
  map_add' := by
    intro x y
    funext i
    simp [mul_add]
  map_smul' := by
    intro a x
    funext i
    simp [mul_assoc, mul_comm, mul_left_comm]

@[simp] theorem diagGate_apply (p : ι → K) (z : ι → K) (i : ι) :
    diagGate p z i = p i * z i := rfl

/-- 对角门控逐点独立：第 i 维输出只依赖第 i 维输入（无跨维依赖）。
    这是 GPU 友善的数学根基：各维度缩放相互独立，可完全并行（embarrassingly parallel），
    无归约 / 跨元素数据依赖，逐元素 kernel 可被 torch.compile 融合。 -/
theorem diagGate_pointwise (p : ι → K) (z w : ι → K) (i : ι)
    (hz : z i = w i) :
    (diagGate p z) i = (diagGate p w) i := by
  simp [hz]

/-- 对角门控的复合 = 逐维权重相乘：diag(p) ∘ diag(q) = diag(λi, p i * q i)。
    多个门控可合并为单次逐元素缩放，故并入 §3.7.3 PSCO 关联扫描的幺半群元时不破坏
    结合律（comp_assoc）、不新增扫描趟次——门控在 GPU 上零额外开销。 -/
theorem diagGate_comp (p q : ι → K) :
    (diagGate p).comp (diagGate q) = diagGate (fun i => p i * q i) := by
  ext z
  ext i
  simp [LinearMap.comp_apply, mul_assoc]

/-- 门控逐维放过（推广 §7 feedback_spares_slow 到软门控）：
    p i = 0 ⟹ (diag(p) z) i = 0——维度 i 被完全保护、反馈零扰动。 -/
theorem diagGate_zero_dim (p : ι → K) (z : ι → K) {i : ι} (hp : p i = 0) :
    (diagGate p z) i = 0 := by
  simp [hp]

end DiagonalGate

/-! ## 8（续）. 有序域下的逐维结论（ℝ） -/

section DiagonalGateOrdered

variable {ι : Type*}

/-- 软门控的逐维单调性：门控权重越大，该维反馈影响越大（对非负分量）。
    p ≤ q（逐点）且 0 ≤ z i ⟹ (diag(p) z) i ≤ (diag(q) z) i。
    这给出「反馈影响随 p_i 连续调节」的定量形式（0=放过、1=全修正、中间连续）。 -/
theorem diagGate_mono (p q : ι → ℝ) (z : ι → ℝ) {i : ι}
    (hpq : p i ≤ q i) (hz0 : 0 ≤ z i) :
    (diagGate p z) i ≤ (diagGate q z) i := by
  simp
  exact mul_le_mul_of_nonneg_right hpq hz0

/-- 软门控不破坏收缩（逐维）：a ∈ [0,1) 且 p ∈ [0,1] ⟹ p·a ∈ [0,1)。
    门控（缩放 p ≤ 1）不放大特征值，故「对所有 Z 维度分层（每维软门控）」仍满足
    ρ < 1（谱夹层 δ < ã_i < 1-γ 保护后仍成立），不动点收敛与 PSCO 扫描全部继承。 -/
theorem soft_gate_preserves_contraction (a p : ℝ) (ha0 : 0 ≤ a) (ha1 : a < 1)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ p * a ∧ p * a < 1 := by
  constructor
  · exact mul_nonneg hp0 ha0
  · have hle : p * a ≤ a := by
      calc
        p * a ≤ 1 * a := mul_le_mul_of_nonneg_right hp1 ha0
        _ = a := by simp
    exact lt_of_le_of_lt hle ha1

end DiagonalGateOrdered

/-! ## 谱夹层相容性（ZHierarchy 与 §3.7.3 谱夹层 δ < ã_i < 1-γ 的冲突检查） -/

section SpectralSandwich

/-- 门控保持谱夹层**上界**：0≤a<u、p≤1 ⟹ p·a < u。
    若门控作用于转移 M（把特征值 a 缩为 p·a），上界 1-γ 仍被保持（p·a ≤ a < u）。
    即 ZHierarchy 不破坏 §3.7.3 的收缩上界，不动点收敛仍成立。 -/
theorem gate_preserves_upper_bound (a p u : ℝ) (ha0 : 0 ≤ a) (hau : a < u) (hp1 : p ≤ 1) :
    p * a < u := by
  have hle : p * a ≤ a := by
    calc
      p * a ≤ 1 * a := mul_le_mul_of_nonneg_right hp1 ha0
      _ = a := by simp
  exact lt_of_le_of_lt hle hau

/-- 门控保持谱夹层**下界**的充要条件：0<δ≤a 时 δ < p·a ⟷ δ/a < p。
    这是 ZHierarchy 与谱夹层下界 δ 的冲突点：若门控作用于转移 M（把特征值 a 缩为 p·a），
    要保持下界 δ < p·a，必须 p > δ/a（因 a≥δ 故 δ/a ≤ 1，是真实约束）。
    **结论**：ZHierarchy 的门控应只作用于**反馈**（不缩放 M 的特征值），从而不触碰下界；
    若把门控也作用到 M 上，则每个受门控维度须满足 p_i > δ/a_i。 -/
theorem gate_preserves_lower_bound (δ a p : ℝ) (hδ0 : 0 < δ) (hδa : δ ≤ a) :
    δ < p * a ↔ δ / a < p := by
  have ha0 : 0 < a := lt_of_lt_of_le hδ0 hδa
  exact (div_lt_iff₀ ha0).symm

end SpectralSandwich

end ZHierarchy
