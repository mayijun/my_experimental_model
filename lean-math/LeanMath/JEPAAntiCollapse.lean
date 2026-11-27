/-
# JEPA 论文防塌缩策略与 math.md 架构的兼容性验证

本文件形式化验证四条来自 JEPA 论文的防塌缩策略与本架构（math.md）的兼容性，
每条给出一条核心数学命题（对应 math.md 的相关契约）：

1. 冻结初始化锚点（Human-JEPA / SALT）：EMA 目标会「追随」在线编码器塌缩
   （唯一不动点 = 目标值 c）；冻结（μ=1）目标不漂移。⟹ 锚点改冻结切断了
   「在线塌缩 → 目标跟塌缩 → 锚点失守」的连锁（math.md §3.4「冻结 Z = EMA 在 μ=1 的硬极限」）。
2. 分布正则（LeWorldModel / LeJEPA / UniJEPA）：VICReg 方差 hinge `relu(1-σ)`
   在 σ≥1 处饱和为 0（无法区分 σ=1 与 σ=10）；高斯正则 `(σ²-1)²` 双端正则、
   零集（σ≥0）恰为 {σ=1}，严格强于 VICReg，且是 σ 的连续函数（不引入量化，相容 §2.6 Z 连续契约）。
3. 块掩码（I-JEPA / V-JEPA）：孤立掩码产生「被掩 patch 贴可见 patch」的抄捷径；
   连续块掩码的块内部两侧皆掩、无抄捷径，强迫全局理解（math.md §2.2.1）。
4. DINO centering：居中 z ↦ z - mean(z) 是平移，精确把塌缩方向（常数表征）映到 0，
   保留所有相对差异（平移等变），故抑制教师输出塌缩（math.md §2.2.8 改进 #3）。
-/

import Mathlib
import LeanMath.AntiCollapse

/-! ## 1. 冻结初始化锚点（Human-JEPA / SALT） -/

namespace Anchor

/-- EMA 一步更新：ξ' = μ·ξ + (1-μ)·θ（math.md §2.2.3 式 31）。 -/
def emaStep (μ ξ θ : ℝ) : ℝ := μ * ξ + (1 - μ) * θ

/-- EMA 迭代 t 步（目标恒为 θ 的简化场景）。 -/
def emaIterate (μ θ ξ0 : ℝ) : ℕ → ℝ
  | 0 => ξ0
  | (t + 1) => emaStep μ (emaIterate μ θ ξ0 t) θ

/-- EMA 的唯一不动点就是目标值：若目标塌缩为常数 c，EMA 教师唯一不动点是 c。
    μ ≠ 1 时 `μξ + (1-μ)c = ξ ⟺ ξ = c` —— 「目标塌缩，教师必跟到塌缩」。 -/
theorem ema_fixedPoint_unique (μ θ ξ : ℝ) (hμ : μ ≠ 1) :
    emaStep μ ξ θ = ξ ↔ ξ = θ := by
  unfold emaStep
  constructor
  · intro h
    have hlin : (1 - μ) * ξ - (1 - μ) * θ = 0 := by linarith
    have hfac : (1 - μ) * (ξ - θ) = 0 := by
      rw [mul_sub]
      exact hlin
    have h1m : (1 - μ) ≠ 0 := by
      intro hz
      have hμ1 : μ = 1 := by linarith
      exact hμ hμ1
    exact sub_eq_zero.mp ((mul_eq_zero.mp hfac).resolve_left h1m)
  · intro hξθ
    rw [hξθ]
    ring

/-- 冻结锚点（μ=1）的 EMA 更新是恒等：不追随目标。 -/
theorem frozen_step_is_identity (ξ θ : ℝ) : emaStep 1 ξ θ = ξ := by
  unfold emaStep
  ring

/-- 冻结锚点序列静止：μ=1 ⟹ ∀t, ξ_t = ξ_0（目标漂移不影响冻结教师）。 -/
theorem frozen_anchor_stationary (θ ξ0 : ℝ) (t : ℕ) :
    emaIterate 1 θ ξ0 t = ξ0 := by
  induction t with
  | zero => rfl
  | succ t ih => simp [emaIterate, emaStep, ih]

/-- EMA 迭代闭式：ξ_t = μ^t·ξ0 + (1-μ^t)·c。|μ|<1 时 μ^t→0，ξ_t→c（塌缩跟随）。
    这是「EMA 追随塌缩目标」的收敛性形式。 -/
theorem ema_iterate_closedForm (μ c ξ0 : ℝ) (t : ℕ) :
    emaIterate μ c ξ0 t = μ ^ t * ξ0 + (1 - μ ^ t) * c := by
  induction t with
  | zero => simp [emaIterate]
  | succ t ih =>
      simp [emaIterate, emaStep, ih]
      rw [pow_succ]
      ring

end Anchor

/-! ## 2. 分布正则（LeWorldModel / LeJEPA / UniJEPA） -/

namespace DistributionReg

open AntiCollapse

/-- 单维高斯正则（χ² 分布正则的 1D 投影）：`(σ² - 1)²`。
    UniJEPA 的 `R(θ)=E[χ²₁((uᵀz)²)]` 要求潜变量沿每个随机方向 u 的投影服从标准正态；
    其「方差=1」的标量投影即 σ=1。它双端正则：σ<1（塌缩）与 σ>1（发散）都被惩罚，
    零集（σ≥0）恰为 {σ=1}。 -/
def gaussPenalty (σ : ℝ) : ℝ := (σ ^ 2 - 1) ^ 2

/-- 高斯正则非负。 -/
theorem gaussPenalty_nonneg (σ : ℝ) : 0 ≤ gaussPenalty σ := by
  unfold gaussPenalty
  nlinarith [sq_nonneg (σ ^ 2 - 1)]

/-- 高斯正则零集：`gaussPenalty σ = 0 ⟺ σ² = 1`。 -/
theorem gaussPenalty_eq_zero_iff (σ : ℝ) : gaussPenalty σ = 0 ↔ σ ^ 2 = 1 := by
  unfold gaussPenalty
  constructor
  · intro h
    have h' : σ ^ 2 - 1 = 0 := sq_eq_zero_iff.mp h
    linarith
  · intro h
    rw [h]
    ring

/-- 塌缩（σ=0）时高斯正则 = 1 > 0：常数解被严格排除（与 VICReg 同向、且更强）。 -/
theorem gaussPenalty_collapse : gaussPenalty 0 = 1 := by
  unfold gaussPenalty
  norm_num

/-- 高斯正则零集（σ≥0）⊂ VICReg 零集：分布正则不弱于 VICReg，二者都排除塌缩。 -/
theorem gauss_zero_implies_vicreg_zero (σ : ℝ) (hσ : 0 ≤ σ) :
    gaussPenalty σ = 0 → varPenalty σ = 0 := by
  intro hg
  have hσ2 : σ ^ 2 = 1 := (gaussPenalty_eq_zero_iff σ).mp hg
  have hσeq : σ = 1 := by
    have hcases : σ = 1 ∨ σ = -1 := sq_eq_one_iff.mp hσ2
    rcases hcases with h1 | hneg
    · exact h1
    · linarith
  rw [hσeq]
  exact (varPenalty_eq_zero_iff 1).mpr le_rfl

/-- 严格更强：σ=2（健康但方差非 1）VICReg 惩罚为 0（已饱和），高斯正则仍 ≠ 0。 -/
theorem gauss_strictly_stronger :
    varPenalty 2 = 0 ∧ gaussPenalty 2 ≠ 0 := by
  constructor
  · exact (varPenalty_eq_zero_iff 2).mpr (by norm_num)
  · unfold gaussPenalty; norm_num

/-- VICReg 方差 hinge 在 σ≥1 处恒为 0（饱和、梯度消失）：无法区分 σ=1 与任意更大 σ。 -/
theorem varPenalty_saturated (σ : ℝ) (hσ : 1 ≤ σ) : varPenalty σ = 0 :=
  (varPenalty_eq_zero_iff σ).mpr hσ

/-- 高斯正则不在有限 σ 处饱和为常数：σ=1 与 σ=2 惩罚不同（双端、梯度不消失）。 -/
theorem gaussPenalty_not_flat : gaussPenalty 1 ≠ gaussPenalty 2 := by
  unfold gaussPenalty
  norm_num

/-- 高斯正则是 σ 的多项式，故连续：不引入量化/离散，相容 §2.6「Z 全程连续」契约。 -/
theorem gaussPenalty_continuous : Continuous gaussPenalty := by
  unfold gaussPenalty
  continuity

end DistributionReg

/-! ## 3. 块掩码（I-JEPA / V-JEPA） -/

namespace BlockMask

/-- 抄捷径位：位置 i 被掩，但至少一个相邻位（i-1 或 i+1）可见，可「抄」相邻内容。 -/
def CopyShortcut (m : ℤ → Prop) (i : ℤ) : Prop :=
  m i ∧ (¬ m (i - 1) ∨ ¬ m (i + 1))

/-- 块掩码 [a,b]：闭区间内全掩。 -/
def Block (a b : ℤ) (i : ℤ) : Prop := a ≤ i ∧ i ≤ b

/-- 块内部位置两侧皆掩：无抄捷径（强迫全局理解，而非抄相邻可见 patch）。 -/
theorem block_interior_no_shortcut (a b i : ℤ) (h : a < i ∧ i < b) :
    ¬ CopyShortcut (Block a b) i := by
  intro hs
  rcases hs with ⟨_hi, hnei⟩
  have hleft : Block a b (i - 1) := by
    constructor <;> omega
  have hright : Block a b (i + 1) := by
    constructor <;> omega
  rcases hnei with hnl | hnr
  · exact hnl hleft
  · exact hnr hright

/-- 对比：孤立掩码（只掩单点 0）产生抄捷径位——0 的两侧都可见、可直接抄。
    逐位置独立 Bernoulli 掩码以正概率产生这种孤立被掩 patch，故有抄捷径；块掩码无。 -/
theorem singleton_has_shortcut : CopyShortcut (fun j : ℤ => j = 0) 0 := by
  constructor
  · rfl
  · left
    intro h
    omega

end BlockMask

/-! ## 4. DINO centering -/

namespace Centering

open AntiCollapse

/-- 居中：z ↦ z - mean(z)（DINO centering 的核心，math.md §3.4 DINOHead）。 -/
noncomputable def centered {n : ℕ} (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  x i - mean x

/-- 居中平移等变：不改变表征间的相对差异，只移除全局均值方向。 -/
theorem centered_preserves_differences {n : ℕ} (x : Fin n → ℝ) (i j : Fin n) :
    centered x i - centered x j = x i - x j := by
  unfold centered
  ring

/-- 常数表征（塌缩）的居中输出恒为 0：centering 把塌缩方向精确映到 0。
    DINO 教师对塌缩的常数输出居中后为 0，softmax(0/τ) 均匀、无锐化目标，
    故「教师学生同时塌缩」的平凡解不再产生自洽的锐化目标。 -/
theorem centered_constant_is_zero {n : ℕ} (hn : n ≠ 0) (a : ℝ) :
    ∀ i : Fin n, centered (fun _ : Fin n => a) i = 0 := by
  intro i
  unfold centered mean
  have hmean : (∑ j : Fin n, a) / (n : ℝ) = a := by
    have hsum : (∑ j : Fin n, a) = (n : ℝ) * a := by
      rw [Finset.sum_const]
      simp
    rw [hsum]
    field_simp [show (n : ℝ) ≠ 0 by exact_mod_cast hn]
  rw [hmean]
  ring

end Centering

/-! ## 5. DINOv3 Gram-anchoring（§3.4） -/

namespace GramAnchor

/-- Gram 矩阵元素：dense 特征 X 的 pairwise 内积（L2 归一化后即余弦）。 -/
def gramEntry {n d : ℕ} (X : Fin n → Fin d → ℝ) (i j : Fin n) : ℝ :=
  ∑ k : Fin d, X i k * X j k

/-- Gram-anchoring 损失（逐对平方差，即 ‖G_S − G_T‖²_F = Σᵢⱼ (G_S[i,j] − G_T[i,j])²）。 -/
def gramAnchorLoss {n d : ℕ} (Xs Xt : Fin n → Fin d → ℝ) : ℝ :=
  ∑ i : Fin n, ∑ j : Fin n, (gramEntry Xs i j - gramEntry Xt i j) ^ 2

/-- 塌缩的 dense 特征（所有 token 相同且归一化 Σₖxₖ²=1）的 Gram 矩阵恒为 1（秩 1 退化）。 -/
theorem collapsed_gram_is_ones {n d : ℕ} (x : Fin d → ℝ)
    (hunit : (∑ k : Fin d, x k * x k) = 1) :
    ∀ i j : Fin n, gramEntry (fun _ : Fin n => x) i j = 1 := by
  intro i j
  simpa [gramEntry] using hunit

/-- Gram-anchoring 排除塌缩：若学生塌缩（全同归一化特征）而 teacher 非塌缩
    （存在一对 token 的 Gram ≠ 1），则 Gram-anchoring 损失 > 0，把学生从塌缩推开。 -/
theorem gram_anchor_excludes_collapse {n d : ℕ} (x : Fin d → ℝ)
    (Xt : Fin n → Fin d → ℝ) (hunit : (∑ k : Fin d, x k * x k) = 1)
    (hnoncol : ∃ i j : Fin n, gramEntry Xt i j ≠ 1) :
    0 < gramAnchorLoss (fun _ : Fin n => x) Xt := by
  unfold gramAnchorLoss
  rcases hnoncol with ⟨i, j, hij⟩
  have hpos_pair :
      0 < (gramEntry (fun _ : Fin n => x) i j - gramEntry Xt i j) ^ 2 := by
    have hg : gramEntry (fun _ : Fin n => x) i j = 1 := collapsed_gram_is_ones x hunit i j
    rw [hg]
    have hdiff : 1 - gramEntry Xt i j ≠ 0 := by
      intro hz
      have hge : gramEntry Xt i j = 1 := by linarith
      exact hij hge
    exact sq_pos_of_ne_zero hdiff
  have hinner_ge :
      (gramEntry (fun _ : Fin n => x) i j - gramEntry Xt i j) ^ 2 ≤
        ∑ j' : Fin n, (gramEntry (fun _ : Fin n => x) i j' - gramEntry Xt i j') ^ 2 := by
    exact Finset.single_le_sum
      (fun j' _ => sq_nonneg (gramEntry (fun _ : Fin n => x) i j' - gramEntry Xt i j'))
      (Finset.mem_univ j)
  have houter_ge :
      (∑ j' : Fin n, (gramEntry (fun _ : Fin n => x) i j' - gramEntry Xt i j') ^ 2) ≤
        ∑ i' : Fin n, ∑ j' : Fin n, (gramEntry (fun _ : Fin n => x) i' j' - gramEntry Xt i' j') ^ 2 := by
    exact Finset.single_le_sum
      (fun i' _ => Finset.sum_nonneg
        (fun j' _ => sq_nonneg (gramEntry (fun _ : Fin n => x) i' j' - gramEntry Xt i' j')))
      (Finset.mem_univ i)
  exact lt_of_lt_of_le hpos_pair (le_trans hinner_ge houter_ge)

end GramAnchor
