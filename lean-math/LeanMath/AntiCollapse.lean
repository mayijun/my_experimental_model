/-
反塌缩正则的数学验证（math.md §2.2.4(b) 式 36 的 VICReg 方差项）。

核心命题（对应本次诊断的塌缩 bug）：BYOL 余弦对齐损失存在平凡退化解——
所有表征坍缩为同一常数（`cos(a,a)=1`，损失为 0）。math.md §2.2.4 的 ⚠️ 说明
已指出「余弦 + EMA 不能排除塌缩」，因此需要 VICReg 方差正则补足缺口。

本文件形式化验证：方差正则 relu(1-σ_j) 在常数塌缩解（σ_j=0）处严格为正（=1），
且其零损失集恰为「σ_j ≥ 1」。故在 λ_var > 0 时，常数塌缩解不可能是零损失极小点。
-/

import Mathlib

namespace AntiCollapse

/-- ReLU：max(x, 0)。 -/
def relu (x : ℝ) : ℝ := max x 0

/-- 单个潜维度的方差惩罚：relu(1 - σ_j)，σ_j 为该维度在 batch 上的样本标准差
    （math.md §2.2.4(b) 式 (36)）。 -/
def varPenalty (σ : ℝ) : ℝ := relu (1 - σ)

/-- relu 非负。 -/
theorem relu_nonneg (x : ℝ) : 0 ≤ relu x := by
  unfold relu
  exact le_max_right x 0

/-- 方差惩罚非负：方差正则永远不会是负损失。 -/
theorem varPenalty_nonneg (σ : ℝ) : 0 ≤ varPenalty σ := relu_nonneg (1 - σ)

/-- 方差惩罚为零 ⟺ 该维标准差 ≥ 1（即「σ_j ≥ 1 即达标」的目标契约）。 -/
theorem varPenalty_eq_zero_iff (σ : ℝ) : varPenalty σ = 0 ↔ 1 ≤ σ := by
  unfold varPenalty relu
  constructor
  · intro h
    by_contra hlt
    have hpos : 0 < 1 - σ := sub_pos.mpr (not_le.mp hlt)
    have hmax : max (1 - σ) 0 = 1 - σ := max_eq_left (le_of_lt hpos)
    rw [hmax] at h
    linarith
  · intro h
    have hle : 1 - σ ≤ 0 := by linarith
    exact max_eq_right hle

/-- 塌缩（σ_j = 0，该维所有样本相同）时方差惩罚 = 1 > 0：
    常数退化解被严格惩罚，无法成为零损失解。 -/
theorem varPenalty_collapse : varPenalty 0 = 1 := by
  unfold varPenalty relu
  norm_num

/-- 有限样本序列（Fin n → ℝ）的均值。 -/
noncomputable def mean {n : ℕ} (x : Fin n → ℝ) : ℝ := (∑ i : Fin n, x i) / (n : ℝ)

/-- 总体方差：Σ(x - μ)² / n。 -/
noncomputable def variance {n : ℕ} (x : Fin n → ℝ) : ℝ :=
  (∑ i : Fin n, (x i - mean x) ^ 2) / (n : ℝ)

/-- 总体标准差。 -/
noncomputable def std {n : ℕ} (x : Fin n → ℝ) : ℝ := Real.sqrt (variance x)

/-- 常数序列的标准差为 0（n>0）。 -/
theorem constant_std_zero {n : ℕ} (hn : n ≠ 0) (a : ℝ) :
    std (fun _ : Fin n => a) = 0 := by
  unfold std variance mean
  have hmean : (∑ i : Fin n, a) / (n : ℝ) = a := by
    have hsum : (∑ i : Fin n, a) = (n : ℝ) * a := by
      rw [Finset.sum_const]
      simp
    rw [hsum]
    field_simp [show (n : ℝ) ≠ 0 by exact_mod_cast hn]
  rw [hmean]
  have hsq : (∑ i : Fin n, (a - a) ^ 2) = 0 := by simp
  rw [hsq]
  simp

/-- 塌缩解（常数表征）的方差惩罚严格为正：把常数解从零损失点中排除。 -/
theorem collapse_not_zero_loss {n : ℕ} (hn : n ≠ 0) (a : ℝ) :
    0 < varPenalty (std (fun _ : Fin n => a)) := by
  rw [constant_std_zero hn a]
  rw [varPenalty_collapse]
  norm_num

end AntiCollapse
