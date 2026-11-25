/-
Shannon 熵（有限字母表，单位 nat）与最大熵定理。

对应 math.md 的式 (1)–(2)：
  H(T) ≤ log |𝒱^L| = L log |𝒱|

本文件从零定义有限分布上的 Shannon 熵，并用 Jensen 不等式证明最大熵定理。
-/

import Mathlib

open scoped BigOperators

namespace Shannon

/-- 有限字母表 `α` 上分布 `p` 的 Shannon 熵（单位 nat，自然对数）。 -/
noncomputable def entropy {α : Type*} [Fintype α] (p : α → ℝ) : ℝ :=
  ∑ a, Real.negMulLog (p a)

section Basic

variable {α : Type*} [Fintype α]

/-- 熵非负：若 `p` 各分量落在 `[0, 1]`，则 `0 ≤ entropy p`。 -/
lemma entropy_nonneg (p : α → ℝ) (hp₀ : ∀ a, 0 ≤ p a) (hp₁ : ∀ a, p a ≤ 1) :
    0 ≤ entropy p := by
  unfold entropy
  exact Finset.sum_nonneg fun a _ => Real.negMulLog_nonneg (hp₀ a) (hp₁ a)

/-- `negMulLog (n⁻¹) = n⁻¹ · log n`（用于把最大熵的 Jensen 结果写成 `log n`）。 -/
lemma negMulLog_inv (n : ℝ) :
    Real.negMulLog n⁻¹ = n⁻¹ * Real.log n := by
  rw [Real.negMulLog, Real.log_inv n]
  ring

/-- 最大熵定理：`n = card α` 个原子的分布熵 ≤ `log n`，等号当且仅当均匀分布。 -/
lemma entropy_le_log_card [Nonempty α] (p : α → ℝ) (hp₀ : ∀ a, 0 ≤ p a)
    (hsum : ∑ a, p a = 1) :
    entropy p ≤ Real.log (Fintype.card α) := by
  classical
  let n : ℕ := Fintype.card α
  let w : α → ℝ := fun _ => (n : ℝ)⁻¹
  have hn_pos : 0 < (n : ℝ) := by
    exact_mod_cast (Fintype.card_pos : 0 < n)
  have hw_nonneg : ∀ a ∈ (Finset.univ : Finset α), 0 ≤ w a := by
    intro a _ha
    dsimp [w]
    positivity
  have hw_sum : (∑ a : α, w a) = 1 := by
    simp only [w]
    rw [Finset.sum_const, Finset.card_univ]
    simp [n, hn_pos.ne']
  have hmem : ∀ a ∈ (Finset.univ : Finset α), p a ∈ Set.Ici (0 : ℝ) := by
    intro a _ha
    exact hp₀ a
  have hj := Real.concaveOn_negMulLog.le_map_sum
    (t := Finset.univ) (w := w) (p := p) hw_nonneg hw_sum hmem
  -- 左端 = n⁻¹ · entropy p，右端 = n⁻¹ · log n
  have hleft : (∑ a : α, w a • Real.negMulLog (p a)) = (n : ℝ)⁻¹ * entropy p := by
    simp only [w, entropy, smul_eq_mul]
    rw [← Finset.mul_sum]
  have hright : Real.negMulLog (∑ a : α, w a • p a) = (n : ℝ)⁻¹ * Real.log (n : ℝ) := by
    calc
      Real.negMulLog (∑ a : α, w a • p a) = Real.negMulLog ((n : ℝ)⁻¹ * ∑ a : α, p a) := by
        congr 1
        simp only [w, smul_eq_mul]
        rw [← Finset.mul_sum]
      _ = Real.negMulLog ((n : ℝ)⁻¹) := by rw [hsum, mul_one]
      _ = (n : ℝ)⁻¹ * Real.log (n : ℝ) := negMulLog_inv (n : ℝ)
  rw [hleft, hright] at hj
  -- 乘回 `n` 得到结论
  have hmul : (n : ℝ) * ((n : ℝ)⁻¹ * entropy p) ≤ (n : ℝ) * ((n : ℝ)⁻¹ * Real.log (n : ℝ)) :=
    mul_le_mul_of_nonneg_left hj (le_of_lt hn_pos)
  simpa [n, mul_assoc, mul_inv_cancel₀ hn_pos.ne'] using hmul

end Basic

end Shannon
