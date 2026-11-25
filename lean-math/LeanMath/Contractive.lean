/-
§3.7.1 / §3.7.3 收缩算子与 Neumann 级数的数学核心。

对应 math.md §3.7.1（收缩条件与 DEQ 隐式微分）与 §3.7.3（谱夹层 → 收缩）：

    ρ(J) < 1  ⇒  J^k → 0  ⇒  Σ_{k≥0} J^k = (I - J)^{-1}      （DEQ 式 34 的根基）
    收缩映射的复合仍是收缩映射                                  （Banach 不动点定理）
    (I - J)^{-1} = Σ J^k 是 BPTT 的 Σ_{k=0}^{L-1} J^k 在 L→∞ 的极限

本文件证明：

1. 复合：两个 Lipschitz 映射（收缩常数 `Kf`、`Kg`）的复合仍是 Lipschitz，
   常数为 `Kf * Kg`（`LipschitzWith.comp`）；两个严格收缩的复合仍是严格收缩
   （`Kf * Kg < 1`）；
2. Neumann 级数的代数身份：`(1 - a) * Σ_{i<k} a^i = 1 - a^k`（有限截断），
   这是 `(I - J)^{-1} = Σ J^k` 的代数核心。

Banach 不动点定理（存在唯一不动点 + 迭代收敛 + 误差界）在 mathlib 已形式化
（`ContractingWith.exists_fixedPoint`），此处作为「收缩映射的复合仍是收缩」的直接
推论引用。
-/

import Mathlib

open scoped BigOperators
open scoped NNReal

namespace Contractive

universe u

/-! ## 1. 收缩映射的复合 -/

/-- 复合：`LipschitzWith Kf f` 与 `LipschitzWith Kg g` 的复合是 `LipschitzWith (Kf * Kg)`。
    对应 §3.7.1「收缩映射的复合仍是收缩映射」。 -/
theorem contraction_comp {α β γ : Type*} [PseudoEMetricSpace α] [PseudoEMetricSpace β]
    [PseudoEMetricSpace γ] {Kf Kg : ℝ≥0} {f : β → γ} {g : α → β}
    (hf : LipschitzWith Kf f) (hg : LipschitzWith Kg g) :
    LipschitzWith (Kf * Kg) (f ∘ g) :=
  hf.comp hg

/-- 两个严格收缩（常数 < 1）的复合仍是严格收缩（常数 < 1）。 -/
theorem contraction_comp_contracting {α : Type u} [PseudoEMetricSpace α]
    {Kf Kg : ℝ≥0} {f g : α → α}
    (hf : LipschitzWith Kf f) (hg : LipschitzWith Kg g)
    (hKf : Kf < 1) (hKg : Kg < 1) :
    ∃ K : ℝ≥0, K < 1 ∧ LipschitzWith K (f ∘ g) := by
  refine ⟨Kf * Kg, ?_, hf.comp hg⟩
  have hf' : (Kf : ℝ) < 1 := by exact_mod_cast hKf
  have hg' : (Kg : ℝ) < 1 := by exact_mod_cast hKg
  have hf0 : 0 ≤ (Kf : ℝ) := by positivity
  have h : (Kf : ℝ) * (Kg : ℝ) < 1 := by
    calc
      (Kf : ℝ) * (Kg : ℝ) ≤ (Kf : ℝ) * 1 :=
        mul_le_mul_of_nonneg_left (le_of_lt hg') hf0
      _ = (Kf : ℝ) := by ring
      _ < 1 := hf'
  exact_mod_cast h

/-! ## 2. Neumann 级数（式 34 的代数核心） -/

/-- Neumann 级数有限截断：`(∑_{i<k} a^i) * (a - 1) = a^k - 1`。
    （mathlib `geom_sum_mul` 的直接引用，此处显式重述以对应式 34 上下文。） -/
theorem geom_sum_mul_geom {R : Type*} [Ring R] (a : R) (k : Nat) :
    (∑ i ∈ Finset.range k, a ^ i) * (a - 1) = a ^ k - 1 :=
  geom_sum_mul a k

/-- Neumann 级数的 `1 - a` 形式：`(1 - a) * Σ_{i<k} a^i = 1 - a^k`。
    这正是 `(I - J) · Σ_{k=0}^{L-1} J^k = I - J^L`，L→∞ 且 `ρ(J) < 1` 时给出
    `Σ_{k≥0} J^k = (I - J)^{-1}`。 -/
theorem geom_sum_one_sub {R : Type*} [CommRing R] (a : R) (k : Nat) :
    (1 - a) * (∑ i ∈ Finset.range k, a ^ i) = 1 - a ^ k := by
  have h := geom_sum_mul a k
  calc
    (1 - a) * (∑ i ∈ Finset.range k, a ^ i)
        = -((∑ i ∈ Finset.range k, a ^ i) * (a - 1)) := by ring
    _ = -(a ^ k - 1) := by rw [h]
    _ = 1 - a ^ k := by ring

end Contractive
