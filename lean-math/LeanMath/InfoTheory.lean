/-
互信息与条件熵的有限字母表形式化，以及确定性信道的核心命题。

对应 math.md 的式 (3) 与式 (15)：
  · I(Z;T) = H(T)    （当 T = g(Z) 为确定性映射时）
  · H(T) = I(T;Y) + H(T|Y)   （语义/措辞分解）

这里从联合分布出发，用链式法则重排定义条件熵与互信息，
并证明确定性标记不增加熵、从而 I(Z;T) = H(T)。
-/

import LeanMath.Entropy

open scoped BigOperators

namespace Shannon

section InfoTheory

variable {α β : Type*} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]

/-- 由联合分布 `p : α × β → ℝ` 得到的 X 边缘分布。 -/
noncomputable def marginalX (p : α × β → ℝ) : α → ℝ := fun a => ∑ b, p (a, b)

/-- 由联合分布 `p : α × β → ℝ` 得到的 Y 边缘分布。 -/
noncomputable def marginalY (p : α × β → ℝ) : β → ℝ := fun b => ∑ a, p (a, b)

/-- 条件熵 H(Y | X) := H(X,Y) − H(X)。 -/
noncomputable def condEntropy (p : α × β → ℝ) : ℝ := entropy p - entropy (marginalX p)

/-- 条件熵 H(X | Y) := H(X,Y) − H(Y)。 -/
noncomputable def condEntropyYX (p : α × β → ℝ) : ℝ := entropy p - entropy (marginalY p)

/-- 互信息 I(X;Y) := H(X) + H(Y) − H(X,Y)。 -/
noncomputable def mutualInformation (p : α × β → ℝ) : ℝ :=
  entropy (marginalX p) + entropy (marginalY p) - entropy p

/-- 确定性映射 `g` 的推前（pushforward）分布 `g_* p`，即文本信道 T 的分布。 -/
noncomputable def map (g : α → β) (p : α → ℝ) : β → ℝ :=
  fun b => ∑ a, if g a = b then p a else 0

/-- 联合分布 (Z, g Z)：`diag g p (a, b) = p a` 当 `b = g a`，否则为 0。 -/
noncomputable def diag (g : α → β) (p : α → ℝ) : α × β → ℝ :=
  fun x => if g x.1 = x.2 then p x.1 else 0

/-! ## 代数恒等式（纯定义推演，对应 math.md 式 (15)） -/

/-- 链式法则重排：H(X) = I(X;Y) + H(X|Y)。 -/
lemma entropy_marginalX_eq_mutualInformation_add_condEntropyYX (p : α × β → ℝ) :
    entropy (marginalX p) = mutualInformation p + condEntropyYX p := by
  unfold mutualInformation condEntropyYX
  ring

/-- I(X;Y) = H(Y) − H(Y|X)。 -/
lemma mutualInformation_eq_entropy_marginalY_sub_condEntropy (p : α × β → ℝ) :
    mutualInformation p = entropy (marginalY p) - condEntropy p := by
  unfold mutualInformation condEntropy
  ring

/-! ## 确定性信道：H(T|Z) = 0 且 I(Z;T) = H(T)（math.md 式 (3)） -/

/-- `diag` 的 X 边缘就是原来的 `p`。 -/
lemma marginalX_diag (g : α → β) (p : α → ℝ) : marginalX (diag g p) = p := by
  funext a
  simp only [marginalX, diag]
  rw [Finset.sum_eq_single (g a)]
  · simp
  · intro b _ hb
    simp [hb.symm]
  · intro h
    simp at h

/-- `diag` 的 Y 边缘就是推前 `map g p`（即文本分布）。 -/
lemma marginalY_diag (g : α → β) (p : α → ℝ) : marginalY (diag g p) = map g p := rfl

/-- 联合熵 H(Z, g Z) = H(Z)：确定性标记不增加熵。 -/
lemma entropy_diag (g : α → β) (p : α → ℝ) : entropy (diag g p) = entropy p := by
  unfold entropy
  rw [← Finset.univ_product_univ, Finset.sum_product]
  apply Finset.sum_congr rfl
  intro a _
  rw [Finset.sum_eq_single (g a)]
  · simp [diag]
  · intro b _ hb
    simp [diag, hb.symm]
  · intro h
    simp at h

/-- 确定性信道下 H(T | Z) = 0。 -/
lemma condEntropy_zero_of_deterministic (g : α → β) (p : α → ℝ) :
    condEntropy (diag g p) = 0 := by
  unfold condEntropy
  rw [entropy_diag, marginalX_diag]
  ring

/-- 确定性信道下 I(Z;T) = H(T)（math.md 式 (3)）。 -/
lemma mutualInformation_eq_entropy_of_deterministic (g : α → β) (p : α → ℝ) :
    mutualInformation (diag g p) = entropy (map g p) := by
  rw [mutualInformation_eq_entropy_marginalY_sub_condEntropy]
  rw [marginalY_diag, condEntropy_zero_of_deterministic]
  ring

end InfoTheory

end Shannon
