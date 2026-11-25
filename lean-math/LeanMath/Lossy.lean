/-
命题 1 / 命题 2 / 命题 10 的**离散核心**：鸽笼原理下的有损性。

对应 math.md §2.1 的命题 1（有损性）、§2.4 的命题 2（失真地板）、§5 的命题 10
（软反馈信号不竭性）。这三条命题的连续版本依赖微分熵与率失真测度论（mathlib 暂缺），
但其**核心结论**都是同一个离散事实：确定性映射 `g : Z → T` 若码本（值域）严格小于
源集，则必非单射——两个不同源点被压进同一个 token，接收方无法唯一还原源。

本文件把这个事实机械化，并据此指出文档措辞的一处可改进点（见 README「问题 5」）：

  文档命题 1 断言「对**任意**有限字母表的确定性 g，在**任意**精细分辨率 δ 下
  H(Z_δ|T) > 0」。这个断言过强：有损性是一个**计数条件**（码本不足以区分源集），
  而非「任意精细分辨率」下的无条件结论。当 δ 尚不精细（|supp Z_δ| ≤ |𝒱|^L）时
  条件熵可以为 0。正确的可判定版本是单向的鸽笼：`card β < card α ⟹ g 非单射`。
  反方向（非单射 ⟹ card β < card α）**不成立**（见下方反例）。
-/

import Mathlib

namespace Lossy

section Pigeonhole

variable {α β : Type*} [Fintype α] [Fintype β] [DecidableEq β]

/-- 命题 1 核心（鸽笼原理）：确定性映射到更小有限集必发生碰撞（非单射）。 -/
theorem exists_collision_of_card_lt (g : α → β) (h : Fintype.card β < Fintype.card α) :
    ∃ a₁ a₂ : α, a₁ ≠ a₂ ∧ g a₁ = g a₂ := by
  by_contra hno
  push_neg at hno                       -- hno : ∀ a₁ a₂, a₁ ≠ a₂ → g a₁ ≠ g a₂
  have hinj : Function.Injective g := by
    intro a₁ a₂ heq
    by_contra hne
    exact hno a₁ a₂ hne heq
  have hle : Fintype.card α ≤ Fintype.card β := Fintype.card_le_of_injective g hinj
  exact not_lt_of_ge hle h

/-- 碰撞 ⟹ 任何「解码器」`d : β → α` 都无法同时还原两个碰撞点（命题 2 失真地板的根）。 -/
theorem decoder_cannot_invert [DecidableEq α] (g : α → β) (d : β → α)
    (hcol : ∃ a₁ a₂ : α, a₁ ≠ a₂ ∧ g a₁ = g a₂) :
    ∃ a : α, d (g a) ≠ a := by
  rcases hcol with ⟨a₁, a₂, hne, hg⟩
  by_cases h₁ : d (g a₁) = a₁
  · refine ⟨a₂, ?_⟩
    have hd₂ : d (g a₂) = a₁ := by
      rw [hg] at h₁                      -- h₁ : d (g a₁) = a₁  →  d (g a₂) = a₁
      exact h₁
    intro h₂                            -- 假设 d (g a₂) = a₂
    rw [hd₂] at h₂                      -- h₂ : a₁ = a₂
    exact hne h₂
  · exact ⟨a₁, h₁⟩

/-- 命题 10 核心：`g` 非单射 ⟹ 存在 `z` 使 `E (g z) ≠ z`（软反馈创新恒非零）。
    这里 `E` 是任意「重编码」函数 `β → α`（自反馈里的 `E ∘ g`）。 -/
theorem non_injective_implies_feedback_nonzero [DecidableEq α] (g : α → β) (E : β → α)
    (hcol : ∃ a₁ a₂ : α, a₁ ≠ a₂ ∧ g a₁ = g a₂) :
    ∃ a : α, E (g a) ≠ a := decoder_cannot_invert g E hcol

/-- 单射与无碰撞等价（这是「非单射」的展开，作为有损性的判定句法糖）。 -/
theorem injective_iff_no_collision (g : α → β) :
    Function.Injective g ↔ ∀ a₁ a₂ : α, g a₁ = g a₂ → a₁ = a₂ := Iff.rfl

/-- 反例：非单射 ⟹ card β < card α 是**假**的。
    令 α = β = Bool（card 相等），`g = const false` 非单射但码本并不更小。
    这证明文档「有损性 ⟺ 码本更小」的双向措辞需要收紧为单向鸽笼。 -/
example : (∃ a₁ a₂ : Bool, a₁ ≠ a₂ ∧ (Function.const false) a₁ = (Function.const false) a₂)
    ∧ ¬ (Fintype.card Bool < Fintype.card Bool) := by
  constructor
  · exact ⟨true, false, by decide⟩
  · exact Nat.lt_irrefl (Fintype.card Bool)

/-- 正确方向（命题 1 的精确表述）：只要源集基数严格超过码本基数，
    任意确定性映射都必有碰撞——与映射具体形式无关。 -/
theorem any_map_lossy_when_source_larger (g : α → β)
    (h : Fintype.card β < Fintype.card α) :
    ∃ a₁ a₂ : α, a₁ ≠ a₂ ∧ g a₁ = g a₂ := exists_collision_of_card_lt g h

end Pigeonhole

end Lossy
