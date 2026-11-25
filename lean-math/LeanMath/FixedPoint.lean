/-!
# 「结论状态是变换链的不动点」的 Lean 4 形式化

对应 math.md §3.7：CoT 是状态依赖变换的轨迹

    z_{i+1} = Ψ_{h_i}(z_i)，

结论状态是这条链的收敛点（不动点）。本文件证明其核心断言：

1. 轨迹在某一步稳定（z_{N+1} = z_N）⇒ z_N 是 Ψ_{h_N} 的不动点；
2. 改分量为零（读模式）⇒ 算子退化为恒等 ⇒ 一切状态都是不动点（对齐 = 恒等）；
3. 具体实例：前驱变换 pred 的不动点是 0，轨迹从任意 n 收敛到 0 且到达后不再变化。
-/

namespace FixedPoint

universe u v

variable {Z : Type u} {H : Type v}

/-- 不动点：Ψ h z = z，即变换作用于 z 后 z 不变。 -/
def IsFixedPoint (Ψ : H → Z → Z) (h : H) (z : Z) : Prop :=
  Ψ h z = z

/-- 定理 1（核心）：轨迹稳定 ⇒ 不动点。
    给定轨迹 z_{i+1} = Ψ_{h_i}(z_i)，若 z_{N+1} = z_N，则 z_N 是 Ψ_{h_N} 的不动点。 -/
theorem stabilized_is_fixed_point
    (Ψ : H → Z → Z) (z : Nat → Z) (h : Nat → H)
    (traj : ∀ i, z (i+1) = Ψ (h i) (z i))
    {N : Nat} (stab : z (N+1) = z N) :
    IsFixedPoint Ψ (h N) (z N) := by
  unfold IsFixedPoint
  rw [← traj N]
  exact stab

/-- 定理 2：改分量为零（读模式）⇒ 算子退化为恒等 ⇒ 一切状态都是不动点。
    这是「对齐 = 恒等变换」的形式化：读模式下的 Ψ 不再改变状态。 -/
theorem read_only_fixes_all {Hr Hc : Type} [Zero Hc]
    (Ψ : Hr × Hc → Z → Z)
    (h_id : ∀ hr z, Ψ (hr, 0) z = z) :
    ∀ hr z, IsFixedPoint Ψ (hr, 0) z := by
  intro hr z
  unfold IsFixedPoint
  exact h_id hr z

/-- 轨迹：算子复合的展开（中间变换过程）。 -/
def trajectory (Ψ : H → Z → Z) (z₀ : Z) (h : Nat → H) : Nat → Z
  | 0     => z₀
  | i + 1 => Ψ (h i) (trajectory Ψ z₀ h i)

/-- 定理 3：轨迹的单步就是一次状态依赖变换。 -/
theorem trajectory_step
    (Ψ : H → Z → Z) (z₀ : Z) (h : Nat → H) (i : Nat) :
    trajectory Ψ z₀ h (i+1) = Ψ (h i) (trajectory Ψ z₀ h i) := by
  rfl

end FixedPoint

-- ============================================================
-- 具体实例：变换 = 前驱 pred，结论状态（不动点）= 0
-- ============================================================
namespace PredExample

/-- 迭代算子（自建，避免依赖外部库）。 -/
def iter {α : Type} (f : α → α) : Nat → α → α
  | 0,   a => a
  | n+1, a => f (iter f n a)

/-- 0 是 pred 的不动点。 -/
theorem pred_zero : Nat.pred 0 = 0 := rfl

/-- 每前进一步，n+1 变 n（轨迹向 0 递减）。 -/
theorem pred_step (n : Nat) : Nat.pred (n+1) = n := rfl

/-- 到达不动点 0 后，轨迹恒为 0（不再变化）。 -/
theorem pred_stays_at_zero (i : Nat) : iter Nat.pred i 0 = 0 := by
  induction i with
  | zero => rfl
  | succ i ih =>
      change Nat.pred (iter Nat.pred i 0) = 0
      rw [ih]
      rfl

/-- 关键：迭代 pred，n + k 在 n 步后变为 k（前 n 步把 n 削到 0）。 -/
theorem iter_pred_add (n k : Nat) : iter Nat.pred n (n + k) = k := by
  induction n generalizing k with
  | zero =>
      simp [iter]
  | succ n ih =>
      change Nat.pred (iter Nat.pred n ((n+1) + k)) = k
      rw [Nat.succ_add_eq_add_succ]
      rw [ih (k+1)]
      rfl

/-- 推论：从 n 出发，迭代 n 次到达不动点 0。 -/
theorem pred_reaches_zero (n : Nat) : iter Nat.pred n n = 0 :=
  iter_pred_add n 0

end PredExample
