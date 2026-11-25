/-
§3.7.3 PSCO（并行扫描收缩算子）的数学核心：仿射幺半群与线性递推闭式。

对应 math.md §3.7.3 式 (30)–(31) 与 §3.9 式 (30)：

    Z^{(k+1)} = Ã_h Z^{(k)} + b_h                         （线性递推）
    (M₁,u₁) ∘ (M₂,u₂) = (M₁∘M₂,  M₁u₂ + u₁)              （仿射复合，式 31）
    Z^{(k)} = Ã_h^k Z^{(0)} + Σ_{j<k} Ã_h^{k-1-j} b_h      （闭式，式 1129）

本文件证明：

1. 仿射对 `(M, u)` 的复合律 `(M₁,u₁)∘(M₂,u₂)=(M₁∘M₂, M₁u₂+u₁)` 确实对应
   **函数复合**（`comp_apply`，表示是同态）；
2. 该复合满足**结合律**并含单位元 `(id, 0)`（`Ω` 是幺半群，Blelloch 扫描正确的根基）；
3. 线性递推的**闭式**（式 1129 的等价升幂形式）。

这正是「算子可以横向量化并行扫描」的数学保证：只要复合律结合、表示同态，
Blelloch 上下两趟就能以任意括号顺序算出同一前缀。
-/

import Mathlib

open scoped BigOperators

namespace PSCO

universe u

/-- 仿射变换 `x ↦ M x + u`，用对 `(M, u)` 表示。 -/
structure Aff (K : Type*) [Field K] (V : Type u) [AddCommGroup V] [Module K V] where
  lin : V →ₗ[K] V
  trans : V

namespace Aff

variable {K : Type*} [Field K]
variable {V : Type u} [AddCommGroup V] [Module K V]

/-- 外延：两个仿射变换相等当且仅当线部与平移部分别相等。 -/
@[ext] theorem ext' {a b : Aff K V} (hlin : a.lin = b.lin) (htrans : a.trans = b.trans) :
    a = b := by
  cases a
  cases b
  simp_all

/-- 仿射变换作为函数。 -/
def toFun (a : Aff K V) : V → V := fun x => a.lin x + a.trans

instance : CoeFun (Aff K V) (fun _ => V → V) := ⟨toFun⟩

/-- 复合：`(M₁,u₁) ∘ (M₂,u₂) = (M₁∘M₂, M₁u₂ + u₁)`，对应式 (31)。 -/
def comp (a b : Aff K V) : Aff K V where
  lin := a.lin.comp b.lin
  trans := a.lin b.trans + a.trans

/-- 单位元：`(id, 0)`。 -/
def id : Aff K V where
  lin := LinearMap.id
  trans := 0

/-! ## 1. 表示同态：仿射对的复合 = 函数的复合 -/

theorem comp_apply (a b : Aff K V) (x : V) :
    (a.comp b) x = a (b x) := by
  simp [comp, toFun]
  abel

/-! ## 2. 幺半群三公理（结合律 + 单位元） -/

theorem comp_assoc (a b c : Aff K V) :
    a.comp (b.comp c) = (a.comp b).comp c := by
  apply ext'
  · rfl
  · simp [comp]; abel

theorem comp_id (a : Aff K V) : a.comp id = a := by
  apply ext'
  · simp [comp, id]
  · simp [comp, id]

theorem id_comp (a : Aff K V) : id.comp a = a := by
  apply ext'
  · simp [comp, id]
  · simp [comp, id]

end Aff

/-! ## 3. 线性递推的闭式（式 1129 的等价升幂形式） -/

variable {K : Type*} [Field K]
variable {V : Type u} [AddCommGroup V] [Module K V]

/-- 线性递推 `Z ↦ M Z + b`（式 30 的 `Ψ_h` 线性骨架）。 -/
def iterate (M : V →ₗ[K] V) (b : V) : Nat → V → V
  | 0,     z => z
  | k + 1, z => M (iterate M b k z) + b

/-- 左复合幂：`powL M k = M ∘ M ∘ ⋯ ∘ M`（k 次），满足 `powL M (k+1) z = M (powL M k z)`。 -/
def powL (M : V →ₗ[K] V) : Nat → V →ₗ[K] V
  | 0     => 1
  | k + 1 => M.comp (powL M k)

/-- 幂的单步展开：`powL M (k+1) z = M (powL M k z)`。 -/
lemma powL_succ_apply (M : V →ₗ[K] V) (k : Nat) (z : V) :
    (powL M (k + 1)) z = M (powL M k z) := rfl

/-- 闭式的升幂形式：`M^k z + Σ_{i<k} M^i b`。
    与文档式 (1129) 的降幂形式 `Σ_{j<k} M^{k-1-j} b` 相等（因 `V` 的加法可交换）。 -/
def closedForm (M : V →ₗ[K] V) (b : V) (k : Nat) (z : V) : V :=
  (powL M k) z + ∑ i ∈ Finset.range k, ((powL M i) b)

/-- 求和移动引理：`Σ_{i<k} M^{i+1} b + b = Σ_{i<k+1} M^i b`。 -/
lemma sum_shift (M : V →ₗ[K] V) (b : V) (k : Nat) :
    (∑ i ∈ Finset.range k, ((powL M (i + 1)) b)) + b
      = ∑ i ∈ Finset.range (k + 1), ((powL M i) b) := by
  rw [Finset.sum_range_succ']
  simp [powL]

/-- 核心定理：线性递推的闭式。
    `Z^{(k)} = M^k Z^{(0)} + Σ_{i<k} M^i b`（升幂），等价于式 (1129) 的降幂形式。 -/
theorem iterate_eq_closedForm (M : V →ₗ[K] V) (b : V) (k : Nat) (z : V) :
    iterate M b k z = closedForm M b k z := by
  induction k generalizing z with
  | zero => simp [iterate, closedForm, powL]
  | succ k ih =>
      rw [iterate, ih]
      unfold closedForm
      rw [map_add, map_sum]
      simp only [← powL_succ_apply]
      rw [add_assoc]
      rw [sum_shift]

end PSCO
