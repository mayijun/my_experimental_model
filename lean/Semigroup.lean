/-!
# 算子幺半群与「重建 / 生成」的形式化

对应 math.md §3.9 式 (29)–(30) 与命题 11：

    Ω = { Ψ : Z → Z },  id ∈ Ω,  Ψ₂ ∘ Ψ₁ ∈ Ω   （复合封闭，式 30）
    重建 = D(id(Z))，  生成 = D(Ψ(Z,ω))，Ψ ≠ id   （式 29）

本文件用纯 Lean（不依赖 mathlib）证明：

1. 自映射全体 `Z → Z` 在函数复合下构成**幺半群**：结合律、单位元 `id`；
2. 命题 11 的二分：`Ψ = id ⟺ 重建`，`Ψ ≠ id ⟺ 生成`，二者互斥且穷尽；
3. **生成 ⟺ 移动了某个点**：非恒等算子存在一点 `z` 使 `Ψ z ≠ z`（生成产出了新状态）；
4. 「描述是单位元邻域」的精确化：`id` 是所有点的公共不动点（承接 `fixed_point.lean`
   的 `read_only_fixes_all`）。

注意：本文件刻意**不引入** mathlib 的 `Monoid` 类实例，而是把幺半群公理当作
定理显式写出来——因为 §3.9 的 `Ω` 是否构成严格意义下的幺半群，正是本文要
**验证**的对象，而非预设。
-/

namespace OperatorMonoid

universe u

/-- 算子空间：`Z` 上的全体自映射。对应式 (30) 的 `Ω = { Ψ : Z → Z }`。 -/
def Op (Z : Type u) : Type u := Z → Z

/-- 恒等算子，即 `Ω` 的单位元。 -/
def idOp {Z : Type u} : Op Z := id

/-- 复合算子。 -/
def comp {Z : Type u} (f g : Op Z) : Op Z := fun z => f (g z)

/-! ## 幺半群三公理（式 30） -/

/-- 结合律：`(f ∘ g) ∘ h = f ∘ (g ∘ h)`。这是 `Ω` 能作并行扫描 / 算子复合的根基。 -/
theorem comp_assoc {Z : Type u} (f g h : Op Z) :
    comp (comp f g) h = comp f (comp g h) := by
  rfl

/-- 左单位：`id ∘ f = f`。 -/
theorem id_comp {Z : Type u} (f : Op Z) : comp idOp f = f := by
  apply funext; intro z; rfl

/-- 右单位：`f ∘ id = f`。 -/
theorem comp_id {Z : Type u} (f : Op Z) : comp f idOp = f := by
  apply funext; intro z; rfl

/-! ## 命题 11：重建 vs 生成 的二分（式 29） -/

/-- 重建 = 恒等算子（`Ψ = id`，谱系 `L₀`）。 -/
def isReconstruction {Z : Type u} (f : Op Z) : Prop := f = idOp

/-- 生成 = 非恒等算子（`Ψ ≠ id`，谱系 `L_{≥1}`）。 -/
def isGeneration {Z : Type u} (f : Op Z) : Prop := f ≠ idOp

/-- 二分互斥穷尽：每个算子要么是重建、要么是生成，且不可兼得。 -/
theorem generation_iff_not_reconstruction {Z : Type u} (f : Op Z) :
    isGeneration f ↔ ¬ isReconstruction f := by
  rfl

/-- 命题 11 核心：非恒等算子 ⟺ 存在被它移动的点（生成了输入之外的新状态）。
    这是「生成 = D(Ψ(Z,ω))、Ψ≠id」的算子学表述。 -/
theorem generation_iff_moves_some_point {Z : Type u} (f : Op Z) :
    isGeneration f ↔ ∃ z : Z, f z ≠ z := by
  constructor
  · intro hf
    apply Classical.byContradiction
    intro hno              -- hno : ¬ (∃ z, f z ≠ z)
    apply hf               -- 目标变为 f = idOp
    apply funext
    intro z
    apply Classical.byContradiction
    intro hz               -- hz : ¬ (f z = z) = f z ≠ z
    exact hno ⟨z, hz⟩
  · intro hex
    intro hf               -- hf : f = idOp
    rcases hex with ⟨z, hz⟩
    rw [hf] at hz          -- hz : z ≠ z
    exact hz rfl

/-- 重建（恒等）不移动任何点——`id` 是每个点的公共不动点。
    对应 math.md「描述是 `Ω` 的单位元邻域：`g_change → 0` 时 `Ψ_h → id`」。 -/
theorem reconstruction_fixes_all {Z : Type u} (f : Op Z) (hf : isReconstruction f) :
    ∀ z : Z, f z = z := by
  intro z
  rw [hf]
  rfl

/-- 「描述 ⊂ 算子」：恒等是算子空间里退化到极致的元素，描述是算子的特例。 -/
theorem id_is_trivial_operator {Z : Type u} : isReconstruction (idOp : Op Z) := rfl

end OperatorMonoid

-- ============================================================
-- 具体实例：`Nat.pred` 是「生成」算子，`id` 是「重建」算子
-- ============================================================
namespace OperatorExample

open OperatorMonoid

/-- `Nat.pred` 移动了点 1（`pred 1 = 0 ≠ 1`），故是生成算子。 -/
example : isGeneration (fun n : Nat => Nat.pred n) := by
  exact (generation_iff_moves_some_point (fun n : Nat => Nat.pred n)).2 ⟨1, by decide⟩

/-- 恒等算子不移动任何点，故是重建算子。 -/
example : isReconstruction (idOp : Op Nat) := rfl

end OperatorExample
