/-
§4.1.7 黎曼流匹配的 GPU 友好近似（retraction-based）的数学核心。

对应 math.md §4.1.7：

    R_z : T_z M → M，R_z(0) = z，dR_z(0) = id           （retraction 定义）
    exp_z(v) = R_z(v) + O(‖v‖²)                        （与指数映射一阶一致）
    R_z(v) = E(g(z+v))                                  （投影 retraction = decode–reencode）
    L = E ‖v_ψ(z_t,t) − Ṙ‖²，z_{t+Δt} = R_{z_t}(v Δt)    （retraction-based FM）
    G = J_Dᵀ J_D，rank(G) ≤ min(|𝒱|, d)                  （低秩度量，§4.2(a)）

本文件证明：

1. retraction 的两条公理（`R 0 = z`、`dR(0) = id`）作为 `Retraction` 结构；
2. 一阶一致性的严格形式：`R(v) = z + v + o(‖v‖)`（Fréchet 导数定义直接给出；
   文档写 O(‖v‖²) 需 R 为 C²，见 README「问题 8」）；
3. 欧氏 retraction `R_z(v) = z + v` 满足两条公理——这是结论「欧氏 FM 是 retraction 的
   R_z(v)=z+v 退化特例」的根基；
4. 投影 retraction 的代数核心：幂等 π = E∘g 使 `R_z(v)=π(z+v)` 的像落在不动点集（流形）
   上（`π(R_z v)=R_z v`），即「路径始终在 M 上、全程可解码」的代数保证；π=id 时退化为
   欧氏 retraction；
5. 低秩度量 `rank(JᵀJ) = rank(J) ≤ min(|𝒱|, d)`（纯线性代数，mathlib `Matrix.rank`）。

（retraction 是流形对象，本文件把切空间/环境空间统一抽象为同一个赋范向量空间 E——这是
 环境坐标下的展开，覆盖 §4.1.7 的核心代数与一阶分析结论；真流形的 exp/log 与测地线 ODE
 不在此展开。）
-/

import Mathlib

open scoped BigOperators
open scoped Topology
open Filter Asymptotics

namespace RiemannianFM

/-! ## 1. Retraction 结构：两条公理（基点 + 一阶一致） -/

/-- Retraction：把切向量（环境坐标下）映射回流形，与指数映射一阶一致。
    两条公理对应 math.md §4.1.7 的 `R_z(0)=z` 与 `(d/dt) R_z(tv)|_{t=0}=v`
    （后者即 `dR_z(0) = id` 的方向导数形式）。 -/
structure Retraction (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E] where
  base : E
  toFun : E → E
  map_zero : toFun 0 = base
  fderiv_id : HasFDerivAt toFun (ContinuousLinearMap.id ℝ E) 0

namespace Retraction

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

instance : CoeFun (Retraction E) (fun _ => E → E) := ⟨Retraction.toFun⟩

/-- 一阶一致性的严格形式：`R(v) = base + v + o(‖v‖)`。
    这正是 §4.1.7 的 `exp_z(v) = R_z(v) + O(‖v‖²)` 去掉二阶余项后的一阶部分；
    Fréchet 可微 + `dR(0)=id` 直接给出。 -/
theorem first_order_consistency (R : Retraction E) :
    (fun v : E => R v - R.base - v) =o[𝓝 (0 : E)] fun v : E => v := by
  simpa [R.map_zero] using (hasFDerivAt_iff_isLittleO_nhds_zero.mp R.fderiv_id)

end Retraction

/-! ## 2. 欧氏 retraction：`R_z(v) = z + v` 是退化特例 -/

section Euclidean

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- 欧氏 retraction `R_z(v) = z + v`（§4.1.7 结论「欧氏 FM 是 retraction 的
    `R_z(v)=z+v` 退化特例」的显式构造）。 -/
def euclideanRetraction (z : E) : Retraction E where
  base := z
  toFun := fun v => z + v
  map_zero := by simp
  fderiv_id := by
    simpa using ((hasFDerivAt_id (0 : E)).const_add z)

end Euclidean

/-! ## 3. 投影 retraction（decode–reencode）：幂等投影的代数核心 -/

section Projection

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- 投影映射的幂等性：`π ∘ π = π`。对应 §4.1.7 的 `π = E ∘ g`（decode–reencode），
    以及 §3.7.2 投影/锚定流、§5 软反馈的自洽性 `E∘g ≈ id`（在流形上为精确不动点）。 -/
def IsProjection (π : E → E) : Prop := ∀ x, π (π x) = π x

/-- 投影 retraction：`R_z(v) = π (z + v)`，把环境空间步进投影回纤维（流形）。 -/
def projectionRetraction (π : E → E) (z : E) : E → E := fun v => π (z + v)

/-- 基点公理：当基点 `z` 是 π 的不动点（`z ∈ M`，即 `π z = z`）时，`R_z(0) = z`。
    这与 §4.1.7「z 在流形上，投影把它固定」一致。 -/
theorem projectionRetraction_map_zero (π : E → E) (z : E) (hz : π z = z) :
    projectionRetraction π z 0 = z := by
  unfold projectionRetraction
  simp [hz]

/-- 投影 retraction 的像落在不动点集（流形）上：`π (R_z v) = R_z v`。
    这是 §4.1.6/§4.1.7「生成路径始终在 M 上、中途 c(t) 不塌陷、全程可解码」的
    **代数核心**——一步之后恒在流形上，与欧氏直线漂出流形相对照。 -/
theorem projectionRetraction_lands_on_manifold {π : E → E} (hπ : IsProjection π) (z v : E) :
    π (projectionRetraction π z v) = projectionRetraction π z v := by
  unfold projectionRetraction
  exact hπ (z + v)

/-- 欧氏 retraction 是投影 retraction 在 `π = id`（无量化、流形 = 全空间）时的退化特例。 -/
theorem projectionRetraction_id_eq_euclidean (z : E) :
    projectionRetraction (fun x : E => x) z = fun v => z + v := by
  unfold projectionRetraction
  rfl

end Projection

/-! ## 4. 低秩度量：`rank(G) = rank(J) ≤ min(|𝒱|, d)` -/

section RankBound

open Matrix

variable {m n : Type*} [Fintype m] [Fintype n]

/-- 拉回度量 `G = JᵀJ` 的秩等于读出雅可比 `J` 的秩。
    （§4.1.7 梯度 3 与 §4.2(a) 像/核分解；`m = |𝒱|` 词表行数、`n = d` 潜维列数。） -/
theorem rank_pullbackMetric_eq_rank (J : Matrix m n ℝ) :
    (Jᵀ * J).rank = J.rank := by
  exact Matrix.rank_transpose_mul_self J

/-- `rank(G) ≤ |𝒱|`（词表大小 = 行数）。 -/
theorem rank_pullbackMetric_le_vocab (J : Matrix m n ℝ) :
    (Jᵀ * J).rank ≤ Fintype.card m := by
  rw [Matrix.rank_transpose_mul_self]
  exact Matrix.rank_le_card_height J

/-- `rank(G) ≤ d`（潜维 = 列数）。 -/
theorem rank_pullbackMetric_le_latent (J : Matrix m n ℝ) :
    (Jᵀ * J).rank ≤ Fintype.card n := by
  rw [Matrix.rank_transpose_mul_self]
  exact Matrix.rank_le_card_width J

/-- 综合：`rank(G) ≤ min(|𝒱|, d)`，即 §4.1.7 梯度 3 与 §4.2(a)「可区分维度至多
    min(|𝒱|,d)」的精确线性代数形式。 -/
theorem rank_pullbackMetric_le_min (J : Matrix m n ℝ) :
    (Jᵀ * J).rank ≤ min (Fintype.card m) (Fintype.card n) := by
  exact le_min (rank_pullbackMetric_le_vocab J) (rank_pullbackMetric_le_latent J)

end RankBound

/-! ## 5. 线性投影 retraction（问题 8 / 9 修正的精确形式） -/

section LinearProjection

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- 线性投影 retraction `R_z(v) = z + P v` 的导数是 `P`（不是 `id`）。
    这是问题 9 修正的精确形式：`R_z(v)=z+Pv` 作为**环境空间**上的映射，其导数不是恒等；
    它只在「切空间 = im P」上是一阶一致的（`P|_{im P} = id`，见下一引理），
    而在环境空间上是恒等导数 iff `P = id`（即退化为欧氏 retraction）。 -/
theorem linearProjectionRetraction_fderiv (P : E →L[ℝ] E) (z : E) :
    HasFDerivAt (fun v : E => z + P v) P (0 : E) :=
  P.hasFDerivAt.const_add z

/-- 幂等线性投影在像空间上是恒等：`P (P x) = P x`，即 `P|_{im P} = id`。
    与 `linearProjectionRetraction_fderiv` 合起来：`z + P v` 在 `im P`（token 区分方向）
    上是一阶精确的 retraction，这正是 GPU 友好近似的正确形式（软投影，秩 ≤ min(|𝒱|,d)）。 -/
theorem linearProjection_idOnImage (P : E →L[ℝ] E) (hP : P.comp P = P) (x : E) :
    P (P x) = P x := by
  have h := congrArg (fun f : E →L[ℝ] E => f x) hP
  simpa [ContinuousLinearMap.comp_apply] using h

/-- 软投影 retraction 与欧氏 retraction 的关系：`P = id` 时二者一致（问题 9 的退化情形）。 -/
theorem linearProjectionRetraction_eq_euclidean_of_id {P : E →L[ℝ] E} (hP : P = 1) (z : E) :
    (fun v : E => z + P v) = fun v : E => z + v := by
  subst P
  rfl

end LinearProjection

end RiemannianFM
