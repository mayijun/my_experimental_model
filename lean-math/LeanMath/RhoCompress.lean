/-
§2.6 token 层 ρ 残差压缩与自适应码率控制（R+λD 的拉格朗日对偶上升）的数学核心。

对应 math.md §2.6：

    ρ_t  = ‖v_t − M_{t-1} k_t‖²                         冗余度 = 预测残差（新奇度）
    b_t  = clamp(a·(ρ_t−μ)/σ + c·ρ_t/μ, b_min, b_max)    比特门控（式 665 的实现代理）
    keep_t = ρ_t > μ + z·σ                                相对阈值丢/留
    R    = mean_t b_t                                     码率
    损失 = D + λ·R   ⟷   约束问题 min D s.t. R ≤ R*        率失真（式 665 的 R+λD）

本文件机械化以下断言（均无 sorry / 无新增公理，纯代数/实内积，不依赖测度论）：

1. 残差几何（实内积空间）：ρ ≥ 0；ρ = 0 ↔ v 被预测 p 精确重建；当预测与残差正交时
   ‖v‖² = ‖p‖² + ρ（能量分解 = 被吸收能量 + 新奇能量，正交性原理）。
2. 实现侧单调性：keep 判定关于 z 单调不增（z 是有效压缩强度旋钮）；比特门控 b 关于 ρ
   单调不减，故码率 R = mean b 随 ρ 下降而下降（记忆吸收冗余 ⇒ 省比特，梯度信号方向正确）。
3. 自适应码率控制的拉格朗日根基（用户选中的「让 λ 自己收敛」方案）：
   (a) KKT 充分性：若 (x*, λ*) 满足「x* 极小化 L(·,λ*)」∧ λ*≥0 ∧ 可行 ∧ 互补松弛
       λ*·(R(x*)−R*)=0，则 x* 是率失真问题 min{D(x) : R(x)≤R*} 的最优解；
   (b) 对偶超梯度：若 x 极小化 L(·,λ)，则 (R(x)−R*) 是凹对偶 q(λ)=min_x L(x,λ) 的
       超梯度——q(λ') ≤ q(λ) + (λ'−λ)·(R(x)−R*)，故「R > R* ⇒ 增大 λ」是正确的上升方向；
   (c) 鞍点 ⟹ KKT ⟹ 原始最优：L(x,λ) 的鞍点恰是约束问题的解；
   (d) 投影更新 λ ← max(0, λ + η·(R(x_λ)−R*)) 的不动点 ⟹ 互补松弛成立（约束紧或 λ=0）。
   即：让 λ 按该更新自调，其不动点恰好命中 KKT 条件 ⇒ 最优码率权衡。迭代收敛到不动点
   是投影超梯度上升的标准结论（凹对偶 + Lipschitz），列为已知标准结果（见「未形式化」）。

与现有形式化的关系：AssociativeMemory.lean 已在任意域/模上证明残差 ρ=0 ↔ 被历史重建
（residual_zero_iff_reconstructed）；本文件在 ℝ 内积空间补充 ρ 的**几何**（非负、平方范数、
正交分解），并新增**率失真优化**与**自适应 λ** 的代数核心，二者不重叠。
-/

import Mathlib

open scoped BigOperators
open scoped InnerProductSpace

namespace RhoCompress

/-! ## 1. 残差几何：ρ = ‖v − p‖² 的恒等式（实内积空间） -/

-- 与 mathlib InnerProductSpace.Basic 的变量声明对齐：InnerProductSpace 是 mixin，
-- 需先给出 (Semi)normedAddCommGroup，否则 ‖·‖ / 减法的实例无法合成。
-- 用 NormedAddCommGroup 而非 Seminormed：rho_zero_iff（ρ=0 ↔ v=p）依赖正定性
-- ‖x‖=0 → x=0，这只在赋范空间成立；inner_self_eq_zero / norm_add_sq 也都在 Normed 节。
variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- 冗余度 ρ = ‖v − p‖²，其中 p 是记忆对当前 token 的预测（p = M_{t-1} k_t）。 -/
def rho (v p : F) : ℝ := ‖v - p‖ ^ 2

/-- ρ 是平方范数，恒非负。 -/
theorem rho_nonneg (v p : F) : 0 ≤ rho v p := by
  exact sq_nonneg _

/-- ρ = 0 当且仅当 v 被预测 p 精确重建（残差为零 ⇔ 该 token 完全冗余、可压）。
    用内积正定性（inner_self_eq_zero）证明，不依赖 seminormed 下不一定成立的 norm_eq_zero。 -/
theorem rho_zero_iff (v p : F) : rho v p = 0 ↔ v = p := by
  have eq : ⟪v - p, v - p⟫_ℝ = ‖v - p‖ ^ 2 := real_inner_self_eq_norm_sq (v - p)
  have h1 : rho v p = 0 ↔ ⟪v - p, v - p⟫_ℝ = 0 := by
    simp [rho, eq]
  have h2 : ⟪v - p, v - p⟫_ℝ = 0 ↔ v - p = 0 :=
    inner_self_eq_zero (x := v - p)
  have h3 : v - p = 0 ↔ v = p := sub_eq_zero
  rw [h1, h2, h3]

/-- 正交性原理：当预测 p 与残差 v−p 正交时，总能量分解为
    「被预测吸收的能量 ‖p‖²」+「残差(新奇)能量 ρ」。
    对最优最小二乘预测严格成立；在线 delta-rule 记忆 M 近似满足（写出方向上投影）。 -/
theorem rho_pythagorean (v p : F) (horth : ⟪p, v - p⟫_ℝ = 0) :
    ‖v‖ ^ 2 = ‖p‖ ^ 2 + rho v p := by
  -- norm_add_sq 直接给出 ‖x+y‖^2 形式；ℝ 上 re ⟪·⟫ = ⟪·⟫，故正交时中项 2*re⟪p,v-p⟫ 为 0。
  -- 必须显式指定 𝕜 := ℝ，否则 InnerProductSpace ?m F 的标量域是元变量、实例合成卡住。
  have hsq : ‖p + (v - p)‖ ^ 2 = ‖p‖ ^ 2 + ‖v - p‖ ^ 2 := by
    have h := norm_add_sq (𝕜 := ℝ) (x := p) (y := v - p)
    simp [horth] at h ⊢
    exact h
  have hv : v = p + (v - p) := by abel
  -- 用 congrArg 只把 ‖v‖ 替换成 ‖p+(v-p)‖，避免 rw 误改右侧 ‖v-p‖ 里的 v。
  have hmain : ‖v‖ ^ 2 = ‖p‖ ^ 2 + ‖v - p‖ ^ 2 := by
    have hnorm : ‖v‖ = ‖p + (v - p)‖ := congrArg (fun x : F => ‖x‖) hv
    have h1 : ‖v‖ ^ 2 = ‖p + (v - p)‖ ^ 2 := by rw [hnorm]
    linarith [h1, hsq]
  simpa [rho] using hmain

/-! ## 2. 实现侧单调性：z（阈值）与 R（码率）是良定义的单调旋钮 -/

/-- 列表均值（空表为 0，与 Lean 实数除零约定一致）。 -/
noncomputable def mean (xs : List ℝ) : ℝ := xs.sum / xs.length

/-- 比特门控 b(ρ) = clamp(a·(ρ−μ)/σ + c·ρ/μ, b_min, b_max)（式 665 的实现代理，
    见 model/encoder.py RhoCompressor.forward）。 -/
noncomputable def b_of_rho (r mu sigma a c bmin bmax : ℝ) : ℝ :=
  max bmin (min bmax (a * (r - mu) / sigma + c * r / mu))

/-- keep 判定（命题形式）：ρ > μ + z·σ（相对阈值，见 model/encoder.py）。 -/
def keep (r mu sigma z : ℝ) : Prop := r > mu + z * sigma

/-- keep 的 Bool 版本，用于 List.filter 计数（实现侧 keep 是 bool mask）。 -/
noncomputable def keepb (r mu sigma z : ℝ) : Bool :=
  -- 直接写条件而非 keep r mu sigma z：keep 是 def，Decidable 实例搜索不会自动展开它。
  if r > mu + z * sigma then true else false

/-- keepb 与 keep 的等价桥。 -/
lemma keepb_iff (r mu sigma z : ℝ) : keepb r mu sigma z = true ↔ keep r mu sigma z := by
  simp [keepb, keep]

/-- z ↑ ⇒ 阈值 μ+zσ ↑ ⇒ keep 更难满足：保留判定关于 z 单调不增。
    故 z 是有效的压缩强度旋钮（z 越大越激进）。
    注意 z1*sigma 是双线性项，需用 nlinarith（linarith 无法处理乘积）。 -/
theorem keep_antitone_in_z (r mu sigma : ℝ) (hsig : 0 ≤ sigma)
    (z1 z2 : ℝ) (hle : z1 ≤ z2) :
    keep r mu sigma z2 → keep r mu sigma z1 := by
  intro h2
  dsimp only [keep] at h2 ⊢
  nlinarith

/-- 列表上保留 token 数关于 z 单调不增：z 越大，保留的 token 越少。 -/
theorem keepCount_antitone_in_z (rhos : List ℝ) (mu sigma : ℝ) (hsig : 0 ≤ sigma)
    (z1 z2 : ℝ) (hle : z1 ≤ z2) :
    (rhos.filter (fun x => keepb x mu sigma z2)).length
    ≤ (rhos.filter (fun x => keepb x mu sigma z1)).length := by
  have hstep : ∀ r, keepb r mu sigma z2 = true → keepb r mu sigma z1 = true := by
    intro r h2
    have h2' : keep r mu sigma z2 := (keepb_iff r mu sigma z2).mp h2
    have h1' : keep r mu sigma z1 := keep_antitone_in_z r mu sigma hsig z1 z2 hle h2'
    exact (keepb_iff r mu sigma z1).mpr h1'
  induction rhos with
  | nil => rfl
  | cons r rs ih =>
    by_cases h2 : keepb r mu sigma z2 = true
    · have h1 : keepb r mu sigma z1 = true := hstep r h2
      simp [List.filter_cons, h2, h1, ih]
    · have h2f : keepb r mu sigma z2 = false := by simpa [Bool.not_eq_true] using h2
      by_cases h1 : keepb r mu sigma z1 = true
      · simp [List.filter_cons, h2f, h1, ih] <;> linarith
      · have h1f : keepb r mu sigma z1 = false := by simpa [Bool.not_eq_true] using h1
        simp [List.filter_cons, h2f, h1f, ih]

/-- 逐点单调函数保 Forall₂ 求和不等式的自包含引理：
    若 l1, l2 逐点 ≤（Forall₂），f 单调不减，则 Σ f(l1) ≤ Σ f(l2)。 -/
lemma sum_map_le_of_forall₂_le (l1 l2 : List ℝ) (f : ℝ → ℝ)
    (hf : ∀ x y : ℝ, x ≤ y → f x ≤ f y)
    (h : List.Forall₂ (· ≤ ·) l1 l2) :
    (l1.map f).sum ≤ (l2.map f).sum := by
  -- 把命题写成 ∀ l1 l2，再对 l1 归纳：这样 l2 留在目标的 ∀ 里，cons 步 ih 才能套到新的 l2'。
  -- （直接 induction l1 时 l2 在上下文中是固定的，ih 不接受 l2'；generalizing 子句在此版本未生效。）
  have H : ∀ (l1 l2 : List ℝ), List.Forall₂ (· ≤ ·) l1 l2 → (l1.map f).sum ≤ (l2.map f).sum := by
    intro l1
    induction l1 with
    | nil =>
      intro l2 h
      have hnil : l2 = [] := List.forall₂_nil_left_iff.mp h
      simp [hnil]
    | cons a t ih =>
      intro l2 h
      rcases List.forall₂_cons_left_iff.mp h with ⟨b, l2', hab, htl, heq⟩
      rw [heq]
      simp [List.map_cons, List.sum_cons]
      exact add_le_add (hf a b hab) (ih l2' htl)
  exact H l1 l2 h

/-- b 关于 ρ 单调不减（σ>0, μ>0, a,c≥0）：记忆吸收冗余 ⇒ ρ↓ ⇒ b↓ ⇒ R↓。
    这是「码率正则梯度方向正确」的代数根基（式 665 的 R+λD）。 -/
theorem b_monotone_in_rho (mu sigma a c bmin bmax : ℝ)
    (hmu : 0 < mu) (hsig : 0 < sigma) (ha : 0 ≤ a) (hc : 0 ≤ c)
    (r1 r2 : ℝ) (hle : r1 ≤ r2) :
    b_of_rho r1 mu sigma a c bmin bmax ≤ b_of_rho r2 mu sigma a c bmin bmax := by
  dsimp only [b_of_rho]
  have h1 : a * (r1 - mu) / sigma ≤ a * (r2 - mu) / sigma := by
    gcongr <;> linarith
  have h2 : c * r1 / mu ≤ c * r2 / mu := by
    gcongr <;> linarith
  have hlin : a * (r1 - mu) / sigma + c * r1 / mu
           ≤ a * (r2 - mu) / sigma + c * r2 / mu := by linarith
  have hmin : min bmax (a * (r1 - mu) / sigma + c * r1 / mu)
            ≤ min bmax (a * (r2 - mu) / sigma + c * r2 / mu) :=
    min_le_min (le_refl bmax) hlin
  exact max_le_max (le_refl bmin) hmin

/-- 平均码率 R = mean_t b_t 关于每个 ρ_t 单调不减：任一 token 的 ρ 增大不会降低 R。
    结合 rho_zero_iff：记忆把 v 学得越准（ρ→0），整体码率越低。 -/
theorem meanRate_monotone_in_rho (rhos1 rhos2 : List ℝ)
    (mu sigma a c bmin bmax : ℝ)
    (hmu : 0 < mu) (hsig : 0 < sigma) (ha : 0 ≤ a) (hc : 0 ≤ c)
    (hforall : List.Forall₂ (· ≤ ·) rhos1 rhos2) :
    let f := fun r : ℝ => b_of_rho r mu sigma a c bmin bmax
    (rhos1.map f).sum / rhos1.length ≤ (rhos2.map f).sum / rhos2.length := by
  dsimp only
  set f : ℝ → ℝ := fun r => b_of_rho r mu sigma a c bmin bmax with hf
  have hmon : ∀ x y : ℝ, x ≤ y → f x ≤ f y := by
    intro x y hxy
    exact b_monotone_in_rho mu sigma a c bmin bmax hmu hsig ha hc x y hxy
  have hsum : (rhos1.map f).sum ≤ (rhos2.map f).sum :=
    sum_map_le_of_forall₂_le rhos1 rhos2 f hmon hforall
  have hlen : rhos1.length = rhos2.length := List.Forall₂.length_eq hforall
  -- 只在目标里把 rhos1.length 重写成 rhos2.length（hsum 不含 .length，不能 at hsum）
  rw [hlen]
  exact div_le_div_of_nonneg_right hsum (Nat.cast_nonneg rhos2.length)

/-! ## 3. 自适应码率控制：率失真约束 min{D : R ≤ R*} 的拉格朗日对偶与 KKT -/

variable {X : Type*} (D R : X → ℝ) (Rstar : ℝ)

/-- 可行性：码率不超过预算 R*。 -/
def Feasible (x : X) : Prop := R x ≤ Rstar

/-- 拉格朗日函数 L(x, lam) = D(x) + lam·(R(x) − R*)，lam ≥ 0。 -/
def Lagrangian (x : X) (lam : ℝ) : ℝ := D x + lam * (R x - Rstar)

/-- x 是 L(·, lam) 在 X 上的极小元（原始方向 / 内循环）。 -/
def IsMinimizer (x : X) (lam : ℝ) : Prop :=
    ∀ y : X, Lagrangian D R Rstar x lam ≤ Lagrangian D R Rstar y lam

/-- x 是原始最优：在所有可行点中失真 D 最小。 -/
def PrimalOptimal (x : X) : Prop :=
    Feasible R Rstar x ∧ ∀ y, Feasible R Rstar y → D x ≤ D y

/-- **KKT 充分性**（代数核心）：若 (x*, lam*) 满足
    (i) x* 极小化 L(·,lam*)、(ii) lam* ≥ 0、(iii) 可行 R(x*) ≤ R*、
    (iv) 互补松弛 lam*·(R(x*)−R*) = 0，则 x* 是率失真问题的最优解。

    意义：只要找到合适的 lam*，内循环极小化 L(·,lam*) 的解 x* 就是「码率 ≤ R* 下失真最小」的解。
    这正是「让 lam 自调到命中 R*」方案的理论合法性根基。 -/
theorem kkt_sufficient (xstar : X) (lamStar : ℝ)
    (hmin : IsMinimizer D R Rstar xstar lamStar)
    (hlam : 0 ≤ lamStar)
    (hfeas : Feasible R Rstar xstar)
    (hcs : lamStar * (R xstar - Rstar) = 0) :
    PrimalOptimal D R Rstar xstar := by
  constructor
  · exact hfeas
  · intro y hy
    have h1 : Lagrangian D R Rstar xstar lamStar ≤ Lagrangian D R Rstar y lamStar := hmin y
    have h2 : Lagrangian D R Rstar y lamStar ≤ D y := by
      dsimp only [Lagrangian, Feasible] at hy ⊢
      have : lamStar * (R y - Rstar) ≤ 0 := by nlinarith
      linarith
    have h3 : D xstar = Lagrangian D R Rstar xstar lamStar := by
      dsimp only [Lagrangian]; linarith [hcs]
    linarith

/-- 对偶函数在 lam 处的值（取一个极小元 x 的 L 值）。 -/
def DualValue (x : X) (lam : ℝ) : ℝ := Lagrangian D R Rstar x lam

/-- **对偶超梯度**：若 x 极小化 L(·,lam)、x' 极小化 L(·,lam')，则
    q(lam') ≤ q(lam) + (lam'−lam)·(R(x)−R*)。
    即 (R(x)−R*) 是凹对偶 q(lam) = min_x L(x,lam) 的超梯度。
    由此「R(x) > R* ⇒ 沿 +lam 方向上升 q」是正确的更新方向（投影超梯度上升）。 -/
theorem dual_supergradient (x x' : X) (lam lam' : ℝ)
    (_hx : IsMinimizer D R Rstar x lam) (hx' : IsMinimizer D R Rstar x' lam') :
    DualValue D R Rstar x' lam' ≤ DualValue D R Rstar x lam + (lam' - lam) * (R x - Rstar) := by
  have h1 : Lagrangian D R Rstar x' lam' ≤ Lagrangian D R Rstar x lam' := hx' x
  dsimp only [DualValue, Lagrangian] at h1 ⊢
  linarith

/-- 鞍点：x* 在 lam* 处极小化 L，lam* 在 [0,∞) 上极大化 L(x*,·)。 -/
def IsSaddle (xstar : X) (lamStar : ℝ) : Prop :=
    IsMinimizer D R Rstar xstar lamStar ∧ 0 ≤ lamStar ∧
    (∀ lam ≥ 0, Lagrangian D R Rstar xstar lam ≤ Lagrangian D R Rstar xstar lamStar)

/-- 鞍点 ⟹ KKT 条件（lam* ≥ 0 ∧ 可行 ∧ 互补松弛）。 -/
theorem saddle_implies_kkt (xstar : X) (lamStar : ℝ)
    (hs : IsSaddle D R Rstar xstar lamStar) :
    0 ≤ lamStar ∧ Feasible R Rstar xstar ∧ lamStar * (R xstar - Rstar) = 0 := by
  rcases hs with ⟨hmin, hlam, hmax⟩
  have hfeas : R xstar ≤ Rstar := by
    have h := hmax (lamStar + 1) (by linarith)
    dsimp only [Lagrangian] at h
    linarith
  have hcs : lamStar * (R xstar - Rstar) = 0 := by
    by_cases hpos : 0 < lamStar
    · have h := hmax 0 (by norm_num)
      dsimp only [Lagrangian] at h
      nlinarith
    · have : lamStar = 0 := by linarith
      simp [this]
  exact ⟨hlam, hfeas, hcs⟩

/-- 鞍点 ⟹ 原始最优（组合 saddle_implies_kkt + kkt_sufficient）。
    即 L(x,lam) 的鞍点恰是率失真约束问题的解。 -/
theorem saddle_implies_primal_optimal (xstar : X) (lamStar : ℝ)
    (hs : IsSaddle D R Rstar xstar lamStar) :
    PrimalOptimal D R Rstar xstar := by
  have hkkt := saddle_implies_kkt D R Rstar xstar lamStar hs
  rcases hkkt with ⟨hlam, hfeas, hcs⟩
  exact kkt_sufficient D R Rstar xstar lamStar hs.1 hlam hfeas hcs

/-- 投影对偶上升更新：lam ← max(0, lam + η·(R(x_lam) − R*))，η > 0。
    「R > R* ⇒ 增大 lam；R < R* ⇒ 减小 lam（但不低于 0）」的精确形式。 -/
def ProjectedUpdate (x : X) (lam η : ℝ) : ℝ := max 0 (lam + η * (R x - Rstar))

/-- **更新不动点 ⟹ KKT 互补松弛**：若 lam* 是 ProjectedUpdate 的不动点（且 x* 在 lam* 处
    极小化 L），则 R(x*) ≤ R* 且 lam*·(R(x*)−R*) = 0。
    即：让 lam 按该式自调直到稳定，其稳定点恰好命中 KKT 条件，由 kkt_sufficient 知最优。 -/
theorem update_fixedpoint_implies_cs (xstar : X) (lamStar η : ℝ)
    (hη : 0 < η)
    (hfp : ProjectedUpdate R Rstar xstar lamStar η = lamStar)
    (_hmin : IsMinimizer D R Rstar xstar lamStar) :
    Feasible R Rstar xstar ∧ lamStar * (R xstar - Rstar) = 0 := by
  set t : ℝ := lamStar + η * (R xstar - Rstar) with ht
  have hunfold : ProjectedUpdate R Rstar xstar lamStar η = max 0 t := by rfl
  rw [hunfold] at hfp
  by_cases h : 0 ≤ t
  · -- t ≥ 0 ⇒ max 0 t = t = lamStar ⇒ η·(R−R*) = 0 ⇒ R = R*
    have hmax : max 0 t = t := max_eq_right h
    have heq : t = lamStar := by rwa [hmax] at hfp
    have hprod : η * (R xstar - Rstar) = 0 := by nlinarith [ht, heq]
    have hrz : R xstar - Rstar = 0 :=
      (mul_eq_zero.mp hprod).resolve_left (ne_of_gt hη)
    have hfeas : R xstar ≤ Rstar := by
      have : R xstar = Rstar := by linarith [hrz]
      exact le_of_eq this
    exact ⟨hfeas, by simp [hrz]⟩
  · -- t < 0 ⇒ max 0 t = 0 = lamStar ⇒ lamStar = 0 且 R < R*
    have hlt : t < 0 := by nlinarith
    have hmax : max 0 t = 0 := max_eq_left (by nlinarith)
    have h0 : 0 = lamStar := by rwa [hmax] at hfp
    have hl0 : lamStar = 0 := h0.symm
    have hfeas : R xstar ≤ Rstar := by nlinarith [ht, hl0, hη, hlt]
    exact ⟨hfeas, by simp [hl0]⟩

end RhoCompress
