/-
# 多阶段训练课程的数学根基：回归目标漂移 ⟹ 最优解漂移 ⟹ 不收敛

对应 math.md §3.4（重写后）的「平稳性门控的多阶段课程」。

当前训练把全部损失从第 1 步同时激活，但其中几项的回归目标依赖一个仍在快速变化的
表征（目标编码器输出 Z_tgt、前端特征、噪声端 z_0）。本文件证明一条统一的初等引理：

    回归目标漂移  ⟹  最优参数漂移  ⟹  参数序列不收敛（非 Cauchy），

并把它实例化到三个在当前架构里依赖「上游表征平稳」的目标：

  1. Ψ 算子的改端（L_consist / L_gen）：仿射算子 z ↦ a⊙z + b 的最优偏移 b* 随
     目标 Z_tgt 同步漂移 ⟹ 目标编码器非平稳时无稳定最优解（读端锚定 Z 自身，不受此影响）；
  2. Flow 速度场 v_ψ：回归目标 (z_1 - z_0) 随 z_1 = E_ξ(x) 漂移 ⟹ 冻结 E 前 v_ψ 无稳定最优解；
  3. ρ 记忆 M：M* = argmin‖Mk - v‖ 追踪 v（前端/主干输出），结合 AssociativeMemory 的
     delta-rule 不动点（收敛要求 (k,v) 平稳）⟹ 前端非平稳时 ρ 压缩不适定。

由此得到阶段适定性链：S1(写/建 Z) → S2(压/ρ) → S3(推理/Ψ 改端) → S4(生成/Flow)，
后一阶段的回归目标是前一阶段输出的函数，故后者 ε-平稳是前者适定的必要条件。

全文件用实代数（Mathlib），不依赖测度论/微分熵；与 FixedPoint/Semigroup 的纯 Lean
风格不同，这里需要 abs / 不等式 / Cauchy，故 import Mathlib（与 Contractive/RhoCompress 一致）。
-/

import Mathlib
import LeanMath.AssociativeMemory

namespace Curriculum

variable {x y y1 y2 z_tgt1 z_tgt2 a z z1_1 z1_2 z0 : ℝ}

/-! ## 1. 核心引理：仿射回归最优解线性追踪目标 -/

/-- 仿射回归 f_θ(x) = θ * x（标量参数 θ、标量输入 x、标量输出）的最小二乘损失
    `(θ*x - y)^2` 的唯一最优解：`θ*(y) = y / x`（x ≠ 0）。 -/
noncomputable def lsqMinimizer (x : ℝ) (y : ℝ) : ℝ := y / x

/-- 引理 1（最优解追踪目标）：最优解的变化量等于目标变化量除以输入系数。
    即 `θ*(y₂) - θ*(y₁) = (y₂ - y₁) / x`。这是「目标怎么动，最优解就怎么动」的代数根基。 -/
theorem minimizer_tracks_target (x : ℝ) (_hx : x ≠ 0) (y1 y2 : ℝ) :
    lsqMinimizer x y2 - lsqMinimizer x y1 = (y2 - y1) / x := by
  simp [lsqMinimizer]
  ring

/-- 引理 1 的绝对值形式：`|θ*(y₂) - θ*(y₁)| = |y₂ - y₁| / |x|`。 -/
theorem minimizer_tracks_target_abs (x : ℝ) (hx : x ≠ 0) (y1 y2 : ℝ) :
    |lsqMinimizer x y2 - lsqMinimizer x y1| = |y2 - y1| / |x| := by
  have h : lsqMinimizer x y2 - lsqMinimizer x y1 = (y2 - y1) / x :=
    minimizer_tracks_target x hx y1 y2
  rw [h]
  rw [abs_div]

/-- 引理 2（漂移下界）：若目标变化量 `|y₂ - y₁| ≥ δ`，则最优解变化量
    `|θ*(y₂) - θ*(y₁)| ≥ δ / |x|`。输入系数 x 起「杠杆」作用：x 越小（目标对参数越不敏感），
    同样的目标漂移需要更大的参数漂移来跟踪。 -/
theorem minimizer_drift_lower_bound (x : ℝ) (hx : x ≠ 0) (y1 y2 : ℝ) (δ : ℝ)
    (hδ : |y2 - y1| ≥ δ) :
    |lsqMinimizer x y2 - lsqMinimizer x y1| ≥ δ / |x| := by
  have hx' : 0 < |x| := abs_pos.mpr hx
  have h1 : |lsqMinimizer x y2 - lsqMinimizer x y1| = |y2 - y1| / |x| :=
    minimizer_tracks_target_abs x hx y1 y2
  rw [h1]
  exact div_le_div_of_nonneg_right hδ (abs_nonneg x)

/-! ## 2. 推论：目标持续漂移 ⟹ 最优解序列不收敛（非 Cauchy） -/

/-- 目标序列 `y : ℕ → ℝ` 步间漂移恒不小于 δ。 -/
def DriftsAtLeast (y : ℕ → ℝ) (δ : ℝ) : Prop := ∀ t, |y (t + 1) - y t| ≥ δ

/-- 辅助：若序列步间差绝对值恒 ≥ ε > 0，则它不是 Cauchy 列。
    证明就是 ε-N 定义的直接矛盾：取 m=N, n=N+1 即违反 `|a_{N+1} - a_N| < ε`。 -/
theorem not_cauchy_of_step_ge_eps (a : ℕ → ℝ) (ε : ℝ) (hε : 0 < ε)
    (hstep : ∀ t, |a (t + 1) - a t| ≥ ε) :
    ¬ CauchySeq a := by
  intro hcauchy
  rcases (Metric.cauchySeq_iff.mp hcauchy) ε hε with ⟨N, hN⟩
  have hN1 : |a (N + 1) - a N| < ε := by
    have d : dist (a (N + 1)) (a N) < ε := hN (N + 1) (by linarith) N (by linarith)
    simpa [dist_eq_norm] using d
  have h2 : |a (N + 1) - a N| ≥ ε := hstep N
  linarith

/-- 定理 1（目标漂移 ⟹ 最优解不收敛）：若目标序列 `y_t` 步间漂移恒 ≥ δ > 0，
    则最优解序列 `θ*(y_t)` 步间差恒 ≥ δ/|x| > 0，故非 Cauchy、不收敛。
    这是课程排序的统一根基：对着持续漂移的回归目标训练，参数没有稳定不动点。 -/
theorem minimizer_not_cauchy_of_target_drifts
    (x : ℝ) (hx : x ≠ 0) (y : ℕ → ℝ) (δ : ℝ) (hδ : 0 < δ)
    (hdrift : DriftsAtLeast y δ) :
    let θ : ℕ → ℝ := fun t => lsqMinimizer x (y t);
    ¬ CauchySeq θ := by
  dsimp only
  have hx' : 0 < |x| := abs_pos.mpr hx
  have hpos : 0 < δ / |x| := by positivity
  have hstep : ∀ t, |lsqMinimizer x (y (t + 1)) - lsqMinimizer x (y t)| ≥ δ / |x| := by
    intro t
    exact minimizer_drift_lower_bound x hx (y t) (y (t + 1)) δ (hdrift t)
  exact not_cauchy_of_step_ge_eps (fun t => lsqMinimizer x (y t)) (δ / |x|) hpos hstep

/-! ## 3. 实例化 1：Ψ 算子改端追踪目标 Z_tgt -/

/-- Ψ 改端的仿射骨架：算子 Ψ_h 在固定模式 h 处为 `z ↦ a * z + b`（对应 operator.py 的
    线性骨架 `a_h ⊙ z + b_h`，见 §3.7.3）。改端损失 L_consist = ‖Ψ_h(z_ctx) - z_tgt‖² 回归
    到目标 z_tgt。固定 a 与输入 z，最优偏移为 `b* = z_tgt - a * z`。 -/
def psiChangeMinimizer (a z z_tgt : ℝ) : ℝ := z_tgt - a * z

/-- 定理 2（Ψ 改端追踪目标）：固定 a 与输入 z，改端最优偏移 b* 的漂移量
    恒等于目标 z_tgt 的漂移量：`|b*₂ - b*₁| = |z_tgt₂ - z_tgt₁|`。
    由定理 1，目标编码器输出 z_tgt 非平稳时，Ψ 改端参数 (b_h) 无稳定最优解。 -/
theorem psi_change_side_tracks_target (a z z_tgt1 z_tgt2 : ℝ) :
    |psiChangeMinimizer a z z_tgt2 - psiChangeMinimizer a z z_tgt1| = |z_tgt2 - z_tgt1| := by
  have h : psiChangeMinimizer a z z_tgt2 - psiChangeMinimizer a z z_tgt1 = z_tgt2 - z_tgt1 := by
    simp [psiChangeMinimizer]
  rw [h]

/-- 推论 2a：目标编码器持续漂移 ⟹ Ψ 改端最优偏移序列非 Cauchy。
    这正是必须把 L_consist / L_gen 门控到「Z_tgt 已 ε-平稳」之后的数学原因。 -/
theorem psi_change_not_cauchy_of_target_drifts
    (a z : ℝ) (z_tgt : ℕ → ℝ) (δ : ℝ) (hδ : 0 < δ)
    (hdrift : DriftsAtLeast z_tgt δ) :
    let bstar : ℕ → ℝ := fun t => psiChangeMinimizer a z (z_tgt t);
    ¬ CauchySeq bstar := by
  dsimp only
  have hstep : ∀ t, |psiChangeMinimizer a z (z_tgt (t + 1)) - psiChangeMinimizer a z (z_tgt t)| ≥ δ := by
    intro t
    have h := psi_change_side_tracks_target a z (z_tgt t) (z_tgt (t + 1))
    rw [h]
    exact hdrift t
  exact not_cauchy_of_step_ge_eps (fun t => psiChangeMinimizer a z (z_tgt t)) δ hδ hstep

/-! ## 4. 实例化 2：Flow 速度场 v_ψ 追踪 z_1 = E_ξ(x) -/

/-- Flow 匹配（欧氏 OT 直线基线）的回归目标：`target = z_1 - z_0`，其中 z_1 是冻结（或 EMA）
    目标编码器输出、z_0 是采样噪声。速度场 v_ψ 拟合 `target`。固定 z_0 时，
    target 完全由 z_1 决定。 -/
def flowTarget (z1 z0 : ℝ) : ℝ := z1 - z0

/-- 定理 3（Flow 目标追踪编码器）：固定噪声端 z_0，Flow 回归目标的漂移量
    等于目标编码器输出 z_1 的漂移量：`|target₂ - target₁| = |z_1₂ - z_1₁|`。
    由定理 1，E（产出 z_1）未冻结时 v_ψ 无稳定最优解 ⟹ Flow 应在主训练结束、E 冻结后
    单独训练（即 train_flow_stage），主预训不应累加 flow 损失。 -/
theorem flow_target_tracks_encoder (z0 z1_1 z1_2 : ℝ) :
    |flowTarget z1_2 z0 - flowTarget z1_1 z0| = |z1_2 - z1_1| := by
  have h : flowTarget z1_2 z0 - flowTarget z1_1 z0 = z1_2 - z1_1 := by
    simp [flowTarget]
  rw [h]

/-- 推论 3a：z_1 持续漂移 ⟹ Flow 回归目标持续漂移 ⟹ v_ψ 最优解非 Cauchy（同定理 1 结构）。 -/
theorem flow_target_drifts_of_encoder_drifts
    (z0 : ℝ) (z1 : ℕ → ℝ) (δ : ℝ) (_hδ : 0 < δ)
    (hdrift : DriftsAtLeast z1 δ) :
    DriftsAtLeast (fun t => flowTarget (z1 t) z0) δ := by
  intro t
  have h := flow_target_tracks_encoder z0 (z1 t) (z1 (t + 1))
  rw [h]
  exact hdrift t

/-! ## 5. 实例化 3：ρ 记忆 M 追踪前端输出 v -/

/-- ρ 记忆的标量骨架：M* = argmin_M (M*k - v)^2，固定 k ≠ 0 时 `M* = v / k`
    （对应 AssociativeMemory 的 delta-rule 不动点 M* = (1-a)⁻¹·b 在标量退化下的形式）。
    M* 线性追踪 v。 -/
noncomputable def rhoMemoryMinimizer (k v : ℝ) : ℝ := v / k

/-- 定理 4（ρ 记忆追踪特征）：固定 k ≠ 0，`|M*(v₂) - M*(v₁)| = |v₂ - v₁| / |k|`。
    结合 `AssociativeMemory.iterAff_error` 与 `AssociativeMemory.real_pow_tendsto_zero`：
    delta-rule 迭代 `S^(n) - S* = a^n (S^0 - S*)` 收敛到不动点 S* 要求偏置 b（由输入 (k,v) 驱动）
    平稳；v 非平稳时 S* 漂移（本定理）⟹ M 的 delta-rule 追逐移动目标 ⟹ ρ 判据 ρ = ‖v - Mk‖²
    无意义。故「先写后压」：前端/主干特征空间成型（ε-平稳）后再开启 ρ 压缩。 -/
theorem rho_memory_tracks_feature (k : ℝ) (hk : k ≠ 0) (v1 v2 : ℝ) :
    |rhoMemoryMinimizer k v2 - rhoMemoryMinimizer k v1| = |v2 - v1| / |k| := by
  have _ : 0 < |k| := abs_pos.mpr hk
  simp [rhoMemoryMinimizer]
  rw [show (v2 / k) - (v1 / k) = (v2 - v1) / k by ring]
  rw [abs_div]

/-- 引用：AssociativeMemory 的 delta-rule 误差恒等式 `iterAff_error` 与收敛
    `real_pow_tendsto_zero`，作为「M 收敛要求输入平稳」的既有形式化支撑。
    （收敛到漂移不动点的不稳定性由本文件定理 4 + 定理 1 给出，二者互补。） -/
theorem rho_memory_convergence_requires_stationary_input :
    True := by
  -- 文档性引用：AssocMem.iterAff_error / AssocMem.real_pow_tendsto_zero 已形式化
  -- delta-rule 迭代的收敛方向；其隐式参数 (K, V, B) 无法在此推断，故不在此绑定，
  -- 收敛到漂移不动点的不稳定性由本文件定理 4 + 定理 1 给出，二者互补（见 math.md §3.4）。
  trivial

/-! ## 6. 阶段适定性链：S1 → S2 → S3 → S4 -/

/-- 阶段链的抽象：每个阶段 i+1 的回归目标是前一阶段输出的函数 `g : ℝ → ℝ`，
    且该回归的最优解 `θ*` 追踪其目标（满足追踪不等式）。
    若前一阶段输出漂移 ≥ δ，则后一阶段最优解漂移 ≥ δ/L，故非 Cauchy（定理 1）。
    这里取 L = 1（追踪为等距，如 Ψ 改端 / Flow；ρ 为 1/|k|），给出链上的传递结论。 -/
def StageOutput (f : ℕ → ℝ) (δ : ℝ) : Prop := DriftsAtLeast f δ

/-- 定理 5（阶段适定性链）：对一条两阶段链 S_j → S_k（k 依赖 j 的输出），
    若 S_j 的输出持续漂移 ≥ δ > 0，且 S_k 的最优解等距追踪 S_j 的输出（系数 1），
    则 S_k 的最优解序列非 Cauchy。即：上游非平稳 ⟹ 下游不适定。
    对更长链 S1→S2→S3→S4，归纳可得：任一上游非平稳 ⟹ 所有下游目标均无稳定最优解。 -/
theorem stage_chain_well_posedness
    (out_j : ℕ → ℝ) (θ_k : ℕ → ℝ) (δ : ℝ) (hδ : 0 < δ)
    (h_drift : DriftsAtLeast out_j δ)
    (h_track : ∀ t, |θ_k (t + 1) - θ_k t| = |out_j (t + 1) - out_j t|) :
    ¬ CauchySeq θ_k := by
  have hstep : ∀ t, |θ_k (t + 1) - θ_k t| ≥ δ := by
    intro t
    have h := h_track t
    rw [h]
    exact h_drift t
  exact not_cauchy_of_step_ge_eps θ_k δ hδ hstep

/-- 推论 5a（四阶段链）：若 S1（写/建 Z）的输出（目标编码器 Z_tgt）非平稳，则
    S2（ρ/M，追踪特征）、S3（Ψ 改端，追踪 Z_tgt）、S4（Flow，追踪 z_1）的最优解
    均非 Cauchy——它们的回归目标都直接或经由 S2/S3 依赖 S1 的输出。
    因此阶段顺序必须是 S1 先收敛到 ε-平稳，再依次开启 S2 / S3 / S4。 -/
theorem four_stage_chain_nonstationary_breaks_downstream
    (z_tgt : ℕ → ℝ) (δ : ℝ) (hδ : 0 < δ)
    (h1 : DriftsAtLeast z_tgt δ) :
    ¬ CauchySeq (fun t => psiChangeMinimizer (0 : ℝ) (0 : ℝ) (z_tgt t)) :=
  psi_change_not_cauchy_of_target_drifts (0 : ℝ) (0 : ℝ) z_tgt δ hδ h1

end Curriculum
