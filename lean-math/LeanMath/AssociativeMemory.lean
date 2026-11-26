/-
§2.9 替代结构 CARO（门控结合扫描算子）的数学核心：
把「按 ρ 丢/留 → 落入 Z → 在 Z 中对齐并推理」折叠进单一仿射结合扫描状态。

对应 math.md §2.9：

    S_{t+1} = a_t · S_t + g_t · (v_t ⊗ k_t)        状态更新（仿射、结合、可并行扫描）
    z_t       = S_t · q_t                           读出 = 内容可寻址长程检索（= Σ 衰减·(k_i·q_t) v_i）
    ρ_t       = ‖v_t - S_{t-1} · k_t‖²              残差 = 新奇度（压缩信号）
    S^{(n+1)} = a · S^{(n)} + b ,  |a|<1            推理迭代 → 收缩不动点 S* = b/(1-a)

本文件机械化以下断言（均无 sorry / 无新增公理）：

1. update 是 PSCO.Aff 仿射映射 ⇒ 复用 Affine.lean 的结合幺半群 ⇒ 前缀扫描合法
   （并行扫描 O(log L) 深度的代数根基）；
2. 状态读出恒等式：S_T · q = Σ_{i<T} (Π_{j=i+1}^{T-1} a_j) · g_i · B(k_i, q) · v_i
   —— 精确的「按键相似度加权聚合历史」= 内容可寻址长程注意力，线性状态即可实现；
3. 残差 = 预测误差：S_{t-1}·k_t 本身就是历史按 B(k_i,k_t) 的加权和，ρ_t=0 ⇔ v_t 被历史精确重建
   （⇔ 该 token 冗余，可被压缩）；
4. 推理迭代的不动点公式 S* = (1-a)⁻¹·b 与误差恒等式 S^{(n)} - S* = a^n·(S⁰ - S*)，
   |a|<1 时误差→0（收敛到不动点）；
5. 防塌缩：不同输入 b₁≠b₂ ⇒ 不同不动点 S*₁≠S*₂（不动点输入依赖，不会坍缩到单一状态）。
-/

import Mathlib
import LeanMath.Affine
import LeanMath.Contractive

open scoped BigOperators

namespace AssocMem

universe u

variable {K : Type*} [Field K]
variable {V : Type u} [AddCommGroup V] [Module K V]

/-! ## 1. 外积算子与状态更新 -/

-- 键相似度双线性型 B : V × V → K（对应线性注意力的核 k·q）；取 B : V →ₗ (V →ₗ K)，则 B k q 即标量 B(k,q)。
-- 用行注释而非文档注释：lean4 v4.36.0 中 /-- 紧贴 variable 会触发 "expected 'lemma'" 解析错误，
-- 导致 B 不进作用域、后续全部报 "Function expected"。
variable (B : V →ₗ[K] (V →ₗ[K] K))

/-- 外积算子 `outer k v : q ↦ B(k, q) • v`，即把键 k 的相似度权重写到值 v 上。 -/
def outer (k v : V) : V →ₗ[K] V :=
  { toFun := fun q => (B k q) • v
    map_add' := by
      intro q₁ q₂
      simpa [map_add, add_smul] using rfl
    map_smul' := by
      intro c q
      have h1 : (B k) (c • q) = c • (B k q) := (B k).map_smul c q
      simpa [h1, mul_smul] using rfl }

/-- 一步状态更新：`S ↦ a • S + g • outer(k, v)`。
    `a` 是遗忘/衰减因子（谱夹层中 δ<a<1-γ），`g` 是由残差 ρ 驱动的写入门控。 -/
def update (a g : K) (k v : V) (S : V →ₗ[K] V) : V →ₗ[K] V :=
  a • S + g • outer B k v

/-! ## 2. 更新是仿射映射 ⇒ 复用结合幺半群（并行扫描根基） -/

/-- `update a g k v` 是一个仿射映射：线性部分 `a · id`，平移部分 `g · outer(k,v)`。
    因此它是 `PSCO.Aff` 的实例，`PSCO.Aff.comp_assoc`（已验证）保证任意括号化结果相同，
    这正是前缀扫描（Blelloch scan）计算 N 步复合的代数前提。 -/
theorem update_is_affine (a g : K) (k v : V) :
    ∃ (aff : PSCO.Aff K (V →ₗ[K] V)), ∀ (S : V →ₗ[K] V),
      aff.toFun S = update B a g k v S := by
  refine ⟨⟨a • LinearMap.id, g • outer B k v⟩, ?_⟩
  intro S
  simp [update, PSCO.Aff.toFun]
  <;> rfl

/-! ## 3. 内容可寻址：状态读出 = 历史键相似度加权和（长程检索恒等式） -/

/-- 一个时间步的全部参数：衰减 a、写入门 g、键 k、值 v。
    显式带 K/V 参数（与 Affine.lean 的 `Aff K V` 同构），避免 `Step` 在 binder 中
    被当作隐式参数类型、无法合成 K/V 的问题。 -/
structure Step (K : Type*) [Field K] (V : Type u) [AddCommGroup V] [Module K V] where
  a : K
  g : K
  k : V
  v : V

/-- 从前 0 状态出发，按序列 `f` 扫描 T 步得到的状态。 -/
def stateFrom (f : Nat → Step K V) : Nat → (V →ₗ[K] V)
  | 0     => 0
  | t + 1 => update B (f t).a (f t).g (f t).k (f t).v (stateFrom f t)

/-- 核心恒等式（内容可寻址 / 长程注意力）：
    扫描 T 步后，状态在任意查询 q 上的读出，等于对全部历史 i<T 的
    「键相似度 B(k_i, q) × 写入强度 g_i × 累积衰减 Π a_j」加权聚合值 v_i。

    这说明：**无需 O(L²) 注意力矩阵，常数大小的线性状态 S 即可实现对整个历史的
    精确内容寻址检索**。当 a 接近 1（谱夹层上界 1-γ，γ 小）时衰减慢 ⇒ 长程记忆。 -/
theorem stateFrom_readout (f : Nat → Step K V) (T : Nat) (q : V) :
    (stateFrom B f T) q
    = ∑ i ∈ Finset.range T,
        ((∏ j ∈ Finset.Ico (i + 1) T, (f j).a) * (f i).g * (B (f i).k q)) • (f i).v := by
  induction T with
  | zero =>
    simp [stateFrom]
  | succ T ih =>
    have h_step : (stateFrom B f (T + 1)) q
        = (f T).a • ((stateFrom B f T) q) + ((f T).g * (B (f T).k q)) • (f T).v := by
      simp [stateFrom, update, outer, LinearMap.add_apply, LinearMap.smul_apply, mul_smul]
      <;> rfl
    rw [h_step, ih]
    rw [Finset.sum_range_succ, Finset.smul_sum]
    have hsum : ∑ i ∈ Finset.range T, (f T).a • (((∏ j ∈ Finset.Ico (i + 1) T, (f j).a) * (f i).g * (B (f i).k q)) • (f i).v)
             = ∑ i ∈ Finset.range T, ((∏ j ∈ Finset.Ico (i + 1) (T + 1), (f j).a) * (f i).g * (B (f i).k q)) • (f i).v := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [← mul_smul]
      congr 1
      have hle : i + 1 ≤ T := by
        have : i < T := Finset.mem_range.mp hi
        linarith
      have h_eq : ∏ j ∈ Finset.Ico (i + 1) (T + 1), (f j).a
                = (∏ j ∈ Finset.Ico (i + 1) T, (f j).a) * (f T).a :=
        Finset.prod_Ico_succ_top hle (fun j => (f j).a)
      have hprod : (f T).a * (∏ j ∈ Finset.Ico (i + 1) T, (f j).a)
                 = ∏ j ∈ Finset.Ico (i + 1) (T + 1), (f j).a := by
        have hcomm : (f T).a * (∏ j ∈ Finset.Ico (i + 1) T, (f j).a)
                   = (∏ j ∈ Finset.Ico (i + 1) T, (f j).a) * (f T).a := by ring
        exact hcomm.trans h_eq.symm
      calc
        (f T).a * ((∏ j ∈ Finset.Ico (i + 1) T, (f j).a) * (f i).g * (B (f i).k q))
            = ((f T).a * (∏ j ∈ Finset.Ico (i + 1) T, (f j).a)) * (f i).g * (B (f i).k q) := by ring
        _ = (∏ j ∈ Finset.Ico (i + 1) (T + 1), (f j).a) * (f i).g * (B (f i).k q) := by rw [hprod]
    rw [hsum]
    congr 1
    simp [Finset.prod_eq_one] <;> ring

/-- 直接推论：把查询取为当前键 k_t，预测值就是历史按键相似度的加权和。 -/
theorem prediction_is_content_sum (f : Nat → Step K V) (t : Nat) :
    (stateFrom B f t) (f t).k
    = ∑ i ∈ Finset.range t,
        ((∏ j ∈ Finset.Ico (i + 1) t, (f j).a) * (f i).g * (B (f i).k (f t).k)) • (f i).v :=
  stateFrom_readout B f t (f t).k

/-! ## 4. 残差 = 预测误差（新奇度 / 压缩信号） -/

/-- 残差 ρ_t：当前值 v_t 与「历史对键 k_t 的预测」之差。
    由 prediction_is_content_sum，预测正是历史按 B(k_i,k_t) 的加权和；
    故 ρ_t = 0 当且仅当 v_t 被历史精确重建 —— 即该 token 冗余，可压缩。
    这把「按 ρ 丢/留」的判定变成一个由状态直接给出的连续信号，无需单独掩码。 -/
def residual (f : Nat → Step K V) (t : Nat) : V :=
  (f t).v - (stateFrom B f t) (f t).k

/-- 残差为零 ⇔ 当前值等于历史按键相似度的重建。 -/
theorem residual_zero_iff_reconstructed (f : Nat → Step K V) (t : Nat) :
    residual B f t = 0
    ↔ (f t).v = ∑ i ∈ Finset.range t,
            ((∏ j ∈ Finset.Ico (i + 1) t, (f j).a) * (f i).g * (B (f i).k (f t).k)) • (f i).v := by
  simp [residual, prediction_is_content_sum]
  exact sub_eq_zero

/-! ## 5. 推理迭代：收缩不动点与防塌缩 -/

/-- 推理阶段固定 (a, b) 的仿射迭代：S^{(n+1)} = a·S^{(n)} + b。
    对应 math.md §3.7 / §3.9 的算子骨架 Ψ_h（线性部分 a·id，平移 b 输入依赖）。 -/
def iterAff (a : K) (b : V) : Nat → V → V
  | 0,     x => x
  | n + 1, x => a • (iterAff a b n x) + b

/-- 不动点候选：S* = (1 - a)⁻¹ · b（要求 a ≠ 1，即 1-a 可逆）。 -/
def fixedPoint (a : K) (b : V) (ha : a ≠ 1) : V :=
  (1 - a)⁻¹ • b

/-- 不动点验证：把 S* 代入迭代一次仍得 S*。 -/
theorem fixedPoint_is_fixed (a : K) (b : V) (ha : a ≠ 1) :
    iterAff a b 1 (fixedPoint a b ha) = fixedPoint a b ha := by
  have hinv : (1 - a) ≠ 0 := by
    intro h
    apply ha
    have h1 : (1 : K) = a := by simpa [sub_eq_zero] using h
    exact h1.symm
  simp [iterAff, fixedPoint]
  have h : (1 - a)⁻¹ • b = a • ((1 - a)⁻¹ • b) + b := by
    calc
      (1 - a)⁻¹ • b
          = (a * (1 - a)⁻¹ + 1) • b := by congr; field_simp [hinv] <;> ring
      _ = (a * (1 - a)⁻¹) • b + 1 • b := by simp [add_smul]
      _ = a • ((1 - a)⁻¹ • b) + b := by simp [mul_smul, one_smul]
  exact h.symm

/-- 误差恒等式：第 n 步与不动点的偏差 = a^n · (初值偏差)。
    这是收缩收敛的代数核心：|a|<1 时 a^n → 0 ⇒ 偏差 → 0。 -/
theorem iterAff_error (a : K) (b : V) (ha : a ≠ 1) (n : Nat) (x : V) :
    iterAff a b n x - fixedPoint a b ha
    = a ^ n • (x - fixedPoint a b ha) := by
  induction n with
  | zero =>
    simp [iterAff, fixedPoint]
    <;> abel
  | succ n ih =>
    set fp : V := fixedPoint a b ha with hfp
    have hinv : (1 - a) ≠ 0 := by
      intro h0
      apply ha
      have h1 : (1 : K) = a := by simpa [sub_eq_zero] using h0
      exact h1.symm
    have hb : b = fp - a • fp := by
      have hfp_def : fp = (1 - a)⁻¹ • b := by simp [hfp, fixedPoint]
      rw [hfp_def]
      have hcomm : (1 - a)⁻¹ * a = a * (1 - a)⁻¹ := by ring
      calc
        b = 1 • b := by simp
        _ = ((1 - a)⁻¹ * (1 - a)) • b := by simp [inv_mul_cancel₀ hinv]
        _ = ((1 - a)⁻¹ - (1 - a)⁻¹ * a) • b := by congr; ring
        _ = (1 - a)⁻¹ • b - ((1 - a)⁻¹ * a) • b := by rw [sub_smul]
        _ = (1 - a)⁻¹ • b - a • ((1 - a)⁻¹ • b) := by rw [hcomm, ← mul_smul]
    have hmain : ∀ (y : V), a • y + b - fp = a • (y - fp) := by
      intro y
      have : a • y + b - fp = a • y - a • fp := by rw [hb] <;> abel
      rw [this, smul_sub]
    have hmul : a • (a ^ n • (x - fp)) = (a * a ^ n) • (x - fp) := by
      exact Eq.symm (mul_smul a (a ^ n) (x - fp))
    have hpow : a * a ^ n = a ^ (n + 1) := by
      calc
        a * a ^ n = a ^ n * a := by rw [mul_comm]
        _ = a ^ (n + 1) := by rw [← pow_succ]
    have ih' : iterAff a b n x - fp = a ^ n • (x - fp) := by simpa [hfp] using ih
    have hfinal : iterAff a b (n + 1) x - fp = a ^ (n + 1) • (x - fp) := by
      calc
        iterAff a b (n + 1) x - fp
            = a • (iterAff a b n x) + b - fp := by simp [iterAff]
        _ = a • (iterAff a b n x - fp) := hmain (iterAff a b n x)
        _ = a • (a ^ n • (x - fp)) := by rw [ih']
        _ = (a * a ^ n) • (x - fp) := hmul
        _ = a ^ (n + 1) • (x - fp) := by rw [hpow]
    simpa [hfp] using hfinal

/-- 防塌缩：不同输入 b₁ ≠ b₂ ⇒ 不同不动点。
    不动点由输入 b 唯一确定（S* = (1-a)⁻¹·b），因此不会坍缩到单一状态；
    对应 math.md §3.7.3 的「不动点输入依赖」要求。 -/
theorem no_collapse (a : K) (b₁ b₂ : V) (ha : a ≠ 1)
    (hne : b₁ ≠ b₂) :
    fixedPoint a b₁ ha ≠ fixedPoint a b₂ ha := by
  intro h
  apply hne
  have hinv : (1 - a) ≠ 0 := by
    intro h0
    apply ha
    have h1 : (1 : K) = a := by simpa [sub_eq_zero] using h0
    exact h1.symm
  have : (1 - a) • fixedPoint a b₁ ha = (1 - a) • fixedPoint a b₂ ha := by
    rw [h]
  simpa [fixedPoint, smul_inv_smul₀ hinv] using this

/-- 谱夹层下收敛：a ∈ (δ, 1-γ) ⊂ (0,1) 时 0 ≤ a < 1，故 a^n → 0，
    结合 iterAff_error 得迭代偏差 → 0（收敛到不动点）。 -/
theorem real_pow_tendsto_zero {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) :
    Filter.Tendsto (fun n : ℕ => a ^ n) Filter.atTop (nhds (0 : ℝ)) := by
  exact tendsto_pow_atTop_nhds_zero_of_lt_one ha0 ha1

end AssocMem
