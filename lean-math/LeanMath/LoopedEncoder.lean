/-
§2.10 替代结构「Looped Encoder」的数学核心：
把 E 主干 → Z 空间的前馈链（N 个不同层 B_N ∘ ⋯ ∘ B_1）折叠为
**权重共享的内部循环**（同一层 B 迭代 T 次：B^∘T）。

对应 math.md §2.10：

    前馈基线：  E_ff(x)  = W_out · (B_N ∘ ⋯ ∘ B_1)(embed(x) + pos)
    回环（MR）： h^{(0)} = 0
                 h^{(t)} = B_θ(h^{(t-1)} + e),   e = embed(x) + pos
                 E_loop(x) = W_out · h^{(T)}

线性化骨架（B 的线性部分 M、输入注入 b = M·e）即仿射迭代 h ↦ M·h + b，
与 §3.7.3 PSCO / §2.9 CARO 的仿射不动点迭代同构，可直接复用已验证结论。

本文件机械化以下断言（均无 sorry / 无新增公理）：

1. **表达力包含**：回环实现的函数 = 同深度前馈（所有层取同一个 f）的函数，
   故回环函数族 ⊆ 同深度前馈函数族（对角切片）。形式化：
   `Function.iterate f T = List.foldr (·∘·) id (List.replicate T f)`。
2. **参数容量计数**：单块参数 p，前馈深度 N 用 N·p，回环深度 T 用 p（与 T 无关）；
   同有效深度（T=N）下回环严格更省参数（N>0, p>0）。
3. **MR-Loop 仿射误差恒等式**（线性推广 AssocMem.iterAff_error）：
   若 fp 是不动点（fp = M·fp + b），则第 t 步偏差 = M^t·(初值偏差)。
4. **反输入塌缩**：输入注入下，M 单射 ⇒ 不同输入 e₁≠e₂ ⇒ 不同不动点 fp₁≠fp₂
   （Z 不坍缩到与输入无关的单一状态，守住 §2.2.4 / §2.6 反塌缩契约）。
5. **梯度衰减（优化代价）**：连续线性 M 的算子范数 ‖M‖≤K<1 时，
   ‖M^n v‖ ≤ K^n·‖v‖——第 t 步回传梯度被几何衰减 K^{T-t} 抑制，
   这是权重共享回环的主要优化代价（有效信用分配深度受 K 限制）。

未形式化（设计分析，如实标注，见 math.md §2.10 与 README）：
渐进复杂度 O(·) 常数因子、墙钟时间、DEQ 隐式微分的数值稳定性、softmax 块的
非线性表达力与「同参更深」的泛化收益——这些依赖实测，非 Lean 可机械化对象。
-/

import Mathlib
import LeanMath.Affine
import LeanMath.AssociativeMemory
import LeanMath.Contractive

open scoped BigOperators
open Asymptotics Filter

namespace LoopedEncoder

universe u

/-! ## 1. 表达力包含：回环 = 同深度前馈的权重绑定对角 -/

variable {α : Type u}

/-- 前馈网：层列表的复合。`feedforward [f₁, …, f_N] = f₁ ∘ ⋯ ∘ f_N`（先 f_N 后 f₁）。
    id 是单位元（0 层 = 恒等）。 -/
def feedforward (fs : List (α → α)) : α → α :=
  List.foldr (fun f g => f ∘ g) id fs

/-- 回环网：单层 f 迭代 T 次（`Function.iterate`）。 -/
def looped (f : α → α) (T : Nat) : α → α := f^[T]

/-- 核心定理（表达力包含）：回环 f^[T] 等于「把 f 复制 T 份作前馈层」的复合。
    即每个回环网实现的函数，都能被一个**同深度、权重全部绑定为 f** 的前馈网实现。
    因此回环函数族是前馈函数族的子集（对角切片）；反向不成立（前馈可让各层相异）。 -/
theorem looped_eq_feedforward_tied (f : α → α) (T : Nat) :
    looped f T = feedforward (List.replicate T f) := by
  funext x
  induction T with
  | zero =>
      simp [looped, feedforward]
  | succ T ih =>
      have h₁ : looped f (T + 1) x = f (looped f T x) := by
        simp [looped, Function.iterate_succ_apply']
      have h₂ : feedforward (List.replicate (T + 1) f) x
                 = f (feedforward (List.replicate T f) x) := by
        simp [feedforward]; rfl
      rw [h₁, h₂, ih]

/-- 推论：任意回环函数都被某个长度为 T 的前馈层列表实现。 -/
theorem looped_realized_by_feedforward (f : α → α) (T : Nat) :
    ∃ (fs : List (α → α)), fs.length = T ∧ feedforward fs = looped f T :=
  ⟨List.replicate T f, by simp, (looped_eq_feedforward_tied f T).symm⟩

/-! ## 2. 参数容量计数 -/

/-- 前馈深度 N、单块参数 p：总参数 N·p（层间不共享）。 -/
def param_ff (N p : Nat) : Nat := N * p

/-- 回环深度 T、单块参数 p：总参数 p（层间共享，与迭代次数 T 无关）。 -/
def param_loop (p : Nat) : Nat := p

/-- 同有效深度（N ≥ 1）下回环不超过前馈的参数量：p ≤ N·p。 -/
theorem loop_uses_no_more_params (N p : Nat) (hN : 1 ≤ N) :
    param_loop p ≤ param_ff N p := by
  simp [param_loop, param_ff]
  nlinarith

/-- N ≥ 2 且 p > 0 时回环严格更省参数（容量代价的核心收益）：p < N·p。
    N = 1 时两者相等（单层无共享收益），故严格收益需至少 2 层。 -/
theorem loop_uses_fewer_params (N p : Nat) (hN : 1 < N) (hp : 0 < p) :
    param_loop p < param_ff N p := by
  simp [param_loop, param_ff]
  nlinarith

/-! ## 3. MR-Loop 仿射不动点：误差恒等式（线性推广） -/

variable {K : Type*} [Field K]
variable {V : Type u} [AddCommGroup V] [Module K V]

/-- MR-Loop（Fan et al. 2025）回环编码器的线性骨架：
    h^{(0)} = 0,  h^{(t)} = M(h^{(t-1)} + e) = M·h^{(t-1)} + M·e。
    每步把输入嵌入 e 重新注入（input injection），避免纯收缩环遗忘输入。
    等价于 `PSCO.iterate M b`，b = M·e。 -/
def mrLoop (M : V →ₗ[K] V) (e : V) : Nat → V → V :=
  PSCO.iterate M (M e)

/-- 线性仿射迭代的误差恒等式（推广 AssocMem.iterAff_error 到一般线性 M）：
    若 fp 是不动点 fp = M·fp + b，则第 t 步与不动点的偏差 = M^t·(初值偏差)。
    当 M 收缩（谱半径 < 1）时 M^t → 0，偏差 → 0（收敛到不动点 = DEQ 极限）。 -/
theorem affine_iter_error (M : V →ₗ[K] V) (b : V) (t : Nat) (x fp : V)
    (hfp : fp = M fp + b) :
    PSCO.iterate M b t x - fp = (PSCO.powL M t) (x - fp) := by
  induction t with
  | zero =>
      simp [PSCO.iterate, PSCO.powL]
  | succ t ih =>
      have h₁ : PSCO.iterate M b (t + 1) x = M (PSCO.iterate M b t x) + b := by
        simp [PSCO.iterate]
      have h₂ : M (PSCO.iterate M b t x) + b - fp = M (PSCO.iterate M b t x - fp) := by
        have hbm : b + M fp = fp := by
          calc b + M fp = M fp + b := by abel
                     _ = fp := hfp.symm
        have hsub : b - fp + M fp = 0 := by
          calc b - fp + M fp = (b + M fp) - fp := by abel
                          _ = fp - fp := by rw [hbm]
                          _ = 0 := by simp
        have hneg : b - fp = -M fp := eq_neg_of_add_eq_zero_left hsub
        calc
          M (PSCO.iterate M b t x) + b - fp
              = M (PSCO.iterate M b t x) + (b - fp) := by abel
          _ = M (PSCO.iterate M b t x) + (-M fp) := by rw [hneg]
          _ = M (PSCO.iterate M b t x) - M fp := by abel
          _ = M (PSCO.iterate M b t x - fp) := by rw [map_sub]
      rw [h₁, h₂, ih]; rfl

/-- 推论（MR-Loop 误差）：MR-Loop 第 t 步与不动点 fp 的偏差 = M^t·(初值偏差)。
    取初值 h^{(0)} = 0，则偏差 = -M^t·fp，t→∞ 时 → 0（M 收缩）。 -/
theorem mrLoop_error (M : V →ₗ[K] V) (e : V) (t : Nat) (fp : V)
    (hfp : fp = M fp + M e) :
    mrLoop M e t 0 - fp = (PSCO.powL M t) (0 - fp) :=
  affine_iter_error M (M e) t 0 fp hfp

/-! ## 4. 反输入塌缩：输入注入守住 Z 对输入的依赖 -/

/-- 反塌缩（输入依赖）：M 单射 ⇒ 不同输入 e₁ ≠ e₂ ⇒ 不同不动点 fp₁ ≠ fp₂。
    这保证回环编码器的输出 Z = W_out·fp 不会坍缩到与输入无关的单一状态——
    守住 §2.2.4（重建损失）与 §2.6（Z 保留可预测结构）的反塌缩契约。
    关键：靠**输入注入** b = M·e 而非纯环 h ↦ M·h；后者不动点恒为 0、与输入无关。 -/
theorem mrLoop_no_input_collapse (M : V →ₗ[K] V) (hM : Function.Injective M)
    (e₁ e₂ : V) (fp₁ fp₂ : V)
    (h1 : fp₁ = M fp₁ + M e₁) (h2 : fp₂ = M fp₂ + M e₂)
    (hne : e₁ ≠ e₂) : fp₁ ≠ fp₂ := by
  intro heq
  have h₁ : M e₁ + M fp₁ = fp₁ := by simpa [add_comm] using h1.symm
  have h₂ : M e₂ + M fp₂ = fp₂ := by simpa [add_comm] using h2.symm
  have hMe1 : M e₁ = fp₁ - M fp₁ := by
    rw [eq_sub_iff_add_eq]; exact h₁
  have hMe2 : M e₂ = fp₂ - M fp₂ := by
    rw [eq_sub_iff_add_eq]; exact h₂
  have hMe : M e₁ = M e₂ := by
    rw [hMe1, hMe2, heq]
  exact hne (hM hMe)

/-! ## 5. 梯度衰减：权重共享回环的优化代价 -/

/-- 梯度衰减（核心优化代价）：连续线性 M 逐次应用 le_opNorm，得 ‖M^n v‖ ≤ ‖M‖^n·‖v‖。
    即第 n 步回传信号被 ‖M‖^n 几何抑制（归纳证明，仅依赖 le_opNorm 次可乘性）。 -/
theorem gradient_decay {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (M : E →L[ℝ] E) (n : Nat) (v : E) :
    ‖(M ^ n) v‖ ≤ ‖M‖ ^ n * ‖v‖ := by
  induction n with
  | zero =>
      simp [one_mul]
  | succ n ih =>
      have hstep : (M ^ (n + 1)) v = M ((M ^ n) v) := by
        rw [pow_succ']; simp
      calc
        ‖(M ^ (n + 1)) v‖ = ‖M ((M ^ n) v)‖ := by rw [hstep]
        _ ≤ ‖M‖ * ‖(M ^ n) v‖ := ContinuousLinearMap.le_opNorm M ((M ^ n) v)
        _ ≤ ‖M‖ * (‖M‖ ^ n * ‖v‖) := by gcongr
        _ = ‖M‖ ^ (n + 1) * ‖v‖ := by
          simp [pow_succ]; ring

/-- 推论：若 ‖M‖ ≤ K < 1，则第 t 步回传信号被 K^{T-t} 几何抑制——
    即权重共享回环的有效信用分配深度受收缩率 K 限制（K 越小，越早步的梯度越难传回）。
    这是相对前馈（各层雅可比独立、可逐层放大/保持）的主要优化代价。 -/
theorem gradient_decay_with_K {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (M : E →L[ℝ] E) (K : ℝ) (hK : ‖M‖ ≤ K) (n : Nat) (v : E) :
    ‖(M ^ n) v‖ ≤ K ^ n * ‖v‖ := by
  have h₂ : ‖M‖ ^ n ≤ K ^ n := by
    gcongr
  calc
    ‖(M ^ n) v‖ ≤ ‖M‖ ^ n * ‖v‖ := gradient_decay M n v
    _ ≤ K ^ n * ‖v‖ := by
      gcongr

/-! ## 6. 复杂度与显存：Asymptotics 形式化

§2.10.3 对比表中「激活显存从 $O(T)$ 降到 $O(1)$」原标为设计断言。本节用 mathlib 的
`Asymptotics.IsBigO` 把 $O(\cdot)$ 落地为定理，并给出 DEQ 极限下「只需留存不动点」的收敛证明。

抽象模型（不绑定具体硬件 / 非线性块）：
- `trajectory f h0 T` = 朴素反传必须全量留存的激活列表 `[h^(0), …, h^(T)]`，长度 `T + 1`；
- `memoryBaseline` / `memoryCheckpoint` / `memoryDEQ` = 三种方案需留存的激活数；
- `evalCount` / `evalCountCheckpoint` = 前向 / 前向+重算 的 f-求值次数。

机械化结论（均无 `sorry` / 无新增公理）：

1. `length_trajectory`：留存轨迹长度精确等于 `T + 1`。
2. `baseline_memory_isBigO_T`：朴素激活显存 $= O(T)$（`IsBigO`，常数 2，阈值 1）。
3. `checkpoint_memory_isBigO_one` / `deq_memory_isBigO_one`：检查点 / DEQ 显存 $= O(1)$。
4. `baseline_memory_unbounded`：朴素显存随 $T\to\infty$ 无界（$\to\mathtt{atTop}$），
   故**不**是 $O(1)$——与检查点/DEQ 的对比是严格的（线性 vs 常数）。
5. `eval_isBigO_T` / `checkpoint_eval_isBigO_T`：前向 $=O(T)$；检查点总代价 $2T=O(T)$
   （用常数因子 2 的重算换显存，渐进算力持平）。
6. `trajectory_determined`：整条轨迹由 $(f, h_0, T)$ 唯一确定 ⇒ 不携带超出该三元组的信息，
   故只存检查点 $h_0$ 即可反向重建任意 $h^{(t)}$——**检查点不丢信息的代数根因**。
7. `mrLoop_tendsTo_fixedPoint`：`‖M‖ < 1` 且 `fp` 是 MR-Loop 不动点时，迭代态按范数
   $\to fp$（拼接 `mrLoop_error` + `gradient_decay` + `AssocMem.real_pow_tendsto_zero`，
   `squeeze_zero`）。极限输出只依赖 $fp$（单个向量），无需留存轨迹——**DEQ $O(1)$ 激活的数学根因**。

仍未形式化（诚实边界，见 math.md §2.10.7 与 README 问题 11）：常数因子的真实墙钟/访存、
kernel 融合、非线性块表达力、DEQ 数值求解精度——这些是工程/经验断言。 -/

/-- 回环迭代产生的完整激活轨迹 `[h^(0), h^(1), …, h^(T)]`，朴素反传必须全量留存。 -/
def trajectory (f : α → α) (h0 : α) (T : Nat) : List α :=
  (List.range' 0 (T + 1)).map (fun t => f^[t] h0)

/-- 留存轨迹长度精确等于 `T + 1`：朴素反传的激活数随迭代深度线性增长。 -/
theorem length_trajectory (f : α → α) (h0 : α) (T : Nat) :
    (trajectory f h0 T).length = T + 1 := by
  simp [trajectory]

/-- 朴素（无检查点）反传需留存的激活数 = 轨迹长度 = T + 1。 -/
def memoryBaseline (T : Nat) : Nat := T + 1
/-- 梯度检查点：只留存一个检查点 h0，其余在反向时重算 ⇒ 留存激活数恒为 1。 -/
def memoryCheckpoint (_T : Nat) : Nat := 1
/-- DEQ：只留存不动点 h* ⇒ 留存激活数恒为 1。 -/
def memoryDEQ (_T : Nat) : Nat := 1

/-- 前向 f-求值次数 = T。 -/
def evalCount (T : Nat) : Nat := T
/-- 检查点方案总 f-求值 = 前向 T + 反向重算 T = 2T（常数因子 2 的算力换显存）。 -/
def evalCountCheckpoint (T : Nat) : Nat := 2 * T

/-- 朴素激活显存 $= O(T)$：`T + 1 ≤ 2T`（`T ≥ 1`），`IsBigO` 常数 2、阈值 1。 -/
theorem baseline_memory_isBigO_T :
    (fun T : Nat => (memoryBaseline T : ℝ)) =O[Filter.atTop] (fun T : Nat => (T : ℝ)) := by
  rw [isBigO_iff_isBigOWith]
  refine ⟨2, ?_⟩
  rw [isBigOWith_iff, eventually_atTop]
  refine ⟨1, fun n hn => ?_⟩
  have hle1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hpos1 : 0 ≤ (n : ℝ) + 1 := by linarith
  have hmain : (n : ℝ) + 1 ≤ 2 * (n : ℝ) := by linarith
  simp only [memoryBaseline, Nat.cast_add, Nat.cast_one]
  rw [Real.norm_of_nonneg hpos1, Real.norm_natCast n]
  exact hmain

/-- 梯度检查点显存 $= O(1)$：留存激活数恒为 1，与迭代次数 T 无关。 -/
theorem checkpoint_memory_isBigO_one :
    (fun T : Nat => (memoryCheckpoint T : ℝ)) =O[Filter.atTop] (fun _ : Nat => (1 : ℝ)) := by
  simpa [memoryCheckpoint] using isBigO_refl (fun _ : Nat => (1 : ℝ)) Filter.atTop

/-- DEQ 显存 $= O(1)$：极限只留存不动点 fp（单个向量），与迭代次数 T 无关。 -/
theorem deq_memory_isBigO_one :
    (fun T : Nat => (memoryDEQ T : ℝ)) =O[Filter.atTop] (fun _ : Nat => (1 : ℝ)) := by
  simpa [memoryDEQ] using isBigO_refl (fun _ : Nat => (1 : ℝ)) Filter.atTop

/-- 朴素显存随 $T\to\infty$ 无界（$\to\mathtt{atTop}$），故**不**是 $O(1)$——
    线性 vs 常数的对比是严格的，不是同阶的记号巧合。 -/
theorem baseline_memory_unbounded :
    Filter.Tendsto (fun T : Nat => (memoryBaseline T : ℝ)) Filter.atTop Filter.atTop := by
  have hbase : Filter.Tendsto (fun n : Nat => (n : ℝ)) Filter.atTop Filter.atTop :=
    tendsto_natCast_atTop_atTop
  have hshift : Filter.Tendsto (fun n : Nat => (n : ℝ) + 1) Filter.atTop Filter.atTop :=
    tendsto_atTop_add_const_right Filter.atTop (1 : ℝ) hbase
  have heq : (fun n : Nat => ((n + 1 : Nat) : ℝ)) = fun n : Nat => (n : ℝ) + 1 := by
    funext n; simp [Nat.cast_add]
  simpa [memoryBaseline, heq] using hshift

/-- 前向算力 $= O(T)$：每层恰求值一次 f，共 T 次。 -/
theorem eval_isBigO_T :
    (fun T : Nat => (evalCount T : ℝ)) =O[Filter.atTop] (fun T : Nat => (T : ℝ)) := by
  simpa [evalCount] using isBigO_refl (fun T : Nat => (T : ℝ)) Filter.atTop

/-- 检查点总算力 $= O(T)$：前向 T + 反向重算 T = 2T，渐进与朴素持平（仅常数因子 2）。 -/
theorem checkpoint_eval_isBigO_T :
    (fun T : Nat => (evalCountCheckpoint T : ℝ)) =O[Filter.atTop] (fun T : Nat => (T : ℝ)) := by
  rw [isBigO_iff_isBigOWith]
  refine ⟨2, ?_⟩
  rw [isBigOWith_iff, eventually_atTop]
  refine ⟨0, fun n _ => ?_⟩
  have hpos2 : 0 ≤ 2 * (n : ℝ) := by positivity
  simp only [evalCountCheckpoint, Nat.cast_mul, Nat.cast_two]
  rw [Real.norm_of_nonneg hpos2, Real.norm_natCast n]

/-- 轨迹由 `(f, h0, T)` 唯一确定 ⇒ 全量轨迹不携带超出该三元组的信息，
    故不必存储：只存检查点 `h0` 即可反向重建任意 `h^(t)`（检查点合法性的代数根因）。 -/
theorem trajectory_determined (f g : α → α) (h0 h0' : α) (T T' : Nat)
    (hfg : f = g) (hh : h0 = h0') (hT : T = T') :
    trajectory f h0 T = trajectory g h0' T' := by
  subst hfg hh hT
  rfl

/-- 连续线性幂的线性映射骨架连接：`PSCO.powL (M.toLinearMap) t v = (M ^ t) v`，
    把 §3 的线性骨架 (`PSCO.powL`) 与赋范空间的连续线性幂 (`M ^ t`) 对接。 -/
lemma powL_toLinearMap {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (M : E →L[ℝ] E) :
    ∀ (t : Nat) (v : E), PSCO.powL (M.toLinearMap) t v = (M ^ t) v := by
  intro t
  induction t with
  | zero =>
    intro v
    simp [PSCO.powL]
  | succ t ih =>
    intro v
    have h₁ : (PSCO.powL (M.toLinearMap) (t + 1)) v
        = M.toLinearMap ((PSCO.powL (M.toLinearMap) t) v) := by
      simp [PSCO.powL]
    rw [h₁, ih v]
    have h₂ : M.toLinearMap ((M ^ t) v) = M ((M ^ t) v) := by simp
    rw [h₂]
    have h₃ : (M ^ (t + 1)) v = M ((M ^ t) v) := by
      rw [pow_succ']; rfl
    exact h₃.symm

/-- DEQ 极限：`‖M‖ < 1` 且 `fp` 是 MR-Loop 不动点时，迭代态按范数收敛到 `fp`。
    拼接 `mrLoop_error`（误差恒等式）+ `gradient_decay`（算子范数幂次界）+
    `AssocMem.real_pow_tendsto_zero`（$K^t\to0$），经 `squeeze_zero` 得证。
    故有效极限输出只依赖 `fp`（单个向量），无需留存整条轨迹——DEQ $O(1)$ 激活的数学根因。 -/
theorem mrLoop_tendsTo_fixedPoint {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (M : E →L[ℝ] E) (e : E) (fp : E)
    (hfp : fp = M fp + M e) (hK : ‖M‖ < 1) :
    Filter.Tendsto (fun t : Nat => mrLoop (M.toLinearMap) e t 0) Filter.atTop (nhds fp) := by
  have hnonneg : 0 ≤ ‖M‖ := norm_nonneg _
  have htz : Filter.Tendsto (fun t : Nat => ‖M‖ ^ t) Filter.atTop (nhds (0 : ℝ)) :=
    AssocMem.real_pow_tendsto_zero hnonneg hK
  have htz2 : Filter.Tendsto (fun t : Nat => ‖M‖ ^ t * ‖fp‖) Filter.atTop (nhds (0 : ℝ)) := by
    have h := htz.const_mul ‖fp‖
    simpa [mul_comm] using h
  have hle : ∀ t : Nat, ‖mrLoop (M.toLinearMap) e t 0 - fp‖ ≤ ‖M‖ ^ t * ‖fp‖ := by
    intro t
    have h1 : mrLoop (M.toLinearMap) e t 0 - fp = PSCO.powL (M.toLinearMap) t (0 - fp) :=
      mrLoop_error (M.toLinearMap) e t fp hfp
    rw [h1, powL_toLinearMap M t (0 - fp), zero_sub]
    simpa [map_neg, norm_neg] using gradient_decay M t fp
  have hle2 : ∀ t : Nat, ‖‖mrLoop (M.toLinearMap) e t 0 - fp‖‖ ≤ ‖M‖ ^ t * ‖fp‖ := by
    intro t
    rw [Real.norm_of_nonneg (norm_nonneg (mrLoop (M.toLinearMap) e t 0 - fp))]
    exact hle t
  have hnorm : Filter.Tendsto (fun t : Nat => ‖mrLoop (M.toLinearMap) e t 0 - fp‖)
      Filter.atTop (nhds (0 : ℝ)) := squeeze_zero_norm hle2 htz2
  have hg : Filter.Tendsto (fun t : Nat => mrLoop (M.toLinearMap) e t 0 - fp)
      Filter.atTop (nhds (0 : E)) := tendsto_zero_iff_norm_tendsto_zero.2 hnorm
  have hfp' : Filter.Tendsto (fun _ : Nat => fp) Filter.atTop (nhds fp) := tendsto_const_nhds
  simpa [sub_add_cancel] using hg.add hfp'

end LoopedEncoder
