# 多阶段潜空间推理多模态模型：端到端架构与数学基础（含核心命题验证）

> 本文按「端到端四阶段流水线」组织：**阶段一编码（§2）→ 阶段二对齐与推理（§3）→ 阶段三解码（§4）→ 阶段四自反馈修正（§5）**。§2.2（JEPA 表征学习）的公式与当前实现（`model/`、`training/` 各模块）对齐；§4.1 Flow Matching 速度场 $v_\psi$ 的欧氏 OT 基线已实现（[flow.py](file:///home/simon-ma/JEPA/model/flow.py) 的 `VelocityField`），测地整流 / RFM（§4.1.6–4.1.8）仍为规划；§2.1、§2.7、§3 引入的 $W, I, V, Y, \phi, M, \Omega, \Psi, E^I, E^V, E^A, R_I, R_V, R_A, K, e_t$ 是概念量，对应 [`plan`](file:///home/simon-ma/JEPA/.trae/plan_mosaic_text_unified_lossy_channel.md) 的 P1–P8，本脚本暂不实现。

---

## 0. 符号表与源码映射

| 数学符号 | 含义 | 源码对应 |
|---|---|---|
| $x, x_c, x_y$ | 干净图文对、掩码上下文、干净目标 | `(img, txt)`；掩码见 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `mask_patches`/`mask_text` |
| $E_\theta$ | 在线编码器（接收梯度，图像/文本各一） | `model.image_online`/`model.text_online`，[jepa.py](file:///home/simon-ma/JEPA/model/jepa.py) 的 `MultimodalJEPA` |
| $E_\xi$ | 目标编码器（仅 EMA 更新，图像/文本各一） | `model.image_target`/`model.text_target`，[jepa.py](file:///home/simon-ma/JEPA/model/jepa.py) 的 `MultimodalJEPA` |
| $P_\phi$ | JEPA 潜变量预测器（图像/文本各一） | `model.image_predictor`/`model.text_predictor`，[representation.py](file:///home/simon-ma/JEPA/model/representation.py) 的 `LatentPredictor` |
| $v_\psi$ | Flow Matching 速度场预测器（欧氏 OT 直线基线） | [flow.py](file:///home/simon-ma/JEPA/model/flow.py) 的 `VelocityField`（已实现；测地整流 / RFM 见 §4.1.6–4.1.8） |
| $D_\eta$ | 潜变量→各模态解码器 | `model.image_decoder`/`model.text_decoder`，[decoder.py](file:///home/simon-ma/JEPA/model/decoder.py) 的 `ImageDecoder`/`TextDecoder` |
| $D$ | 通用解码器记号（$R_I/g/R_V/R_A$ 的并称，§3.9），把 $Z$ 渲染回模态，对重建/生成无差别 | 概念量（§3.9） |
| $H(X)$、$h(X)$ | Shannon 熵 $H(X)=-\sum_x p_x\log_2 p_x$（离散）；微分熵 $h(X)=-\int p_X\log_2 p_X$（连续） | §2.3；此 $H$/$h$ 为熵，区别于 §3.9 的隐含模式编码 $H$、状态 $h$ |
| $I(X;Y)$ | 互信息 $=H(X)+H(Y)-H(X,Y)=H(X)-H(X\mid Y)$ | §2.3–2.5；写作 $I(\cdot;\cdot)$，区别于图像观测 $I$ |
| $D_{\mathrm{KL}}(p\,\|\,q)$ | KL 散度（相对熵）$=\sum_x p_x\log_2\frac{p_x}{q_x}$（离散），$\int p\log_2\frac p q$（连续） | §3.5/§4/§5；$p\ll q$ 时有限，否则 $+\infty$ |
| $\mu$ | EMA 衰减系数 | `ema_decay=0.99`，[config.py](file:///home/simon-ma/JEPA/model/config.py) 的 `MJEPAConfig` |
| $L$ | 潜序列长度（硬对齐要求两模态同长） | `text_len = n_patches = (image_size/patch_size)^2`，随规模伸缩；冒烟 tiny/small 为 16 |
| $d$ | 潜空间维度 | `dim`，随规模伸缩；冒烟 tiny=64/small=128，大规模按 §2.2.7 的 $r=8$ 解耦 |
| $\mathcal{V}$ | 输出词表 | 冒烟用字符级 `CharTokenizer`（[synthetic.py](file:///home/simon-ma/JEPA/data_prepare/synthetic.py)）；大模型接现成 tokenizer（LLaDA 等，见 §2.2.7） |
| $z_0, z_1, z_t$ | 噪声、干净潜变量、时刻 $t$ 插值 | `noise`, `target`, `z_t` |
| $u_t$ | 真实速度场 | `velocity_target` |
| $W$ | 世界状态 | 概念量，本脚本不实现 |
| $I, V$ | 图像 / 视频连续观测 | 概念量，本脚本不实现 |
| $Y$ | 跨模态共享语义因子 | 概念量，可由跨模态对齐或线性探针估计 |
| $g$ | 文本量化器 $Z \mapsto \mathcal{V}^L$ | `generate` 的 argmax 解码，[decoder.py](file:///home/simon-ma/JEPA/model/decoder.py) 的 `TextDecoder` |
| $\mathbf G$ | 解码器诱导的拉回度量 $\mathbf G=J_D^\top J_D$（黎曼度量，区别于量化器 $g$） | 概念量，§2.1.1 / §2.2.6 / §3.7.1 |
| $\mathrm{verify}$ | 客观正确性判定器 $\Omega\to\{0,1\}$（硬反馈的决策过程形式） | 概念量（§4.1.8 / §5 硬反馈），外部决策程序，未内置 |
| $\mathcal{F}_t$ | 文本 $t$ 对应的潜空间纤维 | 概念量，$\mathcal{F}_t = g^{-1}(t)$ |
| $\phi, M$ | 马赛克算子与马赛克观测 $M=\phi(I)$ | 概念量（plan P2），下采样 + 分块平均，未实现 |
| $\Psi_h, H$ | 状态依赖算子、隐含模式编码 $Z'=\Psi_h(Z),\ h=H(x)$ | [operator.py](file:///home/simon-ma/JEPA/model/operator.py) 的 `PsiLayer`（已实现，含 `infer_h`） |
| $h$ | 隐含模式状态（由 $H$ 自推断，含描述/推理倾向） | [operator.py](file:///home/simon-ma/JEPA/model/operator.py) 的 `PsiLayer.infer_h`（已实现） |
| $\mathrm{id}$ | 恒等算子（$Z\to Z$ 的单位元），重建 = $D(\mathrm{id}(Z))$（式 29） | 概念量（§3.9） |
| $\Omega$ | 算子空间（$Z$ 上的自映射半群，$\mathrm{id}\in\Omega$、复合封闭，式 30） | 概念量（§3.5 / §3.9），未实现 |
| $\omega$ | 算子参数 / 指令的变换分量（$\omega\in\Omega$，$Z'=\Psi(Z,\omega)$） | 概念量（plan P5），未实现 |
| $E^I, E^V, E^A$；$R_I, R_V, R_A$ | 各模态编译器 / 解释器 | 概念量（plan P6/P7），未实现 |
| $K, e_t$ | 观测器增益、创新序列 | 概念量（plan P8），未实现 |

---

## 1.0 Lean 形式化对照（成熟结论，后续思考可略过）

本节把文中核心命题 / 公式与 [`lean-math/LeanMath`](file:///home/simon-ma/JEPA/lean-math/LeanMath) 的形式化证明一一对照。凡在此表中的结论均已由 `lake build`（Lean 4.36.0 + mathlib）编译验证、无 `sorry`/`axiom`，视为**已成熟**：后续设计 / 推理直接引用这些结论，**不再重新推导**。完整推导与「发现的问题」见 [lean-math/README.md](file:///home/simon-ma/JEPA/lean-math/README.md)。

| 命题 / 公式 | Lean 证明（文件 : 定理） | 状态 |
|---|---|---|
| 式 (1)–(2) 最大熵 $H(T)\le L\log_2\|\mathcal V\|$ | `Entropy.lean` : `entropy_le_log_card` | ✅ 已证 |
| 式 (3) 确定性信道 $I(Z;T)=H(T)$ | `InfoTheory.lean` : `mutualInformation_eq_entropy_of_deterministic` | ✅ 已证 |
| 式 (15) 链式法则 $H(T)=I(T;Y)+H(T\|Y)$ | `InfoTheory.lean` : `entropy_marginalX_eq_mutualInformation_add_condEntropyYX` | ✅ 已证 |
| 命题 1（有损性，离散核心） | `Lossy.lean` : `exists_collision_of_card_lt` | ✅ 单向鸽笼；「任意分辨率」措辞已收紧（README 问题 5） |
| 命题 2（失真地板，离散核心） | `Lossy.lean` : `decoder_cannot_invert` | ✅ 已证 |
| 命题 10（软反馈创新恒非零） | `Lossy.lean` : `non_injective_implies_feedback_nonzero` | ✅ 已证 |
| §3.7 结论 = 变换链不动点 | `FixedPoint.lean` : `stabilized_is_fixed_point` / `read_only_fixes_all` | ✅ 已证 |
| §3.7.1 收缩复合 + Neumann 级数 | `Contractive.lean` : `contraction_comp` / `geom_sum_one_sub` | ✅ 已证 |
| §3.7.3 仿射幺半群 + 线性递推闭式 | `Affine.lean` : `comp_assoc` / `iterate_eq_closedForm` | ✅ 已证 |
| §2.9 CARO：更新是仿射 ⇒ 复用结合幺半群 ⇒ 并行扫描合法 | `AssociativeMemory.lean` : `update_is_affine` | ✅ 已证 |
| §2.9 (CARO-1) 内容可寻址读出恒等式 $S_T q=\sum_{i<T}(\prod a)g_i B(k_i,q)v_i$ | `AssociativeMemory.lean` : `stateFrom_readout` | ✅ 已证 |
| §2.9 (CARO-2) 残差 $\rho=0\iff v_t$ 被历史精确重建 | `AssociativeMemory.lean` : `prediction_is_content_sum` / `residual_zero_iff_reconstructed` | ✅ 已证 |
| §2.9 (CARO-4) 不动点 $S^*=(1-a)^{-1}b$ + 误差恒等式 + $a^n\to0$ 收敛 | `AssociativeMemory.lean` : `fixedPoint_is_fixed` / `iterAff_error` / `real_pow_tendsto_zero` | ✅ 已证 |
| §2.9 反塌缩：不同输入 $b$ ⇒ 不同不动点（输入依赖） | `AssociativeMemory.lean` : `no_collapse` | ✅ 已证 |
| §2.10 (LE-8) 回环 = 同深度前馈的权重绑定特例 ⇒ 表达力包含 (LE-9) | `LoopedEncoder.lean` : `looped_eq_feedforward_tied` / `looped_realized_by_feedforward` | ✅ 已证 |
| §2.10 参数量：同有效深度下回环 $\le$ 前馈；$N\ge2,p>0$ 时严格省 | `LoopedEncoder.lean` : `loop_uses_no_more_params` / `loop_uses_fewer_params` | ✅ 已证 |
| §2.10 (LE-5) 仿射迭代误差恒等式（线性推广 AssocMem） | `LoopedEncoder.lean` : `affine_iter_error` / `mrLoop_error` | ✅ 已证 |
| §2.10 反输入塌缩：$M$ 单射 $\Rightarrow e_1\neq e_2\Rightarrow$ 不动点不同 | `LoopedEncoder.lean` : `mrLoop_no_input_collapse` | ✅ 已证 |
| §2.10 (LE-7) 梯度衰减 $\|M^n v\|\le\|M\|^n\|v\|\le K^n\|v\|$ | `LoopedEncoder.lean` : `gradient_decay` / `gradient_decay_with_K` | ✅ 已证 |
| §2.10 留存激活轨迹长度 $=T+1$（精确，朴素反传线性增长） | `LoopedEncoder.lean` : `length_trajectory` | ✅ 已证 |
| §2.10 朴素激活显存 $=O(T)$；检查点 / DEQ 显存 $=O(1)$（`IsBigO`） | `LoopedEncoder.lean` : `baseline_memory_isBigO_T` / `checkpoint_memory_isBigO_one` / `deq_memory_isBigO_one` | ✅ 已证 |
| §2.10 朴素显存无界（$\to\mathtt{atTop}$）⇒ 非 $O(1)$，线性 vs 常数对比严格 | `LoopedEncoder.lean` : `baseline_memory_unbounded` | ✅ 已证 |
| §2.10 前向算力 $=O(T)$；检查点总 $2T=O(T)$（常数 2 重算换显存，渐进持平） | `LoopedEncoder.lean` : `eval_isBigO_T` / `checkpoint_eval_isBigO_T` | ✅ 已证 |
| §2.10 轨迹由 $(f,h_0,T)$ 唯一确定 ⇒ 检查点不丢信息（代数根因） | `LoopedEncoder.lean` : `trajectory_determined` | ✅ 已证 |
| §2.10 DEQ 极限：$\|M\|<1$ 时迭代态按范数 $\to fp$（极限只需留存不动点） | `LoopedEncoder.lean` : `mrLoop_tendsTo_fixedPoint` | ✅ 已证 |
| §3.10 (LZR-6) 表达力三层包含：全绑定 $\subseteq$ 状态依赖回环 $\subseteq$ 全独立前馈 | `LoopedZReasoner.lean` : `tied_is_sdl` / `sdl_realized_by_feedforward` | ✅ 已证 |
| §3.10 参数量：$p_\Psi+p_h\le Np_\Psi$（$p_h\le(N-1)p_\Psi$）⇒ 回环省参 | `LoopedZReasoner.lean` : `param_loop_le_free` / `param_loop_lt_free` | ✅ 已证 |
| §3.10 反算子塌缩（模式分离）：固定 $M$ 下 $b_1\neq b_2\Rightarrow$ 不动点不同 | `LoopedZReasoner.lean` : `fixedPoint_diff_b_diff_fp` | ✅ 已证 |
| §3.10 (LZR-5) Neumann 级数极限 $|a|<1\Rightarrow\sum_{i<k}a^i\to(1-a)^{-1}$（BPTT vs DEQ，式 34） | `LoopedZReasoner.lean` : `geometric_series_limit` | ✅ 已证 |
| §3.10 收缩约束代价：$K>1$ 时膨胀算子迭代 $\|K^n x_0\|\to\infty$（不动点不可达） | `LoopedZReasoner.lean` : `expanding_diverges` | ✅ 已证 |
| §3.10 串行深度：扫描 `scanDepth(2T)=scanDepth(T)+1`（对数）vs 展开线性；`scanDepth≤unrolledDepth` | `LoopedZReasoner.lean` : `scanDepth_double` / `scanDepth_le_unrolled` / `scanDepth_isBigO_unrolled` | ✅ 已证 |
| §3.10 DEQ 极限：$\|M\|<1$ 时从任意初值迭代 $\to$ 不动点（推广 §2.10，初值任意） | `LoopedZReasoner.lean` : `reasoner_tendsTo_fixedPoint` | ✅ 已证 |
| §2.10 非线性块表达力 / 「同参更深」泛化 / DEQ 数值稳定 / 墙钟访存 | — | ⚠️ 设计分析，非 Lean 可形式化，见 §2.10.7 与 README |
| 命题 11（重建 $L_0$ vs 生成 $L_{\ge1}$） | `Semigroup.lean` : `generation_iff_not_reconstruction` / `generation_iff_moves_some_point` | ✅ 已证 |
| 命题 7 / 8 / 9（设计 / 诊断断言） | — | ⚠️ 非可形式化定理，见 README 问题 7 |
| §3.7.3 谱夹层 vs §3.7.1 $\rho<1$ | `Contractive.lean` / `Affine.lean` | ⚠️ 谱夹层**强于** $\rho<1$，见 README 问题 6 |
| §2.9 复杂度 $O(\cdot)$ 表 / 「超越 Mamba」经验断言 | — | ⚠️ 设计分析，非 Lean 可形式化（复杂度未机械化），见 §2.9.6 与 README 问题 10 |
| §3.4.2 引理 1 / 定理 1：仿射回归最优解追踪目标；目标漂移 ⟹ 最优解非 Cauchy 不收敛 | `Curriculum.lean` : `minimizer_tracks_target` / `minimizer_drift_lower_bound` / `minimizer_not_cauchy_of_target_drifts` / `not_cauchy_of_step_ge_eps` | ✅ 已证 |
| §3.4.2 实例化：$\Psi$ 改端 / Flow / $\rho$ 记忆的回归目标追踪上游表征 | `Curriculum.lean` : `psi_change_side_tracks_target` / `flow_target_tracks_encoder` / `rho_memory_tracks_feature` + 各 `_not_cauchy` 推论 | ✅ 已证 |
| §3.4.2 定理 5：阶段适定性链 S1→S2→S3→S4（上游非平稳 ⟹ 下游不适定） | `Curriculum.lean` : `stage_chain_well_posedness` / `four_stage_chain_nonstationary_breaks_downstream` | ✅ 已证 |
| §3.4.2 $\varepsilon$ 阈值 / 滑窗 / 阶段比例 / 梯度方差 / 门控等价性 | — | ⚠️ 设计断言，非 Lean 可形式化，见 §3.4.2 与 README 问题 14 |
| §3.11 慢/快谱分离：持久性 $a^k$ 随 $a$ 单调；快特征 $a\in[0,1)$ 衰减到 $0$；快慢比 $(a/b)^k\to0$ | `TemporalHierarchy.lean` : `persistence_mono` / `fast_decay` / `fast_slow_ratio` | ✅ 已证 |
| §3.11 跨步 = 算子幂（加法性）：$M^{a+b}=M^a\circ M^b$（层次可组合） | `TemporalHierarchy.lean` : `powL_add` | ✅ 已证 |
| §3.11 跨步保收缩：严格收缩 $f$ 的 $n$ 次迭代仍是严格收缩（常数 $K^n<1$） | `TemporalHierarchy.lean` : `lipschitz_iterate` / `iterate_still_contracting` | ✅ 已证 |
| §3.11 几何级数伸缩恒等式：$(id-M)\circ(\Sigma_{i<s}M^i)=id-M^s$ | `TemporalHierarchy.lean` : `geom_series_telescope` | ✅ 已证 |
| §3.11 不动点跨步不变性：基层不动点 $z$（$Mz+b=z$）也是任意 stride $s$ 粗算子的不动点 | `TemporalHierarchy.lean` : `fixedPoint_stride_invariant` | ✅ 已证 |
| §3.11 速率-分辨率权衡：$K\in[0,1]$ 时 $K^s\le K$（$s\ge1$，越粗收敛越快） | `TemporalHierarchy.lean` : `rate_mono_stride` | ✅ 已证 |
| §3.11 跨步幂乘法性：$(K^s)^n=K^{s\cdot n}$（残余只依赖总 horizon $s\cdot n$） | `TemporalHierarchy.lean` : `pow_mul_stride` | ✅ 已证 |
| §3.11 残余随总 horizon 单调递减：$H\le H'\Rightarrow K^{H'}\le K^H$ | `TemporalHierarchy.lean` : `residual_horizon_mono` | ✅ 已证 |
| §3.11 迭代次数随 stride 单调递减：$s\le s'\Rightarrow K^{s'\cdot n}\le K^{s\cdot n}$ | `TemporalHierarchy.lean` : `coarse_iterations_mono` | ✅ 已证 |
| §3.12 序列输入递推闭式：$Z^{(t)}=M^t Z^{(0)}+\Sigma_{i<t}M^{t-1-i}x_i$（Z 隐式时间轴） | `TemporalHierarchy.lean` : `iterateSeq_eq_closedForm` | ✅ 已证 |
| §3.13 慢特征保护抬升留存：$0\le a_s\le a_s'\Rightarrow a_s^k\le a_s'^k$（长程记忆下界逐点抬升） | `TemporalHierarchy.lean` : `slow_retention_lifted` | ✅ 已证 |
| §3.13 保护后仍收缩：$a_s'\in[0,1)$ 时 $a_s'^k<1$（$k>0$，不破坏 ρ<1 契约） | `TemporalHierarchy.lean` : `protected_slow_still_contracting` | ✅ 已证 |
| §3.13 保护拉大慢快留存差距：$a_s^k-a_f^k\le a_s'^k-a_f^k$（分离强化） | `TemporalHierarchy.lean` : `slow_fast_gap_widened` | ✅ 已证 |
| §3.13 自反馈修正随 horizon 衰减：$a_f\in[0,1)$ 时 $a_f^k\to0$（短期局部化） | `TemporalHierarchy.lean` : `feedback_correction_decays` | ✅ 已证 |
| §3.13 门控自反馈不扰动慢特征：$\Pi_f(\Pi_s z)=0\Rightarrow (C\circ\Pi_f)(\Pi_s z)=0$ | `TemporalHierarchy.lean` : `feedback_spares_slow` | ✅ 已证 |
| §3.14 ZHierarchy 对角门控逐点独立：第 $i$ 维输出只依赖第 $i$ 维输入（GPU 并行，无跨维依赖） | `TemporalHierarchy.lean` : `diagGate_pointwise` | ✅ 已证 |
| §3.14 ZHierarchy 门控复合 = 逐维权重相乘：$\operatorname{diag}(p)\circ\operatorname{diag}(q)=\operatorname{diag}(p\cdot q)$（可合并） | `TemporalHierarchy.lean` : `diagGate_comp` | ✅ 已证 |
| §3.14 ZHierarchy 门控逐维放过：$p_i=0\Rightarrow(\operatorname{diag}(p)z)_i=0$（推广 feedback_spares_slow） | `TemporalHierarchy.lean` : `diagGate_zero_dim` | ✅ 已证 |
| §3.14 ZHierarchy 软门控逐维单调：$p_i\le q_i$、$0\le z_i$ ⟹ $(\operatorname{diag}(p)z)_i\le(\operatorname{diag}(q)z)_i$ | `TemporalHierarchy.lean` : `diagGate_mono` | ✅ 已证 |
| §3.14 ZHierarchy 软门控不破坏收缩：$a\in[0,1)$、$p\in[0,1]$ ⟹ $pa\in[0,1)$ | `TemporalHierarchy.lean` : `soft_gate_preserves_contraction` | ✅ 已证 |
| §3.14 ZHierarchy 门控保持谱夹层上界：$0\le a<u$、$p\le1$ ⟹ $pa<u$（$u=1-\gamma$） | `TemporalHierarchy.lean` : `gate_preserves_upper_bound` | ✅ 已证 |
| §3.14 ZHierarchy 门控保持谱夹层下界充要条件：$0<\delta\le a$ 时 $\delta<pa\iff\delta/a<p$ | `TemporalHierarchy.lean` : `gate_preserves_lower_bound` | ✅ 已证 |

**成熟要点（直接引用，不重证）**：

- 命题 1 / 2 / 10 的连续版本依赖微分熵与率失真测度论（mathlib 暂缺），但其**可判定核心**是同一离散事实：确定性映射 $g$ 的码本严格小于源集 ⟹ 必非单射 ⟹ 解码 / 重编码无法唯一还原 ⟹ 创新恒非零（`card β < card α`，**单向**鸽笼，反向为假）。
- §3.7 的「结论 = 变换链不动点」、§3.7.3 的 PSCO 仿射幺半群（结合律 + 表示同态 + 线性递推闭式）、§3.9 的「重建 = 恒等 / 生成 = 非恒等」二分均已形式化，是算子谱系 §3.9 与并行扫描正确性的根基。
- §3.7.3 的谱夹层（逐维 $\delta>0$ 下界）**严于** §3.7.1 的收缩条件 $\rho<1$（额外排除负特征值振荡收敛），二者不可等价混用。

---

## 1. 架构总览：端到端四阶段流水线

先明确整篇文档论证的是什么：一个把「多模态输入 → 共享世界表征 → 表征内对齐与推理 → 多模态输出 → 自反馈修正」串成闭环的世界模型，本文当前只实现并验证其中的**文本通道**。

四个阶段：

1. **编码（§2）**：把各模态输入压进一个共享表征空间 $Z = E(W)$——$Z$ 代表世界状态/世界规则的抽象。文本通道用 JEPA 学连续潜表征；$T = g(Z)$ 是这一通道的有损投影。
2. **对齐与推理（§3）**：让不同模态的表征落在同一坐标系 $Z$，并在这个表征空间内做转换与计算。**这两件事合并进同一阶段**——对文字而言，对齐（$T=g(Z)$，描述状态）与推理（$T_{cot}\approx[g(Z^{(0)}),\dots]$，描述算子轨迹）句法上不可分、常混在同一段文字里，由网络自推断的隐含模式 $h$ 分辨。
3. **解码（§4）**：把表征空间里的 $Z$ 转回对应多模态通道的输出。文本通道用条件 Flow Matching 采样 + 位置混合 Transformer 解码器确定性映射回 token。
4. **自反馈修正（§5）**：把解码产物 $T$ 重新压回表征空间 $\tilde Z=E(T)$，与推理预测 $\hat Z$ 比对得到创新 $e=\hat Z-\tilde Z$，据此校正状态（$\hat Z^+=\hat Z-Ke$）与参数。这一步把前三个阶段的开环前向接成闭环——而「文字有损」（命题 1）恰好保证 $e$ 永不枯竭，自反馈总有驱动信号。

**为什么对齐与推理不可分。** 文字有两副面孔，且二者混写：

- **描述性文本** $T=g(Z)$：对应世界当前状态的表征，是 $Z$ 的有损投影（命题 1–4）；
- **推理性文本** $T_{cot}\approx[g(Z^{(0)}),g(Z^{(1)}),\dots]$：对应表征空间的**预测 / 算子轨迹**（CoT），是 $\Psi$ 逐步执行的投影（命题 7、命题 9）。

句法上无法把它们分开——"因为地面湿了，所以刚才下过雨"里同时含状态描述与因果推理，同一个 $Z$ 既是被描述的对象又是被变换的对象。其他模态不同：

- **图像**描述的基本是当前世界状态；**视频**把每个时间点的状态排开，预测 / 推理隐含在帧间演化里，不单独成文字；
- **语音**混杂：既可以是文字对应的语言潜空间（$g$ 的投影），也可以是自然界物理现象的声音（当前状态与世界推演的观测）；
- **机器人控制通道 / 触觉通道**同理——既是状态的读取，又隐含下一步的动作 / 预测。

因此对齐与推理本就不该切成两个阶段：它们共享同一个 $Z$、同一套算子 $\Psi$，只是文字把「描述」与「运算」用同一种符号混写，使对齐目标与推理目标在文本上交织。§3 因而把硬对齐（§3.1–3.4）与算子推理（§3.5–3.8）并置，用统一的「预测框架」（§3.8）把它们归为同一目标的不同采样。

一句话：**编码把世界压进 $Z$，对齐与推理在 $Z$ 内交织闭环，解码把 $Z$ 渲回各模态，自反馈再把产物读回 $Z$ 校正前向——文字既描述状态、又描述状态的变换，而它的有损恰是这闭环永不停转的能量源。**

**关键设计取舍：串行深度是 GPU 并行的判据。** 自回归解码是 $O(L)$（随序列长度线性增长），而 Flow 是 $O(N)$（积分步数，与序列长度解耦）；每一步都是 $B\times L\times d$ 的稠密并行，$L$ 较大时与自回归有数量级差别。这就是选条件 Flow Matching 而非自回归解码的核心原因，SD3 / Flux / Sora / π0 是其大规模背书。生成可并行：Flow 一步到位（可蒸馏到 1–4 步），或 MaskGIT 式迭代并行填充。串行深度的完整对比见 §4.3(a)。

```mermaid
flowchart TB
    subgraph S1["阶段一 · 编码：输入 → 表征空间 Z（多模态通道）"]
        direction TB
        X["干净序列 x"] --> MASK["掩码 M(x)"]
        MASK --> XC["上下文 x_c"]
        XC --> EON["在线编码器 E_θ"]
        X --> ETG["目标编码器 E_ξ<br/>(EMA·无梯度)"]
        EON --> C["上下文表示 c"]
        ETG --> ZY["目标表示 z_y"]
        C --> PRED["预测器 P_φ"]
        PRED --> ZHAT["预测 ẑ"]
        ZHAT -. "1 - cos(ẑ, z_y)" .-> ZY
        EON --> Q1["量化瓶颈 Q<br/>(FSQ / VQ / RVQ)"]
        Q1 --> DEC1["解码器 D_η<br/>(位置混合 Transformer)"]
        DEC1 --> REC["重建 x<br/>(交叉熵 + 方差正则)"]
        IV["图像 I / 视频 V"] -.-> EIV["E^I / E^V<br/>(概念，P6)"]
        EIV -.-> ZENC["共享表征 Z = E(W)"]
        ETG -.-> ZENC
    end

    subgraph S23["阶段二 · 对齐与推理：统一坐标系 + 算子复合（概念，P5–P8）"]
        direction TB
        ZENC --> ALIGN["硬对齐 · 交叉掩码<br/>各模态共用同一 Z"]
        ALIGN --> Z2["统一世界逻辑 Z"]
        Z2 --> PSI["状态依赖算子 Ψ_h(Z) / 闭环 K"]
        PSI --> Z3["变换后表征 Z'"]
        Z2 -. "描述 g(Z) 与<br/>推理 g(Z^(i)) 混写" .-> PSI
    end

    subgraph S4["阶段三 · 解码：Z → 多模态输出"]
        direction TB
        Z3 --> N0["噪声 z_0 ~ N(0, I)"]
        N0 --> ZT["插值 z_t = (1-t)z_0 + t·z_1"]
        Z1["目标潜变量 z_1"] --> ZT
        COND["条件 c（E_θ 冻结）"] --> VEL["速度场 v_ψ(z_t, t, c)"]
        ZT --> VEL
        VEL -. "回归 z_1 - z_0" .-> Z1
        N0 -. "推理 Euler N 步" .-> ZHAT2["采样潜变量 ẑ(1)"]
        ZHAT2 --> Q2["量化 Q"]
        Q2 --> DEC2["解码器 D_η"]
        DEC2 --> TEXT["生成文本 T"]
    end

    subgraph S5["阶段四 · 自反馈修正：重编码 + 创新校正"]
        direction TB
        TEXT --> ENC2["重编码 ẑ̃ = E(T)"]
        Z3 --> ENC2
        ENC2 --> INNOV["创新 e = ẑ - ẑ̃"]
        INNOV --> CORR["状态修正 ẑ⁺ = ẑ - K·e"]
        INNOV -. "‖e‖² / L_consist" .-> PARAM["参数修正 Ψ, E, D_η"]
        CORR --> ZFB["校正后表征（回灌 §3）"]
    end

    S1 --> S23 --> S4 --> S5
```

**(a) 前向流程。**

阶段一 · 编码（表征学习，[losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `forward_losses`）：

1. 对图像/文本施加掩码得到上下文（图像换 `mask_token`、文本换 `mask_id`），在线编码器给出上下文表示 $c$，目标编码器（`no_grad`）给出干净目标表示 $z_y$；
2. 模态各自的预测器 $\hat z=P_\phi(c)$，用 $1-\cos(\hat z,z_y)$ 训练；
3. 解码器重建本模态输入（图像 MSE、文本交叉熵），并做跨模态硬对齐双向重建（文本 token 重建图像、图像 token 重建文本）；
4. InfoNCE 软对齐图像/文本 `[CLS]`，方差正则防维度塌缩，总损失
$$\mathcal L_1=\mathcal L_{\text{JEPA}}+\mathcal L_{\text{recon}}+0.5\,\mathcal L_{\text{cross}}+0.1\,\mathcal L_{\text{align}}+0.1\,\mathcal L_{\text{var}}.$$

其中 $\mathcal L_{\text{cross}}$ 是硬对齐双向重建（图像与文本穿过同一 $Z$），$\mathcal L_{\text{align}}$ 是 InfoNCE 软对齐；权重对应 [config.py](file:///home/simon-ma/JEPA/model/config.py) 的 `cross_weight=0.5`、`align_weight=0.1`、`var_weight=0.1`。

阶段二 · 对齐与推理（跨模态统一 + 表征空间内计算，概念，见 §3）：

1. 各模态编码器 $E^I, E^V, E_\theta$ 把输入压进同一 $Z$；
2. 硬对齐要求 $\hat T=g(E^I(I))\approx T$、$\hat I=R_I(p_\phi(Z\mid T))\approx I$（两条路径穿过同一 $Z$）；
3. 跨模态交叉掩码防止某模态坍缩；本脚本只实现文本通道，$E^I, E^V$ 未实现；
4. 主干 $H$ 从输入 $x$ 自推断隐含模式状态 $h=H(x)\in\mathcal H$，同时承载描述/推理倾向与变换内容；
5. 状态变换 $Z'=\Psi_h(Z)$ 全程在 $Z$ 内闭环；闭环观测器再加 $K e_t$ 修正；
6. token 化 $g$ 降级为可选的可解释性出口，非推理必经环节。

阶段三 · 解码：

1. 冻结 $E_\theta$ 与 $E_\xi$，潜分布静止；
2. $z_1=E_\xi(x)$，$z_0\sim\mathcal N(0,I)$，$z_t=(1-t)z_0+t z_1$；
3. 速度场 $v_\psi(z_t,t,c)$ 回归 $z_1-z_0$，条件 $c$ 经交叉注意力注入；
4. 解码器 $D_\eta$ 用教师强制的 $Q(z_1)$ 重建训练；训练另加 Flow 预测端点 $Q\bigl(z_t+(1-t)v_\psi\bigr)$（对 $v_\psi$ detach，只训解码器）作采样重建，使解码器对采样误差鲁棒；
5. 推理用 Euler 积分 $N$ 步得 $\hat z(1)$，再取 $T=\arg\max D_\eta(\hat z)$。

阶段四 · 自反馈修正（闭环校正，概念，见 §5）：

1. 把生成文本 $T=g(\hat Z)$ 重新编码 $\tilde Z=E(T)$，得到可观测表征；
2. 创新 $e=\hat Z-\tilde Z$，由命题 1 恒非零（文本有损）；
3. 状态修正 $\hat Z^+=\hat Z-Ke$（承接 §3.6 观测器），参数修正把 $\|e\|^2$ 与一致性项 $\mathcal L_{\text{consist}}$（式 48）回灌 $\Psi,E,D_\eta$；
4. 本脚本未实现这一闭环；$e$ 的统计量可由 `analyze.py` 的重编码失真（推论 2）间接观测。

注意：$P_\phi$ 只在 JEPA 阶段使用；生成时由 $v_\psi$ 接替"条件 $\to$ 潜变量"的映射。$E_\theta$ 提供条件，$E_\xi$ 定义目标分布。

**(b) 公式–模块对照。**

| 数学对象 | 模块 | 作用 |
|---|---|---|
| $c=E_\theta(x_c)$，$z_y=E_\xi(x)$ | 双编码器 | 上下文表示 / 目标表示 |
| $\hat z=P_\phi(c)$，$1-\cos(\hat z,z_y)$ | JEPA 预测器 | 潜空间预测（仅 JEPA 阶段） |
| $\mathrm{CE}(D_\eta(E_\theta(x)),x)$ | 位置混合 Transformer 解码器 | 保留 token 信息、防塌缩 |
| $\mathcal L_{\text{var}}$ | 方差正则 | 防止维度塌缩 |
| $z_t=(1-t)z_0+t z_1$ | 线性概率路径 | OT 插值 |
| $v_\psi(z_t,t,c)$ | Flow Transformer | 学速度场 |
| $z_{k+1}=z_k+\Delta t\,v_\psi$ | Euler 采样 | 潜变量生成 |
| $T=\arg\max D_\eta(Z)$ | 解码 | 离散 readout |

**(c) 概率视角。**

当前模型是一个带离散观测的潜变量模型：

$$p(T\mid x_c)\approx\int p_\theta(T\mid Z)\,p_\psi(Z\mid c)\,dZ,\qquad c=E_\theta(x_c),$$

其中 $p_\psi$ 由 Flow 定义，$p_\theta$ 在当前实现中是 $\arg\max$ 的确定性点质量。

**(d) 与后文的关系。**

- §2.3–2.6 把 $T=g(Z)$ 抽象成统一潜空间 $Z$ 的离散有损通道（纤维几何、率失真地板、DIB 摘要性、视觉马赛克同构）；
- §3.1–3.4 讨论多模态如何对齐到同一 $Z$（硬对齐、互补压缩、PAC 训练策略），§3.5–3.8 讨论 $Z$ 内的转换与计算（算子空间、闭环观测器、文字时间性、统一预测框架），二者共享同一预测框架、不可切分；
- §4.2 对齐当前解码实现、§4.3 给出显式离散瓶颈与并行解码、§4.4–4.5 讨论噪声熵源与复杂度分离、§4.6 给出一致性约束。

---

## 2. 阶段一 · 编码：输入 → 表征空间 Z

### 2.1 世界状态与统一潜空间假设

前文总览只处理单一模态（字符）。要陈述"图像、视频、文字都是同一潜空间的不同描述"，必须显式引入世界状态。令世界状态 $W \sim p_W$，共享潜变量

$$Z = E(W) \in \mathbb{R}^{L \times d},$$

并设各模态都是同一 $Z$ 的渲染：

$$
I = R_I(Z) + \varepsilon,\qquad
V = R_V(Z) + \varepsilon,\qquad
T = g(Z) \in \mathcal{V}^L .
$$

其中 $I, V$ 是连续观测（图像、视频），$T$ 是离散 token 序列，$g$ 是文本量化器。本模型只实现最后一条通道：$g$ 为位置混合 Transformer 解码器的 $\arg\max$（[decoder.py](file:///home/simon-ma/JEPA/model/decoder.py) 的 `TextDecoder`）；$R_I, R_V$ 是概念量。

为使命题有意义，需要三条假设：

- **(A1) 共享性**：存在跨模态共享的潜空间，使不同模态的编码落在同一（或可对齐的）子空间。这是"统一潜空间"的经验前提，需通过跨模态检索 / InfoNCE 检验，而不是由本脚本保证。它的实现是阶段二（§3）的主题。
- **(A2) 文本通道是确定性量化器**：$g$ 逐位置确定，故 $H(T \mid Z) = 0$。
- **(A3) 连续通道近可逆**：在足够高精度 / 码率下 $H(Z \mid I) \to 0$。这是图像、视频与文本的关键不对称来源。

记 $Y$ 为**跨模态共享语义因子**（低维、模态无关的量，可用线性探针或跨模态对齐目标估计）。命题如下。

> **命题 1（有损性）**：对任意有限字母表的确定性 $g$，在任意精细分辨率 $\delta$ 下 $H(Z_\delta \mid T) > 0$。文本无法唯一确定连续潜变量。
>
> **命题 2（失真地板）**：$T$ 能达到的失真存在正下界 $D_T \ge D^{\star} > 0$，且该下界只由码率预算 $L\log_2|\mathcal{V}|$ 与潜空间分布决定。
>
> **命题 3（摘要性）**：存在 $Y$ 使语义码率 $I(T;Y)$ 接近 $I(Z;Y)$，同时细节损失 $H(Z\mid T)$ 仍然很大。即文本是**关于 $Y$ 的摘要**，而非随机丢弃。
>
> **命题 4（模态不对称）**：连续通道的失真可随精度趋于 $0$，而文本通道在固定 $L, |\mathcal{V}|$ 下永远高于地板 $D^{\star}$。

> *(Lean 已验证离散核心：命题 1/2/10 见 `Lossy.lean`，其连续版依赖率失真测度论未形式化；「任意精细分辨率」措辞过强，见 §1.0 对照表与 README 问题 5。)*

**选择性有损压缩：编码器压缩与解码器解压缩的对称对。** 上述命题把"有损"当作观测结果；这里把它升格为**设计目标**。编码器与解码器是一对对称的压缩/解压缩对：

$$X \xrightarrow{\ E\ } Z \xrightarrow{\ D\ } \hat X = D(E(X))\neq X.$$

编码器 $E$ 把输入压进低维 $Z$（丢信息），解码器 $D$ 从 $Z$ 恢复能恢复的那部分；二者共享同一 $Z$、互为逆过程的不完全实现，$E$ 的"压缩"与 $D$ 的"解压缩"正是同一有损通道的两端。**关键在丢什么**：理想不是均匀丢，而是只丢 $Y$ 无关的细节。把 $Z$ 的信息拆成"主要脉络" $I(Z;Y)$（与语义因子 $Y$ 的互信息）与"无关细节" $H(Z\mid Y)$（给定 $Y$ 后残余的熵），理想编码器即信息瓶颈：

$$\max_{E}\ I(Z;Y)\quad\text{s.t.}\quad I(X;Z)\le R,$$

或 Lagrangian 形式 $\mathcal L_{IB}=I(X;Z)-\beta I(Z;Y)$。码率预算 $R$ 有限 → 把有限处理能力全花在 $I(Z;Y)$、挤掉 $H(Z\mid Y)$——正是"算力有限、集中在关键点"。于是解码器只能恢复 $Y$ 相关的**骨架**：$\hat X$ 忠实于 $Y$ 相关方向，重建误差 $\|X-D(E(X))\|$ 主要落在 $Y$ 无关方向。**有损不是缺陷而是设计**：人类抽象出主要脉络、忽略无关细节（文字、图像皆然）。对应到本模型——量化瓶颈 $Q$（FSQ/VQ/RVQ，§2.2）是码率约束的落地，JEPA 预测目标 $1-\cos(\hat z,z_y)$ 把 $Z$ 朝"可预测的结构"偏置，而"可预测"正是 $I(Z;Y)$ 在自监督下的代理（§3.7 的"规律、动力学、不变量"即无标签场景的 $Y$）。命题 3 的"摘要性"由此从被动观测升格为主动设计目标。

下面 §2.3–2.5 依次验证命题 1–3（命题 3 同时给出命题 4），§2.7 补上视觉通道，§4.2 说明如何落到本模型实现。

#### 2.1.1 潜空间的非欧性质与路径存在性

前文默认 $Z\in\mathbb{R}^{L\times d}$、flow 的 OT 路径是欧氏直线。这一默认过强：**实际潜空间不是整个 $\mathbb{R}^{L\times d}$，而是世界分布经编码器推前后的支撑集**

$$\mathcal M = \mathrm{supp}(E_\# p_W) \subseteq \mathbb{R}^{L\times d}.$$

由 Sard 定理 / 正则值定理，光滑映射 $E$ 的像的维数 $\le \dim(\text{世界状态流形})\ll L\times d$，故 $\mathcal M$ 泛型地是一个**低维、可能弯曲、可能不连通、可能有洞**的子流形，而非填满环境空间。

**路径不存在的三类来源。**

1. **不连通（$\pi_0(\mathcal M)>1$）**：$z_0,z_1$ 落在不同连通分支时，$\mathcal M$ 内不存在连续路径连接它们——欧氏直线穿过的是 $p(Z)$ 的零密度"真空"。
2. **有洞（$\pi_1(\mathcal M)\ne 0$）**：即使连通，环面/球面等非平凡拓扑使某些"本应直连"的段必须在 $\mathcal M$ 内绕行。
3. **曲率**：即使连通且单连通，弯曲的 $\mathcal M$ 使欧氏直线段 $[z_0,z_1]$ 离开 $\mathcal M$——直线在环境空间"存在"，但不在"可解码的 $Z$ 空间"里。

**引理（最短路径不一定是直线）。** 设 $\mathcal M\subset\mathbb{R}^{L\times d}$ 是弯曲黎曼子流形，$z_0,z_1\in\mathcal M$。则 $\mathcal M$ 上连接二者最短的路径（测地线）一般**不等于**欧氏直线。

*证。* 测地线 $\gamma$ 最小化内在长度 $\ell[\gamma]=\int_0^1\sqrt{\dot\gamma^\top \mathbf G(\gamma)\,\dot\gamma}\,dt$（$\mathbf G=J_D^\top J_D$ 是解码器诱导的拉回度量，记粗体 $\mathbf G$ 以区别于文本量化器 $g$），并满足测地线方程

$$\ddot\gamma + \Gamma(\dot\gamma,\dot\gamma)=0,$$

其中 $\Gamma$ 是 Christoffel 符号（内在曲率）。欧氏直线 $\sigma(t)=(1-t)z_0+t z_1$ 满足 $\ddot\sigma=0$（零外在加速度），但它作为 $\mathcal M$ 上路径成立当且仅当 $\sigma([0,1])\subseteq\mathcal M$——这要求 $\mathcal M$ 关于欧氏结构测地凸，对弯曲子流形泛型不成立。二者重合当且仅当 $\Gamma\equiv 0$（$\mathcal M$ 平坦）。$\blacksquare$

*反例（圆）。* $\mathcal M=S^1$（单位圆），$z_0=(1,0)$、$z_1=(-1,0)$。欧氏直线是过原点的直径（长度 $2$），但原点不在 $S^1$ 上，故这条"直线"在 $\mathcal M$ 中**不存在**；$\mathcal M$ 上最短路径是半圆弧（长度 $\pi$）。直线更短、却不可行——**最短的可行路径是测地线，而非直线**。

**推论（Riemannian Flow Matching）。** 严谨推广是把 OT 直线换成流形测地线：$z_t=\exp_{z_0}\!\big(t\cdot\log_{z_0}(z_1)\big)$，速度场 $v_\psi$ 活在切丛 $T\mathcal M$ 上（Chen & Lipman, ICLR 2024）。

**可测判据。** 路径是否存在于 $\mathcal M$ 可由解码置信度沿路径判定：$c(t)=\max_v p_\theta(v\mid z_t)$。直线穿过 off-manifold 区时中途 $c(t)$ 塌陷；测地线则 $c(t)$ 全程平稳。

### 2.2 JEPA 表征学习（图像/文本双通道，共享 $Z$）

#### 2.2.1 双编码器与上下文构造

设数据分布为 $x \sim p_{\text{data}}$。JEPA 维护**在线/目标**两个角色的编码器，图像与文本各有一对（§0 的 `image_online/text_online`、`image_target/text_target`）：

- 在线编码器 $E_\theta$（可训练），
- 目标编码器 $E_\xi$（参数冻结，仅靠 EMA 追随 $\theta$），

初始化时 $\xi \leftarrow \theta$（[representation.py](file:///home/simon-ma/JEPA/model/representation.py) 的 `update_target`），并对 $E_\xi$ 关闭梯度（[representation.py](file:///home/simon-ma/JEPA/model/representation.py) 的 `update_target`）。

对每个样本 $x$，构造掩码上下文 $x_c = M(x)$，掩码比例为固定值 $m$（[config.py](file:///home/simon-ma/JEPA/model/config.py) 的 `mask_ratio=0.5`）。被遮位置替换为掩码 token（图像换 `mask_token`、文本换 `mask_id`），干净目标 $x_y = x$ 保持不变（[losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `mask_patches`/`mask_text`，均为逐位置独立 Bernoulli：`torch.rand(...) < ratio`）。

**mask 预测的角色与必要性。** mask 不是架构的组成部分，而是**训练阶段的关系建模自监督信号**——表征、预测器、解码器都不「是」mask；推理/生成阶段的 Flow Matching（§4）完全不依赖它（§4 用噪声→潜变量，mask 只在训练时出现）。

目的（§4.5）是让 $Z$ 折叠 token 间的全对全关系——「$z_i,z_j$ 的坐标值承载 $i,j$ 的关系」。mask 预测是逼出这个关系的可微信号：要预测被掩位置的潜表征，模型只能聚合所有可见位置的关系、无捷径可走，因此它与 §4.5「关系由 $O(L^2)$ attention 成对交互承载」是同一数学结构的两面。

mask 不是数学必然，但去掉它需要给出同样「迫使全对全关系 + 并行 + 硬对齐」的替代信号，当前无更强候选：

| 替代自监督目标 | 为何不如 mask |
|---|---|
| 自回归（下一 token） | 串行深度 $O(L)$、GPU 不友好；只利用单向关系，非 §4.5 的全对全 |
| 对比学习 | §3.4 定位为「软对齐」，与本框架「硬对齐」冲突；负样本构造复杂 |
| 去噪/扩散目标 | Flow 阶段已有，学分布而非 token 间关系 |
| 纯重建 | §2.2.4(a) 已有，只测「编码-解码闭环」，不测「可见→被掩」的预测能力 |

结论：「预测性」（从部分观测推断完整语义）是架构的硬要求，mask 是当前满足它的最优实现；未来若出现更强的「关系建模信号」可替换 mask，但那是换训练目标、不是动架构。

**多位置同时掩码的利弊与掩码分布选型。** 当前实现是逐位置独立 Bernoulli 掩码（[losses.py](file:///home/simon-ma/JEPA/training/losses.py#L14-L18) 的 `mask_text` / `mask_patches`：`torch.rand(...) < ratio`），一次前向即同时遮掉约 `ratio` 比例的位置——这正是 JEPA 相对自回归并行预测的核心优势。

利：(i) 并行预测、串行深度 $O(1)$，GPU 友好（§4.3）；(ii) 同时预测多个被掩位置，强迫全对全关系建模（§4.5 目的）；(iii) 每步产生多个位置的预测梯度，样本效率高。

弊：(i) **mask 比例过高时上下文不足**——可见位置数 $=L(1-r)$，短序列（冒烟 $L=16$）下 $r$ 过高时可见位置过少、任务退化为无信息猜测；故固定 $m=0.5$ 偏激进时，建议按 $L$ 收紧上限（$r\le 0.6$ 保证可见位置 $\approx 0.4L$）；(ii) **被掩位置间依赖被忽略**——预测器对每个位置独立输出 $z_i$，未显式建模被掩位置间的协调，这是「并行」换来的代价；(iii) Bernoulli 实际被掩数有方差，任务难度随 batch 漂移。

掩码**分布**比数量更关键——逐位置随机 mask 对图像 patch 太简单（相邻 patch 强相关，模型可「抄」相邻可见 patch 作弊、学不到全局结构），I-JEPA 为此改用 block mask：

| mask 分布 | 适用 | 理由 |
|---|---|---|
| 逐位置随机（当前） | 字符级短文本 | 字符间局部相关弱，作弊空间小 |
| block mask（I-JEPA） | 图像 patch | 阻断「抄相邻 patch」捷径，强迫全局理解 |
| span mask（T5/ERNIE 式） | 未来 BPE 长文本 | 相邻子词相关，掩连续 span 强迫长程依赖 |

建议：图像改 block mask、文本未来切 BPE 后改 span mask，并按序列长度 $L$ 收紧 `mask_ratio` 上限（如 $r\le 0.6$）。

**离散 token ↔ 连续 $Z$ 区域的映射。** 一个离散 token 并不对应 $Z$ 空间中的单个点，而对应一片连续区域。编码方向：$E_\theta$ 把含同一 token 的不同上下文编码成 $Z$ 里散布的一组点，构成该 token 的「编码纤维」$\mathcal F_v$——token 内的连续语义变异（如「红」的深浅、明暗）体现为纤维内的连续自由度。解码方向：§4.2(a) 的线性读出头用超平面划分把 $Z$ 切成 $|V|$ 个凸区域 $\mathcal R_v$，$g(z)=\arg\max_v(w_v^\top z+b_v)$ 把连续 $Z$ 判回离散 token。两者合起来给出「离散符号 ↔ 连续概念区间」的完整结构：$\mathcal R_v$ 是 token 的连续概念区间，$\mathcal F_v$ 是它对应的连续 $Z$ 原像，logits 差距度量边界处的「模糊区」。

#### 2.2.2 预测目标与余弦损失

记上下文表示与目标表示：

$$c = E_\theta(x_c), \qquad z_y = E_\xi(x_y).$$

$z_y$ 在 `torch.no_grad()` 下计算（[losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `forward_losses`），梯度只回传到 $E_\theta$ 与 $P_\phi$。预测器从上下文预测目标潜变量：

$$\hat z = P_\phi(c),$$

对应 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `forward_losses`。损失采用余弦相似度（而非原 I-JEPA 的 $\ell_2$）：

$$\mathcal{L}_{\text{JEPA}} = \mathbb{E}_{x}\left[\, 1 - \cos\!\bigl(\hat z,\; z_y\bigr)\,\right],$$

即 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `cosine_loss`。最小化该式等价于把预测方向对齐到目标表示方向。

#### 2.2.3 EMA 目标更新

每次优化步之后执行（[representation.py](file:///home/simon-ma/JEPA/model/representation.py) 的 `update_target`、[representation.py](file:///home/simon-ma/JEPA/model/representation.py) 的 `update_target`）：

$$\xi \leftarrow \mu\,\xi + (1-\mu)\,\theta, \qquad \mu = 0.99.$$

**为什么 EMA + stop-gradient 是必需的**：若没有 stop-gradient，平凡解 $E_\theta(x) \equiv \text{const}$、$P_\phi(c) \equiv \text{const}$ 可使余弦损失直接为 0，出现**表征塌缩**。stop-gradient 切断了这条捷径：目标 $z_y$ 只能缓慢地经 EMA 移动，在线网络必须去预测一个"缓慢漂移但非平凡"的目标。

#### 2.2.4 防塌缩的两道附加约束（源码中的关键设计）

> ⚠️ 严谨说明：余弦损失 + EMA 本身**并不能完全排除塌缩**——若在线编码器把所有位置映到同一常向量 $a$，预测器输出 $a$，目标经 EMA 也趋于 $a$，仍有 $\cos(a,a)=1$、损失为 0。因此本模型真正的塌缩防线是下面两项。防塌缩在本框架分三层：本节为**表征层**，**模态层**见 §3.4（交叉掩码），**算子层**见 §3.7.3（谱夹层 + 读出头 + 方差），三层总表见 §3.7.3。

**(a) 重建损失（主要防线；文本侧交叉熵、图像侧均方误差）**

$$\mathcal{L}_{\text{recon}} = \mathrm{CE}\!\bigl(D_\eta(E_\theta(x)),\; x\bigr)\ \ \text{（文本）},\qquad \mathcal{L}_{\text{recon}} = \bigl\|D_\eta(E_\theta(x))-x\bigr\|^2\ \ \text{（图像）}.$$

对应 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `forward_losses`（图像 `MSE`、文本 `CE`）。若 $E_\theta(x)$ 在不同位置退化为常数，解码器无法区分 token，交叉熵（或像素 MSE）将爆炸。该项强制表示中保留**位置可辨别的 token 信息**，是抑制塌缩的主力。

**(b) VICReg 风格方差约束（次要防线）**：对每个潜维度 $j$ 在 batch×位置维度上算边际标准差 $\sigma_j$，方差正则

$$\mathcal{L}_{\text{var}} = \mathbb{E}_j\bigl[\mathrm{relu}(1-\sigma_j)\bigr],\qquad \sigma_j = \mathrm{std}\bigl(\mathrm{reshape}(z,\,[-1,d])_{:,j}\bigr),$$

惩罚方差过小的潜空间维度、防止维度塌缩；见 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `variance_loss`。其「方向与率惩罚相反」的辨析见 §4.2(d)（编号式 (36) 亦在 §4.2(d)）。

> **作用对象修正（表征塌缩修复）**：上式 $\sigma_j$ 原仅作用于 token 序列 $Z\in\mathbb R^{L\times d}$（`var_weight`）。但 `[CLS]` 聚合 token 只受软对齐损失 $\mathcal L_{\text{align}}$ 监督，而 BYOL 余弦目标存在平凡退化解（所有 `[CLS]` 投影坍为同一常数、`cos=1`、损失为 0），且 BYOL 投影器以 LayerNorm 替代 BatchNorm（小 batch 更稳）后失去了 BN 的隐式反塌缩——实测 `[CLS]` 投影方差坍到 $\sim10^{-4}$ 而 token 序列 `var` 仍守住（序列受重建损失约束、`[CLS]` 不受）。故把方差正则**同时施加到对齐投影空间 $g_\theta(\texttt{CLS})$**（`align_var_weight`），直接约束塌缩发生的空间。Lean 已验证：`varPenalty(σ)=relu(1-σ)` 在 `σ=0`（常数塌缩）处取严格正值 `1`，故常数解不可能是零损失极小点（见 `lean-math/LeanMath/AntiCollapse.lean`）。

**(c) VICReg 风格协方差约束（维度去冗余）**：对对齐投影表征 $Z_c\in\mathbb R^{B\times d}$（居中后）算 $d\times d$ 协方差矩阵，惩罚非对角元素的平方和

$$\mathcal{L}_{\text{cov}} = \frac1d\sum_{i\neq j} C_{ij}^2,\qquad C = \frac{Z_c^\top Z_c}{B-1},$$

防止不同潜维度退化为彼此冗余（维度塌缩的另一形态）；见 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `covariance_loss`（`align_cov_weight`）。与 (b) 合起来构成 VICReg 的完整「方差 + 去相关」正则（VICReg 三项中的两项；不变量项即 $\mathcal L_{\text{align}}$）。

**(d) CLS 读出头锚定（非退化锚点）**：`[CLS]` 聚合 token 原仅受软对齐损失监督（欠监督 → 退化塌缩），现加读出头 $r(\cdot)$，令 CLS 预测**池化目标 token**（目标编码器 token 序列的均值池化，重建锚定、输入特定）：

$$\mathcal{L}_{\text{cls-readout}} = \|r(\texttt{img\_cls}) - \mathrm{pool}(Z_{y,\text{img}})\|^2 + \|r(\texttt{txt\_cls}) - \mathrm{pool}(Z_{y,\text{txt}})\|^2,$$

给 `[CLS]` 一个与对齐无关的梯度来源（§3.7.3 防线 2「不动点读出头锚定」的具体落地）：目标 token 序列受重建损失约束而输入特定，故其池化非退化，CLS 必须编码输入内容才能预测它，常数塌缩解被排除。见 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `cls_readout`（`cls_readout_weight`）。

阶段一总目标（[losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `forward_losses`，图文双模态各取一组）：

$$\mathcal{L}_1 = \mathcal{L}_{\text{JEPA}} + \mathcal{L}_{\text{recon}} + 0.5\,\mathcal{L}_{\text{cross}} + 0.1\,\mathcal{L}_{\text{align}} + 0.1\,\mathcal{L}_{\text{var}} + 0.1\,\mathcal{L}_{\text{align-var}} + 0.1\,\mathcal{L}_{\text{align-cov}} + 0.3\,\mathcal{L}_{\text{cls-readout}},$$

其中图像/文本各自有 JEPA 余弦损失与自重建（图像 MSE、文本交叉熵）；$\mathcal{L}_{\text{cross}}$ 是跨模态硬对齐双向重建（文本 token 重建图像、图像 token 重建文本，穿过同一 $Z$）；$\mathcal{L}_{\text{align}}$ 是图像/文本 `[CLS]` 的软对齐（§3.4，默认 BYOL）；$\mathcal{L}_{\text{align-var}}$、$\mathcal{L}_{\text{align-cov}}$ 是对齐投影空间的方差/协方差正则（见 (b)(c)）；$\mathcal{L}_{\text{cls-readout}}$ 是 CLS 读出头锚定（见 (d)）。权重对应 [config.py](file:///home/simon-ma/JEPA/model/config.py) 的 `cross_weight=0.5`、`align_weight=0.1`、`var_weight=0.1`、`align_var_weight=0.1`、`align_cov_weight=0.1`、`cls_readout_weight=0.3`。注意：这里**没有**量化器项——量化只发生在 token 读出层（§4.2(f)），$Z$ 全程连续（§2.6）。

#### 2.2.5 编码器主干的候选选型（测试不理想时的备选路径）

§2.2.1–2.2.4 定义的是 JEPA 的**训练目标与结构**（双编码器 + 预测器 + EMA + 防塌缩），与 $E_\theta, E_\xi$ 内部用哪种**序列主干**正交：任何能读入 token 序列、输出连续潜向量 $Z\in\mathbb{R}^{L\times d}$ 的骨干都可替换当前的位置混合 Transformer，训练目标与损失不变。

当前主干是位置混合 Transformer（self-attention）。若实测不理想（内存、单步耗时、长序列扩展性），可按需切换到以下候选：

1. **Mamba-2 / Mamba-3（SSM 选择性扫描）**：输出连续（§2.6 接口契约已确认），"discretization" 是 ZOH 数值积分而非输出离散化。训练 $O(L)$ 时间、$O(1)$ 状态，长序列（$L\gtrsim10^3$）下比注意力省时省内存；短序列（冒烟 $L=16$）无明显收益、常数因子略亏。
2. **混合骨干（注意力 + SSM 交错）**：前几层注意力抓局部/位置敏感模式、后几层 SSM 做长程压缩，兼顾表达力与线性复杂度，工程上常见的折中。
3. **其他线性复杂度序列模型**：RWKV、RetNet、线性注意力（Performer / 线性核）、xLSTM、Hyena 等，同属"把二次注意力换成线性/卷积/递归"一族，选型判据与 Mamba 相同。
4. **纯卷积 / 局部注意力 + 全局池化**：对字符级短序列局部感受野已够，全局信息经池化注入，训练最快但长程依赖最弱。

**切换判据**（什么信号提示该换）：训练显存随 $L$ 二次增长、单步耗时随 $L$ 超线性、或目标序列长度要扩到文档/视频帧级。这些信号出现前，默认维持 Transformer（实现成熟、常数因子优、短序列最快）。

关键：换主干不改变本框架的两条既有契约——(i) $Z$ 全程连续、不施加量化瓶颈（§2.6）；(ii) 有损压缩只发生在 token 层（§2.6）。因此任何候选都只是 $E$ 的内部实现细节，对外接口 $x \mapsto Z$ 不变。

#### 2.2.6 潜预测损失的备选度量（含 GPU 友好性）

§2.2.2 的 cosine 损失 $1-\cos(\hat z,z_y)$ 是**欧氏度量**：它在环境空间 $\mathbb{R}^{L\times d}$ 上算内积，隐含假设 $\mathcal M$ 平坦。但 §2.1.1 已证明 $\mathcal M$ 是弯曲黎曼子流形、正确度量是解码器诱导的拉回度量 $\mathbf G=J_D^\top J_D$，§2.3 进一步说明同一纤维 $\mathcal F_t$ 内的点语义等价（解码到同一 token）。据此给出三个在框架内自洽的备选，均属「备选路径」，实际取舍须实测（同 §2.2.5 与 §3.7.3 的判据原则）。

**(a) 拉回度量距离（黎曼 ℓ2）。**

$$\mathcal{L}_{\text{JEPA}}^{(\mathbf G)} = \mathbb{E}\Big[\big(\hat z - z_y\big)^\top \mathbf G(z_y)\big(\hat z - z_y\big)\Big] = \mathbb{E}\Big[\big\|J_D(z_y)\,(\hat z - z_y)\big\|^2\Big],\qquad \mathbf G(z)=J_D(z)^\top J_D(z).$$

依据：§2.1.1 的度量定义。它是方案 (b) 的一阶线性化——泰勒展开 $\|D(\hat z)-D(z_y)\|^2\approx\|J_D(z_y)(\hat z-z_y)\|^2$。纤维切方向上 $J_D$ 特征值趋 0，误差被自动压 0，同一纤维内判定为「预测正确」，这是 cosine/ℓ2 做不到的。

**(b) 经解码器的 token 分布对齐。**

$$\mathcal{L}_{\text{JEPA}}^{(\text{dec})} = \mathbb{E}\Big[\mathrm{KL}\big(p_\theta(\cdot\mid z_y)\,\big\|\,p_\theta(\cdot\mid\hat z)\big)\Big]\quad\text{或}\quad\mathrm{CE}\big(D(\hat z),\,g(z_y)\big).$$

依据：§2.3 纤维几何 + §2.2.4(a) recon 损失。直接在可观测的 token 分布层对齐，同纤维 $\Rightarrow$ KL=0。

**(c) 拉回度量内积（黎曼 cosine）。**

$$\mathcal{L}_{\text{JEPA}}^{(\cos,\mathbf G)} = \mathbb{E}\left[1 - \frac{\hat z^\top \mathbf G\, z_y}{\sqrt{(\hat z^\top \mathbf G\,\hat z)\,(z_y^\top \mathbf G\, z_y)}}\right].$$

保留 cosine 的尺度无关性，把欧氏内积换成 $\mathbf G$ 内积。

**GPU 友好性对比。** 三者都无跨样本串行依赖，全是 batch 并行逐元素/矩阵乘运算，与 §2.6 的 token 层压缩、§4.3 的串行深度判据一致——没有变长循环。差异只在额外算子的形态：

| 方案 | 额外算子 | 内存 | 额外串行深度 |
|---|---|---|---|
| (a) 拉回距离 | JVP $\|J_D(z_y)\delta\|^2$，$\delta=\hat z-z_y$ | $O(Ld)$（无需显式 $\mathbf G$） | $O(1)$ |
| (b) 解码对齐 | 一次解码器前向 | $O(Ld)$ | $O(1)$ |
| (c) 拉回 cosine | 同 (a) 的 $\mathbf G$ 内积 | $O(Ld)$ | $O(1)$ |

细节：冒烟尺度 $d=64$ 时显式构造 $\mathbf G$（$64\times64$）比 JVP 更省（$d^2$ 标量、一次矩阵乘）；JVP 用 `torch.autograd.grad` / `torch.func.jvp` 批量并行，代价约一次反向。方案 (b) 多一次解码前向，但解码器是 §4.5 的 $O(L)$ 通路 + 位置混合，成本低于 attention 主干。**三者对 GPU 均友好、无串行瓶颈；差别是 (a)(c) 付一次 JVP 代价、(b) 付一次解码前向代价。** 关键工程注意：$\mathbf G$ 随解码器 $D$ 训练漂移，损失的地面度量（ground metric）不稳定，宜在 $D$ 相对冻结（如 Flow 阶段 $E_\xi$ 冻结后）启用，或对 $\mathbf G$ 做 stop-gradient。

#### 2.2.7 前端翻译层设计与潜空间维度 $d$ 的跨规模选型

**(a) 前端翻译层的设计要点。** 当前前端为文本 `nn.Embedding`（离散→连续嵌入）与图像 `patch_embed=Conv2d(stride=patch_size)`（非重叠 patch 化）。受三条契约约束——§4.5「通路/关系分离」（前端是纯通路、逐位置 $O(L)$）、§2.6「输出连续、不量化」、§3.2「薄物理翻译层」——文本 embedding 是离散 token 的唯一合理翻译，数学正确、无需替换。

图像前端有一处与 §2.7 命题 5 的张力需修正：单层非重叠 patch 化是「连续→离散」的提前硬切割，与命题 5 主张的「连续→连续降采样、损失靠分辨率、可逼近无损」不一致。两个改进方向（均属备选、须实测）：

1. **卷积 stem（重叠、多级 stride-2 卷积）** 替代单层非重叠 patch，保持连续降采样、patch 间重叠，消除 patch 边界伪影；
2. **把前端纳入 §2.6 的「先写后压」统一框架**：先连续降采样得稠密特征序列，再由 $\rho=\|v-Mk\|^2$ 判据决定保留位置，使图像前端的「有损」与 token 层的「$\rho$ 压缩」成为同一机制，消除「图像前端提前硬丢信息 vs 文本 token 层按 $\rho$ 自适应丢」的不对称。

注意：编码器前端 embedding（查表）与解码器读出头（线性打分，§4.2(a)）**不对称是正确设计**——前者「离散→连续嵌入」、后者「连续→离散判别」，不应 weight tying。

**(b) 潜空间维度 $d$ 与主干 $d_{\text{hidden}}$ 的比例选型。** 先区分两个易混的「$d$」：

- **主干 hidden 维度 $d_{\text{hidden}}$**：决定参数量与表达力，$P\approx 12\,n\,d_{\text{hidden}}^2$（标准 Transformer、FFN $4d$，不含 embedding 层）；
- **潜空间维度 $d$**：编码器输出 $Z\in\mathbb{R}^{L\times d}$ 的语义瓶颈维度，是 §2.6「已压缩精华」的落点。

当前 tiny/small/base 三者绑定（`dim` 同时是主干宽度与潜空间维度，[config.py](file:///home/simon-ma/JEPA/model/config.py) 的 `PRESETS`），这是小规模的简化；**大规模下应解耦**——主干宽、潜空间窄。

**比例约定：$d = d_{\text{hidden}} / r$，取 $r=4$（即 $d : d_{\text{hidden}} = 1:4$）**，可调范围 $r\in[4,16]$。冲生成（§4 解码/生成、不做 §5 自反馈）时潜空间 $d$ 需承载生成的中间表示，故从居中默认 $r=8$ 收紧到下限 $r=4$（$d$ 翻倍）；纯表征阶段可回到 $r=8$。模型放大后按此比例等比例缩放。依据：

1. **压缩定位**（§2.6）：潜空间是「已压缩精华」，$d$ 应明显小于主干；$r$ 太小（如 1:2）则潜空间冗余、压缩语义弱、信息率偏高；
2. **信息率充分**（§4.4）：$r$ 太大（如 1:32）则潜空间过窄、语义丢失、$\widehat{\mathrm{Var}}(Z\mid T)$ 抬升；
3. **业界 VAE 潜空间压缩比**（latent:hidden）集中在 $1:4$–$1:16$，$1:8$ 是居中、可等比例放大的安全默认；
4. **拉回度量秩约束**（§4.2(a)+§2.2.6）：$\mathrm{rank}(\mathbf G)\le\min(|V|,d)$，在现成多模态词表 $|V|\approx 15\text{万}$ 下 $d\ll |V|$，$\mathbf G$ 满秩、方案 (a)/(c) 不再受词表约束。

主干 $d_{\text{hidden}}$ 按业界成熟经验（GPT-3/LLaMA 同构）随参数量增长，潜空间 $d$ 按 $r=8$ 等比例缩放：

| 规模 | 参数量 | 主干 $d_{\text{hidden}}$ | 层数 $n$ | 潜空间 $d$ | 词表 |
|---|---|---|---|---|---|
| tiny（CPU 冒烟） | ~0.87M | 64 | 2 | 64（冒烟绑定 $d=d_{\text{hidden}}$） | 字符级 |
| small | ~5.7M | 128 | 4 | 128（冒烟绑定） | 字符级 |
| base（冒烟/过渡） | — | 256 | 6 | 256（冒烟绑定，$L=64$） | 字符级 |
| 中型 0.3B | ~0.3B | 1024 | 24 | 256（$=d_{\text{hidden}}/4$） | 现成多模态 |
| 中型 1B | ~1B | 2048 | 24 | 512 | 现成多模态 |
| 中型 3B | ~3B | 3072 | 32 | 768 | 现成多模态 |
| 中型 7B | ~7B | 4096 | 32 | 1024 | 现成多模态 |

（冒烟三档与 [config.py](file:///home/simon-ma/JEPA/model/config.py) 的 `PRESETS` 一致：tiny `dim=64/depth=2`、small `dim=128/depth=4`、base `dim=256/depth=6`，此时潜空间维度尚未从主干解耦。）

**词表（tokenizer）选型——用现成 mask 预测模型词表，不自建 BPE。** 自建 BPE 脱离主流研究生态，直接复用带 mask id 的现成 tokenizer：

- **首选 LLaDA tokenizer**（`GSAI-ML/LLaDA-8B-Instruct` 或 `GSAI-ML/LLaDA-8B-Base`）：LLaDA 本身即 masked diffusion 模型，词表内置专用 mask token `<|md|>`（`mask_token_id=126336`，词表约 12.6 万、基于 Qwen2 扩展）；多模态方向已有 LLaDA-V（CVPR 2026）/ MMaDA 证明该 tokenizer 的 mask 机制可扩展到视觉；
- **备选 Qwen2.5-VL / Qwen3-VL tokenizer**：视觉特殊 token（`<|vision_start|>`、`<|image_pad|>` 等）已内置，但无专用 mask token，需 `add_special_tokens({'mask_token': '<|mask|>'})` 自补。

注意：本架构图像通道走连续潜空间（§2.6「$Z$ 全程连续、不量化」），**不引入** LLaDA-V / MMaDA 的视觉 VQ tokenizer（MAGVIT-v2 / SigLIP-VQ）；统一发生在连续 $Z$ 层（§3.2），不在离散 token 层。故只需文本侧 mask id，视觉侧沿用本模型的 `ImageEncoder` 连续 patch 化。

**LLaDA tokenizer 接入代码（对接 `CharTokenizer` 接口）。** 现有 `train.py` 通过 `tokenizer.pad_id / mask_id / vocab_size / encode / decode` 消费 tokenizer（`data_prepare/synthetic.py` 的 `CharTokenizer` 同构）。LLaDA 用如下包装类即可无缝替换：

```python
"""LLaDA tokenizer：对接现有 CharTokenizer 接口（pad_id/mask_id/vocab_size/encode/decode）。"""
from transformers import AutoTokenizer

class LLaDATokenizer:
    def __init__(self, model="GSAI-ML/LLaDA-8B-Instruct"):
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tok.pad_token is None:            # Qwen2 系兜底：用 eos 作 pad
            self.tok.pad_token = self.tok.eos_token
        if self.tok.mask_token is None:           # LLaDA 词表内置 <|md|>，此处仅兜底
            self.tok.add_special_tokens({"mask_token": "<|md|>"})
        self.pad_id = self.tok.pad_token_id
        self.mask_id = self.tok.mask_token_id     # = 126336
        self.vocab_size = len(self.tok)

    def encode(self, text, length):
        ids = self.tok.encode(text, add_special_tokens=False)[:length]
        return ids + [self.pad_id] * (length - len(ids))

    def decode(self, ids):
        keep = [i for i in ids if i not in (self.pad_id, self.mask_id)]
        return self.tok.decode(keep, skip_special_tokens=True)
```

接入点：`TextEncoder` 的 `nn.Embedding(vocab_size, dim)` 用 `vocab_size=len(tokenizer)`（约 12.6 万），`mask_text` 用 `tokenizer.mask_id` 替换被掩位置；`encode/decode` 与 `CharTokenizer` 完全同构，`train.py` 的 `tokenizer.pad_id/mask_id/vocab_size` 路径无需改动。

**词表扩展对比例的影响**：$d:d_{\text{hidden}}=1:8$ 的比例与 $|V|$ 大小无关（只要 $|V|\gg d$）。词表从字符级（合成语料）→ 12.6 万的实际影响只有两处：(i) embedding 层参数量 $|V|\times d_{\text{hidden}}$（7B 档约 $12.6\text{万}\times4096\approx5.2\text{亿}$，占 ~7.4%，需计入参数量预算或对 embedding 层单独处理）；(ii) 拉回度量秩上界从字符级放宽到 $\min(|V|,d)=d$，方案 (a)/(c) 不再受词表约束。

**结论（3B–7B）**：主干 $d_{\text{hidden}}=3072$（3B）/ $4096$（7B），潜空间 $d=384$（3B）/ $512$（7B），即 $r=8$ 等比例缩放，词表用 LLaDA 现成 tokenizer（内置 mask id）。潜空间 $d$ 的最终值仍须按 §2.2.6 的实验判据实测（$I(Z;Y)$ 饱和点 + $\widehat{\mathrm{Var}}(Z\mid T)$），比例 $r$ 可在 $[4,16]$ 内据实调整。

### 2.3 纤维几何与数据处理不等式（验证有损性）

**(a) 离散码率的硬上界。**
$T$ 取值于有限集 $\mathcal{V}^L$，故

$$H(T) \le L \log_2 |\mathcal{V}|. \tag{1}$$

输出词表大小 $|\mathcal{V}|$ 与序列长度 $L$ 都是**规模相关的超参数**、不是固定常数：冒烟用字符级 `CharTokenizer`（[synthetic.py](file:///home/simon-ma/JEPA/data_prepare/synthetic.py)，$|\mathcal{V}|$ 为合成语料字符集大小、$L=\text{text\_len}=16$）；大模型接现成 tokenizer（LLaDA 等，$|\mathcal{V}|\sim10^5$，见 §2.2.7）。`[MASK]` 只出现在输入上下文、从不进入输出文本，故输出词表为 $|\mathcal{V}|$。于是

$$H(T) \le L \log_2 |\mathcal{V}|. \tag{2}$$

这是一个**与 $Z$ 无关、仅由 $(L,|\mathcal{V}|)$ 决定的有限上界**；它随规模伸缩（冒烟 $L=16,|\mathcal{V}|\approx20$ 时约 $69$ bit；$L=1024,|\mathcal{V}|=10^5$ 时约 $17$ kbit，完整演算见 §2.8）。

**(b) $T$ 是 $Z$ 的确定函数。**
因 $T = g(Z)$，给定 $Z$ 时 $T$ 完全确定，$H(T\mid Z)=0$，故

$$I(Z;T) = H(T) - H(T\mid Z) = H(T) \le L\log_2|\mathcal{V}|. \tag{3}$$

**(c) 连续潜变量的量化熵无界。**
设 $Z$ 有密度。将 $Z$ 量化到分辨率为 $\delta$ 的网格，得离散变量 $Z_\delta$。$Z$ 有 $Ld$ 个实数维度，高分辨率极限下（单位统一为 bit）

$$H(Z_\delta) \approx h(Z) + Ld \log_2 \frac{1}{\delta} \;\xrightarrow{\delta \to 0}\; +\infty. \tag{4}$$

例如 $\delta = 10^{-2}$ 时量化熵增量项 $Ld\log_2 100 \approx 6.64\,Ld$ bit；只要 $d \log_2(1/\delta) > \log_2|\mathcal{V}| - h(Z)/L$（右端随 $\delta\to0$ 趋于 $\log_2|\mathcal{V}|$，几乎总成立，见 §2.8），(4) 就远超 (2)，$\delta\to0$ 时 $H(Z_\delta)\gg H(T)$。这里不能丢掉 $h(Z)/L$ 项：$h(Z)/L=O(d)$，在中等 $\delta$ 下与 $d\log_2(1/\delta)$ 同阶，仅在 $\delta\to0$ 极限下才可忽略。

**(d) 数据处理不等式。**
$Z_\delta$ 与 $T$ 都是 $Z$ 的函数，构成 Markov 链 $Z_\delta \leftarrow Z \to T$（$Z$ 为公共父节点）。由**数据处理不等式**，

$$I(Z_\delta;T) \le I(Z;T) \le L\log_2|\mathcal{V}|. \tag{5}$$

当 $\delta$ 足够小时 $H(Z_\delta) > I(Z_\delta;T)$，于是

$$H(Z_\delta \mid T) = H(Z_\delta) - I(Z_\delta;T) \ge H(Z_\delta) - L\log_2|\mathcal{V}| > 0. \tag{6}$$

> **结论**：给定文本 $T$，$Z$ 在精细分辨率下仍有正的条件熵——存在 $T$ 无法确定的 $Z$ 的细节。$T$ 对 $Z$ 的描述是**严格有损**的，这是不依赖任何训练假设的信息论事实。

**(e) 纤维（fiber）视角。**
$g$ 把潜空间划分成 $|\mathcal{V}|^L$ 个格子

$$\mathcal{F}_t = g^{-1}(t) = \{ z : g(z) = t \},\qquad t \in \mathcal{V}^L .$$

文本只报告 $T$ 落在哪个格子，格子内部的变异不可见。任何由 $T$ 重建 $Z$ 的估计器 $\hat Z(T)$ 都满足 MMSE 下界

$$\mathbb{E}\big\| Z - \hat Z(T) \big\|^2 \;\ge\; \mathbb{E}\big\| Z - \mathbb{E}[Z\mid T] \big\|^2 \;=\; \sum_{j=1}^{Ld} \mathbb{E}\,\mathrm{Var}(Z_j \mid T). \tag{7}$$

> **结论**：失真来自**纤维内变异**，纤维粗细由码本大小 $|\mathcal{V}|^L$ 与潜空间几何共同决定。词表越小、序列越短，纤维越粗。

### 2.4 率失真地板与模态不对称

率失真函数

$$R(D) = \min_{p(\hat z \mid z):\ \mathbb{E}[d(Z,\hat Z)] \le D} I(Z;\hat Z) \tag{8}$$

关于 $D$ 单调不增；对连续 $Z$（有密度、$d$ 取平方误差）有 $R(0) = +\infty$。全文固定 $d$ 为平方误差；若各维尺度差异大，可先按维度方差归一化（马氏距离）。

**文本通道。** 以 $T$ 为描述，任何重建 $\hat Z(T)$ 满足

$$I(Z;\hat Z) \le I(Z;T) = H(T) \le L\log_2|\mathcal{V}|. \tag{9}$$

因此

$$R(D_T) \le L\log_2|\mathcal{V}|
\quad\Longrightarrow\quad
D_T \ \ge\ D^{\star} := \inf\{\, D : R(D) \le L\log_2|\mathcal{V}| \,\}. \tag{10}$$

由 $R(0)=+\infty$ 得 $D^{\star}>0$。注意 $R^{-1}$ 是多值的，故必须用上述"下确界失真"定义，而不能写成简单的反函数（严格说 $R$ 定义域上的极值应取 $\inf$；对连续源 $R$ 连续、下确界可达，故 $\inf$ 即 $\min$）。这把"有损"从定性变成了可计算的量：词表越小、序列越短，$D^{\star}$ 越高。

**视觉通道对比。** 对 $Ld$ 维、每维方差 $\sigma^2$ 的连续源，高斯情形下

$$R_{\text{vis}}(D) = \frac{Ld}{2}\log_2\frac{\sigma^2}{D}, \qquad 0 < D \le \sigma^2, \tag{11}$$

即失真可随码率任意趋近 $0$。文本通道的码率上限 $L\log_2|\mathcal{V}|$ 与 $D$ 无关，因此

$$D_{\text{vis}}(B) \xrightarrow{B\to\infty} 0,
\qquad
D_{\text{text}}(B) \ge D^{\star} > 0
\quad(\text{固定 } L,\ |\mathcal{V}|). \tag{12}$$

> **结论**：图像、视频是"连续对连续"的通道，可以在高码率下逼近无损；文本是"离散有限码本"的通道，存在与码率预算绑定的正失真地板。这就是命题 4 的严格版本。

### 2.5 确定性信息瓶颈——为什么是"摘要"而非随机丢弃

仅有损不足以称为"摘要"。经典信息瓶颈（Tishby 等，1999）为

$$\min_{p(t\mid z)}\ I(Z;T) - \beta\, I(T;Y), \qquad \beta > 0. \tag{13}$$

但本模型的 $g$ 是确定性映射，$H(T\mid Z)=0$，$I(Z;T)=H(T)$，压缩只来自码本基数、**不来自随机噪声**。因此正确的对象不是 (13)，而是**确定性信息瓶颈（DIB，Strouse & Schwab 2017）**的基数约束形式：

$$\max_{g:\ \mathcal{Z}\to\mathcal{C},\ |g(Z)| \le |\mathcal{V}|^L}\ I(g(Z);Y). \tag{14}$$

这里**不能写"$\beta\to\infty$ 的极限"**：在 (13) 中 $\beta\to\infty$ 会丢掉压缩项，最优解退化为 $T$ 是 $Z$ 的**无损编码**（任意可逆 $T$，含 $T=Z$；无限码率、零压缩），与目标恰好相反。(14) 把"码本基数不超过 $|\mathcal{V}|^L$"作为硬约束——注意约束的是 $g$ 的**像集** $|g(Z)|$ 而非陪域 $|\mathcal{C}|$；由 (1) 它蕴含 $H(g(Z))\le \log_2|g(Z)|\le L\log_2|\mathcal{V}|$，这才对应"固定词表 + 固定长度"的架构。

在 (14) 下，$T$ 是把 $Z$ 投影到离散码本、使语义互信息最大的量化器。要区分"语义"与"措辞"，需用 $Y$ 对**码率**做分解。注意式 (15) 是互信息定义式 $I(T;Y)=H(T)-H(T\mid Y)$ 的重排，对**任意**两个离散变量 $(T,Y)$ 恒成立、**不要求**它们同为 $Z$ 的函数；

$$H(T) \;=\; \underbrace{I(T;Y)}_{\text{语义码率}} \;+\; \underbrace{H(T\mid Y)}_{\text{措辞码率}}. \tag{15}$$

两个不同的量必须分清：

- **保留了什么**：$I(T;Y)$ 是文本携带的语义比特；$H(T\mid Y)$ 是与语义无关的表面形式（措辞）比特。
- **丢掉了什么**：$H(Z\mid T)$ 是潜空间中未被文本确定的细节，对应 (7) 的纤维内变异。

"摘要"的精确含义是：$I(T;Y)/H(T)$ 高（码率主要花在语义上），同时 $H(Z\mid T)$ 大（细节大量丢失）。注意 $I(Z;T)$ 是**保留**的量，丢弃量是 $H(Z\mid T)$，二者不可混淆。

> **结论**：文本不是任意量化，而是在固定码率约束下最大化语义互信息 $I(T;Y)$ 的瓶颈变量。这解释了"语义在、措辞丢"的两种读法：若指生成文本的表面形式与真实文本不同，那是 $H(T\mid Y)$ 的差异；若指潜空间中的精细信息无法恢复，那是 $H(Z\mid T)$。

### 2.6 不可压缩内容的生成元化压缩

§2.5 的 DIB 预设"存在 $H(Z\mid Y)$ 可丢"。但当样本是已被人类压到极致的**定理/公理**时，$H(Z\mid Y)\approx 0$——没有无关细节可挤，再套信息瓶颈 $I(X;Z)\le R$ 会把 $I(Z;Y)$ 一起切掉。此时必须换思路：**不可压缩的只是"陈述文本"，不是"内容"**。一个公理的可预测性不在它自身句子里，而在它作为**生成元**产生的轨道里（它推出整个理论）。于是"压缩不可压缩内容"的正确对象不是那个句子，而是"把它识别成生成元"。

**信息论判断。** 把内容拆成"种子" $S$ 与"轨道" $O$（推论）：$H(\text{内容})\approx H(S)+H(O\mid S)$，且轨道可预测 $H(O\mid S)\to 0$（给定种子，推论被决定）。于是压缩 = **存储 $S$、按需生成 $O$**，码率从 $H(\text{内容})$ 降到 $H(S)$。这正是"规律折叠成转移算子"（§3.5）的压缩版：定理 = 算子空间 $\Omega$ 的一个生成元，其轨道由 $\Psi_h$ 重新生成、从不存储。

**两条途径与成熟方案。**

1. **自适应压缩率（成熟）**：按样本熵分配码率，高熵样本不硬压。神经压缩的**熵模型**天然内容自适应——码率 $-\log p(\hat z)$，不可压缩内容熵高、自动获得更多比特。落地件：Ballé et al. 2018（scale hyperprior）、Minnen et al. 2018（autoregressive priors）、Choi et al. 2019（conditional autoencoder 的 per-sample rate $\lambda$）。本框架对应：把码率分配做在 **token 层的丢/留与比特门控**上（见下文「创新编码」），而**不是**对潜空间 $Z$ 追加量化瓶颈——$Z$ 是浓缩后的精华，不再二次有损（见下文「接口契约」）。
2. **填充再压缩 = 生成元识别（半成熟）**：把种子展开成可预测轨道（"填充"），再压缩回种子（"再压缩"）。原型是 **DreamCoder**（Ellis et al. 2021，wake-sleep 库学习：学可复用原语库即生成元）；理论地基是 **MDL / Kolmogorov 复杂度**（最短描述 = 最短生成程序）。本框架对应：§3.7 的显式 CoT（展开）与折叠（压缩）。

**离散 token 场景：变速率压缩的两种形态。** 上文的"变速率压缩"（Ballé 式）原为连续信号设计，搬到 token 数据要区分两种形态：

1. **无损 token 压缩 = 语言模型 + 算术编码**。离散数据的熵模型就是语言模型 $p(x_i\mid x_{<i})$（categorical/softmax），码率 $R=-\sum_i\log p(x_i\mid x_{<i})$ 即每 token 的 NLL；"变速率"自动发生——可预测 token 花少比特、意外 token 花多比特，无需显式 $\lambda$。这是香农"预测=压缩"的离散域原生形态，LLM 训练已在隐式地做；依赖建模从"超先验捕空间依赖"退化为"自回归上下文捕序列依赖"。
2. **有损 token 压缩 = token 层的 $\rho$ 残差压缩，$Z$ 不量化**。token 已是离散、无连续潜变量可量化，故 Ballé 式管线不直接作用于原始 token；有损发生在**输入 token 层**，而非其后的潜空间：

$$x \xrightarrow{\ \text{按 }\rho\text{ 丢/留}\ } \tilde x \xrightarrow{\ E\ } Z\ (\text{精华，连续}) \xrightarrow{\ g\ } \hat x.$$

丢/留与比特由冗余度 $\rho=\|v-Mk\|^2$ 决定（见下文「创新编码」）；token 通道走无损（LM 熵模型），token 层走有损，$Z$ 完整保留。

**接口契约：$Z$ 保持连续，不施加量化瓶颈 $Q$。**（注：本节讨论的语义量化器 FSQ/VQ/RVQ 仅为备选参考，**非强制约束**；主表征路径以连续 $Z$ 为准。）三条理由：

1. **$Z$ 是精华，量化即二次有损**：本节的目标是让 $Z$ 完整承载 $I(Z;Y)$；若再对 $Z$ 施加 FSQ/VQ/RVQ，等于在已压好的精华上再砍一刀，与"有损只发生在 token 层"的定位直接矛盾。
2. **§4 的 Flow 阶段要求连续潜变量**：线性路径 $z_t=(1-t)z_0+tz_1$ 与速度场 $v_\psi$ 的连续回归都要求 $z_1\in\mathbb R^{L\times d}$；$Z$ 一旦离散化，生成退化为离散序列建模，串行深度从 $O(N)$ 恶化到 $O(L)$（§4.3(a)）。
3. **§3 的推理要求连续可微**：状态依赖算子 $\Psi_h$（§3.5）与 CoT 轨迹复合（§3.7）都在连续表征上做；离散瓶颈切断梯度，破坏不动点收敛性质。

编码器 $E$ 取 Transformer 或 Mamba（SSM）均可，**输出都是连续 $Z$**——SSM 里的 "discretization"（ZOH 数值离散化）是微分方程的积分步长处理，不是把输出变成离散符号。若确有存储/传输需求，要区分两类**性质不同的量化**：

1. **语义量化（FSQ / VQ / RVQ）**：把连续 $Z$ 映射到**有限码本索引**，改变表征语义、破坏连续可微，故不作主表征路径上的常设瓶颈，但**保留为备选测试方案**——是否在某处启用、对 loss 与下游的影响如何，需实测观察后再定。两种用法：(i) 作**序列化格式层**（仅落盘/传输时启用，读入后立即反量化回连续 $Z$，要求 $\|\hat Z-Z\|$ 低于阈值）；(ii) 作**实验性瓶颈**（在 §3/§4 某处插入，观察不动点/生成质量的退化幅度，若性价比可接受则保留）。
2. **数值精度量化（FP16 / FP8 / FP4，按需选择）**：只降浮点位宽、**保留 $Z$ 仍是连续实数**，是"运算/传输/存储层面的位宽压缩"而非"语义瓶颈"。位宽越低，$Z$ 的实际比特消耗越小，代价是数值精度与可微稳定性越弱。**可分层混合精度**——编码器层、表征层（$Z$ 与 $\Psi_h$）、解码器层不必统一精度，各自独立选择；判据是"哪层参数容量大、哪层降精度对 loss 的边际影响小（性价比高）"：对容量大且对精度不敏感的层优先降位宽（FP8/FP4），对承载关键语义/长程依赖的层保留高精度（FP16/BF16）。

两者都不改变主契约"$Z$ 全程连续、语义不量化"：语义量化是可测试的**备选方案**（实测证明某处瓶颈的 loss 代价可接受才保留），数值精度量化只降位宽、不改语义、且可按层差异化精度。量化瓶颈 $Q$ 属于实现现状，按本契约应降级为可选的 I/O 组件，主表征路径以连续 $Z$ 为准。

**熵是码率信号，可约损失才是取舍信号。** 前面用 $H(Z\mid Y)$ 表述"丢什么"，但没给出"如何测什么该留"。操作性判据要区分两个角色，**熵≠显著性**：

1. **熵 → 码率分配（rate）**：$H[p]$ 决定给每个 token/潜变量多少比特（变速率压缩），这是 Shannon 的原始用途——低熵少给比特、高熵多给比特；
2. **可约损失 → 取舍/写入判据（salience）**：决定"这块信息值不值得占容量"，用 $\Delta(z)=\mathrm{KL}(p^*\Vert p_{\theta,M})-\mathrm{KL}(p^*\Vert p_{\theta,M\oplus z})$（写入前后、同一验证集上的 loss 差）。

**"测差值、别估熵"是可约损失的操作核心**：$\ell_t=\ell^*_t+r_t$，不可约 $\ell^*_t$ 即条件熵 $H(x_t\mid c)$，可约 $r_t$ 即 KL 散度。但"不可约"是**相对的**——$H(x_t\mid c)=H(x_t\mid c,M)+I(x_t;M\mid c)$，第二项正是表征/记忆能吃掉的部分。这把 §2.1 命题 1 的"有损"从绝对损失修正为"相对当前表征容量的损失"，也把 $I(Z;Y)$ 精确化：**"该留的信息"不是绝对熵，而是"当前表征容量能消掉的那部分可约损失"**。

**可约损失的五种可计算估计。** 核心要求：验证集必须是"未来/他处"的，不能是刚写进去的内容本身：

| 方法 | 公式/做法 | 成本 | 适用 |
|---|---|---|---|
| (1) 重放集差分（金标准） | 维护 buffer $B$，写前后各算一次 loss 差 | $2|B|$ 前向 | 离线/批量 |
| (2) 前视窗口（最便宜） | 写完后在接下来 $W$ 个 token 上算 loss 差 | 无需 buffer | 在线 |
| (3) 双模型差分 | 带/不带记忆两次前向的 loss 差 | (1) 的特例 | 无 buffer 对照 |
| (4) 熵分解上界 | $H_{\text{tot}}-H_{\text{alea}}=\mathrm{MI}$（dropout 采样） | 单前向 | 粗筛 |
| (5) 经验条件熵查表 | 后缀数组统计下一 token 经验分布熵 | 高 | 离线校准 |

其中 (4) 的 $\mathrm{MI}=H[\frac1K\sum_k p_k]-\frac1K\sum_k H[p_k]$ 是"更多数据能消掉多少"的上界，不完全等于"更多容量能消掉多少"。

**比特预算。** 统一单位：$\Delta$ 用 nat，除以 $\ln 2$ 得 bit；理想校准下 $\Delta(\text{bits})\approx I(M;x_{\text{future}}\mid c)$，与目标对齐；每比特收益 = $\Delta/\text{bits}(z)$，按此贪心/背包分配。判据：$\Delta(z)-\lambda\cdot\text{bits}(z)>0$ 才写。

**与本框架的两处对齐**：(1) "测差值"就是 §5 的创新 $e=\hat Z-\tilde Z$ 与 §3.7 的一致性损失 $\mathcal L_{\text{consist}}$ 的同一原理——不可约项抵消后，剩下的差值正是可约/可改进部分；(2) attention sink 反例（低熵但关键）印证"生成元识别"——种子可能低熵，但绝不可因低熵而丢弃；熵只能决定码率、不能决定去留。现成机制（DeltaNet 的 $v-Mk$、Titans 的 surprisal 门控 + momentum/decay）都是"可约损失单步估计 + 预算 + 衰减"的组合，与本节三步方法论同构。

**本文概念与本框架的映射。**

| 本文概念 | 框架已有件 | 对应 |
|---|---|---|
| token 残差 $\rho=\|v-Mk\|^2$ | delta rule 中间量 | 冗余度/可约损失的单步代理 |
| token 层丢/留掩码 | 记忆 $M$ + 掩码 gather | 按 $\rho$ 决定丢留与比特 |
| "测差值" | §5 自反馈、§3.7 $\mathcal L_{\text{consist}}$ | 不可约项抵消的同一原理 |
| 下游 loss 增量 | §5 自反馈闭环 | 失真度量 |
| 生成元/轨道 | 算子空间 $\Omega$、$\Psi_h$ | 存算子、按需生成轨道 |

**创新编码（修正）：token 层残差压缩，$Z$ 是完整精华。** 有损压缩发生在 **token 层**，不是潜空间——$Z$ 是编码器已压好的精华 $I(Z;Y)$，不该再损失；该丢的是 token 层的冗余：

```
token x ──► [写进记忆 M] ──► 残差 ρ = ‖v − Mk‖² ──► 按 ρ 决定丢/留 ──► x̃ ──► E ──► Z(精华，完整)
```

判据是**冗余度 $\rho=\|v-Mk\|^2$**（给定记忆后还重建不了多少），不是注意力——注意力与重要性**反相关**（attention sink、高频虚词注意力高却无内容，实体词注意力低却是关键）。$\rho\approx0$ 说明已被记忆吸收、可丢；$\rho$ 大说明不可预测、必须留。

**先写后压。** token 先写进记忆，再算 $\rho$ 当失真，按 $\rho$ 决定留不留、留几比特：

$$b_t=\mathrm{clamp}\big(a\log\tfrac{1}{p(\rho_t)}+c\,H(x_t\mid\text{rest}),\ b_{\min},b_{\max}\big),\qquad \mathrm{keep}_t=(\rho_t>\epsilon_\rho)\lor\mathrm{is\_sink}(t).$$

（$H_{\text{cond},t}>\epsilon_H$ 不单列：实现取 $H(x_t\mid\text{rest})\approx\rho_t/\mu$（见下文「标定」），与 $\rho_t>\epsilon_\rho$ 单调冗余；真要估计条件熵需逐 token 语言模型的 NLL，与「测差值、别估熵」相冲突，故仅在未来接入逐 token LM 时作为独立判据预留。）

$\rho$ 是 delta rule 的中间量，零额外前向。**压缩率内生自平衡**：$M$ 有空位 → 全吸收 → $\rho\approx0$ → 激进压缩；$M$ 满 → 旧信息被 decay 挤出 → 新 token $\rho$ 上升 → 自动保留更多——压缩器和记忆互为溢出区，稳态自动收敛。

**两个补丁**：(1) sink 保护——位置 0 和开头几个 token 硬编码不丢（softmax 泄洪口）；(2) ε 强制采样——每批保留 2–5% 低 $\rho$ 低熵 token，防"丢弃→学不到→loss 降不下→继续丢"的自我强化退化。**失真度量**用下游 loss 增量 $\Delta L_{\text{downstream}}=\mathbb E[\ell(x|\text{压缩后},M)-\ell(x|\text{原始},M)]$，不用重建 MSE。被丢 token 由解码器从骨架 $Z$ 随机填充——被丢的本来就是冗余，填充不改变大意。

**GPU 并行。** 判据 $\rho_t$ 是逐 token 标量，决策全是逐元素运算、无变长循环：

1. 并行写记忆 + 并行算 $\rho_t=\|v_t-M_{t-1}k_t\|^2$（记忆更新用线性注意力的**并行扫描**，串行深度 $O(\log L)$）；
2. 并行算 $b_t$（逐元素 `clamp`）与 `keep_t`（逐元素布尔掩码）；
3. 并行用 `keep_t` 掩码 gather 得到保留 token $\tilde x$；
4. $\tilde x$ 进编码器 $E$（Transformer 或 Mamba，均可并行）得连续 $Z$。

串行深度 = 记忆扫描 $O(\log L)$ + 决策 $O(1)$ + 编码器 $O(1)$，优于 Flow 的 $O(N)$、远好于自回归 $O(L)$（§4.3(a)）；$\rho$、$b$、`keep` 全 $O(1)$ 逐元素。唯一串行件仍是"编成比特流"的算术编码，本框架训练用可微代理、推理留掩码索引，两者都不碰。

**$b_t$ 与 $\epsilon_\rho$ 的标定（实现）。** 式 665 的两个量在无 next-token LM 的框架里如何取值：

1. **$p(\rho_t)$ 分布 → surprisal 用批内 z-score 代理**。$\log\frac1{p(\rho_t)}$ 需要 $\rho$ 的经验分布，但 $\rho=\|v-Mk\|^2$ 无闭式先验、且绝对量级随初始化 / `head_dim` / `beta` 漂移。操作上用**批内标准化**代理：$\log\frac1{p(\rho)}\approx(\rho-\mu)/\sigma$（$\mu,\sigma$ 为当前 batch 的 $\rho$ 均值/标准差）。它捕获的是"该 token 的 $\rho$ 偏离批内典型值多少"，即相对罕见度；未来可换滑动统计 / 软分位数得到更平稳的分布估计，但初始用批内 z-score 已够（冒烟验证：$b_t$ 落在 clamp 区间内）。
2. **$H(x_t\mid\text{rest})$ → 用 $\rho$ 自身代理**。§2.6 主张"测差值、别估熵"：条件熵 $H(x_t\mid c)$ 等价于 LM 的 NLL，本框架没有逐 token 语言模型，故不直接估计。但 $\rho$ 是单步可约损失代理，且预测误差是条件熵的上界（$\rho\ge H(x_t\mid\text{rest})$ 的代理意义），故取 $H\approx\rho/\mu$（$\rho$ 相对批内背景的倍数），与 surprisal 互补：surprisal 看"离群程度"、$H$ 代理看"绝对预测难度"。
3. **$\epsilon_\rho$ → 相对阈值，绝对阈值不可先验**。$\rho$ 的绝对量级依赖未标定的尺度，直接设固定绝对阈值会在不同 `head_dim`/初始化下失效。改用 $\epsilon_\rho=\mu+z\,\sigma$（$z$ 为相对 z-score）：$z=0$ 即批内均值，约保留半数高 $\rho$ token（冒烟测得 keep≈0.59，含 sink 与 ε 强制采样）。绝对阈值 `rho_eps>0` 仅作为下限兜底。

**中型规模 $z$ 的标定建议**：$z$ 越大越激进（保留越少），越小越保守。初始取 $z=0$（不预判分布偏斜），随后按 `keep` 比例与下游 loss 扫描 $z\in\{-0.5,0,0.5,1.0\}$；目标 keep 落在 30%–70%（配合 sink 保护与 $\epsilon$ 强制采样 2–5% 防自我强化退化）。压缩率内生自平衡（§2.6）：$M$ 满 → $\rho$ 升 → keep 自动升，稳态由 `rho_eps_keep` 与 $\epsilon_H$ 共同锚定。

**码率正则闭环（$R+\lambda D$）。** $b_t$ 不只是诊断量——把它作为码率正则项加进总损失：$\mathcal L_{\text{total}}\leftarrow\mathcal L_{\text{total}}+\lambda_\rho\,\frac1{BL}\sum_t b_t$（`rho_rate_weight`，默认 0 关闭，取在线编码器干净输入的 $b_t$ 均值）。最小化平均码率 $R$ 与重建失真 $D$ 构成经典率失真权衡，梯度经 $b_t\to\rho\to$ 记忆投影 $(k_q,k_v)$ 回传，使记忆学得更会吸收冗余（$\rho\downarrow\Rightarrow b_t\downarrow\Rightarrow$ 更省比特）。与「压缩率内生自平衡」互补：`keep` 门控决定**丢哪些**（salience），码率正则决定**留多少比特**（rate），二者合起来兑现式 665 的完整「可变码率」语义。

**自适应码率控制：$\lambda$ 的拉格朗日对偶上升（让 $\lambda$ 自调到命中 $R^*$）。** 上文把 $\lambda_\rho$ 当作固定超参（默认 0 关闭），这是「手动选一个码率权重」的特例。更原则的做法是把 $\lambda$ 视为率失真约束的拉格朗日对偶乘子，让它在线自收敛到命中目标码率 $R^*$：

$$\min_{x}\ D(x)\quad \text{s.t.}\quad R(x)\le R^*,\qquad \mathcal L(x,\lambda)=D(x)+\lambda\,(R(x)-R^*),\ \lambda\ge 0.$$

这里 $x$ 泛指编码器参数与记忆状态，$D$ 是下游失真、$R=\frac1{BL}\sum_t b_t$ 是平均码率。四条被 Lean 机械化的代数事实（[`lean-math/LeanMath/RhoCompress.lean`](file:///home/simon-ma/JEPA/lean-math/LeanMath/RhoCompress.lean)，`RhoCompress` 命名空间，全部无 `sorry`、无新增公理、纯实内积/代数）保证这条路线自洽：

1. **KKT 充分性**（`kkt_sufficient`）：若 $(x^*,\lambda^*)$ 满足「$x^*$ 极小化 $\mathcal L(\cdot,\lambda^*)$」$\wedge\,\lambda^*\ge0\,\wedge\,R(x^*)\le R^*\,\wedge\,$互补松弛 $\lambda^*(R(x^*)-R^*)=0$，则 $x^*$ 是 $\min\{D:R\le R^*\}$ 的最优解。这是「让 $\lambda$ 自调到命中 $R^*$」合法性的根基——只要内循环在当前 $\lambda$ 下把 $\mathcal L$ 压到最小，外循环把 $\lambda$ 调到让约束取等，最终解就是率失真最优。
2. **对偶超梯度方向正确**（`dual_supergradient`）：若 $x_\lambda$ 极小化 $\mathcal L(\cdot,\lambda)$，则 $(R(x_\lambda)-R^*)$ 是凹对偶 $q(\lambda)=\min_x\mathcal L(x,\lambda)$ 的超梯度：$q(\lambda')\le q(\lambda)+(\lambda'-\lambda)(R(x_\lambda)-R^*)$。因此「$R>R^*\Rightarrow$ 增大 $\lambda$、$R<R^*\Rightarrow$ 减小 $\lambda$」是正确的上升方向。
3. **投影更新的不动点 $=$ KKT 互补松弛**（`update_fixedpoint_implies_cs`）：用投影超梯度上升
   $$\lambda\leftarrow \max\!\big(0,\ \lambda+\eta\,(R(x_\lambda)-R^*)\big),\qquad \eta>0,$$
   则任何不动点 $\lambda^*$ 自动满足 $R(x^*)\le R^*$ 且 $\lambda^*(R(x^*)-R^*)=0$；再由 `kkt_sufficient` 即得最优。迭代收敛到不动点是投影超梯度上升的标准结论（凹对偶 $+$ Lipschitz 超梯度，列为已知标准结果，未在 Lean 中形式化收敛速率）。
4. **鞍点刻画**（`saddle_implies_kkt`、`saddle_implies_primal_optimal`）：$\mathcal L$ 的鞍点恰是约束问题的解——原极小 / 对偶极大同时成立。

单调性根基（梯度方向确实指向「省比特」）由同文件的实现侧定理保证：`keep_antitone_in_z`（$z\uparrow\Rightarrow$ keep 更难满足，$z$ 是有效压缩旋钮）、`b_monotone_in_rho` 与 `meanRate_monotone_in_rho`（$\rho\downarrow\Rightarrow b\downarrow\Rightarrow R\downarrow$，记忆吸收冗余则码率下降）。残差几何侧 `rho_nonneg` / `rho_zero_iff` / `rho_pythagorean`（$\rho=0\Leftrightarrow v$ 被精确重建；预测与残差正交时 $\|v\|^2=\|p\|^2+\rho$）给出 $\rho$ 作为冗余度的语义正当性。

**与本契约的兼容性**：自适应 $\lambda$ 只调整码率正则项的权重（标量 $\lambda_\rho$ 从常数变成在线更新量），**不改动 token 层的丢/留机制（式 665）、不改动 $b_t$ 的标定、不对潜空间 $Z$ 施加任何量化瓶颈**。因此 $Z$ 全程连续、反塌缩约束（§2.1）与 §3/§4 的连续可微要求均不受影响——$\lambda$ 是损失函数上的一维旋钮，不是表征路径上的结构改动。固定 $\lambda$（当前实现，`rho_rate_weight` 默认 0）是该方案 $\lambda$ 取常数的退化情形；将其升级为在线更新只需在 `training/losses.py` 的码率正则处用批内 $R$ 的滑动均值驱动上式，属后续可选实现，本轮仅固化数学契约与形式化验证。

**记忆维度 $D$ 的扩展策略。** 记忆矩阵 $M$ 的 k/v 维度 $D$（=`rho_head_dim`，默认 64）类比 Transformer 的 head_dim：它是**冻结的单头宽度**，不随主干宽度 $d_{\text{hidden}}$ 缩放（中型 0.3b–7b 全程 $D=64$）。$M$ 是 $D\times D$ 方阵、秩 $\le D$，故 $D$ 决定「单头容量」。$D$ 太小只降低压缩率（$M$ 易满 → $\rho$ 上升 → 多保留），不损害正确性；需要扩展的是「总容量」，靠**加 head**（多份独立 $M$，各 $D=64$）而非放大 $D$——放大 $D$ 是 $O(D^2)$ 显存平方增长，加 head 是线性。

**并行路径选择。** 因 $D=64$ 冻结，精度优先路径（矩阵仿射并行扫描，直接矩阵连乘、无求逆，显存 $O(BLD^2)$）在 $D=64$ 下可接受，故为默认实现（`psco.associative_scan_mat`）。显存优先路径（DeltaNet chunked / WY 低秩，$O(BLD)$）仅在 $D$ 需放大时才有必要，现仅记录不实现。

**自推方法论（三步，复用本框架已有件）。**

1. **不可压缩性检测**：读 token 层的条件熵 $H(x_t\mid c)$（等价于 LM 的 NLL）与残差 $\rho_t$，超阈值判为"种子"（定理）而非"描述"，决定走 IB 还是走生成元路线；
2. **存储生成元而非轨道**：种子存入 $\Omega$ 作为生成元，轨道由 $\Psi_h$ 按需生成，码率封顶在 $H(S)$；
3. **自适应码率兜底**：半压缩内容（有冗余但不多）回落到变速率熵模型，纯种子则不再尝试压下去。

一句话：**可压缩内容丢 $H(Z\mid Y)$、留 $I(Z;Y)$（IB）；不可压缩内容本身就是 $I(Z;Y)$ 的种子、丢无可丢，正确操作是识别成生成元、存算子按需生成轨道。有损只发生在 token 层（按 $\rho$ 丢冗余），潜空间 $Z$ 全程连续、不设量化瓶颈。**

**多模态下的私有记忆 M：模态内有损、跨模态统一在 Z。** 当通道多于一个（图像、文本、触觉、控制…）时，记忆 $M$ 的归属决定扩展性。结论是 **$M$ 模态内私有，跨模态统一留给 $Z$**：$\rho=\|v-Mk\|^2$ 判的是「观测冗余」（同一模态内已被记忆吸收的信息），而「语义冗余」（图像和文本说同一件事）应靠 §3.2 的对齐损失在 $Z$ 层挤掉，不该提前到 token 层——不同模态的 token 语义空间不同、不可直接比，共享 $M$ 会容量争抢 + 灾难性遗忘：

```
触觉 x ─► M_触觉(私有) ─► ρ ─► 丢/留 ─► x̃ ─► E(共享主干) ─┐
视觉 x ─► M_视觉(私有) ─► ρ ─► 丢/留 ─► x̃ ─► E(共享主干) ─┼─► Z(跨模态共享潜空间)
文本 x ─► M_文本(私有) ─► ρ ─► 丢/留 ─► x̃ ─► E(共享主干) ─┘
```

这正是 §2.8「薄接口 + 厚共享主干」的实例：物理翻译层与私有 $M$ 是「薄」的（模态特化），主干 $E$ 与潜空间 $Z$ 是「厚」的（全模态共享）。新模态接入 = 薄前端 + 私有 $M$ + 头，纯增量，共享主干与潜空间不动；接入顺序与锚定策略见 §3.4 的 PAC。

### 2.7 视觉通道 $R_I$ 与马赛克同构（plan P2）

§2.3–2.5 只落地了文本通道 $T=g(Z)$。本节把视觉通道 $R_I$ 的最小实例——**马赛克**——补进来，说明它与文本通道在数学上同构，但损失机制不同。

**马赛克通道。** 令马赛克算子 $\phi$ 为「下采样 + 分块平均」：把高清图 $I\in\mathbb R^{H\times W\times C}$ 分成 $B\times B$ 块、每块取均值，得到

$$M = \phi(I) \in \mathbb{R}^{\frac{H}{B}\times\frac{W}{B}\times C}.$$

$\phi$ 与文本量化器 $g$ 同属**多对一映射**：多张不同高清图共享同一张马赛克，故

$$H(I \mid M) > 0. \tag{16}$$

纤维 $\phi^{-1}(M)=\{I:\phi(I)=M\}$ 内的变异不可见——马赛克只报告分块均值（低频），块内高频细节丢失。这与文本纤维 $\mathcal F_t=g^{-1}(t)$ 是同一结构（式 6–7）。

**三层次同构**（对齐式 6–7、式 15、式 41）：

| 层次 | 文本通道 | 马赛克通道 | 数学对象 |
|---|---|---|---|
| 映射 | $T=g(Z)$，$H(Z\mid T)>0$ | $M=\phi(I)$，$H(I\mid M)>0$ | 纤维 $\mathcal F_t$ / $\phi^{-1}(M)$ |
| 逆问题 | $p_\phi(Z\mid T)$ | $p(I\mid M)$ | 后验 $\propto$ 先验 × 一致性（式 41） |
| 信息 | 语义 $I(T;Y)$ vs 措辞 $H(T\mid Y)$ | 结构 vs 高频细节 | 式 15；还原恢复 $Y$，细节是幻觉 |

**关键区别（不可混同）。**

> **命题 5（通道不对称的中间地带）**：马赛克是连续→连续降采样，失真随块大小 $B\to 1$（码率上升）而趋于 $0$（式 11 的 $R_{\text{vis}}(D)$ 情形）；文本是连续→离散有限码本，固定 $L,|\mathcal V|$ 下失真永远高于地板 $D^\star>0$（式 10）。马赛克是 $R_I$ 的低码率切片，补上"近无损 ↔ 离散码本"之间的整段过渡。

含义：视觉通道 $R_I$ 无需离散码本即可有损——连续降采样本身就产生正条件熵 $H(I\mid M)>0$；文本的损失机制是**离散码本基数**。一个靠分辨率（连续尺度），一个靠基数（离散计数）。这正是命题 4 中两类通道不对称的根源。

### 2.8 规模不变性：信息论结论跨规模的统一性验证与数值取舍

§2.3–2.5 的结论（有损性 $H(Z\mid T)>0$、失真地板 $D^\star>0$、噪声熵源）是**结构性的**，不依赖任何具体数字；但其中的上界 $\le L\log_2|\mathcal V|$、量化熵增量 $Ld\log_2(1/\delta)$、噪声熵 $h(z_0)=\frac{Ld}{2}\log_2(2\pi e)$ 都显式含 $(L,d,|\mathcal V|)$。本节把这三个超参数在 §2.2.7 的规模表上扫一遍，验证结论**是否跨规模统一**，并给出**数值取舍**。

**规模表**（$L,d$ 取自 §2.2.7；$|\mathcal V|$ 冒烟为合成语料字符级、大模型为 LLaDA 约 12.6 万）：

| 规模 | $L$ | $d$ | $|\mathcal V|$ | $\log_2|\mathcal V|$ |
|---|---|---|---|---|---|
| tiny | 16 | 64 | ~20 | 4.32 |
| small | 16 | 128 | ~20 | 4.32 |
| base | 64 | 256 | ~20 | 4.32 |
| 0.3B | 1024 | 128 | $1.26\times10^5$ | 16.94 |
| 1B | 1024 | 256 | $1.26\times10^5$ | 16.94 |
| 3B | 4096 | 384 | $1.26\times10^5$ | 16.94 |
| 7B | 4096 | 512 | $1.26\times10^5$ | 16.94 |

**结论只依赖两个无量纲比值**（都与 $L$ 无关，因为 $L$ 在分子分母中抵消）：

**比值 A：有损裕度 $m = \dfrac{d\,\log_2(1/\delta)}{\log_2|\mathcal V|}$**（$\delta$ 取 $10^{-2}$，$\log_2(1/\delta)\approx 6.64$）。它度量式 (6) 中「潜空间量化熵 vs 文本码率」的差：$m>1$ 时 $H(Z_\delta\mid T)>0$。

**比值 B：噪声熵预算 $r = \dfrac{h(z_0)/L}{H(T)/L}\approx\dfrac{d\cdot 2.047}{\log_2|\mathcal V|}$**。它度量 §4.4 中「噪声端熵源 vs 文本端码率」的比。⚠️ 注意 $h(z_0)$ 是**微分熵**（相对量、可为负），不是绝对 bit 数，与离散熵 $H(T)$ 不可直接同单位相除；本比值是**启发式**，其严格版本是走式 (4) 的量化桥：固定与文本分辨率匹配的 $\delta$ 后 $H(Z_{\delta,0})\approx h(z_0)+Ld\log_2(1/\delta)\gg H(T)$，结论不变且更强。

| 规模 | 有损裕度 $m$ | 噪声/文本 $r$ | $H(T)\le L\log_2|\mathcal V|$ | $h(z_0)=Ld\cdot2.047$ |
|---|---|---|---|---|
| tiny | 98× | 30× | 69 bit | 2096 bit |
| small | 197× | 61× | 69 bit | 4192 bit |
| base | 394× | 121× | 277 bit | 33.5 kbit |
| 0.3B | 50× | 15× | 17.3 kbit | 268 kbit |
| 1B | 100× | 31× | 17.3 kbit | 536 kbit |
| 3B | 151× | 46× | 69.4 kbit | 3.22 Mbit |
| 7B | 201× | 62× | 69.4 kbit | 4.29 Mbit |

**结论统一性验证。** 两个比值在所有规模下都 $\gg 1$（$m\in[50,394]$、$r\in[15,121]$），且 $\delta$ 即使放宽到 $10^{-1}$（$\log_2(1/\delta)=3.32$）仍满足 $m\gg1$。因此：

- 命题 1/4（$H(Z\mid T)>0$、$D^\star>0$）：在**所有规模**成立，且裕度随 $d/\log_2|\mathcal V|$ 单调增长；
- §4.4 的「噪声熵预算 ≫ 文本码率」：在所有规模成立，比值 $r\in[15,121]$，即 **1～2 个数量级**（原文「约一个数量级」是冒烟低端值 $r\approx10$ 的低估）；
- 词表从字符级 → 12.6 万时 $\log_2|\mathcal V|$ 只从 $4.3\to16.9$（约 4×），而 $d$ 同步从 $64\to512$（8×），故换大词表**不会破坏**结论，反而在 7B 档把 $m$ 拉回 $\sim200×$。

**数值取舍（设计判据）。**

1. **唯一关键不等式是 $d \gg \log_2|\mathcal V|$**（更精确地 $d \gg \log_2|\mathcal V|/\min\{2.047,\ \log_2(1/\delta)\}$，约 $d\gtrsim 2\log_2|\mathcal V|$）。它同时保证有损性与噪声熵源两条结论。$\log_2|\mathcal V|\in[4,17]$，而 $d\in[64,512]$，富余 $15$～$400×$，无任何规模触碰红线。
2. **$r=d_{\text{hidden}}/d=8$ 是安全的**：$r$ 的下限由「$d$ 不能逼近 $\log_2|\mathcal V|$」决定（7B 档安全下限 $d_{\min}\approx 2\log_2|\mathcal V|\approx 34$，$r=8$ 给的 $d=512$ 富余 15×），故 $r$ 可在 $[4,16]$ 内自由调而结论不变。
3. **$\delta$ 只是分析参数**：$Z$ 全程连续（§2.6），$\delta$ 不进入训练；更细的 $\delta$ 只增大 $m$，结论不依赖 $\delta$ 的精确取值。
4. **$L$ 只定总量、不定方向**：$L$ 同时缩放 $H(T)$ 与 $h(z_0)$，在比值中抵消，不改变定性结论；它决定绝对预算（7B 档 $h(z_0)\approx4.3$ Mbit），对应「文字高压缩、潜空间高冗余」的直觉。

> **结论**：§2.3–2.5 的信息论结论是**规模不变的**——有损性、失真地板、噪声熵源在 tiny 到 7B 全部成立，且安全裕度只随 $d/\log_2|\mathcal V|$ 增长。唯一需要遵守的数值判据是 $d\gg\log_2|\mathcal V|$（几乎自动满足），故跨规模缩放时无需为维持这些结论而引入额外约束。

### 2.9 替代结构：CARO——把「按 ρ 丢/留 → 落入 Z → 在 Z 中对齐并推理」折叠为单一门控结合扫描态

> **动机**。§2.6 的压缩器（记忆 $M$，线性注意力 / DeltaNet 的矩阵仿射扫描）、编码器 $E$（Transformer / Mamba）、§3.7.3 的推理算子 PSCO，**三者底层是同一个仿射结合扫描幺半群**（$x\mapsto a x+b$，见 `Affine.lean` `comp_assoc`）：压缩器把 token 写进一个仿射态、$E$ 又对上下文重新聚合一遍、PSCO 再在 $Z$ 上迭代仿射不动点。三次走过同一代数结构是冗余的。本节约束一个**单一门控结合扫描算子 CARO**（Context-addressed Associative-scan Reasoning Operator），用一个线性态同时承担「压缩 / 表征 / 推理」，替代「按 $\rho$ 丢/留 $\to$ $E$ $\to$ $Z$ $\to$ 对齐并推理」这一整段，在 GPU 的五个复杂度维度上追平或超越 Transformer / Mamba，同时在长程上维持并超越内容可寻址注意力。

#### 2.9.1 定义

令键相似度双线性型 $B:V\times V\to K$（线性注意力的核 $k\cdot q$），外积算子 $\mathrm{outer}_B(k,v):q\mapsto B(k,q)\,v$。**CARO 状态**是一个线性映射 $S_t:V\to V$，由三步构成：

$$
\boxed{\ S_{t+1}=a_t\,S_t+g_t\,\mathrm{outer}_B(k_t,v_t),\qquad z_t=S_t\,q_t,\qquad \rho_t=\|v_t-S_{t-1}\,k_t\|^2\ }
$$

- **状态更新（压缩 + 表征）**：$a_t\in(0,1)$ 是遗忘/衰减（谱夹层中 $\delta<a_t<1-\gamma$，见 §3.7.3）；$g_t\in[0,1]$ 是由残差 $\rho_t$ 驱动的**写入门控**——$\rho_t\approx0$（已被历史吸收）则 $g_t\approx0$（不写，等价于「丢」），$\rho_t$ 大则 $g_t\approx1$（写，等价于「留」）。于是「按 $\rho$ 丢/留」不再是显式掩码 gather，而是连续门控 $g_t$，§2.6 GPU 并行的第 3 步（`keep_t` 掩码 gather 保留 token）被消掉。
- **读出（表征落入 $Z$）**：$z_t=S_t\,q_t$。把 $S_t$ 按递推展开即得核心恒等式

$$
z_t=S_t\,q_t=\sum_{i<t}\Big(\prod_{j=i+1}^{t-1}a_j\Big)\,g_i\,B(k_i,q_t)\,v_i . \tag{CARO-1}
$$

这是对**全部历史**按「键相似度 $B(k_i,q_t)\times$ 写入强度 $g_i\times$ 累积衰减」的加权聚合——即**精确的内容可寻址长程检索**，由一个常数大小的线性态 $S_t$ 实现，无需 $O(L^2)$ 注意力矩阵。$Z=(z_1,\dots,z_L)$ 即编码后的连续表征（$Z$ 全程连续，见契约核对）。

- **残差（新奇度 / 压缩信号）**：$\rho_t=\|v_t-S_{t-1}k_t\|^2$。由 (CARO-1)，$S_{t-1}k_t$ 本身就是「历史按键相似度 $B(k_i,k_t)$ 的加权重建」，故

$$
\rho_t=0\iff v_t=\sum_{i<t}\Big(\prod_{j=i+1}^{t-1}a_j\Big)\,g_i\,B(k_i,k_t)\,v_i\iff v_t\text{ 被历史精确重建}\iff\text{该 token 冗余}. \tag{CARO-2}
$$

这把 §2.6 的「$\rho$ 判冗余」变成状态直接给出的连续信号，并与写入门控 $g_t$ 闭环（$\rho\uparrow\Rightarrow g\uparrow$，压缩率内生自平衡，同 §2.6）。

- **推理（在 $Z$ 中对齐并推理）**：给定读出的 $Z$，推理阶段固定 $(a,b)$ 做仿射迭代

$$
S^{(n+1)}=a\,S^{(n)}+b,\qquad a\in(\delta,1-\gamma)\subset(0,1) \tag{CARO-3}
$$

（线性骨架即 §3.7.3 PSCO 的 $\tilde A_h Z+b_h$，此处取标量/对角 $a$；$b$ 输入依赖，由隐含模式 $h$ 投影，见 §3.5）。$|a|<1$ 时收缩到唯一不动点

$$
S^*=(1-a)^{-1}b,\qquad S^{(n)}-S^*=a^n(S^{(0)}-S^*)\xrightarrow{n\to\infty}0. \tag{CARO-4}
$$

#### 2.9.2 与既有流水线的对应（折叠了什么）

| 既有段 | 原实现 | CARO 对应 |
|---|---|---|
| 按 $\rho$ 丢/留（§2.6） | 记忆 $M$ + 残差 $\rho$ + `keep` 掩码 gather | $g_t=\sigma(\rho_t)$ 连续写入门控；$\rho_t$ 由同一状态读出，零额外前向 |
| 编码器 $E$（Transformer / Mamba） | 对保留 token $\tilde x$ 重聚合上下文得 $Z$ | 读出 $z_t=S_t q_t$（CARO-1）已聚合全部历史，$E$ 的二次聚合被消除 |
| 在 $Z$ 中对齐并推理（§3 / §3.7.3 PSCO） | PSCO 在 $Z$ 上迭代仿射不动点 | 同一状态迭代 (CARO-3)，复用 `Affine.lean` 幺半群与 `Contractive.lean` 收缩 |

一句话：**压缩（$\rho$ / $g$）、表征（$z=Sq$）、推理（$S^{(n)}\to S^*$）是同一个仿射态的三种读法**，不再是三个串行模块。

#### 2.9.3 复杂度对比（追平 / 超越 Transformer 与 Mamba）

记序列长 $n$、表征维 $d$、CARO 单头键/值维 $D$（冻结 64，同 §2.6 `rho_head_dim`，靠加 head 扩容量）。

| 维度 | Transformer | Mamba (SSM) | **CARO（本结构）** | 说明 |
|---|---|---|---|---|
| 训练计算 | $O(n^2 d)$ | $O(n\,d\,N)\approx O(n\,d)$ | **$O(n\,d\,D)\approx O(n\,d)$** | 同 Mamba；$D=64$ 冻结故与 $d$ 解耦，线性于 $n$ |
| 训练显存 | $O(n^2)$（注意力矩阵） | $O(n)$（线性） | **$O(n\,D^2)$ 或 $O(n\,d)$** | 线性于 $n$；矩阵路径 $D=64$ 可接受，低秩路径 $O(n\,d)$（同 §2.6） |
| 推理单步 | $O(n\,d)$（KV cache 随上下文增长） | $O(1)$（仅依赖上一步隐状态） | **$O(1)$** | $S_t$ 是定常线性态，单步只更新 $S$，不依赖历史长度 |
| 推理总耗 | $O(n^2 d)$ | $O(n)$ | **$O(n)$** | 同 Mamba；每步 $O(1)$，$n$ 步共 $O(n)$ |
| 推理显存 | $O(n\,d)$（KV cache） | $O(1)$（常数状态） | **$O(1)$** | 只存当前态 $S$，无 KV cache |

> **诚实声明**：上表的 $O(\cdot)$ 是**设计断言，未在 Lean 中形式化**（复杂度不是 mathlib 已机械化的对象）。「追平 Mamba / 超越 Transformer」指**渐进复杂度**；常数因子、kernel 融合质量、真实 GPU 墙钟时间需实测。参数量 / 显存 / 算力在 tiny–7B 三档的伸缩按 §2.2.7 规模表与 AGENTS 规则执行——$D=64$ 冻结、靠加 head 扩容量，保证第 1 档纯 CPU 可跑（CARO 无 GPU-only 依赖，不进入 `pyproject.toml` 可选 extra）。

#### 2.9.4 长程注意力：维持并超越

- **维持 Transformer 的内容可寻址性**：(CARO-1) 说明 $z_t=S_t q_t$ 精确等于「历史按 $B(k_i,q_t)$ 加权聚合」——这正是注意力 $\mathrm{softmax}(QK^\top)V$ 的**线性化内核形式**（$B$ 替代 softmax），在**线性状态**中实现，无需物化 $n\times n$ 矩阵。因此 CARO 具备 Transformer 「按内容检索任意历史 token」的能力（needle-in-haystack 可解）。
- **超越 Mamba 的长程**：Mamba 的 SSM 递推 $h_t=A h_{t-1}+B x_t$、读出 $y_t=C h_t$ 是**通道/位置固定**的线性递推，不含查询 $q_t$ 对历史的**内容选择**——长程 needle 检索是其已知弱点。CARO 的读出 $S_t q_t$ 显式含 $q_t$，每步按当前查询对全历史做内容加权，故在长程内容寻址上**严格强于 Mamba**。
- **软化 caveat（非形式化）**：$B(k,q)$ 是双线性核，等价于线性注意力；要达到 softmax-attention 的匹配质量需用正核近似（$\exp(k\cdot q)$ 等）或多尺度。这是**工程近似，未在 Lean 中证明**；恒等式 (CARO-1) 对任意双线性 $B$ 精确成立，但 softmax 质量不在形式化范围内。

#### 2.9.5 契约核对（对照 AGENTS 规则与 math.md 既有前提）

1. **$Z$ 保持连续、不引入语义量化瓶颈**：$z_t=S_t q_t$ 是线性态的线性读出，全程 $\mathbb R^d$ 连续值；不施加 FSQ/VQ/RVQ（同 §2.6 接口契约）。数值精度量化（FP8/FP4）只降位宽、不改语义，且可分层（同 §2.6）。✅
2. **反塌缩**：(CARO-4) 不动点 $S^*=(1-a)^{-1}b$ **输入依赖**——不同 $b$ 得不同 $S^*$（`no_collapse`），不会坍缩到单一状态；谱夹层 $a\in(\delta,1-\gamma)$ 封死 $a\to0$ 一步塌缩（同 §3.7.3 五道防线第 1 条）。✅
3. **$d:d_{\text{hidden}}=1:8$**：CARO 表征读出维 $d$ 与主干 $d_{\text{hidden}}$ 维持 §2.2.7 / §2.8 的 $r=8$ 不变；键/值维 $D=64$ 冻结不随 $d$ 缩放（同 §2.6 记忆维度扩展策略）。✅
4. **通路 / 关系分离（§4.5）**：CARO 的状态更新（关系建模，$O(n)$）与通路映射（逐位置读出投影，$O(1)$ 深度）仍分离——读出 $z_t=S_t q_t$ 是关系聚合，其后的读出头 / 量化是逐位置通路，互不混淆。关系从 $O(n^2)$ 降到 $O(n)$，通路仍 $O(L)$ 独立，符合 §4.5 的分离原则。✅
5. **与 §3 对齐推理的衔接**：CARO 折叠的是「编码 → 落入 Z」这一段的表征获取；§3 的硬对齐（§3.1）、交叉掩码（§3.2）、PAC（§3.4）在 $Z$ 之上仍照常进行——CARO 产出连续 $Z$，不改变 §3 的契约，只是把 $Z$ 的生产者从「$E$ 重聚合」换成「同一状态读出」。✅

#### 2.9.6 Lean 形式化（见 §1.0 对照表，全部无 sorry / 无新增公理）

恒等式 (CARO-1)–(CARO-4) 与反塌缩已在 [`lean-math/LeanMath/AssociativeMemory.lean`](file:///home/simon-ma/JEPA/lean-math/LeanMath/AssociativeMemory.lean)（命名空间 `AssocMem`）机械化：

| 断言 | 定理 |
|---|---|
| 更新是仿射映射 ⇒ 复用 `PSCO.Aff` 结合幺半群 ⇒ 并行扫描合法 | `update_is_affine` |
| (CARO-1) 读出 = 历史键相似度加权和（内容可寻址长程检索） | `stateFrom_readout` |
| (CARO-2) 残差 = 预测误差，$\rho=0\iff$ 被历史重建 | `prediction_is_content_sum` / `residual_zero_iff_reconstructed` |
| (CARO-3)(CARO-4) 不动点公式与误差恒等式，$|a|<1$ 收敛 | `fixedPoint_is_fixed` / `iterAff_error` / `real_pow_tendsto_zero` |
| 反塌缩：不同输入 $b$ ⇒ 不同不动点 | `no_collapse` |

> **形式化边界（诚实）**：Lean 证明的是**代数恒等式与收敛性**（任意域 $K$、任意模 $V$、任意双线性型 $B$）。**未形式化**的部分：复杂度 $O(\cdot)$、GPU 友好性、softmax 核近似质量、「超越 Mamba」的经验断言——这些是设计分析，见 [lean-math/README.md](file:///home/simon-ma/JEPA/lean-math/README.md)「发现的问题」。

---

### 2.10 替代结构：Looped Encoder——把「$E$ 主干 $\to$ $Z$」的前馈链折叠为权重共享的内部循环

> **动机**。§3.7.1 已把 Loop Transformer / DEQ 作为 **$Z$ 空间推理算子 $\Psi_h$** 的参考锚点：编码完成后，在 $Z$ 上把同一算子迭代到不动点。本节讨论一个**正交**的替换：把**编码器主干 $E\to Z$ 本身**（§2.2 中 $N$ 个互不相同的 Transformer/Mamba 块 $B_N\circ\cdots\circ B_1$ 加读出头投影到 $Z$）折叠为**权重共享的内部循环**——同一块 $B_\theta$ 迭代 $T$ 次再读出。这是现实中已有的结构：ALBERT（Lan et al. 2020，跨层参数共享）、Universal Transformer（Dehghani et al. 2019，自适应停步 ACT）、MR-Loop / Looped Transformer（Fan et al. 2025, ICLR 2025，输入注入的循环）、DEQ（Bai et al. 2019，不动点 + 隐式微分）。用户问的「在内部建立计算循环以优化空间使用或计算」正是这一类；本节给出其收益/代价的数学推导，并把核心断言形式化到 Lean（[`lean-math/LeanMath/LoopedEncoder.lean`](file:///home/simon-ma/JEPA/lean-math/LeanMath/LoopedEncoder.lean)）。
>
> **与 §3.7.1 的关键区别（勿混）**：§3.7.1 循环的是**推理算子 $\Psi_h$**（在 $Z$ 上、输入依赖、$h$ 自推断）；本节循环的是**编码器主干**（$\text{输入}\to Z$、对每个样本权重固定）。二者可叠加：先用回环编码器产出 $Z$，再在 $Z$ 上用 $\Psi_h$ 推理。本节替换的是「$E$ 主干 $\to$ 落入 $Z$」这一段的**参数组织方式**，不改 $Z$ 的契约，也不改 §3。

#### 2.10.1 定义

记序列长 $L$、主干隐藏维 $d_h$、潜空间维 $d$（$d:d_h=1:8$，见 §2.2.7）。令 $e=\mathrm{embed}(x)+\mathrm{pos}\in\mathbb R^{L\times d_h}$ 为前端嵌入（含位置编码），单块参数数为 $p$。

**前馈基线**（§2.2 现状）：$N$ 个互不共享的块 $B_i$（参数 $\theta_i$，$|\theta_i|=p$）串接，最后线性头投影到 $Z$：

$$
h_0=e,\quad h_{i+1}=B_i(h_i)\ (i=0,\dots,N-1),\qquad Z=W_{\mathrm{out}}\,h_N\in\mathbb R^{L\times d}. \tag{LE-1}
$$

总参数 $P_{\text{ff}}=N\cdot p+P_{\text{embed}}+P_{\text{head}}+P_{\text{pos}}$（embedding / 头 / 位置编码不参与层间共享）。

**回环编码器（MR-Loop 形式）**：单一共享块 $B_\theta$ 迭代 $T$ 次；每步把输入嵌入 $e$ 重新注入（input injection），再读出：

$$
\boxed{\ h^{(0)}=0,\quad h^{(t)}=B_\theta\big(h^{(t-1)}+e\big)\ (t=1,\dots,T),\qquad Z=W_{\mathrm{out}}\,h^{(T)}\ } \tag{LE-2}
$$

总参数 $P_{\text{loop}}=p+P_{\text{embed}}+P_{\text{head}}+P_{\text{pos}}$（块参数只存一份，与 $T$ 无关）。

> **为什么必须输入注入（MR-Loop 而非纯环 $h^{(t)}=B(h^{(t-1)})$）**：若 $B$ 的线性部分 $M$ 收缩（$\rho(M)<1$，见 §2.10.4），纯环的不动点 $h^*$ 满足 $h^*=Mh^*$，即 $h^*=0$，**与初值 $h^{(0)}$ 无关**——也就与输入 $x$ 无关，$Z$ 坍缩到常数（违反 §2.2.4 / §2.6 反塌缩）。输入注入 $h^{(t)}=Mh^{(t-1)}+Me$ 把 $e$ 作为**仿射常数项 $b=Me$** 每步重灌，不动点变为 $h^*=(I-M)^{-1}Me$，**显式依赖 $e$**。这是回环编码器可用的前提，也是 `mrLoop_no_input_collapse`（§2.10.7）要守住的契约。

#### 2.10.2 线性化骨架与收敛（DEQ 极限）

把块 $B_\theta$ 在不动点处线性化，记其线性部分为 $M\in\mathbb R^{d_h\times d_h}$（残差注意力块的 $B(h)=h+Mh+\text{非线性}$，此处取线性骨架；非线性项的处理见形式化边界）。MR-Loop 的线性骨架即仿射迭代

$$
h^{(t)}=M\,h^{(t-1)}+b,\qquad b=M\,e, \tag{LE-3}
$$

与 §3.7.3 PSCO、§2.9 CARO 的仿射不动点迭代**同构**（同一 `Affine.lean` 幺半群）。当 $M$ 严格收缩（算子范数 $\|M\|\le K<1$，或谱半径 $\rho(M)<1$，谱夹层见 §3.7.3）时：

1. **存在唯一不动点** $h^*=Mh^*+b$，由 Neumann 级数给出

$$
h^*=(I-M)^{-1}b=\sum_{j=0}^\infty M^j b,\qquad b=Me. \tag{LE-4}
$$

2. **误差几何衰减**（线性推广 `AssocMem.iterAff_error`，见 `affine_iter_error` / `mrLoop_error`）：

$$
h^{(t)}-h^*=M^t\big(h^{(0)}-h^*\big),\qquad \|h^{(t)}-h^*\|\le K^t\,\|h^{(0)}-h^*\|\xrightarrow{t\to\infty}0. \tag{LE-5}
$$

3. **DEQ 极限**：$T$ 足够大时 $h^{(T)}\approx h^*$，$Z=W_{\mathrm{out}}h^*$ 与 $T$ 无关（收益递减，可早停）。训练反传可用 DEQ 隐式微分（§3.7.1 式 (25)，此处 $J=M$），不展开 $T$ 步 rollout。

> 这正是「优化空间使用」的数学根因：权重共享使**跨循环的梯度检查点 / 隐式微分**合法——前向只需存当前态 $h^{(t)}$ 与不动点 $h^*$，激活显存从 $O(T\cdot L\cdot d_h)$ 降到 $O(L\cdot d_h)$（检查点）乃至 $O(1)$（DEQ 只存 $h^*$）。详见 §2.10.3 对比表。

#### 2.10.3 收益与代价：参数 / 算力 / 串行深度 / 显存

记单块前向算力 $C_{\text{block}}$（FLOPs）、$K=\|M\|<1$ 为收缩率。下表对比前馈基线（深度 $N$）与回环（迭代 $T$、单层共享）。

| 维度 | 前馈 $E_{\text{ff}}$（$N$ 层，不共享） | 回环 $E_{\text{loop}}$（$T$ 迭代，单层共享） | 说明 |
|---|---|---|---|
| **参数量**（容量代价） | $N\cdot p+P_{\text{前端}}$ | $p+P_{\text{前端}}$ | 同有效深度 $T=N$ 时主干参数 $\div N$（`loop_uses_fewer_params`，$N\ge2$）；前端 embed/head/pos 不共享，是省不掉的接口成本 |
| **算力 FLOPs** | $N\cdot C_{\text{block}}$ | $T\cdot C_{\text{block}}$ | 同深度 $T=N$ 时**持平**；省下的参数预算可加**宽**单层（更大 $p$）或加**深** $T$，换取「同参更深」。渐进持平已形式化：`eval_isBigO_T`（前向 $=O(T)$）、`checkpoint_eval_isBigO_T`（检查点总 $2T=O(T)$，仅常数 2 重算） |
| **串行深度**（墙钟关键路径） | $N$ | $T$ | 权重共享**不减少**串行步数，仍是 $T$ 步串行——不能加速单样本前向（并行度不变） |
| **训练显存（激活）** | $O(N\cdot L\cdot d_h)$（无检查点） | 朴素 $O(T\cdot L\cdot d_h)$；**检查点 $O(L\cdot d_h)$**；**DEQ 隐式 $O(1)$** | 空间优化的核心收益：共享权重使跨循环检查点 / 不动点反传合法。已形式化：朴素 $=O(T)$（`baseline_memory_isBigO_T`，且无界 `baseline_memory_unbounded` ⇒ 非 $O(1)$）、检查点/DEQ $=O(1)$（`checkpoint_memory_isBigO_one` / `deq_memory_isBigO_one`）、轨迹可由 $(f,h_0,T)$ 重建不丢信息（`trajectory_determined`）、DEQ 极限态收敛到不动点（`mrLoop_tendsTo_fixedPoint`） |
| **权重访存**（内存带宽瓶颈） | 每步读 $N$ 套不同权重 | 每 $T$ 步只读 $1$ 套权重（可驻留 SRAM/缓存） | 内存带宽受限的加速器上，共享权重显著降低访能、提升吞吐——这是「优化计算」的另一面 |
| **推理** | 固定 $N$ 步 | 可自适应停步（ACT）；收缩时 $T$ 增大收益递减 | 类似 Universal Transformer；$T$ 可按样本难度分配 |
| **表达力（同深度 $T=N$）** | 各层独立（全函数空间） | 权重绑定（对角子集） | 回环函数族 $\subsetneq$ 同深度前馈函数族（`looped_eq_feedforward_tied`），见 §2.10.5 |
| **表达力（同参数预算）** | 深度受限 $N\approx P/p$ | 可设 $T\gg N$（同参数下更深） | 「同参更深」潜在收益；但受 §2.10.4 梯度衰减限制，非免费 |
| **优化难度**（梯度代价） | 各层雅可比独立，可逐层放大/保持 | 共享权重 $\Rightarrow$ 回传 $\sum_t M^{T-t}$，几何衰减 | `gradient_decay_with_K`：第 $t$ 步梯度被 $K^{T-t}$ 抑制，$K$ 越小越早步越难传回——主要优化代价 |

> **诚实声明**：本表的**渐进复杂度 $O(\cdot)$ 已在 Lean 中形式化**（mathlib `Asymptotics.IsBigO`，见 §2.10.7 与 `LoopedEncoder.lean` 第 6 节）：朴素激活显存 $=O(T)$、检查点 / DEQ $=O(1)$、前向 / 检查点算力 $=O(T)$，并证明朴素显存无界（线性 vs 常数对比严格）、轨迹可由 $(f,h_0,T)$ 重建（检查点不丢信息）、DEQ 极限态按范数收敛到不动点。**仍为设计断言、未形式化**的是：常数因子、kernel 融合质量、真实 GPU 墙钟与访存带宽、非线性 transformer 块的表达力与「同参更深」泛化收益——这些需实测。三档规模的取舍见 §2.10.6 第 7 条。

#### 2.10.4 主要代价：梯度衰减与有效信用分配深度

权重共享回环的反传梯度是各迭代步贡献之和。线性化后，第 $t$ 步（$t$ 越小 = 越早）的梯度信号需穿过 $T-t$ 层共享雅可比 $M^\top$：

$$
\frac{\partial \ell}{\partial \theta}\bigg|_{\text{来自 }h^{(t)}} \propto \big(M^\top\big)^{T-t}\,\nabla_{h^{(T)}}\ell. \tag{LE-6}
$$

由算子范数次可乘性（`gradient_decay`，仅依赖 `le_opNorm`）：

$$
\boxed{\ \big\|M^n\,v\big\|\le \|M\|^n\,\|v\|\le K^n\,\|v\|,\quad K=\|M\|<1\ } \tag{LE-7}
$$

故 $K^{T-t}\to0$：**越早步的梯度被指数级抑制**，有效信用分配深度被收缩率 $K$ 限制。这是相对前馈（各层雅可比可独立调整、残差连接可保持范数）的**主要优化代价**——也是为什么必须用**谱夹层** $\delta<K<1-\gamma$（同 §3.7.3）而不是一味取小 $K$：$K$ 太小虽加速不动点收敛（LE-5），却把早期迭代的梯度杀光（LE-7），二者此消彼长。残差连接（$B(h)=h+\cdots$）把 $M$ 写成 $I+A$，把衰减从 $K^n$ 改善为 $(1+\|A\|)^n$ 的可控范围，是工程上缓解该代价的标准手段（未形式化）。

#### 2.10.5 表达力包含：回环 = 同深度前馈的权重绑定对角

形式化断言（`looped_eq_feedforward_tied`，无 `sorry`）：**任意回环网实现的函数，都能被一个同深度、所有权重层绑定为同一 $f$ 的前馈网实现**。令前馈网为层列表的复合 $\mathrm{feedforward}([f_1,\dots,f_N])=f_1\circ\cdots\circ f_N$，回环网为函数迭代 $\mathrm{looped}(f,T)=f^{\circ T}$，则

$$
f^{\circ T}=\mathrm{feedforward}\big(\underbrace{[f,\dots,f]}_{T}\big). \tag{LE-8}
$$

**推论（容量代价的精确表述）**：在固定深度 $T$ 下，回环可实现的函数族是前馈函数族的**子集**（取所有 $f_i=f$ 的对角切片），即

$$
\{\,f^{\circ T}:f\,\}\ \subseteq\ \{\,f_T\circ\cdots\circ f_1:f_i\,\}. \tag{LE-9}
$$

反向不成立：前馈可让各层相异（$f_i\neq f_j$），一般 $g\circ f$（$f\neq g$）不是某个 $h^{\circ 2}$。**故同深度下回环表达力严格弱于前馈**——这就是参数共享省下的容量代价。

**反向收益（非形式化，依赖宽/迭代极限）**：在**固定参数预算**下，回环可把省下的参数用于加宽单层或增大 $T$。充分宽的共享块 + 足够多迭代在函数逼近意义下是通用的（可逼近任意固定深度前馈），故「同参更深」存在表达力收益区间——但这依赖宽度/迭代的渐近论证与实测，不在 Lean 形式化范围内（见 §2.10.7 形式化边界）。

#### 2.10.6 契约核对（对照 AGENTS 规则与 math.md 既有前提）

1. **$Z$ 保持连续、不引入语义量化瓶颈**：$Z=W_{\mathrm{out}}h^{(T)}$ 是连续态的线性读出，全程 $\mathbb R^d$ 连续值；不施加 FSQ/VQ/RVQ（同 §2.6 / §2.9.5 接口契约）。数值精度量化（FP8/FP4）只降位宽不改语义。✅
2. **反塌缩**：输入注入（LE-2）使不动点 $h^*=(I-M)^{-1}Me$ **输入依赖**——$M$ 单射时不同输入 $e_1\neq e_2\Rightarrow$ 不同不动点（`mrLoop_no_input_collapse`），不会坍缩到与输入无关的单一状态；谱夹层封死 $M\to0$ 一步塌缩（同 §3.7.3 五道防线第 1 条）。✅
3. **$d:d_{\text{hidden}}=1:8$**：回环块隐藏维 $d_h$、读出头 $W_{\mathrm{out}}:d_h\to d$ 维持 §2.2.7 的 $r=8$ 不变；循环只改层间参数组织，不改维度比例。✅
4. **通路 / 关系分离（§4.5）**：回环迭代是**关系聚合**（对上下文反复重组），读出 $W_{\mathrm{out}}$ 是逐位置**通路**映射，二者仍分离；关系从「$N$ 个不同块的复合」变为「同一块的 $T$ 次复合」，通路仍 $O(L)$ 独立。✅
5. **与 §3 对齐推理的衔接**：回环只替换 $Z$ 的**生产者**（从 $N$ 层前馈换成单层 $T$ 迭代），产出的仍是连续 $Z$；§3.1 硬对齐、§3.2 交叉掩码、§3.4 PAC 在 $Z$ 之上照常进行，契约不变。✅
6. **与 §3.7.1 的正交性**：§3.7.1 循环 $Z$ 上的推理算子 $\Psi_h$（输入依赖、$h$ 自推断）；本节循环输入$\to Z$ 的编码器主干（样本内权重固定）。二者可叠加且不冲突：$E_{\text{loop}}$ 产 $Z$，$\Psi_h$ 在 $Z$ 上推理。✅
7. **三档规模可扩展性（AGENTS 模型规模规则）**：
   - **小规模（CPU 冒烟，$\le10$M）**：现有 `tiny`/`small` 的 $N$ 本就很小（$N=2\sim4$），回环收益不明显（$N\ge2$ 才严格省参，见 `loop_uses_fewer_params`）；但可直接设 $T=N$ 验证算法正确性（损失下降、不动点收敛、反塌缩），纯 CPU 可跑（回环块是纯 PyTorch，无 GPU-only 依赖，不进入 `pyproject.toml` 可选 extra）。
   - **中型（GPU + 真实数据，0.3B–7B）**：主要受益档。参数 $\div N$ 释放的预算可加宽单层或转给对齐/生成头；长上下文下跨循环梯度检查点把激活显存从 $O(N\cdot L\cdot d_h)$ 降到 $O(L\cdot d_h)$，是训练长序列的关键。支持 bf16/AMP、`torch.compile`、梯度检查点（均为成熟库能力）。
   - **大规模（scaling 极限）**：权重共享是规模化刚需（ALBERT 已实证跨层共享可降参数 ~18×）；DEQ 隐式微分使有效迭代深度 $T$ 可远超显存允许的展开深度，$O(1)$ 激活支撑极深有效深度，用于检验 scaling law 的收益/拐点。若引入 GPU-only 隐式求解器，按 AGENTS 规则放入 `pyproject.toml` 可选 extra，保证第 1 档仍纯 CPU。

#### 2.10.7 Lean 形式化（见 §1.0 对照表，全部无 sorry / 无新增公理）

(LE-3)–(LE-9) 的核心断言已在 [`lean-math/LeanMath/LoopedEncoder.lean`](file:///home/simon-ma/JEPA/lean-math/LeanMath/LoopedEncoder.lean)（命名空间 `LoopedEncoder`）机械化；收敛性复用既有 `Affine.lean` / `Contractive.lean` / `AssociativeMemory.lean`：

| 断言 | 定理 | 对应公式 |
|---|---|---|
| (LE-8) 回环 = 同深度前馈的权重绑定特例 ⇒ 函数族包含 (LE-9) | `looped_eq_feedforward_tied` / `looped_realized_by_feedforward` | §2.10.5 |
| 参数量：同有效深度下回环 $\le$ 前馈；$N\ge2,p>0$ 时严格省 | `loop_uses_no_more_params` / `loop_uses_fewer_params` | §2.10.3 |
| (LE-5) 仿射迭代误差恒等式（线性推广 `AssocMem.iterAff_error`） | `affine_iter_error` / `mrLoop_error` | §2.10.2 |
| 反输入塌缩：$M$ 单射 $\Rightarrow e_1\neq e_2\Rightarrow$ 不动点不同（输入依赖） | `mrLoop_no_input_collapse` | §2.10.2 / 契约核对 2 |
| (LE-7) 梯度衰减 $\|M^n v\|\le\|M\|^n\|v\|\le K^n\|v\|$（优化代价） | `gradient_decay` / `gradient_decay_with_K` | §2.10.4 |
| 留存轨迹长度 $=T+1$；朴素激活显存 $=O(T)$ 且无界（故非 $O(1)$，对比严格） | `length_trajectory` / `baseline_memory_isBigO_T` / `baseline_memory_unbounded` | §2.10.3 |
| 检查点 / DEQ 激活显存 $=O(1)$（`Asymptotics.IsBigO`） | `checkpoint_memory_isBigO_one` / `deq_memory_isBigO_one` | §2.10.3 |
| 前向算力 $=O(T)$；检查点总 $2T=O(T)$（常数 2 重算换显存，渐进持平） | `eval_isBigO_T` / `checkpoint_eval_isBigO_T` | §2.10.3 |
| 轨迹由 $(f,h_0,T)$ 唯一确定 ⇒ 只存检查点即可重建，不丢信息 | `trajectory_determined` | §2.10.3 |
| $\|M\|<1$ 时迭代态按范数 $\to$ 不动点 $fp$（DEQ 极限只需留存 $fp$） | `mrLoop_tendsTo_fixedPoint`（拼 `mrLoop_error` + `gradient_decay` + `AssocMem.real_pow_tendsto_zero`） | §2.10.2 / §2.10.3 |
| $M$ 收缩 $\Rightarrow$ 收敛到唯一不动点（DEQ 极限） | 复用 `Contractive.contraction_comp` + mathlib Banach；标量/对角收敛 `AssocMem.real_pow_tendsto_zero` | §2.10.2 (LE-4)(LE-5) |

> **形式化边界（诚实）**：Lean 证明的是**代数恒等式（函数迭代 = 列表复合）、参数计数、线性仿射误差与输入依赖性、算子范数幂次界，以及渐进复杂度 $O(\cdot)$**（mathlib `Asymptotics.IsBigO`：朴素显存 $=O(T)$、检查点 / DEQ $=O(1)$、前向 / 检查点算力 $=O(T)$、轨迹可由 $(f,h_0,T)$ 重建、DEQ 极限态按范数收敛）——代数结论对任意域 $K$、任意模 $V$、任意（连续）线性 $M$ 成立；复杂度结论对 $T\to\infty$ 的 `Filter.atTop` 成立。**未形式化**的部分：常数因子与真实墙钟 / 访存带宽 / kernel 融合（§2.10.3 表）、非线性 transformer 块的表达力与「同参更深」泛化收益（§2.10.5 反向收益）、DEQ 隐式微分的数值稳定性与求解精度、残差连接对衰减的改善（§2.10.4）、充分宽 / 迭代极限的通用性——均为设计分析，依赖实测，见 [lean-math/README.md](file:///home/simon-ma/JEPA/lean-math/README.md)「发现的问题」。与 §2.9 CARO 一样，线性化骨架是数学分析的对象，真实块的非线性行为需实验验证。

---

## 3. 阶段二 · 对齐与推理：表征空间的统一与计算

### 3.1 从共享性假设到硬对齐

(A1) 共享性至今是纸面假设：$Z = E(W)$ 里，图像、视频、文字都应该是同一世界状态的不同渲染。本节把它变成可检验的实现——不是 CLIP 式软对齐，而是"编译到同一 IR、可互相解释"的硬对齐。

### 3.2 编译器/解释器与交叉掩码（plan P6）

**编译器/解释器框架。** 每个模态有前端编译器（encoder）与后端解释器（decoder），共享同一中间表示 $Z$：

$$I\xrightarrow{E^I} Z \xrightarrow{R_I} \hat I,\qquad
T\xrightarrow{E_\theta} Z \xrightarrow{g} \hat T,\qquad
V\xrightarrow{E^V} Z \xrightarrow{R_V} \hat V,\qquad
A\xrightarrow{E^A} Z \xrightarrow{R_A} \hat A.$$

马赛克 $\phi$、JEPA 掩码、文本量化 $g$ 是同一族"有损编译"在不同通道的实例（§2.7 与本节统一）。

**分层统一编解码器。** 既然各模态的编译器/解释器都在做"有损压缩再还原"，能否统一成一套？答案是**分层统一**：统一的是 $Z$ 及其上的语义层，不是物理信号层。两个硬约束使物理层必须各留：

- **信号拓扑不同**：语音是 1D 高采样波形、图像是 2D 网格、视频是 3D 时空、文字是离散 token，最前端/末端必须模态特化；
- **损失机制不同**（§2.7 通道谱系）：连续通道（图像/视频 VAE）靠分辨率可逼近无损，离散通道（文字 $g$、语音 RVQ）靠基数撞地板 $D^\star$——这是 A2/A3 不对称（$H(T\mid Z)=0$ 而 $H(Z\mid I)\to0$）的固有后果，统一 encoder 抹不平，前端 codec 的目标函数也无法共用。

因此"一套"的正确形态是**薄接口 + 厚共享主干**：物理信号层与 token/patch 序列层模态特化（薄），$Z$ 与语义层（$P_\phi,v_\psi,\Psi,K$）全模态共享（厚）。统一收益随深度递增——越靠近 $Z$ 共享度越高（图像/视频 decoder 的卷积上采样可共享，各模态 encoder 的 transformer 后段可共享），最前端/末端只是薄薄的"物理翻译层"。

一句话：**统一发生在 $Z$ 这一层，不是信号这一层**。每个模态保留薄物理编译器/解释器，$Z$ 之上的预测/生成/指令/闭环用同一套参数。这也是多模态大模型（各模态 tokenizer + 投影对齐到共享 LLM）的泛化，区别在于本框架主张硬对齐而非软对齐。

**硬对齐 vs 软对齐。** 软对齐（CLIP/InfoNCE）只要求 $E^I(I)$ 与 $E^T(T)$ 余弦相近，允许落在不同子空间；硬对齐要求二者**共用同一坐标系**：

$$\hat T = g(E^I(I)) \approx T,\qquad \hat I = R_I\big(p_\phi(Z\mid T)\big) \approx I. \tag{17}$$

两条路径穿过同一 $Z$。对齐是"共享瓶颈 + 双向重建"的**副产品**，不是单独的对比损失。

**跨模态交叉掩码（防模态坍缩）。** 各模态各掩一部分，用"本模态剩余 + 其他模态"联合预测被掩部分：

$$\hat z_I,\ \hat z_T = P_\phi\big(E^I(M_I(I)),\ E^T(M_T(T))\big),\qquad
\mathcal L_{\text{pred}} = \mathcal L(\hat z_I, E^I(I)) + \mathcal L(\hat z_T, E^T(T)). \tag{18}$$

> **命题 6（交叉掩码防坍缩）**：若某模态坍缩到常数，它负责补的跨模态缺失就重建失败（其他模态无法替代，见 §3.3 互补性），梯度把它拽回非平凡。与 JEPA stop-gradient 互补：前者防**模态坍缩**，后者防**单模态表征坍缩**（§2.2.3）。

**数据形态。** 图文对 → 状态空间 $Z$ 对齐（描述角色）；图-指令-图三元组 → 算子空间 $\Omega$ 对齐（指令角色）。纯文本语料无世界状态观测，无法提供对齐。

**课程投喂与锚定对齐。** 对齐本质是让各模态编码器在共享坐标系 $Z$ 上收敛到一致，投喂顺序决定"这个坐标系由谁先锚定、其余如何对齐"。这是一个分布式共识问题：$K$ 方无领导协商（混合随机同时训）一致化慢，先让一个锚模态选出 $Z$、其余 $K-1$ 方各自向固定锚对齐，则退化为 $K-1$ 个目标明确、彼此独立的收敛，更快。三个机制：

- **锚点效应**：先有成熟锚（数据最丰富、结构最清晰的模态，通常图文）再对齐，比同时从零抢坐标系快；
- **梯度干扰**：多模态同时训时各模态在共享 $Z$ 上的梯度互相冲突，顺序投喂降低同时干扰；
- **课程难度**：对齐难度随模态而变（命题 4 不对称）——连续通道（图像，近无损）易、离散通道（文字/语音 RVQ，有地板）中、稀疏高噪通道（触觉）难，先易后难符合课程学习。

但**锚定 ≠ 冻结**：过早硬冻结会让 $Z$ 过拟合到锚模态、后续模态挤不进去。最优是**锚定 + 软化**——$Z$ 不冻结，而是低学习率 / EMA 式缓慢漂移，既保留旧对齐又缓慢扩展容纳新模态（冻结是 EMA 在 $\mu=1$ 的硬极限，见 §2.2.3）。实践：锚模态选图文（或复用预训练视觉/语言模型）；课程顺序连续 → 离散 → 稀疏；用损失加权 + 学习率调度实现软锚定（锚模态低 lr、新模态高 lr）；避免极端分阶段（灾难性遗忘）与纯随机混合（无锚收敛慢）。

### 3.3 互补压缩与多模态 codec（plan P7）

视频 $V$、图片 $I$、文字 $T$ 都是同一世界状态 $W$ 的观测，各自有损，但**互补**：

$$I(V;I,T) > \max\big(I(V;I),\, I(V;T)\big). \tag{19}$$

图片补文字丢的空间，文字补图片丢的时间语义。落到失真上：

$$D(V\mid I,T) < \min\big(D(V\mid I),\, D(V\mid T)\big). \tag{20}$$

**两阶段训练。** 阶段① 模态内学 codec（视频 VAE / MAGVIT，语音 EnCodec / SoundStream 的 RVQ）；阶段② 对齐到 $Z$（复用 §3.2 的交叉掩码 + 双向重建）。关键：语音 RVQ 产出**离散码**，与文字 $g$ 完全同构，同有失真地板 $D^\star$（A2）；视频/图像 VAE 是连续潜空间，与文字不对称（命题 4）。

**对应 §2.7 的通道谱系**：

| 通道 | 类型 | 损失机制 | 失真地板 |
|---|---|---|---|
| 图像/视频（连续 VAE） | 连续→连续 | 分辨率 | 无（可逼近 0） |
| 马赛克 $\phi$ | 连续→连续降采样 | 分辨率 | 无 |
| 语音 RVQ / 文字 $g$ | 连续→离散码本 | 基数 | $D^\star>0$ |

### 3.4 多模态硬对齐的训练策略（PAC）

本节把 §3.2 的硬对齐落地为可执行的训练方法，覆盖四个问题：增量通道、软对齐对照、课程投喂、蒸馏预热。

**增量通道对齐与 stop-gradient。** 新增一个通道（如触觉 $E^A,R_A$）时，第一阶段应**冻结已有 $Z$**、只训新通道。这与 JEPA 的 stop-gradient（§2.2.3）同根：都是"静止锚点 + 自适应追逐方"的非对称结构，但两处不同：

- **目的不同**：JEPA 的 stop-gradient 防**表征塌缩**（同期训练退化成常数）；增量通道冻结 $Z$ 防**灾难性遗忘**（跨期破坏旧对齐）；
- **锚点硬度不同**：冻结 $Z$ 是 EMA 在 $\mu=1$ 的硬极限；JEPA 用 $\mu=0.99$ 因其目标编码器训练早期未收敛，而已收敛的 $Z$ 可放心硬冻结。

最优是两阶段：先硬冻结对齐（保旧），若新模态信息"挤不进" $Z$，再放开用 EMA 式软化微调（纳新）。

**软对齐对照：模态间投影的五个高效机制。** 若改为模态间投影（软对齐，各模态自留潜空间），高效学习投影的五个机制：

1. **枢纽投影**：$K$ 个到共享低维枢纽的投影，替代 $O(K^2)$ 两两投影；
2. **低秩投影**：$W=AB^\top$（$r\ll\min(d_I,d_T)$），参数 $O(r(d_I+d_T))$；
3. **对比学习（InfoNCE）**：学度量而非显式映射，batch 内负样本零生成成本；
4. **交叉注意力**：对齐发生在注意力里，处理变长/局部/维度不匹配（Flamingo/LLaVA 主流）；
5. **分方向**：高→低是有损压缩（DIB，§2.5），低→高是生成（条件 Flow，§4.3(c)），不能一个对称映射两头用。

软对齐更灵活省参，但**多跳跨模态误差累积**（$I\to T\to V$ 链式有损投影）；硬对齐穿过同一 $Z$、组合精确。故本框架选硬对齐，软对齐作对照。

**课程投喂与锚定对齐（PAC）。** §3.2 已给概念基础（锚定 ≠ 冻结、三机制），这里给出完整可执行方案——**渐进锚定课程（Progressive Anchor Curriculum, PAC）**。锚模态选图文（数据最丰富、结构最清晰），$K$ 个模态、总步 $T$：

| 阶段 | 步数 | 锚概率 | $Z$ 策略 | 目标 |
|---|---|---|---|---|
| S0 锚预训练 | 0–20% | 100% | 正常 lr | 建 $Z$ 雏形 |
| S1 锚定对齐 | 20–50% | 70%→40% | 正常 lr | 新模态挤进 |
| S2 均衡软化 | 50–90% | 40%→$1/K$ | $Z$ lr ×0.1 | 均衡防漂移 |
| S3 收尾 | 90–100% | $1/K$ | 微调 | 收敛 |

**私有 M 的「先写后压」warmup。** ρ 判据 $\rho=\|v-Mk\|^2$ 依赖特征空间（$v,k$ 是前端输出），前端/主干未学好时 ρ 判据无意义、$M$ 会乱丢 token 阻碍训练。故各模态接入时先关 `rho_compress`（只训前端 + 主干，让特征空间成型），再开（让 $M$ 在有意义特征上学会判冗余）——「先让主干学会写（编码）、再让 $M$ 学会压（判冗余）」。当前 `rho_compress` 默认 `False` 即为此 warmup 开关。

**采样调度**（锚模态对概率 sigmoid 缓降、新模态均分剩余）：

$$p_{\text{anchor}}(t)=p_\infty+(p_0-p_\infty)\,\sigma\!\Big(\!-\tfrac{t-t_0}{\tau}\Big),\qquad
p_m(t)=\tfrac{1-p_{\text{anchor}}(t)}{K-1}. \tag{21}$$

**交叉掩码递进**：$\rho_{\text{mask}}(t)=\min(0.8,\ 0.2+0.6\cdot t/T)$。

**难负例挖掘**：batch 内按 $\text{sim}(z_I,z_T)$ 取语义近但非配对的硬负例进 InfoNCE 分母。

**分层 lr（软锚定）**：锚编码器低 lr、新编码器高 lr（约 $3\times$）、$Z$ 层 cosine decay 且 S2 起 ×0.1。

**锚预训练的蒸馏预热。** S0 可用成熟多模态模型（CLIP/SigLIP）加速，但**蒸馏只解决语义先验、替代不了可还原 $Z$**（CLIP 是软对齐、不可互相还原）：

1. **权重初始化**（最省）：CLIP 视觉/文本 encoder 初始化 $E^I,E^T$（加投影适配层）；
2. **关系蒸馏**（推荐）：匹配相似度矩阵而非锁死表示值，$$\mathcal L_{\text{distill}}=\mathrm{KL}\Big(\text{softmax}(\text{sim}_s/\tau)\,\big\|\,\text{softmax}(\text{sim}_t/\tau)\Big), \tag{22}$$ 维度/结构不必一致，只继承"谁跟谁配"；
3. **软标签混合**：$\mathcal L=\mathcal L_{\text{recon}}+\beta\,\mathcal L_{\text{distill}}$，$\beta$ 随训练衰减。

主损失始终是双向重建（§3.2 式 17）——蒸馏只做预热，双向重建压实成可还原 $Z$。

#### 3.4.1 与当前实现的偏离（文档审计）

上述 PAC 方案写于「各模态独立编码器 + 独立 $Z$ 生产者」的假设下；模型结构经历共享主干 `share_trunk`、$\Psi_h$ 收缩算子融合进 $Z$、Flow 速度场、私有 $\rho$ 压缩等多次修改后，实现已与本节描述显著偏离。逐条记录（带代码引用）：

1. **`pac_anchor_prob` 仅写日志、不门控批次。** [`training/train.py`](file:///home/simon-ma/JEPA/training/train.py) 计算 $p_{\text{anchor}}(t)$ 后只用于 `writer.add_scalar("pac/p_anchor", ...)` 与打印字符串；训练循环每步恒以 `img, txt = batch_fn()` 取两模态同时前向，锚概率从未决定当步训练哪个模态——S0/S1 的「锚先建 $Z$」在实现中是名义的。
2. **全部损失从第 1 步同时激活，无阶段门控。** [`training/losses.py:forward_losses`](file:///home/simon-ma/JEPA/training/losses.py) 把 JEPA、4 项重建、软对齐、方差、码率、交叉掩码、$\Psi$ 算子 5 项（readout/var/stop/consist/gen）、Flow×2 共 13 项加权求和，无按阶段启用；$\Psi$ 改端（`psi_consist`/`psi_gen`）从第 1 步就对着尚未收敛的目标编码器输出 $Z_{\text{tgt}}$ 回归。
3. **Flow 在主预训与独立阶段双重训练。** `flow_enabled` 时主预训 `forward_losses` 计算并累加 `flow`/`flow_cond` 损失（此时目标编码器 EMA 每步更新、仍在漂移），而 [`train.py:train_flow_stage`](file:///home/simon-ma/JEPA/training/train.py) 又在冻结 $E$ 后单独训 $v_\psi$——同一回归目标被训练两次，且主预训段目标非平稳（适定性问题，见下）。
4. **`rho_compress` 是静态开关，「先写后压」非调度跃迁。** [`model/config.py`](file:///home/simon-ma/JEPA/model/config.py) 中 `rho_compress` 默认 `False`，是一次性布尔配置，无法在训练中按「特征空间成型后开启」自动切换；本节描述的 warmup 需手工改配置重启。
5. **共享主干与 PAC 非对称假设冲突。** 中型档默认 `share_trunk=True`（`model/config.py` PRESETS），图文共享同一条产生 $Z$ 的主干；PAC「冻结已有 $Z$ / 锚编码器低 lr 新编码器高 lr」假设锚与新模态各有独立 $Z$ 生产者，共享主干下 $Z$ 的生产者是同一个，只有薄前端接口私有——「冻结 $Z$」在共享主干下退化为「已有薄接口低 lr、新薄接口高 lr」。
6. **检索/评测嵌入头与训练头曾不一致（已修复）。** 检索评测原用 `model.proj`（仅 InfoNCE 路径训练），默认 `byol` 下该头不被训练 → 检索恒随机；现已统一为 `model.embed`（[jepa.py](file:///home/simon-ma/JEPA/model/jepa.py)），byol/dino 用对齐投影器输出 $g_\theta(\texttt{CLS})$，与训练侧反塌缩正则（§2.2.4(c)）共用同一表征，训练与评测口径一致。

#### 3.4.2 平稳性门控的多阶段课程（Stationarity-Gated Curriculum）

上述偏离的共性是：把**依赖「上游表征已平稳」的回归目标**与上游表征的训练同时进行。这在优化适定性上有问题——可由一条初等引理统一刻画，已 Lean 形式化（见 [`lean-math/LeanMath/Curriculum.lean`](file:///home/simon-ma/JEPA/lean-math/LeanMath/Curriculum.lean)，命名空间 `Curriculum`，`lake build` 通过、无 sorry/axiom）：

> **引理 1（最优解追踪目标，`Curriculum.minimizer_tracks_target`）。** 仿射回归 $f_\theta(x)=\theta\cdot x$ 的最小二乘最优解 $\theta^*(y)=y/x$，其变化量满足
> $$\|\theta^*(y_2)-\theta^*(y_1)\|=\frac{\|y_2-y_1\|}{\|x\|}.$$
> 即「目标怎么动，最优解就怎么动」，输入系数 $x$ 起杠杆作用。

> **定理 1（目标漂移 ⟹ 最优解不收敛，`Curriculum.minimizer_not_cauchy_of_target_drifts`）。** 若回归目标序列 $y_t$ 步间漂移恒 $\ge\delta>0$，则最优解序列 $\theta^*(y_t)$ 步间差恒 $\ge\delta/\|x\|>0$，故非 Cauchy、不收敛（`Curriculum.not_cauchy_of_step_ge_eps`）。

核心结论：**对着持续漂移的回归目标训练，参数没有稳定不动点**。据此检查当前架构中各损失的回归目标：

- **$\Psi$ 改端（$L_{\text{consist}}$ / $L_{\text{gen}}$）。** $\Psi_h$ 的仿射骨架 $z\mapsto a\odot z+b$（[`model/operator.py`](file:///home/simon-ma/JEPA/model/operator.py)，对应 §3.7.3），固定 $a$ 与输入 $z$ 时最优偏移 $b^*=z_{\text{tgt}}-a\odot z$。`Curriculum.psi_change_side_tracks_target` 证 $\|b^*_2-b^*_1\|=\|z_{\text{tgt},2}-z_{\text{tgt},1}\|$：改端最优解随目标编码器输出 $z_{\text{tgt}}$ 同步漂移。故 **$z_{\text{tgt}}$ 非平稳时 $L_{\text{consist}}$/$L_{\text{gen}}$ 无稳定最优解**（`Curriculum.psi_change_not_cauchy_of_target_drifts`）。注意 $\Psi$ **读端**（$L_{\text{stop}}$/$L_{\text{readout}}$/$L_{\text{var}}$）锚定 $\Psi(Z_{\text{clean}})\approx Z_{\text{clean}}$，目标是 $Z$ 自身（只需 $Z$ 非退化，由方差正则保证），不受此影响——可从早期低权重启用。
- **Flow 速度场 $v_\psi$。** 回归目标是 $z_1-z_0$（$z_1=E_\xi(x)$ 为目标编码器输出，$z_0$ 为噪声）。`Curriculum.flow_target_tracks_encoder` 证固定 $z_0$ 时目标漂移 $=\|z_{1,2}-z_{1,1}\|$：$E$ 未冻结时 $v_\psi$ 无稳定最优解（`Curriculum.flow_target_drifts_of_encoder_drifts`）。故 **Flow 应在主训练结束、$E$ 冻结后单独训练（`train_flow_stage`），主预训不应累加 flow 损失**。
- **$\rho$ 记忆 $M$。** $M^*=\arg\min\|Mk-v\|$ 追踪前端输出 $v$（`Curriculum.rho_memory_tracks_feature`）；结合 `AssociativeMemory` 的 delta-rule 不动点（收敛要求 $(k,v)$ 平稳，见 `AssocMem.iterAff_error` / `AssocMem.real_pow_tendsto_zero`），前端/主干非平稳时 $M$ 追逐移动目标、$\rho$ 判据无意义（`Curriculum.rho_memory_convergence_requires_stationary_input`）。故 **「先写后压」：特征空间 $\varepsilon$-平稳后再开 `rho_compress`**。

> **定理 5（阶段适定性链，`Curriculum.stage_chain_well_posedness` / `Curriculum.four_stage_chain_nonstationary_breaks_downstream`）。** 阶段偏序 S1→S2→S3→S4 中，后一阶段的回归目标是前一阶段输出的函数；由定理 1，**上游 $\varepsilon$-平稳是下游适定的必要条件**——任一上游非平稳，所有下游目标均无稳定最优解。

据此给出修正后的四阶段课程（阶段比例为超参，可在 `model/config.py` 调）：

| 阶段 | 步数比例 | 启用损失（相对主损失） | 模态采样 | $\rho$ 状态 | Flow 状态 | EMA / 冻结策略 | 依据 |
|---|---|---|---|---|---|---|---|
| **S1 写 / 建 $Z$** | 0–45% | JEPA + 自重建×2 + 交叉重建×2（硬对齐）+ 软对齐 + 方差 + $\Psi$ 读端（$L_{\text{stop}}$/$L_{\text{readout}}$/$L_{\text{var}}$，×0.3 低权重） | 锚薄接口先接入，$p_{\text{anchor}}$ 1→0.5 缓降（**实际门控**） | 关 | 主预训关 | EMA $\mu=0.99$ 正常更新，目标编码器随在线收敛 | 定理 1：此时仅需 $Z$ 非退化（方差正则），不需 $Z_{\text{tgt}}$ 平稳 |
| **S2 压 / $\rho$** | 45–75% | S1 全部 + 码率（开启 $\rho$） | $p_{\text{anchor}}\to 1/K$ 均衡 | **开**（特征已 $\varepsilon$-平稳） | 主预训关 | EMA 继续 | `Curriculum.rho_memory_tracks_feature`：$v$ 平稳后 $M$ 才有意义 |
| **S3 推理 / $\Psi$ 改端** | 55–100% | S2 全部 + $\Psi$ 改端（$L_{\text{consist}}$/$L_{\text{gen}}$，×1.0） | 均衡 | 开 | 主预训关 | EMA 趋近冻结（$\mu\to 0.999$），$Z_{\text{tgt}}$ 近 $\varepsilon$-平稳 | `Curriculum.psi_change_not_cauchy_of_target_drifts`：$Z_{\text{tgt}}$ 平稳后改端才适定（与 S2 重叠，因二者都需平稳 $Z$ 可联训） |
| **S4 生成 / Flow** | 主训练结束后（独立 `train_flow_stage`） | 仅 flow + flow_cond | 均衡 | 开 | **开** | **冻结 $E_\xi$ 与 $E_\theta$**，只训 $v_\psi$ | `Curriculum.flow_target_drifts_of_encoder_drifts`：$E$ 冻结后 $v_\psi$ 才适定 |

**门控判据（可计算代理）。** 「$Z_{\text{tgt}}$ $\varepsilon$-平稳」用目标编码器输出的 EMA 漂移量代理：$\text{drift}_t=\|\text{EMA}(Z_{\text{tgt}},t)-\text{EMA}(Z_{\text{tgt}},t-w)\|$（滑窗 $w$，如 100 步），或在线/目标 $Z$ 的余弦 $\cos(Z_{\text{online}},Z_{\text{target}})$ 接近 1；当 $\text{drift}_t<\varepsilon$（如 $10^{-3}$）判定进入下一阶段。**〔设计断言〕** 精确阈值、滑窗长度、是否用余弦替代漂移量，均为工程超参，未形式化；定理 1–5 只保证「非平稳 ⟹ 不适定」的必要性方向，不给出阈值。

**共享主干下的适配。** `share_trunk=True` 时锚/新模态共享产生 $Z$ 的主干，只有薄前端接口私有。因此 PAC「冻结已有 $Z$」退化为：新模态接入时只新增薄前端接口（高 lr），共享主干与已有薄接口保持低 lr（软锚定），$Z$ 不硬冻结——与「锚定 ≠ 冻结」（§3.2）一致。**〔设计断言〕** 此适配的最优 lr 比例需实验确定。

**模态门控的近似实现。** 真正的单模态前向（按 $p_{\text{anchor}}$ 跳模态）会破坏 `forward_losses` 的张量形状约定；实现上**以 $p_{\text{anchor}}$ 缩放锚/新模态的损失权重**（$w_{\text{anchor}}=p_{\text{anchor}}$、$w_{\text{new}}=(1-p_{\text{anchor}})/(K-1)$）达到「锚梯度主导」的同等效果，不跳前向。**〔设计断言〕** 损失权重缩放与真正跳前向在梯度上的等价性未形式化。

**与既有契约的兼容性。** 本课程只调整训练目标的**启用顺序与权重门控**，不改变模型结构：$Z$ 全程连续（$\rho$ 压缩作用于 token 层，不触及潜空间，§2.6）、不引入语义量化瓶颈、反塌缩约束（方差正则从 S1 即启用）保持不变。$\Psi$ 的收缩性（`Contractive.contraction_comp`）与读端锚定（$L_{\text{stop}}$）保证 S3 开启改端时不动点不漂移出界。**实现注记**：$\rho$ 的结构仍由 config 静态决定（`rho_compress`，训练前 `--rho-compress` 开启，`RhoCompressor` 参数始终存在，避免切换破坏 checkpoint 结构）；「先写后压」在**课程调度层**落地为三道门控，都从 `curriculum_rho_start` 起生效——(1) **forward 层** `rho_active`（S1 $t<\rho_{\text{start}}$ 时 ρ 压缩休眠：不丢 token、不产生码率信号，S2 起激活丢 token，[`model/encoder.py`](file:///home/simon-ma/JEPA/model/encoder.py) `RhoCompressor.forward(active=...)`）；(2) **参数层** `rho_memory` 独立分组（S1 冻结 `lr=0`、S2 解冻，[`training/train.py`](file:///home/simon-ma/JEPA/training/train.py) `build_param_groups`/`apply_rho_schedule`）；(3) **损失层** `rate` warmup（`curriculum_gates` 的 `rate` 门控，[`training/losses.py`](file:///home/simon-ma/JEPA/training/losses.py)）。`curriculum_enabled=False` 时三道门控全部旁路，等价旧行为（ρ 全程激活）。

> **形式化边界（诚实）。** 已 Lean 形式化：引理 1、定理 1、$\Psi$/Flow/$\rho$ 三实例化、阶段链定理 5（见 `Curriculum.lean`，对照表见 §1.0）。**未形式化（设计断言）**：$\varepsilon$ 阈值与滑窗选取、梯度方差收敛速率、平稳性代理的统计估计误差、阶段比例最优值、损失权重门控与跳前向的梯度等价性、共享主干下 lr 比例——均依赖实测，见 [lean-math/README.md](file:///home/simon-ma/JEPA/lean-math/README.md)「问题」。

§3.1–3.4 讨论多模态如何对齐到同一 $Z$；从本节起讨论 $Z$ 内的转换与计算。二者并置在同一阶段的原因见 §1——文字把「描述（对齐）」与「推理（预测）」用同一种符号混写，靠网络自推断的隐含模式 $h$ 分辨，故对齐目标与推理目标在表征空间上交织、不可切分。

### 3.5 描述与推理：隐含模式 $h$（plan P5）

§2.1–2.5 把文字统一视为状态的有损投影 $T=g(Z)$。但同一 $\mathcal V^L$ 里还有另一类文字——它不是单纯描述状态，而是**作用于**状态（推理/变换）。现实中无法在句法层把「描述」与「推理」切开，二者常混写在同一段文字里，故不设外部二值开关，而是让网络自推断一个隐含模式状态 $h$。

**动机：为什么推理要在 $Z$ 内做，而不是在 token 空间做。** 文字（及任何离散模态）是 $Z$ 的**低码率有损离散化**，而非「低维线性投影」：$g:Z\to\mathcal V^L$ 量化到有限离散点集，作为流形其维数为 $0$，但每个纤维 $\mathcal F_t=g^{-1}(t)$ 是 $Z$ 里的高维区域——瓶颈在**码率** $H(T)\le L\log_2|\mathcal V|$（式 (1)–(2)），不在维度。

在 token 空间推理 = 让算子迭代的每一步都「压缩 → 还原」一次（式 (6) 的 $H(Z^{(i)}\mid T^{(i)})>0$ 每步都亏，细节见 §3.7），且单步信息吞吐被 token 码率卡死（$\le L\log_2|\mathcal V|$）：表达一个 $Ld$ 维连续状态被迫拆成远多于 $N$ 步。这正是普通 LLM 的 CoT 浪费算力的机制，其长串行深度与 §1 解码端的自回归 $O(L)$ 同源，只是瓶颈从「解码」换到「算子迭代」。

三条路径按长度递增：**环境欧氏直线**（离开可解码流形 $\mathcal M$ 的折叠捷径，最短但不可解释）< **$\mathcal M$ 上的测地线 / 收缩流**（$Z$ 内连续迭代，保持可解释的最短路径）< **token 阶梯**（每步量化重投影，最长）。本节与 §3.7 要落到的正是中间那条：推理 = $Z$ 上的算子复合 $\Psi^{(N)}(Z)$。注意「短」来自**去量化瓶颈 + 算子收缩结构**，而非「高维」本身——单纯加维有维度灾难，并不必然更好找路径。

**数学区分。** 令隐含模式空间 $\mathcal H$（连续状态流形）、主干 $H:\mathcal V^L\to\mathcal H$、状态依赖变换 $\Psi:Z\times\mathcal H\to Z$，其中 $h=H(x_{\le t})$ 由网络从输入自推断，同时承载「描述/推理倾向」与「变换内容」：

- 描述 $T_d = g(Z)$：状态的有损投影（命题 1–4），读状态——$h$ 落在「读状态」端，$\Psi_h\approx\mathrm{id}$；
- 推理 $T_{cot}$：$h$ 落在「改状态」端，$Z'=\Psi_h(Z)$，非平凡地把 $Z$ 投影成另一时序/条件下的状态。

> **命题 7（描述/推理的角色由网络自推断）**：$T$ 是描述还是推理不是文本的固有属性，而是**网络自推断的隐含状态 $h$ 决定的角色**——$h$ 落在「读状态」端时 $T_d\approx g(Z)$；$h$ 落在「改状态」端时 $Z'=\Psi_h(Z)$。同一句话的 $h$ 投影出不同分量、走不同路径，无需在文本层预分类，也无需外部标签。

**两条实现路线**（对应 $P_\phi$ 的确定性 vs Flow 的随机性）：

- **路线 1 · $P_\phi$ 泛化（确定性）**：$Z'=P_\phi(Z,h)$，用"隐含模式–变换对"监督。$P_\phi$ 是 $\Psi$ 的无监督先验——掩码自监督已学出潜空间动力学，推理变换学习只是给这动力学贴语义标签。
- **路线 2 · Flow 条件生成（随机）**：$h$ 作交叉注意力条件注入 $v_\psi(z_t,t,h)$，随机性来自噪声熵源（式 44–47）。等价于把式 41 的条件从 $T$ 换成 $h$。

**关键结构。** 变换算子需**可组合结构**：模糊/变色/移动为生成元，复杂推理是它们的组合，对应变换半群 $\Psi(Z,\omega_2\circ\omega_1)=\Psi(\Psi(Z,\omega_1),\omega_2)$。这里 $\omega_1,\omega_2$ 是 $h$ 的变换分量；$h$ 另含控制/指令分量，§3.6 的时序控制 $\omega_t$ 即其表现。该算子与 §3.6 的状态转移 $\Psi(\hat Z_t,\omega_t)$ 共用。

### 3.6 预测-校正闭环观测器（plan P8）

§3.5 的 $\Psi$ 与 §3.2 的编码器合起来仍是**开环**链路：$\hat Z_{t+1}=\Psi(\hat Z_t,\omega_t)$ 预测完不看真实结果。JEPA 已做"误差驱动的参数学习"（离线系统辨识），缺"在线状态修正"闭环。这是卡尔曼滤波视角。

**组件映射。** $\Psi$ = 状态转移 $F$；$R_I/R_V$ = 观测 $H$；$E^I/E^V/E_\theta$ = 观测逆映射；指令 $\omega$ = 控制输入 $u$；新增增益 $K$。

**预测-校正两步。**

$$\hat Z_{t+1}^{-} = \Psi(\hat Z_t,\ \omega_t), \qquad
\hat Z_{t+1} = \hat Z_{t+1}^{-} + K\, e_{t+1}, \qquad
e_{t+1} = E(o_{t+1}) - \hat Z_{t+1}^{-}. \tag{23}$$

$e_{t+1}$ 是**创新**（innovation）。增益 $K$ 决定"信预测 vs 信观测"：大 $K$ 跟随真实、小 $K$ 平滑抗噪，最优为卡尔曼增益。

> **命题 8（创新序列的诊断价值）**：若 $\Psi$ 是世界的充分模型，创新 $e_t$ 应近似白噪声；若 $e_t$ 有自相关或偏置，则 $\Psi$ 存在系统性偏差，可据此定位补数据 / 修 $\Psi$ 的方向。

**多模态统一。** 语音/图片/视频/文字都是同一 $Z$ 的观测，$e$ 在潜空间统一计算，与 §3.2/§3.3 共用编码器，不新增模态通道。

**观测编码与误差反投影的区分。** 现实信号 $o$ 进入 $Z$ 的唯一途径是经编码器 $E$（各模态前端 + 共享主干）：式 (23) 的 $E(o_{t+1})$ 就是对齐信号。对齐不是反馈回路现做的，而是 $E$ 在训练阶段已学成（§3.2 硬对齐 + §3.3 互补压缩 + §2.2.4 recon 防塌缩）；反馈回路只是**复用** $E$ 把观测读回 $Z$ 比对。修正量 $Ke$ 直接加在 $Z$ 上（$\hat Z^+=\hat Z^-+Ke$），**不**反投影回像素/传感器空间重建观测——那是生成式路线（VAE/扩散），本框架不做。故「反馈落在 $Z$ 空间」不等于「信号不经编码器」：准确表述是**观测必经 $E$ 编码进 $Z$，修正量直接作用在 $Z$、不重建观测**。若 $E$ 尚未对齐充分，$E(o)$ 的偏差由硬反馈创新统计（命题 8）反向诊断并驱动 $E$ 继续训练（§5 参数修正），闭环自身校准对齐信号。

**静态知识与动态状态的分层。** 式 (23) 隐含两类量的分离：$\Psi,E,D$ 是**静态知识**（训练学到的世界规律/算子，参数化、慢变），$\hat Z$ 是**动态状态**（当前世界状态的估计，激活、快变）。误差 $e$ 因此有两条注入路径、两个时间尺度：① **状态修正（在线、快）**——$e$ 经增益 $K$ 注入 $\hat Z$，只改状态估计、不动模型；② **参数修正（离线、慢）**——$e$ 的统计量（命题 8：自相关/偏置 ⟹ 模型系统性偏差）驱动训练，修 $\Psi,E,D$。「修正后续所有经 $Z$ 的计算」须区分二者：只求当前状态准走状态修正；若 $\Psi$ 对世界规律的描述本身有偏，仅改状态不够，须走参数修正。

**增量表示与误差注入的设计空间（备选，待决策）。** 把动态状态写成增量形式 $\hat Z = Z_{\text{base}} + \delta$，其中 $\delta$ 是叠加在基准 $Z_{\text{base}}$ 上的小状态分量，实时更新、描述当前世界状态。$\delta$ 通常更低秩、更稀疏，显式追踪「状态相对模型预测的偏移」，对应成熟的残差范式；且 $\delta$ 天然是创新 $e$ 的累积，不引入新参数家族。「小的状态分量」按误差注入的层级，可选成熟方案如下（均属备选、非强制）：

| 层级 | 方案 | 与本框架的关系 |
|---|---|---|
| 状态层（改激活不动模型） | Luenberger 观测器（固定 $K$） | 式 (23) 即其形态 |
| | 卡尔曼滤波（最优 $K$ + 协方差） | §3.6 已用此视角 |
| | 预测编码（Predictive Coding） | 顶层慢先验 + 底层快误差单元，与「静态知识 + 动态增量」同构 |
| | 粒子滤波 / 信念状态 | 维护状态分布而非点估计，多模态动作（抓取多解）时优于单点 $\hat Z$ |
| 参数层（改模型） | fast weights（Schmidhuber） | 慢权重存规律 + 快权重存上下文，状态分量在参数空间的对应物 |
| | LoRA / 低秩增量 $W=W_0+BA$ | 「叠加小分量」的标准实现 |
| | delta rule 增量更新 | 项目 `LinearMemory` 已有 |
| 记忆层（外部存储） | 记忆槽 / KV cache | 动态状态写外部记忆，项目私有 $M$ + $\rho$ 压缩器 |

**待拍板的岔口。** 「小的状态分量」是**激活**还是**参数**？若为**激活**（状态估计 $\hat Z$）→ 走状态层（观测器/KF/预测编码），即 $\hat Z$ 增量表示，简单、在线、实时；若为**参数**（$\Delta\Psi$）→ 走参数层（fast weights / LoRA），即 $\Psi = \Psi_0 + \Delta\Psi$、$\Delta\Psi$ 在线更新，能把「慢参数修正」部分快化——但引入「谁学得快、谁学得慢」的学习率分离问题。此岔口留待后续按实际场景（实时性需求 vs 模型偏差程度）再决策。

### 3.7 文字的时间性与情态：表征空间的可容纳边界

前文把"文字 = 状态投影" $T=g(Z)$ 当作默认假设（§2.1），但自然语言的时间性/情态谱系远比图像/视频宽：文字能描述过去、现在、未来、状态变化、反事实与抽象规律，而图像/视频只能锚定**已观测到的状态**。本节放宽该假设，说明 JEPA 的"表征空间"（预测而非重构）如何容纳这些超出"快照"的内容。

**CoT 与过程对齐。** CoT（思维链）是**算子轨迹**而非**状态快照**：它描述 $\Psi$ 的逐步执行 $T_{cot}\approx\big[g(Z^{(0)}),\,g(Z^{(1)}),\,\dots,\,g(Z^{(N)})\big]$，其中每一步是一次状态依赖变换 $Z^{(i+1)}=\Psi_{h^{(i)}}(Z^{(i)})$，$h^{(i)}=H(Z^{(i)})$ 是第 $i$ 步自推断的隐含模式（原 $\omega^{(i)}$ 即其改分量 $h^{(i)}_{\text{change}}$）。把它当描述文字 $T=g(Z)$ 对齐会产生三处错位：

- **$Y$ 污染**：CoT 的措辞码率 $H(T|Y)$ 巨大，语义码率 $I(T;Y)$ 占比反而低（式 15），套用 DIB 会把过程 token 误当语义保留；
- **$Z$ 时间性污染**：CoT 描述未来状态，塞进"当前快照 $Z$"会混入时间演化；
- **路径无关性缺失**："含/不含思考输出应一致"要求 $g$ 在算子等价类上良定义，但框架未显式建模：

$$g\big(\Psi^{(N)}(Z)\big)\;\approx\;g\big(Z_{\text{final}}\big). \tag{24}$$

对齐做法：CoT 归入算子空间 $\Omega$（§3.5），拆"过程"与"最终答案"两段（过程对齐 $\Omega$ 轨迹、答案对齐 $Z$），并显式加式 24 的路径无关性约束。

**token 化是可解释性出口，不是推理的必经环节。** 上一段只解决了 CoT 的「错位」，更深的问题在于**推理本身就不该在 token 空间进行**。算子轨迹 $Z^{(0)}\to Z^{(1)}\to\cdots\to Z^{(N)}$ 是表征空间里的一条路径，$g$ 只是把每个中间状态 $Z^{(i)}$ 投影成可读 token $T^{(i)}=g(Z^{(i)})$。把 CoT 当作推理的必要形式，等于强制**每一步都做一次有损投影再恢复**：由式 (6) 有 $H(Z^{(i)}\mid T^{(i)})>0$，$g$ 丢掉纤维内细节；下一步若要从 $T^{(i)}$ 继续，又得先重建 $Z^{(i)}$，于是每步都「压缩 → 还原」，累计失真、且白白消耗 $\sum_i |T^{(i)}|$ 个 token。

推理的正确载体是 $Z$ 上的算子复合 $\Psi^{(N)}(Z)=\Psi_{h^{(N-1)}}\circ\cdots\circ\Psi_{h^{(0)}}(Z)$，全程不离开表征空间；$g$ 降级为一个**可选的投影出口**——可选而非不需要：需要时（审计、对齐、人可读、可解释性）才把轨迹的某个节点投影成文字 $g(Z^{(i)})$，不需要时推理直接在 $Z$ 内闭环。这正是式 24 路径无关性的自然实现：$\Psi^{(N)}(Z)$ 与 $g$ 解耦后，「含/不含思考输出」只差一个可选投影，不再影响中间状态。

**结论状态是变换链的不动点。** 轨迹 $Z^{(0)}\to Z^{(1)}\to\cdots\to Z^{(N)}$ 的终点不是额外指定，而是链自身收敛的结果：当改分量 $h^{(i)}_{\text{change}}$ 沿轨迹衰减到零、$\Psi_{h^{(N)}}\approx\mathrm{id}$ 时，$Z$ 不再变化，读分量读出结论 $g(Z^{(N)})$。换言之，**推理 = 改分量沿轨迹递减到零的过程，结论 = 改分量归零后剩下的状态**；中间每一步 $Z^{(i)}$ 都是「到达结论前的中间变换」，只有 $Z^{(N)}$ 才是对齐/输出的目标。

> **命题 9（推理的表征空间内禀性）**：表征空间推理在 $Z$ 上经算子复合闭环，token 化 $g$ 是挂在推理之外的可解释性接口，非推理的必要环节；CoT 的 token 开销是可解释性税，而不是推理本身的信息量。

图像同理：视觉推理也应发生在 $Z$ 上（算子复合），而非「生成像素 / 视觉 token 再推理」——后者同样是把表征空间的变换强制搬到 token 空间，重复「有损投影 → 恢复」的浪费。

**文字的时间性/情态谱系与跨模态可对齐性。** 图像/视频是**存在性**数据（只锚定已观测状态），文字是**命题性**数据（可描述任意时刻、甚至非现实状态）。二者的可对齐性取决于文字描述的对象在 $Z$ 空间里是否是图像/视频能锚定的几何对象：

| 文字类型 | 几何对象 | 能否对齐到图像/视频 $Z$ | 处理 |
|---|---|---|---|
| 当前/过去可观测状态 | 点 $Z_t$ | ✅ 有观测锚 | 正常跨模态对齐 |
| 状态变化 | 轨迹 / 转移 $\Psi$ | ✅ 对齐到算子 | 改架构：算子空间（§3.5） |
| 未来状态 | 点 $Z_{future}$ | ⚠️ 无观测锚，仅 $\Psi$ 前向 | 改架构：$Z$ 时间索引 + 预测 |
| 反事实/假设 | 虚拟点（不存在） | ❌ 无观测 | 筛除或纯文本路径 |
| 抽象规律 | 跨状态不变量 | ❌ 非单一点 | 筛除或独立学习 |

"未来状态"与"反事实"是两码事：前者是真实存在但暂缺观测的点，可经 $\Psi$ 前向逼近（§3.6 预测端）；后者是符号系统"凭空构造"的、$Z$ 里不存在的点，观测锚不住。故既要**改架构**（$Z$ 时间索引、状态/算子/轨迹分离、情态作独立编码轴），又要**筛样本**（跨模态对齐只用"可观测状态描述"，其余走独立路径）。

**潜空间 vs 表征空间。** JEPA 学**表征空间**而非**潜空间**，这一区别改写了"对齐"规则：

- **潜空间（重构型）**：潜变量的意义来自"能还原观测"，故**一个点必须有观测才能被定义**——未来、反事实无观测 → 无法定义 → 无法对齐；
- **表征空间（预测型）**：表示的意义来自"能被预测"，故**一个点只要能"被预测/被构造"就能存在**——未来、反事实可被 $\Psi$ 或组合算子构造 → 能落入表征空间。

于是前文的困境大部分消解：

| 内容 | 在表征空间里的形态 |
|---|---|
| 未来状态 | $\Psi$ 的前向轨道（可达点） |
| 状态变化 | 算子 $\Psi$ 本身 |
| 抽象规律 | 流形的不变量/对称性 |
| 反事实 | 被组合算子构造的"虚拟点" |

这些"额外内容"不是"对齐"出来的，而是**预测任务的副产物**——预测自然把"可预测的结构"（规律、动力学、不变量）沉淀进表征空间。但**反事实的沉淀不免费**：默认的掩码 JEPA（§2.2）只学局部预测，学不到反事实。要容纳它，需扩展预测目标——推理预测 $Z'=\Psi_h(Z)$（§3.5）、假设-后果预测、以及算子空间 $\Omega$ 的**可组合生成元**。若 $\Omega$ 只覆盖"真实发生过的变化"，反事实仍构造不出。

**选择性有损压缩与"可预测的结构"的对应。** 上文"可预测的结构"（规律、动力学、不变量）正是 §2.1 选择性有损压缩里 $I(Z;Y)$ 的自监督代理：编码器 $E$ 通过信息瓶颈只保留可预测（$Y$ 相关）分量、在压缩端有意挤掉不可预测的细节，于是"可预测的结构"被**主动**沉淀进 $Z$、而非随机残留。换言之，表征空间 = 把 $Z$ 的码率预算集中投向"可预测结构"的结果；后续推理 $Z'=\Psi_h(Z)$ 与 Flow $v_\psi$ 都只在这个"可预测骨架"上运算，细节的缺失（命题 1）正是它们得以高效闭环的前提——压缩掉的 $H(Z\mid Y)$ 不再占用算力，保留的 $I(Z;Y)$ 足够支撑变换。

一句话：**潜空间问"这个点有观测吗"，表征空间问"这个点能被预测/构造吗"**。JEPA 用第二个问题替换第一个，把"未来/变化/规律"从对齐死角变成预测的自然产物；反事实能否跟进，取决于预测目标是否覆盖"构造虚拟状态"这一环。

**用 CoT 样本训练表征状态网络。** 给定一批 CoT 样本 $t_0,t_1,\dots,t_N$（$t_0$ 为题、$t_N$ 为答案、中间为推理步），先经编码器把它提升进表征空间 $\tilde Z^{(i)}=E(t_i)$。因 $g$ 有损（命题 1），提升用 EMA 目标编码器 $E_\xi$ 作稳定教师、训练用在线 $E_\theta$（沿用 §2.2 双编码器），避免有损噪声回灌。同一批 CoT 同时给出四个互补训练目标：

1. **逐步算子拟合**（用「过程」）：$\mathcal L_{\text{step}}=\sum_{i=0}^{N-1}\big\|\Psi_{h_i}(\hat Z^{(i)})-\tilde Z^{(i+1)}\big\|^2$，$h_i=H(t_0,\dots,t_i)$；
2. **端点 Flow Matching**（用「端点」）：$\mathcal L_{\text{fm}}=\big\|v_\psi(z_t,t,c)-(z_1-z_0)\big\|^2$，$z_0=\tilde Z^{(0)}$、$z_1=\tilde Z^{(N)}$；
3. **不动点/停止正则**（用「终止」）：$\mathcal L_{\text{stop}}=\big\|\Psi_{h_N}(\tilde Z^{(N)})-\tilde Z^{(N)}\big\|^2$；
4. **路径无关性一致性**（式 24 的焊点）：$\mathcal L_{\text{consist}}=\big\|\Psi^{(N)}(\hat Z^{(0)})-\tilde Z^{(N)}\big\|^2$，$\Psi^{(N)}=\Psi_{h_{N-1}}\circ\cdots\circ\Psi_{h_0}$。

(1) 用过程、(2) 用端点、(3) 用终止、(4) 把过程与端点焊在一起；(2) 与 (4) 对偶——一个走连续流、一个走逐步复合，却必须收敛到同一 $Z^{(N)}$。这正印证「对齐（描述 $g(Z^{(i)})$）与推理（变换 $Z^{(i)}\to Z^{(i+1)}$）是同一批样本、同一个 $Z$ 的两个训练目标，不可切分」。

**混合样本与可变深度训练。** 上列四目标假设"给定纯 CoT"，但现实样本是混合的：推理被展开到什么程度是连续且无标签的。按「可见中间步数 $M$」分三类：

| 类型 | 数据形态 | 可见深度 $M$ | $h$ 状态 |
|---|---|---|---|
| A 纯描述 | 单点 $t=g(Z)$ | 0 | $h_{\text{change}}\to 0$（读端） |
| B 隐式 CoT | 端点对 $(t_0,t_N)$ | 1 | $h_{\text{change}}$ 非平凡（变换折叠） |
| C 显式 CoT | 全轨迹 $(t_0,\dots,t_N)$ | $N$ | 逐步 $h^{(i)}$ 序列 |

**可见深度 $M$ ≠ 有效变换深度 $N_{\text{eff}}$**：隐式 CoT（B）里 $M=1$ 但 $N_{\text{eff}}$ 可很大——多步推理被折叠进单个非平凡的 $\Psi^{(N)}$（"规律折叠成转移算子"）。三类是同一目标在不同可见分辨率下的采样：可见几个中间态就监督几步，其余交给 $h$ 自推断变换深度。无需样本类型标签（$h$ 自推断），变换幅度由端点差自然决定；防模式坍缩靠 $\mathcal L_{\text{stop}}$ 与 A 类对齐损失共同钳制两端。

**隐式 CoT 的中间态是潜变量：边际化 vs 端到端。** B 类样本没有 $Z^{(1:N-1)}$，两种训练实现：

- **边际化**：把中间态当显式潜变量 $Z^{(i)}=\Psi_{h^{(i)}}(Z^{(i-1)})$，引入推断网络 $q_\lambda(Z^{(1:N-1)}\mid t_0,t_N)$，最大化 ELBO $\mathcal L_{\text{ELBO}}=\mathbb E_q[\log p(t_N\mid Z^{(N)})]-\mathrm{KL}(q_\lambda\,\|\,p)$。中间态在训练中被**实例化**（推断/采样），只是数据里不可见；保留离散推理步骤与多路径不确定性，但需额外推断网络、ELBO 有界偏松、有后验坍缩风险。
- **端到端**：不实例化中间态，把 $Z^{(0)}\to Z^{(N)}$ 折叠成单步映射 $\mathcal L_{\text{e2e}}=\|\Phi(Z^{(0)})-\tilde Z^{(N)}\|^2$；Flow Matching 的 OT 直线即其实现——$z_t=(1-t)z_0+t z_1$ 是**几何插值而非推理步**，推理被折进 $v_\psi$。稳定、无推断网络、契合"折叠"本意，但丢弃中间结构、不可审计。

两者与 §4.1.5 的曲率对应：**端到端 = 最短流（零曲率，折叠）；边际化 = 保留曲率/多路径（显式离散推理步）**。C 类显式 CoT 是"展开的教学信号"，把中间结构展示给网络，之后要么折叠进 B 类端到端、要么经边际化保留为可推断轨迹。

**实现层面差异。** 端到端每样本三步、不新增参数：$z_0,z_1$ 编码 → 采 $t$、插值 $z_t=(1-t)z_0+t z_1$ → 回归 $v_\psi(z_t,t,c)\to z_1-z_0$；中间态从不出现在计算图。边际化每样本多一个推断网络 $q_\lambda$ 与逐步 rollout：先 $Z^{(1:N-1)}=q_\lambda(t_0,t_N)$ → 沿 $\Psi_h$ 逐步 $Z^{(i+1)}=\Psi_{h^{(i)}}(Z^{(i)})$ → ELBO 同时逼真重建 $t_N$ 与约束 $q_\lambda$ 轨迹贴近 $\Psi_h$ 轨迹（每步一个 KL）；中间态作为张量被实例化。

**折叠的可解释性代价与「可解释性出口」的缺口。** LLM 的显式 CoT 浪费中间算力，却换来可解释性；把推理折叠成 $Z$ 内的中间变换后，中间过程原则上不可直接解释——但即便把中间态 $Z^{(i)}$ 抽样出来，$g$ 也未必能把它转成有意义文字。原因在纤维几何：$g$ 只"认识"文本纤维 $\mathcal Z_{\text{text}}=\bigcup_t g^{-1}(t)$（$Z$ 中能被文字表达的测度零子集），折叠/最短流把中间态推离纤维，落在 $\mathcal Z_{\text{text}}$ 之外；此时 $g(Z^{(i)})$ 是"硬凑最近纤维"的 argmax，产生无意义 token。这给早先"token 化是可解释性出口"补上一个缺口：**出口存在，中间态却未必能通过它**。三条出路：

- **文字锚定正则**（强约束）：$\mathcal L_{\text{audit}}=\|Z^{(i)}-E(g(Z^{(i)}))\|^2$，强制中间态落在纤维上、$g(Z^{(i)})$ 忠实；代价是加曲率（绕路），与最短流冲突。
- **专用叙述器 $g_{\text{int}}$**（中约束）：另训一个 $Z\to$文字 的"推理叙述器"，用 C 类显式 CoT 配对数据监督，学会把潜态推理翻译成话；推理保持折叠、可解释性由叙述器提供，代价是额外网络。
- **审计用显式 CoT**（弱约束）：不试图解码潜轨迹；可解释性由"审计模式"（读分量激活、轨迹文字锚定）下生成的显式 CoT 提供。

更深一层（命题 1 的推论）：**任何文字解释都有损**——$g_{\text{int}}$ 再好也只能叙述 $Z^{(i)}$ 的"文字可表达投影"，$H(Z^{(i)}\mid T^{(i)})>0$ 恒成立，故折叠推理的"完整可解释性"原则上不可得，只能得到文字能承载的那部分。归结：**可解释性 vs 折叠（最短流）是 $h$ 的读/改分量权衡**——读分量激活→文字锚定→可解释但绕路；改分量主导→折叠→高效但不可解释。这使可解释性成为 $h$ 调控的**可切换模式**（audit mode），而非架构缺陷。

**流形重述（回扣 §2.1.1）。** 上面的"可解释性 vs 折叠"权衡可作几何重述：**可解释 = 路径始终保持在可解码流形 $\mathcal M$ 上**——CoT 的每个中间态 $g(Z^{(i)})$ 是有意义 token 的可解码检查点；**折叠 = 在 $\mathcal M$ 外取欧氏捷径**——端到端 / 最短流穿过 $p(Z)$ 的零密度区，中间态不可解码。于是本节的权衡本质是"沿 $\mathcal M$ 的测地线（多步、可解释）vs 环境空间的欧氏捷径（少步、出流形）"——CoT 的"绕路"不是冗余，而是贴弯曲流形走的测地线本征形状（§2.1.1 引理：最短路径 ≠ 直线）。

#### 3.7.1 与 Loop Transformer / DEQ 的关系（参考）

本节"结论 = 变换链不动点、CoT = 算子轨迹"与近年**循环 Transformer（Loop Transformer）**与**深度平衡模型（DEQ）**高度同构，可作形式化参考锚点。

**Loop Transformer 的近亲关系。** Loop Transformer 把同一段层反复循环 $Z^{(k+1)}=F_\theta(Z^{(k)};x)$，其关键结果（Yang et al., ICLR 2024 "Loop Transformers are Universal In-Context Learners"）：当 $F_\theta$ 为「线性自注意力 + ReLU」时，循环逐次逼近一个不动点，且每一步等价于一步迭代算法（如线性回归的牛顿法/梯度下降），故**循环次数 $L$ 是算法迭代步数、不是层数**。与本框架的对应：

| 本框架（§3.5 / §3.7） | Loop Transformer |
|---|---|
| 状态依赖算子 $\Psi_h(Z)$，$Z'=\Psi_h(Z)$ | 循环层 $F_\theta(Z;x)$ |
| CoT 轨迹 $Z^{(0)}\to\cdots\to Z^{(N)}$ | 循环 $Z^{(0)}\to\cdots\to Z^{(L)}$ |
| 结论 = 不动点（$h_{\text{change}}\to0$ 时 $\Psi_h\approx id$） | 循环收敛到 $Z^*$，从 $Z^*$ 读出 |
| CoT 步数 $N$（串行深度） | 循环次数 $L$（= 算法迭代步数） |

共同主张：**推理/计算不是一次性前向映射，而是在表征空间里把同一算子迭代到不动点**。Lean 里证的 `stabilized_is_fixed_point` 正是 Loop Transformer 已用定理证出的"循环 = 迭代算法"的弱化版。

**DEQ 隐式微分：把表达力与串行深度解耦。** Deep Equilibrium Model（Bai et al., NeurIPS 2019）给出不动点反传的隐式公式。设不动点 $Z^*=\Psi_h(Z^*)$（$h$ 已收敛），标量损失 $\ell$ 依赖 $Z^*$，则

$$\nabla_\theta \ell = \left(\frac{\partial \Psi_h}{\partial \theta}\right)^{\top}\left(I - \frac{\partial \Psi_h}{\partial Z}\Big|_{Z^*}\right)^{-T}\nabla_{Z^*}\ell. \tag{25}$$

其中伴随量 $g^*=\left(I-\frac{\partial \Psi_h}{\partial Z}\big|_{Z^*}\right)^{-T}\nabla_{Z^*}\ell$ 由线性不动点迭代求出，而非展开 $N$ 步：

$$g^{(k+1)} = \left(\frac{\partial \Psi_h}{\partial Z}\Big|_{Z^*}\right)^{\top} g^{(k)} + \nabla_{Z^*}\ell, \tag{26}$$

当谱半径 $\rho\!\left(\frac{\partial \Psi_h}{\partial Z}\big|_{Z^*}\right)<1$（即前向收缩条件，同一条件保证不动点存在且唯一）时收敛。

**意义**：反传不必展开 $O(N)$ 步串行 rollout，只需求一个伴随线性不动点——把"表达力（$N$ 步迭代）"与"反传串行深度"解耦。§3.7 的停止正则 $\mathcal L_{\text{stop}}=\|\Psi_{h_N}(\tilde Z^{(N)})-\tilde Z^{(N)}\|^2$ 正是这个不动点残差；DEQ 视角提示：显式 $N$ 步 rollout 可替换为直接解 $Z^*=\Psi_h(Z^*)$，训练与推理都能绕开逐步反传。

**基于测地整流的重推导。** 式 25 假设 $\mathcal M$ 平坦（$J$ 是环境空间雅可比、$\cdot^\top$ 是欧氏内积的伴随）。按 §2.1.1，$\mathcal M$ 泛型弯曲，隐式微分应在切丛 $T\mathcal M$ 上重述。把算子写成测地流：设 $v_h(Z)=\log_Z(\Psi_h(Z))$ 是 $\Psi_h$ 的测地速度场（从 $Z$ 到 $\Psi_h(Z)$ 的对数映射），则 $\Psi_h(Z)=\exp_Z(v_h(Z))$，且不动点 $Z^*=\Psi_h(Z^*)$ 等价于 $v_h(Z^*)=0$（速度为零 = 不再移动）。

在 $Z^*$ 处线性化：$D\exp_{Z^*}(0)=\mathrm{id}$（切空间上的恒等），故

$$J \;=\; D\Psi_h(Z^*) \;=\; I + \nabla v_h(Z^*),$$

其中 $I$ 是 $T_{Z^*}\mathcal M$ 上的恒等，$\nabla v_h$ 是速度场的协变导数（因 $v_h(Z^*)=0$，Christoffel 项消失，$\nabla v_h=D v_h$）。于是 $I-J=-\nabla v_h(Z^*)$，式 25 的隐式微分重述为

$$\nabla_\theta\ell \;=\; -\big(D_\theta\Psi_h(Z^*)\big)^{\dagger}\;\big((\nabla v_h(Z^*))^{\dagger}\big)^{-1}\;\nabla_{Z^*}^{\mathbf G}\ell,$$

其中 $\nabla_{Z^*}^{\mathbf G}\ell=\mathbf G(Z^*)^{-1}\frac{\partial\ell}{\partial Z}\big|_{Z^*}$ 是**黎曼梯度**（用解码器拉回度量 $\mathbf G=J_D^\top J_D$ 把环境梯度提升到切空间），$\dagger$ 是**度量伴随**（对自同态 $T\to T$：$A^\dagger=\mathbf G^{-1}A^\top \mathbf G$；对 $D_\theta\Psi_h:\mathbb R^p\to T$：$B^\dagger=B^\top \mathbf G$；均非欧氏转置）。收缩条件 $\rho(J)<1$ 要求 $\nabla v_h(Z^*)$ 的特征值落在 $(-2,0)$：负定（收缩）且大于 $-2$（不过冲、不振荡）。

**含义**：测地整流下，隐式微分的关键量从"算子雅可比的逆 $(I-J)^{-1}$"变成"速度场协变导数的伴随逆 $\big((\nabla v_h)^{\dagger}\big)^{-1}$"——后者正是 flow matching 所学的 $v_\theta$ 在不动点处的切线性化。这把 Loop Transformer 的算子不动点与 §4.1.5 的整流流统一到同一对象：$v_\theta$ 的稳定零点。欧氏情形是 $\mathcal M$ 平坦（$\mathbf G=I$、$\nabla=\partial$、$\dagger=\top$）时的退化，式 25 是其特例。⚠️ 关键：$(\nabla v_h)^{-1}$ 必须取**度量伴随**后再作逆，不可省略 $\dagger$。仅当 $v_h$ 是（局部）梯度场（$dv_h^\flat=0$，即 $\nabla v_h$ 自伴）时 $(\nabla v_h)^{\dagger}=\nabla v_h$，二者才相等；一般测地速度场非梯度，伴随不可省。欧氏退化时上式退化为式 25 的 $(\partial_\theta\Psi_h)^\top(I-J)^{-\top}\nabla_Z\ell$。

**可借用的三点结论。**

1. **循环次数 = 算法迭代步数**（Yang et al. 2024）升级 §4.3(a) 的串行深度判据：$N$ 不是该压缩的延迟，而是**表达力资源**——每多一步 ≈ 多一步迭代求解器，能解的算法类别更大；故 CoT 步数应自适应分配（对应 Universal Transformer 的 adaptive halting，Dehghani et al. 2019）。
2. **in-context learning = 迭代求解器**（Giannou et al. 2023; Akyürek et al. 2022）为"$\Psi_h$ 反复复合可实现数值/符号运算"提供可表达性下界。
3. **普适 in-context learner 的证明模板**（Yang et al. 2024）可迁移到"状态依赖算子迭代"的一般情形。

**一处关键差异（勿过度类比）**：Loop Transformer 权重固定、算子固定，程序写在权重里、只朝静态不动点走；本框架的 $\Psi_h$ 是输入依赖、可跨步演化（$h^{(i)}$ 自推断），不动点是"以算子 $h$ 为条件的"，更接近 Universal Transformer + 状态依赖算子。即 **Loop Transformer 是"固定算子迭代到不动点"的特例，本框架是"状态依赖算子迭代到不动点"的一般化**。

#### 3.7.2 可解释最短流的算法族

§3.7 末的"可解释 = 保持在可解码流形 $\mathcal M$ 上"给出一个可操作的问题：**给定端点 $z_0,z_1$，找一条路径，使其上的点尽量可解码（$c(z_t)=\max_v p_\theta(v\mid z_t)$ 高），且总流长尽量短**。这是带纤维约束的最短路径，不是自由测地线（自由测地线即 $\lambda=0$ 的解，落在纤维外、不可解释）。三族现成算法：

1. **离散化 → 图最短路径（Dijkstra / A\* / beam search）**。把"可解释点"离散成 token 序列：节点 $=$ 文本 $t$（代表点 $z_t=E(t)$），边权 $=$ 相邻锚点间流长 $d(z_{t_i},z_{t_{i+1}})$。最短可解释路径 $=$ 该图最短路径。现成实现即 ToT / GoT / MCTS（AlphaProof 式）——C 类显式 CoT 的搜索视角，也是 §2.6 MDL（"最短描述 $=$ 最短路径"）的落地。
2. **投影 / 锚定流（交替投影 + decode-reencode）**。流走一步、再投影回纤维 $z\leftarrow E(g(z))$，反复。即 §3.7 的 $\mathcal L_{\text{audit}}$ 在推理时落地，本质是 retraction（收缩到约束子流形）。现成工具：交替投影（Dykstra）、projected flow、confidence guidance（对低 $c(z_t)$ 施斥力）。
3. **加权测地线 → Eikonal / Fast Marching Method**。把"不可解释区"编码进度量 $\tilde g(z)=w(z)\,g(z)$，$w(z)$ 在低置信区取大。$\tilde g$ 下的测地线自动贴可解释区走。现成算法即 Fast Marching（Sethian）：解 Eikonal 方程 $|\nabla u|_{\tilde g}=1$、沿 $\nabla u$ 回溯。

**三者合一的推荐配方（两段式）**：先离散搜骨架（A\*/beam 得锚点序列 $t_0,\ldots,t_N$），再连续填流（相邻锚点间 Riemannian flow 或 decode-reencode 锚定流），目标

$$\min_{\{t_i\},\gamma} \sum_i d(z_{t_i}, z_{t_{i+1}}) \;+\; \lambda\sum_i \big(\text{off-fiber penalty}\big).$$

其中 $\lambda$ 即 $h$ 的**读/改分量旋钮**：$\lambda\to\infty$ 强制每步贴纤维（可解释、绕路），$\lambda\to0$ 退化为自由测地线（短、出流形）。"最短测地流不一定可解释"在形式上即 $\lambda=0$ 的解落在纤维外，故必须 $\lambda>0$。现成单族算法都只覆盖一段：① 有离散骨架无连续流、② 简单但投影破坏流结构、③ 连续但需可微 $w$；合一封装暂无，最接近的是 Riemannian Flow Matching + confidence guidance。

#### 3.7.3 并行扫描收缩算子（PSCO）：算子迭代的 GPU 加速（备选）

§3.7.1 说 Loop Transformer 是「固定算子迭代到不动点」的特例、本框架是「状态依赖算子迭代到不动点」的一般化。但 §3.7.1 的 DEQ 只把**反传内存**降到 $O(1)$，前向仍要迭代 $O(N)$ 步到不动点；§4.1.5 的 reflow 能蒸馏到 $O(1)$ 步，却丢掉自适应深度与可解释性。本小节给出一个备选实现 PSCO（Parallel-Scan Contractive Operator）：把前向串行深度从 $O(N)$ 压到 $O(\log N)$，同时保留状态依赖算子与自适应深度。它复用 §2.6/§4.3 的「并行扫描」原语与 §3.7.1 的测地线性化，不引入新硬件假设。

**算子拆分。** 把状态依赖算子写成不动点方向上的增量形式：

$$\Psi_h(Z) = Z + \Delta_h(Z),\qquad \Delta_h(Z) = A_h Z + b_h + r_h(Z).$$

- $A_h Z + b_h$ 是**线性骨架**，取 §3.7.1 不动点处的线性化 $J = I + \nabla v_h(Z^*)$，即 $A_h=\nabla v_h(Z^*)$、$b_h$ 吸收常数项——负责迭代收敛的「渐近段」；
- $r_h(Z)$ 是**非线性残差**，负责「瞬态段」里线性化失效的大跳变。

$h$ 仍由网络自推断（§3.5），故 $A_h,b_h$ 随输入变化，状态依赖性保留。

**线性骨架 → 一阶线性递推。** 令 $\tilde A_h = I + A_h$。渐近段（$r_h\approx 0$）退化为线性递推

$$Z^{(k+1)} = \tilde A_h Z^{(k)} + b_h,\qquad Z^{(k)}\in\mathbb R^d.$$

闭式解：

$$Z^{(k)} = \tilde A_h^k Z^{(0)} + \sum_{j=0}^{k-1}\tilde A_h^{\,k-1-j}\,b_h.$$

**并行扫描（associative scan）。** 上式是「仿射变换幺半群」上的前缀和。定义二元结合算子

$$(M_1,u_1)\circ(M_2,u_2)=\big(M_1 M_2,\ M_1 u_2 + u_1\big),$$

初始元素取 $(\tilde A_h,\,b_h)$。对 $N$ 个相同元素做 Blelloch 前缀扫描，一次得到全部前缀 $Z^{(1)},\dots,Z^{(N)}$：工作量 $O(N)$、**串行深度 $O(\log N)$**——这就是 Mamba/S4 训练快的同一原语（对应 §4.3(a) 的串行深度判据）。

**低秩/对角约束（保 $O(Nd)$ 工作量）。** 若 $\tilde A_h$ 是稠密 $d\times d$，幺半群乘法 $M_1M_2$ 是 $O(d^3)$。要保留线性工作量，取对角 $\tilde A_h=\operatorname{diag}(\tilde a_h)$（SSM 风格）或秩一 $\tilde A_h=I+u_h v_h^\top$（矩阵幂走 Woodbury 闭式）。二者都退化为 $O(Nd)$ 逐元素扫描。

**收缩条件（数值稳定性）。** 扫描收敛要求 $\rho(\tilde A_h)<1$，正是 §3.7.1 的收缩条件（$\nabla v_h$ 特征值落在 $(-2,0)$）。训练加可微谱正则钳住：

$$\mathcal L_{\text{spec}}=\operatorname{relu}\big(\|\tilde A_h\|_2-(1-\gamma)\big)^2,\qquad \gamma\in(0,1).$$

**自适应深度（早停免费）。** 扫描同时物化整条轨迹 $Z^{(0..N)}$，停止索引只是一次并行 argmin：

$$\text{stop}=\min\big\{k:\ \|Z^{(k+1)}-Z^{(k)}\|<\epsilon\big\}.$$

$N$ 只是扫描上限：简单样本早停、难题跑满，二者 GPU 深度都是 $O(\log N_{\max})$，不因停得早而更慢（对比 Loop Transformer 固定 $L$ 步的浪费）。

**训练（反传）。** 关联扫描的反向同样是 $O(\log N)$（Mamba 同款），训练串行深度 $O(\log N)$ 而非 BPTT 的 $O(N)$；只关心不动点时改用 DEQ 隐式微分（式 25/26），只存 $Z^*$、$O(1)$ 内存。两段式：瞬态段 $m$ 步显式反传 + 渐近段 DEQ 隐式微分。

**伪代码。**

```text
# ---- 前向：状态依赖算子 + 并行扫描 ----
def psi_h(Z, h):                     # 状态依赖算子 Ψ_h
    Ah, bh = proj_A(h), proj_b(h)    # h -> 线性骨架参数（对角/低秩）
    return Z + Ah @ Z + bh + r_h(Z)  # 线性骨架 + 非线性残差

def scan_linear(Z0, Ah, bh, N):      # 仿射幺半群上的并行前缀扫描
    # 元素 (M,u)；组合 (M1,u1)∘(M2,u2) = (M1 M2, M1 u2 + u1)
    # Blelloch 上下两趟，深度 O(log N)，返回 Zs[k] = Z^(k), k=1..N
    ...

def forward(Z0, h, N, eps, m):
    Z = Z0
    for _ in range(m):               # 瞬态段：少量全非线性步
        Z = psi_h(Z, h)
    Zs = scan_linear(Z, Ah, bh, N)   # 渐近段：并行扫描物化全轨迹
    resid = (Zs[1:] - Zs[:-1]).norm(dim=-1)     # 逐步残差
    stop = int(resid.lt(eps).long().argmax()) if resid.lt(eps).any() else N
    return Zs[stop], stop            # 不动点 + 自适应用到的深度

# ---- 训练：两条反传路径 ----
# 路径A：scan 反向（关联扫描 reverse），O(log N) 深度
# 路径B：只存 Z*，解伴随线性不动点（式 25/26），O(1) 内存（DEQ）
# 谱正则：L_spec = relu(||Ah_tilde||_2 - (1-gamma))^2  钳住收缩
```

**PyTorch 落地要点（适配当前 multimodal_jepa.py）。** 现有实现是 `TransformerBlock`（`nn.MultiheadAttention` + MLP，无 SSM/scan 依赖），PSCO 与它正交，作为**算子迭代层**接入：把 §3.5 的 $\Psi_h$ 实现为「线性骨架 + 残差」模块，替换/包裹 `LatentPredictor` 或作为推理专用头。并行扫描在 PyTorch 有三档实现，按序列长度递进：

1. **纯 PyTorch log-depth 扫描**：用 reshape + 逐级组合手写 Blelloch 两趟，零新依赖；配合 `torch.compile` 融合逐元素 op。当前 `dim ≤ 256`、短序列下完全够用，先用它验证正确性。
2. **chunked scan + `torch.compile`**：把 $N$ 切成 $N/C$ 块，块内串行、块间并行扫描，减少 kernel 启动、提升 occupancy——长序列下的工程折中。
3. **Triton / `pscan` / `mamba-ssm` 内核**：$L\gtrsim 10^3$（对应 §4.2(g) 的 Mamba 收益区）才需要；对角 $\tilde A_h$ 直接复用 `causal-conv1d` 的关联扫描核。

三个工程注意点：(i) **内存**——扫描物化 $O(N)$ 前缀激活，若只出不冻点走 DEQ 路径（$O(1)$），否则对扫描段用 `torch.utils.checkpoint` 重计算；(ii) **精度**——线性骨架 $A_h$ 与 §2.6 的数值精度量化正交，可单独降到 FP8 而不破坏 $Z$ 的连续语义（§2.6 契约）；(iii) **梯度**——对角 $\tilde A_h$ 的扫描是可微的逐元素乘加，`torch.compile` 能自动反向，无需手写 Triton 反传。

**串行深度对比（收束）。**

| 方案 | 前向 | 反传 | 内存 | 自适应深度 | 状态依赖算子 |
|---|---|---|---|---|---|
| Loop Transformer | $O(L)$ | $O(L)$ | $O(L)$ | ✗ | ✗ |
| DEQ | $O(N)$ 解不动点 | 隐式解 | $O(1)$ | ✗ | ✗ |
| Rectified Flow 蒸馏 | $O(1)$ | $O(1)$ | $O(1)$ | ✗ | ✓（折进 $v_\theta$） |
| **PSCO（本方案）** | $O(\log N+m)$ | $O(\log N)$ 或 $O(1)$ | $O(1)$（DEQ 路径） | **✓ 免费** | **✓** |

PSCO 不替代 reflow，而是给 reflow 提供一个**有据可依、能自适应深度的 teacher**；两者共用 $\nabla v_h(Z^*)$ 这一同一对象（§3.7.1）。

**反塌缩机制。** 收缩算子天然把一切拉向同一个吸引子，故 PSCO 的塌缩风险比普通 Transformer 更需显式设防。要分两种塌缩：

| 塌缩类型 | 表现 | 根因 |
|---|---|---|
| 表征塌缩 | 所有输入收敛到同一不动点 $Z^*$ | 强收缩 $\tilde A_h\to0$ 把轨迹一步拉到共享的 $b_h$，输入信息被抹掉（防线见 §2.2.3–2.2.4） |
| 模态坍缩 | 某模态被另一模态/共享表征吞并 | 对齐目标单侧收敛、交叉掩码缺位（防线见 §3.4） |
| 算子塌缩 | $\Psi_h\to\mathrm{id}$（$\tilde A_h\to1$）或常值映射 | 模型学到「不做计算 / 恒等」的最省力解 |

五道防线（前三道已在 [`psco.py`](file:///home/simon-ma/JEPA/model/psco.py) 验证，后两道留作扩展）：

1. **谱夹层（spectral sandwich）** 逐维 $\delta<\tilde a_{h,i}<1-\gamma$（等价于 $\rho(\tilde A_h)<1-\gamma$ 且 $\min_i\tilde a_{h,i}>\delta$）：上界保收敛（§3.7.1 收缩条件），下界 $\delta$ 封死「$\tilde A_h\to0$ 一步塌成一点」这条最阴的路径——只给上界时模型会学 $\tilde A_h\to0$，一步把所有 $Z$ 拉到共享 $b_h$，轨迹不再区分输入。参数化取 $\tilde A_h=\operatorname{diag}\big(\delta+(1-\gamma-\delta)\,\sigma(h)\big)$，逐维钳进 $(\delta,1-\gamma)$。
2. **不动点读出头锚定**：$Z^*$ 必须能预测任务目标 $\mathcal L_{\text{readout}}=\|r(Z^*)-\tilde Z_y\|^2$（$r$ 为读出头）。塌成常数就预测不了，梯度把 $Z^*$ 重新推开。对应 §3.8 的预测目标与 §3.7 的 $\mathcal L_{\text{consist}}$。
3. **方差正则**：对 $Z^*$ 每维标准差设下限 $\mathcal L_{\text{var}}=\mathrm{relu}(1-\sigma_j)$（同 §2.2.4(b) 式 (36)），防维度塌缩。
4. **$h$ 的信息量约束**：$a_h,b_h$ 由 $h$ 自推断；若 $h$ 网络塌缩（所有输入同一 $h$），算子退化为常值。把 $h$ 接一个判别/预测任务（或 DIB 约束 §2.5）强制 $h$ 随输入变化。
5. **轨迹非平凡性**：扫描物化整条轨迹，加「首步位移 / 轨迹长度」下限惩罚，防算子第一步就跳到不动点（瞬态段 $m$ 步被跳过）。对应 §2.6「ε 强制采样」防自我强化退化。

一句话：**塌缩本质是「算子失去输入依赖性」——$\tilde A_h\to0$ 强收缩抹信息，或 $b_h$ 退成常数共享吸引子。谱夹层管 $\tilde A_h$ 的健康区间，读出头锚定强制 $Z^*$ 携带输入信息，方差正则管维度分布，三者合起来保证不动点不塌成一点。**

**文字 → 算子 vs 描述：与 PSCO 的对应。** §3.5 命题 7 说文字的两面性由自推断的 $h$ 决定，本节把这条接到 PSCO 的算子参数上。文字 $T$ 经 $H$ 提升为 $h=(h_{\text{read}},h_{\text{change}})$，改分量 $h_{\text{change}}$ 直接门控算子的非平凡部分：

$$\Psi_h(Z) = Z + g_{\text{change}}\odot\big(A_h Z + b_h + r_h(Z)\big),\qquad g_{\text{change}}=\sigma(W_{\text{ch}}h)\in(0,1)^d.$$

- 描述端（$h_{\text{change}}\to0$）：$g_{\text{change}}\to0$，$\Psi_h\approx\mathrm{id}$，文字是状态投影 $T_d=g(Z)$；
- 算子端（$h_{\text{change}}\approx1$）：$g_{\text{change}}\approx1$，$\Psi_h$ 非平凡，文字是 $\Omega$ 里的一个变换 $Z'=\Psi_h(Z)$。

于是「算子性 = $g_{\text{change}}$ 偏离 0 的程度」，与 §3.7 的停止条件 $h_{\text{change}}\to0$ 天然对齐：改分量归零 ⟺ 变换完成 ⟺ 读出结论。

**算子空间 $\Omega$ 的具体定义。** 取 $Z\in\mathbb R^d$（$d$ 为表征维数），算子空间是「对角收缩仿射 + 非线性残差」这一族映射：

$$\Omega = \Big\{\ \Psi_h:\ \Psi_h(Z)=Z+g_{\text{change}}\odot\big(\operatorname{diag}(a_h)\,Z + b_h + r_h(Z)\big)\ \Big|\ a_h\in(\delta-1,-\gamma)^d,\ b_h\in\mathbb R^d,\ h\in\mathcal H\ \Big\}.$$

- $a_h\in(\delta-1,-\gamma)^d$ 是对角收缩**增量**（对应谱夹层 $\tilde A_h=I+\operatorname{diag}(a_h)\in(\delta,1-\gamma)^d$，见上文），$\rho\big(I+\operatorname{diag}(a_h)\big)<1$ 保证收敛；
- **半群结构**：收缩映射的复合仍是收缩映射（Banach 不动点定理），故 $\Psi_{h_2}\circ\Psi_{h_1}$ 仍收敛到不动点——复杂推理 = 生成元复合（§3.5 的 $\Psi(Z,\omega_2\circ\omega_1)=\Psi(\Psi(Z,\omega_1),\omega_2)$）。注意该复合一般不再具有 $\Omega$ 的对角仿射参数形式，PSCO 以「逐段扫描 + 换 $h$」实现复合，而非字面代入 $\Omega$ 的解析式；
- 描述是 $\Omega$ 的**单位元邻域**：$g_{\text{change}}\to0$ 时 $\Psi_h\to\mathrm{id}\in\Omega$——「描述 ⊂ 算子」是同一套退化结构，与 §3.8「重建是预测的特例」同构：描述是算子的特例，算子是一般情形。

**文字 → $\Omega$ 的伪代码。**

```text
def text_to_operator(ids, H, W_ch, proj_a, proj_b, r):
    h = H(ids)                               # 隐含模式（自推断，无需标签）
    g_change = sigmoid(W_ch @ h)             # 改分量门控 ∈ (0,1)^d
    a_tilde = delta + (1 - gamma - delta) * sigmoid(proj_a(h))  # 全线性图对角 ∈ (δ,1-γ)
    a = a_tilde - 1                                          # 增量对角 ∈ (δ-1,-γ) ⊂ (-1,0)
    b = proj_b(h)                            # 偏移
    def psi(Z):
        return Z + g_change * (a * Z + b + r(Z))   # 描述端 g_change→0 退化为 id
    return psi, g_change                     # psi ∈ Ω；g_change 即「算子性」强度
```

**与 Universal Transformer 自适应停机的区别与借鉴。** Universal Transformer（Dehghani et al. 2019）配 ACT 的自适应停机是**学习驱动、串行判定**：每步用小网络算 per-position 停机概率 $p_t$，累计超过阈值才停，并用 ponder cost 惩罚总步数。PSCO 的早停是**残差驱动、事后判定**：扫描已把整条轨迹 $Z^{(0..N)}$ 物化，停机只是对已算好的轨迹做一次并行 argmin（见上文「自适应深度」）。

| 维度 | ACT | Universal Transformer | PonderNet | PSCO |
|---|---|---|---|---|
| 核心机制 | 累计停机概率 + 剩余质量加权 | 共享 block + per-position 停机 | 几何分布停 + KL 先验 | 收缩算子 + 并行扫描 |
| 串行深度 | $O(N)$ | $O(N)$ | $O(N)$ | $O(\log N+m)$ |
| 停机信号 | 学习 $p_t$（标量） | 学习 $p_t$（per-position） | 学习 $\lambda_n$ | 残差阈值（零参数） |
| 停机语义 | 计算预算耗尽 | 计算预算耗尽 | 先验分布匹配 | 不动点收敛（有保证） |
| 收敛保证 | ✗ | ✗ | ✗ | ✓（Banach 收缩） |
| 算子性质 | 固定权重 | 固定权重 | 固定权重 | 输入依赖 $\Psi_h$ |
| 早停成本 | 串行等 $p_t$ | 串行等 $p_t$ | 串行等 $\lambda_n$ | 免费（轨迹已物化） |
| 训练/内存 | 有偏梯度，$O(N)$ | 有偏梯度，$O(N)$ | KL 正则，$O(N)$ | scan 反传 / DEQ，$O(1)$ |
| 实现复杂度 | 低 | 中 | 中 | 高 |

**各取所长（三点嫁接）。**

1. **ACT → ponder cost / 深度正则**：显式惩罚 stop 分布，防 PSCO 学成「永远跑满 $N$」或「永远一步停」的退化，加 $\mathcal L_{\text{ponder}}=\mathbb E[\text{stop}]$。
2. **PonderNet → 概率加权 + KL 先验**：它的 $\hat Z=\sum_n p_n Z^{(n)}$（按停机概率加权）与「KL 到几何先验」给出**低方差、无偏**的深度正则，比 ACT 的有偏估计干净。PSCO 的软停机掩码应照这个做（$\hat Z=\sum_n\sigma_n Z^{(n)}$，$\sigma_n$ 为软停机概率），而非 ACT 的剩余质量 hack。
3. **Universal Transformer → per-position/per-dim**：把「每位置独立停」向量化成 PSCO 的「每维独立收缩」——对角化 $a_h$ 天然支持，这是三者中唯一能和 PSCO 无缝对接的粒度。

PSCO 相对三者独有、学不来的：**早停免费**（扫描物化轨迹，无需串行等停机概率）、**停机语义统一**（残差归零 = 改分量归零，而非外挂网络）、**$O(\log N)$ 深度**与**收敛保证**。代价是**实现复杂度最高**与**线性化假设**（渐近段需收缩）。

**备选方案与实验判据。** 本小节把自适应深度的四条路线一并列为备选：ACT、Universal Transformer、PonderNet、PSCO。四者不是替代而是互补——**停机策略（借 ACT/PonderNet/UT）、收敛保证（PSCO）、$O(\log N)$ 深度（PSCO）可以合体**。但要诚实：目前所有结论都来自机制分析与 [`psco.py`](file:///home/simon-ma/JEPA/model/psco.py) 的线性骨架原型，**哪一个在真实多模态 JEPA 任务上更好，只能靠具体实验判定**，本小节不作先验取舍。

### 3.8 重建作为预测的特例：统一预测框架

把 §3.7 的"潜空间 vs 表征空间"进一步统一：**重建不是预测的对立面，而是预测的退化特例**——二者是同一预测框架的两端，而非两个范式。

**统一预测框架。** 表征空间的学习目标统一写成：

$$\hat z = P_\phi(c,\ h),\qquad \mathcal L = d\big(\hat z,\ z_y\big). \tag{27}$$

$c=E_\theta(x_c)$ 是上下文表示，$h$ 是自推断的隐含模式（描述/推理倾向），$z_y=E_\xi(x_y)$ 是目标表示，$d$ 是表征空间度量。**不同任务 = 不同的 $(x_c,h,x_y)$ 三元组**：

| 任务 | 上下文 $x_c$ | 隐含模式 $h$ | 目标 $x_y$ |
|---|---|---|---|
| **重建** | $x$（完整） | $h_{\text{desc}}$ | $x$ |
| 掩码补全 | $M(x)$ | $h_{\text{desc}}$ | $x$ |
| 未来预测 | $z_t$ | $h_{\text{desc}}$ | $z_{t+1}$ |
| 推理/变换 | $z_t$ | $h$ | $\Psi_h(z_t)$ |
| 反事实/假设 | 前提 | $h_{hyp}$ | 虚拟后果 |

重建是第一行：上下文 = 完整目标 = 当前状态，且 $h$ 落在描述端。两个退化条件**合起来**才退化为重建：(1) 目标 = 当前自身 $x_y=x$；(2) 描述端 $h=h_{\text{desc}}$。缺一不可——只给 $h=h_{\text{desc}}$ 但目标设 $z_{t+1}$ 是未来预测；目标设 $x$ 但 $h$ 落在改状态端是"按变换回到当前"。

**统一训练目标。** 令 $\mathcal D$ 为"上下文-隐含模式-目标"三元组的采样分布：

$$\mathcal L = \mathbb E_{(x_c,\ h,\ x_y)\sim \mathcal D}\Big[ d\big(P_\phi(c,h),\ z_y\big) \Big]. \tag{28}$$

训练 = **设计 $\mathcal D$ 采样哪些三元组**：只采第一行 → 退化回潜空间/重建路线；采全部五行 → 完整表征空间路线。

**含义。** ① 架构不需二选一——重建是预测框架里"目标=当前、描述端"的一种采样，表征空间**天然包含**重建，而非与之对立；② 潜空间路线的真正问题不是"另一种训练方式"，而是**把特例当全部**（只采第一行，困在"还原已有观测"）；③ 前文各节在统一框架下归位——掩码 JEPA（§2.2）= 第二行、推理/变换（§3.5）= 第四行、反事实（§3.7）= 第五行、防塌缩重建项（§2.2.4）= 把第一行混入训练 batch，它从来不是独立目标，而是同一预测目标的特例采样；④ 从重建走向表征空间，不是换架构，而是**扩大 $\mathcal D$ 的采样范围**——把"预测当前"扩成"预测未来/变化/反事实"，正接上 §3.7"反事实能否沉淀，取决于预测目标是否覆盖构造虚拟状态"。

**采样分布 $\mathcal D$ 的设计策略。** 式 28 把训练归结为设计 $\mathcal D$，具体策略如下：

**(1) 任务权重课程 $\alpha_k(t)$。** 五类任务各有采样权重，随步数按难度递进：

| 任务 $k$ | 难度 | 权重趋势 | 数据来源 |
|---|---|---|---|
| 重建 | 最低（特例） | $0.3\to 0.05$（保底锚） | 任意单模态 |
| 掩码补全 | 低 | $0.5\to 0.3$（主体） | 任意单模态 |
| 未来预测 | 中 | $0\to 0.2$（中期引入） | 时序/状态序列 |
| 推理/变换 | 中高 | $0\to 0.25$（中期引入） | 推理-变换对 |
| 反事实 | 最高 | $0\to 0.1$（后期引入） | 假设-后果对 |

约束 $\sum_k\alpha_k(t)=1$。重建权重**永不归零**——它是"预测的特例锚点"，保留小权重持续提供稳定梯度、防表征塌缩（承接 §2.2.4 的重建防塌缩与 §3.4 的锚定思想）。

**(2) 任务内参数采样。** 每类任务内部还有"难度旋钮"，随课程递增：

- 掩码比例 $m\sim\mathcal U(0.2,0.95)$（§2.2.1，已有）；
- 未来时距 $k$：从 $k=1$ 短程逐步增至 $K_{max}$（长程预测更难）；
- 指令复杂度：从生成元（模糊/变色/移动，§3.5）逐步到组合指令；
- 反事实偏离度：从轻微偏离（近似真实轨迹）逐步到强反事实（虚拟状态）。

**(3) 混合 batch + 锚定。** 每步按 $\alpha_k(t)$ 采样任务类型，再采该任务样本，同一 batch 混合多任务，让共享 $P_\phi,E_\theta$ 同时学到多种预测能力；重建的低权重持续存在 = 锚定，防止预测任务把表征带向塌缩。

**(4) 数据约束下的优先级。** $\mathcal D$ 受数据可得性限制：重建/掩码无监督任意可得，未来预测需时序，指令需变换对，反事实需假设-后果对（稀缺）。故**先做数据丰富的任务（重建+掩码）打底，再逐步引入数据稀缺的高难度任务**——与 §3.4 的课程投喂同构：先易后难、先多后少。

一句话：**$\mathcal D$ 的设计 = "重建保底锚 + 掩码主体 + 未来/指令/反事实按难度与数据可得性递进引入"，每步按课程权重 $\alpha_k(t)$ 混合采样，重建权重永不归零以锚定表征、防塌缩。**

### 3.9 算子谱系：重建到生成的统一

§3.5 引入算子 $\Psi_h$ 与算子空间 $\Omega$，§3.7.3 把它落地为 PSCO，§3.8 把重建归入预测。本节把三者收束成一条**算子谱系**，并把「重建 / 生成」的区分形式化为「算子是否恒等」。

**状态与算子统一在 $Z$ 内。** 算子 $\Psi$ 不是 $Z$ 之外的独立对象，而是 $Z\to Z$ 的自映射：描述文字是 $Z$ 里的**点**（单位元邻域），指令文字是 $Z$ 上的**变换**，二者同属算子空间 $\Omega$。解码器 $D\in\{R_I,g,R_V,R_A\}$ 只把 $Z$ 渲染回模态，不承载「生成与否」的语义——区分重建/生成的，是**解码前 $Z$ 是否被算子变换过**：

$$\text{重建}\ =\ D\big(\mathrm{id}(Z)\big),\qquad
\text{生成}\ =\ D\big(\Psi(Z,\omega)\big),\quad \Psi\neq\mathrm{id}. \tag{29}$$

$\mathrm{id}$ 是恒等算子（$Z$ 不动，$D(Z)\approx$ 输入）；$\Psi(\cdot,\omega)$ 是非恒等算子（$Z\mapsto Z'$ 是新状态，$D(Z')$ 是输入之外的新东西）。于是「解码器做不做生成」是伪问题——**生成是算子赋予的，不是解码器赋予的**。旧方案里生成 = 离散码上的 Flow（须用量化器把 $Z$ 打成离散 token）；本框架生成 = 连续 $Z$ 上的算子，§3.7 的 $\Psi^{(N)}(Z)$ 全程不离开 $Z$，量化器的码空间角色被算子替代。

**算子空间与半群。** 令表征空间 $Z\subset\mathbb R^{L\times d}$，算子空间是 $Z$ 上的一族自映射，在复合下封闭：

$$\Omega\ =\ \big\{\,\Psi:Z\to Z\,\big\},\qquad
\mathrm{id}\in\Omega,\qquad
\Psi_2\circ\Psi_1\in\Omega\ \ (\text{半群}). \tag{30}$$

$\mathrm{id}$ 是单位元；$\Psi_2\circ\Psi_1$ 是复合——复杂生成 = 生成元复合（§3.5 的 $\Psi(Z,\omega_2\circ\omega_1)=\Psi(\Psi(Z,\omega_1),\omega_2)$）。

**算子谱系。** 按「偏离恒等的幅度」给 $\Omega$ 分层，得到一条从重建到生成的退化链：

| 层级 | 算子 | 数学形式 | 源码 | 语义 |
|---|---|---|---|---|
| $L_0$ | 恒等 $\mathrm{id}$ | $\mathrm{id}(Z)=Z$ | 直通 $E\to D$（无算子层） | **重建** |
| $L_1$ | 预测 $P_\phi$ | $Z_{\text{ctx}}\mapsto P_\phi(Z_{\text{ctx}})$ | [LatentPredictor](file:///home/simon-ma/JEPA/model/representation.py) | 掩码补全 / 未来预测 |
| $L_2$ | 收缩 $\Psi_h$ | $\Psi_h(Z)=Z+\Delta_h(Z),\ \rho(J)<1$ | [psco.py](file:///home/simon-ma/JEPA/model/psco.py) | 迭代推理 → 不动点 |
| $L_3$ | 一般算子 $\Psi(\cdot,\omega)$ | $\Psi:Z\times\Omega\to Z,\ \omega\in\Omega$ | plan P5 的 $E^{\text{cmd}}$（暂缓） | 任意变换 / 生成 |

> **命题 11（重建是谱系的 $L_0$，生成是 $L_{\ge1}$）**：重建与生成落在同一算子谱系两端——$\Psi=\mathrm{id}$（$L_0$）即重建，$\Psi\neq\mathrm{id}$（$L_{\ge1}$）即生成。二者不是两个范式，而是同一 $\Omega$ 按变换幅度的退化链
>
> $$\mathrm{id}\ \subset\ P_\phi\ \subset\ \Psi_h\ \subset\ \Psi(\cdot,\omega).$$
>
> 每层是上层的特例：$L_0$ 是 $L_1$ 当「上下文 = 目标」时（§3.8 式 27 第一行）；$L_1$ 是 $L_2$ 当门控收缩到单步时（§3.7.3）；$L_2$ 是 $L_3$ 当 $\Omega$ 取「对角收缩仿射 + 残差」子族时（§3.7.3）。

> *(Lean 已验证：`Semigroup.lean` : `generation_iff_not_reconstruction`（重建 ⟺ 恒等）、`generation_iff_moves_some_point`（生成 ⟺ 非恒等 ⟺ 移动某点）；§3.7.3 仿射幺半群见 `Affine.lean`，见 §1.0 对照表。)*

**幅度度量 $g_{\text{change}}$。** 「重建 vs 生成」的二分由连续量 $g_{\text{change}}=\sigma(W_{\text{ch}}h)\in(0,1)^d$（§3.7.3）度量：

$$\Psi_h(Z)\ =\ Z + g_{\text{change}}\odot\big(A_h Z + b_h + r_h(Z)\big). \tag{31}$$

$g_{\text{change}}\to0$ 时 $\Psi_h\to\mathrm{id}$（$L_0$，描述/重建端，$T_d=g(Z)$）；$g_{\text{change}}\to1$ 时 $\Psi_h$ 非平凡（$L_{\ge1}$，推理/生成端，$Z'=\Psi_h(Z)$）。**算子性 = $g_{\text{change}}$ 偏离 0 的程度**，与 §3.7 的停止条件 $h_{\text{change}}\to0\iff$ 变换完成 $\iff$ 读出结论天然对齐。

**推论（量化器可省的算子学表述）。** 量化器的历史作用之一是把连续 $Z$ 打成离散 token 以支撑「离散码上的生成」。当生成被重定义为「连续 $Z$ 上的算子」后，这一作用被 $\Omega$ 的谱系结构整体替代——生成路径 $D\circ\Psi\circ E$ 全程不离开连续 $Z$，无需离散化；量化器剩余的唯一价值（commitment 防塌缩）由 §2.2.3 的 EMA、§2.2.4 的重建与方差、§3.7.3 的谱夹层接管。故在本框架下量化器非必要，AGENTS.md 的「$Z$ 保持连续、不引入语义量化瓶颈」两条硬约束在算子谱系下自然成立。

---

### 3.10 替代结构：Looped Z-Reasoner——把「Z 内 N 步算子复合 / 解码流」折叠为权重共享的不动点回环

> **动机**。§2.10 把 **Z 的生产者**（编码器主干 $E\to Z$）折叠成了权重共享的内部循环；现实中 Loop Transformer / DEQ / MR-Loop（§2.10 已列文献）走的是同一条路。本节处理**正交的另一端**：把 **Z 内的消费者**——推理算子 $\Psi_h$ 的 $N$ 步复合 $\Psi^{(N)}$（§3.5 / §3.7）与解码端流 $v_\psi$ 的 $N$ 步 Euler 积分（§4.1）——折叠为**单一权重共享算子迭代到不动点**，让计算在模型内部沿 $Z$ 产生回环。§3.7.1 已把 Loop Transformer / DEQ 标为 $\Psi_h$ 的参考锚点，§3.7.3 的 PSCO 是它的并行扫描实现，[`psco.py`](file:///home/simon-ma/JEPA/model/psco.py) 已落地线性骨架原型；但它们都停留在**机制描述**，没有像 §2.10 那样给出**收益 / 代价的数学推导 + Lean 机械化**。本节补上这块，并处理 $\Psi_h$ 的**状态依赖性**（$h$ 跨步自推断）这一让 Z 回环区别于编码器回环的核心难点。
>
> **与 §2.10 的关键区别（勿混）**：
> 1. **循环对象不同**：§2.10 循环「输入 $\to Z$」的编码器块 $B_\theta$（样本内权重固定、$h$ 不跨步）；本节循环「$Z\to Z$」的推理算子 $\Psi_h$（$h$ 由网络自推断、可跨步演化）。
> 2. **独有收益不同**：§2.10 权重共享**不减串行深度**（仍是 $T$ 步串行）；本节配合 PSCO 并行扫描可把前向串行深度从 $O(N)$ 压到 $O(\log N)$（§3.10.3 表）——这是 Z 内计算回环相对编码器回环的**独有算力收益**。
> 3. **独有代价不同**：§2.10 的代价是梯度衰减；本节额外付出两个代价：① **收缩约束**——不动点迭代只对收缩算子收敛，膨胀型算子被排除（`expanding_diverges`，§3.10.4）；② **状态依赖破坏不动点**——$h$ 未收敛时不存在 $Z^*=\Psi_h(Z^*)$，无法用 DEQ 隐式微分，$O(1)$ 显存收益丢失（§3.10.4 第 4 条）。
>
> 二者**正交可叠加**：先用 $E_{\text{loop}}$（§2.10）产出 $Z$，再用 $\Psi_{\text{loop}}$ 在 $Z$ 上推理 / 生成；解码端可把 $N$ 步 Euler 流也折进同一回环（§4.1.5 reflow 是其拉直版）。

#### 3.10.1 定义

记 $Z\in\mathbb R^{L\times d}$（$d:d_{\text{hidden}}=1:8$ 不变，§2.2.7）、单算子参数数 $p_\Psi$、隐含模式网 $H$ 参数数 $p_h$。

**展开基线**（§3.5 / §3.7 现状）：$Z$ 内推理是 $N$ 个可相异的算子实例复合（每步 $h^{(i)}$ 自推断，算子参数也可随步独立）；解码端 Euler 是 $N$ 次独立速度场求值：

$$
Z^{(0)}=Z_{\text{in}},\quad Z^{(k+1)}=\Psi_{\theta_k,\,h^{(k)}}\!\big(Z^{(k)}\big)\ (k=0,\dots,N-1),\qquad Z_{\text{out}}=r\big(Z^{(N)}\big). \tag{LZR-1}
$$

总参数 $P_{\text{ff}}=N\cdot p_\Psi+P_{\text{readout}}$（每步算子各一套参数；读出头不参与步间共享）。解码端同理：$N$ 次 $v_\psi(\cdot,k\Delta t,c)$ 求值，参数 $\psi$ 虽共享但**算子身份随 $t$ 变化**（时间条件化），跨 $t$ 不构成权重绑定意义上的回环。

**回环推理器（Looped Z-Reasoner）**：单一共享算子 $\Psi_\theta$，$h^{(k)}=H(Z^{(k)})$ 跨步自推断，迭代 $T$ 次（或到不动点）后读出：

$$
\boxed{\ Z^{(0)}=Z_{\text{in}},\quad Z^{(k+1)}=\Psi_\theta\!\big(Z^{(k)};\,h^{(k)}\big),\quad h^{(k)}=H\!\big(Z^{(k)}\big)\ (k=0,\dots,T-1),\quad Z_{\text{out}}=r\!\big(Z^{(T)}\big)\ } \tag{LZR-2}
$$

总参数 $P_{\text{loop}}=p_\Psi+p_h+P_{\text{readout}}$（算子只存一份；$h$ 网另算，通常 $p_h\ll p_\Psi$）。

**线性化骨架**（在稳定段、$h$ 收敛到 $h^*$ 处，取 $\Psi_\theta$ 的线性部分，§3.7.3 式 (30)）：

$$
Z^{(k+1)}=M_{h^*}Z^{(k)}+b_{h^*},\qquad M_{h^*}=I+\nabla v_{h^*}(Z^*), \tag{LZR-3}
$$

与 §3.7.3 PSCO、§2.9 CARO、§2.10 MR-Loop 是**同一个仿射不动点迭代**（`Affine.lean` 幺半群），收敛性直接复用 `Contractive` + `AssocMem`。

**两种模式必须分开**（这是 Z 回环与编码器回环的核心数学差异）：

- **(a) 固定模式**（$h^{(k)}\equiv h^*$，描述端 $g_{\text{change}}\to0$ 或推理到达结论段）：退化为单算子仿射迭代 $Z^{(k+1)}=M Z^{(k)}+b$，满足单算子不动点方程 $Z^*=MZ^*+b$，可用 DEQ 隐式微分（式 (25)），$O(1)$ 显存。
- **(b) 状态依赖模式**（$h^{(k)}=H(Z^{(k)})$ 跨步变化，推理进行中）：**非自治**迭代，$Z^{(k+1)}=\Psi_{h^{(k)}}(Z^{(k)})$ 中算子随步改变，**不满足任何单算子不动点方程**；只能保证「轨迹稳定 $\Rightarrow$ 稳定点是终算子的不动点」（`FixedPoint.stabilized_is_fixed_point`，已证）。这一段必须显式展开 $m$ 步（§3.7.3 两段式的瞬态段），$O(1)$ 显存收益**不成立**。

#### 3.10.2 线性化骨架与收敛（DEQ 极限）

**(a) 固定模式。** 当 $\rho(M_{h^*})<1$（§3.7.1 收缩条件，谱夹层 $\delta<\tilde a_{h,i}<1-\gamma$ 是其逐维加强，见 README 问题 6）时：

1. 存在唯一不动点 $Z^*=(I-M_{h^*})^{-1}b_{h^*}=\sum_{j\ge0}M_{h^*}^j b_{h^*}$（Neumann 级数，`Contractive.geom_sum_one_sub`）；
2. 误差几何衰减 $Z^{(k)}-Z^*=M_{h^*}^k(Z^{(0)}-Z^*)$，$\|Z^{(k)}-Z^*\|\le K^k\|Z^{(0)}-Z^*\|$，$K=\|M_{h^*}\|<1$（`reasoner_tendsTo_fixedPoint`，推广自 §2.10 的 `mrLoop_tendsTo_fixedPoint`，初值取任意 $Z_{\text{in}}$ 而非 $0$）；
3. $T$ 足够大时 $Z^{(T)}\approx Z^*$，读出 $r(Z^*)$ 与 $T$ 无关（收益递减，可早停）；反传用 DEQ 隐式微分（式 (25)），只需留存 $Z^*$。

**反算子塌缩（模式分离）**：对固定线性部分 $M$，偏移 $b$ 单射地决定不动点——$b_1\neq b_2\Rightarrow Z^*_1\neq Z^*_2$（`fixedPoint_diff_b_diff_fp`，对任意线性 $M$ 成立，无需收缩）。含义：不同推理指令（不同 $b_h$）给出不同结论不动点，$\Psi_h$ 不会把所有指令坍缩到同一答案。这与 `mrLoop_no_input_collapse`（§2.10，守住 $Z$ 对输入的依赖）形成一对：输入注入守住「$Z$ 依赖输入」，模式分离守住「结论依赖指令」。

**(b) 状态依赖模式。** $h^{(k)}$ 跨步变化时，单算子不动点方程无意义。已证（`FixedPoint.stabilized_is_fixed_point`）：若轨迹在第 $N$ 步稳定（$Z^{(N+1)}=Z^{(N)}$），则 $Z^{(N)}$ 是**终算子** $\Psi_{h^{(N)}}$ 的不动点。这正是 §3.7「结论 $=$ 变换链不动点」的精确成立条件：只有当 $h$ 收敛（改分量 $h_{\text{change}}\to0$、$\Psi_h\approx\mathrm{id}$）时，稳定点才是不动点；推理进行中（$h$ 仍在变）的中间态不是任何算子的不动点，不能用 DEQ 隐式微分。

#### 3.10.3 收益与代价：参数 / 算力 / 串行深度 / 显存 / 表达力 / 优化

下表对比展开基线（深度 $N$）与回环推理器（迭代 $T$、单层算子共享 + 状态依赖 $h$），形式化了的行标注对应 Lean 定理。

| 维度 | 展开基线（$N$ 步，算子可相异） | 回环 $\Psi_{\text{loop}}$（$T$ 迭代，算子共享 + $h$） | 说明 / Lean |
|---|---|---|---|
| **参数量**（容量代价） | $N\cdot p_\Psi+P_{\text{readout}}$ | $p_\Psi+p_h+P_{\text{readout}}$ | $p_h\le(N-1)p_\Psi$ 时省（`param_loop_le_free`）；$h$ 网通常远小于算子网，故显著省；若每步 $h$ 需高容量，$p_h$ 吃掉收益 |
| **算力 FLOPs** | $N\cdot C_\Psi$ | $T\cdot C_\Psi$ | 同深度 $T=N$ 时**渐进持平**（`eval_isBigO_T`，复用 §2.10）；省下的参数预算可加宽单层算子或加深 $T$ |
| **串行深度**（墙钟关键路径） | $O(N)$ | $O(\log T)$（配 PSCO 并行扫描） | **本节独有收益**：扫描深度 `scanDepth` 满足 `scanDepth(2T)=scanDepth(T)+1`（对数增长），而展开 `unrolledDepth(2T)=2·unrolledDepth(T)`（线性增长）；`scanDepth≤unrolledDepth`（`scanDepth_double` / `scanDepth_le_unrolled`）。编码器回环（§2.10）无此收益 |
| **训练显存（激活）** | $O(N\cdot L\cdot d)$ | 朴素 $O(T)$；检查点 / DEQ $O(1)$ | 复用 §2.10：朴素 $=O(T)$（`baseline_memory_isBigO_T`）且无界（`baseline_memory_unbounded`），检查点 / DEQ $=O(1)$（`checkpoint_memory_isBigO_one` / `deq_memory_isBigO_one`）；**仅固定模式 (a) 可用 DEQ**，状态依赖模式 (b) 必须展开 |
| **权重访存** | 每步读 $N$ 套不同权重 | 每 $T$ 步只读 $1$ 套算子（可驻留缓存） | 带宽受限加速器上降访能、提吞吐（同 §2.10） |
| **推理** | 固定 $N$ 步 | 可自适应停步（残差阈值，PSCO 早停免费） | 简单样本早停、难题跑满，GPU 深度都是 $O(\log T_{\max})$（§3.7.3） |
| **表达力（同深度）** | 各步算子独立（全函数空间） | 各步算子 $\in\operatorname{im}\Psi$（$\Psi$ 的值域） | 固定模式 = 全绑定对角子集（§2.10 LE-9）；状态依赖模式 = 各步在 $\operatorname{im}\Psi$ 内可相异，**介于全绑定与全独立之间**：$\text{tied}\subseteq\text{loop}(\Psi)\subseteq\text{free}$（`tied_is_sdl` / `sdl_realized_by_feedforward`） |
| **表达力（同参数预算）** | 深度受限 $N\approx P/p_\Psi$ | 可设 $T\gg N$（同参数下更深） | 「同参更深」潜在收益；但受梯度衰减 + 收缩约束双重限制 |
| **优化难度**（梯度代价） | 各步雅可比独立 | 共享权重 $\Rightarrow$ 回传 $\sum_t M^{T-t}$，几何衰减 | 复用 §2.10 `gradient_decay_with_K`；BPTT 截断 $\sum_{k<L}J^k$ 与 DEQ 完整逆 $(I-J)^{-1}$ 的关系由 Neumann 级数极限刻画（`geometric_series_limit`） |
| **收缩约束**（**新代价**①） | 无（每步可膨胀） | 必须 $\rho(M_h)<1$ 才收敛到唯一吸引不动点 | 膨胀型算子的不动点迭代**发散**：$x\mapsto Kx$（$K>1$）从 $x_0\neq0$ 迭代 $\|K^n x_0\|\to\infty$（`expanding_diverges`）——回环只能求吸引不动点，能表达的算法类受限 |
| **状态依赖代价**（**新代价**②） | 无 | $h$ 未收敛时无单算子不动点，DEQ 失效 | 瞬态段必须展开 $m$ 步（§3.7.3 两段式），$O(1)$ 显存收益在该段丢失；这是 $\Psi_h$ 状态依赖相对 Loop Transformer（固定算子）的额外代价 |
| **数值求解**（**新代价**③） | 无 | DEQ 每前向需 Picard / Anderson 迭代到收敛 | 额外算力 + 求解精度 / 重启策略是工程量（未形式化） |

> **诚实声明**：本表的**渐进复杂度 $O(\cdot)$ 与递推对比已在 Lean 中形式化**（`scanDepth` 对数增长 vs `unrolledDepth` 线性增长；显存 / 算力 big-O 复用 §2.10；表达力三层包含；不动点收敛；梯度衰减；Neumann 级数极限；膨胀发散）。**仍为设计断言、未形式化**：常数因子、真实 GPU 墙钟 / 访存带宽 / kernel 融合、非线性 $\Psi_h$ 的表达力与「同参更深」泛化收益、DEQ 数值稳定性、状态依赖段 $m$ 的最优取值——需实测。三档取舍见 §3.10.6 第 7 条。

#### 3.10.4 主要代价：梯度衰减、收缩约束与状态依赖非自治性

**1. 梯度衰减（复用 §2.10.4）。** 权重共享回环的反传梯度是各步贡献之和。线性化后第 $t$ 步信号穿过 $T-t$ 层共享雅可比 $M^\top$：

$$
\frac{\partial\ell}{\partial\theta}\Big|_{\text{来自 }Z^{(t)}}\propto\big(M^\top\big)^{T-t}\nabla_{Z^{(T)}}\ell,\qquad \|M^n v\|\le\|M\|^n\|v\|\le K^n\|v\|,\ K=\|M\|<1. \tag{LZR-4}
$$

（`gradient_decay_with_K`，仅依赖 `le_opNorm`。）$K$ 越小不动点收敛越快（LZR-3 第 2 条），但早期迭代梯度被 $K^{T-t}$ 杀得越狠——二者此消彼长，故必须谱夹层 $\delta<K<1-\gamma$（§3.7.3）而非一味取小 $K$。

**BPTT vs DEQ 的 Neumann 级数刻画（式 (34) 的形式化）。** §4.1.5 式 (34) 把 $L$ 步 BPTT 梯度写成截断 Neumann 级数 $\sum_{k=0}^{L-1}J^k$，DEQ 写成完整逆 $(I-J)^{-1}$。其代数核心已在 `Contractive.geom_sum_one_sub` 机械化：$(1-a)\sum_{i<k}a^i=1-a^k$；本节补极限：当 $|a|<1$ 时部分和 $\to\frac{1}{1-a}$（`geometric_series_limit`），即

$$
\sum_{k=0}^{L-1}a^k\xrightarrow{L\to\infty}(1-a)^{-1}. \tag{LZR-5}
$$

这就把「有限步 CoT / BPTT = 截断级数（有偏）」与「结论 $=$ 不动点 + DEQ = 完整逆（无偏）」的关系落到了定理：$L\to\infty$ 且 $\rho(J)<1$ 时二者相等（标量版；矩阵版同构，见 §4.1.5 末段「只在 $h$ 固定时严格成立」）。

**2. 收缩约束（表达力 / 容量代价，本节新代价①）。** 回环求不动点要求 $\Psi$ 在不动点附近是收缩（$\rho(J)<1$）。这是展开基线没有的约束——展开 $N$ 步中某些步可以是膨胀的（先把状态推开、后续再拉回），只要整体复合把端点映射好。回环则要求**每一步的迭代映射**都朝向吸引不动点，膨胀算子的不动点**迭代不可达**：

> 取 $f(x)=Kx$（$K>1$），其不动点是 $0$，但从任意 $x_0\neq0$ 迭代 $f^n(x_0)=K^n x_0$，$\|K^n x_0\|\to\infty$（`expanding_diverges`）。

故**回环只能求吸引不动点，能表达的算法类被限制在「收缩复合」内**。谱夹层（§3.7.3 五道防线第 1 条）是缓解：它把算子限制在逐维对角 $(\delta,1-\gamma)$，既保收敛（上界）又封死一步塌缩（下界），但代价是进一步缩小了算子族（排除负特征值振荡与强膨胀，见 README 问题 6）。展开基线无此限制——这是回环为「同参更深 + $O(\log N)$ 深度 + $O(1)$ 显存」付出的**表达力代价**。

**3. 状态依赖非自治性（本节新代价②）。** Loop Transformer / 经典 DEQ 的算子固定（$F_\theta(Z;x)$，$x$ 样本内不变），故始终满足单算子不动点方程。本框架的 $\Psi_h$ 是**状态依赖**的（$h^{(k)}=H(Z^{(k)})$ 跨步演化），推理进行中 $h$ 不收敛 $\Rightarrow$ 无 $Z^*=\Psi_h(Z^*)$ $\Rightarrow$ DEQ 隐式微分（式 (25)）的前提不成立 $\Rightarrow$ 必须展开该段 rollout，$O(1)$ 显存收益丢失。§3.7.3 的两段式（瞬态段 $m$ 步显式 + 渐近段 DEQ）是应对：只有 $h$ 收敛后的渐近段能享受 $O(1)$ 显存。这是 $\Psi_h$ 比 Loop Transformer 多付的代价，也是「状态依赖算子」相对「固定算子迭代」的本质差异（§3.7.1 末段已点出）。

#### 3.10.5 表达力包含的精细版：回环 $=$ 各步取自 $\operatorname{im}\Psi$ 的前馈

§2.10 (LE-9) 证明回环 $f^{\circ T}$ 是同深度前馈的**全绑定对角子集**（所有层取同一个 $f$）。Z 回环因 $h$ 跨步自推断，表达力落在**全绑定**与**全独立**之间。精确表述（纯函数代数，已在 Lean 机械化）：

令前馈网为层列表复合 $\mathrm{feedforward}([f_1,\dots,f_N])=f_1\circ\cdots\circ f_N$；状态依赖回环为 `sdl Ψ h T`：从 $Z^{(0)}$ 起依次作用 $\Psi(h_0),\Psi(h_1),\dots,\Psi(h_{T-1})$；全绑定回环为 $f^{\circ T}$（$h$ 取常值）。则

$$
\underbrace{\{\,f^{\circ T}:f\,\}}_{\text{全绑定（§2.10）}}\ \subseteq\ \underbrace{\{\,\mathrm{sdl}\,\Psi\,h\,T:\Psi,h\,\}}_{\text{状态依赖回环}}\ \subseteq\ \underbrace{\{\,\mathrm{feedforward}([f_1,\dots,f_T]):f_i\,\}}_{\text{全独立前馈}}. \tag{LZR-6}
$$

- 左包含（`tied_is_sdl`）：取 $h$ 为常值函数，全绑定 $f^{\circ T}$ 实现为状态依赖回环；
- 右包含（`sdl_realized_by_feedforward`）：任意状态依赖回环都能被一个层列表实现——层 $i=\Psi(h_i)$，全部落在 $\operatorname{im}\Psi$ 内；
- **反向均不成立**：全独立前馈可让各层取到 $\operatorname{im}\Psi$ 之外的函数；状态依赖回环可让相邻层不同（$h$ 变），故真包含全绑定。

**容量代价的精确含义**：回环省下参数的代价，是把每一步算子限制在 $\operatorname{im}\Psi$ 这一个网络的值域内；$h$ 恢复了「步间可相异」的自由度，但不能突破 $\operatorname{im}\Psi$ 的表达上界。当 $h$ 恒定时退化到 §2.10 的全绑定（最弱表达力）；当 $h$ 跨步充分变化且 $\Psi$ 足够宽时逼近全独立——但逼近程度取决于 $\Psi$ 的宽度与 $h$ 的容量，是「同参数预算下如何分配 $p_\Psi$ 与 $p_h$」的设计问题（未形式化，依赖实测）。

#### 3.10.6 契约核对（对照 AGENTS 规则与 math.md 既有前提）

1. **$Z$ 保持连续、不引入语义量化瓶颈**：回环迭代全程在 $\mathbb R^{L\times d}$ 连续值上进行（LZR-2/3），不施加 FSQ/VQ/RVQ（同 §2.6 / §2.9.5 / §2.10.6 契约 1）。✅
2. **反塌缩**：① 输入侧由 $E$（§2.2.4）守住；② 算子侧由模式分离 `fixedPoint_diff_b_diff_fp`（不同指令 $\Rightarrow$ 不同结论不动点）+ 读出头锚定（$\mathcal L_{\text{readout}}$，§3.7.3 防线 2）+ 方差正则（防线 3）+ 谱夹层（防线 1，封死 $\tilde A_h\to0$ 一步塌缩）共同守住；③ `expanding_diverges` 反向说明收缩约束本身封死了「算子膨胀后失稳」这一塌缩路径。✅
3. **$d:d_{\text{hidden}}=1:8$**：回环只改 $Z$ 内计算的参数组织，不改维度比例；读出头 $r:Z\to$ 输出与算子隐藏维仍按 §2.2.7。✅
4. **通路 / 关系分离（§4.5）**：回环迭代是关系聚合（对 $Z$ 反复重组），读出 $r$ 是逐位置通路映射，二者仍分离；关系从「$N$ 个相异算子复合」变为「同算子族的 $T$ 次复合」，通路仍 $O(L)$ 独立。✅
5. **与 §3 对齐推理的衔接**：回环只替换 $Z$ 内**计算**的组织方式（$N$ 步展开 $\to$ $T$ 步共享迭代），产出的仍是连续 $Z$；§3.1 硬对齐、§3.2 交叉掩码、§3.4 PAC、§3.8 统一预测框架在 $Z$ 之上照常进行，契约不变。✅
6. **与 §2.10 / §3.7.3 的正交性**：§2.10 循环编码器（输入 $\to Z$），本节循环推理算子（$Z\to Z$），§3.7.3 PSCO 是本节的并行扫描实现——三者可叠加：$E_{\text{loop}}$ 产 $Z$，$\Psi_{\text{loop}}$（PSCO 加速）在 $Z$ 上推理，互不冲突。本节给 PSCO 补上 §2.10 级别的收益 / 代价形式化，不替代它。✅
7. **三档规模可扩展性（AGENTS 模型规模规则）**：
   - **小规模（CPU 冒烟，$\le10$M）**：现有 `tiny`/`small` 的 $N$ 很小，回环收益不明显，但可直接设 $T=N$ 验证算法正确性（损失下降、不动点收敛 `reasoner_tendsTo_fixedPoint`、模式分离 `fixedPoint_diff_b_diff_fp`、膨胀发散被谱夹层排除 `expanding_diverges`），纯 CPU 可跑（回环算子是纯 PyTorch，无 GPU-only 依赖，不进入 `pyproject.toml` 可选 extra）。
   - **中型（GPU + 真实数据，0.3B–7B）**：主要受益档。参数 $\div N$ 释放的预算可加宽算子或转给对齐 / 生成头；**$O(\log N)$ 串行深度**（PSCO）是长推理 / 长 CoT 的关键墙钟收益；长上下文下跨循环梯度检查点把激活显存从 $O(N\cdot L\cdot d)$ 降到 $O(L\cdot d)$ 乃至 $O(1)$（DEQ）。支持 bf16/AMP、`torch.compile`、梯度检查点（成熟库能力）。
   - **大规模（scaling 极限）**：权重共享是规模化刚需；DEQ 隐式微分使有效迭代深度 $T$ 可远超显存允许的展开深度，$O(1)$ 激活支撑极深有效推理深度，用于检验 scaling law 的收益 / 拐点与收缩约束下的表达力极限（`expanding_diverges` 划定的算法类边界）。若引入 GPU-only 隐式求解器，按 AGENTS 规则放入 `pyproject.toml` 可选 extra，保证第 1 档仍纯 CPU。

#### 3.10.7 Lean 形式化（见 §1.0 对照表，全部无 sorry / 无新增公理）

(LZR-1)–(LZR-6) 的可证核心已在 [`lean-math/LeanMath/LoopedZReasoner.lean`](file:///home/simon-ma/JEPA/lean-math/LeanMath/LoopedZReasoner.lean)（命名空间 `LoopedZReasoner`）机械化；收敛性复用既有 `Affine` / `Contractive` / `AssociativeMemory` / `FixedPoint` / `LoopedEncoder`：

| 断言 | 定理 | 对应公式 |
|---|---|---|
| (LZR-6) 表达力三层包含：全绑定 $\subseteq$ 状态依赖回环 $\subseteq$ 全独立前馈 | `tied_is_sdl` / `sdl_realized_by_feedforward` | §3.10.5 |
| 参数量：$p_\Psi+p_h\le N p_\Psi$（$p_h\le(N-1)p_\Psi$）⇒ 回环省参 | `param_loop_le_free` / `param_loop_lt_free` | §3.10.3 |
| 反算子塌缩（模式分离）：固定 $M$ 下 $b_1\neq b_2\Rightarrow$ 不动点不同 | `fixedPoint_diff_b_diff_fp` | §3.10.2 |
| (LZR-4) 梯度衰减 $\|M^n v\|\le K^n\|v\|$（优化代价，复用 §2.10） | `gradient_decay_with_K` | §3.10.4 |
| (LZR-5) Neumann 级数极限 $|a|<1\Rightarrow\sum_{i<k}a^i\to(1-a)^{-1}$（BPTT vs DEQ，式 34） | `geometric_series_limit` | §3.10.4 / §4.1.5 |
| 收缩约束代价：$K>1$ 时膨胀算子迭代 $\|K^n x_0\|\to\infty$（不动点不可达） | `expanding_diverges` | §3.10.4 |
| 串行深度：扫描 `scanDepth(2T)=scanDepth(T)+1`（对数）vs 展开 `unrolledDepth(2T)=2·unrolledDepth(T)`（线性）；`scanDepth≤unrolledDepth` | `scanDepth_double` / `scanDepth_le_unrolled` / `scanDepth_isBigO_unrolled` | §3.10.3 |
| DEQ 极限：$\|M\|<1$ 时从任意初值 $Z_{\text{in}}$ 迭代 $\to$ 不动点 $Z^*$（推广 §2.10 `mrLoop_tendsTo_fixedPoint`） | `reasoner_tendsTo_fixedPoint` | §3.10.2 |

> **形式化边界（诚实）**：Lean 证明的是**表达力包含（函数代数）、参数计数不等式、不动点对偏移的单射性、算子范数幂次界、Neumann 级数极限、膨胀发散、串行深度递推对比、big-O 与不动点收敛**——代数结论对任意域 $K$ / 任意模 $V$ 成立；分析结论（极限、big-O）对 $T\to\infty$ / $k\to\infty$ 的 `Filter.atTop` 成立。**未形式化**：常数因子与真实墙钟 / 访存 / kernel 融合、非线性 $\Psi_h$ 表达力与「同参更深」泛化、DEQ 数值稳定性与求解精度、状态依赖段 $m$ 的最优取值、$\operatorname{im}\Psi$ 逼近全独立的程度——均为设计分析，依赖实测，见 [lean-math/README.md](file:///home/simon-ma/JEPA/lean-math/README.md)「问题 12」。线性化骨架（LZR-3）是数学分析对象，真实块的非线性行为需实验验证（同 §2.9 / §2.10 / §3.7.3 的边界声明）。

### 3.11 时间分层与慢/快谱分离（借鉴 H-JEPA）

> **问题来源**。H-JEPA（LeCun 团队 2026，`arXiv:2610.06805`）用「L 层动作条件 JEPA 堆叠、每层在**自己的潜空间**里预测更远期未来」实现时间分层，并报告「高层丢弃快变不可预测细节、保留慢变任务相关状态」。本节把它放进本框架的数学语言，回答两个问题：**(Q1)** 「可预测性 → 保留/丢弃」作为补充判据是否数学有利；**(Q2)** 本框架是否已隐含时间分层、若没有该如何修正才能满足全部既有公式。

#### 3.11.1 Q1：可预测性判据的数学地位——收缩谱上的慢/快分离

§2.6 的 $\rho=\|v-Mk\|^2$ 是**单步**冗余（给定记忆后还剩多少），不区分「可预测时域」。但 §2.1/§3.7 已确立：目标 $I(Z;Y)$ 是**慢变、稳定、可预测**的语义因子，而「可预测」本身带**时域轴**——短时域可预测（快）≠ 长时域可预测（慢）。H-JEPA 的「留慢丢快」正是把这条时域轴显式化。

本框架**不需要引入新的时域参数**：收缩算子的谱本身就是时域轴。对角收缩 $M=\operatorname{diag}(\tilde a_h)$（§3.7.3 谱夹层 $\delta<\tilde a_{h,i}<1-\gamma$）的分量 $i$ 在 horizon $k$ 的持久性为 $\tilde a_{h,i}^k$，它随 $\tilde a_{h,i}$ 单调（`persistence_mono`）、快特征（小 $\tilde a$）以 $a^k\to0$ 衰减（`fast_decay`）、快分量相对慢分量以 $(a/b)^k\to0$ 趋于可忽略（`fast_slow_ratio`）。因此：

- **慢特征 ⟺ 大特征值**（$\tilde a\to1-\gamma$，长时域可预测，高层保留）；
- **快特征 ⟺ 小特征值**（$\tilde a\to\delta$，短时域可预测，低层/丢弃）。

于是「可预测性 → 保留/丢弃」判据的**代数核心是可判定的**：它是谱夹层下界 $\delta$ 的 horizon 化，而非启发式。这与 §2.6「测差值、别估熵」一致——把「差值」从单步 $\rho$ 推广到 horizon 参数化的显著性 $s^{(\tau)}=I(x;Y\mid \tilde x,\text{历史}\tau)$；其**互信息版本是分析命题（未形式化）**（依赖率失真测度论，同 §8.4 命题 12），但代数核心（谱单调 + 快衰减）已由 [`lean-math/LeanMath/TemporalHierarchy.lean`](file:///home/simon-ma/JEPA/lean-math/LeanMath/TemporalHierarchy.lean) 机械化。

#### 3.11.2 Q2：框架是否已隐含时间分层——关联扫描的跨步前缀

**结论：框架未以「独立潜空间堆叠」隐含时间分层，但同一收缩关联扫描的「跨步前缀」即是一层，无需新建任何算子。**

H-JEPA 的做法是堆叠 $L$ 个独立 JEPA（各自 $E^{(\ell)},A^{(\ell)},F^{(\ell)}$、独立潜空间 $Z^{(\ell)}$）。这**违反**本框架两条契约：单一共享 $Z$（§2.6）与「有损只发生在 token 层、$Z$ 全程连续、不量化」（§2.6）。强行照搬会引入 $L$ 个量化瓶颈与对齐边界。

但本框架已有更优的原语：§3.7.3 的关联扫描（Blelloch）**已经物化整条轨迹** $Z^{(0..N)}$。时间分层的第 $\ell$ 层（stride $s_\ell$）只需**读扫描在索引 $\{s_1\cdots s_\ell\cdot t\}$ 处的前缀**——这是 $O(1)$ 的索引选择、零额外前向。层次组合由仿射幺半群的结合性给出：$M^{a+b}=M^a\circ M^b$（`powL_add`），故「走 $a$ 步的层 + 走 $b$ 步的层 = 走 $a+b$ 步」，任意多层嵌套是其反复应用；跨步复合仍仿射、仍收缩（`iterate_still_contracting`：严格收缩 $f$ 的 $n$ 次迭代常数 $K^n<1$），因此**每一层的不动点收敛保证（Banach）由基层直接继承，无需为每层重证**。

#### 3.11.3 更优方案：谱时间分层（Spectral Timescale Hierarchy, STH）

把 3.11.1 与 3.11.2 合并，得到比 H-JEPA 更适合本框架的时间分层方案——**不新增潜空间，只用「收缩谱 + 跨步扫描」**：

1. **谱即层级**：收缩算子 $M$ 的对角谱 $\tilde a_i\in(\delta,1-\gamma)$ 天然把 $Z$ 的维度按「快慢」分层——大特征值维度 = 高层抽象态、小特征值维度 = 低层细节。这与 §3.7.3 谱夹层**同一对象**，反塌缩（下界 $\delta$ 防慢模式坍缩、上界 $1-\gamma$ 保收敛）不新增约束。
2. **跨步扫描即多分辨率**：关联扫描物化全轨迹，第 $\ell$ 层 = 读 stride $s_\ell$ 的前缀，$s_\ell$ 越大越「粗」；层次嵌套由 `powL_add` 保证，任意组合。
3. **GPU 特性**：与 §3.7.3 关联扫描的 $O(\log N)$、§3.10.3 串行深度判据一致——扫描 $O(\log N)$、跨步索引 $O(1)$、无新增 kernel；谱分解是逐元素乘加、`torch.compile` 自动融合。串行深度 $O(\log N)$，优于 H-JEPA 逐层 rollout 的 $O(\sum_\ell H_\ell)$。
4. **契约兼容**：$Z$ 单一、连续、不量化；有损仍只在 token 层（§2.6）；不新增模态边界。

**一句话**：H-JEPA 给本框架的价值不是「独立潜空间堆叠」，而是「时间分层 = 收缩谱的慢/快分离 + 关联扫描的跨步前缀」这一**零新增算子**的设计；本框架已有的收缩幺半群（§2.9/§3.7.3）恰好是它的最优载体，无需照搬 H-JEPA 的多潜空间。

**边界（诚实）**：「慢/快 = 大/小特征值」的对应依赖对角/低秩骨架（§3.7.3 的线性化假设）；真实非线性残差 $r_h(Z)$ 的时域分离是否同样干净需实验验证（同 §3.7.3/§2.9 的边界声明）。互信息版 horizon 显著性 $s^{(\tau)}$ 是分析命题、未形式化（§8.4 同款）。

#### 3.11.4 更深入的推导：不动点的跨步不变性（已 Lean 机械化）

3.11.2 断言「每层结论一致」——这需要一个更强的代数保证：**基层收缩映射的不动点（抽象结论）不随 stride 改变**。本小节给出机械化的证明链（[`TemporalHierarchy.lean`](file:///home/simon-ma/JEPA/lean-math/LeanMath/TemporalHierarchy.lean) 第 4 节）。

**几何级数伸缩恒等式**。对任意线性算子 $M$ 与 stride $s$，几何部分和 $\Sigma_{i<s}M^i$ 满足

$$(id-M)\circ\Big(\textstyle\sum_{i<s}M^i\Big)\;=\;id-M^s \qquad\text{(`geom_series_telescope`)}$$

这是「粗算子不动点方程 = 基层不动点方程」的代数根基：粗算子（stride $s$）是仿射对 $(M^s,\ \Sigma_{i<s}M^i b)$，其不动点方程是 $(id-M^s)z=\Sigma_{i<s}M^i b$；由伸缩恒等式 $(id-M^s)=(id-M)\circ(\Sigma M^i)$，二者在 $(id-M)$ 可逆时**等价**。

**不动点的跨步不变性（正向）**。设 $z$ 是基层不动点：$Mz+b=z$。则对任意 $k$ 迭代稳定 $z=\text{iterate}(M,b,k,z)$（`fixed_point_iterate`），结合闭式（§3.7.3 `iterate_eq_closedForm`）得

$$z \;=\; M^s z+\textstyle\sum_{i<s}M^i b \qquad\text{(`fixedPoint_stride_invariant`)}$$

即 $z$ 同时是任意 stride $s$ 粗算子 $(M^s,\Sigma M^i b)$ 的不动点。**抽象结论（不动点）不依赖时间分辨率**——这是谱时间分层「每层同一 $Z$、同一结论、只差分辨率」的数学保证，也是它与 H-JEPA「每层独立潜空间」的根本区别：H-JEPA 每层收敛到**不同**潜空间的**不同**不动点，本框架每层收敛到**同一** $Z$ 的**同一**不动点。

**速率-分辨率权衡**。对收缩常数 $K\in[0,1]$，$K^s\le K$（$s\ge1$，`rate_mono_stride`）。故 stride 越大（层越粗），Banach 迭代收敛**越快**（常数 $K^s$ 越小），但每步覆盖 horizon 越**长**、时间分辨率越**粗**——一个显式的速率/分辨率权衡，可为每层选最优 stride。

**未形式化的边界**：反向（粗算子不动点 ⟹ 基层不动点）需要 $\Sigma_{i<s}M^i$ 可逆，等价于 $M$ 无 $s$ 次单位根特征值；对严格收缩（谱半径 $\rho(M)<1$）恒成立，但「$\rho(M)<1\Rightarrow\Sigma M^i$ 可逆」的谱论未在此机械化（依赖 §3.7.1 收缩 / Neumann 级数，同 Contractive.lean 的边界声明）。

#### 3.11.5 最优 stride 的闭式与速率-分辨率权衡（已 Lean 机械化）

**问题设置**。给定收缩常数 $K\in(0,1)$、目标残余 $\varepsilon\in(0,1)$，一个 stride $s$ 的层每做一次粗迭代把残余乘以 $K^s$（§3.11.4 `lipschitz_iterate`）。问：达到残余 $\le\varepsilon$ 需要多少次粗迭代 $n$？

**闭式解（迭代次数）**。残余界是 $K^{s\cdot n}$（`pow_mul_stride`：$(K^s)^n=K^{s\cdot n}$，**只依赖总 horizon $s\cdot n$**）。要 $K^{s\cdot n}\le\varepsilon$，取对数（$K<1\Rightarrow\log K<0$）：

$$s\cdot n \;\ge\; \frac{\log(1/\varepsilon)}{\log(1/K)} \;=:\;\Lambda \qquad\Longrightarrow\qquad n_{\min}(s)=\left\lceil\frac{\Lambda}{s}\right\rceil$$

其中 $\Lambda=\log(1/\varepsilon)/\log(1/K)$ 是**总 horizon 预算**，与 stride $s$ 无关。

**关键结构：平权权衡（flat tradeoff）**。由 `pow_mul_stride`，残余界 $K^{s\cdot n}$ 只由乘积 $s\cdot n$ 决定；由 `residual_horizon_mono`，$H\le H'\Rightarrow K^{H'}\le K^H$（残余随总 horizon 单调递减）。故：

- **精度（残余）由总 horizon $\Lambda$ 决定，与 stride 分解无关**——这是 §3.11.4 不动点跨步不变性的定量形式：**无法靠「多分几层」买到精度**。
- **stride 只在「迭代次数 vs 分辨率」间做平权交换**：$n_{\min}(s)=\lceil\Lambda/s\rceil$ 随 $s$ 单调递减（`coarse_iterations_mono`），但总工作量 $s\cdot n_{\min}(s)\approx\Lambda$ 恒定。

**最优 stride 的判定**。既然精度与 stride 无关、总工作量也近似恒定，**最优 stride 不由精度决定，而由下游的时间分辨率需求与显存/带宽决定**：

1. **分辨率下界**：若需分辨时域尺度 $\tau$（以 base step 计），则 $s\le\tau$。
2. **慢特征保真上界**（连接 §3.11.1 谱夹层）：最慢特征 $\tilde a_{\max}=1-\gamma$ 在 stride $s$ 后的持久性为 $(1-\gamma)^s$；若要求它在读出头可见（$\ge$ 阈值 $\theta$），则 $(1-\gamma)^s\ge\theta\Rightarrow s\le\log\theta/\log(1-\gamma)$。
3. **综合最优**（取可行区间最大 stride，最小化迭代/显存）：

$$s^* \;=\; \min\Big(\tau,\ \big\lfloor \tfrac{\log\theta}{\log(1-\gamma)}\big\rfloor\Big)$$

**对本框架的借鉴意义**（承接 §3.11.1 谱时间分层）：

1. **精度与层级解耦**：不动点跨步不变 + 残余只依赖 $\Lambda$ ⟹ 「加层」不能提升抽象结论精度，层级设计只服务于**计算/带宽/分辨率**，不服务于精度。纠正「更深分层 = 更准世界模型」的误区。
2. **最优 stride 由谱夹层给出**：$s^*$ 的上界 $\log\theta/\log(1-\gamma)$ 直接来自谱夹层上界 $1-\gamma$（§3.7.3）——快/慢分离（§3.11.1）与最优 stride（本节）是**同一组谱参数的两种用法**：慢维度（$\tilde a\to1-\gamma$）可粗 stride，快维度（$\tilde a\to\delta$）必须细 stride。
3. **与 $\rho$ 压缩正交**：$\rho$ 压缩（§2.6）是 token 层「写后压」的**步内**冗余（单步内哪些 token 保留），stride 是**跨步**的时间分辨率（几步一读）——两者都是「慢/快」，但作用于不同粒度，不可混淆。
4. **GPU 特性**：`pow_mul_stride` + `coarse_iterations_mono` 说明在 $O(\log N)$ 关联扫描（§3.7.3）下，stride 只影响**物化状态数**（$N/s$）而非并行深度——最优 stride 本质是「在满足分辨率约束下最小化显存/带宽」，与 §3.10.3 串行深度 $O(\log N)$ 相互独立。
5. **落地方向（不在此落地）**：可把 $s^*=\min(\tau,\lfloor\log\theta/\log(1-\gamma)\rfloor)$ 写入 config 作为 `stride_schedule` 默认，由 $\gamma,\theta,\tau$ 三个超参决定，无需手工调 stride。

**未形式化的边界**：$\Lambda$ 与 $s^*$ 的对数/取整闭式依赖 $\log$ 的实数分析（未在此机械化）；Lean 已证的代数核心是 `pow_mul_stride`（残余只依赖总 horizon）、`residual_horizon_mono`（单调递减）、`coarse_iterations_mono`（stride 越大越快），闭式是它们的直接对数推论。

---

## 3.12 Z 空间的隐式时间轴（序列输入注入，非时间分层方案）

> **问题**：能否**不引入时间分层**（§3.11），直接在原方案的单一 $Z$ 中隐式承载视频时间轴，使（1）有时间的视频与（2）无时间的文字在 $Z$ 上对齐，（3）算子在隐式时间轴上运算？本节给出数学回答与机制。

#### 3.12.1 负结果：自治收缩只给「深度」，不给「时间」

原方案的核心算子是**自治仿射收缩** $Z^{(k+1)}=MZ^{(k)}+b$（§2.9/§3.7.3）。它的唯一不动点 $z^*=(I-M)^{-1}b$ 是**时间不变的**（§3.11.4 `fixedPoint_stride_invariant`），且轨道 $Z^{(k)}\to z^*$（`mrLoop_tendsTo_fixedPoint` / §3.7.1）。因此：**把视频「迭代到不动点」必然把时间序列坍缩成单个时间不变摘要**——自治收缩的 $Z$ 不保留时间。

§2.10 的 MR-Loop（输入注入 $h\mapsto Mh+Me$）修复了「输入坍缩」（`mrLoop_no_input_collapse`：不同输入 $e$ ⟹ 不同不动点），但它注入的是**固定**输入 $e$，收敛到的仍是一个**输入依赖但时间不变**的不动点 $\mathrm{fp}(e)$。所以 MR-Loop 给的是**深度**（单输入 $\to$ 摘要），不是**时间**（输入序列 $\to$ 状态序列）。

#### 3.12.2 机制：序列输入注入（把「固定输入」推广为「逐帧输入」）

要让单一 $Z$ 隐式承载时间，唯一缺失的机制是**把 MR-Loop 的固定输入 $e$ 换成逐步变化的输入 $x_t$**：

$$Z^{(t+1)} \;=\; M\,Z^{(t)} + x_t$$

其闭式（`iterateSeq_eq_closedForm`，已 Lean 机械化）为**升幂卷积**：

$$Z^{(t)} \;=\; M^t Z^{(0)} \;+\;\sum_{i<t} M^{\,t-1-i}\, x_i$$

**这是「隐式时间」的数学核心**：$Z^{(t)}$ 携带**完整输入历史** $(x_0,\dots,x_{t-1})$，时间轴被折叠进算子 $M$ 的幂与卷积，而非 $Z$ 的显式维度。故 $Z^{(t)}$ 与文字模态的 $Z$（单向量）**同维、可对齐**——这正是「隐式」的含义：时间不进维度，进递归。

**对齐**：文字（无时间）对齐到视频的**时间不变摘要** $z^*=(I-M)^{-1}\bar{x}$（$\bar{x}$ 为时间平均输入），即自治收缩的不动点；视频的**时间变分量** $\sum M^{t-1-i}(x_i-\bar{x})$ 承载运动/变化，不与文字对齐（文字无时间轴可表达之）。这与 §3.4 跨模态硬对齐一致：跨模态对齐发生在**时间不变内容**上，时间变内容只存在于有时轴的模态内部。

**算子运算**：算子 $M$ 沿 $t$ 逐帧作用（同一 $M$），即 §3.7.3 的关联扫描（PSCO）——时间轴 = 扫描下标，算子 = 扫描的幺半群元。

#### 3.12.3 机制性质（GPU 友善 + 反塌缩）

1. **GPU 友善**：序列注入是 §3.7.3 关联扫描的输入注入形式，$O(\log T)$ 并行深度、无新增 kernel；$x_t$ 的注入是逐元素加，`torch.compile` 自动融合。
2. **不引入量化/新维度**：$Z$ 仍单一、连续、不量化（§2.6），时间不进维度——与「有损只在 token 层」契约一致。
3. **反塌缩**：时间变分量 $\sum M^{t-1-i}x_i$ 使 $Z^{(t)}$ 对输入序列**可区分**（不同帧序列 ⟹ 不同状态轨迹），避免 MR-Loop 单输入时的「时间不变塌缩」；同时 $\|M\|<1$ 保证历史按 $K^{t-i}$ 衰减（`fast_decay`），旧帧影响力有限（因果、有限记忆）。

#### 3.12.4 对本框架的意义与边界

**意义**：原方案**无需时间分层即可隐式承载时间**——只需把 MR-Loop 的固定输入 $e$ 换成逐帧输入 $x_t$（一个薄改动）。这回答了「文字↔视频能否在单一 $Z$ 对齐」：**能**，对齐发生在时间不变摘要 $z^*$ 上。算子沿隐式时间轴的运算就是已有 PSCO 扫描。

**与 §3.11 时间分层的关系**：二者正交——§3.11 的跨步分层是**多时间尺度**（同一序列在不同 stride 下的摘要），本节是**单时间尺度的隐式时间承载**（把时间折进递归）。需要多尺度时叠加 §3.11，不需要时仅用本节机制。

**边界（诚实）**：$z^*=(I-M)^{-1}\bar{x}$ 的「时间平均 $\bar{x}$」是分析/设计量，未机械化；$x_t=E(o_t)$ 逐帧注入假设视频帧可独立编码（帧间依赖由 $M$ 沿时间传递），真实视觉编码器的帧间混合需实验验证（同 §2.7 边界）。闭式 `iterateSeq_eq_closedForm` 已对任意域 $K$、任意模 $V$、任意输入序列 $x:\mathbb{N}\to V$ 机械化。

---

## 3.13 融合方案：单 Z 骨架上的「慢特征保护 + 快特征自反馈」（§3.11 × §3.12 取舍）

> **问题**：§3.11（多时间尺度）提供谱定向「留慢丢快」，但有 `pow_mul_stride` 的平权权衡（不提升精度）+ 独立层参数陷阱；§3.12（单尺度隐式时间 + ρ压缩）简洁、跨模态对齐、GPU 友善，但记忆受收缩谱 $K^t$ 限制、无谱定向丢弃。能否**取二者之长**——在单 Z 隐式时间骨架上，用「慢特征保护」抬升长程记忆下界，用「快特征门控自反馈」修正短期误差且不扰动长期不变特征？本节给出数学形式，核心已在 Lean 机械化。

### 3.13.1 语义约定：慢 = 时间不变，快 = 时间相关

本框架把 §3.11 的「慢/快谱分离」按**语义**重标注（这是融合的关键取舍，不再是单纯的大/小特征值）：

- **慢特征**（时间无关 / 弱相关）：特征值 $a_s$ 接近 $1$，$M$ 作用近似保持 → 对应时间不变摘要 $z^*=(I-M)^{-1}\bar x$（§3.12.2 的文字对齐目标）。
- **快特征**（时间相关 / 短期状态）：特征值 $a_f$ 接近谱夹层下界 $\delta$，$M$ 作用快速衰减 → 对应瞬态运动 / 变化（§3.12.2 的时间变分量 $\sum M^{t-1-i}(x_i-\bar x)$）。

### 3.13.2 机制一：慢特征保护（谱正则，抬升长程记忆下界）

**数学形式**：把慢维特征值从 $a_s$ 抬升到 $a_s'\in[a_s,1)$，快维不变。即受保护转移 $\tilde M$ 的慢维特征值为 $a_s'$（仍 $<1$，保收缩）。这是**对角缩放**（保护算子 $P$ 在慢维 $\times(a_s'/a_s)\ge1$，快维 $\times1$），不引入新维度、不改算子结构。

- **`slow_retention_lifted`**：$0\le a_s\le a_s'\Rightarrow a_s^k\le a_s'^k$。逐 horizon $k$ 的留存下界由 $a_s^k$ **抬升**到 $a_s'^k$——这正是「长程记忆下界抬升」的逐点形式：慢特征不再被收缩以 $a_s^k\to0$ 湮灭，而是以更慢的 $a_s'^k$ 留存。
- **`protected_slow_still_contracting`**：$a_s'\in[0,1)$ 时 $a_s'^k<1$（$k>0$）——保护**不破坏**框架收缩契约（谱夹层 $\delta<\tilde a_i<1-\gamma$ 仍成立，§3.7.1 的 ρ<1 不动点收敛继续成立）。
- **`slow_fast_gap_widened`**：$a_s^k-a_f^k\le a_s'^k-a_f^k$——慢（保留）与快（衰减）的留存**差距被拉大**，分离被强化（§3.11 `fast_slow_ratio` 的「保护强化」版本，用差值避开除法）。

**取舍**：$a_s'\to1^-$ 换取更长留存，但减慢收敛（§3.11 `rate_mono_stride` 的对偶）。$a_s'$ 是「留存 vs 收敛」的旋钮——这是把 §3.11 的「分层」降级为「单旋钮谱正则」的关键：不引入独立层，只调一个超参。

### 3.13.3 机制二：快特征门控自反馈（修正短期、放过长期）

**数学形式**：自反馈修正 $F=C\circ\Pi_f$（`gatedFeedback`）——修正算子 $C$ 先经快投影 $\Pi_f$ **门控**，只作用于快（时间相关）子空间。

- **`feedback_spares_slow`**：$\forall z,\ \Pi_f(\Pi_s z)=0\Rightarrow F(\Pi_s z)=0$——自反馈对慢（时间无关）特征**零扰动**，长程不变内容被保护。
- **`feedback_correction_decays`**：快特征值 $a_f\in[0,1)$ 的修正 $a_f^k\to0$——自反馈的修正**随 horizon 衰减**，只修正近期（短期）误差。

**自反馈模式（§5）下的行为**：短期误差（快子空间）被 $F$ 及时修正；长期不变特征（慢子空间）不受 $F$ 扰动——即「自反馈修正短期、对长程无关特征影响变小（此处为 0）」。这满足用户对自反馈的期望：短期闭环校正有效，长期不变内容不被误改。

### 3.13.4 融合的正确性（对照框架契约）

1. **Z 连续、单空间**：保护与门控都是线性算子，不引入量化 / 新维度，时间仍折进递归（§3.12），有损只在 token 层 ρ压缩——符合 §2.6「有损只在 token 层」契约。
2. **收缩契约**：保护后仍 $\rho<1$（`protected_slow_still_contracting`），不动点收敛（§3.7.1）与 PSCO 关联扫描（§3.7.3）全部继承。
3. **反塌缩**：快维自反馈修正衰减但不恒零（`feedback_correction_decays`），慢维保护防时间不变摘要被收缩湮灭（`slow_retention_lifted`）——双向反塌缩。
4. **GPU 友善**：保护 = 逐元素对角缩放，门控 = 投影复合，均为逐元素 / 线性算子，可并入 PSCO 扫描的幺半群元（§3.7.3），无新增 kernel，`torch.compile` 自动融合。

### 3.13.5 边界（诚实）

- 慢/快谱分裂阈值 $\theta$、保护强度 $a_s'$ 是超参（分析 / 设计量，未机械化）；Lean 已证的是其**代数核心**：留存抬升（`slow_retention_lifted`）、差距拉大（`slow_fast_gap_widened`）、修正衰减（`feedback_correction_decays`）、慢子空间零扰动（`feedback_spares_slow`）。
- 投影 $\Pi_s/\Pi_f$ 的「正交分解」只用假设 $\forall z,\ \Pi_f(\Pi_s z)=0$ 表达（`feedback_spares_slow`），未要求具体正交基——保持抽象、与任意实现解耦。
- 融合是**单 Z 骨架上的谱正则 + 门控**，**不是**引入显式多层（避开 §3.11 独立层参数陷阱），也**不是**丢弃快特征（快维仍存在并承载短期状态，只由自反馈修正、由 ρ压缩在 token 层有损）。

---

## 3.14 ZHierarchy：全 Z 逐维软门控（机制二的推广）

> **问题**：§3.13 的机制二用**二值**投影 $\Pi_f$ 门控（快维权重 1、慢维权重 0）。能否推广到**所有 Z 维度**——每一维独立拥有自己的门控权重 $p_i\in[0,1]$（软门控），即「全 Z 逐维分层」？其优缺点、GPU 友善性、以及与现行架构其它数学原理是否有冲突？本节给出数学形式，命名 **ZHierarchy**，并已 Lean 机械化（含谱夹层相容性检查，§3.14.6）。

### 3.14.1 数学形式：对角门控 diag(p)

把 §3.13 的二值门控 $F=C\circ\Pi_f$ 推广为**对角门控** $G=\operatorname{diag}(p)$：

$$(G z)_i \;=\; p_i\, z_i,\qquad p_i\in[0,1]$$

反馈算子 $F=C\circ G$。$p_i=0$ 表示维度 $i$ 被完全保护（慢特征，反馈零扰动）；$p_i=1$ 表示全修正（快特征）；$0<p_i<1$ 连续调节。二值门控是 $p_i\in\{0,1\}$ 的特例。

这同时把 §3.11 的「离散多时间尺度」与 §3.13 的「慢/快二分」统一为**逐维连续谱**：每个维度 $i$ 是独立的一层，拥有自己的 $($特征值 $a_i$, 门控权重 $p_i)$。

### 3.14.2 GPU 友善（已 Lean 机械化）

1. **逐点独立（embarrassingly parallel）**：`diagGate_pointwise` 证第 $i$ 维输出只依赖第 $i$ 维输入——各维度缩放无跨维数据依赖、无归约，可完全并行，逐元素 kernel 被 `torch.compile` 融合。
2. **门控可合并**：`diagGate_comp` 证 $\operatorname{diag}(p)\circ\operatorname{diag}(q)=\operatorname{diag}(p\cdot q)$——多个门控合并为单次逐元素缩放，并入 §3.7.3 PSCO 关联扫描的幺半群元时不破坏结合律（`comp_assoc`）、不新增扫描趟次，零额外开销。
3. **仍收缩**：`soft_gate_preserves_contraction` 证 $0\le a<1$、$0\le p\le1$ ⟹ $pa\in[0,1)$——全 Z 分层后仍 $\rho<1$，不动点收敛与 PSCO 扫描全部继承。

### 3.14.3 逐维性质（已 Lean 机械化）

- **逐维放过（推广 `feedback_spares_slow`）**：`diagGate_zero_dim` 证 $p_i=0\Rightarrow(Gz)_i=0$——慢维被完全保护。
- **逐维单调（反馈影响连续调节）**：`diagGate_mono` 证 $p_i\le q_i$、$0\le z_i$ ⟹ $(G_p z)_i\le(G_q z)_i$——门控权重越大反馈影响越大，$p_i$ 是「放过 $\leftrightarrow$ 全修正」的连续旋钮。

### 3.14.4 优缺点

**优点**：

1. **消除硬阈值**：§3.13 的慢/快二分依赖阈值 $\theta$，逐维软门控消除这一任意性，保留谱夹层全部信息。
2. **连续可微、可学习**：$p_i\in[0,1]$ 连续，可端到端学习 / 正则（二值 $\Pi_f$ 不可微）。
3. **全谱控制**：每个维度独立 $($留存 $a_i$, 反馈 $p_i)$，统一 §3.11 与 §3.13。
4. **GPU 友善**：见 §3.14.2。

**缺点**：

1. **$n$ 个超参**（每维 $p_i$），需学习门控或正则，优化 / 过拟合负担高于二值的单阈值 $\theta$。
2. **留存与反馈是两个旋钮**：软门控只控「反馈强度」，不直接抬升「留存」（留存由 $a_i$ 决定，需配 §3.13 机制一 `slow_retention_lifted`）。
3. **慢维反馈泄漏风险**：$p_i$ 未正确正则时，慢维 $p_i\not\to0$，长期不变特征会被反馈扰动——二值 $\Pi_f$ 的硬零更稳健。
4. **对角线假设**：$G=\operatorname{diag}(p)$ 要求 $M$ 可对角化（谱域理想化）；真实 $M$ 非对角时需先谱分解或子空间投影，增加实现复杂度。

### 3.14.5 边界（诚实）

- Lean 已证的是对角门控的**代数与有序核心**（逐点独立、可合并、逐维放过 / 单调、保收缩）；$p_i$ 的取值、阈值、学习方式为超参（未机械化）。
- $G=\operatorname{diag}(p)$ 在函数空间 $\iota\to K$ 上定义（逐维天然并行）；对抽象 $V$ 需先谱分解（§3.13.5 已注）。

### 3.14.6 与现行架构其它数学原理的冲突检查（已 Lean 机械化）

ZHierarchy 与 math.md 现行各契约逐条核对：

**无冲突**：

1. **$Z$ 连续、单一、不量化**（§2.6）：$G=\operatorname{diag}(p)$ 是线性缩放，连续、不量化——不违反「有损只在 token 层」契约。
2. **收缩 $\rho<1$**（§3.7.1）：`soft_gate_preserves_contraction` 证 $pa\in[0,1)$——全 Z 分层仍收缩，不动点收敛（Banach）继承。
3. **仿射幺半群结合律 / PSCO 并行扫描**（§3.7.3）：`diagGate_comp` 证门控可合并为单次逐元素缩放，`comp_assoc` 保持——扫描正确性不受影响。
4. **反塌缩**（§2.9/§2.10/§3.10）：快维 $p_i=1$ 保留反馈修正、慢维 $p_i=0$ 由 $M$ 驱动留存——门控只控制反馈、不删减 $M$ 的作用，故不塌缩任何维度的状态演化。
5. **隐式时间**（§3.12）：$p_i$ 时间无关（静态），$G$ 不干扰升幂卷积闭式 $M^t Z^{(0)}+\sum_{i<t}M^{t-1-i}x_i$。

**一个必须满足的条件（冲突点，已证充要条件）**：

- **谱夹层下界 $\delta<\tilde a_i$**（§3.7.3）：`gate_preserves_lower_bound` 证 $0<\delta\le a$ 时 $\delta<pa\iff\delta/a<p$。即**若把门控作用到转移 $M$ 上**（把特征值 $a$ 缩为 $pa$），要保持下界必须 $p>\delta/a$（因 $a\ge\delta$ 故 $\delta/a\le1$，是真实约束）。**ZHierarchy 的设计是门控只作用于反馈**（不缩放 $M$ 的特征值），故下界不受触碰；上界 `gate_preserves_upper_bound` 证 $0\le a<u$、$p\le1$ ⟹ $pa<u$（$u=1-\gamma$），始终成立。

**一个范围假设（非定理，诚实标注）**：

- **对角线假设**：$G=\operatorname{diag}(p)$ 的「逐维」只在 $M$ 可对角化时良定义。核心框架的 PSCO（§3.7.3）对抽象 $V$ 成立（不要求对角化）；ZHierarchy 引入「可对角化」这一更强假设，是唯一超出核心框架的前提，需在实现时先谱分解或子空间投影（§3.13.5 已注）。

---

## 4. 阶段三 · 解码：表征空间 → 多模态输出

> **【实现状态】** §4.1 的 Flow Matching（线性插值 OT 路径、速度场回归、时间嵌入、Euler 采样、软投影 retraction §4.1.7、条件流 §4.1.8）与 §4.2 的解码已在当前实现落地并接入默认训练/采样（`cfg.flow_enabled` / `flow_cond_enabled` / `psi_enabled` 默认 `True`，见 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) `forward_losses` 与 [inference/run.py](file:///home/simon-ma/JEPA/inference/run.py) `euler_sample + make_projection`）。**尚未落地的是 §3.9 式 29 的「生成 = D(Ψ(Z))」路径**：算子 Ψ 的不动点 $z^*=\Psi(Z)$ 目前只在潜空间损失（readout/var/stop/consist，§3.7）中使用，从不送入解码器——解码器只接收原始编码 $Z$（重建）与跨模态 $Z$（硬对齐），不接收 $\Psi(Z)$（生成）。故「算子 → 不动点 → 解码」的生成链路尚未接通；解码/生成路径的选型（自回归、MaskGIT、离散 diffusion 或 Flow）对「如何把 $z^*$ 渲染回 token」仍开放。

### 4.1 Flow Matching 潜空间生成

JEPA 阶段结束后冻结 $E_\theta$，潜空间被固定，Flow 只需拟合这个静止的数据分布。

#### 4.1.1 概率路径与线性（OT）插值

设 $p_0 = \mathcal{N}(0, I)$ 为噪声分布，$p_1$ 为目标编码器诱导的数据分布 $z_1 = E_\xi(x) \sim p_1$。Flow Matching 在两者之间构造一条概率路径 $p_t$（$t\in[0,1]$）。本模型采用**线性插值路径**：

$$z_t = (1-t)\,z_0 + t\,z_1, \qquad z_0 \sim p_0,\; z_1 \sim p_1,\; t \sim \mathcal{U}(0,1).$$

这是高斯与数据分布之间的直线插值路径，采用**独立耦合**（$z_0,z_1$ 各自独立采样，即 Rectified Flow 的默认设定），也是 Conditional Flow Matching 中最简单且稳定的选择。注意它**不是** OT 耦合——OT 是 reflow 迭代的极限（§4.1.5）。

#### 4.1.2 速度场与回归目标

对路径关于 $t$ 求导，得到该路径对应的**真实速度场**：

$$u_t(z_t) = \frac{d z_t}{d t} = z_1 - z_0.$$

源码中即 `velocity_target = target - noise`。模型学习一个条件速度场 $v_\psi(z_t, t, c)$ 去回归它，损失为均方误差：

$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, z_0, z_1}\left[\left\| v_\psi(z_t, t, c) - (z_1 - z_0) \right\|^2\right].$$

条件 $c = E_\theta(x_c)$ 通过 `FlowBlock` 中的交叉注意力注入，使生成的潜变量与可见前缀一致。

#### 4.1.3 时间嵌入

连续时间 $t\in[0,1]$ 经正弦嵌入（标准 transformer 频率约定）后由 MLP 非线性化：

$$\mathrm{emb}(t) = \big[\sin(t\,f_j),\ \cos(t\,f_j)\big]_{j=0}^{d/2-1}, \qquad f_j = 10000^{-2j/d} = \exp\!\left(-\frac{2\ln 10000}{d}\,j\right),$$

频率按对数几何间隔从 $f_0=1$（最高频）降到 $f_{d/2-1}\approx 10000^{-1}$（最低频），覆盖多个数量级、同时捕捉时间的高频变化与低频趋势；再与 $z_t$ 相加后送入主干。

#### 4.1.4 Euler 采样：从噪声积分到潜变量

推理时求解由学习速度场定义的 ODE：

$$\frac{d z}{d t} = v_\psi(z, t, c), \qquad z(0) \sim \mathcal{N}(0, I),$$

用 Euler 法离散，步长 $\Delta t = 1/N$（$N$ = `inference_steps`）：

$$z_{k+1} = z_k + \Delta t \cdot v_\psi(z_k,\; k\Delta t,\; c), \qquad k = 0, 1, \dots, N-1.$$

终点 $z(1)$ 即为生成的连续潜变量 $\hat z$。随后经解码器量化为文本（见 §4.2）。

#### 4.1.5 曲率与直线性：CoT 潜流的最短路径

把 CoT 的潜态轨迹 $Z^{(0)}\to\cdots\to Z^{(N)}$ 看成一条潜流。**「曲线救国」说的是潜空间轨迹的弯曲，不是 token 数**：中间 $Z^{(i)}$ 全落在 $Z^{(0)}$ 到 $Z^{(N)}$ 的直线段上，token 再多潜流也是直的；绕出去再回来，token 再少也有曲率。

**数学刻画。** 轨迹 $z(t)$ 是测地线当且仅当零加速度：$\ddot z_t=0\iff z_t=(1-t)z_0+t z_1\iff v=\text{const}$。曲率 $\kappa$ 度量速度**方向**的改变；「曲线救国」即 $\ddot z_t\ne 0$、方向不断漂移。

**训练阶段：把速度场矫正成直线。** 三个手段：

1. **直线插值 + 独立耦合（§4.1.1 默认，Rectified Flow）**：训练目标本就取直线插值、回归恒定速度 $z_1-z_0$，从定义上把「直线」设为靶子；
2. **矫正流（Rectified Flow / ReFlow）**：迭代「拟合→沿拟合流重采样→再拟合」，每轮把路径拉直一点、运输成本单调降到 OT 下限——这是「把曲线救国的 CoT 逐轮拉直」的算法原型；
3. **直线性正则**：显式惩罚曲率 $\mathcal L_{\text{straight}}=\mathbb E_t\|\ddot z_t\|^2$，等价于惩罚偏离直线 $z_t-((1-t)z_0+t z_1)$。

绕路的中间步骤是「有曲率的训练信号」，训练目标是让 $\Psi_h$ 学会忽略绕路、直取答案；曲率是「冗余推理」的可微度量。

**推理阶段：少步/单步到位。** 直线流速度恒定 → 1 步 Euler 就落到答案（§4.1.4 的 $N$ 可蒸馏到 1–4 步）；弯曲流必须多步积分、每步沿弯走还可能离题。对 CoT 的推论：直线潜流意味着推理时可**不走完中间 token，直接 $\Psi^{(N)}$（或 $v_\psi$ 一步）跳到 $Z^{(N)}$**，中间 $g(Z^{(i)})$ 从「必要步骤」降级为「可选出口」（可解释性税）——与 §3.7 的「token 化是可选出口」一致。

**与不动点、$h$ 的统一。** 改分量方向沿轨迹恒定 = 测地线（最短流）；方向漂移 = 曲率（曲线救国）；改分量衰减到零 = 到达不动点（答案）。于是一句话：**训练 = 把 $h_{\text{change}}$ 的方向矫正到恒定（零曲率）；推理 = 沿这条恒向测地线一步走到 $h_{\text{change}}=0$ 的不动点。**

**把 Loop Transformer 的算子路径拉直：Rectified Flow 的 reflow。** §3.7.1 里 Loop Transformer 的循环 $Z^{(k+1)}=F_\theta(Z^{(k)};x)$ 定义了一条离散弯曲轨迹 $Z^{(0)}\to\cdots\to Z^{(L)}=Z^*$。把这条轨迹拉直正是 **Rectified Flow**（Liu et al., ICLR 2023）的 reflow 操作——取 (起点, 终点) 配对 $(Z^{(0)},Z^{(L)})$，训练速度场匹配 OT 直线的恒定速度：

$$\mathcal L_{\text{reflow}} = \mathbb E\left[\int_0^1\Big\|v_\theta(z_t,t) - \big(Z^{(L)} - Z^{(0)}\big)\Big\|^2\,dt\right],\qquad z_t = (1-t)Z^{(0)} + t\,Z^{(L)}. \tag{32}$$

沿 $v_\theta$ 积分得到的新轨迹 $z^{(1)}_t$ 比原循环轨迹更直；**reflow 可迭代**——用新轨迹端点再配对、再拟合，第 $k$ 轮的运输代价单调不增：

$$\mathbb E\int_0^1\big\|\dot z_t^{(k+1)}\big\|^2\,dt \;\le\; \mathbb E\int_0^1\big\|\dot z_t^{(k)}\big\|^2\,dt, \tag{33}$$

极限是 OT 直线（$\ddot z_t=0$、速度恒定），步数从 $L$ 折叠到 $O(1)$。这正是把 §3.7.1 的"循环 $L$ 步到不动点"折叠成"单步流到 $Z^*$"。但**代价**：直线把中间态推出文本纤维 $\mathcal{Z}_{\text{text}}$，$g(z_t)$ 是 argmax 硬凑出的无意义 token；曲线（CoT）贴着纤维走、每个 $g(Z^{(i)})$ 可表达。于是"拉直"是**曲率预算**旋钮——改分量主导 → 多轮 reflow → 零曲率、少步、不可解释（折叠）；读分量主导 → 保留曲率 + 纤维锚定正则 $\mathcal L_{\text{audit}}=\|Z^{(i)}-E(g(Z^{(i)}))\|^2$ 把中间态拉回纤维 → 多步、可解释（CoT）。reflow 次数是连续刻度，与本节的 $\mathcal L_{\text{straight}}$ 互补。

**Loop Transformer 与 DEQ 在隐式微分上的具体差异。** 二者都是"共享权重 / 不动点"架构，但反传不同，差异可精确写成一个 Neumann 级数。设 $J=\frac{\partial F_\theta}{\partial Z}\big|_{Z^*}$：

| 维度 | Loop Transformer（BPTT） | DEQ（隐式微分） |
|---|---|---|
| 前向 | 显式展开固定 $L$ 步（未必收敛） | 迭代到不动点 $Z^*$（收敛） |
| 反传 | BPTT 展开 $L$ 步，梯度沿共享权重求和 | 隐式函数定理，解伴随线性不动点 |
| 中间态 | 存 $O(L)$ 个激活 | 只存 $Z^*$，$O(1)$ 内存 |
| 关键量 | $\sum_{k=0}^{L-1}J^k$（截断） | $(I-J)^{-1}$（完整逆） |
| 收敛条件 | 防梯度消失 / 爆炸（深度 $L$） | $\rho(J)<1$（收缩） |

BPTT 走 $L$ 步的梯度是 Neumann 级数 $I+J+J^2+\cdots$ 的**截断**，DEQ 的 $(I-J)^{-1}$ 是它的**收敛极限**（忽略转置细节，转置形式见式 25）：

$$\nabla_\theta\mathcal L\Big|_{\text{BPTT},\,L} = \sum_{k=0}^{L-1} \frac{\partial\mathcal L}{\partial Z}\,J^k\,\frac{\partial F}{\partial\theta},\qquad
\nabla_\theta\mathcal L\Big|_{\text{DEQ}} = \frac{\partial\mathcal L}{\partial Z}\,(I-J)^{-1}\,\frac{\partial F}{\partial\theta}. \tag{34}$$

当 $\rho(J)<1$ 时后者是前者的 $L\to\infty$ 极限。对应到本框架：**有限步 CoT（$N$ 步）= Loop Transformer = 截断 Neumann 级数（有偏但可解释）；"结论 = 不动点" + DEQ = 完整逆（无偏但折叠）**。二者在 §3.7.1 的式 25 统一——式 25 的 $(I-\partial\Psi_h/\partial Z)^{-1}$ 正是把 §3.7 的 $N$ 步 rollout 求和推广到收敛极限。注意这层类比只在 $h$ 固定（$h_{\text{change}}\to0$）时严格成立，跨步演化的 $\Psi_{h^{(i)}}$ 不满足单算子 Neumann 级数。

**直线 → 测地线：非欧修正。** 本节"直线 / 零曲率"默认了 $\mathcal M$ 平坦且测地凸。按 §2.1.1，$\mathcal M$ 泛型弯曲，故"最短流"应重述为"$\mathcal M$ 上的测地流"，"零曲率"应指**零内在曲率**（沿切丛 $T\mathcal M$ 平行移动的方向恒定），而非零外在加速度 $\ddot z=0$；reflow 应重述为**测地整流**（geodesic rectification），极限是 OT 测地线而非欧氏直线。欧氏 reflow 只在局部（小测地球 / 低曲率）是一阶近似，全局会离开 $\mathcal M$。这条修正同时解释了 §3.7 的"曲线救国"：CoT 的中间步骤之所以"绕"，正是因为它贴着弯曲的 $\mathcal M$（可解码 / 可解释的流形）走，测地线在环境空间里看就是弯的。

#### 4.1.6 黎曼流匹配（Riemannian Flow Matching）备选：数学意义与 GPU 友好性

§4.1.2 的速度场回归目标 $z_1-z_0$ 是**欧氏直线速度**。但 §2.1.1 已证明 $\mathcal M$ 泛型弯曲，欧氏直线段 $[z_0,z_1]$ 会离开 $\mathcal M$（off-manifold，中途 $c(t)=\max_v p_\theta(v\mid z_t)$ 塌陷）。欧氏 FM 因此回归了一条**在可解码流形上不可行**的路径。

**数学意义。** 严谨推广是把 OT 直线换成流形测地线（§2.1.1 推论）：

$$z_t = \exp_{z_0}\!\big(t\cdot\log_{z_0}(z_1)\big),\qquad v_\psi \in T\mathcal M,$$

速度场活在切丛 $T\mathcal M$ 上，回归目标是测地速度（Chen & Lipman, ICLR 2024）。意义有三：

1. **生成路径始终在 $\mathcal M$ 上**：中间态 $z_t$ 全程可解码、$c(t)$ 平稳，生成质量高——欧氏直线则中途穿过 off-manifold 区；
2. **与 §4.1.5 自洽**：「最短流」应重述为测地流、「零曲率」指零内在曲率（切丛平行移动方向恒定），reflow 极限是 OT 测地线而非欧氏直线（§4.1.5 末段已点出此修正）；
3. **度量统一**：若 §2.2.6 的潜预测损失升级为拉回度量 $\mathbf G$，Flow 速度场回归也应同步用 $\mathbf G$——两处度量统一，避免「预测用黎曼、生成用欧氏」的新矛盾。

**GPU 友好性。** RFM 的代价集中在 $\exp_z$ / $\log_z$（指数/对数映射）的计算：

| 项 | 欧氏 FM（当前） | Riemannian FM |
|---|---|---|
| 回归目标 | $z_1-z_0$，$O(1)$ 逐元素 | 需解测地线 ODE $\ddot\gamma+\Gamma(\dot\gamma,\dot\gamma)=0$ |
| 采样 $z_t$ | 线性插值，$O(1)$ | $\exp_{z_0}(t\log_{z_0}z_1)$，数值积分 |
| 训练并行 | batch 全并行、零串行 | 每对 $(z_0,z_1)$ 独立解测地线，batch 内可并行向量化，但每条测地线是 $K$ 步串行积分 |
| 推理串行深度 | $O(N)$（Euler 步，与 $L$ 解耦） | $O(N\cdot K)$（每步 Euler 外再套测地积分） |

**关键瓶颈**：我们的 $\mathcal M$ 是**隐式定义**的（$\mathbf G=J_D^\top J_D$，无闭式 $\exp/\log$），每一步都要数值求解测地线方程，串行深度从 $O(N)$ 恶化到 $O(NK)$，且 $\mathbf G$ 随解码器漂移使测地线不稳定。因此 RFM **数学上更正确、GPU 上明显更贵**，仅在 $d$ 很小（如冒烟 $d=64$）或流形有闭式 $\exp/\log$（球面/双曲等）时才划算。当前建议：保持欧氏 FM，把 RFM 列为「当 off-manifold 生成退化（中途 $c(t)$ 塌陷）被实测观察到时」的备选，不强制切换。

#### 4.1.7 黎曼流匹配的 GPU 友好近似（retraction-based）

§4.1.6 指出 RFM 的瓶颈是 $\exp_z/\log_z$ 要数值解测地线 ODE，串行深度 $O(NK)$。但存在把「测地线」降级为「一阶 retraction」的近似族，代价从 $O(NK)$ 回到 $O(N)$、每步 $O(1)$，同时保留「路径始终在 $\mathcal M$ 上」的核心收益：

**retraction 定义。** $R_z:T_z\mathcal M\to\mathcal M$ 满足 $R_z(0)=z$ 且 $\frac{d}{dt}R_z(tv)\big|_{t=0}=v$，即与 $\exp_z$ 一阶一致。两条公理（仅 Fréchet 可微）只给出一阶 $R_z(v)=z+v+o(\|v\|)$；要升到二阶 $O(\|v\|^2)$ 需 $R_z$ 为 $C^2$（Taylor 余项被二次项控制）：
$$\exp_z(v) = R_z(v) + O(\|v\|^2)\quad(\text{当 }R_z\text{ 为 }C^2).$$

**GPU 友好的 retraction（软投影）定义。** $\mathcal M$ 隐式由量化器 $g$（argmax）定义。连续流每步用**软投影** retraction，把环境空间步进投影到 token 区分方向——即拉回度量 $\mathbf G=J_D^\top J_D$ 的像空间（$T_z\mathcal M=\operatorname{im}J_D^\top=\operatorname{im}\mathbf G$，§4.2(a)）：

$$R_z(v) = z + P_z v,\qquad P_z = J_D^+ J_D \in \mathbb R^{d\times d},$$

其中 $J_D\in\mathbb R^{|\mathcal V|\times d}$ 是读出层雅可比（第 $v$ 行为 $w_v^\top$，§4.2(a) 式 35），$J_D^+$ 为 Moore–Penrose 伪逆。$P_z$ 是 $\operatorname{im}J_D^\top$（token 区分方向）上的**正交投影**，四条性质：

1. $P_z^2=P_z$（幂等）、$P_z^\top=P_z$（对称）、$\operatorname{rank}P_z=\operatorname{rank}J_D=r\le\min(|\mathcal V|,d)$（§4.2(a)）；
2. 线性 $\Rightarrow C^\infty$；$R_z(0)=z$；$dR_z(0)=P_z$，且 $P_z|_{\operatorname{im}J_D^\top}=\mathrm{id}$（在切空间上恒等）；
3. 故对 $v\in T_z\mathcal M$ 有 $R_z(v)=z+v$——切空间上一阶**精确**（无 $o$ 项），是真正的 retraction；核方向（token 内自由度）被投影掉，不引入伪流；
4. GPU 成本：$P_z$ 为秩 $r$ 矩阵，每步一次 $O(rd)\le O(d\min(|\mathcal V|,d))$ 矩阵-向量乘，全 batch 并行、无 ODE 求解、无串行深度；$P_z$ 随 $z$ 慢变可摊销（缓存/低秩更新，见梯度 3）。

**硬 decode–reencode** $E(g(z+v))$ 是**离散读出/锚定**（snap-to-codebook，§4.2(f) 的 argmax/FSQ），保证最终落在可解码点上；但它是分段常值（Voronoi 胞内 $dR=0$、胞界跳变），**不是** retraction，只用于最终读出一步。

**误差分析（两类正交误差）。** 设每步速度已投影到切空间（$v_\psi(z_t,t)\in T_{z_t}\mathcal M$），则 $P_{z_t}v=v$、$R_{z_t}(v\Delta t)=z_t+v\Delta t$。精确流的一步为 $\exp_{z_t}(v\Delta t)=z_t+v\Delta t+O(\Delta t^2)$（$\exp$ 为 $C^\infty$、$d\exp_{z_t}(0)=\mathrm{id}$），故**局部截断误差**

$$\big\|R_{z_t}(v\Delta t)-\exp_{z_t}(v\Delta t)\big\|=O(\Delta t^2),$$

即软投影是一阶方法（与欧氏 Euler 同阶）；累计 $N=O(\Delta t^{-1})$ 步后**全局误差** $O(\Delta t)$，可经 reflow（§4.1.5）迭代拉直消除。与之**正交**的是最终硬读出的**量化误差**

$$\big\|E(g(z))-z\big\|\le\varepsilon_{\mathrm{quant}},\qquad \varepsilon_{\mathrm{quant}}\sim\text{Voronoi 胞半径}\sim f(|\mathcal V|,d),$$

它不随 $\Delta t\to0$ 消失，只能靠增大 $|\mathcal V|$ 或软量化（FSQ/温度退火，§4.2(f)）压小。两类误差来源独立、可分别控制：

| 误差 | 量级 | 来源 | 消除手段 |
|---|---|---|---|
| 截断（离散化） | 局部 $O(\Delta t^2)$ / 全局 $O(\Delta t)$ | 用 $R_z$ 替代 $\exp$ 的一阶积分器误差 | 减小 $\Delta t$、reflow、高阶积分器（截断测地线，梯度 2） |
| 量化（读出） | $O(1)$（码本分辨率地板） | 硬 argmax snap-to-codebook | 增大 $|\mathcal V|$、FSQ/温度退火软量化 |

GPU 视角：截断误差由 $\Delta t$ 与积分器阶数决定（**算力**换精度），量化误差由码本与软量化决定（**记忆/容量**换精度），二者在 $O(N)$ 串行深度下独立调优。

**retraction-based FM。** 把速度场回归与采样里的 $\exp$ 换成 $R$：
$$\mathcal L=\mathbb E\Big[\big\|v_\psi(z_t,t)-\dot R\big\|^2\Big],\qquad z_{t+\Delta t}=R_{z_t}\!\big(v_\psi(z_t,t)\,\Delta t\big).$$
（截断/量化两类误差的量级与消除手段见上面误差分析；式 33 运输代价单调不增在软投影下仍成立。）三条 GPU 友好的梯度：

1. **软投影 retraction**（上）：$R_z(v)=z+P_z v$，$O(1)$ 一步低秩投影（$r\le\min(|\mathcal V|,d)$），全程在 token 区分方向内；硬读出 $E(g(z))$ 只在最后一步做；
2. **截断测地线（少步 RK）**：$K=1\text{–}2$ 步廉价积分器替代 $K=20\text{–}50$，降常数、不降阶；
3. **低秩度量利用**：$\mathbf G=J_D^\top J_D$ 的秩 $\le\min(|\mathcal V|,d)$（§4.2(a)），只在 $\mathbf G$ 的像空间（token 区分方向）上做流、核空间走欧氏，配合投影 retraction。

**结论。** 欧氏 FM（§4.1.1–4.1.4）是 retraction 的 $R_z(v)=z+v$（$P_z=\mathrm{id}$）退化特例；「欧氏 FM + 软投影 retraction」是最小 GPU 友好的 RFM 近似——把 off-manifold 偏差从「全局漂出 $\mathcal M$」压到「每步 $O(\Delta t^2)$、可被 reflow 吸收」，同时保持 $O(N)$ 串行深度。升级路径：先欧氏 FM（已实现，§4.1.2）→ 软投影 retraction（每步加 $z\leftarrow z+P_z v$，$P_z=J_D^+J_D$，秩 $r\le\min(|\mathcal V|,d)$）→ 需要时再上真 RFM。硬 argmax 读出 $E(g(z))$ 是最终离散化（§4.2(f)），不是连续流的 retraction。

#### 4.1.8 反向合成（自举）：用模型自身轨迹 + 硬反馈过滤生成训练数据

§4.1.5 的 reflow 需要「弯曲轨迹的 (起点, 终点) 配对」，而真实中间态（CoT）数据稀缺（见 §5 硬/软反馈：自然语言 CoT 是软反馈，无客观判据）。模型成型后可以**自己合成**这些轨迹，再经硬反馈过滤，构成自举回路：

**verify 接口（数学定义）。** 设输出空间 $\Omega$（解码产物 $T=g(Z)$ 的取值域）。硬反馈的客观判定器是一个与模型 $\theta$ 无关的**全函数决策过程**
$$\mathrm{verify}:\Omega\to\{0,1\},\qquad \mathrm{verify}(o)=1\iff o\in C,$$
其中 $C\subseteq\Omega$ 是客观定义的「正确集」：

- 形式语言：$\mathrm{verify}(\text{proof})=1$ 当且仅当证明项通过内核类型检查（Lean/mathlib）；
- 程序：$\mathrm{verify}(\text{prog})=1$ 当且仅当执行满足规格（全部测试通过）；
- 物理观测：$\mathrm{verify}(\text{state})=1$ 当且仅当状态与测量一致。

它区别于 §5 的软反馈 $e_{\text{soft}}=\hat Z-E(g(\hat Z))$：软反馈是「自洽性」（$E\circ g\approx\mathrm{id}$），verify 是「正确性」（$o\in C$），二者正交（§5 命题 10）。分级形式 $\mathrm{verify}:\Omega\to[0,1]$（如通过测试比例）作偏分奖励，二元是极限。

**四步自举回路。**

1. **轨迹自举**：用 $\Psi_h$ 迭代（或 §3.7.2 的 ToT/MCTS 搜索）从起点滚出中间态 $Z^{(0)}\to Z^{(1)}\to\cdots\to Z^{(N)}$，$Z^{(i+1)}=\Psi_{h^{(i)}}(Z^{(i)})$。
2. **硬反馈过滤（拒绝采样）**：只保留终点可验证正确的轨迹
   $$q\big(Z^{(1:N)}\mid Z^{(0)}\big) = \frac{p_\theta\big(Z^{(1:N)}\mid Z^{(0)}\big)\,\mathrm{verify}\!\big(g(Z^{(N)})\big)}{\mathcal Z(Z^{(0)})},\qquad \mathcal Z(Z^{(0)})=\mathbb E_{p_\theta}\big[\mathrm{verify}(g(Z^{(N)}))\big],$$
   其中 $\mathcal Z(Z^{(0)})$ 是「该起点可解率」（成功概率）。
3. **reflow 蒸馏**（§4.1.5 式 32）：用自生成配对 $(Z^{(0)},Z^{(N)})$ 拉直轨迹，$\mathcal L_{\text{reflow}}=\mathbb E\int_0^1\|v_\psi(z_t,t)-(Z^{(N)}-Z^{(0)})\|^2 dt$。
4. **迭代不动点**（式 33）：运输代价单调不增，极限为 OT 测地线。

**正确性 / 经济性二分量。** verify 是「正确性」分量的硬锚；reflow 的「最短流 / 码率」是「经济性」分量的锚。二者正交（§5 末的二维 reward）：仅正确性会导致模型学「绕远但正确」的路径，仅经济性会导致「直但错」——自举必须同时保留两个过滤条件。

**冷启动节奏。** 成型前无法自举：先真实自监督语料（§2.2）练出初版 → 少量硬反馈数据（LeanDojo 证明态、程序执行态等）冷启动推理轨迹 → 进入上述自举放大。这解决了「中间态数据稀缺」：模型自己造，verify 只当裁判。

### 4.2 与本模型实现的对齐

**(a) 解码器带位置混合，读出头仍是线性打分，不是欧氏最近邻。**
解码器是「位置混合 Transformer + 线性读出头」（`Decoder`，`decoder_layers` 层 self-attention）：先对量化潜变量做位置间交互，再逐位置线性输出

$$h = \mathrm{Norm}\bigl(\mathrm{Attn}(Q(Z) + \mathrm{pos})\bigr),\qquad
T_j = \arg\max_{v \in \mathcal{V}} \bigl( w_v^{\top} h_j + b_v \bigr), \tag{35}$$

其中 $w_v$ 是读出头的权重第 $v$ 行，$h_j$ 是位置混合后的特征（依赖全部位置的潜变量）。读出头仍是**由超平面划分的 Voronoi 型量化**，一般**不等于**最近邻量化 $\arg\min_v \|h_j - e(v)\|^2$；只有当 $w_v = 2e(v)$、$b_v = -\|e(v)\|^2 + \text{const}$ 时才退化为此形式，本模型没有绑定。因此：

- 不能把 $w_v$ 称为"token embedding"；
- 不能用 $\|Z - e(T)\|^2$ 作为式 (10) 中 $D_T$ 的估计。正确做法是用重编码失真 $\mathbb{E}\|Z - E_\theta(T)\|^2$（Flow 阶段已冻结），或直接用 (7) 的条件方差。

**超平面划分给出的「离散 token ↔ 连续概念区间」结构。** 式 (35) 的 argmax 把读出特征 $h$ 切成 $|\mathcal V|$ 个凸多面体区域

$$\mathcal R_v=\big\{h\ :\ w_v^\top h+b_v\ \ge\ w_u^\top h+b_u,\ \forall u\ne v\big\}.$$

这就是「离散 token ↔ 连续 $Z$ 区域」映射的精确几何形式：token $v$（如「红」）不是 $Z$ 空间里的一个点，而是一片**连续凸区域 $\mathcal R_v$**；区域内部的连续变异（深浅、明暗）被读出层判为同一 token，即「离散符号模糊对应连续概念」。区分度由 logits 差距度量

$$s_v(h)-s_u(h)=(w_v-w_u)^\top h+(b_v-b_u),$$

在 $\mathcal R_v$ 深处为正且大（强判 $v$），在 $\mathcal R_v$ 与 $\mathcal R_u$ 的边界处趋零——正是「红/橙」这类相邻概念的模糊区。

**拉回度量的「像 / 核」分解。** 承接 §2.2.6 的 $g=J_D^\top J_D$（$J_D$ 为读出层雅可比）：$g$ 的**像空间**由 $\{w_v-w_u\}$ 张成，是「token 间区分方向」，决定 $\mathcal R_v$ 的超平面边界；$g$ 的**核空间**是「token 内的连续自由度」，即 $\mathcal R_v$ 内部、读出层看不见的方向。因此 $\mathrm{rank}(g)\le\min(|\mathcal V|,d)$ 精确刻画「连续 $Z$ 被折叠成多少个可区分离散 token」——可区分维度至多 $\min(|\mathcal V|,d)$，其余维度都是 token 内的连续语义。

位置混合使解码器能借助其他位置纠正单个位置的采样误差，从而对 Flow 采样误差更鲁棒。

**(b) 率是架构硬约束，不是损失项。**
本模型没有任何 KL 项或码率惩罚。码率由 $|\mathcal{V}|^L$ 硬性封顶，对应 (14) 的约束集而不是 (13) 的 $\beta$。训练中只有失真项（重建交叉熵：JEPA 阶段 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `forward_losses`、Flow 阶段的教师强制重建；另加 Flow 端点采样重建一路）和防塌缩正则。

**(c) 确定性编码器下 ELBO 不成立。**
若 $q(z\mid x)$ 是点质量（确定性编码器），则 $\mathrm{KL}(q(z\mid x)\|p(z)) = +\infty$，VAE 型 ELBO 只是形式类比，不是本模型的目标。不能写成"训练目标已隐含率–失真结构"：真正的结构是"失真项 + 架构率约束"。

**(d) `variance_loss` 的方向与率惩罚相反。**

$$\mathcal{L}_{\text{var}} = \mathbb{E}_j\bigl[\mathrm{relu}(1 - \sigma_j)\bigr],\qquad
\sigma_j = \mathrm{std}\bigl(\mathrm{reshape}(z,\,[-1, d])_{:,j}\bigr), \tag{36}$$

即 [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `variance_loss`（$\sigma_j$ 是 batch×位置维度上的边际标准差）。它在 $\sigma_j < 1$ 时受罚，作用是**抬高**潜空间各维的熵、防止维度塌缩，方向与"压低 $I(Z;T)$"的率惩罚相反。它与率无关，只是防塌缩正则。

**(e) 潜空间静止的前提。**
Flow 阶段没有再调用 `update_target`，$E_\xi$ 在 Flow 阶段完全冻结，$p_1$ 恒定。式 (7)–(15) 中的 $Z$ 正是这个静止分布的样本，论证因此成立。

**(f) 连续潜变量 → 离散 token 的解码器选型：两族方案。**

把连续 $Z$（或 Flow 采样出的 $\hat z(1)$）还原成离散 token 有两条路线，它们的差异在"离散化发生的位置"：

1. **连续 Flow + 软化取整（本框架推荐）**。保持 §4.1 的连续 OT 路径与速度场 $v_\psi$ 全程在 $\mathbb{R}^{L\times d}$ 上不变，只在**最后读出一步**做离散化。按"硬度"递增的可选离散化算子：

   - **argmax**（当前实现，式 35，最硬）：$T_j=\arg\max_v(w_v^\top h_j+b_v)$，确定性但不可微；
   - **FSQ（Finite Scalar Quantization）**：把连续向量按固定网格舍入到有限索引，梯度用直通估计（STE）回传，训练/推理一致；
   - **Gumbel-softmax / 温度退火**：训练时软化采样可微，温度从高到低退火逼近 argmax，可在早期提供梯度信号；
   - **RVQ（残差 VQ，仅作序列化层）**：多级残差量化把连续 $Z$ 压成短码传输，读入即反量化回连续，**不参与训练梯度**（承接 §2.6 的接口契约）。

   共同点：离散化是**事后读出**，不破坏 §4.1 的直线性与曲率结论。

2. **离散 Flow / 离散扩散（本框架不必要）**：DFM（Campbell 2024）、D3PM、MaskGIT 直接在离散 token 空间定义转移核或掩码流。它们天然处理离散态，但代价是：

   - 速度场换成离散转移率矩阵，放弃连续潜空间的 OT 直线路径，无法复用 §4.1 的最短路径/曲率论证；
   - 与"$Z$ 全程连续、不施加量化瓶颈"的接口契约（§2.6）冲突——离散 Flow 要求潜变量本身就是离散 token；
   - 采样步数、温度退火、去掩码顺序需单独调参，与现有 Flow 阶段不共享。

   结论：本框架 $Z$ 已确立为连续精华、离散化只在 token 读出层发生，故选**连续 Flow + FSQ / 温度退火 argmax**；离散 Flow 仅在要端到端替代自回归 token 生成时才考虑。

**(g) 编码器/解码器都用 Mamba 的对称性与训练优势。**

若 $E_\theta$ 与 $D_\eta$ 的主干都从位置混合 Transformer 换成 Mamba/SSM（把 self-attention 换成选择性扫描），训练与推理两端的变化如下：

- **复杂度对称**：注意力是 $O(L^2)$ 时间、$O(L)$ 缓存；Mamba 选择性扫描是 $O(L)$ 时间、$O(1)$ 隐藏状态。编码与解码两侧都从二次降到线性，序列越长收益越大。
- **训练并行**：Mamba 用关联扫描（associative scan）工作高效并行化——$O(L)$ 总工作量、$O(\log L)$ 串行深度（对应 §4.3 的判据），单层训练不退化；长序列下注意力反而受 KV 内存瓶颈。
- **推理解码加速**：逐 token 读出时 Mamba 每步只更新常数大小的隐藏状态（$O(1)$ per token），注意力则每步重算增长中的上下文（$O(L)$ per token）。
- **对称架构红利**：编码器与解码器同构，可共享扫描核、权重初始化、超参数与算子实现，甚至部分权重绑定。

但要诚实标注边界——"能否节省训练时间"取决于序列长度：

- 冒烟配置（`image_size=32`、`patch_size=8`、`dim=64/128`）序列极短（$L=16$），$L^2$ 项远未成瓶颈，Mamba 不会带来可测的训练加速，反而引入 SSM 的 $A,B,C$ 参数与 ZOH 离散化开销；
- Mamba 的训练优势在长序列（$L \gtrsim 10^3$）才兑现，短序列下 Transformer 常数因子更优、实现更成熟；
- 因此当前字符级短序列不必换 Mamba；若要扩到文档/视频帧级长上下文，Mamba 的线性复杂度与 $O(1)$ 缓存是直接收益。

具体选哪条离散化路线、是否切 Mamba，均需按目标序列长度与规模实测后才能定论。

命题 1–4 的信息论结论：

> $T$ 是 $Z$ 的严格有损描述（式 6），失真地板为正（式 10），且高码率下视觉通道可逼近无损而文本不能（式 12）；文本把有限的码率花在语义码率 $I(T;Y)$ 上（式 15），因而是一个**关于共享语义 $Y$ 的摘要**，而不是 $Z$ 的近似重建。

### 4.3 GPU 并行实现：离散瓶颈与条件后验

命题 1–4 说明"为什么"需要离散瓶颈与模糊后验；本节讨论"怎么算"。判据不是 FLOPs，而是**串行深度**——一次推理从输入到输出必须顺序执行的步数。它决定 GPU 的并行利用率。

两个算子：

- 前向压缩：$p_\theta(T\mid Z)$，把连续潜变量采样成离散 token；
- 反向回溯：$p_\phi(Z\mid T)$，由文本恢复潜变量的**分布**，其后验熵就是"文字背后事件的模糊性"。

训练端两者都无序列依赖（随机 $t$ / 随机 mask 均可并行）；差异在推理端。

**(a) 串行深度对比。**

| 方法 | 训练串行依赖 | 推理串行深度 | 大模型案例 | 方向 |
|---|---|---|---|---|
| 条件 Flow / 扩散 | 无（每样本独立采 $t$） | $O(N)$，与 $L$ 解耦，可蒸馏到 1–4 步 | SD3、Flux.1、Sora、π0、AlphaFold 3、Cosmos | 离散→连续分布 |
| 离散扩散 / MaskGIT | 无（随机 mask） | $O(T_{\text{refine}})$，位置维并行 | MaskGIT、Muse、LLaDA、Gemini Diffusion | 离散→离散 |
| VQ/RVQ + 自回归 | 无（teacher forcing） | $O(L)$，随序列线性增长 | DALL·E 1、Parti、MusicGen、VALL-E | 连续→离散 |
| VQ + MaskGIT | 无 | $O(T_{\text{refine}})$，位置并行 | Muse | 连续→离散 |
| MDN / CVAE | 无 | $O(1)$ | 多为中小规模 | 离散→连续分布 |
| Normalizing Flow | 无 | $O(1)$（层间串行） | Glow 等，非前沿 | 离散→连续分布 |

关键差异：自回归的串行深度随 $L$ 线性增长；flow 的串行深度是步数 $N$，**与序列长度无关**，且每步都是 $B\times L\times d$ 的稠密并行。$L$ 较大时这是数量级的差别。

**(b) 前向：三种可并行的量化。**

**FSQ**（无码本，最省显存）逐维取整并用直通估计梯度；$L_i$ 需为奇数，码字关于 0 对称：

$$\hat z_i=\frac{\mathrm{round}\!\big(\tfrac{L_i-1}{2}\tanh(z_i)\big)}{(L_i-1)/2},\qquad
\hat z_i\leftarrow \tanh(z_i)+\mathrm{sg}\big(\hat z_i-\tanh(z_i)\big). \tag{37}$$

码本大小为 $\prod_i L_i$，计算是纯逐元素，没有 $K$ 维距离矩阵。

**VQ / RVQ**（固定码本，码率可控；码本经 EMA 更新，而非梯度）：

$$k^{\star}=\arg\min_k\|z-e_k\|_2^2,\qquad
\mathcal L_{\mathrm{VQ}}=\gamma\,\big\|z-\mathrm{sg}[e_{k^{\star}}]\big\|^2, \tag{38}$$

其中距离矩阵可用 $d_{bk}=\|z_b\|^2+\|e_k\|^2-2z_be_k^\top$ 一次矩阵乘得到，码本向量用 EMA 追踪分配到该码字的编码器输出。RVQ 再把残差逐级量化：

$$\hat z=\sum_{m=1}^{M}e^{(m)}_{k_m},\qquad
r^{(m+1)}=r^{(m)}-e^{(m)}_{k_m},\qquad
k_m=\arg\min_k\|r^{(m)}-e^{(m)}_k\|^2, \tag{39}$$

$M$ 级之间少量串行、级内全并行。

**随机量化信道**（把"采样"写成概率）：

$$p_\theta(k\mid z)=\mathrm{softmax}_k\!\left(-\frac{\|z-e_k\|^2}{2\tau^2}\right). \tag{40}$$

$\tau$ 越小越接近确定性量化、率越高；$\tau$ 越大越模糊、率越低。它就是式 (10) 中"码率预算 ↔ 失真地板"的可调旋钮。

**(c) 反向：条件 Flow 作为摊销后验。**

把 §4.1 的 Flow Matching 条件从上下文 $c$ 换成文本 $T$，即得 $p_\phi(Z\mid T)$ 的摊销采样器：

$$\mathcal L_{\mathrm{CFM}}=\mathbb E_{t,Z_0,Z_1,T}\Big\|v_\phi\big((1-t)Z_0+tZ_1,\,t\mid T\big)-(Z_1-Z_0)\Big\|^2, \tag{41}$$

推理用 Euler 积分

$$Z^{(k+1)}=Z^{(k)}+\tfrac{1}{N}\,v_\phi\big(Z^{(k)},\tfrac{k}{N}\mid T\big),\qquad k=0,\dots,N-1. \tag{42}$$

训练时 $t$ 逐样本独立采样，没有时间步展开；推理串行深度是 $N$，与 $L$ 无关。蒸馏（一致性模型 / LCM / DMD）可把 $N$ 压到 1–4。

"模糊回溯"的度量：并行采 $M$ 条轨迹，用后验离散度作为 $H(Z\mid T)$ 的经验代理，

$$\widehat{\mathrm{Var}}(Z\mid T)=\frac{1}{M}\sum_{m=1}^{M}\big\|Z^{(m)}-\bar Z\big\|^2,\qquad
\bar Z=\frac{1}{M}\sum_{m=1}^{M}Z^{(m)}. \tag{43}$$

$M$ 个样本落在同一 batch，GPU 并行；$\widehat{\mathrm{Var}}$ 越大，说明该文本对应的潜变量越模糊。

**(d) 离散侧也要并行时。**

若不想让文本 token 本身自回归生成，用 MaskGIT / 离散扩散：每步对所有位置并行预测，按置信度保留前 $k$ 个，串行深度 $O(T_{\text{refine}})$，与 $L$ 解耦。它与 (34)–(37) 的离散码本配合，构成"压缩 + 并行解码"。

**(e) 大模型证据与取舍。**

- **条件 Flow / 扩散**：SD3、Flux.1（rectified flow，连续 VAE 潜变量 + 文本交叉注意力）；Sora、Wan、HunyuanVideo、Cosmos（视频 DiT）；π0（语言+视觉条件、flow matching 动作分布）；AlphaFold 3（连续坐标扩散）；Matcha-TTS（条件 flow matching）。这是"文本条件 → 连续模糊分布"最成熟的路线。
- **VQ/RVQ**：DALL·E 1（dVAE + 自回归）、Parti（ViT-VQGAN + 自回归）、EnCodec / MusicGen / VALL-E（RVQ 音频）。压缩端并行，但自回归解码端串行——并行性来自量化，不来自自回归。
- **MaskGIT / 离散扩散**：Muse、LLaDA、Gemini Diffusion。位置并行，但文本质量仍在追赶自回归。
- **MDN / Normalizing Flow**：串行深度最小，但前沿大模型案例少，多峰与质量不如 flow/扩散。

### 4.4 噪声作为熵源：为什么后验采样必须从噪声出发

式 (41)–(42) 的采样器把 $z_0\sim\mathcal N(0,I)$ "去噪"成潜变量，但噪声的真实角色不是"被去除的脏东西"，而是**熵源**——它为条件后验 $p(Z\mid T)$ 提供宽度。下面把这一点写成与源码对齐的数学事实。

**(a) 点回归的熵坍缩。** 命题 1 断言 $H(Z\mid T)>0$，即给定 $T$ 后 $Z$ 仍是连续分布。若把采样器退化为确定性点回归 $\hat Z=\mathbb E[Z\mid T]$（式 (7) 的 MMSE 最优），它诱导的是点质量 $q(Z\mid T)=\delta(Z-\mathbb E[Z\mid T])$，微分熵为

$$h(q)=-\infty .$$

点质量无法表达后验的宽度——它只报告"最可能的一个点"，把"文字背后有多种可能"整件事抹掉。而 $H(Z\mid T)$（连续版本 $h(Z\mid T)$）要度量的正是这个宽度。

**(b) 噪声是采样器随机性的唯一来源。** 推理起点是 `torch.randn`，即 $z_0\sim\mathcal N(0,I)$；训练端同样以 `torch.randn_like(target)` 注入噪声。采样器 $\Phi_1$ 是 Flow ODE 在 $t\in[0,1]$ 上的积分映射，给定 $T$ 后 $\hat Z=\Phi_1(z_0)$ 的全部随机性来自 $z_0$。连续流可逆，由微分熵的变量替换公式

$$h(\hat Z\mid T)=h(z_0)+\mathbb E_{z_0}\Big[\log\Big|\det\tfrac{\partial\Phi_1(z_0)}{\partial z_0}\Big|\Big]. \tag{44}$$

噪声熵 $h(z_0)$ 是底数，雅可比行列式是流对体积收缩/膨胀的修正。标准高斯每维微分熵 $\tfrac12\log_2(2\pi e)\approx 2.047$ bit，故

$$h(z_0)=\frac{Ld}{2}\log_2(2\pi e). \tag{45}$$

**(c) 噪声熵在收缩流下是输出熵的上界。** 严格成立的是式 (44)；"上界"成立与否完全取决于流 $\Phi_1$ 的雅可比行列式 $\det J$ 的符号。本节把它作为一个**假设（收缩流假设）**：$\Phi_1$ 在数据支撑上体积不膨胀，即 $\mathbb E[\log|\det J|]\le 0$。⚠️ 该假设**不能**由 variance_loss（式 (36)，[losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `variance_loss`）推出——后者约束的是**编码器输出分布各维的边际标准差** $\sigma_j\gtrsim 1$，而 $\det J$ 是**采样器** $\Phi_1$ 的体积变化率，二者是不同对象的性质、无蕴含关系。收缩流假设是「数据支撑上近似体积守恒」的经验性前提，需实测或另加正则保证，而非自动成立。在该假设下代入式 (44) 得

$$h(\hat Z\mid T)\;\le\; h(z_0)=\frac{Ld}{2}\log_2(2\pi e). \tag{46}$$

这就是"噪声作为熵源"的精确含义：**在收缩流（$\mathbb E[\log|\det J|]\le0$，典型情形）下，采样器输出分布的熵不会超过注入噪声的熵（体积守恒时取等号）**。要覆盖后验宽度，噪声必须供给足够熵：

$$h(z_0)\;\gtrsim\; h(Z\mid T). \tag{47}$$

以冒烟尺度 tiny（$L=16$、$d=64$）计，$h(z_0)\approx 16\times 64\times 2.047\approx 2096$ bit，比文本码率上界 $H(T)\le L\log_2|\mathcal V|$（式 (2)）高出约一个数量级以上——噪声端的熵预算远大于文本端，与"文字高压缩、潜空间高冗余"的直觉一致。该比例 $\approx d\cdot 2.047/\log_2|\mathcal V|$ 与 $L$ 无关，各规模下的完整演算见 §2.8。

**(d) 与离散熵的衔接。** 式 (4)–(6) 用网格量化 $Z_\delta$ 定义的是离散熵 $H(Z_\delta\mid T)$；本节 $h(Z\mid T)$ 是连续微分熵。二者由式 (4) 的条件版本联系：

$$H(Z_\delta\mid T)\approx h(Z\mid T)+Ld\,\log_2\tfrac{1}{\delta},$$

固定分辨率 $\delta$ 后只差一个与 $T$ 无关的常数。注意式 (43) 的经验离散度 $\widehat{\mathrm{Var}}(Z\mid T)=\tfrac1M\sum_m\|Z^{(m)}-\bar Z\|^2$ 度量的是**同一文本 $T$ 采样 $M$ 条生成轨迹的跨样本方差**，即连续条件宽度 $h(Z\mid T)$；它**不是** [losses.py](file:///home/simon-ma/JEPA/training/losses.py) 的 `variance_loss`（后者是 `z.reshape(-1,d).std(dim=0)`，度量 batch×位置维度上各维的**边际**标准差、防维度塌缩，对应 $h(Z)$ 的各维宽度）。二者对象不同：式 (43) 测"这段文字背后潜变量有多散"（需对同一 $T$ 重复采样），`variance_loss` 测"不同样本的潜变量各维是否塌缩"，与"文字本身多少 bit"是两回事。

**(e) 熵源 ≠ 被去噪的噪声。** 式 (44)–(47) 表明 $\mathcal N(0,I)$ 的熵最终转化为后验 $\hat Z$ 的宽度，代表"文字说不清的那部分"（$h(Z\mid T)$）。它**不是**压缩时注入、等着被去噪还原的细节：压缩端 $g$ 是确定性的（$H(T\mid Z)=0$，式 A2，argmax 解码见 [decoder.py](file:///home/simon-ma/JEPA/model/decoder.py) 的 `TextDecoder`），随机性全部来自生成端 $z_0$。这与 §4.2(c) 的"确定性编码器下 ELBO 不成立"是同一类辨析——不能把采样器的随机性误读成编码端的噪声。

综合：

> 条件 Flow Matching 在本节判据下最优：训练无串行依赖，推理串行深度 $O(N)$ 与序列长度 $L$ 解耦，且有 SD3 / Flux / Sora / π0 等大规模背书。它正是命题 3 中 $p_\phi(Z\mid T)$ 的实现；其采样离散度 (40) 就是命题 1 中 $H(Z\mid T)$ 的可测代理。前向若要学习型离散瓶颈，配 FSQ / RVQ；若要文本侧也并行，配 MaskGIT。

### 4.5 关系 vs 通路的计算复杂度分离

把「通路映射」与「数据内部关系」分开是有意的：二者在复杂度与并行度上完全不同，混在一起会妨碍各自优化。

**通路映射是逐位置独立的 $O(L)$ 操作。** FSQ（式 37 逐维量化）、马赛克 $\phi$（§2.7 逐块平均）都只看单个位置/块，位置间无耦合，是 $L$ 个独立运算，串行深度 $O(1)$，天然 GPU 并行。注意解码器 $g$ 的读出头虽是逐位置线性打分，但其前有 self-attention 做位置混合，属于下面的关系类 $O(L^2)$，不再是纯通路投影。

**数据内部关系由 self-attention 承载，是 $O(L^2)$ 的成对交互。** 编码器 $E_\theta$ 用 attention 让 $z_i$ 聚合整条序列上下文，token $i$ 与 $j$ 的关系被折叠进 $z_i,z_j$ 的坐标值（分布式表征），而非一条显式边。$Z=(z_1,\dots,z_L)$ 的联合结构 $p(Z)$ 承载关系；JEPA 掩码预测 $\hat z=P_\phi(c)$ 显式学习 $p(z_{\text{masked}}\mid z_{\text{visible}})$，即位置间依赖。

**因此关系是 $O(L^2)$ 瓶颈，通路是 $O(L)$ 廉价投影。** 注意力矩阵是 $L\times L$ 成对计算（计算与显存的主开销），量化与解码读出头是 $L$ 个独立投影，而解码器的位置混合 self-attention 属于关系类。分离二者后各自可独立优化：通路可稀疏化/低精度，attention 可线性化（FlashAttention、线性注意力、Mamba 等）。这正是本节串行深度讨论的根源——关系建模的成本集中在 attention 层，而非通路层。

**有损性正是「关系被丢弃」。** $T=g(Z)$ 逐位置量化只保留符号，位置间关系信息进入纤维 $\mathcal F_t$（式 6），构成 $H(Z\mid T)>0$ 的一部分。生成时关系不是从数据直接搬来，而是 $v_\psi$ 的 attention 每步动态"注入"：起点 $z_0\sim\mathcal N(0,I)$ 各维独立（无关系），经 self-attention 逐步编织成有结构的 $z_1$。

### 4.6 一致性约束：后验采样不漂出纤维（plan P1）

§4.3(c) 式 41 的 $p_\phi(Z\mid T)$ 只做了 CFM 回归，未约束"采样出的 $\hat Z$ 重新量化回去等于 $T$"。数学上，$p_\phi(Z\mid T)$ 应支撑在纤维 $\mathcal F_T=g^{-1}(T)$ 上，否则回溯会漂出 $T$ 对应的潜空间区域。

**一致性项。** 给后验采样加约束

$$\mathcal L_{\text{consist}} = \mathbb E_{Z_0,T}\Big[\mathrm{CE}\big(g(\Phi_1(Z_0)),\, T\big)\Big], \tag{48}$$

其中 $\Phi_1$ 是 Flow ODE 的积分映射（式 42），$Z_0\sim\mathcal N(0,I)$。它强制 $\hat Z=\Phi_1(Z_0)$ 经 $g$ 量化回去等于 $T$，即 $\hat Z\in\mathcal F_T$；soft 版本可用 $-\log p_\theta(T\mid\hat Z)$ 替代 $\mathrm{CE}$。

**数学依据。** 这是超分/还原领域最成熟的经验——下采样再上采样必须回到观测（consistency）。在本框架里它把式 41 的边际 CFM 收紧为**纤维条件**的 CFM：不是学任意 $p(Z\mid T)$，而是学支撑在 $\mathcal F_T$ 上的 $p(Z\mid T)$。缺一致性时，模型可能学会"平均掉纤维内变异"的 $p(Z\mid T)$，其样本 $\hat Z$ 量化后不等于 $T$，违背 $p(T\mid Z)$ 的确定性（A2）。

---

## 5. 阶段四 · 自反馈修正：闭环校正

前三个阶段（编码 §2 → 对齐与推理 §3 → 解码 §4）是一条**开环前向**链路：输入进、输出出，中间没有把产物回灌验证的环节。但解码产物是否可信、算子 $\Psi$ 是否真的模拟了世界，只有把输出重新读回表征空间、与推理预测比对才能发现。这就是第四阶段——**自反馈修正**。

**观测与创新。** 生成端产出 $\hat Z = \Phi_1(Z_0)$（式 42）与 $T = g(\hat Z)$；自反馈把它重新压回表征空间得到观测

$$\tilde Z = E(T), \qquad e = \hat Z - \tilde Z . \tag{49}$$

$e$ 是**创新**——推理表征与可观测表征之间的失配。由命题 1 的 $H(Z \mid T) > 0$，$\tilde Z$ 几乎处处不等于 $\hat Z$，故 $e$ 恒非零：**文本越有损，自反馈的校正空间越大**。

**硬反馈与软反馈。** 上式把解码产物 $T=g(\hat Z)$ 重编码当作「观测」，但须区分两种性质不同的创新：

- **软反馈（自洽性，文无第一）**：$\tilde Z=E(g(\hat Z))$ 是模型**自己解码再重编码**的镜像，不是独立观测。$e_{\text{soft}}=\hat Z-\tilde Z$ 只衡量编解码可逆性 $E\circ g\approx\mathrm{id}$，不衡量世界描述的正确性——一个完全错误但「自洽」的世界模型（$E,g$ 互相还原、却都错）也能让 $e_{\text{soft}}\to 0$。文字无客观对错，故该信号只能校正「可还原性」，不能校正「正确性」。
- **硬反馈（正确性，武无第二）**：§3.6 式 23 的 $e_{\text{hard}}=E(o_{t+1})-\hat Z^-$，观测 $o$ 来自**真实世界下一帧**（图像/视频/触觉/传感器），有客观物理规则不变量作评价体系。预测错 → 下一步观测打脸 → $e_{\text{hard}}$ 非零；命题 8 的「创新白噪声 ⟺ 模型充分」诊断也只在硬反馈下才有客观意义。

因此「创新驱动校正」须按通道区分：物理通道走硬反馈（$e_{\text{hard}}$，正确性），文字通道只能走软反馈（$e_{\text{soft}}$，自洽性）。正确性锚只能来自硬反馈；文字通道贡献的是语义结构与可还原性约束（§4.6 一致性），不贡献正确性判定。

**两类修正。** 同一创新驱动两个层次的校正：

- **状态修正（在线）**：$\hat Z^+ = \hat Z - K e$，$K$ 为增益。这是 §3.6 闭环观测器 $\hat Z_{t+1} = \Psi(\hat Z_t,\omega_t) + K e_t$ 从「推理内部」提升到「系统级」——把解码产物当作一次观测，对预测做卡尔曼式校正。
- **参数修正（训练）**：把 $\|e\|^2$ 与一致性项 $\mathcal L_{\text{consist}}$（式 48）回灌 $\Psi, E, D_\eta$。§4.6 的一致性约束由此从「解码阶段的训练损失」升级为「系统级自反馈信号」。

**与前后阶段的关系。** 编码（§2）定义 $Z$ 与 $E$，推理（§3）提供算子 $\Psi$ 与观测器 $K$，解码（§4）产出可回灌的观测 $T$；自反馈修正（本节）把这四者的误差闭环。它不新增模态通道、不引入新参数家族，只是复用 $E, \Psi, K$ 并把它们接成反馈回路。

> **命题 10（软反馈信号的不竭性，修订）**：若命题 1 成立（$H(Z\mid T)>0$），则软反馈创新 $e_{\text{soft}} = \hat Z - E(g(\hat Z))$ 几乎处处非零，自洽性校正始终有可用的驱动信号；当且仅当通道无损（$H(Z\mid T)=0$）时该信号才可能枯竭。但须限定：**不竭的只是「自洽性信号」，不是「正确性信号」**——$e_{\text{soft}}\to 0$ 只能保证 $E\circ g$ 可还原，不保证 $Z$ 正确描述世界；正确性锚只能来自硬反馈（物理观测 $o$）。故「文字有损」是自洽性校正的能量源，而非正确性校正的能量源。

---

## 6. 落到本模型的可检验推论与实验

由第 2、4、5 节可直接导出：

| 推论 | 实验 | 预期 |
|---|---|---|
| $H(T) \le L\log_2|\mathcal{V}|$（式 2） | 采样生成文本，用 held-out n-gram 或独立 LM 估计 $\hat H(T)$ | $\hat H(T) < L\log_2|\mathcal{V}|$（数值随规模，见 §2.8）；生成样本近确定性，经验熵会偏低，需交叉估计 |
| 失真地板 $D_T \ge D^{\star}$（式 10） | 把生成文本 $T$ 重新编码 $\hat Z = E_\theta(T)$，与真实 $Z$ 比较 | $\mathbb{E}\|Z-\hat Z\|^2$ 显著大于 0；须与随机 token 基线对比，不能只看余弦相似度 |
| 率失真曲线（式 11–12） | 改变词表粒度（字符 → 子词 → 词）重训，画 $(\hat H(T),\, \hat D_T)$ | 文本工作点落在"低码率、高失真"区域，$D$ 随 $|\mathcal{V}|$ 减小而上升 |
| 语义选择性（式 15） | 用跨模态 / 下游探针定义 $Y$，比较 $I(T;Y)$ 与 $H(Z\mid T)$ | 语义码率占比高；细节损失集中在低语义方向，需用探针方向而非裸 $d$ 维度 |

---

## 7. 参考文献

1. Lipman, Y. et al. *Flow Matching for Generative Modeling*, ICLR 2023.（§4.1 概率路径与速度场）
2. Assran, M. et al. *I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, 2023.（§2.2 JEPA）
3. Bardes, A. et al. *VICReg: Variance-Invariance-Covariance Regularization*, ICLR 2022.（§2.2.4 方差约束）
4. Tishby, N., Pereira, F. C., Bialek, W. *The Information Bottleneck Method*, 1999.（§2.5）
5. Shannon, C. E. *Coding theorems for a discrete source with a fidelity criterion*, 1959.（§2.4 率失真）
6. DeepMind. *Language Modeling Is Compression*, arXiv:2309.10668, 2023.（预测模型 ⟺ 压缩器的经验背书）
7. Li, J. et al. *JUICE: Unified Multimodal Modeling in Pretrained Representation Spaces*, arXiv:2610.28801, 2026.（"连续生成文本会丢措辞"的经验观察，对应 §2.5）
8. Strouse, D. J., Schwab, D. J. *The Deterministic Information Bottleneck*, Neural Computation, 2017.（§2.5 的 DIB 形式）
9. Esser, P. et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, ICML 2024.（SD3；§4.3 条件 Flow 的大规模案例）
10. Chang, H. et al. *Masked Generative Image Transformer*, CVPR 2022.（MaskGIT；§4.3 并行解码）
11. Mentzer, F. et al. *Finite Scalar Quantization: VQ-VAE Made Simple*, 2023.（FSQ；§4.3 前向量化）
12. Black, K. et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024.（§4.3 flow matching 工业案例）
13. Défossez, A. et al. *High Fidelity Neural Audio Compression*, 2022.（EnCodec / RVQ；§4.3）
14. Nie, S. et al. *Large Language Diffusion Models*, 2025.（LLaDA；§4.3 离散扩散）
15. Williams, P. L., Beer, R. D. *Nonnegative Decomposition of Multivariate Information*, arXiv:1004.2515, 2010.（§8.3 冗余/协同的部分信息分解 PID）

---

## 8. 附录 · 跨模态 ρ 融合：从「模态内私有丢/留」到「跨模态联合选材」（可行性分析，规划级）

> 本节回应一个具体的扩展设想：在机器人等多模态冗余显著的任务里，§2.6 的 `ρ → 丢/留` 判据能否从「模态内私有」升级为「跨模态联合」，进而把 `ρ → 丢/留 → x̃ → E(共享主干)` 整条链跨模态合并、从多个模态里挑信息掉入 $Z$，从而缩减模型规模。本节只做**数学可行性分析，不改代码**；命题为**分析命题（未形式化）**，区别于 §1.0 的 Lean 已证结论。落地前须过 §8.8 的前提核对与 §8.9 的判据。

### 8.1 问题重述与 §2.6 的既有结论

§2.6 末尾给出当前多模态架构（原文图）：

```
触觉 x ─► M_触觉(私有) ─► ρ ─► 丢/留 ─► x̃ ─► E(共享主干) ─┐
视觉 x ─► M_视觉(私有) ─► ρ ─► 丢/留 ─► x̃ ─► E(共享主干) ─┼─► Z(跨模态共享潜空间)
文本 x ─► M_文本(私有) ─► ρ ─► 丢/留 ─► x̃ ─► E(共享主干) ─┘
```

并下了结论：**「$M$ 模态内私有，跨模态统一留给 $Z$」**——$\rho=\|v-Mk\|^2$ 只判「观测冗余」（同一模态内已被记忆吸收的信息），「语义冗余」（图像与文本说同一件事）应靠 §3.2 对齐损失在 $Z$ 层挤掉，不该提前到 token 层，理由是「不同模态的 token 语义空间不同、不可直接比，共享 $M$ 会容量争抢 + 灾难性遗忘」。

本附录讨论的正是这条结论的**反方向**：当跨模态冗余足够强（机器人场景），能否让 $\rho$ 也跨模态判定，从而在 $E$ 之前就把冗余信息跨模态裁掉。先把命题拆成两个可分别判定的子命题：

1. **(决策可跨模态)** $\rho$（及丢/留）能否以「其他模态 + 本模态记忆」为条件做判定？（§8.4 可证，单调占优）
2. **(存储/前端可合并)** $M$ 与物理前端 $M_m$ 能否随之合并成一个共享模块？（§8.5 不可简单合并，token 空间不同）

### 8.2 信息论根基：共同原因 $W$ 与条件独立

跨模态冗余能被「识破」的前提是存在共同原因。设世界状态 $W$，各模态是其有噪观测：

$$X_m = f_m(W) + \eta_m,\qquad m=1,\dots,K, \tag{50}$$

其中 $\eta_m$ 为模态噪声。由 $W$ 为公共父节点，各模态**给定 $W$ 条件独立**：

$$X_m \perp\!\!\!\perp X_{m'} \mid W,\qquad \forall m\ne m'. \tag{51}$$

(51) 是本节一切结论的地基：它保证「跨模态冗余」有明确的信息论对象——凡是被 $W$ 决定、且被至少两个模态同时携带的那部分信息，就是冗余，可被安全识别并裁剪。

### 8.3 冗余 vs 互补：PID 分解与「机器人先验」

跨模态信息并非全是冗余，也非全是互补。用部分信息分解（PID, Williams & Beer 2010，参考文献 15）把总互信息三分：

$$I(X_{1:K}; W) = I_{\cap} + I_{\mathrm{uniq}} + I_{\mathrm{syn}}, \tag{52}$$

- $I_{\cap}$（冗余）：至少两个模态各自携带、可互相替代的公共信息——**跨模态 $\rho$ 可裁的对象**；
- $I_{\mathrm{uniq}}$（独占）：仅一个模态携带的信息——必须保留；
- $I_{\mathrm{syn}}$（协同）：只在「多模态联合」时才显现的信息——必须联合保留。

§3.3 互补压缩（式 19–20）强调的是 $I_{\mathrm{uniq}}+I_{\mathrm{syn}}$ 那部分（图像补文字丢的空间、文字补图像丢的时间语义）；本附录强调的是 $I_{\cap}$ 那部分。**二者不矛盾，是同一分解的两个分量**——互补性驱动「必须多模态」，冗余性驱动「可以跨模态裁剪」。

「机器人先验」的精确表述：设跨模态冗余占比

$$q = \frac{I_{\cap}}{I(X_{1:K}; W)} \in [0,1]. \tag{53}$$

机器人（视觉 / 本体感觉 / 触觉 / 控制指令常同时编码「末端位姿」）$q$ 大；图文 caption（文字补充图像没有的语义标签）$q$ 偏小。**跨模态 $\rho$ 的收益单调随 $q$ 增长**，$q\to1$ 时退化为「只需一个模态即可覆盖 $W$」的极限。

### 8.4 跨模态 $\rho$ 的单调占优（可行性定理）

把 §2.6 的「可约损失 / 显著性」判据从模态内推广到跨模态。模态 $m$ 位置 $t$ 的显著性：

$$s^{\mathrm{intra}}_{m,t} = I(x_{m,t}; Y \mid \tilde x_m),\qquad
s^{\mathrm{cross}}_{m,t} = I(x_{m,t}; Y \mid \tilde x_m,\ x_{-m}), \tag{54}$$

其中 $\tilde x_m$ 是本模态已留子集、$x_{-m}$ 是其余模态观测。二者都是 §2.5 式 (14) DIB 目标「在固定容量下最大化 $I(Z;Y)$」的单 token 代理；$s^{\mathrm{intra}}$ 对应 §2.6 的 $\rho=\|v-Mk\|^2$（以本模态记忆 $M$ 为条件），$s^{\mathrm{cross}}$ 以「本模态记忆 + 全部其他模态」为条件。

> **命题 12（跨模态裁剪的单调占优，分析命题）**：由数据处理不等式，条件集扩大只能缩小残余互信息，
> $$I(x_{m,t}; Y \mid \tilde x_m,\ x_{-m}) \;\le\; I(x_{m,t}; Y \mid \tilde x_m), \tag{55}$$
> 故 $s^{\mathrm{cross}}_{m,t}\le s^{\mathrm{intra}}_{m,t}$，跨模态显著性集是模态内显著性集的**子集**；当 $I_{\cap}>0$ 时严格更小。于是，在保留相同 $I(Z;Y)$ 的前提下，跨模态裁剪后的入 $E$ 序列长度 $L_{\mathrm{kept}}$ 严格更短。

**规模缩减的排序推论。** 定义三条管线（保留相同 $I(Z;Y)$）：

- $P_0$（现状，§2.6）：只按 $s^{\mathrm{intra}}$ 丢/留；
- $P_1$（跨模态门控，私有 $M$ 保留）：按 $\min(s^{\mathrm{intra}}, s^{\mathrm{cross}})$ 丢/留；
- $P_2$（全合并共享 $M$）：整链合并。

则入 $E$ 序列长度满足

$$L_{\mathrm{kept}}(P_0) \;\ge\; L_{\mathrm{kept}}(P_1) \;\ge\; L_{\mathrm{kept}}(P_2). \tag{56}$$

命题 12 保证「数学上占优」；但 $P_1,P_2$ 引入了新的塌缩模式（§8.6），故**可达到**的保留互信息未必不降——这是规模缩减与训练稳定性之间的取舍，见 §8.9。

### 8.5 关键难点 1：决策可跨模态，存储/前端不可简单合并

命题 12 只证明「$\rho$ 的**条件集**可以跨模态」，不等于「$M$ 和 $M_m$ 可以合并」。§2.6 的 token 空间不可比论据依然成立：不同模态 token 语义空间不同，直接共享 token 级记忆会容量争抢 + 灾难性遗忘。更致命的是一个**循环依赖**：

- 若 $M$ 要跨模态共享，它必须以「共同的语义空间」为 key/value；
- 而这个共同空间正是 $E$ 的产出 $Z$（或 $Z$ 的预览）；
- 于是「在 $E$ 之前用一个共享 $M$ 判冗余」就要求先有一个 mini-$E$ 把各模态投到公共空间——这与「缩减 $E$ 的规模」目标相悖。

因此「跨模态合并」的正确形态是**分而治之**，与 §2.8「薄接口 + 厚共享主干」一致：

- **跨模态门控（可共享、可小）**：一个共享的世界状态摘要 $\hat W$（如对 $Z$ 的浅层预览 / 共享注意力读出），在其上算跨模态冗余 $\rho^{\mathrm{cross}}_{m,t}=\|v_m-\hat W k_m\|^2$；
- **物理前端 $M_m$ 与私有记忆（保持私有、保持薄）**：只在各自 token 空间内做「先写后压」，负责 $\rho^{\mathrm{intra}}_m$。

这同时化解了 §2.6「语义冗余应在 $Z$ 层挤掉」与本附录「语义冗余可提前裁」的表面矛盾：**跨模态冗余确实提前裁了，但裁的判据落在共享语义空间（$\hat W$），而不是在不可比的 token 空间硬比。** §2.6 的「$M$ 私有」是保守基线，本附录的「跨模态门控」是激进扩展，二者在式 (14) 的 DIB 最优处收敛，差别只在把可压缩预算提前到 $E$ 之前兑现了多少。

### 8.6 关键难点 2：跨模态 $\rho$ 的「模态饥饿」——新的塌缩模式

§2.6 的 ε 强制采样防的是「token 级自我强化退化」；跨模态 $\rho$ 引入一个**更粗粒度**的新塌缩模式：

- 模态 $m$ 前端的梯度 $\propto \mathbb{E}[\mathrm{keep}_{m,t}]$（被保留 token 比例）；
- 若某模态的信息总被其他模态覆盖（$s^{\mathrm{cross}}_m$ 恒小），跨模态 $\rho$ 会把它整体裁掉 → 前端梯度 $\to0$ → 该模态编码器塌缩；
- 到推理时若覆盖模态被遮挡 / 高噪，已塌缩模态无法兜底 → 出现「单点失效」。

这与 §3.2 命题 6（交叉掩码防模态坍缩）方向**相反**：交叉掩码强迫每个模态去补其他模态的缺失（把坍缩模态拽回非平凡），跨模态 $\rho$ 却奖励裁掉冗余模态（把「暂时冗余」变成「永久塌缩」）。二者必须**并用**而非互替。

**数学判据与防御。** 保持每个模态的保留比例有正下界：

$$\mathbb{E}[\mathrm{keep}_{m,t}] \ge \epsilon_m > 0,\qquad m=1,\dots,K. \tag{57}$$

(57) 是 §2.6 ε 强制采样从「token 层」到「模态层」的推广（§2.6 是 2–5% 低 $\rho$ 保留，(57) 是每模态最低保留比例），并与 §3.4 PAC 的「分层 lr / 锚定 ≠ 冻结」同源：都是给「信息暂时冗余但结构上必需」的通道留一条梯度通路。

### 8.7 规模缩减到底省在哪

对照 §2.2.7 的参数量公式与三档规则，把「缩减模型规模」拆成可检验的三项：

| 缩减来源 | 量级 | 依据 | 备注 |
|---|---|---|---|
| 入 $E$ 序列长度 $L_{\mathrm{kept}}$ 下降 | 计算 | attention $O(L^2)$，$L_{\mathrm{kept}}<\sum_m L_m$ | 降低单步 FLOPs，**非参数量** |
| $E$ 可缩小而容量不减 | 参数 | $P_E\approx12\,n\,d_{\mathrm{hidden}}^2$（§2.2.7） | 输入去冗余后，同样 $I(Z;Y)$ 用更小 $d_{\mathrm{hidden}}$/n 即可达 |
| 共享跨模态门控 + 薄前端 | 参数 | §8.5 | 省 $K$ 个私有「冗余编码」的重复开销，**唯一真正合并的部件** |

结论：跨模态 $\rho$ 的**直接**收益是计算量（$L_{\mathrm{kept}}$ 下降）与「等容量下更小 $E$」；「合并 $K$ 个私有 $M$」只对**门控层**成立（§8.5），物理前端不能合并。规模三档的伸缩仍由 §2.2.7 的 $r=d_{\mathrm{hidden}}/d$ 与 $P_E$ 同构控制——跨模态 $\rho$ 不改调用约定，只在 $E$ 前多一道共享门控，纯增量、可配置开关（沿用 §2.6 `rho_compress` 的 warmup 开关思路）。

### 8.8 前提核对（对照 math.md 既有契约）

| 契约 | 是否仍满足 | 说明 |
|---|---|---|
| $Z$ 全程连续（§2.6 接口契约） | ✅ | 丢/留发生在 token 层、$E$ 之前，$E$ 输出仍连续 $Z$ |
| 不施加语义量化瓶颈（§2.6） | ✅ | 门控是 token 级布尔掩码，非 $Z$ 量化 |
| 有损只发生在 token 层（§2.6） | ⚠️ 部分修订 | 跨模态 $\rho$ 把「语义冗余」也提前到 token 层裁，§8.5 已论证这不违反 §2.6 本意（判据在共享语义空间） |
| 反塌缩（§2.2.4 / §2.6 ε / §3.2 命题 6） | ⚠️ 需新增 (57) | 跨模态 $\rho$ 引入模态饥饿，须加每模态保留下界 |

### 8.9 结论与落地判据

**可行性结论**：在共同原因 $W$ + 条件独立 (51) 下，跨模态 $\rho$ 数学上单调占优（命题 12、式 55–56），当且仅当跨模态冗余占比 $q=I_{\cap}/I(X_{1:K};W)$ 足够大（机器人场景满足）时带来可观的规模/计算缩减。但它引入两个新约束：**门控必须落在共享语义空间（§8.5），并须加每模态反饥饿下界（§8.6 式 57）**。这不是对 §2.6 的否定，而是把「$M$ 私有」的保守结论在冗余显著场景下放宽的**激进扩展**。

**落地判据（可检验）**：

| 判据 | 实验 | 阈值 |
|---|---|---|
| $q$ 是否够大 | 跨模态预测探针：用其余模态预测本模态 token 的互信息估计 $\hat I_{\cap}$ | $\hat q > 0.5$ 才值得上 $P_1$ |
| 冗余是否可被 $\hat W$ 捕捉 | 共享门控 $\hat W$ 下，$s^{\mathrm{cross}}$ 与 $s^{\mathrm{intra}}$ 的差 | 差显著 > 0（否则 $P_1$ 无增益） |
| 是否触发模态饥饿 | 训练中每模态 $\mathbb{E}[\mathrm{keep}_{m,t}]$ 曲线 | 任意模态跌破 $\epsilon_m$ 即回退 $P_0$ |

**落地顺序**：先 $P_1$（跨模态门控 + 私有 $M$）验证 $q$ 与增益，再评估 $P_2$（共享 $M$）——$P_2$ 的循环依赖（§8.5）未解前不要一步到位。三档伸缩：tiny/small 冒烟用 $P_0$ 验证正确性基线，中型 0.3B–7B 接真实多模态数据时评估 $P_1$ 的 $q$ 与计算收益。
