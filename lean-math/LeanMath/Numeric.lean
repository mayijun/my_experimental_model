/-
数值校验：math.md 中出现的位计数常量。

· (2)  H(T) ≤ L log₂ |V| = 24 log₂ 77 ≈ 150.40 bit
· (4)  H(Z_δ) ≈ h(Z) + Ld log₂(1/δ)，δ=1e-2 时 Ld log₂ 100 ≈ 5102 bit
· (26) 高斯差分熵每维 ½ log₂(2πe) ≈ 2.047，Ld·½ log₂(2πe) ≈ 1572 bit

整数部分用 `norm_num` 精确校验；对数部分用 `Float` 近似显示。
-/

import Mathlib

namespace Numeric

/-- L = 24 个 token、d = 32 维，L·d = 768（式 (2) 与式 (26) 的公共因子）。 -/
example : (24 : ℕ) * 32 = 768 := by norm_num

/-- 文本信道码字数量 |𝒱|^L = 77^24（式 (1) 的指数）。 -/
example : (77 : ℕ) ^ 24 = 1887023914800623499934622000276222741687622241 := by
  norm_num

end Numeric

-- 24 log₂ 77 的近似值（式 (2)，应为 ≈ 150.40 bit）
#eval (24 : Float) * Float.log (77 : Float) / Float.log (2 : Float)

-- 768 log₂ 100 的近似值（式 (4)，应为 ≈ 5102 bit）
#eval (768 : Float) * Float.log (100 : Float) / Float.log (2 : Float)

-- 高斯差分熵每维 ½ log₂(2πe) 的近似值（式 (26)，应为 ≈ 2.047 bit）
#eval (0.5 : Float) * Float.log (2 * Float.pi * Float.exp 1) / Float.log (2 : Float)

-- Ld · ½ log₂(2πe) 的近似值（式 (26)，应为 ≈ 1572 bit）
#eval (768 : Float) * ((0.5 : Float) * Float.log (2 * Float.pi * Float.exp 1) / Float.log (2 : Float))
