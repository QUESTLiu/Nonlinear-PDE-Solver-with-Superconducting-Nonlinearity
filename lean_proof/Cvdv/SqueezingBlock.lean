import Cvdv.FockHermitian
import Mathlib.Analysis.Normed.Algebra.Exponential

/-!
# The quadratic Duffing correction as a squeezing block

This file formalizes manuscript Eqs. (eq:df-v2-v4) and (eq:df-squeezing).
The algebraic generator is identified
first on finite-support Fock space and only then passed through the strict finite sandwich.
The final exponential is therefore the exponential of the sandwiched generator; no claim is
made that projection commutes with an infinite-dimensional squeezing evolution.
-/

open NormedSpace

namespace Cvdv

/-- The quadratic counter-rotating term `V₂ = 6 (a² + a†²)`. -/
noncomputable def fockV2 : Module.End ℂ FockAlg :=
  (6 : ℂ) • (lowering ^ 2 + raising ^ 2)

/-- The number-changing quartic term
`V₄ = a†⁴ + 4 a†³ a + a⁴ + 4 a† a³`. -/
noncomputable def fockV4 : Module.End ℂ FockAlg :=
  raising ^ 4 + 4 * (raising ^ 3 * lowering) + lowering ^ 4
    + 4 * (raising * lowering ^ 3)

/-- Manuscript Eq. (eq:df-v2-v4): the counter-rotating polynomial is the sum of its
quadratic and quartic number-changing terms. -/
theorem fockCounterRotating_eq_fockV2_add_fockV4 :
    fockCounterRotating = fockV2 + fockV4 := by
  simp only [fockCounterRotating, fockV2, fockV4, Algebra.smul_def, map_ofNat]
  noncomm_ring

/-- Eq. (eq:df-v2-v4) after taking the strict finite Fock sandwich. -/
theorem cutoffCounterRotating_eq_cutoffV2_add_cutoffV4 (N : ℕ) :
    fockSandwichCLM N fockCounterRotating =
      fockSandwichCLM N fockV2 + fockSandwichCLM N fockV4 := by
  rw [fockCounterRotating_eq_fockV2_add_fockV4, map_add]

/-- The algebraic squeezing generator
`(1 / 2) (conj ζ a² - ζ a†²)`. -/
noncomputable def squeezeGenerator (zeta : ℂ) : Module.End ℂ FockAlg :=
  (1 / 2 : ℂ) • ((starRingEnd ℂ) zeta • lowering ^ 2 - zeta • raising ^ 2)

/-- For real `λ` and `Δt`, the short-time quadratic Duffing generator is the squeezing
generator with parameter `ζ = 12 i λ Δt`. -/
theorem fockV2_generator_eq_squeezeGenerator (lam dt : ℝ) :
    (-Complex.I * (lam : ℂ) * (dt : ℂ)) • fockV2 =
      squeezeGenerator ((12 : ℂ) * Complex.I * (lam : ℂ) * (dt : ℂ)) := by
  simp only [fockV2, squeezeGenerator, map_mul, map_ofNat, Complex.conj_I,
    Complex.conj_ofReal, smul_add, smul_smul]
  module

/-- The two generators remain equal after taking the strict finite sandwich. -/
theorem cutoffV2_generator_eq_squeezeGenerator (N : ℕ) (lam dt : ℝ) :
    (-Complex.I * (lam : ℂ) * (dt : ℂ)) • fockSandwichCLM N fockV2 =
      fockSandwichCLM N
        (squeezeGenerator ((12 : ℂ) * Complex.I * (lam : ℂ) * (dt : ℂ))) := by
  rw [← map_smul, fockV2_generator_eq_squeezeGenerator]

/-- Manuscript Eq. (eq:df-squeezing) on the chosen cutoff: exponentiating the strictly
sandwiched quadratic generator gives the same bounded evolution as exponentiating the
strictly sandwiched squeezing generator. -/
theorem exp_cutoffV2_eq_exp_squeezeGenerator (N : ℕ) (lam dt : ℝ) :
    NormedSpace.exp
        ((-Complex.I * (lam : ℂ) * (dt : ℂ)) • fockSandwichCLM N fockV2) =
      NormedSpace.exp (fockSandwichCLM N
        (squeezeGenerator ((12 : ℂ) * Complex.I * (lam : ℂ) * (dt : ℂ)))) := by
  rw [cutoffV2_generator_eq_squeezeGenerator]

end Cvdv
