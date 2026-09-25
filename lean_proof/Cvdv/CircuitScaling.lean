import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.FieldSimp
import Mathlib.Data.Complex.Basic
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Analysis.Real.Sqrt

/-!
# Quadrature commutation relation and the circuit-to-lattice rescaling

Formalization of manuscript Eq. (eq:sg-quadratures) and of the rescaling that carries
Eq. (eq:jj-array) into Eq. (eq:sg-lattice).

`quadrature_ccr` is Eq. (eq:sg-quadratures).  In a ring carrying the canonical commutation
relation `x * y = y * x + 1` (`x` the annihilation operator `a`, `y` the creation operator
`a†`), the quadratures `φ = (a + a†)/√(2ω)` and `Π = i√(ω/2)(a† - a)` satisfy
`φΠ - Πφ = i` for every basis frequency `ω > 0`.  The proof separates the ring identity
`(x + y)(y - x) - (y - x)(x + y) = 2` from the scalar identity
`(√(2ω))⁻¹ · i√(ω/2) · 2 = i` (`quadrature_coeff`), which reduces to `√(2ω) = 2√(ω/2)`.

`jj_rescale` is the coefficient bookkeeping behind the identification
`H_JJ = 8E_C · H_SG` with `E_SG = E_J/(8E_C)` and `κ = E_L/(8E_C)`.  In an abstract
`ℂ`-module, `n2` stands for the charging sum `∑ n̂_j²`, `v` for the Josephson cosine sum,
and `w` for the gradient sum, so the operator content is untouched and only the scalar
coefficients move.
-/

namespace Cvdv

section Quadratures

variable {R : Type*} [Ring R] [Algebra ℂ R]

/-- Scalar bookkeeping for Eq. (eq:sg-quadratures): the coefficient collected from the two
quadrature prefactors and the ring commutator equals `i`, since `√(2ω) = 2√(ω/2)`. -/
theorem quadrature_coeff {omega : ℝ} (hw : 0 < omega) :
    (((Real.sqrt (2 * omega) : ℝ) : ℂ))⁻¹ * (Complex.I * (Real.sqrt (omega / 2) : ℝ)) * 2
      = Complex.I := by
  have h4 : Real.sqrt (2 * omega) = 2 * Real.sqrt (omega / 2) := by
    rw [show (2 : ℝ) * omega = 2 ^ 2 * (omega / 2) by ring,
      Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2 ^ 2) (omega / 2),
      Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 2)]
  have hpos : 0 < Real.sqrt (omega / 2) := Real.sqrt_pos.mpr (half_pos hw)
  have hne : ((Real.sqrt (omega / 2) : ℝ) : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hpos.ne'
  rw [h4]
  push_cast
  field_simp

/-- Eq. (eq:sg-quadratures): for every basis frequency `ω > 0`, the quadrature pair
`φ = (x + y)/√(2ω)` and `Π = i√(ω/2)(y - x)` built from a canonical pair
`x * y = y * x + 1` satisfies the canonical relation `φΠ - Πφ = i`. -/
theorem quadrature_ccr (x y : R) (h : x * y = y * x + 1) (omega : ℝ) (hw : 0 < omega) :
    ((((Real.sqrt (2 * omega) : ℝ) : ℂ))⁻¹ • (x + y)) *
          ((Complex.I * (Real.sqrt (omega / 2) : ℝ)) • (y - x)) -
        ((Complex.I * (Real.sqrt (omega / 2) : ℝ)) • (y - x)) *
          ((((Real.sqrt (2 * omega) : ℝ) : ℂ))⁻¹ • (x + y)) =
      algebraMap ℂ R Complex.I := by
  have hcore : (x + y) * (y - x) - (y - x) * (x + y) = 2 := by
    have expand : (x + y) * (y - x) - (y - x) * (x + y) = 2 * (x * y) - 2 * (y * x) := by
      noncomm_ring
    rw [expand, h, mul_add, mul_one, add_sub_cancel_left]
  rw [smul_mul_smul_comm, smul_mul_smul_comm,
    mul_comm (Complex.I * ((Real.sqrt (omega / 2) : ℝ) : ℂ))
      ((((Real.sqrt (2 * omega) : ℝ) : ℂ))⁻¹),
    ← smul_sub, hcore,
    show (2 : R) = algebraMap ℂ R 2 from (map_ofNat (algebraMap ℂ R) 2).symm,
    Algebra.smul_def, ← map_mul, quadrature_coeff hw]

/-- Multisite form of Eq. (eq:sg-quadratures).  For an indexed CCR family with a common
basis frequency `ω > 0`, the quadratures at sites `j` and `k` satisfy
`[φ_j, Π_k] = i δ_jk`.  The three off-diagonal hypotheses state that annihilators,
creators, and an annihilator and creator at distinct sites commute. -/
theorem quadrature_ccr_multisite {ι : Type*} [DecidableEq ι]
    (a adag : ι → R)
    (hdiag : ∀ j, a j * adag j = adag j * a j + 1)
    (hann : ∀ j k, j ≠ k → a j * a k = a k * a j)
    (hcre : ∀ j k, j ≠ k → adag j * adag k = adag k * adag j)
    (hcross : ∀ j k, j ≠ k → a j * adag k = adag k * a j)
    (omega : ℝ) (hw : 0 < omega) (j k : ι) :
    ((((Real.sqrt (2 * omega) : ℝ) : ℂ))⁻¹ • (a j + adag j)) *
          ((Complex.I * (Real.sqrt (omega / 2) : ℝ)) • (adag k - a k)) -
        ((Complex.I * (Real.sqrt (omega / 2) : ℝ)) • (adag k - a k)) *
          ((((Real.sqrt (2 * omega) : ℝ) : ℂ))⁻¹ • (a j + adag j)) =
      if j = k then algebraMap ℂ R Complex.I else 0 := by
  by_cases hjk : j = k
  · subst k
    simpa using quadrature_ccr (a j) (adag j) (hdiag j) omega hw
  · rw [if_neg hjk]
    have hcore :
        (a j + adag j) * (adag k - a k) - (adag k - a k) * (a j + adag j) = 0 := by
      have expand :
          (a j + adag j) * (adag k - a k) - (adag k - a k) * (a j + adag j) =
            (a j * adag k - adag k * a j) - (a j * a k - a k * a j) +
              (adag j * adag k - adag k * adag j) - (adag j * a k - a k * adag j) := by
        noncomm_ring
      rw [expand, hcross j k hjk, hann j k hjk, hcre j k hjk,
        hcross k j (fun hkj ↦ hjk hkj.symm)]
      noncomm_ring
    rw [smul_mul_smul_comm, smul_mul_smul_comm,
      mul_comm (Complex.I * ((Real.sqrt (omega / 2) : ℝ) : ℂ))
        ((((Real.sqrt (2 * omega) : ℝ) : ℂ))⁻¹),
      ← smul_sub, hcore, smul_zero]

end Quadratures

section Rescaling

/-- Eq. (eq:jj-array) to Eq. (eq:sg-lattice): the physical array Hamiltonian
`4E_C n2 + E_J v + (E_L/2) w` equals `8E_C` times the dimensionless sine-Gordon form
`(1/2) n2 + E_SG v + (κ/2) w` with `E_SG = E_J/(8E_C)` and `κ = E_L/(8E_C)`.  Here `n2`
stands for the charging sum, `v` for the Josephson cosine sum, and `w` for the gradient
sum, and only the scalar coefficients are rearranged. -/
theorem jj_rescale {M : Type*} [AddCommGroup M] [Module ℂ M] (n2 v w : M)
    (EC EJ EL : ℂ) (hEC : EC ≠ 0) :
    (4 * EC) • n2 + EJ • v + (EL / 2) • w
      = (8 * EC) • ((1 / 2 : ℂ) • n2 + (EJ / (8 * EC)) • v + (EL / (16 * EC)) • w) := by
  rw [smul_add, smul_add, smul_smul, smul_smul, smul_smul,
    show (8 * EC) * (1 / 2 : ℂ) = 4 * EC by ring,
    show (8 * EC) * (EJ / (8 * EC)) = EJ by field_simp,
    show (8 * EC) * (EL / (16 * EC)) = EL / 2 by field_simp; ring]

end Rescaling

end Cvdv
