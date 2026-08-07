import Cvdv.CircuitScaling
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.Module

/-!
# Duffing Hamiltonian in ladder operators

Algebraic reduction of manuscript Eq. (eq:df-ham) with `ℏ = 1`.
-/

namespace Cvdv

section DuffingLadder

variable {R : Type*} [Ring R] [Algebra ℂ R]

/-- Position quadrature used in Eq. (eq:df-ham). -/
noncomputable def duffingPosition (omega0 : ℝ) (a adag : R) : R :=
  (((Real.sqrt (2 * omega0) : ℝ) : ℂ))⁻¹ • (a + adag)

/-- Momentum quadrature used in Eq. (eq:df-ham). -/
noncomputable def duffingMomentum (omega0 : ℝ) (a adag : R) : R :=
  (Complex.I * (Real.sqrt (omega0 / 2) : ℝ)) • (adag - a)

private lemma position_coeff_sq {omega0 : ℝ} (hw : 0 < omega0) :
    ((((Real.sqrt (2 * omega0) : ℝ) : ℂ))⁻¹) ^ 2 =
      (((1 / (2 * omega0) : ℝ) : ℂ)) := by
  norm_cast
  rw [inv_pow, Real.sq_sqrt (by positivity)]
  rw [one_div]

private lemma momentum_coeff_sq {omega0 : ℝ} (hw : 0 < omega0) :
    (Complex.I * (Real.sqrt (omega0 / 2) : ℝ)) ^ 2 =
      (((-(omega0 / 2) : ℝ) : ℂ)) := by
  rw [mul_pow, Complex.I_sq, neg_one_mul, ← Complex.ofReal_pow,
    Real.sq_sqrt (by positivity)]
  push_cast
  ring

private lemma position_coeff_four {omega0 : ℝ} (hw : 0 < omega0) :
    ((((Real.sqrt (2 * omega0) : ℝ) : ℂ))⁻¹) ^ 4 =
      (((1 / (4 * omega0 ^ 2) : ℝ) : ℂ)) := by
  calc
    ((((Real.sqrt (2 * omega0) : ℝ) : ℂ))⁻¹) ^ 4 =
        (((((Real.sqrt (2 * omega0) : ℝ) : ℂ))⁻¹) ^ 2) ^ 2 := by ring
    _ = ((((1 / (2 * omega0) : ℝ) : ℂ))) ^ 2 := by rw [position_coeff_sq hw]
    _ = (((1 / (4 * omega0 ^ 2) : ℝ) : ℂ)) := by
      push_cast
      field_simp [ne_of_gt hw]
      ring

private lemma harmonic_core (a adag : R) (hccr : a * adag = adag * a + 1) :
    (a + adag) ^ 2 - (adag - a) ^ 2 =
      (4 : ℂ) • (adag * a) + (2 : ℂ) • (1 : R) := by
  rw [Algebra.smul_def, Algebra.smul_def, map_ofNat, map_ofNat, mul_one]
  have expand :
      (a + adag) ^ 2 - (adag - a) ^ 2 = 2 * (a * adag) + 2 * (adag * a) := by
    noncomm_ring
  rw [expand, hccr]
  noncomm_ring
  simp

/-- Harmonic part of manuscript Eq. (eq:df-ham): with `ℏ = 1`, the quadrature
Hamiltonian is `ω0 (a† a + 1/2)`. -/
theorem duffing_harmonic_ladder (a adag : R) (hccr : a * adag = adag * a + 1)
    (omega0 : ℝ) (hw : 0 < omega0) :
    (1 / 2 : ℂ) • (duffingMomentum omega0 a adag) ^ 2 +
        (((omega0 ^ 2 / 2 : ℝ) : ℂ)) • (duffingPosition omega0 a adag) ^ 2 =
      (omega0 : ℂ) • (adag * a + (1 / 2 : ℂ) • (1 : R)) := by
  simp only [duffingMomentum, duffingPosition, smul_pow, smul_smul]
  rw [momentum_coeff_sq hw, position_coeff_sq hw]
  have hp :
      (1 / 2 : ℂ) * (((-(omega0 / 2) : ℝ) : ℂ)) = (((-omega0 / 4 : ℝ) : ℂ)) := by
    push_cast
    ring
  have hq :
      (((omega0 ^ 2 / 2 : ℝ) : ℂ)) * (((1 / (2 * omega0) : ℝ) : ℂ)) =
        (((omega0 / 4 : ℝ) : ℂ)) := by
    push_cast
    field_simp [ne_of_gt hw]
    ring
  rw [hp, hq]
  calc
    (((-omega0 / 4 : ℝ) : ℂ)) • (adag - a) ^ 2 +
          (((omega0 / 4 : ℝ) : ℂ)) • (a + adag) ^ 2 =
        (((omega0 / 4 : ℝ) : ℂ)) • ((a + adag) ^ 2 - (adag - a) ^ 2) := by
          module
    _ = (((omega0 / 4 : ℝ) : ℂ)) •
          ((4 : ℂ) • (adag * a) + (2 : ℂ) • (1 : R)) := by
          rw [harmonic_core a adag hccr]
    _ = (omega0 : ℂ) • (adag * a + (1 / 2 : ℂ) • (1 : R)) := by
          module

/-- Quartic part of manuscript Eq. (eq:df-ham): substituting
`q = (a + a†) / √(2 ω0)` turns `(β/4) q⁴` into
`(β/(16 ω0²)) (a + a†)⁴`. -/
theorem duffing_quartic_ladder (a adag : R) (omega0 beta : ℝ) (hw : 0 < omega0) :
    (((beta / 4 : ℝ) : ℂ)) • (duffingPosition omega0 a adag) ^ 4 =
      (((beta / (16 * omega0 ^ 2) : ℝ) : ℂ)) • (a + adag) ^ 4 := by
  simp only [duffingPosition, smul_pow, smul_smul]
  rw [position_coeff_four hw]
  have hcoeff :
      (((beta / 4 : ℝ) : ℂ)) * (((1 / (4 * omega0 ^ 2) : ℝ) : ℂ)) =
        (((beta / (16 * omega0 ^ 2) : ℝ) : ℂ)) := by
    push_cast
    field_simp [ne_of_gt hw]
    ring
  rw [hcoeff]

/-- Complete algebraic reduction in manuscript Eq. (eq:df-ham), in units `ℏ = 1`. -/
theorem duffing_ladder_reduction (a adag : R) (hccr : a * adag = adag * a + 1)
    (omega0 beta : ℝ) (hw : 0 < omega0) :
    (1 / 2 : ℂ) • (duffingMomentum omega0 a adag) ^ 2 +
          (((omega0 ^ 2 / 2 : ℝ) : ℂ)) • (duffingPosition omega0 a adag) ^ 2 +
        (((beta / 4 : ℝ) : ℂ)) • (duffingPosition omega0 a adag) ^ 4 =
      (omega0 : ℂ) • (adag * a + (1 / 2 : ℂ) • (1 : R)) +
        (((beta / (16 * omega0 ^ 2) : ℝ) : ℂ)) • (a + adag) ^ 4 := by
  rw [duffing_harmonic_ladder a adag hccr omega0 hw,
    duffing_quartic_ladder a adag omega0 beta hw]

end DuffingLadder

end Cvdv
