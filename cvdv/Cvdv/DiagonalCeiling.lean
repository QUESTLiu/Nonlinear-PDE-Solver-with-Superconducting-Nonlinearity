import Mathlib.Analysis.SpecialFunctions.Complex.Arg
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Data.Complex.BigOperators
import Mathlib.Data.Matrix.Mul

/-!
# Fidelity ceiling of number-diagonal unitaries

This file formalizes manuscript Eq. (df-diagonal-ceiling). A number-diagonal unitary acts on
Fock coefficients as multiplication by unit phases `exp (θ n * I)`, so the overlap between a
state with coefficients `c` and the phased state with coefficients `d` is
`∑ n, conj (c n) * exp (θ n * I) * d n` over a finite (truncated) index set.

Main results:
* `diag_overlap_le`: for every phase profile `θ` the modulus of the overlap is at most
  `∑ n, ‖c n‖ * ‖d n‖`.
* `exists_diag_overlap_eq`: the bound is attained by the phase choice
  `θ n = -arg (conj (c n) * d n)`.
* `isGreatest_diag_overlap_sq`: consequently `(∑ n, ‖c n‖ * ‖d n‖) ^ 2` is the greatest value
  of the squared overlap over all phase profiles, the fidelity ceiling of Eq.
  (df-diagonal-ceiling).

All results are fully proved; there is no `sorry`.
-/

open Finset
open scoped ComplexConjugate

namespace Cvdv

variable {ι : Type*} [Fintype ι] (c d : ι → ℂ)

section NumberDiagonalUnitary

variable [DecidableEq ι]

/-- A finite number-diagonal unitary: a diagonal matrix whose entries all have
unit modulus. -/
def IsNumberDiagonalUnitary (U : Matrix ι ι ℂ) : Prop :=
  ∃ z : ι → ℂ, (∀ n, ‖z n‖ = 1) ∧ U = Matrix.diagonal z

omit [Fintype ι] in
/-- Every number-diagonal unitary is a real phase profile on the number basis. -/
theorem IsNumberDiagonalUnitary.exists_phaseProfile {U : Matrix ι ι ℂ}
    (hU : IsNumberDiagonalUnitary U) :
    ∃ θ : ι → ℝ, U = Matrix.diagonal fun n => Complex.exp (θ n * Complex.I) := by
  rcases hU with ⟨z, hz, rfl⟩
  refine ⟨fun n => (z n).arg, ?_⟩
  congr 1
  funext n
  rw [← Complex.norm_mul_exp_arg_mul_I (z n), hz n]
  simp

/-- Consequently a number-diagonal unitary acts coefficientwise by its phase
profile. -/
theorem IsNumberDiagonalUnitary.exists_phaseProfile_mulVec {U : Matrix ι ι ℂ}
    (hU : IsNumberDiagonalUnitary U) :
    ∃ θ : ι → ℝ, ∀ v n,
      Matrix.mulVec U v n = Complex.exp (θ n * Complex.I) * v n := by
  rcases hU.exists_phaseProfile with ⟨θ, rfl⟩
  exact ⟨θ, fun v n => Matrix.mulVec_diagonal _ _ _⟩

omit [Fintype ι] in
/-- Conversely, every real phase profile defines a number-diagonal unitary. -/
theorem isNumberDiagonalUnitary_diagonal_phaseProfile (θ : ι → ℝ) :
    IsNumberDiagonalUnitary
      (Matrix.diagonal fun n => Complex.exp (θ n * Complex.I)) := by
  refine ⟨fun n => Complex.exp (θ n * Complex.I), ?_, rfl⟩
  intro n
  exact Complex.norm_exp_ofReal_mul_I (θ n)

end NumberDiagonalUnitary

/-- Multiplying a complex number by the unit phase cancelling its argument yields its modulus,
viewed as a complex number. -/
lemma mul_exp_neg_arg_mul_I (z : ℂ) :
    z * Complex.exp ((-z.arg : ℝ) * Complex.I) = (‖z‖ : ℂ) :=
  calc z * Complex.exp ((-z.arg : ℝ) * Complex.I)
      = (‖z‖ : ℂ) * Complex.exp ((z.arg : ℝ) * Complex.I) *
        Complex.exp ((-z.arg : ℝ) * Complex.I) := by
        rw [Complex.norm_mul_exp_arg_mul_I]
    _ = (‖z‖ : ℂ) * Complex.exp ((z.arg : ℝ) * Complex.I + (-z.arg : ℝ) * Complex.I) := by
        rw [mul_assoc, Complex.exp_add]
    _ = (‖z‖ : ℂ) := by
        rw [show ((z.arg : ℝ) : ℂ) * Complex.I + ((-z.arg : ℝ) : ℂ) * Complex.I = 0 by
          push_cast; ring]
        rw [Complex.exp_zero, mul_one]

/-- Upper bound on the overlap achievable by a number-diagonal unitary: the modulus of
`∑ n, conj (c n) * exp (θ n * I) * d n` is at most the sum of the products of the
amplitude moduli. -/
theorem diag_overlap_le (θ : ι → ℝ) :
    ‖∑ n, (starRingEnd ℂ) (c n) * Complex.exp (θ n * Complex.I) * d n‖ ≤
      ∑ n, ‖c n‖ * ‖d n‖ := by
  refine le_trans (norm_sum_le _ _) (le_of_eq (Finset.sum_congr rfl fun n _ => ?_))
  rw [norm_mul, norm_mul, Complex.norm_exp_ofReal_mul_I, mul_one, Complex.norm_conj]

/-- The phase-profile bound therefore applies to every number-diagonal
unitary acting on the coefficient vector `d`. -/
theorem numberDiagonalUnitary_overlap_le [DecidableEq ι]
    (U : Matrix ι ι ℂ) (hU : IsNumberDiagonalUnitary U) :
    ‖∑ n, (starRingEnd ℂ) (c n) * Matrix.mulVec U d n‖ ≤
      ∑ n, ‖c n‖ * ‖d n‖ := by
  rcases hU.exists_phaseProfile_mulVec with ⟨θ, hθ⟩
  simpa only [hθ, mul_assoc] using diag_overlap_le c d θ

/-- The upper bound of `diag_overlap_le` is attained: the phase choice
`θ n = -arg (conj (c n) * d n)` aligns every term of the overlap sum along the positive
real axis. -/
theorem exists_diag_overlap_eq :
    ∃ θ : ι → ℝ,
      ‖∑ n, (starRingEnd ℂ) (c n) * Complex.exp (θ n * Complex.I) * d n‖ =
        ∑ n, ‖c n‖ * ‖d n‖ := by
  refine ⟨fun n => -((starRingEnd ℂ) (c n) * d n).arg, ?_⟩
  show ‖∑ n, (starRingEnd ℂ) (c n) *
      Complex.exp ((-((starRingEnd ℂ) (c n) * d n).arg : ℝ) * Complex.I) * d n‖ =
    ∑ n, ‖c n‖ * ‖d n‖
  have hsum : ∑ n, (starRingEnd ℂ) (c n) *
      Complex.exp ((-((starRingEnd ℂ) (c n) * d n).arg : ℝ) * Complex.I) * d n =
      ((∑ n, ‖c n‖ * ‖d n‖ : ℝ) : ℂ) := by
    rw [Complex.ofReal_sum]
    refine Finset.sum_congr rfl fun n _ => ?_
    rw [mul_right_comm, mul_exp_neg_arg_mul_I, norm_mul, Complex.norm_conj]
  rw [hsum]
  exact Complex.norm_of_nonneg
    (Finset.sum_nonneg fun n _ => mul_nonneg (norm_nonneg _) (norm_nonneg _))

/-- Fidelity ceiling of number-diagonal unitaries, manuscript Eq. (df-diagonal-ceiling):
`(∑ n, ‖c n‖ * ‖d n‖) ^ 2` is the greatest value of the squared overlap over all phase
profiles `θ`. -/
theorem isGreatest_diag_overlap_sq :
    IsGreatest
      (Set.range fun θ : ι → ℝ =>
        ‖∑ n, (starRingEnd ℂ) (c n) * Complex.exp (θ n * Complex.I) * d n‖ ^ 2)
      ((∑ n, ‖c n‖ * ‖d n‖) ^ 2) := by
  constructor
  · obtain ⟨θ, hθ⟩ := exists_diag_overlap_eq c d
    exact ⟨θ, by simp only [hθ]⟩
  · rintro x ⟨θ, rfl⟩
    exact pow_le_pow_left₀ (norm_nonneg _) (diag_overlap_le c d θ) 2

end Cvdv
