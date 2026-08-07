import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Complex.Norm
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Topology.Algebra.InfiniteSum.NatInt
import Mathlib.Topology.Algebra.InfiniteSum.Ring

/-!
# Coherent-state quadrature moments and the zero-point variance

Formalization of the coherent-state moment computations behind manuscript Eq. (sg-zpf): in a
coherent state the phase quadrature `φ = (a + a†)/√(2 ω_b)` has variance `1/(2 ω_b)`.

We work in the Fock-sequence representation.  The coherent amplitudes are
`cohAmp α n = exp (-‖α‖²/2) · αⁿ / √(n!)`, and the three Fock-sum moments are

* `aExpect_cohAmp` : `⟨a⟩ = α`,
* `nExpect_cohAmp` : `⟨n⟩ = ‖α‖²`,
* `aSqExpect_cohAmp` : `⟨a²⟩ = α²`.

The variance corollary `cohVariance` is the resulting pure-algebra identity
`⟨φ²⟩ - ⟨φ⟩² = 1/(2 ω)`, with `⟨φ²⟩ = (⟨a²⟩ + ⟨a†²⟩ + 2⟨n⟩ + 1)/(2 ω)` and
`⟨φ⟩ = (α + conj α)/√(2 ω)` substituted from the moment computations.
-/

namespace Cvdv

noncomputable section

/-- Fock amplitude of the coherent state with complex amplitude `alpha`:
`⟨n|α⟩ = exp (-‖α‖²/2) αⁿ / √(n!)`. -/
def cohAmp (alpha : ℂ) (n : ℕ) : ℂ :=
  Complex.exp (-((‖alpha‖ : ℂ) ^ 2) / 2) * alpha ^ n / (Real.sqrt n.factorial : ℝ)

/-- Fock-sum expectation of the annihilation operator `a` in the state `psi`. -/
def aExpect (psi : ℕ → ℂ) : ℂ :=
  ∑' n : ℕ, (starRingEnd ℂ) (psi n) * (Real.sqrt (n + 1) : ℝ) * psi (n + 1)

/-- Fock-sum expectation of the number operator `a† a` in the state `psi`. -/
def nExpect (psi : ℕ → ℂ) : ℝ :=
  ∑' n : ℕ, (n : ℝ) * ‖psi n‖ ^ 2

/-- Fock-sum expectation of `a²` in the state `psi`. -/
def aSqExpect (psi : ℕ → ℂ) : ℂ :=
  ∑' n : ℕ, (starRingEnd ℂ) (psi n) * (Real.sqrt ((n + 1) * (n + 2)) : ℝ) * psi (n + 2)

/-- The exponential series in `ℂ`, summed as a `tsum`. -/
lemma coh_tsum_exp_complex (x : ℂ) :
    ∑' n : ℕ, x ^ n / (n.factorial : ℂ) = Complex.exp x := by
  rw [Complex.exp_eq_exp_ℂ]
  exact (NormedSpace.expSeries_div_hasSum_exp x).tsum_eq

/-- The exponential series in `ℝ`, summed as a `tsum`. -/
lemma coh_tsum_exp_real (r : ℝ) :
    ∑' n : ℕ, r ^ n / (n.factorial : ℝ) = Real.exp r := by
  rw [Real.exp_eq_exp_ℝ]
  exact (NormedSpace.expSeries_div_hasSum_exp r).tsum_eq

/-- Conjugate of a coherent amplitude: the Gaussian prefactor and the square root are real. -/
lemma conj_cohAmp (alpha : ℂ) (n : ℕ) :
    (starRingEnd ℂ) (cohAmp alpha n)
      = Complex.exp (-((‖alpha‖ : ℂ) ^ 2) / 2) * (starRingEnd ℂ) alpha ^ n
          / (Real.sqrt (n.factorial : ℝ) : ℂ) := by
  simp only [cohAmp, map_div₀, map_mul, map_pow, map_neg, map_ofNat, Complex.conj_ofReal,
    ← Complex.exp_conj]

/-- Term-collapse identity for `⟨a⟩`: the square-root and factorial factors cancel, leaving one
term of the exponential series. -/
lemma cohAmp_term_a (alpha : ℂ) (n : ℕ) :
    (starRingEnd ℂ) (cohAmp alpha n) * (Real.sqrt (n + 1) : ℝ) * cohAmp alpha (n + 1)
      = Complex.exp (-((‖alpha‖ : ℂ) ^ 2)) * alpha
          * (((‖alpha‖ : ℂ) ^ 2) ^ n / (n.factorial : ℂ)) := by
  have hfacpos : (0 : ℝ) < (n.factorial : ℝ) := by exact_mod_cast n.factorial_pos
  have ht : ((Real.sqrt (n.factorial : ℝ) : ℝ) : ℂ) ≠ 0 := by
    exact_mod_cast (Real.sqrt_pos.mpr hfacpos).ne'
  have hs : ((Real.sqrt ((n : ℝ) + 1) : ℝ) : ℂ) ≠ 0 := by
    exact_mod_cast (Real.sqrt_pos.mpr (by positivity : (0 : ℝ) < (n : ℝ) + 1)).ne'
  have htt : ((Real.sqrt (n.factorial : ℝ) : ℝ) : ℂ) * ((Real.sqrt (n.factorial : ℝ) : ℝ) : ℂ)
      = (n.factorial : ℂ) := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt hfacpos.le]
    norm_cast
  have hfacR : Real.sqrt (((n + 1).factorial : ℝ))
      = Real.sqrt ((n : ℝ) + 1) * Real.sqrt ((n.factorial : ℝ)) := by
    rw [Nat.factorial_succ]
    push_cast
    exact Real.sqrt_mul (by positivity) _
  calc (starRingEnd ℂ) (cohAmp alpha n) * (Real.sqrt (n + 1) : ℝ) * cohAmp alpha (n + 1)
      = Complex.exp (-((‖alpha‖ : ℂ) ^ 2) / 2) * Complex.exp (-((‖alpha‖ : ℂ) ^ 2) / 2)
          * (((starRingEnd ℂ) alpha ^ n * alpha ^ n) * alpha) / (n.factorial : ℂ) := by
        rw [conj_cohAmp, ← htt]
        simp only [cohAmp, hfacR, Complex.ofReal_mul, pow_succ]
        field_simp
    _ = Complex.exp (-((‖alpha‖ : ℂ) ^ 2)) * alpha
          * (((‖alpha‖ : ℂ) ^ 2) ^ n / (n.factorial : ℂ)) := by
        rw [← Complex.exp_add, add_halves, ← mul_pow, Complex.conj_mul']
        ring

/-- Term-collapse identity for `⟨a²⟩`: the double square-root collapses the two-step factorial
ratio, leaving `α²` times one term of the exponential series. -/
lemma cohAmp_term_aSq (alpha : ℂ) (n : ℕ) :
    (starRingEnd ℂ) (cohAmp alpha n) * (Real.sqrt ((n + 1) * (n + 2)) : ℝ) * cohAmp alpha (n + 2)
      = Complex.exp (-((‖alpha‖ : ℂ) ^ 2)) * alpha ^ 2
          * (((‖alpha‖ : ℂ) ^ 2) ^ n / (n.factorial : ℂ)) := by
  have hfacpos : (0 : ℝ) < (n.factorial : ℝ) := by exact_mod_cast n.factorial_pos
  have ht : ((Real.sqrt (n.factorial : ℝ) : ℝ) : ℂ) ≠ 0 := by
    exact_mod_cast (Real.sqrt_pos.mpr hfacpos).ne'
  have hs1 : ((Real.sqrt ((n : ℝ) + 1) : ℝ) : ℂ) ≠ 0 := by
    exact_mod_cast (Real.sqrt_pos.mpr (by positivity : (0 : ℝ) < (n : ℝ) + 1)).ne'
  have hs2 : ((Real.sqrt ((n : ℝ) + 2) : ℝ) : ℂ) ≠ 0 := by
    exact_mod_cast (Real.sqrt_pos.mpr (by positivity : (0 : ℝ) < (n : ℝ) + 2)).ne'
  have htt : ((Real.sqrt (n.factorial : ℝ) : ℝ) : ℂ) * ((Real.sqrt (n.factorial : ℝ) : ℝ) : ℂ)
      = (n.factorial : ℂ) := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt hfacpos.le]
    norm_cast
  have h2 : ((n + 2).factorial : ℝ) = ((n : ℝ) + 2) * (((n : ℝ) + 1) * (n.factorial : ℝ)) := by
    rw [show n + 2 = n + 1 + 1 by omega, Nat.factorial_succ, Nat.factorial_succ]
    push_cast
    ring
  have hfac2R : Real.sqrt (((n + 2).factorial : ℝ))
      = Real.sqrt ((n : ℝ) + 2) * (Real.sqrt ((n : ℝ) + 1) * Real.sqrt ((n.factorial : ℝ))) := by
    rw [h2, Real.sqrt_mul (by positivity), Real.sqrt_mul (by positivity)]
  have hS12 : Real.sqrt (((n : ℝ) + 1) * ((n : ℝ) + 2))
      = Real.sqrt ((n : ℝ) + 1) * Real.sqrt ((n : ℝ) + 2) :=
    Real.sqrt_mul (by positivity) _
  calc (starRingEnd ℂ) (cohAmp alpha n) * (Real.sqrt ((n + 1) * (n + 2)) : ℝ)
        * cohAmp alpha (n + 2)
      = Complex.exp (-((‖alpha‖ : ℂ) ^ 2) / 2) * Complex.exp (-((‖alpha‖ : ℂ) ^ 2) / 2)
          * (((starRingEnd ℂ) alpha ^ n * alpha ^ n) * alpha ^ 2) / (n.factorial : ℂ) := by
        rw [conj_cohAmp, ← htt]
        simp only [cohAmp, hfac2R, hS12, Complex.ofReal_mul, pow_add]
        field_simp
    _ = Complex.exp (-((‖alpha‖ : ℂ) ^ 2)) * alpha ^ 2
          * (((‖alpha‖ : ℂ) ^ 2) ^ n / (n.factorial : ℂ)) := by
        rw [← Complex.exp_add, add_halves, ← mul_pow, Complex.conj_mul']
        ring

/-- **R1**: in a coherent state, `⟨a⟩ = α`. -/
theorem aExpect_cohAmp (alpha : ℂ) : aExpect (cohAmp alpha) = alpha := by
  have h : aExpect (cohAmp alpha)
      = ∑' n : ℕ, Complex.exp (-((‖alpha‖ : ℂ) ^ 2)) * alpha
          * (((‖alpha‖ : ℂ) ^ 2) ^ n / (n.factorial : ℂ)) := by
    simp only [aExpect]
    exact tsum_congr fun n => cohAmp_term_a alpha n
  rw [h, tsum_mul_left, coh_tsum_exp_complex, mul_right_comm, ← Complex.exp_add,
    neg_add_cancel, Complex.exp_zero, one_mul]

/-- **R3**: in a coherent state, `⟨a²⟩ = α²`. -/
theorem aSqExpect_cohAmp (alpha : ℂ) : aSqExpect (cohAmp alpha) = alpha ^ 2 := by
  have h : aSqExpect (cohAmp alpha)
      = ∑' n : ℕ, Complex.exp (-((‖alpha‖ : ℂ) ^ 2)) * alpha ^ 2
          * (((‖alpha‖ : ℂ) ^ 2) ^ n / (n.factorial : ℂ)) := by
    simp only [aSqExpect]
    exact tsum_congr fun n => cohAmp_term_aSq alpha n
  rw [h, tsum_mul_left, coh_tsum_exp_complex, mul_right_comm, ← Complex.exp_add,
    neg_add_cancel, Complex.exp_zero, one_mul]

/-- Squared norm of a coherent amplitude: the Poisson weight `exp (-‖α‖²) ‖α‖^(2n) / n!`. -/
lemma norm_cohAmp_sq (alpha : ℂ) (n : ℕ) :
    ‖cohAmp alpha n‖ ^ 2 = Real.exp (-(‖alpha‖ ^ 2)) * (‖alpha‖ ^ 2) ^ n / (n.factorial : ℝ) := by
  have harg : (-((‖alpha‖ : ℂ) ^ 2) / 2) = ((-(‖alpha‖ ^ 2) / 2 : ℝ) : ℂ) := by
    push_cast
    ring
  have hnorm : ‖cohAmp alpha n‖
      = Real.exp (-(‖alpha‖ ^ 2) / 2) * ‖alpha‖ ^ n / Real.sqrt (n.factorial : ℝ) := by
    simp only [cohAmp]
    rw [harg, norm_div, norm_mul, Complex.norm_exp, Complex.ofReal_re, norm_pow,
      Complex.norm_of_nonneg (Real.sqrt_nonneg _)]
  rw [hnorm, div_pow, mul_pow, Real.sq_sqrt (Nat.cast_nonneg _),
    pow_two (Real.exp (-(‖alpha‖ ^ 2) / 2)), ← Real.exp_add, add_halves, pow_right_comm]

/-- The `n`-th term of the number-operator sum in a coherent state. -/
def cohNTerm (r : ℝ) (n : ℕ) : ℝ :=
  n * (Real.exp (-r) * r ^ n / (n.factorial : ℝ))

lemma cohNTerm_zero (r : ℝ) : cohNTerm r 0 = 0 := by
  simp [cohNTerm]

/-- Shift identity: the `n`-weight cancels one factor of the factorial. -/
lemma cohNTerm_succ (r : ℝ) (n : ℕ) :
    cohNTerm r (n + 1) = Real.exp (-r) * r * (r ^ n / (n.factorial : ℝ)) := by
  have h1 : ((n : ℝ) + 1) ≠ 0 := by positivity
  have h2 : ((n.factorial : ℝ)) ≠ 0 := by exact_mod_cast n.factorial_ne_zero
  simp only [cohNTerm, Nat.factorial_succ, pow_succ]
  push_cast
  field_simp

lemma cohNTerm_succ_summable (r : ℝ) : Summable fun n : ℕ => cohNTerm r (n + 1) :=
  ((Real.summable_pow_div_factorial r).mul_left (Real.exp (-r) * r)).congr
    fun n => (cohNTerm_succ r n).symm

/-- **R2**: in a coherent state, `⟨n⟩ = ‖α‖²`. -/
theorem nExpect_cohAmp (alpha : ℂ) : nExpect (cohAmp alpha) = ‖alpha‖ ^ 2 := by
  have h : nExpect (cohAmp alpha) = ∑' n : ℕ, cohNTerm (‖alpha‖ ^ 2) n := by
    simp only [nExpect]
    refine tsum_congr fun n => ?_
    rw [norm_cohAmp_sq]
    simp only [cohNTerm]
  have h2 : ∑' n : ℕ, cohNTerm (‖alpha‖ ^ 2) (n + 1)
      = ∑' n : ℕ, Real.exp (-(‖alpha‖ ^ 2)) * ‖alpha‖ ^ 2
          * ((‖alpha‖ ^ 2) ^ n / (n.factorial : ℝ)) :=
    tsum_congr fun n => cohNTerm_succ _ n
  rw [h, tsum_eq_zero_add' (cohNTerm_succ_summable (‖alpha‖ ^ 2)), cohNTerm_zero, zero_add, h2,
    tsum_mul_left, coh_tsum_exp_real, mul_right_comm, ← Real.exp_add, neg_add_cancel,
    Real.exp_zero, one_mul]

/-- **R4** (manuscript Eq. (sg-zpf)): the coherent-state variance of the phase quadrature
`φ = (a + a†)/√(2 ω)` equals the zero-point value `1/(2 ω)`.  The first summand is `⟨φ²⟩` with
the moments `⟨a²⟩ = α²`, `⟨a†²⟩ = conj α ²`, `⟨a† a⟩ = ‖α‖²` substituted, the second is `⟨φ⟩²`. -/
theorem cohVariance (alpha : ℂ) {omega : ℝ} (homega : 0 < omega) :
    ((alpha ^ 2 + (starRingEnd ℂ) alpha ^ 2 + 2 * (‖alpha‖ : ℂ) ^ 2 + 1) / (2 * omega)).re
      - ((alpha + (starRingEnd ℂ) alpha) / (Real.sqrt (2 * omega) : ℝ)).re ^ 2
      = 1 / (2 * omega) := by
  have h2 : (0 : ℝ) < 2 * omega := by linarith
  have hnum : alpha ^ 2 + (starRingEnd ℂ) alpha ^ 2 + 2 * (‖alpha‖ : ℂ) ^ 2 + 1
      = ((4 * alpha.re ^ 2 + 1 : ℝ) : ℂ) := by
    have hsq : alpha ^ 2 + (starRingEnd ℂ) alpha ^ 2 + 2 * (‖alpha‖ : ℂ) ^ 2
        = (alpha + (starRingEnd ℂ) alpha) ^ 2 := by
      rw [← Complex.conj_mul']
      ring
    rw [hsq, Complex.add_conj]
    push_cast
    ring
  have hden : (2 : ℂ) * (omega : ℂ) = ((2 * omega : ℝ) : ℂ) := by
    push_cast
    ring
  rw [hnum, Complex.add_conj, hden, ← Complex.ofReal_div, ← Complex.ofReal_div,
    Complex.ofReal_re, Complex.ofReal_re, div_pow, Real.sq_sqrt h2.le, div_sub_div_same]
  have hone : 4 * alpha.re ^ 2 + 1 - (2 * alpha.re) ^ 2 = 1 := by ring
  rw [hone]

end

end Cvdv
