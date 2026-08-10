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

Formalization of the coherent-state moment computations behind manuscript Eq. `eq:sg-zpf`: in a
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

/-- **R4** (manuscript Eq. `eq:sg-zpf`): the coherent-state variance of the phase quadrature
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

/-!
### Operator-level quadrature moments (manuscript Secs. `subsec:encoding`, `subsec:sg-quantization`)

Upgrade of the scalar identity `cohVariance` to an operator-level derivation.  The phase
quadrature `φ = (a + a†)/√(2 ω_b)` acts on Fock-amplitude sequences through `phiApply`, and the
moments `⟨φ⟩`, `⟨φ²⟩` are genuine Fock sums (`phiExpect`, `phiSqExpect`).  Evaluated on the
coherent amplitudes `cohAmp ξ` they give `⟨φ⟩ = (ξ + conj ξ)/√(2 ω)` (`phiExpect_cohAmp`) and
`⟨φ²⟩ = (ξ² + conj ξ ² + 2 ‖ξ‖² + 1)/(2 ω)` (`phiSqExpect_cohAmp`), whence the zero-point
variance `⟨φ²⟩ - ⟨φ⟩² = 1/(2 ω)` of manuscript Eq. `eq:sg-zpf` (`cohPhiVariance`) with no scalar
substitution hypotheses left.
-/

namespace Cvdv

noncomputable section

/-- Action of the phase quadrature `φ = (a + a†)/√(2 ω)` on a Fock-amplitude sequence:
`(φ ψ)ₙ = (√(n+1) ψₙ₊₁ + √n ψₙ₋₁)/√(2 ω)`.  At `n = 0` the ladder-down coefficient `√0 = 0`
kills the out-of-range `ψ₋₁` term, so the `ℕ`-truncated index `n - 1` is harmless. -/
def phiApply (omega : ℝ) (psi : ℕ → ℂ) (n : ℕ) : ℂ :=
  ((Real.sqrt (n + 1) : ℝ) * psi (n + 1) + (Real.sqrt n : ℝ) * psi (n - 1))
    / (Real.sqrt (2 * omega) : ℝ)

/-- Fock-sum expectation `⟨φ⟩ = ∑ conj ψₙ · (φψ)ₙ` of the phase quadrature. -/
def phiExpect (omega : ℝ) (psi : ℕ → ℂ) : ℂ :=
  ∑' n : ℕ, (starRingEnd ℂ) (psi n) * phiApply omega psi n

/-- Fock-sum expectation `⟨φ²⟩ = ∑ conj ψₙ · (φ(φψ))ₙ` of the squared phase quadrature. -/
def phiSqExpect (omega : ℝ) (psi : ℕ → ℂ) : ℂ :=
  ∑' n : ℕ, (starRingEnd ℂ) (psi n) * phiApply omega (phiApply omega psi) n

/-- Ladder-down (`a†`-pattern) block of a quadrature sum: `conj ψₙ √n ψₙ₋₁`. -/
def phiDownTerm (psi : ℕ → ℂ) (n : ℕ) : ℂ :=
  (starRingEnd ℂ) (psi n) * (Real.sqrt n : ℝ) * psi (n - 1)

/-- Double-ladder-up (`a²`-pattern) block of the `⟨φ²⟩` sum, with the square root kept in the
factored form produced by two quadrature applications. -/
def phiUp2Term (psi : ℕ → ℂ) (n : ℕ) : ℂ :=
  (starRingEnd ℂ) (psi n) * ((Real.sqrt (n + 1) * Real.sqrt (n + 2) : ℝ) : ℂ) * psi (n + 2)

/-- Diagonal (`a a† + a† a`) block of the `⟨φ²⟩` sum: `conj ψₙ (2n+1) ψₙ`. -/
def phiDiagTerm (psi : ℕ → ℂ) (n : ℕ) : ℂ :=
  (starRingEnd ℂ) (psi n) * ((2 * n + 1 : ℝ) : ℂ) * psi n

/-- Double-ladder-down (`a†²`-pattern) block of the `⟨φ²⟩` sum.  The real subtraction `n - 1`
inside the square root makes the coefficient vanish at `n = 0, 1`, so the `ℕ`-truncated index
`n - 2` is harmless. -/
def phiDown2Term (psi : ℕ → ℂ) (n : ℕ) : ℂ :=
  (starRingEnd ℂ) (psi n) * ((Real.sqrt (n - 1) * Real.sqrt n : ℝ) : ℂ) * psi (n - 2)

/-- The complex exponential series at the real point `‖ξ‖²` is summable. -/
lemma cohExpSeries_summable (xi : ℂ) :
    Summable fun n : ℕ => ((‖xi‖ : ℂ) ^ 2) ^ n / (n.factorial : ℂ) := by
  refine (Complex.summable_ofReal.mpr (Real.summable_pow_div_factorial (‖xi‖ ^ 2))).congr
    fun n => ?_
  push_cast
  ring

/-- The `n = 0` ladder-down term vanishes because `√0 = 0`. -/
lemma phiDownTerm_zero (psi : ℕ → ℂ) : phiDownTerm psi 0 = 0 := by
  simp [phiDownTerm]

/-- Term collapse for the `a†` block: one step down the ladder against `cohAmp` produces
`conj ξ` times one term of the exponential series (the conjugate of `cohAmp_term_a`). -/
lemma phiDownTerm_cohAmp_succ (xi : ℂ) (n : ℕ) :
    phiDownTerm (cohAmp xi) (n + 1)
      = Complex.exp (-((‖xi‖ : ℂ) ^ 2)) * (starRingEnd ℂ) xi
          * (((‖xi‖ : ℂ) ^ 2) ^ n / (n.factorial : ℂ)) := by
  have h := congrArg (starRingEnd ℂ) (cohAmp_term_a xi n)
  simp only [map_mul, map_div₀, map_pow, map_neg, Complex.conj_conj, Complex.conj_ofReal,
    map_natCast, ← Complex.exp_conj] at h
  have hcast : ((n + 1 : ℕ) : ℝ) = (n : ℝ) + 1 := by push_cast; ring
  simp only [phiDownTerm, Nat.add_sub_cancel, hcast]
  linear_combination h

/-- Summability of the shifted `a†`-block terms in a coherent state. -/
lemma phiDownTerm_succ_summable (xi : ℂ) :
    Summable fun n : ℕ => phiDownTerm (cohAmp xi) (n + 1) :=
  ((cohExpSeries_summable xi).mul_left
      (Complex.exp (-((‖xi‖ : ℂ) ^ 2)) * (starRingEnd ℂ) xi)).congr
    fun n => (phiDownTerm_cohAmp_succ xi n).symm

/-- Summability of the `a†`-block terms in a coherent state. -/
lemma phiDownTerm_summable (xi : ℂ) :
    Summable fun n : ℕ => phiDownTerm (cohAmp xi) n :=
  (summable_nat_add_iff (f := fun n => phiDownTerm (cohAmp xi) n) 1).mp
    (phiDownTerm_succ_summable xi)

/-- The `a†` block sums to `conj ξ` in a coherent state:
`∑ conj ψₙ √n ψₙ₋₁ = conj ξ` for `ψ = cohAmp ξ`. -/
lemma phiDownTerm_tsum_cohAmp (xi : ℂ) :
    ∑' n : ℕ, phiDownTerm (cohAmp xi) n = (starRingEnd ℂ) xi := by
  rw [tsum_eq_zero_add' (phiDownTerm_succ_summable xi), phiDownTerm_zero, zero_add,
    tsum_congr fun n => phiDownTerm_cohAmp_succ xi n, tsum_mul_left, coh_tsum_exp_complex,
    mul_right_comm, ← Complex.exp_add, neg_add_cancel, Complex.exp_zero, one_mul]

/-- Sesquilinear expansion of a `⟨φ⟩` summand into the ladder-up (`a`) and ladder-down (`a†`)
blocks, divided by `√(2 ω)`. -/
lemma phiExpect_term (omega : ℝ) (psi : ℕ → ℂ) (n : ℕ) :
    (starRingEnd ℂ) (psi n) * phiApply omega psi n
      = ((starRingEnd ℂ) (psi n) * (Real.sqrt (n + 1) : ℝ) * psi (n + 1)
          + phiDownTerm psi n) / (Real.sqrt (2 * omega) : ℝ) := by
  simp only [phiApply, phiDownTerm]
  ring

/-- **Operator-level `⟨φ⟩`** (manuscript Secs. `subsec:encoding`, `subsec:sg-quantization`):
in the coherent state `|ξ⟩` the phase quadrature has mean `⟨φ⟩ = (ξ + conj ξ)/√(2 ω)`. -/
theorem phiExpect_cohAmp (xi : ℂ) {omega : ℝ} (_homega : 0 < omega) :
    phiExpect omega (cohAmp xi)
      = (xi + (starRingEnd ℂ) xi) / (Real.sqrt (2 * omega) : ℝ) := by
  have hup : Summable fun n : ℕ =>
      (starRingEnd ℂ) (cohAmp xi n) * (Real.sqrt (n + 1) : ℝ) * cohAmp xi (n + 1) :=
    ((cohExpSeries_summable xi).mul_left (Complex.exp (-((‖xi‖ : ℂ) ^ 2)) * xi)).congr
      fun n => (cohAmp_term_a xi n).symm
  calc phiExpect omega (cohAmp xi)
      = ∑' n : ℕ, ((starRingEnd ℂ) (cohAmp xi n) * (Real.sqrt (n + 1) : ℝ) * cohAmp xi (n + 1)
          + phiDownTerm (cohAmp xi) n) / (Real.sqrt (2 * omega) : ℝ) :=
        tsum_congr fun n => phiExpect_term omega (cohAmp xi) n
    _ = (∑' n : ℕ, ((starRingEnd ℂ) (cohAmp xi n) * (Real.sqrt (n + 1) : ℝ) * cohAmp xi (n + 1)
          + phiDownTerm (cohAmp xi) n)) / (Real.sqrt (2 * omega) : ℝ) := tsum_div_const
    _ = ((∑' n : ℕ, (starRingEnd ℂ) (cohAmp xi n) * (Real.sqrt (n + 1) : ℝ) * cohAmp xi (n + 1))
          + ∑' n : ℕ, phiDownTerm (cohAmp xi) n) / (Real.sqrt (2 * omega) : ℝ) := by
        rw [Summable.tsum_add hup (phiDownTerm_summable xi)]
    _ = (xi + (starRingEnd ℂ) xi) / (Real.sqrt (2 * omega) : ℝ) := by
        rw [show (∑' n : ℕ, (starRingEnd ℂ) (cohAmp xi n) * (Real.sqrt (n + 1) : ℝ)
            * cohAmp xi (n + 1)) = xi from aExpect_cohAmp xi, phiDownTerm_tsum_cohAmp]

/-- The factored square root of `phiUp2Term` recombines into the `aSqExpect` summand. -/
lemma phiUp2Term_eq (psi : ℕ → ℂ) (n : ℕ) :
    phiUp2Term psi n
      = (starRingEnd ℂ) (psi n) * (Real.sqrt ((n + 1) * (n + 2)) : ℝ) * psi (n + 2) := by
  simp only [phiUp2Term]
  rw [Real.sqrt_mul (by positivity : (0 : ℝ) ≤ (n : ℝ) + 1)]

/-- Summability of the `a²`-block terms in a coherent state. -/
lemma phiUp2Term_summable (xi : ℂ) :
    Summable fun n : ℕ => phiUp2Term (cohAmp xi) n := by
  refine ((cohExpSeries_summable xi).mul_left
      (Complex.exp (-((‖xi‖ : ℂ) ^ 2)) * xi ^ 2)).congr fun n => ?_
  rw [phiUp2Term_eq]
  exact (cohAmp_term_aSq xi n).symm

/-- The `a²` block sums to `ξ²` in a coherent state. -/
lemma phiUp2Term_tsum_cohAmp (xi : ℂ) :
    ∑' n : ℕ, phiUp2Term (cohAmp xi) n = xi ^ 2 := by
  rw [tsum_congr fun n => phiUp2Term_eq (cohAmp xi) n]
  exact aSqExpect_cohAmp xi

/-- Diagonal collapse: `conj ψₙ (2n+1) ψₙ` is the real scalar `(2n+1) ‖ψₙ‖²`. -/
lemma phiDiagTerm_eq (psi : ℕ → ℂ) (n : ℕ) :
    phiDiagTerm psi n = (((2 * (n : ℝ) + 1) * ‖psi n‖ ^ 2 : ℝ) : ℂ) := by
  simp only [phiDiagTerm]
  rw [mul_right_comm, Complex.conj_mul']
  push_cast
  ring

/-- Summability of the Poisson weights `‖⟨n|ξ⟩‖²`. -/
lemma cohAmp_normSq_summable (xi : ℂ) :
    Summable fun n : ℕ => ‖cohAmp xi n‖ ^ 2 := by
  refine ((Real.summable_pow_div_factorial (‖xi‖ ^ 2)).mul_left
      (Real.exp (-(‖xi‖ ^ 2)))).congr fun n => ?_
  rw [norm_cohAmp_sq, mul_div_assoc]

/-- Summability of the number-weighted Poisson weights `n ‖⟨n|ξ⟩‖²`. -/
lemma cohNExpect_summable (xi : ℂ) :
    Summable fun n : ℕ => (n : ℝ) * ‖cohAmp xi n‖ ^ 2 := by
  refine ((summable_nat_add_iff (f := cohNTerm (‖xi‖ ^ 2)) 1).mp
      (cohNTerm_succ_summable (‖xi‖ ^ 2))).congr fun n => ?_
  simp only [cohNTerm, norm_cohAmp_sq]

/-- Summability of the second-factorial-moment Poisson weights
`n (n - 1) ‖⟨n|ξ⟩‖²`. -/
lemma cohNSecondFactorial_summable (xi : ℂ) :
    Summable fun n : ℕ =>
      (n : ℝ) * ((n : ℝ) - 1) * ‖cohAmp xi n‖ ^ 2 := by
  refine (summable_nat_add_iff (f := fun n : ℕ =>
    (n : ℝ) * ((n : ℝ) - 1) * ‖cohAmp xi n‖ ^ 2) 2).mp ?_
  refine ((Real.summable_pow_div_factorial (‖xi‖ ^ 2)).mul_left
    (Real.exp (-(‖xi‖ ^ 2)) * (‖xi‖ ^ 2) ^ 2)).congr fun n => ?_
  rw [norm_cohAmp_sq, show n + 2 = n + 1 + 1 by omega,
    Nat.factorial_succ, Nat.factorial_succ]
  push_cast
  rw [pow_add]
  field_simp
  ring

/-- Summability of the squared-number-weighted Poisson weights
`n² ‖⟨n|ξ⟩‖²`. -/
lemma cohNSqExpect_summable (xi : ℂ) :
    Summable fun n : ℕ => (n : ℝ) ^ 2 * ‖cohAmp xi n‖ ^ 2 := by
  refine ((cohNSecondFactorial_summable xi).add (cohNExpect_summable xi)).congr
    fun n => ?_
  ring

/-- Normalization of the coherent state: `∑ ‖⟨n|ξ⟩‖² = 1`. -/
lemma cohAmp_normSq_tsum (xi : ℂ) : ∑' n : ℕ, ‖cohAmp xi n‖ ^ 2 = 1 := by
  rw [tsum_congr fun n : ℕ => show ‖cohAmp xi n‖ ^ 2
      = Real.exp (-(‖xi‖ ^ 2)) * ((‖xi‖ ^ 2) ^ n / (n.factorial : ℝ)) by
        rw [norm_cohAmp_sq, mul_div_assoc],
    tsum_mul_left, coh_tsum_exp_real, ← Real.exp_add, neg_add_cancel, Real.exp_zero]

/-- Weighted normalization sum: `∑ (2n+1) ‖⟨n|ξ⟩‖² = 2 ‖ξ‖² + 1`. -/
lemma cohWeightSum (xi : ℂ) :
    ∑' n : ℕ, (2 * (n : ℝ) + 1) * ‖cohAmp xi n‖ ^ 2 = 2 * ‖xi‖ ^ 2 + 1 := by
  rw [tsum_congr fun n : ℕ => show (2 * (n : ℝ) + 1) * ‖cohAmp xi n‖ ^ 2
      = 2 * ((n : ℝ) * ‖cohAmp xi n‖ ^ 2) + ‖cohAmp xi n‖ ^ 2 by ring,
    Summable.tsum_add ((cohNExpect_summable xi).mul_left 2) (cohAmp_normSq_summable xi),
    tsum_mul_left,
    show (∑' n : ℕ, (n : ℝ) * ‖cohAmp xi n‖ ^ 2) = ‖xi‖ ^ 2 from nExpect_cohAmp xi,
    cohAmp_normSq_tsum]

/-- Summability of the diagonal-block terms in a coherent state. -/
lemma phiDiagTerm_summable (xi : ℂ) :
    Summable fun n : ℕ => phiDiagTerm (cohAmp xi) n := by
  have hg : Summable fun n : ℕ => (2 * (n : ℝ) + 1) * ‖cohAmp xi n‖ ^ 2 := by
    refine (((cohNExpect_summable xi).mul_left 2).add (cohAmp_normSq_summable xi)).congr
      fun n => ?_
    ring
  exact (Complex.summable_ofReal.mpr hg).congr fun n => (phiDiagTerm_eq (cohAmp xi) n).symm

/-- The diagonal block sums to `2 ‖ξ‖² + 1` in a coherent state. -/
lemma phiDiagTerm_tsum_cohAmp (xi : ℂ) :
    ∑' n : ℕ, phiDiagTerm (cohAmp xi) n = 2 * (‖xi‖ : ℂ) ^ 2 + 1 := by
  rw [tsum_congr fun n => phiDiagTerm_eq (cohAmp xi) n, ← Complex.ofReal_tsum, cohWeightSum]
  norm_cast

/-- The `n = 0` double-ladder-down term vanishes. -/
lemma phiDown2Term_zero (psi : ℕ → ℂ) : phiDown2Term psi 0 = 0 := by
  simp [phiDown2Term]

/-- The `n = 1` double-ladder-down term vanishes because `√(1-1) = 0`. -/
lemma phiDown2Term_one (psi : ℕ → ℂ) : phiDown2Term psi 1 = 0 := by
  simp [phiDown2Term]

/-- Term collapse for the `a†²` block: two steps down the ladder against `cohAmp` produce
`conj ξ ²` times one term of the exponential series (the conjugate of `cohAmp_term_aSq`). -/
lemma phiDown2Term_cohAmp_add_two (xi : ℂ) (n : ℕ) :
    phiDown2Term (cohAmp xi) (n + 2)
      = Complex.exp (-((‖xi‖ : ℂ) ^ 2)) * (starRingEnd ℂ) xi ^ 2
          * (((‖xi‖ : ℂ) ^ 2) ^ n / (n.factorial : ℂ)) := by
  have h := congrArg (starRingEnd ℂ) (cohAmp_term_aSq xi n)
  simp only [map_mul, map_div₀, map_pow, map_neg, Complex.conj_conj, Complex.conj_ofReal,
    map_natCast, ← Complex.exp_conj] at h
  have hb : ((n + 2 : ℕ) : ℝ) = (n : ℝ) + 2 := by push_cast; ring
  have hs : Real.sqrt ((n : ℝ) + 2 - 1) * Real.sqrt ((n : ℝ) + 2)
      = Real.sqrt (((n : ℝ) + 1) * ((n : ℝ) + 2)) := by
    rw [show (n : ℝ) + 2 - 1 = (n : ℝ) + 1 by ring,
      ← Real.sqrt_mul (by positivity : (0 : ℝ) ≤ (n : ℝ) + 1)]
  simp only [phiDown2Term, Nat.add_sub_cancel, hb, hs]
  linear_combination h

/-- Summability of the doubly shifted `a†²`-block terms in a coherent state. -/
lemma phiDown2Term_add_two_summable (xi : ℂ) :
    Summable fun n : ℕ => phiDown2Term (cohAmp xi) (n + 2) :=
  ((cohExpSeries_summable xi).mul_left
      (Complex.exp (-((‖xi‖ : ℂ) ^ 2)) * (starRingEnd ℂ) xi ^ 2)).congr
    fun n => (phiDown2Term_cohAmp_add_two xi n).symm

/-- Summability of the singly shifted `a†²`-block terms in a coherent state. -/
lemma phiDown2Term_add_one_summable (xi : ℂ) :
    Summable fun n : ℕ => phiDown2Term (cohAmp xi) (n + 1) := by
  have h : Summable fun n : ℕ => phiDown2Term (cohAmp xi) (n + 1 + 1) :=
    (phiDown2Term_add_two_summable xi).congr fun n => rfl
  exact (summable_nat_add_iff (f := fun n => phiDown2Term (cohAmp xi) (n + 1)) 1).mp h

/-- Summability of the `a†²`-block terms in a coherent state. -/
lemma phiDown2Term_summable (xi : ℂ) :
    Summable fun n : ℕ => phiDown2Term (cohAmp xi) n :=
  (summable_nat_add_iff (f := fun n => phiDown2Term (cohAmp xi) n) 1).mp
    (phiDown2Term_add_one_summable xi)

/-- The `a†²` block sums to `conj ξ ²` in a coherent state. -/
lemma phiDown2Term_tsum_cohAmp (xi : ℂ) :
    ∑' n : ℕ, phiDown2Term (cohAmp xi) n = (starRingEnd ℂ) xi ^ 2 := by
  rw [tsum_eq_zero_add' (phiDown2Term_add_one_summable xi), phiDown2Term_zero, zero_add,
    tsum_eq_zero_add' (f := fun n => phiDown2Term (cohAmp xi) (n + 1))
      ((phiDown2Term_add_two_summable xi).congr fun n => rfl)]
  simp only [zero_add, phiDown2Term_one,
    show ∀ k : ℕ, k + 1 + 1 = k + 2 from fun k => rfl]
  rw [tsum_congr fun n => phiDown2Term_cohAmp_add_two xi n, tsum_mul_left,
    coh_tsum_exp_complex, mul_right_comm, ← Complex.exp_add, neg_add_cancel, Complex.exp_zero,
    one_mul]

/-- Sesquilinear expansion of a `⟨φ²⟩` summand into the `a²`, diagonal, and `a†²` ladder
blocks: `conj ψₙ (φ(φψ))ₙ = (T_{a²}(n) + T_diag(n) + T_{a†²}(n)) / (2 ω)`. -/
lemma phiSq_term (omega : ℝ) (homega : 0 < omega) (psi : ℕ → ℂ) (n : ℕ) :
    (starRingEnd ℂ) (psi n) * phiApply omega (phiApply omega psi) n
      = (phiUp2Term psi n + phiDiagTerm psi n + phiDown2Term psi n) / (2 * (omega : ℂ)) := by
  have h2 : (0 : ℝ) < 2 * omega := by linarith
  have hcc : ((Real.sqrt (2 * omega) : ℝ) : ℂ) * ((Real.sqrt (2 * omega) : ℝ) : ℂ)
      = 2 * (omega : ℂ) := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt h2.le]
    push_cast
    ring
  match n with
  | 0 =>
    have e1 : (0 : ℕ) - 1 = 0 := rfl
    have e3 : (0 : ℕ) + 1 = 1 := rfl
    have e4 : (0 : ℕ) + 2 = 2 := rfl
    have er1 : (1 : ℝ) + 1 = 2 := by norm_num
    simp only [phiApply, phiUp2Term, phiDiagTerm, phiDown2Term, e1, e3, e4,
      Nat.cast_zero, Nat.cast_one, zero_add, er1, Real.sqrt_zero, Real.sqrt_one,
      Complex.ofReal_zero, Complex.ofReal_one, zero_mul, mul_zero,
      one_mul, mul_one, add_zero]
    simp only [div_div, ← mul_div_assoc]
    rw [hcc]
    congr 1
    ring
  | 1 =>
    have e1 : (1 : ℕ) - 1 = 0 := rfl
    have e3 : (1 : ℕ) + 1 = 2 := rfl
    have e4 : (1 : ℕ) + 2 = 3 := rfl
    have e5 : (2 : ℕ) - 1 = 1 := rfl
    have e8 : (0 : ℕ) + 1 = 1 := rfl
    have er1 : (1 : ℝ) + 1 = 2 := by norm_num
    have er2 : (2 : ℝ) + 1 = 3 := by norm_num
    have er3 : (1 : ℝ) + 2 = 3 := by norm_num
    have hsq2 : ((Real.sqrt 2 : ℝ) : ℂ) * ((Real.sqrt 2 : ℝ) : ℂ) = 2 := by
      rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
      norm_num
    simp only [phiApply, phiUp2Term, phiDiagTerm, phiDown2Term, e1, e3, e4, e5, e8,
      Nat.cast_zero, Nat.cast_one, Nat.cast_ofNat, zero_add, er1, er2, er3, Real.sqrt_zero,
      Real.sqrt_one, sub_self, Complex.ofReal_zero, Complex.ofReal_one, Complex.ofReal_mul,
      Complex.ofReal_ofNat, zero_mul, mul_zero, one_mul, mul_one, add_zero]
    simp only [← add_div, div_div, ← mul_div_assoc]
    rw [hcc]
    congr 1
    linear_combination ((starRingEnd ℂ) (psi 1) * psi 1) * hsq2
  | (m + 2) =>
    have e1 : m + 2 - 1 = m + 1 := rfl
    have e2 : m + 2 - 2 = m := rfl
    have e3 : m + 2 + 1 = m + 3 := rfl
    have e4 : m + 3 - 1 = m + 2 := rfl
    have e5 : m + 3 + 1 = m + 4 := rfl
    have e6 : m + 1 - 1 = m := rfl
    have e7 : m + 1 + 1 = m + 2 := rfl
    have ha1 : ((m : ℝ) + 2) + 1 = (m : ℝ) + 3 := by ring
    have ha2 : ((m : ℝ) + 3) + 1 = (m : ℝ) + 4 := by ring
    have ha3 : ((m : ℝ) + 2) + 2 = (m : ℝ) + 4 := by ring
    have ha4 : ((m : ℝ) + 1) + 1 = (m : ℝ) + 2 := by ring
    have ha5 : ((m : ℝ) + 2) - 1 = (m : ℝ) + 1 := by ring
    have hq : ((Real.sqrt ((m : ℝ) + 2) : ℝ) : ℂ) * ((Real.sqrt ((m : ℝ) + 2) : ℝ) : ℂ)
        = (m : ℂ) + 2 := by
      rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity)]
      push_cast
      ring
    have hr : ((Real.sqrt ((m : ℝ) + 3) : ℝ) : ℂ) * ((Real.sqrt ((m : ℝ) + 3) : ℝ) : ℂ)
        = (m : ℂ) + 3 := by
      rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity)]
      push_cast
      ring
    simp only [phiApply, phiUp2Term, phiDiagTerm, phiDown2Term, e1, e2, e3, e4, e5, e6, e7]
    push_cast
    simp only [ha1, ha2, ha3, ha4, ha5]
    simp only [← add_div, div_div, ← mul_div_assoc]
    rw [hcc]
    congr 1
    linear_combination ((starRingEnd ℂ) (psi (m + 2)) * psi (m + 2)) * hq
      + ((starRingEnd ℂ) (psi (m + 2)) * psi (m + 2)) * hr

/-- **Operator-level `⟨φ²⟩`** (manuscript Secs. `subsec:encoding`, `subsec:sg-quantization`):
in the coherent state `|ξ⟩` the squared phase quadrature has mean
`⟨φ²⟩ = (ξ² + conj ξ ² + 2 ‖ξ‖² + 1)/(2 ω)`. -/
theorem phiSqExpect_cohAmp (xi : ℂ) {omega : ℝ} (homega : 0 < omega) :
    phiSqExpect omega (cohAmp xi)
      = (xi ^ 2 + (starRingEnd ℂ) xi ^ 2 + 2 * (‖xi‖ : ℂ) ^ 2 + 1) / (2 * omega) := by
  calc phiSqExpect omega (cohAmp xi)
      = ∑' n : ℕ, (phiUp2Term (cohAmp xi) n + phiDiagTerm (cohAmp xi) n
          + phiDown2Term (cohAmp xi) n) / (2 * (omega : ℂ)) :=
        tsum_congr fun n => phiSq_term omega homega (cohAmp xi) n
    _ = (∑' n : ℕ, (phiUp2Term (cohAmp xi) n + phiDiagTerm (cohAmp xi) n
          + phiDown2Term (cohAmp xi) n)) / (2 * (omega : ℂ)) := tsum_div_const
    _ = ((∑' n : ℕ, phiUp2Term (cohAmp xi) n) + (∑' n : ℕ, phiDiagTerm (cohAmp xi) n)
          + ∑' n : ℕ, phiDown2Term (cohAmp xi) n) / (2 * (omega : ℂ)) := by
        rw [Summable.tsum_add ((phiUp2Term_summable xi).add (phiDiagTerm_summable xi))
            (phiDown2Term_summable xi),
          Summable.tsum_add (phiUp2Term_summable xi) (phiDiagTerm_summable xi)]
    _ = (xi ^ 2 + (2 * (‖xi‖ : ℂ) ^ 2 + 1) + (starRingEnd ℂ) xi ^ 2) / (2 * (omega : ℂ)) := by
        rw [phiUp2Term_tsum_cohAmp, phiDiagTerm_tsum_cohAmp, phiDown2Term_tsum_cohAmp]
    _ = (xi ^ 2 + (starRingEnd ℂ) xi ^ 2 + 2 * (‖xi‖ : ℂ) ^ 2 + 1) / (2 * omega) := by ring

/-- **Operator-level zero-point variance** (manuscript Eq. `eq:sg-zpf`): in the coherent state
`|ξ⟩`, `⟨φ²⟩ - ⟨φ⟩² = 1/(2 ω)` — the coherent-state variance of `φ = (a + a†)/√(2 ω)` is the
zero-point value `1/(2 ω)`, now derived from the Fock-sum moments themselves rather than from
substituted scalars (compare `cohVariance`). -/
theorem cohPhiVariance (xi : ℂ) {omega : ℝ} (homega : 0 < omega) :
    phiSqExpect omega (cohAmp xi) - phiExpect omega (cohAmp xi) ^ 2 = 1 / (2 * omega) := by
  rw [phiSqExpect_cohAmp xi homega, phiExpect_cohAmp xi homega]
  have hsq : ((Real.sqrt (2 * omega) : ℝ) : ℂ) ^ 2 = 2 * (omega : ℂ) := by
    rw [← Complex.ofReal_pow, Real.sq_sqrt (by linarith : (0 : ℝ) ≤ 2 * omega)]
    push_cast
    ring
  rw [div_pow, hsq, div_sub_div_same,
    show xi ^ 2 + (starRingEnd ℂ) xi ^ 2 + 2 * (‖xi‖ : ℂ) ^ 2 + 1
        - (xi + (starRingEnd ℂ) xi) ^ 2 = 1 by
      linear_combination (-2 : ℂ) * Complex.mul_conj' xi]

/-- Fock-sum expectation of the creation operator `a†` in the state `psi`. -/
def phiADagExpect (psi : ℕ → ℂ) : ℂ :=
  ∑' n : ℕ, phiDownTerm psi n

/-- In a coherent state, `⟨a†⟩ = conj ⟨a⟩ (= conj ξ)`. -/
theorem phiADagExpect_cohAmp (xi : ℂ) :
    phiADagExpect (cohAmp xi) = (starRingEnd ℂ) (aExpect (cohAmp xi)) := by
  rw [aExpect_cohAmp]
  exact phiDownTerm_tsum_cohAmp xi

/-- Fock-sum expectation of `a†²` in the state `psi`. -/
def phiADagSqExpect (psi : ℕ → ℂ) : ℂ :=
  ∑' n : ℕ, phiDown2Term psi n

/-- In a coherent state, `⟨a†²⟩ = conj ⟨a²⟩ (= conj ξ ²)`. -/
theorem phiADagSqExpect_cohAmp (xi : ℂ) :
    phiADagSqExpect (cohAmp xi) = (starRingEnd ℂ) (aSqExpect (cohAmp xi)) := by
  rw [aSqExpect_cohAmp, map_pow]
  exact phiDown2Term_tsum_cohAmp xi

end

end Cvdv
