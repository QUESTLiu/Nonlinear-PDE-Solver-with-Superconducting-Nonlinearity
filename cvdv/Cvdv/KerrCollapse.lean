import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Topology.Algebra.InfiniteSum.Ring
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Positivity

/-!
# Coherent-state first-moment collapse under Kerr evolution

Formalization of the first-moment collapse formula of manuscript Sec. 2.1: under the Kerr
Hamiltonian `H_K = ω n + K n²` an initial coherent state `|ξ⟩` evolves so that its first
moment `⟨a⟩` has modulus `‖ξ‖ · exp(n̄ (cos 2Kt − 1))` with `n̄ = ‖ξ‖²` the mean occupation.

States are represented as Fock-coefficient sequences `ℕ → ℂ`.  `coherentAmp ξ` is the
coherent-state amplitude sequence, `kerrEnergy ω K n = ω n + K n²` the Kerr spectrum,
`kerrState` the time-evolved sequence, and `firstMoment ψ = ∑' n, conj (ψ n) √(n+1) ψ (n+1)`
the explicit matrix-element sum for `⟨a⟩`.

Main results:
* `kerr_firstMoment`: the closed form
  `⟨a⟩ = ξ · exp(−(ω+K)t i) · exp(‖ξ‖² (exp(−2Kt i) − 1))`;
* `kerr_firstMoment_norm`: the manuscript display
  `‖⟨a⟩‖ = ‖ξ‖ · exp(‖ξ‖² (cos 2Kt − 1))`.

All results are fully proved, with no placeholders.
-/

namespace Cvdv

/-- Fock coefficients of the coherent state `|ξ⟩`. -/
noncomputable def coherentAmp (xi : ℂ) (n : ℕ) : ℂ :=
  Complex.exp (-((‖xi‖ : ℂ)^2)/2) * xi^n / (Real.sqrt n.factorial : ℝ)

/-- Kerr spectrum `ω n + K n²`. -/
noncomputable def kerrEnergy (omega K : ℝ) (n : ℕ) : ℝ := omega * n + K * n^2

/-- Coherent state evolved for time `t` under the Kerr Hamiltonian. -/
noncomputable def kerrState (xi : ℂ) (omega K t : ℝ) (n : ℕ) : ℂ :=
  coherentAmp xi n * Complex.exp (-(kerrEnergy omega K n * t) * Complex.I)

/-- First moment `⟨a⟩` of a Fock-coefficient sequence, as the matrix-element sum. -/
noncomputable def firstMoment (psi : ℕ → ℂ) : ℂ :=
  ∑' n : ℕ, (starRingEnd ℂ) (psi n) * (Real.sqrt (n+1) : ℝ) * psi (n+1)

/-- `√((n+1)!) = √(n+1) · √(n!)`. -/
lemma sqrt_factorial_succ (n : ℕ) :
    Real.sqrt (n+1).factorial = Real.sqrt (n+1 : ℝ) * Real.sqrt n.factorial := by
  rw [Nat.factorial_succ]
  push_cast
  rw [Real.sqrt_mul (by positivity)]

/-- Coherent-state part of the matrix element: the `√(n+1)` ladder factor cancels the
factorial mismatch, leaving `ξ e^{-‖ξ‖²} ‖ξ‖^{2n} / n!`. -/
lemma coherent_moment_term (xi : ℂ) (n : ℕ) :
    (starRingEnd ℂ) (coherentAmp xi n) * (Real.sqrt (n+1) : ℝ) * coherentAmp xi (n+1)
      = xi * Complex.exp (-((‖xi‖:ℂ)^2)) * ((‖xi‖:ℂ)^2)^n / (n.factorial : ℂ) := by
  have hfac_pos : (0:ℝ) < (n.factorial : ℝ) := by exact_mod_cast n.factorial_pos
  have hsR : Real.sqrt n.factorial * Real.sqrt (n+1).factorial
      = Real.sqrt ((n:ℝ)+1) * (n.factorial : ℝ) := by
    rw [sqrt_factorial_succ n, mul_left_comm, Real.mul_self_sqrt hfac_pos.le]
  have hcast : (Real.sqrt n.factorial : ℂ) * (Real.sqrt (n+1).factorial : ℂ)
      = (Real.sqrt ((n:ℝ)+1) : ℂ) * (n.factorial : ℂ) := by
    exact_mod_cast hsR
  have hq0 : (Real.sqrt ((n:ℝ)+1) : ℂ) ≠ 0 := by
    have h : (0:ℝ) < Real.sqrt ((n:ℝ)+1) := Real.sqrt_pos.mpr (by positivity)
    exact_mod_cast h.ne'
  have hqC : (Real.sqrt ((n:ℝ)+1) : ℂ) * (Real.sqrt n.factorial : ℂ)⁻¹
        * (Real.sqrt (n+1).factorial : ℂ)⁻¹
      = (n.factorial : ℂ)⁻¹ := by
    rw [mul_assoc, ← mul_inv, hcast, mul_inv, ← mul_assoc, mul_inv_cancel₀ hq0, one_mul]
  have harg : -((‖xi‖:ℂ)^2)/2 = ((-(‖xi‖^2)/2 : ℝ) : ℂ) := by push_cast; ring
  have hconjA : (starRingEnd ℂ) (Complex.exp (-((‖xi‖:ℂ)^2)/2))
      = Complex.exp (-((‖xi‖:ℂ)^2)/2) := by
    rw [harg, ← Complex.exp_conj, Complex.conj_ofReal]
  have hA : Complex.exp (-((‖xi‖:ℂ)^2)/2) * Complex.exp (-((‖xi‖:ℂ)^2)/2)
      = Complex.exp (-((‖xi‖:ℂ)^2)) := by
    rw [← Complex.exp_add]
    congr 1
    ring
  have hxipow : ((starRingEnd ℂ) xi * xi)^n = ((‖xi‖:ℂ)^2)^n := by
    rw [Complex.conj_mul']
  simp only [coherentAmp]
  rw [map_div₀, map_mul, map_pow, Complex.conj_ofReal, hconjA]
  linear_combination
    ((starRingEnd ℂ) xi ^ n * xi ^ (n+1) * (Real.sqrt ((n:ℝ)+1) : ℂ)
        * (Real.sqrt n.factorial : ℂ)⁻¹ * (Real.sqrt (n+1).factorial : ℂ)⁻¹) * hA
    + (xi * Complex.exp (-((‖xi‖:ℂ)^2)) * (Real.sqrt ((n:ℝ)+1) : ℂ)
        * (Real.sqrt n.factorial : ℂ)⁻¹ * (Real.sqrt (n+1).factorial : ℂ)⁻¹) * hxipow
    + (xi * Complex.exp (-((‖xi‖:ℂ)^2)) * ((‖xi‖:ℂ)^2)^n) * hqC

/-- Phase part of the matrix element: the energy difference
`E_{n+1} − E_n = (ω + K) + 2Kn` splits into an `n`-independent phase and the `n`-th
power of the collapse phase `e^{-2Kt i}`. -/
lemma kerr_phase (omega K t : ℝ) (n : ℕ) :
    (starRingEnd ℂ) (Complex.exp (-(kerrEnergy omega K n * t) * Complex.I))
        * Complex.exp (-(kerrEnergy omega K (n+1) * t) * Complex.I)
      = Complex.exp (-((omega + K) * t) * Complex.I)
          * Complex.exp (-(2*K*t) * Complex.I) ^ n := by
  rw [← Complex.exp_conj, ← Complex.exp_nat_mul, ← Complex.exp_add, ← Complex.exp_add]
  congr 1
  simp only [kerrEnergy, map_mul, map_neg, Complex.conj_ofReal, Complex.conj_I]
  push_cast
  ring

/-- Closed form of the `n`-th matrix element of the Kerr-evolved coherent state. -/
lemma kerr_summand (xi : ℂ) (omega K t : ℝ) (n : ℕ) :
    (starRingEnd ℂ) (kerrState xi omega K t n) * (Real.sqrt (n+1) : ℝ)
        * kerrState xi omega K t (n+1)
      = xi * Complex.exp (-((omega + K) * t) * Complex.I)
          * Complex.exp (-((‖xi‖:ℂ)^2))
          * ((‖xi‖:ℂ)^2 * Complex.exp (-(2*K*t) * Complex.I))^n / (n.factorial : ℂ) := by
  have h1 := coherent_moment_term xi n
  have h2 := kerr_phase omega K t n
  simp only [kerrState]
  rw [map_mul]
  linear_combination
    ((starRingEnd ℂ) (Complex.exp (-(kerrEnergy omega K n * t) * Complex.I))
        * Complex.exp (-(kerrEnergy omega K (n+1) * t) * Complex.I)) * h1
    + (xi * Complex.exp (-((‖xi‖:ℂ)^2)) * ((‖xi‖:ℂ)^2)^n / (n.factorial : ℂ)) * h2

/-- The exponential series in `ℂ`: `∑' n, w^n / n! = exp w`. -/
lemma tsum_pow_div_factorial_eq_exp (w : ℂ) :
    ∑' n : ℕ, w ^ n / (n.factorial : ℂ) = Complex.exp w := by
  rw [Complex.exp_eq_exp_ℂ, NormedSpace.exp_eq_tsum_div]

/-- Manuscript Sec. 2.1: closed form of the first moment of a coherent state evolved
under the Kerr Hamiltonian `ω n + K n²`. -/
theorem kerr_firstMoment (xi : ℂ) (omega K t : ℝ) :
    firstMoment (kerrState xi omega K t)
      = xi * Complex.exp (-((omega + K) * t) * Complex.I)
          * Complex.exp ((‖xi‖:ℂ)^2 * (Complex.exp (-(2*K*t) * Complex.I) - 1)) := by
  have hterm : ∀ n : ℕ,
      (starRingEnd ℂ) (kerrState xi omega K t n) * (Real.sqrt (n+1) : ℝ)
          * kerrState xi omega K t (n+1)
        = (xi * Complex.exp (-((omega + K) * t) * Complex.I) * Complex.exp (-((‖xi‖:ℂ)^2)))
            * (((‖xi‖:ℂ)^2 * Complex.exp (-(2*K*t) * Complex.I))^n / (n.factorial : ℂ)) := by
    intro n
    rw [kerr_summand xi omega K t n]
    ring
  have hexp : Complex.exp (-((‖xi‖:ℂ)^2))
        * Complex.exp ((‖xi‖:ℂ)^2 * Complex.exp (-(2*K*t) * Complex.I))
      = Complex.exp ((‖xi‖:ℂ)^2 * (Complex.exp (-(2*K*t) * Complex.I) - 1)) := by
    rw [← Complex.exp_add]
    congr 1
    ring
  unfold firstMoment
  rw [tsum_congr hterm, tsum_mul_left, tsum_pow_div_factorial_eq_exp, mul_assoc, hexp]

/-- Manuscript Sec. 2.1 display: the modulus of the first moment collapses as
`‖⟨a⟩‖ = ‖ξ‖ · exp(n̄ (cos 2Kt − 1))` with `n̄ = ‖ξ‖²`. -/
theorem kerr_firstMoment_norm (xi : ℂ) (omega K t : ℝ) :
    ‖firstMoment (kerrState xi omega K t)‖
      = ‖xi‖ * Real.exp (‖xi‖^2 * (Real.cos (2*K*t) - 1)) := by
  have h1 : ‖Complex.exp (-((omega + K) * t) * Complex.I)‖ = 1 := by
    have harg : -((omega + K) * t : ℂ) * Complex.I
        = ((-((omega + K) * t) : ℝ) : ℂ) * Complex.I := by
      push_cast; ring
    rw [harg, Complex.norm_exp_ofReal_mul_I]
  have h2 : ‖Complex.exp ((‖xi‖:ℂ)^2 * (Complex.exp (-(2*K*t) * Complex.I) - 1))‖
      = Real.exp (‖xi‖^2 * (Real.cos (2*K*t) - 1)) := by
    rw [Complex.norm_exp]
    congr 1
    have harg2 : (-(2*K*t) * Complex.I : ℂ) = ((-(2*K*t) : ℝ) : ℂ) * Complex.I := by
      push_cast; ring
    have hsq : ((‖xi‖:ℂ)^2) = ((‖xi‖^2 : ℝ) : ℂ) := by push_cast; ring
    rw [harg2, hsq, Complex.re_ofReal_mul, Complex.sub_re, Complex.one_re,
      Complex.exp_ofReal_mul_I_re, Real.cos_neg]
  rw [kerr_firstMoment, norm_mul, norm_mul, h1, mul_one, h2]

end Cvdv
