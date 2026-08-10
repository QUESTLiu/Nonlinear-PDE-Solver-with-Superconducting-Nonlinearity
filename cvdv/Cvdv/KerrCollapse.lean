import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Topology.Algebra.InfiniteSum.Ring
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Positivity
import Cvdv.CoherentQuadrature

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

/-!
### Collapse and revival (manuscript Sec. 2.1, l.296-311; conservation sentence l.1325)

Completion of the Kerr collapse/revival paragraph: `⟨n⟩` conservation, the small-time
Gaussian envelope with `O(t⁴)` remainder, the first return of the first-moment magnitude
at `t_rev = π/|K|`, and the `e^{-1}` collapse-time identity.
-/

/-- Definition bridge: the coherent amplitudes `coherentAmp` of this file coincide with
`cohAmp` of `Cvdv.CoherentQuadrature` (both are `⟨n|ξ⟩ = e^{-‖ξ‖²/2} ξⁿ/√(n!)`). -/
theorem kerr_coherentAmp_eq_cohAmp (xi : ℂ) : coherentAmp xi = cohAmp xi := rfl

/-- Number-diagonal evolution only multiplies each Fock amplitude by a unimodular phase:
`‖(kerrState ξ ω K t) n‖ = ‖cohAmp ξ n‖`. -/
lemma kerr_norm_kerrState (xi : ℂ) (omega K t : ℝ) (n : ℕ) :
    ‖kerrState xi omega K t n‖ = ‖cohAmp xi n‖ := by
  have h1 : ‖Complex.exp (-(kerrEnergy omega K n * t) * Complex.I)‖ = 1 := by
    have harg : -(kerrEnergy omega K n * t : ℂ) * Complex.I
        = ((-(kerrEnergy omega K n * t) : ℝ) : ℂ) * Complex.I := by
      push_cast; ring
    rw [harg, Complex.norm_exp_ofReal_mul_I]
  simp only [kerrState]
  rw [norm_mul, h1, mul_one, kerr_coherentAmp_eq_cohAmp]

/-- Manuscript l.311 and l.1325: because the Kerr Hamiltonian is a function only of `n̂`,
it conserves the mean occupation: `⟨n⟩ = ‖ξ‖²` at every time `t` ("even though no energy
has been lost"). -/
theorem kerr_nExpect_invariant (xi : ℂ) (omega K t : ℝ) :
    nExpect (kerrState xi omega K t) = ‖xi‖ ^ 2 := by
  have h : nExpect (kerrState xi omega K t) = nExpect (cohAmp xi) := by
    simp only [nExpect]
    exact tsum_congr fun n => by rw [kerr_norm_kerrState]
  rw [h, nExpect_cohAmp]

/-- Manuscript l.303-305: the collapse exponent `n̄ (cos 2Kt − 1)` agrees with the Gaussian
exponent `−2 n̄ K² t²` up to an `O(t⁴)` remainder as `t → 0`, i.e.
`n̄ (cos 2Kt − 1) + 2 n̄ K² t² = O(t⁴)` near `t = 0`. -/
theorem kerr_envelope_isBigO (xi : ℂ) (K : ℝ) :
    (fun t : ℝ => ‖xi‖ ^ 2 * (Real.cos (2 * K * t) - 1) + 2 * ‖xi‖ ^ 2 * K ^ 2 * t ^ 2)
      =O[nhds 0] fun t : ℝ => t ^ 4 := by
  rw [Asymptotics.isBigO_iff]
  refine ⟨‖xi‖ ^ 2 * K ^ 4 * (5 / 6), ?_⟩
  have habs : ∀ᶠ u in nhds (0 : ℝ), |u| ≤ 1 := by
    filter_upwards [Icc_mem_nhds (by norm_num : (-1 : ℝ) < 0) (by norm_num : (0 : ℝ) < 1)]
      with u hu
    exact abs_le.mpr ⟨hu.1, hu.2⟩
  have hcont : Filter.Tendsto (fun t : ℝ => 2 * K * t) (nhds 0) (nhds 0) := by
    simpa using (continuous_id.const_mul (2 * K)).tendsto 0
  filter_upwards [hcont.eventually habs] with t ht
  have hb := Real.cos_bound ht
  have heq : ‖xi‖ ^ 2 * (Real.cos (2 * K * t) - 1) + 2 * ‖xi‖ ^ 2 * K ^ 2 * t ^ 2
      = ‖xi‖ ^ 2 * (Real.cos (2 * K * t) - (1 - (2 * K * t) ^ 2 / 2)) := by ring
  have h4 : ∀ a : ℝ, |a| ^ 4 = a ^ 4 := fun a => by
    rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, sq_abs, ← pow_mul]
  rw [Real.norm_eq_abs, Real.norm_eq_abs, heq, abs_mul,
    abs_of_nonneg (by positivity : (0 : ℝ) ≤ ‖xi‖ ^ 2)]
  calc ‖xi‖ ^ 2 * |Real.cos (2 * K * t) - (1 - (2 * K * t) ^ 2 / 2)|
      ≤ ‖xi‖ ^ 2 * (|2 * K * t| ^ 4 * (5 / 96)) :=
        mul_le_mul_of_nonneg_left hb (by positivity)
    _ = ‖xi‖ ^ 2 * K ^ 4 * (5 / 6) * |t ^ 4| := by
        rw [h4, abs_of_nonneg (by positivity : (0 : ℝ) ≤ t ^ 4)]
        ring

/-- Manuscript l.310: the first-moment magnitude first returns at `t_rev = π/|K|`.
At `t_rev` the modulus equals `‖ξ‖`, and for every `0 < t < t_rev` it is strictly
below `‖ξ‖`. -/
theorem kerr_first_return (xi : ℂ) (omega K : ℝ) (hK : K ≠ 0) (hxi : xi ≠ 0) :
    ‖firstMoment (kerrState xi omega K (Real.pi / |K|))‖ = ‖xi‖ ∧
      ∀ t ∈ Set.Ioo (0 : ℝ) (Real.pi / |K|),
        ‖firstMoment (kerrState xi omega K t)‖ < ‖xi‖ := by
  have hKpos : (0 : ℝ) < |K| := abs_pos.mpr hK
  have hxin : (0 : ℝ) < ‖xi‖ := norm_pos_iff.mpr hxi
  constructor
  · rw [kerr_firstMoment_norm]
    have hcos : Real.cos (2 * K * (Real.pi / |K|)) = 1 := by
      have habs : |2 * K * (Real.pi / |K|)| = 2 * Real.pi := by
        rw [abs_mul, abs_mul, abs_two, abs_of_pos (div_pos Real.pi_pos hKpos)]
        field_simp
      rw [← Real.cos_abs, habs, Real.cos_two_pi]
    rw [hcos]
    simp
  · intro t ht
    rw [kerr_firstMoment_norm]
    have habs : |2 * K * t| < 2 * Real.pi := by
      have h1 : |2 * K * t| = 2 * |K| * t := by
        rw [abs_mul, abs_mul, abs_two, abs_of_pos ht.1]
      have h2 : t * |K| < Real.pi := (lt_div_iff₀ hKpos).mp ht.2
      rw [h1]; nlinarith
    have hcos_lt : Real.cos (2 * K * t) < 1 := by
      have hne : Real.cos (2 * K * t) ≠ 1 := fun h =>
        mul_ne_zero (mul_ne_zero two_ne_zero hK) ht.1.ne'
          ((Real.cos_eq_one_iff_of_lt_of_lt (neg_lt_of_abs_lt habs)
            (lt_of_abs_lt habs)).mp h)
      exact lt_of_le_of_ne (Real.cos_le_one _) hne
    have hexp : Real.exp (‖xi‖ ^ 2 * (Real.cos (2 * K * t) - 1)) < 1 := by
      rw [Real.exp_lt_one_iff]
      exact mul_neg_of_pos_of_neg (by positivity) (by linarith)
    exact mul_lt_of_lt_one_right hxin hexp

/-- Manuscript l.307-309: at the collapse time `t_c = 1/(|K|·√(2 n̄))` with `n̄ = ‖ξ‖²`,
the Gaussian exponent `2 n̄ K² t²` equals `1`, so `t_c` is the `e^{-1}` timescale of the
small-time envelope. -/
theorem kerr_collapse_time (xi : ℂ) (K : ℝ) (hK : K ≠ 0) (hxi : xi ≠ 0) :
    2 * ‖xi‖ ^ 2 * K ^ 2 * (1 / (|K| * Real.sqrt (2 * ‖xi‖ ^ 2))) ^ 2 = 1 := by
  have hxin : (0 : ℝ) < ‖xi‖ := norm_pos_iff.mpr hxi
  have h2n : (0 : ℝ) < 2 * ‖xi‖ ^ 2 := mul_pos two_pos (pow_pos hxin 2)
  have hne : K ^ 2 * (2 * ‖xi‖ ^ 2) ≠ 0 := mul_ne_zero (pow_ne_zero 2 hK) h2n.ne'
  rw [one_div, inv_pow, mul_pow, sq_abs, Real.sq_sqrt h2n.le]
  rw [show (2 : ℝ) * ‖xi‖ ^ 2 * K ^ 2 = K ^ 2 * (2 * ‖xi‖ ^ 2) by ring,
    mul_inv_cancel₀ hne]

/-- Manuscript l.307-309, `e^{-1}` reading: at the collapse time
`t_c = 1/(|K|·√(2 n̄))` the small-time Gaussian envelope factor
`exp(−2 n̄ K² t²)` equals `exp(−1)` exactly.  This chains `kerr_collapse_time`
to the envelope exponent of `kerr_envelope_isBigO`, so the `e^{-1}` collapse
timescale is a Lean statement rather than a docstring remark. -/
theorem kerr_collapse_time_exp (xi : ℂ) (K : ℝ) (hK : K ≠ 0) (hxi : xi ≠ 0) :
    Real.exp (-(2 * ‖xi‖ ^ 2 * K ^ 2 * (1 / (|K| * Real.sqrt (2 * ‖xi‖ ^ 2))) ^ 2))
      = Real.exp (-1) := by
  rw [kerr_collapse_time xi K hK hxi]

/-- The coherent-state Kerr-energy series is summable. -/
lemma kerr_energy_cohAmp_summable (xi : ℂ) (omega K : ℝ) :
    Summable fun n : ℕ => kerrEnergy omega K n * ‖cohAmp xi n‖ ^ 2 := by
  refine (((cohNExpect_summable xi).mul_left omega).add
    ((cohNSqExpect_summable xi).mul_left K)).congr fun n => ?_
  simp only [kerrEnergy]
  ring

/-- The Kerr-energy series remains summable after number-diagonal evolution. -/
theorem kerr_energyExpect_summable (xi : ℂ) (omega K t : ℝ) :
    Summable fun n : ℕ =>
      kerrEnergy omega K n * ‖kerrState xi omega K t n‖ ^ 2 :=
  (kerr_energy_cohAmp_summable xi omega K).congr fun n => by
    rw [kerr_norm_kerrState]

/-- Summable form of number-diagonal energy conservation: termwise equality
transfers summability and equality of the two energy sums. -/
theorem kerr_energyExpect_invariant_of_summable (xi : ℂ) (omega K t s : ℝ)
    (ht : Summable fun n : ℕ =>
      kerrEnergy omega K n * ‖kerrState xi omega K t n‖ ^ 2) :
    Summable (fun n : ℕ =>
      kerrEnergy omega K n * ‖kerrState xi omega K s n‖ ^ 2) ∧
      (∑' n : ℕ, kerrEnergy omega K n * ‖kerrState xi omega K t n‖ ^ 2) =
        ∑' n : ℕ, kerrEnergy omega K n * ‖kerrState xi omega K s n‖ ^ 2 := by
  have hterm : ∀ n : ℕ,
      kerrEnergy omega K n * ‖kerrState xi omega K t n‖ ^ 2 =
        kerrEnergy omega K n * ‖kerrState xi omega K s n‖ ^ 2 := fun n => by
    rw [kerr_norm_kerrState, kerr_norm_kerrState]
  exact ⟨ht.congr hterm, tsum_congr hterm⟩

/-- Manuscript l.311 ("even though no energy has been lost"), energy version of the
conservation sentence: because the Kerr evolution is number-diagonal, each Fock
amplitude keeps its modulus, so the summable energy expectation
`⟨H_K⟩ = ∑ₙ (ω n + K n²) |ψₙ|²` is the same at any two times `t` and `s`. -/
theorem kerr_energyExpect_invariant (xi : ℂ) (omega K t s : ℝ) :
    (∑' n : ℕ, kerrEnergy omega K n * ‖kerrState xi omega K t n‖ ^ 2)
      = ∑' n : ℕ, kerrEnergy omega K n * ‖kerrState xi omega K s n‖ ^ 2 :=
  (kerr_energyExpect_invariant_of_summable xi omega K t s
    (kerr_energyExpect_summable xi omega K t)).2

/-- The time-invariant energy expectation evaluated against the initial coherent
amplitudes: `⟨H_K⟩(t) = ∑ₙ (ω n + K n²) |⟨n|ξ⟩|²` for every `t`. -/
theorem kerr_energyExpect_eq_initial (xi : ℂ) (omega K t : ℝ) :
    (∑' n : ℕ, kerrEnergy omega K n * ‖kerrState xi omega K t n‖ ^ 2)
      = ∑' n : ℕ, kerrEnergy omega K n * ‖cohAmp xi n‖ ^ 2 :=
  tsum_congr fun n => by rw [kerr_norm_kerrState]

end Cvdv
