import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Normed.Operator.Bilinear
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Tactic.NoncommRing
import Cvdv.StateGeometry

/-!
# Duhamel bound for the rotating-wave-approximation error

This file proves phase-free abstract Duhamel estimates for bounded operators on
a complex Hilbert space (the finite Fock space of the paper).  The
energy-shifted statements matching manuscript Eqs. `eq:df-duhamel` and
`eq:df-bound` are the corollaries in `Cvdv.EnergyShift` and
`Cvdv.FiniteDuffing`.

* `Cvdv.duhamel_identity` is the Duhamel integral identity
  `exp (t • C) x - exp (t • A) x = ∫ s in 0..t, exp ((t-s) • C) ((C - A) (exp (s • A) x))`.
* `Cvdv.duhamel_norm_bound` bounds the difference in norm when `exp (s • C)`
  is isometric.
* `Cvdv.isometry_exp_of_skew` shows that a generator with purely imaginary
  quadratic form generates an isometric flow, and
  `Cvdv.skew_of_selfAdjoint` verifies this for `(-I) • H` with `H` symmetric.
* `Cvdv.duhamel_rwa_bound` is the phase-free abstract norm estimate for the
  `H + λV` and `H` evolutions of `ψ₀`.
* `Cvdv.duhamel_infidelity_bound` is its phase-free, phase-invariant
  infidelity consequence.

All results are fully proved, with no placeholders.
-/

open NormedSpace

namespace Cvdv

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]

/-- Derivative of the orbit `s ↦ exp (s • A) x` of a bounded operator `A`. -/
lemma hasDerivAt_exp_smul_apply (A : E →L[ℂ] E) (x : E) (s : ℝ) :
    HasDerivAt (fun u : ℝ => exp (u • A) x) (A (exp (s • A) x)) s := by
  have hop : HasDerivAt (fun u : ℝ => exp (u • A)) (A * exp (s • A)) s :=
    hasDerivAt_exp_smul_const' A s
  have happ :=
    ((ContinuousLinearMap.apply ℂ E x).restrictScalars ℝ).hasFDerivAt.comp_hasDerivAt s hop
  simpa [Function.comp_def] using happ

set_option linter.unusedVariables false in
/-- L1, the Duhamel identity: the difference of two operator flows applied to a
vector is the integral of the interaction term.  The hypothesis `ht` is kept to
match the manuscript statement, although the identity holds for every `t`. -/
theorem duhamel_identity (A C : E →L[ℂ] E) (x : E) (t : ℝ) (ht : 0 ≤ t) :
    exp (t • C) x - exp (t • A) x
      = ∫ s in (0:ℝ)..t, exp ((t - s) • C) ((C - A) (exp (s • A) x)) := by
  -- Operator-level derivative of the interpolating product.
  have key : ∀ s : ℝ,
      HasDerivAt (fun u : ℝ => exp ((t - u) • C) * exp (u • A))
        (-(exp ((t - s) • C) * ((C - A) * exp (s • A)))) s := by
    intro s
    have h1 : HasDerivAt (fun u : ℝ => t - u) (-1) s := (hasDerivAt_id s).const_sub t
    have h2 : HasDerivAt (fun u : ℝ => exp (u • C)) (exp ((t - s) • C) * C) (t - s) :=
      hasDerivAt_exp_smul_const C (t - s)
    have h3 : HasDerivAt (fun u : ℝ => exp ((t - u) • C))
        ((-1 : ℝ) • (exp ((t - s) • C) * C)) s := by
      simpa [Function.comp_def] using h2.scomp s h1
    have h4 : HasDerivAt (fun u : ℝ => exp (u • A)) (A * exp (s • A)) s :=
      hasDerivAt_exp_smul_const' A s
    have h5 : HasDerivAt (fun u : ℝ => exp ((t - u) • C) * exp (u • A))
        (((-1 : ℝ) • (exp ((t - s) • C) * C)) * exp (s • A)
          + exp ((t - s) • C) * (A * exp (s • A))) s := h3.mul h4
    have harith : ((-1 : ℝ) • (exp ((t - s) • C) * C)) * exp (s • A)
        + exp ((t - s) • C) * (A * exp (s • A))
        = -(exp ((t - s) • C) * ((C - A) * exp (s • A))) := by
      rw [neg_one_smul]
      noncomm_ring
    rw [harith] at h5
    exact h5
  -- Continuity of the integrand, at the operator level.
  have hcontA : Continuous fun s : ℝ => exp (s • A) :=
    (differentiable_exp_smul_const (𝕂 := ℝ) A).continuous
  have hcontC : Continuous fun s : ℝ => exp ((t - s) • C) :=
    ((differentiable_exp_smul_const (𝕂 := ℝ) C).continuous).comp
      (continuous_const.sub continuous_id)
  have hcontI : Continuous fun s : ℝ => exp ((t - s) • C) * ((C - A) * exp (s • A)) :=
    hcontC.mul (continuous_const.mul hcontA)
  -- Fundamental theorem of calculus for the interpolating product.
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
      (f := fun u : ℝ => exp ((t - u) • C) * exp (u • A))
      (f' := fun s : ℝ => -(exp ((t - s) • C) * ((C - A) * exp (s • A))))
      (fun s _ => key s) ((hcontI.neg).intervalIntegrable 0 t)
  have hop : (∫ s in (0:ℝ)..t, exp ((t - s) • C) * ((C - A) * exp (s • A)))
      = exp (t • C) - exp (t • A) := by
    rw [intervalIntegral.integral_neg] at hFTC
    have h' := neg_eq_iff_eq_neg.mp hFTC
    rw [h']
    simp [NormedSpace.exp_zero]
  -- Apply the operator identity to the vector `x`.
  calc exp (t • C) x - exp (t • A) x
      = (exp (t • C) - exp (t • A)) x := by simp
    _ = (∫ s in (0:ℝ)..t, exp ((t - s) • C) * ((C - A) * exp (s • A))) x := by rw [hop]
    _ = ∫ s in (0:ℝ)..t, (exp ((t - s) • C) * ((C - A) * exp (s • A))) x :=
        ContinuousLinearMap.intervalIntegral_apply (hcontI.intervalIntegrable 0 t) x
    _ = ∫ s in (0:ℝ)..t, exp ((t - s) • C) ((C - A) (exp (s • A) x)) := by
        refine intervalIntegral.integral_congr fun s _ => ?_
        simp

/-- L2, the Duhamel norm bound: if the flow of `C` is isometric, the norm of
the difference of the flows is bounded by the integral of the interaction
norm. -/
theorem duhamel_norm_bound (A C : E →L[ℂ] E) (x : E) (t : ℝ) (ht : 0 ≤ t)
    (hC_iso : ∀ (s : ℝ) (v : E), ‖exp (s • C) v‖ = ‖v‖) :
    ‖exp (t • C) x - exp (t • A) x‖
      ≤ ∫ s in (0:ℝ)..t, ‖(C - A) (exp (s • A) x)‖ := by
  rw [duhamel_identity A C x t ht]
  refine (intervalIntegral.norm_integral_le_integral_norm ht).trans_eq ?_
  refine intervalIntegral.integral_congr fun s _ => ?_
  exact hC_iso (t - s) ((C - A) (exp (s • A) x))

/-- L3, skew generators are isometric: if `re ⟪A v, v⟫ = 0` for every `v`, the
flow of `A` conserves the norm. -/
theorem isometry_exp_of_skew (A : E →L[ℂ] E)
    (hA : ∀ v : E, RCLike.re (inner ℂ (A v) v) = 0) (s : ℝ) (v : E) :
    ‖exp (s • A) v‖ = ‖v‖ := by
  -- The squared norm has vanishing derivative along the flow.
  have hderiv : ∀ r : ℝ,
      HasDerivAt (fun r' : ℝ => (inner ℂ (exp (r' • A) v) (exp (r' • A) v)).re) 0 r := by
    intro r
    have hu : HasDerivAt (fun r' : ℝ => exp (r' • A) v) (A (exp (r • A) v)) r :=
      hasDerivAt_exp_smul_apply A v r
    have hinner := hu.inner ℂ hu
    have hre := Complex.reCLM.hasFDerivAt.comp_hasDerivAt r hinner
    have hzero : (inner ℂ (exp (r • A) v) (A (exp (r • A) v))
        + inner ℂ (A (exp (r • A) v)) (exp (r • A) v)).re = 0 := by
      have h1 : inner ℂ (exp (r • A) v) (A (exp (r • A) v))
          = (starRingEnd ℂ) (inner ℂ (A (exp (r • A) v)) (exp (r • A) v)) :=
        (inner_conj_symm (exp (r • A) v) (A (exp (r • A) v))).symm
      have h2 : (inner ℂ (A (exp (r • A) v)) (exp (r • A) v)).re = 0 := by
        have h := hA (exp (r • A) v)
        rwa [RCLike.re_to_complex] at h
      rw [Complex.add_re, h1, Complex.conj_re, h2, add_zero]
    simpa [Function.comp_def, Complex.reCLM_apply, hzero] using hre
  -- Hence the squared norm is constant.
  have hconst := is_const_of_deriv_eq_zero
      (f := fun r' : ℝ => (inner ℂ (exp (r' • A) v) (exp (r' • A) v)).re)
      (fun r => (hderiv r).differentiableAt) (fun r => (hderiv r).deriv) s 0
  have h0 : exp ((0 : ℝ) • A) v = v := by
    simp [NormedSpace.exp_zero]
  have e1 : (inner ℂ (exp (s • A) v) (exp (s • A) v)).re = ‖exp (s • A) v‖ ^ 2 := by
    have h := inner_self_eq_norm_sq (𝕜 := ℂ) (exp (s • A) v)
    rwa [RCLike.re_to_complex] at h
  have e2 : (inner ℂ v v).re = ‖v‖ ^ 2 := by
    have h := inner_self_eq_norm_sq (𝕜 := ℂ) v
    rwa [RCLike.re_to_complex] at h
  have hconst' : (inner ℂ (exp (s • A) v) (exp (s • A) v)).re
      = (inner ℂ (exp ((0:ℝ) • A) v) (exp ((0:ℝ) • A) v)).re := hconst
  rw [h0] at hconst'
  have hsq : ‖exp (s • A) v‖ ^ 2 = ‖v‖ ^ 2 := by
    rw [← e1, ← e2]
    exact hconst'
  have h := congrArg Real.sqrt hsq
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _)] at h

omit [CompleteSpace E] in
/-- L4, self-adjoint packaging: for a symmetric operator `H` the generator
`(-I) • H` has purely imaginary quadratic form. -/
theorem skew_of_selfAdjoint (H : E →L[ℂ] E)
    (hH : ∀ v w : E, inner ℂ (H v) w = inner ℂ v (H w)) (v : E) :
    RCLike.re (inner ℂ (((-Complex.I) • H) v) v) = 0 := by
  have hsym : (starRingEnd ℂ) (inner ℂ (H v) v) = inner ℂ (H v) v := by
    nth_rewrite 2 [hH v v]
    exact inner_conj_symm v (H v)
  have him : (inner ℂ (H v) v).im = 0 := Complex.conj_eq_iff_im.mp hsym
  simp only [smul_apply, inner_smul_left, map_neg, Complex.conj_I, neg_neg,
    RCLike.re_to_complex, Complex.mul_re, Complex.I_re, Complex.I_im, him]
  ring

omit [CompleteSpace E] in
/-- The symmetric combination `H + λ V` of two symmetric operators, with a
real coupling `λ`. -/
lemma isSymm_add_real_smul (H V : E →L[ℂ] E)
    (hH : ∀ v w : E, inner ℂ (H v) w = inner ℂ v (H w))
    (hV : ∀ v w : E, inner ℂ (V v) w = inner ℂ v (V w)) (lam : ℝ) (v w : E) :
    inner ℂ ((H + (lam : ℂ) • V) v) w = inner ℂ v ((H + (lam : ℂ) • V) w) := by
  simp only [add_apply, smul_apply, inner_add_left, inner_add_right,
    inner_smul_left, inner_smul_right, Complex.conj_ofReal]
  rw [hH v w, hV v w]

omit [CompleteSpace E] in
/-- The real-time flow `exp (s • ((-I) • B))` matches the physics form
`exp ((-I s) • B)`. -/
lemma exp_real_smul_neg_I_smul (B : E →L[ℂ] E) (s : ℝ) :
    exp (s • ((-Complex.I) • B)) = exp ((-Complex.I * s) • B) := by
  congr 1
  rw [← smul_one_smul ℂ s ((-Complex.I) • B), Complex.real_smul, mul_one, smul_smul,
    mul_comm ((s : ℂ)) (-Complex.I)]

/-- L5, the phase-free abstract form of manuscript Eq. (df-duhamel), whose E₀-phase corollaries live in `Cvdv.EnergyShift` and `Cvdv.FiniteDuffing`: the error of dropping the coupling term
`λ V` from the generator is bounded by the integral of the interaction norm
along the unperturbed evolution. -/
theorem duhamel_rwa_bound (H V : E →L[ℂ] E)
    (hH : ∀ v w : E, inner ℂ (H v) w = inner ℂ v (H w))
    (hV : ∀ v w : E, inner ℂ (V v) w = inner ℂ v (V w))
    (lam : ℝ) (psi0 : E) (t : ℝ) (ht : 0 ≤ t) :
    ‖exp ((-Complex.I * t) • (H + (lam : ℂ) • V)) psi0
        - exp ((-Complex.I * t) • H) psi0‖
      ≤ |lam| * ∫ s in (0:ℝ)..t, ‖V (exp ((-Complex.I * s) • H) psi0)‖ := by
  have hCiso : ∀ (s : ℝ) (v : E),
      ‖exp (s • ((-Complex.I) • (H + (lam : ℂ) • V))) v‖ = ‖v‖ :=
    fun s v => isometry_exp_of_skew _
      (skew_of_selfAdjoint _ (isSymm_add_real_smul H V hH hV lam)) s v
  have hb := duhamel_norm_bound ((-Complex.I) • H)
      ((-Complex.I) • (H + (lam : ℂ) • V)) psi0 t ht hCiso
  have hCA : (-Complex.I) • (H + (lam : ℂ) • V) - (-Complex.I) • H
      = (-Complex.I * (lam : ℂ)) • V := by
    rw [← smul_sub, add_sub_cancel_left, smul_smul]
  calc ‖exp ((-Complex.I * t) • (H + (lam : ℂ) • V)) psi0
        - exp ((-Complex.I * t) • H) psi0‖
      = ‖exp (t • ((-Complex.I) • (H + (lam : ℂ) • V))) psi0
          - exp (t • ((-Complex.I) • H)) psi0‖ := by
        rw [exp_real_smul_neg_I_smul, exp_real_smul_neg_I_smul]
    _ ≤ ∫ s in (0:ℝ)..t,
          ‖((-Complex.I) • (H + (lam : ℂ) • V) - (-Complex.I) • H)
            (exp (s • ((-Complex.I) • H)) psi0)‖ := hb
    _ = ∫ s in (0:ℝ)..t, |lam| * ‖V (exp ((-Complex.I * s) • H) psi0)‖ := by
        refine intervalIntegral.integral_congr fun s _ => ?_
        rw [hCA, smul_apply, norm_smul, exp_real_smul_neg_I_smul]
        congr 1
        rw [norm_mul, norm_neg, Complex.norm_I, one_mul, Complex.norm_real,
          Real.norm_eq_abs]
    _ = |lam| * ∫ s in (0:ℝ)..t, ‖V (exp ((-Complex.I * s) • H) psi0)‖ :=
        intervalIntegral.integral_const_mul _ _

/-- L6, the phase-free abstract form of manuscript Eq. (df-bound), whose E₀-phase corollaries live in `Cvdv.EnergyShift` and `Cvdv.FiniteDuffing`: the infidelity of the perturbed and
unperturbed evolutions of a normalized state is bounded by the square of the
Duhamel integral. -/
theorem duhamel_infidelity_bound (H V : E →L[ℂ] E)
    (hH : ∀ v w : E, inner ℂ (H v) w = inner ℂ v (H w))
    (hV : ∀ v w : E, inner ℂ (V v) w = inner ℂ v (V w))
    (lam : ℝ) (psi0 : E) (t : ℝ) (ht : 0 ≤ t) (hpsi : ‖psi0‖ = 1) :
    1 - ‖inner ℂ (exp ((-Complex.I * t) • H) psi0)
          (exp ((-Complex.I * t) • (H + (lam : ℂ) • V)) psi0)‖ ^ 2
      ≤ (|lam| * ∫ s in (0:ℝ)..t, ‖V (exp ((-Complex.I * s) • H) psi0)‖) ^ 2 := by
  have hu : ‖exp ((-Complex.I * t) • H) psi0‖ = 1 := by
    rw [← exp_real_smul_neg_I_smul,
      isometry_exp_of_skew _ (skew_of_selfAdjoint H hH) t psi0, hpsi]
  have hv : ‖exp ((-Complex.I * t) • (H + (lam : ℂ) • V)) psi0‖ = 1 := by
    rw [← exp_real_smul_neg_I_smul,
      isometry_exp_of_skew _
        (skew_of_selfAdjoint _ (isSymm_add_real_smul H V hH hV lam)) t psi0, hpsi]
  have h1 := one_sub_fidelity_le_norm_sub_sq
      (exp ((-Complex.I * t) • H) psi0)
      (exp ((-Complex.I * t) • (H + (lam : ℂ) • V)) psi0) hu hv
  have h2 : ‖exp ((-Complex.I * t) • H) psi0
      - exp ((-Complex.I * t) • (H + (lam : ℂ) • V)) psi0‖
      ≤ |lam| * ∫ s in (0:ℝ)..t, ‖V (exp ((-Complex.I * s) • H) psi0)‖ := by
    rw [norm_sub_rev]
    exact duhamel_rwa_bound H V hH hV lam psi0 t ht
  exact h1.trans (pow_le_pow_left₀ (norm_nonneg _) h2 2)

end Cvdv
