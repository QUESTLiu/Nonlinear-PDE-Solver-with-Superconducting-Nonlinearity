import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Analysis.Normed.Algebra.Exponential
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Tactic.NoncommRing
import Cvdv.Duhamel
import Cvdv.Telescoping

/-!
# First-order Trotter product-formula error bound

This file formalizes the first-order (Lie) product-formula error estimate that
backs the manuscript's cost claim for Hamiltonian simulation (Sec. 2.2,
Eq. (eta-C)): a single splitting step of length `t/r` carries an error of order
`‖[H₁, H₂]‖ (t/r)²`, so `r = O(‖[H₁, H₂]‖ t² / δ)` steps reach accuracy `δ`.

The estimates are proved in a real Banach algebra `A` with the normed-algebra
exponential, then specialized to bounded operators on a complex Hilbert space.

* `Cvdv.norm_exp_le_exp_norm` bounds the exponential by the scalar exponential
  of the norm.
* `Cvdv.commutator_exp_smul_integral` is the integral identity for the
  commutator of an element with an exponential,
  `X exp(sY) - exp(sY) X = ∫ u in 0..s, exp(uY) [X, Y] exp((s-u)Y)`.
* `Cvdv.exp_mul_exp_sub_exp_add` is the one-step defect identity,
  `exp X exp Y - exp (X+Y) = ∫ s in 0..1,
    exp(sX) (X exp(sY) - exp(sY) X) exp((1-s)(X+Y))`.
* `Cvdv.norm_exp_mul_exp_sub_exp_add_le` is the classical one-step bound
  `‖exp X exp Y - exp (X+Y)‖ ≤ (1/2) ‖[X, Y]‖ exp(‖X‖ + ‖Y‖)`.
* `Cvdv.norm_trotter_pow_sub_exp_le` is the Trotter theorem in the algebra:
  under the contraction hypotheses for one step, the `r`-step product deviates
  from `exp (X+Y)` by at most `(1/(2r)) ‖[X, Y]‖ exp((‖X‖ + ‖Y‖)/r)`.
* `Cvdv.norm_trotter_pow_sub_exp_le_of_selfAdjoint` is the paper-facing
  corollary for symmetric `H₁`, `H₂` on a complex Hilbert space, with the
  contraction hypotheses discharged by unitarity of the skew exponentials.

The one-step and `r`-step algebra bounds carry a `NormOneClass A` hypothesis,
used only through `‖x ^ n‖ ≤ ‖x‖ ^ n` in the series bound for the exponential.
Bounded operators on a nontrivial space satisfy it, and the Hilbert corollary
absorbs the trivial-space case by a separate (degenerate) argument, so its
statement carries no extra hypothesis.
-/

open NormedSpace

open scoped Nat

namespace Cvdv

section BanachAlgebra

variable {A : Type*} [NormedRing A] [NormedAlgebra ℝ A] [CompleteSpace A]

omit [CompleteSpace A] in
/-- Series bound for the normed-algebra exponential,
`‖exp x‖ ≤ Real.exp ‖x‖`.  Mathlib has this for `ℂ` but not for a general
Banach algebra, so it is proved here from the defining series. -/
theorem norm_exp_le_exp_norm [NormOneClass A] (x : A) : ‖exp x‖ ≤ Real.exp ‖x‖ := by
  have h1 : exp x = ∑' n : ℕ, ((n ! : ℝ))⁻¹ • x ^ n := congrFun (exp_eq_tsum (𝕂 := ℝ)) x
  have hs1 : Summable fun n : ℕ => ‖((n ! : ℝ))⁻¹ • x ^ n‖ :=
    norm_expSeries_summable' (𝕂 := ℝ) x
  have hs2 : Summable fun n : ℕ => ((n ! : ℝ))⁻¹ * ‖x‖ ^ n := by
    simpa [div_eq_inv_mul] using Real.summable_pow_div_factorial ‖x‖
  have h2 : Real.exp ‖x‖ = ∑' n : ℕ, ((n ! : ℝ))⁻¹ * ‖x‖ ^ n := by
    rw [Real.exp_eq_exp_ℝ]
    simpa [smul_eq_mul] using congrFun (exp_eq_tsum (𝕂 := ℝ) (𝔸 := ℝ)) ‖x‖
  rw [h1, h2]
  refine (norm_tsum_le_tsum_norm hs1).trans (Summable.tsum_le_tsum (fun n => ?_) hs1 hs2)
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity : (0:ℝ) ≤ ((n ! : ℝ))⁻¹)]
  exact mul_le_mul_of_nonneg_left (norm_pow_le x n) (by positivity)

/-- The exponential of a natural multiple is the power of the exponential.
Mathlib's `NormedSpace.exp_nsmul` asks for a `NormedAlgebra ℚ` structure that
operator algebras do not carry by instance, so the statement is rederived from
`exp_add_of_commute_of_mem_ball` over `ℝ`. -/
theorem exp_nsmul_eq_pow (n : ℕ) (z : A) : exp (n • z) = exp z ^ n := by
  induction n with
  | zero => rw [zero_smul, pow_zero, exp_zero]
  | succ n ih =>
    rw [succ_nsmul, pow_succ, ← ih]
    exact exp_add_of_commute_of_mem_ball (𝕂 := ℝ) ((Commute.refl z).smul_left n)
      ((expSeries_radius_eq_top ℝ A).symm ▸ edist_lt_top _ _)
      ((expSeries_radius_eq_top ℝ A).symm ▸ edist_lt_top _ _)

set_option linter.unusedVariables false in
/-- T1, the commutator-with-exponential integral identity.  The hypothesis
`hs` matches the intended range of the manuscript argument, although the
identity holds for every `s`. -/
theorem commutator_exp_smul_integral (X Y : A) (s : ℝ) (hs : 0 ≤ s) :
    X * exp (s • Y) - exp (s • Y) * X
      = ∫ u in (0:ℝ)..s, exp (u • Y) * (X * Y - Y * X) * exp ((s - u) • Y) := by
  -- Derivative of the interpolating product `u ↦ exp(uY) X exp((s-u)Y)`.
  have key : ∀ u : ℝ,
      HasDerivAt (fun v : ℝ => exp (v • Y) * X * exp ((s - v) • Y))
        (exp (u • Y) * (Y * X - X * Y) * exp ((s - u) • Y)) u := by
    intro u
    have h1 : HasDerivAt (fun v : ℝ => s - v) (-1) u := (hasDerivAt_id u).const_sub s
    have h2 : HasDerivAt (fun v : ℝ => exp (v • Y)) (exp ((s - u) • Y) * Y) (s - u) :=
      hasDerivAt_exp_smul_const Y (s - u)
    have h3 : HasDerivAt (fun v : ℝ => exp ((s - v) • Y))
        ((-1 : ℝ) • (exp ((s - u) • Y) * Y)) u := by
      simpa [Function.comp_def] using h2.scomp u h1
    have h4 : HasDerivAt (fun v : ℝ => exp (v • Y)) (exp (u • Y) * Y) u :=
      hasDerivAt_exp_smul_const Y u
    have h5 : HasDerivAt (fun v : ℝ => exp (v • Y) * X) (exp (u • Y) * Y * X) u :=
      h4.mul_const X
    have h6 := h5.mul h3
    have harith : exp (u • Y) * Y * X * exp ((s - u) • Y)
        + exp (u • Y) * X * ((-1 : ℝ) • (exp ((s - u) • Y) * Y))
        = exp (u • Y) * (Y * X - X * Y) * exp ((s - u) • Y) := by
      have hc : exp ((s - u) • Y) * Y = Y * exp ((s - u) • Y) :=
        (((Commute.refl Y).smul_left (s - u)).exp_left).eq
      rw [neg_one_smul, hc]
      noncomm_ring
    rw [harith] at h6
    exact h6
  -- Interval integrability of the integrand.
  have hcY : Continuous fun u : ℝ => exp (u • Y) :=
    (differentiable_exp_smul_const (𝕂 := ℝ) Y).continuous
  have hcR : Continuous fun u : ℝ => exp ((s - u) • Y) :=
    hcY.comp (continuous_const.sub continuous_id)
  have hcont : Continuous fun u : ℝ =>
      exp (u • Y) * (Y * X - X * Y) * exp ((s - u) • Y) :=
    (hcY.mul continuous_const).mul hcR
  -- Fundamental theorem of calculus.
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
      (f := fun v : ℝ => exp (v • Y) * X * exp ((s - v) • Y))
      (f' := fun u : ℝ => exp (u • Y) * (Y * X - X * Y) * exp ((s - u) • Y))
      (fun u _ => key u) (hcont.intervalIntegrable 0 s)
  have hint : (∫ u in (0:ℝ)..s, exp (u • Y) * (X * Y - Y * X) * exp ((s - u) • Y))
      = -(∫ u in (0:ℝ)..s, exp (u • Y) * (Y * X - X * Y) * exp ((s - u) • Y)) := by
    rw [← intervalIntegral.integral_neg]
    refine intervalIntegral.integral_congr fun u _ => ?_
    noncomm_ring
  rw [hint, hFTC]
  simp [NormedSpace.exp_zero]

/-- T2, the one-step defect integral identity for the Lie product formula. -/
theorem exp_mul_exp_sub_exp_add (X Y : A) :
    exp X * exp Y - exp (X + Y)
      = ∫ s in (0:ℝ)..1,
          exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y)) := by
  -- Derivative of the interpolation `s ↦ exp(sX) exp(sY) exp((1-s)(X+Y))`.
  have key : ∀ s : ℝ,
      HasDerivAt (fun v : ℝ => exp (v • X) * exp (v • Y) * exp ((1 - v) • (X + Y)))
        (exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y))) s := by
    intro s
    have h1 : HasDerivAt (fun v : ℝ => 1 - v) (-1) s := (hasDerivAt_id s).const_sub 1
    have h2 : HasDerivAt (fun v : ℝ => exp (v • (X + Y)))
        (exp ((1 - s) • (X + Y)) * (X + Y)) (1 - s) :=
      hasDerivAt_exp_smul_const (X + Y) (1 - s)
    have h3 : HasDerivAt (fun v : ℝ => exp ((1 - v) • (X + Y)))
        ((-1 : ℝ) • (exp ((1 - s) • (X + Y)) * (X + Y))) s := by
      simpa [Function.comp_def] using h2.scomp s h1
    have h4 : HasDerivAt (fun v : ℝ => exp (v • X)) (X * exp (s • X)) s :=
      hasDerivAt_exp_smul_const' X s
    have h5 : HasDerivAt (fun v : ℝ => exp (v • Y)) (exp (s • Y) * Y) s :=
      hasDerivAt_exp_smul_const Y s
    have h6 : HasDerivAt (fun v : ℝ => exp (v • X) * exp (v • Y) * exp ((1 - v) • (X + Y)))
        ((X * exp (s • X) * exp (s • Y) + exp (s • X) * (exp (s • Y) * Y))
            * exp ((1 - s) • (X + Y))
          + exp (s • X) * exp (s • Y) * ((-1 : ℝ) • (exp ((1 - s) • (X + Y)) * (X + Y)))) s :=
      (h4.mul h5).mul h3
    have harith :
        (X * exp (s • X) * exp (s • Y) + exp (s • X) * (exp (s • Y) * Y))
            * exp ((1 - s) • (X + Y))
          + exp (s • X) * exp (s • Y) * ((-1 : ℝ) • (exp ((1 - s) • (X + Y)) * (X + Y)))
        = exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y)) := by
      have hc1 : X * exp (s • X) = exp (s • X) * X :=
        (((Commute.refl X).smul_right s).exp_right).eq
      have hc2 : exp ((1 - s) • (X + Y)) * (X + Y) = (X + Y) * exp ((1 - s) • (X + Y)) :=
        (((Commute.refl (X + Y)).smul_left (1 - s)).exp_left).eq
      rw [neg_one_smul, hc2, hc1]
      noncomm_ring
    rw [harith] at h6
    exact h6
  -- Interval integrability of the integrand.
  have hcX : Continuous fun s : ℝ => exp (s • X) :=
    (differentiable_exp_smul_const (𝕂 := ℝ) X).continuous
  have hcY : Continuous fun s : ℝ => exp (s • Y) :=
    (differentiable_exp_smul_const (𝕂 := ℝ) Y).continuous
  have hcR : Continuous fun s : ℝ => exp ((1 - s) • (X + Y)) :=
    ((differentiable_exp_smul_const (𝕂 := ℝ) (X + Y)).continuous).comp
      (continuous_const.sub continuous_id)
  have hcont : Continuous fun s : ℝ =>
      exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y)) :=
    (hcX.mul ((continuous_const.mul hcY).sub (hcY.mul continuous_const))).mul hcR
  -- Fundamental theorem of calculus over `[0, 1]`.
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
      (f := fun v : ℝ => exp (v • X) * exp (v • Y) * exp ((1 - v) • (X + Y)))
      (f' := fun s : ℝ =>
        exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y)))
      (fun s _ => key s) (hcont.intervalIntegrable 0 1)
  rw [hFTC]
  simp [NormedSpace.exp_zero]

/-- Pointwise norm bound for the inner commutator factor, obtained from T1 by
bounding the integrand by the constant `Real.exp (s ‖Y‖) ‖[X, Y]‖`. -/
theorem norm_commutator_exp_smul_le [NormOneClass A] (X Y : A) (s : ℝ) (hs : 0 ≤ s) :
    ‖X * exp (s • Y) - exp (s • Y) * X‖
      ≤ s * Real.exp (s * ‖Y‖) * ‖X * Y - Y * X‖ := by
  rw [commutator_exp_smul_integral X Y s hs]
  have hb : ∀ u ∈ Set.uIoc (0:ℝ) s,
      ‖exp (u • Y) * (X * Y - Y * X) * exp ((s - u) • Y)‖
        ≤ Real.exp (s * ‖Y‖) * ‖X * Y - Y * X‖ := by
    intro u hu
    rw [Set.uIoc_of_le hs] at hu
    obtain ⟨hu0, hus⟩ := hu
    have h1 : ‖exp (u • Y)‖ ≤ Real.exp (u * ‖Y‖) := by
      refine (norm_exp_le_exp_norm _).trans_eq ?_
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hu0.le]
    have h2 : ‖exp ((s - u) • Y)‖ ≤ Real.exp ((s - u) * ‖Y‖) := by
      refine (norm_exp_le_exp_norm _).trans_eq ?_
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by linarith : (0:ℝ) ≤ s - u)]
    calc ‖exp (u • Y) * (X * Y - Y * X) * exp ((s - u) • Y)‖
        ≤ ‖exp (u • Y) * (X * Y - Y * X)‖ * ‖exp ((s - u) • Y)‖ := norm_mul_le _ _
      _ ≤ (‖exp (u • Y)‖ * ‖X * Y - Y * X‖) * ‖exp ((s - u) • Y)‖ :=
          mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _)
      _ ≤ (Real.exp (u * ‖Y‖) * ‖X * Y - Y * X‖) * Real.exp ((s - u) * ‖Y‖) :=
          mul_le_mul (mul_le_mul_of_nonneg_right h1 (norm_nonneg _)) h2 (norm_nonneg _)
            (mul_nonneg (Real.exp_pos _).le (norm_nonneg _))
      _ = Real.exp (s * ‖Y‖) * ‖X * Y - Y * X‖ := by
          rw [mul_right_comm, ← Real.exp_add,
            show u * ‖Y‖ + (s - u) * ‖Y‖ = s * ‖Y‖ from by ring]
  refine (intervalIntegral.norm_integral_le_of_norm_le_const hb).trans (le_of_eq ?_)
  rw [sub_zero, abs_of_nonneg hs]
  ring

/-- T3, the classical one-step estimate for the Lie product formula. -/
theorem norm_exp_mul_exp_sub_exp_add_le [NormOneClass A] (X Y : A) :
    ‖exp X * exp Y - exp (X + Y)‖
      ≤ (1 / 2) * ‖X * Y - Y * X‖ * Real.exp (‖X‖ + ‖Y‖) := by
  rw [exp_mul_exp_sub_exp_add X Y]
  have hcX : Continuous fun s : ℝ => exp (s • X) :=
    (differentiable_exp_smul_const (𝕂 := ℝ) X).continuous
  have hcY : Continuous fun s : ℝ => exp (s • Y) :=
    (differentiable_exp_smul_const (𝕂 := ℝ) Y).continuous
  have hcR : Continuous fun s : ℝ => exp ((1 - s) • (X + Y)) :=
    ((differentiable_exp_smul_const (𝕂 := ℝ) (X + Y)).continuous).comp
      (continuous_const.sub continuous_id)
  have hg : Continuous fun s : ℝ =>
      exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y)) :=
    (hcX.mul ((continuous_const.mul hcY).sub (hcY.mul continuous_const))).mul hcR
  -- Pointwise bound `s ↦ s ‖[X, Y]‖ exp(‖X‖ + ‖Y‖)` on `[0, 1]`; the three
  -- exponential factors merge because the exponents sum to `‖X‖ + ‖Y‖`.
  have hbound : ∀ s ∈ Set.Icc (0:ℝ) 1,
      ‖exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y))‖
        ≤ s * (‖X * Y - Y * X‖ * Real.exp (‖X‖ + ‖Y‖)) := by
    intro s hs
    obtain ⟨hs0, hs1⟩ := hs
    have h1 : ‖exp (s • X)‖ ≤ Real.exp (s * ‖X‖) := by
      refine (norm_exp_le_exp_norm _).trans_eq ?_
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hs0]
    have h2 := norm_commutator_exp_smul_le X Y s hs0
    have h3 : ‖exp ((1 - s) • (X + Y))‖ ≤ Real.exp ((1 - s) * (‖X‖ + ‖Y‖)) := by
      refine (norm_exp_le_exp_norm _).trans ?_
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by linarith : (0:ℝ) ≤ 1 - s)]
      exact Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left (norm_add_le X Y) (by linarith))
    have hmerge : Real.exp (s * ‖X‖) * Real.exp (s * ‖Y‖)
        * Real.exp ((1 - s) * (‖X‖ + ‖Y‖)) = Real.exp (‖X‖ + ‖Y‖) := by
      rw [← Real.exp_add, ← Real.exp_add]
      congr 1
      ring
    calc ‖exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y))‖
        ≤ ‖exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X)‖ * ‖exp ((1 - s) • (X + Y))‖ :=
          norm_mul_le _ _
      _ ≤ (‖exp (s • X)‖ * ‖X * exp (s • Y) - exp (s • Y) * X‖) * ‖exp ((1 - s) • (X + Y))‖ :=
          mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _)
      _ ≤ (Real.exp (s * ‖X‖) * (s * Real.exp (s * ‖Y‖) * ‖X * Y - Y * X‖))
            * Real.exp ((1 - s) * (‖X‖ + ‖Y‖)) := by
          refine mul_le_mul (mul_le_mul h1 h2 (norm_nonneg _) (Real.exp_pos _).le) h3
            (norm_nonneg _) ?_
          exact mul_nonneg (Real.exp_pos _).le
            (mul_nonneg (mul_nonneg hs0 (Real.exp_pos _).le) (norm_nonneg _))
      _ = s * (‖X * Y - Y * X‖ * Real.exp (‖X‖ + ‖Y‖)) := by
          rw [show Real.exp (s * ‖X‖) * (s * Real.exp (s * ‖Y‖) * ‖X * Y - Y * X‖)
                * Real.exp ((1 - s) * (‖X‖ + ‖Y‖))
              = s * ‖X * Y - Y * X‖ * (Real.exp (s * ‖X‖) * Real.exp (s * ‖Y‖)
                * Real.exp ((1 - s) * (‖X‖ + ‖Y‖))) from by ring, hmerge]
          ring
  calc ‖∫ s in (0:ℝ)..1,
        exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y))‖
      ≤ ∫ s in (0:ℝ)..1,
          ‖exp (s • X) * (X * exp (s • Y) - exp (s • Y) * X) * exp ((1 - s) • (X + Y))‖ :=
        intervalIntegral.norm_integral_le_integral_norm zero_le_one
    _ ≤ ∫ s in (0:ℝ)..1, s * (‖X * Y - Y * X‖ * Real.exp (‖X‖ + ‖Y‖)) :=
        intervalIntegral.integral_mono_on zero_le_one (hg.norm.intervalIntegrable 0 1)
          ((continuous_id.mul continuous_const).intervalIntegrable 0 1) hbound
    _ = (1 / 2) * ‖X * Y - Y * X‖ * Real.exp (‖X‖ + ‖Y‖) := by
        rw [intervalIntegral.integral_mul_const, integral_id]
        ring

/-- T4, the first-order Trotter bound in a Banach algebra.  The hypotheses
`hS` and `hT` state that one exact step and one split step are contractions,
which holds for the unitary exponentials of the manuscript.  Combined with the
telescoping bound, `r` steps of length `1/r` give the `1/(2r)` error decay of
manuscript Eq. (eta-C). -/
theorem norm_trotter_pow_sub_exp_le [NormOneClass A] (X Y : A) (r : ℕ) (hr : 1 ≤ r)
    (hS : ‖exp ((r : ℝ)⁻¹ • (X + Y))‖ ≤ 1)
    (hT : ‖exp ((r : ℝ)⁻¹ • X) * exp ((r : ℝ)⁻¹ • Y)‖ ≤ 1) :
    ‖(exp ((r : ℝ)⁻¹ • X) * exp ((r : ℝ)⁻¹ • Y)) ^ r - exp (X + Y)‖
      ≤ (1 / (2 * r)) * ‖X * Y - Y * X‖ * Real.exp ((‖X‖ + ‖Y‖) / r) := by
  have hr0 : (r : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.one_le_iff_ne_zero.mp hr)
  have hrinv : (0:ℝ) ≤ (r : ℝ)⁻¹ := by positivity
  -- The exact evolution is the `r`-th power of the exact step.
  have hpow : exp (X + Y) = exp ((r : ℝ)⁻¹ • (X + Y)) ^ r := by
    rw [← exp_nsmul_eq_pow]
    congr 1
    rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul, mul_inv_cancel₀ hr0, one_smul]
  have hsum : (r : ℝ)⁻¹ • X + (r : ℝ)⁻¹ • Y = (r : ℝ)⁻¹ • (X + Y) := (smul_add _ _ _).symm
  -- One-step estimate at the rescaled generators.
  have hstep := norm_exp_mul_exp_sub_exp_add_le ((r : ℝ)⁻¹ • X) ((r : ℝ)⁻¹ • Y)
  rw [hsum] at hstep
  have hcomm : ((r : ℝ)⁻¹ • X) * ((r : ℝ)⁻¹ • Y) - ((r : ℝ)⁻¹ • Y) * ((r : ℝ)⁻¹ • X)
      = ((r : ℝ)⁻¹ * (r : ℝ)⁻¹) • (X * Y - Y * X) := by
    rw [smul_mul_smul_comm, smul_mul_smul_comm, ← smul_sub]
  have hX' : ‖(r : ℝ)⁻¹ • X‖ = (r : ℝ)⁻¹ * ‖X‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hrinv]
  have hY' : ‖(r : ℝ)⁻¹ • Y‖ = (r : ℝ)⁻¹ * ‖Y‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hrinv]
  rw [hcomm, hX', hY'] at hstep
  have hKnorm : ‖((r : ℝ)⁻¹ * (r : ℝ)⁻¹) • (X * Y - Y * X)‖
      = (r : ℝ)⁻¹ * (r : ℝ)⁻¹ * ‖X * Y - Y * X‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  rw [hKnorm] at hstep
  have harg : (r : ℝ)⁻¹ * ‖X‖ + (r : ℝ)⁻¹ * ‖Y‖ = (‖X‖ + ‖Y‖) / r := by ring
  rw [harg] at hstep
  calc ‖(exp ((r : ℝ)⁻¹ • X) * exp ((r : ℝ)⁻¹ • Y)) ^ r - exp (X + Y)‖
      = ‖(exp ((r : ℝ)⁻¹ • X) * exp ((r : ℝ)⁻¹ • Y)) ^ r
          - exp ((r : ℝ)⁻¹ • (X + Y)) ^ r‖ := by rw [hpow]
    _ ≤ r * ‖exp ((r : ℝ)⁻¹ • X) * exp ((r : ℝ)⁻¹ • Y) - exp ((r : ℝ)⁻¹ • (X + Y))‖ :=
        norm_pow_sub_pow_le _ _ hT hS r
    _ ≤ r * ((1 / 2) * ((r : ℝ)⁻¹ * (r : ℝ)⁻¹ * ‖X * Y - Y * X‖)
          * Real.exp ((‖X‖ + ‖Y‖) / r)) :=
        mul_le_mul_of_nonneg_left hstep (Nat.cast_nonneg r)
    _ = (1 / (2 * r)) * ‖X * Y - Y * X‖ * Real.exp ((‖X‖ + ‖Y‖) / r) := by
        field_simp

end BanachAlgebra

section HilbertSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]

omit [CompleteSpace E] in
/-- Alignment of the real rescaling with the complex generator, in the style of
`Cvdv.exp_real_smul_neg_I_smul`. -/
lemma real_smul_complex_smul (a : ℝ) (c : ℂ) (B : E →L[ℂ] E) :
    a • (c • B) = (c * a) • B := by
  rw [← smul_one_smul ℂ a (c • B), Complex.real_smul, mul_one, smul_smul, mul_comm (a : ℂ) c]

/-- The exponential of `(-i u) H` with `H` symmetric and `u` real is a
contraction in operator norm, by the isometry machinery of `Duhamel.lean`. -/
lemma norm_exp_neg_I_smul_le_one (H : E →L[ℂ] E)
    (hH : ∀ v w : E, inner ℂ (H v) w = inner ℂ v (H w)) (u : ℝ) :
    ‖exp ((-Complex.I * u) • H)‖ ≤ 1 := by
  rw [← exp_real_smul_neg_I_smul]
  refine ContinuousLinearMap.opNorm_le_bound _ zero_le_one fun v => ?_
  calc ‖exp (u • ((-Complex.I) • H)) v‖
      = ‖v‖ := isometry_exp_of_skew _ (skew_of_selfAdjoint H hH) u v
    _ ≤ 1 * ‖v‖ := le_of_eq (one_mul _).symm

/-- T5, the paper-facing Trotter bound (manuscript Sec. 2.2, Eq. (eta-C)):
for symmetric `H₁`, `H₂` on a complex Hilbert space, `r` first-order Trotter
steps for time `t` deviate from the exact evolution by at most
`(t²/(2r)) ‖[H₁, H₂]‖ exp(|t| (‖H₁‖ + ‖H₂‖)/r)`. -/
theorem norm_trotter_pow_sub_exp_le_of_selfAdjoint (H1 H2 : E →L[ℂ] E)
    (hH1 : ∀ v w : E, inner ℂ (H1 v) w = inner ℂ v (H1 w))
    (hH2 : ∀ v w : E, inner ℂ (H2 v) w = inner ℂ v (H2 w))
    (t : ℝ) (r : ℕ) (hr : 1 ≤ r) :
    ‖(exp ((-Complex.I * t * (r : ℝ)⁻¹) • H1)
        * exp ((-Complex.I * t * (r : ℝ)⁻¹) • H2)) ^ r
        - exp ((-Complex.I * t) • (H1 + H2))‖
      ≤ (t ^ 2 / (2 * r)) * ‖H1 * H2 - H2 * H1‖
          * Real.exp (|t| * (‖H1‖ + ‖H2‖) / r) := by
  rcases subsingleton_or_nontrivial E with hE | hE
  · -- On the trivial space every operator vanishes and both sides degenerate.
    haveI : Subsingleton (E →L[ℂ] E) :=
      ⟨fun f g => ContinuousLinearMap.ext fun v => Subsingleton.elim _ _⟩
    have hz : ∀ f : E →L[ℂ] E, ‖f‖ = 0 := fun f => by
      rw [Subsingleton.elim f (0 : E →L[ℂ] E)]; exact norm_zero
    simp only [hz]
    positivity
  · have hsymm : ∀ v w : E, inner ℂ ((H1 + H2) v) w = inner ℂ v ((H1 + H2) w) := by
      intro v w
      simp only [add_apply, inner_add_left, inner_add_right]
      rw [hH1 v w, hH2 v w]
    have hcnorm : ‖-Complex.I * (t : ℂ)‖ = |t| := by
      rw [norm_mul, norm_neg, Complex.norm_I, one_mul, Complex.norm_real, Real.norm_eq_abs]
    have hscalar : (-Complex.I * (t : ℂ)) * (((r : ℝ)⁻¹ : ℝ) : ℂ)
        = -Complex.I * ((t * (r : ℝ)⁻¹ : ℝ) : ℂ) := by
      push_cast
      ring
    have hX := real_smul_complex_smul ((r : ℝ)⁻¹) (-Complex.I * t) H1
    have hY := real_smul_complex_smul ((r : ℝ)⁻¹) (-Complex.I * t) H2
    have hXY : (-Complex.I * t) • H1 + (-Complex.I * t) • H2
        = (-Complex.I * t) • (H1 + H2) := (smul_add _ _ _).symm
    -- Contraction hypotheses from unitarity of the skew exponentials.
    have hS : ‖exp ((r : ℝ)⁻¹ • ((-Complex.I * t) • H1 + (-Complex.I * t) • H2))‖ ≤ 1 := by
      rw [hXY, real_smul_complex_smul, hscalar]
      exact norm_exp_neg_I_smul_le_one (H1 + H2) hsymm _
    have hT : ‖exp ((r : ℝ)⁻¹ • ((-Complex.I * t) • H1))
        * exp ((r : ℝ)⁻¹ • ((-Complex.I * t) • H2))‖ ≤ 1 := by
      refine (norm_mul_le _ _).trans (mul_le_one₀ ?_ (norm_nonneg _) ?_)
      · rw [hX, hscalar]
        exact norm_exp_neg_I_smul_le_one H1 hH1 _
      · rw [hY, hscalar]
        exact norm_exp_neg_I_smul_le_one H2 hH2 _
    have h4 := norm_trotter_pow_sub_exp_le ((-Complex.I * t) • H1) ((-Complex.I * t) • H2)
      r hr hS hT
    rw [hX, hY, hXY] at h4
    -- Scalar bookkeeping for the commutator and the norms.
    have hcomm : ((-Complex.I * t) • H1) * ((-Complex.I * t) • H2)
        - ((-Complex.I * t) • H2) * ((-Complex.I * t) • H1)
        = ((-Complex.I * t) * (-Complex.I * t)) • (H1 * H2 - H2 * H1) := by
      rw [smul_mul_smul_comm, smul_mul_smul_comm, ← smul_sub]
    have hncomm : ‖((-Complex.I * t) • H1) * ((-Complex.I * t) • H2)
        - ((-Complex.I * t) • H2) * ((-Complex.I * t) • H1)‖
        = t ^ 2 * ‖H1 * H2 - H2 * H1‖ := by
      rw [hcomm, norm_smul, norm_mul, hcnorm, abs_mul_abs_self, ← pow_two]
    have hnX : ‖(-Complex.I * t) • H1‖ = |t| * ‖H1‖ := by rw [norm_smul, hcnorm]
    have hnY : ‖(-Complex.I * t) • H2‖ = |t| * ‖H2‖ := by rw [norm_smul, hcnorm]
    rw [hncomm, hnX, hnY] at h4
    have harg : (|t| * ‖H1‖ + |t| * ‖H2‖) / (r : ℝ) = |t| * (‖H1‖ + ‖H2‖) / (r : ℝ) := by
      ring
    rw [harg] at h4
    refine h4.trans (le_of_eq ?_)
    ring

/-- Explicit paper-facing step budget for the first-order product formula.
If the step count makes the exponential correction at most `exp 1` and the
resulting commutator budget is at most `δ`, then the Trotter error is at most `δ`. -/
theorem norm_trotter_pow_sub_exp_le_delta_of_selfAdjoint (H1 H2 : E →L[ℂ] E)
    (hH1 : ∀ v w : E, inner ℂ (H1 v) w = inner ℂ v (H1 w))
    (hH2 : ∀ v w : E, inner ℂ (H2 v) w = inner ℂ v (H2 w))
    (t δ : ℝ) (r : ℕ) (hr : 1 ≤ r) (_hδ : 0 < δ)
    (hscale : |t| * (‖H1‖ + ‖H2‖) ≤ r)
    (hbudget : Real.exp 1 * t ^ 2 * ‖H1 * H2 - H2 * H1‖ ≤ 2 * δ * r) :
    ‖(exp ((-Complex.I * t * (r : ℝ)⁻¹) • H1)
        * exp ((-Complex.I * t * (r : ℝ)⁻¹) • H2)) ^ r
        - exp ((-Complex.I * t) • (H1 + H2))‖ ≤ δ := by
  have hrpos : (0 : ℝ) < r := by exact_mod_cast Nat.zero_lt_of_lt hr
  have hexp : Real.exp (|t| * (‖H1‖ + ‖H2‖) / r) ≤ Real.exp 1 :=
    Real.exp_le_exp.mpr ((div_le_one hrpos).2 hscale)
  have hbudget' :
      Real.exp 1 * t ^ 2 * ‖H1 * H2 - H2 * H1‖ / (2 * r) ≤ δ := by
    rw [div_le_iff₀ (by positivity : (0 : ℝ) < 2 * r)]
    nlinarith [hbudget]
  refine (norm_trotter_pow_sub_exp_le_of_selfAdjoint H1 H2 hH1 hH2 t r hr).trans ?_
  calc
    (t ^ 2 / (2 * r)) * ‖H1 * H2 - H2 * H1‖
          * Real.exp (|t| * (‖H1‖ + ‖H2‖) / r)
        ≤ (t ^ 2 / (2 * r)) * ‖H1 * H2 - H2 * H1‖ * Real.exp 1 := by
            exact mul_le_mul_of_nonneg_left hexp
              (mul_nonneg (div_nonneg (sq_nonneg t) (by positivity)) (norm_nonneg _))
    _ = Real.exp 1 * t ^ 2 * ‖H1 * H2 - H2 * H1‖ / (2 * r) := by ring
    _ ≤ δ := hbudget'

end HilbertSpace

end Cvdv
