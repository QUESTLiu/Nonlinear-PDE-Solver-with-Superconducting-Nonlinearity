import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.Analysis.Calculus.Deriv.Inv

/-!
# Normalized expectations for non-Hermitian evolution

This file proves the finite-dimensional identity used in manuscript
Eq. (eq:lle-normalized-expectation).  The state obeys
`ψ' = -(L + iH) ψ`, where `L` and `H` are symmetric and `O` is bounded and time-independent.
-/

namespace Cvdv

noncomputable section

open Complex
open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]
  [NormedSpace ℝ E]

/-- The normalized expectation of `O` in a nonzero, not necessarily normalized state. -/
def normalizedExpectation (O : E →L[ℂ] E) (ψ : E) : ℂ :=
  ⟪ψ, O ψ⟫_ℂ / ⟪ψ, ψ⟫_ℂ

omit [NormedSpace ℝ E] in
private lemma inner_self_ne_zero {ψ : E} (hψ : ψ ≠ 0) : ⟪ψ, ψ⟫_ℂ ≠ 0 := by
  rw [inner_self_eq_norm_sq_to_K]
  exact pow_ne_zero 2 (ofReal_ne_zero.mpr (norm_ne_zero_iff.mpr hψ))

omit [NormedSpace ℝ E] in
private lemma generator_inner_numerator
    (L H O : E →L[ℂ] E) (ψ : E)
    (hL : (L : E →ₗ[ℂ] E).IsSymmetric)
    (hH : (H : E →ₗ[ℂ] E).IsSymmetric) :
    ⟪ψ, O (-((L + I • H) ψ))⟫_ℂ + ⟪-((L + I • H) ψ), O ψ⟫_ℂ =
      I * ⟪ψ, (H * O - O * H) ψ⟫_ℂ - ⟪ψ, (L * O + O * L) ψ⟫_ℂ := by
  simp only [add_apply, smul_apply, map_add, map_neg, map_smul, inner_neg_right,
    inner_neg_left, inner_add_right, inner_add_left, inner_sub_right, inner_smul_right,
    inner_smul_left, conj_I, neg_mul, sub_apply, mul_apply_eq_comp]
  rw [hL.apply_clm, hH.apply_clm]
  ring

omit [NormedSpace ℝ E] in
private lemma generator_inner_denominator
    (L H : E →L[ℂ] E) (ψ : E)
    (hL : (L : E →ₗ[ℂ] E).IsSymmetric)
    (hH : (H : E →ₗ[ℂ] E).IsSymmetric) :
    ⟪ψ, -((L + I • H) ψ)⟫_ℂ + ⟪-((L + I • H) ψ), ψ⟫_ℂ =
      -2 * ⟪ψ, L ψ⟫_ℂ := by
  simp only [add_apply, smul_apply, inner_neg_right, inner_neg_left, inner_add_right,
    inner_add_left, inner_smul_right, inner_smul_left, conj_I, neg_mul]
  rw [hL.apply_clm, hH.apply_clm]
  ring

/-- Manuscript Eq. (eq:lle-normalized-expectation) at one time `t`.

The theorem is analytic: it starts from the state derivative and returns the derivative of the
normalized expectation.  No mean-field closure is used. -/
theorem normalizedExpectation_hasDerivAt
    (L H O : E →L[ℂ] E) (ψ : ℝ → E) (t : ℝ)
    (hψ0 : ψ t ≠ 0)
    (hψ : HasDerivAt ψ (-((L + I • H) (ψ t))) t)
    (hL : (L : E →ₗ[ℂ] E).IsSymmetric)
    (hH : (H : E →ₗ[ℂ] E).IsSymmetric) :
    HasDerivAt (fun s ↦ normalizedExpectation O (ψ s))
      (I * normalizedExpectation (H * O - O * H) (ψ t)
        - normalizedExpectation (L * O + O * L) (ψ t)
        + 2 * normalizedExpectation L (ψ t) * normalizedExpectation O (ψ t)) t := by
  let v : E := -((L + I • H) (ψ t))
  have hOψ : HasDerivAt (fun s ↦ O (ψ s)) (O v) t :=
    by
      have hOψ0 : HasDerivAt ((O.restrictScalars ℝ) ∘ ψ)
          ((O.restrictScalars ℝ) v) t := by
        simpa only [v] using (O.restrictScalars ℝ).hasFDerivAt.comp_hasDerivAt t hψ
      simpa only [Function.comp_def, ContinuousLinearMap.coe_restrictScalars'] using hOψ0
  have hnum : HasDerivAt (fun s ↦ ⟪ψ s, O (ψ s)⟫_ℂ)
      (⟪ψ t, O v⟫_ℂ + ⟪v, O (ψ t)⟫_ℂ) t :=
    hψ.inner ℂ hOψ
  have hden : HasDerivAt (fun s ↦ ⟪ψ s, ψ s⟫_ℂ)
      (⟪ψ t, v⟫_ℂ + ⟪v, ψ t⟫_ℂ) t :=
    hψ.inner ℂ hψ
  have hquot := by
    with_reducible_and_instances
      exact hnum.div hden (inner_self_ne_zero hψ0)
  rw [show v = -((L + I • H) (ψ t)) by rfl,
    generator_inner_numerator L H O (ψ t) hL hH,
    generator_inner_denominator L H (ψ t) hL hH] at hquot
  with_reducible_and_instances
    convert hquot using 1
    · rfl
    · funext s
      simp only [normalizedExpectation, Pi.div_apply]
    · unfold normalizedExpectation
      field_simp [inner_self_ne_zero hψ0]
      ring

end

end Cvdv
