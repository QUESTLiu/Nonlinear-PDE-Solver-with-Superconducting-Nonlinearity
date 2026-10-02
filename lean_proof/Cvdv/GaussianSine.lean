import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap
import Mathlib.Analysis.Complex.Trigonometric

/-!
# Debye-Waller identity for the Gaussian measure

Formalization of manuscript Eq. (debye-waller): for a Gaussian state the expectation of the
sine of the phase quadrature equals `exp (-v/2) * sin μ`, where `μ` is the mean phase and `v`
the variance.  Since the measurement statistics of the phase quadrature of a Gaussian state
are exactly Gaussian, the mathematical content is the probability identity
`∫ x, sin x ∂(gaussianReal μ v) = exp (-v/2) * sin μ`, proved here as
`integral_sin_gaussianReal` together with the cosine companion `integral_cos_gaussianReal`.

Both follow from the characteristic function of the real Gaussian measure
(`ProbabilityTheory.charFun_gaussianReal`) evaluated at `t = 1`: the complex identity
`∫ x, exp (x * I) ∂(gaussianReal μ v) = exp (μ * I - v / 2)` is isolated as
`integral_cexp_mul_I_gaussianReal`, and the two real identities are its imaginary and real
parts.  Integrability of `x ↦ exp (x * I)` against any finite measure is supplied by
`integrable_cexp_mul_I`, using that the integrand has norm one.
-/

open MeasureTheory ProbabilityTheory Complex

open scoped NNReal

/-- The unimodular function `x ↦ exp (x * I)` is integrable against any finite measure. -/
lemma integrable_cexp_mul_I (P : Measure ℝ) [IsFiniteMeasure P] :
    Integrable (fun x : ℝ ↦ Complex.exp (x * I)) P := by
  refine (integrable_const (1 : ℝ)).mono'
    ((Complex.continuous_exp.comp
      (Complex.continuous_ofReal.mul continuous_const)).aestronglyMeasurable) ?_
  filter_upwards with x
  exact le_of_eq (Complex.norm_exp_ofReal_mul_I x)

/-- The characteristic function of `gaussianReal μ v` at `t = 1`, written as a Bochner
integral: `∫ x, exp (x * I) ∂(gaussianReal μ v) = exp (μ * I - v / 2)`. -/
theorem integral_cexp_mul_I_gaussianReal (mu : ℝ) (v : ℝ≥0) :
    ∫ x : ℝ, Complex.exp (x * I) ∂(gaussianReal mu v)
      = Complex.exp (mu * I - (v : ℝ) / 2) := by
  have h := charFun_gaussianReal (μ := mu) (v := v) 1
  rw [charFun_apply_real] at h
  simpa using h

/-- **Debye-Waller identity**, manuscript Eq. (debye-waller): the expectation of the sine
under the Gaussian measure with mean `μ` and variance `v` is `exp (-v/2) * sin μ`. -/
theorem integral_sin_gaussianReal (mu : ℝ) (v : ℝ≥0) :
    ∫ x, Real.sin x ∂(gaussianReal mu v) = Real.exp (-(v : ℝ) / 2) * Real.sin mu := by
  have h := integral_im (μ := gaussianReal mu v) (integrable_cexp_mul_I _)
  simp only [RCLike.im_to_complex, Complex.exp_ofReal_mul_I_im] at h
  rw [h, integral_cexp_mul_I_gaussianReal, Complex.exp_im]
  have harg : ((mu : ℂ) * I - (v : ℝ) / 2) = ((-(v : ℝ) / 2 : ℝ) : ℂ) + (mu : ℂ) * I := by
    push_cast
    ring
  rw [harg]
  simp

/-- Cosine companion of the Debye-Waller identity: the expectation of the cosine under the
Gaussian measure with mean `μ` and variance `v` is `exp (-v/2) * cos μ`. -/
theorem integral_cos_gaussianReal (mu : ℝ) (v : ℝ≥0) :
    ∫ x, Real.cos x ∂(gaussianReal mu v) = Real.exp (-(v : ℝ) / 2) * Real.cos mu := by
  have h := integral_re (μ := gaussianReal mu v) (integrable_cexp_mul_I _)
  simp only [RCLike.re_to_complex, Complex.exp_ofReal_mul_I_re] at h
  rw [h, integral_cexp_mul_I_gaussianReal, Complex.exp_re]
  have harg : ((mu : ℂ) * I - (v : ℝ) / 2) = ((-(v : ℝ) / 2 : ℝ) : ℂ) + (mu : ℂ) * I := by
    push_cast
    ring
  rw [harg]
  simp
