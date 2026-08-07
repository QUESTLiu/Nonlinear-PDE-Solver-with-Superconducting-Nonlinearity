import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Complex.Arg
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Tactic.Linarith

/-!
# State-distance and fidelity inequalities

This file formalizes the state-geometry estimates used in the manuscript's error
analysis. For normalized states `u v` in a complex inner product space, the
squared distance expands as `‖u - v‖ ^ 2 = 2 - 2 * re ⟪u, v⟫`
(`norm_sub_sq_of_norm_one`), and one ingredient of manuscript Eq. (eq:df-bound) follows:
the infidelity `1 - ‖⟪u, v⟫‖ ^ 2` is bounded by the squared norm distance
(`one_sub_fidelity_le_norm_sub_sq`). A global phase can always be chosen so
that the squared distance equals `2 - 2 * ‖⟪u, v⟫‖` exactly
(`exists_phase_norm_sub_sq`). Finally, the algebraic identity
`1 - c ^ 2 = D ^ 2 * (1 - D ^ 2 / 4)` for `D ^ 2 = 2 - 2 * c` from the
Strang-splitting remark is proved in `one_sub_sq_eq_of_sq_eq_two_sub`.

All results are fully proved, with no placeholders.
-/

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]

/-- For unit vectors the squared distance is `2 - 2 * re ⟪u, v⟫`. -/
lemma norm_sub_sq_of_norm_one (u v : E) (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) :
    ‖u - v‖ ^ 2 = 2 - 2 * RCLike.re (inner ℂ u v) := by
  rw [norm_sub_sq (𝕜 := ℂ), hu, hv]
  ring

/-- State-geometry ingredient of manuscript Eq. (eq:df-bound): for normalized states the infidelity
`1 - F = 1 - ‖⟪u, v⟫‖ ^ 2` is bounded by the squared norm distance. -/
theorem one_sub_fidelity_le_norm_sub_sq (u v : E) (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) :
    1 - ‖inner ℂ u v‖ ^ 2 ≤ ‖u - v‖ ^ 2 := by
  have hc : ‖inner ℂ u v‖ ≤ 1 := by
    have h := norm_inner_le_norm (𝕜 := ℂ) u v
    rwa [hu, hv, mul_one] at h
  have hre : RCLike.re (inner ℂ u v) ≤ ‖inner ℂ u v‖ :=
    RCLike.re_le_norm (inner ℂ u v)
  rw [norm_sub_sq_of_norm_one u v hu hv]
  nlinarith [sq_nonneg (1 - ‖inner ℂ u v‖), hc, hre]

/-- Phase alignment: for unit vectors there is a global phase `θ` making the
squared distance equal to `2 - 2 * ‖⟪u, v⟫‖` exactly. -/
theorem exists_phase_norm_sub_sq (u v : E) (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) :
    ∃ θ : ℝ, ‖u - Complex.exp (θ * Complex.I) • v‖ ^ 2 = 2 - 2 * ‖inner ℂ u v‖ := by
  have key : ∀ r a : ℝ,
      Complex.exp ((-a : ℝ) * Complex.I) * ((r : ℂ) * Complex.exp ((a : ℝ) * Complex.I))
        = (r : ℂ) := by
    intro r a
    have h0 : ((-a : ℝ) : ℂ) * Complex.I + (a : ℂ) * Complex.I = 0 := by
      push_cast
      ring
    rw [mul_comm (r : ℂ), ← mul_assoc, ← Complex.exp_add, h0, Complex.exp_zero, one_mul]
  refine ⟨-Complex.arg (inner ℂ u v), ?_⟩
  have hv' : ‖Complex.exp ((-Complex.arg (inner ℂ u v) : ℝ) * Complex.I) • v‖ = 1 := by
    rw [norm_smul, Complex.norm_exp_ofReal_mul_I, hv, mul_one]
  have hinner :
      inner ℂ u (Complex.exp ((-Complex.arg (inner ℂ u v) : ℝ) * Complex.I) • v)
        = (‖inner ℂ u v‖ : ℂ) := by
    rw [inner_smul_right]
    generalize inner ℂ u v = w
    nth_rw 2 [← Complex.norm_mul_exp_arg_mul_I w]
    exact key ‖w‖ (Complex.arg w)
  rw [norm_sub_sq_of_norm_one u _ hu hv', hinner, RCLike.re_to_complex, Complex.ofReal_re]

/-- Strang infidelity identity: if `D ^ 2 = 2 - 2 * c` then
`1 - c ^ 2 = D ^ 2 * (1 - D ^ 2 / 4)`. -/
theorem one_sub_sq_eq_of_sq_eq_two_sub (c D : ℝ) (hD : D ^ 2 = 2 - 2 * c) :
    1 - c ^ 2 = D ^ 2 * (1 - D ^ 2 / 4) := by
  rw [hD]
  ring

/-- Manuscript Eq. (eq:scaled-moment-recovery): if `Ψ = α • v` and
`v = √p • ψcond`, then every continuous-linear observable has expectation
`⟪Ψ, O Ψ⟫ = ‖α‖² p ⟪ψcond, O ψcond⟫`. -/
theorem scaled_moment_recovery (O : E →L[ℂ] E) (v psiCond Psi : E)
    (alpha : ℂ) (p : ℝ) (hp : 0 ≤ p)
    (hcond : v = (Real.sqrt p : ℂ) • psiCond) (hPsi : Psi = alpha • v) :
    inner ℂ Psi (O Psi) =
      ((‖alpha‖ ^ 2 * p : ℝ) : ℂ) * inner ℂ psiCond (O psiCond) := by
  calc
    inner ℂ Psi (O Psi) =
        inner ℂ (alpha • ((Real.sqrt p : ℂ) • psiCond))
          (O (alpha • ((Real.sqrt p : ℂ) • psiCond))) := by rw [hPsi, hcond]
    _ = (((starRingEnd ℂ) alpha * alpha) *
          ((Real.sqrt p : ℂ) * (Real.sqrt p : ℂ))) * inner ℂ psiCond (O psiCond) := by
      simp only [map_smul, inner_smul_left, inner_smul_right, Complex.conj_ofReal]
      ring
    _ = ((‖alpha‖ ^ 2 * p : ℝ) : ℂ) * inner ℂ psiCond (O psiCond) := by
      rw [Complex.conj_mul', ← Complex.ofReal_mul, Real.mul_self_sqrt hp]
      push_cast
      ring
