import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Symmetric
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

/-- Postselection form of `scaled_moment_recovery`: when `psiCond` is
normalized, `v = √p • psiCond` also proves that `p = ⟪v, v⟫`, so the scalar in
the recovered moment is the postselection probability rather than an
independent parameter. -/
theorem scaled_moment_recovery_postselection (O : E →L[ℂ] E)
    (v psiCond Psi : E) (alpha : ℂ) (p : ℝ) (hp : 0 ≤ p)
    (hpsiCond : ‖psiCond‖ = 1)
    (hcond : v = (Real.sqrt p : ℂ) • psiCond) (hPsi : Psi = alpha • v) :
    inner ℂ v v = (p : ℂ) ∧
      inner ℂ Psi (O Psi) =
        ((‖alpha‖ ^ 2 * p : ℝ) : ℂ) * inner ℂ psiCond (O psiCond) := by
  constructor
  · rw [hcond, inner_smul_left, inner_smul_right, Complex.conj_ofReal,
      inner_self_eq_norm_sq_to_K, hpsiCond]
    norm_num
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt hp]
  · exact scaled_moment_recovery O v psiCond Psi alpha p hp hcond hPsi

/-- Concrete phase alignment (manuscript, Sec. on multi-site Strang validation):
the specific global phase `θ⋆ = -arg ⟪ψ_ref, ψ_Δt⟫` achieves
`‖u - e^{iθ⋆} • v‖ ^ 2 = 2 - 2 * ‖⟪u, v⟫‖` for unit vectors `u v`. This is the
explicit-witness form of `exists_phase_norm_sub_sq`. -/
theorem phase_norm_sub_sq (u v : E) (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) :
    ‖u - Complex.exp ((-Complex.arg (inner ℂ u v) : ℝ) * Complex.I) • v‖ ^ 2
      = 2 - 2 * ‖inner ℂ u v‖ := by
  have key : ∀ r a : ℝ,
      Complex.exp ((-a : ℝ) * Complex.I) * ((r : ℂ) * Complex.exp ((a : ℝ) * Complex.I))
        = (r : ℂ) := by
    intro r a
    have h0 : ((-a : ℝ) : ℂ) * Complex.I + (a : ℂ) * Complex.I = 0 := by
      push_cast
      ring
    rw [mul_comm (r : ℂ), ← mul_assoc, ← Complex.exp_add, h0, Complex.exp_zero, one_mul]
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

/-- Variance identity behind manuscript Eq. (eq:df-state-ratio): for a normalized
state `ψ` and observable `A` with expectation `c = ⟪ψ, A ψ⟫`, the centered vector
`A ψ - c • ψ` has squared norm `‖A ψ‖ ^ 2 - ‖c‖ ^ 2` (the variance
`⟨A†A⟩ - |⟨A⟩|²`, which is `⟨A²⟩ - ⟨A⟩²` for symmetric `A`). -/
theorem centered_norm_sq (A : E →L[ℂ] E) (psi : E) (hpsi : ‖psi‖ = 1) :
    ‖A psi - (inner ℂ psi (A psi)) • psi‖ ^ 2
      = ‖A psi‖ ^ 2 - ‖inner ℂ psi (A psi)‖ ^ 2 := by
  set c := inner ℂ psi (A psi) with hc
  have hip : inner ℂ (A psi) psi = (starRingEnd ℂ) c := by
    rw [hc, ← inner_conj_symm]
  have hre : RCLike.re (inner ℂ (A psi) (c • psi)) = ‖c‖ ^ 2 := by
    rw [inner_smul_right, hip, RCLike.mul_conj]
    exact RCLike.re_ofReal_pow ‖c‖ 2
  rw [norm_sub_sq (𝕜 := ℂ), hre, norm_smul, hpsi, mul_one]
  ring

/-- Real-scalar-shift invariance behind manuscript Eq. (eq:df-state-ratio):
subtracting the expectation value removes any real multiple of the identity,
i.e. the centered vector of `A + r • 1` equals the centered vector of `A`
(the manuscript's "components that generate only an overall phase"). -/
theorem centered_shift_real_id (A : E →L[ℂ] E) (psi : E) (hpsi : ‖psi‖ = 1) (r : ℝ) :
    (A + (r : ℂ) • ContinuousLinearMap.id ℂ E) psi
        - (inner ℂ psi ((A + (r : ℂ) • ContinuousLinearMap.id ℂ E) psi)) • psi
      = A psi - (inner ℂ psi (A psi)) • psi := by
  have happ : (A + (r : ℂ) • ContinuousLinearMap.id ℂ E) psi
      = A psi + (r : ℂ) • psi := by
    simp
  have hself : inner ℂ psi psi = (1 : ℂ) := by
    rw [inner_self_eq_norm_sq_to_K, hpsi]
    norm_num
  rw [happ, inner_add_right, inner_smul_right, hself, mul_one, add_smul]
  abel

/-- Manuscript Eq. (eq:df-state-ratio), Hermitian form of the variance identity: for a
symmetric observable `A` the squared norm `‖A ψ‖ ^ 2 = ⟪A ψ, A ψ⟫` equals the
second-moment matrix element `re ⟪ψ, A (A ψ)⟫ = ⟨A²⟩`, so the centered vector has
squared norm `⟨A²⟩ - |⟨A⟩|²` — the manuscript's standard deviation
`σ_ψ(A)² = ⟨A²⟩ - ⟨A⟩²` for Hermitian `A`. -/
theorem centered_norm_sq_symmetric (A : E →L[ℂ] E)
    (hA : LinearMap.IsSymmetric (A : E →ₗ[ℂ] E)) (psi : E) (hpsi : ‖psi‖ = 1) :
    ‖A psi - (inner ℂ psi (A psi)) • psi‖ ^ 2
      = RCLike.re (inner ℂ psi (A (A psi))) - ‖inner ℂ psi (A psi)‖ ^ 2 := by
  have hAA : RCLike.re (inner ℂ psi (A (A psi))) = ‖A psi‖ ^ 2 := by
    rw [← hA.apply_clm psi (A psi)]
    exact inner_self_eq_norm_sq (𝕜 := ℂ) (A psi)
  rw [centered_norm_sq A psi hpsi, hAA]
