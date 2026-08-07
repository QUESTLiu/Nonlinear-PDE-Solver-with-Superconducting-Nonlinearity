import Cvdv.Duhamel

/-!
# Additive energy shifts in the Duhamel bound

An additive real multiple of the identity changes a Schrödinger evolution only
by a global phase.  This file combines that fact with the bounded-operator
Duhamel estimates in `Cvdv.Duhamel`.
-/

open NormedSpace

namespace Cvdv

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]

/-- Adding the real energy offset `E0 • 1` multiplies the evolution by the
global phase `exp (-i t E0)`. -/
theorem exp_energy_shift_apply (H : E →L[ℂ] E) (E0 t : ℝ) (psi : E) :
    exp ((-Complex.I * t) • (H + (E0 : ℂ) • (1 : E →L[ℂ] E))) psi
      = Complex.exp (-Complex.I * (t * E0)) •
          exp ((-Complex.I * t) • H) psi := by
  have harg :
      (-Complex.I * t) • (H + (E0 : ℂ) • (1 : E →L[ℂ] E))
        = (-Complex.I * t) • H
          + algebraMap ℂ (E →L[ℂ] E) (-Complex.I * (t * E0)) := by
    rw [smul_add, Algebra.algebraMap_eq_smul_one, smul_smul]
    congr 1
    ring_nf
  rw [harg]
  rw [NormedSpace.exp_add_of_commute_of_mem_ball (𝕂 := ℂ)
    (Algebra.commutes (-Complex.I * (t * E0)) ((-Complex.I * t) • H)).symm
    ((NormedSpace.expSeries_radius_eq_top ℂ (E →L[ℂ] E)).symm ▸ edist_lt_top _ _)
    ((NormedSpace.expSeries_radius_eq_top ℂ (E →L[ℂ] E)).symm ▸ edist_lt_top _ _)]
  rw [← NormedSpace.algebraMap_exp_comm]
  simp [Algebra.algebraMap_eq_smul_one, ← Complex.exp_eq_exp_ℂ]

omit [CompleteSpace E] in
/-- The identity continuous linear map is symmetric. -/
lemma isSymm_identity_clm (v w : E) :
    inner ℂ ((1 : E →L[ℂ] E) v) w = inner ℂ v ((1 : E →L[ℂ] E) w) := by
  simp

/-- A scalar energy shift does not change the interaction norm along the
unperturbed orbit. -/
lemma norm_apply_exp_energy_shift (H V : E →L[ℂ] E) (E0 s : ℝ) (psi : E) :
    ‖V (exp ((-Complex.I * s) • (H + (E0 : ℂ) • (1 : E →L[ℂ] E))) psi)‖
      = ‖V (exp ((-Complex.I * s) • H) psi)‖ := by
  rw [exp_energy_shift_apply, map_smul, norm_smul, Complex.norm_exp]
  simp

/-- Phase-aligned Duhamel bound for a Hamiltonian split
`Hfull = (H + E0 I) + λ V`. -/
theorem duhamel_phase_aligned_bound_of_split (Hfull H V : E →L[ℂ] E)
    (hH : ∀ v w : E, inner ℂ (H v) w = inner ℂ v (H w))
    (hV : ∀ v w : E, inner ℂ (V v) w = inner ℂ v (V w))
    (E0 lam : ℝ)
    (hfull : Hfull = (H + (E0 : ℂ) • (1 : E →L[ℂ] E)) + (lam : ℂ) • V)
    (psi0 : E) (t : ℝ) (ht : 0 ≤ t) :
    ‖exp ((-Complex.I * t) • Hfull) psi0
        - Complex.exp (-Complex.I * (t * E0)) • exp ((-Complex.I * t) • H) psi0‖
      ≤ |lam| * ∫ s in (0 : ℝ)..t, ‖V (exp ((-Complex.I * s) • H) psi0)‖ := by
  subst Hfull
  have hHshift : ∀ v w : E,
      inner ℂ ((H + (E0 : ℂ) • (1 : E →L[ℂ] E)) v) w
        = inner ℂ v ((H + (E0 : ℂ) • (1 : E →L[ℂ] E)) w) :=
    isSymm_add_real_smul H (1 : E →L[ℂ] E) hH isSymm_identity_clm E0
  have hbound := duhamel_rwa_bound
    (H + (E0 : ℂ) • (1 : E →L[ℂ] E)) V hHshift hV lam psi0 t ht
  calc
    ‖exp ((-Complex.I * t) •
          ((H + (E0 : ℂ) • (1 : E →L[ℂ] E)) + (lam : ℂ) • V)) psi0
        - Complex.exp (-Complex.I * (t * E0)) • exp ((-Complex.I * t) • H) psi0‖
        = ‖exp ((-Complex.I * t) •
            ((H + (E0 : ℂ) • (1 : E →L[ℂ] E)) + (lam : ℂ) • V)) psi0
          - exp ((-Complex.I * t) •
            (H + (E0 : ℂ) • (1 : E →L[ℂ] E))) psi0‖ := by
              rw [exp_energy_shift_apply]
    _ ≤ |lam| * ∫ s in (0 : ℝ)..t,
          ‖V (exp ((-Complex.I * s) •
            (H + (E0 : ℂ) • (1 : E →L[ℂ] E))) psi0)‖ := hbound
    _ = |lam| * ∫ s in (0 : ℝ)..t, ‖V (exp ((-Complex.I * s) • H) psi0)‖ := by
      congr 1
      refine intervalIntegral.integral_congr fun s _ => ?_
      exact norm_apply_exp_energy_shift H V E0 s psi0

/-- Phase-invariant infidelity bound for a full Hamiltonian satisfying
`Hfull = (H + E0 I) + λ V`.  The global phase from `E0` drops out. -/
theorem duhamel_infidelity_bound_of_split (Hfull H V : E →L[ℂ] E)
    (hH : ∀ v w : E, inner ℂ (H v) w = inner ℂ v (H w))
    (hV : ∀ v w : E, inner ℂ (V v) w = inner ℂ v (V w))
    (E0 lam : ℝ)
    (hfull : Hfull = (H + (E0 : ℂ) • (1 : E →L[ℂ] E)) + (lam : ℂ) • V)
    (psi0 : E) (t : ℝ) (ht : 0 ≤ t) (hpsi : ‖psi0‖ = 1) :
    1 - ‖inner ℂ (exp ((-Complex.I * t) • H) psi0)
          (exp ((-Complex.I * t) • Hfull) psi0)‖ ^ 2
      ≤ (|lam| * ∫ s in (0 : ℝ)..t, ‖V (exp ((-Complex.I * s) • H) psi0)‖) ^ 2 := by
  subst Hfull
  have hHshift : ∀ v w : E,
      inner ℂ ((H + (E0 : ℂ) • (1 : E →L[ℂ] E)) v) w
        = inner ℂ v ((H + (E0 : ℂ) • (1 : E →L[ℂ] E)) w) :=
    isSymm_add_real_smul H (1 : E →L[ℂ] E) hH isSymm_identity_clm E0
  have hbound := duhamel_infidelity_bound
    (H + (E0 : ℂ) • (1 : E →L[ℂ] E)) V hHshift hV lam psi0 t ht hpsi
  have hfidelity :
      ‖inner ℂ
          (exp ((-Complex.I * t) • (H + (E0 : ℂ) • (1 : E →L[ℂ] E))) psi0)
          (exp ((-Complex.I * t) •
            ((H + (E0 : ℂ) • (1 : E →L[ℂ] E)) + (lam : ℂ) • V)) psi0)‖
        = ‖inner ℂ (exp ((-Complex.I * t) • H) psi0)
          (exp ((-Complex.I * t) •
            ((H + (E0 : ℂ) • (1 : E →L[ℂ] E)) + (lam : ℂ) • V)) psi0)‖ := by
    rw [exp_energy_shift_apply, inner_smul_left, norm_mul, Complex.norm_conj,
      Complex.norm_exp]
    simp
  have hinteraction :
      (∫ s in (0 : ℝ)..t,
          ‖V (exp ((-Complex.I * s) •
            (H + (E0 : ℂ) • (1 : E →L[ℂ] E))) psi0)‖)
        = ∫ s in (0 : ℝ)..t, ‖V (exp ((-Complex.I * s) • H) psi0)‖ := by
    refine intervalIntegral.integral_congr fun s _ => ?_
    exact norm_apply_exp_energy_shift H V E0 s psi0
  rw [hfidelity, hinteraction] at hbound
  exact hbound

end Cvdv
