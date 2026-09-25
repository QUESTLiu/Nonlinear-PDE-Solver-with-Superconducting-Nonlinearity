import Cvdv.EnergyShift
import Cvdv.FockHermitian
import Cvdv.TrotterBound

/-!
# The finite-cutoff Duffing--Kerr theorem chain

The Duffing polynomial is formed on algebraic Fock space before taking the strict finite
sandwich.  No product of sandwiched ladder operators is used.
-/

open NormedSpace

namespace Cvdv

/-- The full algebraic Duffing Hamiltonian
`ω0 (n + 1/2) + λ (a + a†)⁴`. -/
noncomputable def fockDuffingHamiltonian (ω0 lam : ℝ) : Module.End ℂ FockAlg :=
  (ω0 : ℂ) • fockNumber + ((ω0 / 2 : ℝ) : ℂ) • 1
    + (lam : ℂ) • (lowering + raising) ^ 4

/-- The number-conserving Duffing--Kerr Hamiltonian
`(ω0 + 6λ) n + 6λ n²`. -/
noncomputable def fockDuffingKerrHamiltonian (ω0 lam : ℝ) : Module.End ℂ FockAlg :=
  fockKerrHamiltonian (ω0 + 6 * lam) (6 * lam)

/-- The additive constant discarded by the Kerr dynamics. -/
noncomputable def fockDuffingEnergyShift (ω0 lam : ℝ) : ℝ := ω0 / 2 + 3 * lam

/-- Exact Duffing--Kerr split on algebraic Fock space. -/
theorem fockDuffingHamiltonian_split (ω0 lam : ℝ) :
    fockDuffingHamiltonian ω0 lam =
      fockDuffingKerrHamiltonian ω0 lam
        + (fockDuffingEnergyShift ω0 lam : ℂ) • 1
        + (lam : ℂ) • fockCounterRotating := by
  rw [fockDuffingHamiltonian, add_pow_four_split_smul lowering raising lowering_mul_raising]
  simp only [fockDuffingKerrHamiltonian, fockKerrHamiltonian, fockNumber,
    fockDuffingEnergyShift, fockCounterRotating]
  push_cast
  simp only [pow_two]
  module

/-- Strict finite sandwich of the full Duffing Hamiltonian. -/
noncomputable def cutoffDuffingHamiltonian (N : ℕ) (ω0 lam : ℝ) :
    FockCutoff N →L[ℂ] FockCutoff N :=
  fockSandwichCLM N (fockDuffingHamiltonian ω0 lam)

/-- Strict finite sandwich of the Kerr Hamiltonian `H_K`. -/
noncomputable def cutoffDuffingKerrHamiltonian (N : ℕ) (ω0 lam : ℝ) :
    FockCutoff N →L[ℂ] FockCutoff N :=
  fockSandwichCLM N (fockDuffingKerrHamiltonian ω0 lam)

/-- Strict finite sandwich of the counter-rotating polynomial. -/
noncomputable def cutoffCounterRotating (N : ℕ) :
    FockCutoff N →L[ℂ] FockCutoff N :=
  fockSandwichCLM N fockCounterRotating

/-- The projected correction Hamiltonian `λ Π_N V_cr Π_N`. -/
noncomputable def cutoffDuffingCorrection (N : ℕ) (lam : ℝ) :
    FockCutoff N →L[ℂ] FockCutoff N :=
  (lam : ℂ) • cutoffCounterRotating N

/-- Paper-facing strict-cutoff Duffing split.  It follows by linearity of the sandwich;
no multiplicativity of the sandwich is assumed. -/
theorem cutoffDuffingHamiltonian_split (N : ℕ) (ω0 lam : ℝ) :
    cutoffDuffingHamiltonian N ω0 lam =
      cutoffDuffingKerrHamiltonian N ω0 lam
        + (fockDuffingEnergyShift ω0 lam : ℂ) • 1
        + (lam : ℂ) • cutoffCounterRotating N := by
  have h := congrArg (fockSandwichCLM N) (fockDuffingHamiltonian_split ω0 lam)
  simpa only [cutoffDuffingHamiltonian, cutoffDuffingKerrHamiltonian,
    cutoffCounterRotating, map_add, map_smul, fockSandwichCLM_one] using h

/-- Exact `E₀` phase factor in the sentence following manuscript Eq. (eq:df-hybrid-step).
The theorem concerns the full reduced evolution, not its symmetric product formula. -/
theorem cutoffDuffing_exp_eq_phase_smul_exp_reduced (N : ℕ) (ω0 lam t : ℝ)
    (psi : FockCutoff N) :
    exp ((-Complex.I * t) • cutoffDuffingHamiltonian N ω0 lam) psi =
      Complex.exp (-Complex.I * (t * fockDuffingEnergyShift ω0 lam)) •
        exp ((-Complex.I * t) •
          (cutoffDuffingKerrHamiltonian N ω0 lam + cutoffDuffingCorrection N lam)) psi := by
  rw [cutoffDuffingHamiltonian_split]
  have hHamiltonian :
      cutoffDuffingKerrHamiltonian N ω0 lam
          + (fockDuffingEnergyShift ω0 lam : ℂ) • 1
          + (lam : ℂ) • cutoffCounterRotating N =
        (cutoffDuffingKerrHamiltonian N ω0 lam + cutoffDuffingCorrection N lam)
          + (fockDuffingEnergyShift ω0 lam : ℂ) • 1 := by
    simp only [cutoffDuffingCorrection]
    abel
  rw [hHamiltonian, exp_energy_shift_apply]

/-- The strict-cutoff Duffing--Kerr Hamiltonian is symmetric. -/
theorem cutoffDuffingKerrHamiltonian_isSymmetric (N : ℕ) (ω0 lam : ℝ) :
    (cutoffDuffingKerrHamiltonian N ω0 lam :
      FockCutoff N →ₗ[ℂ] FockCutoff N).IsSymmetric := by
  intro v w
  change inner ℂ (cutoffDuffingKerrHamiltonian N ω0 lam v) w =
    inner ℂ v (cutoffDuffingKerrHamiltonian N ω0 lam w)
  have hn := fockSandwichCLM_number_isSymmetric N v w
  change inner ℂ ((fockSandwichCLM N fockNumber) v) w =
    inner ℂ v ((fockSandwichCLM N fockNumber) w) at hn
  have hn2 := fockSandwichCLM_number_sq_isSymmetric N v w
  change inner ℂ ((fockSandwichCLM N (fockNumber ^ 2)) v) w =
    inner ℂ v ((fockSandwichCLM N (fockNumber ^ 2)) w) at hn2
  simp only [cutoffDuffingKerrHamiltonian, fockDuffingKerrHamiltonian,
    fockKerrHamiltonian, map_add, map_smul, add_apply, smul_apply,
    inner_add_left, inner_add_right, inner_smul_left, inner_smul_right,
    Complex.conj_ofReal]
  rw [hn, hn2]
  simp only [map_mul, Complex.conj_ofReal]
  have h6 : (starRingEnd ℂ) (6 : ℂ) = 6 := map_natCast (starRingEnd ℂ) 6
  rw [h6]

/-- The strict-cutoff counter-rotating polynomial is symmetric. -/
theorem cutoffCounterRotating_isSymmetric (N : ℕ) :
    (cutoffCounterRotating N : FockCutoff N →ₗ[ℂ] FockCutoff N).IsSymmetric := by
  simpa only [cutoffCounterRotating] using
    fockSandwichCLM_counterRotating_isSymmetric N

/-- The strict-cutoff full Duffing Hamiltonian is symmetric. -/
theorem cutoffDuffingHamiltonian_isSymmetric (N : ℕ) (ω0 lam : ℝ) :
    (cutoffDuffingHamiltonian N ω0 lam :
      FockCutoff N →ₗ[ℂ] FockCutoff N).IsSymmetric := by
  have hshift := isSymm_add_real_smul
    (cutoffDuffingKerrHamiltonian N ω0 lam) 1
    (cutoffDuffingKerrHamiltonian_isSymmetric N ω0 lam)
    (isSymm_identity_clm (E := FockCutoff N)) (fockDuffingEnergyShift ω0 lam)
  have hfull := isSymm_add_real_smul
    (cutoffDuffingKerrHamiltonian N ω0 lam
      + (fockDuffingEnergyShift ω0 lam : ℂ) • 1)
    (cutoffCounterRotating N) hshift (cutoffCounterRotating_isSymmetric N) lam
  rw [cutoffDuffingHamiltonian_split]
  exact hfull

/-- Phase-aligned Duhamel estimate for the strict-cutoff Duffing--Kerr split. -/
theorem cutoffDuffing_phase_aligned_bound (N : ℕ) (ω0 lam : ℝ)
    (psi0 : FockCutoff N) (t : ℝ) (ht : 0 ≤ t) :
    ‖exp ((-Complex.I * t) • cutoffDuffingHamiltonian N ω0 lam) psi0
        - Complex.exp (-Complex.I * (t * fockDuffingEnergyShift ω0 lam)) •
          exp ((-Complex.I * t) • cutoffDuffingKerrHamiltonian N ω0 lam) psi0‖
      ≤ |lam| * ∫ s in (0 : ℝ)..t,
          ‖cutoffCounterRotating N
            (exp ((-Complex.I * s) • cutoffDuffingKerrHamiltonian N ω0 lam) psi0)‖ := by
  exact duhamel_phase_aligned_bound_of_split
    (cutoffDuffingHamiltonian N ω0 lam)
    (cutoffDuffingKerrHamiltonian N ω0 lam) (cutoffCounterRotating N)
    (cutoffDuffingKerrHamiltonian_isSymmetric N ω0 lam)
    (cutoffCounterRotating_isSymmetric N)
    (fockDuffingEnergyShift ω0 lam) lam
    (cutoffDuffingHamiltonian_split N ω0 lam) psi0 t ht

/-- Phase-invariant infidelity estimate for the strict-cutoff Duffing--Kerr split. -/
theorem cutoffDuffing_infidelity_bound (N : ℕ) (ω0 lam : ℝ)
    (psi0 : FockCutoff N) (t : ℝ) (ht : 0 ≤ t) (hpsi : ‖psi0‖ = 1) :
    1 - ‖inner ℂ
          (exp ((-Complex.I * t) • cutoffDuffingKerrHamiltonian N ω0 lam) psi0)
          (exp ((-Complex.I * t) • cutoffDuffingHamiltonian N ω0 lam) psi0)‖ ^ 2
      ≤ (|lam| * ∫ s in (0 : ℝ)..t,
          ‖cutoffCounterRotating N
            (exp ((-Complex.I * s) • cutoffDuffingKerrHamiltonian N ω0 lam) psi0)‖) ^ 2 := by
  exact duhamel_infidelity_bound_of_split
    (cutoffDuffingHamiltonian N ω0 lam)
    (cutoffDuffingKerrHamiltonian N ω0 lam) (cutoffCounterRotating N)
    (cutoffDuffingKerrHamiltonian_isSymmetric N ω0 lam)
    (cutoffCounterRotating_isSymmetric N)
    (fockDuffingEnergyShift ω0 lam) lam
    (cutoffDuffingHamiltonian_split N ω0 lam) psi0 t ht hpsi

/-- Manuscript Eq. (df-C-N): the norm of the strict sandwich of
`[H_K, λ V_cr]`. -/
noncomputable def cutoffDuffingCommutatorBudget (N : ℕ) (ω0 lam : ℝ) : ℝ :=
  ‖fockSandwichCLM N
      (fockDuffingKerrHamiltonian ω0 lam * ((lam : ℂ) • fockCounterRotating)
        - ((lam : ℂ) • fockCounterRotating) * fockDuffingKerrHamiltonian ω0 lam)‖

/-- Because the Kerr Hamiltonian is number diagonal, the strict sandwich of its
commutator is exactly the commutator of the strict sandwiches. -/
theorem cutoffDuffingCommutatorBudget_eq (N : ℕ) (ω0 lam : ℝ) :
    cutoffDuffingCommutatorBudget N ω0 lam =
      ‖cutoffDuffingKerrHamiltonian N ω0 lam * cutoffDuffingCorrection N lam
        - cutoffDuffingCorrection N lam * cutoffDuffingKerrHamiltonian N ω0 lam‖ := by
  unfold cutoffDuffingCommutatorBudget cutoffDuffingKerrHamiltonian
    fockDuffingKerrHamiltonian cutoffDuffingCorrection cutoffCounterRotating
  rw [fockSandwichCLM_kerr_commutator]
  simp only [map_smul]

/-- First-order Trotter bound for the Kerr Hamiltonian plus the projected correction.
The exactly factored `E0 I` changes the full Duffing evolution only by the phase handled above. -/
theorem cutoffDuffing_trotter_bound (N : ℕ) (ω0 lam t : ℝ)
    (r : ℕ) (hr : 1 ≤ r) :
    ‖(exp ((-Complex.I * t * (r : ℝ)⁻¹) • cutoffDuffingKerrHamiltonian N ω0 lam)
        * exp ((-Complex.I * t * (r : ℝ)⁻¹) • cutoffDuffingCorrection N lam)) ^ r
        - exp ((-Complex.I * t) •
            (cutoffDuffingKerrHamiltonian N ω0 lam + cutoffDuffingCorrection N lam))‖
      ≤ (t ^ 2 / (2 * r)) * cutoffDuffingCommutatorBudget N ω0 lam
          * Real.exp (|t| *
            (‖cutoffDuffingKerrHamiltonian N ω0 lam‖
              + ‖cutoffDuffingCorrection N lam‖) / r) := by
  have hcorrection : ∀ v w : FockCutoff N,
      inner ℂ (cutoffDuffingCorrection N lam v) w =
        inner ℂ v (cutoffDuffingCorrection N lam w) := by
    intro v w
    simp only [cutoffDuffingCorrection, smul_apply, inner_smul_left,
      inner_smul_right, Complex.conj_ofReal]
    have hV := cutoffCounterRotating_isSymmetric N v w
    change inner ℂ (cutoffCounterRotating N v) w =
      inner ℂ v (cutoffCounterRotating N w) at hV
    rw [hV]
  have h := norm_trotter_pow_sub_exp_le_of_selfAdjoint
    (cutoffDuffingKerrHamiltonian N ω0 lam) (cutoffDuffingCorrection N lam)
    (cutoffDuffingKerrHamiltonian_isSymmetric N ω0 lam) hcorrection t r hr
  rw [← cutoffDuffingCommutatorBudget_eq N ω0 lam] at h
  exact h

end Cvdv
