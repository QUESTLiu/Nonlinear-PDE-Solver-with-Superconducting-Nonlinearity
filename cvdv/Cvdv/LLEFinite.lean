import Cvdv.FockHermitian
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Permutation
import Mathlib.Logic.Equiv.Fin.Rotate

/-!
# Finite-matrix structural facts for the Lugiato--Lefever model

This file treats every cutoff operator as a finite complex matrix.
It proves the finite structural conditions and the local cutoff ladder-norm bounds.

No exact-LCHS kernel or operator-valued integral is formalized here.
-/

namespace Cvdv

open scoped BigOperators ComplexOrder Matrix Matrix.Norms.L2Operator

noncomputable section

section LLEFinite

variable {ι σ : Type*} [Fintype ι] [DecidableEq ι]
  [Fintype σ] [DecidableEq σ]

/-- The strict finite-Fock lowering matrix is a cyclic permutation followed by a diagonal weight.
The zero diagonal weight removes the cyclic wraparound. -/
theorem fockSandwichMatrix_lowering_eq_permMatrix_mul_diagonal (N : ℕ) :
    fockSandwichMatrix N lowering =
      (finRotate N).permMatrix ℂ *
        Matrix.diagonal (fun j : Fin N ↦ (Real.sqrt j.val : ℂ)) := by
  ext i j
  rw [fockSandwichMatrix_apply, lowering_single, Matrix.mul_diagonal]
  simp only [Finsupp.smul_single, smul_eq_mul, mul_one, one_smul,
    PEquiv.toMatrix_apply, Equiv.toPEquiv_apply, finRotate_apply,
    Option.mem_def, Option.some.injEq, ite_mul, one_mul, zero_mul]
  by_cases hj0 : j.val = 0
  · simp [hj0]
  · haveI : NeZero N := j.neZero
    have hj : j ≠ 0 := by
      intro h
      apply hj0
      simp [h]
    have hindex : j.val - 1 = i.val ↔ i + 1 = j := by
      constructor
      · intro h
        have hi : i = (finRotate N).symm j := by
          apply Fin.ext
          rw [coe_finRotate_symm_of_ne_zero hj]
          exact h.symm
        rw [← finRotate_apply i]
        exact (finRotate N).apply_eq_iff_eq_symm_apply.mpr hi
      · intro h
        have hi : i = (finRotate N).symm j :=
          (finRotate N).apply_eq_iff_eq_symm_apply.mp (by simpa [finRotate_apply] using h)
        have hval := congrArg Fin.val hi
        rw [coe_finRotate_symm_of_ne_zero hj] at hval
        exact hval.symm
    simp only [Finsupp.single_apply, hindex]

/-- The strict finite-Fock lowering matrix obeys the cutoff bound used in the manuscript. -/
theorem norm_fockSandwichMatrix_lowering_le_sqrt_cutoff (N : ℕ) :
    ‖fockSandwichMatrix N lowering‖ ≤ Real.sqrt N := by
  rw [fockSandwichMatrix_lowering_eq_permMatrix_mul_diagonal]
  have hdiag :
      ‖Matrix.diagonal (fun j : Fin N ↦ (Real.sqrt j.val : ℂ))‖ ≤ Real.sqrt N := by
    rw [Matrix.l2_opNorm_diagonal]
    refine (pi_norm_le_iff_of_nonneg (Real.sqrt_nonneg N)).mpr fun j ↦ ?_
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (Real.sqrt_nonneg j.val)]
    exact Real.sqrt_le_sqrt (by exact_mod_cast (Nat.le_of_lt j.isLt))
  calc
    ‖(finRotate N).permMatrix ℂ *
        Matrix.diagonal (fun j : Fin N ↦ (Real.sqrt j.val : ℂ))‖
        ≤ ‖(finRotate N).permMatrix ℂ‖ *
          ‖Matrix.diagonal (fun j : Fin N ↦ (Real.sqrt j.val : ℂ))‖ :=
      Matrix.l2_opNorm_mul _ _
    _ ≤ 1 * ‖Matrix.diagonal (fun j : Fin N ↦ (Real.sqrt j.val : ℂ))‖ :=
      mul_le_mul_of_nonneg_right (Matrix.permMatrix_l2_opNorm_le (finRotate N)) (norm_nonneg _)
    _ ≤ 1 * Real.sqrt N := mul_le_mul_of_nonneg_left hdiag zero_le_one
    _ = Real.sqrt N := one_mul _

/-- The strict finite-Fock raising matrix is the adjoint of the lowering matrix. -/
theorem fockSandwichMatrix_raising_eq_lowering_conjTranspose (N : ℕ) :
    fockSandwichMatrix N raising = (fockSandwichMatrix N lowering)ᴴ := by
  ext i j
  rw [fockSandwichMatrix_apply, Matrix.conjTranspose_apply, fockSandwichMatrix_apply]
  have h := lowering_raising_isAdjointPair
    (Finsupp.single i.val 1) (Finsupp.single j.val 1)
  rw [fockAlgebraForm_single_right, fockAlgebraForm_single_left] at h
  simpa using congrArg star h.symm

/-- The strict finite-Fock raising matrix obeys the same cutoff norm bound. -/
theorem norm_fockSandwichMatrix_raising_le_sqrt_cutoff (N : ℕ) :
    ‖fockSandwichMatrix N raising‖ ≤ Real.sqrt N := by
  rw [fockSandwichMatrix_raising_eq_lowering_conjTranspose,
    Matrix.l2_opNorm_conjTranspose]
  exact norm_fockSandwichMatrix_lowering_le_sqrt_cutoff N

/-- Total occupation represented directly on one finite system space. -/
def finiteLLETotalNumber (a : ι → Matrix σ σ ℂ) : Matrix σ σ ℂ :=
  ∑ j, (a j)ᴴ * a j

/-- The coherent-drive contribution in the manuscript convention. -/
def finiteLLEDrive (a : ι → Matrix σ σ ℂ) (F : ι → ℂ) : Matrix σ σ ℂ :=
  Complex.I • ∑ j, ((F j) • (a j)ᴴ - (starRingEnd ℂ) (F j) • a j)

/-- The finite LLE Hamiltonian.  `dispersion` abbreviates `D₂/(2h²)`. -/
def finiteLLEHamiltonian (a : ι → Matrix σ σ ℂ) (next : ι → ι)
    (delta dispersion g : ℝ) (F : ι → ℂ) : Matrix σ σ ℂ :=
  delta • finiteLLETotalNumber a
    + dispersion • ∑ j, (a (next j) - a j)ᴴ * (a (next j) - a j)
    + (g / 2) • ∑ j, ((a j) ^ 2)ᴴ * (a j) ^ 2
    + finiteLLEDrive a F

/-- The finite loss operator. -/
def finiteLLELoss (a : ι → Matrix σ σ ℂ) (kappa : ℝ) : Matrix σ σ ℂ :=
  (kappa / 2) • finiteLLETotalNumber a

/-- The loss--drive commutator in the closed form used for the LCHS cost bound. -/
def finiteLLEMixedCommutator (a : ι → Matrix σ σ ℂ) (kappa : ℝ)
    (F : ι → ℂ) : Matrix σ σ ℂ :=
  (Complex.I * (kappa / 2 : ℝ)) •
    ∑ j, ((F j) • (a j)ᴴ + (starRingEnd ℂ) (F j) • a j)

/-- The finite generator `A = L + iH`. -/
def finiteLLEGenerator (a : ι → Matrix σ σ ℂ) (next : ι → ι)
    (delta dispersion g kappa : ℝ) (F : ι → ℂ) : Matrix σ σ ℂ :=
  finiteLLELoss a kappa + Complex.I • finiteLLEHamiltonian a next delta dispersion g F

omit [Fintype σ] [DecidableEq σ] in
private theorem finiteLLEDrive_site_isHermitian
    (A : Matrix σ σ ℂ) (z : ℂ) :
    (Complex.I • (z • Aᴴ - (starRingEnd ℂ) z • A)).IsHermitian := by
  let B : Matrix σ σ ℂ := (Complex.I * z) • Aᴴ
  have hB : (B + Bᴴ).IsHermitian := Matrix.isHermitian_add_transpose_self B
  simpa [B, sub_eq_add_neg, smul_add, smul_smul] using hB

omit [DecidableEq ι] [Fintype σ] [DecidableEq σ] in
/-- The coherent-drive matrix is Hermitian for arbitrary complex pump amplitudes. -/
theorem finiteLLEDrive_isHermitian (a : ι → Matrix σ σ ℂ) (F : ι → ℂ) :
    (finiteLLEDrive a F).IsHermitian := by
  rw [finiteLLEDrive, Finset.smul_sum]
  exact (isSelfAdjoint_sum Finset.univ fun j _ ↦
    (finiteLLEDrive_site_isHermitian (a j) (F j)).isSelfAdjoint).isHermitian

omit [DecidableEq ι] in
/-- Manuscript Eq. `eq:lle-native-hamiltonian` is Hermitian on a finite system space. -/
theorem finiteLLEHamiltonian_isHermitian (a : ι → Matrix σ σ ℂ) (next : ι → ι)
    (delta dispersion g : ℝ) (F : ι → ℂ) :
    (finiteLLEHamiltonian a next delta dispersion g F).IsHermitian := by
  have hnumber : (finiteLLETotalNumber a).IsHermitian := by
    rw [finiteLLETotalNumber]
    exact (isSelfAdjoint_sum Finset.univ fun j _ ↦
      (Matrix.isHermitian_conjTranspose_mul_self (a j)).isSelfAdjoint).isHermitian
  have hdispersion :
      (∑ j, (a (next j) - a j)ᴴ * (a (next j) - a j)).IsHermitian :=
    (isSelfAdjoint_sum Finset.univ fun j _ ↦
      (Matrix.isHermitian_conjTranspose_mul_self (a (next j) - a j)).isSelfAdjoint).isHermitian
  have hkerr : (∑ j, ((a j) ^ 2)ᴴ * (a j) ^ 2).IsHermitian :=
    (isSelfAdjoint_sum Finset.univ fun j _ ↦
      (Matrix.isHermitian_conjTranspose_mul_self ((a j) ^ 2)).isSelfAdjoint).isHermitian
  unfold finiteLLEHamiltonian
  exact (((hnumber.smul (IsSelfAdjoint.all delta)).add
    (hdispersion.smul (IsSelfAdjoint.all dispersion))).add
      (hkerr.smul (IsSelfAdjoint.all (g / 2)))).add (finiteLLEDrive_isHermitian a F)

omit [DecidableEq ι] [DecidableEq σ] in
/-- Manuscript Eq. `eq:lle-loss-operator` is positive semidefinite for `κ ≥ 0`. -/
theorem finiteLLELoss_posSemidef (a : ι → Matrix σ σ ℂ) (kappa : ℝ)
    (hkappa : 0 ≤ kappa) :
    (finiteLLELoss a kappa).PosSemidef := by
  unfold finiteLLELoss finiteLLETotalNumber
  exact (Matrix.posSemidef_sum Finset.univ fun j _ ↦
    Matrix.posSemidef_conjTranspose_mul_self (a j)).smul (α := ℝ)
      (div_nonneg hkappa (by positivity))

omit [DecidableEq ι] in
/-- The manuscript `C_LCHS` estimate from a uniform cutoff ladder-norm bound. -/
theorem norm_finiteLLEMixedCommutator_le (a : ι → Matrix σ σ ℂ) (kappa r : ℝ)
    (F : ι → ℂ) (hkappa : 0 ≤ kappa)
    (ha : ∀ j, ‖a j‖ ≤ r) :
    ‖finiteLLEMixedCommutator a kappa F‖ ≤
      kappa * r * ∑ j, ‖F j‖ := by
  rw [finiteLLEMixedCommutator, norm_smul]
  have hscalar : ‖Complex.I * (kappa / 2 : ℝ)‖ = kappa / 2 := by
    rw [norm_mul, Complex.norm_I, one_mul, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonneg (div_nonneg hkappa (by positivity))]
  rw [hscalar]
  calc
    kappa / 2 * ‖∑ j, ((F j) • (a j)ᴴ + (starRingEnd ℂ) (F j) • a j)‖
        ≤ kappa / 2 * ∑ j, ‖(F j) • (a j)ᴴ + (starRingEnd ℂ) (F j) • a j‖ := by
          gcongr
          exact norm_sum_le _ _
    _ ≤ kappa / 2 * ∑ j, (2 * (‖F j‖ * r)) := by
      gcongr with j
      calc
        ‖(F j) • (a j)ᴴ + (starRingEnd ℂ) (F j) • a j‖
            ≤ ‖(F j) • (a j)ᴴ‖ + ‖(starRingEnd ℂ) (F j) • a j‖ := norm_add_le _ _
        _ = ‖F j‖ * ‖a j‖ + ‖F j‖ * ‖a j‖ := by
          rw [norm_smul, norm_smul, Matrix.l2_opNorm_conjTranspose]
          change ‖F j‖ * ‖a j‖ + ‖star (F j)‖ * ‖a j‖ = _
          rw [norm_star]
        _ ≤ ‖F j‖ * r + ‖F j‖ * r := by gcongr <;> exact ha j
        _ = 2 * (‖F j‖ * r) := by ring
    _ = kappa * r * ∑ j, ‖F j‖ := by
      simp_rw [show ∀ j, 2 * (‖F j‖ * r) = (2 * r) * ‖F j‖ by
        intro j
        ring]
      rw [← Finset.mul_sum]
      ring

omit [DecidableEq ι] in
/-- The cutoff form of the manuscript bound, assuming `‖aⱼ‖ ≤ √n_cut`. -/
theorem norm_finiteLLEMixedCommutator_le_sqrt_cutoff
    (a : ι → Matrix σ σ ℂ) (kappa : ℝ) (ncut : ℕ) (F : ι → ℂ)
    (hkappa : 0 ≤ kappa) (ha : ∀ j, ‖a j‖ ≤ Real.sqrt ncut) :
    ‖finiteLLEMixedCommutator a kappa F‖ ≤
      kappa * Real.sqrt ncut * ∑ j, ‖F j‖ :=
  norm_finiteLLEMixedCommutator_le a kappa (Real.sqrt ncut) F hkappa ha

omit [DecidableEq ι] in
/-- The finite-matrix structural prerequisites supplied to an exact-LCHS theorem.

This theorem does not assert existence or admissibility of a kernel.
-/
theorem finiteLLE_structural_prerequisites (a : ι → Matrix σ σ ℂ) (next : ι → ι)
    (delta dispersion g kappa : ℝ) (F : ι → ℂ) (hkappa : 0 ≤ kappa) :
    (finiteLLELoss a kappa).PosSemidef ∧
      (finiteLLEHamiltonian a next delta dispersion g F).IsHermitian ∧
      finiteLLEGenerator a next delta dispersion g kappa F =
        finiteLLELoss a kappa +
          Complex.I • finiteLLEHamiltonian a next delta dispersion g F := by
  exact ⟨finiteLLELoss_posSemidef a kappa hkappa,
    finiteLLEHamiltonian_isHermitian a next delta dispersion g F, rfl⟩

end LLEFinite

end

end Cvdv
