import Cvdv.FockCommutator
import Mathlib.LinearAlgebra.Matrix.Hermitian

/-!
# Hermiticity of strict finite Fock sandwiches

The ladder-operator polynomials are formed on the algebraic Fock space before projection.
Their formal adjoint relations are proved with the standard finite-support sesquilinear form;
only then are they transferred to strict finite sandwiches.
-/

namespace Cvdv

/-- The quartic position polynomial, formed before the strict cutoff. -/
noncomputable def fockQuartic : Module.End ℂ FockAlg :=
  (lowering + raising) ^ 4

/-- The counter-rotating polynomial from manuscript Eq. (df-vcr). -/
noncomputable def fockCounterRotating : Module.End ℂ FockAlg :=
  raising ^ 4 + 4 * (raising ^ 3 * lowering) + lowering ^ 4
    + 4 * (raising * lowering ^ 3) + 6 * lowering ^ 2 + 6 * raising ^ 2

/-- The standard algebraic Fock-space form, `B x y = ∑_i conj(y_i) x_i`. -/
noncomputable def fockAlgebraForm :
    FockAlg →ₗ[ℂ] FockAlg →ₛₗ[starRingEnd ℂ] ℂ where
  toFun x :=
    (Finsupp.lsum ℂ) fun i ↦
      { toFun := fun c ↦ star c * x i
        map_add' := by intro a b; simp [add_mul]
        map_smul' := by intro a b; simp [mul_assoc] }
  map_add' x y := by
    apply LinearMap.ext
    intro z
    change (z.sum fun i a ↦ star a * (x i + y i)) =
      (z.sum fun i a ↦ star a * x i) + z.sum fun i a ↦ star a * y i
    rw [← Finsupp.sum_add]
    apply Finsupp.sum_congr
    intro i hi
    exact mul_add _ _ _
  map_smul' c x := by
    apply LinearMap.ext
    intro z
    change (z.sum fun i a ↦ star a * (c * x i)) = c * z.sum fun i a ↦ star a * x i
    rw [Finsupp.mul_sum]
    apply Finsupp.sum_congr
    intro i hi
    ring

@[simp]
theorem fockAlgebraForm_single (i j : ℕ) (c d : ℂ) :
    fockAlgebraForm (Finsupp.single i c) (Finsupp.single j d) =
      if j = i then star d * c else 0 := by
  simp only [fockAlgebraForm]
  by_cases h : j = i
  · subst j
    simp
  · simp [h]

/-- It suffices to check a formal adjoint relation on number-state basis vectors. -/
lemma fock_isAdjointPair_of_single {A B : Module.End ℂ FockAlg}
    (h : ∀ i c j d,
      fockAlgebraForm (A (Finsupp.single i c)) (Finsupp.single j d) =
        fockAlgebraForm (Finsupp.single i c) (B (Finsupp.single j d))) :
    LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm A B := by
  intro x
  induction x using Finsupp.induction_linear with
  | zero => intro y; simp
  | add x x' hx hx' =>
      intro y
      calc
        fockAlgebraForm (A (x + x')) y = fockAlgebraForm (A x + A x') y := by
          rw [A.map_add]
        _ = fockAlgebraForm (A x) y + fockAlgebraForm (A x') y := by
          rw [fockAlgebraForm.map_add, LinearMap.add_apply]
        _ = fockAlgebraForm x (B y) + fockAlgebraForm x' (B y) :=
          congrArg₂ (fun a b ↦ a + b) (hx y) (hx' y)
        _ = fockAlgebraForm (x + x') (B y) := by
          rw [fockAlgebraForm.map_add, LinearMap.add_apply]
  | single i c =>
      intro y
      induction y using Finsupp.induction_linear with
      | zero => simp
      | add y y' hy hy' =>
          calc
            fockAlgebraForm (A (Finsupp.single i c)) (y + y') =
                fockAlgebraForm (A (Finsupp.single i c)) y
                  + fockAlgebraForm (A (Finsupp.single i c)) y' := by rw [map_add]
            _ = fockAlgebraForm (Finsupp.single i c) (B y)
                  + fockAlgebraForm (Finsupp.single i c) (B y') :=
              congrArg₂ (fun a b ↦ a + b) hy hy'
            _ = fockAlgebraForm (Finsupp.single i c) (B y + B y') := by rw [map_add]
            _ = fockAlgebraForm (Finsupp.single i c) (B (y + y')) := by rw [B.map_add]
      | single j d => exact h i c j d

theorem lowering_raising_isAdjointPair :
    LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm lowering raising := by
  apply fock_isAdjointPair_of_single
  intro i c j d
  simp only [lowering_single, raising_single, map_smul]
  cases i with
  | zero => simp
  | succ k =>
      by_cases h : j = k
      · subst j
        simp [mul_comm]
      · simp [h]

theorem raising_lowering_isAdjointPair :
    LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm raising lowering := by
  apply fock_isAdjointPair_of_single
  intro i c j d
  simp only [lowering_single, raising_single, map_smul]
  cases j with
  | zero => simp
  | succ k =>
      by_cases h : k = i
      · subst k
        simp [mul_comm]
      · simp [h]

theorem fockPosition_isAdjointPair :
    LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm
      ((lowering + raising : Module.End ℂ FockAlg) : FockAlg → FockAlg)
      ((lowering + raising : Module.End ℂ FockAlg) : FockAlg → FockAlg) := by
  intro x y
  calc
    fockAlgebraForm ((lowering + raising) x) y =
        fockAlgebraForm (lowering x + raising x) y := rfl
    _ = fockAlgebraForm (lowering x) y + fockAlgebraForm (raising x) y := by
      rw [fockAlgebraForm.map_add, LinearMap.add_apply]
    _ = fockAlgebraForm x (raising y) + fockAlgebraForm x (lowering y) :=
      congrArg₂ (fun a b ↦ a + b)
        (lowering_raising_isAdjointPair x y) (raising_lowering_isAdjointPair x y)
    _ = fockAlgebraForm x (lowering y) + fockAlgebraForm x (raising y) := add_comm _ _
    _ = fockAlgebraForm x (lowering y + raising y) := by rw [map_add]
    _ = fockAlgebraForm x ((lowering + raising) y) := rfl

theorem fockNumber_isAdjointPair :
    LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm fockNumber fockNumber := by
  simpa only [fockNumber] using
    raising_lowering_isAdjointPair.mul lowering_raising_isAdjointPair

theorem fockNumber_sq_isAdjointPair :
    LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm
      ((fockNumber ^ 2 : Module.End ℂ FockAlg) : FockAlg → FockAlg)
      ((fockNumber ^ 2 : Module.End ℂ FockAlg) : FockAlg → FockAlg) := by
  simpa only [pow_two] using fockNumber_isAdjointPair.mul fockNumber_isAdjointPair

theorem fockQuartic_isAdjointPair :
    LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm fockQuartic fockQuartic := by
  have h₂ := fockPosition_isAdjointPair.mul fockPosition_isAdjointPair
  have h₄ := h₂.mul h₂
  simpa only [fockQuartic, pow_succ, pow_zero, one_mul, mul_one, mul_assoc] using h₄

@[simp]
theorem fockAlgebraForm_single_right (x : FockAlg) (i : ℕ) :
    fockAlgebraForm x (Finsupp.single i 1) = x i := by
  simp [fockAlgebraForm]

@[simp]
theorem fockAlgebraForm_single_left (i : ℕ) (x : FockAlg) :
    fockAlgebraForm (Finsupp.single i 1) x = star (x i) := by
  classical
  change (x.sum fun j c ↦ star c * Finsupp.single i 1 j) = star (x i)
  rw [Finsupp.sum_eq_single i]
  · simp
  · intro j hj hji
    simp [hji]
  · intro hi
    simp

/-- A self-adjoint algebraic operator has a Hermitian strict sandwich. -/
theorem fockSandwichMatrix_isHermitian_of_isAdjointPair (N : ℕ)
    (A : Module.End ℂ FockAlg)
    (hA : LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm A A) :
    Matrix.IsHermitian (fockSandwichMatrix N A) := by
  rw [Matrix.IsHermitian.ext_iff]
  intro i j
  rw [fockSandwichMatrix_apply, fockSandwichMatrix_apply]
  have h := hA (Finsupp.single j.val 1) (Finsupp.single i.val 1)
  simpa using h.symm

theorem fockSandwichMatrix_number_isHermitian (N : ℕ) :
    Matrix.IsHermitian (fockSandwichMatrix N fockNumber) :=
  fockSandwichMatrix_isHermitian_of_isAdjointPair N fockNumber fockNumber_isAdjointPair

theorem fockSandwichMatrix_number_sq_isHermitian (N : ℕ) :
    Matrix.IsHermitian (fockSandwichMatrix N (fockNumber ^ 2)) :=
  fockSandwichMatrix_isHermitian_of_isAdjointPair N _ fockNumber_sq_isAdjointPair

theorem fockSandwichMatrix_quartic_isHermitian (N : ℕ) :
    Matrix.IsHermitian (fockSandwichMatrix N fockQuartic) :=
  fockSandwichMatrix_isHermitian_of_isAdjointPair N _ fockQuartic_isAdjointPair

theorem fockQuartic_split :
    fockQuartic =
      (6 : ℂ) • (fockNumber ^ 2) + (6 : ℂ) • fockNumber + (3 : ℂ) • 1
        + fockCounterRotating := by
  simpa only [fockQuartic, fockNumber, fockCounterRotating, pow_two] using
    (add_pow_four_split_smul lowering raising lowering_mul_raising)

theorem fockSandwichMatrix_counterRotating_eq (N : ℕ) :
    fockSandwichMatrix N fockCounterRotating =
      fockSandwichMatrix N fockQuartic
        - (6 : ℂ) • fockSandwichMatrix N (fockNumber ^ 2)
        - (6 : ℂ) • fockSandwichMatrix N fockNumber
        - (3 : ℂ) • 1 := by
  have h := congrArg (fockSandwichMatrix N) fockQuartic_split
  simp only [map_add, map_smul, fockSandwichMatrix_one] at h
  rw [h]
  module

theorem fockSandwichMatrix_counterRotating_isHermitian (N : ℕ) :
    Matrix.IsHermitian (fockSandwichMatrix N fockCounterRotating) := by
  rw [fockSandwichMatrix_counterRotating_eq]
  exact (((fockSandwichMatrix_quartic_isHermitian N).sub
    ((fockSandwichMatrix_number_sq_isHermitian N).smul (by simp))).sub
      ((fockSandwichMatrix_number_isHermitian N).smul (by simp))).sub
        (Matrix.isHermitian_one.smul (by simp))

private theorem fockSandwichCLM_isSymmetric_of_isHermitian (N : ℕ)
    (A : Module.End ℂ FockAlg)
    (hA : Matrix.IsHermitian (fockSandwichMatrix N A)) :
    (fockSandwichCLM N A : FockCutoff N →ₗ[ℂ] FockCutoff N).IsSymmetric := by
  have hCLM : IsSelfAdjoint
      (Matrix.toEuclideanCLM (n := Fin N) (𝕜 := ℂ) (fockSandwichMatrix N A)) :=
    hA.isSelfAdjoint.map (Matrix.toEuclideanCLM (n := Fin N) (𝕜 := ℂ))
  exact hCLM.isSymmetric

theorem fockSandwichCLM_number_isSymmetric (N : ℕ) :
    (fockSandwichCLM N fockNumber : FockCutoff N →ₗ[ℂ] FockCutoff N).IsSymmetric :=
  fockSandwichCLM_isSymmetric_of_isHermitian N _
    (fockSandwichMatrix_number_isHermitian N)

theorem fockSandwichCLM_number_sq_isSymmetric (N : ℕ) :
    (fockSandwichCLM N (fockNumber ^ 2) :
      FockCutoff N →ₗ[ℂ] FockCutoff N).IsSymmetric :=
  fockSandwichCLM_isSymmetric_of_isHermitian N _
    (fockSandwichMatrix_number_sq_isHermitian N)

theorem fockSandwichCLM_quartic_isSymmetric (N : ℕ) :
    (fockSandwichCLM N fockQuartic : FockCutoff N →ₗ[ℂ] FockCutoff N).IsSymmetric :=
  fockSandwichCLM_isSymmetric_of_isHermitian N _
    (fockSandwichMatrix_quartic_isHermitian N)

theorem fockSandwichCLM_counterRotating_isSymmetric (N : ℕ) :
    (fockSandwichCLM N fockCounterRotating :
      FockCutoff N →ₗ[ℂ] FockCutoff N).IsSymmetric :=
  fockSandwichCLM_isSymmetric_of_isHermitian N _
    (fockSandwichMatrix_counterRotating_isHermitian N)

end Cvdv
