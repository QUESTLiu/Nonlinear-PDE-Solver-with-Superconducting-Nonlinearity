import Cvdv.FockCutoff

/-!
# Strict-cutoff commutators with the Kerr Hamiltonian

The Fock number operator and every polynomial in it preserve each number state.  Consequently,
strict sandwiching preserves left and right multiplication by the Kerr Hamiltonian, even though
strict sandwiching is not multiplicative for general algebraic operators.
-/

namespace Cvdv

/-- The algebraic Fock number operator `n = a† a`. -/
noncomputable def fockNumber : Module.End ℂ FockAlg := raising * lowering

@[simp]
theorem fockNumber_single (n : ℕ) (c : ℂ) :
    fockNumber (Finsupp.single n c) = (n : ℂ) • Finsupp.single n c := by
  cases n with
  | zero => simp [fockNumber, Module.End.mul_apply]
  | succ n =>
      simp only [fockNumber, Module.End.mul_apply, lowering_single, raising_single,
        Nat.succ_sub_one, Finsupp.smul_single, smul_eq_mul, mul_one]
      congr 1
      push_cast
      rw [mul_assoc, ← Complex.ofReal_mul,
        Real.mul_self_sqrt (by positivity : (0 : ℝ) ≤ n + 1)]
      push_cast
      ring

@[simp]
theorem fockNumber_apply (x : FockAlg) (n : ℕ) :
    fockNumber x n = (n : ℂ) * x n := by
  classical
  induction x using Finsupp.induction with
  | zero => simp
  | @single_add p c x hp hc ih =>
      rw [map_add]
      change fockNumber (Finsupp.single p c) n + fockNumber x n =
        (n : ℂ) * (Finsupp.single p c n + x n)
      rw [fockNumber_single]
      by_cases h : p = n
      · subst n
        simp [ih, mul_add]
      · simp [h, ih]

@[simp]
theorem fockNumber_sq_apply (x : FockAlg) (n : ℕ) :
    (fockNumber ^ 2) x n = (n : ℂ) ^ 2 * x n := by
  simp [pow_two, Module.End.mul_apply, fockNumber_apply]
  ring

/-- The algebraic Kerr Hamiltonian `ω n + K n²`. -/
noncomputable def fockKerrHamiltonian (ω K : ℂ) : Module.End ℂ FockAlg :=
  ω • fockNumber + K • fockNumber ^ 2

@[simp]
theorem fockKerrHamiltonian_apply (ω K : ℂ) (x : FockAlg) (n : ℕ) :
    fockKerrHamiltonian ω K x n = (ω * n + K * n ^ 2) * x n := by
  simp [fockKerrHamiltonian, fockNumber_apply, fockNumber_sq_apply]
  ring

/-- The strict sandwich of a number-basis diagonal algebraic operator is its retained diagonal
matrix. -/
theorem fockSandwichMatrix_eq_diagonal_of_apply (N : ℕ)
    (D : Module.End ℂ FockAlg) (d : ℕ → ℂ)
    (hD : ∀ x n, D x n = d n * x n) :
    fockSandwichMatrix N D = Matrix.diagonal (fun i : Fin N ↦ d i.val) := by
  classical
  ext i j
  rw [fockSandwichMatrix_apply, hD]
  by_cases h : i = j
  · subst j
    simp
  · have hval : j.val ≠ i.val := fun hji ↦ h (Fin.ext hji.symm)
    simp [h, hval]

/-- Strict matrix sandwiching preserves left multiplication by an algebraic operator that is
diagonal in the number basis. -/
theorem fockSandwichMatrix_mul_left_of_diagonal (N : ℕ)
    (D A : Module.End ℂ FockAlg) (d : ℕ → ℂ)
    (hD : ∀ x n, D x n = d n * x n) :
    fockSandwichMatrix N (D * A) =
      fockSandwichMatrix N D * fockSandwichMatrix N A := by
  classical
  ext i j
  rw [fockSandwichMatrix_apply, Module.End.mul_apply, hD,
    fockSandwichMatrix_eq_diagonal_of_apply N D d hD, Matrix.diagonal_mul,
    fockSandwichMatrix_apply]

/-- Strict matrix sandwiching preserves right multiplication by an algebraic operator that is
diagonal in the number basis. -/
theorem fockSandwichMatrix_mul_right_of_diagonal (N : ℕ)
    (A D : Module.End ℂ FockAlg) (d : ℕ → ℂ)
    (hD : ∀ x n, D x n = d n * x n) :
    fockSandwichMatrix N (A * D) =
      fockSandwichMatrix N A * fockSandwichMatrix N D := by
  classical
  ext i j
  rw [fockSandwichMatrix_apply, Module.End.mul_apply]
  have hsingle : D (Finsupp.single j.val 1) = d j.val • Finsupp.single j.val 1 := by
    ext n
    rw [hD]
    by_cases h : j.val = n
    · subst n
      simp
    · simp [h]
  rw [hsingle, map_smul,
    fockSandwichMatrix_eq_diagonal_of_apply N D d hD, Matrix.mul_diagonal,
    fockSandwichMatrix_apply]
  change d j.val * A (Finsupp.single j.val 1) i.val =
    A (Finsupp.single j.val 1) i.val * d j.val
  exact mul_comm _ _

/-- Strict matrix sandwiching preserves the commutator with the Kerr Hamiltonian. -/
theorem fockSandwichMatrix_kerr_commutator (N : ℕ) (ω K : ℂ)
    (A : Module.End ℂ FockAlg) :
    fockSandwichMatrix N (fockKerrHamiltonian ω K * A - A * fockKerrHamiltonian ω K) =
      fockSandwichMatrix N (fockKerrHamiltonian ω K) * fockSandwichMatrix N A -
        fockSandwichMatrix N A * fockSandwichMatrix N (fockKerrHamiltonian ω K) := by
  calc
    _ = fockSandwichMatrix N (fockKerrHamiltonian ω K * A) -
        fockSandwichMatrix N (A * fockKerrHamiltonian ω K) :=
      map_sub (fockSandwichMatrix N) _ _
    _ = _ := congrArg₂ (· - ·)
      (fockSandwichMatrix_mul_left_of_diagonal N _ _ _
        (fockKerrHamiltonian_apply ω K))
      (fockSandwichMatrix_mul_right_of_diagonal N _ _ _
        (fockKerrHamiltonian_apply ω K))

/-- Paper-facing bounded-operator form of strict cutoff compatibility with the Kerr commutator. -/
theorem fockSandwichCLM_kerr_commutator (N : ℕ) (ω K : ℂ)
    (A : Module.End ℂ FockAlg) :
    fockSandwichCLM N (fockKerrHamiltonian ω K * A - A * fockKerrHamiltonian ω K) =
      fockSandwichCLM N (fockKerrHamiltonian ω K) * fockSandwichCLM N A -
        fockSandwichCLM N A * fockSandwichCLM N (fockKerrHamiltonian ω K) := by
  have h := congrArg
    (Matrix.toEuclideanCLM (n := Fin N) (𝕜 := ℂ))
    (fockSandwichMatrix_kerr_commutator N ω K A)
  simpa [fockSandwichCLM] using h

end Cvdv
