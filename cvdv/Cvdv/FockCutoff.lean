import Cvdv.FockAlgebra
import Mathlib.Analysis.CStarAlgebra.Matrix

/-!
# Finite Fock cutoffs of algebraic operators

This file implements the manuscript convention that an operator polynomial is constructed on the
algebraic Fock space before the finite cutoff is applied.  The algebraic space is `ℕ →₀ ℂ` and
the retained space has basis indexed by `Fin N`.

For an algebraic endomorphism `A`, its strict finite sandwich is

`fockProject N ∘ A ∘ fockEmbed N`.

It is then represented both as an `N × N` matrix and as a continuous endomorphism of
`EuclideanSpace ℂ (Fin N)`.  The sandwich is linear in `A`, but it is not multiplicative: the
products and powers below always remain inside the sandwich until after the infinite algebraic
identity has been established.
-/

namespace Cvdv

/-- Algebraic Fock space: finite linear combinations of number states. -/
abbrev FockAlg := ℕ →₀ ℂ

/-- Algebraic retained subspace with number states `0, ..., N - 1`. -/
abbrev FockCutoffAlg (N : ℕ) := Fin N →₀ ℂ

/-- The retained finite-dimensional Hilbert space. -/
abbrev FockCutoff (N : ℕ) := EuclideanSpace ℂ (Fin N)

/-- Include the first `N` number states into the algebraic Fock space. -/
noncomputable def fockEmbed (N : ℕ) : FockCutoffAlg N →ₗ[ℂ] FockAlg :=
  Finsupp.lmapDomain ℂ ℂ (fun i : Fin N ↦ i.val)

/-- Restrict an algebraic Fock vector to its first `N` coefficients. -/
noncomputable def fockProject (N : ℕ) : FockAlg →ₗ[ℂ] FockCutoffAlg N :=
  Finsupp.lcomapDomain (R := ℂ) (M := ℂ) (fun i : Fin N ↦ i.val) Fin.val_injective

@[simp]
theorem fockProject_apply (N : ℕ) (x : FockAlg) (i : Fin N) :
    fockProject N x i = x i.val :=
  rfl

@[simp]
theorem fockProject_fockEmbed (N : ℕ) :
    (fockProject N).comp (fockEmbed N) = LinearMap.id := by
  ext x i
  simp [fockProject, fockEmbed]

/-- The strict finite sandwich `P_N A I_N`, kept as an algebraic endomorphism. -/
noncomputable def fockSandwichAlg (N : ℕ) :
    Module.End ℂ FockAlg →ₗ[ℂ] Module.End ℂ (FockCutoffAlg N) where
  toFun A := (fockProject N).comp (A.comp (fockEmbed N))
  map_add' A B := by ext x i; rfl
  map_smul' c A := by ext x i; rfl

@[simp]
theorem fockSandwichAlg_apply (N : ℕ) (A : Module.End ℂ FockAlg)
    (x : FockCutoffAlg N) (i : Fin N) :
    fockSandwichAlg N A x i = A (fockEmbed N x) i.val :=
  rfl

@[simp]
theorem fockSandwichAlg_one (N : ℕ) : fockSandwichAlg N 1 = 1 := by
  ext x i
  simp [fockSandwichAlg, fockProject, fockEmbed]

/-- Matrix elements of the strict sandwich in the retained number-state basis. -/
noncomputable def fockSandwichMatrix (N : ℕ) :
    Module.End ℂ FockAlg →ₗ[ℂ] Matrix (Fin N) (Fin N) ℂ where
  toFun A i j := fockSandwichAlg N A (Finsupp.single j 1) i
  map_add' A B := by ext i j; rfl
  map_smul' c A := by ext i j; rfl

@[simp]
theorem fockSandwichMatrix_apply (N : ℕ) (A : Module.End ℂ FockAlg) (i j : Fin N) :
    fockSandwichMatrix N A i j = A (Finsupp.single j.val 1) i.val := by
  change fockSandwichAlg N A (Finsupp.single j 1) i = _
  rw [fockSandwichAlg_apply]
  simp [fockEmbed]

@[simp]
theorem fockSandwichMatrix_one (N : ℕ) : fockSandwichMatrix N 1 = 1 := by
  ext i j
  rw [fockSandwichMatrix_apply]
  by_cases h : i = j
  · subst j
    simp
  · have hval : j.val ≠ i.val := fun hji ↦ h (Fin.ext hji.symm)
    simp [h, hval]

/-- The strict sandwich as a bounded operator on the retained Hilbert space.  Continuity is
automatic in finite dimension. -/
noncomputable def fockSandwichCLM (N : ℕ) :
    Module.End ℂ FockAlg →ₗ[ℂ] (FockCutoff N →L[ℂ] FockCutoff N) where
  toFun A := Matrix.toEuclideanCLM (n := Fin N) (𝕜 := ℂ) (fockSandwichMatrix N A)
  map_add' A B := by simp
  map_smul' c A := by simp

@[simp]
theorem fockSandwichCLM_apply (N : ℕ) (A : Module.End ℂ FockAlg) (x : FockCutoff N) :
    WithLp.ofLp (fockSandwichCLM N A x) =
      Matrix.mulVec (fockSandwichMatrix N A) (WithLp.ofLp x) := by
  rfl

@[simp]
theorem fockSandwichCLM_one (N : ℕ) : fockSandwichCLM N 1 = 1 := by
  simp [fockSandwichCLM]

/-- The concrete quartic normal-ordering identity after taking strict finite matrix elements.
Every product and power is formed on `FockAlg` before `fockSandwichMatrix` is applied. -/
theorem fock_add_pow_four_split_sandwich_matrix (N : ℕ) :
    fockSandwichMatrix N ((lowering + raising) ^ 4) =
      fockSandwichMatrix N
        (6 * ((raising * lowering) * (raising * lowering)) + 6 * (raising * lowering) + 3 +
          (raising ^ 4 + 4 * (raising ^ 3 * lowering) + lowering ^ 4
            + 4 * (raising * lowering ^ 3) + 6 * lowering ^ 2 + 6 * raising ^ 2)) := by
  exact congrArg (fockSandwichMatrix N) fock_add_pow_four_split

/-- The concrete quartic normal-ordering identity as bounded operators on the finite Hilbert
space.  This theorem does not replace sandwiched products by products of sandwiched factors. -/
theorem fock_add_pow_four_split_sandwich (N : ℕ) :
    fockSandwichCLM N ((lowering + raising) ^ 4) =
      (6 : ℂ) • fockSandwichCLM N ((raising * lowering) * (raising * lowering))
        + (6 : ℂ) • fockSandwichCLM N (raising * lowering)
        + (3 : ℂ) • 1
        + fockSandwichCLM N
            (raising ^ 4 + 4 * (raising ^ 3 * lowering) + lowering ^ 4
              + 4 * (raising * lowering ^ 3) + 6 * lowering ^ 2 + 6 * raising ^ 2) := by
  have h := congrArg (fockSandwichCLM N)
    (add_pow_four_split_smul lowering raising lowering_mul_raising)
  simpa only [map_add, map_smul, fockSandwichCLM_one] using h

end Cvdv
