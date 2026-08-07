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

namespace Cvdv

/-- The manuscript's strict-sandwich convention is not vacuous: the sandwich is genuinely
nonmultiplicative, `(Π_N q̂ Π_N)^4 ≠ Π_N q̂^4 Π_N` (the sentence following the display
defining `Ĥ₄^{(N)} = Π_N λ (â + â†)⁴ Π_N`).  The witness is the smallest cutoff `N = 1`,
whose retained space is spanned by `|0⟩`: the sandwiched matrix of `â + â†` is the `1 × 1`
zero matrix (the `(0,0)` entry of `â + â†` vanishes), so its fourth power is zero, while the
sandwich of `(â + â†)⁴` has `(0,0)` entry `⟨0|(â + â†)⁴|0⟩ = 3`.  The physical quadrature is
`q̂ = (â + â†)/√(2ω₀)`; its scalar prefactor is irrelevant to nonmultiplicativity, so the
witness is stated for `â + â†` itself. -/
theorem cutoffSandwich_pow_four_ne :
    (fockSandwichMatrix 1 (lowering + raising)) ^ 4 ≠
      fockSandwichMatrix 1 ((lowering + raising) ^ 4) := by
  have hA0 : (lowering + raising) (Finsupp.single 0 1) = Finsupp.single 1 1 := by
    simp
  have hA1 : (lowering + raising) (Finsupp.single 1 1) =
      Finsupp.single 0 1 + (Real.sqrt 2 : ℂ) • Finsupp.single 2 1 := by
    simp
    norm_num
  have hA2 : (lowering + raising) (Finsupp.single 2 1) =
      (Real.sqrt 2 : ℂ) • Finsupp.single 1 1 + (Real.sqrt 3 : ℂ) • Finsupp.single 3 1 := by
    simp
    norm_num
  have hA3 : (lowering + raising) (Finsupp.single 3 1) =
      (Real.sqrt 3 : ℂ) • Finsupp.single 2 1 + (Real.sqrt 4 : ℂ) • Finsupp.single 4 1 := by
    simp
    norm_num
  have hpow : (((lowering + raising) ^ 4) (Finsupp.single 0 1)) 0 = 3 := by
    have h4 : (lowering + raising) ^ 4 =
        (((lowering + raising) * (lowering + raising)) * (lowering + raising)) *
          (lowering + raising) := by
      rw [pow_succ, pow_succ, pow_succ, pow_one]
    rw [h4]
    simp only [Module.End.mul_apply]
    simp only [map_add, map_smul, hA0, hA1, hA2, hA3]
    simp
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
    norm_num
  have hL : fockSandwichMatrix 1 (lowering + raising) = 0 := by
    ext i j
    have hi : i = 0 := Subsingleton.elim i 0
    have hj : j = 0 := Subsingleton.elim j 0
    subst hi
    subst hj
    rw [fockSandwichMatrix_apply, Matrix.zero_apply]
    simp [hA0]
  intro hcontra
  have h00 : ((fockSandwichMatrix 1 (lowering + raising)) ^ 4) 0 0 = 3 := by
    rw [hcontra, fockSandwichMatrix_apply]
    simpa using hpow
  rw [hL, zero_pow (by norm_num : (4 : ℕ) ≠ 0)] at h00
  norm_num [Matrix.zero_apply] at h00

/-- Nonzero-scalar transfer of sandwich nonmultiplicativity: if the strict sandwich of a
fourth power differs from the fourth power of the sandwich for `A`, the same holds for
`c • A` with any nonzero scalar `c`, because both sides simply pick up `c⁴`. -/
theorem cutoffSandwich_pow_four_ne_smul {N : ℕ} {A : Module.End ℂ FockAlg} {c : ℂ}
    (hc : c ≠ 0)
    (hA : (fockSandwichMatrix N A) ^ 4 ≠ fockSandwichMatrix N (A ^ 4)) :
    (fockSandwichMatrix N (c • A)) ^ 4 ≠ fockSandwichMatrix N ((c • A) ^ 4) := by
  intro hcontra
  apply hA
  have hL : (fockSandwichMatrix N (c • A)) ^ 4
      = c ^ 4 • (fockSandwichMatrix N A) ^ 4 := by
    rw [map_smul, smul_pow]
  have hR : fockSandwichMatrix N ((c • A) ^ 4)
      = c ^ 4 • fockSandwichMatrix N (A ^ 4) := by
    rw [smul_pow, map_smul]
  rw [hL, hR] at hcontra
  exact smul_right_injective _ (pow_ne_zero 4 hc) hcontra

/-- Manuscript lines 1215-1217 for the physical quadrature itself: with
`q̂ = (â + â†)/√(2ω₀)` at any `ω₀ > 0`, the strict sandwich is genuinely
nonmultiplicative, `(Π_N q̂ Π_N)⁴ ≠ Π_N q̂⁴ Π_N`, again at the smallest cutoff `N = 1`.
The two `(0,0)` entries are `0` and `3/(4ω₀²)`; the statement follows from the unscaled
witness `cutoffSandwich_pow_four_ne` by the nonzero-scalar transfer lemma with
`c = 1/√(2ω₀)`. -/
theorem cutoffSandwich_pow_four_ne_quadrature (omega0 : ℝ) (h : 0 < omega0) :
    (fockSandwichMatrix 1
        (((1 / Real.sqrt (2 * omega0) : ℝ) : ℂ) • (lowering + raising))) ^ 4 ≠
      fockSandwichMatrix 1
        ((((1 / Real.sqrt (2 * omega0) : ℝ) : ℂ) • (lowering + raising)) ^ 4) := by
  have hs : (0 : ℝ) < Real.sqrt (2 * omega0) := Real.sqrt_pos.mpr (by linarith)
  have hc : ((1 / Real.sqrt (2 * omega0) : ℝ) : ℂ) ≠ 0 := by
    exact_mod_cast (one_div_pos.mpr hs).ne'
  exact cutoffSandwich_pow_four_ne_smul hc cutoffSandwich_pow_four_ne

end Cvdv
