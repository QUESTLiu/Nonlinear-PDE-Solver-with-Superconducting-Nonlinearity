import Mathlib.Analysis.Normed.Algebra.MatrixExponential
import Mathlib.LinearAlgebra.Matrix.Kronecker

/-!
# Lugiato--Lefever loss-layer factorization

This file proves the finite commuting exponential factorization used in
Eq. (eq:lle-number-controlled-displacements). It does not construct a hardware
gate decomposition or prove a product-formula error bound.
-/

namespace Cvdv

open scoped BigOperators Kronecker Matrix

noncomputable section

section Factorization

variable {ι ιa ιs : Type*}
  [Fintype ι] [DecidableEq ι]
  [Fintype ιa] [DecidableEq ιa]
  [Fintype ιs] [DecidableEq ιs]

omit [DecidableEq ιa] [DecidableEq ιs] in
/-- Joint operators with the same ancillary factor commute when their system factors commute. -/
theorem commute_kronecker_same_left (x : Matrix ιa ιa ℂ)
    {A B : Matrix ιs ιs ℂ} (hAB : Commute A B) :
    Commute (x ⊗ₖ A) (x ⊗ₖ B) := by
  rw [Commute]
  calc
    (x ⊗ₖ A) * (x ⊗ₖ B) = (x * x) ⊗ₖ (A * B) :=
      (Matrix.mul_kronecker_mul x x A B).symm
    _ = (x * x) ⊗ₖ (B * A) := by rw [hAB.eq]
    _ = (x ⊗ₖ B) * (x ⊗ₖ A) := Matrix.mul_kronecker_mul x x B A

omit [DecidableEq ι] in
/-- The exponential of a finite sum of commuting loss-site generators factorizes exactly. -/
theorem lle_loss_exponential_factorization (x : Matrix ιa ιa ℂ)
    (n : ι → Matrix ιs ιs ℂ) (c : ℂ)
    (hn : ∀ j k, Commute (n j) (n k)) :
    NormedSpace.exp (((c • x) ⊗ₖ (∑ j, n j))) =
      Finset.univ.noncommProd
        (fun j => NormedSpace.exp ((c • x) ⊗ₖ n j))
        (fun j _ k _ _ => (commute_kronecker_same_left (c • x) (hn j k)).exp) := by
  have hsum :
      (c • x) ⊗ₖ (∑ j, n j) = ∑ j, (c • x) ⊗ₖ n j := by
    ext ⟨ia, is⟩ ⟨ja, js⟩
    simp only [Matrix.kronecker_apply, Matrix.sum_apply, Matrix.smul_apply]
    rw [Finset.mul_sum]
  rw [hsum]
  exact Matrix.exp_sum_of_commute Finset.univ
    (fun j => (c • x) ⊗ₖ n j)
    (fun j _ k _ _ => commute_kronecker_same_left (c • x) (hn j k))

end Factorization

section Displacement

variable {ιa ιs : Type*}
  [Fintype ιa] [DecidableEq ιa]
  [Fintype ιs] [DecidableEq ιs]

/-- The dimensionless ancillary position quadrature used in the manuscript. -/
def lleAncillaX (b : Matrix ιa ιa ℂ) : Matrix ιa ιa ℂ :=
  (((1 / Real.sqrt 2 : ℝ) : ℂ)) • (b + bᴴ)

/-- Matrix form of an operator-controlled ancillary displacement. -/
def lleControlledDisplacement (b : Matrix ιa ιa ℂ) (Γ : Matrix ιs ιs ℂ) :
    Matrix (ιa × ιs) (ιa × ιs) ℂ :=
  NormedSpace.exp (bᴴ ⊗ₖ Γ - b ⊗ₖ Γᴴ)

/-- The displacement parameter in Eq. (eq:lle-number-controlled-displacements),
with `τ = κ Δt / 2`. -/
def lleLossDisplacementParameter (τ : ℝ) (n : Matrix ιs ιs ℂ) : Matrix ιs ιs ℂ :=
  (-Complex.I * ((τ / Real.sqrt 2 : ℝ) : ℂ)) • n

/-- With the manuscript normalization `xₐ = (bₐ + bₐ†)/√2`, one loss-site
exponential is exactly the corresponding operator-controlled displacement. -/
theorem lle_loss_site_exponential_eq_controlledDisplacement
    (b : Matrix ιa ιa ℂ) (n : Matrix ιs ιs ℂ) (hn : n.IsHermitian) (τ : ℝ) :
    NormedSpace.exp (((-Complex.I * (τ : ℂ)) • lleAncillaX b) ⊗ₖ n) =
      lleControlledDisplacement b (lleLossDisplacementParameter τ n) := by
  unfold lleControlledDisplacement
  apply congrArg NormedSpace.exp
  have hparam :
      (lleLossDisplacementParameter τ n)ᴴ =
        (Complex.I * ((τ / Real.sqrt 2 : ℝ) : ℂ)) • n := by
    rw [lleLossDisplacementParameter, Matrix.conjTranspose_smul, hn.eq]
    simp
  rw [hparam]
  ext ⟨ia, is⟩ ⟨ja, js⟩
  simp only [lleAncillaX, lleLossDisplacementParameter, Matrix.kronecker_apply,
    Matrix.smul_apply, Matrix.add_apply, Matrix.sub_apply]
  push_cast
  ring

end Displacement

end

end Cvdv
