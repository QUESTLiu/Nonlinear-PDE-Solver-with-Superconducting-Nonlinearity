import Cvdv.FockAlgebra
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.Module
import Mathlib.Data.Complex.Basic
import Mathlib.Algebra.Algebra.Defs

/-!
# Quadratic generators close the first-moment hierarchy

Formalization of the algebraic core of manuscript Sec. (subsec:encoding): for a quadratic element of
the CCR algebra, the commutator with a mode operator is affine in the mode operators,
so the Heisenberg equations of the first moments form a closed linear system.

As in `Cvdv.FockAlgebra`, the canonical commutation relation is encoded in an abstract
ring by `x` (annihilation) and `y` (creation) with `x * y = y * x + 1`.  Five helper
commutators are proved first, in the orientation `A * B - B * A`:
`[y x, x] = -x`, `[y², x] = -2 y`, `[y, x] = -1`, `[y x, y] = y`, `[x², y] = 2 x`.

For the general quadratic generator
`H = a • (y x) + b • (y y) + c • (x x) + d • y + e • x` over an algebra `[Algebra ℂ R]`,
the main results compute
`closure_x : H x - x H = (-a) • x + (-(2 b)) • y + algebraMap ℂ R (-d)` and
`closure_y : H y - y H = a • y + (2 c) • x + algebraMap ℂ R e`,
so the commutator of `H` with either mode operator is again affine in `x`, `y`, `1`.

Scope: this is the closure direction of the paper's assertion — a quadratic generator
keeps the first-moment equations closed.  The converse half of the manuscript's iff
sentence is state-dependent and is not claimed here; the failure of closure for the
quartic interaction is quantified separately in `Cvdv.FockAlgebra`.
-/

namespace Cvdv

section CCRCommutators

variable {R : Type*} [Ring R]

/-- Commutator of the number element with the annihilator: `[y x, x] = -x`. -/
theorem comm_number_ann (x y : R) (h : x * y = y * x + 1) :
    (y * x) * x - x * (y * x) = -x := by
  calc (y * x) * x - x * (y * x) = (y * x) * x - (x * y) * x := by noncomm_ring
    _ = (y * x) * x - (y * x + 1) * x := by rw [h]
    _ = -x := by noncomm_ring

/-- Commutator of the squared creator with the annihilator: `[y², x] = -2 y`. -/
theorem comm_creation_sq_ann (x y : R) (h : x * y = y * x + 1) :
    y ^ 2 * x - x * y ^ 2 = -(2 * y) := by
  rw [mul_creation_sq x y h]
  noncomm_ring

/-- Commutator of the creator with the annihilator: `[y, x] = -1`. -/
theorem comm_creation_ann (x y : R) (h : x * y = y * x + 1) :
    y * x - x * y = -1 := by
  rw [h]
  noncomm_ring

/-- Commutator of the number element with the creator: `[y x, y] = y`. -/
theorem comm_number_creation (x y : R) (h : x * y = y * x + 1) :
    (y * x) * y - y * (y * x) = y := by
  calc (y * x) * y - y * (y * x) = y * (x * y) - y * (y * x) := by noncomm_ring
    _ = y * (y * x + 1) - y * (y * x) := by rw [h]
    _ = y := by noncomm_ring

/-- Commutator of the squared annihilator with the creator: `[x², y] = 2 x`. -/
theorem comm_ann_sq_creation (x y : R) (h : x * y = y * x + 1) :
    x ^ 2 * y - y * x ^ 2 = 2 * x := by
  rw [sq_mul_creation x y h]
  noncomm_ring

end CCRCommutators

section QuadraticClosure

variable {R : Type*} [Ring R] [Algebra ℂ R]

/-- Sec. (subsec:encoding), annihilator closure for the general quadratic
generator `H = a • (y x) + b • (y y) + c • (x x) + d • y + e • x`, the commutator
`H x - x H` is affine in the mode operators:
`H x - x H = (-a) • x + (-(2 b)) • y + algebraMap ℂ R (-d)`. -/
theorem closure_x (x y : R) (h : x * y = y * x + 1) (a b c d e : ℂ) :
    (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x) * x
      - x * (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x)
      = (-a) • x + (-(2 * b)) • y + algebraMap ℂ R (-d) := by
  have h1 := comm_number_ann x y h
  have h2 := comm_creation_sq_ann x y h
  have h3 := comm_creation_ann x y h
  have hc : (x * x) * x - x * (x * x) = (0 : R) := by noncomm_ring
  have htwo : (2 : R) * y = (2 : ℂ) • y := by rw [Algebra.smul_def, map_ofNat]
  calc (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x) * x
        - x * (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x)
      = a • ((y * x) * x - x * (y * x)) + b • (y ^ 2 * x - x * y ^ 2)
          + c • ((x * x) * x - x * (x * x)) + d • (y * x - x * y)
          + e • (x * x - x * x) := by
        simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm, pow_two]
        module
    _ = a • (-x) + b • (-(2 * y)) + c • (0 : R) + d • (-1 : R)
          + e • (x * x - x * x) := by
        rw [h1, h2, h3, hc]
    _ = (-a) • x + (-(2 * b)) • y + algebraMap ℂ R (-d) := by
        rw [sub_self, htwo, Algebra.algebraMap_eq_smul_one]
        module

/-- Manuscript Sec. (subsec:encoding), closure of the creator equation: for the general quadratic
generator `H = a • (y x) + b • (y y) + c • (x x) + d • y + e • x`, the commutator
`H y - y H` is affine in the mode operators:
`H y - y H = a • y + (2 c) • x + algebraMap ℂ R e`. -/
theorem closure_y (x y : R) (h : x * y = y * x + 1) (a b c d e : ℂ) :
    (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x) * y
      - y * (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x)
      = a • y + (2 * c) • x + algebraMap ℂ R e := by
  have h4 := comm_number_creation x y h
  have h5 := comm_ann_sq_creation x y h
  have he : x * y - y * x = (1 : R) := by rw [h]; noncomm_ring
  have hb : (y * y) * y - y * (y * y) = (0 : R) := by noncomm_ring
  have htwo : (2 : R) * x = (2 : ℂ) • x := by rw [Algebra.smul_def, map_ofNat]
  calc (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x) * y
        - y * (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x)
      = a • ((y * x) * y - y * (y * x)) + b • ((y * y) * y - y * (y * y))
          + c • (x ^ 2 * y - y * x ^ 2) + d • (y * y - y * y)
          + e • (x * y - y * x) := by
        simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm, pow_two]
        module
    _ = a • y + b • (0 : R) + c • (2 * x) + d • (y * y - y * y) + e • (1 : R) := by
        rw [h4, h5, hb, he]
    _ = a • y + (2 * c) • x + algebraMap ℂ R e := by
        rw [sub_self, htwo, Algebra.algebraMap_eq_smul_one]
        module

/-- Image of `closure_x` under a complex-linear functional.
The identity term remains `E 1` without normalization. -/
theorem closure_x_expectation (E : R →ₗ[ℂ] ℂ) (x y : R)
    (h : x * y = y * x + 1) (a b c d e : ℂ) :
    E ((a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x) * x
        - x * (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x))
      = (-a) • E x + (-(2 * b)) • E y + (-d) • E 1 := by
  rw [closure_x x y h a b c d e, map_add, map_add, map_smul, map_smul,
    Algebra.algebraMap_eq_smul_one, map_smul]

/-- Image of `closure_y` under a complex-linear functional.
The identity term remains `E 1` without normalization. -/
theorem closure_y_expectation (E : R →ₗ[ℂ] ℂ) (x y : R)
    (h : x * y = y * x + 1) (a b c d e : ℂ) :
    E ((a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x) * y
        - y * (a • (y * x) + b • (y * y) + c • (x * x) + d • y + e • x))
      = a • E y + (2 * c) • E x + e • E 1 := by
  rw [closure_y x y h a b c d e, map_add, map_add, map_smul, map_smul,
    Algebra.algebraMap_eq_smul_one, map_smul]

end QuadraticClosure

end Cvdv
