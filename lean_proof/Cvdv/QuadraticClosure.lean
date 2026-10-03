import Cvdv.FockAlgebra
import Mathlib.Algebra.Module.BigOperators
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
sentence is state-dependent and is not claimed here, and the non-closure direction is
witnessed by `nonclosure_witness` below: on the Fock representation, the commutator of
the squared number operator with the annihilator is not an affine combination of the
mode operators and the identity.
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

noncomputable section

section MultimodeClosure

variable {ι R : Type*} [Fintype ι] [DecidableEq ι] [Ring R] [Algebra ℂ R]

/-- Multimode canonical commutation relations for finite families of annihilators `x j`
and creators `y j`: `x j * y k = y k * x j + δⱼₖ`, with annihilators commuting among
themselves and creators commuting among themselves (manuscript Sec. (subsec:encoding)). -/
structure MultiCCR (x y : ι → R) : Prop where
  ann_creation : ∀ j k, x j * y k = y k * x j + if j = k then 1 else 0
  ann_ann : ∀ j k, x j * x k = x k * x j
  creation_creation : ∀ j k, y j * y k = y k * y j

omit [Fintype ι] [Algebra ℂ R] in
/-- Two annihilators commute: `[xⱼ, xₘ] = 0`. -/
theorem multi_comm_ann_ann (x y : ι → R) (h : MultiCCR x y) (j m : ι) :
    x j * x m - x m * x j = 0 := by
  rw [h.ann_ann j m]; exact sub_self _

omit [Fintype ι] [Algebra ℂ R] in
/-- Two creators commute: `[yⱼ, yₘ] = 0`. -/
theorem multi_comm_creation_creation (x y : ι → R) (h : MultiCCR x y) (j m : ι) :
    y j * y m - y m * y j = 0 := by
  rw [h.creation_creation j m]; exact sub_self _

omit [Fintype ι] [Algebra ℂ R] in
/-- Creator against annihilator: `[yⱼ, xₘ] = -δₘⱼ`. -/
theorem multi_comm_creation_ann (x y : ι → R) (h : MultiCCR x y) (j m : ι) :
    y j * x m - x m * y j = -(if m = j then (1 : R) else 0) := by
  rw [h.ann_creation m j]; noncomm_ring

omit [Fintype ι] [Algebra ℂ R] in
/-- Annihilator against creator: `[xⱼ, yₘ] = δⱼₘ`. -/
theorem multi_comm_ann_creation (x y : ι → R) (h : MultiCCR x y) (j m : ι) :
    x j * y m - y m * x j = if j = m then (1 : R) else 0 := by
  rw [h.ann_creation j m]; noncomm_ring

omit [Fintype ι] [Algebra ℂ R] in
/-- Number-conserving bilinear against an annihilator: `[yⱼxₖ, xₘ] = -δₘⱼ xₖ`. -/
theorem multi_comm_bilinear_ann (x y : ι → R) (h : MultiCCR x y) (j k m : ι) :
    (y j * x k) * x m - x m * (y j * x k) = if m = j then -(x k) else 0 := by
  have hmj := h.ann_creation m j
  calc (y j * x k) * x m - x m * (y j * x k)
      = (y j * x k) * x m - (x m * y j) * x k := by noncomm_ring
    _ = (y j * x k) * x m - (y j * x m + if m = j then (1 : R) else 0) * x k := by
        rw [hmj]
    _ = y j * (x k * x m) - y j * (x m * x k)
          - (if m = j then (1 : R) else 0) * x k := by noncomm_ring
    _ = y j * (x k * x m) - y j * (x k * x m)
          - (if m = j then (1 : R) else 0) * x k := by rw [h.ann_ann m k]
    _ = -((if m = j then (1 : R) else 0) * x k) := by noncomm_ring
    _ = if m = j then -(x k) else 0 := by split_ifs <;> simp

omit [Fintype ι] [Algebra ℂ R] in
/-- Number-conserving bilinear against a creator: `[yⱼxₖ, yₘ] = δₖₘ yⱼ`. -/
theorem multi_comm_bilinear_creation (x y : ι → R) (h : MultiCCR x y) (j k m : ι) :
    (y j * x k) * y m - y m * (y j * x k) = if k = m then y j else 0 := by
  have hkm := h.ann_creation k m
  calc (y j * x k) * y m - y m * (y j * x k)
      = y j * (x k * y m) - (y m * y j) * x k := by noncomm_ring
    _ = y j * (y m * x k + if k = m then (1 : R) else 0) - (y m * y j) * x k := by
        rw [hkm]
    _ = y j * (y m * x k) - (y m * y j) * x k
          + y j * (if k = m then (1 : R) else 0) := by noncomm_ring
    _ = y j * (y m * x k) - (y j * y m) * x k
          + y j * (if k = m then (1 : R) else 0) := by rw [h.creation_creation m j]
    _ = y j * (if k = m then (1 : R) else 0) := by noncomm_ring
    _ = if k = m then y j else 0 := by split_ifs <;> simp

omit [Fintype ι] [Algebra ℂ R] in
/-- Anomalous creator pair against an annihilator: `[yⱼyₖ, xₘ] = -δₘₖ yⱼ - δₘⱼ yₖ`. -/
theorem multi_comm_creation_pair_ann (x y : ι → R) (h : MultiCCR x y) (j k m : ι) :
    (y j * y k) * x m - x m * (y j * y k)
      = (if m = k then -(y j) else 0) + (if m = j then -(y k) else 0) := by
  have hmj := h.ann_creation m j
  have hmk := h.ann_creation m k
  calc (y j * y k) * x m - x m * (y j * y k)
      = (y j * y k) * x m - (x m * y j) * y k := by noncomm_ring
    _ = (y j * y k) * x m - (y j * x m + if m = j then (1 : R) else 0) * y k := by
        rw [hmj]
    _ = (y j * y k) * x m - y j * (x m * y k)
          - (if m = j then (1 : R) else 0) * y k := by noncomm_ring
    _ = (y j * y k) * x m - y j * (y k * x m + if m = k then (1 : R) else 0)
          - (if m = j then (1 : R) else 0) * y k := by rw [hmk]
    _ = -(y j * (if m = k then (1 : R) else 0))
          - (if m = j then (1 : R) else 0) * y k := by noncomm_ring
    _ = (if m = k then -(y j) else 0) + (if m = j then -(y k) else 0) := by
        split_ifs <;> simp [sub_eq_add_neg]

omit [Fintype ι] [Algebra ℂ R] in
/-- Anomalous annihilator pair against a creator: `[xⱼxₖ, yₘ] = δⱼₘ xₖ + δₖₘ xⱼ`. -/
theorem multi_comm_ann_pair_creation (x y : ι → R) (h : MultiCCR x y) (j k m : ι) :
    (x j * x k) * y m - y m * (x j * x k)
      = (if j = m then x k else 0) + (if k = m then x j else 0) := by
  have hjm := h.ann_creation j m
  have hkm := h.ann_creation k m
  calc (x j * x k) * y m - y m * (x j * x k)
      = x j * (x k * y m) - y m * (x j * x k) := by noncomm_ring
    _ = x j * (y m * x k + if k = m then (1 : R) else 0) - y m * (x j * x k) := by
        rw [hkm]
    _ = (x j * y m) * x k - y m * (x j * x k)
          + x j * (if k = m then (1 : R) else 0) := by noncomm_ring
    _ = (y m * x j + if j = m then (1 : R) else 0) * x k - y m * (x j * x k)
          + x j * (if k = m then (1 : R) else 0) := by rw [hjm]
    _ = (if j = m then (1 : R) else 0) * x k
          + x j * (if k = m then (1 : R) else 0) := by noncomm_ring
    _ = (if j = m then x k else 0) + (if k = m then x j else 0) := by
        split_ifs <;> simp

omit [Fintype ι] [Algebra ℂ R] in
/-- Anomalous annihilator pair against an annihilator: `[xⱼxₖ, xₘ] = 0`. -/
theorem multi_comm_ann_pair_ann (x y : ι → R) (h : MultiCCR x y) (j k m : ι) :
    (x j * x k) * x m - x m * (x j * x k) = 0 := by
  calc (x j * x k) * x m - x m * (x j * x k)
      = x j * (x k * x m) - (x m * x j) * x k := by noncomm_ring
    _ = x j * (x m * x k) - (x j * x m) * x k := by rw [h.ann_ann k m, h.ann_ann m j]
    _ = 0 := by noncomm_ring

omit [Fintype ι] [Algebra ℂ R] in
/-- Anomalous creator pair against a creator: `[yⱼyₖ, yₘ] = 0`. -/
theorem multi_comm_creation_pair_creation (x y : ι → R) (h : MultiCCR x y) (j k m : ι) :
    (y j * y k) * y m - y m * (y j * y k) = 0 := by
  calc (y j * y k) * y m - y m * (y j * y k)
      = y j * (y k * y m) - (y m * y j) * y k := by noncomm_ring
    _ = y j * (y m * y k) - (y j * y m) * y k := by
        rw [h.creation_creation k m, h.creation_creation m j]
    _ = 0 := by noncomm_ring

/-- Commutation with a sum is the sum of the commutators. -/
private theorem multi_comm_sum {α S : Type*} [Ring S] (s : Finset α) (f : α → S)
    (z : S) :
    (∑ j ∈ s, f j) * z - z * ∑ j ∈ s, f j = ∑ j ∈ s, (f j * z - z * f j) := by
  rw [Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]

/-- A sum of an index-independent `if` is the `if` of the sum. -/
private theorem multi_sum_ite_const {α S : Type*} [Ring S] (P : Prop) [Decidable P]
    (s : Finset α) (f : α → S) :
    (∑ j ∈ s, if P then f j else 0) = if P then ∑ j ∈ s, f j else 0 := by
  split_ifs <;> simp

/-- The general index-quadratic generator
`H = ∑ⱼₖ (aⱼₖ yⱼxₖ + bⱼₖ yⱼyₖ + cⱼₖ xⱼxₖ) + ∑ⱼ (dⱼ yⱼ + eⱼ xⱼ)`:
the diagonal `aⱼⱼ` are onsite number terms, the off-diagonal `aⱼₖ` are cross-site
hopping bilinears, `bⱼₖ`/`cⱼₖ` are anomalous pair bilinears, and `dⱼ`/`eⱼ` are linear
drive terms.  The sine–Gordon lattice Hamiltonian of manuscript Sec. (sec:sg) at
`E_J = 0` — onsite harmonic terms, nearest-neighbor quadratic coupling, and linear
terms — is of this form. -/
def multiQuadGen (x y : ι → R) (a b c : ι → ι → ℂ) (d e : ι → ℂ) : R :=
  (∑ j, ∑ k, (a j k • (y j * x k) + b j k • (y j * y k) + c j k • (x j * x k)))
    + ∑ j, (d j • y j + e j • x j)

omit [Fintype ι] in
private theorem multi_quad_term_comm_ann (x y : ι → R) (h : MultiCCR x y)
    (a b c : ι → ι → ℂ) (j k m : ι) :
    (a j k • (y j * x k) + b j k • (y j * y k) + c j k • (x j * x k)) * x m
        - x m * (a j k • (y j * x k) + b j k • (y j * y k) + c j k • (x j * x k))
      = (if m = j then a j k • -(x k) else 0)
          + ((if m = k then b j k • -(y j) else 0)
            + (if m = j then b j k • -(y k) else 0)) := by
  calc (a j k • (y j * x k) + b j k • (y j * y k) + c j k • (x j * x k)) * x m
        - x m * (a j k • (y j * x k) + b j k • (y j * y k) + c j k • (x j * x k))
      = a j k • ((y j * x k) * x m - x m * (y j * x k))
          + b j k • ((y j * y k) * x m - x m * (y j * y k))
          + c j k • ((x j * x k) * x m - x m * (x j * x k)) := by
        simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm]
        module
    _ = a j k • (if m = j then -(x k) else 0)
          + b j k • ((if m = k then -(y j) else 0) + (if m = j then -(y k) else 0))
          + c j k • (0 : R) := by
        rw [multi_comm_bilinear_ann x y h j k m,
          multi_comm_creation_pair_ann x y h j k m,
          multi_comm_ann_pair_ann x y h j k m]
    _ = (if m = j then a j k • -(x k) else 0)
          + ((if m = k then b j k • -(y j) else 0)
            + (if m = j then b j k • -(y k) else 0)) := by
        split_ifs <;> simp [smul_add]

omit [Fintype ι] in
private theorem multi_quad_term_comm_creation (x y : ι → R) (h : MultiCCR x y)
    (a b c : ι → ι → ℂ) (j k m : ι) :
    (a j k • (y j * x k) + b j k • (y j * y k) + c j k • (x j * x k)) * y m
        - y m * (a j k • (y j * x k) + b j k • (y j * y k) + c j k • (x j * x k))
      = (if k = m then a j k • y j else 0)
          + ((if j = m then c j k • x k else 0)
            + (if k = m then c j k • x j else 0)) := by
  calc (a j k • (y j * x k) + b j k • (y j * y k) + c j k • (x j * x k)) * y m
        - y m * (a j k • (y j * x k) + b j k • (y j * y k) + c j k • (x j * x k))
      = a j k • ((y j * x k) * y m - y m * (y j * x k))
          + b j k • ((y j * y k) * y m - y m * (y j * y k))
          + c j k • ((x j * x k) * y m - y m * (x j * x k)) := by
        simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm]
        module
    _ = a j k • (if k = m then y j else 0)
          + b j k • (0 : R)
          + c j k • ((if j = m then x k else 0) + (if k = m then x j else 0)) := by
        rw [multi_comm_bilinear_creation x y h j k m,
          multi_comm_creation_pair_creation x y h j k m,
          multi_comm_ann_pair_creation x y h j k m]
    _ = (if k = m then a j k • y j else 0)
          + ((if j = m then c j k • x k else 0)
            + (if k = m then c j k • x j else 0)) := by
        split_ifs <;> simp [smul_add]

omit [Fintype ι] in
private theorem multi_lin_term_comm_ann (x y : ι → R) (h : MultiCCR x y)
    (d e : ι → ℂ) (j m : ι) :
    (d j • y j + e j • x j) * x m - x m * (d j • y j + e j • x j)
      = if m = j then d j • (-1 : R) else 0 := by
  calc (d j • y j + e j • x j) * x m - x m * (d j • y j + e j • x j)
      = d j • (y j * x m - x m * y j) + e j • (x j * x m - x m * x j) := by
        simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm]
        module
    _ = d j • (-(if m = j then (1 : R) else 0)) + e j • (0 : R) := by
        rw [multi_comm_creation_ann x y h j m, multi_comm_ann_ann x y h j m]
    _ = if m = j then d j • (-1 : R) else 0 := by
        split_ifs <;> simp

omit [Fintype ι] in
private theorem multi_lin_term_comm_creation (x y : ι → R) (h : MultiCCR x y)
    (d e : ι → ℂ) (j m : ι) :
    (d j • y j + e j • x j) * y m - y m * (d j • y j + e j • x j)
      = if j = m then e j • (1 : R) else 0 := by
  calc (d j • y j + e j • x j) * y m - y m * (d j • y j + e j • x j)
      = d j • (y j * y m - y m * y j) + e j • (x j * y m - y m * x j) := by
        simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm]
        module
    _ = d j • (0 : R) + e j • (if j = m then (1 : R) else 0) := by
        rw [multi_comm_creation_creation x y h j m, multi_comm_ann_creation x y h j m]
    _ = if j = m then e j • (1 : R) else 0 := by
        split_ifs <;> simp

/-- Manuscript Sec. (subsec:encoding) (Hamiltonians of at most quadratic order close the
first-moment equations exactly), multimode annihilator closure: for the index-quadratic
generator `H = multiQuadGen x y a b c d e`, the commutator `H xₘ - xₘ H` is again affine
in the mode operators,
`H xₘ - xₘ H = ∑ₖ (-aₘₖ) • xₖ + ∑ⱼ (-(bⱼₘ + bₘⱼ)) • yⱼ + (-dₘ) • 1`.
In particular the first-moment Heisenberg hierarchy of the `E_J = 0` sine–Gordon lattice
(manuscript Sec. (sec:sg)) closes exactly. -/
theorem multi_closure_x (x y : ι → R) (h : MultiCCR x y)
    (a b c : ι → ι → ℂ) (d e : ι → ℂ) (m : ι) :
    multiQuadGen x y a b c d e * x m - x m * multiQuadGen x y a b c d e
      = (∑ k, (-(a m k)) • x k) + (∑ j, (-(b j m + b m j)) • y j)
          + algebraMap ℂ R (-(d m)) := by
  have hsplit : ∀ P L z : R, (P + L) * z - z * (P + L)
      = (P * z - z * P) + (L * z - z * L) := fun P L z => by noncomm_ring
  have hQ : (∑ j, ∑ k, (a j k • (y j * x k) + b j k • (y j * y k)
          + c j k • (x j * x k))) * x m
        - x m * ∑ j, ∑ k, (a j k • (y j * x k) + b j k • (y j * y k)
          + c j k • (x j * x k))
      = (∑ k, (-(a m k)) • x k) + ∑ j, (-(b j m + b m j)) • y j := by
    rw [multi_comm_sum]
    calc ∑ j, ((∑ k, (a j k • (y j * x k) + b j k • (y j * y k)
            + c j k • (x j * x k))) * x m
          - x m * ∑ k, (a j k • (y j * x k) + b j k • (y j * y k)
            + c j k • (x j * x k)))
        = ∑ j, ∑ k, ((if m = j then a j k • -(x k) else 0)
            + ((if m = k then b j k • -(y j) else 0)
              + (if m = j then b j k • -(y k) else 0))) := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [multi_comm_sum]
          exact Finset.sum_congr rfl fun k _ =>
            multi_quad_term_comm_ann x y h a b c j k m
      _ = (∑ k, a m k • -(x k))
            + ((∑ j, b j m • -(y j)) + ∑ k, b m k • -(y k)) := by
          simp only [Finset.sum_add_distrib, multi_sum_ite_const, Finset.sum_ite_eq,
            Finset.mem_univ, if_true]
      _ = (∑ k, (-(a m k)) • x k) + ∑ j, (-(b j m + b m j)) • y j := by
          rw [← Finset.sum_add_distrib]
          congr 1
          · exact Finset.sum_congr rfl fun k _ => by module
          · exact Finset.sum_congr rfl fun j _ => by module
  have hL : (∑ j, (d j • y j + e j • x j)) * x m
        - x m * ∑ j, (d j • y j + e j • x j)
      = algebraMap ℂ R (-(d m)) := by
    rw [multi_comm_sum]
    calc ∑ j, ((d j • y j + e j • x j) * x m - x m * (d j • y j + e j • x j))
        = ∑ j, (if m = j then d j • (-1 : R) else 0) :=
          Finset.sum_congr rfl fun j _ => multi_lin_term_comm_ann x y h d e j m
      _ = d m • (-1 : R) := by rw [Finset.sum_ite_eq]; simp
      _ = algebraMap ℂ R (-(d m)) := by
          rw [Algebra.algebraMap_eq_smul_one]
          module
  unfold multiQuadGen
  rw [hsplit, hQ, hL]

/-- Manuscript Sec. (subsec:encoding), multimode creator closure: for the index-quadratic
generator `H = multiQuadGen x y a b c d e`, the commutator `H yₘ - yₘ H` is again affine
in the mode operators,
`H yₘ - yₘ H = ∑ⱼ aⱼₘ • yⱼ + ∑ₖ (cₘₖ + cₖₘ) • xₖ + eₘ • 1`. -/
theorem multi_closure_y (x y : ι → R) (h : MultiCCR x y)
    (a b c : ι → ι → ℂ) (d e : ι → ℂ) (m : ι) :
    multiQuadGen x y a b c d e * y m - y m * multiQuadGen x y a b c d e
      = (∑ j, a j m • y j) + (∑ k, (c m k + c k m) • x k)
          + algebraMap ℂ R (e m) := by
  have hsplit : ∀ P L z : R, (P + L) * z - z * (P + L)
      = (P * z - z * P) + (L * z - z * L) := fun P L z => by noncomm_ring
  have hQ : (∑ j, ∑ k, (a j k • (y j * x k) + b j k • (y j * y k)
          + c j k • (x j * x k))) * y m
        - y m * ∑ j, ∑ k, (a j k • (y j * x k) + b j k • (y j * y k)
          + c j k • (x j * x k))
      = (∑ j, a j m • y j) + ∑ k, (c m k + c k m) • x k := by
    rw [multi_comm_sum]
    calc ∑ j, ((∑ k, (a j k • (y j * x k) + b j k • (y j * y k)
            + c j k • (x j * x k))) * y m
          - y m * ∑ k, (a j k • (y j * x k) + b j k • (y j * y k)
            + c j k • (x j * x k)))
        = ∑ j, ∑ k, ((if k = m then a j k • y j else 0)
            + ((if j = m then c j k • x k else 0)
              + (if k = m then c j k • x j else 0))) := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [multi_comm_sum]
          exact Finset.sum_congr rfl fun k _ =>
            multi_quad_term_comm_creation x y h a b c j k m
      _ = (∑ j, a j m • y j)
            + ((∑ k, c m k • x k) + ∑ j, c j m • x j) := by
          simp only [Finset.sum_add_distrib, multi_sum_ite_const, Finset.sum_ite_eq',
            Finset.mem_univ, if_true]
      _ = (∑ j, a j m • y j) + ∑ k, (c m k + c k m) • x k := by
          rw [← Finset.sum_add_distrib]
          congr 1
          exact Finset.sum_congr rfl fun k _ => by module
  have hL : (∑ j, (d j • y j + e j • x j)) * y m
        - y m * ∑ j, (d j • y j + e j • x j)
      = algebraMap ℂ R (e m) := by
    rw [multi_comm_sum]
    calc ∑ j, ((d j • y j + e j • x j) * y m - y m * (d j • y j + e j • x j))
        = ∑ j, (if j = m then e j • (1 : R) else 0) :=
          Finset.sum_congr rfl fun j _ => multi_lin_term_comm_creation x y h d e j m
      _ = e m • (1 : R) := by rw [Finset.sum_ite_eq']; simp
      _ = algebraMap ℂ R (e m) := (Algebra.algebraMap_eq_smul_one _).symm
  unfold multiQuadGen
  rw [hsplit, hQ, hL]

/-- Image of `multi_closure_x` under a complex-linear functional: the first-moment
equation of mode `m` involves only the first moments `E (x k)`, `E (y j)`, and `E 1`.
The identity term remains `E 1` without normalization. -/
theorem multi_closure_x_expectation (E : R →ₗ[ℂ] ℂ) (x y : ι → R) (h : MultiCCR x y)
    (a b c : ι → ι → ℂ) (d e : ι → ℂ) (m : ι) :
    E (multiQuadGen x y a b c d e * x m - x m * multiQuadGen x y a b c d e)
      = (∑ k, (-(a m k)) • E (x k)) + (∑ j, (-(b j m + b m j)) • E (y j))
          + (-(d m)) • E 1 := by
  rw [multi_closure_x x y h a b c d e m]
  simp only [map_add, map_sum, map_smul, Algebra.algebraMap_eq_smul_one]

/-- Image of `multi_closure_y` under a complex-linear functional: the first-moment
equation of mode `m` involves only the first moments `E (y j)`, `E (x k)`, and `E 1`.
The identity term remains `E 1` without normalization. -/
theorem multi_closure_y_expectation (E : R →ₗ[ℂ] ℂ) (x y : ι → R) (h : MultiCCR x y)
    (a b c : ι → ι → ℂ) (d e : ι → ℂ) (m : ι) :
    E (multiQuadGen x y a b c d e * y m - y m * multiQuadGen x y a b c d e)
      = (∑ j, a j m • E (y j)) + (∑ k, (c m k + c k m) • E (x k))
          + e m • E 1 := by
  rw [multi_closure_y x y h a b c d e m]
  simp only [map_add, map_sum, map_smul, Algebra.algebraMap_eq_smul_one]

end MultimodeClosure

section NonClosure

/-- Manuscript Sec. (subsec:encoding) (nonlinear Hamiltonians generally couple first
moments to higher moments): under the CCR `x y = y x + 1` the commutator of the squared
number element `n² = (y x)²` with the annihilator is `[n², x] = -2 y x² - x`, whose
leading term `y x²` is not affine in the mode operators. -/
theorem nonclosure_comm {R : Type*} [Ring R] (x y : R) (h : x * y = y * x + 1) :
    (y * x) * (y * x) * x - x * ((y * x) * (y * x))
      = -(2 : R) • (y * (x * x)) - x := by
  have hx : x * (y * x) = (y * x) * x + x := by
    calc x * (y * x) = (x * y) * x := by noncomm_ring
      _ = (y * x + 1) * x := by rw [h]
      _ = (y * x) * x + x := by noncomm_ring
  calc (y * x) * (y * x) * x - x * ((y * x) * (y * x))
      = (y * x) * (y * x) * x - (x * (y * x)) * (y * x) := by noncomm_ring
    _ = (y * x) * (y * x) * x - ((y * x) * x + x) * (y * x) := by rw [hx]
    _ = (y * x) * (y * x) * x - (y * x) * (x * (y * x)) - x * (y * x) := by
        noncomm_ring
    _ = (y * x) * (y * x) * x - (y * x) * ((y * x) * x + x)
          - ((y * x) * x + x) := by rw [hx]
    _ = -(2 : R) • (y * (x * x)) - x := by
        rw [smul_eq_mul]
        noncomm_ring

/-- The non-closure commutator with the scalar action expanded into ring operations. -/
private theorem nonclosure_comm_mul {R : Type*} [Ring R] (x y : R)
    (h : x * y = y * x + 1) :
    (y * x) * (y * x) * x - x * ((y * x) * (y * x))
      = -(y * (x * x)) - y * (x * x) - x := by
  rw [nonclosure_comm x y h, smul_eq_mul]
  noncomm_ring

/-- Non-closure witness on the concrete Fock representation: there are no complex
scalars `a`, `b`, `c` with `[n̂², â] = a • â + b • â† + c • 1` as endomorphisms of
`ℕ →₀ ℂ`, where `â = lowering`, `â† = raising`, `n̂ = raising * lowering`.  Evaluating
both sides on the basis vectors `e₁` and `e₂` forces `a = -1` and `a = -3`
simultaneously.  This witnesses the manuscript's statement that nonlinear generators
couple first moments to higher moments (Sec. (subsec:encoding)); conversely, as
`E_J → 0` the lattice Hamiltonian is quadratic and `multi_closure_x`/`multi_closure_y`
apply (Sec. (sec:sg)). -/
theorem nonclosure_witness :
    ¬ ∃ a b c : ℂ,
      (raising * lowering) * (raising * lowering) * lowering
          - lowering * ((raising * lowering) * (raising * lowering))
        = a • lowering + b • raising + c • (1 : Module.End ℂ (ℕ →₀ ℂ)) := by
  rintro ⟨a, b, c, hab⟩
  rw [nonclosure_comm_mul lowering raising lowering_mul_raising] at hab
  have h1 := DFunLike.congr_fun (DFunLike.congr_fun hab (Finsupp.single 1 (1 : ℂ))) 0
  have h2 := DFunLike.congr_fun (DFunLike.congr_fun hab (Finsupp.single 2 (1 : ℂ))) 1
  simp at h1 h2
  rw [← h1] at h2
  have hs : (Real.sqrt 2 : ℝ) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr (by norm_num))
  have hsC : ((Real.sqrt 2 : ℝ) : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hs
  have h3 : ((Real.sqrt 2 : ℝ) : ℂ) = 0 := by linear_combination (-(1 : ℂ) / 2) * h2
  exact hsC h3

end NonClosure

section SGFreeLattice

variable {ι R : Type*} [Fintype ι] [DecidableEq ι] [Ring R] [Algebra ℂ R]

/-!
### The `E_J = 0` sine–Gordon lattice Hamiltonian as a `multiQuadGen` instance

Manuscript Sec. (sec:sg), Eq. (eq:sg-lattice), at `E_J = 0`, together with the
quadratic-limit sentence of Sec. (subsec:sg-classical) ("As `E_J → 0` the only
nonlinear term vanishes and the mean-field closure becomes exact"): with the
onsite cosine absent, the lattice Hamiltonian
`H = ∑_j Π̂_j²/2 + (κ/2) ∑_j (φ̂_{next j} − φ̂_j)²`, written in the site
quadratures of Eq. (eq:sg-quadratures), is an index-quadratic generator.  Below
it is exhibited as `multiQuadGen` at explicit coefficient matrices (plus a
scalar zero-point constant that commutes with everything), so
`multi_closure_x`/`multi_closure_y` apply verbatim.
-/

/-- Site quadrature `φ̂_j = (â_j + â_j†)/√(2 ω_b)` of manuscript
Eq. (eq:sg-quadratures), written for an abstract multimode CCR family with
annihilators `x` and creators `y`. -/
def sgSitePhi (omega : ℝ) (x y : ι → R) (j : ι) : R :=
  ((1 / Real.sqrt (2 * omega) : ℝ) : ℂ) • (x j + y j)

/-- Site momentum `Π̂_j = i √(ω_b/2) (â_j† − â_j)` of manuscript
Eq. (eq:sg-quadratures). -/
def sgSitePi (omega : ℝ) (x y : ι → R) (j : ι) : R :=
  (Complex.I * ((Real.sqrt (omega / 2) : ℝ) : ℂ)) • (y j - x j)

/-- The `E_J = 0` sine–Gordon lattice Hamiltonian of manuscript
Eq. (eq:sg-lattice): `H = ∑_j Π̂_j²/2 + (κ/2) ∑_j (φ̂_{next j} − φ̂_j)²`, the
onsite cosine term being absent in the `E_J → 0` limit.  The bond map `next`
encodes the boundary convention discussed after the display (periodic
boundaries: the cyclic successor). -/
def sgLatticeGenFree (omega kappa : ℝ) (next : ι → ι) (x y : ι → R) : R :=
  (∑ j, ((1 : ℂ) / 2) • (sgSitePi omega x y j * sgSitePi omega x y j))
    + ((kappa : ℂ) / 2) •
        ∑ j, ((sgSitePhi omega x y (next j) - sgSitePhi omega x y j)
          * (sgSitePhi omega x y (next j) - sgSitePhi omega x y j))

/-- Bond incidence scalar `δ_{m, next j} − δ_{m, j}` of the nearest-neighbor
coupling. -/
def sgFreeBondDelta (next : ι → ι) (j m : ι) : ℂ :=
  (if m = next j then 1 else 0) - (if m = j then 1 else 0)

/-- Coefficient matrix of the `yₘxₖ` bilinears of the `E_J = 0` lattice
generator: onsite `ω/2` plus the incidence quadratic form of the bond map. -/
def sgFreeCoeffA (omega kappa : ℝ) (next : ι → ι) (m k : ι) : ℂ :=
  (if m = k then (omega : ℂ) / 2 else 0)
    + (kappa : ℂ) / (2 * omega)
        * ∑ j, sgFreeBondDelta next j m * sgFreeBondDelta next j k

/-- Coefficient matrix of the anomalous `yₘyₖ` and `xₘxₖ` bilinears of the
`E_J = 0` lattice generator (the two matrices coincide): onsite `−ω/4` plus
half the incidence quadratic form. -/
def sgFreeCoeffB (omega kappa : ℝ) (next : ι → ι) (m k : ι) : ℂ :=
  (if m = k then -((omega : ℂ) / 4) else 0)
    + (kappa : ℂ) / (4 * omega)
        * ∑ j, sgFreeBondDelta next j m * sgFreeBondDelta next j k

/-- Scalar (zero-point) part of the `E_J = 0` lattice generator: `ω/4` per site
plus `κ/(2ω)` per bond with distinct endpoints. -/
def sgFreeConst (omega kappa : ℝ) (next : ι → ι) : ℂ :=
  (Fintype.card ι : ℂ) * ((omega : ℂ) / 4)
    + (kappa : ℂ) / (2 * omega) * ∑ j, (1 - if next j = j then (1 : ℂ) else 0)

private theorem sgFree_delta_smul_sum (next : ι → ι) (j : ι) (v : ι → R) :
    (∑ m, sgFreeBondDelta next j m • v m) = v (next j) - v j := by
  simp only [sgFreeBondDelta, sub_smul, ite_smul, one_smul, zero_smul,
    Finset.sum_sub_distrib, Finset.sum_ite_eq', Finset.mem_univ, if_true]

private theorem sgFree_delta_pair_collapse (next : ι → ι) (j : ι) (v w : ι → R) :
    (∑ m, ∑ k, (sgFreeBondDelta next j m * sgFreeBondDelta next j k) • (v m * w k))
      = (v (next j) - v j) * (w (next j) - w j) := by
  rw [← sgFree_delta_smul_sum next j v, ← sgFree_delta_smul_sum next j w,
    Finset.sum_mul_sum]
  exact Finset.sum_congr rfl fun m _ => Finset.sum_congr rfl fun k _ =>
    (smul_mul_smul_comm _ _ _ _).symm

private theorem sgFree_coupling_collapse (c : ℂ) (next : ι → ι) (v w : ι → R) :
    (∑ m, ∑ k,
        (c * ∑ j, sgFreeBondDelta next j m * sgFreeBondDelta next j k) • (v m * w k))
      = ∑ j, c • ((v (next j) - v j) * (w (next j) - w j)) := by
  have h1 : ∀ m k : ι,
      (c * ∑ j, sgFreeBondDelta next j m * sgFreeBondDelta next j k) • (v m * w k)
        = ∑ j, c • ((sgFreeBondDelta next j m * sgFreeBondDelta next j k)
            • (v m * w k)) := by
    intro m k
    rw [Finset.mul_sum, Finset.sum_smul]
    exact Finset.sum_congr rfl fun j _ => by rw [mul_smul]
  calc (∑ m, ∑ k,
        (c * ∑ j, sgFreeBondDelta next j m * sgFreeBondDelta next j k) • (v m * w k))
      = ∑ m, ∑ k, ∑ j, c • ((sgFreeBondDelta next j m * sgFreeBondDelta next j k)
          • (v m * w k)) :=
        Finset.sum_congr rfl fun m _ => Finset.sum_congr rfl fun k _ => h1 m k
    _ = ∑ m, ∑ j, ∑ k, c • ((sgFreeBondDelta next j m * sgFreeBondDelta next j k)
          • (v m * w k)) :=
        Finset.sum_congr rfl fun m _ => Finset.sum_comm
    _ = ∑ j, ∑ m, ∑ k, c • ((sgFreeBondDelta next j m * sgFreeBondDelta next j k)
          • (v m * w k)) := Finset.sum_comm
    _ = ∑ j, c • ((v (next j) - v j) * (w (next j) - w j)) := by
        refine Finset.sum_congr rfl fun j _ => ?_
        simp only [← Finset.smul_sum]
        rw [sgFree_delta_pair_collapse]

private theorem sgFree_diag_collapse (c : ℂ) (M : ι → ι → R) :
    (∑ m, ∑ k, (if m = k then c else 0) • M m k) = ∑ m, c • M m m := by
  refine Finset.sum_congr rfl fun m _ => ?_
  simp [ite_smul, zero_smul, Finset.sum_ite_eq]

private theorem sgFree_multiQuadGen_expand (omega kappa : ℝ)
    (next : ι → ι) (x y : ι → R) :
    multiQuadGen x y (sgFreeCoeffA omega kappa next) (sgFreeCoeffB omega kappa next)
        (sgFreeCoeffB omega kappa next) 0 0
      = (∑ j, (((omega : ℂ) / 2) • (y j * x j) + (-((omega : ℂ) / 4)) • (y j * y j)
            + (-((omega : ℂ) / 4)) • (x j * x j)))
        + ∑ j, (((kappa : ℂ) / (2 * omega))
              • ((y (next j) - y j) * (x (next j) - x j))
            + ((kappa : ℂ) / (4 * omega))
              • ((y (next j) - y j) * (y (next j) - y j))
            + ((kappa : ℂ) / (4 * omega))
              • ((x (next j) - x j) * (x (next j) - x j))) := by
  simp only [multiQuadGen, sgFreeCoeffA, sgFreeCoeffB, Pi.zero_apply, zero_smul,
    add_zero, Finset.sum_const_zero, add_smul, Finset.sum_add_distrib]
  rw [sgFree_diag_collapse ((omega : ℂ) / 2) (fun m k => y m * x k),
    sgFree_diag_collapse (-((omega : ℂ) / 4)) (fun m k => y m * y k),
    sgFree_diag_collapse (-((omega : ℂ) / 4)) (fun m k => x m * x k),
    sgFree_coupling_collapse ((kappa : ℂ) / (2 * omega)) next y x,
    sgFree_coupling_collapse ((kappa : ℂ) / (4 * omega)) next y y,
    sgFree_coupling_collapse ((kappa : ℂ) / (4 * omega)) next x x]
  abel

omit [Fintype ι] in
private theorem sgFree_kinetic_site (omega : ℝ) (homega : 0 < omega)
    (x y : ι → R) (h : MultiCCR x y) (j : ι) :
    ((1 : ℂ) / 2) • (sgSitePi omega x y j * sgSitePi omega x y j)
      = (((omega : ℂ) / 2) • (y j * x j) + (-((omega : ℂ) / 4)) • (y j * y j)
          + (-((omega : ℂ) / 4)) • (x j * x j))
        + algebraMap ℂ R ((omega : ℂ) / 4) := by
  have hxy : x j * y j = y j * x j + 1 := by simpa using h.ann_creation j j
  have hring : (y j - x j) * (y j - x j)
      = y j * y j + x j * x j - (y j * x j + y j * x j) - 1 := by
    calc (y j - x j) * (y j - x j)
        = y j * y j + x j * x j - y j * x j - x j * y j := by noncomm_ring
      _ = y j * y j + x j * x j - y j * x j - (y j * x j + 1) := by rw [hxy]
      _ = y j * y j + x j * x j - (y j * x j + y j * x j) - 1 := by noncomm_ring
  have hs : ((Real.sqrt (omega / 2) : ℝ) : ℂ) * ((Real.sqrt (omega / 2) : ℝ) : ℂ)
      = ((omega : ℂ) / 2) := by
    rw [← Complex.ofReal_mul,
      Real.mul_self_sqrt (div_nonneg homega.le (by norm_num))]
    push_cast
    ring
  have hc2 : (Complex.I * ((Real.sqrt (omega / 2) : ℝ) : ℂ))
        * (Complex.I * ((Real.sqrt (omega / 2) : ℝ) : ℂ)) = -((omega : ℂ) / 2) := by
    calc (Complex.I * ((Real.sqrt (omega / 2) : ℝ) : ℂ))
          * (Complex.I * ((Real.sqrt (omega / 2) : ℝ) : ℂ))
        = Complex.I * Complex.I * (((Real.sqrt (omega / 2) : ℝ) : ℂ)
            * ((Real.sqrt (omega / 2) : ℝ) : ℂ)) := by ring
      _ = -((omega : ℂ) / 2) := by rw [Complex.I_mul_I, hs]; ring
  have hmul : sgSitePi omega x y j * sgSitePi omega x y j
      = (-((omega : ℂ) / 2)) • ((y j - x j) * (y j - x j)) := by
    simp only [sgSitePi]
    rw [smul_mul_smul_comm, hc2]
  rw [hmul, hring, smul_smul, Algebra.algebraMap_eq_smul_one]
  have hcoef : (1 : ℂ) / 2 * -((omega : ℂ) / 2) = -((omega : ℂ) / 4) := by ring
  rw [hcoef]
  module

omit [Fintype ι] in
private theorem sgFree_bond_site (omega kappa : ℝ) (homega : 0 < omega)
    (next : ι → ι) (x y : ι → R) (h : MultiCCR x y) (j : ι) :
    ((kappa : ℂ) / 2) • ((sgSitePhi omega x y (next j) - sgSitePhi omega x y j)
        * (sgSitePhi omega x y (next j) - sgSitePhi omega x y j))
      = (((kappa : ℂ) / (2 * omega)) • ((y (next j) - y j) * (x (next j) - x j))
          + ((kappa : ℂ) / (4 * omega)) • ((y (next j) - y j) * (y (next j) - y j))
          + ((kappa : ℂ) / (4 * omega)) • ((x (next j) - x j) * (x (next j) - x j)))
        + algebraMap ℂ R ((kappa : ℂ) / (2 * omega)
            * (1 - if next j = j then 1 else 0)) := by
  have homegaC : (omega : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr homega.ne'
  have hphi : sgSitePhi omega x y (next j) - sgSitePhi omega x y j
      = ((1 / Real.sqrt (2 * omega) : ℝ) : ℂ)
          • ((x (next j) - x j) + (y (next j) - y j)) := by
    simp only [sgSitePhi, ← smul_sub]
    congr 1
    abel
  have hs2 : ((1 / Real.sqrt (2 * omega) : ℝ) : ℂ)
        * ((1 / Real.sqrt (2 * omega) : ℝ) : ℂ) = 1 / (2 * (omega : ℂ)) := by
    have hR : (1 / Real.sqrt (2 * omega)) * (1 / Real.sqrt (2 * omega))
        = 1 / (2 * omega) := by
      rw [div_mul_div_comm, one_mul,
        Real.mul_self_sqrt (mul_nonneg (by norm_num) homega.le)]
    rw [← Complex.ofReal_mul, hR]
    push_cast
    ring
  have c1 : x (next j) * y (next j) = y (next j) * x (next j) + 1 := by
    simpa using h.ann_creation (next j) (next j)
  have c4 : x j * y j = y j * x j + 1 := by simpa using h.ann_creation j j
  have c2 : x (next j) * y j
      = y j * x (next j) + (if next j = j then 1 else 0) := h.ann_creation (next j) j
  have c3 : x j * y (next j)
      = y (next j) * x j + (if next j = j then 1 else 0) := by
    have h' := h.ann_creation j (next j)
    have hswap : (if j = next j then (1 : R) else 0)
        = (if next j = j then (1 : R) else 0) := by simp [eq_comm]
    rwa [hswap] at h'
  have hXY : (x (next j) - x j) * (y (next j) - y j)
      = (y (next j) - y j) * (x (next j) - x j)
          + (((1 : R) + 1) - ((if next j = j then (1 : R) else 0)
              + (if next j = j then (1 : R) else 0))) := by
    calc (x (next j) - x j) * (y (next j) - y j)
        = x (next j) * y (next j) - x (next j) * y j - x j * y (next j)
            + x j * y j := by noncomm_ring
      _ = (y (next j) * x (next j) + 1)
            - (y j * x (next j) + (if next j = j then 1 else 0))
            - (y (next j) * x j + (if next j = j then 1 else 0))
            + (y j * x j + 1) := by rw [c1, c2, c3, c4]
      _ = (y (next j) - y j) * (x (next j) - x j)
            + (((1 : R) + 1) - ((if next j = j then (1 : R) else 0)
                + (if next j = j then (1 : R) else 0))) := by noncomm_ring
  have hsq : ((x (next j) - x j) + (y (next j) - y j))
        * ((x (next j) - x j) + (y (next j) - y j))
      = (y (next j) - y j) * (x (next j) - x j)
          + (y (next j) - y j) * (x (next j) - x j)
          + (y (next j) - y j) * (y (next j) - y j)
          + (x (next j) - x j) * (x (next j) - x j)
          + (((1 : R) + 1) - ((if next j = j then (1 : R) else 0)
              + (if next j = j then (1 : R) else 0))) := by
    calc ((x (next j) - x j) + (y (next j) - y j))
          * ((x (next j) - x j) + (y (next j) - y j))
        = (x (next j) - x j) * (x (next j) - x j)
            + (x (next j) - x j) * (y (next j) - y j)
            + (y (next j) - y j) * (x (next j) - x j)
            + (y (next j) - y j) * (y (next j) - y j) := by noncomm_ring
      _ = _ := by rw [hXY]; noncomm_ring
  have hconst : (((1 : R) + 1) - ((if next j = j then (1 : R) else 0)
        + (if next j = j then (1 : R) else 0)))
      = algebraMap ℂ R (2 - ((if next j = j then (1 : ℂ) else 0)
          + (if next j = j then (1 : ℂ) else 0))) := by
    split_ifs <;> simp [map_ofNat] <;> norm_num
  have hcoef : (kappa : ℂ) / 2 * (1 / (2 * (omega : ℂ))) = (kappa : ℂ) / (4 * omega) := by
    field_simp
    ring
  have hone : (kappa : ℂ) / (2 * omega) = (kappa : ℂ) / (4 * omega) * 2 := by
    field_simp
    ring
  rw [hphi, smul_mul_smul_comm, hs2, smul_smul, hcoef, hsq, hconst, hone]
  simp only [Algebra.algebraMap_eq_smul_one]
  module

/-- Manuscript Sec. (sec:sg) at `E_J = 0` (the quadratic-limit sentence of
Sec. (subsec:sg-classical): "As `E_J → 0` the only nonlinear term vanishes and the
mean-field closure becomes exact"): the `E_J = 0` sine–Gordon lattice Hamiltonian
of Eq. (eq:sg-lattice), written in the site quadratures of Eq. (eq:sg-quadratures)
over any multimode CCR family, equals the index-quadratic generator `multiQuadGen`
at the explicit coefficient matrices `sgFreeCoeffA`/`sgFreeCoeffB` (no linear
terms), plus the scalar zero-point constant `sgFreeConst`.  The closure theorems
`multi_closure_x`/`multi_closure_y` therefore apply to it verbatim. -/
theorem sgLatticeGenFree_eq_multiQuadGen (omega kappa : ℝ) (homega : 0 < omega)
    (next : ι → ι) (x y : ι → R) (h : MultiCCR x y) :
    sgLatticeGenFree omega kappa next x y
      = multiQuadGen x y (sgFreeCoeffA omega kappa next) (sgFreeCoeffB omega kappa next)
          (sgFreeCoeffB omega kappa next) 0 0
        + algebraMap ℂ R (sgFreeConst omega kappa next) := by
  rw [sgFree_multiQuadGen_expand]
  simp only [sgLatticeGenFree]
  rw [Finset.sum_congr rfl fun j _ => sgFree_kinetic_site omega homega x y h j,
    Finset.smul_sum,
    Finset.sum_congr rfl fun j _ => sgFree_bond_site omega kappa homega next x y h j]
  simp only [Finset.sum_add_distrib, ← map_sum, sgFreeConst, Finset.sum_const,
    Finset.card_univ, nsmul_eq_mul, Finset.mul_sum, map_add]
  abel

/-- First-moment closure of the `E_J = 0` sine–Gordon lattice, annihilator side:
`multi_closure_x` applied through `sgLatticeGenFree_eq_multiQuadGen` (the
zero-point constant commutes away), so the commutator of the `E_J = 0` lattice
Hamiltonian with each `xₘ` is affine in the mode operators — the exact closure
asserted by manuscript Sec. (subsec:encoding) for the `E_J → 0` limit of
Sec. (sec:sg). -/
theorem sgLatticeGenFree_closure_x (omega kappa : ℝ) (homega : 0 < omega)
    (next : ι → ι) (x y : ι → R) (h : MultiCCR x y) (m : ι) :
    sgLatticeGenFree omega kappa next x y * x m
        - x m * sgLatticeGenFree omega kappa next x y
      = (∑ k, (-(sgFreeCoeffA omega kappa next m k)) • x k)
          + ∑ j, (-(sgFreeCoeffB omega kappa next j m
              + sgFreeCoeffB omega kappa next m j)) • y j := by
  rw [sgLatticeGenFree_eq_multiQuadGen omega kappa homega next x y h]
  have hz := Algebra.commutes (A := R) (sgFreeConst omega kappa next) (x m)
  have hcomm : (multiQuadGen x y (sgFreeCoeffA omega kappa next)
          (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0
        + algebraMap ℂ R (sgFreeConst omega kappa next)) * x m
        - x m * (multiQuadGen x y (sgFreeCoeffA omega kappa next)
          (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0
        + algebraMap ℂ R (sgFreeConst omega kappa next))
      = multiQuadGen x y (sgFreeCoeffA omega kappa next)
          (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0 * x m
        - x m * multiQuadGen x y (sgFreeCoeffA omega kappa next)
          (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0 := by
    calc _ = (multiQuadGen x y (sgFreeCoeffA omega kappa next)
            (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0 * x m
          - x m * multiQuadGen x y (sgFreeCoeffA omega kappa next)
            (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0)
          + (algebraMap ℂ R (sgFreeConst omega kappa next) * x m
            - x m * algebraMap ℂ R (sgFreeConst omega kappa next)) := by noncomm_ring
      _ = _ := by rw [hz, sub_self, add_zero]
  rw [hcomm, multi_closure_x x y h (sgFreeCoeffA omega kappa next)
    (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0 m]
  simp

/-- First-moment closure of the `E_J = 0` sine–Gordon lattice, creator side:
`multi_closure_y` applied through `sgLatticeGenFree_eq_multiQuadGen`. -/
theorem sgLatticeGenFree_closure_y (omega kappa : ℝ) (homega : 0 < omega)
    (next : ι → ι) (x y : ι → R) (h : MultiCCR x y) (m : ι) :
    sgLatticeGenFree omega kappa next x y * y m
        - y m * sgLatticeGenFree omega kappa next x y
      = (∑ j, sgFreeCoeffA omega kappa next j m • y j)
          + ∑ k, (sgFreeCoeffB omega kappa next m k
              + sgFreeCoeffB omega kappa next k m) • x k := by
  rw [sgLatticeGenFree_eq_multiQuadGen omega kappa homega next x y h]
  have hz := Algebra.commutes (A := R) (sgFreeConst omega kappa next) (y m)
  have hcomm : (multiQuadGen x y (sgFreeCoeffA omega kappa next)
          (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0
        + algebraMap ℂ R (sgFreeConst omega kappa next)) * y m
        - y m * (multiQuadGen x y (sgFreeCoeffA omega kappa next)
          (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0
        + algebraMap ℂ R (sgFreeConst omega kappa next))
      = multiQuadGen x y (sgFreeCoeffA omega kappa next)
          (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0 * y m
        - y m * multiQuadGen x y (sgFreeCoeffA omega kappa next)
          (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0 := by
    calc _ = (multiQuadGen x y (sgFreeCoeffA omega kappa next)
            (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0 * y m
          - y m * multiQuadGen x y (sgFreeCoeffA omega kappa next)
            (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0)
          + (algebraMap ℂ R (sgFreeConst omega kappa next) * y m
            - y m * algebraMap ℂ R (sgFreeConst omega kappa next)) := by noncomm_ring
      _ = _ := by rw [hz, sub_self, add_zero]
  rw [hcomm, multi_closure_y x y h (sgFreeCoeffA omega kappa next)
    (sgFreeCoeffB omega kappa next) (sgFreeCoeffB omega kappa next) 0 0 m]
  simp

end SGFreeLattice

end

end Cvdv
