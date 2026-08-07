import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Module
import Mathlib.Data.Complex.Basic
import Mathlib.Algebra.Algebra.Defs
import Mathlib.Algebra.Module.LinearMap.End
import Mathlib.Analysis.Real.Sqrt
import Mathlib.LinearAlgebra.Finsupp.LSum

/-!
# Normal-ordering split of the quartic interaction under the CCR

Formalization of manuscript Eqs. (df-vcr) and (df-split)/(df-fock-kerr).  In an abstract
unital ring, the canonical commutation relation is encoded by two elements `x` (the
annihilation operator `a`) and `y` (the creation operator `a†`) subject to
`x * y = y * x + 1`; the number element is `y * x`.

`add_pow_four_split` is the central algebraic identity: under the CCR,
`(x + y) ^ 4 = 6 n² + 6 n + 3 + V_cr` with `n = y * x` and
`V_cr = y⁴ + 4 y³ x + x⁴ + 4 y x³ + 6 x² + 6 y²`, Eq. (df-vcr).  The proof squares the
elementary identity `(x + y) ² = x² + y² + 2 y x + 1` and normal-orders every cross term
through a short staircase of commutation lemmas, each obtained from the defining relation
by a `calc` chain.

`duffing_normal_ordered` is the scalar bookkeeping of Eq. (df-fock-kerr): for an algebra
over `ℂ` and `λ = β / (16 ω₀²)`, the Duffing generator `ω₀ n + λ (x + y)⁴` equals
`ω_eff n + K_eff n² + 3 λ + λ V_cr` with `ω_eff = ω₀ + 3 β / (8 ω₀²)` and
`K_eff = 3 β / (8 ω₀²)`; both correction coefficients equal `6 λ`.

The final section realizes the CCR concretely on finitely supported sequences `ℕ →₀ ℂ`:
`lowering` sends `e_(k+1)` to `√(k+1) e_k` and `e_0` to `0`, `raising` sends `e_n` to
`√(n+1) e_(n+1)`, and `lowering_mul_raising` proves `a a† = a† a + 1` in the endomorphism
ring, so `fock_add_pow_four_split` transfers the quartic split to the Fock-space operators.
-/

namespace Cvdv

section CCR

variable {R : Type*} [Ring R]

/-- Commuting the annihilator past a squared creator: `x y² = y² x + 2 y`. -/
theorem mul_creation_sq (x y : R) (h : x * y = y * x + 1) :
    x * y ^ 2 = y ^ 2 * x + 2 * y := by
  calc x * y ^ 2 = (x * y) * y := by noncomm_ring
    _ = (y * x + 1) * y := by rw [h]
    _ = y * (x * y) + y := by noncomm_ring
    _ = y * (y * x + 1) + y := by rw [h]
    _ = y ^ 2 * x + 2 * y := by noncomm_ring

/-- Commuting a squared annihilator past the creator: `x² y = y x² + 2 x`. -/
theorem sq_mul_creation (x y : R) (h : x * y = y * x + 1) :
    x ^ 2 * y = y * x ^ 2 + 2 * x := by
  calc x ^ 2 * y = x * (x * y) := by noncomm_ring
    _ = x * (y * x + 1) := by rw [h]
    _ = (x * y) * x + x := by noncomm_ring
    _ = (y * x + 1) * x + x := by rw [h]
    _ = y * x ^ 2 + 2 * x := by noncomm_ring

/-- Normal ordering of `x² y²`. -/
theorem sq_mul_creation_sq (x y : R) (h : x * y = y * x + 1) :
    x ^ 2 * y ^ 2 = y ^ 2 * x ^ 2 + 4 * (y * x) + 2 := by
  calc x ^ 2 * y ^ 2 = x * (x * y ^ 2) := by noncomm_ring
    _ = x * (y ^ 2 * x + 2 * y) := by rw [mul_creation_sq x y h]
    _ = (x * y ^ 2) * x + 2 * (x * y) := by noncomm_ring
    _ = (y ^ 2 * x + 2 * y) * x + 2 * (y * x + 1) := by rw [mul_creation_sq x y h, h]
    _ = (y ^ 2 * x + 2 * y) * x + (2 * (y * x) + 2) := by rw [mul_add, mul_one]
    _ = y ^ 2 * x ^ 2 + 4 * (y * x) + 2 := by noncomm_ring

/-- The squared number element in normal order: `n² = y² x² + n`. -/
theorem number_sq (x y : R) (h : x * y = y * x + 1) :
    (y * x) * (y * x) = y ^ 2 * x ^ 2 + y * x := by
  calc (y * x) * (y * x) = y * (x * y) * x := by noncomm_ring
    _ = y * (y * x + 1) * x := by rw [h]
    _ = y ^ 2 * x ^ 2 + y * x := by noncomm_ring

/-- Normal ordering of `x² n`. -/
theorem sq_mul_number (x y : R) (h : x * y = y * x + 1) :
    x ^ 2 * (y * x) = y * x ^ 3 + 2 * x ^ 2 := by
  calc x ^ 2 * (y * x) = (x ^ 2 * y) * x := by noncomm_ring
    _ = (y * x ^ 2 + 2 * x) * x := by rw [sq_mul_creation x y h]
    _ = y * x ^ 3 + 2 * x ^ 2 := by noncomm_ring

/-- Normal ordering of `n y²`. -/
theorem number_mul_creation_sq (x y : R) (h : x * y = y * x + 1) :
    (y * x) * y ^ 2 = y ^ 3 * x + 2 * y ^ 2 := by
  calc (y * x) * y ^ 2 = y * (x * y ^ 2) := by noncomm_ring
    _ = y * (y ^ 2 * x + 2 * y) := by rw [mul_creation_sq x y h]
    _ = y ^ 3 * x + 2 * y ^ 2 := by noncomm_ring

/-- The elementary square: `(x + y)² = x² + y² + 2 n + 1`. -/
theorem add_sq_split (x y : R) (h : x * y = y * x + 1) :
    (x + y) ^ 2 = x ^ 2 + y ^ 2 + 2 * (y * x) + 1 := by
  calc (x + y) ^ 2 = x ^ 2 + x * y + y * x + y ^ 2 := by noncomm_ring
    _ = x ^ 2 + (y * x + 1) + y * x + y ^ 2 := by rw [h]
    _ = x ^ 2 + y ^ 2 + 2 * (y * x) + 1 := by noncomm_ring

/-- The fully normal-ordered expansion of the quartic power. -/
theorem add_pow_four_normal (x y : R) (h : x * y = y * x + 1) :
    (x + y) ^ 4 =
      x ^ 4 + 4 * (y * x ^ 3) + 6 * (y ^ 2 * x ^ 2) + 4 * (y ^ 3 * x) + y ^ 4
        + 6 * x ^ 2 + 12 * (y * x) + 6 * y ^ 2 + 3 := by
  have expand : (x + y) ^ 4 =
      x ^ 2 * x ^ 2 + x ^ 2 * y ^ 2 + 2 * (x ^ 2 * (y * x)) + x ^ 2
        + (y ^ 2 * x ^ 2 + y ^ 2 * y ^ 2 + 2 * (y ^ 2 * (y * x)) + y ^ 2)
        + (2 * ((y * x) * x ^ 2) + 2 * ((y * x) * y ^ 2) + 4 * ((y * x) * (y * x))
            + 2 * (y * x))
        + (x ^ 2 + y ^ 2 + 2 * (y * x) + 1) := by
    calc (x + y) ^ 4 = ((x + y) ^ 2) * ((x + y) ^ 2) := by noncomm_ring
      _ = (x ^ 2 + y ^ 2 + 2 * (y * x) + 1) * (x ^ 2 + y ^ 2 + 2 * (y * x) + 1) := by
          rw [add_sq_split x y h]
      _ = _ := by noncomm_ring
  calc (x + y) ^ 4
      = x ^ 4 + 4 * (y * x ^ 3) + 6 * (y ^ 2 * x ^ 2) + 4 * (y ^ 3 * x) + y ^ 4
          + 6 * x ^ 2 + 12 * (y * x) + 6 * y ^ 2 + 2 + 1 := by
        rw [expand, sq_mul_creation_sq x y h, sq_mul_number x y h,
          number_mul_creation_sq x y h, number_sq x y h]
        noncomm_ring
    _ = x ^ 4 + 4 * (y * x ^ 3) + 6 * (y ^ 2 * x ^ 2) + 4 * (y ^ 3 * x) + y ^ 4
          + 6 * x ^ 2 + 12 * (y * x) + 6 * y ^ 2 + 3 := by
        rw [add_assoc]; norm_num

/-- Manuscript Eqs. (df-vcr)/(df-split): with the CCR `x y = y x + 1`,
`(x + y)⁴ = 6 n² + 6 n + 3 + V_cr`, where `n = y x` and
`V_cr = y⁴ + 4 y³ x + x⁴ + 4 y x³ + 6 x² + 6 y²`. -/
theorem add_pow_four_split (x y : R) (h : x * y = y * x + 1) :
    (x + y) ^ 4 =
      6 * ((y * x) * (y * x)) + 6 * (y * x) + 3 +
        (y ^ 4 + 4 * (y ^ 3 * x) + x ^ 4 + 4 * (y * x ^ 3) + 6 * x ^ 2 + 6 * y ^ 2) := by
  rw [add_pow_four_normal x y h, number_sq x y h]
  noncomm_ring

end CCR

section DuffingCoefficients

variable {R : Type*} [Ring R] [Algebra ℂ R]

/-- The quartic split of `add_pow_four_split` with the outer coefficients expressed as
`ℂ`-scalar multiplications, ready for coefficient bookkeeping. -/
theorem add_pow_four_split_smul (x y : R) (h : x * y = y * x + 1) :
    (x + y) ^ 4 =
      (6 : ℂ) • ((y * x) * (y * x)) + (6 : ℂ) • (y * x) + (3 : ℂ) • (1 : R) +
        (y ^ 4 + 4 * (y ^ 3 * x) + x ^ 4 + 4 * (y * x ^ 3) + 6 * x ^ 2 + 6 * y ^ 2) := by
  rw [add_pow_four_split x y h]
  simp only [Algebra.smul_def, map_ofNat, mul_one]

/-- Manuscript Eq. (df-fock-kerr): for `λ = β / (16 ω₀²)`, the Duffing generator
`ω₀ n + λ (x + y)⁴` splits into an effective harmonic part, a Kerr part, a central
constant, and the counter-rotating polynomial, with
`ω_eff = ω₀ + 3 β / (8 ω₀²)` and `K_eff = 3 β / (8 ω₀²)`; both corrections equal `6 λ`. -/
theorem duffing_normal_ordered (x y : R) (h : x * y = y * x + 1)
    (omega0 beta lam : ℂ) (homega : omega0 ≠ 0)
    (hlam : lam = beta / (16 * omega0 ^ 2)) :
    omega0 • (y * x) + lam • (x + y) ^ 4 =
      (omega0 + 3 * beta / (8 * omega0 ^ 2)) • (y * x)
        + (3 * beta / (8 * omega0 ^ 2)) • ((y * x) * (y * x))
        + (3 * lam) • (1 : R)
        + lam • (y ^ 4 + 4 * (y ^ 3 * x) + x ^ 4 + 4 * (y * x ^ 3)
            + 6 * x ^ 2 + 6 * y ^ 2) := by
  have hcoef : 3 * beta / (8 * omega0 ^ 2) = 6 * lam := by
    rw [hlam]
    field_simp
    ring
  rw [hcoef, add_pow_four_split_smul x y h]
  module

/-- The effective frequency and Kerr corrections coincide: both equal `6 λ`. -/
theorem omega_eff_eq_kerr_eff (omega0 beta lam : ℂ) (homega : omega0 ≠ 0)
    (hlam : lam = beta / (16 * omega0 ^ 2)) :
    3 * beta / (8 * omega0 ^ 2) = 6 * lam := by
  rw [hlam]
  field_simp
  ring

end DuffingCoefficients

section FockRealization

/-- The Fock lowering (annihilation) operator on finitely supported sequences:
`e_(k+1) ↦ √(k+1) e_k` and `e_0 ↦ 0`.  Uniformly, `e_n ↦ √n e_(n-1)` with truncated
subtraction, since `√0 = 0`. -/
noncomputable def lowering : Module.End ℂ (ℕ →₀ ℂ) :=
  Finsupp.lift (ℕ →₀ ℂ) ℂ ℕ fun n => (Real.sqrt n : ℂ) • Finsupp.single (n - 1) 1

/-- The Fock raising (creation) operator: `e_n ↦ √(n+1) e_(n+1)`. -/
noncomputable def raising : Module.End ℂ (ℕ →₀ ℂ) :=
  Finsupp.lift (ℕ →₀ ℂ) ℂ ℕ fun n => (Real.sqrt (n + 1) : ℂ) • Finsupp.single (n + 1) 1

@[simp] theorem lowering_single (n : ℕ) (c : ℂ) :
    lowering (Finsupp.single n c) = c • (Real.sqrt n : ℂ) • Finsupp.single (n - 1) 1 := by
  simp [lowering]

@[simp] theorem raising_single (n : ℕ) (c : ℂ) :
    raising (Finsupp.single n c) = c • (Real.sqrt (n + 1) : ℂ) • Finsupp.single (n + 1) 1 := by
  simp [raising]

/-- The canonical commutation relation for the concrete Fock operators: on finitely
supported sequences, `a a† = a† a + 1`.  This instantiates the hypothesis of
`add_pow_four_split` for the actual Fock-space annihilation and creation operators. -/
theorem lowering_mul_raising :
    lowering * raising = raising * lowering + 1 := by
  apply Finsupp.lhom_ext
  intro n c
  cases n with
  | zero =>
      simp [Module.End.mul_apply, LinearMap.add_apply, Module.End.one_apply,
        Real.sqrt_zero, Real.sqrt_one]
  | succ k =>
      have h1 : ((Real.sqrt ((k : ℝ) + 1 + 1) : ℝ) : ℂ) * ((Real.sqrt ((k : ℝ) + 1 + 1) : ℝ) : ℂ)
          = (k : ℂ) + 1 + 1 := by
        rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity)]
        push_cast
        ring
      have h2 : ((Real.sqrt ((k : ℝ) + 1) : ℝ) : ℂ) * ((Real.sqrt ((k : ℝ) + 1) : ℝ) : ℂ)
          = (k : ℂ) + 1 := by
        rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity)]
        push_cast
        ring
      simp only [Module.End.mul_apply, LinearMap.add_apply, Module.End.one_apply,
        raising_single, lowering_single, Nat.add_sub_cancel, Finsupp.smul_single,
        smul_eq_mul, mul_one]
      push_cast
      rw [← Finsupp.single_add]
      congr 1
      linear_combination c * h1 - c * h2

/-- The quartic normal-ordering split for the concrete Fock-space operators, obtained by
instantiating `add_pow_four_split` at the endomorphism ring of `ℕ →₀ ℂ`. -/
theorem fock_add_pow_four_split :
    (lowering + raising) ^ 4 =
      6 * ((raising * lowering) * (raising * lowering)) + 6 * (raising * lowering) + 3 +
        (raising ^ 4 + 4 * (raising ^ 3 * lowering) + lowering ^ 4
          + 4 * (raising * lowering ^ 3) + 6 * lowering ^ 2 + 6 * raising ^ 2) :=
  add_pow_four_split lowering raising lowering_mul_raising

end FockRealization

end Cvdv
