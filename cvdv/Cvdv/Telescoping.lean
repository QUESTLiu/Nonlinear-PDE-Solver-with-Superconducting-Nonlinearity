import Mathlib.Analysis.Normed.Ring.Basic
import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.Complex.Trigonometric

/-!
# Telescoping composition bound

Formalization of manuscript Eq. (sg-spectral-composition): for elements `a`, `b` of a
normed ring with `‖a‖ ≤ 1` and `‖b‖ ≤ 1`, the composed error obeys
`‖a ^ r - b ^ r‖ ≤ r * ‖a - b‖`, so the error of `r` composed approximate steps grows
at most linearly in `r`.  Mathlib provides no norm bound of this shape for a
noncommutative normed ring (`sub_dvd_pow_sub_pow` is the commutative divisibility
statement), so `norm_pow_sub_pow_le` is proved by induction from the factorization
`a ^ (n + 1) - b ^ (n + 1) = a ^ n * (a - b) + (a ^ n - b ^ n) * b` together with
`norm_mul_le`, `norm_pow_le'`, and the triangle inequality.

`norm_pow_sub_exp_smul_pow_le` is the phase-factored corollary used in the manuscript:
if a single step of the compiled unitary matches the target up to a global phase
`exp (i θ)`, then after `r` steps the phase accumulates to `exp (i r θ)` and the same
linear bound holds.  It follows from `smul_pow`, `Complex.exp_nat_mul`, and
`Complex.norm_exp_ofReal_mul_I`.
-/

open Complex

/-- Multiplication by a power of a norm-at-most-one element does not increase the norm.
The zeroth power is handled separately because a general normed ring need not satisfy
`‖1‖ = 1`. -/
theorem norm_pow_mul_le_of_norm_le_one {A : Type*} [NormedRing A] {a : A} (ha : ‖a‖ ≤ 1)
    (c : A) (n : ℕ) : ‖a ^ n * c‖ ≤ ‖c‖ := by
  cases n with
  | zero => simp
  | succ m =>
    have hpow : ‖a ^ (m + 1)‖ ≤ 1 :=
      (norm_pow_le' a m.succ_pos).trans (pow_le_one₀ (norm_nonneg a) ha)
    calc ‖a ^ (m + 1) * c‖ ≤ ‖a ^ (m + 1)‖ * ‖c‖ := norm_mul_le _ _
      _ ≤ 1 * ‖c‖ := mul_le_mul_of_nonneg_right hpow (norm_nonneg c)
      _ = ‖c‖ := one_mul _

/-- Telescoping bound, manuscript Eq. (sg-spectral-composition): for norm-at-most-one
elements of a normed ring, `‖a ^ r - b ^ r‖ ≤ r * ‖a - b‖`. -/
theorem norm_pow_sub_pow_le {A : Type*} [NormedRing A] (a b : A)
    (ha : ‖a‖ ≤ 1) (hb : ‖b‖ ≤ 1) (r : ℕ) :
    ‖a ^ r - b ^ r‖ ≤ r * ‖a - b‖ := by
  induction r with
  | zero => simp
  | succ n ih =>
    have key : a ^ (n + 1) - b ^ (n + 1) = a ^ n * (a - b) + (a ^ n - b ^ n) * b := by
      rw [mul_sub, sub_mul, pow_succ, pow_succ]
      abel
    calc ‖a ^ (n + 1) - b ^ (n + 1)‖
        = ‖a ^ n * (a - b) + (a ^ n - b ^ n) * b‖ := by rw [key]
      _ ≤ ‖a ^ n * (a - b)‖ + ‖(a ^ n - b ^ n) * b‖ := norm_add_le _ _
      _ ≤ ‖a - b‖ + ‖a ^ n - b ^ n‖ * ‖b‖ :=
          add_le_add (norm_pow_mul_le_of_norm_le_one ha _ n) (norm_mul_le _ _)
      _ ≤ ‖a - b‖ + (n * ‖a - b‖) * 1 :=
          add_le_add le_rfl (mul_le_mul ih hb (norm_nonneg b) (by positivity))
      _ = ((n + 1 : ℕ) : ℝ) * ‖a - b‖ := by push_cast; ring

/-- Phase-factored corollary of `norm_pow_sub_pow_le`: if `U` approximates the target
up to a global phase `exp (i θ)` in one step, then `r` composed steps accumulate the
phase `exp (i r θ)` and the error still grows at most linearly in `r`. -/
theorem norm_pow_sub_exp_smul_pow_le {A : Type*} [NormedRing A] [NormedAlgebra ℂ A]
    (U V : A) (hU : ‖U‖ ≤ 1) (hV : ‖V‖ ≤ 1) (θ : ℝ) (r : ℕ) :
    ‖U ^ r - Complex.exp (r * θ * Complex.I) • V ^ r‖ ≤
      r * ‖U - Complex.exp (θ * Complex.I) • V‖ := by
  have hnorm : ‖Complex.exp (θ * Complex.I) • V‖ ≤ 1 := by
    rw [norm_smul, Complex.norm_exp_ofReal_mul_I, one_mul]
    exact hV
  have hpow : (Complex.exp (θ * Complex.I) • V) ^ r
      = Complex.exp (r * θ * Complex.I) • V ^ r := by
    rw [smul_pow, ← Complex.exp_nat_mul, mul_assoc]
  calc ‖U ^ r - Complex.exp (r * θ * Complex.I) • V ^ r‖
      = ‖U ^ r - (Complex.exp (θ * Complex.I) • V) ^ r‖ := by rw [hpow]
    _ ≤ r * ‖U - Complex.exp (θ * Complex.I) • V‖ :=
        norm_pow_sub_pow_le U _ hU hnorm r
