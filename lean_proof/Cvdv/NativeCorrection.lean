import Cvdv.Telescoping
import Mathlib.Analysis.Normed.Operator.Basic
import Mathlib.Logic.Function.Iterate
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.NoncommRing

/-!
# Native-correction composition

The input-family bound samples the exact reference orbit. The five-factor
bound combines inner and outer defects under explicit contraction hypotheses.
The values of the local cubic constants are external to these statements.
-/

namespace Cvdv

/-- Accumulation of action errors on the exact reference orbit. -/
theorem iterate_error_le_action_sum {E : Type*} [NormedAddCommGroup E]
    (W V : E → E) (hW : ∀ x y, ‖W x - W y‖ ≤ ‖x - y‖)
    (x : E) (r : ℕ) :
    ‖W^[r] x - V^[r] x‖ ≤
      ∑ k ∈ Finset.range r, ‖W (V^[k] x) - V (V^[k] x)‖ := by
  induction r with
  | zero => simp
  | succ r ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', Finset.sum_range_succ]
    calc
      ‖W (W^[r] x) - V (V^[r] x)‖ ≤
          ‖W (W^[r] x) - W (V^[r] x)‖ + ‖W (V^[r] x) - V (V^[r] x)‖ :=
        by simpa only [dist_eq_norm] using dist_triangle (W (W^[r] x)) (W (V^[r] x)) (V (V^[r] x))
      _ ≤ ‖W^[r] x - V^[r] x‖ + ‖W (V^[r] x) - V (V^[r] x)‖ :=
        add_le_add (hW _ _) le_rfl
      _ ≤ _ := add_le_add ih le_rfl

/-- Projection residuals remain in the output space of the action. -/
theorem restricted_action_le {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℂ E] (D P : E →L[ℂ] E) (ε : ℝ)
    (hD : ‖D‖ ≤ 2) (hRestricted : ∀ x, ‖D (P x)‖ ≤ ε * ‖P x‖)
    (x : E) : ‖D x‖ ≤ ε * ‖P x‖ + 2 * ‖x - P x‖ := by
  have split : D x = D (P x) + D (x - P x) := by
    rw [← map_add]
    congr 1
    abel
  calc
    ‖D x‖ = ‖D (P x) + D (x - P x)‖ := by rw [split]
    _ ≤ ‖D (P x)‖ + ‖D (x - P x)‖ := norm_add_le _ _
    _ ≤ ε * ‖P x‖ + ‖D‖ * ‖x - P x‖ :=
      add_le_add (hRestricted x) (D.le_opNorm _)
    _ ≤ _ := by gcongr

/-- A contracted sandwich preserves an action difference bound. -/
theorem norm_sandwich_difference_le {A : Type*} [NormedRing A]
    (a b x y : A) (ha : ‖a‖ ≤ 1) (hb : ‖b‖ ≤ 1) :
    ‖a * x * b - a * y * b‖ ≤ ‖x - y‖ := by
  have h : a * x * b - a * y * b = a * (x - y) * b := by noncomm_ring
  rw [h]
  calc
    ‖a * (x - y) * b‖ ≤ ‖a * (x - y)‖ * ‖b‖ := norm_mul_le _ _
    _ ≤ (‖a‖ * ‖x - y‖) * ‖b‖ :=
      mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _)
    _ ≤ (1 * ‖x - y‖) * 1 := by gcongr
    _ = ‖x - y‖ := by ring

/-- The actual five factors are controlled by inner and outer split defects. -/
theorem five_factor_error_le {A : Type*} [NormedRing A]
    (a b c bc target : A) (ha : ‖a‖ ≤ 1) (εinner εouter : ℝ)
    (hinner : ‖b * c * b - bc‖ ≤ εinner)
    (houter : ‖a * bc * a - target‖ ≤ εouter) :
    ‖a * (b * c * b) * a - target‖ ≤ εinner + εouter := by
  calc
    ‖a * (b * c * b) * a - target‖ ≤
        ‖a * (b * c * b) * a - a * bc * a‖ + ‖a * bc * a - target‖ :=
      by simpa only [dist_eq_norm] using dist_triangle (a * (b * c * b) * a) (a * bc * a) target
    _ ≤ ‖b * c * b - bc‖ + ‖a * bc * a - target‖ :=
      add_le_add (norm_sandwich_difference_le a a _ bc ha ha) le_rfl
    _ ≤ _ := add_le_add hinner houter

/-- Repetition of a five-factor step with an established local error bound. -/
theorem five_factor_pow_error_le {A : Type*} [NormedRing A]
    (a b c bc target : A) (ha : ‖a‖ ≤ 1) (εinner εouter : ℝ)
    (hinner : ‖b * c * b - bc‖ ≤ εinner)
    (houter : ‖a * bc * a - target‖ ≤ εouter)
    (hstep : ‖a * (b * c * b) * a‖ ≤ 1) (htarget : ‖target‖ ≤ 1) (r : ℕ) :
    ‖(a * (b * c * b) * a)^r - target^r‖ ≤ r * (εinner + εouter) := by
  exact (norm_pow_sub_pow_le _ _ hstep htarget r).trans
    (mul_le_mul_of_nonneg_left
      (five_factor_error_le a b c bc target ha εinner εouter hinner houter)
      (Nat.cast_nonneg r))

end Cvdv
