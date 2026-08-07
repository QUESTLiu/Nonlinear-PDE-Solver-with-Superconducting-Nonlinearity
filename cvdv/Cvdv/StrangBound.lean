import Cvdv.TrotterBound
import Mathlib.Analysis.Analytic.Order
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.Matrix.Hermitian

/-!
# Local and global error of the second-order Strang product formula

This file proves the local cubic error of the symmetric product

`exp (t X / 2) * exp (t Y) * exp (t X / 2)`

for bounded generators in a real Banach algebra. The local cubic defect and a
telescoping contraction argument give global `O(n⁻²)` convergence. In the
finite-cutoff LLE application, the Banach algebra is the algebra of bounded
operators on the joint ancillary-system cutoff space. Concrete matrix wrappers
use `Matrix.toEuclideanCLM`. No theorem here concerns an unbounded oscillator
generator before the ancillary cutoff is imposed.
-/

open NormedSpace
open Asymptotics
open scoped Topology

namespace Cvdv

section BanachAlgebra

variable {A : Type*} [NormedRing A] [NormedAlgebra ℝ A] [CompleteSpace A]

/-- One symmetric Strang step for two bounded generators. -/
noncomputable def strangStep (X Y : A) (t : ℝ) : A :=
  exp (t • ((1 / 2 : ℝ) • X)) * exp (t • Y) * exp (t • ((1 / 2 : ℝ) • X))

/-- Difference between one Strang step and the exact exponential. -/
noncomputable def strangDefect (X Y : A) (t : ℝ) : A :=
  strangStep X Y t - exp (t • (X + Y))

private theorem hasDerivAt_strangDefect (X Y : A) (t : ℝ) :
    HasDerivAt (strangDefect X Y)
      (let H := (1 / 2 : ℝ) • X
       let eH := exp (t • H)
       let eY := exp (t • Y)
       (((H * eH) * eY + eH * (Y * eY)) * eH + (eH * eY) * (H * eH))
         - (X + Y) * exp (t • (X + Y))) t := by
  let H : A := (1 / 2 : ℝ) • X
  have hH : HasDerivAt (fun s : ℝ => exp (s • H)) (H * exp (t • H)) t :=
    hasDerivAt_exp_smul_const' H t
  have hY : HasDerivAt (fun s : ℝ => exp (s • Y)) (Y * exp (t • Y)) t :=
    hasDerivAt_exp_smul_const' Y t
  have hXY : HasDerivAt (fun s : ℝ => exp (s • (X + Y)))
      ((X + Y) * exp (t • (X + Y))) t :=
    hasDerivAt_exp_smul_const' (X + Y) t
  change HasDerivAt
    (fun s : ℝ => exp (s • H) * exp (s • Y) * exp (s • H) - exp (s • (X + Y)))
    ((((H * exp (t • H)) * exp (t • Y) + exp (t • H) * (Y * exp (t • Y))) *
          exp (t • H) +
        (exp (t • H) * exp (t • Y)) * (H * exp (t • H))) -
      (X + Y) * exp (t • (X + Y))) t
  exact ((hH.mul hY).mul hH).sub hXY

private theorem analyticAt_strangDefect (X Y : A) :
    AnalyticAt ℝ (strangDefect X Y) 0 := by
  let H : A := (1 / 2 : ℝ) • X
  have hlinearH : AnalyticAt ℝ (fun t : ℝ => t • H) 0 :=
    analyticAt_id.smul analyticAt_const
  have hlinearY : AnalyticAt ℝ (fun t : ℝ => t • Y) 0 :=
    analyticAt_id.smul analyticAt_const
  have hlinearXY : AnalyticAt ℝ (fun t : ℝ => t • (X + Y)) 0 :=
    analyticAt_id.smul analyticAt_const
  have hH : AnalyticAt ℝ (fun t : ℝ => exp (t • H)) 0 := by
    simpa only [Function.comp_def] using
      (NormedSpace.exp_analytic (𝕂 := ℝ) (𝔸 := A) (0 : A)).comp_of_eq hlinearH (by simp)
  have hY : AnalyticAt ℝ (fun t : ℝ => exp (t • Y)) 0 := by
    simpa only [Function.comp_def] using
      (NormedSpace.exp_analytic (𝕂 := ℝ) (𝔸 := A) (0 : A)).comp_of_eq hlinearY (by simp)
  have hXY : AnalyticAt ℝ (fun t : ℝ => exp (t • (X + Y))) 0 := by
    simpa only [Function.comp_def] using
      (NormedSpace.exp_analytic (𝕂 := ℝ) (𝔸 := A) (0 : A)).comp_of_eq hlinearXY (by simp)
  change AnalyticAt ℝ
    (fun t : ℝ => exp (t • H) * exp (t • Y) * exp (t • H) - exp (t • (X + Y))) 0
  exact ((hH.mul hY).mul hH).sub hXY

/-- A symmetric Strang step has local operator-norm error `O(t³)` for bounded generators.

The result is stated in a Banach algebra and therefore applies to continuous
linear operators on any complete normed space. The finite-matrix wrapper below
uses `Matrix.toEuclideanCLM`. No commutativity or self-adjointness assumption is
needed for the local order statement. -/
theorem strangDefect_isBigO_cubic (X Y : A) :
    (strangDefect X Y) =O[𝓝 0] (fun t : ℝ => t ^ 3) := by
  let H : A := (1 / 2 : ℝ) • X
  let eH : ℝ → A := fun t => exp (t • H)
  let eY : ℝ → A := fun t => exp (t • Y)
  let eXY : ℝ → A := fun t => exp (t • (X + Y))
  let D : ℝ → A := fun t =>
    (H * eH t) * eY t * eH t +
      eH t * (Y * eY t) * eH t +
      (eH t * eY t) * (H * eH t) - (X + Y) * eXY t
  have hX : X = H + H := by
    dsimp only [H]
    rw [← add_smul]
    norm_num
  have hdefectDeriv : ∀ t, HasDerivAt (strangDefect X Y) (D t) t := by
    intro t
    simpa only [D, H, eH, eY, eXY, add_mul] using hasDerivAt_strangDefect X Y t
  have hDzero : D 0 = 0 := by
    simp only [D, eH, eY, eXY, zero_smul, exp_zero, mul_one, one_mul]
    rw [hX]
    noncomm_ring
  have hDderiv : HasDerivAt D 0 0 := by
    have hH : HasDerivAt eH H 0 := by
      simpa only [eH, zero_smul, exp_zero, mul_one] using
        hasDerivAt_exp_smul_const' H (0 : ℝ)
    have hY : HasDerivAt eY Y 0 := by
      simpa only [eY, zero_smul, exp_zero, mul_one] using
        hasDerivAt_exp_smul_const' Y (0 : ℝ)
    have hXY : HasDerivAt eXY (X + Y) 0 := by
      simpa only [eXY, zero_smul, exp_zero, mul_one] using
        hasDerivAt_exp_smul_const' (X + Y) (0 : ℝ)
    have hterm1 := (((hasDerivAt_const (0 : ℝ) H).mul hH).mul hY).mul hH
    have hterm2 := (hH.mul ((hasDerivAt_const (0 : ℝ) Y).mul hY)).mul hH
    have hterm3 := (hH.mul hY).mul ((hasDerivAt_const (0 : ℝ) H).mul hH)
    have hexact := (hasDerivAt_const (0 : ℝ) (X + Y)).mul hXY
    have hcombined := ((hterm1.add hterm2).add hterm3).sub hexact
    simp only [Pi.mul_apply] at hcombined
    change HasDerivAt
      (fun t =>
        (H * eH t) * eY t * eH t + eH t * (Y * eY t) * eH t +
          (eH t * eY t) * (H * eH t) - (X + Y) * eXY t)
      0 0
    apply hcombined.congr_deriv
    simp only [eH, eY, eXY, zero_smul, exp_zero, mul_one, one_mul]
    rw [hX]
    noncomm_ring
  have hderiv : deriv (strangDefect X Y) = D := by
    funext t
    exact (hdefectDeriv t).deriv
  have hzero : ∀ i < 3, iteratedDeriv i (strangDefect X Y) 0 = 0 := by
    intro i hi
    interval_cases i
    · simp only [iteratedDeriv_zero, strangDefect, strangStep, zero_smul, exp_zero,
        mul_one, sub_self]
    · rw [iteratedDeriv_one, hderiv]
      exact hDzero
    · rw [show 2 = 1 + 1 by norm_num, iteratedDeriv_succ, iteratedDeriv_one, hderiv]
      exact hDderiv.deriv
  obtain ⟨F, hF, hEq⟩ := (analyticAt_strangDefect X Y).exists_eventuallyEq_sum_add_pow_mul 3
  have hsum : ∀ t : ℝ,
      (∑ i ∈ Finset.range 3,
        (t ^ i / i.factorial) • iteratedDeriv i (strangDefect X Y) 0) = 0 := by
    intro t
    apply Finset.sum_eq_zero
    intro i hi
    rw [hzero i (Finset.mem_range.mp hi), smul_zero]
  have hfactor : strangDefect X Y =ᶠ[𝓝 0] fun t => t ^ 3 • F t := by
    filter_upwards [hEq] with t ht
    rw [ht, hsum]
    exact zero_add _
  have hbounded : F =O[𝓝 0] (fun _ : ℝ => (1 : ℝ)) :=
    hF.continuousAt.isBigO_one ℝ
  have hproduct := (isBigO_refl (fun t : ℝ => t ^ 3) (𝓝 0)).smul hbounded
  have hcubic : (fun t => t ^ 3 • F t) =O[𝓝 0] (fun t : ℝ => t ^ 3) := by
    simpa only [smul_eq_mul, mul_one] using hproduct
  exact hcubic.congr' hfactor.symm (Filter.EventuallyEq.rfl)

/-- A local cubic Strang defect gives global quadratic convergence under the
usual contraction hypotheses for the split and exact one-step evolutions.

The statement is for a fixed total generator `X + Y`: the `n`-step
approximation uses step size `1 / n`. It controls only product-formula error
for this bounded generator. -/
theorem strang_pow_sub_exp_isBigO_inv_sq_of_contraction (X Y : A)
    (hSplit : ∀ n : ℕ, 1 ≤ n → ‖strangStep X Y (n : ℝ)⁻¹‖ ≤ 1)
    (hExact : ∀ n : ℕ, 1 ≤ n → ‖exp ((n : ℝ)⁻¹ • (X + Y))‖ ≤ 1) :
    (fun n : ℕ =>
      (strangStep X Y (n : ℝ)⁻¹) ^ n - exp (X + Y))
      =O[Filter.atTop] (fun n : ℕ => ((n : ℝ)⁻¹) ^ 2) := by
  have hlocal := (strangDefect_isBigO_cubic X Y).comp_tendsto
    (tendsto_inv_atTop_nhds_zero_nat (𝕜 := ℝ))
  change (fun n : ℕ => strangDefect X Y (n : ℝ)⁻¹)
      =O[Filter.atTop] (fun n : ℕ => ((n : ℝ)⁻¹) ^ 3) at hlocal
  obtain ⟨c, hcpos, hc⟩ := hlocal.exists_pos
  rw [isBigO_iff]
  refine ⟨c, ?_⟩
  filter_upwards [Filter.eventually_atTop.2 ⟨1, fun n hn => hn⟩, hc.bound] with n hn hdefect
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.one_le_iff_ne_zero.mp hn)
  have hpow : exp (X + Y) = exp ((n : ℝ)⁻¹ • (X + Y)) ^ n := by
    rw [← exp_nsmul_eq_pow]
    congr 1
    rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul, mul_inv_cancel₀ hn0, one_smul]
  calc
    ‖(strangStep X Y (n : ℝ)⁻¹) ^ n - exp (X + Y)‖ =
        ‖(strangStep X Y (n : ℝ)⁻¹) ^ n -
          exp ((n : ℝ)⁻¹ • (X + Y)) ^ n‖ := by rw [hpow]
    _ ≤ n * ‖strangStep X Y (n : ℝ)⁻¹ - exp ((n : ℝ)⁻¹ • (X + Y))‖ :=
      norm_pow_sub_pow_le _ _ (hSplit n hn) (hExact n hn) n
    _ = n * ‖strangDefect X Y (n : ℝ)⁻¹‖ := by
      rfl
    _ ≤ n * (c * ‖((n : ℝ)⁻¹) ^ 3‖) :=
      mul_le_mul_of_nonneg_left hdefect (Nat.cast_nonneg n)
    _ = c * ‖((n : ℝ)⁻¹) ^ 2‖ := by
      rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (by positivity),
        abs_of_nonneg (by positivity)]
      field_simp

end BanachAlgebra

section HilbertSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]

/-- A symmetric Strang step built from two Hermitian bounded operators is a
contraction. -/
lemma norm_strangStep_neg_I_smul_le_one (H1 H2 : E →L[ℂ] E)
    (hH1 : ∀ v w : E, inner ℂ (H1 v) w = inner ℂ v (H1 w))
    (hH2 : ∀ v w : E, inner ℂ (H2 v) w = inner ℂ v (H2 w))
    (u : ℝ) :
    ‖strangStep ((-Complex.I) • H1) ((-Complex.I) • H2) u‖ ≤ 1 := by
  unfold strangStep
  have hhalf :
      ‖exp (u • ((1 / 2 : ℝ) • ((-Complex.I) • H1)))‖ ≤ 1 := by
    rw [real_smul_complex_smul, real_smul_complex_smul]
    have hscalar :
        (-Complex.I * ((1 / 2 : ℝ) : ℂ)) * (u : ℂ) =
          -Complex.I * ((u / 2 : ℝ) : ℂ) := by
      push_cast
      ring
    rw [hscalar]
    exact norm_exp_neg_I_smul_le_one H1 hH1 (u / 2)
  have hmiddle : ‖exp (u • ((-Complex.I) • H2))‖ ≤ 1 := by
    rw [real_smul_complex_smul]
    exact norm_exp_neg_I_smul_le_one H2 hH2 u
  calc
    ‖exp (u • ((1 / 2 : ℝ) • ((-Complex.I) • H1))) *
        exp (u • ((-Complex.I) • H2)) *
        exp (u • ((1 / 2 : ℝ) • ((-Complex.I) • H1)))‖
        ≤ ‖exp (u • ((1 / 2 : ℝ) • ((-Complex.I) • H1))) *
            exp (u • ((-Complex.I) • H2))‖ *
          ‖exp (u • ((1 / 2 : ℝ) • ((-Complex.I) • H1)))‖ := norm_mul_le _ _
    _ ≤ (‖exp (u • ((1 / 2 : ℝ) • ((-Complex.I) • H1)))‖ *
          ‖exp (u • ((-Complex.I) • H2))‖) *
        ‖exp (u • ((1 / 2 : ℝ) • ((-Complex.I) • H1)))‖ :=
      mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _)
    _ ≤ (1 * 1) * 1 := by
      gcongr
    _ = 1 := by norm_num

end HilbertSpace

section FiniteCutoff

variable {ιa ιs : Type*} [Fintype ιa] [Fintype ιs]

/-- Local cubic Strang error on a finite joint ancillary-system cutoff space.

`Hsys` represents `Iₐ ⊗ H_LLE` and `Hloss` represents `xₐ ⊗ L_loss` after both
cutoffs have been imposed. -/
theorem lleFiniteCutoff_strangDefect_isBigO_cubic
    (Hsys Hloss :
      EuclideanSpace ℂ (ιa × ιs) →L[ℂ] EuclideanSpace ℂ (ιa × ιs)) :
    (strangDefect ((-Complex.I) • Hsys) ((-Complex.I) • Hloss))
      =O[𝓝 0] (fun t : ℝ => t ^ 3) :=
  strangDefect_isBigO_cubic _ _

/-- For fixed cutoffs and fixed physical time `T`, the repeated finite-cutoff
Strang formula has global operator-norm error `O(n⁻²)`.

This compares with the exponential of the same finite truncated dilation. It
does not include kernel, ancillary-cutoff, postselection, or hardware error. -/
theorem lleFiniteCutoff_strangPow_isBigO_inv_sq
    (Hsys Hloss :
      EuclideanSpace ℂ (ιa × ιs) →L[ℂ] EuclideanSpace ℂ (ιa × ιs))
    (hHsys : ∀ v w, inner ℂ (Hsys v) w = inner ℂ v (Hsys w))
    (hHloss : ∀ v w, inner ℂ (Hloss v) w = inner ℂ v (Hloss w))
    (T : ℝ) :
    (fun n : ℕ =>
      (strangStep ((-Complex.I) • ((T : ℂ) • Hsys))
        ((-Complex.I) • ((T : ℂ) • Hloss)) (n : ℝ)⁻¹) ^ n -
      exp ((-Complex.I) • ((T : ℂ) • (Hsys + Hloss))))
      =O[Filter.atTop] (fun n : ℕ => ((n : ℝ)⁻¹) ^ 2) := by
  have hTHsys : ∀ v w,
      inner ℂ (((T : ℂ) • Hsys) v) w = inner ℂ v (((T : ℂ) • Hsys) w) :=
    LinearMap.IsSymmetric.smul (Complex.conj_ofReal T) hHsys
  have hTHloss : ∀ v w,
      inner ℂ (((T : ℂ) • Hloss) v) w = inner ℂ v (((T : ℂ) • Hloss) w) :=
    LinearMap.IsSymmetric.smul (Complex.conj_ofReal T) hHloss
  have hsum :
      (-Complex.I) • ((T : ℂ) • Hsys) + (-Complex.I) • ((T : ℂ) • Hloss) =
        (-Complex.I) • ((T : ℂ) • (Hsys + Hloss)) := by
    rw [smul_add, smul_add]
  rw [← hsum]
  apply strang_pow_sub_exp_isBigO_inv_sq_of_contraction
  · intro n _hn
    exact norm_strangStep_neg_I_smul_le_one ((T : ℂ) • Hsys) ((T : ℂ) • Hloss)
      hTHsys hTHloss (n : ℝ)⁻¹
  · intro n _hn
    rw [hsum, real_smul_complex_smul]
    have hsumSymm : ∀ v w,
        inner ℂ (((T : ℂ) • (Hsys + Hloss)) v) w =
          inner ℂ v (((T : ℂ) • (Hsys + Hloss)) w) :=
      LinearMap.IsSymmetric.smul (Complex.conj_ofReal T)
        (LinearMap.IsSymmetric.add hHsys hHloss)
    exact norm_exp_neg_I_smul_le_one ((T : ℂ) • (Hsys + Hloss))
      hsumSymm (n : ℝ)⁻¹

variable [DecidableEq ιa] [DecidableEq ιs]

/-- Concrete finite-matrix specialization of the local cubic Strang theorem.
The matrix algebra acts on the joint cutoff `EuclideanSpace` through the
operator-norm-preserving equivalence `Matrix.toEuclideanCLM`. -/
theorem lleFiniteCutoffMatrix_strangDefect_isBigO_cubic
    (Hsys Hloss : Matrix (ιa × ιs) (ιa × ιs) ℂ) :
    (strangDefect
      ((-Complex.I) •
        ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs)) Hsys))
      ((-Complex.I) •
        ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs)) Hloss)))
      =O[𝓝 0] (fun t : ℝ => t ^ 3) :=
  lleFiniteCutoff_strangDefect_isBigO_cubic _ _

/-- Concrete finite-matrix specialization of global second-order Strang
convergence for Hermitian finite truncated dilation terms. -/
theorem lleFiniteCutoffMatrix_strangPow_isBigO_inv_sq
    (Hsys Hloss : Matrix (ιa × ιs) (ιa × ιs) ℂ)
    (hHsys : Hsys.IsHermitian) (hHloss : Hloss.IsHermitian) (T : ℝ) :
    (fun n : ℕ =>
      (strangStep
        ((-Complex.I) • ((T : ℂ) •
          ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs)) Hsys)))
        ((-Complex.I) • ((T : ℂ) •
          ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs)) Hloss)))
        (n : ℝ)⁻¹) ^ n -
      exp ((-Complex.I) • ((T : ℂ) •
        (((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs)) Hsys) +
          ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs)) Hloss)))))
      =O[Filter.atTop] (fun n : ℕ => ((n : ℝ)⁻¹) ^ 2) := by
  apply lleFiniteCutoff_strangPow_isBigO_inv_sq
  · exact Matrix.isSymmetric_toEuclideanLin_iff.mpr hHsys
  · exact Matrix.isSymmetric_toEuclideanLin_iff.mpr hHloss

end FiniteCutoff

end Cvdv
