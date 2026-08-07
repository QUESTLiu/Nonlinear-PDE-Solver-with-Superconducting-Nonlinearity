import Cvdv.FiniteDuffing
import Cvdv.StrangBound

/-!
# Second-order Strang bounds for the strict-cutoff Duffing--Kerr split

This file instantiates the abstract symmetric-product-formula theorems of
`StrangBound.lean` at the strict-cutoff Duffing split of manuscript
Eq. (eq:df-C-N): `X = Π_N Ĥ_K Π_N` and `Y = λ Π_N V̂_cr Π_N`, the two
Hamiltonian sectors of `cutoffDuffingHamiltonian_split`.  It formalizes the
sentence following Eq. (eq:df-C-N): unlike the worst-case first-order
product-formula estimate controlled by `C_N`, a symmetric product formula for
this split has a different (second-order) error scaling, with local defect
`O(Δt³)` and global error `O(n⁻²)`.
-/

open NormedSpace
open Asymptotics
open scoped Topology

namespace Cvdv

/-- The symmetric Strang step for the strict-cutoff Duffing--Kerr split has a
local operator-norm defect of cubic order in the step size: the second-order
scaling of the sentence after manuscript Eq. (eq:df-C-N). -/
theorem duffingStrang_defect_isBigO_cubic (N : ℕ) (ω0 lam : ℝ) :
    (strangDefect ((-Complex.I) • cutoffDuffingKerrHamiltonian N ω0 lam)
      ((-Complex.I) • ((lam : ℂ) • cutoffCounterRotating N)))
      =O[𝓝 0] (fun t : ℝ => t ^ 3) :=
  strangDefect_isBigO_cubic _ _

/-- Auxiliary global second-order Strang bound on an arbitrary complex Hilbert
space, in the plumbing of `lleFiniteCutoff_strangPow_isBigO_inv_sq` but with
the joint cutoff Euclidean space generalized so that it applies to the
`FockCutoff` operator algebra. -/
private theorem duffingStrang_pow_aux {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℂ E] [CompleteSpace E] (H1 H2 : E →L[ℂ] E)
    (h1 : ∀ v w, inner ℂ (H1 v) w = inner ℂ v (H1 w))
    (h2 : ∀ v w, inner ℂ (H2 v) w = inner ℂ v (H2 w)) (T : ℝ) :
    (fun n : ℕ =>
      (strangStep ((-Complex.I) • ((T : ℂ) • H1))
        ((-Complex.I) • ((T : ℂ) • H2)) (n : ℝ)⁻¹) ^ n -
      exp ((-Complex.I) • ((T : ℂ) • (H1 + H2))))
      =O[Filter.atTop] (fun n : ℕ => ((n : ℝ)⁻¹) ^ 2) := by
  have hTH1 : ∀ v w,
      inner ℂ (((T : ℂ) • H1) v) w = inner ℂ v (((T : ℂ) • H1) w) :=
    LinearMap.IsSymmetric.smul (Complex.conj_ofReal T) h1
  have hTH2 : ∀ v w,
      inner ℂ (((T : ℂ) • H2) v) w = inner ℂ v (((T : ℂ) • H2) w) :=
    LinearMap.IsSymmetric.smul (Complex.conj_ofReal T) h2
  have hsum :
      (-Complex.I) • ((T : ℂ) • H1) + (-Complex.I) • ((T : ℂ) • H2) =
        (-Complex.I) • ((T : ℂ) • (H1 + H2)) := by
    rw [smul_add, smul_add]
  rw [← hsum]
  apply strang_pow_sub_exp_isBigO_inv_sq_of_contraction
  · intro n _hn
    exact norm_strangStep_neg_I_smul_le_one ((T : ℂ) • H1) ((T : ℂ) • H2)
      hTH1 hTH2 (n : ℝ)⁻¹
  · intro n _hn
    rw [hsum, real_smul_complex_smul]
    have hsumSymm : ∀ v w,
        inner ℂ (((T : ℂ) • (H1 + H2)) v) w =
          inner ℂ v (((T : ℂ) • (H1 + H2)) w) :=
      LinearMap.IsSymmetric.smul (Complex.conj_ofReal T)
        (LinearMap.IsSymmetric.add h1 h2)
    exact norm_exp_neg_I_smul_le_one ((T : ℂ) • (H1 + H2))
      hsumSymm (n : ℝ)⁻¹

/-- For a fixed cutoff `N` and fixed physical time `T`, the repeated symmetric
Strang formula for the strict-cutoff Duffing--Kerr split
`Π_N Ĥ_K Π_N + λ Π_N V̂_cr Π_N` converges to the exact split evolution with
global operator-norm error `O(n⁻²)`: the second-order error scaling of the
symmetric product formula asserted after manuscript Eq. (eq:df-C-N).  Both
Hermiticity side conditions are discharged from the strict-sandwich symmetry
theorems of `FiniteDuffing.lean`. -/
theorem duffingStrang_pow_isBigO_inv_sq (N : ℕ) (ω0 lam T : ℝ) :
    (fun n : ℕ =>
      (strangStep
        ((-Complex.I) • ((T : ℂ) • cutoffDuffingKerrHamiltonian N ω0 lam))
        ((-Complex.I) • ((T : ℂ) • ((lam : ℂ) • cutoffCounterRotating N)))
        (n : ℝ)⁻¹) ^ n -
      exp ((-Complex.I) • ((T : ℂ) •
        (cutoffDuffingKerrHamiltonian N ω0 lam +
          (lam : ℂ) • cutoffCounterRotating N))))
      =O[Filter.atTop] (fun n : ℕ => ((n : ℝ)⁻¹) ^ 2) := by
  have hKerr : ∀ v w : FockCutoff N,
      inner ℂ (cutoffDuffingKerrHamiltonian N ω0 lam v) w =
        inner ℂ v (cutoffDuffingKerrHamiltonian N ω0 lam w) := fun v w =>
    cutoffDuffingKerrHamiltonian_isSymmetric N ω0 lam v w
  have hCorr : ∀ v w : FockCutoff N,
      inner ℂ (((lam : ℂ) • cutoffCounterRotating N) v) w =
        inner ℂ v (((lam : ℂ) • cutoffCounterRotating N) w) :=
    LinearMap.IsSymmetric.smul (Complex.conj_ofReal lam)
      (cutoffCounterRotating_isSymmetric N)
  exact duffingStrang_pow_aux (cutoffDuffingKerrHamiltonian N ω0 lam)
    ((lam : ℂ) • cutoffCounterRotating N) hKerr hCorr T

end Cvdv
