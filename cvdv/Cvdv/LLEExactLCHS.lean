import Cvdv.LLEFinite
import Mathlib.Analysis.Normed.Algebra.MatrixExponential
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

/-!
# Conditional exact-LCHS wrapper for finite LLE matrices

This file records the exact-LCHS integral identity and normalized oscillator
factorization as hypotheses. It proves only the resulting scaled position-space
postselection identity. It does not construct an admissible kernel or an
infinite-dimensional oscillator realization.
-/

namespace Cvdv

open MeasureTheory
open scoped ComplexOrder Matrix.Norms.L2Operator

noncomputable section

section LLEExactLCHS

variable {σ : Type*} [Fintype σ] [DecidableEq σ]

/-- The unitary branch (e^{-it(xL+H)}) in the exact-LCHS integral. -/
def lleLCHSBranch (t x : ℝ) (L H : Matrix σ σ ℂ) : Matrix σ σ ℂ :=
  NormedSpace.exp
    ((-Complex.I * (t : ℂ)) • ((x : ℂ) • L + H))

/-- The target finite-matrix map (e^{-t(L+iH)}). -/
def lleLCHSTarget (t : ℝ) (L H : Matrix σ σ ℂ) : Matrix σ σ ℂ :=
  NormedSpace.exp ((-(t : ℂ)) • (L + Complex.I • H))

/-- The operator-valued exact-LCHS integral for a supplied kernel. -/
def lleLCHSIntegral (kernel : ℝ → ℂ) (t : ℝ)
    (L H : Matrix σ σ ℂ) : Matrix σ σ ℂ :=
  ∫ x, kernel x • lleLCHSBranch t x L H ∂volume

/-- The external exact-LCHS input: one kernel supplies the operator identity at every
nonnegative time. -/
def HasUniformExactLCHSIdentity (kernel : ℝ → ℂ)
    (L H : Matrix σ σ ℂ) : Prop :=
  Integrable kernel volume ∧
    ∀ t : ℝ, 0 ≤ t →
      Integrable (fun x ↦ kernel x • lleLCHSBranch t x L H) volume ∧
      lleLCHSIntegral kernel t L H = lleLCHSTarget t L H

/-- Exact kernel admissibility for a positive-semidefinite accretive part and a
Hermitian Hamiltonian part.  The exact integral identity remains an assumption. -/
def IsAdmissibleExactLCHSKernel (kernel : ℝ → ℂ)
    (L H : Matrix σ σ ℂ) : Prop :=
  L.PosSemidef ∧ H.IsHermitian ∧ HasUniformExactLCHSIdentity kernel L H

/-- A normalized oscillator wavefunction in the position representation. -/
def IsNormalizedOscillatorWavefunction (ξ : ℝ → ℂ) : Prop :=
  MemLp ξ 2 volume ∧
    ∫ x, ‖ξ x‖ ^ 2 ∂volume = 1

/-- The normalized factorization (g=\alpha_g\phi_g^*\psi_g). -/
def IsNormalizedOscillatorFactorization (kernel : ℝ → ℂ) (alpha : ℂ)
    (phi psi : ℝ → ℂ) : Prop :=
  IsNormalizedOscillatorWavefunction phi ∧
    IsNormalizedOscillatorWavefunction psi ∧
    ∀ᵐ x ∂volume, kernel x = alpha * (star (phi x) * psi x)

/-- The position-space form of the unscaled oscillator postselection block. -/
def lleOscillatorPostselectedBlock (phi psi : ℝ → ℂ) (t : ℝ)
    (L H : Matrix σ σ ℂ) : Matrix σ σ ℂ :=
  ∫ x, (star (phi x) * psi x) • lleLCHSBranch t x L H ∂volume

/-- Under explicit exact-kernel and normalized-factorization hypotheses, the scaled
position-space postselection block equals the target nonunitary map. -/
theorem lle_scaled_postselectedMap_eq_target
    (kernel : ℝ → ℂ) (alpha : ℂ) (phi psi : ℝ → ℂ) (t : ℝ)
    (L H : Matrix σ σ ℂ)
    (ht : 0 ≤ t)
    (hKernel : IsAdmissibleExactLCHSKernel kernel L H)
    (hFactor : IsNormalizedOscillatorFactorization kernel alpha phi psi) :
    alpha • lleOscillatorPostselectedBlock phi psi t L H =
      lleLCHSTarget t L H := by
  calc
    alpha • lleOscillatorPostselectedBlock phi psi t L H =
        ∫ x, alpha • ((star (phi x) * psi x) • lleLCHSBranch t x L H) ∂volume := by
      rw [lleOscillatorPostselectedBlock, integral_smul]
    _ = lleLCHSIntegral kernel t L H := by
      unfold lleLCHSIntegral
      apply integral_congr_ae
      filter_upwards [hFactor.2.2] with x hx
      rw [smul_smul, ← hx]
    _ = lleLCHSTarget t L H := (hKernel.2.2.2 t ht).2

/-- The finite LLE structural theorem supplies positivity and Hermiticity, leaving
only the uniform exact-kernel identity and normalized factorization as external inputs. -/
theorem finiteLLE_scaled_postselectedMap_eq_target
    {ι : Type*} [Fintype ι]
    (a : ι → Matrix σ σ ℂ) (next : ι → ι)
    (delta dispersion g kappa : ℝ) (F : ι → ℂ) (hkappa : 0 ≤ kappa)
    (kernel : ℝ → ℂ) (alpha : ℂ) (phi psi : ℝ → ℂ) (t : ℝ) (ht : 0 ≤ t)
    (hExact : HasUniformExactLCHSIdentity kernel (finiteLLELoss a kappa)
      (finiteLLEHamiltonian a next delta dispersion g F))
    (hFactor : IsNormalizedOscillatorFactorization kernel alpha phi psi) :
    alpha • lleOscillatorPostselectedBlock phi psi t (finiteLLELoss a kappa)
        (finiteLLEHamiltonian a next delta dispersion g F) =
      NormedSpace.exp ((-(t : ℂ)) •
        finiteLLEGenerator a next delta dispersion g kappa F) := by
  classical
  have hStructural :=
    finiteLLE_structural_prerequisites a next delta dispersion g kappa F hkappa
  calc
    alpha • lleOscillatorPostselectedBlock phi psi t (finiteLLELoss a kappa)
          (finiteLLEHamiltonian a next delta dispersion g F) =
        lleLCHSTarget t (finiteLLELoss a kappa)
          (finiteLLEHamiltonian a next delta dispersion g F) :=
      lle_scaled_postselectedMap_eq_target kernel alpha phi psi t
        (finiteLLELoss a kappa) (finiteLLEHamiltonian a next delta dispersion g F) ht
        ⟨hStructural.1, hStructural.2.1, hExact⟩ hFactor
    _ = NormedSpace.exp ((-(t : ℂ)) •
        finiteLLEGenerator a next delta dispersion g kappa F) := by
      rw [lleLCHSTarget, hStructural.2.2]

end LLEExactLCHS

end

end Cvdv
