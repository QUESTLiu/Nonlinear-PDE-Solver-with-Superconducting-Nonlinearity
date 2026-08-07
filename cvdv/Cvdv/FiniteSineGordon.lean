import Cvdv.SiteLift
import Cvdv.FockHermitian
import Cvdv.Telescoping
import Mathlib.Analysis.Normed.Algebra.MatrixExponential
import Mathlib.LinearAlgebra.UnitaryGroup

/-!
# The finite sine-Gordon lattice module

Formalization of the finite-cutoff sine-Gordon lattice of manuscript
Sec. `sec:sg`.  The single-site quadratures follow the quantization
convention of Eq. `eq:sg-quadratures`,

`φ̂_j = (â_j + â_j†)/√(2ω_b)`, `Π̂_j = i √(ω_b/2) (â_j† − â_j)`,

realized here as strict finite Fock sandwiches (`fockSandwichMatrix`)
of the algebraic ladder polynomials, in the convention of
`Cvdv.FockCutoff`.  The onsite Josephson cosine of the lattice
Hamiltonian Eq. `eq:sg-lattice` is written as the single-site lift
(`siteLift`) of the single-mode matrix cosine
`cos A = (e^{iA} + e^{-iA})/2`; the lemma `siteLift_matCos` proves that
this lift **equals** the multimode matrix cosine of the lifted site
quadrature, the manuscript's onsite `cos φ̂_j`.  The generator
`A_SG = iH_SG` of Eq. `eq:sg-classification` is purely Hermitian, which
is the content of `sgCutoffHamiltonian_isHermitian`.  The deformation
Hamiltonians `ΔH_dis` and `ΔH_2` of Eq. `eq:sg-deformations` and the
double sine-Gordon target of Eq. `eq:sg-double-target` are formalized
with their Hermiticity proofs.  The SNAP gate of Eq. `eq:df-snap-d` is
proved unitary; the manuscript displacement `D(α) = exp(αâ† − α*â)` is
exhibited concretely at the cutoff by the truncated generator
`sgDispGen N α = α·(sandwich of â†) − α*·(sandwich of â)`, proved
skew-Hermitian (`sgDispGen_conjTranspose`), so its exponential is
unitary (`sgDispGen_disp_mem_unitaryGroup`).  Layered products in the
ansatz of Eqs. `eq:snapd-ansatz` / `eq:df-ansatz` are unitary, unitary
matrices are `ℓ²`-operator-norm contractions, and the phase-aligned
telescoping bound of Eq. `eq:sg-spectral-composition` is instantiated
for the concrete compiled objects with both contraction premises proved
rather than assumed (`sgSpectral_composition_bound`), including the
amplitude-parametrized form whose layers are literally
`D(α_ℓ) SNAP(θ_ℓ)` (`sgSpectral_composition_bound_disp`).

All matrix norms are the `ℓ²` operator norm
(scoped `Matrix.Norms.L2Operator`), as in `Cvdv.SiteLift`.
-/

namespace Cvdv

open NormedSpace
open scoped BigOperators Matrix Matrix.Norms.L2Operator

noncomputable section

section SingleMode

/-- A real number, coerced to `ℂ`, is self-adjoint.  Coefficient plumbing for
the Hermiticity proofs below. -/
theorem sgMat_ofReal_isSelfAdjoint (c : ℝ) : IsSelfAdjoint ((c : ℝ) : ℂ) :=
  Complex.conj_ofReal c

/-- The strict-cutoff position quadrature of manuscript Eq. `eq:sg-quadratures`,
`φ̂ = (â + â†)/√(2ω_b)`: the algebraic polynomial `â + â†` is formed on the full
Fock space and sandwiched to the retained `N`-level space before the real
prefactor `1/√(2ω_b)` is applied. -/
def sgCutoffPhi (N : ℕ) (omega : ℝ) : Matrix (Fin N) (Fin N) ℂ :=
  ((1 / Real.sqrt (2 * omega) : ℝ) : ℂ) • fockSandwichMatrix N (lowering + raising)

/-- For a positive basis frequency `ω_b > 0`, the cutoff position quadrature
`φ̂` of Eq. `eq:sg-quadratures` is Hermitian: the strict sandwich of `â + â†`
is Hermitian and the prefactor `1/√(2ω_b)` is real. -/
theorem sgCutoffPhi_isHermitian (N : ℕ) {omega : ℝ} (_homega : 0 < omega) :
    (sgCutoffPhi N omega).IsHermitian :=
  (fockSandwichMatrix_isHermitian_of_isAdjointPair N _
    fockPosition_isAdjointPair).smul (sgMat_ofReal_isSelfAdjoint _)

/-- The algebraic momentum polynomial `i(â† − â)` is formally self-adjoint for
the standard Fock-space sesquilinear form: the adjoint relations of the ladder
pair combine with the conjugation of the prefactor `i`. -/
theorem sgCutoffPi_base_isAdjointPair :
    LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm
      ((Complex.I • (raising - lowering) : Module.End ℂ FockAlg) : FockAlg → FockAlg)
      ((Complex.I • (raising - lowering) : Module.End ℂ FockAlg) : FockAlg → FockAlg) := by
  intro x y
  have hrl := raising_lowering_isAdjointPair x y
  have hlr := lowering_raising_isAdjointPair x y
  have happ : ∀ z : FockAlg,
      (Complex.I • (raising - lowering) : Module.End ℂ FockAlg) z =
        Complex.I • (raising z - lowering z) := fun z => rfl
  have hL : fockAlgebraForm ((Complex.I • (raising - lowering) :
        Module.End ℂ FockAlg) x) y =
      Complex.I *
        (fockAlgebraForm (raising x) y - fockAlgebraForm (lowering x) y) := by
    rw [happ, map_smul, map_sub]
    simp [smul_eq_mul]
  have hR : fockAlgebraForm x ((Complex.I • (raising - lowering) :
        Module.End ℂ FockAlg) y) =
      -Complex.I *
        (fockAlgebraForm x (raising y) - fockAlgebraForm x (lowering y)) := by
    rw [happ, LinearMap.map_smulₛₗ, map_sub]
    simp [Complex.conj_I, smul_eq_mul]
  rw [hL, hR, hrl, hlr]
  ring

/-- The strict-cutoff momentum quadrature of manuscript Eq. `eq:sg-quadratures`,
`Π̂ = i √(ω_b/2) (â† − â)`: the skew polynomial `i(â† − â)` is formed on the
full Fock space and sandwiched to the retained `N`-level space before the real
prefactor `√(ω_b/2)` is applied. -/
def sgCutoffPi (N : ℕ) (omega : ℝ) : Matrix (Fin N) (Fin N) ℂ :=
  ((Real.sqrt (omega / 2) : ℝ) : ℂ) •
    fockSandwichMatrix N (Complex.I • (raising - lowering))

/-- The cutoff momentum quadrature in the manuscript's displayed form
`Π̂ = (i √(ω_b/2)) · (sandwich of â† − â)`: the prefactor `i` may be moved
through the linear sandwich. -/
theorem sgCutoffPi_eq (N : ℕ) (omega : ℝ) :
    sgCutoffPi N omega =
      (Complex.I * (Real.sqrt (omega / 2) : ℝ)) •
        fockSandwichMatrix N (raising - lowering) := by
  rw [sgCutoffPi, map_smul, smul_smul, mul_comm]

/-- For a positive basis frequency `ω_b > 0`, the cutoff momentum quadrature
`Π̂` of Eq. `eq:sg-quadratures` is Hermitian: the sandwich of the formally
self-adjoint polynomial `i(â† − â)` is Hermitian and `√(ω_b/2)` is real. -/
theorem sgCutoffPi_isHermitian (N : ℕ) {omega : ℝ} (_homega : 0 < omega) :
    (sgCutoffPi N omega).IsHermitian :=
  (fockSandwichMatrix_isHermitian_of_isAdjointPair N _
    sgCutoffPi_base_isAdjointPair).smul (sgMat_ofReal_isSelfAdjoint _)

/-- The matrix cosine `cos A = (e^{iA} + e^{-iA})/2`, used for the onsite
Josephson term `cos φ̂_j` of manuscript Eq. `eq:sg-lattice` and the deformation
harmonics `cos φ̂_j`, `cos 2φ̂_j` of Eq. `eq:sg-deformations`. -/
def sgMatCos {d : ℕ} (A : Matrix (Fin d) (Fin d) ℂ) : Matrix (Fin d) (Fin d) ℂ :=
  ((1 : ℂ) / 2) •
    (NormedSpace.exp (Complex.I • A) + NormedSpace.exp ((-Complex.I) • A))

/-- The matrix cosine of a Hermitian matrix is Hermitian: conjugate-transposing
`e^{iA}` gives `e^{-iA}` when `Aᴴ = A`, and the symmetrized sum is fixed. -/
theorem sgMatCos_isHermitian {d : ℕ} {A : Matrix (Fin d) (Fin d) ℂ}
    (hA : A.IsHermitian) : (sgMatCos A).IsHermitian := by
  have hI : (Complex.I • A)ᴴ = (-Complex.I) • A := by
    rw [Matrix.conjTranspose_smul, hA.eq]
    congr 1
    simp
  have hI' : ((-Complex.I) • A)ᴴ = Complex.I • A := by
    rw [Matrix.conjTranspose_smul, hA.eq]
    congr 1
    simp
  show _ = _
  rw [sgMatCos, Matrix.conjTranspose_smul, Matrix.conjTranspose_add,
    ← Matrix.exp_conjTranspose, ← Matrix.exp_conjTranspose, hI, hI',
    add_comm (NormedSpace.exp ((-Complex.I) • A)) (NormedSpace.exp (Complex.I • A))]
  congr 1
  simp

end SingleMode

section Multimode

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- A finite sum of Hermitian matrices is Hermitian. -/
theorem sgMat_sum_isHermitian {m : Type*} {κ : Type*} (s : Finset κ)
    {M : κ → Matrix m m ℂ} (h : ∀ i ∈ s, (M i).IsHermitian) :
    (∑ i ∈ s, M i).IsHermitian := by
  show _ = _
  rw [Matrix.conjTranspose_sum]
  exact Finset.sum_congr rfl fun i hi => (h i hi).eq

/-- The finite-cutoff sine-Gordon lattice Hamiltonian of manuscript
Eq. `eq:sg-lattice`,

`H_SG = ∑_j [ Π̂_j²/2 + E_J (1 − cos φ̂_j) ] + (κ/2) ∑_j (φ̂_{j+1} − φ̂_j)²`,

on the multimode configuration space with every site truncated to `N` Fock
levels.  The site quadratures are the single-site lifts of the strict-cutoff
quadratures of Eq. `eq:sg-quadratures`.  Design note: the local cosine is
written as the single-site lift of the single-mode matrix cosine
`sgMatCos (sgCutoffPhi N omega)`; by `siteLift_matCos` this **equals** the
multimode matrix cosine of the lifted quadrature, the manuscript's onsite
`cos φ̂_j`.  The nearest-neighbor bond `j ↦ next j` abstracts the boundary
condition of the sum `∑_j (φ̂_{j+1} − φ̂_j)²`: `next = j + 1` cyclically
encodes the periodic lattice, and the manuscript's open chain with
`N_site − 1` physical links (Eq. `eq:sg-lattice`) is encoded by a bond map
that fixes the final site (`next j_end = j_end`), so that the extra bond term
`(φ̂_{next j_end} − φ̂_{j_end})²` vanishes identically. -/
def sgCutoffHamiltonian (N : ℕ) (next : ι → ι) (EJ kappa omega : ℝ) :
    Matrix (ι → Fin N) (ι → Fin N) ℂ :=
  ∑ j : ι,
      (((1 / 2 : ℝ) : ℂ) • siteLift j (sgCutoffPi N omega) ^ 2
        + (EJ : ℂ) • (1 - siteLift j (sgMatCos (sgCutoffPhi N omega))))
    + ((kappa / 2 : ℝ) : ℂ) •
        ∑ j : ι,
          (siteLift (next j) (sgCutoffPhi N omega)
            - siteLift j (sgCutoffPhi N omega)) ^ 2

/-- The finite-cutoff sine-Gordon lattice Hamiltonian is Hermitian for real
couplings `E_J`, `κ` and positive basis frequency `ω_b > 0` — the finite-cutoff
form of the manuscript classification Eq. `eq:sg-classification`
(`A_SG = iH_SG` with `H_SG` purely Hermitian).  Each kinetic term is the square
of a lifted Hermitian quadrature, each onsite term is a real multiple of
`1 − cos φ̂_j` with `cos φ̂_j` Hermitian, and each bond term is the square of a
difference of lifted Hermitian quadratures. -/
theorem sgCutoffHamiltonian_isHermitian (N : ℕ) (next : ι → ι)
    (EJ kappa : ℝ) {omega : ℝ} (homega : 0 < omega) :
    (sgCutoffHamiltonian (ι := ι) N next EJ kappa omega).IsHermitian := by
  have hPi := sgCutoffPi_isHermitian N homega
  have hPhi := sgCutoffPhi_isHermitian N homega
  have hCos := sgMatCos_isHermitian hPhi
  refine Matrix.IsHermitian.add ?_ ?_
  · refine sgMat_sum_isHermitian _ fun j _ => Matrix.IsHermitian.add ?_ ?_
    · exact ((siteLift_isHermitian j hPi).pow 2).smul
        (sgMat_ofReal_isSelfAdjoint _)
    · exact (Matrix.isHermitian_one.sub (siteLift_isHermitian j hCos)).smul
        (sgMat_ofReal_isSelfAdjoint EJ)
  · refine Matrix.IsHermitian.smul ?_ (sgMat_ofReal_isSelfAdjoint _)
    refine sgMat_sum_isHermitian _ fun j _ => ?_
    exact ((siteLift_isHermitian (next j) hPhi).sub
      (siteLift_isHermitian j hPhi)).pow 2

/-- The junction-disorder deformation of manuscript Eq. `eq:sg-deformations`,
`ΔH_dis = ∑_j δ_j (1 − cos φ̂_j)`, with a site-dependent real disorder profile
`δ : ι → ℝ`, on the multimode cutoff space. -/
def sgCutoffDeltaHdis (N : ℕ) (delta : ι → ℝ) (omega : ℝ) :
    Matrix (ι → Fin N) (ι → Fin N) ℂ :=
  ∑ j : ι, (delta j : ℂ) • (1 - siteLift j (sgMatCos (sgCutoffPhi N omega)))

/-- The junction-disorder deformation `ΔH_dis` of Eq. `eq:sg-deformations` is
Hermitian for any real disorder profile and positive basis frequency. -/
theorem sgCutoffDeltaHdis_isHermitian (N : ℕ) (delta : ι → ℝ) {omega : ℝ}
    (homega : 0 < omega) :
    (sgCutoffDeltaHdis (ι := ι) N delta omega).IsHermitian := by
  have hCos := sgMatCos_isHermitian (sgCutoffPhi_isHermitian N homega)
  refine sgMat_sum_isHermitian _ fun j _ => ?_
  exact (Matrix.isHermitian_one.sub (siteLift_isHermitian j hCos)).smul
    (sgMat_ofReal_isSelfAdjoint (delta j))

/-- The double sine-Gordon deformation of manuscript Eq. `eq:sg-deformations`,
`ΔH_2 = λ ∑_j (1 − cos 2φ̂_j)`, on the multimode cutoff space: the doubled
harmonic is the matrix cosine of `2φ̂`, lifted site by site. -/
def sgCutoffDeltaH2 (N : ℕ) (lambda omega : ℝ) :
    Matrix (ι → Fin N) (ι → Fin N) ℂ :=
  (lambda : ℂ) •
    ∑ j : ι, (1 - siteLift j (sgMatCos ((2 : ℂ) • sgCutoffPhi N omega)))

/-- The double sine-Gordon deformation `ΔH_2` of Eq. `eq:sg-deformations` is
Hermitian for real `λ` and positive basis frequency: `2φ̂` is Hermitian, hence
so is its matrix cosine and each lifted onsite term. -/
theorem sgCutoffDeltaH2_isHermitian (N : ℕ) (lambda : ℝ) {omega : ℝ}
    (homega : 0 < omega) :
    (sgCutoffDeltaH2 (ι := ι) N lambda omega).IsHermitian := by
  have h2 : IsSelfAdjoint (2 : ℂ) := by
    simp [IsSelfAdjoint]
  have hCos := sgMatCos_isHermitian
    ((sgCutoffPhi_isHermitian N homega).smul h2)
  refine Matrix.IsHermitian.smul ?_ (sgMat_ofReal_isSelfAdjoint lambda)
  refine sgMat_sum_isHermitian _ fun j _ => ?_
  exact Matrix.isHermitian_one.sub (siteLift_isHermitian j hCos)

/-- The double sine-Gordon target Hamiltonian of manuscript
Eq. `eq:sg-double-target`, `H_target = H_SG + λ ∑_j (1 − cos 2φ̂_j)`, on the
multimode cutoff space (the manuscript's benchmark instance has two sites; the
statement here is for any finite site set). -/
def sgCutoffHtarget (N : ℕ) (next : ι → ι) (EJ kappa lambda omega : ℝ) :
    Matrix (ι → Fin N) (ι → Fin N) ℂ :=
  sgCutoffHamiltonian N next EJ kappa omega + sgCutoffDeltaH2 N lambda omega

/-- The double sine-Gordon target Hamiltonian of Eq. `eq:sg-double-target` is
Hermitian for real couplings and positive basis frequency. -/
theorem sgCutoffHtarget_isHermitian (N : ℕ) (next : ι → ι)
    (EJ kappa lambda : ℝ) {omega : ℝ} (homega : 0 < omega) :
    (sgCutoffHtarget (ι := ι) N next EJ kappa lambda omega).IsHermitian :=
  (sgCutoffHamiltonian_isHermitian N next EJ kappa homega).add
    (sgCutoffDeltaH2_isHermitian N lambda homega)

end Multimode

section Unitaries

/-- The SNAP gate of manuscript Eq. `eq:df-snap-d`,
`SNAP(θ) = ∑_n e^{iθ_n} |n⟩⟨n|`, as a diagonal matrix on the truncated
oscillator space. -/
def sgSnap {d : ℕ} (theta : Fin d → ℝ) : Matrix (Fin d) (Fin d) ℂ :=
  Matrix.diagonal fun k => Complex.exp (Complex.I * theta k)

/-- The SNAP gate is unitary: each diagonal phase `e^{iθ_n}` is unimodular. -/
theorem sgSnap_mem_unitaryGroup {d : ℕ} (theta : Fin d → ℝ) :
    sgSnap theta ∈ Matrix.unitaryGroup (Fin d) ℂ := by
  rw [Matrix.mem_unitaryGroup_iff, Matrix.star_eq_conjTranspose, sgSnap,
    Matrix.diagonal_conjTranspose, Matrix.diagonal_mul_diagonal]
  ext i j
  rcases eq_or_ne i j with rfl | hij
  · rw [Matrix.diagonal_apply_eq, Matrix.one_apply_eq]
    simp only [Pi.star_apply]
    rw [show star (Complex.exp (Complex.I * theta i)) =
        Complex.exp ((starRingEnd ℂ) (Complex.I * theta i)) from
      (Complex.exp_conj _).symm, ← Complex.exp_add]
    have hzero : Complex.I * theta i + (starRingEnd ℂ) (Complex.I * theta i) = 0 := by
      simp [Complex.conj_ofReal]
    rw [hzero, Complex.exp_zero]
  · rw [Matrix.diagonal_apply_ne _ hij, Matrix.one_apply_ne hij]

/-- The compiled displacement-type factor: the matrix exponential of an
arbitrary generator.  For a skew-Hermitian generator `S` (in particular any
skew-Hermitian truncation of `αâ† − α*â`, Eq. `eq:df-snap-d`) it is unitary,
which is the content of `sgDisp_mem_unitaryGroup`. -/
def sgDisp {d : ℕ} (S : Matrix (Fin d) (Fin d) ℂ) : Matrix (Fin d) (Fin d) ℂ :=
  NormedSpace.exp S

/-- The exponential of a skew-Hermitian matrix (`Sᴴ = −S`) is unitary — the
truncated displacement gate `D(α) = exp(αâ† − α*â)` of Eq. `eq:df-snap-d` for
any skew-Hermitian truncation of its generator. -/
theorem sgDisp_mem_unitaryGroup {d : ℕ} {S : Matrix (Fin d) (Fin d) ℂ}
    (hS : Sᴴ = -S) : sgDisp S ∈ Matrix.unitaryGroup (Fin d) ℂ := by
  letI : NormedAlgebra ℚ (Matrix (Fin d) (Fin d) ℂ) :=
    NormedAlgebra.restrictScalars ℚ ℂ _
  exact exp_mem_unitary_of_mem_skewAdjoint
    (by rwa [skewAdjoint.mem_iff, Matrix.star_eq_conjTranspose])

/-- The single-step evolution `exp(−i t H)` generated by a Hermitian `H` is
unitary: the generator `(−i t) H` is skew-Hermitian. -/
theorem sgDisp_exp_neg_I_smul_mem_unitaryGroup {d : ℕ}
    {H : Matrix (Fin d) (Fin d) ℂ} (hH : H.IsHermitian) (t : ℝ) :
    NormedSpace.exp ((-Complex.I * (t : ℂ)) • H) ∈
      Matrix.unitaryGroup (Fin d) ℂ := by
  have hskew : ((-Complex.I * (t : ℂ)) • H)ᴴ = -((-Complex.I * (t : ℂ)) • H) := by
    rw [Matrix.conjTranspose_smul, hH.eq, ← neg_smul]
    congr 1
    simp [Complex.conj_ofReal]
  exact sgDisp_mem_unitaryGroup hskew

end Unitaries

section Spectral

/-- A unitary matrix is an `ℓ²`-operator-norm contraction: the C*-identity
gives `‖U‖² = ‖U†U‖ = ‖1‖`, and the identity operator has norm at most one.
No nonemptiness hypothesis is needed (for `d = 0` both sides degenerate). -/
theorem sgSpectral_norm_le_one_of_mem_unitaryGroup {d : ℕ}
    {U : Matrix (Fin d) (Fin d) ℂ}
    (hU : U ∈ Matrix.unitaryGroup (Fin d) ℂ) : ‖U‖ ≤ 1 := by
  have hstar : star U * U = 1 := (Unitary.mem_iff.mp hU).1
  have hsq : ‖U‖ * ‖U‖ = ‖(1 : Matrix (Fin d) (Fin d) ℂ)‖ := by
    rw [← CStarRing.norm_star_mul_self, hstar]
  have hone : ‖(1 : Matrix (Fin d) (Fin d) ℂ)‖ ≤ 1 := by
    rw [Matrix.cstar_norm_def, map_one]
    exact ContinuousLinearMap.norm_id_le
  nlinarith [norm_nonneg U, hsq, hone]

/-- The `L`-layer SNAP+displacement ansatz of manuscript
Eqs. `eq:snapd-ansatz` / `eq:df-ansatz`,
`U_SD(θ, α) = ∏_{ℓ=1}^{L} D(α_ℓ) SNAP(θ_ℓ)`, with the displacement of layer
`ℓ` given as the exponential of a generator `S ℓ`. -/
def sgSpectralAnsatz {d L : ℕ} (theta : Fin L → Fin d → ℝ)
    (S : Fin L → Matrix (Fin d) (Fin d) ℂ) : Matrix (Fin d) (Fin d) ℂ :=
  (List.ofFn fun ℓ => sgDisp (S ℓ) * sgSnap (theta ℓ)).prod

/-- The compiled SNAP+displacement ansatz of Eq. `eq:snapd-ansatz` is unitary
whenever every displacement generator is skew-Hermitian: each layer is a
product of two unitaries and the unitary group is closed under products. -/
theorem sgSpectralAnsatz_mem_unitaryGroup {d L : ℕ} (theta : Fin L → Fin d → ℝ)
    {S : Fin L → Matrix (Fin d) (Fin d) ℂ} (hS : ∀ ℓ, (S ℓ)ᴴ = -(S ℓ)) :
    sgSpectralAnsatz theta S ∈ Matrix.unitaryGroup (Fin d) ℂ := by
  refine Submonoid.list_prod_mem _ fun x hx => ?_
  rw [List.mem_ofFn] at hx
  obtain ⟨ℓ, rfl⟩ := hx
  exact mul_mem (sgDisp_mem_unitaryGroup (hS ℓ)) (sgSnap_mem_unitaryGroup (theta ℓ))

/-- The single-mode local double sine-Gordon deformation block generator of
Eqs. `eq:sg-deformations` / `eq:sg-double-target`,
`ΔH_loc = λ (1 − cos 2φ̂)`, on the truncated oscillator space: the generator
of the local target block `U_target^{(loc)} = exp(−i ΔH_loc Δt)`. -/
def sgSpectralDeltaH2Loc (N : ℕ) (lambda omega : ℝ) : Matrix (Fin N) (Fin N) ℂ :=
  (lambda : ℂ) • (1 - sgMatCos ((2 : ℂ) • sgCutoffPhi N omega))

/-- The local double sine-Gordon deformation generator is Hermitian for real
`λ` and positive basis frequency. -/
theorem sgSpectralDeltaH2Loc_isHermitian (N : ℕ) (lambda : ℝ) {omega : ℝ}
    (homega : 0 < omega) : (sgSpectralDeltaH2Loc N lambda omega).IsHermitian := by
  have h2 : IsSelfAdjoint (2 : ℂ) := by
    simp [IsSelfAdjoint]
  exact (Matrix.isHermitian_one.sub
    (sgMatCos_isHermitian ((sgCutoffPhi_isHermitian N homega).smul h2))).smul
      (sgMat_ofReal_isSelfAdjoint lambda)

/-- Manuscript Eq. `eq:sg-spectral-composition` for the concrete compiled
objects: with `U_target^{(loc)} = exp(−i ΔH_loc Δt)` the exact local double
sine-Gordon deformation block (Eqs. `eq:sg-deformations`,
`eq:sg-double-target`) and `U_comp = U_SD(θ, α)` the `L`-layer
SNAP+displacement ansatz of Eqs. `eq:snapd-ansatz` / `eq:df-ansatz`, and with
the global phase `ϑ_HS` fixed by the Hilbert–Schmidt overlap, `r` repetitions
obey the telescoping bound

`‖(U_target^{(loc)})^r − e^{i r ϑ_HS} U_comp^r‖₂ ≤ r ε₂`,

where `ε₂ = ‖U_target^{(loc)} − e^{i ϑ_HS} U_comp‖₂`.  Both contraction
premises of the telescoping lemma are proved, not assumed: the target block is
the exponential of `−i Δt` times a Hermitian generator, and the ansatz is a
product of SNAP and displacement unitaries. -/
theorem sgSpectral_composition_bound (N : ℕ) (lambda : ℝ) {omega : ℝ}
    (homega : 0 < omega) (dt thetaHS : ℝ) {L : ℕ}
    (theta : Fin L → Fin N → ℝ) {S : Fin L → Matrix (Fin N) (Fin N) ℂ}
    (hS : ∀ ℓ, (S ℓ)ᴴ = -(S ℓ)) (r : ℕ) :
    ‖NormedSpace.exp
        ((-Complex.I * (dt : ℂ)) • sgSpectralDeltaH2Loc N lambda omega) ^ r
      - Complex.exp (r * thetaHS * Complex.I) • sgSpectralAnsatz theta S ^ r‖
    ≤ r * ‖NormedSpace.exp
        ((-Complex.I * (dt : ℂ)) • sgSpectralDeltaH2Loc N lambda omega)
      - Complex.exp (thetaHS * Complex.I) • sgSpectralAnsatz theta S‖ := by
  apply norm_pow_sub_exp_smul_pow_le
  · exact sgSpectral_norm_le_one_of_mem_unitaryGroup
      (sgDisp_exp_neg_I_smul_mem_unitaryGroup
        (sgSpectralDeltaH2Loc_isHermitian N lambda homega) dt)
  · exact sgSpectral_norm_le_one_of_mem_unitaryGroup
      (sgSpectralAnsatz_mem_unitaryGroup theta hS)

end Spectral

section CosineLift

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The matrix cosine `cos A = (e^{iA} + e^{-iA})/2` on an arbitrary finite
index type — the multimode-configuration-space form of `sgMatCos`, which is
stated for `Fin d`-indexed matrices only. -/
def sgMatCosMulti {m : Type*} [Fintype m] [DecidableEq m]
    (A : Matrix m m ℂ) : Matrix m m ℂ :=
  ((1 : ℂ) / 2) •
    (NormedSpace.exp (Complex.I • A) + NormedSpace.exp ((-Complex.I) • A))

/-- On `Fin d`-indexed matrices the general-index matrix cosine coincides with
the single-mode `sgMatCos`. -/
theorem sgMatCosMulti_eq_sgMatCos {d : ℕ} (A : Matrix (Fin d) (Fin d) ℂ) :
    sgMatCosMulti A = sgMatCos A :=
  rfl

/-- The single-site lift commutes with the matrix cosine: the lifted
single-mode cosine `siteLift j (cos A)` **equals** the multimode matrix cosine
of the lifted operator, `cos (siteLift j A)`.  This is the identification of
the onsite terms `cos φ̂_j`, `cos 2φ̂_j` of Eqs. `eq:sg-lattice` /
`eq:sg-deformations` (defined in `sgCutoffHamiltonian`, `sgCutoffDeltaHdis`,
`sgCutoffDeltaH2` as lifted single-mode cosines) with the cosine of the lifted
site quadrature, proved in Lean via `siteLift_exp` rather than asserted as a
design convention. -/
theorem siteLift_matCos {N : ℕ} (j : ι) (A : Matrix (Fin N) (Fin N) ℂ) :
    siteLift j (sgMatCos A) = sgMatCosMulti (siteLift j A) := by
  simp only [sgMatCos, sgMatCosMulti, siteLift_smul, siteLift_add, siteLift_exp]

/-- The onsite Josephson cosine of the lattice Hamiltonian
Eq. `eq:sg-lattice` is the multimode matrix cosine of the lifted site
quadrature `φ̂_j`. -/
theorem sgCutoffHamiltonian_onsite_cos (N : ℕ) (omega : ℝ) (j : ι) :
    siteLift j (sgMatCos (sgCutoffPhi N omega)) =
      sgMatCosMulti (siteLift j (sgCutoffPhi N omega)) :=
  siteLift_matCos j _

end CosineLift

section Displacement

/-- Conjugate transposition of a strict finite Fock sandwich along a formal
adjoint pair: if `A` and `B` are adjoint to each other for the Fock
sesquilinear form, the sandwich of `A` conjugate-transposes to the sandwich of
`B`. -/
theorem sgSandwich_conjTranspose_of_isAdjointPair (N : ℕ)
    {A B : Module.End ℂ FockAlg}
    (hAB : LinearMap.IsAdjointPair fockAlgebraForm fockAlgebraForm A B) :
    (fockSandwichMatrix N A)ᴴ = fockSandwichMatrix N B := by
  ext i j
  rw [Matrix.conjTranspose_apply, fockSandwichMatrix_apply, fockSandwichMatrix_apply]
  have h := hAB (Finsupp.single i.val 1) (Finsupp.single j.val 1)
  rw [fockAlgebraForm_single_right, fockAlgebraForm_single_left] at h
  rw [h, star_star]

/-- The strict sandwich of the raising operator conjugate-transposes to the
sandwich of the lowering operator: `(â†)† = â` survives the strict cutoff. -/
theorem sgSandwich_raising_conjTranspose (N : ℕ) :
    (fockSandwichMatrix N raising)ᴴ = fockSandwichMatrix N lowering :=
  sgSandwich_conjTranspose_of_isAdjointPair N raising_lowering_isAdjointPair

/-- The strict sandwich of the lowering operator conjugate-transposes to the
sandwich of the raising operator: `â† = (â)†` survives the strict cutoff. -/
theorem sgSandwich_lowering_conjTranspose (N : ℕ) :
    (fockSandwichMatrix N lowering)ᴴ = fockSandwichMatrix N raising :=
  sgSandwich_conjTranspose_of_isAdjointPair N lowering_raising_isAdjointPair

/-- The truncated displacement generator of the manuscript gate
`D(α) = exp(αâ† − α*â)` (Eqs. `eq:snapd-ansatz`, `eq:df-snap-d`): the
amplitude `α` times the strict `N`-level sandwich of `â†` minus its conjugate
times the sandwich of `â`. -/
def sgDispGen (N : ℕ) (alpha : ℂ) : Matrix (Fin N) (Fin N) ℂ :=
  alpha • fockSandwichMatrix N raising -
    (starRingEnd ℂ) alpha • fockSandwichMatrix N lowering

/-- The truncated displacement generator `αâ† − α*â` is skew-Hermitian: the
sandwiches of `â†` and `â` are conjugate transposes of each other and the
amplitudes conjugate accordingly. -/
theorem sgDispGen_conjTranspose (N : ℕ) (alpha : ℂ) :
    (sgDispGen N alpha)ᴴ = -sgDispGen N alpha := by
  simp only [sgDispGen, Matrix.conjTranspose_sub, Matrix.conjTranspose_smul,
    sgSandwich_raising_conjTranspose, sgSandwich_lowering_conjTranspose,
    starRingEnd_apply, star_star, neg_sub]

/-- The truncated displacement gate `D(α) = exp(αâ† − α*â)` of
Eq. `eq:df-snap-d` is unitary: its concrete generator `sgDispGen N α` is
skew-Hermitian. -/
theorem sgDispGen_disp_mem_unitaryGroup (N : ℕ) (alpha : ℂ) :
    sgDisp (sgDispGen N alpha) ∈ Matrix.unitaryGroup (Fin N) ℂ :=
  sgDisp_mem_unitaryGroup (sgDispGen_conjTranspose N alpha)

/-- The `L`-layer SNAP+displacement ansatz of Eqs. `eq:snapd-ansatz` /
`eq:df-ansatz` with the concrete truncated displacement generators
`αℓ â† − αℓ* â` is unitary. -/
theorem sgSpectralAnsatz_disp_mem_unitaryGroup {N L : ℕ}
    (theta : Fin L → Fin N → ℝ) (alpha : Fin L → ℂ) :
    sgSpectralAnsatz theta (fun ℓ => sgDispGen N (alpha ℓ)) ∈
      Matrix.unitaryGroup (Fin N) ℂ :=
  sgSpectralAnsatz_mem_unitaryGroup theta fun ℓ => sgDispGen_conjTranspose N (alpha ℓ)

/-- Manuscript Eq. `eq:sg-spectral-composition` with the compiled ansatz given
literally as in Eq. `eq:snapd-ansatz`: the layers are
`D(α_ℓ) SNAP(θ_ℓ) = exp(α_ℓ â† − α_ℓ* â) SNAP(θ_ℓ)`, parametrized by complex
amplitudes `α : Fin L → ℂ` and phase lists `θ : Fin L → Fin N → ℝ`, and the
target block is the exact local double sine-Gordon deformation exponential.
Both contraction premises are proved for these concrete compiled blocks. -/
theorem sgSpectral_composition_bound_disp (N : ℕ) (lambda : ℝ) {omega : ℝ}
    (homega : 0 < omega) (dt thetaHS : ℝ) {L : ℕ}
    (theta : Fin L → Fin N → ℝ) (alpha : Fin L → ℂ) (r : ℕ) :
    ‖NormedSpace.exp
        ((-Complex.I * (dt : ℂ)) • sgSpectralDeltaH2Loc N lambda omega) ^ r
      - Complex.exp (r * thetaHS * Complex.I) •
          sgSpectralAnsatz theta (fun ℓ => sgDispGen N (alpha ℓ)) ^ r‖
    ≤ r * ‖NormedSpace.exp
        ((-Complex.I * (dt : ℂ)) • sgSpectralDeltaH2Loc N lambda omega)
      - Complex.exp (thetaHS * Complex.I) •
          sgSpectralAnsatz theta (fun ℓ => sgDispGen N (alpha ℓ))‖ :=
  sgSpectral_composition_bound N lambda homega dt thetaHS theta
    (fun ℓ => sgDispGen_conjTranspose N (alpha ℓ)) r

end Displacement

end

end Cvdv
