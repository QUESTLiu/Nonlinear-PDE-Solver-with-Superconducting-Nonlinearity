import Cvdv.SiteLift
import Cvdv.LLEHardwareCompilation
import Cvdv.LLEFinite
import Mathlib.Algebra.Lie.OfAssociative
import Mathlib.Tactic.NoncommRing

/-!
# Multimode spectral compilation and the exact LCHS mixed commutator

This file lifts the single-mode strict-cutoff spectral compilation of
`Cvdv.LLEHardwareCompilation` to the full multimode system space using the
single-site lift of `Cvdv.SiteLift`.

* The full-system occupation projectors `P_{j,m}` of manuscript
  Sec. `subsec:lle-resources` (the paragraph after
  Eq. `eq:lle-number-controlled-displacements`) are realized as `lleMM_P j m`,
  with their exclusive-label products, the resolution of the identity, and the
  spectral decomposition of the lifted number matrix.
* The abstract projector-family compiler is instantiated at this family, so
  one system-number-controlled ancillary displacement factorizes into
  projector-selective displacements built from genuine full-system projectors,
  including the optimized variant that omits the `m = 0` identity factor
  (the `n_cut - 1` factor count of Eq. `eq:lle-resource-tuple`).
* The multisite loss layer of Eq. `eq:lle-number-controlled-displacements`
  factorizes into one number-controlled displacement per site, with the
  multimode number matrices `siteLift j (lleCutoffNumberMatrix n)`.
* The multimode cutoff ladder matrices `lleMM_lowering`/`lleMM_raising` obey
  the exact norm identity `‖siteLift j a‖ = ‖a‖ ≤ √n_cut`, giving the
  manuscript estimate `C_LCHS ≤ κ √n_cut ∑_j |F_j|` after
  Eq. `eq:lle-mixed-commutator` for the genuine multimode ladder family.
* The mixed commutator `[L_loss, H_F]` of Eq. `eq:lle-mixed-commutator` is
  proved to be *exactly* `finiteLLEMixedCommutator` for the concrete multimode
  cutoff ladders: the strict-cutoff number-ladder commutators
  `[n̄, ā] = -ā` and `[n̄, ā†] = ā†` hold exactly, so the number-conserving
  sectors of the Hamiltonian commute with the loss operator and only the
  coherent drive survives.

No claim is made here about hardware availability of a projector-selective
displacement, and no operator-valued LCHS integral is formalized.
-/

namespace Cvdv

open scoped BigOperators Kronecker Matrix Matrix.Norms.L2Operator

noncomputable section

section MultimodeProjectors

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {n : ℕ}

/-- The single-site lift commutes with finite sums of cutoff matrices. -/
theorem lleMM_siteLift_sum {α : Type*} (s : Finset α) (j : ι)
    (A : α → Matrix (Fin n) (Fin n) ℂ) :
    siteLift j (∑ m ∈ s, A m) = ∑ m ∈ s, siteLift j (A m) := by
  induction s using Finset.cons_induction with
  | empty => simp
  | cons a s ha ih => rw [Finset.sum_cons, Finset.sum_cons, siteLift_add, ih]

/-- The single-site lift commutes with negation. -/
theorem lleMM_siteLift_neg (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j (-A) = -siteLift j A := by
  have h := siteLift_sub j 0 A
  simpa using h

/-- The full-system projector `P_{j,m}` onto occupation `m` at site `j`, acting
as the identity on every other site (manuscript Sec. `subsec:lle-resources`). -/
def lleMM_P (j : ι) (m : Fin n) : Matrix (ι → Fin n) (ι → Fin n) ℂ :=
  siteLift j (lleCutoffOccupationProjector m)

/-- Each full-system occupation projector is Hermitian. -/
theorem lleMM_P_isHermitian (j : ι) (m : Fin n) : (lleMM_P j m).IsHermitian :=
  siteLift_isHermitian j (lle_cutoff_occupation_projector_isHermitian m)

/-- Exclusive-label products of the full-system occupation projectors:
`P_{j,m} P_{j,k} = δ_{mk} P_{j,m}` (manuscript Sec. `subsec:lle-resources`). -/
theorem lleMM_P_orthogonal (j : ι) (m k : Fin n) :
    lleMM_P j m * lleMM_P j k = if m = k then lleMM_P j m else 0 := by
  rw [lleMM_P, lleMM_P, siteLift_mul, lle_cutoff_occupation_projector_mul]
  by_cases hmk : m = k
  · rw [if_pos hmk, if_pos hmk]
  · rw [if_neg hmk, if_neg hmk, siteLift_zero]

/-- Each full-system occupation projector is idempotent. -/
theorem lleMM_P_idempotent (j : ι) (m : Fin n) :
    lleMM_P j m * lleMM_P j m = lleMM_P j m := by
  rw [lleMM_P_orthogonal, if_pos rfl]

/-- The full-system projectors satisfy the occupation-projector predicate of
the abstract compiler. -/
theorem lleMM_P_isOccupationProjector (j : ι) (m : Fin n) :
    IsLLEOccupationProjector (lleMM_P j m) :=
  ⟨lleMM_P_isHermitian j m, lleMM_P_idempotent j m⟩

/-- Same-site full-system occupation projectors commute. -/
theorem lleMM_P_commute (j : ι) (m k : Fin n) :
    Commute (lleMM_P j m) (lleMM_P j k) := by
  rw [commute_iff_eq, lleMM_P_orthogonal, lleMM_P_orthogonal]
  by_cases hmk : m = k
  · rw [if_pos hmk, if_pos hmk.symm, hmk]
  · rw [if_neg hmk, if_neg (Ne.symm hmk)]

/-- Resolution of the identity on the full system space:
`∑_m P_{j,m} = I_sys` (manuscript Sec. `subsec:lle-resources`). -/
theorem lleMM_P_sum_one (j : ι) : ∑ m : Fin n, lleMM_P j m = 1 := by
  have h := lleMM_siteLift_sum Finset.univ j
    fun m : Fin n => lleCutoffOccupationProjector m
  rw [lle_cutoff_occupation_projectors_sum_eq_one] at h
  simpa [lleMM_P] using h.symm

/-- Spectral decomposition of the lifted number matrix:
`n̂_j = ∑_m m P_{j,m}` with genuine full-system projectors
(manuscript Sec. `subsec:lle-resources`). -/
theorem lleMM_number_spectral_decomposition (j : ι) :
    siteLift j (lleCutoffNumberMatrix n) =
      ∑ m : Fin n, ((m.val : ℝ) : ℂ) • lleMM_P j m := by
  rw [lle_cutoff_number_matrix_spectral_decomposition, lleMM_siteLift_sum]
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [siteLift_smul, lleMM_P]

/-- The lifted number matrix is Hermitian. -/
theorem lleMM_cutoffNumberMatrix_isHermitian (n : ℕ) :
    (lleCutoffNumberMatrix n).IsHermitian := by
  rw [lleCutoffNumberMatrix, Matrix.IsHermitian, Matrix.diagonal_conjTranspose]
  congr 1
  funext m
  simp

/-- Lifted number matrices at arbitrary sites commute. -/
theorem lleMM_number_commute (j k : ι) :
    Commute (siteLift j (lleCutoffNumberMatrix n))
      (siteLift k (lleCutoffNumberMatrix n)) := by
  rcases eq_or_ne j k with rfl | hjk
  · exact Commute.refl _
  · exact (commute_iff_eq _ _).mpr (siteLift_comm hjk _ _)

end MultimodeProjectors

section MultimodeCompilation

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {n : ℕ}
  {ιa : Type*} [Fintype ιa] [DecidableEq ιa]

/-- Multimode spectral compilation: one system-number-controlled ancillary
displacement on the full system space factorizes into projector-selective
displacements over the genuine full-system projectors `P_{j,m}` (manuscript
Sec. `subsec:lle-resources`, unoptimized compiler with `n_cut` factors). -/
theorem lleMM_spectral_compilation_eq_number_controlled_displacement
    (b : Matrix ιa ιa ℂ) (j : ι) (τ : ℝ) :
    lleSpectralDisplacementCompilation b (fun m : Fin n => lleMM_P j m)
        (fun m : Fin n => (m.val : ℝ)) τ (fun m k => lleMM_P_commute j m k) =
      lleControlledDisplacement b
        (lleLossDisplacementParameter τ (siteLift j (lleCutoffNumberMatrix n))) := by
  apply lle_spectral_compilation_eq_number_controlled_displacement
  · exact fun m => lleMM_P_isOccupationProjector j m
  · exact lleMM_number_spectral_decomposition j

/-- A zero-weight multimode spectral factor is the identity gate. -/
theorem lleMM_projector_selective_displacement_of_zero_weight
    (b : Matrix ιa ιa ℂ) (j : ι) (m : Fin n) (τ : ℝ) (hm : m.val = 0) :
    lleProjectorSelectiveDisplacement b (lleMM_P j m) (m.val : ℝ) τ = 1 := by
  simp [lleProjectorSelectiveDisplacement, hm]

/-- The optimized multimode spectral compiler: one projector-selective
displacement per nontrivial occupation label, `n_cut - 1` abstract factors
per site as in Eq. `eq:lle-resource-tuple`. -/
def lleMM_optimizedSpectralDisplacementCompilation
    (b : Matrix ιa ιa ℂ) (j : ι) (τ : ℝ) :
    Matrix (ιa × (ι → Fin n)) (ιa × (ι → Fin n)) ℂ :=
  (lleNontrivialOccupationLabels n).noncommProd
    (fun m => lleProjectorSelectiveDisplacement b (lleMM_P j m) (m.val : ℝ) τ)
    (fun m _ k _ _ =>
      lle_projector_selective_displacements_commute b (lleMM_P j m) (lleMM_P j k)
        (m.val : ℝ) (k.val : ℝ) τ (lleMM_P_commute j m k))

/-- Omitting the `m = 0` identity factor leaves the compiled multimode matrix
unchanged. -/
theorem lleMM_optimized_spectral_compilation_eq_unoptimized
    (b : Matrix ιa ιa ℂ) (j : ι) (τ : ℝ) :
    lleMM_optimizedSpectralDisplacementCompilation (n := n) b j τ =
      lleSpectralDisplacementCompilation b (fun m : Fin n => lleMM_P j m)
        (fun m : Fin n => (m.val : ℝ)) τ (fun m k => lleMM_P_commute j m k) := by
  cases n with
  | zero =>
      simp [lleMM_optimizedSpectralDisplacementCompilation,
        lleNontrivialOccupationLabels, lleSpectralDisplacementCompilation]
  | succ N =>
      let f : Fin (N + 1) → Matrix (ιa × (ι → Fin (N + 1))) (ιa × (ι → Fin (N + 1))) ℂ :=
        fun m => lleProjectorSelectiveDisplacement b (lleMM_P j m) (m.val : ℝ) τ
      have herase := Finset.noncommProd_erase_mul
        (Finset.univ : Finset (Fin (N + 1))) (Finset.mem_univ (0 : Fin (N + 1))) f
        (fun m _ k _ _ =>
          lle_projector_selective_displacements_commute b
            (lleMM_P j m) (lleMM_P j k) (m.val : ℝ) (k.val : ℝ) τ
            (lleMM_P_commute j m k))
      have hzero : f (0 : Fin (N + 1)) = 1 :=
        lleMM_projector_selective_displacement_of_zero_weight b j 0 τ rfl
      rw [hzero, mul_one] at herase
      simpa [lleMM_optimizedSpectralDisplacementCompilation,
        lleNontrivialOccupationLabels_succ_eq_erase_zero,
        lleSpectralDisplacementCompilation, f] using herase

/-- The optimized multimode compiler implements the number-controlled
displacement of Eq. `eq:lle-number-controlled-displacements` exactly, with
`n_cut - 1` projector-selective factors built from full-system projectors. -/
theorem lleMM_optimized_spectral_compilation_eq_number_controlled_displacement
    (b : Matrix ιa ιa ℂ) (j : ι) (τ : ℝ) :
    lleMM_optimizedSpectralDisplacementCompilation (n := n) b j τ =
      lleControlledDisplacement b
        (lleLossDisplacementParameter τ (siteLift j (lleCutoffNumberMatrix n))) := by
  rw [lleMM_optimized_spectral_compilation_eq_unoptimized,
    lleMM_spectral_compilation_eq_number_controlled_displacement]

/-- Number-controlled displacements at arbitrary sites commute, since the
lifted number matrices commute. -/
theorem lleMM_controlled_displacements_commute
    (b : Matrix ιa ιa ℂ) (τ : ℝ) (j k : ι) :
    Commute
      (lleControlledDisplacement b
        (lleLossDisplacementParameter τ (siteLift j (lleCutoffNumberMatrix n))))
      (lleControlledDisplacement b
        (lleLossDisplacementParameter τ (siteLift k (lleCutoffNumberMatrix n)))) := by
  rw [← lle_loss_site_exponential_eq_controlledDisplacement b _
      (siteLift_isHermitian j (lleMM_cutoffNumberMatrix_isHermitian n)) τ,
    ← lle_loss_site_exponential_eq_controlledDisplacement b _
      (siteLift_isHermitian k (lleMM_cutoffNumberMatrix_isHermitian n)) τ]
  exact (commute_kronecker_same_left ((-Complex.I * (τ : ℂ)) • lleAncillaX b)
    (lleMM_number_commute j k)).exp

/-- Multisite loss-layer factorization of
Eq. `eq:lle-number-controlled-displacements`: the joint loss exponential over
the full multimode number operator splits exactly into one number-controlled
ancillary displacement per site. -/
theorem lleMM_loss_layer_factorization (b : Matrix ιa ιa ℂ) (τ : ℝ) :
    NormedSpace.exp (((-Complex.I * (τ : ℂ)) • lleAncillaX b) ⊗ₖ
        (∑ j : ι, siteLift j (lleCutoffNumberMatrix n))) =
      Finset.univ.noncommProd
        (fun j : ι => lleControlledDisplacement b
          (lleLossDisplacementParameter τ (siteLift j (lleCutoffNumberMatrix n))))
        (fun j _ k _ _ => lleMM_controlled_displacements_commute b τ j k) := by
  rw [lle_loss_exponential_factorization (lleAncillaX b)
    (fun j : ι => siteLift j (lleCutoffNumberMatrix n)) (-Complex.I * (τ : ℂ))
    (fun j k => lleMM_number_commute j k)]
  exact Finset.noncommProd_congr rfl
    (fun j _ => lle_loss_site_exponential_eq_controlledDisplacement b
      (siteLift j (lleCutoffNumberMatrix n))
      (siteLift_isHermitian j (lleMM_cutoffNumberMatrix_isHermitian n)) τ) _

end MultimodeCompilation

section MultimodeLadder

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The multimode strict-cutoff lowering matrix `ā_j`: the single-mode cutoff
lowering matrix acting at site `j`, identity elsewhere. -/
def lleMM_lowering (j : ι) (n : ℕ) : Matrix (ι → Fin n) (ι → Fin n) ℂ :=
  siteLift j (fockSandwichMatrix n lowering)

/-- The multimode strict-cutoff raising matrix `ā_j†`. -/
def lleMM_raising (j : ι) (n : ℕ) : Matrix (ι → Fin n) (ι → Fin n) ℂ :=
  siteLift j (fockSandwichMatrix n raising)

/-- The multimode raising matrix is the adjoint of the multimode lowering
matrix. -/
theorem lleMM_raising_eq_conjTranspose (j : ι) (n : ℕ) :
    lleMM_raising j n = (lleMM_lowering j n)ᴴ := by
  rw [lleMM_raising, lleMM_lowering, siteLift_conjTranspose,
    fockSandwichMatrix_raising_eq_lowering_conjTranspose]

/-- Multimode form of the manuscript bound `‖ā_j Π_sys‖ ≤ √n_cut`: the exact
norm preservation of the single-site lift transfers the single-mode cutoff
bound with no loss. -/
theorem lleMM_norm_lowering_le_sqrt_cutoff (j : ι) (n : ℕ) :
    ‖lleMM_lowering j n‖ ≤ Real.sqrt n := by
  rw [lleMM_lowering, siteLift_l2_opNorm]
  exact norm_fockSandwichMatrix_lowering_le_sqrt_cutoff n

/-- Multimode form of the manuscript bound `‖ā_j† Π_sys‖ ≤ √n_cut`. -/
theorem lleMM_norm_raising_le_sqrt_cutoff (j : ι) (n : ℕ) :
    ‖lleMM_raising j n‖ ≤ Real.sqrt n := by
  rw [lleMM_raising, siteLift_l2_opNorm]
  exact norm_fockSandwichMatrix_raising_le_sqrt_cutoff n

/-- The manuscript estimate `C_LCHS ≤ κ √n_cut ∑_j |F_j|` after
Eq. `eq:lle-mixed-commutator`, for the genuine multimode cutoff ladder
family. -/
theorem lleMM_norm_mixedCommutator_le_sqrt_cutoff
    (kappa : ℝ) (n : ℕ) (F : ι → ℂ) (hkappa : 0 ≤ kappa) :
    ‖finiteLLEMixedCommutator (fun j : ι => lleMM_lowering j n) kappa F‖ ≤
      kappa * Real.sqrt n * ∑ j, ‖F j‖ :=
  norm_finiteLLEMixedCommutator_le_sqrt_cutoff _ kappa n F hkappa
    fun j => lleMM_norm_lowering_le_sqrt_cutoff j n

end MultimodeLadder

section MixedCommutator

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Entrywise description of the single-mode strict-cutoff lowering matrix. -/
theorem lleMM_lowering_matrix_apply (n : ℕ) (i j : Fin n) :
    fockSandwichMatrix n lowering i j =
      if finRotate n i = j then ((Real.sqrt j.val : ℝ) : ℂ) else 0 := by
  rw [fockSandwichMatrix_lowering_eq_permMatrix_mul_diagonal, Matrix.mul_diagonal]
  simp only [PEquiv.toMatrix_apply, Equiv.toPEquiv_apply, Option.mem_def,
    Option.some.injEq, ite_mul, one_mul, zero_mul]

/-- Under the strict cutoff, `ā† ā` is exactly the diagonal number matrix
with spectrum `0, ..., n - 1`. -/
theorem lleMM_conjTranspose_mul_lowering (n : ℕ) :
    (fockSandwichMatrix n lowering)ᴴ * fockSandwichMatrix n lowering =
      lleCutoffNumberMatrix n := by
  have hPP : ((finRotate n).permMatrix ℂ)ᴴ * (finRotate n).permMatrix ℂ = 1 := by
    rw [Matrix.conjTranspose_permMatrix, ← Matrix.permMatrix_mul, mul_inv_cancel,
      Matrix.permMatrix_one]
  rw [fockSandwichMatrix_lowering_eq_permMatrix_mul_diagonal, Matrix.conjTranspose_mul,
    Matrix.mul_assoc, ← Matrix.mul_assoc (((finRotate n).permMatrix ℂ)ᴴ), hPP,
    Matrix.one_mul, Matrix.diagonal_conjTranspose, Matrix.diagonal_mul_diagonal,
    lleCutoffNumberMatrix]
  congr 1
  funext m
  rw [Pi.star_apply, Complex.star_def, Complex.conj_ofReal, ← Complex.ofReal_mul,
    Real.mul_self_sqrt (Nat.cast_nonneg _), Complex.ofReal_natCast]

/-- The exact strict-cutoff number-lowering commutator `[n̄, ā] = -ā`: the
top-projector defect cancels because the removed wraparound entry carries the
weight `√0 = 0`. -/
theorem lleMM_number_mul_lowering_sub (n : ℕ) :
    lleCutoffNumberMatrix n * fockSandwichMatrix n lowering -
      fockSandwichMatrix n lowering * lleCutoffNumberMatrix n =
      -fockSandwichMatrix n lowering := by
  ext i j
  simp only [Matrix.sub_apply, Matrix.neg_apply, lleCutoffNumberMatrix,
    Matrix.diagonal_mul, Matrix.mul_diagonal, lleMM_lowering_matrix_apply]
  by_cases h : finRotate n i = j
  · rw [if_pos h]
    by_cases hj0 : j.val = 0
    · rw [hj0]
      simp
    · haveI : NeZero n := j.neZero
      have hj : j ≠ 0 := by
        intro hcon
        apply hj0
        simp [hcon]
      have hi : i = (finRotate n).symm j :=
        ((finRotate n).apply_eq_iff_eq_symm_apply).mp h
      have hival : i.val = j.val - 1 := by
        rw [hi]
        exact coe_finRotate_symm_of_ne_zero hj
      have h1 : 1 ≤ j.val := Nat.one_le_iff_ne_zero.mpr hj0
      have hcast : (i.val : ℂ) = (j.val : ℂ) - 1 := by
        rw [hival, Nat.cast_sub h1, Nat.cast_one]
      rw [hcast]
      ring
  · rw [if_neg h]
    simp

/-- The exact strict-cutoff number-raising commutator `[n̄, ā†] = ā†`. -/
theorem lleMM_number_mul_raising_sub (n : ℕ) :
    lleCutoffNumberMatrix n * (fockSandwichMatrix n lowering)ᴴ -
      (fockSandwichMatrix n lowering)ᴴ * lleCutoffNumberMatrix n =
      (fockSandwichMatrix n lowering)ᴴ := by
  have h := congrArg Matrix.conjTranspose (lleMM_number_mul_lowering_sub n)
  rw [Matrix.conjTranspose_sub, Matrix.conjTranspose_neg, Matrix.conjTranspose_mul,
    Matrix.conjTranspose_mul, (lleMM_cutoffNumberMatrix_isHermitian n).eq] at h
  rw [← neg_sub, h, neg_neg]

/-- The total number matrix of the multimode ladder family is the sum of the
lifted single-mode number matrices. -/
theorem lleMM_totalNumber_eq (n : ℕ) :
    finiteLLETotalNumber (fun j : ι => lleMM_lowering j n) =
      ∑ j : ι, siteLift j (lleCutoffNumberMatrix n) := by
  simp only [finiteLLETotalNumber, lleMM_lowering]
  exact Finset.sum_congr rfl fun j _ => by
    rw [siteLift_conjTranspose, siteLift_mul, lleMM_conjTranspose_mul_lowering]

/-- The total number matrix decomposes over the genuine full-system occupation
projectors, `N_tot = ∑_j ∑_m m P_{j,m}`. -/
theorem lleMM_totalNumber_spectral_decomposition (n : ℕ) :
    finiteLLETotalNumber (fun j : ι => lleMM_lowering j n) =
      ∑ j : ι, ∑ m : Fin n, ((m.val : ℝ) : ℂ) • lleMM_P j m := by
  rw [lleMM_totalNumber_eq]
  exact Finset.sum_congr rfl fun j _ => lleMM_number_spectral_decomposition j

/-- Leibniz rule for the commutator bracket of an associative ring. -/
private theorem lleMM_lie_mul {α : Type*} [Ring α] (x y z : α) :
    ⁅x, y * z⁆ = ⁅x, y⁆ * z + y * ⁅x, z⁆ := by
  simp only [Ring.lie_def]
  noncomm_ring

/-- The ring commutator vanishes on the diagonal. -/
private theorem lleMM_lie_self {α : Type*} [Ring α] (x : α) : ⁅x, x⁆ = 0 := by
  rw [Ring.lie_def, sub_self]

/-- The ring commutator is additive in its second argument. -/
private theorem lleMM_lie_add {α : Type*} [Ring α] (x y z : α) :
    ⁅x, y + z⁆ = ⁅x, y⁆ + ⁅x, z⁆ := by
  simp only [Ring.lie_def]
  noncomm_ring

/-- The ring commutator is subtractive in its second argument. -/
private theorem lleMM_lie_sub {α : Type*} [Ring α] (x y z : α) :
    ⁅x, y - z⁆ = ⁅x, y⁆ - ⁅x, z⁆ := by
  simp only [Ring.lie_def]
  noncomm_ring

/-- The ring commutator distributes over finite sums in the first argument. -/
private theorem lleMM_sum_lie {α β : Type*} [Ring β] (s : Finset α)
    (f : α → β) (m : β) :
    ⁅∑ i ∈ s, f i, m⁆ = ∑ i ∈ s, ⁅f i, m⁆ := by
  simp only [Ring.lie_def, Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]

/-- The ring commutator distributes over finite sums in the second argument. -/
private theorem lleMM_lie_sum {α β : Type*} [Ring β] (s : Finset α)
    (x : β) (f : α → β) :
    ⁅x, ∑ i ∈ s, f i⁆ = ∑ i ∈ s, ⁅x, f i⁆ := by
  simp only [Ring.lie_def, Finset.mul_sum, Finset.sum_mul, ← Finset.sum_sub_distrib]

/-- The ring commutator commutes with scalars in the second argument. -/
private theorem lleMM_lie_smul {R β : Type*} [Ring β] [Monoid R]
    [DistribMulAction R β] [SMulCommClass R β β] [IsScalarTower R β β]
    (c : R) (x m : β) :
    ⁅x, c • m⁆ = c • ⁅x, m⁆ := by
  simp only [Ring.lie_def, mul_smul_comm, smul_mul_assoc, smul_sub]

/-- The ring commutator commutes with scalars in the first argument. -/
private theorem lleMM_smul_lie {R β : Type*} [Ring β] [Monoid R]
    [DistribMulAction R β] [SMulCommClass R β β] [IsScalarTower R β β]
    (c : R) (x m : β) :
    ⁅c • x, m⁆ = c • ⁅x, m⁆ := by
  simp only [Ring.lie_def, smul_mul_assoc, mul_smul_comm, smul_sub]

/-- The exact multimode number-lowering commutator `[N_tot, ā_j] = -ā_j`:
off-site number matrices commute with `ā_j` and the on-site commutator is the
exact single-mode identity. -/
theorem lleMM_totalNumber_lie_lowering (n : ℕ) (j : ι) :
    ⁅finiteLLETotalNumber (fun k : ι => lleMM_lowering k n), lleMM_lowering j n⁆ =
      -lleMM_lowering j n := by
  rw [lleMM_totalNumber_eq, lleMM_sum_lie,
    Fintype.sum_eq_single j fun k hkj => by
      rw [Ring.lie_def, lleMM_lowering, siteLift_comm hkj, sub_self]]
  rw [lleMM_lowering, Ring.lie_def, siteLift_mul, siteLift_mul, ← siteLift_sub,
    lleMM_number_mul_lowering_sub, lleMM_siteLift_neg]

/-- The exact multimode number-raising commutator `[N_tot, ā_j†] = ā_j†`. -/
theorem lleMM_totalNumber_lie_raising (n : ℕ) (j : ι) :
    ⁅finiteLLETotalNumber (fun k : ι => lleMM_lowering k n), (lleMM_lowering j n)ᴴ⁆ =
      (lleMM_lowering j n)ᴴ := by
  rw [lleMM_totalNumber_eq, lleMM_sum_lie,
    Fintype.sum_eq_single j fun k hkj => by
      rw [lleMM_lowering, siteLift_conjTranspose, Ring.lie_def, siteLift_comm hkj,
        sub_self]]
  rw [lleMM_lowering, siteLift_conjTranspose, Ring.lie_def, siteLift_mul, siteLift_mul,
    ← siteLift_sub, lleMM_number_mul_raising_sub]

section AbstractLadder

variable {σ : Type*} [Fintype σ] [DecidableEq σ]

omit [DecidableEq ι] in
/-- Abstract form of the manuscript identity Eq. `eq:lle-mixed-commutator`:
for any cutoff ladder family whose total number matrix satisfies the exact
commutators `[N_tot, a_j] = -a_j` and `[N_tot, a_j†] = a_j†`, the
loss--Hamiltonian commutator is exactly the mixed commutator
`[L_loss, H_F] = i (κ/2) ∑_j (F_j a_j† + F_j* a_j)`: the detuning, dispersion,
and Kerr sectors are number conserving and drop out, and only the coherent
drive survives. -/
theorem lleMM_loss_lie_hamiltonian_of_ladder
    (a : ι → Matrix σ σ ℂ) (next : ι → ι)
    (delta dispersion g kappa : ℝ) (F : ι → ℂ)
    (hlow : ∀ j, ⁅finiteLLETotalNumber a, a j⁆ = -a j)
    (hraise : ∀ j, ⁅finiteLLETotalNumber a, (a j)ᴴ⁆ = (a j)ᴴ) :
    ⁅finiteLLELoss a kappa, finiteLLEHamiltonian a next delta dispersion g F⁆ =
      finiteLLEMixedCommutator a kappa F := by
  have hdisp : ⁅finiteLLETotalNumber a,
      ∑ j, (a (next j) - a j)ᴴ * (a (next j) - a j)⁆ = 0 := by
    rw [lleMM_lie_sum]
    refine Finset.sum_eq_zero fun j _ => ?_
    rw [lleMM_lie_mul, Matrix.conjTranspose_sub, lleMM_lie_sub, lleMM_lie_sub,
      hraise (next j), hraise j, hlow (next j), hlow j]
    noncomm_ring
  have hkerr : ⁅finiteLLETotalNumber a, ∑ j, ((a j) ^ 2)ᴴ * (a j) ^ 2⁆ = 0 := by
    rw [lleMM_lie_sum]
    refine Finset.sum_eq_zero fun j _ => ?_
    rw [pow_two, Matrix.conjTranspose_mul, lleMM_lie_mul, lleMM_lie_mul, lleMM_lie_mul,
      hlow j, hraise j]
    noncomm_ring
  have hdrive : ⁅finiteLLETotalNumber a, finiteLLEDrive a F⁆ =
      Complex.I • ∑ j, (F j • (a j)ᴴ + (starRingEnd ℂ) (F j) • a j) := by
    rw [finiteLLEDrive, lleMM_lie_smul, lleMM_lie_sum]
    congr 1
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [lleMM_lie_sub, lleMM_lie_smul, lleMM_lie_smul, hlow j, hraise j, smul_neg,
      sub_neg_eq_add]
  rw [finiteLLELoss, lleMM_smul_lie, finiteLLEHamiltonian, lleMM_lie_add, lleMM_lie_add,
    lleMM_lie_add, lleMM_lie_smul, lleMM_lie_smul, lleMM_lie_smul, lleMM_lie_self, hdisp,
    hkerr, hdrive, smul_zero, smul_zero, smul_zero, zero_add, zero_add, zero_add,
    finiteLLEMixedCommutator, ← smul_assoc]
  rw [show ((kappa / 2 : ℝ) • Complex.I : ℂ) = Complex.I * ((kappa / 2 : ℝ) : ℂ) by
    rw [Complex.real_smul, mul_comm]]

end AbstractLadder

/-- The `C_LCHS` bridge for the concrete multimode cutoff ladders (manuscript
Eq. `eq:lle-mixed-commutator`): the loss--Hamiltonian commutator bracket of the
finite LLE model built from `lleMM_lowering` is exactly the closed-form mixed
commutator. -/
theorem lleMM_loss_lie_hamiltonian_eq_mixedCommutator
    (n : ℕ) (next : ι → ι) (delta dispersion g kappa : ℝ) (F : ι → ℂ) :
    ⁅finiteLLELoss (fun j : ι => lleMM_lowering j n) kappa,
      finiteLLEHamiltonian (fun j : ι => lleMM_lowering j n)
        next delta dispersion g F⁆ =
      finiteLLEMixedCommutator (fun j : ι => lleMM_lowering j n) kappa F :=
  lleMM_loss_lie_hamiltonian_of_ladder _ next delta dispersion g kappa F
    (fun j => lleMM_totalNumber_lie_lowering n j)
    (fun j => lleMM_totalNumber_lie_raising n j)

/-- Product form of the exact `C_LCHS` identity:
`L_loss H_F - H_F L_loss = i (κ/2) ∑_j (F_j ā_j† + F_j* ā_j)` for the concrete
multimode cutoff ladders. -/
theorem lleMM_loss_hamiltonian_commutator_eq_mixedCommutator
    (n : ℕ) (next : ι → ι) (delta dispersion g kappa : ℝ) (F : ι → ℂ) :
    finiteLLELoss (fun j : ι => lleMM_lowering j n) kappa *
        finiteLLEHamiltonian (fun j : ι => lleMM_lowering j n)
          next delta dispersion g F -
      finiteLLEHamiltonian (fun j : ι => lleMM_lowering j n)
          next delta dispersion g F *
        finiteLLELoss (fun j : ι => lleMM_lowering j n) kappa =
      finiteLLEMixedCommutator (fun j : ι => lleMM_lowering j n) kappa F := by
  rw [← Ring.lie_def]
  exact lleMM_loss_lie_hamiltonian_eq_mixedCommutator n next delta dispersion g kappa F

/-- The manuscript estimate `C_LCHS ≤ κ √n_cut ∑_j |F_j|` applied to the actual
loss--Hamiltonian commutator of the concrete multimode model, not merely to the
closed-form surrogate. -/
theorem lleMM_norm_loss_hamiltonian_commutator_le
    (n : ℕ) (next : ι → ι) (delta dispersion g kappa : ℝ) (F : ι → ℂ)
    (hkappa : 0 ≤ kappa) :
    ‖finiteLLELoss (fun j : ι => lleMM_lowering j n) kappa *
        finiteLLEHamiltonian (fun j : ι => lleMM_lowering j n)
          next delta dispersion g F -
      finiteLLEHamiltonian (fun j : ι => lleMM_lowering j n)
          next delta dispersion g F *
        finiteLLELoss (fun j : ι => lleMM_lowering j n) kappa‖ ≤
      kappa * Real.sqrt n * ∑ j, ‖F j‖ := by
  rw [lleMM_loss_hamiltonian_commutator_eq_mixedCommutator]
  exact lleMM_norm_mixedCommutator_le_sqrt_cutoff kappa n F hkappa

end MixedCommutator

section ManuscriptParameter

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {n : ℕ}
  {ιa : Type*} [Fintype ιa] [DecidableEq ιa]

/-- At the manuscript identification `τ = κ Δt / 2`, the displacement parameter
of Eq. (eq:lle-number-controlled-displacements) takes literally the displayed
form `-i κ Δt/(2√2) · n̂` (main.tex 2053-2056). -/
theorem lleLossDisplacementParameter_kappa_deltaT {σ : Type*}
    (kappa deltaT : ℝ) (m : Matrix σ σ ℂ) :
    lleLossDisplacementParameter (kappa * deltaT / 2) m =
      (-Complex.I * ((kappa * deltaT / (2 * Real.sqrt 2) : ℝ) : ℂ)) • m := by
  rw [lleLossDisplacementParameter, div_div]

/-- Number-controlled displacements carrying the literal manuscript parameter
`-i κ Δt/(2√2) n̂_j` commute across sites. -/
theorem lleMM_controlled_displacements_commute_kappa_deltaT
    (b : Matrix ιa ιa ℂ) (kappa deltaT : ℝ) (j k : ι) :
    Commute
      (lleControlledDisplacement b
        ((-Complex.I * ((kappa * deltaT / (2 * Real.sqrt 2) : ℝ) : ℂ)) •
          siteLift j (lleCutoffNumberMatrix n)))
      (lleControlledDisplacement b
        ((-Complex.I * ((kappa * deltaT / (2 * Real.sqrt 2) : ℝ) : ℂ)) •
          siteLift k (lleCutoffNumberMatrix n))) := by
  simp only [← lleLossDisplacementParameter_kappa_deltaT]
  exact lleMM_controlled_displacements_commute b (kappa * deltaT / 2) j k

/-- The optimized multimode spectral compiler at `τ = κ Δt / 2`: the compiled
gate is the number-controlled displacement whose parameter is literally the
manuscript's `-i κ Δt/(2√2) n̂_j` (main.tex 2053-2056). -/
theorem lleMM_optimized_spectral_compilation_kappa_deltaT
    (b : Matrix ιa ιa ℂ) (j : ι) (kappa deltaT : ℝ) :
    lleMM_optimizedSpectralDisplacementCompilation (n := n) b j (kappa * deltaT / 2) =
      lleControlledDisplacement b
        ((-Complex.I * ((kappa * deltaT / (2 * Real.sqrt 2) : ℝ) : ℂ)) •
          siteLift j (lleCutoffNumberMatrix n)) := by
  rw [lleMM_optimized_spectral_compilation_eq_number_controlled_displacement,
    lleLossDisplacementParameter_kappa_deltaT]

/-- Multisite loss-layer factorization at `τ = κ Δt / 2`: every displacement
factor carries the literal manuscript parameter `-i κ Δt/(2√2) n̂_j`. -/
theorem lleMM_loss_layer_factorization_kappa_deltaT
    (b : Matrix ιa ιa ℂ) (kappa deltaT : ℝ) :
    NormedSpace.exp
        (((-Complex.I * ((kappa * deltaT / 2 : ℝ) : ℂ)) • lleAncillaX b) ⊗ₖ
          (∑ j : ι, siteLift j (lleCutoffNumberMatrix n))) =
      Finset.univ.noncommProd
        (fun j : ι => lleControlledDisplacement b
          ((-Complex.I * ((kappa * deltaT / (2 * Real.sqrt 2) : ℝ) : ℂ)) •
            siteLift j (lleCutoffNumberMatrix n)))
        (fun j _ k _ _ =>
          lleMM_controlled_displacements_commute_kappa_deltaT b kappa deltaT j k) := by
  rw [lleMM_loss_layer_factorization b (kappa * deltaT / 2)]
  exact Finset.noncommProd_congr rfl
    (fun j _ => by rw [lleLossDisplacementParameter_kappa_deltaT]) _

/-- Fully literal form of Eq. (eq:lle-number-controlled-displacements):
`e^{-i Δt x_a ⊗ L_loss} = ∏_j D_a(-i κ Δt/(2√2) n̂_j)` for the concrete
multimode cutoff ladders, with `L_loss = (κ/2) N_tot` the finite loss
operator itself on the left-hand side. -/
theorem lleMM_loss_exponential_eq_controlled_displacements
    (b : Matrix ιa ιa ℂ) (kappa deltaT : ℝ) :
    NormedSpace.exp ((-Complex.I * (deltaT : ℂ)) •
        (lleAncillaX b ⊗ₖ finiteLLELoss (fun j : ι => lleMM_lowering j n) kappa)) =
      Finset.univ.noncommProd
        (fun j : ι => lleControlledDisplacement b
          ((-Complex.I * ((kappa * deltaT / (2 * Real.sqrt 2) : ℝ) : ℂ)) •
            siteLift j (lleCutoffNumberMatrix n)))
        (fun j _ k _ _ =>
          lleMM_controlled_displacements_commute_kappa_deltaT b kappa deltaT j k) := by
  have harg : (-Complex.I * (deltaT : ℂ)) •
      (lleAncillaX b ⊗ₖ finiteLLELoss (fun j : ι => lleMM_lowering j n) kappa) =
    ((-Complex.I * ((kappa * deltaT / 2 : ℝ) : ℂ)) • lleAncillaX b) ⊗ₖ
      (∑ j : ι, siteLift j (lleCutoffNumberMatrix n)) := by
    rw [finiteLLELoss, lleMM_totalNumber_eq]
    ext ⟨ia, is⟩ ⟨ja, js⟩
    simp only [Matrix.smul_apply, Matrix.kronecker_apply, smul_eq_mul,
      Complex.real_smul]
    push_cast
    ring
  rw [harg]
  exact lleMM_loss_layer_factorization_kappa_deltaT b kappa deltaT

end ManuscriptParameter

end

end Cvdv
