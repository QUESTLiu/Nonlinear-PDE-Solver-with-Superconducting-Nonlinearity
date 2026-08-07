import Cvdv.LLEDisplacement
import Mathlib.Data.Matrix.Basis

/-!
# Spectral compilation of one LLE number-controlled displacement

The first section proves the generic finite-matrix factorization under an
assumed decomposition `n = ∑ m, λ m • P m`.  The second section constructs
the strict-cutoff occupation projectors on `Fin n`, proves their exclusive
spectral identities, and specializes the compiler to the cutoff number matrix.
It also records gate counts in the projector-selective abstract gate basis.

The result below proves the finite-matrix product identity.  It does not assert
physical availability, calibration, timing, or noise performance of a
projector-selective displacement.
-/

namespace Cvdv

open scoped BigOperators Kronecker Matrix

noncomputable section

section

variable {ιm ιa ιs : Type*}
  [Fintype ιm] [DecidableEq ιm]
  [Fintype ιa] [DecidableEq ιa]
  [Fintype ιs] [DecidableEq ιs]

/-- A finite-matrix occupation projector is Hermitian and idempotent. -/
def IsLLEOccupationProjector (P : Matrix ιs ιs ℂ) : Prop :=
  P.IsHermitian ∧ P * P = P

/-- One gate in the abstract compilation basis: an ancillary displacement
selected by one weighted occupation projector. -/
def lleProjectorSelectiveDisplacement (b : Matrix ιa ιa ℂ)
    (P : Matrix ιs ιs ℂ) (weight τ : ℝ) : Matrix (ιa × ιs) (ιa × ιs) ℂ :=
  NormedSpace.exp
    (((-Complex.I * (τ : ℂ)) • lleAncillaX b) ⊗ₖ ((weight : ℂ) • P))

/-- Projector-selective gates commute when their system projectors commute. -/
theorem lle_projector_selective_displacements_commute
    (b : Matrix ιa ιa ℂ) (P Q : Matrix ιs ιs ℂ)
    (weight mu τ : ℝ) (hPQ : Commute P Q) :
    Commute (lleProjectorSelectiveDisplacement b P weight τ)
      (lleProjectorSelectiveDisplacement b Q mu τ) := by
  unfold lleProjectorSelectiveDisplacement
  exact (commute_kronecker_same_left
    ((-Complex.I * (τ : ℂ)) • lleAncillaX b)
    ((hPQ.smul_left (weight : ℂ)).smul_right (mu : ℂ))).exp

/-- Product of one projector-selective gate per spectral label. -/
def lleSpectralDisplacementCompilation (b : Matrix ιa ιa ℂ)
    (P : ιm → Matrix ιs ιs ℂ) (weight : ιm → ℝ) (τ : ℝ)
    (hcomm : ∀ m k, Commute (P m) (P k)) :
    Matrix (ιa × ιs) (ιa × ιs) ℂ :=
  Finset.univ.noncommProd
    (fun m => lleProjectorSelectiveDisplacement b (P m) (weight m) τ)
    (fun m _ k _ _ =>
      lle_projector_selective_displacements_commute b (P m) (P k)
        (weight m) (weight k) τ (hcomm m k))

/-- The unoptimized abstract factor count is one projector-selective displacement per label. -/
def lleSpectralDisplacementGateCount : ℕ :=
  Fintype.card ιm

/-- One abstract projector-selective factor is the corresponding operator-controlled
displacement in the notation of `Cvdv.LLEDisplacement`. -/
theorem lle_projector_selective_displacement_eq_controlledDisplacement
    (b : Matrix ιa ιa ℂ) (P : Matrix ιs ιs ℂ)
    (weight τ : ℝ) (hP : IsLLEOccupationProjector P) :
    lleProjectorSelectiveDisplacement b P weight τ =
      lleControlledDisplacement b
        (lleLossDisplacementParameter τ ((weight : ℂ) • P)) := by
  unfold lleProjectorSelectiveDisplacement
  exact lle_loss_site_exponential_eq_controlledDisplacement b
    ((weight : ℂ) • P) (hP.1.smul (by
      change star (weight : ℂ) = (weight : ℂ)
      exact Complex.conj_ofReal weight)) τ

omit [DecidableEq ιm] in
/-- Spectral compilation theorem for one number-controlled displacement.

The assumptions state that `P m` are commuting occupation projectors and that
their real-weighted sum is the system number matrix `n`. -/
theorem lle_spectral_compilation_eq_number_controlled_displacement
    (b : Matrix ιa ιa ℂ) (n : Matrix ιs ιs ℂ)
    (P : ιm → Matrix ιs ιs ℂ) (weight : ιm → ℝ) (τ : ℝ)
    (hP : ∀ m, IsLLEOccupationProjector (P m))
    (hcomm : ∀ m k, Commute (P m) (P k))
    (hn : n = ∑ m, ((weight m : ℝ) : ℂ) • P m) :
    lleSpectralDisplacementCompilation b P weight τ hcomm =
      lleControlledDisplacement b (lleLossDisplacementParameter τ n) := by
  have hweightedHermitian :
      ∀ m, (((weight m : ℝ) : ℂ) • P m).IsHermitian :=
    fun m => (hP m).1.smul (by
      change star (weight m : ℂ) = (weight m : ℂ)
      exact Complex.conj_ofReal (weight m))
  have hsumHermitian :
      (∑ m, (((weight m : ℝ) : ℂ) • P m)).IsHermitian :=
    (isSelfAdjoint_sum Finset.univ fun m _ =>
      (hweightedHermitian m).isSelfAdjoint).isHermitian
  have hweightedCommute :
      ∀ m k, Commute (((weight m : ℝ) : ℂ) • P m)
        (((weight k : ℝ) : ℂ) • P k) :=
    fun m k => ((hcomm m k).smul_left (weight m : ℂ)).smul_right (weight k : ℂ)
  have hfactor := lle_loss_exponential_factorization
    (x := lleAncillaX b)
    (n := fun m => (((weight m : ℝ) : ℂ) • P m))
    (c := -Complex.I * (τ : ℂ)) hweightedCommute
  calc
    lleSpectralDisplacementCompilation b P weight τ hcomm =
        NormedSpace.exp
          (((-Complex.I * (τ : ℂ)) • lleAncillaX b) ⊗ₖ
            (∑ m, (((weight m : ℝ) : ℂ) • P m))) := by
              simpa [lleSpectralDisplacementCompilation,
                lleProjectorSelectiveDisplacement] using hfactor.symm
    _ = NormedSpace.exp
          (((-Complex.I * (τ : ℂ)) • lleAncillaX b) ⊗ₖ n) := by rw [← hn]
    _ = lleControlledDisplacement b (lleLossDisplacementParameter τ n) :=
      lle_loss_site_exponential_eq_controlledDisplacement b n
        (hn.symm ▸ hsumHermitian) τ

end

section StrictCutoffSpectralCompilation

variable {ιa : Type*} [Fintype ιa] [DecidableEq ιa]

/-- The projector onto one occupation basis state in a strict `n`-level cutoff. -/
def lleCutoffOccupationProjector {n : ℕ} (m : Fin n) :
    Matrix (Fin n) (Fin n) ℂ :=
  Matrix.single m m 1

@[simp]
theorem lle_cutoff_occupation_projector_isHermitian {n : ℕ} (m : Fin n) :
    (lleCutoffOccupationProjector m).IsHermitian := by
  simp [lleCutoffOccupationProjector, Matrix.IsHermitian]

@[simp]
theorem lle_cutoff_occupation_projector_idempotent {n : ℕ} (m : Fin n) :
    lleCutoffOccupationProjector m * lleCutoffOccupationProjector m =
      lleCutoffOccupationProjector m := by
  simp [lleCutoffOccupationProjector]

theorem lle_cutoff_occupation_projector_isOccupationProjector {n : ℕ} (m : Fin n) :
    IsLLEOccupationProjector (lleCutoffOccupationProjector m) :=
  ⟨lle_cutoff_occupation_projector_isHermitian m,
    lle_cutoff_occupation_projector_idempotent m⟩

@[simp]
theorem lle_cutoff_occupation_projectors_orthogonal {n : ℕ} {m k : Fin n}
    (hmk : m ≠ k) :
    lleCutoffOccupationProjector m * lleCutoffOccupationProjector k = 0 := by
  simp [lleCutoffOccupationProjector, hmk]

/-- The product of two cutoff occupation projectors has exact exclusive-label semantics. -/
theorem lle_cutoff_occupation_projector_mul {n : ℕ} (m k : Fin n) :
    lleCutoffOccupationProjector m * lleCutoffOccupationProjector k =
      if m = k then lleCutoffOccupationProjector m else 0 := by
  by_cases hmk : m = k
  · subst k
    simp
  · simp [hmk]

theorem lle_cutoff_occupation_projectors_commute {n : ℕ} (m k : Fin n) :
    Commute (lleCutoffOccupationProjector m) (lleCutoffOccupationProjector k) := by
  rw [commute_iff_eq]
  rw [lle_cutoff_occupation_projector_mul, lle_cutoff_occupation_projector_mul]
  by_cases hmk : m = k
  · simp [hmk]
  · simp [hmk, Ne.symm hmk]

/-- The cutoff occupation projectors resolve the identity matrix. -/
theorem lle_cutoff_occupation_projectors_sum_eq_one (n : ℕ) :
    ∑ m : Fin n, lleCutoffOccupationProjector m = 1 := by
  simpa [lleCutoffOccupationProjector] using
    (Matrix.sum_single_one (m := Fin n) (α := ℂ))

/-- The strict-cutoff number matrix with spectrum `0, ..., n - 1`. -/
def lleCutoffNumberMatrix (n : ℕ) : Matrix (Fin n) (Fin n) ℂ :=
  Matrix.diagonal fun m ↦ (m.val : ℂ)

/-- Exact spectral decomposition of the strict-cutoff number matrix. -/
theorem lle_cutoff_number_matrix_spectral_decomposition (n : ℕ) :
    lleCutoffNumberMatrix n =
      ∑ m : Fin n, (((m.val : ℝ) : ℂ) • lleCutoffOccupationProjector m) := by
  symm
  calc
    ∑ m : Fin n, (((m.val : ℝ) : ℂ) • lleCutoffOccupationProjector m) =
        ∑ m : Fin n, Matrix.single m m (m.val : ℂ) := by
          apply Finset.sum_congr rfl
          intro m _
          simp [lleCutoffOccupationProjector]
    _ = Matrix.diagonal (fun m : Fin n ↦ (m.val : ℂ)) :=
      Matrix.sum_single_eq_diagonal _
    _ = lleCutoffNumberMatrix n := rfl

/-- Concrete strict-cutoff specialization of the spectral compiler. -/
theorem lle_cutoff_spectral_compilation_eq_number_controlled_displacement
    (b : Matrix ιa ιa ℂ) (n : ℕ) (τ : ℝ) :
    lleSpectralDisplacementCompilation b
        (fun m : Fin n ↦ lleCutoffOccupationProjector m)
        (fun m : Fin n ↦ (m.val : ℝ)) τ
        (fun m k ↦ lle_cutoff_occupation_projectors_commute m k) =
      lleControlledDisplacement b
        (lleLossDisplacementParameter τ (lleCutoffNumberMatrix n)) := by
  apply lle_spectral_compilation_eq_number_controlled_displacement
  · exact fun m ↦ lle_cutoff_occupation_projector_isOccupationProjector m
  · exact lle_cutoff_number_matrix_spectral_decomposition n

/-- The unoptimized strict-cutoff compiler emits one abstract factor per label. -/
theorem lle_cutoff_spectral_displacement_gateCount_eq (n : ℕ) :
    lleSpectralDisplacementGateCount (ιm := Fin n) = n := by
  simp [lleSpectralDisplacementGateCount]

/-- Occupation labels whose displacement weight is nonzero. -/
def lleNontrivialOccupationLabels (n : ℕ) : Finset (Fin n) :=
  Finset.univ.filter fun m ↦ m.val ≠ 0

/-- A zero-weight spectral factor is the identity and requires no abstract gate. -/
theorem lle_cutoff_projector_selective_displacement_of_zero_weight
    (b : Matrix ιa ιa ℂ) {n : ℕ} (m : Fin n) (τ : ℝ) (hm : m.val = 0) :
    lleProjectorSelectiveDisplacement b (lleCutoffOccupationProjector m) (m.val : ℝ) τ = 1 := by
  simp [lleProjectorSelectiveDisplacement, hm]

theorem lleNontrivialOccupationLabels_succ_eq_erase_zero (n : ℕ) :
    lleNontrivialOccupationLabels (n + 1) =
      Finset.univ.erase (0 : Fin (n + 1)) := by
  ext m
  simp only [lleNontrivialOccupationLabels, Finset.mem_filter, Finset.mem_univ,
    true_and, Finset.mem_erase]
  constructor
  · intro hmval
    refine ⟨?_, trivial⟩
    intro hmzero
    apply hmval
    simp [hmzero]
  · rintro ⟨hm, _⟩ hmval
    apply hm
    apply Fin.ext
    exact hmval

/-- Exactly `n - 1` cutoff occupation labels have nonzero displacement weight. -/
theorem card_lleNontrivialOccupationLabels (n : ℕ) :
    (lleNontrivialOccupationLabels n).card = n - 1 := by
  cases n with
  | zero => simp [lleNontrivialOccupationLabels]
  | succ n =>
      rw [lleNontrivialOccupationLabels_succ_eq_erase_zero,
        Finset.card_erase_of_mem (Finset.mem_univ 0)]
      simp

/-- Strict-cutoff compilation after omitting the zero-weight identity factor. -/
def lleOptimizedCutoffSpectralDisplacementCompilation
    (b : Matrix ιa ιa ℂ) (n : ℕ) (τ : ℝ) :
    Matrix (ιa × Fin n) (ιa × Fin n) ℂ :=
  (lleNontrivialOccupationLabels n).noncommProd
    (fun m ↦ lleProjectorSelectiveDisplacement b
      (lleCutoffOccupationProjector m) (m.val : ℝ) τ)
    (fun m _ k _ _ ↦
      lle_projector_selective_displacements_commute b
        (lleCutoffOccupationProjector m) (lleCutoffOccupationProjector k)
        (m.val : ℝ) (k.val : ℝ) τ
        (lle_cutoff_occupation_projectors_commute m k))

/-- Omitting the zero-weight identity factor leaves the compiled matrix unchanged. -/
theorem lle_optimized_cutoff_spectral_compilation_eq_unoptimized
    (b : Matrix ιa ιa ℂ) (n : ℕ) (τ : ℝ) :
    lleOptimizedCutoffSpectralDisplacementCompilation b n τ =
      lleSpectralDisplacementCompilation b
        (fun m : Fin n ↦ lleCutoffOccupationProjector m)
        (fun m : Fin n ↦ (m.val : ℝ)) τ
        (fun m k ↦ lle_cutoff_occupation_projectors_commute m k) := by
  cases n with
  | zero =>
      simp [lleOptimizedCutoffSpectralDisplacementCompilation,
        lleNontrivialOccupationLabels, lleSpectralDisplacementCompilation]
  | succ n =>
      let f : Fin (n + 1) → Matrix (ιa × Fin (n + 1)) (ιa × Fin (n + 1)) ℂ :=
        fun m ↦ lleProjectorSelectiveDisplacement b
          (lleCutoffOccupationProjector m) (m.val : ℝ) τ
      have herase := Finset.noncommProd_erase_mul
        (Finset.univ : Finset (Fin (n + 1))) (Finset.mem_univ (0 : Fin (n + 1))) f
        (fun m _ k _ _ ↦
          lle_projector_selective_displacements_commute b
            (lleCutoffOccupationProjector m) (lleCutoffOccupationProjector k)
            (m.val : ℝ) (k.val : ℝ) τ
            (lle_cutoff_occupation_projectors_commute m k))
      have hzero : f (0 : Fin (n + 1)) = 1 := by
        dsimp [f]
        exact lle_cutoff_projector_selective_displacement_of_zero_weight b 0 τ rfl
      rw [hzero, mul_one] at herase
      simpa [lleOptimizedCutoffSpectralDisplacementCompilation,
        lleNontrivialOccupationLabels_succ_eq_erase_zero,
        lleSpectralDisplacementCompilation, f] using herase

/-- The optimized exact compiler implements the strict-cutoff number-controlled displacement. -/
theorem lle_optimized_cutoff_spectral_compilation_eq_number_controlled_displacement
    (b : Matrix ιa ιa ℂ) (n : ℕ) (τ : ℝ) :
    lleOptimizedCutoffSpectralDisplacementCompilation b n τ =
      lleControlledDisplacement b
        (lleLossDisplacementParameter τ (lleCutoffNumberMatrix n)) := by
  rw [lle_optimized_cutoff_spectral_compilation_eq_unoptimized,
    lle_cutoff_spectral_compilation_eq_number_controlled_displacement]

/-- Projector-selective factors for all sites and product-formula layers before
omitting zero weight. -/
theorem lle_unoptimized_multisite_layer_gateCount
    (M n_t n : ℕ) :
    M * n_t * lleSpectralDisplacementGateCount (ιm := Fin n) = M * n_t * n := by
  rw [lle_cutoff_spectral_displacement_gateCount_eq]

/-- Nontrivial projector-selective factors for all sites and product-formula layers. -/
theorem lle_optimized_multisite_layer_gateCount
    (M n_t n : ℕ) :
    M * n_t * (lleNontrivialOccupationLabels n).card = M * n_t * (n - 1) := by
  rw [card_lleNontrivialOccupationLabels]

end StrictCutoffSpectralCompilation


end


end Cvdv
