import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.Normed.Algebra.MatrixExponential
import Mathlib.Data.Matrix.Block
import Mathlib.LinearAlgebra.Matrix.Hermitian
import Mathlib.LinearAlgebra.Matrix.Reindex
import Mathlib.Topology.Algebra.Module.FiniteDimension

/-!
# Single-site operator lifts on the finite multimode cutoff space

Shared infrastructure for the finite lattice models of the manuscript: the
sine-Gordon lattice Hamiltonian (Sec. `sec:sg`, Eq. `eq:sg-lattice`), the
Lugiato--Lefever native Hamiltonian and ladder operators (Sec. `sec:lle`,
Eq. `eq:lle-native-hamiltonian`), and the multimode cutoff projector
Eq. `eq:lle-system-projector`.  Each of these places a single-mode cutoff
operator on one site `j` of a finite lattice, acting as the identity on all
other sites.  With every mode truncated to `n` Fock levels
(`eq:lle-system-projector`), the multimode space is indexed by occupation
configurations `f : ι → Fin n`, and the lift of a single-mode matrix `A`
at site `j` is the matrix `siteLift j A`.

The file proves that `siteLift j` is a unital star-algebra embedding
(additive, multiplicative, unit-, adjoint- and Hermiticity-preserving), that
lifts at distinct sites commute, and that the lift preserves the `ℓ²`
operator norm exactly, so single-site cutoff norm bounds such as
`‖a_j‖ ≤ √n_cut` transfer verbatim to the multimode space.  The route to the
norm statement is the block-diagonal decomposition of the configuration space
along the off-site configurations.
-/

namespace Cvdv

open scoped BigOperators Matrix Matrix.Norms.L2Operator

noncomputable section

section SiteLift

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {n : ℕ}

/-- The lift of a single-mode cutoff matrix `A` onto the multimode configuration
space, acting as `A` on the mode at site `j` and as the identity on all other
sites. -/
def siteLift (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) :
    Matrix (ι → Fin n) (ι → Fin n) ℂ :=
  Matrix.of fun f g => if ∀ k, k ≠ j → f k = g k then A (f j) (g j) else 0

/-- Entrywise description of the single-site lift. -/
theorem siteLift_apply (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) (f g : ι → Fin n) :
    siteLift j A f g = if ∀ k, k ≠ j → f k = g k then A (f j) (g j) else 0 :=
  rfl

/-- The single-site lift of the zero matrix is zero. -/
@[simp]
theorem siteLift_zero (j : ι) :
    siteLift j (0 : Matrix (Fin n) (Fin n) ℂ) = 0 := by
  ext f g
  by_cases h : ∀ k, k ≠ j → f k = g k <;> simp [siteLift_apply, h]

/-- The single-site lift is additive. -/
theorem siteLift_add (j : ι) (A B : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j (A + B) = siteLift j A + siteLift j B := by
  ext f g
  simp only [siteLift_apply, Matrix.add_apply]
  by_cases h : ∀ k, k ≠ j → f k = g k
  · rw [if_pos h, if_pos h, if_pos h]
  · rw [if_neg h, if_neg h, if_neg h, add_zero]

/-- The single-site lift commutes with complex scalar multiplication. -/
theorem siteLift_smul (j : ι) (c : ℂ) (A : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j (c • A) = c • siteLift j A := by
  ext f g
  by_cases h : ∀ k, k ≠ j → f k = g k <;>
    simp [siteLift_apply, h, Matrix.smul_apply]

/-- The single-site lift preserves differences. -/
theorem siteLift_sub (j : ι) (A B : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j (A - B) = siteLift j A - siteLift j B := by
  ext f g
  simp only [siteLift_apply, Matrix.sub_apply]
  by_cases h : ∀ k, k ≠ j → f k = g k
  · rw [if_pos h, if_pos h, if_pos h]
  · rw [if_neg h, if_neg h, if_neg h, sub_zero]

/-- Block-diagonal normal form of the single-site lift: splitting each
configuration into its value at site `j` and its off-site restriction, the
lift is the block-diagonal matrix with one copy of `A` per off-site
configuration. -/
theorem siteLift_eq_reindex_blockDiagonal (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j A =
      Matrix.reindex (Equiv.funSplitAt j (Fin n)).symm (Equiv.funSplitAt j (Fin n)).symm
        (Matrix.blockDiagonal fun _ : {k : ι // k ≠ j} → Fin n => A) := by
  ext f g
  have hcond : ((fun k : {k : ι // k ≠ j} => f k.1) = fun k : {k : ι // k ≠ j} => g k.1) ↔
      ∀ k, k ≠ j → f k = g k := by
    rw [funext_iff, Subtype.forall]
  simp only [Matrix.reindex_apply, Matrix.submatrix_apply, Equiv.symm_symm,
    Matrix.blockDiagonal_apply, Equiv.funSplitAt_apply, siteLift_apply]
  by_cases h : ∀ k, k ≠ j → f k = g k
  · rw [if_pos h, if_pos (hcond.mpr h)]
  · rw [if_neg h, if_neg fun hc => h (hcond.mp hc)]

/-- The single-site lift of the identity is the identity. -/
@[simp]
theorem siteLift_one (j : ι) :
    siteLift j (1 : Matrix (Fin n) (Fin n) ℂ) = 1 := by
  rw [siteLift_eq_reindex_blockDiagonal, Matrix.reindex_apply]
  have h1 : (fun _ : {k : ι // k ≠ j} → Fin n => (1 : Matrix (Fin n) (Fin n) ℂ)) =
      (1 : ({k : ι // k ≠ j} → Fin n) → Matrix (Fin n) (Fin n) ℂ) := rfl
  rw [h1, Matrix.blockDiagonal_one, Equiv.symm_symm, Matrix.submatrix_one_equiv]

/-- Same-site lifts multiply as the lifted product: the off-site delta
constraints collapse the configuration sum. -/
theorem siteLift_mul (j : ι) (A B : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j A * siteLift j B = siteLift j (A * B) := by
  simp only [siteLift_eq_reindex_blockDiagonal, Matrix.reindex_apply, Equiv.symm_symm,
    Matrix.submatrix_mul_equiv, ← Matrix.blockDiagonal_mul]

/-- The single-site lift intertwines the conjugate transpose. -/
theorem siteLift_conjTranspose (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) :
    (siteLift j A)ᴴ = siteLift j Aᴴ := by
  ext f g
  simp only [Matrix.conjTranspose_apply, siteLift_apply]
  by_cases h : ∀ k, k ≠ j → f k = g k
  · rw [if_pos h, if_pos fun k hk => (h k hk).symm]
  · rw [if_neg h, if_neg fun hc => h fun k hk => (hc k hk).symm, star_zero]

/-- The single-site lift of a Hermitian matrix is Hermitian. -/
theorem siteLift_isHermitian (j : ι) {A : Matrix (Fin n) (Fin n) ℂ}
    (hA : A.IsHermitian) : (siteLift j A).IsHermitian := by
  change (siteLift j A)ᴴ = siteLift j A
  rw [siteLift_conjTranspose, hA.eq]

/-- Closed form of the product of lifts at two distinct sites: a single
double-site matrix element survives the configuration sum. -/
theorem siteLift_mul_siteLift_of_ne {j k : ι} (hjk : j ≠ k)
    (A B : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j A * siteLift k B =
      Matrix.of fun f g =>
        if ∀ l, l ≠ j → l ≠ k → f l = g l then A (f j) (g j) * B (f k) (g k)
        else 0 := by
  ext f g
  rw [Matrix.mul_apply, Matrix.of_apply]
  by_cases hc : ∀ l, l ≠ j → l ≠ k → f l = g l
  · rw [if_pos hc]
    rw [Finset.sum_eq_single (Function.update g k (f k))]
    · have h1 : ∀ l, l ≠ j → f l = Function.update g k (f k) l := by
        intro l hl
        by_cases hlk : l = k
        · rw [hlk, Function.update_self]
        · rw [Function.update_of_ne hlk]
          exact hc l hl hlk
      have h2 : ∀ l, l ≠ k → Function.update g k (f k) l = g l := fun l hl =>
        Function.update_of_ne hl _ _
      rw [siteLift_apply, siteLift_apply, if_pos h1, if_pos h2,
        Function.update_of_ne hjk, Function.update_self]
    · intro h _ hne
      rw [siteLift_apply, siteLift_apply]
      by_cases h1 : ∀ l, l ≠ j → f l = h l
      · by_cases h2 : ∀ l, l ≠ k → h l = g l
        · exact absurd (funext fun l => by
            by_cases hlk : l = k
            · rw [hlk, Function.update_self]
              exact (h1 k (Ne.symm hjk)).symm
            · rw [Function.update_of_ne hlk]
              exact h2 l hlk) hne
        · rw [if_neg h2, mul_zero]
      · rw [if_neg h1, zero_mul]
    · intro habs
      exact absurd (Finset.mem_univ _) habs
  · rw [if_neg hc]
    refine Finset.sum_eq_zero fun h _ => ?_
    rw [siteLift_apply, siteLift_apply]
    by_cases h1 : ∀ l, l ≠ j → f l = h l
    · by_cases h2 : ∀ l, l ≠ k → h l = g l
      · exact absurd (fun l hlj hlk => (h1 l hlj).trans (h2 l hlk)) hc
      · rw [if_neg h2, mul_zero]
    · rw [if_neg h1, zero_mul]

/-- Lifts at distinct sites commute: they act on disjoint coordinates. -/
theorem siteLift_comm {j k : ι} (hjk : j ≠ k)
    (A B : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j A * siteLift k B = siteLift k B * siteLift j A := by
  rw [siteLift_mul_siteLift_of_ne hjk, siteLift_mul_siteLift_of_ne hjk.symm]
  ext f g
  simp only [Matrix.of_apply]
  have hcond : (∀ l, l ≠ j → l ≠ k → f l = g l) ↔ ∀ l, l ≠ k → l ≠ j → f l = g l :=
    ⟨fun h l h1 h2 => h l h2 h1, fun h l h1 h2 => h l h2 h1⟩
  by_cases hc : ∀ l, l ≠ j → l ≠ k → f l = g l
  · rw [if_pos hc, if_pos (hcond.mp hc), mul_comm]
  · rw [if_neg hc, if_neg fun h => hc (hcond.mpr h)]

/-! ### `ℓ²` operator-norm preservation

The lift acts block-diagonally over off-site configurations, so it preserves
the `ℓ²` operator norm exactly.  The auxiliary lemmas below transport the norm
through the reindexing equivalence and through the block-diagonal form. -/

/-- Composing the coordinates with an equivalence of index types preserves the
Euclidean norm. -/
private theorem siteLiftAux_norm_toLp_comp_equiv {l m : Type*} [Fintype l] [Fintype m]
    (e : l ≃ m) (v : m → ℂ) :
    ‖(WithLp.toLp 2 (v ∘ e) : EuclideanSpace ℂ l)‖ =
      ‖(WithLp.toLp 2 v : EuclideanSpace ℂ m)‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  congr 1
  simp only [Function.comp_apply]
  exact e.sum_comp fun i => ‖v i‖ ^ 2

/-- The Euclidean norm of a matrix-vector product is controlled by the `ℓ²`
operator norm. -/
private theorem siteLiftAux_norm_toLp_mulVec_le {m : Type*} [Fintype m] [DecidableEq m]
    (M : Matrix m m ℂ) (v : m → ℂ) :
    ‖(WithLp.toLp 2 (M *ᵥ v) : EuclideanSpace ℂ m)‖ ≤
      ‖M‖ * ‖(WithLp.toLp 2 v : EuclideanSpace ℂ m)‖ :=
  calc ‖(WithLp.toLp 2 (M *ᵥ v) : EuclideanSpace ℂ m)‖
      = ‖Matrix.toEuclideanCLM (𝕜 := ℂ) M (WithLp.toLp 2 v)‖ := by
        rw [Matrix.toEuclideanCLM_toLp]
    _ ≤ ‖Matrix.toEuclideanCLM (𝕜 := ℂ) M‖ * ‖(WithLp.toLp 2 v : EuclideanSpace ℂ m)‖ :=
        ContinuousLinearMap.le_opNorm _ _
    _ = ‖M‖ * ‖(WithLp.toLp 2 v : EuclideanSpace ℂ m)‖ := by
        rw [Matrix.l2_opNorm_toEuclideanCLM]

/-- Reindexing a square matrix by an equivalence does not increase the `ℓ²`
operator norm. -/
private theorem siteLiftAux_l2_opNorm_reindex_le {l m : Type*}
    [Fintype l] [DecidableEq l] [Fintype m] [DecidableEq m]
    (e : m ≃ l) (M : Matrix m m ℂ) :
    ‖Matrix.reindex e e M‖ ≤ ‖M‖ := by
  rw [Matrix.cstar_norm_def]
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg M) fun x => ?_
  have happ : Matrix.toEuclideanCLM (𝕜 := ℂ) (Matrix.reindex e e M) x =
      WithLp.toLp 2 ((M *ᵥ (WithLp.ofLp x ∘ e)) ∘ e.symm) := by
    have h1 : Matrix.toEuclideanCLM (𝕜 := ℂ) (Matrix.reindex e e M) x =
        WithLp.toLp 2 (Matrix.reindex e e M *ᵥ WithLp.ofLp x) := rfl
    rw [h1, Matrix.reindex_apply, Matrix.submatrix_mulVec_equiv]
    simp
  rw [happ, siteLiftAux_norm_toLp_comp_equiv e.symm]
  calc ‖(WithLp.toLp 2 (M *ᵥ (WithLp.ofLp x ∘ e)) : EuclideanSpace ℂ m)‖
      ≤ ‖M‖ * ‖(WithLp.toLp 2 (WithLp.ofLp x ∘ e) : EuclideanSpace ℂ m)‖ :=
        siteLiftAux_norm_toLp_mulVec_le M _
    _ = ‖M‖ * ‖x‖ := by
        rw [siteLiftAux_norm_toLp_comp_equiv e (WithLp.ofLp x)]

/-- Reindexing a square matrix by an equivalence preserves the `ℓ²` operator
norm. -/
private theorem siteLiftAux_l2_opNorm_reindex {l m : Type*}
    [Fintype l] [DecidableEq l] [Fintype m] [DecidableEq m]
    (e : m ≃ l) (M : Matrix m m ℂ) :
    ‖Matrix.reindex e e M‖ = ‖M‖ := by
  refine le_antisymm (siteLiftAux_l2_opNorm_reindex_le e M) ?_
  have h := siteLiftAux_l2_opNorm_reindex_le e.symm (Matrix.reindex e e M)
  have h2 : Matrix.reindex e.symm e.symm (Matrix.reindex e e M) = M := by
    rw [← Matrix.reindex_symm]
    exact (Matrix.reindex e e).symm_apply_apply M
  rwa [h2] at h

/-- Entry formula for the block-diagonal matrix-vector product. -/
private theorem siteLiftAux_blockDiagonal_mulVec {o m : Type*} [Fintype o] [DecidableEq o]
    [Fintype m] (A : Matrix m m ℂ) (v : m × o → ℂ) (i : m) (k : o) :
    (Matrix.blockDiagonal (fun _ : o => A) *ᵥ v) (i, k) =
      (A *ᵥ fun j => v (j, k)) i := by
  rw [Matrix.mulVec_apply_eq_sum, Matrix.mulVec_apply_eq_sum, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun j _ => ?_
  have hterm : ∀ k' : o,
      Matrix.blockDiagonal (fun _ : o => A) (i, k) (j, k') * v (j, k') =
        if k = k' then A i j * v (j, k') else 0 := by
    intro k'
    by_cases hk : k = k'
    · subst hk
      rw [Matrix.blockDiagonal_apply_eq, if_pos rfl]
    · rw [Matrix.blockDiagonal_apply_ne _ _ _ hk, zero_mul, if_neg hk]
  simp_rw [hterm]
  rw [Finset.sum_ite_eq]
  exact if_pos (Finset.mem_univ k)

/-- A vector supported on one off-site block has the norm of its block. -/
private theorem siteLiftAux_norm_toLp_single_block {o m : Type*} [Fintype o] [DecidableEq o]
    [Fintype m] (k₀ : o) (u : m → ℂ) :
    ‖(WithLp.toLp 2 (fun p : m × o => if p.2 = k₀ then u p.1 else 0) :
        EuclideanSpace ℂ (m × o))‖ = ‖(WithLp.toLp 2 u : EuclideanSpace ℂ m)‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  congr 1
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hterm : ∀ k : o,
      ‖(WithLp.toLp 2 (fun p : m × o => if p.2 = k₀ then u p.1 else 0) :
        EuclideanSpace ℂ (m × o)) (i, k)‖ ^ 2 = if k = k₀ then ‖u i‖ ^ 2 else 0 := by
    intro k
    by_cases hk : k = k₀ <;> simp [hk]
  simp_rw [hterm]
  rw [Finset.sum_ite_eq' Finset.univ k₀ fun _ => ‖u i‖ ^ 2]
  simp

/-- The `ℓ²` operator norm of a constant block-diagonal matrix is at most the
block norm. -/
private theorem siteLiftAux_l2_opNorm_blockDiagonal_le {o m : Type*}
    [Fintype o] [DecidableEq o] [Fintype m] [DecidableEq m] (A : Matrix m m ℂ) :
    ‖Matrix.blockDiagonal (fun _ : o => A)‖ ≤ ‖A‖ := by
  rw [Matrix.cstar_norm_def]
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg A) fun x => ?_
  refine (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp ?_
  have happ : Matrix.toEuclideanCLM (𝕜 := ℂ) (Matrix.blockDiagonal fun _ : o => A) x =
      WithLp.toLp 2 (Matrix.blockDiagonal (fun _ : o => A) *ᵥ WithLp.ofLp x) := rfl
  rw [happ, mul_pow, EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq]
  calc ∑ p : m × o,
        ‖(WithLp.toLp 2 (Matrix.blockDiagonal (fun _ : o => A) *ᵥ WithLp.ofLp x) :
          EuclideanSpace ℂ (m × o)) p‖ ^ 2
      = ∑ k : o, ∑ i : m, ‖(A *ᵥ fun j => WithLp.ofLp x (j, k)) i‖ ^ 2 := by
        rw [Fintype.sum_prod_type, Finset.sum_comm]
        refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun i _ => ?_
        rw [PiLp.toLp_apply, siteLiftAux_blockDiagonal_mulVec]
    _ = ∑ k : o,
        ‖(WithLp.toLp 2 (A *ᵥ fun j => WithLp.ofLp x (j, k)) : EuclideanSpace ℂ m)‖ ^ 2 := by
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [EuclideanSpace.norm_sq_eq]
    _ ≤ ∑ k : o,
        (‖A‖ * ‖(WithLp.toLp 2 (fun j => WithLp.ofLp x (j, k)) : EuclideanSpace ℂ m)‖) ^ 2 := by
        refine Finset.sum_le_sum fun k _ => ?_
        exact pow_le_pow_left₀ (norm_nonneg _) (siteLiftAux_norm_toLp_mulVec_le A _) 2
    _ = ‖A‖ ^ 2 * ∑ p : m × o, ‖x p‖ ^ 2 := by
        simp_rw [mul_pow, EuclideanSpace.norm_sq_eq]
        rw [← Finset.mul_sum, Fintype.sum_prod_type, Finset.sum_comm]

/-- The block norm is at most the `ℓ²` operator norm of a nonempty constant
block-diagonal matrix. -/
private theorem siteLiftAux_le_l2_opNorm_blockDiagonal {o m : Type*}
    [Fintype o] [DecidableEq o] [Nonempty o] [Fintype m] [DecidableEq m]
    (A : Matrix m m ℂ) :
    ‖A‖ ≤ ‖Matrix.blockDiagonal (fun _ : o => A)‖ := by
  rw [Matrix.cstar_norm_def (A := A)]
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _) fun x => ?_
  obtain ⟨k₀⟩ := ‹Nonempty o›
  have hbd : (Matrix.blockDiagonal (fun _ : o => A) *ᵥ
        fun p : m × o => if p.2 = k₀ then WithLp.ofLp x p.1 else 0) =
      fun p : m × o => if p.2 = k₀ then (A *ᵥ WithLp.ofLp x) p.1 else 0 := by
    funext p
    obtain ⟨i, k⟩ := p
    rw [siteLiftAux_blockDiagonal_mulVec]
    change (A *ᵥ fun j => if k = k₀ then WithLp.ofLp x j else 0) i =
      if k = k₀ then (A *ᵥ WithLp.ofLp x) i else 0
    by_cases hk : k = k₀
    · rw [if_pos hk]
      congr 1
      funext j
      rw [if_pos hk]
    · rw [if_neg hk]
      have harg : (fun j : m => if k = k₀ then WithLp.ofLp x j else 0) =
          fun _ : m => (0 : ℂ) := by
        funext j
        rw [if_neg hk]
      rw [harg, Matrix.mulVec_apply_eq_sum]
      simp
  have happ : Matrix.toEuclideanCLM (𝕜 := ℂ) A x =
      WithLp.toLp 2 (A *ᵥ WithLp.ofLp x) := rfl
  rw [happ]
  calc ‖(WithLp.toLp 2 (A *ᵥ WithLp.ofLp x) : EuclideanSpace ℂ m)‖
      = ‖(WithLp.toLp 2 (Matrix.blockDiagonal (fun _ : o => A) *ᵥ
            fun p : m × o => if p.2 = k₀ then WithLp.ofLp x p.1 else 0) :
          EuclideanSpace ℂ (m × o))‖ := by
        rw [hbd, siteLiftAux_norm_toLp_single_block]
    _ ≤ ‖Matrix.blockDiagonal (fun _ : o => A)‖ *
          ‖(WithLp.toLp 2 (fun p : m × o => if p.2 = k₀ then WithLp.ofLp x p.1 else 0) :
            EuclideanSpace ℂ (m × o))‖ :=
        siteLiftAux_norm_toLp_mulVec_le _ _
    _ = ‖Matrix.blockDiagonal (fun _ : o => A)‖ * ‖x‖ := by
        rw [siteLiftAux_norm_toLp_single_block k₀ (WithLp.ofLp x)]

/-- The single-site lift preserves the `ℓ²` operator norm exactly: the lift is
block-diagonal over off-site configurations with every block equal to `A`. -/
theorem siteLift_l2_opNorm (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) :
    ‖siteLift j A‖ = ‖A‖ := by
  rw [siteLift_eq_reindex_blockDiagonal, siteLiftAux_l2_opNorm_reindex]
  rcases isEmpty_or_nonempty ({k : ι // k ≠ j} → Fin n) with h | h
  · have hFn : IsEmpty (Fin n) := by
      by_contra hcon
      rw [not_isEmpty_iff] at hcon
      obtain ⟨b⟩ := hcon
      exact h.false fun _ => b
    have hA : A = 0 := by
      ext i k
      exact (hFn.false i).elim
    have hbd : (Matrix.blockDiagonal fun _ : {k : ι // k ≠ j} → Fin n => A) = 0 := by
      ext p q
      exact (hFn.false p.1).elim
    rw [hbd, hA, norm_zero, norm_zero]
  · exact le_antisymm (siteLiftAux_l2_opNorm_blockDiagonal_le A)
      (siteLiftAux_le_l2_opNorm_blockDiagonal A)

/-- Single-mode cutoff norm bounds transfer verbatim to the multimode lift. -/
theorem siteLift_l2_opNorm_le (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) :
    ‖siteLift j A‖ ≤ ‖A‖ :=
  le_of_eq (siteLift_l2_opNorm j A)

/-! ### The lift as a continuous unital ring homomorphism

`siteLift j` is additive, multiplicative and unit-preserving, hence a unital
ring homomorphism; it is linear over `ℂ` between finite-dimensional spaces,
hence continuous.  A continuous unital ring homomorphism commutes with the
exponential series, so the lift of a single-mode matrix exponential is the
multimode exponential of the lifted generator.  This is the mechanism that
identifies the lifted single-mode matrix cosine with the multimode cosine of
the lifted quadrature in the sine-Gordon lattice Hamiltonian
(Eqs. `eq:sg-lattice`, `eq:sg-deformations`). -/

/-- The single-site lift packaged as a unital ring homomorphism from
single-mode cutoff matrices to multimode configuration-space matrices. -/
def siteLiftRingHom (j : ι) :
    Matrix (Fin n) (Fin n) ℂ →+* Matrix (ι → Fin n) (ι → Fin n) ℂ where
  toFun := siteLift j
  map_one' := siteLift_one j
  map_mul' A B := (siteLift_mul j A B).symm
  map_zero' := siteLift_zero j
  map_add' := siteLift_add j

/-- The ring-homomorphism packaging acts as the single-site lift. -/
@[simp]
theorem siteLiftRingHom_apply (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) :
    siteLiftRingHom j A = siteLift j A :=
  rfl

/-- The single-site lift packaged as a `ℂ`-linear map. -/
def siteLiftLinearMap (j : ι) :
    Matrix (Fin n) (Fin n) ℂ →ₗ[ℂ] Matrix (ι → Fin n) (ι → Fin n) ℂ where
  toFun := siteLift j
  map_add' := siteLift_add j
  map_smul' := siteLift_smul j

/-- The single-site lift is continuous: it is `ℂ`-linear and the single-mode
cutoff matrix space is finite-dimensional. -/
theorem siteLift_continuous (j : ι) :
    Continuous fun A : Matrix (Fin n) (Fin n) ℂ => siteLift j A :=
  (siteLiftLinearMap (ι := ι) j).continuous_of_finiteDimensional

/-- The single-site lift commutes with the matrix exponential: the lift is a
continuous unital ring homomorphism, so it maps the exponential series of a
single-mode cutoff generator to the exponential series of its multimode lift.
Single-mode compiled exponentials (displacements, SNAP phases, matrix cosines)
therefore lift verbatim to the multimode configuration space. -/
theorem siteLift_exp (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j (NormedSpace.exp A) = NormedSpace.exp (siteLift j A) := by
  letI : NormedAlgebra ℚ (Matrix (Fin n) (Fin n) ℂ) :=
    NormedAlgebra.restrictScalars ℚ ℂ _
  letI : NormedAlgebra ℚ (Matrix (ι → Fin n) (ι → Fin n) ℂ) :=
    NormedAlgebra.restrictScalars ℚ ℂ _
  exact NormedSpace.map_exp (siteLiftRingHom j) (siteLift_continuous j) A

end SiteLift

end

end Cvdv
