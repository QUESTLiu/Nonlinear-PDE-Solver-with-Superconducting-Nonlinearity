import Cvdv.StrangBound
import Cvdv.LLEFinite
import Cvdv.LLEDisplacement

/-!
# Kronecker structure of the truncated LLE dilation

This file proves the finite Kronecker-product facts needed for the truncated
dilation `x_a ⊗ L_loss + I_a ⊗ H_LLE` of manuscript Eq. (eq:lle-dilation):

* Hermiticity of Kronecker products and of the truncated ancillary quadrature
  `x_a` of Eq. (eq:lle-dilation);
* the mixed commutator identity
  `[X ⊗ L, I ⊗ H] = X ⊗ [L, H]`;
* exact multiplicativity of the `l²` operator norm under the Kronecker
  product, hence the cutoff-dependent amplification of `C_LCHS` by the norm of
  the truncated ancillary quadrature stated after
  Eq. (eq:combined-cost-diagnostics);
* the instantiation of the second-order Strang bounds for the step
  Eq. (eq:lle-strang-step) at the dilation generators
  `I_a ⊗ H_LLE` and `x_a ⊗ L_loss`.
-/

namespace Cvdv

open WithLp NormedSpace
open scoped Kronecker Matrix Matrix.Norms.L2Operator Topology

open Asymptotics

noncomputable section

section KroneckerHermitian

variable {ιa ιs : Type*} [Fintype ιa] [DecidableEq ιa] [Fintype ιs] [DecidableEq ιs]

omit [Fintype ιa] [DecidableEq ιa] [Fintype ιs] [DecidableEq ιs] in
/-- The adjoint of a Kronecker product is the Kronecker product of the adjoints. -/
theorem kron_conjTranspose (A : Matrix ιa ιa ℂ) (B : Matrix ιs ιs ℂ) :
    (A ⊗ₖ B)ᴴ = Aᴴ ⊗ₖ Bᴴ :=
  Matrix.conjTranspose_kronecker A B

omit [Fintype ιa] [DecidableEq ιa] [Fintype ιs] [DecidableEq ιs] in
/-- A Kronecker product of Hermitian factors is Hermitian. -/
theorem kron_isHermitian {A : Matrix ιa ιa ℂ} {B : Matrix ιs ιs ℂ}
    (hA : A.IsHermitian) (hB : B.IsHermitian) : (A ⊗ₖ B).IsHermitian :=
  show (A ⊗ₖ B)ᴴ = A ⊗ₖ B by
    rw [Matrix.conjTranspose_kronecker, hA.eq, hB.eq]

omit [DecidableEq ιs] in
/-- The mixed dilation commutator collapses to a single Kronecker factor:
`[X ⊗ L, I ⊗ H] = X ⊗ [L, H]`.  This is the structure behind the
`C_LCHS` diagnostic of the truncated dilation. -/
theorem kron_commutator (X : Matrix ιa ιa ℂ) (L H : Matrix ιs ιs ℂ) :
    (X ⊗ₖ L) * ((1 : Matrix ιa ιa ℂ) ⊗ₖ H)
        - ((1 : Matrix ιa ιa ℂ) ⊗ₖ H) * (X ⊗ₖ L) =
      X ⊗ₖ (L * H - H * L) := by
  rw [← Matrix.mul_kronecker_mul, ← Matrix.mul_kronecker_mul, mul_one, one_mul]
  ext p q
  simp only [Matrix.sub_apply, Matrix.kroneckerMap_apply, mul_sub]

omit [Fintype ιa] [DecidableEq ιa] in
/-- The truncated dimensionless ancillary position quadrature
`x_a = (b_a + b_a†)/√2` of Eq. (eq:lle-dilation) is Hermitian. -/
theorem lleAncillaX_isHermitian (b : Matrix ιa ιa ℂ) :
    (lleAncillaX b).IsHermitian := by
  have hsum : (b + bᴴ).IsHermitian := Matrix.isHermitian_add_transpose_self b
  exact hsum.smul (show star (((1 / Real.sqrt 2 : ℝ) : ℂ)) = _ from
    Complex.conj_ofReal _)

end KroneckerHermitian

section KroneckerNorm

variable {ιa ιs : Type*} [Fintype ιa] [DecidableEq ιa] [Fintype ιs] [DecidableEq ιs]

omit [DecidableEq ιa] [DecidableEq ιs] in
/-- A Kronecker product acts on product vectors factorwise. -/
theorem kron_mulVec (A : Matrix ιa ιa ℂ) (B : Matrix ιs ιs ℂ)
    (x : ιa → ℂ) (y : ιs → ℂ) :
    (A ⊗ₖ B) *ᵥ (fun p : ιa × ιs => x p.1 * y p.2) =
      fun p : ιa × ιs => (A *ᵥ x) p.1 * (B *ᵥ y) p.2 := by
  funext p
  obtain ⟨i, j⟩ := p
  simp only [Matrix.mulVec, dotProduct, Matrix.kroneckerMap_apply]
  rw [Fintype.sum_prod_type, Finset.sum_mul_sum]
  exact Finset.sum_congr rfl fun k _ =>
    Finset.sum_congr rfl fun l _ => by ring

omit [DecidableEq ιa] [DecidableEq ιs] in
/-- The Euclidean norm of a Kronecker product vector factorizes. -/
theorem kron_norm_prodVec (x : ιa → ℂ) (y : ιs → ℂ) :
    ‖(toLp 2 (fun p : ιa × ιs => x p.1 * y p.2) :
        EuclideanSpace ℂ (ιa × ιs))‖ =
      ‖(toLp 2 x : EuclideanSpace ℂ ιa)‖ * ‖(toLp 2 y : EuclideanSpace ℂ ιs)‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq, EuclideanSpace.norm_eq,
    ← Real.sqrt_mul (by positivity), Finset.sum_mul_sum]
  congr 1
  rw [Fintype.sum_prod_type]
  exact Finset.sum_congr rfl fun i _ =>
    Finset.sum_congr rfl fun j _ => by
      simp only [norm_mul, mul_pow]

/-- Matrix-vector bound through the `l²` operator norm, in `toLp` form. -/
private theorem kron_norm_toLp_mulVec_le (B : Matrix ιs ιs ℂ) (w : ιs → ℂ) :
    ‖(toLp 2 (B *ᵥ w) : EuclideanSpace ℂ ιs)‖ ≤
      ‖B‖ * ‖(toLp 2 w : EuclideanSpace ℂ ιs)‖ := by
  rw [← Matrix.toEuclideanCLM_toLp]
  calc
    ‖Matrix.toEuclideanCLM (𝕜 := ℂ) B (toLp 2 w)‖
        ≤ ‖Matrix.toEuclideanCLM (𝕜 := ℂ) B‖ * ‖(toLp 2 w : EuclideanSpace ℂ ιs)‖ :=
      (Matrix.toEuclideanCLM (𝕜 := ℂ) B).le_opNorm _
    _ = ‖B‖ * ‖(toLp 2 w : EuclideanSpace ℂ ιs)‖ := by
      rw [Matrix.l2_opNorm_toEuclideanCLM]

omit [DecidableEq ιs] in
/-- Elementwise action of `I ⊗ B` on a joint vector. -/
theorem kron_one_kronecker_mulVec (B : Matrix ιs ιs ℂ) (v : ιa × ιs → ℂ) :
    ((1 : Matrix ιa ιa ℂ) ⊗ₖ B) *ᵥ v =
      fun p : ιa × ιs => (B *ᵥ fun l => v (p.1, l)) p.2 := by
  funext p
  obtain ⟨i, j⟩ := p
  simp only [Matrix.mulVec, dotProduct, Matrix.kroneckerMap_apply,
    Matrix.one_apply]
  rw [Fintype.sum_prod_type, Finset.sum_comm]
  simp [ite_mul]

omit [DecidableEq ιa] in
/-- Elementwise action of `A ⊗ I` on a joint vector. -/
theorem kron_kronecker_one_mulVec (A : Matrix ιa ιa ℂ) (v : ιa × ιs → ℂ) :
    (A ⊗ₖ (1 : Matrix ιs ιs ℂ)) *ᵥ v =
      fun p : ιa × ιs => (A *ᵥ fun k => v (k, p.2)) p.1 := by
  funext p
  obtain ⟨i, j⟩ := p
  simp only [Matrix.mulVec, dotProduct, Matrix.kroneckerMap_apply,
    Matrix.one_apply]
  rw [Fintype.sum_prod_type]
  simp [mul_ite, ite_mul]

/-- `l²` operator-norm contraction of the ancilla-trivial factor. -/
theorem kron_one_kronecker_l2_opNorm_le (B : Matrix ιs ιs ℂ) :
    ‖(1 : Matrix ιa ιa ℂ) ⊗ₖ B‖ ≤ ‖B‖ := by
  rw [← Matrix.l2_opNorm_toEuclideanCLM]
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg B) fun z => ?_
  refine (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp ?_
  set T := Matrix.toEuclideanCLM (𝕜 := ℂ) ((1 : Matrix ιa ιa ℂ) ⊗ₖ B) with hT
  have happ : ∀ p : ιa × ιs,
      (T z) p = (B *ᵥ fun l => (ofLp z) (p.1, l)) p.2 := by
    intro p
    have h1 : (T z) p = (((1 : Matrix ιa ιa ℂ) ⊗ₖ B) *ᵥ ofLp z) p :=
      congrFun (Matrix.ofLp_toEuclideanCLM _ z) p
    rw [h1, kron_one_kronecker_mulVec]
  calc
    ‖T z‖ ^ 2 = ∑ p : ιa × ιs, ‖(T z) p‖ ^ 2 := EuclideanSpace.norm_sq_eq _
    _ = ∑ i : ιa, ∑ j : ιs, ‖(B *ᵥ fun l => (ofLp z) (i, l)) j‖ ^ 2 := by
      rw [Fintype.sum_prod_type]
      exact Finset.sum_congr rfl fun i _ =>
        Finset.sum_congr rfl fun j _ => by rw [happ (i, j)]
    _ = ∑ i : ιa,
        ‖(toLp 2 (B *ᵥ fun l => (ofLp z) (i, l)) : EuclideanSpace ℂ ιs)‖ ^ 2 := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [EuclideanSpace.norm_sq_eq]
    _ ≤ ∑ i : ιa,
        (‖B‖ * ‖(toLp 2 (fun l => (ofLp z) (i, l)) : EuclideanSpace ℂ ιs)‖) ^ 2 := by
      refine Finset.sum_le_sum fun i _ => ?_
      exact pow_le_pow_left₀ (norm_nonneg _) (kron_norm_toLp_mulVec_le B _) 2
    _ = ‖B‖ ^ 2 * ∑ i : ιa, ∑ l : ιs, ‖(ofLp z) (i, l)‖ ^ 2 := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [mul_pow, EuclideanSpace.norm_sq_eq]
    _ = ‖B‖ ^ 2 * ‖z‖ ^ 2 := by
      rw [EuclideanSpace.norm_sq_eq z, Fintype.sum_prod_type]
    _ = (‖B‖ * ‖z‖) ^ 2 := (mul_pow _ _ 2).symm

/-- `l²` operator-norm contraction of the system-trivial factor. -/
theorem kron_kronecker_one_l2_opNorm_le (A : Matrix ιa ιa ℂ) :
    ‖A ⊗ₖ (1 : Matrix ιs ιs ℂ)‖ ≤ ‖A‖ := by
  rw [← Matrix.l2_opNorm_toEuclideanCLM]
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg A) fun z => ?_
  refine (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp ?_
  set T := Matrix.toEuclideanCLM (𝕜 := ℂ) (A ⊗ₖ (1 : Matrix ιs ιs ℂ)) with hT
  have happ : ∀ p : ιa × ιs,
      (T z) p = (A *ᵥ fun k => (ofLp z) (k, p.2)) p.1 := by
    intro p
    have h1 : (T z) p = ((A ⊗ₖ (1 : Matrix ιs ιs ℂ)) *ᵥ ofLp z) p :=
      congrFun (Matrix.ofLp_toEuclideanCLM _ z) p
    rw [h1, kron_kronecker_one_mulVec]
  calc
    ‖T z‖ ^ 2 = ∑ p : ιa × ιs, ‖(T z) p‖ ^ 2 := EuclideanSpace.norm_sq_eq _
    _ = ∑ j : ιs, ∑ i : ιa, ‖(A *ᵥ fun k => (ofLp z) (k, j)) i‖ ^ 2 := by
      rw [Fintype.sum_prod_type, Finset.sum_comm]
      exact Finset.sum_congr rfl fun j _ =>
        Finset.sum_congr rfl fun i _ => by rw [happ (i, j)]
    _ = ∑ j : ιs,
        ‖(toLp 2 (A *ᵥ fun k => (ofLp z) (k, j)) : EuclideanSpace ℂ ιa)‖ ^ 2 := by
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [EuclideanSpace.norm_sq_eq]
    _ ≤ ∑ j : ιs,
        (‖A‖ * ‖(toLp 2 (fun k => (ofLp z) (k, j)) : EuclideanSpace ℂ ιa)‖) ^ 2 := by
      refine Finset.sum_le_sum fun j _ => ?_
      exact pow_le_pow_left₀ (norm_nonneg _) (kron_norm_toLp_mulVec_le A _) 2
    _ = ‖A‖ ^ 2 * ∑ j : ιs, ∑ k : ιa, ‖(ofLp z) (k, j)‖ ^ 2 := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [mul_pow, EuclideanSpace.norm_sq_eq]
    _ = ‖A‖ ^ 2 * ‖z‖ ^ 2 := by
      rw [EuclideanSpace.norm_sq_eq z, Fintype.sum_prod_type, Finset.sum_comm]
    _ = (‖A‖ * ‖z‖) ^ 2 := (mul_pow _ _ 2).symm

/-- Submultiplicative direction of Kronecker norm multiplicativity. -/
theorem kron_l2_opNorm_le (A : Matrix ιa ιa ℂ) (B : Matrix ιs ιs ℂ) :
    ‖A ⊗ₖ B‖ ≤ ‖A‖ * ‖B‖ := by
  have hfact : A ⊗ₖ B =
      (A ⊗ₖ (1 : Matrix ιs ιs ℂ)) * ((1 : Matrix ιa ιa ℂ) ⊗ₖ B) := by
    rw [← Matrix.mul_kronecker_mul, mul_one, one_mul]
  rw [hfact]
  calc
    ‖(A ⊗ₖ (1 : Matrix ιs ιs ℂ)) * ((1 : Matrix ιa ιa ℂ) ⊗ₖ B)‖
        ≤ ‖A ⊗ₖ (1 : Matrix ιs ιs ℂ)‖ * ‖(1 : Matrix ιa ιa ℂ) ⊗ₖ B‖ :=
      Matrix.l2_opNorm_mul _ _
    _ ≤ ‖A‖ * ‖B‖ :=
      mul_le_mul (kron_kronecker_one_l2_opNorm_le A)
        (kron_one_kronecker_l2_opNorm_le B) (norm_nonneg _) (norm_nonneg _)

/-- Attained direction of Kronecker norm multiplicativity, without any
compactness argument: product test vectors saturate the bound after two
`opNorm_le_bound` rescalings. -/
theorem kron_l2_opNorm_mul_le (A : Matrix ιa ιa ℂ) (B : Matrix ιs ιs ℂ) :
    ‖A‖ * ‖B‖ ≤ ‖A ⊗ₖ B‖ := by
  classical
  have hstep : ∀ (x : EuclideanSpace ℂ ιa) (y : EuclideanSpace ℂ ιs),
      ‖Matrix.toEuclideanCLM (𝕜 := ℂ) A x‖ *
          ‖Matrix.toEuclideanCLM (𝕜 := ℂ) B y‖ ≤
        ‖A ⊗ₖ B‖ * (‖x‖ * ‖y‖) := by
    intro x y
    have hx : Matrix.toEuclideanCLM (𝕜 := ℂ) A x = toLp 2 (A *ᵥ ofLp x) := rfl
    have hy : Matrix.toEuclideanCLM (𝕜 := ℂ) B y = toLp 2 (B *ᵥ ofLp y) := rfl
    have hz :
        Matrix.toEuclideanCLM (𝕜 := ℂ) (A ⊗ₖ B)
            (toLp 2 (fun p : ιa × ιs => (ofLp x) p.1 * (ofLp y) p.2)) =
          toLp 2 (fun p : ιa × ιs => (A *ᵥ ofLp x) p.1 * (B *ᵥ ofLp y) p.2) := by
      rw [Matrix.toEuclideanCLM_toLp, kron_mulVec]
    calc
      ‖Matrix.toEuclideanCLM (𝕜 := ℂ) A x‖ *
          ‖Matrix.toEuclideanCLM (𝕜 := ℂ) B y‖
          = ‖(toLp 2 (fun p : ιa × ιs => (A *ᵥ ofLp x) p.1 * (B *ᵥ ofLp y) p.2) :
              EuclideanSpace ℂ (ιa × ιs))‖ := by
            rw [hx, hy, kron_norm_prodVec]
      _ = ‖Matrix.toEuclideanCLM (𝕜 := ℂ) (A ⊗ₖ B)
            (toLp 2 (fun p : ιa × ιs => (ofLp x) p.1 * (ofLp y) p.2))‖ := by
        rw [hz]
      _ ≤ ‖Matrix.toEuclideanCLM (𝕜 := ℂ) (A ⊗ₖ B)‖ *
          ‖(toLp 2 (fun p : ιa × ιs => (ofLp x) p.1 * (ofLp y) p.2) :
              EuclideanSpace ℂ (ιa × ιs))‖ :=
        (Matrix.toEuclideanCLM (𝕜 := ℂ) (A ⊗ₖ B)).le_opNorm _
      _ = ‖A ⊗ₖ B‖ * (‖x‖ * ‖y‖) := by
        rw [Matrix.l2_opNorm_toEuclideanCLM, kron_norm_prodVec]
  have hstep2 : ∀ y : EuclideanSpace ℂ ιs,
      ‖A‖ * ‖Matrix.toEuclideanCLM (𝕜 := ℂ) B y‖ ≤ ‖A ⊗ₖ B‖ * ‖y‖ := by
    intro y
    have hbound :
        ‖((‖Matrix.toEuclideanCLM (𝕜 := ℂ) B y‖ : ℝ) : ℂ) •
            Matrix.toEuclideanCLM (𝕜 := ℂ) A‖ ≤ ‖A ⊗ₖ B‖ * ‖y‖ := by
      refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun x => ?_
      rw [show (((‖Matrix.toEuclideanCLM (𝕜 := ℂ) B y‖ : ℝ) : ℂ) •
            Matrix.toEuclideanCLM (𝕜 := ℂ) A) x =
          ((‖Matrix.toEuclideanCLM (𝕜 := ℂ) B y‖ : ℝ) : ℂ) •
            Matrix.toEuclideanCLM (𝕜 := ℂ) A x from rfl,
        norm_smul, Complex.norm_real,
        Real.norm_eq_abs, abs_of_nonneg (norm_nonneg _)]
      calc
        ‖Matrix.toEuclideanCLM (𝕜 := ℂ) B y‖ *
            ‖Matrix.toEuclideanCLM (𝕜 := ℂ) A x‖
            ≤ ‖A ⊗ₖ B‖ * (‖x‖ * ‖y‖) := by
          rw [mul_comm]; exact hstep x y
        _ = ‖A ⊗ₖ B‖ * ‖y‖ * ‖x‖ := by ring
    rw [norm_smul, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonneg (norm_nonneg _), Matrix.l2_opNorm_toEuclideanCLM] at hbound
    calc
      ‖A‖ * ‖Matrix.toEuclideanCLM (𝕜 := ℂ) B y‖
          = ‖Matrix.toEuclideanCLM (𝕜 := ℂ) B y‖ * ‖A‖ := mul_comm _ _
      _ ≤ ‖A ⊗ₖ B‖ * ‖y‖ := hbound
  have hfinal :
      ‖((‖A‖ : ℝ) : ℂ) • Matrix.toEuclideanCLM (𝕜 := ℂ) B‖ ≤ ‖A ⊗ₖ B‖ := by
    refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _) fun y => ?_
    rw [show (((‖A‖ : ℝ) : ℂ) • Matrix.toEuclideanCLM (𝕜 := ℂ) B) y =
          ((‖A‖ : ℝ) : ℂ) • Matrix.toEuclideanCLM (𝕜 := ℂ) B y from rfl,
      norm_smul, Complex.norm_real,
      Real.norm_eq_abs, abs_of_nonneg (norm_nonneg _)]
    exact hstep2 y
  rwa [norm_smul, Complex.norm_real, Real.norm_eq_abs,
    abs_of_nonneg (norm_nonneg _), Matrix.l2_opNorm_toEuclideanCLM] at hfinal

/-- Exact multiplicativity of the `l²` operator norm under Kronecker products. -/
theorem kron_l2_opNorm (A : Matrix ιa ιa ℂ) (B : Matrix ιs ιs ℂ) :
    ‖A ⊗ₖ B‖ = ‖A‖ * ‖B‖ :=
  le_antisymm (kron_l2_opNorm_le A B) (kron_l2_opNorm_mul_le A B)

/-- The `C_LCHS` amplification stated after Eq. (eq:combined-cost-diagnostics):
in the finite ancillary oscillator representation, the dilation commutator norm
is the system commutator norm multiplied by the cutoff-dependent norm of the
truncated ancillary quadrature factor `X`. -/
theorem kron_commutator_norm (X : Matrix ιa ιa ℂ) (L H : Matrix ιs ιs ℂ) :
    ‖(X ⊗ₖ L) * ((1 : Matrix ιa ιa ℂ) ⊗ₖ H)
        - ((1 : Matrix ιa ιa ℂ) ⊗ₖ H) * (X ⊗ₖ L)‖ =
      ‖X‖ * ‖L * H - H * L‖ := by
  rw [kron_commutator, kron_l2_opNorm]

end KroneckerNorm

section LLEDilation

variable {ι ιa ιs : Type*} [Fintype ι] [DecidableEq ι]
  [Fintype ιa] [DecidableEq ιa] [Fintype ιs] [DecidableEq ιs]

omit [DecidableEq ι] in
/-- Local cubic Strang defect for the truncated dilation of
Eq. (eq:lle-dilation): system generator `I_a ⊗ H_LLE` on the half steps and
loss generator `x_a ⊗ L_loss` in the middle of Eq. (eq:lle-strang-step). -/
theorem lleDilation_strangDefect_isBigO_cubic
    (b : Matrix ιa ιa ℂ) (a : ι → Matrix ιs ιs ℂ) (next : ι → ι)
    (delta dispersion g kappa : ℝ) (F : ι → ℂ) :
    (strangDefect
      ((-Complex.I) •
        ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs))
          ((1 : Matrix ιa ιa ℂ) ⊗ₖ finiteLLEHamiltonian a next delta dispersion g F)))
      ((-Complex.I) •
        ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs))
          (lleAncillaX b ⊗ₖ finiteLLELoss a kappa))))
      =O[𝓝 0] (fun t : ℝ => t ^ 3) :=
  lleFiniteCutoffMatrix_strangDefect_isBigO_cubic _ _

omit [DecidableEq ι] in
/-- Manuscript Eq. (eq:lle-strang-step): for fixed system and ancillary
cutoffs and fixed physical time `T`, the repeated symmetric step for the
truncated dilation `x_a ⊗ L_loss + I_a ⊗ H_LLE` converges to the exact
truncated evolution with global operator-norm error `O(n_t⁻²)`.  Hermiticity
of both dilation terms is discharged from the Kronecker lemmas above. -/
theorem lleDilation_strangPow_isBigO_inv_sq
    (b : Matrix ιa ιa ℂ) (a : ι → Matrix ιs ιs ℂ) (next : ι → ι)
    (delta dispersion g kappa : ℝ) (F : ι → ℂ) (hkappa : 0 ≤ kappa) (T : ℝ) :
    (fun n : ℕ =>
      (strangStep
        ((-Complex.I) • ((T : ℂ) •
          ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs))
            ((1 : Matrix ιa ιa ℂ) ⊗ₖ
              finiteLLEHamiltonian a next delta dispersion g F))))
        ((-Complex.I) • ((T : ℂ) •
          ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs))
            (lleAncillaX b ⊗ₖ finiteLLELoss a kappa))))
        (n : ℝ)⁻¹) ^ n -
      exp ((-Complex.I) • ((T : ℂ) •
        (((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs))
            ((1 : Matrix ιa ιa ℂ) ⊗ₖ
              finiteLLEHamiltonian a next delta dispersion g F)) +
          ((Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ιa × ιs))
            (lleAncillaX b ⊗ₖ finiteLLELoss a kappa))))))
      =O[Filter.atTop] (fun n : ℕ => ((n : ℝ)⁻¹) ^ 2) := by
  apply lleFiniteCutoffMatrix_strangPow_isBigO_inv_sq
  · exact kron_isHermitian Matrix.isHermitian_one
      (finiteLLEHamiltonian_isHermitian a next delta dispersion g F)
  · exact kron_isHermitian (lleAncillaX_isHermitian b)
      (finiteLLELoss_posSemidef a kappa hkappa).1

omit [DecidableEq ι] in
/-- Concrete form of the `C_LCHS` amplification sentence after
Eq. (eq:combined-cost-diagnostics): for the truncated dilation
`x_a ⊗ L_loss + I_a ⊗ H_LLE` of Eq. (eq:lle-dilation), the `l²` operator norm
of the dilation splitting commutator `[x_a ⊗ L_loss, I_a ⊗ H_LLE]` equals the
norm of the truncated ancillary quadrature `x_a` times the norm of the
truncated commutator `[L_loss, H_LLE]`, the finite-matrix realization of
`C_LCHS` used throughout the LLE section. -/
theorem lleDilation_commutator_norm
    (b : Matrix ιa ιa ℂ) (a : ι → Matrix ιs ιs ℂ) (next : ι → ι)
    (delta dispersion g kappa : ℝ) (F : ι → ℂ) :
    ‖(lleAncillaX b ⊗ₖ finiteLLELoss a kappa) *
          ((1 : Matrix ιa ιa ℂ) ⊗ₖ finiteLLEHamiltonian a next delta dispersion g F) -
        ((1 : Matrix ιa ιa ℂ) ⊗ₖ finiteLLEHamiltonian a next delta dispersion g F) *
          (lleAncillaX b ⊗ₖ finiteLLELoss a kappa)‖ =
      ‖lleAncillaX b‖ *
        ‖finiteLLELoss a kappa * finiteLLEHamiltonian a next delta dispersion g F -
            finiteLLEHamiltonian a next delta dispersion g F * finiteLLELoss a kappa‖ :=
  kron_commutator_norm _ _ _

end LLEDilation

end

end Cvdv
