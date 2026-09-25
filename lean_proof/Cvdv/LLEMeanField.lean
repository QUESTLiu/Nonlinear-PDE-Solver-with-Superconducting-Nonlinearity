import Cvdv.SiteLift
import Cvdv.LLEFinite
import Cvdv.LLENormalizedExpectation

/-!
# The Lugiato--Lefever mean-field chain

This file formalizes the strict-cutoff mean-field chain of manuscript
Sec. `subsec:lle-mean-field`:

* the truncated single-mode canonical commutation relation of the strict-cutoff
  ladder matrices, and its multimode form
  `[a_j, a_k†] = δ_jk (I - n_cut P_j^top)` (the display after "The cutoff
  ladder operators obey");
* the exact commutator of the finite LLE Hamiltonian
  (Eq. `eq:lle-native-hamiltonian`) with a cutoff ladder operator, including
  the explicit strict-cutoff corrections proportional to the top-level
  projector `P_j^top`, and the loss anticommutator built from
  `[N_tot, a_j] = -a_j`;
* the exact cutoff first-moment equation Eq. `eq:lle-first-moment-exact`,
  obtained by specializing Eq. `eq:lle-normalized-expectation`
  (`normalizedExpectation_hasDerivAt`) at `O = a_j`, with the connected
  moments `C_j^loss`, `C_j^(3)` and the cutoff-boundary term `B_j^cut`
  defined exactly as in the manuscript;
* the conditional mean-field corollary Eq. `eq:lle-mean-field-discrete`:
  when the three remainders vanish, the normalized first moment obeys the
  semidiscrete LLE.
-/

namespace Cvdv

open scoped BigOperators Matrix InnerProductSpace

noncomputable section

/-! ### Single-mode strict-cutoff ladder algebra -/

section LLEMeanFieldSingleMode

/-- The top-level projector `P^top = |n_cut - 1⟩⟨n_cut - 1|` of the strict local
cutoff, manuscript display defining `P_j^top` (single-mode factor). -/
def lleMF_topProj (n : ℕ) : Matrix (Fin n) (Fin n) ℂ :=
  Matrix.diagonal fun i => if (i : ℕ) = n - 1 then 1 else 0

/-- Entry formula for the strict-cutoff lowering matrix: one nonzero
superdiagonal with entries `√j`. -/
private theorem lleMF_lowering_apply (n : ℕ) (i j : Fin n) :
    fockSandwichMatrix n lowering i j =
      if (j : ℕ) = (i : ℕ) + 1 then ((Real.sqrt (j : ℕ) : ℝ) : ℂ) else 0 := by
  rw [fockSandwichMatrix_apply, lowering_single]
  simp only [one_smul, Finsupp.smul_apply, smul_eq_mul, Finsupp.single_apply]
  by_cases hj : (j : ℕ) = (i : ℕ) + 1
  · rw [if_pos (by omega : (j : ℕ) - 1 = (i : ℕ)), if_pos hj, mul_one]
  · rw [if_neg hj]
    rcases Nat.eq_zero_or_pos (j : ℕ) with h0 | hpos
    · simp [h0]
    · rw [if_neg (by omega : ¬((j : ℕ) - 1 = (i : ℕ))), mul_zero]

/-- Entry formula for the adjoint of the strict-cutoff lowering matrix. -/
private theorem lleMF_lowering_conjTranspose_apply (n : ℕ) (i j : Fin n) :
    (fockSandwichMatrix n lowering)ᴴ i j =
      if (i : ℕ) = (j : ℕ) + 1 then ((Real.sqrt (i : ℕ) : ℝ) : ℂ) else 0 := by
  rw [Matrix.conjTranspose_apply, lleMF_lowering_apply]
  by_cases h : (i : ℕ) = (j : ℕ) + 1
  · rw [if_pos h]
    simp
  · rw [if_neg h, star_zero]

/-- The strict-cutoff number matrix is diagonal with entries `0, 1, ..., n - 1`. -/
private theorem lleMF_number_eq_diagonal (n : ℕ) :
    (fockSandwichMatrix n lowering)ᴴ * fockSandwichMatrix n lowering =
      Matrix.diagonal fun i : Fin n => ((i : ℕ) : ℂ) := by
  ext i j
  rw [Matrix.mul_apply]
  by_cases hij : i = j
  · subst hij
    rw [Matrix.diagonal_apply_eq]
    rcases Nat.eq_zero_or_pos (i : ℕ) with h0 | hpos
    · rw [Finset.sum_eq_zero fun k _ => by
        rw [lleMF_lowering_conjTranspose_apply,
          if_neg (by omega : ¬((i : ℕ) = (k : ℕ) + 1)), zero_mul]]
      rw [h0]
      simp
    · have hi1 : (i : ℕ) - 1 < n := by have := i.isLt; omega
      rw [Finset.sum_eq_single (⟨(i : ℕ) - 1, hi1⟩ : Fin n)]
      · rw [lleMF_lowering_conjTranspose_apply, lleMF_lowering_apply]
        have hval : ((⟨(i : ℕ) - 1, hi1⟩ : Fin n) : ℕ) = (i : ℕ) - 1 := rfl
        have hcond : (i : ℕ) = ((⟨(i : ℕ) - 1, hi1⟩ : Fin n) : ℕ) + 1 := by
          rw [hval]
          omega
        rw [if_pos hcond, ← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity)]
        simp
      · intro k _ hk
        rw [lleMF_lowering_conjTranspose_apply]
        have : ¬((i : ℕ) = (k : ℕ) + 1) := by
          intro hcon
          exact hk (Fin.ext (show (k : ℕ) = (i : ℕ) - 1 by omega))
        rw [if_neg this, zero_mul]
      · exact fun h => absurd (Finset.mem_univ _) h
  · rw [Matrix.diagonal_apply_ne _ hij]
    refine Finset.sum_eq_zero fun k _ => ?_
    rw [lleMF_lowering_conjTranspose_apply, lleMF_lowering_apply]
    by_cases h1 : (i : ℕ) = (k : ℕ) + 1
    · by_cases h2 : (j : ℕ) = (k : ℕ) + 1
      · exact absurd (Fin.ext (by omega : (i : ℕ) = (j : ℕ))) hij
      · rw [if_neg h2, mul_zero]
    · rw [if_neg h1, zero_mul]

/-- The reversed product `L Lᴴ` is diagonal with entries `1, ..., n - 1, 0`:
the top level is annihilated by the strict cutoff. -/
private theorem lleMF_LR_eq_diagonal (n : ℕ) :
    fockSandwichMatrix n lowering * (fockSandwichMatrix n lowering)ᴴ =
      Matrix.diagonal fun i : Fin n =>
        if (i : ℕ) + 1 < n then ((i : ℕ) : ℂ) + 1 else 0 := by
  ext i j
  rw [Matrix.mul_apply]
  by_cases hij : i = j
  · subst hij
    rw [Matrix.diagonal_apply_eq]
    by_cases hi : (i : ℕ) + 1 < n
    · rw [if_pos hi, Finset.sum_eq_single (⟨(i : ℕ) + 1, hi⟩ : Fin n)]
      · rw [lleMF_lowering_apply, lleMF_lowering_conjTranspose_apply]
        have hcond : ((⟨(i : ℕ) + 1, hi⟩ : Fin n) : ℕ) = (i : ℕ) + 1 := rfl
        rw [if_pos hcond, ← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity),
          hcond]
        push_cast
        ring
      · intro k _ hk
        rw [lleMF_lowering_apply]
        have : ¬((k : ℕ) = (i : ℕ) + 1) := by
          intro hcon
          exact hk (Fin.ext hcon)
        rw [if_neg this, zero_mul]
      · exact fun h => absurd (Finset.mem_univ _) h
    · rw [if_neg hi]
      refine Finset.sum_eq_zero fun k _ => ?_
      rw [lleMF_lowering_apply]
      rw [if_neg (by have := k.isLt; omega : ¬((k : ℕ) = (i : ℕ) + 1)), zero_mul]
  · rw [Matrix.diagonal_apply_ne _ hij]
    refine Finset.sum_eq_zero fun k _ => ?_
    rw [lleMF_lowering_apply, lleMF_lowering_conjTranspose_apply]
    by_cases h1 : (k : ℕ) = (i : ℕ) + 1
    · by_cases h2 : (k : ℕ) = (j : ℕ) + 1
      · exact absurd (Fin.ext (by omega : (i : ℕ) = (j : ℕ))) hij
      · rw [if_neg h2, mul_zero]
    · rw [if_neg h1, zero_mul]

/-- Manuscript truncated CCR, single mode: the strict-cutoff ladder matrices obey
`[a, a†] = I - n_cut P^top`, the display after "The cutoff ladder operators obey"
(with `δ_jk = 1`).  Below the top level the commutator is the identity; at the
top level the entry is `1 - n_cut`. -/
theorem lleMF_truncated_ccr_single (n : ℕ) :
    fockSandwichMatrix n lowering * fockSandwichMatrix n raising -
        fockSandwichMatrix n raising * fockSandwichMatrix n lowering =
      1 - (n : ℂ) • lleMF_topProj n := by
  rw [fockSandwichMatrix_raising_eq_lowering_conjTranspose, lleMF_LR_eq_diagonal,
    lleMF_number_eq_diagonal, lleMF_topProj]
  ext i j
  rw [Matrix.sub_apply, Matrix.sub_apply, Matrix.smul_apply]
  by_cases hij : i = j
  · subst hij
    rw [Matrix.diagonal_apply_eq, Matrix.diagonal_apply_eq, Matrix.diagonal_apply_eq,
      Matrix.one_apply_eq]
    by_cases hi : (i : ℕ) + 1 < n
    · rw [if_pos hi, if_neg (by omega : ¬((i : ℕ) = n - 1)), smul_zero]
      ring
    · have h1 : (i : ℕ) = n - 1 := by have := i.isLt; omega
      have h2 : (n : ℂ) = ((i : ℕ) : ℂ) + 1 := by
        have hn : n = (i : ℕ) + 1 := by have := i.isLt; omega
        exact_mod_cast congrArg (Nat.cast : ℕ → ℂ) hn
      rw [if_neg hi, if_pos h1, h2, smul_eq_mul]
      ring
  · rw [Matrix.diagonal_apply_ne _ hij, Matrix.diagonal_apply_ne _ hij,
      Matrix.diagonal_apply_ne _ hij, Matrix.one_apply_ne hij]
    simp

/-- Reversed single-mode truncated CCR: `a† a - a a† = n_cut P^top - I`. -/
private theorem lleMF_ccr_single_reversed (n : ℕ) :
    (fockSandwichMatrix n lowering)ᴴ * fockSandwichMatrix n lowering -
        fockSandwichMatrix n lowering * (fockSandwichMatrix n lowering)ᴴ =
      (n : ℂ) • lleMF_topProj n - 1 := by
  have h := lleMF_truncated_ccr_single n
  rw [fockSandwichMatrix_raising_eq_lowering_conjTranspose] at h
  rw [← neg_sub, h, neg_sub]

/-- The top-level projector annihilates the range of the lowering matrix:
`P^top a = 0`, because `a` maps into levels `≤ n_cut - 2`. -/
private theorem lleMF_topProj_mul_lowering (n : ℕ) :
    lleMF_topProj n * fockSandwichMatrix n lowering = 0 := by
  ext i j
  rw [lleMF_topProj, Matrix.diagonal_mul, lleMF_lowering_apply, Matrix.zero_apply]
  by_cases hi : (i : ℕ) = n - 1
  · rw [if_neg (by have := j.isLt; have := i.isLt; omega : ¬((j : ℕ) = (i : ℕ) + 1)),
      mul_zero]
  · rw [if_neg hi, zero_mul]

/-- Conjugation transport for the one-step shift: a diagonal matrix moves
through the lowering matrix by shifting its diagonal one level down. -/
private theorem lleMF_diagonal_mul_lowering_eq (n : ℕ) (d d' : Fin n → ℂ)
    (h : ∀ i j : Fin n, (j : ℕ) = (i : ℕ) + 1 → d i = d' j) :
    Matrix.diagonal d * fockSandwichMatrix n lowering =
      fockSandwichMatrix n lowering * Matrix.diagonal d' := by
  ext i j
  rw [Matrix.diagonal_mul, Matrix.mul_diagonal, lleMF_lowering_apply]
  by_cases hc : (j : ℕ) = (i : ℕ) + 1
  · rw [if_pos hc, h i j hc]
    ring
  · rw [if_neg hc, mul_zero, zero_mul]

private theorem lleMF_shift_cast (n : ℕ) :
    ∀ i j : Fin n, (j : ℕ) = (i : ℕ) + 1 → ((i : ℕ) : ℂ) = ((j : ℕ) : ℂ) - 1 := by
  intro i j h
  have hc : ((j : ℕ) : ℂ) = ((i : ℕ) : ℂ) + 1 := by exact_mod_cast h
  rw [hc]
  ring

/-- The exact single-mode number commutator on the cutoff space:
`[a† a, a] = -a` (no strict-cutoff correction, since `P^top a = 0`). -/
private theorem lleMF_number_comm_lowering (n : ℕ) :
    (fockSandwichMatrix n lowering)ᴴ * fockSandwichMatrix n lowering *
        fockSandwichMatrix n lowering -
      fockSandwichMatrix n lowering *
        ((fockSandwichMatrix n lowering)ᴴ * fockSandwichMatrix n lowering) =
      -fockSandwichMatrix n lowering := by
  rw [lleMF_number_eq_diagonal,
    lleMF_diagonal_mul_lowering_eq n (fun i => ((i : ℕ) : ℂ))
      (fun i => ((i : ℕ) : ℂ) - 1) (lleMF_shift_cast n),
    ← mul_sub, Matrix.diagonal_sub]
  have hfun : (fun i : Fin n => (((i : ℕ) : ℂ) - 1) - ((i : ℕ) : ℂ)) =
      fun _ : Fin n => (-1 : ℂ) := by
    funext i
    ring
  rw [hfun]
  have h1 : Matrix.diagonal (fun _ : Fin n => (-1 : ℂ)) =
      -(1 : Matrix (Fin n) (Fin n) ℂ) := by
    ext i k
    by_cases h : i = k
    · subst h
      simp
    · simp [Matrix.diagonal_apply_ne _ h, Matrix.one_apply_ne h]
  rw [h1, mul_neg_one]

/-- The strict-cutoff Kerr site matrix `(a²)† a²` is diagonal with entries
`m (m - 1)`: the sandwiched quartic stays inside the cutoff. -/
private theorem lleMF_kerr_site_eq_diagonal (n : ℕ) :
    ((fockSandwichMatrix n lowering) ^ 2)ᴴ * (fockSandwichMatrix n lowering) ^ 2 =
      Matrix.diagonal fun i : Fin n => ((i : ℕ) : ℂ) * (((i : ℕ) : ℂ) - 1) := by
  rw [pow_two, Matrix.conjTranspose_mul]
  have hassoc : (fockSandwichMatrix n lowering)ᴴ * (fockSandwichMatrix n lowering)ᴴ *
      (fockSandwichMatrix n lowering * fockSandwichMatrix n lowering) =
      (fockSandwichMatrix n lowering)ᴴ *
        ((fockSandwichMatrix n lowering)ᴴ * fockSandwichMatrix n lowering *
          fockSandwichMatrix n lowering) := by
    noncomm_ring
  rw [hassoc, lleMF_number_eq_diagonal,
    lleMF_diagonal_mul_lowering_eq n (fun i => ((i : ℕ) : ℂ))
      (fun i => ((i : ℕ) : ℂ) - 1) (lleMF_shift_cast n),
    ← mul_assoc, lleMF_number_eq_diagonal, Matrix.diagonal_mul_diagonal]

private theorem lleMF_shift_cast_sq (n : ℕ) :
    ∀ i j : Fin n, (j : ℕ) = (i : ℕ) + 1 →
      ((i : ℕ) : ℂ) * (((i : ℕ) : ℂ) - 1) =
        (((j : ℕ) : ℂ) - 1) * (((j : ℕ) : ℂ) - 2) := by
  intro i j h
  have hc : ((j : ℕ) : ℂ) = ((i : ℕ) : ℂ) + 1 := by exact_mod_cast h
  rw [hc]
  ring

/-- The exact single-mode Kerr commutator on the cutoff space:
`[(a²)† a², a] = -2 a† a²` (the strict-cutoff correction `P^top a† a²`
vanishes identically). -/
private theorem lleMF_kerr_comm_lowering (n : ℕ) :
    ((fockSandwichMatrix n lowering) ^ 2)ᴴ * (fockSandwichMatrix n lowering) ^ 2 *
        fockSandwichMatrix n lowering -
      fockSandwichMatrix n lowering *
        (((fockSandwichMatrix n lowering) ^ 2)ᴴ * (fockSandwichMatrix n lowering) ^ 2) =
      (-2 : ℂ) • ((fockSandwichMatrix n lowering)ᴴ * (fockSandwichMatrix n lowering) ^ 2) := by
  rw [lleMF_kerr_site_eq_diagonal,
    lleMF_diagonal_mul_lowering_eq n
      (fun i => ((i : ℕ) : ℂ) * (((i : ℕ) : ℂ) - 1))
      (fun i => (((i : ℕ) : ℂ) - 1) * (((i : ℕ) : ℂ) - 2)) (lleMF_shift_cast_sq n),
    ← mul_sub, Matrix.diagonal_sub]
  have hfun : (fun i : Fin n =>
      ((((i : ℕ) : ℂ) - 1) * (((i : ℕ) : ℂ) - 2)) -
        ((i : ℕ) : ℂ) * (((i : ℕ) : ℂ) - 1)) =
      fun i : Fin n => (-2 : ℂ) * (((i : ℕ) : ℂ) - 1) := by
    funext i
    ring
  rw [hfun]
  have hsmul : Matrix.diagonal (fun i : Fin n => (-2 : ℂ) * (((i : ℕ) : ℂ) - 1)) =
      (-2 : ℂ) • Matrix.diagonal (fun i : Fin n => ((i : ℕ) : ℂ) - 1) := by
    rw [← Matrix.diagonal_smul]
    congr 1
  rw [hsmul, Matrix.mul_smul]
  congr 1
  rw [← lleMF_diagonal_mul_lowering_eq n (fun i => ((i : ℕ) : ℂ))
      (fun i => ((i : ℕ) : ℂ) - 1) (lleMF_shift_cast n),
    ← lleMF_number_eq_diagonal, pow_two, mul_assoc]

end LLEMeanFieldSingleMode

/-! ### Multimode cutoff ladders and the δ_jk truncated CCR -/

section LLEMeanFieldMultimode

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The cutoff ladder operator of mode `j` on the multimode cutoff space:
the single-site lift of the strict-cutoff lowering matrix, manuscript
Eq. `eq:lle-system-projector` context (`a_j` with local cutoff `n`). -/
def lleMFLadder (n : ℕ) (j : ι) : Matrix (ι → Fin n) (ι → Fin n) ℂ :=
  siteLift j (fockSandwichMatrix n lowering)

/-- The multimode top-level projector
`P_j^top = (|n_cut - 1⟩⟨n_cut - 1|)_j ⊗ I_{≠j}` of the manuscript. -/
def lleMFTopProj (n : ℕ) (j : ι) : Matrix (ι → Fin n) (ι → Fin n) ℂ :=
  siteLift j (lleMF_topProj n)

private theorem lleMF_siteLift_neg {n : ℕ} (j : ι) (A : Matrix (Fin n) (Fin n) ℂ) :
    siteLift j (-A) = -siteLift j A := by
  rw [← neg_one_smul ℂ A, siteLift_smul, neg_one_smul]

/-- Cutoff ladder operators at any two sites commute. -/
private theorem lleMF_ladder_mul_comm (n : ℕ) (m j : ι) :
    lleMFLadder n m * lleMFLadder n j = lleMFLadder n j * lleMFLadder n m := by
  by_cases h : m = j
  · rw [h]
  · exact siteLift_comm h _ _

/-- The reversed multimode truncated CCR:
`a_m† a_j - a_j a_m† = δ_mj (n_cut P_j^top - I)`. -/
private theorem lleMF_adj_comm (n : ℕ) (m j : ι) :
    (lleMFLadder n m)ᴴ * lleMFLadder n j - lleMFLadder n j * (lleMFLadder n m)ᴴ =
      if m = j then (n : ℂ) • lleMFTopProj n j - 1 else 0 := by
  by_cases h : m = j
  · subst h
    rw [if_pos rfl, lleMFLadder, lleMFTopProj, siteLift_conjTranspose, siteLift_mul,
      siteLift_mul, ← siteLift_sub, lleMF_ccr_single_reversed, siteLift_sub,
      siteLift_smul, siteLift_one]
  · rw [if_neg h, lleMFLadder, lleMFLadder, siteLift_conjTranspose,
      siteLift_comm h, sub_self]

/-- The lifted top projector annihilates the ladder operator at the same site. -/
private theorem lleMF_topProjSite_mul_ladder_self (n : ℕ) (j : ι) :
    lleMFTopProj n j * lleMFLadder n j = 0 := by
  rw [lleMFTopProj, lleMFLadder, siteLift_mul, lleMF_topProj_mul_lowering, siteLift_zero]

/-- Manuscript truncated CCR display of Sec. `subsec:lle-mean-field`:
`[a_j, a_k†] = δ_jk (I - n_cut P_j^top)` for the strict local cutoff.  The
off-diagonal case is the commutation of lifts at disjoint sites; the diagonal
case is the single-mode truncated CCR transported through the site lift. -/
theorem lleMF_truncated_ccr (n : ℕ) (j k : ι) :
    lleMFLadder n j * (lleMFLadder n k)ᴴ - (lleMFLadder n k)ᴴ * lleMFLadder n j =
      if j = k then 1 - (n : ℂ) • lleMFTopProj n j else 0 := by
  by_cases hjk : j = k
  · subst hjk
    rw [if_pos rfl, lleMFLadder, lleMFTopProj, siteLift_conjTranspose, siteLift_mul,
      siteLift_mul, ← siteLift_sub]
    have h := lleMF_truncated_ccr_single n
    rw [fockSandwichMatrix_raising_eq_lowering_conjTranspose] at h
    rw [h, siteLift_sub, siteLift_one, siteLift_smul]
  · rw [if_neg hjk, lleMFLadder, lleMFLadder, siteLift_conjTranspose,
      siteLift_comm hjk, sub_self]

/-! ### Commutator inputs for the first-moment equation -/

/-- Manuscript identity `[N_tot, a_j] = -a_j`, exact on the strict cutoff space:
the total number matrix of the cutoff ladder family commutes with `a_j` up to
`-a_j`, with no strict-cutoff correction. -/
theorem lleMF_totalNumber_comm_ladder (n : ℕ) (j : ι) :
    finiteLLETotalNumber (lleMFLadder n) * lleMFLadder n j -
        lleMFLadder n j * finiteLLETotalNumber (lleMFLadder n) =
      -lleMFLadder n j := by
  unfold finiteLLETotalNumber
  rw [Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]
  have h0 : ∀ k ∈ Finset.univ, k ≠ j →
      (lleMFLadder n k)ᴴ * lleMFLadder n k * lleMFLadder n j -
        lleMFLadder n j * ((lleMFLadder n k)ᴴ * lleMFLadder n k) = 0 := by
    intro k _ hkj
    have hl : (lleMFLadder n k)ᴴ * lleMFLadder n k =
        siteLift k ((fockSandwichMatrix n lowering)ᴴ * fockSandwichMatrix n lowering) := by
      rw [lleMFLadder, siteLift_conjTranspose, siteLift_mul]
    rw [hl, lleMFLadder, siteLift_comm hkj, sub_self]
  rw [Finset.sum_eq_single j h0 (fun h => absurd (Finset.mem_univ j) h)]
  rw [lleMFLadder, siteLift_conjTranspose]
  simp only [siteLift_mul]
  rw [← siteLift_sub, lleMF_number_comm_lowering, lleMF_siteLift_neg]

/-- The dispersion-sum commutator with a cutoff ladder operator, on the
periodic lattice (`next` a permutation with inverse `next.symm`):
`[Σ_k (a_{k+1} - a_k)†(a_{k+1} - a_k), a_j]
  = a_{j+1} + a_{j-1} - 2 a_j - n_cut P_j^top (a_{j+1} + a_{j-1})`.
The last term is the strict-cutoff correction entering `B_j^cut`. -/
theorem lleMF_dispersion_comm_ladder (n : ℕ) (next : Equiv.Perm ι) (j : ι) :
    (∑ k, (lleMFLadder n (next k) - lleMFLadder n k)ᴴ *
        (lleMFLadder n (next k) - lleMFLadder n k)) * lleMFLadder n j -
      lleMFLadder n j *
        ∑ k, (lleMFLadder n (next k) - lleMFLadder n k)ᴴ *
          (lleMFLadder n (next k) - lleMFLadder n k) =
      lleMFLadder n (next j) + lleMFLadder n (next.symm j) - (2 : ℂ) • lleMFLadder n j -
        (n : ℂ) • (lleMFTopProj n j *
          (lleMFLadder n (next j) + lleMFLadder n (next.symm j))) := by
  rw [Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]
  have hterm : ∀ k : ι,
      (lleMFLadder n (next k) - lleMFLadder n k)ᴴ *
          (lleMFLadder n (next k) - lleMFLadder n k) * lleMFLadder n j -
        lleMFLadder n j *
          ((lleMFLadder n (next k) - lleMFLadder n k)ᴴ *
            (lleMFLadder n (next k) - lleMFLadder n k)) =
      (if next k = j
        then ((n : ℂ) • lleMFTopProj n j - 1) *
          (lleMFLadder n (next k) - lleMFLadder n k) else 0) -
      (if k = j
        then ((n : ℂ) • lleMFTopProj n j - 1) *
          (lleMFLadder n (next k) - lleMFLadder n k) else 0) := by
    intro k
    have hcommX : (lleMFLadder n (next k) - lleMFLadder n k) * lleMFLadder n j =
        lleMFLadder n j * (lleMFLadder n (next k) - lleMFLadder n k) := by
      rw [sub_mul, mul_sub, lleMF_ladder_mul_comm n (next k) j,
        lleMF_ladder_mul_comm n k j]
    have hXd : (lleMFLadder n (next k) - lleMFLadder n k)ᴴ * lleMFLadder n j -
        lleMFLadder n j * (lleMFLadder n (next k) - lleMFLadder n k)ᴴ =
        (if next k = j then (n : ℂ) • lleMFTopProj n j - 1 else 0) -
          (if k = j then (n : ℂ) • lleMFTopProj n j - 1 else 0) := by
      rw [Matrix.conjTranspose_sub]
      calc ((lleMFLadder n (next k))ᴴ - (lleMFLadder n k)ᴴ) * lleMFLadder n j -
            lleMFLadder n j * ((lleMFLadder n (next k))ᴴ - (lleMFLadder n k)ᴴ) =
          ((lleMFLadder n (next k))ᴴ * lleMFLadder n j -
              lleMFLadder n j * (lleMFLadder n (next k))ᴴ) -
            ((lleMFLadder n k)ᴴ * lleMFLadder n j -
              lleMFLadder n j * (lleMFLadder n k)ᴴ) := by noncomm_ring
        _ = _ := by rw [lleMF_adj_comm n (next k) j, lleMF_adj_comm n k j]
    calc (lleMFLadder n (next k) - lleMFLadder n k)ᴴ *
            (lleMFLadder n (next k) - lleMFLadder n k) * lleMFLadder n j -
          lleMFLadder n j *
            ((lleMFLadder n (next k) - lleMFLadder n k)ᴴ *
              (lleMFLadder n (next k) - lleMFLadder n k)) =
        ((lleMFLadder n (next k) - lleMFLadder n k)ᴴ * lleMFLadder n j -
            lleMFLadder n j * (lleMFLadder n (next k) - lleMFLadder n k)ᴴ) *
          (lleMFLadder n (next k) - lleMFLadder n k) := by
          rw [mul_assoc, hcommX, ← mul_assoc, ← mul_assoc, ← sub_mul]
      _ = _ := by
        rw [hXd, sub_mul, ite_mul, ite_mul]
        simp only [zero_mul]
  simp only [hterm]
  rw [Finset.sum_sub_distrib]
  have hcond : ∀ k : ι, (next k = j) ↔ (k = next.symm j) := fun k =>
    next.apply_eq_iff_eq_symm_apply
  simp only [hcond]
  rw [Finset.sum_ite_eq_of_mem' Finset.univ (next.symm j) _ (Finset.mem_univ _),
    Finset.sum_ite_eq_of_mem' Finset.univ j _ (Finset.mem_univ _),
    Equiv.apply_symm_apply]
  have hP0 := lleMF_topProjSite_mul_ladder_self n j
  simp only [sub_mul, Matrix.smul_mul, one_mul, mul_sub, mul_add, hP0]
  module

/-- The Kerr-sum commutator with a cutoff ladder operator is exactly
`[Σ_k (a_k²)† a_k², a_j] = -2 a_j† a_j²`: the strict-cutoff Kerr correction
vanishes identically, which is why `B_j^cut` contains no Kerr term. -/
theorem lleMF_kerr_sum_comm_ladder (n : ℕ) (j : ι) :
    (∑ k, ((lleMFLadder n k) ^ 2)ᴴ * (lleMFLadder n k) ^ 2) * lleMFLadder n j -
      lleMFLadder n j * ∑ k, ((lleMFLadder n k) ^ 2)ᴴ * (lleMFLadder n k) ^ 2 =
      (-2 : ℂ) • ((lleMFLadder n j)ᴴ * lleMFLadder n j ^ 2) := by
  have hsq : ∀ k : ι, lleMFLadder n k ^ 2 =
      siteLift k (fockSandwichMatrix n lowering ^ 2) := by
    intro k
    rw [pow_two, pow_two, lleMFLadder, siteLift_mul]
  rw [Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]
  have h0 : ∀ k ∈ Finset.univ, k ≠ j →
      ((lleMFLadder n k) ^ 2)ᴴ * (lleMFLadder n k) ^ 2 * lleMFLadder n j -
        lleMFLadder n j * (((lleMFLadder n k) ^ 2)ᴴ * (lleMFLadder n k) ^ 2) = 0 := by
    intro k _ hkj
    have hl : ((lleMFLadder n k) ^ 2)ᴴ * (lleMFLadder n k) ^ 2 =
        siteLift k (((fockSandwichMatrix n lowering) ^ 2)ᴴ *
          (fockSandwichMatrix n lowering) ^ 2) := by
      rw [hsq, siteLift_conjTranspose, siteLift_mul]
    rw [hl, lleMFLadder, siteLift_comm hkj, sub_self]
  rw [Finset.sum_eq_single j h0 (fun h => absurd (Finset.mem_univ j) h)]
  have hatom : (lleMFLadder n j)ᴴ * lleMFLadder n j ^ 2 =
      siteLift j ((fockSandwichMatrix n lowering)ᴴ *
        fockSandwichMatrix n lowering ^ 2) := by
    rw [hsq, lleMFLadder, siteLift_conjTranspose, siteLift_mul]
  have hl : ((lleMFLadder n j) ^ 2)ᴴ * (lleMFLadder n j) ^ 2 =
      siteLift j (((fockSandwichMatrix n lowering) ^ 2)ᴴ *
        (fockSandwichMatrix n lowering) ^ 2) := by
    rw [hsq, siteLift_conjTranspose, siteLift_mul]
  rw [hatom, hl, lleMFLadder, siteLift_mul, siteLift_mul, ← siteLift_sub,
    lleMF_kerr_comm_lowering, siteLift_smul]

/-- The coherent-drive commutator with a cutoff ladder operator:
`[F drive, a_j] = i F_j (n_cut P_j^top - I)`.  This carries the strict-cutoff
drive correction `-n_cut F_j P_j^top` of `B_j^cut`. -/
theorem lleMF_drive_comm_ladder (n : ℕ) (F : ι → ℂ) (j : ι) :
    finiteLLEDrive (lleMFLadder n) F * lleMFLadder n j -
        lleMFLadder n j * finiteLLEDrive (lleMFLadder n) F =
      (Complex.I * F j) • ((n : ℂ) • lleMFTopProj n j - 1) := by
  unfold finiteLLEDrive
  rw [Matrix.smul_mul, Matrix.mul_smul, ← smul_sub, Finset.sum_mul, Finset.mul_sum,
    ← Finset.sum_sub_distrib]
  have hterm : ∀ k : ι,
      (F k • (lleMFLadder n k)ᴴ - (starRingEnd ℂ) (F k) • lleMFLadder n k) *
          lleMFLadder n j -
        lleMFLadder n j *
          (F k • (lleMFLadder n k)ᴴ - (starRingEnd ℂ) (F k) • lleMFLadder n k) =
      if k = j then F k • ((n : ℂ) • lleMFTopProj n j - 1) else 0 := by
    intro k
    calc (F k • (lleMFLadder n k)ᴴ - (starRingEnd ℂ) (F k) • lleMFLadder n k) *
            lleMFLadder n j -
          lleMFLadder n j *
            (F k • (lleMFLadder n k)ᴴ - (starRingEnd ℂ) (F k) • lleMFLadder n k) =
        F k • ((lleMFLadder n k)ᴴ * lleMFLadder n j -
            lleMFLadder n j * (lleMFLadder n k)ᴴ) -
          (starRingEnd ℂ) (F k) • (lleMFLadder n k * lleMFLadder n j -
            lleMFLadder n j * lleMFLadder n k) := by
          simp only [sub_mul, mul_sub, Matrix.smul_mul, Matrix.mul_smul, smul_sub]
          abel
      _ = if k = j then F k • ((n : ℂ) • lleMFTopProj n j - 1) else 0 := by
          rw [lleMF_adj_comm n k j, lleMF_ladder_mul_comm n k j, sub_self,
            smul_zero, sub_zero]
          by_cases hkj : k = j <;> simp [hkj]
  simp only [hterm]
  rw [Finset.sum_ite_eq_of_mem' Finset.univ j _ (Finset.mem_univ _), smul_smul]

private theorem lleMF_realSmul {p q : Type*} (r : ℝ) (M : Matrix p q ℂ) :
    r • M = ((r : ℝ) : ℂ) • M := by
  ext i j
  simp [Matrix.smul_apply, Complex.real_smul]

/-- The exact commutator of the finite LLE Hamiltonian
(Eq. `eq:lle-native-hamiltonian`, instantiated at the cutoff ladder family)
with the cutoff ladder operator `a_j`:

`[H_LLE, a_j] = -Δ a_j + (D₂/2h²) (a_{j+1} + a_{j-1} - 2 a_j
  - n_cut P_j^top (a_{j+1} + a_{j-1})) - g a_j† a_j² + i F_j (n_cut P_j^top - I)`.

Every strict-cutoff correction is proportional to the top-level projector,
matching the manuscript's `B_j^cut` bookkeeping; the Kerr term has no cutoff
correction. -/
theorem lleMF_hamiltonian_comm_ladder (n : ℕ) (next : Equiv.Perm ι)
    (delta dispersion g : ℝ) (F : ι → ℂ) (j : ι) :
    finiteLLEHamiltonian (lleMFLadder n) (⇑next) delta dispersion g F * lleMFLadder n j -
        lleMFLadder n j *
          finiteLLEHamiltonian (lleMFLadder n) (⇑next) delta dispersion g F =
      -((delta : ℂ) • lleMFLadder n j) +
        (dispersion : ℂ) •
          (lleMFLadder n (next j) + lleMFLadder n (next.symm j) -
            (2 : ℂ) • lleMFLadder n j -
            (n : ℂ) • (lleMFTopProj n j *
              (lleMFLadder n (next j) + lleMFLadder n (next.symm j)))) -
        (g : ℂ) • ((lleMFLadder n j)ᴴ * lleMFLadder n j ^ 2) +
        (Complex.I * F j) • ((n : ℂ) • lleMFTopProj n j - 1) := by
  unfold finiteLLEHamiltonian
  have hsplit : ∀ W X Y Z M : Matrix (ι → Fin n) (ι → Fin n) ℂ,
      (W + X + Y + Z) * M - M * (W + X + Y + Z) =
        (W * M - M * W) + (X * M - M * X) + (Y * M - M * Y) + (Z * M - M * Z) := by
    intros
    noncomm_ring
  rw [hsplit]
  have h1 : delta • finiteLLETotalNumber (lleMFLadder n) * lleMFLadder n j -
      lleMFLadder n j * (delta • finiteLLETotalNumber (lleMFLadder n)) =
      -((delta : ℂ) • lleMFLadder n j) := by
    rw [Matrix.smul_mul, Matrix.mul_smul, ← smul_sub, lleMF_totalNumber_comm_ladder,
      lleMF_realSmul, smul_neg]
  have h2 : dispersion • (∑ k, (lleMFLadder n (next k) - lleMFLadder n k)ᴴ *
        (lleMFLadder n (next k) - lleMFLadder n k)) * lleMFLadder n j -
      lleMFLadder n j * (dispersion • ∑ k, (lleMFLadder n (next k) - lleMFLadder n k)ᴴ *
        (lleMFLadder n (next k) - lleMFLadder n k)) =
      (dispersion : ℂ) •
        (lleMFLadder n (next j) + lleMFLadder n (next.symm j) -
          (2 : ℂ) • lleMFLadder n j -
          (n : ℂ) • (lleMFTopProj n j *
            (lleMFLadder n (next j) + lleMFLadder n (next.symm j)))) := by
    rw [Matrix.smul_mul, Matrix.mul_smul, ← smul_sub,
      lleMF_dispersion_comm_ladder n next j, lleMF_realSmul]
  have h3 : (g / 2) • (∑ k, ((lleMFLadder n k) ^ 2)ᴴ * (lleMFLadder n k) ^ 2) *
        lleMFLadder n j -
      lleMFLadder n j *
        ((g / 2) • ∑ k, ((lleMFLadder n k) ^ 2)ᴴ * (lleMFLadder n k) ^ 2) =
      -((g : ℂ) • ((lleMFLadder n j)ᴴ * lleMFLadder n j ^ 2)) := by
    rw [Matrix.smul_mul, Matrix.mul_smul, ← smul_sub, lleMF_kerr_sum_comm_ladder,
      lleMF_realSmul, smul_smul, ← neg_smul]
    congr 1
    push_cast
    ring
  rw [h1, h2, h3, lleMF_drive_comm_ladder]
  abel

/-- The exact loss anticommutator `{L_loss, a_j} = κ N_tot a_j + (κ/2) a_j`,
from `[N_tot, a_j] = -a_j`.  This is the anticommutator entering
Eq. `eq:lle-normalized-expectation` at `O = a_j`. -/
theorem lleMF_loss_anticomm_ladder (n : ℕ) (kappa : ℝ) (j : ι) :
    finiteLLELoss (lleMFLadder n) kappa * lleMFLadder n j +
        lleMFLadder n j * finiteLLELoss (lleMFLadder n) kappa =
      (kappa : ℂ) • (finiteLLETotalNumber (lleMFLadder n) * lleMFLadder n j) +
        ((kappa : ℂ) / 2) • lleMFLadder n j := by
  have hAN : lleMFLadder n j * finiteLLETotalNumber (lleMFLadder n) =
      finiteLLETotalNumber (lleMFLadder n) * lleMFLadder n j + lleMFLadder n j := by
    have h := lleMF_totalNumber_comm_ladder n j
    have h2 : lleMFLadder n j * finiteLLETotalNumber (lleMFLadder n) -
        finiteLLETotalNumber (lleMFLadder n) * lleMFLadder n j = lleMFLadder n j := by
      rw [← neg_sub, h, neg_neg]
    rw [sub_eq_iff_eq_add] at h2
    rw [h2, add_comm]
  unfold finiteLLELoss
  rw [lleMF_realSmul, Matrix.smul_mul, Matrix.mul_smul, hAN]
  push_cast
  module

end LLEMeanFieldMultimode

/-! ### Normalized expectations of cutoff matrices -/

section LLEMeanFieldExpectation

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]

private theorem lleMF_nexp_add (S T : E →L[ℂ] E) (ψ : E) :
    normalizedExpectation (S + T) ψ =
      normalizedExpectation S ψ + normalizedExpectation T ψ := by
  unfold normalizedExpectation
  rw [add_apply, inner_add_right, add_div]

private theorem lleMF_nexp_sub (S T : E →L[ℂ] E) (ψ : E) :
    normalizedExpectation (S - T) ψ =
      normalizedExpectation S ψ - normalizedExpectation T ψ := by
  unfold normalizedExpectation
  rw [sub_apply, inner_sub_right, sub_div]

private theorem lleMF_nexp_neg (S : E →L[ℂ] E) (ψ : E) :
    normalizedExpectation (-S) ψ = -normalizedExpectation S ψ := by
  unfold normalizedExpectation
  rw [neg_apply, inner_neg_right, neg_div]

private theorem lleMF_nexp_smul (c : ℂ) (S : E →L[ℂ] E) (ψ : E) :
    normalizedExpectation (c • S) ψ = c * normalizedExpectation S ψ := by
  unfold normalizedExpectation
  rw [smul_apply, inner_smul_right, mul_div_assoc]

private theorem lleMF_nexp_one {ψ : E} (hψ : ψ ≠ 0) :
    normalizedExpectation (1 : E →L[ℂ] E) ψ = 1 := by
  unfold normalizedExpectation
  rw [one_apply_eq_self, div_self]
  rw [Ne, inner_self_eq_zero]
  exact hψ

end LLEMeanFieldExpectation

section LLEMeanFieldMoments

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The normalized expectation `⟨O⟩ = ⟨ψ|O|ψ⟩ / ⟨ψ|ψ⟩` of a cutoff matrix in a
(not necessarily normalized) multimode state, manuscript
Eq. `eq:lle-normalized-expectation` left-hand side at fixed time. -/
def lleMF_expect {m : Type*} [Fintype m] [DecidableEq m]
    (M : Matrix m m ℂ) (ψ : EuclideanSpace ℂ m) : ℂ :=
  normalizedExpectation (Matrix.toEuclideanCLM (𝕜 := ℂ) M) ψ

/-- The normalized first moment `α_j = ⟨a_j⟩` of the postselected branch. -/
def lleMF_alpha (n : ℕ) (j : ι) (ψ : EuclideanSpace ℂ (ι → Fin n)) : ℂ :=
  lleMF_expect (lleMFLadder n j) ψ

/-- The loss connected moment
`C_j^loss = ⟨N_tot a_j⟩ - ⟨N_tot⟩ α_j` of the manuscript. -/
def lleMF_Closs (n : ℕ) (j : ι) (ψ : EuclideanSpace ℂ (ι → Fin n)) : ℂ :=
  lleMF_expect (finiteLLETotalNumber (lleMFLadder n) * lleMFLadder n j) ψ -
    lleMF_expect (finiteLLETotalNumber (lleMFLadder n)) ψ * lleMF_alpha n j ψ

/-- The Kerr cubic connected moment
`C_j^(3) = ⟨a_j† a_j a_j⟩ - |α_j|² α_j` of the manuscript
(with `|α|² α = conj α · α · α`). -/
def lleMF_C3 (n : ℕ) (j : ι) (ψ : EuclideanSpace ℂ (ι → Fin n)) : ℂ :=
  lleMF_expect ((lleMFLadder n j)ᴴ * lleMFLadder n j ^ 2) ψ -
    (starRingEnd ℂ) (lleMF_alpha n j ψ) * lleMF_alpha n j ψ * lleMF_alpha n j ψ

/-- The cutoff-boundary term of the manuscript:
`B_j^cut = -n_cut F_j ⟨P_j^top⟩ - i (D₂ n_cut / 2h²) ⟨P_j^top (a_{j+1} + a_{j-1})⟩`,
with `dispersion` abbreviating `D₂/(2h²)`. -/
def lleMF_Bcut (n : ℕ) (next : Equiv.Perm ι) (dispersion : ℝ) (F : ι → ℂ)
    (j : ι) (ψ : EuclideanSpace ℂ (ι → Fin n)) : ℂ :=
  -(n : ℂ) * F j * lleMF_expect (lleMFTopProj n j) ψ -
    Complex.I * ((dispersion : ℂ) * (n : ℂ)) *
      lleMF_expect (lleMFTopProj n j *
        (lleMFLadder n (next j) + lleMFLadder n (next.symm j))) ψ

/-- Manuscript Eq. `eq:lle-first-moment-exact`, the exact cutoff first-moment
equation.  If the (unnormalized) state obeys `ψ' = -(L_loss + i H_LLE) ψ`
with the finite LLE loss and Hamiltonian instantiated at the cutoff ladder
family, then the normalized first moment `α_j` satisfies

`α̇_j = -(κ/2) α_j - iΔ α_j + i (D₂/2h²)(α_{j+1} - 2α_j + α_{j-1})
  - i g |α_j|² α_j + F_j - κ C_j^loss - i g C_j^(3) + B_j^cut`,

with the connected moments and the cutoff-boundary term defined exactly as in
the manuscript.  No mean-field closure is used. -/
theorem lleMF_first_moment_exact (n : ℕ) (next : Equiv.Perm ι)
    (delta dispersion g kappa : ℝ) (F : ι → ℂ) (j : ι)
    (ψ : ℝ → EuclideanSpace ℂ (ι → Fin n)) (t : ℝ)
    (hψ0 : ψ t ≠ 0)
    (hψ : HasDerivAt ψ
      (-(((Matrix.toEuclideanCLM (𝕜 := ℂ) (finiteLLELoss (lleMFLadder n) kappa) +
          Complex.I • Matrix.toEuclideanCLM (𝕜 := ℂ)
            (finiteLLEHamiltonian (lleMFLadder n) (⇑next) delta dispersion g F) :
          EuclideanSpace ℂ (ι → Fin n) →L[ℂ] EuclideanSpace ℂ (ι → Fin n)))
        (ψ t))) t) :
    HasDerivAt (fun s => lleMF_alpha n j (ψ s))
      (-((kappa : ℂ) / 2) * lleMF_alpha n j (ψ t) -
        Complex.I * (delta : ℂ) * lleMF_alpha n j (ψ t) +
        Complex.I * (dispersion : ℂ) *
          (lleMF_alpha n (next j) (ψ t) - 2 * lleMF_alpha n j (ψ t) +
            lleMF_alpha n (next.symm j) (ψ t)) -
        Complex.I * (g : ℂ) *
          ((starRingEnd ℂ) (lleMF_alpha n j (ψ t)) * lleMF_alpha n j (ψ t) *
            lleMF_alpha n j (ψ t)) +
        F j -
        (kappa : ℂ) * lleMF_Closs n j (ψ t) -
        Complex.I * (g : ℂ) * lleMF_C3 n j (ψ t) +
        lleMF_Bcut n next dispersion F j (ψ t)) t := by
  have hNherm : (finiteLLETotalNumber (lleMFLadder (ι := ι) n)).IsHermitian := by
    unfold finiteLLETotalNumber
    exact (isSelfAdjoint_sum Finset.univ fun k _ =>
      (Matrix.isHermitian_conjTranspose_mul_self (lleMFLadder n k)).isSelfAdjoint).isHermitian
  have hLherm : (finiteLLELoss (lleMFLadder (ι := ι) n) kappa).IsHermitian := by
    unfold finiteLLELoss
    exact hNherm.smul (IsSelfAdjoint.all (kappa / 2))
  have hHherm := finiteLLEHamiltonian_isHermitian (lleMFLadder (ι := ι) n) (⇑next)
    delta dispersion g F
  have hLsym : ((Matrix.toEuclideanCLM (𝕜 := ℂ)
      (finiteLLELoss (lleMFLadder n) kappa) :
        EuclideanSpace ℂ (ι → Fin n) →L[ℂ] EuclideanSpace ℂ (ι → Fin n)) :
        EuclideanSpace ℂ (ι → Fin n) →ₗ[ℂ] EuclideanSpace ℂ (ι → Fin n)).IsSymmetric := by
    have hCLM : IsSelfAdjoint (Matrix.toEuclideanCLM (𝕜 := ℂ)
        (finiteLLELoss (lleMFLadder n) kappa)) :=
      hLherm.isSelfAdjoint.map (Matrix.toEuclideanCLM (n := ι → Fin n) (𝕜 := ℂ))
    exact hCLM.isSymmetric
  have hHsym : ((Matrix.toEuclideanCLM (𝕜 := ℂ)
      (finiteLLEHamiltonian (lleMFLadder n) (⇑next) delta dispersion g F) :
        EuclideanSpace ℂ (ι → Fin n) →L[ℂ] EuclideanSpace ℂ (ι → Fin n)) :
        EuclideanSpace ℂ (ι → Fin n) →ₗ[ℂ] EuclideanSpace ℂ (ι → Fin n)).IsSymmetric := by
    have hCLM : IsSelfAdjoint (Matrix.toEuclideanCLM (𝕜 := ℂ)
        (finiteLLEHamiltonian (lleMFLadder n) (⇑next) delta dispersion g F)) :=
      hHherm.isSelfAdjoint.map (Matrix.toEuclideanCLM (n := ι → Fin n) (𝕜 := ℂ))
    exact hCLM.isSymmetric
  have hderiv := normalizedExpectation_hasDerivAt
    (Matrix.toEuclideanCLM (𝕜 := ℂ) (finiteLLELoss (lleMFLadder n) kappa))
    (Matrix.toEuclideanCLM (𝕜 := ℂ)
      (finiteLLEHamiltonian (lleMFLadder n) (⇑next) delta dispersion g F))
    (Matrix.toEuclideanCLM (𝕜 := ℂ) (lleMFLadder n j))
    ψ t hψ0 hψ hLsym hHsym
  have hHO := congrArg (Matrix.toEuclideanCLM (𝕜 := ℂ))
    (lleMF_hamiltonian_comm_ladder n next delta dispersion g F j)
  rw [map_sub, map_mul, map_mul] at hHO
  have hLO := congrArg (Matrix.toEuclideanCLM (𝕜 := ℂ))
    (lleMF_loss_anticomm_ladder n kappa j)
  rw [map_add, map_mul, map_mul] at hLO
  have hLossEq : finiteLLELoss (lleMFLadder (ι := ι) n) kappa =
      ((kappa : ℂ) / 2) • finiteLLETotalNumber (lleMFLadder n) := by
    unfold finiteLLELoss
    rw [lleMF_realSmul]
    congr 1
    push_cast
    ring
  rw [hHO, hLO, hLossEq] at hderiv
  simp only [map_add, map_sub, map_neg, map_smul, map_one] at hderiv
  simp only [lleMF_nexp_add, lleMF_nexp_sub, lleMF_nexp_neg, lleMF_nexp_smul,
    lleMF_nexp_one hψ0] at hderiv
  refine hderiv.congr_deriv ?_
  simp only [lleMF_alpha, lleMF_Closs, lleMF_C3, lleMF_Bcut, lleMF_expect]
  linear_combination (F j * ((n : ℂ) *
    normalizedExpectation (Matrix.toEuclideanCLM (𝕜 := ℂ) (lleMFTopProj n j)) (ψ t) - 1)) *
    Complex.I_sq

/-- Manuscript Eq. `eq:lle-mean-field-discrete`, the conditional mean-field
corollary: if the loss connected moment, the Kerr cubic connected moment, and
the cutoff-boundary term all vanish at time `t`, the normalized first moment
obeys the semidiscrete LLE

`α̇_j = -(κ/2) α_j - iΔ α_j + i (D₂/2h²)(α_{j+1} - 2α_j + α_{j-1})
  - i g |α_j|² α_j + F_j`. -/
theorem lleMF_mean_field_discrete (n : ℕ) (next : Equiv.Perm ι)
    (delta dispersion g kappa : ℝ) (F : ι → ℂ) (j : ι)
    (ψ : ℝ → EuclideanSpace ℂ (ι → Fin n)) (t : ℝ)
    (hψ0 : ψ t ≠ 0)
    (hψ : HasDerivAt ψ
      (-(((Matrix.toEuclideanCLM (𝕜 := ℂ) (finiteLLELoss (lleMFLadder n) kappa) +
          Complex.I • Matrix.toEuclideanCLM (𝕜 := ℂ)
            (finiteLLEHamiltonian (lleMFLadder n) (⇑next) delta dispersion g F) :
          EuclideanSpace ℂ (ι → Fin n) →L[ℂ] EuclideanSpace ℂ (ι → Fin n)))
        (ψ t))) t)
    (hCloss : lleMF_Closs n j (ψ t) = 0)
    (hC3 : lleMF_C3 n j (ψ t) = 0)
    (hBcut : lleMF_Bcut n next dispersion F j (ψ t) = 0) :
    HasDerivAt (fun s => lleMF_alpha n j (ψ s))
      (-((kappa : ℂ) / 2) * lleMF_alpha n j (ψ t) -
        Complex.I * (delta : ℂ) * lleMF_alpha n j (ψ t) +
        Complex.I * (dispersion : ℂ) *
          (lleMF_alpha n (next j) (ψ t) - 2 * lleMF_alpha n j (ψ t) +
            lleMF_alpha n (next.symm j) (ψ t)) -
        Complex.I * (g : ℂ) *
          ((starRingEnd ℂ) (lleMF_alpha n j (ψ t)) * lleMF_alpha n j (ψ t) *
            lleMF_alpha n j (ψ t)) +
        F j) t := by
  have h := lleMF_first_moment_exact n next delta dispersion g kappa F j ψ t hψ0 hψ
  rw [hCloss, hC3, hBcut] at h
  simpa using h

end LLEMeanFieldMoments

end

end Cvdv
