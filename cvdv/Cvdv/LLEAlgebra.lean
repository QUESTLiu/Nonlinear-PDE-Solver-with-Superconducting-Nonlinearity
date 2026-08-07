import Cvdv.QuadraticClosure
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# Lugiato--Lefever algebraic identities

This file formalizes Eqs. (eq:lle-kerr), (eq:lle-number-conserving-commutators),
and (eq:lle-mixed-commutator) for a finite indexed family satisfying the multimode CCR.
It does not construct the multimode cutoff or prove operator norm bounds.
-/

namespace Cvdv

open scoped BigOperators

noncomputable section

section LLEAlgebra

variable {ι R : Type*} [Fintype ι] [DecidableEq ι] [Ring R] [Algebra ℂ R]

/-- Canonical commutation relations for a finite family of modes. -/
structure MultimodeCCR (a adag : ι → R) : Prop where
  ann_creation : ∀ j k, a j * adag k = adag k * a j + if j = k then 1 else 0
  ann_ann : ∀ j k, a j * a k = a k * a j
  creation_creation : ∀ j k, adag j * adag k = adag k * adag j

def lleNumber (a adag : ι → R) (j : ι) : R := adag j * a j

def lleTotalNumber (a adag : ι → R) : R := ∑ j, lleNumber a adag j

def lleDetuning (a adag : ι → R) (delta : ℂ) : R :=
  delta • lleTotalNumber a adag

def lleDispersion (a adag : ι → R) (next : ι → ι) (dispersion : ℂ) : R :=
  dispersion • ∑ j, (adag (next j) - adag j) * (a (next j) - a j)

def lleKerr (a adag : ι → R) (g : ℂ) : R :=
  (g / 2) • ∑ j, adag j ^ 2 * a j ^ 2

def lleDrive (a adag : ι → R) (F : ι → ℂ) : R :=
  Complex.I • ∑ j, ((F j) • adag j - (starRingEnd ℂ) (F j) • a j)

def lleLoss (a adag : ι → R) (kappa : ℂ) : R :=
  (kappa / 2) • lleTotalNumber a adag

def lleHamiltonian (a adag : ι → R) (next : ι → ι)
    (delta dispersion g : ℂ) (F : ι → ℂ) : R :=
  lleDetuning a adag delta + lleDispersion a adag next dispersion
    + lleKerr a adag g + lleDrive a adag F

omit [Fintype ι] [Algebra ℂ R] in
private theorem same_mode_ccr {a adag : ι → R} (h : MultimodeCCR a adag) (j : ι) :
    a j * adag j = adag j * a j + 1 := by
  simpa using h.ann_creation j j

omit [Fintype ι] [Algebra ℂ R] in
/-- At one mode, `a†²a² = n(n - 1)`. -/
theorem lle_kerr_site_eq_number_falling (a adag : ι → R) (h : MultimodeCCR a adag)
    (j : ι) :
    adag j ^ 2 * a j ^ 2 =
      lleNumber a adag j * (lleNumber a adag j - 1) := by
  have hsquare := number_sq (a j) (adag j) (same_mode_ccr h j)
  rw [lleNumber]
  calc
    adag j ^ 2 * a j ^ 2 = (adag j * a j) * (adag j * a j) - adag j * a j := by
      rw [hsquare]
      noncomm_ring
    _ = (adag j * a j) * (adag j * a j - 1) := by noncomm_ring

/-- The multimode Kerr sum has the number-operator form in Eq. (eq:lle-kerr). -/
theorem lle_kerr_eq_number_falling (a adag : ι → R) (h : MultimodeCCR a adag)
    (g : ℂ) :
    lleKerr a adag g =
      (g / 2) • ∑ j, lleNumber a adag j * (lleNumber a adag j - 1) := by
  unfold lleKerr
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  exact lle_kerr_site_eq_number_falling a adag h j

omit [Fintype ι] [Algebra ℂ R] in
private theorem mode_number_comm_lowering (a adag : ι → R) (h : MultimodeCCR a adag)
    (j k : ι) :
    lleNumber a adag j * a k - a k * lleNumber a adag j =
      if j = k then -a k else 0 := by
  by_cases hjk : j = k
  · subst k
    simp only [if_pos]
    simpa [lleNumber] using comm_number_ann (a j) (adag j) (same_mode_ccr h j)
  · rw [if_neg hjk]
    have hcross : a k * adag j = adag j * a k := by
      simpa [if_neg (Ne.symm hjk)] using h.ann_creation k j
    rw [lleNumber]
    calc
      (adag j * a j) * a k - a k * (adag j * a j) =
          adag j * (a j * a k) - (a k * adag j) * a j := by noncomm_ring
      _ = adag j * (a k * a j) - (adag j * a k) * a j := by
        rw [h.ann_ann j k, hcross]
      _ = 0 := by noncomm_ring

omit [Fintype ι] [Algebra ℂ R] in
private theorem mode_number_comm_creation (a adag : ι → R) (h : MultimodeCCR a adag)
    (j k : ι) :
    lleNumber a adag j * adag k - adag k * lleNumber a adag j =
      if j = k then adag k else 0 := by
  by_cases hjk : j = k
  · subst k
    simp only [if_pos]
    simpa [lleNumber] using comm_number_creation (a j) (adag j) (same_mode_ccr h j)
  · rw [if_neg hjk]
    have hcross : a j * adag k = adag k * a j := by
      simpa [if_neg hjk] using h.ann_creation j k
    rw [lleNumber]
    calc
      (adag j * a j) * adag k - adag k * (adag j * a j) =
          adag j * (a j * adag k) - (adag k * adag j) * a j := by noncomm_ring
      _ = adag j * (adag k * a j) - (adag j * adag k) * a j := by
        rw [hcross, h.creation_creation k j]
      _ = 0 := by noncomm_ring

omit [Algebra ℂ R] in
/-- The total number commutator with a lowering operator is `-aₖ`. -/
theorem lle_total_number_comm_lowering (a adag : ι → R) (h : MultimodeCCR a adag)
    (k : ι) :
    lleTotalNumber a adag * a k - a k * lleTotalNumber a adag = -a k := by
  rw [lleTotalNumber, Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]
  simp_rw [mode_number_comm_lowering a adag h]
  simp

omit [Algebra ℂ R] in
/-- The total number commutator with a creation operator is `a†ₖ`. -/
theorem lle_total_number_comm_creation (a adag : ι → R) (h : MultimodeCCR a adag)
    (k : ι) :
    lleTotalNumber a adag * adag k - adag k * lleTotalNumber a adag = adag k := by
  rw [lleTotalNumber, Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]
  simp_rw [mode_number_comm_creation a adag h]
  simp

omit [Algebra ℂ R] in
/-- Every number-conserving bilinear `a†ⱼaₖ` commutes with total number. -/
theorem lle_total_number_comm_bilinear (a adag : ι → R) (h : MultimodeCCR a adag)
    (j k : ι) :
    lleTotalNumber a adag * (adag j * a k) -
        (adag j * a k) * lleTotalNumber a adag = 0 := by
  calc
    lleTotalNumber a adag * (adag j * a k) -
          (adag j * a k) * lleTotalNumber a adag =
        (lleTotalNumber a adag * adag j - adag j * lleTotalNumber a adag) * a k
          + adag j * (lleTotalNumber a adag * a k - a k * lleTotalNumber a adag) := by
            noncomm_ring
    _ = adag j * a k + adag j * (-a k) := by
      rw [lle_total_number_comm_creation a adag h j,
        lle_total_number_comm_lowering a adag h k]
    _ = 0 := by noncomm_ring

omit [DecidableEq ι] in
private theorem comm_scalar_sum_eq_zero (N : R) (c : ℂ) (f : ι → R)
    (hf : ∀ j, N * f j - f j * N = 0) :
    N * (c • ∑ j, f j) - (c • ∑ j, f j) * N = 0 := by
  calc
    N * (c • ∑ j, f j) - (c • ∑ j, f j) * N =
        c • (N * (∑ j, f j) - (∑ j, f j) * N) := by
          simp only [mul_smul_comm, smul_mul_assoc, smul_sub]
    _ = c • ∑ j, (N * f j - f j * N) := by
      rw [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_sub_distrib]
    _ = 0 := by simp [hf]

omit [DecidableEq ι] in
/-- Total number commutes with the detuning term. -/
theorem lle_total_number_comm_detuning (a adag : ι → R) (delta : ℂ) :
    lleTotalNumber a adag * lleDetuning a adag delta -
        lleDetuning a adag delta * lleTotalNumber a adag = 0 := by
  unfold lleDetuning
  calc
    lleTotalNumber a adag * (delta • lleTotalNumber a adag) -
          (delta • lleTotalNumber a adag) * lleTotalNumber a adag =
        delta • (lleTotalNumber a adag * lleTotalNumber a adag -
          lleTotalNumber a adag * lleTotalNumber a adag) := by
            simp only [mul_smul_comm, smul_mul_assoc, smul_sub]
    _ = 0 := by simp

omit [Algebra ℂ R] in
private theorem lle_total_number_comm_dispersion_site
    (a adag : ι → R) (h : MultimodeCCR a adag) (next : ι → ι) (j : ι) :
    lleTotalNumber a adag * ((adag (next j) - adag j) * (a (next j) - a j)) -
        ((adag (next j) - adag j) * (a (next j) - a j)) *
          lleTotalNumber a adag = 0 := by
  have hcreation :
      lleTotalNumber a adag * (adag (next j) - adag j) -
          (adag (next j) - adag j) * lleTotalNumber a adag =
        adag (next j) - adag j := by
    calc
      lleTotalNumber a adag * (adag (next j) - adag j) -
            (adag (next j) - adag j) * lleTotalNumber a adag =
          (lleTotalNumber a adag * adag (next j) -
              adag (next j) * lleTotalNumber a adag) -
            (lleTotalNumber a adag * adag j - adag j * lleTotalNumber a adag) := by
              noncomm_ring
      _ = adag (next j) - adag j := by
        rw [lle_total_number_comm_creation a adag h (next j),
          lle_total_number_comm_creation a adag h j]
  have hlowering :
      lleTotalNumber a adag * (a (next j) - a j) -
          (a (next j) - a j) * lleTotalNumber a adag =
        -(a (next j) - a j) := by
    calc
      lleTotalNumber a adag * (a (next j) - a j) -
            (a (next j) - a j) * lleTotalNumber a adag =
          (lleTotalNumber a adag * a (next j) -
              a (next j) * lleTotalNumber a adag) -
            (lleTotalNumber a adag * a j - a j * lleTotalNumber a adag) := by
              noncomm_ring
      _ = -a (next j) - (-a j) := by
        rw [lle_total_number_comm_lowering a adag h (next j),
          lle_total_number_comm_lowering a adag h j]
      _ = -(a (next j) - a j) := by noncomm_ring
  calc
    lleTotalNumber a adag * ((adag (next j) - adag j) * (a (next j) - a j)) -
          ((adag (next j) - adag j) * (a (next j) - a j)) *
            lleTotalNumber a adag =
        (lleTotalNumber a adag * (adag (next j) - adag j) -
            (adag (next j) - adag j) * lleTotalNumber a adag) *
              (a (next j) - a j)
          + (adag (next j) - adag j) *
            (lleTotalNumber a adag * (a (next j) - a j) -
              (a (next j) - a j) * lleTotalNumber a adag) := by
                noncomm_ring
    _ = (adag (next j) - adag j) * (a (next j) - a j) +
        (adag (next j) - adag j) * (-(a (next j) - a j)) := by
          rw [hcreation, hlowering]
    _ = 0 := by noncomm_ring

/-- Total number commutes with the nearest-neighbor dispersion sum. -/
theorem lle_total_number_comm_dispersion (a adag : ι → R) (h : MultimodeCCR a adag)
    (next : ι → ι) (dispersion : ℂ) :
    lleTotalNumber a adag * lleDispersion a adag next dispersion -
        lleDispersion a adag next dispersion * lleTotalNumber a adag = 0 := by
  unfold lleDispersion
  exact comm_scalar_sum_eq_zero (lleTotalNumber a adag) dispersion
    (fun j ↦ (adag (next j) - adag j) * (a (next j) - a j))
    (lle_total_number_comm_dispersion_site a adag h next)

omit [Algebra ℂ R] in
private theorem lle_total_number_comm_kerr_site
    (a adag : ι → R) (h : MultimodeCCR a adag) (j : ι) :
    lleTotalNumber a adag * (adag j ^ 2 * a j ^ 2) -
        (adag j ^ 2 * a j ^ 2) * lleTotalNumber a adag = 0 := by
  have hnumber :
      lleTotalNumber a adag * lleNumber a adag j -
          lleNumber a adag j * lleTotalNumber a adag = 0 := by
    simpa [lleNumber] using lle_total_number_comm_bilinear a adag h j j
  have hfalling :
      lleTotalNumber a adag * (lleNumber a adag j - 1) -
          (lleNumber a adag j - 1) * lleTotalNumber a adag = 0 := by
    calc
      lleTotalNumber a adag * (lleNumber a adag j - 1) -
            (lleNumber a adag j - 1) * lleTotalNumber a adag =
          lleTotalNumber a adag * lleNumber a adag j -
            lleNumber a adag j * lleTotalNumber a adag := by noncomm_ring
      _ = 0 := hnumber
  rw [lle_kerr_site_eq_number_falling a adag h j]
  calc
    lleTotalNumber a adag *
          (lleNumber a adag j * (lleNumber a adag j - 1)) -
        (lleNumber a adag j * (lleNumber a adag j - 1)) *
          lleTotalNumber a adag =
      (lleTotalNumber a adag * lleNumber a adag j -
          lleNumber a adag j * lleTotalNumber a adag) *
            (lleNumber a adag j - 1)
        + lleNumber a adag j *
          (lleTotalNumber a adag * (lleNumber a adag j - 1) -
            (lleNumber a adag j - 1) * lleTotalNumber a adag) := by
              noncomm_ring
    _ = 0 := by rw [hnumber, hfalling]; noncomm_ring

/-- Total number commutes with the multimode Kerr term. -/
theorem lle_total_number_comm_kerr (a adag : ι → R) (h : MultimodeCCR a adag)
    (g : ℂ) :
    lleTotalNumber a adag * lleKerr a adag g -
        lleKerr a adag g * lleTotalNumber a adag = 0 := by
  unfold lleKerr
  exact comm_scalar_sum_eq_zero (lleTotalNumber a adag) (g / 2)
    (fun j ↦ adag j ^ 2 * a j ^ 2)
    (lle_total_number_comm_kerr_site a adag h)

/-- Manuscript Eq. (eq:lle-number-conserving-commutators). -/
theorem lle_number_conserving_commutators (a adag : ι → R) (h : MultimodeCCR a adag)
    (next : ι → ι) (delta dispersion g : ℂ) :
    (lleTotalNumber a adag * lleDetuning a adag delta -
        lleDetuning a adag delta * lleTotalNumber a adag = 0) ∧
    (lleTotalNumber a adag * lleDispersion a adag next dispersion -
        lleDispersion a adag next dispersion * lleTotalNumber a adag = 0) ∧
    (lleTotalNumber a adag * lleKerr a adag g -
        lleKerr a adag g * lleTotalNumber a adag = 0) := by
  exact ⟨lle_total_number_comm_detuning a adag delta,
    lle_total_number_comm_dispersion a adag h next dispersion,
    lle_total_number_comm_kerr a adag h g⟩

private theorem lle_total_number_comm_drive_site
    (a adag : ι → R) (h : MultimodeCCR a adag) (F : ι → ℂ) (j : ι) :
    lleTotalNumber a adag *
          ((F j) • adag j - (starRingEnd ℂ) (F j) • a j) -
        ((F j) • adag j - (starRingEnd ℂ) (F j) • a j) *
          lleTotalNumber a adag =
      (F j) • adag j + (starRingEnd ℂ) (F j) • a j := by
  calc
    lleTotalNumber a adag *
          ((F j) • adag j - (starRingEnd ℂ) (F j) • a j) -
        ((F j) • adag j - (starRingEnd ℂ) (F j) • a j) *
          lleTotalNumber a adag =
      (F j) • (lleTotalNumber a adag * adag j -
          adag j * lleTotalNumber a adag) -
        (starRingEnd ℂ) (F j) • (lleTotalNumber a adag * a j -
          a j * lleTotalNumber a adag) := by
            simp only [mul_sub, sub_mul, mul_smul_comm, smul_mul_assoc]
            module
    _ = (F j) • adag j - (starRingEnd ℂ) (F j) • (-a j) := by
      rw [lle_total_number_comm_creation a adag h j,
        lle_total_number_comm_lowering a adag h j]
    _ = (F j) • adag j + (starRingEnd ℂ) (F j) • a j := by module

/-- The coherent drive is the only term whose total-number commutator is nonzero. -/
theorem lle_total_number_comm_drive (a adag : ι → R) (h : MultimodeCCR a adag)
    (F : ι → ℂ) :
    lleTotalNumber a adag * lleDrive a adag F -
        lleDrive a adag F * lleTotalNumber a adag =
      Complex.I • ∑ j, ((F j) • adag j + (starRingEnd ℂ) (F j) • a j) := by
  unfold lleDrive
  calc
    lleTotalNumber a adag *
          (Complex.I • ∑ j, ((F j) • adag j - (starRingEnd ℂ) (F j) • a j)) -
        (Complex.I • ∑ j, ((F j) • adag j - (starRingEnd ℂ) (F j) • a j)) *
          lleTotalNumber a adag =
      Complex.I •
        (lleTotalNumber a adag *
            (∑ j, ((F j) • adag j - (starRingEnd ℂ) (F j) • a j)) -
          (∑ j, ((F j) • adag j - (starRingEnd ℂ) (F j) • a j)) *
            lleTotalNumber a adag) := by
              simp only [mul_smul_comm, smul_mul_assoc, smul_sub]
    _ = Complex.I • ∑ j,
        (lleTotalNumber a adag *
            ((F j) • adag j - (starRingEnd ℂ) (F j) • a j) -
          ((F j) • adag j - (starRingEnd ℂ) (F j) • a j) *
            lleTotalNumber a adag) := by
              rw [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_sub_distrib]
    _ = Complex.I • ∑ j,
        ((F j) • adag j + (starRingEnd ℂ) (F j) • a j) := by
          congr 1
          apply Finset.sum_congr rfl
          intro j _
          exact lle_total_number_comm_drive_site a adag h F j

private theorem lle_total_number_comm_hamiltonian_eq_drive
    (a adag : ι → R) (h : MultimodeCCR a adag) (next : ι → ι)
    (delta dispersion g : ℂ) (F : ι → ℂ) :
    lleTotalNumber a adag * lleHamiltonian a adag next delta dispersion g F -
        lleHamiltonian a adag next delta dispersion g F * lleTotalNumber a adag =
      lleTotalNumber a adag * lleDrive a adag F -
        lleDrive a adag F * lleTotalNumber a adag := by
  have hdetuning := lle_total_number_comm_detuning a adag delta
  have hdispersion := lle_total_number_comm_dispersion a adag h next dispersion
  have hkerr := lle_total_number_comm_kerr a adag h g
  unfold lleHamiltonian
  calc
    lleTotalNumber a adag *
          (lleDetuning a adag delta + lleDispersion a adag next dispersion +
            lleKerr a adag g + lleDrive a adag F) -
        (lleDetuning a adag delta + lleDispersion a adag next dispersion +
            lleKerr a adag g + lleDrive a adag F) * lleTotalNumber a adag =
      (lleTotalNumber a adag * lleDetuning a adag delta -
          lleDetuning a adag delta * lleTotalNumber a adag) +
        (lleTotalNumber a adag * lleDispersion a adag next dispersion -
          lleDispersion a adag next dispersion * lleTotalNumber a adag) +
        (lleTotalNumber a adag * lleKerr a adag g -
          lleKerr a adag g * lleTotalNumber a adag) +
        (lleTotalNumber a adag * lleDrive a adag F -
          lleDrive a adag F * lleTotalNumber a adag) := by noncomm_ring
    _ = lleTotalNumber a adag * lleDrive a adag F -
        lleDrive a adag F * lleTotalNumber a adag := by
          rw [hdetuning, hdispersion, hkerr]
          noncomm_ring

/-- The loss commutator with the full Hamiltonian equals its drive contribution. -/
theorem lle_loss_comm_hamiltonian_eq_drive
    (a adag : ι → R) (h : MultimodeCCR a adag) (next : ι → ι)
    (delta dispersion g kappa : ℂ) (F : ι → ℂ) :
    lleLoss a adag kappa * lleHamiltonian a adag next delta dispersion g F -
        lleHamiltonian a adag next delta dispersion g F * lleLoss a adag kappa =
      lleLoss a adag kappa * lleDrive a adag F -
        lleDrive a adag F * lleLoss a adag kappa := by
  unfold lleLoss
  calc
    ((kappa / 2) • lleTotalNumber a adag) *
          lleHamiltonian a adag next delta dispersion g F -
        lleHamiltonian a adag next delta dispersion g F *
          ((kappa / 2) • lleTotalNumber a adag) =
      (kappa / 2) •
        (lleTotalNumber a adag * lleHamiltonian a adag next delta dispersion g F -
          lleHamiltonian a adag next delta dispersion g F * lleTotalNumber a adag) := by
            simp only [smul_mul_assoc, mul_smul_comm, smul_sub]
    _ = (kappa / 2) •
        (lleTotalNumber a adag * lleDrive a adag F -
          lleDrive a adag F * lleTotalNumber a adag) := by
            rw [lle_total_number_comm_hamiltonian_eq_drive a adag h next
              delta dispersion g F]
    _ = ((kappa / 2) • lleTotalNumber a adag) * lleDrive a adag F -
        lleDrive a adag F * ((kappa / 2) • lleTotalNumber a adag) := by
          simp only [smul_mul_assoc, mul_smul_comm, smul_sub]

/-- The exact loss-drive commutator. -/
theorem lle_loss_comm_drive (a adag : ι → R) (h : MultimodeCCR a adag)
    (kappa : ℂ) (F : ι → ℂ) :
    lleLoss a adag kappa * lleDrive a adag F -
        lleDrive a adag F * lleLoss a adag kappa =
      (Complex.I * (kappa / 2)) •
        ∑ j, ((F j) • adag j + (starRingEnd ℂ) (F j) • a j) := by
  unfold lleLoss
  calc
    ((kappa / 2) • lleTotalNumber a adag) * lleDrive a adag F -
          lleDrive a adag F * ((kappa / 2) • lleTotalNumber a adag) =
        (kappa / 2) •
          (lleTotalNumber a adag * lleDrive a adag F -
            lleDrive a adag F * lleTotalNumber a adag) := by
              simp only [smul_mul_assoc, mul_smul_comm, smul_sub]
    _ = (kappa / 2) •
        (Complex.I • ∑ j,
          ((F j) • adag j + (starRingEnd ℂ) (F j) • a j)) := by
            rw [lle_total_number_comm_drive a adag h F]
    _ = (Complex.I * (kappa / 2)) •
        ∑ j, ((F j) • adag j + (starRingEnd ℂ) (F j) • a j) := by
          rw [smul_smul, mul_comm]

/-- Manuscript Eq. (eq:lle-mixed-commutator). -/
theorem lle_mixed_commutator
    (a adag : ι → R) (h : MultimodeCCR a adag) (next : ι → ι)
    (delta dispersion g kappa : ℂ) (F : ι → ℂ) :
    lleLoss a adag kappa * lleHamiltonian a adag next delta dispersion g F -
        lleHamiltonian a adag next delta dispersion g F * lleLoss a adag kappa =
      (Complex.I * (kappa / 2)) •
        ∑ j, ((F j) • adag j + (starRingEnd ℂ) (F j) • a j) := by
  calc
    lleLoss a adag kappa * lleHamiltonian a adag next delta dispersion g F -
          lleHamiltonian a adag next delta dispersion g F * lleLoss a adag kappa =
        lleLoss a adag kappa * lleDrive a adag F -
          lleDrive a adag F * lleLoss a adag kappa := by
            exact lle_loss_comm_hamiltonian_eq_drive a adag h next
              delta dispersion g kappa F
    _ = (Complex.I * (kappa / 2)) •
        ∑ j, ((F j) • adag j + (starRingEnd ℂ) (F j) • a j) := by
          exact lle_loss_comm_drive a adag h kappa F

end LLEAlgebra

end

end Cvdv
