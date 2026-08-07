import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.Calculus.LHopital
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.Calculus.ContDiff.Deriv

/-!
# Classical sine-Gordon lattice mechanics

Formalization of the classical lattice-mechanics claims of the manuscript:

* Eq. (eq:sg-lattice): the open-chain sine-Gordon lattice Hamiltonian
  `H_SG = ∑_j [Π_j²/2 + E_J (1 - cos φ_j)] + (κ/2) ∑_bonds (φ_{j+1} - φ_j)²`,
  formalized as `sgLatticeEnergy` with the open-boundary bond set of the
  `N_site - 1` physical links.
* The Hamilton equations displayed after Eq. (eq:sg-lattice) and the
  open-boundary equation of motion Eq. (eq:sg-classical-eom) of
  Sec. (subsec:sg-classical): the partial derivative of `H_SG` in `Π_j` is
  `Π_j` (`sgLatticeEnergy_hasDerivAt_Pi`), and the partial derivative in
  `φ_j` is `E_J sin φ_j - κ ∑_{k ∈ nn(j)} (φ_k - φ_j)`, where the neighbor
  set contains two sites in the interior
  (`sgLatticeEnergy_hasDerivAt_phi_interior`, three-point stencil
  `φ_{j+1} - 2 φ_j + φ_{j-1}`) and exactly one site at each open endpoint
  (`sgLatticeEnergy_hasDerivAt_phi_left`, `sgLatticeEnergy_hasDerivAt_phi_right`),
  so that the bulk stencil is not applied incorrectly at an open boundary.
* The central-difference continuum sentence after Eq. (eq:sg-lattice):
  `(φ(x+h) - 2 φ(x) + φ(x-h))/h² → ∂_xx φ(x)` as `h → 0`
  (`sgCentral_tendsto_secondDeriv`, with the `ContDiffAt ℝ 2` corollary
  `sgCentral_tendsto_secondDeriv_of_contDiffAt`).
-/

namespace Cvdv

open scoped BigOperators
open Set Filter

noncomputable section

section SGClassical

/-- The lower-index endpoint (site `b`) of the `b`-th nearest-neighbor bond of the
open chain of `M` sites.  The open chain has the `M - 1` physical links
`b — b+1` for `b = 0, …, M-2`, matching the open-boundary convention stated
after Eq. (eq:sg-lattice). -/
def sgLatticeBondLo (M : ℕ) (b : Fin (M - 1)) : Fin M :=
  ⟨(b : ℕ), by have := b.isLt; omega⟩

/-- The upper-index endpoint (site `b + 1`) of the `b`-th nearest-neighbor bond of the
open chain of `M` sites. -/
def sgLatticeBondHi (M : ℕ) (b : Fin (M - 1)) : Fin M :=
  ⟨(b : ℕ) + 1, by have := b.isLt; omega⟩

/-- Eq. (eq:sg-lattice), open boundary conditions: the sine-Gordon lattice Hamiltonian
`H_SG = ∑_j [Π_j²/2 + E_J (1 - cos φ_j)] + (κ/2) ∑_j (φ_{j+1} - φ_j)²`
on the open chain of `M` sites.  The onsite sum runs over all `M` sites; the
coupling sum runs over the `M - 1` physical nearest-neighbor links only
(no periodic wraparound), as stated in the manuscript for open boundaries. -/
def sgLatticeEnergy (M : ℕ) (EJ kappa : ℝ) (phi Pi : Fin M → ℝ) : ℝ :=
  (∑ j : Fin M, (1 / 2 * Pi j ^ 2 + EJ * (1 - Real.cos (phi j)))) +
    kappa / 2 * ∑ b : Fin (M - 1),
      (phi (sgLatticeBondHi M b) - phi (sgLatticeBondLo M b)) ^ 2

/-- Derivative of the square function, in the normalization `(s²)' = 2s`, used for the
kinetic and bond terms of `sgLatticeEnergy`. -/
theorem sgLattice_hasDerivAt_sq (a : ℝ) :
    HasDerivAt (fun s : ℝ => s ^ 2) (2 * a) a := by
  simpa using hasDerivAt_pow 2 a

/-- First Hamilton equation (displayed after Eq. (eq:sg-lattice)):
the partial derivative of `H_SG` with respect to the momentum `Π_j`, evaluated at the
configuration `(φ, Π)`, equals `Π_j`; hence `φ̇_j = ∂H/∂Π_j = Π_j`. -/
theorem sgLatticeEnergy_hasDerivAt_Pi (M : ℕ) (EJ kappa : ℝ) (phi Pi : Fin M → ℝ)
    (j : Fin M) :
    HasDerivAt (fun s => sgLatticeEnergy M EJ kappa phi (Function.update Pi j s))
      (Pi j) (Pi j) := by
  have h1 : ∀ k : Fin M, HasDerivAt
      (fun s => 1 / 2 * Function.update Pi j s k ^ 2 + EJ * (1 - Real.cos (phi k)))
      (if k = j then Pi j else 0) (Pi j) := by
    intro k
    rcases eq_or_ne k j with hk | hk
    · subst hk
      simp only [Function.update_self]
      have h := ((sgLattice_hasDerivAt_sq (Pi k)).const_mul (1 / 2 : ℝ)).add_const
        (EJ * (1 - Real.cos (phi k)))
      have hv : (1 / 2 : ℝ) * (2 * Pi k) = Pi k := by ring
      rwa [hv] at h
    · simp only [Function.update_of_ne hk, if_neg hk]
      exact hasDerivAt_const _ _
  have hsum : HasDerivAt
      (fun s => ∑ k : Fin M,
        (1 / 2 * Function.update Pi j s k ^ 2 + EJ * (1 - Real.cos (phi k))))
      (∑ k : Fin M, if k = j then Pi j else 0) (Pi j) :=
    HasDerivAt.fun_sum fun k _ => h1 k
  have htotal := hsum.add_const
    (kappa / 2 * ∑ b : Fin (M - 1),
      (phi (sgLatticeBondHi M b) - phi (sgLatticeBondLo M b)) ^ 2)
  have hval : (∑ k : Fin M, if k = j then Pi j else 0) = Pi j := by simp
  rw [hval] at htotal
  simpa only [sgLatticeEnergy] using htotal

/-- Auxiliary form of the second Hamilton equation: the partial derivative of `H_SG`
with respect to `φ_j` is `E_J sin φ_j` plus `κ/2` times a sum of per-bond
contributions, each nonzero only when the bond touches site `j`. -/
theorem sgLatticeEnergy_hasDerivAt_phi_aux (M : ℕ) (EJ kappa : ℝ) (phi Pi : Fin M → ℝ)
    (j : Fin M) :
    HasDerivAt (fun s => sgLatticeEnergy M EJ kappa (Function.update phi j s) Pi)
      (EJ * Real.sin (phi j) +
        kappa / 2 * ∑ b : Fin (M - 1),
          (if sgLatticeBondHi M b = j then 2 * (phi j - phi (sgLatticeBondLo M b))
           else if sgLatticeBondLo M b = j then
             -(2 * (phi (sgLatticeBondHi M b) - phi j))
           else 0)) (phi j) := by
  have h1 : ∀ k : Fin M, HasDerivAt
      (fun s => 1 / 2 * Pi k ^ 2 + EJ * (1 - Real.cos (Function.update phi j s k)))
      (if k = j then EJ * Real.sin (phi j) else 0) (phi j) := by
    intro k
    rcases eq_or_ne k j with hk | hk
    · subst hk
      simp only [Function.update_self]
      have hc : HasDerivAt (fun s : ℝ => 1 - Real.cos s) (Real.sin (phi k)) (phi k) := by
        simpa using (Real.hasDerivAt_cos (phi k)).const_sub 1
      exact (hc.const_mul EJ).const_add (1 / 2 * Pi k ^ 2)
    · simp only [Function.update_of_ne hk, if_neg hk]
      exact hasDerivAt_const _ _
  have h2 : ∀ b : Fin (M - 1), HasDerivAt
      (fun s => (Function.update phi j s (sgLatticeBondHi M b) -
        Function.update phi j s (sgLatticeBondLo M b)) ^ 2)
      (if sgLatticeBondHi M b = j then 2 * (phi j - phi (sgLatticeBondLo M b))
       else if sgLatticeBondLo M b = j then -(2 * (phi (sgLatticeBondHi M b) - phi j))
       else 0) (phi j) := by
    intro b
    have hne : sgLatticeBondHi M b ≠ sgLatticeBondLo M b := by
      simp [sgLatticeBondHi, sgLatticeBondLo, Fin.ext_iff]
    rcases eq_or_ne (sgLatticeBondHi M b) j with hhi | hhi
    · have hlo : sgLatticeBondLo M b ≠ j := fun h => hne (hhi.trans h.symm)
      simp only [hhi, Function.update_self, Function.update_of_ne hlo]
      have h := (sgLattice_hasDerivAt_sq (phi j - phi (sgLatticeBondLo M b))).comp
        (phi j) ((hasDerivAt_id (phi j)).sub_const (phi (sgLatticeBondLo M b)))
      have hv : 2 * (phi j - phi (sgLatticeBondLo M b)) * 1
          = 2 * (phi j - phi (sgLatticeBondLo M b)) := mul_one _
      rwa [hv] at h
    · rcases eq_or_ne (sgLatticeBondLo M b) j with hlo | hlo
      · simp only [hlo, Function.update_self, Function.update_of_ne hhi, if_neg hhi]
        have h := (sgLattice_hasDerivAt_sq (phi (sgLatticeBondHi M b) - phi j)).comp
          (phi j) ((hasDerivAt_id (phi j)).const_sub (phi (sgLatticeBondHi M b)))
        have hv : 2 * (phi (sgLatticeBondHi M b) - phi j) * (-1)
            = -(2 * (phi (sgLatticeBondHi M b) - phi j)) := by ring
        rwa [hv] at h
      · simp only [Function.update_of_ne hhi, Function.update_of_ne hlo,
          if_neg hhi, if_neg hlo]
        exact hasDerivAt_const _ _
  have honsite : HasDerivAt
      (fun s => ∑ k : Fin M,
        (1 / 2 * Pi k ^ 2 + EJ * (1 - Real.cos (Function.update phi j s k))))
      (∑ k : Fin M, if k = j then EJ * Real.sin (phi j) else 0) (phi j) :=
    HasDerivAt.fun_sum fun k _ => h1 k
  have hbond : HasDerivAt
      (fun s => kappa / 2 * ∑ b : Fin (M - 1),
        (Function.update phi j s (sgLatticeBondHi M b) -
          Function.update phi j s (sgLatticeBondLo M b)) ^ 2)
      (kappa / 2 * ∑ b : Fin (M - 1),
        (if sgLatticeBondHi M b = j then 2 * (phi j - phi (sgLatticeBondLo M b))
         else if sgLatticeBondLo M b = j then
           -(2 * (phi (sgLatticeBondHi M b) - phi j))
         else 0)) (phi j) :=
    (HasDerivAt.fun_sum fun b _ => h2 b).const_mul (kappa / 2)
  have htotal := honsite.fun_add hbond
  have hval : (∑ k : Fin M, if k = j then EJ * Real.sin (phi j) else 0)
      = EJ * Real.sin (phi j) := by simp
  rw [hval] at htotal
  simpa only [sgLatticeEnergy] using htotal

/-- Second Hamilton equation, interior site (displayed after Eq. (eq:sg-lattice) and
Eq. (eq:sg-classical-eom) with the two-neighbor set `nn(j) = {j-1, j+1}`): for
`0 < j < M - 1`, the partial derivative of `H_SG` with respect to `φ_j` equals
`E_J sin φ_j - κ (φ_{j+1} - 2 φ_j + φ_{j-1})`; hence
`Π̇_j = -∂H/∂φ_j = -E_J sin φ_j + κ (φ_{j+1} - 2 φ_j + φ_{j-1})`. -/
theorem sgLatticeEnergy_hasDerivAt_phi_interior (M : ℕ) (EJ kappa : ℝ)
    (phi Pi : Fin M → ℝ) (j : Fin M) (h0 : 0 < (j : ℕ)) (hM : (j : ℕ) + 1 < M) :
    HasDerivAt (fun s => sgLatticeEnergy M EJ kappa (Function.update phi j s) Pi)
      (EJ * Real.sin (phi j) -
        kappa * (phi ⟨(j : ℕ) + 1, hM⟩ - 2 * phi j + phi ⟨(j : ℕ) - 1, by omega⟩))
      (phi j) := by
  have key := sgLatticeEnergy_hasDerivAt_phi_aux M EJ kappa phi Pi j
  have hbprev : (j : ℕ) - 1 < M - 1 := by omega
  have hbcur : (j : ℕ) < M - 1 := by omega
  have hhi_iff : ∀ b : Fin (M - 1),
      sgLatticeBondHi M b = j ↔ b = ⟨(j : ℕ) - 1, hbprev⟩ := by
    intro b
    simp only [sgLatticeBondHi, Fin.ext_iff]
    omega
  have hlo_iff : ∀ b : Fin (M - 1),
      sgLatticeBondLo M b = j ↔ b = ⟨(j : ℕ), hbcur⟩ := by
    intro b
    simp only [sgLatticeBondLo, Fin.ext_iff]
  have hne : (⟨(j : ℕ) - 1, hbprev⟩ : Fin (M - 1)) ≠ ⟨(j : ℕ), hbcur⟩ := by
    simp only [ne_eq, Fin.mk.injEq]
    omega
  have hpt : ∀ b : Fin (M - 1),
      (if sgLatticeBondHi M b = j then 2 * (phi j - phi (sgLatticeBondLo M b))
       else if sgLatticeBondLo M b = j then -(2 * (phi (sgLatticeBondHi M b) - phi j))
       else 0)
      = (if b = ⟨(j : ℕ) - 1, hbprev⟩ then 2 * (phi j - phi (sgLatticeBondLo M b))
          else 0)
        + (if b = ⟨(j : ℕ), hbcur⟩ then -(2 * (phi (sgLatticeBondHi M b) - phi j))
          else 0) := by
    intro b
    rcases eq_or_ne b ⟨(j : ℕ) - 1, hbprev⟩ with hb | hb
    · have hb2 : b ≠ ⟨(j : ℕ), hbcur⟩ := by rw [hb]; exact hne
      rw [if_pos ((hhi_iff b).mpr hb), if_pos hb, if_neg hb2, add_zero]
    · rcases eq_or_ne b ⟨(j : ℕ), hbcur⟩ with hb2 | hb2
      · rw [if_neg (fun h => hb ((hhi_iff b).mp h)), if_pos ((hlo_iff b).mpr hb2),
          if_neg hb, if_pos hb2, zero_add]
      · rw [if_neg (fun h => hb ((hhi_iff b).mp h)),
          if_neg (fun h => hb2 ((hlo_iff b).mp h)), if_neg hb, if_neg hb2, add_zero]
  have hsum : (∑ b : Fin (M - 1),
      (if sgLatticeBondHi M b = j then 2 * (phi j - phi (sgLatticeBondLo M b))
       else if sgLatticeBondLo M b = j then -(2 * (phi (sgLatticeBondHi M b) - phi j))
       else 0))
      = 2 * (phi j - phi ⟨(j : ℕ) - 1, by omega⟩)
        + -(2 * (phi ⟨(j : ℕ) + 1, hM⟩ - phi j)) := by
    rw [Finset.sum_congr rfl fun b _ => hpt b, Finset.sum_add_distrib]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, if_true]
    rfl
  rw [hsum] at key
  convert key using 1
  ring

/-- Second Hamilton equation, left open endpoint (Eq. (eq:sg-classical-eom) with the
one-neighbor set `nn(0) = {1}`): the partial derivative of `H_SG` with respect to
`φ_0` equals `E_J sin φ_0 - κ (φ_1 - φ_0)`, i.e. the open boundary keeps only the
single physical link `0 — 1` and the bulk three-point stencil is not applied. -/
theorem sgLatticeEnergy_hasDerivAt_phi_left (M : ℕ) (hM : 2 ≤ M) (EJ kappa : ℝ)
    (phi Pi : Fin M → ℝ) :
    HasDerivAt
      (fun s => sgLatticeEnergy M EJ kappa (Function.update phi ⟨0, by omega⟩ s) Pi)
      (EJ * Real.sin (phi ⟨0, by omega⟩) -
        kappa * (phi ⟨1, by omega⟩ - phi ⟨0, by omega⟩))
      (phi ⟨0, by omega⟩) := by
  have key := sgLatticeEnergy_hasDerivAt_phi_aux M EJ kappa phi Pi ⟨0, by omega⟩
  have hb0 : (0 : ℕ) < M - 1 := by omega
  have hhi : ∀ b : Fin (M - 1), sgLatticeBondHi M b ≠ (⟨0, by omega⟩ : Fin M) := by
    intro b
    simp only [sgLatticeBondHi, ne_eq, Fin.mk.injEq]
    omega
  have hlo_iff : ∀ b : Fin (M - 1),
      sgLatticeBondLo M b = (⟨0, by omega⟩ : Fin M) ↔ b = ⟨0, hb0⟩ := by
    intro b
    simp only [sgLatticeBondLo, Fin.ext_iff]
  have hpt : ∀ b : Fin (M - 1),
      (if sgLatticeBondHi M b = (⟨0, by omega⟩ : Fin M) then
        2 * (phi ⟨0, by omega⟩ - phi (sgLatticeBondLo M b))
       else if sgLatticeBondLo M b = (⟨0, by omega⟩ : Fin M) then
        -(2 * (phi (sgLatticeBondHi M b) - phi ⟨0, by omega⟩))
       else 0)
      = if b = ⟨0, hb0⟩ then
          -(2 * (phi (sgLatticeBondHi M b) - phi ⟨0, by omega⟩))
        else 0 := by
    intro b
    rcases eq_or_ne b ⟨0, hb0⟩ with hb | hb
    · rw [if_neg (hhi b), if_pos ((hlo_iff b).mpr hb), if_pos hb]
    · rw [if_neg (hhi b), if_neg (fun h => hb ((hlo_iff b).mp h)), if_neg hb]
  have hsum : (∑ b : Fin (M - 1),
      (if sgLatticeBondHi M b = (⟨0, by omega⟩ : Fin M) then
        2 * (phi ⟨0, by omega⟩ - phi (sgLatticeBondLo M b))
       else if sgLatticeBondLo M b = (⟨0, by omega⟩ : Fin M) then
        -(2 * (phi (sgLatticeBondHi M b) - phi ⟨0, by omega⟩))
       else 0))
      = -(2 * (phi ⟨1, by omega⟩ - phi ⟨0, by omega⟩)) := by
    rw [Finset.sum_congr rfl fun b _ => hpt b]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, if_true]
    rfl
  rw [hsum] at key
  convert key using 1
  ring

/-- Second Hamilton equation, right open endpoint (Eq. (eq:sg-classical-eom) with the
one-neighbor set `nn(M-1) = {M-2}`): the partial derivative of `H_SG` with respect to
`φ_{M-1}` equals `E_J sin φ_{M-1} - κ (φ_{M-2} - φ_{M-1})`, i.e. the open boundary
keeps only the single physical link `M-2 — M-1`. -/
theorem sgLatticeEnergy_hasDerivAt_phi_right (M : ℕ) (hM : 2 ≤ M) (EJ kappa : ℝ)
    (phi Pi : Fin M → ℝ) :
    HasDerivAt
      (fun s => sgLatticeEnergy M EJ kappa
        (Function.update phi ⟨M - 1, by omega⟩ s) Pi)
      (EJ * Real.sin (phi ⟨M - 1, by omega⟩) -
        kappa * (phi ⟨M - 2, by omega⟩ - phi ⟨M - 1, by omega⟩))
      (phi ⟨M - 1, by omega⟩) := by
  have key := sgLatticeEnergy_hasDerivAt_phi_aux M EJ kappa phi Pi ⟨M - 1, by omega⟩
  have hbR : M - 2 < M - 1 := by omega
  have hlo : ∀ b : Fin (M - 1),
      sgLatticeBondLo M b ≠ (⟨M - 1, by omega⟩ : Fin M) := by
    intro b
    have := b.isLt
    simp only [sgLatticeBondLo, ne_eq, Fin.mk.injEq]
    omega
  have hhi_iff : ∀ b : Fin (M - 1),
      sgLatticeBondHi M b = (⟨M - 1, by omega⟩ : Fin M) ↔ b = ⟨M - 2, hbR⟩ := by
    intro b
    have := b.isLt
    simp only [sgLatticeBondHi, Fin.ext_iff]
    omega
  have hpt : ∀ b : Fin (M - 1),
      (if sgLatticeBondHi M b = (⟨M - 1, by omega⟩ : Fin M) then
        2 * (phi ⟨M - 1, by omega⟩ - phi (sgLatticeBondLo M b))
       else if sgLatticeBondLo M b = (⟨M - 1, by omega⟩ : Fin M) then
        -(2 * (phi (sgLatticeBondHi M b) - phi ⟨M - 1, by omega⟩))
       else 0)
      = if b = ⟨M - 2, hbR⟩ then
          2 * (phi ⟨M - 1, by omega⟩ - phi (sgLatticeBondLo M b))
        else 0 := by
    intro b
    rcases eq_or_ne b ⟨M - 2, hbR⟩ with hb | hb
    · rw [if_pos ((hhi_iff b).mpr hb), if_pos hb]
    · rw [if_neg (fun h => hb ((hhi_iff b).mp h)), if_neg (hlo b), if_neg hb]
  have hsum : (∑ b : Fin (M - 1),
      (if sgLatticeBondHi M b = (⟨M - 1, by omega⟩ : Fin M) then
        2 * (phi ⟨M - 1, by omega⟩ - phi (sgLatticeBondLo M b))
       else if sgLatticeBondLo M b = (⟨M - 1, by omega⟩ : Fin M) then
        -(2 * (phi (sgLatticeBondHi M b) - phi ⟨M - 1, by omega⟩))
       else 0))
      = 2 * (phi ⟨M - 1, by omega⟩ - phi ⟨M - 2, by omega⟩) := by
    rw [Finset.sum_congr rfl fun b _ => hpt b]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, if_true]
    rfl
  rw [hsum] at key
  convert key using 1
  ring

/-- Central-difference consistency (the continuum sentence after Eq. (eq:sg-lattice)):
if `f` has derivative `f'` on a neighborhood `(x - δ, x + δ)` and `f'` has derivative
`l` at `x` (so `l` is the second derivative of `f` at `x`), then the second central
difference `(f (x+h) - 2 f x + f (x-h)) / h²` tends to `l` as `h → 0⁺`.
The proof is one application of L'Hôpital's rule, reducing the quotient to the
symmetric first difference `(f' (x+h) - f' (x-h)) / (2h)` of the derivative. -/
theorem sgCentral_tendsto_secondDeriv (f f' : ℝ → ℝ) (x l δ : ℝ) (hδ : 0 < δ)
    (hf' : ∀ y ∈ Set.Ioo (x - δ) (x + δ), HasDerivAt f (f' y) y)
    (hf'' : HasDerivAt f' l x) :
    Filter.Tendsto (fun h => (f (x + h) - 2 * f x + f (x - h)) / h ^ 2)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds l) := by
  -- Derivative of the numerator on (0, δ).
  have hff' : ∀ h ∈ Set.Ioo (0 : ℝ) δ,
      HasDerivAt (fun h => f (x + h) - 2 * f x + f (x - h))
        (f' (x + h) * 1 + f' (x - h) * (-1)) h := by
    intro h hh
    have hxp : x + h ∈ Set.Ioo (x - δ) (x + δ) := by
      constructor <;> [linarith [hh.1]; linarith [hh.2]]
    have hxm : x - h ∈ Set.Ioo (x - δ) (x + δ) := by
      constructor <;> [linarith [hh.2]; linarith [hh.1]]
    have hA : HasDerivAt (fun h' : ℝ => f (x + h')) (f' (x + h) * 1) h :=
      (hf' (x + h) hxp).comp h ((hasDerivAt_id h).const_add x)
    have hB : HasDerivAt (fun h' : ℝ => f (x - h')) (f' (x - h) * (-1)) h :=
      (hf' (x - h) hxm).comp h ((hasDerivAt_id h).const_sub x)
    exact (hA.sub_const (2 * f x)).add hB
  -- Derivative of the denominator on (0, δ).
  have hgg' : ∀ h ∈ Set.Ioo (0 : ℝ) δ, HasDerivAt (fun h : ℝ => h ^ 2) (2 * h) h :=
    fun h _ => sgLattice_hasDerivAt_sq h
  have hg' : ∀ h ∈ Set.Ioo (0 : ℝ) δ, (2 : ℝ) * h ≠ 0 := by
    intro h hh
    have : (0 : ℝ) < 2 * h := by linarith [hh.1]
    exact ne_of_gt this
  -- Both numerator and denominator vanish as `h → 0⁺`.
  have hxmem : x ∈ Set.Ioo (x - δ) (x + δ) := by constructor <;> linarith
  have hmap : Tendsto (fun h : ℝ => x + h) (nhdsWithin 0 (Set.Ioi 0)) (nhds x) := by
    have h1 : Tendsto (fun h : ℝ => x + h) (nhds 0) (nhds (x + 0)) :=
      (continuous_const.add continuous_id).tendsto 0
    simpa using h1.mono_left nhdsWithin_le_nhds
  have hmap' : Tendsto (fun h : ℝ => x - h) (nhdsWithin 0 (Set.Ioi 0)) (nhds x) := by
    have h1 : Tendsto (fun h : ℝ => x - h) (nhds 0) (nhds (x - 0)) :=
      (continuous_const.sub continuous_id).tendsto 0
    simpa using h1.mono_left nhdsWithin_le_nhds
  have hcontf : ContinuousAt f x := (hf' x hxmem).continuousAt
  have hfa : Tendsto (fun h => f (x + h) - 2 * f x + f (x - h))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
    have hA : Tendsto (fun h : ℝ => f (x + h)) (nhdsWithin 0 (Set.Ioi 0))
        (nhds (f x)) := hcontf.tendsto.comp hmap
    have hB : Tendsto (fun h : ℝ => f (x - h)) (nhdsWithin 0 (Set.Ioi 0))
        (nhds (f x)) := hcontf.tendsto.comp hmap'
    have hC : Tendsto (fun h : ℝ => f (x + h) - 2 * f x + f (x - h))
        (nhdsWithin 0 (Set.Ioi 0)) (nhds (f x - 2 * f x + f x)) :=
      (hA.sub_const (2 * f x)).add hB
    have h0 : f x - 2 * f x + f x = 0 := by ring
    rwa [h0] at hC
  have hga : Tendsto (fun h : ℝ => h ^ 2) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
    have h1 : Tendsto (fun h : ℝ => h ^ 2) (nhds 0) (nhds ((0 : ℝ) ^ 2)) :=
      (continuous_pow 2).tendsto 0
    simpa using h1.mono_left nhdsWithin_le_nhds
  -- The quotient of derivatives tends to `l`, via the slope characterization of `f''`.
  have hslope := hasDerivAt_iff_tendsto_slope.mp hf''
  have hmapne : Tendsto (fun h : ℝ => x + h) (nhdsWithin 0 (Set.Ioi 0))
      (nhdsWithin x {x}ᶜ) := by
    rw [tendsto_nhdsWithin_iff]
    refine ⟨hmap, ?_⟩
    filter_upwards [self_mem_nhdsWithin] with h hh
    have hpos : (0 : ℝ) < h := hh
    simp only [Set.mem_compl_iff, Set.mem_singleton_iff]
    intro hcontra
    have : h = 0 := by linarith
    linarith
  have hmapne' : Tendsto (fun h : ℝ => x - h) (nhdsWithin 0 (Set.Ioi 0))
      (nhdsWithin x {x}ᶜ) := by
    rw [tendsto_nhdsWithin_iff]
    refine ⟨hmap', ?_⟩
    filter_upwards [self_mem_nhdsWithin] with h hh
    have hpos : (0 : ℝ) < h := hh
    simp only [Set.mem_compl_iff, Set.mem_singleton_iff]
    intro hcontra
    have : h = 0 := by linarith
    linarith
  have hplus : Tendsto (fun h : ℝ => (f' (x + h) - f' x) / h)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds l) := by
    have h1 := hslope.comp hmapne
    have heq : (slope f' x ∘ fun h : ℝ => x + h)
        = fun h : ℝ => (f' (x + h) - f' x) / h := by
      funext h
      simp only [Function.comp_apply, slope_def_field]
      rw [show x + h - x = h by ring]
    rwa [heq] at h1
  have hminus : Tendsto (fun h : ℝ => (f' x - f' (x - h)) / h)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds l) := by
    have h1 := hslope.comp hmapne'
    have heq : (slope f' x ∘ fun h : ℝ => x - h)
        = fun h : ℝ => (f' x - f' (x - h)) / h := by
      funext h
      simp only [Function.comp_apply, slope_def_field]
      rw [show x - h - x = -h by ring, div_neg, ← neg_div, neg_sub]
    rwa [heq] at h1
  have hdiv : Tendsto (fun h : ℝ => (f' (x + h) * 1 + f' (x - h) * (-1)) / (2 * h))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds l) := by
    have hcomb := (hplus.add hminus).div_const 2
    have hl2 : (l + l) / 2 = l := by ring
    rw [hl2] at hcomb
    have heq : (fun h : ℝ => ((f' (x + h) - f' x) / h + (f' x - f' (x - h)) / h) / 2)
        = fun h : ℝ => (f' (x + h) * 1 + f' (x - h) * (-1)) / (2 * h) := by
      funext h
      rcases eq_or_ne h 0 with rfl | hh
      · simp
      · field_simp
        ring
    rwa [heq] at hcomb
  exact HasDerivAt.lhopital_zero_right_on_Ioo hδ hff' hgg' hg' hfa hga hdiv

/-- Central-difference consistency in `ContDiffAt` form: if `f` is `C²` at `x`, then
`(f (x+h) - 2 f x + f (x-h)) / h² → (deriv (deriv f)) x` as `h → 0⁺`, which is the
manuscript's statement that the three-point stencil `(φ_{j+1} - 2 φ_j + φ_{j-1})/b²`
recovers `∂_xx φ` in the continuum limit `b → 0`. -/
theorem sgCentral_tendsto_secondDeriv_of_contDiffAt (f : ℝ → ℝ) (x : ℝ)
    (hf : ContDiffAt ℝ 2 f x) :
    Filter.Tendsto (fun h => (f (x + h) - 2 * f x + f (x - h)) / h ^ 2)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds (deriv (deriv f) x)) := by
  obtain ⟨u, hu, hfu⟩ := hf.contDiffOn (le_refl 2) (by simp)
  obtain ⟨δ, hδ, hball⟩ := Metric.mem_nhds_iff.mp hu
  have hsub : Set.Ioo (x - δ) (x + δ) ⊆ u := by
    rw [← Real.ball_eq_Ioo]; exact hball
  have hopen : IsOpen (Set.Ioo (x - δ) (x + δ)) := isOpen_Ioo
  have hfIoo : ContDiffOn ℝ 2 f (Set.Ioo (x - δ) (x + δ)) := hfu.mono hsub
  have hdiff : DifferentiableOn ℝ f (Set.Ioo (x - δ) (x + δ)) :=
    hfIoo.differentiableOn (by norm_num)
  have hf' : ∀ y ∈ Set.Ioo (x - δ) (x + δ), HasDerivAt f (deriv f y) y := fun y hy =>
    (hdiff.differentiableAt (hopen.mem_nhds hy)).hasDerivAt
  have hd1 : ContDiffOn ℝ 1 (deriv f) (Set.Ioo (x - δ) (x + δ)) :=
    hfIoo.deriv_of_isOpen hopen (by norm_num)
  have hxmem : x ∈ Set.Ioo (x - δ) (x + δ) := by constructor <;> linarith
  have hf'' : HasDerivAt (deriv f) (deriv (deriv f) x) x :=
    ((hd1.differentiableOn (by norm_num)).differentiableAt
      (hopen.mem_nhds hxmem)).hasDerivAt
  exact sgCentral_tendsto_secondDeriv f (deriv f) x (deriv (deriv f) x) δ hδ hf' hf''

end SGClassical

end

end Cvdv
