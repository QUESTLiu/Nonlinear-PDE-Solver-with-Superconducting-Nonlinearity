#!/usr/bin/env python3
"""
Finite-Fock-cutoff hardware benchmark for a cubic--quintic NLSE / Kerr lattice.

Story tested:
  H_target = H_native + H_corr
  H_native = onsite Kerr + nearest-neighbor hopping
  H_corr   = non-native quintic/three-body number interaction

The script compares:
  1) native-only evolution:         exp(-i H_native t)
  2) ideal finite-cutoff target:    exp(-i (H_native+H_corr) t)
  3) hybrid-corrected Strang route: repeated native evolution + diagonal correction pulses

It also compares finite-cutoff results against a larger reference cutoff to expose
hardware truncation error.

No pandas dependency. Matplotlib is optional; CSV/NPZ are always written.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class Params:
    sites: int
    cutoff: int
    ref_cutoff: int
    omega: float
    J: float
    U: float
    W_values: Sequence[float]
    T: float
    n_time: int
    trotter_steps: int
    amp: float
    sigma: float
    k0: float
    periodic: bool
    outdir: str


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def destroy(cutoff: int) -> sp.csr_matrix:
    """Single-mode annihilation operator in an n=0,...,cutoff-1 Fock basis."""
    vals = np.sqrt(np.arange(1, cutoff, dtype=float))
    return sp.diags(vals, offsets=1, shape=(cutoff, cutoff), format="csr")


def local_op(op: sp.spmatrix, site: int, sites: int, cutoff: int) -> sp.csr_matrix:
    """Embed a single-site operator into a tensor-product oscillator lattice."""
    mats: List[sp.spmatrix] = []
    ident = sp.identity(cutoff, format="csr", dtype=complex)
    for j in range(sites):
        mats.append(op if j == site else ident)
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out.tocsr()


def build_ops(sites: int, cutoff: int) -> Tuple[List[sp.csr_matrix], List[sp.csr_matrix]]:
    a = destroy(cutoff).astype(complex)
    n = (a.conjugate().transpose() @ a).tocsr()
    a_ops = [local_op(a, j, sites, cutoff) for j in range(sites)]
    n_ops = [local_op(n, j, sites, cutoff) for j in range(sites)]
    return a_ops, n_ops


def build_hamiltonians(
    sites: int,
    cutoff: int,
    omega: float,
    J: float,
    U: float,
    W: float,
    periodic: bool = False,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Build finite-cutoff sparse Hamiltonians.

    H_native = sum_j [omega n_j + U/2 n_j(n_j-1)]
               - J sum_<j,j+1> (a_j^dag a_{j+1}+h.c.)

    H_corr = W/6 sum_j n_j(n_j-1)(n_j-2)

    The diagonal of H_corr is returned separately for fast SNAP-like correction phases.
    """
    dim = cutoff ** sites
    I = sp.identity(dim, format="csr", dtype=complex)
    a_ops, n_ops = build_ops(sites, cutoff)

    H_native = sp.csr_matrix((dim, dim), dtype=complex)
    H_corr = sp.csr_matrix((dim, dim), dtype=complex)

    for n_j in n_ops:
        H_native += omega * n_j
        H_native += 0.5 * U * (n_j @ (n_j - I))
        H_corr += (W / 6.0) * (n_j @ (n_j - I) @ (n_j - 2.0 * I))

    edges = [(j, j + 1) for j in range(sites - 1)]
    if periodic and sites > 2:
        edges.append((sites - 1, 0))

    for j, k in edges:
        hop = a_ops[j].conjugate().transpose() @ a_ops[k]
        H_native += -J * (hop + hop.conjugate().transpose())

    H_target = (H_native + H_corr).tocsr()
    H_corr_diag = H_corr.diagonal().astype(complex)
    return H_native.tocsr(), H_corr.tocsr(), H_target, H_corr_diag


def coherent_coeff(alpha: complex, cutoff: int) -> np.ndarray:
    """Truncated coherent state coefficients, normalized inside the cutoff."""
    coeff = np.empty(cutoff, dtype=complex)
    coeff[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, cutoff):
        coeff[n] = coeff[n - 1] * alpha / math.sqrt(n)
    norm = np.linalg.norm(coeff)
    if norm == 0:
        raise ValueError("zero coherent-state norm after truncation")
    return coeff / norm


def product_state(local_states: Sequence[np.ndarray]) -> np.ndarray:
    psi = local_states[0]
    for v in local_states[1:]:
        psi = np.kron(psi, v)
    return psi.astype(complex) / np.linalg.norm(psi)


def initial_coherent_wavepacket(
    sites: int,
    cutoff: int,
    amp: float,
    sigma: float,
    k0: float,
) -> np.ndarray:
    """Product of coherent states with Gaussian envelope and phase gradient."""
    center = 0.5 * (sites - 1)
    local = []
    for j in range(sites):
        envelope = math.exp(-((j - center) ** 2) / (2.0 * sigma ** 2))
        phase = np.exp(1j * k0 * (j - center))
        alpha_j = amp * envelope * phase
        local.append(coherent_coeff(alpha_j, cutoff))
    return product_state(local)


def evolve_exact(H: sp.csr_matrix, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return states psi(t_i)=exp(-i H t_i) psi0 for all times."""
    A = (-1j) * H
    return expm_multiply(A, psi0, start=float(times[0]), stop=float(times[-1]), num=len(times), endpoint=True)


def evolve_hybrid_strang(
    H_native: sp.csr_matrix,
    H_corr_diag: np.ndarray,
    psi0: np.ndarray,
    T: float,
    n_time: int,
    trotter_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hardware-inspired corrected evolution.

    One correction step:
      exp(-i H_native dt/2) exp(-i H_corr dt) exp(-i H_native dt/2)

    H_native is the analog/native Kerr-lattice block.
    H_corr is a diagonal number-selective correction pulse, analogous to a SNAP/number-phase layer.
    """
    if trotter_steps % (n_time - 1) != 0:
        raise ValueError("For simple sampling, choose trotter_steps divisible by n_time-1.")

    sample_every = trotter_steps // (n_time - 1)
    dt = T / trotter_steps
    psi = psi0.astype(complex).copy()
    states = [psi.copy()]
    times = [0.0]

    phase_corr = np.exp(-1j * H_corr_diag * dt)
    A_half = (-1j * H_native) * (0.5 * dt)

    for step in range(1, trotter_steps + 1):
        psi = expm_multiply(A_half, psi)
        psi *= phase_corr
        psi = expm_multiply(A_half, psi)
        psi /= np.linalg.norm(psi)
        if step % sample_every == 0:
            states.append(psi.copy())
            times.append(step * dt)

    return np.asarray(times), np.asarray(states)


def basis_tuples(sites: int, cutoff: int) -> Iterable[Tuple[int, ...]]:
    return itertools.product(range(cutoff), repeat=sites)


def flat_index(occ: Sequence[int], cutoff: int) -> int:
    idx = 0
    for n in occ:
        idx = idx * cutoff + int(n)
    return idx


def embed_state(psi_small: np.ndarray, sites: int, cutoff_small: int, cutoff_ref: int) -> np.ndarray:
    """Embed a state from smaller local cutoff into the larger reference Fock space."""
    if cutoff_small == cutoff_ref:
        return psi_small.copy()
    out = np.zeros(cutoff_ref ** sites, dtype=complex)
    for occ in basis_tuples(sites, cutoff_small):
        i_small = flat_index(occ, cutoff_small)
        i_ref = flat_index(occ, cutoff_ref)
        out[i_ref] = psi_small[i_small]
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    n1 = np.linalg.norm(psi)
    n2 = np.linalg.norm(phi)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(abs(np.vdot(psi / n1, phi / n2)) ** 2)


def max_infidelity(states_a: np.ndarray, states_b: np.ndarray) -> Tuple[float, float, np.ndarray]:
    fids = np.array([fidelity(a, b) for a, b in zip(states_a, states_b)], dtype=float)
    infids = 1.0 - fids
    imax = int(np.argmax(infids))
    return float(infids[imax]), float(fids[-1]), fids


def photon_tail_probability(psi_ref: np.ndarray, sites: int, cutoff: int, cutoff_ref: int) -> float:
    """Probability weight in reference state outside the smaller local cutoff."""
    if cutoff >= cutoff_ref:
        return 0.0
    prob_inside = 0.0
    for occ in basis_tuples(sites, cutoff_ref):
        if all(n < cutoff for n in occ):
            idx = flat_index(occ, cutoff_ref)
            prob_inside += abs(psi_ref[idx]) ** 2
    return float(max(0.0, 1.0 - prob_inside))


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_plot(outdir: str, rows: List[Dict[str, object]]) -> None:
    """Optional plotting. The run succeeds even if matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[plot] matplotlib unavailable; skipping PNG plots: {exc}")
        return

    cutoffs = sorted({int(r["cutoff"]) for r in rows})
    Wvals = sorted({float(r["W"]) for r in rows})

    # Plot max infidelity versus W for each cutoff.
    for key, ylabel, fname in [
        ("max_infid_native_vs_ref", "max infidelity: native only", "max_infid_native_vs_W.png"),
        ("max_infid_hybrid_vs_ref", "max infidelity: hybrid corrected", "max_infid_hybrid_vs_W.png"),
        ("max_infid_cutoff_floor", "max infidelity: finite-cutoff target", "max_infid_cutoff_floor_vs_W.png"),
    ]:
        plt.figure()
        for nc in cutoffs:
            xs = []
            ys = []
            for W in Wvals:
                match = [r for r in rows if int(r["cutoff"]) == nc and abs(float(r["W"]) - W) < 1e-15]
                if match:
                    xs.append(W)
                    ys.append(float(match[0][key]))
            plt.semilogy(xs, ys, marker="o", label=f"Ncut={nc}")
        plt.xlabel("quintic correction strength W")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()

    # Plot cutoff convergence at largest W.
    Wmax = max(Wvals)
    plt.figure()
    xs = []
    native = []
    hybrid = []
    floor = []
    for nc in cutoffs:
        match = [r for r in rows if int(r["cutoff"]) == nc and abs(float(r["W"]) - Wmax) < 1e-15]
        if match:
            xs.append(nc)
            native.append(float(match[0]["max_infid_native_vs_ref"]))
            hybrid.append(float(match[0]["max_infid_hybrid_vs_ref"]))
            floor.append(float(match[0]["max_infid_cutoff_floor"]))
    plt.semilogy(xs, native, marker="o", label="native only")
    plt.semilogy(xs, hybrid, marker="o", label="hybrid corrected")
    plt.semilogy(xs, floor, marker="o", label="finite-cutoff floor")
    plt.xlabel("local Fock cutoff")
    plt.ylabel(f"max infidelity at W={Wmax:g}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cutoff_convergence_at_largest_W.png"), dpi=200)
    plt.close()


def run(params: Params, cutoffs: Sequence[int]) -> None:
    os.makedirs(params.outdir, exist_ok=True)
    times = np.linspace(0.0, params.T, params.n_time)

    fieldnames = [
        "sites", "cutoff", "ref_cutoff", "dim", "ref_dim", "W", "omega", "J", "U", "T",
        "n_time", "trotter_steps", "periodic", "initial_amp", "initial_sigma", "initial_k0",
        "max_infid_native_vs_ref", "final_fid_native_vs_ref",
        "max_infid_cutoff_floor", "final_fid_cutoff_floor",
        "max_infid_hybrid_vs_ref", "final_fid_hybrid_vs_ref",
        "max_infid_hybrid_vs_finite_target", "final_fid_hybrid_vs_finite_target",
        "final_ref_tail_prob_outside_cutoff", "runtime_seconds"
    ]
    all_rows: List[Dict[str, object]] = []

    for W in params.W_values:
        t0_W = time.time()
        print(f"\n=== W = {W:g}: building reference cutoff {params.ref_cutoff} ===", flush=True)
        Hn_ref, Hc_ref, Ht_ref, _ = build_hamiltonians(
            params.sites, params.ref_cutoff, params.omega, params.J, params.U, W, params.periodic
        )
        psi0_ref = initial_coherent_wavepacket(
            params.sites, params.ref_cutoff, params.amp, params.sigma, params.k0
        )
        ref_states = evolve_exact(Ht_ref, psi0_ref, times)
        print(f"Reference evolution done in {time.time()-t0_W:.2f} s", flush=True)

        # Save reference final state for reproducibility/debugging.
        np.savez_compressed(
            os.path.join(params.outdir, f"reference_W_{W:.6g}.npz"),
            times=times,
            final_state=ref_states[-1],
            sites=params.sites,
            ref_cutoff=params.ref_cutoff,
            W=W,
        )

        for cutoff in cutoffs:
            t0 = time.time()
            print(f"--- cutoff = {cutoff} ---", flush=True)
            Hn, Hc, Ht, Hc_diag = build_hamiltonians(
                params.sites, cutoff, params.omega, params.J, params.U, W, params.periodic
            )
            psi0 = initial_coherent_wavepacket(params.sites, cutoff, params.amp, params.sigma, params.k0)

            native_states = evolve_exact(Hn, psi0, times)
            finite_target_states = evolve_exact(Ht, psi0, times)
            t_hybrid, hybrid_states = evolve_hybrid_strang(
                Hn, Hc_diag, psi0, params.T, params.n_time, params.trotter_steps
            )
            if not np.allclose(t_hybrid, times):
                raise RuntimeError("Internal time grid mismatch.")

            native_emb = np.asarray([embed_state(s, params.sites, cutoff, params.ref_cutoff) for s in native_states])
            target_emb = np.asarray([embed_state(s, params.sites, cutoff, params.ref_cutoff) for s in finite_target_states])
            hybrid_emb = np.asarray([embed_state(s, params.sites, cutoff, params.ref_cutoff) for s in hybrid_states])

            max_nat, final_nat, fid_nat = max_infidelity(ref_states, native_emb)
            max_floor, final_floor, fid_floor = max_infidelity(ref_states, target_emb)
            max_hyb, final_hyb, fid_hyb = max_infidelity(ref_states, hybrid_emb)
            max_trot, final_trot, fid_trot = max_infidelity(finite_target_states, hybrid_states)
            tail_final = photon_tail_probability(ref_states[-1], params.sites, cutoff, params.ref_cutoff)

            # Save time traces for this parameter pair.
            np.savez_compressed(
                os.path.join(params.outdir, f"trace_cutoff_{cutoff}_W_{W:.6g}.npz"),
                times=times,
                fid_native_vs_ref=fid_nat,
                fid_cutoff_floor=fid_floor,
                fid_hybrid_vs_ref=fid_hyb,
                fid_hybrid_vs_finite_target=fid_trot,
                cutoff=cutoff,
                ref_cutoff=params.ref_cutoff,
                W=W,
            )

            row = {
                "sites": params.sites,
                "cutoff": cutoff,
                "ref_cutoff": params.ref_cutoff,
                "dim": cutoff ** params.sites,
                "ref_dim": params.ref_cutoff ** params.sites,
                "W": W,
                "omega": params.omega,
                "J": params.J,
                "U": params.U,
                "T": params.T,
                "n_time": params.n_time,
                "trotter_steps": params.trotter_steps,
                "periodic": int(params.periodic),
                "initial_amp": params.amp,
                "initial_sigma": params.sigma,
                "initial_k0": params.k0,
                "max_infid_native_vs_ref": max_nat,
                "final_fid_native_vs_ref": final_nat,
                "max_infid_cutoff_floor": max_floor,
                "final_fid_cutoff_floor": final_floor,
                "max_infid_hybrid_vs_ref": max_hyb,
                "final_fid_hybrid_vs_ref": final_hyb,
                "max_infid_hybrid_vs_finite_target": max_trot,
                "final_fid_hybrid_vs_finite_target": final_trot,
                "final_ref_tail_prob_outside_cutoff": tail_final,
                "runtime_seconds": time.time() - t0,
            }
            all_rows.append(row)
            write_csv(os.path.join(params.outdir, "summary.csv"), all_rows, fieldnames)
            print(
                f"cutoff={cutoff:2d}, W={W:g}, "
                f"native max infid={max_nat:.3e}, "
                f"hybrid max infid={max_hyb:.3e}, "
                f"cutoff floor={max_floor:.3e}, "
                f"hybrid-vs-finite-target={max_trot:.3e}",
                flush=True,
            )

    maybe_plot(params.outdir, all_rows)
    print(f"\nDone. Results written to: {params.outdir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=3)
    ap.add_argument("--cutoffs", type=str, default="4,6,8,10,12")
    ap.add_argument("--ref-cutoff", type=int, default=16)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--J", type=float, default=0.5, help="Use J=0.5 for a=1 so Ja^2=1/2.")
    ap.add_argument("--U", type=float, default=0.8, help="Native cubic/Kerr nonlinearity strength.")
    ap.add_argument("--W-values", type=str, default="0,0.05,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--n-time", type=int, default=81, help="Number of saved times. trotter_steps must be divisible by n_time-1.")
    ap.add_argument("--trotter-steps", type=int, default=160)
    ap.add_argument("--amp", type=float, default=1.25)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--k0", type=float, default=0.6)
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--outdir", type=str, default="results/nlse_fock_cutoff_hybrid")
    args = ap.parse_args()

    cutoffs = parse_int_list(args.cutoffs)
    if any(c <= 1 for c in cutoffs):
        raise ValueError("All cutoffs must be > 1.")
    if args.ref_cutoff < max(cutoffs):
        raise ValueError("ref-cutoff must be >= max(cutoffs).")
    if args.trotter_steps % (args.n_time - 1) != 0:
        raise ValueError("Choose trotter-steps divisible by n-time-1.")

    params = Params(
        sites=args.sites,
        cutoff=-1,
        ref_cutoff=args.ref_cutoff,
        omega=args.omega,
        J=args.J,
        U=args.U,
        W_values=parse_float_list(args.W_values),
        T=args.T,
        n_time=args.n_time,
        trotter_steps=args.trotter_steps,
        amp=args.amp,
        sigma=args.sigma,
        k0=args.k0,
        periodic=args.periodic,
        outdir=args.outdir,
    )
    run(params, cutoffs)


if __name__ == "__main__":
    main()
