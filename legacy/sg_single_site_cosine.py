#!/usr/bin/env python3
r"""
Single-site quantum cosine oscillator benchmark for Sine-Gordon native nonlinearity.

Hamiltonian:
    H_cos = 1/2 pi^2 + EJ (1 - cos(phi))

with oscillator convention:
    phi = a + a^\dagger
    pi  = -i(a - a^\dagger)
    [phi, pi] = 2i

The code compares:
    1. Native cosine Hamiltonian, treated as the exact reference.
    2. Quartic polynomial approximation.
    3. Sixth-order polynomial approximation.
    4. Trotterized native cosine evolution.

Outputs:
    - Per-case .npz time-series file
    - Per-case metrics.csv
    - Per-case summary.json
    - Figures for fidelity, infidelity, and <phi(t)>
    - Optional aggregate_summary.csv after all array jobs finish
"""

import argparse
import csv
import glob
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import eigh

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def hermitize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.conj().T)


def destroy_operator(n_cut: int) -> np.ndarray:
    """Create truncated annihilation operator."""
    a = np.zeros((n_cut, n_cut), dtype=np.complex128)
    for n in range(1, n_cut):
        a[n - 1, n] = np.sqrt(n)
    return a


def coherent_state(alpha: complex, n_cut: int) -> np.ndarray:
    """
    Truncated coherent state.

    |alpha> = exp(-|alpha|^2/2) sum_n alpha^n / sqrt(n!) |n>

    The vector is renormalized after truncation.
    """
    psi = np.zeros(n_cut, dtype=np.complex128)
    psi[0] = np.exp(-0.5 * abs(alpha) ** 2)

    for n in range(1, n_cut):
        psi[n] = psi[n - 1] * alpha / np.sqrt(n)

    norm = np.linalg.norm(psi)
    if norm < 1e-14:
        raise RuntimeError(
            "Initial coherent state has near-zero norm after truncation."
        )

    return psi / norm


def matrix_function_hermitian(op: np.ndarray, func) -> np.ndarray:
    """
    Compute f(op) for Hermitian op using eigendecomposition.
    This is used for cos(phi).
    """
    evals, evecs = eigh(hermitize(op))
    return hermitize((evecs * func(evals)) @ evecs.conj().T)


def build_operators(n_cut: int, EJ: float) -> Dict[str, np.ndarray]:
    """
    Build operators and Hamiltonians for one oscillator.
    """
    a = destroy_operator(n_cut)
    adag = a.conj().T

    I = np.eye(n_cut, dtype=np.complex128)

    phi = hermitize(a + adag)
    pi = hermitize(-1j * (a - adag))

    phi2 = hermitize(phi @ phi)
    phi4 = hermitize(phi2 @ phi2)
    phi6 = hermitize(phi4 @ phi2)

    kinetic = hermitize(0.5 * (pi @ pi))

    cos_phi = matrix_function_hermitian(phi, np.cos)

    V_cos = hermitize(EJ * (I - cos_phi))
    V4 = hermitize(EJ * (0.5 * phi2 - phi4 / 24.0))
    V6 = hermitize(EJ * (0.5 * phi2 - phi4 / 24.0 + phi6 / 720.0))

    H_cos = hermitize(kinetic + V_cos)
    H4 = hermitize(kinetic + V4)
    H6 = hermitize(kinetic + V6)

    n_op = hermitize(adag @ a)

    return {
        "a": a,
        "adag": adag,
        "I": I,
        "phi": phi,
        "pi": pi,
        "n": n_op,
        "kinetic": kinetic,
        "V_cos": V_cos,
        "V4": V4,
        "V6": V6,
        "H_cos": H_cos,
        "H4": H4,
        "H6": H6,
    }


def evolve_exact(H: np.ndarray, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Exact evolution under a time-independent finite-dimensional Hamiltonian.

    Returns:
        states[t_index, n]
    """
    evals, evecs = eigh(hermitize(H))
    coeff0 = evecs.conj().T @ psi0

    phase = np.exp(-1j * np.outer(times, evals))
    coeff_t = phase * coeff0[None, :]

    states = coeff_t @ evecs.T
    return states


def unitary_from_hamiltonian(H: np.ndarray, dt: float) -> np.ndarray:
    evals, evecs = eigh(hermitize(H))
    return (evecs * np.exp(-1j * dt * evals)) @ evecs.conj().T


def evolve_trotter_strang(
    kinetic: np.ndarray,
    potential: np.ndarray,
    psi0: np.ndarray,
    n_steps: int,
    dt: float,
) -> np.ndarray:
    """
    Second-order Strang splitting:

        U(dt) = exp(-i K dt/2) exp(-i V dt) exp(-i K dt/2)

    Reference target is the exact native cosine Hamiltonian.
    """
    U_K_half = unitary_from_hamiltonian(kinetic, 0.5 * dt)
    U_V = unitary_from_hamiltonian(potential, dt)

    U_step = U_K_half @ U_V @ U_K_half

    n_cut = psi0.size
    states = np.zeros((n_steps + 1, n_cut), dtype=np.complex128)
    states[0] = psi0

    psi = psi0.copy()
    for step in range(1, n_steps + 1):
        psi = U_step @ psi

        # Protect against small numerical drift.
        psi = psi / np.linalg.norm(psi)
        states[step] = psi

    return states


def expectation(states: np.ndarray, op: np.ndarray) -> np.ndarray:
    vals = np.einsum("ti,ij,tj->t", states.conj(), op, states)
    return np.real_if_close(vals)


def fidelity(states_ref: np.ndarray, states_test: np.ndarray) -> np.ndarray:
    overlaps = np.einsum("ti,ti->t", states_ref.conj(), states_test)
    return np.abs(overlaps) ** 2


def relative_energy_drift(energy: np.ndarray) -> float:
    e0 = float(np.real(energy[0]))
    denom = max(abs(e0), 1e-12)
    return float(np.max(np.abs(np.real(energy) - e0)) / denom)


def max_abs_error(x_ref: np.ndarray, x_test: np.ndarray) -> float:
    return float(np.max(np.abs(np.real(x_ref) - np.real(x_test))))


def make_plots(
    case_dir: str,
    times: np.ndarray,
    F4: np.ndarray,
    F6: np.ndarray,
    Ftrot: np.ndarray,
    phi_cos: np.ndarray,
    phi4: np.ndarray,
    phi6: np.ndarray,
    phi_trot: np.ndarray,
    n_cut: int,
    phi0: float,
    pi0: float,
) -> None:
    # Fidelity plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(times, F4, label="Quartic vs cosine")
    plt.plot(times, F6, label="Sixth-order vs cosine")
    plt.plot(times, Ftrot, label="Strang Trotter vs cosine")
    plt.xlabel("Time")
    plt.ylabel("Fidelity")
    plt.title(f"Single-site cosine oscillator: n_cut={n_cut}, phi0={phi0}, pi0={pi0}")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "fidelity_vs_time.png"), dpi=250)
    plt.close()

    # Infidelity plot
    plt.figure(figsize=(7, 4.5))
    eps = 1e-16
    plt.semilogy(times, np.maximum(1.0 - F4, eps), label="Quartic")
    plt.semilogy(times, np.maximum(1.0 - F6, eps), label="Sixth-order")
    plt.semilogy(times, np.maximum(1.0 - Ftrot, eps), label="Strang Trotter")
    plt.xlabel("Time")
    plt.ylabel("Infidelity")
    plt.title(f"Infidelity: n_cut={n_cut}, phi0={phi0}, pi0={pi0}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "infidelity_vs_time.png"), dpi=250)
    plt.close()

    # Field expectation plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(times, np.real(phi_cos), label="Native cosine reference")
    plt.plot(times, np.real(phi4), "--", label="Quartic")
    plt.plot(times, np.real(phi6), "-.", label="Sixth-order")
    plt.plot(times, np.real(phi_trot), ":", label="Strang Trotter")
    plt.xlabel("Time")
    plt.ylabel(r"$\langle \hat{\phi}(t) \rangle$")
    plt.title(f"Field expectation: n_cut={n_cut}, phi0={phi0}, pi0={pi0}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "phi_expectation_vs_time.png"), dpi=250)
    plt.close()


def write_timeseries_csv(
    path: str,
    times: np.ndarray,
    F4: np.ndarray,
    F6: np.ndarray,
    Ftrot: np.ndarray,
    phi_cos: np.ndarray,
    phi4: np.ndarray,
    phi6: np.ndarray,
    phi_trot: np.ndarray,
    n_cos: np.ndarray,
    E_cos_ref: np.ndarray,
    E_cos_trot: np.ndarray,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "F_quartic",
                "F_sixth",
                "F_trotter",
                "phi_cos",
                "phi_quartic",
                "phi_sixth",
                "phi_trotter",
                "n_cos",
                "E_cos_reference",
                "E_cos_trotter",
            ]
        )
        for i in range(len(times)):
            writer.writerow(
                [
                    float(times[i]),
                    float(F4[i]),
                    float(F6[i]),
                    float(Ftrot[i]),
                    float(np.real(phi_cos[i])),
                    float(np.real(phi4[i])),
                    float(np.real(phi6[i])),
                    float(np.real(phi_trot[i])),
                    float(np.real(n_cos[i])),
                    float(np.real(E_cos_ref[i])),
                    float(np.real(E_cos_trot[i])),
                ]
            )


def write_metrics_csv(path: str, metrics: Dict[str, object]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def run_case(args: argparse.Namespace) -> None:
    ncut_list = parse_int_list(args.ncut_list)
    phi0_list = parse_float_list(args.phi0_list)

    cases: List[Tuple[int, float]] = []
    for n_cut in ncut_list:
        for phi0 in phi0_list:
            cases.append((n_cut, phi0))

    n_cases = len(cases)

    if args.case_index < 0 or args.case_index >= n_cases:
        raise ValueError(
            f"case_index={args.case_index} is out of range. "
            f"Valid range is 0 to {n_cases - 1}."
        )

    n_cut, phi0 = cases[args.case_index]
    pi0 = args.pi0

    case_dir = os.path.join(
        args.outdir,
        f"case_{args.case_index:03d}_ncut_{n_cut}_phi0_{phi0:g}_pi0_{pi0:g}",
    )
    ensure_dir(case_dir)

    print("=" * 80)
    print("Single-site Sine-Gordon cosine oscillator benchmark")
    print(f"case_index = {args.case_index} / {n_cases - 1}")
    print(f"n_cut      = {n_cut}")
    print(f"phi0       = {phi0}")
    print(f"pi0        = {pi0}")
    print(f"EJ         = {args.EJ}")
    print(f"T          = {args.T}")
    print(f"n_steps    = {args.n_steps}")
    print(f"outdir     = {case_dir}")
    print("=" * 80, flush=True)

    start = time.time()

    ops = build_operators(n_cut=n_cut, EJ=args.EJ)

    # Coherent state with <phi> = phi0 and <pi> = pi0.
    alpha = 0.5 * (phi0 + 1j * pi0)
    psi0 = coherent_state(alpha, n_cut)

    times = np.linspace(0.0, args.T, args.n_steps + 1)
    dt = args.T / args.n_steps

    # Exact evolutions
    states_cos = evolve_exact(ops["H_cos"], psi0, times)
    states4 = evolve_exact(ops["H4"], psi0, times)
    states6 = evolve_exact(ops["H6"], psi0, times)

    # Trotterized native cosine
    states_trot = evolve_trotter_strang(
        kinetic=ops["kinetic"],
        potential=ops["V_cos"],
        psi0=psi0,
        n_steps=args.n_steps,
        dt=dt,
    )

    # Observables
    F4 = fidelity(states_cos, states4)
    F6 = fidelity(states_cos, states6)
    Ftrot = fidelity(states_cos, states_trot)

    phi_cos = expectation(states_cos, ops["phi"])
    phi4 = expectation(states4, ops["phi"])
    phi6 = expectation(states6, ops["phi"])
    phi_trot = expectation(states_trot, ops["phi"])

    pi_cos = expectation(states_cos, ops["pi"])
    n_cos = expectation(states_cos, ops["n"])

    E_cos_ref = expectation(states_cos, ops["H_cos"])
    E_cos_trot = expectation(states_trot, ops["H_cos"])

    elapsed = time.time() - start

    metrics = {
        "case_index": args.case_index,
        "n_cases": n_cases,
        "n_cut": n_cut,
        "EJ": args.EJ,
        "phi0": phi0,
        "pi0": pi0,
        "alpha_real": float(np.real(alpha)),
        "alpha_imag": float(np.imag(alpha)),
        "T": args.T,
        "n_steps": args.n_steps,
        "dt": dt,
        "final_F_quartic": float(F4[-1]),
        "final_F_sixth": float(F6[-1]),
        "final_F_trotter": float(Ftrot[-1]),
        "max_infidelity_quartic": float(np.max(1.0 - F4)),
        "max_infidelity_sixth": float(np.max(1.0 - F6)),
        "max_infidelity_trotter": float(np.max(1.0 - Ftrot)),
        "max_phi_error_quartic": max_abs_error(phi_cos, phi4),
        "max_phi_error_sixth": max_abs_error(phi_cos, phi6),
        "max_phi_error_trotter": max_abs_error(phi_cos, phi_trot),
        "cos_energy_drift_reference": relative_energy_drift(E_cos_ref),
        "cos_energy_drift_trotter": relative_energy_drift(E_cos_trot),
        "max_mean_photon_number_reference": float(np.max(np.real(n_cos))),
        "walltime_seconds": float(elapsed),
    }

    # Save compact arrays.
    np.savez_compressed(
        os.path.join(case_dir, "timeseries_data.npz"),
        times=times,
        F_quartic=F4,
        F_sixth=F6,
        F_trotter=Ftrot,
        phi_cos=phi_cos,
        phi_quartic=phi4,
        phi_sixth=phi6,
        phi_trotter=phi_trot,
        pi_cos=pi_cos,
        n_cos=n_cos,
        E_cos_ref=E_cos_ref,
        E_cos_trot=E_cos_trot,
        psi0=psi0,
    )

    write_timeseries_csv(
        os.path.join(case_dir, "timeseries.csv"),
        times,
        F4,
        F6,
        Ftrot,
        phi_cos,
        phi4,
        phi6,
        phi_trot,
        n_cos,
        E_cos_ref,
        E_cos_trot,
    )

    with open(os.path.join(case_dir, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    write_metrics_csv(os.path.join(case_dir, "metrics.csv"), metrics)

    if args.make_plots:
        make_plots(
            case_dir=case_dir,
            times=times,
            F4=F4,
            F6=F6,
            Ftrot=Ftrot,
            phi_cos=phi_cos,
            phi4=phi4,
            phi6=phi6,
            phi_trot=phi_trot,
            n_cut=n_cut,
            phi0=phi0,
            pi0=pi0,
        )

    print("Finished case.")
    print(json.dumps(metrics, indent=2), flush=True)


def aggregate_results(args: argparse.Namespace) -> None:
    """
    Combine all metrics.csv files into one aggregate_summary.csv.
    Also makes simple summary plots.
    """
    ensure_dir(args.outdir)

    metric_files = sorted(glob.glob(os.path.join(args.outdir, "case_*", "metrics.csv")))

    if not metric_files:
        raise RuntimeError(f"No metrics.csv files found under {args.outdir}/case_*")

    rows: List[Dict[str, str]] = []
    for mf in metric_files:
        with open(mf, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    def to_float(row: Dict[str, str], key: str) -> float:
        return float(row[key])

    rows.sort(key=lambda r: (int(r["n_cut"]), float(r["phi0"])))

    out_csv = os.path.join(args.outdir, "aggregate_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote aggregate summary: {out_csv}")

    # Summary plots
    ncuts = sorted({int(r["n_cut"]) for r in rows})
    phi0s = sorted({float(r["phi0"]) for r in rows})

    for metric_name, ylabel, filename in [
        (
            "max_infidelity_quartic",
            "Max infidelity: quartic",
            "aggregate_quartic_max_infid.png",
        ),
        (
            "max_infidelity_sixth",
            "Max infidelity: sixth-order",
            "aggregate_sixth_max_infid.png",
        ),
        (
            "max_infidelity_trotter",
            "Max infidelity: Strang Trotter",
            "aggregate_trotter_max_infid.png",
        ),
        (
            "max_phi_error_quartic",
            r"Max $|\Delta \langle \phi \rangle|$: quartic",
            "aggregate_quartic_phi_error.png",
        ),
        (
            "max_phi_error_sixth",
            r"Max $|\Delta \langle \phi \rangle|$: sixth-order",
            "aggregate_sixth_phi_error.png",
        ),
    ]:
        plt.figure(figsize=(7, 4.5))
        for ncut in ncuts:
            xs = []
            ys = []
            for phi0 in phi0s:
                match = [
                    r
                    for r in rows
                    if int(r["n_cut"]) == ncut and abs(float(r["phi0"]) - phi0) < 1e-12
                ]
                if match:
                    xs.append(phi0)
                    ys.append(to_float(match[0], metric_name))

            if xs:
                plt.semilogy(xs, ys, marker="o", label=f"n_cut={ncut}")

        plt.xlabel(r"Initial field displacement $\phi_0$")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, filename), dpi=250)
        plt.close()

    print("Aggregate plots written.")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        default=os.environ.get("OUTDIR", "results/sg_single_site"),
        help="Output directory.",
    )

    parser.add_argument(
        "--case-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
        help="Case index. Defaults to SLURM_ARRAY_TASK_ID if available.",
    )

    parser.add_argument(
        "--ncut-list",
        type=str,
        default="20,30,40",
        help="Comma-separated Fock cutoffs.",
    )

    parser.add_argument(
        "--phi0-list",
        type=str,
        default="0.5,1.0,2.0,3.0",
        help="Comma-separated initial field displacements.",
    )

    parser.add_argument(
        "--pi0",
        type=float,
        default=0.0,
        help="Initial momentum expectation.",
    )

    parser.add_argument(
        "--EJ",
        type=float,
        default=1.0,
        help="Josephson/cosine energy scale.",
    )

    parser.add_argument(
        "--T",
        type=float,
        default=5.0,
        help="Final simulation time.",
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=800,
        help="Number of time steps. Used as sampling grid and Trotter steps.",
    )

    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Make per-case plots.",
    )

    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate all case metrics after Slurm array jobs finish.",
    )

    args = parser.parse_args()

    ensure_dir(args.outdir)

    if args.aggregate:
        aggregate_results(args)
    else:
        run_case(args)


if __name__ == "__main__":
    main()
