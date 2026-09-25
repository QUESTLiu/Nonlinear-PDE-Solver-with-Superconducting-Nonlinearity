#!/usr/bin/env python3
"""
Two-site native Josephson / Sine-Gordon embedding benchmark.

Hamiltonian:
    H_cos =
        sum_j [ 1/2 Pi_j^2 + EJ (1 - cos(phi_j)) ]
        + kappa/2 (phi_2 - phi_1)^2

with oscillator quadrature convention:
    phi = a + a^\dagger
    Pi  = -i(a - a^\dagger)
    [phi, Pi] = 2i

This script compares:
    1. Exact native cosine Hamiltonian
    2. Quartic polynomial truncation
    3. Sixth-order polynomial truncation
    4. Strang-split native cosine evolution

Outputs per case:
    - timeseries.csv
    - metrics.csv
    - summary.json
    - timeseries_data.npz
    - figures

Default parameter grid:
    n_cut in {12, 16, 20}
    kappa in {0.1, 0.5, 1.0}

Default initial state:
    phi1_0 = 3.0, phi2_0 = 0.0
    Pi1_0 = Pi2_0 = 0.0
"""

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
from scipy.linalg import eigh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def hermitize(mat):
    return 0.5 * (mat + mat.conj().T)


def destroy_operator(n_cut):
    a = np.zeros((n_cut, n_cut), dtype=np.complex128)
    for n in range(1, n_cut):
        a[n - 1, n] = np.sqrt(n)
    return a


def coherent_state(alpha, n_cut):
    psi = np.zeros(n_cut, dtype=np.complex128)
    psi[0] = np.exp(-0.5 * abs(alpha) ** 2)

    for n in range(1, n_cut):
        psi[n] = psi[n - 1] * alpha / np.sqrt(n)

    norm = np.linalg.norm(psi)
    if norm < 1e-14:
        raise RuntimeError("Truncated coherent state has near-zero norm.")

    return psi / norm


def matrix_function_hermitian(op, func):
    evals, evecs = eigh(hermitize(op))
    return hermitize((evecs * func(evals)) @ evecs.conj().T)


def kron2(A, B):
    return np.kron(A, B)


def build_two_site_operators(n_cut, EJ, kappa):
    """
    Build two-site operators.

    Site ordering:
        Hilbert space = site 1 tensor site 2.
    """
    a = destroy_operator(n_cut)
    adag = a.conj().T

    I1 = np.eye(n_cut, dtype=np.complex128)

    phi = hermitize(a + adag)
    Pi = hermitize(-1j * (a - adag))

    phi2_local = hermitize(phi @ phi)
    phi4_local = hermitize(phi2_local @ phi2_local)
    phi6_local = hermitize(phi4_local @ phi2_local)

    Pi2_local = hermitize(Pi @ Pi)

    cos_phi_local = matrix_function_hermitian(phi, np.cos)

    # Full two-site operators
    I = kron2(I1, I1)

    phi1 = kron2(phi, I1)
    phi2 = kron2(I1, phi)

    Pi1 = kron2(Pi, I1)
    Pi2 = kron2(I1, Pi)

    n_local = hermitize(adag @ a)
    n1 = kron2(n_local, I1)
    n2 = kron2(I1, n_local)

    kinetic = hermitize(0.5 * kron2(Pi2_local, I1) + 0.5 * kron2(I1, Pi2_local))

    V_cos = hermitize(
        EJ * (kron2(I1 - cos_phi_local, I1) + kron2(I1, I1 - cos_phi_local))
    )

    V4 = hermitize(
        EJ * (
            kron2(0.5 * phi2_local - phi4_local / 24.0, I1)
            + kron2(I1, 0.5 * phi2_local - phi4_local / 24.0)
        )
    )

    V6 = hermitize(
        EJ * (
            kron2(
                0.5 * phi2_local - phi4_local / 24.0 + phi6_local / 720.0,
                I1,
            )
            + kron2(
                I1,
                0.5 * phi2_local - phi4_local / 24.0 + phi6_local / 720.0,
            )
        )
    )

    dphi = hermitize(phi2 - phi1)
    coupling = hermitize(0.5 * kappa * (dphi @ dphi))

    H_cos = hermitize(kinetic + V_cos + coupling)
    H4 = hermitize(kinetic + V4 + coupling)
    H6 = hermitize(kinetic + V6 + coupling)

    return {
        "I": I,
        "phi1": phi1,
        "phi2": phi2,
        "Pi1": Pi1,
        "Pi2": Pi2,
        "n1": n1,
        "n2": n2,
        "dphi2": hermitize(dphi @ dphi),
        "kinetic": kinetic,
        "V_cos_total": hermitize(V_cos + coupling),
        "V4_total": hermitize(V4 + coupling),
        "V6_total": hermitize(V6 + coupling),
        "coupling": coupling,
        "H_cos": H_cos,
        "H4": H4,
        "H6": H6,
    }


def evolve_exact(H, psi0, times):
    """
    Exact finite-dimensional evolution.

        |psi(t)> = exp(-i H t) |psi0>

    Returns:
        states[t_index, basis_index]
    """
    evals, evecs = eigh(hermitize(H))
    coeff0 = evecs.conj().T @ psi0

    phase = np.exp(-1j * np.outer(times, evals))
    coeff_t = phase * coeff0[None, :]

    states = coeff_t @ evecs.T

    # Normalize against small numerical drift
    norms = np.linalg.norm(states, axis=1)
    states = states / norms[:, None]

    return states


def unitary_from_hamiltonian(H, dt):
    evals, evecs = eigh(hermitize(H))
    return (evecs * np.exp(-1j * dt * evals)) @ evecs.conj().T


def evolve_trotter_strang(kinetic, potential, psi0, n_steps, dt):
    """
    Strang splitting:

        exp[-i(T+V)dt] ~ exp[-iT dt/2] exp[-iV dt] exp[-iT dt/2]

    Here:
        T = kinetic
        V = native cosine onsite potential + coupling term

    The coupling is included in V because it depends only on phi operators.
    """
    U_T_half = unitary_from_hamiltonian(kinetic, 0.5 * dt)
    U_V = unitary_from_hamiltonian(potential, dt)

    U_step = U_T_half @ U_V @ U_T_half

    dim = psi0.size
    states = np.zeros((n_steps + 1, dim), dtype=np.complex128)
    states[0] = psi0

    psi = psi0.copy()
    for step in range(1, n_steps + 1):
        psi = U_step @ psi
        psi = psi / np.linalg.norm(psi)
        states[step] = psi

    return states


def expectation(states, op):
    vals = np.einsum("ti,ij,tj->t", states.conj(), op, states)
    return np.real_if_close(vals)


def fidelity(states_ref, states_test):
    overlaps = np.einsum("ti,ti->t", states_ref.conj(), states_test)
    return np.abs(overlaps) ** 2


def relative_energy_drift(energy):
    e0 = float(np.real(energy[0]))
    denom = max(abs(e0), 1e-12)
    return float(np.max(np.abs(np.real(energy) - e0)) / denom)


def field_vector_error(phi1_ref, phi2_ref, phi1_test, phi2_test):
    diff = np.sqrt((np.real(phi1_ref - phi1_test)) ** 2 + (np.real(phi2_ref - phi2_test)) ** 2)
    ref_norm = np.sqrt(np.real(phi1_ref) ** 2 + np.real(phi2_ref) ** 2)
    denom = max(float(np.max(ref_norm)), 1e-12)
    return diff / denom


def make_plots(case_dir, times, data, n_cut, kappa, phi1_0, phi2_0):
    title_suffix = (
        r"$n_{\rm cut}=%d$, $\kappa=%.3g$, $\phi_{1,0}=%.2g$, $\phi_{2,0}=%.2g$"
        % (n_cut, kappa, phi1_0, phi2_0)
    )

    # Infidelity
    plt.figure(figsize=(7.2, 4.6))
    eps = 1e-16
    plt.semilogy(times, np.maximum(1.0 - data["F4"], eps), label="Quartic truncation")
    plt.semilogy(times, np.maximum(1.0 - data["F6"], eps), label="Sixth-order truncation")
    plt.semilogy(times, np.maximum(1.0 - data["Ftrot"], eps), label="Trotterized native cosine")
    plt.xlabel("Time")
    plt.ylabel("Infidelity")
    plt.title("Two-site native Josephson benchmark: " + title_suffix)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "infidelity_vs_time.png"), dpi=250)
    plt.close()

    # Fidelity
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(times, data["F4"], label="Quartic vs cosine")
    plt.plot(times, data["F6"], label="Sixth-order vs cosine")
    plt.plot(times, data["Ftrot"], label="Trotterized native cosine vs exact")
    plt.xlabel("Time")
    plt.ylabel("Fidelity")
    plt.title("Two-site fidelity: " + title_suffix)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "fidelity_vs_time.png"), dpi=250)
    plt.close()

    # phi1
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(times, np.real(data["phi1_cos"]), label="Exact native cosine")
    plt.plot(times, np.real(data["phi1_4"]), "--", label="Quartic truncation")
    plt.plot(times, np.real(data["phi1_6"]), "-.", label="Sixth-order truncation")
    plt.plot(times, np.real(data["phi1_trot"]), ":", label="Trotterized native cosine")
    plt.xlabel("Time")
    plt.ylabel(r"$\langle \hat{\phi}_1(t)\rangle$")
    plt.title("Site 1 field expectation: " + title_suffix)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "phi1_expectation_vs_time.png"), dpi=250)
    plt.close()

    # phi2
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(times, np.real(data["phi2_cos"]), label="Exact native cosine")
    plt.plot(times, np.real(data["phi2_4"]), "--", label="Quartic truncation")
    plt.plot(times, np.real(data["phi2_6"]), "-.", label="Sixth-order truncation")
    plt.plot(times, np.real(data["phi2_trot"]), ":", label="Trotterized native cosine")
    plt.xlabel("Time")
    plt.ylabel(r"$\langle \hat{\phi}_2(t)\rangle$")
    plt.title("Site 2 field expectation: " + title_suffix)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "phi2_expectation_vs_time.png"), dpi=250)
    plt.close()

    # Field vector error
    plt.figure(figsize=(7.2, 4.6))
    plt.semilogy(times, np.maximum(data["err_phi_4"], eps), label="Quartic truncation")
    plt.semilogy(times, np.maximum(data["err_phi_6"], eps), label="Sixth-order truncation")
    plt.semilogy(times, np.maximum(data["err_phi_trot"], eps), label="Trotterized native cosine")
    plt.xlabel("Time")
    plt.ylabel(r"Relative field-vector error")
    plt.title("Two-site field-vector error: " + title_suffix)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "field_vector_error_vs_time.png"), dpi=250)
    plt.close()

    # Coupling observable
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(times, np.real(data["dphi2_cos"]), label="Exact native cosine")
    plt.plot(times, np.real(data["dphi2_4"]), "--", label="Quartic truncation")
    plt.plot(times, np.real(data["dphi2_6"]), "-.", label="Sixth-order truncation")
    plt.plot(times, np.real(data["dphi2_trot"]), ":", label="Trotterized native cosine")
    plt.xlabel("Time")
    plt.ylabel(r"$\langle(\hat{\phi}_2-\hat{\phi}_1)^2\rangle$")
    plt.title("Coupling observable: " + title_suffix)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "coupling_observable_vs_time.png"), dpi=250)
    plt.close()


def write_timeseries_csv(path, times, data):
    headers = [
        "time",
        "F_quartic",
        "F_sixth",
        "F_trotter",
        "infid_quartic",
        "infid_sixth",
        "infid_trotter",
        "phi1_cos",
        "phi2_cos",
        "phi1_quartic",
        "phi2_quartic",
        "phi1_sixth",
        "phi2_sixth",
        "phi1_trotter",
        "phi2_trotter",
        "field_vector_error_quartic",
        "field_vector_error_sixth",
        "field_vector_error_trotter",
        "coupling_cos",
        "coupling_quartic",
        "coupling_sixth",
        "coupling_trotter",
        "n1_cos",
        "n2_cos",
        "E_cos_reference",
        "E_cos_trotter",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(len(times)):
            writer.writerow([
                float(times[i]),
                float(data["F4"][i]),
                float(data["F6"][i]),
                float(data["Ftrot"][i]),
                float(1.0 - data["F4"][i]),
                float(1.0 - data["F6"][i]),
                float(1.0 - data["Ftrot"][i]),
                float(np.real(data["phi1_cos"][i])),
                float(np.real(data["phi2_cos"][i])),
                float(np.real(data["phi1_4"][i])),
                float(np.real(data["phi2_4"][i])),
                float(np.real(data["phi1_6"][i])),
                float(np.real(data["phi2_6"][i])),
                float(np.real(data["phi1_trot"][i])),
                float(np.real(data["phi2_trot"][i])),
                float(np.real(data["err_phi_4"][i])),
                float(np.real(data["err_phi_6"][i])),
                float(np.real(data["err_phi_trot"][i])),
                float(np.real(data["dphi2_cos"][i])),
                float(np.real(data["dphi2_4"][i])),
                float(np.real(data["dphi2_6"][i])),
                float(np.real(data["dphi2_trot"][i])),
                float(np.real(data["n1_cos"][i])),
                float(np.real(data["n2_cos"][i])),
                float(np.real(data["E_cos_ref"][i])),
                float(np.real(data["E_cos_trot"][i])),
            ])


def write_metrics_csv(path, metrics):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def run_case(args):
    ncut_list = parse_int_list(args.ncut_list)
    kappa_list = parse_float_list(args.kappa_list)

    cases = []
    for n_cut in ncut_list:
        for kappa in kappa_list:
            cases.append((n_cut, kappa))

    n_cases = len(cases)

    if args.case_index < 0 or args.case_index >= n_cases:
        raise ValueError(
            "case_index=%d is out of range. Valid range: 0 to %d."
            % (args.case_index, n_cases - 1)
        )

    n_cut, kappa = cases[args.case_index]

    case_dir = os.path.join(
        args.outdir,
        "case_%03d_ncut_%d_kappa_%g_phi1_%g_phi2_%g"
        % (args.case_index, n_cut, kappa, args.phi1_0, args.phi2_0),
    )
    ensure_dir(case_dir)

    print("=" * 80)
    print("Two-site native Josephson / Sine-Gordon benchmark")
    print("case_index = %d / %d" % (args.case_index, n_cases - 1))
    print("n_cut      = %d" % n_cut)
    print("kappa      = %.12g" % kappa)
    print("EJ         = %.12g" % args.EJ)
    print("phi1_0     = %.12g" % args.phi1_0)
    print("phi2_0     = %.12g" % args.phi2_0)
    print("Pi1_0      = %.12g" % args.Pi1_0)
    print("Pi2_0      = %.12g" % args.Pi2_0)
    print("T          = %.12g" % args.T)
    print("n_steps    = %d" % args.n_steps)
    print("outdir     = %s" % case_dir)
    print("=" * 80, flush=True)

    t_start = time.time()

    ops = build_two_site_operators(n_cut=n_cut, EJ=args.EJ, kappa=kappa)

    alpha1 = 0.5 * (args.phi1_0 + 1j * args.Pi1_0)
    alpha2 = 0.5 * (args.phi2_0 + 1j * args.Pi2_0)

    psi1 = coherent_state(alpha1, n_cut)
    psi2 = coherent_state(alpha2, n_cut)
    psi0 = np.kron(psi1, psi2)
    psi0 = psi0 / np.linalg.norm(psi0)

    times = np.linspace(0.0, args.T, args.n_steps + 1)
    dt = args.T / args.n_steps

    # Exact evolutions
    states_cos = evolve_exact(ops["H_cos"], psi0, times)
    states4 = evolve_exact(ops["H4"], psi0, times)
    states6 = evolve_exact(ops["H6"], psi0, times)

    # Trotterized native-cosine evolution
    states_trot = evolve_trotter_strang(
        kinetic=ops["kinetic"],
        potential=ops["V_cos_total"],
        psi0=psi0,
        n_steps=args.n_steps,
        dt=dt,
    )

    # Fidelities
    F4 = fidelity(states_cos, states4)
    F6 = fidelity(states_cos, states6)
    Ftrot = fidelity(states_cos, states_trot)

    # Field observables
    phi1_cos = expectation(states_cos, ops["phi1"])
    phi2_cos = expectation(states_cos, ops["phi2"])

    phi1_4 = expectation(states4, ops["phi1"])
    phi2_4 = expectation(states4, ops["phi2"])

    phi1_6 = expectation(states6, ops["phi1"])
    phi2_6 = expectation(states6, ops["phi2"])

    phi1_trot = expectation(states_trot, ops["phi1"])
    phi2_trot = expectation(states_trot, ops["phi2"])

    err_phi_4 = field_vector_error(phi1_cos, phi2_cos, phi1_4, phi2_4)
    err_phi_6 = field_vector_error(phi1_cos, phi2_cos, phi1_6, phi2_6)
    err_phi_trot = field_vector_error(phi1_cos, phi2_cos, phi1_trot, phi2_trot)

    # Coupling observable
    dphi2_cos = expectation(states_cos, ops["dphi2"])
    dphi2_4 = expectation(states4, ops["dphi2"])
    dphi2_6 = expectation(states6, ops["dphi2"])
    dphi2_trot = expectation(states_trot, ops["dphi2"])

    # Occupation
    n1_cos = expectation(states_cos, ops["n1"])
    n2_cos = expectation(states_cos, ops["n2"])

    # Energy
    E_cos_ref = expectation(states_cos, ops["H_cos"])
    E_cos_trot = expectation(states_trot, ops["H_cos"])

    data = {
        "F4": F4,
        "F6": F6,
        "Ftrot": Ftrot,
        "phi1_cos": phi1_cos,
        "phi2_cos": phi2_cos,
        "phi1_4": phi1_4,
        "phi2_4": phi2_4,
        "phi1_6": phi1_6,
        "phi2_6": phi2_6,
        "phi1_trot": phi1_trot,
        "phi2_trot": phi2_trot,
        "err_phi_4": err_phi_4,
        "err_phi_6": err_phi_6,
        "err_phi_trot": err_phi_trot,
        "dphi2_cos": dphi2_cos,
        "dphi2_4": dphi2_4,
        "dphi2_6": dphi2_6,
        "dphi2_trot": dphi2_trot,
        "n1_cos": n1_cos,
        "n2_cos": n2_cos,
        "E_cos_ref": E_cos_ref,
        "E_cos_trot": E_cos_trot,
    }

    elapsed = time.time() - t_start

    metrics = {
        "case_index": args.case_index,
        "n_cases": n_cases,
        "n_cut": n_cut,
        "hilbert_dim": n_cut ** 2,
        "EJ": args.EJ,
        "kappa": kappa,
        "phi1_0": args.phi1_0,
        "phi2_0": args.phi2_0,
        "Pi1_0": args.Pi1_0,
        "Pi2_0": args.Pi2_0,
        "T": args.T,
        "n_steps": args.n_steps,
        "dt": dt,
        "final_F_quartic": float(F4[-1]),
        "final_F_sixth": float(F6[-1]),
        "final_F_trotter": float(Ftrot[-1]),
        "max_infidelity_quartic": float(np.max(1.0 - F4)),
        "max_infidelity_sixth": float(np.max(1.0 - F6)),
        "max_infidelity_trotter": float(np.max(1.0 - Ftrot)),
        "max_field_vector_error_quartic": float(np.max(err_phi_4)),
        "max_field_vector_error_sixth": float(np.max(err_phi_6)),
        "max_field_vector_error_trotter": float(np.max(err_phi_trot)),
        "final_field_vector_error_quartic": float(err_phi_4[-1]),
        "final_field_vector_error_sixth": float(err_phi_6[-1]),
        "final_field_vector_error_trotter": float(err_phi_trot[-1]),
        "max_coupling_observable_cos": float(np.max(np.real(dphi2_cos))),
        "max_coupling_error_quartic": float(np.max(np.abs(np.real(dphi2_cos - dphi2_4)))),
        "max_coupling_error_sixth": float(np.max(np.abs(np.real(dphi2_cos - dphi2_6)))),
        "max_coupling_error_trotter": float(np.max(np.abs(np.real(dphi2_cos - dphi2_trot)))),
        "cos_energy_drift_reference": relative_energy_drift(E_cos_ref),
        "cos_energy_drift_trotter": relative_energy_drift(E_cos_trot),
        "max_mean_n1_reference": float(np.max(np.real(n1_cos))),
        "max_mean_n2_reference": float(np.max(np.real(n2_cos))),
        "max_total_mean_n_reference": float(np.max(np.real(n1_cos + n2_cos))),
        "walltime_seconds": float(elapsed),
    }

    # Save data
    np.savez_compressed(
        os.path.join(case_dir, "timeseries_data.npz"),
        times=times,
        psi0=psi0,
        **data
    )

    write_timeseries_csv(os.path.join(case_dir, "timeseries.csv"), times, data)
    write_metrics_csv(os.path.join(case_dir, "metrics.csv"), metrics)

    with open(os.path.join(case_dir, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.make_plots:
        make_plots(
            case_dir=case_dir,
            times=times,
            data=data,
            n_cut=n_cut,
            kappa=kappa,
            phi1_0=args.phi1_0,
            phi2_0=args.phi2_0,
        )

    print("Finished case.")
    print(json.dumps(metrics, indent=2), flush=True)


def aggregate_results(args):
    ensure_dir(args.outdir)
    metric_files = sorted(glob.glob(os.path.join(args.outdir, "case_*", "metrics.csv")))

    if not metric_files:
        raise RuntimeError("No metrics.csv files found under %s/case_*" % args.outdir)

    rows = []
    for mf in metric_files:
        with open(mf, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    rows.sort(key=lambda r: (int(r["n_cut"]), float(r["kappa"])))

    out_csv = os.path.join(args.outdir, "aggregate_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Wrote aggregate summary:", out_csv)

    # Simple aggregate plots
    ncuts = sorted({int(r["n_cut"]) for r in rows})
    kappas = sorted({float(r["kappa"]) for r in rows})

    plot_specs = [
        ("max_infidelity_quartic", "Max infidelity: quartic", "aggregate_quartic_max_infid.png"),
        ("max_infidelity_sixth", "Max infidelity: sixth-order", "aggregate_sixth_max_infid.png"),
        ("max_infidelity_trotter", "Max infidelity: native Trotter", "aggregate_trotter_max_infid.png"),
        ("max_field_vector_error_quartic", "Max field-vector error: quartic", "aggregate_quartic_field_error.png"),
        ("max_field_vector_error_sixth", "Max field-vector error: sixth-order", "aggregate_sixth_field_error.png"),
        ("max_field_vector_error_trotter", "Max field-vector error: native Trotter", "aggregate_trotter_field_error.png"),
    ]

    for metric_name, ylabel, filename in plot_specs:
        plt.figure(figsize=(7.2, 4.6))
        for ncut in ncuts:
            xs = []
            ys = []
            for kappa in kappas:
                match = [
                    r for r in rows
                    if int(r["n_cut"]) == ncut and abs(float(r["kappa"]) - kappa) < 1e-12
                ]
                if match:
                    xs.append(kappa)
                    ys.append(float(match[0][metric_name]))

            if xs:
                plt.semilogy(xs, ys, marker="o", label="n_cut=%d" % ncut)

        plt.xlabel(r"Coupling $\kappa$")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, filename), dpi=250)
        plt.close()

    print("Aggregate plots written.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        default=os.environ.get("OUTDIR", "results/sg_two_site_native_jj"),
    )

    parser.add_argument(
        "--case-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )

    parser.add_argument(
        "--ncut-list",
        type=str,
        default=os.environ.get("NCUT_LIST", "12,16,20"),
    )

    parser.add_argument(
        "--kappa-list",
        type=str,
        default=os.environ.get("KAPPA_LIST", "0.1,0.5,1.0"),
    )

    parser.add_argument(
        "--EJ",
        type=float,
        default=float(os.environ.get("EJ", "1.0")),
    )

    parser.add_argument(
        "--phi1-0",
        type=float,
        default=float(os.environ.get("PHI1_0", "3.0")),
    )

    parser.add_argument(
        "--phi2-0",
        type=float,
        default=float(os.environ.get("PHI2_0", "0.0")),
    )

    parser.add_argument(
        "--Pi1-0",
        type=float,
        default=float(os.environ.get("PI1_0", "0.0")),
    )

    parser.add_argument(
        "--Pi2-0",
        type=float,
        default=float(os.environ.get("PI2_0", "0.0")),
    )

    parser.add_argument(
        "--T",
        type=float,
        default=float(os.environ.get("T_FINAL", "5.0")),
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=int(os.environ.get("N_STEPS", "800")),
    )

    parser.add_argument(
        "--make-plots",
        action="store_true",
    )

    parser.add_argument(
        "--aggregate",
        action="store_true",
    )

    args = parser.parse_args()
    ensure_dir(args.outdir)

    if args.aggregate:
        aggregate_results(args)
    else:
        run_case(args)


if __name__ == "__main__":
    main()
