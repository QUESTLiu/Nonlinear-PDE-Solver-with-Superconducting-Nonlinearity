#!/usr/bin/env python3
"""
Two-site native Josephson / Sine-Gordon programmable correction-layer benchmark.

Target Hamiltonian:
    H_target = H_native + DeltaH

where

    H_native =
        sum_j [ 1/2 Pi_j^2 + EJ (1 - cos(phi_j)) ]
        + kappa/2 (phi_2 - phi_1)^2

and

    DeltaH = f1 phi_1 + f2 phi_2.

This tests the CV-DV augmentation story:

    native Josephson cosine supplies the nonlinear core;
    programmable correction layer supplies non-native perturbations.

Comparisons:
    1. Exact H_target                         reference
    2. Exact H_native                         native-only, no correction
    3. Strang Trotter H_target                native + correction implementation
    4. Quartic polynomial target approximation
    5. Sixth-order polynomial target approximation

Default scan:
    n_cut = 20
    kappa = 1.0
    f in {0.05, 0.1, 0.2, 0.4}
    f1 = f
    f2 = -f
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


def build_two_site_operators(n_cut, EJ, kappa, f1, f2):
    """
    Hilbert space ordering:
        H = H_site1 tensor H_site2.
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

    # Full operators
    phi1 = kron2(phi, I1)
    phi2 = kron2(I1, phi)

    Pi1 = kron2(Pi, I1)
    Pi2 = kron2(I1, Pi)

    n_local = hermitize(adag @ a)
    n1 = kron2(n_local, I1)
    n2 = kron2(I1, n_local)

    kinetic = hermitize(
        0.5 * kron2(Pi2_local, I1)
        + 0.5 * kron2(I1, Pi2_local)
    )

    V_cos = hermitize(
        EJ * (
            kron2(I1 - cos_phi_local, I1)
            + kron2(I1, I1 - cos_phi_local)
        )
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
                0.5 * phi2_local
                - phi4_local / 24.0
                + phi6_local / 720.0,
                I1,
            )
            + kron2(
                I1,
                0.5 * phi2_local
                - phi4_local / 24.0
                + phi6_local / 720.0,
            )
        )
    )

    dphi = hermitize(phi2 - phi1)
    coupling = hermitize(0.5 * kappa * (dphi @ dphi))

    bias = hermitize(f1 * phi1 + f2 * phi2)

    # Hamiltonians
    H_native = hermitize(kinetic + V_cos + coupling)
    H_target = hermitize(H_native + bias)

    H4_target = hermitize(kinetic + V4 + coupling + bias)
    H6_target = hermitize(kinetic + V6 + coupling + bias)

    V_target = hermitize(V_cos + coupling + bias)

    return {
        "phi1": phi1,
        "phi2": phi2,
        "Pi1": Pi1,
        "Pi2": Pi2,
        "n1": n1,
        "n2": n2,
        "dphi2": hermitize(dphi @ dphi),
        "kinetic": kinetic,
        "V_target": V_target,
        "bias": bias,
        "coupling": coupling,
        "H_native": H_native,
        "H_target": H_target,
        "H4_target": H4_target,
        "H6_target": H6_target,
    }


def evolve_exact(H, psi0, times):
    evals, evecs = eigh(hermitize(H))
    coeff0 = evecs.conj().T @ psi0
    phase = np.exp(-1j * np.outer(times, evals))
    coeff_t = phase * coeff0[None, :]
    states = coeff_t @ evecs.T
    states = states / np.linalg.norm(states, axis=1)[:, None]
    return states


def unitary_from_hamiltonian(H, dt):
    evals, evecs = eigh(hermitize(H))
    return (evecs * np.exp(-1j * dt * evals)) @ evecs.conj().T


def evolve_trotter_strang(kinetic, potential, psi0, n_steps, dt):
    """
    Strang splitting:
        exp[-i(T+V)dt] ~ exp[-iTdt/2] exp[-iVdt] exp[-iTdt/2]

    Here V contains:
        native cosine + coupling + programmable bias correction.
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
    diff = np.sqrt(
        (np.real(phi1_ref - phi1_test)) ** 2
        + (np.real(phi2_ref - phi2_test)) ** 2
    )
    ref_norm = np.sqrt(np.real(phi1_ref) ** 2 + np.real(phi2_ref) ** 2)
    denom = max(float(np.max(ref_norm)), 1e-12)
    return diff / denom


def make_plots(case_dir, times, data, n_cut, kappa, f1, f2):
    title = (
        r"$n_{\rm cut}=%d$, $\kappa=%.3g$, $f_1=%.3g$, $f_2=%.3g$"
        % (n_cut, kappa, f1, f2)
    )

    eps = 1e-16

    # Infidelity
    plt.figure(figsize=(7.2, 4.6))
    plt.semilogy(times, np.maximum(1.0 - data["F_native"], eps), label="Native-only")
    plt.semilogy(times, np.maximum(1.0 - data["F_aug"], eps), label="Native + correction")
    plt.semilogy(times, np.maximum(1.0 - data["F4"], eps), label="Quartic target")
    plt.semilogy(times, np.maximum(1.0 - data["F6"], eps), label="Sixth-order target")
    plt.xlabel("Time")
    plt.ylabel("Infidelity")
    plt.title("Correction-layer benchmark: " + title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "infidelity_vs_time.png"), dpi=250)
    plt.close()

    # Field-vector error
    plt.figure(figsize=(7.2, 4.6))
    plt.semilogy(times, np.maximum(data["err_native"], eps), label="Native-only")
    plt.semilogy(times, np.maximum(data["err_aug"], eps), label="Native + correction")
    plt.semilogy(times, np.maximum(data["err_4"], eps), label="Quartic target")
    plt.semilogy(times, np.maximum(data["err_6"], eps), label="Sixth-order target")
    plt.xlabel("Time")
    plt.ylabel(r"Relative field-vector error")
    plt.title("Field-vector error: " + title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "field_vector_error_vs_time.png"), dpi=250)
    plt.close()

    # Site 1 field
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(times, np.real(data["phi1_target"]), label="Exact target")
    plt.plot(times, np.real(data["phi1_native"]), "--", label="Native-only")
    plt.plot(times, np.real(data["phi1_aug"]), ":", label="Native + correction")
    plt.plot(times, np.real(data["phi1_4"]), "-.", label="Quartic target")
    plt.plot(times, np.real(data["phi1_6"]), label="Sixth-order target")
    plt.xlabel("Time")
    plt.ylabel(r"$\langle \hat{\phi}_1(t)\rangle$")
    plt.title("Site 1 field: " + title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "phi1_expectation_vs_time.png"), dpi=250)
    plt.close()

    # Site 2 field
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(times, np.real(data["phi2_target"]), label="Exact target")
    plt.plot(times, np.real(data["phi2_native"]), "--", label="Native-only")
    plt.plot(times, np.real(data["phi2_aug"]), ":", label="Native + correction")
    plt.plot(times, np.real(data["phi2_4"]), "-.", label="Quartic target")
    plt.plot(times, np.real(data["phi2_6"]), label="Sixth-order target")
    plt.xlabel("Time")
    plt.ylabel(r"$\langle \hat{\phi}_2(t)\rangle$")
    plt.title("Site 2 field: " + title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "phi2_expectation_vs_time.png"), dpi=250)
    plt.close()

    # Coupling observable
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(times, np.real(data["dphi2_target"]), label="Exact target")
    plt.plot(times, np.real(data["dphi2_native"]), "--", label="Native-only")
    plt.plot(times, np.real(data["dphi2_aug"]), ":", label="Native + correction")
    plt.plot(times, np.real(data["dphi2_4"]), "-.", label="Quartic target")
    plt.plot(times, np.real(data["dphi2_6"]), label="Sixth-order target")
    plt.xlabel("Time")
    plt.ylabel(r"$\langle(\hat{\phi}_2-\hat{\phi}_1)^2\rangle$")
    plt.title("Coupling observable: " + title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "coupling_observable_vs_time.png"), dpi=250)
    plt.close()


def write_timeseries_csv(path, times, data):
    headers = [
        "time",
        "F_native_only",
        "F_augmented",
        "F_quartic_target",
        "F_sixth_target",
        "infid_native_only",
        "infid_augmented",
        "infid_quartic_target",
        "infid_sixth_target",
        "field_error_native_only",
        "field_error_augmented",
        "field_error_quartic_target",
        "field_error_sixth_target",
        "phi1_target",
        "phi2_target",
        "phi1_native_only",
        "phi2_native_only",
        "phi1_augmented",
        "phi2_augmented",
        "phi1_quartic_target",
        "phi2_quartic_target",
        "phi1_sixth_target",
        "phi2_sixth_target",
        "coupling_target",
        "coupling_native_only",
        "coupling_augmented",
        "coupling_quartic_target",
        "coupling_sixth_target",
        "E_target_reference",
        "E_target_augmented",
        "n1_target",
        "n2_target",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(len(times)):
            writer.writerow([
                float(times[i]),
                float(data["F_native"][i]),
                float(data["F_aug"][i]),
                float(data["F4"][i]),
                float(data["F6"][i]),
                float(1.0 - data["F_native"][i]),
                float(1.0 - data["F_aug"][i]),
                float(1.0 - data["F4"][i]),
                float(1.0 - data["F6"][i]),
                float(data["err_native"][i]),
                float(data["err_aug"][i]),
                float(data["err_4"][i]),
                float(data["err_6"][i]),
                float(np.real(data["phi1_target"][i])),
                float(np.real(data["phi2_target"][i])),
                float(np.real(data["phi1_native"][i])),
                float(np.real(data["phi2_native"][i])),
                float(np.real(data["phi1_aug"][i])),
                float(np.real(data["phi2_aug"][i])),
                float(np.real(data["phi1_4"][i])),
                float(np.real(data["phi2_4"][i])),
                float(np.real(data["phi1_6"][i])),
                float(np.real(data["phi2_6"][i])),
                float(np.real(data["dphi2_target"][i])),
                float(np.real(data["dphi2_native"][i])),
                float(np.real(data["dphi2_aug"][i])),
                float(np.real(data["dphi2_4"][i])),
                float(np.real(data["dphi2_6"][i])),
                float(np.real(data["E_target_ref"][i])),
                float(np.real(data["E_target_aug"][i])),
                float(np.real(data["n1_target"][i])),
                float(np.real(data["n2_target"][i])),
            ])


def write_metrics_csv(path, metrics):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def run_case(args):
    ncut_list = parse_int_list(args.ncut_list)
    kappa_list = parse_float_list(args.kappa_list)
    f_list = parse_float_list(args.f_list)

    cases = []
    for n_cut in ncut_list:
        for kappa in kappa_list:
            for f in f_list:
                f1 = args.f1_scale * f
                f2 = args.f2_scale * f
                cases.append((n_cut, kappa, f, f1, f2))

    n_cases = len(cases)

    if args.case_index < 0 or args.case_index >= n_cases:
        raise ValueError(
            "case_index=%d is out of range. Valid range: 0 to %d."
            % (args.case_index, n_cases - 1)
        )

    n_cut, kappa, f, f1, f2 = cases[args.case_index]

    case_dir = os.path.join(
        args.outdir,
        "case_%03d_ncut_%d_kappa_%g_f_%g_f1_%g_f2_%g"
        % (args.case_index, n_cut, kappa, f, f1, f2),
    )
    ensure_dir(case_dir)

    print("=" * 80)
    print("Two-site programmable correction-layer benchmark")
    print("case_index = %d / %d" % (args.case_index, n_cases - 1))
    print("n_cut      = %d" % n_cut)
    print("kappa      = %.12g" % kappa)
    print("EJ         = %.12g" % args.EJ)
    print("f          = %.12g" % f)
    print("f1         = %.12g" % f1)
    print("f2         = %.12g" % f2)
    print("phi1_0     = %.12g" % args.phi1_0)
    print("phi2_0     = %.12g" % args.phi2_0)
    print("pi1_0      = %.12g" % args.pi1_0)
    print("pi2_0      = %.12g" % args.pi2_0)
    print("T          = %.12g" % args.T)
    print("n_steps    = %d" % args.n_steps)
    print("outdir     = %s" % case_dir)
    print("=" * 80, flush=True)

    t_start = time.time()

    ops = build_two_site_operators(
        n_cut=n_cut,
        EJ=args.EJ,
        kappa=kappa,
        f1=f1,
        f2=f2,
    )

    alpha1 = 0.5 * (args.phi1_0 + 1j * args.pi1_0)
    alpha2 = 0.5 * (args.phi2_0 + 1j * args.pi2_0)

    psi1 = coherent_state(alpha1, n_cut)
    psi2 = coherent_state(alpha2, n_cut)
    psi0 = np.kron(psi1, psi2)
    psi0 = psi0 / np.linalg.norm(psi0)

    times = np.linspace(0.0, args.T, args.n_steps + 1)
    dt = args.T / args.n_steps

    # Reference and model evolutions
    states_target = evolve_exact(ops["H_target"], psi0, times)
    states_native = evolve_exact(ops["H_native"], psi0, times)
    states4 = evolve_exact(ops["H4_target"], psi0, times)
    states6 = evolve_exact(ops["H6_target"], psi0, times)

    states_aug = evolve_trotter_strang(
        kinetic=ops["kinetic"],
        potential=ops["V_target"],
        psi0=psi0,
        n_steps=args.n_steps,
        dt=dt,
    )

    # Fidelities relative to exact target
    F_native = fidelity(states_target, states_native)
    F_aug = fidelity(states_target, states_aug)
    F4 = fidelity(states_target, states4)
    F6 = fidelity(states_target, states6)

    # Field observables
    phi1_target = expectation(states_target, ops["phi1"])
    phi2_target = expectation(states_target, ops["phi2"])

    phi1_native = expectation(states_native, ops["phi1"])
    phi2_native = expectation(states_native, ops["phi2"])

    phi1_aug = expectation(states_aug, ops["phi1"])
    phi2_aug = expectation(states_aug, ops["phi2"])

    phi1_4 = expectation(states4, ops["phi1"])
    phi2_4 = expectation(states4, ops["phi2"])

    phi1_6 = expectation(states6, ops["phi1"])
    phi2_6 = expectation(states6, ops["phi2"])

    err_native = field_vector_error(phi1_target, phi2_target, phi1_native, phi2_native)
    err_aug = field_vector_error(phi1_target, phi2_target, phi1_aug, phi2_aug)
    err_4 = field_vector_error(phi1_target, phi2_target, phi1_4, phi2_4)
    err_6 = field_vector_error(phi1_target, phi2_target, phi1_6, phi2_6)

    # Coupling observable
    dphi2_target = expectation(states_target, ops["dphi2"])
    dphi2_native = expectation(states_native, ops["dphi2"])
    dphi2_aug = expectation(states_aug, ops["dphi2"])
    dphi2_4 = expectation(states4, ops["dphi2"])
    dphi2_6 = expectation(states6, ops["dphi2"])

    # Occupation
    n1_target = expectation(states_target, ops["n1"])
    n2_target = expectation(states_target, ops["n2"])

    # Target energy drift
    E_target_ref = expectation(states_target, ops["H_target"])
    E_target_aug = expectation(states_aug, ops["H_target"])

    data = {
        "F_native": F_native,
        "F_aug": F_aug,
        "F4": F4,
        "F6": F6,
        "phi1_target": phi1_target,
        "phi2_target": phi2_target,
        "phi1_native": phi1_native,
        "phi2_native": phi2_native,
        "phi1_aug": phi1_aug,
        "phi2_aug": phi2_aug,
        "phi1_4": phi1_4,
        "phi2_4": phi2_4,
        "phi1_6": phi1_6,
        "phi2_6": phi2_6,
        "err_native": err_native,
        "err_aug": err_aug,
        "err_4": err_4,
        "err_6": err_6,
        "dphi2_target": dphi2_target,
        "dphi2_native": dphi2_native,
        "dphi2_aug": dphi2_aug,
        "dphi2_4": dphi2_4,
        "dphi2_6": dphi2_6,
        "n1_target": n1_target,
        "n2_target": n2_target,
        "E_target_ref": E_target_ref,
        "E_target_aug": E_target_aug,
    }

    elapsed = time.time() - t_start

    metrics = {
        "case_index": args.case_index,
        "n_cases": n_cases,
        "n_cut": n_cut,
        "hilbert_dim": n_cut ** 2,
        "EJ": args.EJ,
        "kappa": kappa,
        "f": f,
        "f1": f1,
        "f2": f2,
        "phi1_0": args.phi1_0,
        "phi2_0": args.phi2_0,
        "pi1_0": args.pi1_0,
        "pi2_0": args.pi2_0,
        "T": args.T,
        "n_steps": args.n_steps,
        "dt": dt,
        "final_F_native_only": float(F_native[-1]),
        "final_F_augmented": float(F_aug[-1]),
        "final_F_quartic_target": float(F4[-1]),
        "final_F_sixth_target": float(F6[-1]),
        "max_infidelity_native_only": float(np.max(1.0 - F_native)),
        "max_infidelity_augmented": float(np.max(1.0 - F_aug)),
        "max_infidelity_quartic_target": float(np.max(1.0 - F4)),
        "max_infidelity_sixth_target": float(np.max(1.0 - F6)),
        "max_field_error_native_only": float(np.max(err_native)),
        "max_field_error_augmented": float(np.max(err_aug)),
        "max_field_error_quartic_target": float(np.max(err_4)),
        "max_field_error_sixth_target": float(np.max(err_6)),
        "max_coupling_error_native_only": float(np.max(np.abs(np.real(dphi2_target - dphi2_native)))),
        "max_coupling_error_augmented": float(np.max(np.abs(np.real(dphi2_target - dphi2_aug)))),
        "max_coupling_error_quartic_target": float(np.max(np.abs(np.real(dphi2_target - dphi2_4)))),
        "max_coupling_error_sixth_target": float(np.max(np.abs(np.real(dphi2_target - dphi2_6)))),
        "target_energy_drift_reference": relative_energy_drift(E_target_ref),
        "target_energy_drift_augmented": relative_energy_drift(E_target_aug),
        "max_mean_n1_target": float(np.max(np.real(n1_target))),
        "max_mean_n2_target": float(np.max(np.real(n2_target))),
        "max_total_mean_n_target": float(np.max(np.real(n1_target + n2_target))),
        "walltime_seconds": float(elapsed),
    }

    np.savez_compressed(
        os.path.join(case_dir, "timeseries_data.npz"),
        times=times,
        psi0=psi0,
        **data
    )

    write_timeseries_csv(os.path.join(case_dir, "timeseries.csv"), times, data)
    write_metrics_csv(os.path.join(case_dir, "metrics.csv"), metrics)

    with open(os.path.join(case_dir, "summary.json"), "w") as fjson:
        json.dump(metrics, fjson, indent=2)

    if args.make_plots:
        make_plots(
            case_dir=case_dir,
            times=times,
            data=data,
            n_cut=n_cut,
            kappa=kappa,
            f1=f1,
            f2=f2,
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

    rows.sort(key=lambda r: (int(r["n_cut"]), float(r["kappa"]), float(r["f"])))

    out_csv = os.path.join(args.outdir, "aggregate_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Wrote aggregate summary:", out_csv)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        default=os.environ.get("OUTDIR", "results/sg_two_site_bias_correction"),
    )

    parser.add_argument(
        "--case-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )

    parser.add_argument(
        "--ncut-list",
        type=str,
        default=os.environ.get("NCUT_LIST", "20"),
    )

    parser.add_argument(
        "--kappa-list",
        type=str,
        default=os.environ.get("KAPPA_LIST", "1.0"),
    )

    parser.add_argument(
        "--f-list",
        type=str,
        default=os.environ.get("F_LIST", "0.05,0.1,0.2,0.4"),
    )

    parser.add_argument(
        "--f1-scale",
        type=float,
        default=float(os.environ.get("F1_SCALE", "1.0")),
    )

    parser.add_argument(
        "--f2-scale",
        type=float,
        default=float(os.environ.get("F2_SCALE", "-1.0")),
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
        "--pi1-0",
        type=float,
        default=float(os.environ.get("PI1_0", "0.0")),
    )

    parser.add_argument(
        "--pi2-0",
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
