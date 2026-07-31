#!/usr/bin/env python3
"""
duffing_snapd_scan.py

Native Kerr + squeezing + SNAP+D correction benchmark for the Duffing oscillator.

What it computes
----------------
1. Exact full Duffing evolution:
      U_D(T) = exp[-i T H_D]

2. Native Kerr/RWA evolution:
      U_K(T) = exp[-i T H_K]

3. Kerr + Gaussian squeezing correction:
      [ exp(-i dt H_K) exp(-i dt lambda V2) ]^nt

4. Kerr + Gaussian squeezing + variational SNAP+D quartic correction:
      [ exp(-i dt H_K) exp(-i dt lambda V2) U_SNAPD ]^nt

The variational SNAP+D block is trained in situ inside a fixed Kerr+squeezing 
symmetric step. The loss compares the repeated hybrid trajectory against the 
exact truncated Duffing trajectory at all intermediate time points. 
Therefore the trained block is a dressed quartic/counter-rotating correction, 
not a standalone approximation to exp[-i dt lambda V4].

Outputs
-------
- summary.csv
- fidelity_times.npz
- snapd_params.npz
- several PNG figures

Dependencies
------------
numpy scipy matplotlib pandas
"""

import argparse
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Basic oscillator construction
# -----------------------------

def annihilation(N: int) -> np.ndarray:
    a = np.zeros((N, N), dtype=np.complex128)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a


def coherent_state(N: int, alpha: complex) -> np.ndarray:
    """Truncated normalized coherent state."""
    coeffs = np.array([alpha**n / np.sqrt(math.factorial(n)) for n in range(N)], dtype=np.complex128)
    coeffs *= np.exp(-0.5 * abs(alpha)**2)
    norm = la.norm(coeffs)
    if norm == 0:
        raise ValueError("Coherent-state norm vanished.")
    return coeffs / norm


def displacement_operator(a: np.ndarray, alpha: complex) -> np.ndarray:
    adag = a.conj().T
    return la.expm(alpha * adag - np.conj(alpha) * a)


def snap_operator(theta: np.ndarray) -> np.ndarray:
    return np.diag(np.exp(1j * theta))


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    psi = psi / la.norm(psi)
    phi = phi / la.norm(phi)
    return float(abs(np.vdot(psi, phi))**2)


def state_infidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return 1.0 - fidelity(psi, phi)


@dataclass
class DuffingOperators:
    N: int
    omega0: float
    beta: float
    a: np.ndarray
    adag: np.ndarray
    n: np.ndarray
    q: np.ndarray
    p: np.ndarray
    H_duff: np.ndarray
    H_kerr: np.ndarray
    Vcr: np.ndarray
    V2: np.ndarray
    V4: np.ndarray
    lam: float
    omega_eff: float
    K_eff: float


def build_duffing_operators(N: int, omega0: float, beta: float) -> DuffingOperators:
    a = annihilation(N)
    adag = a.conj().T
    n = adag @ a
    q = (a + adag) / np.sqrt(2.0 * omega0)
    p = 1j * np.sqrt(omega0 / 2.0) * (adag - a)

    H_duff = 0.5 * (p @ p) + 0.5 * omega0**2 * (q @ q) + (beta / 4.0) * (q @ q @ q @ q)

    lam = beta / (16.0 * omega0**2)

    # Decomposition of (a + a^\dagger)^4 = 6n^2 + 6n + 3 + Vcr
    V2 = 6.0 * (a @ a + adag @ adag)
    V4 = (
        adag @ adag @ adag @ adag
        + 4.0 * adag @ adag @ adag @ a
        + a @ a @ a @ a
        + 4.0 * adag @ a @ a @ a
    )
    Vcr = V2 + V4

    K_eff = 3.0 * beta / (8.0 * omega0**2)
    omega_eff = omega0 + 3.0 * beta / (8.0 * omega0**2)

    # Up to a constant energy shift.
    H_kerr = omega_eff * n + K_eff * (n @ n)

    return DuffingOperators(
        N=N, omega0=omega0, beta=beta, a=a, adag=adag, n=n, q=q, p=p,
        H_duff=H_duff, H_kerr=H_kerr, Vcr=Vcr, V2=V2, V4=V4,
        lam=lam, omega_eff=omega_eff, K_eff=K_eff
    )


# -----------------------------
# SNAP+D ansatz and optimization
# -----------------------------

def unpack_params(x: np.ndarray, layers: int, N: int):
    """Each layer: Re alpha, Im alpha, theta_0,...,theta_{N-1}."""
    params = []
    idx = 0
    for _ in range(layers):
        alpha = x[idx] + 1j * x[idx + 1]
        idx += 2
        theta = x[idx:idx + N]
        idx += N
        params.append((alpha, theta))
    return params


def snapd_unitary(a: np.ndarray, x: np.ndarray, layers: int, N: int) -> np.ndarray:
    U = np.eye(N, dtype=np.complex128)
    for alpha, theta in unpack_params(x, layers, N):
        # Convention: one layer = D(alpha) SNAP(theta)
        U = displacement_operator(a, alpha) @ snap_operator(theta) @ U
    return U


def training_loss(
    x: np.ndarray,
    a: np.ndarray,
    U_K_half: np.ndarray,
    U_sq_half: np.ndarray,
    train_states: list[np.ndarray],
    train_targets: list[np.ndarray],
    layers: int,
    N: int,
    num_steps: int,
    reg_alpha: float = 1e-5,
) -> float:
    """
    Multi-step residual trajectory training objective.

    This trains the SNAP+D block inside the repeated symmetric hybrid step:

        [U_K_half U_sq_half U_SNAPD U_sq_half U_K_half]^num_steps |psi>

    to match the full Duffing trajectory:

        exp(-i t H_duff) |psi>

    at all intermediate time steps.
    """
    U_snapd = snapd_unitary(a, x, layers, N)
    U_step = U_K_half @ U_sq_half @ U_snapd @ U_sq_half @ U_K_half

    fids = []

    for psi0, target_traj in zip(train_states, train_targets):
        psi = psi0.copy()

        for k in range(1, num_steps + 1):
            psi = U_step @ psi
            psi = psi / la.norm(psi)

            target = target_traj[k]
            fids.append(fidelity(target, psi))

        # Extra weight on final-time fidelity.
        fids.append(fidelity(target_traj[-1], psi))

    alpha_penalty = 0.0
    for alpha, _theta in unpack_params(x, layers, N):
        alpha_penalty += abs(alpha)**2

    return float(1.0 - np.mean(fids) + reg_alpha * alpha_penalty)

def optimize_snapd(
    a: np.ndarray,
    U_K_half: np.ndarray,
    U_sq_half: np.ndarray,
    train_states: list[np.ndarray],
    train_targets: list[np.ndarray],
    layers: int,
    N: int,
    num_steps: int,
    seed: int,
    n_restarts: int,
    maxiter: int,
    init_scale_alpha: float = 0.05,
    init_scale_theta: float = 0.05,
):
    rng = np.random.default_rng(seed)
    dim = layers * (N + 2)

    best = None
    history = []

    initial_points = [np.zeros(dim)]

    for _ in range(n_restarts):
        x0 = np.zeros(dim)
        for ell in range(layers):
            start = ell * (N + 2)
            x0[start:start+2] = init_scale_alpha * rng.normal(size=2)
            x0[start+2:start+2+N] = init_scale_theta * rng.normal(size=N)
        initial_points.append(x0)

    for ridx, x0 in enumerate(initial_points):
        res = opt.minimize(
            training_loss,
            x0,
            args=(
                a,
                U_K_half,
                U_sq_half,
                train_states,
                train_targets,
                layers,
                N,
                num_steps,
            ),
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            },
        )

        history.append({
            "restart": ridx,
            "loss": float(res.fun),
            "nit": int(res.nit),
            "success": bool(res.success),
            "message": str(res.message),
        })

        if best is None or res.fun < best.fun:
            best = res

    return best.x, float(best.fun), history

# -----------------------------
# Evolution and diagnostics
# -----------------------------

def evolve_times(U_step: np.ndarray, psi0: np.ndarray, num_steps: int):
    states = [psi0]
    psi = psi0.copy()
    for _ in range(num_steps):
        psi = U_step @ psi
        psi = psi / la.norm(psi)
        states.append(psi)
    return np.array(states)


def exact_times(H: np.ndarray, psi0: np.ndarray, times: np.ndarray):
    states = []
    evals, evecs = la.eigh(H)
    coeff = evecs.conj().T @ psi0
    for t in times:
        psi = evecs @ (np.exp(-1j * evals * t) * coeff)
        states.append(psi / la.norm(psi))
    return np.array(states)


def expectations(states: np.ndarray, op: np.ndarray):
    return np.array([np.vdot(s, op @ s) for s in states])


def plot_fidelity(times, fk, fksq, fsnapd, outpath):
    plt.figure(figsize=(7, 4.5))
    plt.plot(times, fk, label="Native Kerr")
    plt.plot(times, fksq, label="Kerr + squeezing")
    plt.plot(times, fsnapd, label="Kerr + squeezing + SNAP+D")
    plt.xlabel("Time")
    plt.ylabel("Fidelity with full Duffing")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()


def plot_observable(times, exact, kerr, snapd, ylabel, outpath):
    plt.figure(figsize=(7, 4.5))
    plt.plot(times, exact, label="Full Duffing")
    plt.plot(times, kerr, "--", label="Native Kerr")
    plt.plot(times, snapd, "-.", label="SNAP+D corrected")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--nt", type=int, default=50)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--train-xis", type=str, default="0.0,0.5,1.0,1.5")
    parser.add_argument("--test-xis", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-restarts", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=400)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_xis = [float(x) for x in args.train_xis.split(",") if x.strip()]
    test_xis = [float(x) for x in args.test_xis.split(",") if x.strip()]

    ops = build_duffing_operators(args.N, args.omega0, args.beta)

    dt = args.T / args.nt
    times = np.linspace(0.0, args.T, args.nt + 1)


    # Full-step components
    U_K_step = la.expm(-1j * dt * ops.H_kerr)
    U_sq_step = la.expm(-1j * dt * ops.lam * ops.V2)

    # Half-step components for symmetric Trotterization
    U_K_half = la.expm(-1j * 0.5 * dt * ops.H_kerr)
    U_sq_half = la.expm(-1j * 0.5 * dt * ops.lam * ops.V2)

    train_states = [coherent_state(args.N, xi) for xi in train_xis]

    # Exact Duffing trajectories for trajectory-level residual training.
    train_targets = [
        exact_times(ops.H_duff, psi, times)
        for psi in train_states
    ]

    xopt, best_loss, history = optimize_snapd(
        ops.a,
        U_K_half,
        U_sq_half,
        train_states,
        train_targets,
        args.layers,
        args.N,
        args.nt,
        seed=args.seed,
        n_restarts=args.n_restarts,
        maxiter=args.maxiter,
    )

    U_snapd = snapd_unitary(ops.a, xopt, args.layers, args.N)

    # Symmetric Kerr+squeezing step
    U_ksq_step = U_K_half @ U_sq_step @ U_K_half

    # Multi-step residual-trained symmetric hybrid step
    U_ksq_snapd = U_K_half @ U_sq_half @ U_snapd @ U_sq_half @ U_K_half

    rows = []
    all_data = {}

    for xi in test_xis:
        psi0 = coherent_state(args.N, xi)

        states_D = exact_times(ops.H_duff, psi0, times)
        states_K = exact_times(ops.H_kerr, psi0, times)
        states_KSQ = evolve_times(U_ksq_step, psi0, args.nt)
        states_ALG = evolve_times(U_ksq_snapd, psi0, args.nt)

        F_K = np.array([fidelity(states_D[i], states_K[i]) for i in range(len(times))])
        F_KSQ = np.array([fidelity(states_D[i], states_KSQ[i]) for i in range(len(times))])
        F_ALG = np.array([fidelity(states_D[i], states_ALG[i]) for i in range(len(times))])

        q_D = np.real(expectations(states_D, ops.q))
        q_K = np.real(expectations(states_K, ops.q))
        q_ALG = np.real(expectations(states_ALG, ops.q))

        n_D = np.real(expectations(states_D, ops.n))
        n_K = np.real(expectations(states_K, ops.n))
        n_ALG = np.real(expectations(states_ALG, ops.n))

        label = f"xi_{xi:.3f}".replace(".", "p")
        all_data[f"{label}_F_K"] = F_K
        all_data[f"{label}_F_KSQ"] = F_KSQ
        all_data[f"{label}_F_SNAPD"] = F_ALG
        all_data[f"{label}_q_D"] = q_D
        all_data[f"{label}_q_K"] = q_K
        all_data[f"{label}_q_SNAPD"] = q_ALG
        all_data[f"{label}_n_D"] = n_D
        all_data[f"{label}_n_K"] = n_K
        all_data[f"{label}_n_SNAPD"] = n_ALG

        plot_fidelity(times, F_K, F_KSQ, F_ALG, outdir / f"fidelity_beta{args.beta}_xi{xi}.png")
        plot_observable(times, q_D, q_K, q_ALG, r"$\langle q(t)\rangle$", outdir / f"q_beta{args.beta}_xi{xi}.png")
        plot_observable(times, n_D, n_K, n_ALG, r"$\langle n(t)\rangle$", outdir / f"n_beta{args.beta}_xi{xi}.png")

        rows.append({
            "N": args.N,
            "omega0": args.omega0,
            "beta": args.beta,
            "T": args.T,
            "nt": args.nt,
            "dt": dt,
            "layers": args.layers,
            "xi": xi,
            "mean_n_initial": xi**2,
            "snapd_training_loss": best_loss,
            "final_F_K": F_K[-1],
            "final_F_KSQ": F_KSQ[-1],
            "final_F_SNAPD": F_ALG[-1],
            "max_infidelity_K": float(np.max(1.0 - F_K)),
            "max_infidelity_KSQ": float(np.max(1.0 - F_KSQ)),
            "max_infidelity_SNAPD": float(np.max(1.0 - F_ALG)),
            "improvement_maxinfid_K_over_SNAPD": float(np.max(1.0 - F_K) - np.max(1.0 - F_ALG)),
        })

    pd.DataFrame(rows).to_csv(outdir / "summary.csv", index=False)
    pd.DataFrame(history).to_csv(outdir / "optimizer_history.csv", index=False)

    np.savez_compressed(
        outdir / "fidelity_times.npz",
        times=times,
        **all_data
    )
    np.savez_compressed(
        outdir / "snapd_params.npz",
        xopt=xopt,
        beta=args.beta,
        N=args.N,
        layers=args.layers,
        dt=dt,
        best_loss=best_loss,
    )

    with open(outdir / "run_metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("DONE")
    print(f"outdir = {outdir}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
