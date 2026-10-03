#!/usr/bin/env python3
"""Shared CLI, validation, metadata, and observer helpers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import platform
import socket
import sys

import numpy as np

if __package__:
    from . import sglib as sg
else:
    import sglib as sg


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--E-J", dest="E_J", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--n-site", type=int, default=2)
    parser.add_argument("--n-cut", type=int, default=20)
    parser.add_argument(
        "--omega",
        type=float,
        default=-1.0,
        help="local oscillator scale; <=0 uses sqrt(E_J + 2*kappa)",
    )
    parser.add_argument(
        "--init",
        choices=["vacuum", "coherent_single", "coherent_uniform", "kink"],
        default="coherent_single",
    )
    parser.add_argument("--phi0", type=float, default=3.0)
    parser.add_argument("--pi0", type=float, default=0.0)
    parser.add_argument(
        "--omega-time",
        type=float,
        default=None,
        help="frequency for tau=Omega_time*t, independent of the basis",
    )
    parser.add_argument("--site", type=int, default=0)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--n-obs", type=int, default=201)
    parser.add_argument("--pbc", action="store_true")
    parser.add_argument("--use-offsets", action="store_true")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--tag", default="")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=-1.0,
        help="abort if conservative split-operator estimate exceeds this; "
        "<=0 reads SG_MEMORY_LIMIT_GB or only reports",
    )
    return parser


def _validate(args) -> None:
    if args.E_J < 0:
        raise ValueError("E_J must be nonnegative")
    if args.kappa < 0:
        raise ValueError("kappa must be nonnegative")
    if args.n_site < 1:
        raise ValueError("n_site must be positive")
    if args.n_cut < 4:
        raise ValueError("n_cut must be at least 4")
    if args.T <= 0 or args.dt <= 0:
        raise ValueError("T and dt must be positive")
    if args.n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if not 0 <= args.site < args.n_site:
        raise ValueError("site is outside the lattice")
    if args.pbc and args.n_site < 3:
        raise ValueError("periodic boundaries require at least three sites")
    if args.init == "kink" and (args.E_J <= 0 or args.kappa <= 0):
        raise ValueError("kink initialization requires E_J > 0 and kappa > 0")


def setup(args, default_tag: str):
    _validate(args)
    omega = (
        float(args.omega)
        if args.omega > 0
        else float(np.sqrt(max(args.E_J + 2.0 * args.kappa, 1e-12)))
    )
    omega_time = (
        float(np.sqrt(args.E_J + 2 * args.kappa))
        if args.omega_time is None
        else float(args.omega_time)
    )
    if not np.isfinite(omega_time) or omega_time < 0:
        raise ValueError("time-scale frequency must be finite and nonnegative")
    basis = sg.build_mode_basis(args.n_cut, omega)
    predicted = sg.estimate_split_memory_gb(args.n_cut, args.n_site)
    limit = float(args.max_memory_gb)
    if limit <= 0:
        try:
            limit = float(os.environ.get("SG_MEMORY_LIMIT_GB", "-1"))
        except ValueError as exc:
            raise ValueError("SG_MEMORY_LIMIT_GB must be numeric") from exc
    print(
        f"[setup] sites={args.n_site} cutoff={args.n_cut} "
        f"dimension={args.n_cut**args.n_site:,} omega={omega:.6g}",
        flush=True,
    )
    print(
        f"[setup] conservative split-memory estimate ~{predicted:.2f} GiB",
        flush=True,
    )
    if limit > 0 and predicted > limit:
        raise MemoryError(
            f"estimated working memory {predicted:.2f} GiB exceeds "
            f"configured limit {limit:.2f} GiB"
        )

    tag = args.tag.strip() or default_tag
    params = {
        "E_J": float(args.E_J),
        "kappa": float(args.kappa),
        "n_site": int(args.n_site),
        "n_cut": int(args.n_cut),
        "omega": omega,
        "omega_b": omega,
        "Omega_time": omega_time,
        "lambda2": float(getattr(args, "lambda2", 0.0)),
        "operator_convention": "finite spectral: pi_N squared and cos(m phi_N)",
        "T_requested": float(args.T),
        "dt_requested": float(args.dt),
        "n_obs_requested": int(args.n_obs),
        "pbc": bool(args.pbc),
        "boundary_condition": "periodic" if args.pbc else "open",
        "initial_condition_requested": str(args.init),
        "phi0_requested": float(args.phi0),
        "pi0_requested": float(args.pi0),
        "initial_site": int(args.site),
        "use_offsets_requested": bool(args.use_offsets),
        "basis_frequency_convention": (
            "user supplied"
            if args.omega > 0
            else "bulk harmonic scale sqrt(E_J+2*kappa)"
        ),
        "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "lsf_job_index": os.environ.get("LSB_JOBINDEX"),
        "estimated_split_memory_gib": predicted,
    }
    semi = {
        "delta_phi_sq": basis.delta_phi_sq,
        "E_J_eff": float(args.E_J * np.exp(-0.5 * basis.delta_phi_sq)),
    }
    params.update(semi)
    return {
        "basis": basis,
        "w": omega,
        "Omega_time": omega_time,
        "phi_op": basis.phi,
        "pi_op": basis.pi,
        "num_op": basis.num,
        "semi": semi,
        "params": params,
        "tag": tag,
    }


def steps_and_stride(T: float, dt_requested: float, n_obs: int):
    n_steps = max(1, int(round(T / dt_requested)))
    dt_actual = T / n_steps
    if not np.isclose(dt_actual, dt_requested, rtol=1e-12, atol=1e-15):
        print(
            f"[setup] adjusted dt from {dt_requested:.12g} to {dt_actual:.12g} "
            "so the final time is exact",
            flush=True,
        )
    # Existing drivers expect only (n_steps, stride); setting the caller's dt
    # is not possible here, so require divisibility rather than silently using
    # a mismatched final time.
    if not np.isclose(n_steps * dt_requested, T, rtol=1e-11, atol=1e-13):
        raise ValueError("T/dt must be integral for trajectory runs")
    stride = max(1, int(round(n_steps / (n_obs - 1))))
    return n_steps, stride


def new_store():
    return {"t": [], "phi": [], "pi": [], "n": [], "leak": []}


def make_observer(phi_op, pi_op, num_op, store, offsets):
    offsets = np.asarray(offsets, dtype=float)
    n_site = offsets.size
    n_cut = phi_op.shape[0]

    def observer(t, psi):
        norm = np.linalg.norm(psi.reshape(-1))
        if norm == 0:
            raise FloatingPointError("observer received a zero state")
        state = psi / norm
        phi = np.array([sg.expect_mode(state, phi_op, j).real for j in range(n_site)])
        pi = np.array([sg.expect_mode(state, pi_op, j).real for j in range(n_site)])
        num = np.array([sg.expect_mode(state, num_op, j).real for j in range(n_site)])
        top = np.zeros((n_cut, n_cut), dtype=float)
        top[-1, -1] = 1.0
        leakage = max(float(sg.expect_mode(state, top, j).real) for j in range(n_site))
        store["t"].append(float(t))
        store["phi"].append(phi + offsets)
        store["pi"].append(pi)
        store["n"].append(num)
        store["leak"].append(leakage)

    return observer


def stack(store):
    return {key: np.asarray(value) for key, value in store.items()}
