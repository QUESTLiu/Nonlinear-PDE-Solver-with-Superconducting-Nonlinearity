#!/usr/bin/env python3
"""
Section 4.4 -- Padded-space validation of the compiled hybrid Duffing correction.

This script performs the additional numerical study needed to compare a
compiled SNAP-plus-displacement correction directly with a converged
N_ref=80 Duffing reference.

Physics conventions
-------------------
    hbar = m = 1
    lambda = beta / (16 omega0**2)

    H_D = omega0 (n + 1/2) + lambda (a + a^dagger)^4
    H_D = H_K + lambda V_2 + lambda V_4

    V_2 = 6 (a^2 + a^dagger^2)
    V_4 = a^dagger^4 + a^4
          + 4 (a^dagger^3 a + a^dagger a^3)

The compiled block approximates exp(-i lambda V_4 dt) with

    U_SD = D(alpha_{L-1}) SNAP(theta_{L-1}) ...
           D(alpha_0)     SNAP(theta_0).

The same block is reused at every time step.  It is trained on low-energy
states encountered by exact split trajectories for the requested training
coherent amplitudes.  The full hybrid step is

    U_hyb(dt) =
        exp(-i H_K dt/2)
        exp(-i lambda V_2 dt/2)
        U_SD
        exp(-i lambda V_2 dt/2)
        exp(-i H_K dt/2).

The numerical simulation is carried out in a padded Hilbert space of
dimension D_pad.  SNAP phases are active only on levels 0,...,d_active-1;
displacements act in all D_pad levels.  The corrected padded-space
trajectory is compared with the independently evolved N_ref=80 Duffing
reference embedded in the padded space.

Outputs
-------
    compilation_restarts.csv
    compilation_training_states.csv
    hybrid_time_series.csv
    hybrid_summary.csv
    metadata.json
    best_compiler_parameters.npz
    validation_verdict.md
    padded_hybrid_validation.pdf/png   (when --plot is used)

Dependencies
------------
    numpy, scipy
    matplotlib is optional and is used only with --plot.

The compiler uses an analytic reverse-mode gradient.  Displacements are
applied with scipy.sparse.linalg.expm_multiply during optimization, so the
optimizer never differentiates a dense matrix exponential.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, eye as sparse_eye
from scipy.sparse.linalg import expm_multiply


Array = np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compile the quartic number-changing Duffing correction in a "
            "padded Fock space and compare it with an N=80 reference."
        )
    )
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.10)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--reference-dim", type=int, default=80)
    p.add_argument("--pad-dim", type=int, default=96)
    p.add_argument("--active-dim", type=int, default=48)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument(
        "--train-xi",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.25],
        help="Coherent amplitudes used to build compilation snapshots.",
    )
    p.add_argument(
        "--test-xi",
        type=float,
        nargs="+",
        default=[1.75, 2.00],
        help="Coherent amplitudes used only for final validation.",
    )
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=1200)
    p.add_argument("--gtol", type=float, default=1.0e-9)
    p.add_argument("--ftol", type=float, default=1.0e-13)
    p.add_argument(
        "--init-scale",
        type=float,
        default=0.025,
        help="Standard deviation of random initial parameters.",
    )
    p.add_argument(
        "--alpha-max",
        type=float,
        default=1.5,
        help="Bound on the real and imaginary parts of each displacement.",
    )
    p.add_argument(
        "--alpha-regularization",
        type=float,
        default=1.0e-8,
        help="Small regularizer that discourages unnecessarily large displacements.",
    )
    p.add_argument("--seed", type=int, default=240731)
    p.add_argument(
        "--edge-levels",
        type=int,
        default=8,
        help="Number of highest padded levels used for the boundary diagnostic.",
    )
    p.add_argument(
        "--max-reference-infidelity",
        type=float,
        default=1.0e-6,
        help="Pass threshold for exact D_pad versus the N_ref reference.",
    )
    p.add_argument(
        "--max-edge-population",
        type=float,
        default=1.0e-10,
        help="Pass threshold for population in the highest padded levels.",
    )
    p.add_argument(
        "--max-split-infidelity",
        type=float,
        default=5.0e-4,
        help="Pass threshold for exact-V4 split versus N_ref Duffing.",
    )
    p.add_argument(
        "--max-hybrid-infidelity",
        type=float,
        default=2.0e-2,
        help="Requested maximum compiled-hybrid infidelity.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/duffing_padded_hybrid"),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Use an existing best_compiler_parameters.npz as restart 0.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Create PDF and PNG figures if matplotlib is installed.",
    )
    p.add_argument(
        "--gradient-check",
        action="store_true",
        help="Check selected analytic-gradient entries before optimization.",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.omega0 <= 0.0:
        raise ValueError("--omega0 must be positive")
    if args.beta < 0.0:
        raise ValueError("--beta must be nonnegative")
    if args.T <= 0.0 or args.steps < 1:
        raise ValueError("--T must be positive and --steps must be >= 1")
    if args.reference_dim < 8:
        raise ValueError("--reference-dim must be at least 8")
    if args.pad_dim < args.reference_dim:
        raise ValueError("--pad-dim must be >= --reference-dim")
    if not (4 <= args.active_dim < args.pad_dim):
        raise ValueError("--active-dim must satisfy 4 <= active-dim < pad-dim")
    if args.layers < 1:
        raise ValueError("--layers must be >= 1")
    if args.restarts < 1 or args.maxiter < 1:
        raise ValueError("--restarts and --maxiter must be >= 1")
    if args.alpha_max <= 0.0:
        raise ValueError("--alpha-max must be positive")
    if not args.train_xi or not args.test_xi:
        raise ValueError("--train-xi and --test-xi cannot be empty")
    if min(args.train_xi + args.test_xi) < 0.0:
        raise ValueError("This script expects nonnegative real coherent amplitudes")
    if not (1 <= args.edge_levels < args.pad_dim):
        raise ValueError("--edge-levels must be between 1 and pad-dim-1")
    thresholds = [
        args.max_reference_infidelity,
        args.max_edge_population,
        args.max_split_infidelity,
        args.max_hybrid_infidelity,
    ]
    if any(value <= 0.0 for value in thresholds):
        raise ValueError("All validation thresholds must be positive")


def ladder(dim: int) -> Array:
    a = np.zeros((dim, dim), dtype=np.complex128)
    a[np.arange(dim - 1), np.arange(1, dim)] = np.sqrt(
        np.arange(1, dim, dtype=float)
    )
    return a


def coherent_state(dim: int, xi: float) -> Tuple[Array, float]:
    """Normalized projected coherent state and exact discarded probability."""
    mu = float(xi * xi)
    coeff = np.empty(dim, dtype=np.complex128)
    coeff[0] = math.exp(-0.5 * mu)
    for n in range(1, dim):
        coeff[n] = coeff[n - 1] * xi / math.sqrt(n)
    retained = float(np.vdot(coeff, coeff).real)
    discarded = max(0.0, 1.0 - retained)
    coeff /= math.sqrt(retained)
    return coeff, discarded


def hermitian_exponential(h: Array, coefficient: complex) -> Array:
    """Return exp(coefficient * h) for Hermitian h."""
    vals, vecs = np.linalg.eigh(h)
    return (vecs * np.exp(coefficient * vals)[None, :]) @ vecs.conj().T


def evolve_hermitian(h: Array, psi0: Array, times: Array) -> Array:
    vals, vecs = np.linalg.eigh(h)
    amplitudes = vecs.conj().T @ psi0
    return vecs @ (
        amplitudes[:, None] * np.exp(-1j * vals[:, None] * times[None, :])
    )


def hard_sandwiched_operators(
    dim: int, omega0: float, beta: float, degree_pad: int = 4
) -> Dict[str, Array | float]:
    """
    Construct Pi_dim A Pi_dim in a dim+4 space.

    Forming (a_dim+a_dim^dagger)^4 directly would introduce the wrong
    upper-boundary algebra.  The extension followed by slicing implements
    the intended hard sandwich.
    """
    ext = dim + degree_pad
    a = ladder(ext)
    ad = a.conj().T
    n = ad @ a
    ident = np.eye(ext, dtype=np.complex128)
    x = a + ad
    a2 = a @ a
    ad2 = ad @ ad
    a3 = a2 @ a
    ad3 = ad2 @ ad
    a4 = a2 @ a2
    ad4 = ad2 @ ad2

    lam = beta / (16.0 * omega0**2)
    k_eff = 3.0 * beta / (8.0 * omega0**2)
    omega_eff = omega0 + k_eff

    v2 = 6.0 * (a2 + ad2)
    v4 = ad4 + a4 + 4.0 * (ad3 @ a + ad @ a3)
    vcr = v2 + v4
    hd = omega0 * (n + 0.5 * ident) + lam * (x @ x @ x @ x)
    hk = (
        omega_eff * n
        + k_eff * (n @ n)
        + (0.5 * omega0 + 3.0 * lam) * ident
    )

    sl = np.s_[:dim, :dim]
    out: Dict[str, Array | float] = {
        "a": a[sl],
        "ad": ad[sl],
        "n": n[sl],
        "H_D": hd[sl],
        "H_K": hk[sl],
        "V_2": v2[sl],
        "V_4": v4[sl],
        "V_cr": vcr[sl],
        "lambda": lam,
        "K_eff": k_eff,
        "omega_eff": omega_eff,
    }
    residual = np.linalg.norm(
        np.asarray(out["H_D"])
        - np.asarray(out["H_K"])
        - lam * (np.asarray(out["V_2"]) + np.asarray(out["V_4"])),
        ord=2,
    )
    out["split_residual_norm"] = float(residual)
    return out


@dataclass
class FixedGates:
    k_half: Array
    v2_half: Array
    v2_full: Array
    u4_exact: Array
    exact_split_step: Array
    k_v2_step: Array


def build_fixed_gates(
    ops: Dict[str, Array | float], dt: float
) -> FixedGates:
    hk = np.asarray(ops["H_K"])
    v2 = np.asarray(ops["V_2"])
    v4 = np.asarray(ops["V_4"])
    lam = float(ops["lambda"])

    # H_K is diagonal in the number basis, but the general diagonal
    # expression below keeps the code transparent.
    k_half = np.diag(np.exp(-0.5j * dt * np.real(np.diag(hk))))
    v2_half = hermitian_exponential(v2, -0.5j * lam * dt)
    v2_full = hermitian_exponential(v2, -1.0j * lam * dt)
    u4_exact = hermitian_exponential(v4, -1.0j * lam * dt)

    exact_split = k_half @ v2_half @ u4_exact @ v2_half @ k_half
    k_v2_step = k_half @ v2_full @ k_half
    return FixedGates(
        k_half=k_half,
        v2_half=v2_half,
        v2_full=v2_full,
        u4_exact=u4_exact,
        exact_split_step=exact_split,
        k_v2_step=k_v2_step,
    )


def build_training_snapshots(
    dim: int,
    xis: Sequence[float],
    steps: int,
    gates: FixedGates,
) -> Tuple[Array, Array, List[dict]]:
    """
    Return states immediately before and after the exact V4 block.

    Snapshots are taken from exact symmetric-split trajectories, not from
    the N_ref Duffing trajectory.  This gives the compiler the states on
    which the repeated block will actually act.
    """
    before: List[Array] = []
    after: List[Array] = []
    rows: List[dict] = []
    for xi in xis:
        psi, discarded = coherent_state(dim, xi)
        for step in range(1, steps + 1):
            x = gates.v2_half @ (gates.k_half @ psi)
            y = gates.u4_exact @ x
            before.append(x)
            after.append(y)
            rows.append(
                {
                    "xi": xi,
                    "step": step,
                    "initial_discarded_probability": discarded,
                    "input_norm": float(np.vdot(x, x).real),
                    "target_norm": float(np.vdot(y, y).real),
                }
            )
            psi = gates.k_half @ (gates.v2_half @ y)
    return np.column_stack(before), np.column_stack(after), rows


def unpack_parameters(
    vector: Array, layers: int, active_dim: int
) -> Tuple[Array, Array]:
    n_theta = layers * active_dim
    theta = vector[:n_theta].reshape(layers, active_dim)
    alpha_xy = vector[n_theta:].reshape(layers, 2)
    return theta, alpha_xy


def pack_parameters(theta: Array, alpha_xy: Array) -> Array:
    return np.concatenate([theta.ravel(), alpha_xy.ravel()])


class SnapDisplacementCompiler:
    """State-subspace compiler with an analytic reverse-mode gradient."""

    def __init__(
        self,
        dim: int,
        active_dim: int,
        layers: int,
        inputs: Array,
        targets: Array,
        alpha_regularization: float,
    ) -> None:
        self.dim = dim
        self.active_dim = active_dim
        self.layers = layers
        self.inputs = np.asarray(inputs, dtype=np.complex128)
        self.targets = np.asarray(targets, dtype=np.complex128)
        self.n_states = self.inputs.shape[1]
        self.alpha_regularization = float(alpha_regularization)

        a_dense = ladder(dim)
        ad_dense = a_dense.conj().T
        self.a = csc_matrix(a_dense)
        self.ad = csc_matrix(ad_dense)
        self.ident = sparse_eye(dim, dtype=np.complex128, format="csc")
        self.last_metrics: Dict[str, float] = {}
        self.calls = 0

    def _forward(
        self, theta: Array, alpha_xy: Array
    ) -> Tuple[Array, List[Tuple[Array, Array, Array, Array, csc_matrix]]]:
        x = self.inputs
        cache: List[Tuple[Array, Array, Array, Array, csc_matrix]] = []
        for layer in range(self.layers):
            snap = np.ones(self.dim, dtype=np.complex128)
            snap[: self.active_dim] = np.exp(1j * theta[layer])
            after_snap = snap[:, None] * x
            xr, yi = alpha_xy[layer]
            alpha = complex(xr, yi)
            generator = alpha * self.ad - np.conj(alpha) * self.a
            after_disp = expm_multiply(generator, after_snap, traceA=0.0)
            cache.append((x, after_snap, after_disp, snap, generator))
            x = after_disp
        return x, cache

    def value_and_grad(self, vector: Array) -> Tuple[float, Array]:
        theta, alpha_xy = unpack_parameters(
            vector, self.layers, self.active_dim
        )
        output, cache = self._forward(theta, alpha_xy)

        overlaps = np.sum(np.conj(self.targets) * output, axis=0)
        per_state_fidelity = np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)
        state_loss = 1.0 - float(np.mean(per_state_fidelity))
        reg = self.alpha_regularization * float(np.mean(alpha_xy**2))
        objective = state_loss + reg

        # For real parameters, df = 2 Re Tr(G^dagger dX).
        adjoint = -(
            self.targets * overlaps[None, :] / float(self.n_states)
        )
        grad_theta = np.zeros_like(theta)
        grad_alpha = np.zeros_like(alpha_xy)

        for layer in range(self.layers - 1, -1, -1):
            before, _, after_disp, snap, generator = cache[layer]
            xr, yi = alpha_xy[layer]

            # Exact Weyl derivatives away from the padded boundary:
            # dD/dx = (a^dagger-a+i y) D
            # dD/dy = (i(a^dagger+a)-i x) D.
            dx_action = (
                (self.ad - self.a + 1j * yi * self.ident) @ after_disp
            )
            dy_action = (
                (1j * (self.ad + self.a) - 1j * xr * self.ident)
                @ after_disp
            )
            grad_alpha[layer, 0] = 2.0 * float(
                np.real(np.vdot(adjoint, dx_action))
            )
            grad_alpha[layer, 1] = 2.0 * float(
                np.real(np.vdot(adjoint, dy_action))
            )

            after_displacement_adjoint = expm_multiply(
                -generator, adjoint, traceA=0.0
            )
            theta_action = (
                1j
                * snap[: self.active_dim, None]
                * before[: self.active_dim]
            )
            grad_theta[layer] = 2.0 * np.real(
                np.sum(
                    np.conj(
                        after_displacement_adjoint[: self.active_dim]
                    )
                    * theta_action,
                    axis=1,
                )
            )
            adjoint = np.conj(snap)[:, None] * after_displacement_adjoint

        if self.alpha_regularization:
            grad_alpha += (
                2.0
                * self.alpha_regularization
                * alpha_xy
                / float(alpha_xy.size)
            )

        coherent_trace = np.sum(overlaps)
        coherent_trace_fidelity = float(
            np.abs(coherent_trace) ** 2 / self.n_states**2
        )
        pop_above_active = np.sum(
            np.abs(output[self.active_dim :, :]) ** 2, axis=0
        )
        self.last_metrics = {
            "objective": float(objective),
            "mean_training_infidelity": float(1.0 - np.mean(per_state_fidelity)),
            "max_training_infidelity": float(
                1.0 - np.min(per_state_fidelity)
            ),
            "coherent_trace_fidelity": coherent_trace_fidelity,
            "mean_population_above_active_after_block": float(
                np.mean(pop_above_active)
            ),
            "max_population_above_active_after_block": float(
                np.max(pop_above_active)
            ),
            "max_abs_alpha": float(np.max(np.hypot(
                alpha_xy[:, 0], alpha_xy[:, 1]
            ))),
        }
        self.calls += 1
        return float(objective), pack_parameters(grad_theta, grad_alpha)

    def metrics(self, vector: Array, edge_levels: int) -> Dict[str, float]:
        theta, alpha_xy = unpack_parameters(
            vector, self.layers, self.active_dim
        )
        output, _ = self._forward(theta, alpha_xy)
        overlaps = np.sum(np.conj(self.targets) * output, axis=0)
        state_fid = np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)
        coherent_trace_fidelity = float(
            np.abs(np.sum(overlaps)) ** 2 / self.n_states**2
        )
        return {
            "mean_training_fidelity": float(np.mean(state_fid)),
            "min_training_fidelity": float(np.min(state_fid)),
            "max_training_infidelity": float(1.0 - np.min(state_fid)),
            "coherent_trace_fidelity": coherent_trace_fidelity,
            "mean_population_above_active_after_block": float(
                np.mean(np.sum(
                    np.abs(output[self.active_dim :, :]) ** 2, axis=0
                ))
            ),
            "max_population_above_active_after_block": float(
                np.max(np.sum(
                    np.abs(output[self.active_dim :, :]) ** 2, axis=0
                ))
            ),
            "max_edge_population_after_block": float(
                np.max(np.sum(np.abs(output[-edge_levels:, :]) ** 2, axis=0))
            ),
            "max_abs_alpha": float(
                np.max(np.hypot(alpha_xy[:, 0], alpha_xy[:, 1]))
            ),
        }


def finite_difference_gradient_check(
    compiler: SnapDisplacementCompiler,
    vector: Array,
    rng: np.random.Generator,
    count: int = 10,
    epsilon: float = 2.0e-6,
) -> None:
    f0, analytic = compiler.value_and_grad(vector)
    indices = rng.choice(vector.size, size=min(count, vector.size), replace=False)
    print(f"Gradient check at objective {f0:.12e}")
    worst = 0.0
    for index in indices:
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        fp, _ = compiler.value_and_grad(plus)
        fm, _ = compiler.value_and_grad(minus)
        numeric = (fp - fm) / (2.0 * epsilon)
        scale = max(1.0e-12, abs(numeric), abs(analytic[index]))
        relative = abs(numeric - analytic[index]) / scale
        worst = max(worst, relative)
        print(
            f"  parameter {index:4d}: analytic={analytic[index]: .6e}, "
            f"numeric={numeric: .6e}, relative={relative:.3e}"
        )
    if worst > 5.0e-4:
        raise RuntimeError(
            f"Gradient check failed: worst relative discrepancy {worst:.3e}"
        )


def write_csv(
    path: Path, fieldnames: Iterable[str], rows: Iterable[dict]
) -> None:
    fields = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def initial_parameters(
    rng: np.random.Generator,
    layers: int,
    active_dim: int,
    scale: float,
    identity: bool,
) -> Array:
    if identity:
        theta = np.zeros((layers, active_dim), dtype=float)
        alpha = np.zeros((layers, 2), dtype=float)
    else:
        theta = rng.normal(scale=scale, size=(layers, active_dim))
        alpha = rng.normal(scale=scale, size=(layers, 2))
        # Zero total displacement is a useful weak prior near the identity.
        alpha -= np.mean(alpha, axis=0, keepdims=True)
    return pack_parameters(theta, alpha)


def save_parameters(
    path: Path,
    vector: Array,
    layers: int,
    active_dim: int,
    metadata: Dict[str, float | int | str],
) -> None:
    theta, alpha_xy = unpack_parameters(vector, layers, active_dim)
    np.savez_compressed(
        path,
        theta=theta,
        alpha_real=alpha_xy[:, 0],
        alpha_imag=alpha_xy[:, 1],
        parameter_vector=vector,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def load_parameter_vector(
    path: Path, layers: int, active_dim: int
) -> Array:
    with np.load(path, allow_pickle=False) as data:
        if "parameter_vector" in data:
            vector = np.asarray(data["parameter_vector"], dtype=float)
        else:
            theta = np.asarray(data["theta"], dtype=float)
            alpha = np.column_stack(
                [
                    np.asarray(data["alpha_real"], dtype=float),
                    np.asarray(data["alpha_imag"], dtype=float),
                ]
            )
            vector = pack_parameters(theta, alpha)
    expected = layers * active_dim + 2 * layers
    if vector.size != expected:
        raise ValueError(
            f"{path} has {vector.size} parameters; expected {expected}"
        )
    return vector


def compile_block(
    args: argparse.Namespace,
    compiler: SnapDisplacementCompiler,
    output_dir: Path,
) -> Tuple[Array, List[dict]]:
    rng = np.random.default_rng(args.seed)
    n_theta = args.layers * args.active_dim
    bounds = [(-math.pi, math.pi)] * n_theta + [
        (-args.alpha_max, args.alpha_max)
    ] * (2 * args.layers)

    checkpoint = output_dir / "best_compiler_parameters.npz"
    resumed_vector: Array | None = None
    if args.resume and checkpoint.exists():
        resumed_vector = load_parameter_vector(
            checkpoint, args.layers, args.active_dim
        )
        print(f"Resuming restart 0 from {checkpoint}")

    best_vector: Array | None = None
    best_objective = math.inf
    restart_rows: List[dict] = []

    for restart in range(args.restarts):
        if restart == 0 and resumed_vector is not None:
            x0 = resumed_vector.copy()
            init_kind = "resume"
        else:
            x0 = initial_parameters(
                rng,
                args.layers,
                args.active_dim,
                args.init_scale,
                identity=(restart == 0 and resumed_vector is None),
            )
            init_kind = "identity" if restart == 0 else "random"

        if args.gradient_check and restart == 0:
            finite_difference_gradient_check(compiler, x0, rng)

        initial_objective, _ = compiler.value_and_grad(x0)
        start = time.perf_counter()
        iteration = 0
        callback_best = initial_objective

        def callback(intermediate: Array) -> None:
            nonlocal iteration, callback_best
            iteration += 1
            value, _ = compiler.value_and_grad(intermediate)
            if value < callback_best:
                callback_best = value
            if iteration == 1 or iteration % 25 == 0:
                print(
                    f"  restart={restart} iter={iteration:4d} "
                    f"objective={value:.6e} best={callback_best:.6e}",
                    flush=True,
                )

        print(
            f"Starting restart {restart + 1}/{args.restarts} "
            f"({init_kind}); objective={initial_objective:.6e}",
            flush=True,
        )
        result = minimize(
            compiler.value_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": args.maxiter,
                "gtol": args.gtol,
                "ftol": args.ftol,
                "maxls": 30,
                "maxcor": 20,
            },
        )
        elapsed = time.perf_counter() - start
        metrics = compiler.metrics(result.x, args.edge_levels)
        row = {
            "restart": restart,
            "initialization": init_kind,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "initial_objective": float(initial_objective),
            "final_objective": float(result.fun),
            "elapsed_seconds": float(elapsed),
            **metrics,
        }
        restart_rows.append(row)
        print(
            f"Finished restart {restart}: objective={result.fun:.6e}, "
            f"mean training F={metrics['mean_training_fidelity']:.10f}, "
            f"min training F={metrics['min_training_fidelity']:.10f}, "
            f"elapsed={elapsed:.1f} s",
            flush=True,
        )

        if float(result.fun) < best_objective:
            best_objective = float(result.fun)
            best_vector = np.asarray(result.x, dtype=float).copy()
            save_parameters(
                checkpoint,
                best_vector,
                args.layers,
                args.active_dim,
                {
                    "best_objective": best_objective,
                    "restart": restart,
                    "mean_training_fidelity": metrics[
                        "mean_training_fidelity"
                    ],
                    "min_training_fidelity": metrics[
                        "min_training_fidelity"
                    ],
                },
            )

        write_csv(
            output_dir / "compilation_restarts.csv",
            restart_rows[0].keys(),
            restart_rows,
        )

    if best_vector is None:
        raise RuntimeError("Compilation produced no parameter vector")
    return best_vector, restart_rows


def dense_ansatz_unitary(
    dim: int,
    active_dim: int,
    layers: int,
    vector: Array,
) -> Array:
    theta, alpha_xy = unpack_parameters(vector, layers, active_dim)
    a = ladder(dim)
    ad = a.conj().T
    unitary = np.eye(dim, dtype=np.complex128)
    for layer in range(layers):
        snap = np.ones(dim, dtype=np.complex128)
        snap[:active_dim] = np.exp(1j * theta[layer])
        alpha = complex(alpha_xy[layer, 0], alpha_xy[layer, 1])
        displacement = expm(alpha * ad - np.conj(alpha) * a)
        unitary = displacement @ (snap[:, None] * unitary)
    return unitary


def fidelity(a: Array, b: Array) -> float:
    value = float(np.abs(np.vdot(a, b)) ** 2)
    return min(1.0, max(0.0, value))


def expectation_n(psi: Array) -> float:
    occupations = np.arange(psi.size, dtype=float)
    return float(np.sum(occupations * np.abs(psi) ** 2).real)


def embed_state(psi: Array, target_dim: int) -> Array:
    if psi.size > target_dim:
        raise ValueError("Cannot embed a state into a smaller dimension")
    out = np.zeros(target_dim, dtype=np.complex128)
    out[: psi.size] = psi
    return out


def simulate_compiled_trajectory(
    psi0: Array,
    steps: int,
    gates: FixedGates,
    u_sd: Array,
    active_dim: int,
    edge_levels: int,
) -> Tuple[Array, Dict[str, float]]:
    states = [psi0.copy()]
    psi = psi0.copy()
    max_edge = float(np.sum(np.abs(psi[-edge_levels:]) ** 2))
    max_above_active = float(np.sum(np.abs(psi[active_dim:]) ** 2))
    max_norm_error = abs(float(np.vdot(psi, psi).real) - 1.0)

    for _ in range(steps):
        psi = gates.k_half @ psi
        psi = gates.v2_half @ psi
        psi = u_sd @ psi
        max_edge = max(
            max_edge, float(np.sum(np.abs(psi[-edge_levels:]) ** 2))
        )
        max_above_active = max(
            max_above_active, float(np.sum(np.abs(psi[active_dim:]) ** 2))
        )
        psi = gates.v2_half @ psi
        psi = gates.k_half @ psi
        max_edge = max(
            max_edge, float(np.sum(np.abs(psi[-edge_levels:]) ** 2))
        )
        max_above_active = max(
            max_above_active, float(np.sum(np.abs(psi[active_dim:]) ** 2))
        )
        max_norm_error = max(
            max_norm_error, abs(float(np.vdot(psi, psi).real) - 1.0)
        )
        states.append(psi.copy())

    return np.column_stack(states), {
        "max_edge_population_during_compiled_trajectory": max_edge,
        "max_population_above_active_during_compiled_trajectory": (
            max_above_active
        ),
        "max_norm_error_during_compiled_trajectory": max_norm_error,
    }


def apply_repeated_step(psi0: Array, step: Array, steps: int) -> Array:
    states = [psi0.copy()]
    psi = psi0.copy()
    for _ in range(steps):
        psi = step @ psi
        states.append(psi.copy())
    return np.column_stack(states)


def diagonal_kerr_trajectory(
    hk: Array, psi0: Array, times: Array
) -> Array:
    energies = np.real(np.diag(hk))
    return psi0[:, None] * np.exp(
        -1j * energies[:, None] * times[None, :]
    )


def max_infidelity(reference: Array, candidate: Array) -> float:
    overlaps = np.sum(np.conj(reference) * candidate, axis=0)
    return float(np.max(1.0 - np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)))


def evaluate(
    args: argparse.Namespace,
    ops_ref: Dict[str, Array | float],
    ops_pad: Dict[str, Array | float],
    gates_pad: FixedGates,
    best_vector: Array,
) -> Tuple[List[dict], List[dict]]:
    times = np.linspace(0.0, args.T, args.steps + 1)
    u_sd = dense_ansatz_unitary(
        args.pad_dim,
        args.active_dim,
        args.layers,
        best_vector,
    )
    compiled_step = (
        gates_pad.k_half
        @ gates_pad.v2_half
        @ u_sd
        @ gates_pad.v2_half
        @ gates_pad.k_half
    )

    time_rows: List[dict] = []
    summary_rows: List[dict] = []
    for xi in args.test_xi:
        psi_ref0, discarded_ref = coherent_state(args.reference_dim, xi)
        psi_pad0, discarded_pad = coherent_state(args.pad_dim, xi)

        reference_small = evolve_hermitian(
            np.asarray(ops_ref["H_D"]), psi_ref0, times
        )
        reference = np.column_stack(
            [
                embed_state(reference_small[:, index], args.pad_dim)
                for index in range(reference_small.shape[1])
            ]
        )
        exact_pad = evolve_hermitian(
            np.asarray(ops_pad["H_D"]), psi_pad0, times
        )
        native_kerr = diagonal_kerr_trajectory(
            np.asarray(ops_pad["H_K"]), psi_pad0, times
        )
        kerr_v2 = apply_repeated_step(
            psi_pad0, gates_pad.k_v2_step, args.steps
        )
        exact_split = apply_repeated_step(
            psi_pad0, gates_pad.exact_split_step, args.steps
        )
        compiled, boundary = simulate_compiled_trajectory(
            psi_pad0,
            args.steps,
            gates_pad,
            u_sd,
            args.active_dim,
            args.edge_levels,
        )

        for index, t in enumerate(times):
            ref = reference[:, index]
            row = {
                "xi": xi,
                "step": index,
                "time": float(t),
                "fidelity_native_kerr_vs_N80": fidelity(
                    ref, native_kerr[:, index]
                ),
                "fidelity_kerr_v2_vs_N80": fidelity(
                    ref, kerr_v2[:, index]
                ),
                "fidelity_exact_split_vs_N80": fidelity(
                    ref, exact_split[:, index]
                ),
                "fidelity_compiled_hybrid_vs_N80": fidelity(
                    ref, compiled[:, index]
                ),
                "fidelity_compiled_vs_exact_split": fidelity(
                    exact_split[:, index], compiled[:, index]
                ),
                "fidelity_exact_Dpad_vs_N80": fidelity(
                    ref, exact_pad[:, index]
                ),
                "mean_n_N80_reference": expectation_n(ref),
                "mean_n_compiled_hybrid": expectation_n(
                    compiled[:, index]
                ),
                "population_above_active_compiled": float(
                    np.sum(np.abs(compiled[args.active_dim :, index]) ** 2)
                ),
                "edge_population_compiled": float(
                    np.sum(
                        np.abs(compiled[-args.edge_levels :, index]) ** 2
                    )
                ),
                "norm_compiled": float(
                    np.vdot(compiled[:, index], compiled[:, index]).real
                ),
            }
            time_rows.append(row)

        summary = {
                "xi": xi,
                "mean_n_initial": xi * xi,
                "reference_dimension": args.reference_dim,
                "padded_dimension": args.pad_dim,
                "active_dimension": args.active_dim,
                "initial_discarded_probability_N80": discarded_ref,
                "initial_discarded_probability_Dpad": discarded_pad,
                "max_infidelity_native_kerr_vs_N80": max_infidelity(
                    reference, native_kerr
                ),
                "final_infidelity_native_kerr_vs_N80": (
                    1.0 - fidelity(reference[:, -1], native_kerr[:, -1])
                ),
                "max_infidelity_kerr_v2_vs_N80": max_infidelity(
                    reference, kerr_v2
                ),
                "final_infidelity_kerr_v2_vs_N80": (
                    1.0 - fidelity(reference[:, -1], kerr_v2[:, -1])
                ),
                "max_infidelity_exact_split_vs_N80": max_infidelity(
                    reference, exact_split
                ),
                "final_infidelity_exact_split_vs_N80": (
                    1.0 - fidelity(reference[:, -1], exact_split[:, -1])
                ),
                "max_infidelity_compiled_hybrid_vs_N80": max_infidelity(
                    reference, compiled
                ),
                "final_infidelity_compiled_hybrid_vs_N80": (
                    1.0 - fidelity(reference[:, -1], compiled[:, -1])
                ),
                "max_infidelity_compiled_vs_exact_split": max_infidelity(
                    exact_split, compiled
                ),
                "final_infidelity_compiled_vs_exact_split": (
                    1.0
                    - fidelity(exact_split[:, -1], compiled[:, -1])
                ),
                "max_infidelity_exact_Dpad_vs_N80": max_infidelity(
                    reference, exact_pad
                ),
                "final_infidelity_exact_Dpad_vs_N80": (
                    1.0 - fidelity(reference[:, -1], exact_pad[:, -1])
                ),
                "max_reference_mean_n": float(
                    max(expectation_n(reference[:, j]) for j in range(
                        reference.shape[1]
                    ))
                ),
                "max_compiled_mean_n": float(
                    max(expectation_n(compiled[:, j]) for j in range(
                        compiled.shape[1]
                    ))
                ),
                **boundary,
        }
        summary.update(
            {
                "padding_reference_pass": bool(
                    summary["max_infidelity_exact_Dpad_vs_N80"]
                    <= args.max_reference_infidelity
                ),
                "padding_boundary_pass": bool(
                    summary[
                        "max_edge_population_during_compiled_trajectory"
                    ]
                    <= args.max_edge_population
                ),
                "product_formula_pass": bool(
                    summary["max_infidelity_exact_split_vs_N80"]
                    <= args.max_split_infidelity
                ),
                "compiled_improves_native": bool(
                    summary["max_infidelity_compiled_hybrid_vs_N80"]
                    < summary["max_infidelity_native_kerr_vs_N80"]
                ),
                "compiled_accuracy_pass": bool(
                    summary["max_infidelity_compiled_hybrid_vs_N80"]
                    <= args.max_hybrid_infidelity
                ),
            }
        )
        summary["overall_validation_pass"] = bool(
            summary["padding_reference_pass"]
            and summary["padding_boundary_pass"]
            and summary["product_formula_pass"]
            and summary["compiled_improves_native"]
            and summary["compiled_accuracy_pass"]
        )
        summary_rows.append(summary)
    return time_rows, summary_rows


def create_plot(
    output_dir: Path,
    time_rows: Sequence[dict],
    test_xi: Sequence[float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--plot was requested, but matplotlib is not installed"
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(
        1, len(test_xi), figsize=(3.6 * len(test_xi), 3.05), sharey=True
    )
    if len(test_xi) == 1:
        axes = [axes]

    styles = [
        ("fidelity_native_kerr_vs_N80", "Native Kerr", "#777777", "--"),
        (
            "fidelity_kerr_v2_vs_N80",
            r"Kerr + analytic $V_2$",
            "#E69F00",
            "-.",
        ),
        (
            "fidelity_exact_split_vs_N80",
            r"Exact-$V_4$ split",
            "#009E73",
            ":",
        ),
        (
            "fidelity_compiled_hybrid_vs_N80",
            r"Compiled hybrid",
            "#0072B2",
            "-",
        ),
    ]
    for axis, xi in zip(axes, test_xi):
        rows = sorted(
            [row for row in time_rows if float(row["xi"]) == float(xi)],
            key=lambda row: float(row["time"]),
        )
        t = np.array([float(row["time"]) for row in rows])
        for key, label, color, linestyle in styles:
            values = np.array([float(row[key]) for row in rows])
            axis.plot(
                t,
                values,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                marker="o" if key == "fidelity_compiled_hybrid_vs_N80" else None,
                markersize=3.2,
            )
        axis.set_title(rf"$\xi={xi:g}$")
        axis.set_xlabel(r"$t$")
        axis.set_xlim(float(t[0]), float(t[-1]))
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel(r"Fidelity to $N=80$ Duffing reference")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "padded_hybrid_validation.pdf")
    fig.savefig(output_dir / "padded_hybrid_validation.png", dpi=400)
    plt.close(fig)


def verdict_text(
    args: argparse.Namespace,
    compiler_metrics: Dict[str, float],
    summary_rows: Sequence[dict],
) -> str:
    lines = [
        "# Padded hybrid Duffing validation",
        "",
        "## Configuration",
        "",
        f"- Reference Duffing dimension: N_ref = {args.reference_dim}",
        f"- Padded circuit dimension: D_pad = {args.pad_dim}",
        f"- Active SNAP dimension: d = {args.active_dim}",
        f"- SNAP+displacement layers: L = {args.layers}",
        f"- Symmetric time steps: r = {args.steps}",
        f"- dt = {args.T / args.steps:.12g}",
        "",
        "## Compilation",
        "",
        (
            "- Mean training-state fidelity of one compiled V4 block: "
            f"{compiler_metrics['mean_training_fidelity']:.10f}"
        ),
        (
            "- Worst training-state fidelity of one compiled V4 block: "
            f"{compiler_metrics['min_training_fidelity']:.10f}"
        ),
        (
            "- Maximum training-state population in the top padded levels: "
            f"{compiler_metrics['max_edge_population_after_block']:.3e}"
        ),
        "",
        "## Direct comparison with the N=80 Duffing reference",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"### xi = {float(row['xi']):g}",
                "",
                (
                    "- Maximum native-Kerr infidelity: "
                    f"{float(row['max_infidelity_native_kerr_vs_N80']):.6e}"
                ),
                (
                    "- Maximum exact-V4 split infidelity: "
                    f"{float(row['max_infidelity_exact_split_vs_N80']):.6e}"
                ),
                (
                    "- Maximum compiled-hybrid infidelity: "
                    f"{float(row['max_infidelity_compiled_hybrid_vs_N80']):.6e}"
                ),
                (
                    "- Maximum compiled-vs-exact-split infidelity: "
                    f"{float(row['max_infidelity_compiled_vs_exact_split']):.6e}"
                ),
                (
                    "- Maximum exact-D_pad vs N=80 infidelity: "
                    f"{float(row['max_infidelity_exact_Dpad_vs_N80']):.6e}"
                ),
                (
                    "- Maximum top-edge population during compiled evolution: "
                    f"{float(row['max_edge_population_during_compiled_trajectory']):.3e}"
                ),
                (
                    "- Validation status: "
                    f"{'PASS' if row['overall_validation_pass'] else 'FAIL'}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation rule",
            "",
            (
                "A converged hybrid claim requires all of the following: "
                "(i) exact-D_pad and N=80 Duffing agree; "
                "(ii) top-edge population is negligible; "
                "(iii) the exact-V4 split error is below the requested "
                "accuracy, showing that the time step is adequate; and "
                "(iv) the compiled hybrid remains close to the exact-V4 "
                "split and to the N=80 reference.  If (iii) fails, increase "
                "the number of time steps.  If (iv) fails while (iii) "
                "passes, increase compilation depth, iterations, or "
                "restarts."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dt = args.T / args.steps

    print("Building hard-sandwiched reference and padded operators...")
    ops_ref = hard_sandwiched_operators(
        args.reference_dim, args.omega0, args.beta
    )
    ops_pad = hard_sandwiched_operators(
        args.pad_dim, args.omega0, args.beta
    )
    print(
        "Split residuals: "
        f"N_ref={float(ops_ref['split_residual_norm']):.3e}, "
        f"D_pad={float(ops_pad['split_residual_norm']):.3e}"
    )
    gates_pad = build_fixed_gates(ops_pad, dt)

    inputs, targets, training_rows = build_training_snapshots(
        args.pad_dim,
        args.train_xi,
        args.steps,
        gates_pad,
    )
    write_csv(
        args.output_dir / "compilation_training_states.csv",
        training_rows[0].keys(),
        training_rows,
    )
    print(
        f"Compiling on {inputs.shape[1]} relevant padded-space states "
        f"({len(args.train_xi)} amplitudes x {args.steps} steps)..."
    )
    compiler = SnapDisplacementCompiler(
        args.pad_dim,
        args.active_dim,
        args.layers,
        inputs,
        targets,
        args.alpha_regularization,
    )
    best_vector, restart_rows = compile_block(
        args, compiler, args.output_dir
    )
    compiler_metrics = compiler.metrics(best_vector, args.edge_levels)

    print("Evaluating compiled trajectories against the N=80 reference...")
    time_rows, summary_rows = evaluate(
        args, ops_ref, ops_pad, gates_pad, best_vector
    )
    write_csv(
        args.output_dir / "hybrid_time_series.csv",
        time_rows[0].keys(),
        time_rows,
    )
    write_csv(
        args.output_dir / "hybrid_summary.csv",
        summary_rows[0].keys(),
        summary_rows,
    )

    metadata = {
        "omega0": args.omega0,
        "beta": args.beta,
        "lambda": float(ops_pad["lambda"]),
        "T": args.T,
        "steps": args.steps,
        "dt": dt,
        "reference_dimension": args.reference_dim,
        "padded_dimension": args.pad_dim,
        "active_dimension": args.active_dim,
        "layers": args.layers,
        "training_amplitudes": list(args.train_xi),
        "test_amplitudes": list(args.test_xi),
        "restarts": args.restarts,
        "maxiter": args.maxiter,
        "seed": args.seed,
        "edge_levels": args.edge_levels,
        "thresholds": {
            "max_reference_infidelity": args.max_reference_infidelity,
            "max_edge_population": args.max_edge_population,
            "max_split_infidelity": args.max_split_infidelity,
            "max_hybrid_infidelity": args.max_hybrid_infidelity,
        },
        "split_residual_reference": float(ops_ref["split_residual_norm"]),
        "split_residual_padded": float(ops_pad["split_residual_norm"]),
        "compiler_metrics": compiler_metrics,
        "best_restart": int(
            min(restart_rows, key=lambda row: row["final_objective"])[
                "restart"
            ]
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "command": " ".join(sys.argv),
    }
    with (args.output_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    verdict = verdict_text(args, compiler_metrics, summary_rows)
    (args.output_dir / "validation_verdict.md").write_text(
        verdict, encoding="utf-8"
    )

    if args.plot:
        create_plot(args.output_dir, time_rows, args.test_xi)

    print("\nValidation summary")
    print("------------------")
    print(
        "One-block training fidelity: "
        f"mean={compiler_metrics['mean_training_fidelity']:.10f}, "
        f"min={compiler_metrics['min_training_fidelity']:.10f}"
    )
    for row in summary_rows:
        print(
            f"xi={float(row['xi']):g}: "
            "max infidelity vs N=80 "
            f"native={float(row['max_infidelity_native_kerr_vs_N80']):.6e}, "
            f"exact-split={float(row['max_infidelity_exact_split_vs_N80']):.6e}, "
            f"compiled={float(row['max_infidelity_compiled_hybrid_vs_N80']):.6e}; "
            "compiled-vs-split="
            f"{float(row['max_infidelity_compiled_vs_exact_split']):.6e}; "
            "edge="
            f"{float(row['max_edge_population_during_compiled_trajectory']):.3e}; "
            f"status={'PASS' if row['overall_validation_pass'] else 'FAIL'}"
        )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
