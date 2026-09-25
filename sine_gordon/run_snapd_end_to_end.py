#!/usr/bin/env python3
"""Section 3.7 -- End-to-end, coherence-constrained SNAP+D deformation benchmark.

This driver tests the two-site double sine-Gordon target

    H_target = H_SG + lambda * sum_j (1 - cos(2 phi_j))

against three approximations:

1. native-only continuous evolution under H_SG;
2. a second-order native/correction product formula using the exact local
   correction block;
3. the same product formula with each local correction block replaced by a
   compiled SNAP+D circuit.

The native half steps are treated as hardware-native evolution.  One local
correction block is applied per product-formula step,

    U_step = exp(-i H_SG dt/2) U_corr exp(-i H_SG dt/2),

and the two sites receive the same local correction in parallel.  Therefore
the circuit-time estimate is r*L*t_layer, not 2*r*L*t_layer and not
N_site*r*L*t_layer.

The main scan is compiled at ``--screen-cut``.  The best coherence-feasible
point is then recompiled at the requested validation cutoffs, and target-state
cutoff convergence is evaluated by embedding the lower Fock basis into the
higher one.

Example
-------
python run_snapd_end_to_end.py \
    --lambda 0.3 --r-values 2,4,5,10 --layers 1,2,4,6 \
    --screen-cut 20 --validation-cuts 16,24 \
    --n-restarts 12 --maxiter 4000 --workers 12 --resume \
    --outdir results --tag snapd_e2e_lambda_0p3
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import tempfile
import time
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply

if __package__:
    from . import sglib as sg
    from .run_snapd import _run_restart, build_ansatz
else:
    import sglib as sg
    from run_snapd import _run_restart, build_ansatz


CHECKPOINT_VERSION = 1


def _parse_int_list(text: str, name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


def _safe_token(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([_json_safe(row) for row in rows])
        temporary = handle.name
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary = handle.name
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _trace(matrix: csr_matrix) -> complex:
    return complex(np.asarray(matrix.diagonal()).sum())


def _unit_norm(states: np.ndarray) -> np.ndarray:
    array = np.asarray(states, dtype=np.complex128)
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        if norm == 0 or not np.isfinite(norm):
            raise FloatingPointError("state has invalid norm")
        return array / norm
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise FloatingPointError("trajectory contains an invalid state")
    return array / norms[:, None]


def build_two_site_problem(
    n_cut: int,
    omega: float,
    E_J: float,
    kappa: float,
    coupling: float,
    convention: str = "spectral",
) -> dict[str, Any]:
    """Construct mutually consistent local and two-site Hamiltonians."""
    if n_cut < 4:
        raise ValueError("n_cut must be at least 4")
    if omega <= 0:
        raise ValueError("omega must be positive")
    if E_J < 0 or kappa < 0 or coupling < 0:
        raise ValueError("Hamiltonian strengths must be nonnegative")

    basis = sg.build_mode_basis(n_cut, omega)
    identity = eye(n_cut, dtype=np.complex128, format="csr")
    phi = csr_matrix(basis.phi)
    phi_sq = phi @ phi

    native_local = basis.kinetic + (
        basis.U_phi
        @ np.diag(E_J * (1.0 - np.cos(basis.lam_phi)))
        @ basis.U_phi.conj().T
    )
    correction_local = (
        basis.U_phi
        @ np.diag(coupling * (1.0 - np.cos(2.0 * basis.lam_phi)))
        @ basis.U_phi.conj().T
    )
    if convention == "physical_sandwich":
        if __package__:
            from .native_correction import cosine_sandwich, sandwich_local_operators
        else:
            from native_correction import cosine_sandwich, sandwich_local_operators
        q2,p2=sandwich_local_operators(basis)
        phi_sq=csr_matrix(q2)
        native_local=.5*p2+E_J*(np.eye(n_cut)-cosine_sandwich(n_cut,omega))
        correction_local=coupling*(np.eye(n_cut)-cosine_sandwich(n_cut,omega,2))
    elif convention != "spectral":
        raise ValueError("convention must be spectral or physical_sandwich")
    native_with_gradient_self = csr_matrix(native_local) + 0.5 * kappa * phi_sq
    H_native = (
        kron(native_with_gradient_self, identity, format="csr")
        + kron(identity, native_with_gradient_self, format="csr")
        - kappa * kron(phi, phi, format="csr")
    )
    correction_local_sparse = csr_matrix(correction_local)
    H_correction = (
        kron(correction_local_sparse, identity, format="csr")
        + kron(identity, correction_local_sparse, format="csr")
    )
    H_target = H_native + H_correction

    for name, operator in (
        ("H_native", H_native),
        ("H_correction", H_correction),
        ("H_target", H_target),
    ):
        residual = operator - operator.getH()
        if residual.nnz and np.max(np.abs(residual.data)) > 2e-11:
            raise FloatingPointError(f"{name} is not Hermitian")

    return {
        "basis": basis,
        "a": sg.annihilation(n_cut),
        "correction_local": np.asarray(correction_local),
        "H_native": H_native.tocsr(),
        "H_target": H_target.tocsr(),
        "native_local_with_bond": native_with_gradient_self.toarray(),
        "bond_strength": kappa,
        "operator_convention": convention,
    }


def exact_trajectory(
    Hamiltonian: csr_matrix,
    psi0: np.ndarray,
    T: float,
    r: int,
) -> np.ndarray:
    """Return states at the r+1 stroboscopic times, including t=0."""
    generator = -1j * Hamiltonian
    states = expm_multiply(
        generator,
        np.asarray(psi0).reshape(-1),
        start=0.0,
        stop=float(T),
        num=int(r) + 1,
        endpoint=True,
        traceA=-1j * _trace(Hamiltonian),
    )
    return _unit_norm(states)


def apply_local_both(local_unitary: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Apply U tensor U to one state or a matrix whose columns are states."""
    U = np.asarray(local_unitary, dtype=np.complex128)
    n_cut = U.shape[0]
    if U.shape != (n_cut, n_cut):
        raise ValueError("local unitary must be square")
    values = np.asarray(states, dtype=np.complex128)
    was_vector = values.ndim == 1
    matrix = values.reshape(n_cut * n_cut, -1)
    output = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        tensor = matrix[:, column].reshape(n_cut, n_cut)
        output[:, column] = (U @ tensor @ U.T).reshape(-1)
    return output[:, 0] if was_vector else output


def product_formula_trajectories(
    H_native: csr_matrix,
    psi0: np.ndarray,
    T: float,
    r: int,
    local_blocks: list[np.ndarray],
) -> list[np.ndarray]:
    """Propagate several correction choices through the same native halves."""
    if not local_blocks:
        return []
    dt = float(T) / int(r)
    states = np.column_stack(
        [np.asarray(psi0, dtype=np.complex128).reshape(-1)] * len(local_blocks)
    )
    histories = [[states[:, index].copy()] for index in range(states.shape[1])]
    half_generator = -0.5j * dt * H_native
    half_trace = -0.5j * dt * _trace(H_native)
    for _ in range(r):
        states = expm_multiply(
            half_generator,
            states,
            traceA=half_trace,
        )
        for index, block in enumerate(local_blocks):
            states[:, index] = apply_local_both(block, states[:, index])
        states = expm_multiply(
            half_generator,
            states,
            traceA=half_trace,
        )
        norms = np.linalg.norm(states, axis=0)
        if np.any(norms == 0) or not np.all(np.isfinite(norms)):
            raise FloatingPointError("product formula produced an invalid state")
        states /= norms[None, :]
        for index in range(states.shape[1]):
            histories[index].append(states[:, index].copy())
    return [_unit_norm(np.asarray(history)) for history in histories]


def trajectory_infidelity(
    target: np.ndarray,
    approximation: np.ndarray,
) -> np.ndarray:
    target = _unit_norm(target)
    approximation = _unit_norm(approximation)
    if target.shape != approximation.shape:
        raise ValueError("trajectory shapes do not match")
    overlaps = np.einsum("ti,ti->t", target.conj(), approximation)
    return np.clip(1.0 - np.abs(overlaps) ** 2, 0.0, 1.0)


def trajectory_leakage(states: np.ndarray, n_cut: int) -> np.ndarray:
    tensors = _unit_norm(states).reshape(-1, n_cut, n_cut)
    first = np.sum(np.abs(tensors[:, -1, :]) ** 2, axis=1)
    second = np.sum(np.abs(tensors[:, :, -1]) ** 2, axis=1)
    return np.maximum(first, second)


def trajectory_phi(states: np.ndarray, phi: np.ndarray) -> np.ndarray:
    tensors = _unit_norm(states).reshape(-1, phi.shape[0], phi.shape[0])
    output = np.empty((tensors.shape[0], 2), dtype=float)
    for index, tensor in enumerate(tensors):
        output[index, 0] = float(
            np.real(np.vdot(tensor, phi @ tensor))
        )
        output[index, 1] = float(
            np.real(np.vdot(tensor, tensor @ phi.T))
        )
    return output


def embed_trajectory(states: np.ndarray, low_cut: int, high_cut: int) -> np.ndarray:
    if low_cut > high_cut:
        raise ValueError("low_cut cannot exceed high_cut")
    low = _unit_norm(states).reshape(-1, low_cut, low_cut)
    high = np.zeros(
        (low.shape[0], high_cut, high_cut), dtype=np.complex128
    )
    high[:, :low_cut, :low_cut] = low
    return _unit_norm(high.reshape(low.shape[0], -1))


def feasible_pairs(
    r_values: list[int],
    layers: list[int],
    t_layer_us: float,
    coherence_budget_us: float,
) -> list[tuple[int, int]]:
    pairs = []
    for r in r_values:
        for layer in layers:
            circuit_time = r * layer * t_layer_us
            if circuit_time <= coherence_budget_us * (1.0 + 1e-12):
                pairs.append((r, layer))
    return pairs


def resize_ansatz_vector(
    vector: np.ndarray,
    layer: int,
    old_cut: int,
    new_cut: int,
) -> np.ndarray:
    """Map displacement parameters and common SNAP phases to a new cutoff."""
    source = np.asarray(vector, dtype=float)
    expected = layer * (old_cut + 2)
    if source.shape != (expected,):
        raise ValueError(
            f"warm-start vector has size {source.size}; expected {expected}"
        )
    destination = np.zeros(layer * (new_cut + 2), dtype=float)
    common = min(old_cut, new_cut)
    for index in range(layer):
        old_start = index * (old_cut + 2)
        new_start = index * (new_cut + 2)
        destination[new_start : new_start + 2] = source[
            old_start : old_start + 2
        ]
        destination[
            new_start + 2 : new_start + 2 + common
        ] = source[old_start + 2 : old_start + 2 + common]
    return destination


def _compile_signature(
    *,
    n_cut: int,
    coupling: float,
    dt: float,
    layer: int,
    n_restarts: int,
    maxiter: int,
    seed: int,
    alpha_bound: float,
    omega: float,
) -> str:
    return json.dumps(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "ansatz": "left_product_D_then_SNAP",
            "target": "second_harmonic",
            "n_cut": n_cut,
            "coupling": coupling,
            "dt": dt,
            "layer": layer,
            "n_restarts": n_restarts,
            "maxiter": maxiter,
            "seed": seed,
            "alpha_bound": alpha_bound,
            "omega": omega,
        },
        sort_keys=True,
    )


def _load_compile_checkpoint(
    path: Path,
    signature: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        found = str(data["signature"].item())
        if found != signature:
            raise RuntimeError(
                f"stale or incompatible checkpoint: {path}. "
                "Use a different --tag or move the old checkpoint."
            )
        return {
            "x": data["x"].copy(),
            "infidelity": float(data["infidelity"].item()),
            "spectral_error": float(data["spectral_error"].item()),
            "optimization_seconds": float(data["optimization_seconds"].item()),
            "success_count": int(data["success_count"].item()),
            "messages": str(data["messages"].item()),
            "resumed": True,
        }


def compile_local_block(
    *,
    U_target: np.ndarray,
    a: np.ndarray,
    n_cut: int,
    coupling: float,
    dt: float,
    layer: int,
    n_restarts: int,
    maxiter: int,
    workers: int,
    seed: int,
    alpha_bound: float,
    omega: float,
    checkpoint: Path,
    resume: bool,
    warm_start: np.ndarray | None,
) -> dict[str, Any]:
    signature = _compile_signature(
        n_cut=n_cut,
        coupling=coupling,
        dt=dt,
        layer=layer,
        n_restarts=n_restarts,
        maxiter=maxiter,
        seed=seed,
        alpha_bound=alpha_bound,
        omega=omega,
    )
    if resume:
        saved = _load_compile_checkpoint(checkpoint, signature)
        if saved is not None:
            print(
                f"[compile] resumed nc={n_cut} dt={dt:g} L={layer} "
                f"infid={saved['infidelity']:.4e}",
                flush=True,
            )
            return saved

    starts: list[np.ndarray] = []
    parameter_count = layer * (n_cut + 2)
    for restart in range(n_restarts):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    seed,
                    n_cut,
                    int(round(coupling * 10**8)),
                    int(round(dt * 10**8)),
                    layer,
                    restart,
                ]
            )
        )
        x0 = np.concatenate(
            [
                np.concatenate(
                    [rng.normal(0.0, 0.3, 2), rng.normal(0.0, 0.5, n_cut)]
                )
                for _ in range(layer)
            ]
        )
        if warm_start is not None and warm_start.size <= parameter_count:
            if restart in (0, 1):
                x0[:] = 0.0
                x0[: warm_start.size] = warm_start
                if restart == 1:
                    x0 += rng.normal(0.0, 0.02, parameter_count)
        starts.append(x0)

    payloads = [
        (
            U_target,
            layer,
            n_cut,
            a,
            x0,
            maxiter,
            alpha_bound,
        )
        for x0 in starts
    ]
    start_time = time.perf_counter()
    outcomes: list[tuple[float, np.ndarray, bool, str]] = []
    worker_count = max(1, min(workers, n_restarts))
    if worker_count == 1:
        for restart, payload in enumerate(payloads):
            try:
                outcomes.append(_run_restart(payload))
            except Exception as exc:  # keep other independent restarts alive
                print(
                    f"[compile] restart {restart} failed: {exc}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            future_to_restart = {
                pool.submit(_run_restart, payload): restart
                for restart, payload in enumerate(payloads)
            }
            for future in as_completed(future_to_restart):
                restart = future_to_restart[future]
                try:
                    outcomes.append(future.result())
                except Exception as exc:
                    print(
                        f"[compile] restart {restart} failed: {exc}",
                        flush=True,
                    )
    elapsed = time.perf_counter() - start_time
    if not outcomes:
        raise RuntimeError(
            f"all optimizer restarts failed for nc={n_cut}, dt={dt}, L={layer}"
        )
    best = min(outcomes, key=lambda item: item[0])
    best_infid, best_x, _, _ = best
    U_compiled = build_ansatz(best_x, layer, n_cut, a)
    spectral_error = sg.phase_aligned_operator_distance(U_target, U_compiled)
    success_count = sum(bool(item[2]) for item in outcomes)
    messages = " | ".join(sorted(set(item[3] for item in outcomes)))
    result = {
        "x": best_x.copy(),
        "infidelity": float(best_infid),
        "spectral_error": float(spectral_error),
        "optimization_seconds": float(elapsed),
        "success_count": int(success_count),
        "messages": messages,
        "resumed": False,
    }
    _atomic_npz(
        checkpoint,
        {
            "signature": np.array(signature),
            "x": best_x,
            "infidelity": np.array(best_infid),
            "spectral_error": np.array(spectral_error),
            "optimization_seconds": np.array(elapsed),
            "success_count": np.array(success_count),
            "messages": np.array(messages),
        },
    )
    print(
        f"[compile] nc={n_cut} dt={dt:g} L={layer} "
        f"infid={best_infid:.4e} eps2={spectral_error:.4e} "
        f"success={success_count}/{len(outcomes)} time={elapsed:.1f}s",
        flush=True,
    )
    return result


def evaluate_point(
    *,
    problem: dict[str, Any],
    psi0: np.ndarray,
    target_trajectory: np.ndarray,
    native_trajectory: np.ndarray,
    coupling: float,
    T: float,
    r: int,
    layer: int,
    compiled_x: np.ndarray,
    block_infidelity: float,
    block_spectral_error: float,
    circuit_time_us: float,
    coherence_budget_us: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_cut = problem["basis"].n_cut
    dt = T / r
    U_exact_local = expm(-1j * problem["correction_local"] * dt)
    U_compiled_local = build_ansatz(
        compiled_x, layer, n_cut, problem["a"]
    )
    exact_correction, compiled_correction = product_formula_trajectories(
        problem["H_native"],
        psi0,
        T,
        r,
        [U_exact_local, U_compiled_local],
    )

    native_infidelity = trajectory_infidelity(
        target_trajectory, native_trajectory
    )
    exact_infidelity = trajectory_infidelity(
        target_trajectory, exact_correction
    )
    hybrid_infidelity = trajectory_infidelity(
        target_trajectory, compiled_correction
    )
    compilation_only = trajectory_infidelity(
        exact_correction, compiled_correction
    )
    target_leakage = trajectory_leakage(target_trajectory, n_cut)
    hybrid_leakage = trajectory_leakage(compiled_correction, n_cut)
    native_max = float(np.max(native_infidelity))
    hybrid_max = float(np.max(hybrid_infidelity))
    improvement = (
        float(native_max / hybrid_max)
        if hybrid_max > 0
        else float("inf")
    )
    row = {
        "lambda": coupling,
        "n_cut": n_cut,
        "T": T,
        "r": r,
        "dt": dt,
        "L": layer,
        "rL": r * layer,
        "circuit_time_us": circuit_time_us,
        "coherence_budget_us": coherence_budget_us,
        "coherence_feasible": circuit_time_us <= coherence_budget_us * (
            1.0 + 1e-12
        ),
        "block_infidelity": block_infidelity,
        "block_spectral_error": block_spectral_error,
        "repeated_operator_error_bound": min(
            2.0, r * block_spectral_error
        ),
        "native_max_infidelity": native_max,
        "exact_correction_max_infidelity": float(np.max(exact_infidelity)),
        "hybrid_max_infidelity": hybrid_max,
        "compilation_only_max_infidelity": float(
            np.max(compilation_only)
        ),
        "improvement_ratio": improvement,
        "target_max_leakage": float(np.max(target_leakage)),
        "hybrid_max_leakage": float(np.max(hybrid_leakage)),
        "native_final_infidelity": float(native_infidelity[-1]),
        "exact_correction_final_infidelity": float(exact_infidelity[-1]),
        "hybrid_final_infidelity": float(hybrid_infidelity[-1]),
    }
    arrays = {
        "time": np.linspace(0.0, T, r + 1),
        "native_infidelity": native_infidelity,
        "exact_correction_infidelity": exact_infidelity,
        "hybrid_infidelity": hybrid_infidelity,
        "compilation_only_infidelity": compilation_only,
        "target_leakage": target_leakage,
        "hybrid_leakage": hybrid_leakage,
        "target_phi": trajectory_phi(
            target_trajectory, problem["basis"].phi
        ),
        "hybrid_phi": trajectory_phi(
            compiled_correction, problem["basis"].phi
        ),
    }
    return row, arrays


def _candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    passes = bool(row["passes_positive_screen"])
    return (
        0 if passes else 1,
        float(row["hybrid_max_infidelity"]),
        -float(row["improvement_ratio"]),
        float(row["circuit_time_us"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lambda", dest="coupling", type=float, required=True)
    parser.add_argument("--E-J", dest="E_J", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.35)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--phi0", type=float, default=1.0)
    parser.add_argument("--pi0", type=float, default=0.0)
    parser.add_argument("--initial-site", type=int, default=0, choices=[0, 1])
    parser.add_argument("--r-values", default="2,4,5,10")
    parser.add_argument("--layers", default="1,2,4,6")
    parser.add_argument("--screen-cut", type=int, default=20)
    parser.add_argument("--validation-cuts", default="16,24")
    parser.add_argument("--n-restarts", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--alpha-bound", type=float, default=3.0)
    parser.add_argument("--chi-mhz", type=float, default=1.5)
    parser.add_argument("--t-displacement-us", type=float, required=True,
                        help="explicit displacement duration, including zero only for an idealized lower bound")
    parser.add_argument("--t2-anc-us", type=float, default=80.0)
    parser.add_argument("--coherence-fraction", type=float, default=0.25)
    parser.add_argument("--native-threshold", type=float, default=1e-2)
    parser.add_argument("--hybrid-threshold", type=float, default=1e-3)
    parser.add_argument("--improvement-threshold", type=float, default=10.0)
    parser.add_argument("--leakage-tol", type=float, default=1e-6)
    parser.add_argument("--cutoff-tol", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.coupling <= 0:
        raise ValueError("--lambda must be positive")
    if args.E_J < 0 or args.kappa < 0:
        raise ValueError("--E-J and --kappa must be nonnegative")
    if args.omega <= 0 or args.T <= 0:
        raise ValueError("--omega and --T must be positive")
    if args.screen_cut < 4:
        raise ValueError("--screen-cut must be at least 4")
    if args.n_restarts < 1 or args.maxiter < 1:
        raise ValueError("--n-restarts and --maxiter must be positive")
    if args.chi_mhz <= 0 or args.t2_anc_us <= 0:
        raise ValueError("hardware time scales must be positive")
    if args.t_displacement_us < 0:
        raise ValueError("--t-displacement-us cannot be negative")
    if not 0 < args.coherence_fraction <= 1:
        raise ValueError("--coherence-fraction must lie in (0,1]")

    r_values = _parse_int_list(args.r_values, "--r-values")
    layers = _parse_int_list(args.layers, "--layers")
    validation_cuts = _parse_int_list(
        args.validation_cuts, "--validation-cuts"
    )
    if args.screen_cut in validation_cuts:
        validation_cuts.remove(args.screen_cut)
    all_validation_cuts = sorted(
        set([args.screen_cut] + validation_cuts)
    )

    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("LSB_DJOB_NUMPROC", "1"))
    workers = max(1, min(workers, args.n_restarts))
    t_snap_us = 1.0 / args.chi_mhz
    t_layer_us = t_snap_us + args.t_displacement_us
    coherence_budget_us = args.coherence_fraction * args.t2_anc_us
    pairs = feasible_pairs(
        r_values, layers, t_layer_us, coherence_budget_us
    )
    if not pairs:
        raise RuntimeError(
            "no (r,L) pair fits the requested coherence budget"
        )
    excluded = [
        (r, layer)
        for r in r_values
        for layer in layers
        if (r, layer) not in pairs
    ]

    tag = args.tag.strip() or (
        f"snapd_e2e_lambda_{_safe_token(args.coupling)}"
    )
    if not all(character.isalnum() or character in "._-" for character in tag):
        raise ValueError("--tag may contain only letters, digits, '.', '_', '-'")
    outdir = Path(args.outdir).expanduser().resolve()
    compile_dir = outdir / f"{tag}_compile"
    outdir.mkdir(parents=True, exist_ok=True)
    compile_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[setup] lambda={args.coupling:g} screen_cut={args.screen_cut} "
        f"T={args.T:g} phi0={args.phi0:g}",
        flush=True,
    )
    print(
        f"[setup] t_SNAP={t_snap_us:.6g} us "
        f"t_layer={t_layer_us:.6g} us "
        f"budget={coherence_budget_us:.6g} us",
        flush=True,
    )
    print(f"[setup] feasible pairs ({len(pairs)}): {pairs}", flush=True)
    print(f"[setup] excluded pairs: {excluded}", flush=True)
    print(
        f"[setup] optimizer workers={workers} "
        f"restarts={args.n_restarts} maxiter={args.maxiter}",
        flush=True,
    )
    if args.t_displacement_us == 0:
        print(
            "[setup] WARNING displacement duration is zero; circuit times "
            "are SNAP-only lower bounds",
            flush=True,
        )
    if args.dry_run:
        return

    problem = build_two_site_problem(
        args.screen_cut,
        args.omega,
        args.E_J,
        args.kappa,
        args.coupling,
    )
    psi0, _, _, _, initial_info = sg.make_initial_condition(
        "coherent_single",
        2,
        args.screen_cut,
        args.omega,
        args.E_J,
        args.kappa,
        args.phi0,
        args.pi0,
        args.initial_site,
        False,
    )
    psi0 = psi0.reshape(-1)

    target_by_r: dict[int, np.ndarray] = {}
    native_by_r: dict[int, np.ndarray] = {}
    for r in r_values:
        target_by_r[r] = exact_trajectory(
            problem["H_target"], psi0, args.T, r
        )
        native_by_r[r] = exact_trajectory(
            problem["H_native"], psi0, args.T, r
        )

    summary_rows: list[dict[str, Any]] = []
    trajectory_arrays: dict[str, np.ndarray] = {}
    compiled_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for r in r_values:
        dt = args.T / r
        U_target_local = expm(
            -1j * problem["correction_local"] * dt
        )
        previous_x: np.ndarray | None = None
        for layer in layers:
            if (r, layer) not in pairs:
                continue
            checkpoint = compile_dir / (
                f"nc{args.screen_cut}_r{r}_L{layer}.npz"
            )
            compiled = compile_local_block(
                U_target=U_target_local,
                a=problem["a"],
                n_cut=args.screen_cut,
                coupling=args.coupling,
                dt=dt,
                layer=layer,
                n_restarts=args.n_restarts,
                maxiter=args.maxiter,
                workers=workers,
                seed=args.seed,
                alpha_bound=args.alpha_bound,
                omega=args.omega,
                checkpoint=checkpoint,
                resume=args.resume,
                warm_start=previous_x,
            )
            previous_x = compiled["x"]
            compiled_by_pair[(r, layer)] = compiled
            circuit_time = r * layer * t_layer_us
            row, arrays = evaluate_point(
                problem=problem,
                psi0=psi0,
                target_trajectory=target_by_r[r],
                native_trajectory=native_by_r[r],
                coupling=args.coupling,
                T=args.T,
                r=r,
                layer=layer,
                compiled_x=compiled["x"],
                block_infidelity=compiled["infidelity"],
                block_spectral_error=compiled["spectral_error"],
                circuit_time_us=circuit_time,
                coherence_budget_us=coherence_budget_us,
            )
            row.update(
                optimizer_seconds=compiled["optimization_seconds"],
                optimizer_success_count=compiled["success_count"],
                optimizer_restarts=args.n_restarts,
                optimizer_resumed=compiled["resumed"],
            )
            row["passes_positive_screen"] = bool(
                row["coherence_feasible"]
                and row["native_max_infidelity"] >= args.native_threshold
                and row["hybrid_max_infidelity"] <= args.hybrid_threshold
                and row["improvement_ratio"] >= args.improvement_threshold
                and row["target_max_leakage"] <= args.leakage_tol
                and row["hybrid_max_leakage"] <= args.leakage_tol
            )
            row["selected_candidate"] = False
            summary_rows.append(row)
            prefix = f"screen_r{r}_L{layer}_"
            trajectory_arrays.update(
                {prefix + key: value for key, value in arrays.items()}
            )
            print(
                f"[screen] r={r:2d} L={layer:2d} "
                f"native={row['native_max_infidelity']:.3e} "
                f"exact={row['exact_correction_max_infidelity']:.3e} "
                f"hybrid={row['hybrid_max_infidelity']:.3e} "
                f"improvement={row['improvement_ratio']:.2f} "
                f"time={circuit_time:.2f}us "
                f"pass={row['passes_positive_screen']}",
                flush=True,
            )

    candidate = min(summary_rows, key=_candidate_key)
    candidate["selected_candidate"] = True
    candidate_r = int(candidate["r"])
    candidate_layer = int(candidate["L"])
    print(
        f"[select] r={candidate_r} L={candidate_layer} "
        f"hybrid={candidate['hybrid_max_infidelity']:.3e} "
        f"improvement={candidate['improvement_ratio']:.2f} "
        f"passes={candidate['passes_positive_screen']}",
        flush=True,
    )

    validation_rows: list[dict[str, Any]] = []
    validation_target_states: dict[int, np.ndarray] = {}
    validation_target_phi: dict[int, np.ndarray] = {}
    if not args.skip_validation:
        previous_cut_x: np.ndarray | None = None
        previous_cut: int | None = None
        for n_cut in all_validation_cuts:
            validation_problem = (
                problem
                if n_cut == args.screen_cut
                else build_two_site_problem(
                    n_cut,
                    args.omega,
                    args.E_J,
                    args.kappa,
                    args.coupling,
                )
            )
            validation_psi0, _, _, _, _ = sg.make_initial_condition(
                "coherent_single",
                2,
                n_cut,
                args.omega,
                args.E_J,
                args.kappa,
                args.phi0,
                args.pi0,
                args.initial_site,
                False,
            )
            validation_psi0 = validation_psi0.reshape(-1)
            validation_target = exact_trajectory(
                validation_problem["H_target"],
                validation_psi0,
                args.T,
                candidate_r,
            )
            validation_native = exact_trajectory(
                validation_problem["H_native"],
                validation_psi0,
                args.T,
                candidate_r,
            )
            if n_cut == args.screen_cut:
                compiled = compiled_by_pair[
                    (candidate_r, candidate_layer)
                ]
            else:
                validation_U_target = expm(
                    -1j
                    * validation_problem["correction_local"]
                    * (args.T / candidate_r)
                )
                checkpoint = compile_dir / (
                    f"nc{n_cut}_r{candidate_r}_L{candidate_layer}.npz"
                )
                compiled = compile_local_block(
                    U_target=validation_U_target,
                    a=validation_problem["a"],
                    n_cut=n_cut,
                    coupling=args.coupling,
                    dt=args.T / candidate_r,
                    layer=candidate_layer,
                    n_restarts=args.n_restarts,
                    maxiter=args.maxiter,
                    workers=workers,
                    seed=args.seed,
                    alpha_bound=args.alpha_bound,
                    omega=args.omega,
                    checkpoint=checkpoint,
                    resume=args.resume,
                    warm_start=(
                        resize_ansatz_vector(
                            previous_cut_x,
                            candidate_layer,
                            previous_cut,
                            n_cut,
                        )
                        if previous_cut_x is not None
                        and previous_cut is not None
                        else None
                    ),
                )
            previous_cut_x = compiled["x"]
            previous_cut = n_cut
            row, arrays = evaluate_point(
                problem=validation_problem,
                psi0=validation_psi0,
                target_trajectory=validation_target,
                native_trajectory=validation_native,
                coupling=args.coupling,
                T=args.T,
                r=candidate_r,
                layer=candidate_layer,
                compiled_x=compiled["x"],
                block_infidelity=compiled["infidelity"],
                block_spectral_error=compiled["spectral_error"],
                circuit_time_us=candidate_r
                * candidate_layer
                * t_layer_us,
                coherence_budget_us=coherence_budget_us,
            )
            row.update(
                selected_candidate=True,
                optimizer_seconds=compiled["optimization_seconds"],
                optimizer_success_count=compiled["success_count"],
                optimizer_restarts=args.n_restarts,
                optimizer_resumed=compiled["resumed"],
                cutoff_pair="",
                target_cutoff_max_infidelity="",
                target_cutoff_max_phi_change="",
                cutoff_pass="",
            )
            validation_rows.append(row)
            validation_target_states[n_cut] = validation_target
            validation_target_phi[n_cut] = arrays["target_phi"]
            prefix = f"validation_nc{n_cut}_"
            trajectory_arrays.update(
                {prefix + key: value for key, value in arrays.items()}
            )

        for lower, higher in zip(
            all_validation_cuts[:-1], all_validation_cuts[1:]
        ):
            embedded = embed_trajectory(
                validation_target_states[lower], lower, higher
            )
            cutoff_infidelity = trajectory_infidelity(
                validation_target_states[higher], embedded
            )
            phi_change = float(
                np.max(
                    np.abs(
                        validation_target_phi[higher]
                        - validation_target_phi[lower]
                    )
                )
            )
            higher_row = next(
                row for row in validation_rows if int(row["n_cut"]) == higher
            )
            higher_row["cutoff_pair"] = f"{lower}->{higher}"
            higher_row["target_cutoff_max_infidelity"] = float(
                np.max(cutoff_infidelity)
            )
            higher_row["target_cutoff_max_phi_change"] = phi_change
            higher_row["cutoff_pass"] = bool(
                phi_change <= args.cutoff_tol
                and higher_row["target_max_leakage"] <= args.leakage_tol
                and higher_row["hybrid_max_leakage"] <= args.leakage_tol
            )
            trajectory_arrays[
                f"cutoff_{lower}_to_{higher}_target_infidelity"
            ] = cutoff_infidelity
            print(
                f"[cutoff] {lower}->{higher} "
                f"target_infid={np.max(cutoff_infidelity):.3e} "
                f"phi_change={phi_change:.3e} "
                f"target_leak={higher_row['target_max_leakage']:.3e} "
                f"hybrid_leak={higher_row['hybrid_max_leakage']:.3e} "
                f"pass={higher_row['cutoff_pass']}",
                flush=True,
            )

    metadata = {
        "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "lsf_job_index": os.environ.get("LSB_JOBINDEX"),
        "E_J": args.E_J,
        "kappa": args.kappa,
        "omega": args.omega,
        "lambda": args.coupling,
        "T": args.T,
        "phi0": args.phi0,
        "pi0": args.pi0,
        "initial_site": args.initial_site,
        "initial_condition": initial_info,
        "r_values": r_values,
        "layers": layers,
        "screen_cut": args.screen_cut,
        "validation_cuts": all_validation_cuts,
        "feasible_pairs": pairs,
        "excluded_pairs": excluded,
        "n_restarts": args.n_restarts,
        "maxiter": args.maxiter,
        "workers": workers,
        "seed": args.seed,
        "alpha_bound": args.alpha_bound,
        "chi_mhz": args.chi_mhz,
        "t_snap_us": t_snap_us,
        "t_displacement_us": args.t_displacement_us,
        "t_layer_us": t_layer_us,
        "t2_anc_us": args.t2_anc_us,
        "coherence_fraction": args.coherence_fraction,
        "coherence_budget_us": coherence_budget_us,
        "timing_is_snap_only_lower_bound": args.t_displacement_us == 0,
        "timing_scope": "programmable local correction only; native waits, switching, preparation and readout are additional",
        "hardware_feasibility": "not_evaluated",
        "site_blocks_assumed_parallel": True,
        "native_threshold": args.native_threshold,
        "hybrid_threshold": args.hybrid_threshold,
        "improvement_threshold": args.improvement_threshold,
        "leakage_tolerance": args.leakage_tol,
        "cutoff_tolerance": args.cutoff_tol,
        "selected_candidate": candidate,
        "validation_performed": not args.skip_validation,
    }
    _atomic_csv(outdir / f"{tag}_summary.csv", summary_rows)
    if validation_rows:
        _atomic_csv(
            outdir / f"{tag}_validation.csv", validation_rows
        )
    _atomic_npz(
        outdir / f"{tag}_trajectories.npz", trajectory_arrays
    )
    _atomic_json(outdir / f"{tag}.json", metadata)
    print(
        f"[save] wrote {outdir / f'{tag}_summary.csv'}",
        flush=True,
    )
    if validation_rows:
        print(
            f"[save] wrote {outdir / f'{tag}_validation.csv'}",
            flush=True,
        )
    print(
        f"[save] wrote {outdir / f'{tag}_trajectories.npz'}",
        flush=True,
    )
    print(f"[save] wrote {outdir / f'{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
