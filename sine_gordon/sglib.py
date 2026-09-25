#!/usr/bin/env python3
"""Numerical utilities for the finite-cutoff sine-Gordon experiments.

The implementation uses dense local operators but never constructs the full
many-body Hamiltonian.  A wavefunction is stored as an ``n_cut``-by-... tensor;
local operators are applied one axis at a time and the potential is diagonal in
the tensor-product phase basis.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def annihilation(n_cut: int) -> np.ndarray:
    a = np.zeros((n_cut, n_cut), dtype=np.complex128)
    if n_cut > 1:
        a[np.arange(n_cut - 1), np.arange(1, n_cut)] = np.sqrt(
            np.arange(1, n_cut)
        )
    return a


def mode_operators(n_cut: int, omega: float):
    if n_cut < 2:
        raise ValueError("n_cut must be at least 2")
    if omega <= 0:
        raise ValueError("omega must be positive")
    a = annihilation(n_cut)
    ad = a.conj().T
    phi = (a + ad) / np.sqrt(2.0 * omega)
    pi = -1j * np.sqrt(omega / 2.0) * (a - ad)
    num = ad @ a
    return phi, pi, num


@dataclass(frozen=True)
class ModeBasis:
    n_cut: int
    omega: float
    phi: np.ndarray
    pi: np.ndarray
    num: np.ndarray
    lam_phi: np.ndarray
    U_phi: np.ndarray
    kinetic: np.ndarray
    delta_phi_sq: float


def build_mode_basis(n_cut: int, omega: float) -> ModeBasis:
    phi, pi, num = mode_operators(n_cut, omega)
    lam_phi, U_phi = np.linalg.eigh(phi)
    kinetic = 0.5 * (pi @ pi)
    vacuum = np.zeros(n_cut, dtype=np.complex128)
    vacuum[0] = 1.0
    mean = np.vdot(vacuum, phi @ vacuum)
    variance = np.vdot(vacuum, (phi @ phi) @ vacuum) - mean * mean
    return ModeBasis(
        n_cut=n_cut,
        omega=float(omega),
        phi=phi,
        pi=pi,
        num=num,
        lam_phi=lam_phi,
        U_phi=U_phi,
        kinetic=kinetic,
        delta_phi_sq=float(np.real(variance)),
    )


def _apply_axis(op: np.ndarray, psi: np.ndarray, axis: int) -> np.ndarray:
    """Apply a local matrix to one tensor axis without forming a Kronecker product."""
    moved = np.moveaxis(psi, axis, 0)
    shape = moved.shape
    out = op @ moved.reshape(shape[0], -1)
    return np.moveaxis(out.reshape(shape), 0, axis)


def expect_mode(psi: np.ndarray, op: np.ndarray, site: int) -> complex:
    applied = _apply_axis(op, psi, site)
    return np.vdot(psi.reshape(-1), applied.reshape(-1))


def infidelity(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0 or nb == 0:
        raise ValueError("cannot compute fidelity of a zero vector")
    overlap = np.vdot(av, bv) / (na * nb)
    return float(np.clip(1.0 - abs(overlap) ** 2, 0.0, 1.0))


def phase_aligned_operator_distance(
    target: np.ndarray, approximation: np.ndarray
) -> float:
    """Return a reproducibly global-phase-aligned spectral-norm distance.

    Trace infidelity is a useful average metric, but it does not by itself
    control the worst-case action of a compiled unitary.  This phase-aligned
    spectral-norm distance does, and it composes through the telescoping
    inequality ``||U^r-V^r||_2 <= r ||U-V||_2``.

    The phase is fixed by the Hilbert--Schmidt overlap.  This does not assert
    that it is the phase minimizing the spectral norm.
    """
    target = np.asarray(target, dtype=np.complex128)
    approximation = np.asarray(approximation, dtype=np.complex128)
    if (
        target.ndim != 2
        or target.shape != approximation.shape
        or target.shape[0] != target.shape[1]
    ):
        raise ValueError("operator inputs must be square matrices of equal shape")
    overlap = np.trace(target.conj().T @ approximation)
    phase = 1.0 if abs(overlap) == 0 else np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(target - phase * approximation, 2))


def estimate_memory_gb(
    n_cut: int, n_site: int, n_vectors: int = 3, safety: float = 1.0
) -> float:
    amplitudes = int(n_cut) ** int(n_site)
    return float(safety * n_vectors * amplitudes * 16 / 1024**3)


def estimate_split_memory_gb(n_cut: int, n_site: int) -> float:
    # State, phase-basis state, local-matmul output, potential grid, and
    # transient NumPy/SciPy workspaces.  The multiplier is deliberately
    # conservative because BLAS/allocator behavior varies between Hazel nodes.
    state_gb = estimate_memory_gb(n_cut, n_site, n_vectors=1)
    return max(0.25, 9.0 * state_gb + 0.5)


def _coherent_state(alpha: complex, n_cut: int) -> np.ndarray:
    coeff = np.empty(n_cut, dtype=np.complex128)
    coeff[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, n_cut):
        coeff[n] = coeff[n - 1] * alpha / np.sqrt(n)
    norm = np.linalg.norm(coeff)
    if norm == 0:
        raise ValueError("coherent-state truncation underflowed")
    return coeff / norm


def _product_state(local_states: list[np.ndarray]) -> np.ndarray:
    psi = local_states[0]
    for state in local_states[1:]:
        psi = np.multiply.outer(psi, state)
    return np.asarray(psi, dtype=np.complex128)


def make_initial_condition(
    kind: str,
    n_site: int,
    n_cut: int,
    omega: float,
    E_J: float,
    kappa: float,
    phi0: float,
    pi0: float,
    site: int,
    use_offsets: bool,
):
    if n_site < 1:
        raise ValueError("n_site must be positive")
    if not 0 <= site < n_site:
        raise ValueError(f"site must lie in [0, {n_site - 1}]")

    phis = np.zeros(n_site, dtype=float)
    pis = np.zeros(n_site, dtype=float)
    info: dict[str, float | str | bool] = {"initial_condition": kind}

    if kind == "vacuum":
        pass
    elif kind == "coherent_single":
        phis[site] = phi0
        pis[site] = pi0
    elif kind == "coherent_uniform":
        phis[:] = phi0
        pis[:] = pi0
    elif kind == "kink":
        if E_J <= 0 or kappa <= 0:
            raise ValueError("kink initialization requires E_J > 0 and kappa > 0")
        width = np.sqrt(kappa / E_J)
        x = np.arange(n_site, dtype=float) - 0.5 * (n_site - 1)
        phis = 4.0 * np.arctan(np.exp(x / width))
        pis[:] = pi0
        info.update(
            kink_width_sites=float(width),
            sites_per_width=float(n_site / width),
        )
    else:
        raise ValueError(f"unknown initial condition: {kind}")

    offsets = phis.copy() if use_offsets else np.zeros(n_site, dtype=float)
    local_phi = phis - offsets
    local_states = []
    for ph, mom in zip(local_phi, pis):
        alpha = np.sqrt(omega / 2.0) * ph + 1j * mom / np.sqrt(2.0 * omega)
        local_states.append(_coherent_state(alpha, n_cut))
    psi = _product_state(local_states)
    info["use_offsets"] = bool(use_offsets)
    info["initial_norm"] = float(np.linalg.norm(psi.reshape(-1)))
    return psi, phis, pis, offsets, info


def _taylor_one_minus_cos(x: np.ndarray, degree: int) -> np.ndarray:
    if degree < 2:
        return np.zeros_like(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    for power in range(2, degree + 1, 2):
        out += ((-1) ** (power // 2 + 1)) * x**power / math.factorial(power)
    return out


def onsite_vals_with_offsets(
    kind: str,
    lam_phi: np.ndarray,
    E_J: float,
    n_site: int,
    offsets: np.ndarray,
    degree: int | None = None,
    phi_max: float | None = None,
) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=float)
    if offsets.shape != (n_site,):
        raise ValueError("offsets must have shape (n_site,)")
    x = offsets[:, None] + np.asarray(lam_phi, dtype=float)[None, :]
    if kind == "native":
        vals = E_J * (1.0 - np.cos(x))
    elif kind == "taylor":
        if degree is None:
            raise ValueError("Taylor approximation requires degree")
        vals = E_J * _taylor_one_minus_cos(x, degree)
    elif kind == "chebyshev":
        if degree is None or phi_max is None or phi_max <= 0:
            raise ValueError("Chebyshev approximation requires degree and phi_max > 0")
        approximant = Chebyshev.interpolate(
            lambda z: E_J * (1.0 - np.cos(z)),
            deg=degree,
            domain=[-phi_max, phi_max],
        )
        vals = approximant(x)
    else:
        raise ValueError(f"unknown onsite potential: {kind}")
    return np.asarray(vals, dtype=float)


def _phase_grid(
    lam_phi: np.ndarray,
    onsite: np.ndarray,
    offsets: np.ndarray,
    kappa: float,
    pbc: bool,
) -> np.ndarray:
    n_site, n_cut = onsite.shape
    shape = (n_cut,) * n_site
    potential = np.zeros(shape, dtype=np.float64)
    fields = []
    for j in range(n_site):
        axis_shape = [1] * n_site
        axis_shape[j] = n_cut
        physical_phi = (lam_phi + offsets[j]).reshape(axis_shape)
        fields.append(physical_phi)
        potential += onsite[j].reshape(axis_shape)

    bonds = [(j, j + 1) for j in range(n_site - 1)]
    if pbc and n_site > 2:
        bonds.append((n_site - 1, 0))
    for left, right in bonds:
        potential += 0.5 * kappa * (fields[right] - fields[left]) ** 2
    return potential


class SplitOperator:
    """Second-order split evolution for kinetic plus phase-diagonal potential."""

    def __init__(
        self,
        basis: ModeBasis,
        n_site: int,
        kappa: float,
        onsite: np.ndarray,
        pbc: bool,
        offsets: np.ndarray | None = None,
    ):
        self.basis = basis
        self.n_site = int(n_site)
        self.kappa = float(kappa)
        self.onsite = np.asarray(onsite, dtype=float)
        self.pbc = bool(pbc)
        self.offsets = (
            np.zeros(self.n_site, dtype=float)
            if offsets is None
            else np.asarray(offsets, dtype=float)
        )
        if self.onsite.shape != (self.n_site, basis.n_cut):
            raise ValueError("onsite has incompatible shape")
        self.potential = _phase_grid(
            basis.lam_phi, self.onsite, self.offsets, self.kappa, self.pbc
        )
        self._cached_dt: float | None = None
        self._kin_half: np.ndarray | None = None
        self._pot_phase: np.ndarray | None = None

    def _prepare(self, dt: float) -> None:
        if self._cached_dt == dt:
            return
        self._kin_half = expm(-0.5j * dt * self.basis.kinetic)
        self._pot_phase = np.exp(-1j * dt * self.potential)
        self._cached_dt = float(dt)

    def _to_phase(self, psi: np.ndarray) -> np.ndarray:
        out = psi
        transform = self.basis.U_phi.conj().T
        for axis in range(self.n_site):
            out = _apply_axis(transform, out, axis)
        return out

    def _from_phase(self, psi: np.ndarray) -> np.ndarray:
        out = psi
        for axis in range(self.n_site):
            out = _apply_axis(self.basis.U_phi, out, axis)
        return out

    def step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        self._prepare(dt)
        assert self._kin_half is not None and self._pot_phase is not None
        out = psi
        for axis in range(self.n_site):
            out = _apply_axis(self._kin_half, out, axis)
        out = self._to_phase(out)
        out *= self._pot_phase
        out = self._from_phase(out)
        for axis in range(self.n_site):
            out = _apply_axis(self._kin_half, out, axis)
        return out

    def evolve(
        self,
        psi0: np.ndarray,
        dt: float,
        n_steps: int,
        stride: int | None = None,
        observer=None,
    ):
        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative")
        psi = np.asarray(psi0, dtype=np.complex128).copy()
        stride = n_steps + 1 if stride is None else max(1, int(stride))
        if observer is not None:
            observer(0.0, psi)
        for step in range(1, n_steps + 1):
            psi = self.step(psi, dt)
            if observer is not None and (step % stride == 0 or step == n_steps):
                observer(step * dt, psi)
        norm = np.linalg.norm(psi.reshape(-1))
        if norm == 0 or not np.isfinite(norm):
            raise FloatingPointError("evolution produced an invalid state")
        psi /= norm
        return psi, n_steps * dt


class HamiltonianApply:
    """Matrix-free application of the unsplit lattice Hamiltonian."""

    def __init__(
        self,
        basis: ModeBasis,
        n_site: int,
        kappa: float,
        onsite: np.ndarray,
        pbc: bool,
        offsets: np.ndarray | None = None,
    ):
        self.basis = basis
        self.n_site = int(n_site)
        self.shape = (basis.n_cut,) * self.n_site
        self.size = basis.n_cut**self.n_site
        self.offsets = (
            np.zeros(self.n_site, dtype=float)
            if offsets is None
            else np.asarray(offsets, dtype=float)
        )
        self.potential = _phase_grid(
            basis.lam_phi,
            np.asarray(onsite, dtype=float),
            self.offsets,
            float(kappa),
            bool(pbc),
        )

    def _to_phase(self, psi: np.ndarray) -> np.ndarray:
        out = psi
        for axis in range(self.n_site):
            out = _apply_axis(self.basis.U_phi.conj().T, out, axis)
        return out

    def _from_phase(self, psi: np.ndarray) -> np.ndarray:
        out = psi
        for axis in range(self.n_site):
            out = _apply_axis(self.basis.U_phi, out, axis)
        return out

    def __call__(self, psi: np.ndarray) -> np.ndarray:
        tensor = np.asarray(psi, dtype=np.complex128).reshape(self.shape)
        out = np.zeros_like(tensor)
        for axis in range(self.n_site):
            out += _apply_axis(self.basis.kinetic, tensor, axis)
        phase = self._to_phase(tensor)
        phase *= self.potential
        out += self._from_phase(phase)
        return out


def _arnoldi_step(H: HamiltonianApply, psi: np.ndarray, dt: float, m: int):
    shape = psi.shape
    vec = psi.reshape(-1)
    beta = np.linalg.norm(vec)
    if beta == 0:
        raise ValueError("zero initial vector")
    n = vec.size
    m = max(2, min(int(m), n))
    V = np.empty((m + 1, n), dtype=np.complex128)
    hess = np.zeros((m + 1, m), dtype=np.complex128)
    V[0] = vec / beta
    k = m
    for j in range(m):
        w = H(V[j].reshape(shape)).reshape(-1)
        # Two-pass modified Gram-Schmidt is stable enough for the small Krylov
        # dimensions used by these validation runs.
        for _ in range(2):
            coeff = V[: j + 1].conj() @ w
            hess[: j + 1, j] += coeff
            w -= coeff @ V[: j + 1]
        hess[j + 1, j] = np.linalg.norm(w)
        if hess[j + 1, j] < 1e-14:
            k = j + 1
            break
        V[j + 1] = w / hess[j + 1, j]
    small = hess[:k, :k]
    coeff = expm(-1j * dt * small)[:, 0] * beta
    out = coeff @ V[:k]
    err = float(abs(hess[k, k - 1] * coeff[-1])) if k == m else 0.0
    return out.reshape(shape), err


def krylov_evolve(
    H: HamiltonianApply,
    psi0: np.ndarray,
    dt: float,
    n_steps: int,
    krylov_m: int,
    stride: int | None = None,
    observer=None,
):
    psi = np.asarray(psi0, dtype=np.complex128).copy()
    stride = n_steps + 1 if stride is None else max(1, int(stride))
    if observer is not None:
        observer(0.0, psi)
    max_err = 0.0
    for step in range(1, n_steps + 1):
        psi, err = _arnoldi_step(H, psi, dt, krylov_m)
        max_err = max(max_err, err)
        norm = np.linalg.norm(psi.reshape(-1))
        if norm == 0 or not np.isfinite(norm):
            raise FloatingPointError("Krylov evolution produced an invalid state")
        psi /= norm
        if observer is not None and (step % stride == 0 or step == n_steps):
            observer(step * dt, psi)
    return psi, max_err


def classical_sine_gordon(
    phis0: np.ndarray,
    pis0: np.ndarray,
    E_J: float,
    kappa: float,
    times: np.ndarray,
    pbc: bool,
    E_J_eff: float | None = None,
    lambda2: float = 0.,
    lambda2_eff: float | None = None,
):
    phis0 = np.asarray(phis0, dtype=float)
    pis0 = np.asarray(pis0, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    n_site = phis0.size
    coupling_pairs = [(j, j + 1) for j in range(n_site - 1)]
    if pbc and n_site > 2:
        coupling_pairs.append((n_site - 1, 0))
    strength = float(E_J if E_J_eff is None else E_J_eff)
    strength2 = float(lambda2 if lambda2_eff is None else lambda2_eff)

    def rhs(_t, y):
        phi = y[:n_site]
        pi = y[n_site:]
        force = -strength * np.sin(phi) - 2 * strength2 * np.sin(2 * phi)
        for left, right in coupling_pairs:
            delta = phi[right] - phi[left]
            force[left] += kappa * delta
            force[right] -= kappa * delta
        return np.concatenate((pi, force))

    y0 = np.concatenate((phis0, pis0))
    if times[-1] == times[0]:
        y = y0[:, None]
    else:
        sol = solve_ivp(
            rhs,
            (float(times[0]), float(times[-1])),
            y0,
            t_eval=times,
            method="DOP853",
            rtol=2e-10,
            atol=2e-12,
        )
        if not sol.success:
            raise RuntimeError(f"classical integration failed: {sol.message}")
        y = sol.y
    return y[:n_site].T, y[n_site:].T


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json_safe(value):
    """Recursively replace non-finite floating values by strict JSON null."""
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


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def _write_csv_groups(outdir: Path, tag: str, arrays: dict[str, np.ndarray]) -> None:
    groups: dict[int, list[tuple[str, np.ndarray]]] = {}
    for name, value in arrays.items():
        arr = np.asarray(value)
        if arr.ndim in (1, 2) and arr.shape[0] > 0 and arr.dtype != object:
            groups.setdefault(arr.shape[0], []).append((name, arr))
    for length, items in groups.items():
        columns: list[tuple[str, np.ndarray]] = []
        for name, arr in items:
            if arr.ndim == 1:
                columns.append((name, arr))
            else:
                for index in range(arr.shape[1]):
                    columns.append((f"{name}_{index}", arr[:, index]))
        if not columns:
            continue
        suffix = "" if len(groups) == 1 else f"_n{length}"
        path = outdir / f"{tag}{suffix}.csv"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=outdir, delete=False
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([name for name, _ in columns])
            writer.writerows(zip(*(col for _, col in columns)))
            temp_name = handle.name
        os.replace(temp_name, path)


def save_run(
    outdir: str | os.PathLike,
    tag: str,
    params: dict,
    write_csv: bool,
    arrays: dict,
) -> None:
    destination = Path(outdir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    clean_tag = "".join(c if c.isalnum() or c in "._-" else "_" for c in tag)
    if not clean_tag:
        raise ValueError("tag contains no usable characters")

    npz_path = destination / f"{clean_tag}.npz"
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".npz", delete=False) as f:
        temp_npz = f.name
    try:
        np.savez_compressed(temp_npz, **arrays)
        os.replace(temp_npz, npz_path)
    finally:
        if os.path.exists(temp_npz):
            os.unlink(temp_npz)

    metadata = dict(params)
    metadata["result_npz"] = str(npz_path)
    _atomic_json(destination / f"{clean_tag}.json", metadata)
    if write_csv:
        _write_csv_groups(destination, clean_tag, arrays)
    print(f"[save] wrote {npz_path}", flush=True)
