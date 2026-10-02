"""Double sine-Gordon circuits on an open chain of N sites.

General-N version of the two-site comparison in dsg_step_scan.py. The chain has E_J = 1, kappa = 1,
omega_b = sqrt(3), T = 5.2/sqrt(3), and the finite spectral convention of the repository (one
truncated mode per site, functions of phi evaluated through the eigenbasis of the truncated phi).
Operators act axis by axis, so no N-site matrix is ever formed, and exp(-i tau H) is applied by an
adaptive Lanczos propagator with full reorthogonalization and a posteriori error control. The initial
state is the product coherent state with <phi> = (phi0, 0, ..., 0) and <Pi> = (pi0, 0, ..., 0).

Routes and order codes are those of dsg_step_scan.py:
  hybrid       H_SG | Delta H with trigonometric gates for Delta H
  kin_pot      H_kin | (H_bond + V_J + Delta H)
  gauss_block  (H_kin + H_bond) | (V_J + Delta H), with the canonical N-mode Gaussian step written as
               exp(-i phi^T A phi) exp(-i Pi^T B Pi) exp(-i phi^T A phi), A and B from the normal
               modes of the bond Laplacian
Order code 1 or 2 uses first- or second-order trigonometric blocks for every harmonic, and code 3
uses second order for the Josephson harmonic and first order for the lambda2 harmonic.

Usage, from the repository root:
    python double_sine_gordon_code/dsg_chain.py references --sites 3 --cuts 96,128 --r 5,6,7
    python double_sine_gordon_code/dsg_chain.py scan --sites 3 --cuts 96,128 --route hybrid --orders 1 --r 5,6,7
    python double_sine_gordon_code/dsg_chain.py merge --sites 3 --cuts 96,128
Fresh Gaussian parts and merge outputs use a _merged suffix. The merge command excludes
unmerged Gaussian archives. References are cached in DSG_CHAIN_CACHE (default /tmp/dsg_chain_cache).
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, expm_multiply

LANCZOS_TOL = 1e-11

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sine_gordon_code import native_correction as nc  # noqa: E402
from sine_gordon_code import sglib as sg  # noqa: E402
from double_sine_gordon_code.gaussian_sequence import merged_gaussian_readouts  # noqa: E402

E_J, KAPPA, OMEGA_B = 1.0, 1.0, np.sqrt(3.0)
T = 5.2 / np.sqrt(3.0)
EPS = np.sqrt(1e-3)
ORDER_CODES = {1: (1, 1), 2: (2, 2), 3: (2, 1)}
CACHE = Path(os.environ.get("DSG_CHAIN_CACHE", "/tmp/dsg_chain_cache"))
DATA = Path(__file__).resolve().parents[1] / "double_sine_gordon_data"


def distance(a, b):
    a, b = a.ravel(), b.ravel()
    overlap = np.vdot(a, b)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * abs(overlap))))


def lanczos_propagate(apply, v, tau, tol=None, m_max=40):
    """exp(-i tau H) v for Hermitian H, by adaptive Krylov substeps with full reorthogonalization."""
    tol = LANCZOS_TOL if tol is None else tol
    out = np.asarray(v, dtype=complex).ravel().copy()
    if np.linalg.norm(out) == 0.0 or tau == 0.0:
        return out, 0
    t_left, dt, matvecs = tau, tau, 0
    while t_left > 1e-15 * abs(tau):
        dt = min(dt, t_left)
        beta0 = np.linalg.norm(out)
        basis = [out / beta0]
        alpha, beta = [], []
        converged = False
        for j in range(m_max):
            w = apply(basis[j])
            matvecs += 1
            a = np.vdot(basis[j], w).real
            w = w - a * basis[j] - (beta[-1] * basis[j - 1] if j > 0 else 0.0)
            for q in basis:
                w -= np.vdot(q, w) * q
            b = np.linalg.norm(w)
            alpha.append(a)
            tri = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
            coeff = expm(-1j * dt * tri)[:, 0]
            # H Q_m=Q_m T_m+b*q_next*e_m^T: b*|coeff[-1]| is the
            # endpoint Schrodinger residual for a normalized starting vector.
            # The acceptance rule uses it as a local indicator; it does not
            # integrate that residual to certify the accumulated state error.
            if b * abs(coeff[-1]) < tol * dt / abs(tau) or b < 1e-14:
                converged = True
                break
            beta.append(b)
            basis.append(w / b)
        if not converged:
            dt *= 0.5
            continue
        out = beta0 * sum(c * q for c, q in zip(coeff, basis[: len(coeff)]))
        t_left -= dt
        dt = min(1.25 * dt, t_left) if t_left > 0 else dt
    return out, matvecs


class Chain:
    """Operators of the N-site double sine-Gordon chain at one local cutoff."""

    def __init__(self, sites, n, lam2, prep=(1.0, 0.0)):
        self.N, self.n, self.lam2 = sites, n, lam2
        b = sg.build_mode_basis(n, OMEGA_B)
        self.basis = b
        U, lam = b.U_phi, b.lam_phi
        fun = lambda f: (U * f(lam)) @ U.conj().T  # noqa: E731
        self.phi = b.phi
        self.lam_phi, self.U_phi = lam, U
        self.lam_pi, self.U_pi = np.linalg.eigh(b.pi)
        self.bonds = [(j, j + 1) for j in range(sites - 1)]
        degree = [sum(j in bond for bond in self.bonds) for j in range(sites)]
        # Each bond contributes kappa*phi_j^2/2 at both endpoints. The cross
        # term -kappa*phi_i*phi_j is applied separately in apply_h.
        self.native_local = [
            b.kinetic
            + fun(lambda x, d=d: E_J * (1 - np.cos(x)) + 0.5 * KAPPA * d * x**2)
            for d in degree
        ]
        self.corr_local = fun(lambda x: lam2 * (1 - np.cos(2 * x)))
        self.target_local = [m + self.corr_local for m in self.native_local]
        laplacian = np.zeros((sites, sites))
        for i, j in self.bonds:
            laplacian[i, i] += 1
            laplacian[j, j] += 1
            laplacian[i, j] -= 1
            laplacian[j, i] -= 1
        self.stiffness = KAPPA * laplacian
        psi = sg.make_initial_condition(
            "coherent_single", sites, n, OMEGA_B, 1.0, 1.0, prep[0], prep[1], 0, False
        )[0]
        self.psi0 = np.asarray(psi, dtype=complex).reshape((n,) * sites)
        self.psi0 /= np.linalg.norm(self.psi0)
        self.vertex = (U * np.exp(1j * lam)) @ U.conj().T

    # ------------------------------------------------------------------ Hamiltonians
    def apply_h(self, state, local):
        out = np.zeros(state.shape, dtype=complex)
        for j in range(self.N):
            out += sg._apply_axis(local[j], state, j)
        for i, j in self.bonds:
            out -= KAPPA * sg._apply_axis(
                self.phi, sg._apply_axis(self.phi, state, i), j
            )
        return out

    def operator(self, local, columns):
        shape = (self.n,) * self.N + ((columns,) if columns > 1 else ())
        dim = self.n**self.N

        def mv(x):
            x = np.asarray(x)
            k = x.shape[1] if x.ndim == 2 else 1
            y = self.apply_h(
                x.reshape((self.n,) * self.N + ((k,) if k > 1 else ())), local
            )
            return y.reshape(dim, k) if x.ndim == 2 else y.reshape(dim)

        trace = self.n ** (self.N - 1) * sum(
            np.trace(m).real for m in local
        ) - KAPPA * sum(
            np.trace(self.phi).real ** 2 * self.n ** (self.N - 2) for _ in self.bonds
        )
        op = LinearOperator(
            (dim, dim),
            matvec=mv,
            rmatvec=mv,
            matmat=mv,
            rmatmat=mv,
            dtype=np.complex128,
        )
        return op, trace, shape

    def propagate(self, state, tau, local, tol=None):
        """exp(-i tau H) on a state with an optional trailing qubit axis (Lanczos, column by column)."""
        system = (self.n,) * self.N

        def apply(x):
            return self.apply_h(x.reshape(system), local).reshape(-1)

        if state.ndim == self.N + 1:
            out = np.empty(state.shape, complex)
            for c in range(state.shape[-1]):
                out[..., c] = lanczos_propagate(apply, state[..., c], tau, tol)[
                    0
                ].reshape(system)
            return out
        return lanczos_propagate(apply, state, tau, tol)[0].reshape(system)

    def propagate_expm(self, state, tau, local):
        """exp(-i tau H) by scipy expm_multiply, used only to check the Lanczos propagator."""
        columns = state.shape[-1] if state.ndim == self.N + 1 else 1
        op, trace, shape = self.operator(local, columns)
        flat = (
            state.reshape(self.n**self.N, columns) if columns > 1 else state.reshape(-1)
        )
        out = expm_multiply(-1j * tau * op, flat, traceA=-1j * tau * trace)
        return np.asarray(out).reshape(state.shape)

    # ------------------------------------------------------------------ observables
    def observables(self, state):
        """Connected vertex correlation of sites 1, 2 and Renyi-2 entropy of site 1 (qubit traced out)."""
        v1 = sg._apply_axis(self.vertex, state, 0)
        v2 = sg._apply_axis(self.vertex.conj().T, state, 1)
        pair = sg._apply_axis(self.vertex, v2, 0)
        c12 = np.vdot(state, pair) - np.vdot(state, v1) * np.vdot(state, v2)
        matrix = state.reshape(self.n, -1)
        rho1 = matrix @ matrix.conj().T
        return complex(c12), float(-np.log(np.vdot(rho1, rho1).real))

    def boundary(self, state, edge=4):
        """Population with at least one mode in its four highest levels."""
        # Complement of the all-modes-interior box counts simultaneous boundary
        # occupations once. Small population is a truncation diagnostic, not a
        # bound on the difference from an infinite-dimensional evolution.
        prob = np.abs(state) ** 2
        if state.ndim == self.N + 1:
            prob = prob.sum(axis=-1)
        return float(prob.sum() - prob[(slice(0, self.n - edge),) * self.N].sum())

    # ------------------------------------------------------------------ circuit pieces
    def diag_phase(self, state, U, phases):
        rotated = state
        for j in range(self.N):
            rotated = sg._apply_axis(U.conj().T, rotated, j)
        rotated = phases.reshape(phases.shape + (1,) * (state.ndim - self.N)) * rotated
        for j in range(self.N):
            rotated = sg._apply_axis(U, rotated, j)
        return rotated

    def grid(self, values):
        return np.meshgrid(*([values] * self.N), indexing="ij")

    def quadratic_generator(self, values, matrix):
        """x^T M x on the product grid of eigenvalues x_j."""
        g = self.grid(values)
        return sum(
            matrix[i, j] * g[i] * g[j] for i in range(self.N) for j in range(self.N)
        )

    def gaussian_generators(self, tau):
        w2, modes = np.linalg.eigh(self.stiffness)
        w = np.sqrt(np.clip(w2, 0.0, None))
        # The open chain has a free center-of-mass mode: its position shear
        # vanishes and sin(w*tau)/(2*w) has the limit tau/2 at w=0.
        a = np.where(w > 1e-12, 0.5 * w * np.tan(0.5 * w * tau), 0.0)
        bb = np.where(
            w > 1e-12, np.sin(w * tau) / (2 * np.where(w > 1e-12, w, 1.0)), 0.5 * tau
        )
        A = modes @ np.diag(a) @ modes.T
        B = modes @ np.diag(bb) @ modes.T
        return (
            self.quadratic_generator(self.lam_phi, A),
            self.quadratic_generator(self.lam_pi, B),
            A,
            B,
        )

    def gaussian_factors(self, tau):
        q_phi, q_pi, A, B = self.gaussian_generators(tau)
        return np.exp(-1j * q_phi), np.exp(-1j * q_pi), A, B

    def gaussian_readouts(self, initial, h, r, order, s, observe):
        """Complete readout circuits with the merged intervals counted by counts()."""
        blocks = [
            nc.trig_block(self.phi, m, h * coefficient, o, s)
            for m, coefficient, o in zip((1, 2), (E_J, self.lam2), ORDER_CODES[order])
        ]

        def potential(state, checkpoint):
            for site in range(self.N):
                for block in blocks:
                    state = nc.apply_joint_local(block, state, site)
                    checkpoint(state)
            return state

        return merged_gaussian_readouts(
            initial,
            r,
            self.gaussian_generators(h / 2)[:2],
            self.gaussian_generators(h)[:2],
            lambda state, q: self.diag_phase(state, self.U_phi, np.exp(-1j * q)),
            lambda state, q: self.diag_phase(state, self.U_pi, np.exp(-1j * q)),
            potential,
            observe,
        )

    def make_step(self, route, h, order, s, observe):
        """One outer step; Gaussian half steps reproduce the archived unmerged sequence."""
        order_j, order_2 = ORDER_CODES[order]
        blocks = {}
        if route != "hybrid":
            blocks[1] = nc.trig_block(self.phi, 1, h * E_J, order_j, s)
        blocks[2] = nc.trig_block(self.phi, 2, h * self.lam2, order_2, s)

        def potential(state):
            for site in range(self.N):
                for m in (1, 2):
                    if m in blocks:
                        state = nc.apply_joint_local(blocks[m], state, site)
                        observe(state)
            return state

        if route == "hybrid":
            if order == 3:
                raise ValueError("order code 3 does not apply to the hybrid circuit")

            def half(state):
                return self.propagate(state, h / 2, self.native_local)
        elif route == "gauss_block":
            gphi, gpi, _, _ = self.gaussian_factors(h / 2)

            def half(state):
                return self.diag_phase(
                    self.diag_phase(
                        self.diag_phase(state, self.U_phi, gphi), self.U_pi, gpi
                    ),
                    self.U_phi,
                    gphi,
                )
        elif route == "kin_pot":
            kin_half = expm(-0.5j * h * self.basis.kinetic)
            g = self.grid(self.lam_phi)
            bond_phase = np.exp(
                -0.25j * h * KAPPA * sum((g[i] - g[j]) ** 2 for i, j in self.bonds)
            )

            def half(state):
                for j in range(self.N):
                    state = sg._apply_axis(kin_half, state, j)
                return self.diag_phase(state, self.U_phi, bond_phase)

            def half_back(state):
                state = self.diag_phase(state, self.U_phi, bond_phase)
                for j in range(self.N):
                    state = sg._apply_axis(kin_half, state, j)
                return state
        else:
            raise ValueError(route)
        back = half_back if route == "kin_pot" else half

        def step(state):
            out = half(state)
            observe(out)
            out = potential(out)
            out = back(out)
            observe(out)
            return out

        return step

    def counts(self, route, r, order, s):
        h = T / r
        coefficients = (0.0, self.lam2) if route == "hybrid" else (E_J, self.lam2)
        orders = ORDER_CODES[order]
        pulses = []
        for site in range(self.N):
            for harmonic, (coefficient, o) in enumerate(zip(coefficients, orders), 1):
                if coefficient == 0:
                    continue
                single = tuple(coefficient if m == harmonic else 0.0 for m in (1, 2))
                for pulse in nc.trig_schedule(1, OMEGA_B, single, h, o, s):
                    pulse = dict(pulse)
                    if pulse["site"] != -1:
                        pulse["site"] = site
                    pulses.append(pulse)
        del (
            pulses
        )  # the qubit basis change is applied once per circuit (trig_counts.py)
        from double_sine_gordon_code import trig_counts

        c = trig_counts.trig_counts(self.N, coefficients, r, orders, s)
        n_bonds = len(self.bonds)
        if route == "kin_pot":
            c["Gaussian"] = self.N * (r + 1) + (self.N + n_bonds) * r
        elif route == "gauss_block":
            _, _, A, B = self.gaussian_factors(h / 2)
            per_factor = lambda M: self.N + sum(
                abs(M[i, j]) > 1e-12
                for i in range(self.N)
                for j in range(i + 1, self.N)
            )  # noqa: E731
            c["Gaussian"] = (r + 1) * per_factor(B) + (r + 2) * per_factor(A)
        else:
            c["Gaussian"] = 0
        return c, (r + 1 if route == "hybrid" else 0)


def tag(sites, lam2, prep):
    return f"N{sites}_lambda{lam2:g}_prep{prep[0]:g}_{prep[1]:g}"


def reference_dir(sites, n, lam2, prep):
    return CACHE / f"target_{tag(sites, lam2, prep)}_n{n}"


def compute_references(sites, cuts, lam2, prep, steps):
    """Exact target states at every outer-step time of every requested r, in one forward sweep."""
    times = sorted({round(k * T / r, 14) for r in steps for k in range(r + 1)})
    for n in cuts:
        folder = reference_dir(sites, n, lam2, prep)
        meta_path = folder / "meta.json"
        if meta_path.exists():
            stored = {round(t, 14) for t in json.loads(meta_path.read_text())["times"]}
            if all(t in stored for t in times):
                continue
        folder.mkdir(parents=True, exist_ok=True)
        chain = Chain(sites, n, lam2, prep)
        started = time.perf_counter()
        states = np.lib.format.open_memmap(
            folder / "states.npy",
            mode="w+",
            dtype=np.complex128,
            shape=(len(times),) + chain.psi0.shape,
        )
        state, t_now, c12, s2 = chain.psi0.copy(), 0.0, [], []
        for i, t in enumerate(times):
            if t > t_now:
                state = chain.propagate(state, t - t_now, chain.target_local)
                state /= np.linalg.norm(state)
                t_now = t
            states[i] = state
            c, s_ = chain.observables(state)
            c12.append([c.real, c.imag])
            s2.append(s_)
        states.flush()
        del states
        check = chain.propagate(chain.psi0, T, chain.target_local, tol=1e-13)
        solver = distance(check / np.linalg.norm(check), state)
        meta_path.write_text(
            json.dumps(
                dict(
                    times=times,
                    C12=c12,
                    S2=s2,
                    solver=solver,
                    lanczos_tol=LANCZOS_TOL,
                    wall_seconds=time.perf_counter() - started,
                )
            )
        )
        print(
            f"reference N={sites} n={n} lambda2={lam2:g} prep={prep} times={len(times)} solver={solver:.1e} "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )


def load_reference(sites, n, lam2, prep, r):
    folder = reference_dir(sites, n, lam2, prep)
    meta = json.loads((folder / "meta.json").read_text())
    index = {round(t, 14): i for i, t in enumerate(meta["times"])}
    rows = [index[round(k * T / r, 14)] for k in range(r + 1)]
    states = np.load(folder / "states.npy", mmap_mode="r")
    return dict(
        states=[np.array(states[i]) for i in rows],
        C12=np.array([complex(*meta["C12"][i]) for i in rows]),
        S2=np.array([meta["S2"][i] for i in rows]),
        solver=float(meta["solver"]),
    )


def run_circuit(chain, route, r, order, s, reference):
    joint = np.zeros(chain.psi0.shape + (2,), complex)
    joint[..., 0] = chain.psi0
    peak = [chain.boundary(joint)]

    def observe(state):
        peak[0] = max(peak[0], chain.boundary(state))

    readouts = None
    if route == "gauss_block":
        readouts = chain.gaussian_readouts(joint, T / r, r, order, s, observe)
        next(readouts)
    else:
        step = chain.make_step(route, T / r, order, s, observe)
    rows, states = [], []
    for k in range(r + 1):
        flat = joint.reshape(-1, 2)
        ref = reference["states"][k].reshape(-1)
        fidelity = float(np.sum(np.abs(ref.conj() @ flat) ** 2))
        c12, s2 = chain.observables(joint)
        rows.append(
            dict(
                bures=float(np.sqrt(max(0.0, 2 - 2 * np.sqrt(min(1.0, fidelity))))),
                ancilla=float(np.sum(np.abs(flat[:, 1]) ** 2)),
                C12=c12,
                S2=s2,
                C12_error=abs(c12 - reference["C12"][k]),
                S2_error=abs(s2 - reference["S2"][k]),
            )
        )
        states.append(joint.copy())
        if k < r:
            joint = next(readouts) if readouts is not None else step(joint)
    return rows, states, peak[0]


def evaluate(sites, cuts, lam2, prep, route, r, order, s):
    low, high = cuts
    refs = {n: load_reference(sites, n, lam2, prep, r) for n in cuts}
    out = {}
    for n in cuts:
        chain = Chain(sites, n, lam2, prep)
        out[n] = run_circuit(chain, route, r, order, s, refs[n])
        if n == low:
            counts, waits = chain.counts(route, r, order, s)
    rows_low, states_low, peak_low = out[low]
    rows_high, states_high, peak_high = out[high]
    pad = 0.0
    for a, b in zip(states_low, states_high):
        embedded = np.zeros(b.shape, complex)
        embedded[(slice(0, low),) * sites + (slice(None),)] = a
        pad = max(pad, distance(embedded, b))
    ref_margin = 0.0
    for a, b in zip(refs[low]["states"], refs[high]["states"]):
        embedded = np.zeros(b.shape, complex)
        embedded[(slice(0, low),) * sites] = a
        ref_margin = max(ref_margin, distance(embedded, b))
    solver = max(refs[low]["solver"], refs[high]["solver"])
    cref = float(np.max(np.abs(refs[low]["C12"] - refs[high]["C12"])))
    sref = float(np.max(np.abs(refs[low]["S2"] - refs[high]["S2"])))
    cp = max(abs(a["C12"] - b["C12"]) for a, b in zip(rows_low, rows_high))
    sp = max(abs(a["S2"] - b["S2"]) for a, b in zip(rows_low, rows_high))
    uc, us = cref + cp, sref + sp
    max_bures = max(row["bures"] for row in rows_low)
    total = max_bures + ref_margin + solver + pad
    boundary = max(peak_low, peak_high)
    c12 = max(row["C12_error"] for row in rows_low)
    s2 = max(row["S2_error"] for row in rows_low)
    acceptance = dict(
        state=bool(total <= EPS and ref_margin <= EPS / 10 and solver <= EPS / 10),
        boundary=bool(boundary <= 1e-6),
        C12=bool(uc <= 1e-3 and c12 + uc <= 0.01),
        S2=bool(us <= 5e-3 and s2 + us <= 0.02),
    )
    return dict(
        sites=sites,
        lambda2=lam2,
        route=route,
        r=r,
        order=order,
        block_orders=list(ORDER_CODES[order]),
        s=s,
        gaussian_sequence="merged_intervals" if route == "gauss_block" else None,
        preparation=list(prep),
        cutoffs=list(cuts),
        primitive_counts={k: int(v) for k, v in counts.items()},
        total_programmable_primitives=int(sum(counts.values())),
        native_waits=waits,
        state_error=total,
        max_bures=max_bures,
        margins=dict(reference=ref_margin, solver=solver, padding=pad, C12=uc, S2=us),
        C12_error=c12 + uc,
        S2_error=s2 + us,
        boundary_population=boundary,
        boundary_low=peak_low,
        boundary_high=peak_high,
        max_ancilla_excited=max(row["ancilla"] for row in rows_low),
        acceptance=acceptance,
        all_criteria=bool(all(acceptance.values())),
    )


def environment():
    return dict(
        python=sys.executable,
        python_version=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        threads=os.environ.get("OPENBLAS_NUM_THREADS"),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("job", choices=["references", "scan", "merge", "time"])
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--cuts", default="96,128")
    parser.add_argument("--lam2", type=float, default=0.3)
    parser.add_argument("--route", choices=["hybrid", "kin_pot", "gauss_block"])
    parser.add_argument("--orders", default="1")
    parser.add_argument("--r", default="")
    parser.add_argument("--s", type=int, default=1)
    parser.add_argument("--prep", default="1,0")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    cuts = tuple(int(x) for x in args.cuts.split(","))
    prep = tuple(float(x) for x in args.prep.split(","))
    steps = [int(x) for x in args.r.split(",") if x]
    parts = DATA / f"dsg_chain_parts_N{args.sites}_cut{cuts[0]}_{cuts[1]}"
    if args.job == "time":
        for n in cuts:
            chain = Chain(args.sites, n, args.lam2, prep)
            started = time.perf_counter()
            chain.propagate(chain.psi0, T / 12, chain.native_local)
            print(
                f"N={args.sites} n={n}: native half step at r=6 took {time.perf_counter() - started:.1f} s",
                flush=True,
            )
        return
    if args.job == "references":
        compute_references(args.sites, cuts, args.lam2, prep, steps)
        return
    if args.job == "scan":
        parts.mkdir(parents=True, exist_ok=True)
        compute_references(args.sites, cuts, args.lam2, prep, steps)
        results = []
        for r in steps:
            for order in [int(x) for x in args.orders.split(",")]:
                started = time.perf_counter()
                row = evaluate(
                    args.sites, cuts, args.lam2, prep, args.route, r, order, args.s
                )
                row["wall_seconds"] = time.perf_counter() - started
                results.append(row)
                acc = (
                    "PASS"
                    if row["all_criteria"]
                    else "fail "
                    + ",".join(k for k, v in row["acceptance"].items() if not v)
                )
                print(
                    f"N={args.sites} lambda2={args.lam2:g} {args.route} r={r} o={order} prims={row['total_programmable_primitives']} "
                    f"state={row['state_error']:.4f} C12={row['C12_error']:.4f} S2={row['S2_error']:.4f} "
                    f"bnd={row['boundary_population']:.2e} ref={row['margins']['reference']:.1e} "
                    f"pad={row['margins']['padding']:.1e} {acc} ({row['wall_seconds']:.0f} s)",
                    flush=True,
                )
                sequence_tag = "_merged" if args.route == "gauss_block" else ""
                name = f"{args.route}_{tag(args.sites, args.lam2, prep)}{args.tag}{sequence_tag}.json"
                (parts / name).write_text(
                    json.dumps(dict(environment=environment(), rows=results), indent=1)
                )
        return
    unique = {}
    for path in sorted(parts.glob("*.json")):
        for row in json.loads(path.read_text())["rows"]:
            if (
                row["route"] == "gauss_block"
                and row.get("gaussian_sequence") != "merged_intervals"
            ):
                continue
            unique[
                (
                    row["route"],
                    row["lambda2"],
                    row["r"],
                    row["order"],
                    tuple(row["preparation"]),
                )
            ] = row
    rows = list(unique.values())
    best = {}
    for row in rows:
        if tuple(row["preparation"]) != (1.0, 0.0) or not row["all_criteria"]:
            continue
        key = (row["route"], row["lambda2"])
        if key not in best or (
            row["total_programmable_primitives"],
            row["state_error"],
        ) < (best[key]["total_programmable_primitives"], best[key]["state_error"]):
            best[key] = row
    summary = dict(
        description=__doc__.split("\n\n")[0],
        sites=args.sites,
        cutoffs=list(cuts),
        rows=rows,
        cheapest_passing=[
            dict(
                route=k[0],
                lambda2=k[1],
                r=v["r"],
                order=v["order"],
                primitives=v["total_programmable_primitives"],
                state_error=v["state_error"],
            )
            for k, v in sorted(best.items())
        ],
        environment=environment(),
    )
    out = DATA / f"dsg_chain_N{args.sites}_cut{cuts[0]}_{cuts[1]}_merged.json"
    out.write_text(json.dumps(summary, indent=1))
    for item in summary["cheapest_passing"]:
        print(item)


if __name__ == "__main__":
    main()
