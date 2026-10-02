"""Primitive counts of the trigonometric blocks with the qubit basis change moved out of every factor.

The elementary factors of the trigonometric construction are E_+(v) = U e^{ivZ} U^+ and
E_-(v) = U^+ e^{ivZ} U with U = e^{iA (x) X / 2}. The earlier schedule wrote U = R_y(pi/2) CD_Z R_y(-pi/2)
and applied these rotations in every factor. With B = R_y(pi/2) R_z(pi/2), which maps Z to X and Y to Z
under conjugation, and D_A = e^{iA (x) Z / 2} = CD_Z,

    E_+(v) = B D_A R_y(-2v) D_A^+ B^+,    E_-(v) = B D_A^+ R_y(-2v) D_A B^+,

so the products B^+ B between consecutive factors cancel, and B commutes with the native and Gaussian
blocks, which act on the system alone. A circuit therefore applies B^+ once at the start and B once at the
end, and each factor needs two controlled displacements and one R_y before merging. The circuit unitary is
unchanged, so every simulated error stays valid.

This module builds the rewritten pulse schedule, merges it with the rules of the original counting
(sine_gordon_code.native_correction.merge_pulses), checks it against the exact factors on a small cutoff, and
recounts the stored circuits. The old counts are kept in each row under "primitive_counts_conjugated".

Usage, from the repository root (interpreter /Users/zhen002/miniconda3/envs/cvdv/bin/python):
    python double_sine_gordon_code/trig_counts.py verify
    python double_sine_gordon_code/trig_counts.py recount
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sine_gordon_code import native_correction as nc  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "double_sine_gordon_data"
OMEGA_B = np.sqrt(3.0)
ORDER_CODES = {1: (1, 1), 2: (2, 2), 3: (2, 1)}
BASIS = [
    dict(gate="Rz", site=-1, value=np.pi / 2),
    dict(gate="Ry", site=-1, value=np.pi / 2),
]  # B, time order
BASIS_INV = [
    dict(gate="Ry", site=-1, value=-np.pi / 2),
    dict(gate="Rz", site=-1, value=-np.pi / 2),
]  # B^+, time order


def basis_schedule(n_site, coefficients, h, orders, s, omega=OMEGA_B):
    """Time-ordered pulses of one potential step with B^+ ... B around every elementary factor."""
    pulses = []

    def elementary(site, harmonic, v, sign):
        amplitude = harmonic / (2 * np.sqrt(2 * omega))
        pulses.extend(dict(p) for p in BASIS_INV)
        pulses.append(dict(gate="CD_Z", site=site, value=-sign * amplitude))
        pulses.append(dict(gate="Ry", site=-1, value=-2 * v))
        pulses.append(dict(gate="CD_Z", site=site, value=sign * amplitude))
        pulses.extend(dict(p) for p in BASIS)

    for site in range(n_site):
        for harmonic, (coefficient, order) in enumerate(zip(coefficients, orders), 1):
            if coefficient == 0:
                continue
            u = h * coefficient
            for _ in range(s):
                if order == 1:
                    elementary(site, harmonic, u / (2 * s), -1)
                    elementary(site, harmonic, u / (2 * s), 1)
                else:
                    elementary(site, harmonic, u / (4 * s), 1)
                    elementary(site, harmonic, u / (2 * s), -1)
                    elementary(site, harmonic, u / (4 * s), 1)
    return pulses


def trig_counts(n_site, coefficients, r, orders, s=1):
    """Merged counts of r potential steps; B B^+ at the r - 1 step boundaries cancel through the system blocks."""
    merged = nc.merge_pulses(basis_schedule(n_site, coefficients, 1.0 / r, orders, s))
    if [(p["gate"], p["value"]) for p in merged[:2]] != [
        ("Ry", -np.pi / 2),
        ("Rz", -np.pi / 2),
    ] or [(p["gate"], p["value"]) for p in merged[-2:]] != [
        ("Rz", np.pi / 2),
        ("Ry", np.pi / 2),
    ]:
        raise ValueError("a step must start with B^+ and end with B")
    counts = Counter()
    for p in merged:
        counts[p["gate"]] += r
    counts["Ry"] -= 2 * (r - 1)
    counts["Rz"] -= 2 * (r - 1)
    return dict(counts)


def recount(row, n_site):
    """New counts of a stored row; the Gaussian count is unchanged."""
    route, r, order, s = row["route"], row["r"], row["order"], row.get("s", 1)
    lam2 = row.get("lambda2", 0.3)
    coefficients = (0.0, lam2) if route == "hybrid" else (1.0, lam2)
    counts = trig_counts(n_site, coefficients, r, ORDER_CODES[order], s)
    counts["Gaussian"] = row["primitive_counts"].get("Gaussian", 0)
    return counts


# --------------------------------------------------------------------------------------- checks
def _pulse_unitary(pulses, phis, omega=OMEGA_B):
    """Joint unitary of a pulse list on (site modes) x qubit, qubit last, time order left to right."""
    n = phis[0].shape[0]
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)
    eye_sys = np.eye(n ** len(phis))
    total = np.eye(2 * n ** len(phis), dtype=complex)
    for p in pulses:
        if p["gate"] in ("Ry", "Rz"):
            G = Y if p["gate"] == "Ry" else Z
            U = np.kron(eye_sys, expm(-0.5j * p["value"] * G))
        else:
            # CD_Z(alpha) with alpha = i m / (2 sqrt(2 omega)) scaled: exp[i m phi (x) Z / 2] for value = m/(2 sqrt(2 omega))
            m = p["value"] * 2 * np.sqrt(2 * omega)
            ops = [np.eye(n)] * len(phis)
            ops[p["site"]] = phis[p["site"]]
            # Tensor factors follow site order; the qubit is appended last.
            A = ops[0]
            for o in ops[1:]:
                A = np.kron(A, o)
            U = expm(0.5j * m * np.kron(A, Z))
        total = U @ total
    return total


def verify():
    n = 10
    phi = np.diag(np.linspace(-2.0, 2.0, n)) + 0.3 * (
        np.eye(n, k=1) + np.eye(n, k=-1)
    )  # a finite Hermitian quadrature
    worst = 0.0
    for orders in ((1, 1), (2, 2), (2, 1)):
        for s in (1, 2):
            for coefficients in ((0.0, 0.3), (1.0, 0.3)):
                h = 0.37
                pulses = nc.merge_pulses(basis_schedule(1, coefficients, h, orders, s))
                U_new = _pulse_unitary(pulses, [phi])
                # exact product of the factors of the original construction (sine_gordon_code.native_correction.trig_block)
                U_ref = np.eye(2 * n, dtype=complex)
                for harmonic, (c, o) in enumerate(zip(coefficients, orders), 1):
                    if c != 0:
                        U_ref = nc.trig_block(phi, harmonic, h * c, o, s) @ U_ref
                        U_ref = U_ref * np.exp(
                            1j * h * c
                        )  # the scalar phase e^{-iu} of Q is bookkeeping, not a pulse
                worst = max(worst, float(np.abs(U_new - U_ref).max()))
    print(f"rewritten pulses reproduce the exact trigonometric blocks to {worst:.1e}")
    verify_circuits()
    # formula check against the stored counts: new = CD + Rz(old) + Gaussian + 4
    bad = 0
    for path in sorted(DATA.glob("dsg_step_scan_parts*/*.json")):
        for row in json.loads(path.read_text())["rows"]:
            old = row.get("primitive_counts_conjugated", row["primitive_counts"])
            new = recount(dict(row, primitive_counts=old), 2)
            if (
                sum(new.values())
                != old["CD_Z"] + old["Rz"] + old.get("Gaussian", 0) + 4
            ):
                bad += 1
    print("rows whose new count differs from CD + Rz + Gaussian + 4:", bad)


def verify_circuits(seed=7):
    """Whole circuits on 2 and 3 sites: B^+ once, B once, and a system unitary between the outer steps.

    The rewritten pulses, with B B^+ removed at every inner step boundary as trig_counts() assumes, are
    compared with the exact factors of the original construction interleaved with the same system unitary,
    which stands in for the native and Gaussian blocks. Also checks that the pulse count equals trig_counts().
    """
    rng = np.random.default_rng(seed)

    def hermitian(n, scale):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return scale * (a + a.conj().T) / 2

    worst, count_bad, cases = 0.0, 0, 0
    for n_site, n in ((2, 5), (3, 4)):
        phis = [
            np.diag(np.linspace(-2.0, 2.0, n)) + 0.3 * hermitian(n, 1.0)
            for _ in range(n_site)
        ]
        embedded = [
            np.kron(np.kron(np.eye(n**j), phis[j]), np.eye(n ** (n_site - j - 1)))
            for j in range(n_site)
        ]
        system = np.kron(expm(-1j * hermitian(n**n_site, 0.5)), np.eye(2))
        dim = 2 * n**n_site
        for orders in ((1, 1), (2, 2), (2, 1)):
            for s in (1, 2):
                for coefficients in ((0.0, 0.3), (1.0, 0.3)):
                    for r in (1, 3):
                        h = 1.0 / r
                        step = nc.merge_pulses(
                            basis_schedule(n_site, coefficients, h, orders, s)
                        )
                        U_step = np.eye(dim, dtype=complex)
                        for j in range(n_site):
                            for harmonic, (c, o) in enumerate(
                                zip(coefficients, orders), 1
                            ):
                                if c != 0:
                                    U_step = (
                                        nc.trig_block(
                                            embedded[j], harmonic, h * c, o, s
                                        )
                                        * np.exp(1j * h * c)
                                        @ U_step
                                    )
                        U_ref = np.eye(dim, dtype=complex)
                        U_new = np.eye(dim, dtype=complex)
                        n_pulses = 0
                        for k in range(r):
                            pulses = step[
                                (2 if k > 0 else 0) : (
                                    len(step) - 2 if k < r - 1 else len(step)
                                )
                            ]
                            n_pulses += len(pulses)
                            U_ref = U_step @ U_ref
                            U_new = _pulse_unitary(pulses, phis) @ U_new
                            if k < r - 1:
                                U_ref = system @ U_ref
                                U_new = system @ U_new
                        phase = np.vdot(U_ref.ravel(), U_new.ravel())
                        worst = max(
                            worst,
                            float(np.abs(U_new - phase / abs(phase) * U_ref).max()),
                        )
                        count_bad += n_pulses != sum(
                            trig_counts(n_site, coefficients, r, orders, s).values()
                        )
                        cases += 1
    print(
        f"{cases} whole circuits on 2 and 3 sites reproduce the original construction to {worst:.1e} "
        f"(global phase removed); pulse-count mismatches: {count_bad}"
    )


# ------------------------------------------------------------------------------------- recount
def _update(path, n_site, lam2=None):
    data = json.loads(path.read_text())
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else None
    if rows is None:
        return 0
    changed = 0
    for row in rows:
        if row.get("route") not in ("hybrid", "gauss_block", "kin_pot"):
            continue
        if lam2 is not None:
            row.setdefault("lambda2", lam2)
        old = row.get("primitive_counts_conjugated", row["primitive_counts"])
        row["primitive_counts_conjugated"] = old
        row["total_programmable_primitives_conjugated"] = int(sum(old.values()))
        new = recount(dict(row, primitive_counts=old), n_site)
        row["primitive_counts"] = new
        row["total_programmable_primitives"] = int(sum(new.values()))
        changed += 1
    path.write_text(json.dumps(data, indent=1))
    return changed


def recount_all():
    jobs = [(p, 2, 0.3) for p in sorted((DATA / "dsg_step_scan_parts").glob("*.json"))]
    jobs += [
        (p, 2, 0.1)
        for p in sorted((DATA / "dsg_step_scan_parts_lambda0.1").glob("*.json"))
    ]
    jobs += [
        (p, 2, 0.6)
        for p in sorted((DATA / "dsg_step_scan_parts_lambda0.6").glob("*.json"))
    ]
    jobs += [
        (p, 2, None) for p in sorted((DATA / "dsg_common_times_parts").glob("*.json"))
    ]
    jobs += [
        (DATA / "dsg_boundary_check_lambda0.6.json", 2, 0.6),
        (DATA / "dsg_boundary_check_lambda0.6_prep0.75_0.25.json", 2, 0.6),
    ]
    jobs += [
        (p, 2, None)
        for p in sorted((DATA / "dsg_chain_parts_N2_cut128_160").glob("*.json"))
    ]
    jobs += [
        (p, 3, None)
        for p in sorted((DATA / "dsg_chain_parts_N3_cut128_160").glob("*.json"))
    ]
    for path, n_site, lam2 in jobs:
        if path.exists():
            print(path.relative_to(DATA), _update(path, n_site, lam2), "rows")


if __name__ == "__main__":
    {"verify": verify, "recount": recount_all}[sys.argv[1]]()
