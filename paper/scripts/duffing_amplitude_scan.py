"""Duffing native-Kerr diagnostics along a fine coherent-amplitude scan.

Reproduces the single-mode quantities quoted in the Duffing section
(maximum native infidelity, uncentered Duhamel bound, native-reference
residual bound, number-diagonal ceiling, state ratio r_psi, and cutoff
convergence) from the repository's operator constructors. The run takes
seconds on one core and writes ../data/duffing_amplitude_scan.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from duffing import duffing_padded_hybrid_validation as df  # noqa: E402
from pra.metrics import native_residual_integral, projective_distances  # noqa: E402

OMEGA0, BETA, T = 1.0, 0.10, 1.0
TIMES = np.linspace(0.0, T, 401)
N_REF, N_CHECK = 80, 64
XIS = np.round(np.arange(0.25, 2.5001, 0.05), 2)


def trajectories(ops, eig, xi):
    psi, _ = df.coherent_state(ops["H_D"].shape[0], xi)
    vals, vecs = eig
    target = vecs @ ((vecs.conj().T @ psi)[:, None] * np.exp(-1j * vals[:, None] * TIMES))
    native = psi[:, None] * np.exp(-1j * np.diag(ops["H_K"]).real[:, None] * TIMES)
    return psi, target, native


def diagnostics(n):
    ops = df.hard_sandwiched_operators(n, OMEGA0, BETA)
    eig = np.linalg.eigh(ops["H_D"])
    residual = ops["H_D"] - ops["H_K"]
    lam_vcr = ops["lambda"] * ops["V_cr"]
    rows = []
    for xi in XIS:
        psi, target, native = trajectories(ops, eig, float(xi))
        dist = projective_distances(target, native)
        infid = dist**2 * (1 - dist**2 / 4)
        bounds = native_residual_integral(residual, native, TIMES)
        ceiling = float(np.sum(np.abs(target[:, -1]) * np.abs(psi)) ** 2)
        # r_psi(t) = sigma(lambda V_cr) / sigma(H_D) on the exact target trajectory
        def sigma(op, states):
            act = op @ states
            mean = np.sum(states.conj() * act, axis=0)
            return np.linalg.norm(act - states * mean[None, :], axis=0)
        r_psi = sigma(lam_vcr, target) / sigma(ops["H_D"], target)
        rows.append(dict(
            xi=float(xi), n0=float(xi**2),
            max_native_infidelity=float(infid.max()),
            duhamel_infidelity_bound=float(bounds["uncentered_integral"][-1] ** 2),
            centered_infidelity_bound=float(bounds["centered_integral"][-1] ** 2),
            diagonal_ceiling_T=ceiling,
            max_r_psi=float(r_psi.max()),
        ))
    return ops, eig, rows


def loglog_slope(x, y):
    x, y = np.log(np.asarray(x)), np.log(np.asarray(y))
    return float(np.polyfit(x, y, 1)[0])


def crossing(xs, ys, level=1.0):
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 < level <= y1:
            return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return None


def main():
    ops, eig, rows = diagnostics(N_REF)
    _, _, rows_check = diagnostics(N_CHECK)
    sel = [r for r in rows if r["n0"] > 1.0]
    fit = dict(
        infidelity_exponent=loglog_slope([r["n0"] for r in sel], [r["max_native_infidelity"] for r in sel]),
        duhamel_exponent=loglog_slope([r["n0"] for r in sel], [r["duhamel_infidelity_bound"] for r in sel]),
        centered_exponent=loglog_slope([r["n0"] for r in sel], [r["centered_infidelity_bound"] for r in sel]),
        points=len(sel),
    )
    xs = [r["xi"] for r in rows]
    unity = dict(
        duhamel_xi=crossing(xs, [r["duhamel_infidelity_bound"] for r in rows]),
        centered_xi=crossing(xs, [r["centered_infidelity_bound"] for r in rows]),
    )
    # cutoff convergence of the exact target at the two benchmark amplitudes
    conv = {}
    for xi in (1.75, 2.0):
        _, ref, _ = trajectories(ops, eig, xi)
        entries = []
        for n in (24, 28, 32, 36, 40, 44, 48, 56, 64, 72):
            o = df.hard_sandwiched_operators(n, OMEGA0, BETA)
            e = np.linalg.eigh(o["H_D"])
            _, tgt, _ = trajectories(o, e, xi)
            emb = np.vstack([tgt, np.zeros((N_REF - n, tgt.shape[1]))])
            d = projective_distances(ref, emb)
            entries.append(dict(N=n, max_infidelity=float((d**2 * (1 - d**2 / 4)).max())))
        conv[str(xi)] = entries
    cutoff_diff = max(abs(a["max_native_infidelity"] - b["max_native_infidelity"]) for a, b in zip(rows, rows_check))
    bench = {str(r["xi"]): r for r in rows if r["xi"] in (1.75, 2.0)}
    out = dict(
        description="Duffing native-Kerr diagnostics, omega0=1, beta=0.1, T=1, 401-point grid, strict physical sandwich",
        N_ref=N_REF, N_check=N_CHECK, K_eff=float(ops["K_eff"]), lambda_=float(ops["lambda"]),
        rows=rows, fit=fit, unity_crossing=unity, benchmark=bench,
        cutoff_convergence=conv, max_infidelity_change_N64_to_N80=cutoff_diff,
        python=sys.executable, numpy=np.__version__,
    )
    path = Path(__file__).resolve().parents[1] / "data" / "duffing_amplitude_scan.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(fit=fit, unity=unity, bench=bench, cutoff_diff=cutoff_diff), indent=1))
    for xi, entries in conv.items():
        print(xi, [(e["N"], f'{e["max_infidelity"]:.2e}') for e in entries])


if __name__ == "__main__":
    main()
