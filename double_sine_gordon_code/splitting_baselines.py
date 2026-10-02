"""Outer splitting error of hybrid and programmable product formulas for the double sine-Gordon lattice.

Two open sites, E_J = 1, kappa = 1, omega_b = sqrt(3), preparation <phi> = (1, 0), <Pi> = (0, 0),
T = 5.2/sqrt(3), finite spectral convention, built with the repository operators of the archived
runs. Every block is applied exactly, so the only error is the outer Strang splitting
A|B = exp(-iA h/2) exp(-iB h) exp(-iA h/2) with h = T/r. Routes:

  hybrid        H_SG | Delta H                                   (native lattice + correction)
  hybrid_ret    H_SG(E_J + 4 lambda2) | retuned correction       (retuned native lattice)
  kin_pot       K | (bond + cosines)                             (kinetic/potential split)
  gauss_cos     (K + bond) | cosines                             (exact two-mode Gaussian block)
  gauss_harm    (K + bond + harmonic part) | anharmonic part      (harmonic reference)

The harmonic part is (E_J + 4 lambda2) phi_j^2 / 2, the curvature of the onsite potential at 0.
The error is the maximum projective distance from the exact same-cutoff target over the outer-step
times. Usage, from the repository root:

    /Users/zhen002/miniconda3/envs/cvdv/bin/python double_sine_gordon_code/splitting_baselines.py
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common_code.run_pilot import _sg_initial  # noqa: E402
from double_sine_gordon_code.run_snapd_end_to_end import build_two_site_problem  # noqa: E402

EJ, KAPPA, OMEGA_B = 1.0, 1.0, np.sqrt(3.0)
T = 5.2 / np.sqrt(3.0)
OUT = Path(__file__).resolve().parents[1] / "double_sine_gordon_data" / "splitting_baselines.json"


def distance(x, y):
    overlap = abs(np.vdot(x.ravel(), y.ravel())) / (np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * overlap)))


def scan(n, lam2, steps, routes, kappa=KAPPA):
    problem = build_two_site_problem(n, OMEGA_B, EJ, kappa, lam2)
    basis = problem["basis"]
    lam, U = basis.lam_phi, basis.U_phi
    ident = eye(n, format="csr")
    psi0 = np.asarray(_sg_initial(basis)).reshape(n, n)
    phi = csr_matrix(basis.phi)
    kin = csr_matrix(basis.kinetic)
    phi2 = phi @ phi
    bond_op = 0.5 * kappa * (kron(phi2, ident) + kron(ident, phi2) - 2 * kron(phi, phi))
    gauss = (kron(kin, ident) + kron(ident, kin) + bond_op).tocsr()
    curvature = EJ + 4 * lam2
    gauss_h = (gauss + 0.5 * curvature * (kron(phi2, ident) + kron(ident, phi2))).tocsr()
    H_native = problem["H_native"]
    H_ret = build_two_site_problem(n, OMEGA_B, EJ + 4 * lam2, kappa, 0.0)["H_native"]
    L1, L2 = np.meshgrid(lam, lam, indexing="ij")
    cosines = EJ * (2 - np.cos(L1) - np.cos(L2)) + lam2 * (2 - np.cos(2 * L1) - np.cos(2 * L2))
    correction = lam2 * (2 - np.cos(2 * L1) - np.cos(2 * L2))
    correction_ret = correction - 4 * lam2 * (2 - np.cos(L1) - np.cos(L2))
    bond = 0.5 * kappa * (L1 - L2) ** 2
    anharmonic = cosines - 0.5 * curvature * (L1 ** 2 + L2 ** 2)

    def diag(p, potential, tau):
        rotated = U.conj().T @ p @ U.conj()
        return U @ (np.exp(-1j * tau * potential) * rotated) @ U.T

    def prop(A, p, tau):
        return expm_multiply(-1j * tau * A, p.ravel()).reshape(n, n)

    results = {name: [] for name in routes}
    for r in steps:
        h = T / r
        kin_half = expm(-0.5j * h * basis.kinetic)
        step = {
            "hybrid": lambda p: prop(H_native, diag(prop(H_native, p, h / 2), correction, h), h / 2),
            "hybrid_ret": lambda p: prop(H_ret, diag(prop(H_ret, p, h / 2), correction_ret, h), h / 2),
            "kin_pot": lambda p: kin_half @ diag(kin_half @ p @ kin_half.T, bond + cosines, h) @ kin_half.T,
            "gauss_cos": lambda p: prop(gauss, diag(prop(gauss, p, h / 2), cosines, h), h / 2),
            "gauss_harm": lambda p: prop(gauss_h, diag(prop(gauss_h, p, h / 2), anharmonic, h), h / 2),
        }
        exact = expm_multiply(-1j * problem["H_target"], psi0.ravel(), start=0.0, stop=T, num=r + 1, endpoint=True)
        for name in routes:
            p, worst = psi0, 0.0
            for j in range(1, r + 1):
                p = step[name](p)
                worst = max(worst, distance(p, exact[j].reshape(n, n)))
            results[name].append(worst)
    return results


def main():
    started = time.perf_counter()
    steps = [5, 7, 10, 14, 20, 28, 40]
    all_routes = ["hybrid", "hybrid_ret", "kin_pot", "gauss_cos", "gauss_harm"]
    record = {
        "description": "maximum projective distance over outer-step times, exact blocks, two open sites, E_J=1, kappa=1",
        "T": T,
        "scan": {"cutoff": 80, "lambda2": 0.3, "r": steps, "errors": scan(80, 0.3, steps, all_routes)},
        "cutoff_check": {"cutoff": 64, "lambda2": 0.3, "r": [10, 20],
                         "errors": scan(64, 0.3, [10, 20], ["hybrid", "kin_pot", "gauss_cos"])},
        "lambda2_check": {"cutoff": 80, "lambda2": 0.15, "r": [10, 20],
                          "errors": scan(80, 0.15, [10, 20], ["hybrid", "kin_pot", "gauss_cos"])},
    }
    record["environment"] = {"python": sys.executable, "python_version": platform.python_version(),
                             "numpy": np.__version__, "scipy": scipy.__version__,
                             "wall_seconds": time.perf_counter() - started}
    OUT.write_text(json.dumps(record, indent=1))
    scan_errors = record["scan"]["errors"]
    for i, r in enumerate(steps):
        hyb = scan_errors["hybrid"][i]
        print(f"r={r:2d} " + " ".join(f"{k}={v[i]:.5f}" for k, v in scan_errors.items())
              + "  ratios " + " ".join(f"{k}/hybrid={v[i] / hyb:.2f}" for k, v in scan_errors.items() if k != "hybrid"))
    for key in ("cutoff_check", "lambda2_check"):
        print(key, json.dumps(record[key]["errors"]))
    print(f"wall {record['environment']['wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()
