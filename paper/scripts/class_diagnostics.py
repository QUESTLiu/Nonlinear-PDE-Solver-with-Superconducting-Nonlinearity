"""Operator-level class diagnostics (eta_H, C_corr) at declared finite cutoffs.

Duffing: strict physical sandwich from the repository constructor.
Double sine-Gordon: two open sites in the finite spectral convention
(pi_N^2 and cos(m phi_N) from the finite quadratures), E_J=1, lambda2=0.3,
kappa=1, omega_b=sqrt(3). Small dense matrices only; runs in seconds.
Writes ../data/class_diagnostics.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from duffing import duffing_padded_hybrid_validation as df  # noqa: E402


def centered_norm(h):
    e = np.linalg.eigvalsh(h)
    return float((e[-1] - e[0]) / 2)


def spec(h):
    return float(np.linalg.norm(h, 2))


def duffing(n, beta=0.1, omega0=1.0):
    ops = df.hard_sandwiched_operators(n, omega0, beta)
    h_t = np.asarray(ops["H_D"])
    h_n = np.asarray(ops["H_K"])  # includes the scalar E0 I
    dh = h_t - h_n
    return dict(N=n, dH_centered=centered_norm(dh), H_centered=centered_norm(h_t),
                eta_H=centered_norm(dh) / centered_norm(h_t), C_corr=spec(h_n @ dh - dh @ h_n))


def dsg(n, ej=1.0, lam2=0.3, kappa=1.0, omega_b=np.sqrt(3.0)):
    a = np.diag(np.sqrt(np.arange(1, n)), 1).astype(complex)
    ad = a.conj().T
    phi = (a + ad) / np.sqrt(2 * omega_b)
    pi = 1j * np.sqrt(omega_b / 2) * (ad - a)
    w, v = np.linalg.eigh(phi)
    cos1 = (v * np.cos(w)) @ v.conj().T
    cos2 = (v * np.cos(2 * w)) @ v.conj().T
    eye = np.eye(n)
    kin = pi @ pi / 2
    site = kin + ej * (eye - cos1)
    res = lam2 * (eye - cos2)
    k = np.kron
    h_nat = k(site, eye) + k(eye, site) + kappa / 2 * np.linalg.matrix_power(k(phi, eye) - k(eye, phi), 2)
    dh = k(res, eye) + k(eye, res)
    h_t = h_nat + dh
    return dict(N=n, dim=n * n, dH_centered=centered_norm(dh), H_centered=centered_norm(h_t),
                eta_H=centered_norm(dh) / centered_norm(h_t), C_corr=spec(h_nat @ dh - dh @ h_nat))


def main():
    out = dict(duffing=[duffing(n) for n in (48, 64, 80)], double_sine_gordon=[dsg(n) for n in (16, 24, 32, 48)])
    eps = np.sqrt(1e-3)
    out["class_I_threshold_T_times_dH"] = eps / 3
    out["duffing_T"] = 1.0
    out["dsg_T"] = 5.2 / np.sqrt(3)
    (Path(__file__).resolve().parents[1] / "data" / "class_diagnostics.json").write_text(json.dumps(out, indent=1))
    for key in ("duffing", "double_sine_gordon"):
        for r in out[key]:
            print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
