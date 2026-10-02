"""Onsite curvature averaged along the native, target, and retuned double sine-Gordon trajectories.

Two open sites, E_J = 1, kappa = 1, lambda2 = 0.3, omega_b = sqrt(3), T = 5.2/sqrt(3), product
coherent preparation <phi> = (1, 0), finite spectral convention at local cutoff 128. Each lattice is
evolved with its own Hamiltonian (original native E_J = 1, target with lambda2, retuned E_J' = E_J + 4
lambda2 without lambda2) on a 129-point grid, and the expectation of the onsite curvature V''(phi_j)
of that Hamiltonian is averaged over the grid and both sites. Writes data/dsg_curvature.json.

Usage, from the repository root:
    python double_sine_gordon_code/dsg_curvature.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common_code.run_pilot import _sg_initial  # noqa: E402
from sine_gordon_code import native_correction as nc  # noqa: E402
from double_sine_gordon_code.run_snapd_end_to_end import build_two_site_problem  # noqa: E402

E_J, KAPPA, LAM2, OMEGA_B, T, N_CUT = (
    1.0,
    1.0,
    0.3,
    np.sqrt(3.0),
    5.2 / np.sqrt(3.0),
    128,
)
LATTICES = {
    "native": (E_J, 0.0),
    "target": (E_J, LAM2),
    "retuned": (E_J + 4 * LAM2, 0.0),
}


def main():
    out = {}
    for name, (ej, lam) in LATTICES.items():
        problem = build_two_site_problem(N_CUT, OMEGA_B, ej, KAPPA, lam)
        basis = problem["basis"]
        psi0 = _sg_initial(basis, 1.0, 0.0)
        states = expm_multiply(
            -1j * problem["H_target"],
            psi0.reshape(-1),
            start=0.0,
            stop=T,
            num=129,
            endpoint=True,
        )
        cos1, cos2, c12, s2 = [], [], [], []
        for state in states:
            amplitudes = (
                basis.U_phi.conj().T @ state.reshape(N_CUT, N_CUT) @ basis.U_phi.conj()
            )
            prob = np.abs(amplitudes) ** 2
            marginals = (prob.sum(axis=1), prob.sum(axis=0))
            cos1.append([m @ np.cos(basis.lam_phi) for m in marginals])
            cos2.append([m @ np.cos(2 * basis.lam_phi) for m in marginals])
            obs = nc.two_site_observables(state.reshape(N_CUT, N_CUT), basis)
            c12.append(abs(obs["C12_real"] + 1j * obs["C12_imag"]))
            s2.append(obs["S2"])
        cos1, cos2 = np.array(cos1), np.array(cos2)
        curvature = ej * cos1 + 4 * lam * cos2
        # This is <V''(phi_j)> for each lattice's own evolving state, not
        # V''(<phi_j>) or a normal-mode frequency. mean() weights all stored
        # endpoints equally; it is a sampled average rather than time quadrature.
        out[name] = dict(
            E_J=ej,
            lambda2=lam,
            mean_curvature=float(curvature.mean()),
            mean_curvature_sites=curvature.mean(axis=0).tolist(),
            min_curvature=float(curvature.min()),
            max_abs_C12=float(max(c12)),
            max_S2=float(max(s2)),
            mean_cos2phi_site1=float(cos2[:, 0].mean()),
            min_cos2phi_site1=float(cos2[:, 0].min()),
        )
        print(
            name,
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in out[name].items()
            },
            flush=True,
        )
    out["description"] = __doc__.split("\n\n")[0]
    out["cutoff"] = N_CUT
    (
        Path(__file__).resolve().parents[1]
        / "double_sine_gordon_data"
        / "dsg_curvature.json"
    ).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
