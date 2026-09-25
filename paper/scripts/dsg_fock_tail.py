"""Population of the double sine-Gordon target above Fock level 60 of any mode.

Reads the cached exact targets at local cutoff 128 (lambda2 = 0.3, E_J = kappa = 1, preparation
<phi> = (1, 0) or (1, 0, 0)): the two-site target of dsg_step_scan.py on the grid t_k = kT/20
(DSG_SCAN_CACHE) and the three-site target of dsg_chain.py on the union of its outer-step grids
(DSG_CHAIN_CACHE). For every stored time it computes 1 - sum_{n_j <= 60 for all j} |psi_n|^2, the
population with at least one mode above level 60, and writes the maxima to data/dsg_fock_tail.json.

Usage, from the repository root (interpreter /Users/zhen002/miniconda3/envs/cvdv/bin/python):
    python paper/scripts/dsg_fock_tail.py
"""

import json
import os
from pathlib import Path

import numpy as np

LEVEL = 60
DATA = Path(__file__).resolve().parents[1] / "data"
SCAN_CACHE = Path(os.environ.get("DSG_SCAN_CACHE", "/tmp/dsg_scan_cache"))
CHAIN_CACHE = Path(os.environ.get("DSG_CHAIN_CACHE", "/tmp/dsg_chain_cache"))


def tail(state, sites, n):
    prob = np.abs(np.asarray(state).reshape((n,) * sites)) ** 2
    inside = prob[(slice(0, LEVEL + 1),) * sites].sum()
    return float(prob.sum() - inside)


def main():
    out = dict(description=__doc__.split("\n\n")[0], level=LEVEL)
    with np.load(SCAN_CACHE / "target_N128_r20_prep1_0.npz") as a:
        values = [tail(s, 2, 128) for s in a["states"]]
    out["two_sites"] = dict(cutoff=128, times=len(values), max_population_above=max(values))
    folder = CHAIN_CACHE / "target_N3_lambda0.3_prep1_0_n128"
    states = np.load(folder / "states.npy", mmap_mode="r")
    values = [tail(states[i], 3, 128) for i in range(states.shape[0])]
    out["three_sites"] = dict(cutoff=128, times=len(values), max_population_above=max(values))
    for key in ("two_sites", "three_sites"):
        print(key, out[key])
    (DATA / "dsg_fock_tail.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
