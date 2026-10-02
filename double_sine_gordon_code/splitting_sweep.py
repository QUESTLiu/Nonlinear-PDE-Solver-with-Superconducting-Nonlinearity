"""Outer splitting error of the hybrid and programmable splits across lambda2 and kappa.

Same two-site model, preparation, basis and horizon as splitting_baselines.py (E_J = 1,
omega_b = sqrt(3), T = 5.2/sqrt(3)), with exact blocks, local cutoff 64, and r = 10 and 20.
The extreme point (lambda2 = 0.6, kappa = 2) is repeated at local cutoff 80. Usage, from the
repository root:

    /Users/zhen002/miniconda3/envs/cvdv/bin/python double_sine_gordon_code/splitting_sweep.py
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splitting_baselines import scan  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "double_sine_gordon_data" / "splitting_sweep.json"
ROUTES = ["hybrid", "kin_pot", "gauss_cos"]


def main():
    started = time.perf_counter()
    points = []
    for kappa in (0.5, 1.0, 2.0):
        for lam2 in (0.1, 0.2, 0.3, 0.45, 0.6):
            errors = scan(64, lam2, [10, 20], ROUTES, kappa=kappa)
            points.append(dict(kappa=kappa, lambda2=lam2, cutoff=64, r=[10, 20], errors=errors))
            ratios = {k: [v[i] / errors["hybrid"][i] for i in range(2)] for k, v in errors.items() if k != "hybrid"}
            print(f"kappa={kappa} lambda2={lam2} hybrid={errors['hybrid']} ratios={ratios}", flush=True)
    check = scan(80, 0.6, [10, 20], ROUTES, kappa=2.0)
    record = dict(description=__doc__.split("\n\n")[0], points=points,
                  cutoff_check=dict(kappa=2.0, lambda2=0.6, cutoff=80, r=[10, 20], errors=check),
                  environment=dict(python=sys.executable, python_version=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, wall_seconds=time.perf_counter() - started))
    OUT.write_text(json.dumps(record, indent=1))
    print("cutoff check", json.dumps(check), f"wall {record['environment']['wall_seconds']:.0f} s")


if __name__ == "__main__":
    main()
