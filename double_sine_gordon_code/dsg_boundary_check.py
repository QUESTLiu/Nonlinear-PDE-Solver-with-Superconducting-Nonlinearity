"""Re-evaluate boundary-limited double sine-Gordon circuits at reference and padding cutoffs 160/192.

At lambda2 = 0.6 the Gaussian-block circuits with mixed block orders (Josephson harmonic second
order, lambda2 harmonic first order) and r = 10 to 12 meet the state and observable criteria of
dsg_step_scan.py at cutoffs 128/160 but exceed the boundary limit 1e-6 at padding 128. This script
applies the same acceptance rule (dsg_step_scan.evaluate, same solver margin) with CUTS = (160, 192)
to those circuits and to r = 9, and writes data/dsg_boundary_check_lambda0.6_merged.json. About 70 s per
reference at cutoff 192 and a few seconds per circuit.

Usage, from the repository root (interpreter /Users/zhen002/miniconda3/envs/cvdv/bin/python):
    python double_sine_gordon_code/dsg_boundary_check.py [0.75,0.25 10]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsg_step_scan as scan  # noqa: E402
import dsg_step_scan_lambda as scan_lambda  # noqa: E402

LAM2, ROUTE, ORDER, STEPS = 0.6, "gauss_block", 3, (9, 10, 11, 12)


def main():
    # optional arguments: preparation <phi_1>,<Pi_1> and outer-step numbers, e.g. "0.75,0.25 10"
    prep = tuple(float(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (1.0, 0.0)
    steps = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else STEPS
    scan_lambda.configure(LAM2)
    scan.CUTS = (160, 192)
    rows = []
    for r in steps:
        scan.compute_references([r], prep)
        started = time.perf_counter()
        row = scan.evaluate(ROUTE, r, ORDER, 1, prep)
        row["lambda2"] = LAM2
        row["wall_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(f"lambda2={LAM2:g} cutoffs 160/192 {ROUTE} r={r} o={ORDER} prims={row['total_programmable_primitives']} "
              f"state={row['state_error']:.4f} C12={row['C12_error']:.4f} S2={row['S2_error']:.4f} "
              f"boundary={row['boundary_low']:.2e}/{row['boundary_high']:.2e} {'PASS' if row['all_criteria'] else row['acceptance']}",
              flush=True)
    out = scan.DATA / f"dsg_boundary_check_lambda{LAM2:g}{'' if prep == (1.0, 0.0) else '_prep' + scan.prep_tag(prep)}_merged.json"
    out.write_text(json.dumps(dict(description=__doc__.split("\n\n")[0], rows=rows, environment=scan.environment()), indent=1,
                              default=lambda x: [x.real, x.imag] if isinstance(x, complex) else x))


if __name__ == "__main__":
    main()
