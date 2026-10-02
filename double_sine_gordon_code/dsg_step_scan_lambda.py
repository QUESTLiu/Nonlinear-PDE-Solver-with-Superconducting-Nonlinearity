"""Step-count scan of the double sine-Gordon circuits at another lambda2.

Runs the circuits and the acceptance rule of dsg_step_scan.py (same model, preparation,
cutoffs 128/160, s = 1) with the second-harmonic strength lambda2 changed, and writes
data/dsg_step_scan_lambda{lambda2}_merged.json. Usage, from the repository root
(interpreter /Users/zhen002/miniconda3/envs/cvdv/bin/python):

    python double_sine_gordon_code/dsg_step_scan_lambda.py references --lam2 0.1 --r 2,3,4
    python double_sine_gordon_code/dsg_step_scan_lambda.py scan --lam2 0.1 --route hybrid --orders 1,2 --r 2,3,4 [--prep 1,0]
    python double_sine_gordon_code/dsg_step_scan_lambda.py merge --lam2 0.1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsg_step_scan as scan  # noqa: E402


def configure(lam2):
    scan.LAM2 = lam2
    scan.CACHE = (
        Path(os.environ.get("DSG_SCAN_CACHE", "/tmp/dsg_scan_cache"))
        / f"lambda{lam2:g}"
    )
    scan.PARTS = scan.DATA / f"dsg_step_scan_parts_lambda{lam2:g}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("job", choices=["references", "scan", "merge"])
    parser.add_argument("--lam2", type=float, required=True)
    parser.add_argument("--route", choices=["hybrid", "kin_pot", "gauss_block"])
    parser.add_argument("--orders", default="1,2")
    parser.add_argument("--r", default="")
    parser.add_argument("--s", type=int, default=1)
    parser.add_argument("--prep", default="1,0")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    configure(args.lam2)
    prep = tuple(float(x) for x in args.prep.split(","))
    steps = [int(x) for x in args.r.split(",") if x]
    if args.job == "references":
        scan.compute_references(steps, prep)
        return
    if args.job == "scan":
        scan.PARTS.mkdir(parents=True, exist_ok=True)
        scan.compute_references(steps, prep)
        results = []
        for r in steps:
            for order in [int(x) for x in args.orders.split(",")]:
                started = time.perf_counter()
                row = scan.evaluate(args.route, r, order, args.s, prep)
                row["lambda2"] = args.lam2
                row["wall_seconds"] = time.perf_counter() - started
                results.append(row)
                acc = (
                    "PASS"
                    if row["all_criteria"]
                    else "fail "
                    + ",".join(k for k, v in row["acceptance"].items() if not v)
                )
                print(
                    f"lambda2={args.lam2:g} {args.route} r={r} o={order} s={args.s} prims={row['total_programmable_primitives']} "
                    f"state={row['state_error']:.4f} C12={row['C12_error']:.4f} S2={row['S2_error']:.4f} "
                    f"bnd={row['boundary_population']:.2e} ref={row['margins']['reference']:.1e} {acc} ({row['wall_seconds']:.0f} s)",
                    flush=True,
                )
                sequence_tag = "_merged" if args.route == "gauss_block" else ""
                name = f"{args.route}_prep{scan.prep_tag(prep)}{args.tag}{sequence_tag}.json"
                (scan.PARTS / name).write_text(
                    json.dumps(
                        dict(environment=scan.environment(), rows=results), indent=1
                    )
                )
        return
    unique = {}
    for path in sorted(scan.PARTS.glob("*.json")):
        for row in json.loads(path.read_text())["rows"]:
            if (
                row["route"] == "gauss_block"
                and row.get("gaussian_sequence") != "merged_intervals"
            ):
                continue
            unique[
                (
                    row["route"],
                    row["r"],
                    row["order"],
                    row["s"],
                    tuple(row["preparation"]),
                )
            ] = row
    rows = list(unique.values())
    best = {}
    for row in rows:
        if row["preparation"] != [1.0, 0.0] or not row["all_criteria"]:
            continue
        key = row["route"]
        if key not in best or (
            row["total_programmable_primitives"],
            row["state_error"],
        ) < (best[key]["total_programmable_primitives"], best[key]["state_error"]):
            best[key] = row
    summary = dict(
        description=f"dsg_step_scan.py circuits and acceptance rule at lambda2 = {args.lam2:g}",
        lambda2=args.lam2,
        rows=rows,
        cheapest_passing={
            k: dict(
                r=v["r"],
                order=v["order"],
                s=v["s"],
                primitives=v["total_programmable_primitives"],
                state_error=v["state_error"],
            )
            for k, v in best.items()
        },
        environment=scan.environment(),
    )
    out = scan.DATA / f"dsg_step_scan_lambda{args.lam2:g}_merged.json"
    out.write_text(json.dumps(summary, indent=1))
    for k, v in summary["cheapest_passing"].items():
        print(k, v)


if __name__ == "__main__":
    main()
