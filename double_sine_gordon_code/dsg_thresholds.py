"""Cheapest double sine-Gordon circuits when the state threshold is tightened.

Reads every evaluated circuit of the step scans (dsg_step_scan.py and dsg_step_scan_lambda.py, parts
directories under data/) and, for each infidelity threshold I* in {1e-3, 5e-4, 2.5e-4, 1e-4}, finds
the cheapest circuit of each route whose state error is at most sqrt(I*), whose reference margin is at
most sqrt(I*)/10, and which meets the unchanged boundary, C12, and S2 criteria. Only the primary
preparation is used. Circuits that exceed the boundary limit only at padding 128 are replaced by their
re-evaluation at cutoffs 160/192 when data/dsg_boundary_check_lambda*.json holds one. Writes
data/dsg_thresholds.json.

Usage, from the repository root (interpreter /Users/zhen002/miniconda3/envs/cvdv/bin/python):
    python double_sine_gordon_code/dsg_thresholds.py
"""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "double_sine_gordon_data"
PARTS = {0.1: DATA / "dsg_step_scan_parts_lambda0.1", 0.3: DATA / "dsg_step_scan_parts", 0.6: DATA / "dsg_step_scan_parts_lambda0.6"}
THRESHOLDS = (1e-3, 5e-4, 2.5e-4, 1e-4)
ROUTES = ("hybrid", "gauss_block", "kin_pot")


def load(directory, lam2):
    unique = {}
    for path in sorted(directory.glob("*.json")):
        for row in json.loads(path.read_text())["rows"]:
            if row["preparation"] == [1.0, 0.0] and row["s"] == 1:
                unique[(row["route"], row["r"], row["order"])] = dict(row, cutoffs=[128, 160])
    # circuits that exceed the boundary limit only at padding 128 and were re-evaluated at cutoffs 160/192
    check = DATA / f"dsg_boundary_check_lambda{lam2:g}.json"
    if check.exists():
        for row in json.loads(check.read_text())["rows"]:
            key = (row["route"], row["r"], row["order"])
            if row["all_criteria"] and key in unique and not unique[key]["all_criteria"] and not unique[key]["acceptance"]["boundary"]:
                unique[key] = dict(row, cutoffs=[160, 192])
    return list(unique.values())


def passes(row, infidelity):
    eps = np.sqrt(infidelity)
    acc = row["acceptance"]
    return bool(row["state_error"] <= eps and row["margins"]["reference"] <= eps / 10 and acc["boundary"] and acc["C12"] and acc["S2"])


def main():
    out = dict(description=__doc__.split("\n\n")[0], results=[])
    for lam2, directory in PARTS.items():
        rows = load(directory, lam2)
        for infidelity in THRESHOLDS:
            entry = dict(lambda2=lam2, infidelity=infidelity)
            for route in ROUTES:
                candidates = [r for r in rows if r["route"] == route and passes(r, infidelity)]
                cheaper = [r for r in rows if r["route"] == route]
                if not candidates:
                    entry[route] = None
                    continue
                best = min(candidates, key=lambda r: (r["total_programmable_primitives"], r["state_error"]))
                # every cheaper circuit of the route must have been evaluated and failed, otherwise the minimum is not established
                unevaluated_gap = best["r"] - 1 not in {r["r"] for r in cheaper if r["order"] == best["order"]}
                entry[route] = dict(r=best["r"], order=best["order"], primitives=best["total_programmable_primitives"],
                                    state_error=best["state_error"], cutoffs=best["cutoffs"], previous_step_evaluated=not unevaluated_gap)
            if entry["hybrid"] and entry["gauss_block"]:
                entry["saving_gauss_block"] = entry["gauss_block"]["primitives"] / entry["hybrid"]["primitives"]
            if entry["hybrid"] and entry["kin_pot"]:
                entry["saving_kin_pot"] = entry["kin_pot"]["primitives"] / entry["hybrid"]["primitives"]
            out["results"].append(entry)
            text = " ".join(f"{k}={v['primitives']}(r{v['r']},o{v['order']}{'' if v['previous_step_evaluated'] else ',gap'})" if v else f"{k}=none"
                            for k, v in ((k, entry[k]) for k in ROUTES))
            print(f"lambda2={lam2:g} I*={infidelity:g}: {text} saving GB={entry.get('saving_gauss_block', float('nan')):.2f} "
                  f"KP={entry.get('saving_kin_pot', float('nan')):.2f}")
    (DATA / "dsg_thresholds.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
