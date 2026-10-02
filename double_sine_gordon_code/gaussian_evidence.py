"""Read merged-circuit error bounds without changing the unmerged archives.

The scalar error fields returned by updated_row are UPPER BOUNDS for Gaussian
rows. Their lower bounds and evidence label travel with them. passes_at refuses
an unresolved decision, so a missing interval cannot silently become a failed
competitor when a consumer selects the cheapest tested circuit.
"""

from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "common_data/records/gaussian_revision"


def candidate_key(row):
    return (
        row.get("sites", 2),
        row.get("lambda2", 0.3),
        tuple(row["preparation"]),
        tuple(row["cutoffs"]),
        row.get("K", 0),
        row["r"],
        row["order"],
        row["s"],
    )


@lru_cache(maxsize=1)
def evidence():
    results = {}
    for path in sorted(RECORDS.glob("g[0-9][0-9][0-9].json")):
        record = json.loads(path.read_text())
        source = ROOT / record["source"]
        row = json.loads(source.read_text())["rows"][record["source_row"]]
        results[candidate_key(row)] = record
    return results


def updated_row(row):
    if (
        row["route"] != "gauss_block"
        or row.get("gaussian_sequence") == "merged_intervals"
    ):
        return row
    record = evidence()[candidate_key(row)]
    if record["acceptance"]["overall"] == "INCONCLUSIVE":
        raise ValueError(f"Unresolved merged Gaussian circuit {record['candidate_id']}")
    out = dict(
        row,
        gaussian_sequence="merged_intervals",
        evidence_id=record["candidate_id"],
        error_evidence="upper bounds propagated from archived raw errors",
        bounds=record["bounds"],
        margins={
            k: v[-1] if isinstance(v, list) else v for k, v in record["margins"].items()
        },
        boundary_population=record["boundary_population"],
        acceptance={
            k: None if v == "INCONCLUSIVE" else v == "PASS"
            for k, v in record["acceptance"].items()
            if k != "overall"
        },
        all_criteria=record["acceptance"]["overall"] == "PASS",
    )
    for name, bounds in record["bounds"].items():
        out[name], out[name + "_lower"] = bounds[1], bounds[0]
    for name in (
        "max_bures",
        "max_ancilla_excited",
        "min_system_purity",
        "gaussian_step_check",
        "wall_seconds",
    ):
        out.pop(name, None)
    if "series" in row:
        series = dict(row["series"], **record["new_series"])
        delta = record["delta_by_cutoff"][str(row["cutoffs"][0])]
        for name, changes in (
            ("bures", delta),
            ("C12_error", record["observable_delta"]["C12"]),
            ("S2_error", record["observable_delta"]["S2"]),
        ):
            series[name + "_lower"] = [
                max(0.0, b - d) for b, d in zip(row["series"][name], changes)
            ]
            series[name] = [b + d for b, d in zip(row["series"][name], changes)]
        series["error_evidence"] = "upper bounds"
        out["series"] = series
    return out


def passes_at(row, eps, allowance=0.0):
    acc, m = row["acceptance"], row["margins"]
    checks = [acc[name] for name in ("boundary", "C12", "S2")]
    if False in checks:
        return False
    if m["reference"] > eps / 10 or m["solver"] > eps / 10:
        return False
    low, high = row.get("bounds", {}).get(
        "state_error", [row["state_error"], row["state_error"]]
    )
    if high + allowance <= eps and None not in checks:
        return True
    if low + allowance > eps:
        return False
    raise ValueError(f"State interval overlaps threshold for {candidate_key(row)}")
