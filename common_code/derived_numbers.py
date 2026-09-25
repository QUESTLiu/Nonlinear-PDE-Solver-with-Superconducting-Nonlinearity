"""Collect the values quoted in the manuscript from the archived records.

Reads the archived records (no simulation) plus duffing_data/duffing_amplitude_scan.json and
writes ../data/derived_numbers.json. Every quoted value in the text should be
traceable to an entry here or to the named summary files.
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SG_DATA = REPO / "sine_gordon_data"
DF_DATA = REPO / "duffing_data"
COMMON_DATA = REPO / "common_data"


def e3_record(ej, prep, n):
    for d in (SG_DATA / "formal" / "E3", SG_DATA / "stage1" / "E3"):
        p = d / f"E3_EJ{ej:g}_{prep}_N{n}"
        if Path(str(p) + ".npz").exists():
            return p
    raise FileNotFoundError(ej, prep, n)


def sg_classical():
    out = {}
    for prep, ejs in (("primary", (0.25, 0.5, 1, 2, 4, 8, 16, 32)), ("fixed_variance", (8, 32))):
        for ej in ejs:
            p = e3_record(ej, prep, 80)
            meta = json.loads(Path(str(p) + ".json").read_text())
            with np.load(str(p) + ".npz") as a:
                amp = float(meta["oscillation_amplitude"])
                dev_cl = np.abs(a["mean_phi"] - a["phi_cl"]) / amp
                dev_dw = np.abs(a["mean_phi"] - a["phi_dw"]) / amp
                out[f"{prep}_EJ{ej:g}"] = dict(
                    E_J=ej, preparation=prep, amplitude=amp,
                    D_max=float(dev_cl.max()), D_max_DW=float(dev_dw.max()),
                    D_mean=float(dev_cl.mean()), D_mean_DW=float(dev_dw.mean()),
                    delta_phi=float(meta["delta_phi"]), force_renormalization_pct=float(meta["force_renormalization_pct"]),
                    tau_E=meta.get("tau_ehrenfest_classical"), tau_E_DW=meta.get("tau_ehrenfest_dw"),
                    max_top_population=float(meta["max_leakage"]),
                )
    return out


def polynomial():
    rows = {}
    for phi0 in ("0.5", "1", "2", "3"):
        for n in (48, 64, 80):
            with np.load(SG_DATA / "formal" / "polynomial" / f"E3_poly_phi{phi0}_N{n}.npz") as a:
                degs = a["degrees"]
                entry = {}
                for m in ("taylor", "chebyshev_spectral", "chebyshev_accessed"):
                    ok = (a[f"{m}_max"] < 1e-3) & (~a[f"{m}_cutoff_limited"])
                    entry[m] = int(degs[ok][0]) if ok.any() else None
                    entry[m + "_max_top_population_accepted"] = float(a[f"{m}_leakage"][ok].max()) if ok.any() else None
                rows[f"phi{phi0}_N{n}"] = entry
    return rows


def main():
    scan = json.loads((DF_DATA / "duffing_amplitude_scan.json").read_text())
    summary = json.loads((COMMON_DATA / "analysis_summary.json").read_text())
    out = dict(
        sg_classical=sg_classical(),
        polynomial_min_degree=polynomial(),
        duffing_scan_fit=scan["fit"], duffing_unity=scan["unity_crossing"], duffing_benchmark=scan["benchmark"],
        E1_counts=summary["E1_primary_verdict_counts"],
    )
    (COMMON_DATA / "derived_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["sg_classical"]["primary_EJ1"], indent=1))
    for k, v in out["sg_classical"].items():
        print(k, round(v["D_max"], 4), round(v["D_max_DW"], 4), round(v["D_mean"], 4), round(v["D_mean_DW"], 4), v["tau_E"], v["max_top_population"])
    for k, v in out["polynomial_min_degree"].items():
        print(k, v["taylor"], v["chebyshev_spectral"], v["chebyshev_accessed"], v["taylor_max_top_population_accepted"], v["chebyshev_spectral_max_top_population_accepted"], v["chebyshev_accessed_max_top_population_accepted"])


if __name__ == "__main__":
    main()
