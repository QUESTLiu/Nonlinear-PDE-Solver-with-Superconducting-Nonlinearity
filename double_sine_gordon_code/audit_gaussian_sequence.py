"""Propagate archived Gaussian evidence to the counted, merged circuit.

This bounded revision check never constructs target references. For normalized
joint pure states a,b, min_phase ||a-phase*b|| is their projective distance and
bounds the reduced Bures distance by contractivity of partial trace. Thus an
archived raw error b_old gives max(0,b_old-delta) <= b_new <= b_old+delta.
The newly measured padding margin is added on both sides, along with the
archived reference and solver margins. These remain empirical margins.

Complex C12 and ancilla-traced S2 use the same triangle/reverse-triangle
argument. Their new cutoff differences are measured directly. The old total
observable margin bounds its reference/solver part from above; subtracting the
replayed old padding difference bounds that part from below. Keeping this
interval avoids subtractive roundoff in very small archived margins.

Run `manifest`, then `run --group two_site|chain|common` through run_batch.py.
The manifest fixes all candidate/source/claim dependencies before propagation.
"""

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from double_sine_gordon_code import dsg_step_scan as scan
from double_sine_gordon_code.dsg_chain import Chain
from double_sine_gordon_code.gaussian_evidence import candidate_key

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "double_sine_gordon_data"
MANIFEST = ROOT / "common_data/manifests/gaussian_revision_candidates.json"
OUTPUT = ROOT / "common_data/records/gaussian_revision"


def make_manifest():
    if MANIFEST.exists():
        raise FileExistsError(
            "Preserve the fixed candidate manifest used by the comparison records"
        )
    paths = [
        DATA / f"dsg_step_scan{suffix}.json"
        for suffix in ("", "_lambda0.1", "_lambda0.6")
    ]
    paths += sorted(DATA.glob("dsg_boundary_check_lambda*.json"))
    paths += [DATA / "dsg_chain_N3_cut128_160.json", DATA / "dsg_common_times.json"]
    candidates = []
    for path in paths:
        for index, row in enumerate(json.loads(path.read_text())["rows"]):
            if row["route"] != "gauss_block":
                continue
            group = (
                "chain"
                if row.get("sites", 2) == 3
                else "common"
                if "K" in row
                else "two_site"
            )
            selected = (
                row.get("sites", 2),
                row["lambda2"],
                row["r"],
                row["order"],
                row["cutoffs"][0],
            ) in {
                (2, 0.1, 6, 3, 128),
                (2, 0.3, 7, 3, 128),
                (2, 0.6, 10, 3, 160),
                (3, 0.3, 8, 3, 128),
            }
            claims = (
                ["Table III: three-site minimum among evaluated candidates"]
                if group == "chain"
                else ["Appendix J: common-time selected circuits and tested neighbors"]
                if group == "common"
                else [
                    "Table III and Fig. 6: matched-accuracy resources",
                    "Table VII: tightened thresholds",
                ]
            )
            if row["preparation"] != [1.0, 0.0]:
                claims = ["Table III: second-preparation checks"]
            elif group == "two_site" and row["lambda2"] == 0.3:
                claims.append("Fig. 7: all gate mixes, allowances 0/0.005/0.010/0.020")
            required = group != "chain" or row["total_programmable_primitives"] <= 382
            candidates.append(
                dict(
                    id=f"g{len(candidates):03d}",
                    group=group,
                    key=list(candidate_key(row)),
                    source=str(path.relative_to(ROOT)),
                    source_row=index,
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    selected=selected,
                    claims=claims,
                    required=required,
                    reason="selected or evaluated competitor"
                    if required
                    else "cost exceeds selected three-site count; no three-site timing claim",
                    check="old/new states at both cutoffs; new padding; complex C12; traced S2; merged checkpoints",
                    priority=0 if selected else 1,
                )
            )
    MANIFEST.write_text(
        json.dumps(
            dict(
                sequence="merged_intervals",
                candidates=candidates,
                scope="All archived two-site Gaussian candidates, boundary rechecks and common-time cases; all three-site candidates no costlier than 382.",
                state_thresholds=[math.sqrt(x) for x in (1e-3, 5e-4, 2.5e-4, 1e-4)],
                allowances=[0.0, 0.005, 0.01, 0.02],
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        f"Fixed {len(candidates)} dependencies; {sum(c['required'] for c in candidates)} require propagation."
    )


def distance(a, b):
    """Stable projective distance, avoiding cancellation in sqrt(2-2|overlap|)."""
    a, b = a.ravel(), b.ravel()
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    overlap = np.vdot(b, a)
    phase = overlap / abs(overlap) if overlap else 1.0
    return float(np.linalg.norm(a - phase * b))


def pair_readouts(row, n, observe):
    sites, r, order, s = row.get("sites", 2), row["r"], row["order"], row["s"]
    chain = Chain(sites, n, row["lambda2"], tuple(row["preparation"]))
    initial = np.zeros(chain.psi0.shape + (2,), complex)
    initial[..., 0] = chain.psi0
    if sites == 2:
        scan.LAM2 = row["lambda2"]
        problem = scan.build_two_site_problem(
            n, scan.OMEGA_B, scan.E_J, scan.KAPPA, scan.LAM2
        )

        def factories(h, steps):
            old = scan.make_gauss_block_step(problem, h, order, s, lambda state: None)[
                0
            ]
            new = scan.gauss_block_readouts(
                problem, initial, h, steps, order, s, observe, final_only="K" in row
            )
            return old, new
    else:

        def factories(h, steps):
            return (
                chain.make_step("gauss_block", h, order, s, lambda state: None),
                chain.gaussian_readouts(initial, h, steps, order, s, observe),
            )

    yield initial, initial, chain
    if "K" in row:
        for k in range(1, row["K"] + 1):
            steps = max(1, math.ceil(k * r / row["K"] - 1e-12))
            old_step, new_readouts = factories(k * scan.T / row["K"] / steps, steps)
            old = initial
            for _ in range(steps):
                old = old_step(old)
            for new in new_readouts:
                pass
            yield old, new, chain
    else:
        old_step, new_readouts = factories(scan.T / r, r)
        next(new_readouts)
        old = initial
        for new in new_readouts:
            old = old_step(old)
            yield old, new, chain


def interval(raw, delta):
    """Max over readouts when raw series exists, otherwise use two maxima."""
    raw, delta = np.asarray(raw), np.asarray(delta)
    if raw.ndim == 0:
        delta = delta.max()
    return [float(np.maximum(0.0, raw - delta).max()), float((raw + delta).max())]


def verdict(bounds, threshold):
    return (
        "PASS"
        if bounds[1] <= threshold
        else "FAIL"
        if bounds[0] > threshold
        else "INCONCLUSIVE"
    )


def acceptance(row, eps=scan.EPS, allowance=0.0):
    bounds, margins = row["bounds"], row["margins"]
    state = verdict([x + allowance for x in bounds["state_error"]], eps)
    if margins["reference"] > eps / 10 or margins["solver"] > eps / 10:
        state = "FAIL"
    statuses = dict(
        state=state, boundary="PASS" if row["boundary_population"] <= 1e-6 else "FAIL"
    )
    for name, threshold, u_limit in (("C12", 0.01, 0.001), ("S2", 0.02, 0.005)):
        error_status = verdict(bounds[name + "_error"], threshold)
        margin_status = verdict(margins[name], u_limit)
        statuses[name] = (
            "FAIL"
            if "FAIL" in (error_status, margin_status)
            else "PASS"
            if error_status == margin_status == "PASS"
            else "INCONCLUSIVE"
        )
    statuses["overall"] = (
        "FAIL"
        if "FAIL" in statuses.values()
        else "PASS"
        if all(x == "PASS" for x in statuses.values())
        else "INCONCLUSIVE"
    )
    return statuses


def audit(entry):
    path = ROOT / entry["source"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["source_sha256"]:
        raise ValueError(f"Archived source changed: {path}")
    row = json.loads(path.read_text())["rows"][entry["source_row"]]
    sites, low, high = row.get("sites", 2), *row["cutoffs"]
    old_obs, new_obs, deltas, peaks = {}, {}, {}, {}
    low_states, new_pad, norm_drift = [], 0.0, 0.0
    for n in (low, high):
        peak = [0.0]

        def observe(state):
            # Same highest-four-level union as the archived boundary protocol.
            p = np.abs(state) ** 2
            peak[0] = max(peak[0], float(p.sum() - p[(slice(0, n - 4),) * sites].sum()))

        oo, no, dd = [], [], []
        for k, (old, new, chain) in enumerate(pair_readouts(row, n, observe)):
            observe(new)
            norm_drift = max(
                norm_drift,
                abs(float(np.linalg.norm(new)) - 1),
                abs(float(np.linalg.norm(old)) - 1),
            )
            oo.append(chain.observables(old))
            no.append(chain.observables(new))
            dd.append(distance(old, new))
            if n == low:
                low_states.append(new.copy())
            else:
                embedded = np.zeros_like(new)
                embedded[(slice(0, low),) * sites] = low_states[k]
                new_pad = max(new_pad, distance(embedded, new))
        old_obs[n], new_obs[n], deltas[n], peaks[n] = (
            np.array(oo),
            np.array(no),
            dd,
            peak[0],
        )
    if norm_drift > 1e-10:
        raise ValueError(f"Unitary norm drift {norm_drift} exceeds replay accuracy")
    series = row.get("series", {})
    replay_residual = {}
    for name, old_series in (
        (
            "C12",
            np.array(series.get("C12_real", []))
            + 1j * np.array(series.get("C12_imag", [])),
        ),
        ("S2", np.array(series.get("S2", []))),
    ):
        if len(old_series):
            idx = 0 if name == "C12" else 1
            replay_residual[name] = float(
                np.max(abs(old_series - old_obs[low][:, idx]))
            )
            if replay_residual[name] > 1e-9:
                raise ValueError(
                    f"Replay does not reproduce archive: {name} {replay_residual[name]}"
                )
    m = row["margins"]
    state_margin = m["reference"] + m["solver"] + new_pad
    state_bounds = [
        x + state_margin
        for x in interval(series.get("bures", row["max_bures"]), deltas[low])
    ]
    bounds = dict(state_error=state_bounds)
    margins = dict(reference=m["reference"], solver=m["solver"], padding=new_pad)
    observable_delta = {}
    for j, name in enumerate(("C12", "S2")):
        delta = abs(new_obs[low][:, j] - old_obs[low][:, j])
        old_padding = float(np.max(abs(old_obs[low][:, j] - old_obs[high][:, j])))
        new_padding = float(np.max(abs(new_obs[low][:, j] - new_obs[high][:, j])))
        margins[name] = [
            max(0.0, m[name] - old_padding) + new_padding,
            m[name] + new_padding,
        ]
        raw = series.get(name + "_error", row[name + "_error"] - m[name])
        bounds[name + "_error"] = [
            x + u for x, u in zip(interval(raw, delta), margins[name])
        ]
        observable_delta[name] = [float(x) for x in delta]
    result = dict(
        candidate_id=entry["id"],
        key=entry["key"],
        gaussian_sequence="merged_intervals",
        evidence="old raw errors plus measured old/new triangle bounds; empirical reference/solver margins",
        source=entry["source"],
        source_row=entry["source_row"],
        source_sha256=entry["source_sha256"],
        bounds=bounds,
        margins=margins,
        boundary_population=max(peaks.values()),
        boundary_by_cutoff=peaks,
        delta_by_cutoff=deltas,
        observable_delta=observable_delta,
        max_norm_drift=norm_drift,
        replay_residual=replay_residual,
        primitive_counts=row["primitive_counts"],
        total_programmable_primitives=row["total_programmable_primitives"],
        new_series=dict(
            C12_real=new_obs[low][:, 0].real.tolist(),
            C12_imag=new_obs[low][:, 0].imag.tolist(),
            S2=new_obs[low][:, 1].real.tolist(),
        ),
    )
    result["acceptance"] = acceptance(result)
    result["threshold_status"] = {
        str(x): acceptance(result, math.sqrt(x)) for x in (1e-3, 5e-4, 2.5e-4, 1e-4)
    }
    result["allowance_status"] = {
        str(x): acceptance(result, allowance=x) for x in (0.0, 0.005, 0.01, 0.02)
    }
    return result


def selected_statistics(entries, output_dir):
    """Readout qubit populations and reduced-system purities for Table III."""
    output = output_dir / "selected_readout_statistics.json"
    if output.exists():
        raise FileExistsError(output)
    results = []
    for entry in entries:
        if not entry["selected"] or entry["group"] == "common":
            continue
        source = ROOT / entry["source"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["source_sha256"]:
            raise ValueError(source)
        row = json.loads(source.read_text())["rows"][entry["source_row"]]
        n, sites = row["cutoffs"][0], row.get("sites", 2)
        chain = Chain(sites, n, row["lambda2"], tuple(row["preparation"]))
        initial = np.zeros(chain.psi0.shape + (2,), complex)
        initial[..., 0] = chain.psi0
        started = time.perf_counter()
        if sites == 2:
            scan.LAM2 = row["lambda2"]
            problem = scan.build_two_site_problem(
                n, scan.OMEGA_B, scan.E_J, scan.KAPPA, scan.LAM2
            )
            states = scan.gauss_block_readouts(
                problem,
                initial,
                scan.T / row["r"],
                row["r"],
                row["order"],
                row["s"],
                lambda state: None,
            )
        else:
            states = chain.gaussian_readouts(
                initial,
                scan.T / row["r"],
                row["r"],
                row["order"],
                row["s"],
                lambda state: None,
            )
        ancilla, impurity = [], []
        for state in states:
            flat = state.reshape(-1, 2)
            gram = flat.conj().T @ flat
            ancilla.append(float(gram[1, 1].real))
            impurity.append(float(1 - np.vdot(gram, gram).real))
        results.append(
            dict(
                candidate_id=entry["id"],
                cutoff=n,
                key=entry["key"],
                max_ancilla_excited=max(ancilla),
                max_system_impurity=max(impurity),
                ancilla=ancilla,
                system_impurity=impurity,
                wall_seconds=time.perf_counter() - started,
            )
        )
        print(entry["id"], max(ancilla), max(impurity), flush=True)
        output.write_text(
            json.dumps(
                dict(gaussian_sequence="merged_intervals", rows=results), indent=2
            )
            + "\n"
        )
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("manifest", "run", "statistics"))
    parser.add_argument("--group", choices=("two_site", "chain", "common"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.job == "manifest":
        make_manifest()
        return
    entries = json.loads(MANIFEST.read_text())["candidates"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.job == "statistics":
        selected_statistics(entries, args.output_dir)
        return
    for entry in sorted(entries, key=lambda x: (x["priority"], x["id"])):
        if not entry["required"] or entry["group"] != args.group:
            continue
        output = args.output_dir / (entry["id"] + ".json")
        if output.exists():
            raise FileExistsError(output)
        started = time.perf_counter()
        result = audit(entry)
        result["wall_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            entry["id"],
            entry["key"],
            result["acceptance"]["overall"],
            result["bounds"]["state_error"],
            f"{result['wall_seconds']:.2f}s",
            flush=True,
        )
        gc.collect()


if __name__ == "__main__":
    main()
