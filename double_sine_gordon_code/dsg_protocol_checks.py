"""Protocol checks of the two-site double sine-Gordon circuits of dsg_step_scan.py.

common  Evaluates a circuit at the common readout times t_k = k T / K (default K = 20) instead of
        its own outer-step times. Each requested time t_k is reached by its own circuit of
        n_k = ceil(t_k / h) equal outer steps of length t_k / n_k <= h, with h = T / r, so no
        partial step is needed and no readout circuit costs more primitives than the circuit to
        T, whose count is unchanged. The acceptance rule is that of dsg_step_scan.py with every
        maximum taken over the common times (state, observables, margins) and the boundary
        population monitored in every readout circuit.
gauss   Replays the archived unmerged Gaussian-block circuit and compares, on every state that enters a Gaussian
        factor product Phi(a) Q Phi(a), the factorized step with the exact truncated block
        exp[-i tau (H_kin + H_bond)] of the finite spectral model (sparse matrix exponential).
merge   Collects current common-time parts into data/dsg_common_times_merged.json.

Usage, from the repository root:
    python double_sine_gordon_code/dsg_protocol_checks.py common --route hybrid --orders 1 --r 5,6,7 [--lam2 0.3] [--K 20]
    python double_sine_gordon_code/dsg_protocol_checks.py gauss --r 7 --orders 3 [--lam2 0.3]
    python double_sine_gordon_code/dsg_protocol_checks.py merge
Reference trajectories are cached as in dsg_step_scan.py (DSG_SCAN_CACHE).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsg_step_scan as scan  # noqa: E402
import dsg_step_scan_lambda as scan_lambda  # noqa: E402

PARTS = scan.DATA / "dsg_common_times_parts"


def make_step(route, problem, h, order, s, observe):
    if route == "gauss_block":
        return scan.make_gauss_block_step(problem, h, order, s, observe)[0]
    if route == "kin_pot" and order == 3:
        return scan.make_kin_pot_step(problem, h, order, s, observe)
    if route == "hybrid" and order == 3:
        raise ValueError(
            "the hybrid circuit compiles one harmonic, so order code 3 does not apply"
        )
    return scan.nc.make_step(
        problem,
        scan.E_J,
        scan.KAPPA,
        scan.LAM2,
        h,
        {"hybrid": "H_trig", "kin_pot": "G_trig"}[route],
        order,
        s,
        observer=observe,
    )


def readout_schedule(r, K):
    """Step number and step length of the circuit for each common time t_k = k T / K."""
    schedule = []
    for k in range(1, K + 1):
        t = k * scan.T / K
        n_k = max(1, math.ceil(k * r / K - 1e-12))
        schedule.append((k, t, n_k, t / n_k))
    return schedule


def run_common(route, n, r, order, s, prep, K, reference):
    problem = scan.build_two_site_problem(
        n, scan.OMEGA_B, scan.E_J, scan.KAPPA, scan.LAM2
    )
    basis = problem["basis"]
    initial = np.zeros((n, n, 2), complex)
    initial[:, :, 0] = scan._sg_initial(basis, *prep)
    peak = scan.nc.populations(initial)

    def observe(state):
        for key, value in scan.nc.populations(state).items():
            peak[key] = max(peak[key], value)

    rows, states = [], [initial.copy()]
    obs0 = scan.nc.two_site_observables(initial, basis, joint=True)
    rows.append(
        dict(
            k=0,
            t=0.0,
            steps=0,
            h=0.0,
            bures=scan.nc.reduced_metrics(initial, reference["states"][0])[
                "bures_distance"
            ],
            ancilla=0.0,
            purity=1.0,
            C12=obs0["C12_real"] + 1j * obs0["C12_imag"],
            S2=obs0["S2"],
            C12_error=abs(
                obs0["C12_real"] + 1j * obs0["C12_imag"] - reference["C12"][0]
            ),
            S2_error=abs(obs0["S2"] - reference["S2"][0]),
        )
    )
    for k, t, n_k, h_k in readout_schedule(r, K):
        joint = initial.copy()
        if route == "gauss_block":
            for joint in scan.gauss_block_readouts(
                problem, joint, h_k, n_k, order, s, observe, final_only=True
            ):
                pass
        else:
            step = make_step(route, problem, h_k, order, s, observe)
            for _ in range(n_k):
                joint = step(joint)
                observe(joint)
        metrics = scan.nc.reduced_metrics(joint, reference["states"][k])
        obs = scan.nc.two_site_observables(joint, basis, joint=True)
        c12 = obs["C12_real"] + 1j * obs["C12_imag"]
        rows.append(
            dict(
                k=k,
                t=t,
                steps=n_k,
                h=h_k,
                bures=metrics["bures_distance"],
                ancilla=metrics["ancilla_excited_population"],
                purity=metrics["system_purity"],
                C12=c12,
                S2=obs["S2"],
                C12_error=abs(c12 - reference["C12"][k]),
                S2_error=abs(obs["S2"] - reference["S2"][k]),
            )
        )
        states.append(joint)
    return rows, np.array(states), peak


def evaluate_common(route, r, order, s, prep, K):
    low, high = scan.CUTS
    scan.compute_references([K], prep)
    refs = {n: scan.load_reference(n, K, prep) for n in scan.CUTS}
    out = {n: run_common(route, n, r, order, s, prep, K, refs[n]) for n in scan.CUTS}
    rows_low, states_low, peak_low = out[low]
    rows_high, states_high, peak_high = out[high]
    embedded = np.zeros((K + 1, high, high, 2), complex)
    embedded[:, :low, :low, :] = states_low
    pad = max(
        scan.projective_distance(a.ravel(), b.ravel())
        for a, b in zip(embedded, states_high)
    )
    target_low = np.zeros((K + 1, high, high), complex)
    target_low[:, :low, :low] = refs[low]["states"].reshape(K + 1, low, low)
    ref_margin = max(
        scan.projective_distance(a.ravel(), b)
        for a, b in zip(target_low, refs[high]["states"])
    )
    cref = float(np.max(abs(refs[low]["C12"] - refs[high]["C12"])))
    sref = float(np.max(abs(refs[low]["S2"] - refs[high]["S2"])))
    cp = max(abs(a["C12"] - b["C12"]) for a, b in zip(rows_low, rows_high))
    sp = max(abs(a["S2"] - b["S2"]) for a, b in zip(rows_low, rows_high))
    uc, us = cref + scan.SOLVER["C12"] + cp, sref + scan.SOLVER["S2"] + sp
    max_bures = max(row["bures"] for row in rows_low)
    total = max_bures + ref_margin + scan.SOLVER["state"] + pad
    boundary = max(
        peak_low["boundary_band_population"], peak_high["boundary_band_population"]
    )
    c12 = max(row["C12_error"] for row in rows_low)
    s2 = max(row["S2_error"] for row in rows_low)
    acceptance = dict(
        state=bool(
            total <= scan.EPS
            and ref_margin <= scan.EPS / 10
            and scan.SOLVER["state"] <= scan.EPS / 10
        ),
        boundary=bool(boundary <= 1e-6),
        C12=bool(uc <= 1e-3 and c12 + uc <= 0.01),
        S2=bool(us <= 5e-3 and s2 + us <= 0.02),
    )
    counts, waits = scan.primitive_counts(route, r, order, s)
    return dict(
        route=route,
        r=r,
        order=order,
        block_orders=list(scan.ORDER_CODES[order]),
        s=s,
        preparation=list(prep),
        gaussian_sequence="merged_intervals" if route == "gauss_block" else None,
        lambda2=scan.LAM2,
        cutoffs=list(scan.CUTS),
        K=K,
        protocol="common readout times t_k = k T / K, each reached by "
        "ceil(t_k / h) equal steps",
        primitive_counts=counts,
        total_programmable_primitives=int(sum(counts.values())),
        native_waits=waits,
        state_error=total,
        max_bures=max_bures,
        argmax_time=float(
            rows_low[int(np.argmax([row["bures"] for row in rows_low]))]["t"]
        ),
        margins=dict(
            reference=ref_margin,
            solver=scan.SOLVER["state"],
            padding=pad,
            C12=uc,
            S2=us,
        ),
        C12_error=c12 + uc,
        S2_error=s2 + us,
        boundary_population=boundary,
        max_ancilla_excited=max(row["ancilla"] for row in rows_low),
        min_system_purity=min(row["purity"] for row in rows_low),
        acceptance=acceptance,
        all_criteria=bool(all(acceptance.values())),
        series=dict(
            t=[row["t"] for row in rows_low],
            steps=[row["steps"] for row in rows_low],
            bures=[row["bures"] for row in rows_low],
            C12_error=[row["C12_error"] for row in rows_low],
            S2_error=[row["S2_error"] for row in rows_low],
        ),
    )


def gauss_check(r, order, s, prep):
    """Archived unmerged-sequence diagnostic, not validation of the merged production circuit."""
    result = {}
    for n in scan.CUTS:
        problem = scan.build_two_site_problem(
            n, scan.OMEGA_B, scan.E_J, scan.KAPPA, scan.LAM2
        )
        basis = problem["basis"]
        exact_generator = (
            -0.5j
            * (scan.T / r)
            * scan.build_two_site_problem(n, scan.OMEGA_B, 0.0, scan.KAPPA, 0.0)[
                "H_native"
            ]
        )
        step, gaussian = scan.make_gauss_block_step(
            problem, scan.T / r, order, s, lambda state: None
        )
        joint = np.zeros((n, n, 2), complex)
        joint[:, :, 0] = scan._sg_initial(basis, *prep)
        worst, worst_boundary, entries = 0.0, 0.0, 0

        def compare(state):
            nonlocal worst, worst_boundary, entries
            factorized = gaussian(state)
            exact = np.stack(
                [
                    expm_multiply(exact_generator, state[:, :, q].ravel()).reshape(n, n)
                    for q in (0, 1)
                ],
                axis=-1,
            )
            worst = max(worst, float(np.linalg.norm((factorized - exact).ravel())))
            worst_boundary = max(
                worst_boundary, scan.nc.populations(state)["boundary_band_population"]
            )
            entries += 1
            return factorized

        # One outer step is Gaussian, potential blocks, Gaussian. Replay it with the comparison at both Gaussian factors.
        order_j, order_2 = scan.ORDER_CODES[order]
        blocks = {
            1: scan.nc.trig_block(basis.phi, 1, scan.T / r * scan.E_J, order_j, s),
            2: scan.nc.trig_block(basis.phi, 2, scan.T / r * scan.LAM2, order_2, s),
        }
        for _ in range(r):
            out = compare(joint)
            for site in (0, 1):
                for m in (1, 2):
                    out = scan.nc.apply_joint_local(blocks[m], out, site)
            joint = compare(out)
        result[n] = dict(
            entry_states=entries,
            max_difference=worst,
            max_entry_boundary_population=worst_boundary,
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("job", choices=["common", "gauss", "merge"])
    parser.add_argument("--route", choices=["hybrid", "kin_pot", "gauss_block"])
    parser.add_argument("--orders", default="1")
    parser.add_argument("--r", default="")
    parser.add_argument("--s", type=int, default=1)
    parser.add_argument("--prep", default="1,0")
    parser.add_argument("--lam2", type=float, default=0.3)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--cuts", default="128,160", help="reference and padding cutoffs"
    )
    args = parser.parse_args()
    if args.lam2 != 0.3:
        scan_lambda.configure(args.lam2)
    scan.CUTS = tuple(int(x) for x in args.cuts.split(","))
    prep = tuple(float(x) for x in args.prep.split(","))
    steps = [int(x) for x in args.r.split(",") if x]
    orders = [int(x) for x in args.orders.split(",")]
    if args.job == "common":
        PARTS.mkdir(parents=True, exist_ok=True)
        results = []
        for r in steps:
            for order in orders:
                started = time.perf_counter()
                row = evaluate_common(args.route, r, order, args.s, prep, args.K)
                row["wall_seconds"] = time.perf_counter() - started
                results.append(row)
                acc = (
                    "PASS"
                    if row["all_criteria"]
                    else "fail "
                    + ",".join(k for k, v in row["acceptance"].items() if not v)
                )
                print(
                    f"lambda2={scan.LAM2:g} K={args.K} {args.route} r={r} o={order} prims={row['total_programmable_primitives']} "
                    f"state={row['state_error']:.4f} (max at t={row['argmax_time']:.3f}) C12={row['C12_error']:.4f} "
                    f"S2={row['S2_error']:.4f} bnd={row['boundary_population']:.2e} {acc} ({row['wall_seconds']:.0f} s)",
                    flush=True,
                )
                cuts_tag = (
                    ""
                    if scan.CUTS == (128, 160)
                    else f"_cut{scan.CUTS[0]}_{scan.CUTS[1]}"
                )
                sequence_tag = "_merged" if args.route == "gauss_block" else ""
                name = f"{args.route}_lambda{scan.LAM2:g}_K{args.K}_prep{scan.prep_tag(prep)}{cuts_tag}{args.tag}{sequence_tag}.json"
                (PARTS / name).write_text(
                    json.dumps(
                        dict(environment=scan.environment(), rows=results),
                        indent=1,
                        default=lambda x: [x.real, x.imag]
                        if isinstance(x, complex)
                        else x,
                    )
                )
        return
    if args.job == "gauss":
        out = {}
        for r in steps:
            for order in orders:
                started = time.perf_counter()
                res = gauss_check(r, order, args.s, prep)
                out[f"r{r}_o{order}"] = res
                print(
                    f"lambda2={scan.LAM2:g} gauss_block r={r} o={order}: "
                    + ", ".join(
                        f"N={n}: {v['entry_states']} entries, max |factorized - exact| = {v['max_difference']:.2e}, "
                        f"max entry boundary population = {v['max_entry_boundary_population']:.1e}"
                        for n, v in res.items()
                    )
                    + f" ({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
        path = scan.DATA / "dsg_gauss_factor_check.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        previous.setdefault(f"lambda{scan.LAM2:g}", {}).update(out)
        previous["environment"] = scan.environment()
        path.write_text(json.dumps(previous, indent=1))
        return
    unique = {}
    for path in sorted(PARTS.glob("*.json")):
        for row in json.loads(path.read_text())["rows"]:
            if (
                row["route"] == "gauss_block"
                and row.get("gaussian_sequence") != "merged_intervals"
            ):
                continue
            unique[
                (
                    row["lambda2"],
                    row["K"],
                    row["route"],
                    row["r"],
                    row["order"],
                    tuple(row["preparation"]),
                    tuple(row["cutoffs"]),
                )
            ] = row
    rows = sorted(
        unique.values(),
        key=lambda x: (x["lambda2"], x["K"], x["route"], x["order"], x["r"]),
    )
    best = {}
    for row in rows:
        if row["preparation"] != [1.0, 0.0] or not row["all_criteria"]:
            continue
        key = f"lambda{row['lambda2']:g}_K{row['K']}_{row['route']}_cut{row['cutoffs'][0]}_{row['cutoffs'][1]}"
        if (
            key not in best
            or row["total_programmable_primitives"] < best[key]["primitives"]
        ):
            best[key] = dict(
                r=row["r"],
                order=row["order"],
                primitives=row["total_programmable_primitives"],
                state_error=row["state_error"],
            )
    summary = dict(
        description=__doc__.split("\n\n")[0],
        rows=rows,
        cheapest_passing=best,
        environment=scan.environment(),
    )
    (scan.DATA / "dsg_common_times_merged.json").write_text(
        json.dumps(summary, indent=1)
    )
    for key, value in best.items():
        print(key, value)


if __name__ == "__main__":
    main()
