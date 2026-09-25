"""Step-count scan of double sine-Gordon correction circuits at padding cutoffs 128 and 160.

Two open sites, E_J = 1, kappa = 1, lambda2 = 0.3, omega_b = sqrt(3), T = 5.2/sqrt(3), finite
spectral convention, and the repository operators of the archived comparison. Three routes are
compiled with one internal subdivision (s = 1). The order code 1 or 2 uses first- or second-order
trigonometric blocks for every harmonic, and code 3 (programmable routes only) uses second-order
blocks for the Josephson harmonic and first-order blocks for the lambda2 harmonic:

  hybrid       H_SG | Delta H, native lattice plus trigonometric correction (repository route H_trig)
  kin_pot      H_kin | (H_bond + V_J + Delta H), kinetic/potential split (repository route G_trig)
  gauss_block  (H_kin + H_bond) | (V_J + Delta H), where the exact two-mode Gaussian step
               exp[-i tau (H_kin + H_bond)] = Phi(a) Q Phi(a) is written as a bond-type quadratic
               phase Phi in the phases and a quadratic phase Q in the momenta (three primitives each)

Every circuit is judged by the acceptance rule of the archived comparison, applied at cutoffs
128/160: state error (reduced Bures distance maximized over the outer-step times, plus reference,
solver and padding margins) at most sqrt(1e-3), C12 and S2 errors within 0.01 and 0.02 including
their margins, and boundary population of the four highest levels below 1e-6 at both cutoffs.
The solver margins are the archived DOP853 comparison at cutoff 160 (double_sine_gordon_data/sg160).

Usage, from the repository root (interpreter /Users/zhen002/miniconda3/envs/cvdv/bin/python):
    python double_sine_gordon_code/dsg_step_scan.py references --r 5,6,7 [--prep 1,0]
    python double_sine_gordon_code/dsg_step_scan.py scan --route gauss_block --orders 1,2,3 --r 5,6,7 [--prep 1,0]
    python double_sine_gordon_code/dsg_step_scan.py merge
Reference trajectories are cached in the directory given by DSG_SCAN_CACHE (default /tmp/dsg_scan_cache).
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common_code.metrics import projective_distance  # noqa: E402
from common_code.run_pilot import _sg_initial  # noqa: E402
from sine_gordon_code import native_correction as nc  # noqa: E402
from sine_gordon_code import sglib as sg  # noqa: E402
from double_sine_gordon_code.run_snapd_end_to_end import build_two_site_problem, exact_trajectory  # noqa: E402

E_J, KAPPA, LAM2, OMEGA_B = 1.0, 1.0, 0.3, np.sqrt(3.0)
T = 5.2 / np.sqrt(3.0)
CUTS = (128, 160)
SOLVER = {"state": 6.850423817979153e-09, "C12": 6.40067722469277e-12, "S2": 6.767145871444669e-12}
EPS = np.sqrt(1e-3)
CACHE = Path(os.environ.get("DSG_SCAN_CACHE", "/tmp/dsg_scan_cache"))
DATA = Path(__file__).resolve().parents[1] / "double_sine_gordon_data"
PARTS = DATA / "dsg_step_scan_parts"
ORDER_CODES = {1: (1, 1), 2: (2, 2), 3: (2, 1)}  # (order of the E_J harmonic, order of the lambda2 harmonic)


def prep_tag(prep):
    return f"{prep[0]:g}_{prep[1]:g}"


def reference_path(n, r, prep):
    return CACHE / f"target_N{n}_r{r}_prep{prep_tag(prep)}.npz"


def compute_references(steps, prep):
    CACHE.mkdir(parents=True, exist_ok=True)
    for n in CUTS:
        problem = build_two_site_problem(n, OMEGA_B, E_J, KAPPA, LAM2)
        basis = problem["basis"]
        initial = _sg_initial(basis, *prep)
        for r in steps:
            path = reference_path(n, r, prep)
            if path.exists():
                continue
            started = time.perf_counter()
            target = exact_trajectory(problem["H_target"], initial, T, r)
            obs = [nc.two_site_observables(state.reshape(n, n), basis) for state in target]
            np.savez(path, states=target, C12=np.array([o["C12_real"] + 1j * o["C12_imag"] for o in obs]),
                     S2=np.array([o["S2"] for o in obs]))
            print(f"reference N={n} r={r} prep={prep} {time.perf_counter() - started:.1f} s", flush=True)


def gaussian_step_factors(basis, tau):
    """Phases of Phi(a) (phi basis) and Q (momentum basis) with Phi Q Phi = exp[-i tau (H_kin + H_bond)]."""
    omega = np.sqrt(2 * KAPPA)
    b = np.sin(omega * tau) / (2 * omega)
    a = 0.5 * omega * np.tan(0.5 * omega * tau)
    L1, L2 = np.meshgrid(basis.lam_phi, basis.lam_phi, indexing="ij")
    lam_pi = np.linalg.eigvalsh(basis.pi)
    P1, P2 = np.meshgrid(lam_pi, lam_pi, indexing="ij")
    phase_phi = np.exp(-0.5j * a * (L1 - L2) ** 2)
    phase_pi = np.exp(-1j * ((tau / 4 + b / 2) * (P1 ** 2 + P2 ** 2) + (tau / 2 - b) * P1 * P2))
    return phase_phi, phase_pi


def make_gauss_block_step(problem, h, order, s, observer):
    basis = problem["basis"]
    U_phi = basis.U_phi
    _, U_pi = np.linalg.eigh(basis.pi)
    order_j, order_2 = ORDER_CODES[order]
    blocks = {1: nc.trig_block(basis.phi, 1, h * E_J, order_j, s), 2: nc.trig_block(basis.phi, 2, h * LAM2, order_2, s)}
    phase_phi, phase_pi = gaussian_step_factors(basis, h / 2)

    def diag(state, U, phase):
        rotated = sg._apply_axis(U.conj().T, sg._apply_axis(U.conj().T, state, 0), 1)
        return sg._apply_axis(U, sg._apply_axis(U, phase[:, :, None] * rotated, 0), 1)

    def gaussian(state):
        return diag(diag(diag(state, U_phi, phase_phi), U_pi, phase_pi), U_phi, phase_phi)

    def step(state, record=True):
        out = gaussian(state)
        if record:
            observer(out)
        for site in (0, 1):
            for m in (1, 2):
                out = nc.apply_joint_local(blocks[m], out, site)
                if record:
                    observer(out)
        out = gaussian(out)
        if record:
            observer(out)
        return out

    return step, gaussian


def make_kin_pot_step(problem, h, order, s, observer):
    """Kinetic/potential split of the repository route G_trig with per-harmonic block orders."""
    basis = problem["basis"]
    order_j, order_2 = ORDER_CODES[order]
    blocks = {1: nc.trig_block(basis.phi, 1, h * E_J, order_j, s), 2: nc.trig_block(basis.phi, 2, h * LAM2, order_2, s)}
    kinetic_half = expm(-0.5j * h * basis.kinetic)
    bond_phase = np.exp(-0.25j * h * KAPPA * (basis.lam_phi[:, None] - basis.lam_phi[None, :]) ** 2)

    def kinetic(state):
        return sg._apply_axis(kinetic_half, sg._apply_axis(kinetic_half, state, 0), 1)

    def bond(state):
        rotated = sg._apply_axis(basis.U_phi.conj().T, sg._apply_axis(basis.U_phi.conj().T, state, 0), 1)
        return sg._apply_axis(basis.U_phi, sg._apply_axis(basis.U_phi, bond_phase[:, :, None] * rotated, 0), 1)

    def step(state, record=True):
        out = bond(kinetic(state))
        if record:
            observer(out)
        for site in (0, 1):
            for m in (1, 2):
                out = nc.apply_joint_local(blocks[m], out, site)
                if record:
                    observer(out)
        out = kinetic(bond(out))
        if record:
            observer(out)
        return out

    return step


def trig_schedule_orders(coefficients, h, orders, s):
    """nc.trig_schedule for two sites with a separate block order for each harmonic."""
    pulses = []
    for site in range(2):
        for harmonic, (coefficient, order) in enumerate(zip(coefficients, orders), 1):
            if coefficient == 0:
                continue
            single = tuple(coefficient if m == harmonic else 0.0 for m in (1, 2))
            for pulse in nc.trig_schedule(1, OMEGA_B, single, h, order, s):
                pulse = dict(pulse)
                if pulse["site"] != -1:
                    pulse["site"] = site
                pulses.append(pulse)
    return pulses


def primitive_counts_conjugated(route, r, order, s):
    """Counts of the earlier schedule, which conjugates every factor by R_y(+-pi/2) (kept for provenance)."""
    h = T / r
    coefficients = (0.0, LAM2) if route == "hybrid" else (E_J, LAM2)
    merged = nc.merge_pulses(trig_schedule_orders(coefficients, h, ORDER_CODES[order], s))
    first, last = merged[0], merged[-1]
    if first["gate"] != "Ry" or last["gate"] != "Ry" or first["value"] + last["value"] != 0:
        raise ValueError("outer-step qubit cancellation premise failed")
    counts = {key: r * value for key, value in Counter(p["gate"] for p in merged).items()}
    counts["Ry"] -= 2 * (r - 1)
    counts["Gaussian"] = {"hybrid": 0, "kin_pot": 5 * r + 2, "gauss_block": 6 * r + 9}[route]
    return counts, (r + 1 if route == "hybrid" else 0)


def primitive_counts(route, r, order, s):
    """Counts with the qubit basis change applied once per circuit (trig_counts.py)."""
    import trig_counts  # noqa: E402
    coefficients = (0.0, LAM2) if route == "hybrid" else (E_J, LAM2)
    counts = trig_counts.trig_counts(2, coefficients, r, ORDER_CODES[order], s)
    counts["Gaussian"] = {"hybrid": 0, "kin_pot": 5 * r + 2, "gauss_block": 6 * r + 9}[route]
    return counts, (r + 1 if route == "hybrid" else 0)


def run_circuit(route, n, r, order, s, prep, reference):
    problem = build_two_site_problem(n, OMEGA_B, E_J, KAPPA, LAM2)
    basis = problem["basis"]
    joint = np.zeros((n, n, 2), complex)
    joint[:, :, 0] = _sg_initial(basis, *prep)
    peak = nc.populations(joint)

    def observe(state):
        for key, value in nc.populations(state).items():
            peak[key] = max(peak[key], value)

    h = T / r
    if route == "gauss_block":
        step, gaussian = make_gauss_block_step(problem, h, order, s, observe)
        # consistency of the decomposed Gaussian step with the exact truncated block
        test = np.zeros_like(joint)
        test[:, :, 0] = joint[:, :, 0]
        exact_block = build_two_site_problem(n, OMEGA_B, 0.0, KAPPA, 0.0)["H_native"]
        exact = expm_multiply(-0.5j * h * exact_block, test[:, :, 0].ravel()).reshape(n, n)
        gauss_check = projective_distance(exact.ravel(), gaussian(test)[:, :, 0].ravel())
    elif route == "kin_pot" and order == 3:
        step = make_kin_pot_step(problem, h, order, s, observe)
        gauss_check = None
    else:
        if route == "hybrid" and order == 3:
            raise ValueError("the hybrid circuit compiles one harmonic, so order code 3 does not apply")
        step = nc.make_step(problem, E_J, KAPPA, LAM2, h, {"hybrid": "H_trig", "kin_pot": "G_trig"}[route],
                            order, s, observer=observe)
        gauss_check = None
    rows, states = [], []
    for k in range(r + 1):
        metrics = nc.reduced_metrics(joint, reference["states"][k])
        obs = nc.two_site_observables(joint, basis, joint=True)
        c12 = obs["C12_real"] + 1j * obs["C12_imag"]
        rows.append(dict(t=k * h, bures=metrics["bures_distance"], ancilla=metrics["ancilla_excited_population"],
                         purity=metrics["system_purity"], C12=c12, S2=obs["S2"],
                         C12_error=abs(c12 - reference["C12"][k]), S2_error=abs(obs["S2"] - reference["S2"][k])))
        states.append(joint.copy())
        if k < r:
            joint = step(joint)
    return rows, np.array(states), peak, gauss_check


def load_reference(n, r, prep):
    with np.load(reference_path(n, r, prep)) as a:
        return {key: a[key] for key in a.files}


def evaluate(route, r, order, s, prep):
    low, high = CUTS
    refs = {n: load_reference(n, r, prep) for n in CUTS}
    out = {}
    for n in CUTS:
        out[n] = run_circuit(route, n, r, order, s, prep, refs[n])
    rows_low, states_low, peak_low, gauss_check = out[low]
    rows_high, states_high, peak_high, _ = out[high]
    embedded = np.zeros((r + 1, high, high, 2), complex)
    embedded[:, :low, :low, :] = states_low
    pad = max(projective_distance(a.ravel(), b.ravel()) for a, b in zip(embedded, states_high))
    target_low = np.zeros((r + 1, high, high), complex)
    target_low[:, :low, :low] = refs[low]["states"].reshape(r + 1, low, low)
    ref_margin = max(projective_distance(a.ravel(), b) for a, b in zip(target_low, refs[high]["states"]))
    cref = float(np.max(abs(refs[low]["C12"] - refs[high]["C12"])))
    sref = float(np.max(abs(refs[low]["S2"] - refs[high]["S2"])))
    cp = max(abs(a["C12"] - b["C12"]) for a, b in zip(rows_low, rows_high))
    sp = max(abs(a["S2"] - b["S2"]) for a, b in zip(rows_low, rows_high))
    uc, us = cref + SOLVER["C12"] + cp, sref + SOLVER["S2"] + sp
    max_bures = max(row["bures"] for row in rows_low)
    total = max_bures + ref_margin + SOLVER["state"] + pad
    boundary = max(peak_low["boundary_band_population"], peak_high["boundary_band_population"])
    c12 = max(row["C12_error"] for row in rows_low)
    s2 = max(row["S2_error"] for row in rows_low)
    acceptance = dict(state=bool(total <= EPS and ref_margin <= EPS / 10 and SOLVER["state"] <= EPS / 10),
                      boundary=bool(boundary <= 1e-6), C12=bool(uc <= 1e-3 and c12 + uc <= 0.01),
                      S2=bool(us <= 5e-3 and s2 + us <= 0.02))
    counts, waits = primitive_counts(route, r, order, s)
    return dict(route=route, r=r, order=order, block_orders=list(ORDER_CODES[order]), s=s, preparation=list(prep), cutoffs=list(CUTS),
                primitive_counts=counts, total_programmable_primitives=int(sum(counts.values())), native_waits=waits,
                state_error=total, max_bures=max_bures, margins=dict(reference=ref_margin, solver=SOLVER["state"], padding=pad,
                                                                     C12=uc, S2=us),
                C12_error=c12 + uc, S2_error=s2 + us, boundary_population=boundary,
                boundary_low=peak_low["boundary_band_population"], boundary_high=peak_high["boundary_band_population"],
                max_ancilla_excited=max(row["ancilla"] for row in rows_low),
                min_system_purity=min(row["purity"] for row in rows_low), gaussian_step_check=gauss_check,
                acceptance=acceptance, all_criteria=bool(all(acceptance.values())),
                series=dict(t=[row["t"] for row in rows_low], bures=[row["bures"] for row in rows_low],
                            C12_error=[row["C12_error"] for row in rows_low], S2_error=[row["S2_error"] for row in rows_low],
                            C12_real=[float(row["C12"].real) for row in rows_low], C12_imag=[float(row["C12"].imag) for row in rows_low],
                            S2=[row["S2"] for row in rows_low]))


def environment():
    return dict(python=sys.executable, python_version=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                threads=os.environ.get("OPENBLAS_NUM_THREADS"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", choices=["references", "scan", "merge"])
    parser.add_argument("--route", choices=["hybrid", "kin_pot", "gauss_block"])
    parser.add_argument("--orders", default="1,2")
    parser.add_argument("--r", default="")
    parser.add_argument("--s", type=int, default=1)
    parser.add_argument("--prep", default="1,0")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    prep = tuple(float(x) for x in args.prep.split(","))
    steps = [int(x) for x in args.r.split(",") if x]
    if args.job == "references":
        compute_references(steps, prep)
        return
    if args.job == "scan":
        PARTS.mkdir(parents=True, exist_ok=True)
        compute_references(steps, prep)
        results = []
        for r in steps:
            for order in [int(x) for x in args.orders.split(",")]:
                started = time.perf_counter()
                row = evaluate(args.route, r, order, args.s, prep)
                row["wall_seconds"] = time.perf_counter() - started
                results.append(row)
                acc = "PASS" if row["all_criteria"] else "fail " + ",".join(k for k, v in row["acceptance"].items() if not v)
                print(f"{args.route} r={r} o={order} s={args.s} prims={row['total_programmable_primitives']} "
                      f"state={row['state_error']:.4f} C12={row['C12_error']:.4f} S2={row['S2_error']:.4f} "
                      f"bnd={row['boundary_population']:.2e} {acc} ({row['wall_seconds']:.0f} s)", flush=True)
                name = f"{args.route}_prep{prep_tag(prep)}{args.tag}.json"
                (PARTS / name).write_text(json.dumps(dict(environment=environment(), rows=results), indent=1))
        return
    unique = {}
    for path in sorted(PARTS.glob("*.json")):
        for row in json.loads(path.read_text())["rows"]:
            unique[(row["route"], row["r"], row["order"], row["s"], tuple(row["preparation"]))] = row
    rows = list(unique.values())
    best = {}
    for row in rows:
        if row["preparation"] != [1.0, 0.0] or not row["all_criteria"]:
            continue
        key = row["route"]
        if key not in best or (row["total_programmable_primitives"], row["state_error"]) < (
                best[key]["total_programmable_primitives"], best[key]["state_error"]):
            best[key] = row
    summary = dict(description=__doc__.split("\n\n")[0], rows=rows,
                   cheapest_passing={k: dict(r=v["r"], order=v["order"], s=v["s"], primitives=v["total_programmable_primitives"],
                                             state_error=v["state_error"]) for k, v in best.items()},
                   environment=environment())
    (DATA / "dsg_step_scan.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary["cheapest_passing"].items():
        print(k, v)


if __name__ == "__main__":
    main()
