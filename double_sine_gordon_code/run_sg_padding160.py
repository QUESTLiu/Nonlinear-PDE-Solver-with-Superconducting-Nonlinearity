"""Focused 128/160 refinement of three existing trigonometric circuits."""

import time
import json
import shutil
import numpy as np
from scipy.integrate import solve_ivp
from double_sine_gordon_code.run_snapd_end_to_end import (
    build_two_site_problem,
    exact_trajectory,
    embed_trajectory,
)
from common_code.run_pilot import _sg_initial, _sg_observations, evaluate_sg
from common_code.metrics import projective_distance
from common_code.records import ROOT, write_json, provenance, peak_memory_bytes


def main():
    out = ROOT / "double_sine_gordon_data/sg160"
    out.mkdir(parents=True, exist_ok=True)
    reference = out / "reference"
    reference.mkdir(exist_ok=False)
    comparison = out / "comparison"
    comparison.mkdir(exist_ok=False)
    started, cpu = time.perf_counter(), time.process_time()
    identity = provenance()
    horizon = 5.2 / np.sqrt(3)
    times = np.linspace(0, horizon, 257)
    old = ROOT / "double_sine_gordon_data/sg128/reference"
    old_summary = json.loads((old / "summary.json").read_text())
    lower_record = next(
        row
        for row in old_summary["records"]
        if row["N"] == 128 and row["lambda2"] == 0.3
    )
    shutil.copyfile(
        old / "E4_reference_lambda0.3_N128.npz",
        reference / "E4_reference_lambda0.3_N128.npz",
    )
    problem = build_two_site_problem(160, np.sqrt(3), 1.0, 1.0, 0.3)
    initial = _sg_initial(problem["basis"])
    exact = exact_trajectory(problem["H_target"], initial, horizon, 256)
    observed = _sg_observations(exact, problem["basis"])
    checked = solve_ivp(
        lambda t, x: -1j * (problem["H_target"] @ x),
        (0.0, horizon),
        initial.ravel(),
        method="DOP853",
        t_eval=times,
        rtol=2e-11,
        atol=2e-13,
    )
    if not checked.success:
        raise RuntimeError(checked.message)
    solver_observed = _sg_observations(checked.y.T, problem["basis"])
    c = lambda data: data["C12_real"] + 1j * data["C12_imag"]
    solver = dict(
        state_distance=max(
            projective_distance(a, b) for a, b in zip(exact, checked.y.T)
        ),
        C12=float(np.max(abs(c(observed) - c(solver_observed)))),
        S2=float(np.max(abs(observed["S2"] - solver_observed["S2"]))),
    )
    steps = exact_trajectory(problem["H_target"], initial, horizon, 20)
    cache = {(160, 20, (1.0, 0.0)): steps, (160, 10, (1.0, 0.0)): steps[::2]}
    with np.load(reference / "E4_reference_lambda0.3_N128.npz") as data:
        low_steps = data["target_step_states"].copy()
        low_obs = {
            key: data["target_" + key].copy() for key in ("C12_real", "C12_imag", "S2")
        }
    cache[(128, 10, (1.0, 0.0))] = low_steps
    margins = dict(
        state_distance=max(
            projective_distance(a, b)
            for a, b in zip(embed_trajectory(low_steps, 128, 160), steps[::2])
        ),
        C12=float(np.max(abs(c(low_obs) - c(observed)))),
        S2=float(np.max(abs(low_obs["S2"] - observed["S2"]))),
    )
    np.savez_compressed(
        reference / "E4_reference_lambda0.3_N160.npz",
        t=times,
        step_t=np.linspace(0, horizon, 11),
        target_step_states=steps[::2],
        target_20step_states=steps,
        **{"target_" + key: value for key, value in observed.items()},
    )
    resolved = (
        margins["state_distance"] + solver["state_distance"] <= np.sqrt(0.001) / 10
        and margins["C12"] + solver["C12"] <= 0.001
        and margins["S2"] + solver["S2"] <= 0.005
    )
    write_json(
        reference / "summary.json",
        dict(
            compared_cutoffs=[128, 160],
            records=[lower_record, dict(N=160, lambda2=0.3, solver=solver)],
            empirical_cutoff_estimates=margins,
            resolution_ok=resolved,
            provenance=identity,
        ),
    )
    if not resolved:
        raise RuntimeError("reference refinement does not resolve original criteria")
    del exact, checked, solver_observed
    source = ROOT / "double_sine_gordon_data/sg_formal/comparison"
    reused = json.loads((source / "summary.json").read_text())
    reused["directory"] = source
    # Trigonometric circuits do not consume a SNAP parameter vector.
    vector = np.zeros(27)
    rows = []
    for r, configs in (
        (10, [("H_trig", 2, 1)]),
        (20, [("G_trig", 1, 1), ("G_trig", 2, 1)]),
    ):
        result = evaluate_sg(
            comparison,
            vector,
            1,
            r,
            reference,
            configs,
            cache,
            extended_names=True,
            deadline=started + 900,
            validation_cutoffs=(128, 160),
            reuse_low=reused,
        )
        rows.extend(result["configurations"])
    write_json(
        out / "summary.json",
        dict(
            configurations=rows,
            validation_cutoffs=[128, 160],
            reference=margins,
            solver=solver,
            scope="User-requested same-circuit padding refinement after the fixed formal scan. Three representative configurations only; no new training or threshold changes.",
            wall_seconds=time.perf_counter() - started,
            cpu_seconds=time.process_time() - cpu,
            peak_memory_bytes=peak_memory_bytes(),
            provenance=identity,
        ),
    )


if __name__ == "__main__":
    main()
