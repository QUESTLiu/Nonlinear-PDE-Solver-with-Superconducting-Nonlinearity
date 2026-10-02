"""Frozen E2 comparison. Restart selection uses training loss only."""

import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.sparse.linalg import expm_multiply

from duffing_code import duffing_padded_hybrid_validation as df
from duffing_code.native_comparison import (
    constructive_pair,
    kerr_snap,
    residual_factors,
)
from common_code.metrics import projective_distance, family_action_sum
from common_code.records import provenance, write_json, peak_memory_bytes


def evaluation_inputs():
    inputs = []
    for alpha, name in [
        (1.75, "real1.75"),
        (2.0, "real2"),
        (1.75j, "imag1.75"),
        (2j, "imag2"),
        (1.75 * np.exp(1j * np.pi / 4), "diagonal1.75"),
        (2 * np.exp(1j * np.pi / 4), "diagonal2"),
    ]:
        inputs.append(
            (
                name,
                df.coherent_state(80, alpha)[0],
                "development" if name.startswith("real") else "independent_validation",
            )
        )
    fock = np.eye(80, dtype=complex)
    for name, state in [
        ("vacuum", fock[:, 0]),
        ("one", fock[:, 1]),
        ("zero_plus_one", (fock[:, 0] + fock[:, 1]) / np.sqrt(2)),
        ("zero_plus_i_two", (fock[:, 0] + 1j * fock[:, 2]) / np.sqrt(2)),
    ]:
        inputs.append((name, state, "independent_validation"))
    return inputs


def evaluate_pair(
    out,
    r,
    layers,
    vector,
    inputs,
    deadline=None,
    boundary_at_both_cutoffs=False,
    padding_cutoffs=(96, 112),
):
    low_pad, high_pad = padding_cutoffs
    if not 80 <= low_pad < high_pad:
        raise ValueError("padding cutoffs must increase and contain the N80 reference")
    h = 1 / r
    times = np.linspace(0, 1, r + 1)
    common = np.arange(0, r + 1, r // 10)
    operators = {
        n: df.hard_sandwiched_operators(n, 1.0, 0.1)
        for n in sorted({64, 80, low_pad, high_pad})
    }
    gates = {n: df.build_fixed_gates(operators[n], h) for n in padding_cutoffs}
    residual = {
        n: df.dense_ansatz_unitary(n, 48, layers, vector) for n in padding_cutoffs
    }
    factors = {n: residual_factors(n, 48, layers, vector) for n in padding_cutoffs}
    w = (
        gates[low_pad].k_half
        @ gates[low_pad].v2_half
        @ residual[low_pad]
        @ gates[low_pad].v2_half
        @ gates[low_pad].k_half
    )
    skhalf = kerr_snap(np.diag(operators[low_pad]["H_K"]).real, h / 2, 48)[1]
    sk = (
        skhalf
        @ gates[low_pad].v2_half
        @ residual[low_pad]
        @ gates[low_pad].v2_half
        @ skhalf
    )
    target_step = df.hermitian_exponential(operators[low_pad]["H_D"], -1j * h)
    secondary = None
    if (r, layers) == (10, 8):
        secondary = {
            name: float(
                np.linalg.norm(
                    (matrix - np.exp(1j * vector[-1]) * target_step)[:, :24], 2
                )
            )
            for name, matrix in (("hybrid", w), ("replacement", sk))
        }
    rows = []
    for label, state, role in inputs:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("Duffing validation time cap reached")
        low = state[:64].copy()
        low /= np.linalg.norm(low)
        reference = {
            64: df.evolve_hermitian(operators[64]["H_D"], low, times),
            80: df.evolve_hermitian(operators[80]["H_D"], state, times),
        }
        reference_margin = max(
            projective_distance(
                df.embed_state(reference[64][:, j], 80), reference[80][:, j]
            )
            for j in range(r + 1)
        )
        independent = expm_multiply(
            -1j * operators[80]["H_D"],
            state,
            start=0.0,
            stop=1.0,
            num=r + 1,
            traceA=-1j * np.trace(operators[80]["H_D"]),
        )
        solver_margin = max(
            projective_distance(reference[80][:, j], independent[j])
            for j in range(r + 1)
        )
        pairs = {
            n: constructive_pair(
                operators[n],
                gates[n],
                residual[n],
                df.embed_state(state, n),
                h,
                r,
                48,
                factors[n],
            )
            for n in padding_cutoffs
        }
        initial = df.embed_state(state, low_pad)
        same_space_reference = df.apply_repeated_step(initial, target_step, r)
        controls = {
            "native": df.diagonal_kerr_trajectory(
                operators[low_pad]["H_K"], initial, times
            ),
            "gaussian": df.apply_repeated_step(initial, gates[low_pad].k_v2_step, r),
            "exact_split": df.apply_repeated_step(
                initial, gates[low_pad].exact_split_step, r
            ),
        }
        arrays = {
            "t": times,
            "reference_access_population": np.sum(abs(reference[80][24:]) ** 2, axis=0),
        }
        for name, states in controls.items():
            errors = np.array(
                [
                    projective_distance(
                        df.embed_state(reference[80][:, j], low_pad), states[:, j]
                    )
                    for j in range(r + 1)
                ]
            )
            arrays[name + "_distance"] = errors
            rows.append(
                dict(
                    input=label,
                    input_role=role,
                    route=name,
                    max_common_distance=float(errors[common].max()),
                    max_endpoint_distance=float(errors.max()),
                )
            )
        for name, matrix in (("hybrid", w), ("replacement", sk)):
            states = pairs[low_pad][name]
            error = np.array(
                [
                    projective_distance(
                        df.embed_state(reference[80][:, j], low_pad), states[:, j]
                    )
                    for j in range(r + 1)
                ]
            )
            padding_distances = np.array(
                [
                    projective_distance(
                        df.embed_state(states[:, j], high_pad),
                        pairs[high_pad][name][:, j],
                    )
                    for j in range(r + 1)
                ]
            )
            padding = float(padding_distances.max())
            actions = family_action_sum(
                lambda x: matrix @ x,
                lambda x: target_step @ x,
                initial,
                r,
                phase=float(vector[-1]),
            )
            same_space_error = np.array(
                [
                    projective_distance(same_space_reference[:, j], states[:, j])
                    for j in range(r + 1)
                ]
            )
            total = float(
                error[common].max() + reference_margin + solver_margin + padding
            )
            pop = pairs[low_pad]["gate_populations"][name]
            high_pop = pairs[high_pad]["gate_populations"][name]
            both_boundary = max(
                pop["boundary_population"], high_pop["boundary_population"]
            )
            boundary = (
                both_boundary
                if boundary_at_both_cutoffs
                else pop["boundary_population"]
            )
            acceptance = {
                str(th): bool(
                    total <= np.sqrt(th)
                    and reference_margin <= np.sqrt(th) / 10
                    and solver_margin <= np.sqrt(th) / 10
                    and boundary <= 1e-6
                )
                for th in (1e-2, 1e-3, 1e-4)
            }
            rows.append(
                dict(
                    input=label,
                    input_role=role,
                    route=name,
                    max_common_distance=float(error[common].max()),
                    max_endpoint_distance=float(error.max()),
                    reference_margin=reference_margin,
                    solver_margin=solver_margin,
                    padding_margin=padding,
                    total_amplitude_estimate=total,
                    action_sum=float(actions.sum()),
                    same_space_final_distance=float(same_space_error[-1]),
                    max_same_space_distance=float(same_space_error.max()),
                    **pop,
                    boundary_population_high_cutoff=high_pop["boundary_population"],
                    boundary_population_both_cutoffs=both_boundary,
                    boundary_acceptance_scope="both cutoffs"
                    if boundary_at_both_cutoffs
                    else "lower cutoff",
                    acceptance=acceptance,
                    replacement_bound=pairs[low_pad]["replacement_bound"]
                    if name == "replacement"
                    else None,
                )
            )
            arrays[name + "_distance"] = error
            arrays[name + "_action_errors"] = actions
            arrays[name + "_same_space_distance"] = same_space_error
            arrays[name + "_padding_distance"] = padding_distances
        np.savez_compressed(out / f"{label}.npz", **arrays)
        write_json(
            out / f"{label}_gate_locations.json",
            dict(
                replacement_locations=pairs[low_pad]["locations"],
                bound=pairs[low_pad]["replacement_bound"],
            ),
        )
    summary = {
        name: {
            str(th): all(
                row["acceptance"][str(th)] for row in rows if row["route"] == name
            )
            for th in (1e-2, 1e-3, 1e-4)
        }
        for name in ("hybrid", "replacement")
    }
    return dict(
        r=r,
        layers=layers,
        padding_cutoffs=list(padding_cutoffs),
        reference_cutoffs=[64, 80],
        replacement_bound_cutoff=low_pad,
        rows=rows,
        family_acceptance=summary,
        secondary_rectangular_norm=secondary,
        resources=dict(
            system_modes=1,
            residual_snap=r * layers,
            residual_displacements=r * layers,
            gaussian_half_pulses=2 * r,
            merged_native_waits=r + 1,
            replacement_extra_snap=r + 1,
            native_simulated_time=1.0,
        ),
        evidence_scope="finite-family numerical validation with empirical reference and padding differences",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.outdir.mkdir(parents=True, exist_ok=args.resume)
    if (args.outdir / "summary.json").exists():
        raise FileExistsError("completed comparisons cannot be resumed")
    identity = provenance()
    started, cpu = time.perf_counter(), time.process_time()
    runs = []
    chosen = []
    optimization_spent = 0.0
    for r in (10, 20):
        ops = df.hard_sandwiched_operators(96, 1.0, 0.1)
        gates = df.build_fixed_gates(ops, 1 / r)
        target_step = df.hermitian_exponential(ops["H_D"], -1j / r)
        x, y, training_rows = df.build_training_snapshots(
            96, [1.0, 1.5, 2.25], r, gates, target_step, 80
        )
        for layers in (4, 8, 12):
            candidates = []
            for seed in (240731, 240732, 240733):
                run_dir = args.outdir / f"r{r}_L{layers}_seed{seed}"
                run_dir.mkdir(exist_ok=args.resume)
                if args.resume and (run_dir / "optimization.json").exists():
                    info = json.loads((run_dir / "optimization.json").read_text())
                    with np.load(run_dir / "optimizer.npz") as data:
                        vector = data["parameter_vector"].copy()
                elif (r, layers, seed) == (10, 8, 240731):
                    with np.load(args.pilot / "E2_optimizer.npz") as data:
                        vector = data["parameter_vector"].copy()
                    info = json.loads((args.pilot / "summary.json").read_text())[
                        "optimization"
                    ]
                    shutil.copyfile(
                        args.pilot / "E2_optimizer.npz", run_dir / "optimizer.npz"
                    )
                    info = dict(
                        info, reused_pilot=True, reuse_basis=protocol["E2_pilot_reuse"]
                    )
                else:
                    compiler = df.SnapDisplacementCompiler(96, 48, layers, x, y, 1e-8)
                    remaining = 16200 - optimization_spent
                    if remaining <= 0:
                        raise RuntimeError("total E2 optimization cap reached")
                    vector, info = df.optimize_common_phase(
                        compiler, seed, min(900.0, remaining), run_dir / "optimizer.npz"
                    )
                optimization_spent += info["elapsed_seconds"]
                write_json(run_dir / "optimization.json", info)
                runs.append(dict(info, r=r, layers=layers, seed=seed))
                candidates.append((info["objective"], seed, vector))
                write_json(
                    args.outdir / "progress.json",
                    dict(
                        phase="training",
                        runs=runs,
                        optimization_seconds=optimization_spent,
                    ),
                )
            loss, seed, vector = min(candidates, key=lambda item: item[0])
            chosen.append((r, layers, seed, vector))
    # The complete selection is written before the independent family is constructed.
    write_json(
        args.outdir / "selection.json",
        dict(
            criterion="minimum training objective within (r,L)",
            selected=[dict(r=r, layers=l, seed=s) for r, l, s, v in chosen],
        ),
    )
    inputs = evaluation_inputs()
    comparisons = []
    for r, layers, seed, vector in chosen:
        directory = args.outdir / f"evaluation_r{r}_L{layers}"
        directory.mkdir()
        comparison = evaluate_pair(directory, r, layers, vector, inputs)
        comparison["selected_seed"] = seed
        write_json(directory / "summary.json", comparison)
        comparisons.append(comparison)
        write_json(
            args.outdir / "progress.json",
            dict(
                phase="validation",
                completed=len(comparisons),
                total=6,
                optimization_seconds=optimization_spent,
            ),
        )
    write_json(
        args.outdir / "summary.json",
        dict(
            protocol=protocol,
            provenance=identity,
            runs=runs,
            comparisons=comparisons,
            optimization_seconds=optimization_spent,
            wall_seconds=time.perf_counter() - started,
            cpu_seconds=time.process_time() - cpu,
            peak_memory_bytes=peak_memory_bytes(),
        ),
    )


if __name__ == "__main__":
    main()
