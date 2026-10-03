"""Bounded check of the single W=0.3 Kerr-lattice example.

From the repository root:
    python common_code/check_number_diagonal.py

The original legacy CLI supplies the cutoff comparison. The independent phase
check constructs integer occupation numbers and binomial coefficients instead
of ladder operators: theta_n = -W dt choose(n, 3). Operators on distinct sites
commute, so their tensor-product SNAP phases equal exp(-i dt H_corr). The
optional step refinement compares only cutoff 12 at 320 steps with its own
finite-cutoff target. For Strang splitting the state error is O(dt^2), hence
the leading infidelity is O(dt^4) and should fall by about 16 when dt is halved.
The whole check is limited to 120 seconds and a 1 GiB sampled worker RSS.
Results are written to common_data/number_diagonal_check, with the legacy
diagnostic previews in its plots subdirectory.
"""

import csv
import importlib.metadata
import json
import math
import os
import platform
import resource
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "common_data" / "number_diagonal_check"
SOURCE = REPO / "legacy" / "nlse_fock_cutoff_hybrid.py"
PYTHON = sys.executable
COMMAND = [
    PYTHON,
    str(SOURCE.relative_to(REPO)),
    "--sites",
    "3",
    "--cutoffs",
    "6,8,10,12",
    "--ref-cutoff",
    "16",
    "--omega",
    "0",
    "--J",
    "0.5",
    "--U",
    "0.8",
    "--W-values",
    "0.3",
    "--T",
    "4",
    "--n-time",
    "81",
    "--trotter-steps",
    "160",
    "--amp",
    "1.25",
    "--sigma",
    "0.9",
    "--k0",
    "0.6",
    "--outdir",
    str(OUT),
]


def write_record(record):
    (OUT / "validation.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def main():
    started = time.monotonic()
    deadline = started + 120
    OUT.mkdir(exist_ok=True)
    record = dict(
        status="running",
        cwd=str(REPO),
        command=shlex.join(COMMAND),
        validation_command=f"{PYTHON} common_code/check_number_diagonal.py",
        validation_source=str(Path(__file__).resolve().relative_to(REPO)),
        source=str(SOURCE.relative_to(REPO)),
        environment=dict(
            python=sys.executable, python_version=platform.python_version()
        ),
        resource_limits=dict(total_wall_seconds=120, sampled_worker_rss_bytes=1024**3),
        parameters=dict(
            sites=3,
            cutoffs=[6, 8, 10, 12],
            reference_cutoff=16,
            omega=0.0,
            J=0.5,
            U=0.8,
            W=0.3,
            T=4.0,
            steps=160,
            saved_times=81,
            amp=1.25,
            sigma=0.9,
            k0=0.6,
            boundary="open",
        ),
        metric_definitions=dict(
            max_infid_native_vs_ref="Maximum 1-|overlap|^2 on 81 saved times, embedded native state versus Nref=16 target.",
            max_infid_cutoff_floor="Embedded finite-cutoff target versus Nref=16 target on the same times.",
            max_infid_hybrid_vs_ref="Embedded Strang state versus Nref=16 target on the same times.",
            max_infid_hybrid_vs_finite_target="Strang state versus the target at the same cutoff on the same times.",
            reference_status="Finite reference at cutoff 16, without a higher-cutoff or independent-solver bound.",
        ),
    )
    write_record(record)
    (OUT / "artifact_audit.json").unlink(missing_ok=True)
    environment = dict(os.environ, MPLBACKEND="Agg")
    record["environment"]["MPLBACKEND"] = "Agg"
    record["environment"]["OPENBLAS_NUM_THREADS"] = environment.get(
        "OPENBLAS_NUM_THREADS"
    )
    print("Running bounded legacy command:", shlex.join(COMMAND), flush=True)
    peak_rss = 0
    batch_started = time.monotonic()
    with (OUT / "legacy_run.log").open("w") as log:
        process = subprocess.Popen(
            COMMAND,
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            rss_text = subprocess.run(
                ["/bin/ps", "-o", "rss=", "-p", str(process.pid)],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if rss_text:
                peak_rss = max(peak_rss, int(rss_text) * 1024)
            if time.monotonic() >= deadline or peak_rss >= 1024**3:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record.update(
                    status="stopped_at_resource_limit",
                    sampled_worker_peak_rss_bytes=peak_rss,
                    elapsed_seconds=time.monotonic() - started,
                )
                write_record(record)
                raise SystemExit("Stopped at the declared resource limit")
            time.sleep(0.25)
    record["legacy_run"] = dict(
        returncode=process.returncode,
        wall_seconds=time.monotonic() - batch_started,
        sampled_peak_rss_bytes=peak_rss,
    )
    if process.returncode:
        record.update(
            status="legacy_failed", elapsed_seconds=time.monotonic() - started
        )
        write_record(record)
        raise SystemExit(process.returncode)
    print("Legacy batch completed", record["legacy_run"], flush=True)
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    for name in (
        "max_infid_native_vs_W.png",
        "max_infid_hybrid_vs_W.png",
        "max_infid_cutoff_floor_vs_W.png",
        "cutoff_convergence_at_largest_W.png",
    ):
        (OUT / name).replace(plots / name)

    def timeout(signum, frame):
        record.update(
            status="stopped_at_resource_limit",
            elapsed_seconds=time.monotonic() - started,
        )
        write_record(record)
        raise SystemExit("Stopped at the total wall-time limit")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
    sys.path.insert(0, str(REPO))
    import numpy as np
    import scipy
    from legacy import nlse_fock_cutoff_hybrid as model

    record["environment"].update(
        numpy=np.__version__,
        scipy=scipy.__version__,
        matplotlib=importlib.metadata.version("matplotlib"),
        numpy_origin=np.__file__,
        scipy_origin=scipy.__file__,
    )
    with (OUT / "summary.csv").open() as handle:
        record["legacy_results"] = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]

    phase_started = time.monotonic()
    phase_rows = []
    for cutoff in (6, 8, 10, 12, 16):
        native, correction, target, diagonal = model.build_hamiltonians(
            3, cutoff, 0, 0.5, 0.8, 0.3, False
        )
        occupations = np.array([math.comb(n, 3) for n in range(cutoff)], dtype=int)
        expected_energy = (
            0.3
            * (
                occupations[:, None, None]
                + occupations[None, :, None]
                + occupations[None, None, :]
            ).ravel()
        )
        local_phase = np.exp(-1j * 0.3 * (4 / 160) * occupations)
        tensor_phase = np.kron(np.kron(local_phase, local_phase), local_phase)
        phase_rows.append(
            dict(
                cutoff=cutoff,
                target_dimension=target.shape[0],
                target_nnz=target.nnz,
                maximum_diagonal_energy_difference=float(
                    np.max(abs(diagonal - expected_energy))
                ),
                maximum_tensor_phase_difference=float(
                    np.max(abs(tensor_phase - np.exp(-1j * (4 / 160) * diagonal)))
                ),
                maximum_phase_modulus_defect=float(np.max(abs(abs(tensor_phase) - 1))),
                correction_is_diagonal=bool(
                    (correction - scipy.sparse.diags(diagonal)).nnz == 0
                ),
            )
        )
    record["independent_phase_check"] = dict(
        dt=4 / 160,
        SNAP_convention="S(theta)=diag(exp(i theta_n))",
        theta_n="-W dt n(n-1)(n-2)/6",
        rows=phase_rows,
        wall_seconds=time.monotonic() - phase_started,
    )

    if deadline - time.monotonic() > 10:
        refine_started = time.monotonic()
        native, _, target, diagonal = model.build_hamiltonians(
            3, 12, 0, 0.5, 0.8, 0.3, False
        )
        initial = model.initial_coherent_wavepacket(3, 12, 1.25, 0.9, 0.6)
        times = np.linspace(0, 4, 81)
        exact = model.evolve_exact(target, initial, times)
        refine_times, refined = model.evolve_hybrid_strang(
            native, diagonal, initial, 4, 81, 320
        )
        maximum, final_fidelity, fidelities = model.max_infidelity(exact, refined)
        coarse = next(row for row in record["legacy_results"] if row["cutoff"] == 12)
        np.savez_compressed(
            OUT / "refinement_cutoff12_steps320.npz",
            times=refine_times,
            fid_hybrid_vs_finite_target=fidelities,
        )
        record["step_refinement"] = dict(
            status="completed",
            cutoff=12,
            steps=320,
            dt=4 / 320,
            max_infid_hybrid_vs_finite_target=maximum,
            final_fid_hybrid_vs_finite_target=final_fidelity,
            coarse_to_fine_infidelity_ratio=coarse["max_infid_hybrid_vs_finite_target"]
            / maximum,
            maximum_target_norm_defect=float(
                np.max(abs(np.linalg.norm(exact, axis=1) - 1))
            ),
            maximum_refined_norm_defect=float(
                np.max(abs(np.linalg.norm(refined, axis=1) - 1))
            ),
            wall_seconds=time.monotonic() - refine_started,
        )
    else:
        record["step_refinement"] = dict(status="not_run_insufficient_time")
    signal.setitimer(signal.ITIMER_REAL, 0)
    rss_scale = 1 if sys.platform == "darwin" else 1024
    record["process_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
    )
    record.update(status="completed", elapsed_seconds=time.monotonic() - started)
    write_record(record)

    # Check that the archived traces contain the reported sampled maxima.
    metric_fields = {
        "fid_native_vs_ref": "max_infid_native_vs_ref",
        "fid_cutoff_floor": "max_infid_cutoff_floor",
        "fid_hybrid_vs_ref": "max_infid_hybrid_vs_ref",
        "fid_hybrid_vs_finite_target": "max_infid_hybrid_vs_finite_target",
    }
    trace_checks = []
    for row in record["legacy_results"]:
        cutoff = int(row["cutoff"])
        with np.load(OUT / f"trace_cutoff_{cutoff}_W_0.3.npz") as trace:
            if not np.array_equal(trace["times"], np.linspace(0.0, 4.0, 81)):
                raise ValueError(f"Unexpected stored time grid at cutoff {cutoff}")
            difference = max(
                abs(float(np.max(1 - trace[series])) - row[stored])
                for series, stored in metric_fields.items()
            )
            if difference != 0:
                raise ValueError(
                    f"Stored maxima differ from the summary at cutoff {cutoff}"
                )
            trace_checks.append(
                dict(
                    cutoff=cutoff,
                    saved_time_count=81,
                    maximum_summary_difference=difference,
                )
            )
    refinement_matches = None
    if record["step_refinement"]["status"] == "completed":
        with np.load(OUT / "refinement_cutoff12_steps320.npz") as trace:
            refinement_matches = (
                np.array_equal(trace["times"], np.linspace(0.0, 4.0, 81))
                and float(np.max(1 - trace["fid_hybrid_vs_finite_target"]))
                == record["step_refinement"]["max_infid_hybrid_vs_finite_target"]
            )
        if not refinement_matches:
            raise ValueError("Refinement trace differs from its summary")
    files = {
        str(path.relative_to(OUT)): dict(
            bytes=path.stat().st_size,
        )
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "artifact_audit.json"
    }
    audit = dict(
        status="PASS",
        metric_trace_checks=trace_checks,
        refinement_trace_matches_summary=refinement_matches,
        matplotlib_version=record["environment"]["matplotlib"],
        total_archived_bytes_before_audit=sum(
            entry["bytes"] for entry in files.values()
        ),
        files=files,
    )
    (OUT / "artifact_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
