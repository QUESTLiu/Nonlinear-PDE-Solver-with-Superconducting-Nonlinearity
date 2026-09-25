#!/usr/bin/env python3
"""
Section 4.3 -- Validate the Duffing--Kerr split and produce publication-ready diagnostics.

The script uses only NumPy and the Python standard library.  It deliberately
does not require pandas, scipy, or matplotlib.  If gnuplot is available, the
companion LSF script converts the generated plotting file to PDF and PNG.

Conventions
-----------
* hbar = m = 1.
* N denotes the Hilbert-space dimension:
      Pi_N = sum_{n=0}^{N-1} |n><n|.
* All hard-truncated matrices are constructed as Pi_N A Pi_N from an extended
  Fock space.  This avoids the incorrect upper-boundary algebra obtained by
  forming (a_N + a_N^dagger)^4 directly in an N-dimensional space.
* The norm table fixes the energy zero exactly as
      H_D = omega0 (n + 1/2) + lambda (a + a^dagger)^4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Duffing correction norms, convergence, and bounds."
    )
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--nt", type=int, default=401)
    parser.add_argument("--dim", type=int, default=80)
    parser.add_argument(
        "--bench-xi", type=float, nargs="+", default=[1.75, 2.00]
    )
    parser.add_argument("--xi-min", type=float, default=0.50)
    parser.add_argument("--xi-max", type=float, default=2.50)
    parser.add_argument("--n-xi", type=int, default=17)
    parser.add_argument(
        "--norm-cutoffs", type=int, nargs="+", default=[16, 24, 32, 40, 48]
    )
    parser.add_argument(
        "--convergence-cutoffs",
        type=int,
        nargs="+",
        default=[16, 24, 32, 40, 48, 64],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/duffing_correction_validation"),
    )
    return parser.parse_args()


def ladder(dim: int) -> np.ndarray:
    a = np.zeros((dim, dim), dtype=np.complex128)
    n = np.arange(1, dim, dtype=float)
    a[np.arange(dim - 1), np.arange(1, dim)] = np.sqrt(n)
    return a


def operators(
    dim: int, omega0: float, beta: float, pad: int = 4
) -> Dict[str, np.ndarray | float]:
    """Return Pi_dim A Pi_dim matrices built in an extended space."""
    if dim < 5:
        raise ValueError("dim must be at least 5")
    ext = dim + pad
    a_ext = ladder(ext)
    ad_ext = a_ext.conj().T
    n_ext = ad_ext @ a_ext
    eye_ext = np.eye(ext, dtype=np.complex128)
    x_ext = a_ext + ad_ext

    a2 = a_ext @ a_ext
    ad2 = ad_ext @ ad_ext
    a3 = a2 @ a_ext
    ad3 = ad2 @ ad_ext
    a4 = a2 @ a2
    ad4 = ad2 @ ad2

    v_ext = (
        ad4
        + a4
        + 4.0 * (ad3 @ a_ext + ad_ext @ a3)
        + 6.0 * (a2 + ad2)
    )

    lam = beta / (16.0 * omega0**2)
    k_eff = 3.0 * beta / (8.0 * omega0**2)
    omega_eff = omega0 + k_eff

    hd_ext = (
        omega0 * (n_ext + 0.5 * eye_ext)
        + lam * (x_ext @ x_ext @ x_ext @ x_ext)
    )
    hk_ext = (
        omega_eff * n_ext
        + k_eff * (n_ext @ n_ext)
        + (0.5 * omega0 + 3.0 * lam) * eye_ext
    )
    delta_ext = lam * v_ext
    comm_ext = hk_ext @ delta_ext - delta_ext @ hk_ext

    sl = np.s_[:dim, :dim]
    return {
        "H_D": hd_ext[sl],
        "H_K": hk_ext[sl],
        "V_cr": v_ext[sl],
        "Delta": delta_ext[sl],
        "commutator": comm_ext[sl],
        "n": n_ext[sl],
        "lambda": lam,
        "K_eff": k_eff,
        "omega_eff": omega_eff,
    }


def spectral_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def coherent_state(dim: int, xi: float) -> Tuple[np.ndarray, float]:
    """Return normalized truncated coherent state and exact discarded weight."""
    mu = float(xi * xi)
    coeff = np.empty(dim, dtype=np.complex128)
    coeff[0] = math.exp(-0.5 * mu)
    for n in range(1, dim):
        coeff[n] = coeff[n - 1] * xi / math.sqrt(n)
    retained = float(np.vdot(coeff, coeff).real)
    discarded = max(0.0, 1.0 - retained)
    coeff /= math.sqrt(retained)
    return coeff, discarded


def eigensystem(hamiltonian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(hamiltonian)
    return evals, evecs


def evolve_from_eigensystem(
    evals: np.ndarray,
    evecs: np.ndarray,
    psi0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    amplitudes = evecs.conj().T @ psi0
    phases = np.exp(-1j * evals[:, None] * times[None, :])
    return evecs @ (amplitudes[:, None] * phases)


def evolve_diagonal(
    energies: np.ndarray, psi0: np.ndarray, times: np.ndarray
) -> np.ndarray:
    return psi0[:, None] * np.exp(-1j * energies[:, None] * times[None, :])


def cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    dt = np.diff(times)
    out[1:] = np.cumsum(0.5 * dt * (values[:-1] + values[1:]))
    return out


def trajectory_metrics(
    ops: Dict[str, np.ndarray | float],
    psi0: np.ndarray,
    times: np.ndarray,
    evals_hd: np.ndarray | None = None,
    evecs_hd: np.ndarray | None = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    hd = np.asarray(ops["H_D"])
    hk = np.asarray(ops["H_K"])
    delta = np.asarray(ops["Delta"])
    vcr = np.asarray(ops["V_cr"])
    n_op = np.asarray(ops["n"])
    lam = float(ops["lambda"])

    if evals_hd is None or evecs_hd is None:
        evals_hd, evecs_hd = eigensystem(hd)
    psi_d = evolve_from_eigensystem(evals_hd, evecs_hd, psi0, times)
    hk_energies = np.real(np.diag(hk))
    psi_k = evolve_diagonal(hk_energies, psi0, times)

    overlaps = np.sum(np.conj(psi_d) * psi_k, axis=0)
    fidelity = np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)
    infidelity = 1.0 - fidelity

    delta_psi = delta @ psi_d
    hd_psi = hd @ psi_d
    mean_delta = np.sum(np.conj(psi_d) * delta_psi, axis=0)
    mean_hd = np.sum(np.conj(psi_d) * hd_psi, axis=0)
    centered_delta = delta_psi - psi_d * mean_delta[None, :]
    centered_hd = hd_psi - psi_d * mean_hd[None, :]
    delta_speed = np.linalg.norm(centered_delta, axis=0)
    total_speed = np.linalg.norm(centered_hd, axis=0)
    projective_ratio = np.divide(
        delta_speed,
        total_speed,
        out=np.full_like(delta_speed, np.nan),
        where=total_speed > 100.0 * np.finfo(float).eps,
    )

    v_on_k = vcr @ psi_k
    v_action = np.linalg.norm(v_on_k, axis=0)
    duhamel_integral = cumulative_trapezoid(v_action, times)
    duhamel_infidelity_bound = (abs(lam) * duhamel_integral) ** 2

    n_psi = n_op @ psi_d
    n_expect = np.real(np.sum(np.conj(psi_d) * n_psi, axis=0))
    top_levels = min(4, psi_d.shape[0])
    top_population = np.sum(np.abs(psi_d[-top_levels:, :]) ** 2, axis=0)

    summary = {
        "max_native_infidelity": float(np.max(infidelity)),
        "final_native_infidelity": float(infidelity[-1]),
        "max_projective_correction_ratio": float(np.nanmax(projective_ratio)),
        "initial_projective_correction_ratio": float(projective_ratio[0]),
        "final_duhamel_infidelity_bound": float(duhamel_infidelity_bound[-1]),
        "max_duhamel_infidelity_bound": float(
            np.max(duhamel_infidelity_bound)
        ),
        "max_top4_population": float(np.max(top_population)),
        "min_n_expect": float(np.min(n_expect)),
        "max_n_expect": float(np.max(n_expect)),
    }
    series = {
        "psi_D": psi_d,
        "psi_K": psi_k,
        "fidelity": fidelity,
        "infidelity": infidelity,
        "projective_ratio": projective_ratio,
        "v_action_on_kerr": v_action,
        "duhamel_infidelity_bound": duhamel_infidelity_bound,
        "n_expect": n_expect,
        "top4_population": top_population,
    }
    return summary, series


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    field_list = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def gnuplot_script(output_dir: Path) -> str:
    data = (output_dir / "xi_scan.csv").as_posix()
    pdf = (output_dir / "duffing_benchmark_scan.pdf").as_posix()
    png = (output_dir / "duffing_benchmark_scan.png").as_posix()
    common = f"""
set datafile separator comma
set key top left opaque
set border linewidth 1.2
set tics out
set grid ytics lc rgb '#dddddd' lw 0.8
set multiplot layout 1,2 margins 0.09,0.98,0.18,0.92 spacing 0.12
set xlabel 'Coherent amplitude, {{/Symbol x}}'
set xrange [0.45:2.55]
set arrow 1 from 1.75,graph 0 to 1.75,graph 1 nohead dt 2 lc rgb '#777777'
set arrow 2 from 2.00,graph 0 to 2.00,graph 1 nohead dt 2 lc rgb '#777777'
set ylabel 'Maximum native infidelity'
set logscale y
plot '{data}' using (column('xi')):(column('max_native_infidelity')) \\
     with linespoints lw 2.2 pt 7 ps 0.7 lc rgb '#0072B2' notitle
unset logscale y
set ylabel 'Maximum projective correction ratio'
plot '{data}' using (column('xi')):(column('max_projective_correction_ratio')) \\
     with linespoints lw 2.2 pt 5 ps 0.7 lc rgb '#D55E00' notitle
unset multiplot
"""
    return f"""
set terminal pdfcairo enhanced color font 'Arial,10' size 7.2in,3.2in
set output '{pdf}'
{common}
set terminal pngcairo enhanced color font 'Arial,12' size 2160,960
set output '{png}'
{common}
"""


def main() -> None:
    args = parse_args()
    if args.nt < 2:
        raise ValueError("--nt must be at least 2")
    if args.dim <= max(args.bench_xi) ** 2:
        raise ValueError("--dim is too small for the requested coherent states")
    if any(n > args.dim for n in args.convergence_cutoffs):
        raise ValueError("All convergence cutoffs must be <= --dim")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, args.T, args.nt)
    main_ops = operators(args.dim, args.omega0, args.beta)
    evals_main, evecs_main = eigensystem(np.asarray(main_ops["H_D"]))

    residual = spectral_norm(
        np.asarray(main_ops["H_D"])
        - np.asarray(main_ops["H_K"])
        - np.asarray(main_ops["Delta"])
    )

    norm_rows: List[dict] = []
    for n_cut in args.norm_cutoffs:
        ops_n = operators(n_cut, args.omega0, args.beta)
        norm_rows.append(
            {
                "N_dimension": n_cut,
                "max_occupation": n_cut - 1,
                "eta_sandwiched": spectral_norm(np.asarray(ops_n["Delta"]))
                / spectral_norm(np.asarray(ops_n["H_D"])),
                "commutator_norm_sandwiched": spectral_norm(
                    np.asarray(ops_n["commutator"])
                ),
                "split_residual_norm": spectral_norm(
                    np.asarray(ops_n["H_D"])
                    - np.asarray(ops_n["H_K"])
                    - np.asarray(ops_n["Delta"])
                ),
            }
        )
    write_csv(
        args.output_dir / "operator_norms.csv",
        norm_rows[0].keys(),
        norm_rows,
    )

    benchmark_rows: List[dict] = []
    time_rows: List[dict] = []
    quadrature_rows: List[dict] = []
    for xi in args.bench_xi:
        psi0, discarded = coherent_state(args.dim, xi)
        summary, series = trajectory_metrics(
            main_ops, psi0, times, evals_main, evecs_main
        )
        benchmark_rows.append(
            {
                "xi": xi,
                "mean_n_initial": xi * xi,
                "dimension": args.dim,
                "initial_discarded_probability": discarded,
                **summary,
            }
        )
        for stride in [8, 4, 2, 1]:
            indices = np.arange(0, args.nt, stride, dtype=int)
            if indices[-1] != args.nt - 1:
                indices = np.append(indices, args.nt - 1)
            sampled_times = times[indices]
            sampled_action = series["v_action_on_kerr"][indices]
            integral = float(
                np.sum(
                    0.5
                    * np.diff(sampled_times)
                    * (sampled_action[:-1] + sampled_action[1:])
                )
            )
            quadrature_rows.append(
                {
                    "xi": xi,
                    "number_of_time_points": int(indices.size),
                    "maximum_time_step": float(np.max(np.diff(sampled_times))),
                    "duhamel_infidelity_bound_at_T": float(
                        (abs(float(main_ops["lambda"])) * integral) ** 2
                    ),
                }
            )
        for idx, time in enumerate(times):
            time_rows.append(
                {
                    "xi": xi,
                    "time": float(time),
                    "native_infidelity": float(series["infidelity"][idx]),
                    "projective_correction_ratio": float(
                        series["projective_ratio"][idx]
                    ),
                    "duhamel_infidelity_bound": float(
                        series["duhamel_infidelity_bound"][idx]
                    ),
                    "vcr_action_on_kerr": float(
                        series["v_action_on_kerr"][idx]
                    ),
                    "n_expect_duffing": float(series["n_expect"][idx]),
                    "top4_population": float(series["top4_population"][idx]),
                }
            )
    write_csv(
        args.output_dir / "benchmark_summary.csv",
        benchmark_rows[0].keys(),
        benchmark_rows,
    )
    write_csv(
        args.output_dir / "benchmark_time_series.csv",
        time_rows[0].keys(),
        time_rows,
    )
    write_csv(
        args.output_dir / "time_quadrature_convergence.csv",
        quadrature_rows[0].keys(),
        quadrature_rows,
    )

    xi_rows: List[dict] = []
    for xi in np.linspace(args.xi_min, args.xi_max, args.n_xi):
        psi0, discarded = coherent_state(args.dim, float(xi))
        summary, _ = trajectory_metrics(
            main_ops, psi0, times, evals_main, evecs_main
        )
        xi_rows.append(
            {
                "xi": float(xi),
                "mean_n_initial": float(xi * xi),
                "initial_discarded_probability": discarded,
                **summary,
            }
        )
    write_csv(args.output_dir / "xi_scan.csv", xi_rows[0].keys(), xi_rows)

    convergence_rows: List[dict] = []
    for xi in args.bench_xi:
        psi_ref0, _ = coherent_state(args.dim, xi)
        _, ref_series = trajectory_metrics(
            main_ops, psi_ref0, times, evals_main, evecs_main
        )
        psi_ref = ref_series["psi_D"]
        n_ref = ref_series["n_expect"]

        for n_cut in args.convergence_cutoffs:
            ops_n = operators(n_cut, args.omega0, args.beta)
            psi0_n, discarded_n = coherent_state(n_cut, xi)
            evals_n, evecs_n = eigensystem(np.asarray(ops_n["H_D"]))
            summary_n, series_n = trajectory_metrics(
                ops_n, psi0_n, times, evals_n, evecs_n
            )
            overlap_ref = np.sum(
                np.conj(psi_ref[:n_cut, :]) * series_n["psi_D"], axis=0
            )
            state_infidelity_ref = 1.0 - np.clip(
                np.abs(overlap_ref) ** 2, 0.0, 1.0
            )
            convergence_rows.append(
                {
                    "xi": xi,
                    "N_dimension": n_cut,
                    "reference_dimension": args.dim,
                    "initial_discarded_probability": discarded_n,
                    "max_top4_population": summary_n["max_top4_population"],
                    "max_state_infidelity_vs_reference": float(
                        np.max(state_infidelity_ref)
                    ),
                    "max_abs_n_error_vs_reference": float(
                        np.max(np.abs(series_n["n_expect"] - n_ref))
                    ),
                    "max_native_infidelity": summary_n[
                        "max_native_infidelity"
                    ],
                    "final_duhamel_infidelity_bound": summary_n[
                        "final_duhamel_infidelity_bound"
                    ],
                }
            )
    write_csv(
        args.output_dir / "cutoff_convergence.csv",
        convergence_rows[0].keys(),
        convergence_rows,
    )

    metadata = {
        "omega0": args.omega0,
        "beta": args.beta,
        "lambda": float(main_ops["lambda"]),
        "omega_eff": float(main_ops["omega_eff"]),
        "K_eff": float(main_ops["K_eff"]),
        "T": args.T,
        "nt": args.nt,
        "dimension": args.dim,
        "Pi_N_definition": "sum_{n=0}^{N-1} |n><n|",
        "matrix_convention": "hard truncated Pi_N A Pi_N",
        "energy_zero": "H_D = omega0(n+1/2)+lambda(a+a^dagger)^4",
        "split_residual_norm_at_main_dimension": residual,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "plot_duffing_benchmark.gp").write_text(
        gnuplot_script(args.output_dir), encoding="utf-8"
    )

    verdict_lines = [
        "# Duffing correction validation",
        "",
        f"- omega0 = {args.omega0:g}, beta = {args.beta:g}, "
        f"lambda = {float(main_ops['lambda']):.8g}",
        f"- T = {args.T:g}, nt = {args.nt}, reference dimension = {args.dim}",
        f"- Exact-split residual norm = {residual:.3e}",
        "",
        "## Benchmark states",
        "",
    ]
    for row in benchmark_rows:
        verdict_lines.append(
            f"- xi={row['xi']:.3g}: max native infidelity "
            f"{row['max_native_infidelity']:.6g}; max projective correction "
            f"ratio {row['max_projective_correction_ratio']:.6g}; final "
            f"Duhamel infidelity bound "
            f"{row['final_duhamel_infidelity_bound']:.6g}; max top-four-level "
            f"population {row['max_top4_population']:.3e}."
        )
    verdict_lines.extend(
        [
            "",
            "The operator norms are cutoff-level diagnostics, not estimates of",
            "the correction experienced by either coherent-state trajectory.",
            "Use cutoff_convergence.csv before quoting any trajectory result.",
            "",
        ]
    )
    (args.output_dir / "validation_verdict.md").write_text(
        "\n".join(verdict_lines), encoding="utf-8"
    )

    print(f"Wrote validation outputs to: {args.output_dir.resolve()}")
    print(f"Exact split residual norm: {residual:.3e}")
    for row in benchmark_rows:
        print(
            f"xi={row['xi']:.3g}: max(1-F_K)={row['max_native_infidelity']:.6g}, "
            f"max r_psi={row['max_projective_correction_ratio']:.6g}, "
            f"Duhamel bound(T)={row['final_duhamel_infidelity_bound']:.6g}"
        )


if __name__ == "__main__":
    main()
