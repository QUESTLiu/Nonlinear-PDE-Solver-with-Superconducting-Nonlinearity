"""Write results/analysis_summary.json and diagnostic figures (in common_data/diagnostic_figures) from the archived numerical records."""

from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "common_data/diagnostic_figures"
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#777777"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.3,
    }
)
provenance = {}
summary = {}


def save(fig, name, sources, meaning):
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    provenance[name] = dict(
        meaning=meaning,
        sources={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command="/Users/zhen002/miniconda3/envs/cvdv/bin/python -m common_code.plot_results",
        python="/Users/zhen002/miniconda3/envs/cvdv/bin/python",
        python_version="3.13.0",
        numpy=np.__version__,
        matplotlib=matplotlib.__version__,
    )


def duffing_native():
    directory = ROOT / "duffing_data/formal/E1_retry"
    data = json.loads((directory / "summary.json").read_text())
    points = [x for x in data["points"] if x["T"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), layout="constrained")
    categories = {
        "sufficient": (GREEN, "o"),
        "observed_failure": (ORANGE, "x"),
        "inconclusive": (GRAY, "s"),
    }
    counts = {k: 0 for k in categories}
    for row in points:
        verdict = row["verdicts"][0]["verdict"]
        color, marker = categories[verdict]
        counts[verdict] += 1
        axes[0].scatter(row["xi"], row["beta"], color=color, marker=marker, s=45)
    axes[0].set(
        xlabel=r"Coherent amplitude $\xi$",
        ylabel=r"Duffing coefficient $\beta$",
        title=r"(a) Numerical verdict, $\mathcal{I}_\star=10^{-3}$",
        xticks=[0.5, 1, 1.75, 2, 2.5],
        yticks=[0, 0.02, 0.05, 0.1, 0.2],
    )
    axes[0].legend(
        handles=[
            Line2D([], [], color=c, marker=m, linestyle="", label=label)
            for label, (c, m) in zip(
                ["Sufficient", "Observed failure", "Inconclusive"], categories.values()
            )
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.83),
        frameon=False,
        fontsize=7,
    )
    selected = [x for x in points if x["beta"] == 0.1]
    x = np.array([p["xi"] for p in selected])
    axes[1].semilogy(
        x,
        [p["max_native_infidelity"] for p in selected],
        "o-",
        color=BLUE,
        label="Native infidelity",
    )
    axes[1].semilogy(
        x,
        [min(1, p["native_bound"] ** 2) for p in selected],
        "s--",
        color=ORANGE,
        label=r"$\min(1,B_\psi^2)$",
    )
    raw = []
    sources = [directory / "summary.json"]
    for p in selected:
        path = directory / p["records"][-1]
        sources.append(path)
        with np.load(path) as record:
            raw.append(min(1, float(record["uncentered_integral"][-1] ** 2)))
    axes[1].semilogy(x, raw, "^:", color=GRAY, label="Uncentered bound")
    axes[1].axhline(0.001, color="black", linestyle=":", linewidth=0.8)
    axes[1].set(
        xlabel=r"Coherent amplitude $\xi$",
        ylabel="Infidelity or bound",
        title=r"(b) $\beta=0.10$, $T=1$, $N=80$",
    )
    axes[1].legend(loc="lower right")
    summary["E1_primary_verdict_counts"] = counts
    summary["E1_benchmark"] = [p for p in selected if p["xi"] in (1.75, 2.0)]
    save(
        fig,
        "duffing_native",
        sources,
        "Fixed finite-grid native assessment and bound tightness, with empirical numerical margins.",
    )


def e3_path(ej, condition, n):
    name = f"E3_EJ{ej:g}_{condition}_N{n}"
    for directory in (
        ROOT / "sine_gordon_data/formal/E3",
        ROOT / "sine_gordon_data/stage1/E3",
    ):
        if (directory / (name + ".npz")).exists():
            return directory / name
    raise FileNotFoundError(name)


def classical():
    conditions = [(ej, "primary") for ej in (0.25, 0.5, 1, 2, 4, 8, 16, 32)] + [
        (8, "fixed_variance"),
        (32, "fixed_variance"),
    ]
    rows = []
    sources = []
    for ej, condition in conditions:
        values = {}
        params = {}
        for n in (48, 64, 80):
            path = e3_path(ej, condition, n)
            sources.extend([path.with_suffix(".npz"), path.with_suffix(".json")])
            # with_suffix would truncate decimal EJ tokens, so append extensions.
            npz = Path(str(path) + ".npz")
            meta = Path(str(path) + ".json")
            sources[-2:] = [npz, meta]
            with np.load(npz) as a:
                values[n] = {key: a[key] for key in a.files}
            params[n] = json.loads(meta.read_text())
            params[n] = params[n].get("params", params[n])
        A = params[80]["oscillation_amplitude"]
        differences = [
            float(np.max(abs(values[a]["phi_q"] - values[b]["phi_q"])) / A)
            for a, b in ((48, 64), (64, 80))
        ]
        time_errors = [
            p.get("time_step_convergence_normalized")
            for p in params.values()
            if p.get("time_step_convergence_normalized") is not None
        ]
        time_error = max(time_errors) if time_errors else None
        bare = float(np.max(abs(values[80]["phi_q"] - values[80]["phi_cl"])) / A)
        dressed = float(np.max(abs(values[80]["phi_q"] - values[80]["phi_dw"])) / A)
        rows.append(
            dict(
                E_J=ej,
                preparation=condition,
                cutoff_changes=differences,
                time_refinement=time_error,
                numerical_resolution_ok=bool(
                    time_error is not None and max(differences + [time_error]) <= 0.005
                ),
                D_max=bare,
                D_max_DW=dressed,
                tau_crossing=params[80]["tau_ehrenfest_classical"],
                tau_bracket=params[80]["tau_ehrenfest_bracket_classical"],
                omega_b=params[80]["omega_b"],
                Omega_time=params[80]["Omega_time"],
            )
        )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), layout="constrained")
    primary = [x for x in rows if x["preparation"] == "primary"]
    control = [x for x in primary if x["E_J"] == 1] + [
        x for x in rows if x["preparation"] == "fixed_variance"
    ]
    x = [p["E_J"] for p in primary]
    axes[0].plot(
        x, [p["D_max"] for p in primary], "o-", color=BLUE, label="Bare classical"
    )
    axes[0].plot(
        x,
        [p["D_max_DW"] for p in primary],
        "s--",
        color=ORANGE,
        label="Initial-variance DW",
    )
    axes[0].axhline(0.1, color=GRAY, linestyle=":")
    axes[0].set(
        xscale="log",
        xlabel=r"$E_J$",
        ylabel="Maximum normalized field error",
        title="(a) Primary preparation",
    )
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.26), borderaxespad=0)
    for points, color, marker, label in (
        (primary, BLUE, "o", "Primary"),
        (control, ORANGE, "s", "Fixed variance"),
    ):
        axes[1].plot(
            [p["E_J"] for p in points],
            [5.2 if p["tau_crossing"] is None else p["tau_crossing"] for p in points],
            marker + "-",
            color=color,
            label=label,
        )
    axes[1].axhline(5.2, color=GRAY, linestyle=":", linewidth=0.8)
    axes[1].set(
        xscale="log",
        xlabel=r"$E_J$",
        ylabel=r"First crossing $\tau_E$",
        title="(b) Preparation sensitivity",
    )
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.26), borderaxespad=0)
    p = e3_path(1, "primary", 80)
    with np.load(str(p) + ".npz") as a:
        for key, color, label in (
            ("closure_momentum", BLUE, "Closure residual"),
            ("delta_NG_1", ORANGE, "Non-Gaussian residual"),
            ("cutoff_momentum", GRAY, "Finite-commutator residual"),
        ):
            axes[2].semilogy(
                a["tau"],
                np.maximum(np.max(abs(a[key]), axis=1), 1e-16),
                color=color,
                label=label,
            )
    axes[2].set(
        xlabel=r"Scaled time $\tau$",
        ylabel="Residual magnitude",
        title=r"(c) $E_J=1$, $N=80$",
    )
    axes[2].legend(
        fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.26), borderaxespad=0
    )
    summary["E3"] = rows
    save(
        fig,
        "sg_classical",
        sources,
        "Classical validity, fixed-initial-variance control, and closure versus cutoff diagnostics. Any log-floor is 1e-16.",
    )


def duffing_resources():
    source = ROOT / "duffing_data/formal/E2/summary.json"
    if not source.exists():
        return
    data = json.loads(source.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), layout="constrained")
    summaries = []
    for comp in data["comparisons"]:
        for route, color in (("hybrid", BLUE), ("replacement", ORANGE)):
            cost = comp["resources"]["residual_snap"] + (
                comp["resources"]["replacement_extra_snap"]
                if route == "replacement"
                else 0
            )
            marker = "o" if comp["r"] == 10 else "s"
            rows = [r for r in comp["rows"] if r["route"] == route]
            for ax, scope in zip(axes, ("development", "all")):
                selected = [
                    r for r in rows if scope == "all" or r["input_role"] == scope
                ]
                ax.scatter(
                    cost,
                    max(r["total_amplitude_estimate"] for r in selected),
                    color=color,
                    marker=marker,
                    s=35,
                )
            summaries.append(
                dict(
                    r=comp["r"],
                    L=comp["layers"],
                    route=route,
                    ideal_snap_count=cost,
                    worst=max(r["total_amplitude_estimate"] for r in rows),
                    worst_input=max(rows, key=lambda r: r["total_amplitude_estimate"])[
                        "input"
                    ],
                    acceptance=comp["family_acceptance"][route],
                )
            )
    for ax, title in zip(
        axes, ("(a) Two development inputs", "(b) Complete ten-input family")
    ):
        ax.set(
            yscale="log",
            xlabel="Ideal SNAP gates",
            ylabel="Amplitude estimate including margins",
            title=title,
        )
        ax.axhline(np.sqrt(0.001), color=GRAY, linestyle=":")
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, linestyle="", label="H"),
            Line2D([], [], marker="o", color=ORANGE, linestyle="", label=r"$S_K$"),
            Line2D(
                [], [], marker="s", color=GRAY, linestyle="", label="Square: 20 steps"
            ),
        ]
    )
    summary["E2"] = summaries
    save(
        fig,
        "duffing_resources",
        [source],
        "All six training-selected matched H/S_K pairs, development and complete input-family accuracy.",
    )


def sg_convergence():
    path = ROOT / "double_sine_gordon_data/sg128/reference/summary.json"
    if not path.exists():
        return
    high = json.loads(path.read_text())
    earlier = [
        ROOT / "double_sine_gordon_data/stage1/E4_reference/summary.json",
        ROOT
        / "double_sine_gordon_data/stage1_continuation/E4_reference64/summary.json",
    ]
    summaries = [json.loads(p.read_text()) for p in earlier] + [high]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), layout="constrained")
    values = [d["empirical_total_estimates"] for d in summaries]
    cutoff_labels = ["32→48", "48→64", "96→128"]
    latest = ROOT / "double_sine_gordon_data/sg160/reference/summary.json"
    if latest.exists():
        refined = json.loads(latest.read_text())
        m = refined["empirical_cutoff_estimates"]
        solver = refined["records"][-1]["solver"]
        values.append({key: m[key] + solver[key] for key in ("state_distance", "C12")})
        cutoff_labels.append("128→160")
        earlier.append(latest)
    x = np.arange(len(values))
    axes[0].semilogy(
        x,
        [d["state_distance"] for d in values],
        "o-",
        color=BLUE,
        label="State distance change (a)",
    )
    axes[0].semilogy(
        x, [d["C12"] for d in values], "s--", color=ORANGE, label=r"$C_{12}$ change (a)"
    )
    axes[0].axhline(
        np.sqrt(0.001) / 10, color=GRAY, linestyle=":", label="State resolution"
    )
    axes[0].set(
        xlabel="Compared local cutoffs",
        ylabel="Adjacent-cutoff change",
        title="(a) Cutoff convergence",
    )
    axes[0].set_xticks(x, cutoff_labels, rotation=25)
    n = 128
    raw = path.parent / f"E4_reference_lambda0.3_N{n}.npz"
    with np.load(raw) as a:
        axes[1].plot(a["t"], a["target_C12_real"], color=BLUE, label="Target")
        axes[1].plot(a["t"], a["native_C12_real"], "--", color=ORANGE, label="Native")
        axes[1].plot(
            a["t"],
            a["retuned_native_C12_real"],
            ":",
            color=GREEN,
            label="Retuned native",
        )
        for route, color, style, label in (
            ("target", BLUE, "-", "Target"),
            ("native", ORANGE, "--", "Native"),
            ("retuned_native", GREEN, ":", "Retuned native"),
        ):
            axes[2].plot(a["t"], a[route + "_S2"], style, color=color, label=label)
    axes[1].set(
        xlabel="Simulated time",
        ylabel=r"$\mathrm{Re}\,C_{12}$",
        title=r"(b) $E_J=1$, $\lambda_2=0.3$, $N=128$",
    )
    axes[2].set(xlabel="Simulated time", ylabel=r"$S_2$", title="(c) Entanglement")
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(
        handles + h2, labels + l2, loc="outside lower center", ncol=3, fontsize=8
    )
    summary["E4_reference128"] = {
        k: v for k, v in high.items() if k not in ("provenance", "records")
    }
    save(
        fig,
        "sg_convergence",
        earlier + [path, raw],
        "Reference convergence and resolved quantum observable signal. The state and observable thresholds are separate.",
    )


def polynomial():
    directory = ROOT / "sine_gordon_data/formal/polynomial"
    rows = []
    sources = []
    for phi in (0.5, 1, 2, 3):
        for n in (48, 64, 80):
            stem = directory / f"E3_poly_phi{phi:g}_N{n}"
            meta = Path(str(stem) + ".json")
            raw = Path(str(stem) + ".npz")
            data = json.loads(meta.read_text())
            sources.extend([meta, raw])
            row = dict(phi0=phi, N=n)
            with np.load(raw) as values:
                for family in ("taylor", "chebyshev_spectral", "chebyshev_accessed"):
                    row[family] = dict(
                        minimum_tested_degree=data["d_star_" + family],
                        cutoff_limited_degrees=values["degrees"][
                            values[family + "_cutoff_limited"]
                        ].tolist(),
                    )
            rows.append(row)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), layout="constrained")
    raw = directory / "E3_poly_phi3_N80.npz"
    with np.load(raw) as a:
        for family, color, marker, label in (
            ("taylor", BLUE, "o", "Taylor"),
            ("chebyshev_spectral", ORANGE, "s", "Spectral Chebyshev"),
            ("chebyshev_accessed", GREEN, "^", "Accessed Chebyshev"),
        ):
            for ax, key in zip(axes, ("max", "potential_error", "force_error")):
                ax.semilogy(
                    a["degrees"],
                    np.maximum(a[family + "_" + key], 1e-16),
                    marker + "-",
                    color=color,
                    markersize=3,
                    label=label,
                )
            failed = a[family + "_cutoff_limited"]
            axes[0].scatter(
                a["degrees"][failed],
                a[family + "_max"][failed],
                marker="x",
                color="black",
                s=40,
                zorder=5,
            )
    for ax, ylabel, title in zip(
        axes,
        (
            "Maximum sampled infidelity",
            "Spectral potential error",
            "Spectral force error",
        ),
        ("(a) Trajectory", "(b) Potential", "(c) Force"),
    ):
        ax.set(xlabel="Polynomial degree", ylabel=ylabel, title=title)
    axes[0].axhline(0.001, color=GRAY, linestyle=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    summary["polynomial"] = rows
    save(
        fig,
        "sg_polynomial",
        sources,
        "Auxiliary one-site representation comparison. Black crosses mark cutoff-limited candidates. Accessed intervals use the exact trajectory retrospectively; operator errors use the complete finite spectrum.",
    )


def sg_compiled():
    source = ROOT / "double_sine_gordon_data/sg_formal/comparison/summary.json"
    if not source.exists():
        return
    data = json.loads(source.read_text())
    rows = []
    sources = [source]
    for point in data["configurations"]:
        if point["cutoff"] != 96:
            continue
        path = source.parent / (point["name"] + "_schedule.json")
        sources.append(path)
        schedule = json.loads(path.read_text())
        r = point["r"]
        counts = Counter(pulse["gate"] for pulse in schedule["per_step_merged"])
        counts = {key: r * value for key, value in counts.items()}
        if "trig" in point["route"]:
            first, last = (
                schedule["per_step_merged"][0],
                schedule["per_step_merged"][-1],
            )
            if (
                first["gate"] != "Ry"
                or last["gate"] != "Ry"
                or first["value"] + last["value"] != 0
            ):
                raise ValueError("outer-step qubit cancellation premise failed")
            counts["Ry"] -= 2 * (r - 1)
        counts.update(
            SNAP=r * schedule["local_snap_per_step"],
            displacement=r * schedule["local_displacement_per_step"],
            Gaussian=schedule["full_circuit_merged"]["gaussian_primitives"],
        )
        other = next(
            x
            for x in data["configurations"]
            if x["name"] == point["name"].replace("_N96", "_N128")
        )
        boundary = max(
            x["max_population_at_monitored_locations"]["boundary_band_population"]
            for x in (point, other)
        )
        verdict = point["numerical_acceptance"]
        rows.append(
            dict(
                name=point["name"],
                route=point["route"],
                r=r,
                L=point.get("layers"),
                order=point["order"],
                s=point["subdivisions"],
                primitive_counts=counts,
                total_programmable_primitives=sum(counts.values()),
                native_waits=schedule["full_circuit_merged"]["native_waits"],
                control_qubits=schedule["coherently_reused_qubits"],
                amplitude=verdict["amplitude_total"],
                C12=point["max_C12_error"] + point["empirical_margins"]["C12"],
                S2=point["max_S2_error"] + point["empirical_margins"]["S2"],
                action_sum=point["action_sum"],
                boundary_population=boundary,
                acceptance={key: verdict[key] for key in ("state", "C12", "S2")},
                all_criteria=all(verdict[key] for key in ("state", "C12", "S2"))
                and boundary <= 1e-6,
            )
        )
    colors = {"H_trig": BLUE, "G_trig": ORANGE, "H_trig_ret": GREEN, "H_SNAP": GRAY}
    labels = {
        "H_trig": r"$H_{\mathrm{trig}}$",
        "G_trig": r"$G_{\mathrm{trig}}$",
        "H_trig_ret": r"$H_{\mathrm{trig,ret}}$",
        "H_SNAP": r"$H_{\mathrm{SNAP}}$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.45), layout="constrained")
    for row in rows:
        color = colors[row["route"]]
        marker = "o" if row["r"] == 10 else "s"
        filled = row["order"] == 2 or row["route"] == "H_SNAP"
        for ax, key in zip(axes, ("amplitude", "C12")):
            ax.scatter(
                row["total_programmable_primitives"],
                row[key],
                marker=marker,
                facecolors=color if filled else "none",
                edgecolors=color,
                s=25,
                linewidths=0.8,
            )
        if row["boundary_population"] > 1e-6:
            axes[0].scatter(
                row["total_programmable_primitives"],
                row["amplitude"],
                marker="x",
                color="black",
                s=13,
                linewidths=0.6,
            )
    refinement = ROOT / "double_sine_gordon_data/sg160/summary.json"
    refined = []
    if refinement.exists():
        sources.append(refinement)
        for point in json.loads(refinement.read_text())["configurations"]:
            if point["cutoff"] != 128:
                continue
            original = next(
                row
                for row in rows
                if row["name"] == point["name"].replace("_N128", "_N96")
            )
            verdict = point["numerical_acceptance"]
            updated = dict(
                original,
                name=point["name"],
                action_sum=point["action_sum"],
                amplitude=verdict["amplitude_total"],
                C12=point["max_C12_error"] + point["empirical_margins"]["C12"],
                S2=point["max_S2_error"] + point["empirical_margins"]["S2"],
                boundary_population=verdict["boundary_population_both_cutoffs"],
                acceptance={key: verdict[key] for key in ("state", "C12", "S2")},
                all_criteria=all(verdict[key] for key in ("state", "C12", "S2")),
                validation_cutoffs=[128, 160],
            )
            refined.append(updated)
            for ax, key in zip(axes, ("amplitude", "C12")):
                ax.scatter(
                    updated["total_programmable_primitives"],
                    updated[key],
                    marker="*",
                    facecolors=colors[updated["route"]],
                    edgecolors="black",
                    s=90,
                    linewidths=0.6,
                    zorder=5,
                )
    for ax, ylabel, title, threshold in zip(
        axes,
        ("Bures estimate including margins", r"$|\Delta C_{12}|$ including margin"),
        ("(a) State accuracy", "(b) Connected correlation"),
        (np.sqrt(0.001), 0.01),
    ):
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Programmable primitives after merging",
            ylabel=ylabel,
            title=title,
        )
        ax.axhline(threshold, color="black", linestyle=":", linewidth=0.8)
    handles = [
        Line2D([], [], marker="o", color=colors[key], linestyle="", label=labels[key])
        for key in colors
    ]
    if refined:
        handles.append(
            Line2D(
                [],
                [],
                marker="*",
                color="black",
                linestyle="",
                label="128/160 refinement",
            )
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8)
    selected = {}
    for route in colors:
        eligible = [
            row for row in rows if row["route"] == route and row["all_criteria"]
        ]
        selected[route] = (
            min(
                eligible,
                key=lambda row: (
                    row["total_programmable_primitives"],
                    row["amplitude"],
                ),
            )
            if eligible
            else None
        )
    summary["E4_compiled"] = dict(
        rows=rows,
        refinement_128_160=refined,
        minimum_programmable_count_among_tested=selected,
        optimization_seconds=data["optimization_seconds"],
        validation_seconds=data["validation_seconds"],
        additional_merging="Cancel the last and first inverse Ry rotations across each system-only outer-step separator, removing 2(r-1) pulses. Raw per-block schedules remain archived.",
    )
    save(
        fig,
        "sg_compiled",
        sources,
        "Complete predefined two-site circuit comparison at the common 11-point grid. Costs are unweighted ideal primitive counts, not physical durations. Crosses mark boundary-population failures.",
    )


def refinement_tables():
    stages = []
    for name, relative in (
        ("original", "sg_formal/comparison"),
        ("paired_frames", "sg_refinement/paired"),
        ("larger_fit", "sg_size/expanded"),
        ("continuation", "sg_continuation/fit"),
    ):
        path = ROOT / "double_sine_gordon_data" / relative / "summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows = [
            row
            for row in data["configurations"]
            if row["route"] == "H_SNAP" and "numerical_acceptance" in row
        ]
        stages.append(
            dict(
                stage=name,
                source=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                training_dimensions=data["protocol"].get(
                    "training_dimensions", [48, 24, 16]
                ),
                r=sorted({row["r"] for row in rows}),
                L=sorted({row["layers"] for row in rows}),
                optimizer_runs=len(data["runs"]),
                selected_circuits=len(rows),
                optimization_seconds=data["optimization_seconds"],
                amplitude_range=[
                    min(row["numerical_acceptance"]["amplitude_total"] for row in rows),
                    max(row["numerical_acceptance"]["amplitude_total"] for row in rows),
                ],
                passing={
                    key: sum(row["numerical_acceptance"][key] for row in rows)
                    for key in ("state", "C12", "S2")
                },
            )
        )
    summary["SG_SNAP_refinements"] = stages
    path = ROOT / "double_sine_gordon_data/sg_independent/check/summary.json"
    if path.exists():
        data = json.loads(path.read_text())
        rows = []
        for point in data["configurations"]:
            if "numerical_acceptance" not in point:
                continue
            rows.append(
                dict(
                    route=point["route"],
                    r=point["r"],
                    order=point["order"],
                    raw_common_distance=point["max_bures_distance"],
                    raw_endpoint_distance=point["max_endpoint_bures_distance"],
                    amplitude=point["numerical_acceptance"]["amplitude_total"],
                    C12=point["max_C12_error"] + point["empirical_margins"]["C12"],
                    S2=point["max_S2_error"] + point["empirical_margins"]["S2"],
                    acceptance=point["numerical_acceptance"],
                )
            )
        summary["SG_second_preparation"] = dict(
            rows=rows,
            preparation=data["preparation"],
            reference=data["reference"],
            solver=data["solver"],
            source=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    path = ROOT / "duffing_data/duffing_refinement/comparison/summary.json"
    if path.exists():
        data = json.loads(path.read_text())
        rows = []
        for comparison in data["comparisons"]:
            for role in ("development", "independent_validation"):
                for route in ("hybrid", "replacement"):
                    group = [
                        row
                        for row in comparison["rows"]
                        if row["input_role"] == role and row["route"] == route
                    ]
                    worst = max(group, key=lambda row: row["total_amplitude_estimate"])
                    rows.append(
                        dict(
                            arm=comparison["arm"],
                            L=comparison["layers"],
                            selected_seed=comparison["selected_seed"],
                            role=role,
                            route=route,
                            inputs=len(group),
                            amplitude=worst["total_amplitude_estimate"],
                            worst_input=worst["input"],
                            max_raw_distance=max(
                                row["max_common_distance"] for row in group
                            ),
                            max_reference=max(row["reference_margin"] for row in group),
                            max_solver=max(row["solver_margin"] for row in group),
                            max_padding=max(row["padding_margin"] for row in group),
                            max_boundary=max(
                                row["boundary_population_both_cutoffs"] for row in group
                            ),
                            acceptance=comparison["scope_acceptance"][role][route],
                        )
                    )
        summary["Duffing_refinement"] = dict(
            rows=rows,
            optimization_seconds=data["optimization_seconds"],
            source=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    path = ROOT / "duffing_data/duffing_padding/comparison/summary.json"
    if path.exists():
        data = json.loads(path.read_text())
        rows = []
        for comparison in data["comparisons"]:
            for route in ("hybrid", "replacement"):
                group = [row for row in comparison["rows"] if row["route"] == route]
                rows.append(
                    dict(
                        cutoffs=comparison["padding_cutoffs"],
                        route=route,
                        inputs=len(group),
                        amplitude=max(row["total_amplitude_estimate"] for row in group),
                        padding=max(row["padding_margin"] for row in group),
                        boundary=max(
                            row["boundary_population_both_cutoffs"] for row in group
                        ),
                        acceptance=comparison["family_acceptance"][route],
                    )
                )
        summary["Duffing_padding"] = dict(
            rows=rows,
            source=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            default_reproduction_difference=data["reproduced_default_max_difference"],
        )


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    duffing_native()
    classical()
    duffing_resources()
    sg_convergence()
    polynomial()
    sg_compiled()
    refinement_tables()
    (FIG / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (ROOT / "common_data/analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
