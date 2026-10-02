"""Conditional duration tradeoffs from the stored two-site DSG step scan.

Run from the repository root with the interpreter declared in README.md:
    python double_sine_gordon_code/dsg_control_tradeoffs.py

This script reads dsg_step_scan.json and its merged-Gaussian comparison evidence. It selects the primary preparation
(1, 0), lambda2 = 0.3, and s = 1, and writes a new JSON record and figure.

The additional amplitude allowance epsilon_add is a bound assumed between an
implemented state and its ideal circuit state at the sampled times. By the
Bures triangle inequality, e_B + epsilon_add <= sqrt(1e-3) is a sufficient
state-error budget if e_B bounds the ideal-circuit error. The archived e_B
includes empirical reference, solver, and padding margins, so this use remains
conditional. All original observable and boundary tests must already pass.
Their acceptance is not extended to a perturbed circuit by this selection.

For a serial circuit i, let p_i = sum_g n_ig tau_g and let tau_pair be the
combined entering and leaving overhead of each native interval. The duration
is p_i for a programmable circuit and p_i + (r_i + 1) tau_pair + t_native for
a hybrid circuit. The Strang waits h/2, h, ..., h, h/2 total T, so t_native is
common to all hybrid candidates. It can therefore be added after minimizing.
Hybrid is faster when 0 <= t_native < min_G p_i - min_H(p_i + n_wait tau_pair).
The minimizations are restricted to the evaluated candidates passing the
specified budget. A negative right-hand side leaves no such native duration.

The figure uses tau_CD = tau_Ry = tau_Rz = tau_0, w_G = tau_G/tau_0,
and w_sw = tau_pair/tau_0. These are constant durations per ideal gate type,
not device measurements or a model of duration-dependent control errors.

For optional observable diagnostics, a Bures separation delta implies trace
distance D <= delta sqrt(1 - delta^2/4). Unit-norm vertex operators give
|Delta C12| <= 6D by expanding its three expectation factors. Partial trace
contracts D, and |Delta Tr(rho_1^2)| <= 4D follows from Holder's inequality.
With ideal minimum purity p_min, |Delta S2| <= -log(1 - 4D/p_min), provided
4D < p_min. These conservative state-to-observable bounds only concern the
stored readout times. They do not bound populations inside the pulse sequence.
"""

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from double_sine_gordon_code.gaussian_evidence import updated_row, passes_at

SOURCE = REPO / "double_sine_gordon_data" / "dsg_step_scan.json"
OUTPUT = REPO / "double_sine_gordon_data" / "dsg_control_tradeoffs.json"
EPS = math.sqrt(1e-3)
GATES = ("CD_Z", "Ry", "Rz", "Gaussian")
ROUTES = ("hybrid", "gauss_block")


def candidate_id(row):
    return f"{row['route']}:r{row['r']}:o{row['order']}"


def eligible(rows, route, epsilon_add):
    return [
        row
        for row in rows
        if row["route"] == route and passes_at(row, EPS, epsilon_add)
    ]


def duration(row, gate_times, pair_time):
    return (
        sum(row["primitive_counts"][gate] * gate_times[gate] for gate in GATES)
        + row["native_waits"] * pair_time
    )


def best(rows, route, epsilon_add, gate_times, pair_time):
    candidates = eligible(rows, route, epsilon_add)
    if not candidates:
        raise ValueError(f"No evaluated {route} candidate at epsilon_add={epsilon_add}")
    return min(
        candidates,
        key=lambda row: (
            duration(row, gate_times, pair_time),
            row["state_error"],
            row["r"],
            row["order"],
        ),
    )


def timing_point(rows, epsilon_add, w_gaussian, w_switch):
    gate_times = dict(CD_Z=1.0, Ry=1.0, Rz=1.0, Gaussian=w_gaussian)
    hybrid = best(rows, "hybrid", epsilon_add, gate_times, w_switch)
    programmable = best(rows, "gauss_block", epsilon_add, gate_times, w_switch)
    hybrid_time = duration(hybrid, gate_times, w_switch)
    programmable_time = duration(programmable, gate_times, w_switch)
    return dict(
        w_G=w_gaussian,
        w_sw=w_switch,
        hybrid=candidate_id(hybrid),
        gauss_block=candidate_id(programmable),
        hybrid_control_time=hybrid_time,
        gauss_block_time=programmable_time,
        critical_native_time=programmable_time - hybrid_time,
    )


def summary(row, epsilon_add):
    distance = epsilon_add * math.sqrt(1 - epsilon_add**2 / 4)
    purity = math.exp(-max(row["series"]["S2"]))
    extra_entropy = (
        -math.log1p(-4 * distance / purity) if 4 * distance < purity else None
    )
    return dict(
        candidate=candidate_id(row),
        r=row["r"],
        order=row["order"],
        error_evidence=row.get("error_evidence", "direct numerical estimate"),
        primitive_counts=row["primitive_counts"],
        total_programmable_primitives=row["total_programmable_primitives"],
        native_waits=row["native_waits"],
        e_B=row["state_error"],
        e_B_plus_allowance=row["state_error"] + epsilon_add,
        remaining_state_budget=EPS - row["state_error"] - epsilon_add,
        C12_error=row["C12_error"],
        S2_error=row["S2_error"],
        conditional_C12_error_bound=row["C12_error"] + 6 * distance,
        conditional_S2_error_bound=(
            row["S2_error"] + extra_entropy if extra_entropy is not None else None
        ),
        ideal_minimum_site_purity=purity,
    )


def load_rows():
    rows = [
        updated_row(row)
        for row in json.loads(SOURCE.read_text())["rows"]
        if row["preparation"] == [1.0, 0.0]
        and row["s"] == 1
        and row["lambda2"] == 0.3
        and row["route"] in ROUTES
    ]
    for row in rows:
        margins = row["margins"]
        acceptance = dict(
            state=(
                row["state_error"] <= EPS
                and margins["reference"] <= EPS / 10
                and margins["solver"] <= EPS / 10
            ),
            boundary=row["boundary_population"] <= 1e-6,
            C12=margins["C12"] <= 1e-3 and row["C12_error"] <= 0.01,
            S2=margins["S2"] <= 5e-3 and row["S2_error"] <= 0.02,
        )
        if (
            acceptance != row["acceptance"]
            or all(acceptance.values()) != row["all_criteria"]
        ):
            raise ValueError(f"Inconsistent archived acceptance: {candidate_id(row)}")
        if (
            sum(row["primitive_counts"].values())
            != row["total_programmable_primitives"]
        ):
            raise ValueError(f"Inconsistent primitive count: {candidate_id(row)}")
        if row["native_waits"] != (row["r"] + 1 if row["route"] == "hybrid" else 0):
            raise ValueError(f"Inconsistent native count: {candidate_id(row)}")
    if len({candidate_id(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate candidates in the selected scan")
    return rows


def make_figure(rows, gate_curve, switch_curve):
    # Reuse the existing plotting style without invoking its figure-generation main.
    sys.path.insert(0, str(REPO))
    from common_code import make_figures as style

    plt, np = style.plt, style.np
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw=dict(wspace=0.28))
    hybrid_color, gaussian_color = style.C_HYBRID, "#A64D8B"

    ax = axes[0]
    # Add the analytic corner to the display grid while preserving every stored point.
    gate_corner = timing_point(rows, 0.01, 5 / 3, 0)
    gate_display = sorted(gate_curve + [gate_corner], key=lambda point: point["w_G"])
    x = np.array([point["w_G"] for point in gate_display])
    y = np.array([point["critical_native_time"] for point in gate_display])
    ax.fill_between(x, 0, y, color=hybrid_color, alpha=0.10)
    ax.plot(x, y, color=hybrid_color, lw=1.7)
    ax.axvline(5 / 3, color=".55", lw=0.7, ls=":")
    ax.plot(
        5 / 3,
        gate_corner["critical_native_time"],
        "o",
        ms=4.5,
        mfc="white",
        mec=hybrid_color,
        mew=1.1,
    )
    ax.text(0, 1.16, "(a) Gaussian gate duration", transform=ax.transAxes, va="bottom")
    ax.text(
        0,
        1.045,
        r"$\epsilon_{\rm add}=0.010$, $w_{\rm sw}=0$",
        transform=ax.transAxes,
        va="bottom",
        fontsize=8,
    )
    ax.text(5 / 6, 404, "GB: $r=9$\norders $2/1$", ha="center", va="top", fontsize=8)
    ax.text(17 / 6, 404, "GB: $r=8$\norders $2/2$", ha="center", va="top", fontsize=8)
    ax.text(
        1.1, 295, "Gaussian block faster", ha="center", color=gaussian_color, fontsize=8
    )
    ax.text(2, 90, "Hybrid faster", ha="center", color=hybrid_color)
    ax.set(xlabel=r"Gaussian duration $w_G=\tau_G/\tau_0$", xlim=(0, 4), ylim=(0, 420))
    ax.set_xticks([0, 5 / 3, 3, 4], ["0", r"$5/3$", "3", "4"])
    ax.set_yticks([0, 100, 200, 300, 400])

    ax = axes[1]
    zero_crossing = timing_point(rows, 0.005, 1, 149 / 7)
    switch_display = sorted(
        switch_curve + [zero_crossing], key=lambda point: point["w_sw"]
    )
    x = np.array([point["w_sw"] for point in switch_display])
    y = np.array([point["critical_native_time"] for point in switch_display])
    ax.axhspan(-35, 0, facecolor=".965", zorder=0)
    ax.fill_between(
        x, 0, y, where=y >= 0, color=hybrid_color, alpha=0.10, interpolate=True
    )
    ax.plot(x, y, color=hybrid_color, lw=1.7)
    ax.axvline(14, color=".55", lw=0.7, ls=":")
    ax.axhline(0, color=".35", lw=0.7)
    ax.plot(
        14,
        timing_point(rows, 0.005, 1, 14)["critical_native_time"],
        "o",
        ms=4.5,
        mfc="white",
        mec=hybrid_color,
        mew=1.1,
    )
    ax.plot(
        149 / 7, zero_crossing["critical_native_time"], "o", ms=3.5, color=hybrid_color
    )
    ax.text(
        0, 1.16, "(b) Native switching overhead", transform=ax.transAxes, va="bottom"
    )
    ax.text(
        0,
        1.045,
        r"$\epsilon_{\rm add}=0.005$, $w_G=1$",
        transform=ax.transAxes,
        va="bottom",
        fontsize=8,
    )
    ax.text(7, 203, "Hybrid: $r=7$\norder $1$", ha="center", va="top", fontsize=8)
    ax.text(19.5, 203, "Hybrid: $r=6$\norder $2$", ha="center", va="top", fontsize=8)
    ax.text(
        17, 102, "Gaussian block faster", ha="center", color=gaussian_color, fontsize=8
    )
    ax.text(5.8, 32, "Hybrid faster", ha="center", color=hybrid_color)
    ax.set(
        xlabel=r"Native overhead $w_{\rm sw}=\tau_{\rm pair}/\tau_0$",
        xlim=(0, 25),
        ylim=(-35, 215),
    )
    ax.set_xticks([0, 7, 14, 149 / 7, 25], ["0", "7", "14", r"$149/7$", "25"])
    ax.set_yticks([-25, 0, 50, 100, 150, 200])
    for ax in axes:
        ax.set_ylabel(r"Native duration $t_{\rm nat}/\tau_0$")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.81)
    stem = REPO / "paper" / "figures" / "fig_dsg_control_tradeoffs"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": None, "Producer": None})
    fig.savefig(stem.with_suffix(".png"), dpi=220, metadata={"Software": None})
    plt.close(fig)
    return dict(
        numpy=np.__version__,
        matplotlib=style.matplotlib.__version__,
        numpy_origin=np.__file__,
        matplotlib_origin=style.matplotlib.__file__,
    )


def main():
    rows = load_rows()
    unit = dict.fromkeys(GATES, 1.0)
    budgets = []
    for allowance in (0.0, 0.005, 0.01, 0.02):
        selected = {
            route: summary(best(rows, route, allowance, unit, 0), allowance)
            for route in ROUTES
        }
        selected["epsilon_add"] = allowance
        selected["primitive_count_ratio"] = (
            selected["gauss_block"]["total_programmable_primitives"]
            / selected["hybrid"]["total_programmable_primitives"]
        )
        budgets.append(selected)
    gate_curve = [timing_point(rows, 0.01, k / 20, 0) for k in range(81)]
    switch_curve = [timing_point(rows, 0.005, 1, k / 4) for k in range(101)]
    environment = make_figure(rows, gate_curve, switch_curve)
    environment.update(python=sys.executable, python_version=platform.python_version())
    output = dict(
        description="Conditional serial-duration comparison within the evaluated two-site DSG candidates.",
        source=str(SOURCE.relative_to(REPO)),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        gaussian_evidence={
            row["evidence_id"]: hashlib.sha256(
                (
                    REPO
                    / "common_data/records/gaussian_revision"
                    / (row["evidence_id"] + ".json")
                ).read_bytes()
            ).hexdigest()
            for row in rows
            if "evidence_id" in row
        },
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        command="python double_sine_gordon_code/dsg_control_tradeoffs.py",
        environment=environment,
        scope=dict(
            lambda2=0.3,
            preparation=[1.0, 0.0],
            s=1,
            cutoffs=[128, 160],
            state_threshold=EPS,
            readout_times="Each circuit's own outer-step grid jT/r.",
            numerical_status="Archived resolution differences are empirical margins.",
            implementation_status="State allowance is assumed, not obtained from a noise simulation.",
            timing_model="One ancilla, serial ideal gates with a fixed duration per gate type.",
            native_overhead="tau_pair includes entering and leaving each of r+1 native intervals.",
            observable_status="Original observable and boundary tests apply to ideal circuits only.",
        ),
        evaluated_steps={
            route: {
                str(order): sorted(
                    row["r"]
                    for row in rows
                    if row["route"] == route and row["order"] == order
                )
                for order in sorted(
                    {row["order"] for row in rows if row["route"] == route}
                )
            }
            for route in ROUTES
        },
        candidates=[
            summary(row, 0)
            | dict(all_criteria=row["all_criteria"], margins=row["margins"])
            for row in rows
        ],
        allowance_selection=budgets,
        gate_duration_curve=dict(epsilon_add=0.01, w_sw=0, points=gate_curve),
        native_overhead_curve=dict(epsilon_add=0.005, w_G=1, points=switch_curve),
        analytic_boundaries=dict(
            gate_duration="epsilon_add=0.01, w_sw=0: min(128+63*w_G, 138+57*w_G)",
            native_overhead="epsilon_add=0.005, w_G=1: max(163-8*w_sw, 149-7*w_sw)",
            criterion="Hybrid is faster for 0 <= t_native/tau_0 < critical_native_time.",
        ),
    )
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    for budget in budgets:
        print(
            f"epsilon_add={budget['epsilon_add']:g}: "
            f"hybrid={budget['hybrid']['total_programmable_primitives']} "
            f"gauss_block={budget['gauss_block']['total_programmable_primitives']} "
            f"ratio={budget['primitive_count_ratio']:.6f}"
        )
    print(OUTPUT.relative_to(REPO))


if __name__ == "__main__":
    main()
