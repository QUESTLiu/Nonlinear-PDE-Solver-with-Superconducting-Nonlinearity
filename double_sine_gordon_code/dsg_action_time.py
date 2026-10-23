"""Serial control times from pulse counts and generator actions.

Run from the repository root in the environment declared in README.md:
    python double_sine_gordon_code/dsg_action_time.py

This reads the primary two-site step scan and merged-Gaussian evidence. It
does not propagate states or fit gates. The same empirical accuracy tests are
used for every route. Additional implementation error is an assumed budget.

For a pulse exp(-i gamma G), time is a per-call overhead plus |gamma|/rate.
CD uses |alpha| and rotations use their full angle theta. Quadratic gates use
X=sqrt(omega_b)*phi and P=Pi/sqrt(omega_b), with coefficients multiplying
X_j^2, P_j^2, X_1 X_2, or P_1 P_2 (no implicit factor 1/2). These actions
can be combined only after assigning an inverse rate to each generator.

Gaussian intervals have lengths h/2,h,...,h,h/2. Position shears commute with
the potential and merge by adding their coefficients. The scalar formulas
below follow Appendix F of the manuscript. No finite-matrix Gaussian group
identity is used. The fixed quadrature normalization is part of the model.
"""

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from double_sine_gordon_code.gaussian_evidence import updated_row, passes_at
from double_sine_gordon_code.trig_counts import basis_schedule, ORDER_CODES
from sine_gordon_code.native_correction import merge_pulses

SOURCE = REPO / "double_sine_gordon_data/dsg_step_scan.json"
OUTPUT = REPO / "double_sine_gordon_data/dsg_action_time.json"
T = 5.2 / math.sqrt(3)
OMEGA_B = math.sqrt(3)
KINDS = ("CD_Z", "Ry", "Rz", "X2", "P2", "XX", "PP")
ROUTES = ("hybrid", "gauss_block", "kin_pot")
ALLOWANCES = (0.0, 0.005, 0.010, 0.020)
EPS = math.sqrt(1e-3)


def trig_pulses(row, total_time=T):
    """Actual merged pulses, leaving system intervals between potential steps."""
    r = row["r"]
    coefficients = (0.0, row["lambda2"]) if row["route"] == "hybrid" else (1.0, row["lambda2"])
    step = merge_pulses(basis_schedule(2, coefficients, total_time/r,
                                     ORDER_CODES[row["order"]], row["s"]))
    pulses = []
    for k in range(r):
        # Only the two basis-change rotations at each end commute through
        # system evolution. CD pulses remain separated by that evolution.
        pulses.extend(step[(2 if k else 0):(-2 if k < r-1 else None)])
    return pulses


def shear_coefficients(tau, kappa=1.0, omega_b=OMEGA_B):
    """Coefficients in Q_phi and Q_Pi, in the normalized X/P basis."""
    if tau <= 0 or kappa <= 0 or omega_b <= 0:
        raise ValueError("Positive time, coupling and basis frequency are required")
    omega = math.sqrt(2*kappa)
    if omega*tau >= math.pi:
        raise ValueError("Gaussian shear outside the branch connected to zero")
    a = omega*math.tan(omega*tau/2)/2
    b = math.sin(omega*tau)/(2*omega)
    return dict(X2=a/(2*omega_b), XX=-a/omega_b,
                P2=omega_b*(tau/4+b/2), PP=omega_b*(tau/2-b))


def gaussian_pulses(route, r, total_time=T, kappa=1.0, omega_b=OMEGA_B):
    """Quadratic primitive coefficients after the manuscript's exact merges."""
    if r < 1 or int(r) != r or total_time <= 0:
        raise ValueError("Positive integer steps and positive time are required")
    if route == "hybrid":
        return []
    h = total_time/r
    intervals = [h/2] + [h]*(r-1) + [h/2]
    pulses = []
    if route == "kin_pot":
        for tau in intervals:
            pulses.extend([("P2", omega_b*tau/2)]*2)
        for _ in range(r):
            pulses.extend([("X2", kappa*h/(2*omega_b))]*2)
            pulses.append(("XX", -kappa*h/omega_b))
        return pulses
    if route != "gauss_block":
        raise ValueError(f"Unknown route {route}")
    coefficients = [shear_coefficients(tau,kappa,omega_b) for tau in intervals]
    q_factors = [coefficients[0]]
    q_factors += [{key:left[key]+right[key] for key in ("X2","XX")}
                  for left,right in zip(coefficients[:-1],coefficients[1:])]
    q_factors += [coefficients[-1]]
    for q in q_factors:
        pulses.extend([("X2",q["X2"])]*2)
        pulses.append(("XX",q["XX"]))
    for p in coefficients:
        pulses.extend([("P2",p["P2"])]*2)
        pulses.append(("PP",p["PP"]))
    return pulses


def pulse_resources(row):
    pulses = [(p["gate"],float(p["value"])) for p in trig_pulses(row)]
    pulses += gaussian_pulses(row["route"],row["r"])
    counts = dict.fromkeys(KINDS,0)
    actions = dict.fromkeys(KINDS,0.0)
    for kind,value in pulses:
        counts[kind] += 1
        actions[kind] += abs(value)
    for kind in ("CD_Z","Ry","Rz"):
        if counts[kind] != row["primitive_counts"][kind]:
            raise ValueError(f"Pulse count mismatch: {kind}, {row['route']}, {row['r']}")
    if sum(counts[k] for k in ("X2","P2","XX","PP")) != row["primitive_counts"]["Gaussian"]:
        raise ValueError("Gaussian primitive count mismatch")
    if sum(counts.values()) != row["total_programmable_primitives"]:
        raise ValueError("Total primitive count mismatch")
    return counts,actions


def load_candidates():
    candidates=[]
    for source_row in json.loads(SOURCE.read_text())["rows"]:
        if source_row["preparation"] != [1.0,0.0] or source_row["s"] != 1 or source_row["lambda2"] != 0.3:
            continue
        row=updated_row(source_row)
        counts,actions=pulse_resources(row)
        candidates.append(dict(
            id=f"{row['route']}:r{row['r']}:o{row['order']}", route=row["route"],
            r=row["r"],order=row["order"],counts=counts,actions=actions,
            native_waits=row["native_waits"],state_error=row["state_error"],
            state_error_interval=row.get("bounds",{}).get("state_error",[row["state_error"]]*2),
            margins=row["margins"],C12_error=row["C12_error"],S2_error=row["S2_error"],
            boundary_population=row["boundary_population"],
            evidence_id=row.get("evidence_id"),
            eligible={str(b):bool(passes_at(row,EPS,b)) for b in ALLOWANCES}))
    return candidates


def control_time(candidate, overhead, inverse_rates):
    overheads = overhead if isinstance(overhead, dict) else dict.fromkeys(KINDS, overhead)
    if set(overheads) != set(KINDS) or any(not math.isfinite(v) or v < 0 for v in overheads.values()):
        raise ValueError("Every generator needs a finite, nonnegative pulse overhead")
    if set(inverse_rates) != set(KINDS) or any(not math.isfinite(v) or v < 0 for v in inverse_rates.values()):
        raise ValueError("Every generator needs a finite, nonnegative inverse rate")
    return sum(overheads[k]*candidate["counts"][k] +
               inverse_rates[k]*candidate["actions"][k] for k in KINDS)


def select(candidates, allowance, overhead, inverse_rates, switch=0.0):
    if not math.isfinite(switch) or switch < 0:
        raise ValueError("Switching overhead must be finite and nonnegative")
    selected={}
    for route in ROUTES:
        pool=[c for c in candidates if c["route"]==route and c["eligible"][str(allowance)]]
        if not pool:
            raise ValueError(f"No accepted {route} candidate at allowance {allowance}")
        def duration(c):
            return control_time(c,overhead,inverse_rates)+c["native_waits"]*switch
        c=min(pool,key=lambda c:(duration(c),c["state_error"],c["r"],c["order"]))
        selected[route]=dict(id=c["id"],r=c["r"],order=c["order"],
                             time_without_native=duration(c))
    programmable=min((selected[r] for r in ("gauss_block","kin_pot")),
                     key=lambda x:x["time_without_native"])
    return dict(routes=selected,programmable=programmable["id"],
                critical_native_time=programmable["time_without_native"]-
                selected["hybrid"]["time_without_native"])


def rates(gaussian=1.0,cross=None):
    result=dict.fromkeys(KINDS,1.0)
    result.update(X2=gaussian,P2=gaussian,XX=gaussian,PP=gaussian)
    if cross is not None:
        result.update(XX=cross,PP=cross)
    return result


def transition_points(candidates, allowance, overhead, gaussian_range=(0.1, 1000.0)):
    """All affine crossings, so plotted route minima include their exact corners."""
    lo, hi = gaussian_range
    pool = [c for c in candidates if c["eligible"][str(allowance)]]
    lines = [(control_time(c, overhead, rates(0)),
              sum(c["actions"][k] for k in ("X2", "P2", "XX", "PP"))) for c in pool]
    points = {lo, hi}
    for i, (a, b) in enumerate(lines):
        for c, d in lines[i+1:]:
            if b != d:
                x = (c-a)/(b-d)
                if lo < x < hi:
                    points.add(x)
    return sorted(points)


def switching_boundary(candidates, allowance, overhead, inverse_rates):
    pool = [c for c in candidates if c["eligible"][str(allowance)]]
    programmable = min(control_time(c, overhead, inverse_rates) for c in pool
                       if c["route"] != "hybrid")
    hybrid = [c for c in pool if c["route"] == "hybrid"]
    roots = [(programmable-control_time(c, overhead, inverse_rates))/c["native_waits"]
             for c in hybrid]
    points = {0.0, max(roots)}
    for i, left in enumerate(hybrid):
        for right in hybrid[i+1:]:
            delta = left["native_waits"]-right["native_waits"]
            if delta:
                x = (control_time(right, overhead, inverse_rates)-
                     control_time(left, overhead, inverse_rates))/delta
                if x >= 0:
                    points.add(x)
    return dict(zero_native_limit=max(roots), knots=sorted(points))


def make_figures(record):
    from common_code import make_figures as style

    plt = style.plt
    figure_dir = REPO / "paper/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    candidates = record["candidates"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.80, wspace=0.36)
    ax = axes[0]
    grid = sorted(set(np.geomspace(0.1, 1000, 201)) |
                  set(transition_points(candidates, 0.005, 0.0)))
    difference = []
    for g in grid:
        selected = select(candidates, 0.005, 0.0, rates(g))["routes"]
        difference.append(selected["gauss_block"]["time_without_native"]-
                          selected["kin_pot"]["time_without_native"])
    ax.axhline(0, color="0.35", lw=0.7)
    ax.plot(grid, difference, color=style.C_GAUSS, lw=1.7)
    ax.set_xscale("log")
    ax.set(xlim=(0.1, 1000), ylim=(-45, 35),
           xlabel=r"Time per Gaussian coefficient $u_G$",
           ylabel=r"$(p_{\rm GB}^{\min}-p_{\rm KP}^{\min})/\tau_A$")
    ax.text(0.03, 0.12, "Gaussian block faster", transform=ax.transAxes,
            color=style.C_GAUSS, fontsize=8)
    ax.text(0.56, 0.89, "Kinetic-potential faster", transform=ax.transAxes,
            color=style.C_PROG, fontsize=8)
    style.panel(ax, "(a) Programmable route selection", y=1.17)
    ax.text(0, 1.04, r"$\epsilon_{\rm add}=0.005$, $u_0=0$", transform=ax.transAxes)

    ax = axes[1]
    for overhead, color, ls in ((0.0, style.C_HYBRID, "-"), (1.0, style.C_PROG, "--")):
        boundary = switching_boundary(candidates, 0.005, overhead, rates())
        grid = sorted(set(np.linspace(0, 30, 151)) |
                      {v for v in boundary["knots"] if 0 <= v <= 30})
        values = [select(candidates, 0.005, overhead, rates(), v)["critical_native_time"]
                  for v in grid]
        ax.plot(grid, values, color=color, ls=ls, lw=1.7, label=f"$u_0={overhead:g}$")
        ax.plot(boundary["zero_native_limit"], 0, "o", ms=4, color=color,
                clip_on=False, zorder=5)
    ax.axhline(0, color="0.35", lw=0.7)
    ax.set(xlim=(0, 30), ylim=(0, 235),
           xlabel=r"Native switching overhead $\tau_{\rm pair}/\tau_A$",
           ylabel=r"Break-even native time $t_{\rm nat}^{\rm crit}/\tau_A$")
    style.panel(ax, "(b) Gate overhead and native time", y=1.17)
    ax.text(0, 1.04, r"$\epsilon_{\rm add}=0.005$, $u_G=1$", transform=ax.transAxes)
    ax.legend(loc="upper right")
    stem = figure_dir / "fig_dsg_action_time"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": None, "Producer": None})
    fig.savefig(stem.with_suffix(".png"), dpi=220, metadata={"Software": None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.84, wspace=0.34)
    steps = list(range(4, 25))
    for order, ls in ((1, "-"), (2, "--")):
        values = []
        for r in steps:
            row = dict(route="hybrid", r=r, order=order, s=1, lambda2=0.3)
            pulses = trig_pulses(row)
            values.append(sum(abs(p["value"]) for p in pulses if p["gate"] == "CD_Z"))
        axes[0].plot(steps, values, ls=ls, color=style.C_HYBRID,
                     label=f"Hybrid, order {order}")
    axes[0].set(xlabel=r"Outer steps $r$", ylabel=r"Displacement action $A_{\rm CD}$",
                xlim=(4, 24), ylim=(0, 160))
    axes[0].legend(loc="upper left", fontsize=8)
    style.panel(axes[0], "(a) Displacement action", y=1.06)
    for route, label, color, ls in (("gauss_block", "Gaussian block", style.C_GAUSS, "-"),
                                    ("kin_pot", "Kinetic-potential", style.C_PROG, "--")):
        values = [sum(abs(v) for _, v in gaussian_pulses(route, r)) for r in steps]
        axes[1].plot(steps, values, color=color, ls=ls, label=label)
    axes[1].set(xlabel=r"Outer steps $r$", ylabel="Total normalized quadratic action",
                xlim=(4, 24))
    axes[1].legend(loc="upper right", fontsize=8)
    style.panel(axes[1], "(b) Quadratic action", y=1.06)
    stem2 = figure_dir / "fig_dsg_action_components"
    fig.savefig(stem2.with_suffix(".pdf"), metadata={"Creator": None, "Producer": None})
    fig.savefig(stem2.with_suffix(".png"), dpi=220, metadata={"Software": None})
    plt.close(fig)
    return dict(matplotlib=style.matplotlib.__version__, matplotlib_origin=style.matplotlib.__file__,
                figures=[str(s.relative_to(REPO))+".pdf" for s in (stem, stem2)])


def main():
    candidates=load_candidates()
    scans=[]
    for allowance in ALLOWANCES:
        for overhead in (0.0,0.2,1.0,5.0):
            for gaussian in np.geomspace(0.1,1000,161):
                scans.append(dict(allowance=allowance,overhead=overhead,
                                  gaussian=float(gaussian),switch=0.0,
                                  **select(candidates,allowance,overhead,rates(gaussian))))
    switching=[]
    for allowance in ALLOWANCES:
        for overhead in (0.0,0.2,1.0,5.0):
            for switch in np.linspace(0,50,201):
                switching.append(dict(allowance=allowance,overhead=overhead,gaussian=1.0,
                                      switch=float(switch),
                                      **select(candidates,allowance,overhead,rates(),switch)))
    sensitivity=[]
    for allowance in ALLOWANCES:
        for local in (0.1,1.0,10.0,100.0):
            for cross in (0.1,1.0,10.0,100.0):
                for overhead in (0.0,1.0):
                    sensitivity.append(dict(allowance=allowance,overhead=overhead,
                                            gaussian_local=local,gaussian_cross=cross,
                                            **select(candidates,allowance,overhead,rates(local,cross))))
    inputs=[SOURCE,Path(__file__),REPO/"double_sine_gordon_code/trig_counts.py",
            REPO/"sine_gordon_code/native_correction.py",REPO/"double_sine_gordon_code/gaussian_evidence.py",
            REPO/"common_code/make_figures.py",REPO/"requirements.txt"]
    records = sorted((REPO/"common_data/records/gaussian_revision").glob("g[0-9][0-9][0-9].json"))
    inputs += records
    inputs += [REPO/json.loads(path.read_text())["source"] for path in records]
    record=dict(
        parameters=dict(EJ=1.0,kappa=1.0,lambda2=0.3,omega_b=OMEGA_B,T=T,
                        preparation=[1,0],s=1,cutoffs=[128,160],allowances=ALLOWANCES),
        scope=dict(candidate_set="TGIFS primary preparation in the archived step scan, all three routes",
                   accuracy="Unchanged empirical acceptance at each circuit's outer-step readouts",
                   timing="Serial, per-call overhead plus absolute generator coefficient times inverse rate",
                   normalization="X=sqrt(omega_b)*phi, P=Pi/sqrt(omega_b), G coefficients have no implicit 1/2",
                   assumptions="Native drift is controlled during ideal pulses; rates are sensitivity parameters",
                   implementation_error="Assumed allowance, no duration-dependent noise or pulse simulation",
                   rate_slices="CD and qubit inverse rates equal tau_A, Gaussian rates varied in tau_A units",
                   overhead_slices="Common per-primitive overhead u_0*tau_A, native pair overhead separate",
                   sign_and_phase="Equal rates for both signs, Rz charged as a physical rotation in displayed slices",
                   empty_pool="Selection stops with an explicit error when any route has no accepted candidate",
                   optimality="Minimum over the evaluated step-scan candidates only"),
        environment=dict(python=sys.executable,python_version=platform.python_version(),
                         numpy=np.__version__,numpy_origin=np.__file__),
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        command="python double_sine_gordon_code/dsg_action_time.py",
        input_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))},
        candidates=candidates,gaussian_rate_scan=scans,switching_scan=switching,
        gaussian_rate_sensitivity=sensitivity,
        switching_boundaries={str(b): switching_boundary(candidates,0.005,b,rates())
                              for b in (0.0,1.0)})
    record["figure_environment"] = make_figures(record)
    OUTPUT.write_text(json.dumps(record,indent=2)+"\n")
    print(f"{len(candidates)} candidates, {len(scans)+len(switching)+len(sensitivity)} algebraic selections")
    for overhead in (0.0,1.0):
        for gaussian in (1.0,10.0,100.0,1000.0):
            result=select(candidates,0.005,overhead,rates(gaussian))
            print(overhead,gaussian,result)
    print(OUTPUT)


if __name__=="__main__":
    main()
