"""Figures for the hardware-native simulation manuscript.

All data come from the archived records in the *_data folders and from
../data/duffing_amplitude_scan.json (produced by duffing_amplitude_scan.py).
No simulation is run here. Usage, from the repository root:

    /Users/zhen002/miniconda3/envs/cvdv/bin/python common_code/make_figures.py [--preview DIR]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SG_DATA = REPO / "sine_gordon_data"
DF_DATA = REPO / "duffing_data"
DSG_DATA = REPO / "double_sine_gordon_data"
COMMON_DATA = REPO / "common_data"
FIG = REPO / "paper" / "figures"

COL1, COL2 = 3.375, 7.0  # PRA single- and double-column widths (in)

# Route colours are shared by every figure (Okabe-Ito palette).
C_TARGET = "#000000"
C_NATIVE = "#8C8C8C"
C_RETUNED = "#009E73"
C_GAUSS = "#CC79A7"
C_HYBRID = "#0072B2"
C_HYBRID_RET = "#009E73"
C_PROG = "#D55E00"
C_SNAP = "#E69F00"
C_DW = "#009E73"
C_FIXED = "#CC79A7"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.3,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


def panel(ax, label, x=0.035, y=0.955, **kw):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=8, **kw)


def save(fig, name, preview):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if preview:
        fig.savefig(Path(preview) / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- SG classical
def e3(ej, prep, n=80):
    for d in (SG_DATA / "formal" / "E3", SG_DATA / "stage1" / "E3"):
        p = d / f"E3_EJ{ej:g}_{prep}_N{n}"
        if Path(str(p) + ".npz").exists():
            meta = json.loads(Path(str(p) + ".json").read_text())
            with np.load(str(p) + ".npz") as a:
                arrays = {k: a[k] for k in a.files}
            return meta, arrays
    raise FileNotFoundError(ej, prep, n)


def fig_sg_classical(preview):
    ejs = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.95), gridspec_kw=dict(wspace=0.5))

    meta, a = e3(1, "primary")
    amp = meta["oscillation_amplitude"]
    tau = a["tau"]
    d_cl = np.max(np.abs(a["mean_phi"] - a["phi_cl"]), axis=1) / amp
    d_dw = np.max(np.abs(a["mean_phi"] - a["phi_dw"]), axis=1) / amp
    ax = axes[0]
    ax.axhline(0.1, color="0.55", lw=0.6, ls=":")
    ax.plot(tau, d_cl, color=C_TARGET, label="bare classical")
    ax.plot(tau, d_dw, color=C_DW, ls="--", label="DW reference")
    for t_e, c in ((meta["tau_ehrenfest_classical"], C_TARGET), (meta["tau_ehrenfest_dw"], C_DW)):
        ax.plot([t_e], [0.1], marker="v", ms=4, color=c, clip_on=False)
    ax.set(xlim=(0, 5.2), ylim=(0, 0.27), xlabel=r"$\tau=\Omega t$", ylabel=r"$d_{\rm MF}(\tau)$")
    ax.text(0.15, 0.106, r"$\varepsilon^\star_{\rm MF}$", fontsize=7, color="0.35", va="bottom")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
    panel(ax, r"(a) $E_J=1$")

    rows = [e3(ej, "primary") for ej in ejs]
    dmax = np.array([np.max(np.abs(x["mean_phi"] - x["phi_cl"])) / m["oscillation_amplitude"] for m, x in rows])
    dmean = np.array([np.mean(np.abs(x["mean_phi"] - x["phi_cl"])) / m["oscillation_amplitude"] for m, x in rows])
    wmax = np.array([np.max(np.abs(x["mean_phi"] - x["phi_dw"])) / m["oscillation_amplitude"] for m, x in rows])
    wmean = np.array([np.mean(np.abs(x["mean_phi"] - x["phi_dw"])) / m["oscillation_amplitude"] for m, x in rows])
    ax = axes[1]
    ax.plot(ejs, dmax, "o-", color=C_TARGET, label=r"$D_{\max}$, bare")
    ax.plot(ejs, dmean, "o:", color=C_TARGET, mfc="white", label=r"$\bar D$, bare")
    ax.plot(ejs, wmax, "s-", color=C_DW, label=r"$D_{\max}$, DW")
    ax.plot(ejs, wmean, "s:", color=C_DW, mfc="white", label=r"$\bar D$, DW")
    ax.axhline(0.1, color="0.55", lw=0.6, ls=":")
    ax.set(xscale="log", yscale="log", ylim=(8e-4, 2.0), xlabel=r"$E_J$", ylabel="normalized deviation")
    ax.set_xticks([0.25, 1, 4, 16], labels=["0.25", "1", "4", "16"])
    ax.legend(loc="lower left", ncol=1, handlelength=1.6, labelspacing=0.25)
    panel(ax, "(b)")

    ax = axes[2]
    tau_p = np.array([m["tau_ehrenfest_classical"] for m, _ in rows])
    tau_w = [m["tau_ehrenfest_dw"] for m, _ in rows]
    ax.axhline(5.2, color="0.55", lw=0.6, ls=":")
    ax.plot(ejs, tau_p, "o-", color=C_TARGET, label="bare")
    fixed = [(1, rows[2][0]["tau_ehrenfest_classical"])] + [(ej, e3(ej, "fixed_variance")[0]["tau_ehrenfest_classical"]) for ej in (8, 32)]
    ax.plot([f[0] for f in fixed], [f[1] for f in fixed], "^--", color=C_FIXED, label="fixed variance")
    crossed = [(ej, t) for ej, t in zip(ejs, tau_w) if t is not None]
    censored = [ej for ej, t in zip(ejs, tau_w) if t is None]
    ax.plot([c[0] for c in crossed], [c[1] for c in crossed], "s", color=C_DW, label="DW")
    ax.plot(censored, [5.2] * len(censored), "s", color=C_DW, mfc="white", clip_on=False)
    ax.set(xscale="log", ylim=(0.0, 6.4), xlabel=r"$E_J$", ylabel=r"first crossing $\tau_{\rm E}$")
    ax.set_xticks([0.25, 1, 4, 16], labels=["0.25", "1", "4", "16"])
    ax.legend(loc="lower left", labelspacing=0.25, borderaxespad=0.3, fontsize=6.5, handlelength=1.8)
    panel(ax, "(c)", y=0.965)

    # (d) terms of the momentum equation at E_J = 1: closure residual, its non-Gaussian part, and the
    # finite-commutator residual of the cutoff (sine_gordon_code.native_correction.moment_diagnostics)
    ax = axes[3]
    floor = 1e-16
    series = (("closure_momentum", 1.0, C_TARGET, "-", "closure", (2.2, 1.5)),
              ("delta_NG_1", meta["E_J"], C_DW, "--", "non-Gaussian\npart", (0.35, 3e-8)),
              ("cutoff_momentum", 1.0, C_NATIVE, "-", "finite\ncutoff", (0.35, 3e-13)))
    for key, scale, color, ls, label, (x_text, y_text) in series:
        values = np.maximum(scale * np.max(np.abs(a[key]), axis=1), floor)
        ax.plot(tau, values, color=color, ls=ls)
        ax.text(x_text, y_text, label, color=color, fontsize=6.5, ha="left", va="center", linespacing=1.0)
    ax.set(xlim=(0, 5.2), yscale="log", ylim=(floor, 1e3), xlabel=r"$\tau=\Omega t$", ylabel="momentum residual")
    ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
    panel(ax, r"(d) $E_J=1$")
    save(fig, "fig_sg_classical", preview)


# ----------------------------------------------------------------- polynomial
def fig_sg_polynomial(preview):
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.75), gridspec_kw=dict(wspace=0.42))
    styles = {
        "taylor": ("Taylor", C_TARGET, "o"),
        "chebyshev_spectral": ("Chebyshev, spectral", "#0072B2", "s"),
        "chebyshev_accessed": ("Chebyshev, accessed", "#E69F00", "^"),
    }
    with np.load(SG_DATA / "formal" / "polynomial" / "E3_poly_phi3_N64.npz") as a:
        degs = a["degrees"]
        ax = axes[0]
        for key, (lab, c, mk) in styles.items():
            y = np.clip(a[f"{key}_max"], 1e-9, None)
            lim = a[f"{key}_cutoff_limited"]
            ax.plot(degs, y, "-", color=c, lw=0.8)
            ax.plot(degs[~lim], y[~lim], mk, color=c, label=lab)
            if lim.any():
                ax.plot(degs[lim], y[lim], mk, color=c, mfc="white")
    ax.axhline(1e-3, color="0.55", lw=0.6, ls=":")
    ax.set(yscale="log", ylim=(1e-8, 2), xlabel="degree $d$", ylabel=r"$\mathcal{I}_d$")
    ax.set_xticks([4, 8, 12, 16, 20, 24])
    panel(ax, r"(a) $\phi_0=3$", x=0.42)

    ax = axes[1]
    phis = [0.5, 1, 2, 3]
    for key, (lab, c, mk) in styles.items():
        ds = []
        for phi in ("0.5", "1", "2", "3"):
            with np.load(SG_DATA / "formal" / "polynomial" / f"E3_poly_phi{phi}_N64.npz") as a:
                ok = (a[f"{key}_max"] < 1e-3) & (~a[f"{key}_cutoff_limited"])
                ds.append(int(a["degrees"][ok][0]))
        ax.plot(phis, ds, mk + "-", color=c, lw=0.8, label=lab)
    ax.set(xlabel=r"$\phi_0$", ylabel=r"$d^\star$", ylim=(2, 16))
    ax.set_xticks(phis, labels=["0.5", "1", "2", "3"])
    ax.set_yticks([4, 8, 12, 16])
    panel(ax, "(b)")
    h = [Line2D([], [], color=c, marker=mk, lw=0.8, label=lab) for lab, c, mk in styles.values()]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=6.5, columnspacing=0.9, handlelength=1.5)
    save(fig, "fig_sg_polynomial", preview)


# ------------------------------------------------------------------- Duffing
def fig_duffing_native(preview):
    scan = json.loads((DF_DATA / "duffing_amplitude_scan.json").read_text())
    rows = scan["rows"]
    xi = np.array([r["xi"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.0), gridspec_kw=dict(wspace=0.36, width_ratios=[1.05, 0.9, 1.15]))

    ax = axes[0]
    for x0 in (1.75, 2.0):
        ax.vlines(x0, 5e-4, 1.6, color="0.85", lw=0.8, zorder=0)
    ax.plot(xi, [r["max_native_infidelity"] for r in rows], color=C_TARGET, label=r"${\rm max}_t\,\mathcal{I}_{\rm K}$")
    ax.plot(xi, np.minimum(1, [r["duhamel_infidelity_bound"] for r in rows]), color=C_NATIVE, ls="--", label="uncentered bound")
    ax.plot(xi, np.minimum(1, [r["centered_infidelity_bound"] for r in rows]), color=C_HYBRID, ls="-.", label=r"residual bound $B_\psi^2$")
    ax.plot(xi, [1 - r["diagonal_ceiling_T"] for r in rows], color=C_RETUNED, ls=":", lw=1.3, label=r"$1-F^{\max}_{\rm diag}(T)$")
    ax.axhline(1e-3, color="0.55", lw=0.6, ls=":")
    ax.set(yscale="log", xlim=(0.25, 2.5), ylim=(5e-4, 1.6), xlabel=r"$\xi$", ylabel="infidelity or bound")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.3, handlelength=1.9, labelspacing=0.2, columnspacing=0.8, borderaxespad=0.1)
    panel(ax, "(a)")

    ax = axes[1]
    for key, c, mk in (("1.75", C_HYBRID, "o"), ("2.0", C_PROG, "s")):
        e = scan["cutoff_convergence"][key]
        ax.plot([x["N"] for x in e], [x["max_infidelity"] for x in e], mk + "-", color=c, label=rf"$\xi={float(key):.2f}$")
    ax.set(yscale="log", xlabel=r"cutoff $N$", ylabel=r"cutoff infidelity $\epsilon_N$", ylim=(1e-9, 3e-2))
    ax.legend(loc="upper right")
    panel(ax, "(b)", x=0.05, y=0.1)

    ax = axes[2]
    grid = json.loads((DF_DATA / "formal" / "E1_retry" / "summary.json").read_text())["points"]
    betas, xis = [0.02, 0.05, 0.1, 0.2], [0.5, 1.0, 1.75, 2.0, 2.5]
    val = np.full((len(betas), len(xis)), np.nan)
    certified = {}
    for p in grid:
        if p["T"] != 1.0 or p["beta"] == 0:
            continue
        i, j = betas.index(p["beta"]), xis.index(p["xi"])
        val[i, j] = p["max_native_infidelity"]
        certified[(i, j)] = [v["verdict"] for v in p["verdicts"]]  # thresholds 1e-3, 1e-2, 1e-4
    im = ax.imshow(val, origin="lower", cmap="Greys", norm=LogNorm(1e-4, 1), aspect="auto", alpha=0.75)
    for (i, j), v in np.ndenumerate(val):
        txt = np.format_float_positional(v, precision=2, fractional=False, unique=False, trim="k")
        ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white" if v > 0.1 else "black")
        verdicts = certified[(i, j)]
        if verdicts[0] == "sufficient":
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=C_HYBRID, lw=1.3))
        elif verdicts[1] == "sufficient":
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=C_HYBRID, lw=0.9, ls="--"))
    ax.set_xticks(range(len(xis)), labels=["0.5", "1", "1.75", "2", "2.5"])
    ax.set_yticks(range(len(betas)), labels=["0.02", "0.05", "0.1", "0.2"])
    ax.tick_params(which="both", top=False, right=False, length=0)
    ax.set(xlabel=r"$\xi$", ylabel=r"$\beta$")
    ax.set_title(r"(c) ${\rm max}_t\,\mathcal{I}_{\rm K}$ at $T=1$", fontsize=8, loc="left", pad=3)
    save(fig, "fig_duffing_native", preview)


def fig_duffing_correction(preview):
    s = json.loads((DF_DATA / "duffing_refinement" / "comparison" / "evaluation_phase_coverage" / "summary.json").read_text())
    routes = {}
    for r in s["rows"]:
        routes.setdefault(r["route"], {})[r["input"]] = r
    inputs = list(routes["native"].keys())
    dev = [k for k in inputs if routes["native"][k]["input_role"] == "development"]
    new = [k for k in inputs if routes["native"][k]["input_role"] != "development"]
    order = sorted(dev, key=lambda k: routes["native"][k]["max_common_distance"]) + \
        sorted(new, key=lambda k: routes["native"][k]["max_common_distance"])
    x = np.arange(len(order)) + np.array([0] * len(dev) + [1] * len(new))
    fig, ax = plt.subplots(figsize=(COL1, 2.15))
    spec = [
        ("native", "native Kerr", C_NATIVE, "x", {}),
        ("gaussian", "Kerr + squeezing", C_NATIVE, "+", {}),
        ("exact_split", "exact split", C_TARGET, "D", dict(mfc="white", ms=3.0, mew=0.7)),
        ("hybrid", r"$U_{\rm hyb}$: 11 Kerr waits, 120 SNAP", C_HYBRID, "o", dict(ms=4.2)),
        ("replacement", r"$U_{\rm rep}$: waits$\,\to\,$SNAP, 131 SNAP", C_PROG, "o", dict(mfc="none", ms=6.0, mew=0.8)),
    ]
    for key, lab, c, mk, kw in spec:
        y = [routes[key][k]["max_common_distance"] for k in order]
        ax.plot(x, y, mk, color=c, label=lab, ls="", **kw)
    ax.axhline(np.sqrt(1e-3), color="0.4", lw=0.7, ls=":")
    ax.text(x[-1] + 0.6, np.sqrt(1e-3) * 1.12, r"$\sqrt{10^{-3}}$", ha="right", va="bottom", fontsize=6.5, color="0.35")
    ax.axvline(len(dev), color="0.8", lw=0.6)
    ax.set(yscale="log", ylim=(6e-5, 1.5), xlim=(-0.8, x[-1] + 0.8), ylabel=r"${\rm max}_j\, d_{\rm pr}(t_j)$")
    ax.set_xticks([np.mean(x[: len(dev)]), np.mean(x[len(dev):])], labels=["development inputs (10)", "held-out coherent states (12)"])
    ax.tick_params(axis="x", which="both", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.33), ncol=2, fontsize=6.5, columnspacing=0.8, handletextpad=0.3)
    save(fig, "fig_duffing_correction", preview)


def fig_duffing_training(preview):
    """Earlier real-amplitude training against the phase-covering training (appendix)."""
    e2 = json.loads((DF_DATA / "formal" / "E2" / "summary.json").read_text())
    new = json.loads((DF_DATA / "duffing_refinement" / "comparison" / "evaluation_phase_coverage" / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    names = ["real1.75", "real2", "diagonal1.75", "diagonal2", "imag1.75", "imag2", "vacuum", "one", "zero_plus_one", "zero_plus_i_two"]
    labels = [r"$1.75$", r"$2$", r"$1.75e^{i\pi/4}$", r"$2e^{i\pi/4}$", r"$1.75i$", r"$2i$", r"$|0\rangle$", r"$|1\rangle$", r"$|0\rangle{+}|1\rangle$", r"$|0\rangle{+}i|2\rangle$"]
    comp = [c for c in e2["comparisons"] if c["r"] == 10 and c["layers"] == 12][0]
    old = {r["input"]: r["max_common_distance"] for r in comp["rows"] if r["route"] == "hybrid"}
    nat = {r["input"]: r["max_common_distance"] for r in new["rows"] if r["route"] == "native"}
    hyb = {r["input"]: r["max_common_distance"] for r in new["rows"] if r["route"] == "hybrid"}
    x = np.arange(len(names))
    ax.plot(x, [nat[k] for k in names], "x", color=C_NATIVE, label="native Kerr")
    if old:
        ax.plot(x, [old[k] for k in names], "s", color=C_SNAP, label="trained on real amplitudes")
    ax.plot(x, [hyb[k] for k in names], "o", color=C_HYBRID, label="trained on eight phases")
    ax.axhline(np.sqrt(1e-3), color="0.4", lw=0.7, ls=":")
    ax.set(yscale="log", ylim=(5e-4, 4), ylabel=r"${\rm max}_j\, d_{\rm pr}(t_j)$")
    ax.set_xticks(x, labels=labels, rotation=40, ha="right", fontsize=6.5)
    ax.tick_params(axis="x", which="both", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=6.5, columnspacing=0.8, handletextpad=0.3)
    save(fig, "fig_duffing_training", preview)


# ------------------------------------------------------------ double sine-Gordon
def _scan():
    return json.loads((DSG_DATA / "dsg_step_scan.json").read_text())


def _cheapest(scan, route, prep=(1.0, 0.0)):
    rows = [r for r in scan["rows"] if r["route"] == route and tuple(r["preparation"]) == prep and r["all_criteria"]]
    return min(rows, key=lambda r: (r["total_programmable_primitives"], r["state_error"]))


ROUTE_STYLE = {
    "hybrid": (C_HYBRID, "hybrid", "o"),
    "gauss_block": (C_GAUSS, "Gaussian block", "D"),
    "kin_pot": (C_PROG, "kinetic-potential", "s"),
}


def fig_dsg_dynamics(preview):
    with np.load(DSG_DATA / "sg128" / "reference" / "E4_reference_lambda0.3_N128.npz") as a:
        ref = {k: a[k] for k in a.files}
    t = ref["t"]
    summ = json.loads((DSG_DATA / "sg128" / "reference" / "summary.json").read_text())
    t_e = [r for r in summ["records"] if r["lambda2"] == 0.3 and r["N"] == 128][0]["mean_field_crossing"]["time"]
    scan = _scan()
    best = {route: _cheapest(scan, route) for route in ROUTE_STYLE}
    mod = lambda re, im: np.abs(np.asarray(re) + 1j * np.asarray(im))

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.95), gridspec_kw=dict(wspace=0.34))
    for ax in axes:
        ax.axvspan(0, t_e, color="0.92", lw=0, zorder=0)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel(r"$t$")
    hyb = best["hybrid"]["series"]
    ax = axes[0]
    ax.plot(t, mod(ref["target_C12_real"], ref["target_C12_imag"]), color=C_TARGET)
    ax.plot(t, mod(ref["native_C12_real"], ref["native_C12_imag"]), color=C_NATIVE, ls="--")
    ax.plot(t, mod(ref["retuned_native_C12_real"], ref["retuned_native_C12_imag"]), color=C_RETUNED, ls="-.")
    ax.plot(hyb["t"], mod(hyb["C12_real"], hyb["C12_imag"]), "o", color=C_HYBRID, ms=3.2)
    ax.set(ylabel=r"$|C_{12}|$", ylim=(-0.01, 0.32))
    panel(ax, "(a)")
    ax = axes[1]
    ax.plot(t, ref["target_S2"], color=C_TARGET)
    ax.plot(t, ref["native_S2"], color=C_NATIVE, ls="--")
    ax.plot(t, ref["retuned_native_S2"], color=C_RETUNED, ls="-.")
    ax.plot(hyb["t"], hyb["S2"], "o", color=C_HYBRID, ms=3.2)
    ax.set(ylabel=r"$S_2$", ylim=(-0.01, 0.27))
    panel(ax, "(b)")
    ax = axes[2]
    ax.plot(t[1:], ref["native_distance"][1:], color=C_NATIVE, ls="--")
    ax.plot(t[1:], ref["retuned_native_distance"][1:], color=C_RETUNED, ls="-.")
    for route, (c, lab, mk) in ROUTE_STYLE.items():
        ser = best[route]["series"]
        ax.plot(ser["t"][1:], ser["bures"][1:], mk + "-", color=c, ms=2.8 if mk != "o" else 3.2, lw=0.8)
    ax.axhline(np.sqrt(1e-3), color="0.4", lw=0.6, ls=":")
    ax.set(yscale="log", ylim=(1e-3, 1), ylabel=r"distance from target")
    panel(ax, "(c)")
    h = [Line2D([], [], color=C_TARGET, label="target"),
         Line2D([], [], color=C_NATIVE, ls="--", label=r"native only, $E_J$"),
         Line2D([], [], color=C_RETUNED, ls="-.", label=r"native only, $E_J+4\lambda_2$"),
         plt.Rectangle((0, 0), 1, 1, fc="0.92", ec="none", label=r"$t<t_{\rm E}$")]
    for route, (c, lab, mk) in ROUTE_STYLE.items():
        b = best[route]
        h.append(Line2D([], [], color=c, marker=mk, ms=3.0, lw=0.8,
                        label=rf"{lab}, $r={b['r']}$ ({b['total_programmable_primitives']})"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=4, fontsize=6.8, columnspacing=1.2, handlelength=2.0)
    save(fig, "fig_dsg_dynamics", preview)


def fig_dsg_resources(preview):
    scan = _scan()
    split = json.loads((DSG_DATA / "splitting_baselines.json").read_text())
    sweep = json.loads((DSG_DATA / "splitting_sweep.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.25), gridspec_kw=dict(wspace=0.36, width_ratios=[1.45, 1.0, 1.0]))

    ax = axes[0]
    for route, (c, lab, mk) in ROUTE_STYLE.items():
        for order, fill, fs in ((1, "white", "full"), (2, c, "full"), (3, c, "left")):
            rows = sorted([r for r in scan["rows"] if r["route"] == route and r["order"] == order and r["s"] == 1
                           and tuple(r["preparation"]) == (1.0, 0.0)], key=lambda r: r["r"])
            if not rows:
                continue
            xs = [r["total_programmable_primitives"] for r in rows]
            ys = [r["state_error"] for r in rows]
            ax.plot(xs, ys, "-", color=c, lw=0.6, alpha=0.6)
            ax.plot(xs, ys, mk, color=c, mfc=fill, mfcalt="white", fillstyle=fs, ms=3.4, mew=0.7, ls="")
            for r, x, y in zip(rows, xs, ys):
                if not r["all_criteria"] and r["acceptance"]["state"]:
                    ax.plot(x, y, "x", color="k", ms=3.0, mew=0.6)
        best = _cheapest(scan, route)
        ax.plot(best["total_programmable_primitives"], best["state_error"], "*", color="k", mfc=c, ms=8, mew=0.5, zorder=5)
        offset, ha, va = {"hybrid": ((0, -8), "center", "top"), "gauss_block": ((-7, -5), "right", "top"),
                          "kin_pot": ((13, 8), "left", "bottom")}[route]
        ax.annotate(f"{best['total_programmable_primitives']}", (best["total_programmable_primitives"], best["state_error"]),
                    xytext=offset, textcoords="offset points", fontsize=6.5, ha=ha, va=va, color=c)
    ax.axhline(np.sqrt(1e-3), color="0.4", lw=0.7, ls=":")
    ax.set(xscale="log", yscale="log", xlabel="ideal programmable primitives", ylabel="state error incl. margins")
    panel(ax, "(a)")

    ax = axes[1]
    T = split["T"]
    sc = split["scan"]
    hs = [T / r for r in sc["r"]]
    styles = [("hybrid", C_HYBRID, "o-", r"$H_{\rm SG}|\Delta H$"), ("hybrid_ret", C_HYBRID_RET, "^-", "retuned"),
              ("gauss_cos", C_GAUSS, "D-", "Gaussian block"), ("kin_pot", C_PROG, "s-", "kinetic-potential"),
              ("gauss_harm", "0.55", "v:", "harmonic block")]
    for key, c, st, lab in styles:
        ax.plot(hs, sc["errors"][key], st, color=c, ms=2.8, lw=0.8)
    hh = np.array([0.075, 0.45])
    ax.plot(hh, 1.2e-3 * (hh / 0.1) ** 2, color="0.5", lw=0.6, ls="--")
    ax.text(0.085, 2.9e-4, r"$\propto h^2$", fontsize=6.5, color="0.4")
    ax.set(xscale="log", yscale="log", xlabel=r"outer step $h=T/r$", ylabel="outer splitting error", ylim=(2e-4, 0.4))
    ax.set_xticks([0.1, 0.2, 0.4], labels=["0.1", "0.2", "0.4"])
    ax.set_xticks([], minor=True)
    panel(ax, "(b)")

    ax = axes[2]
    lam = sorted({p["lambda2"] for p in sweep["points"]})
    for kappa, ls in ((0.5, ":"), (1.0, "-"), (2.0, "--")):
        pts = sorted([p for p in sweep["points"] if p["kappa"] == kappa], key=lambda p: p["lambda2"])
        g = [p["errors"]["gauss_cos"][1] / p["errors"]["hybrid"][1] for p in pts]
        k = [p["errors"]["kin_pot"][1] / p["errors"]["hybrid"][1] for p in pts]
        ax.plot([p["lambda2"] for p in pts], g, "D" + ls, color=C_GAUSS, ms=1.6, lw=0.8)
        ax.plot([p["lambda2"] for p in pts], k, "s" + ls, color=C_PROG, ms=1.6, lw=0.8)
    # compiled ratios at the original threshold, including the lambda2 = 0.6 Gaussian-block circuit that is resolved
    # at cutoffs 160/192 (dsg_thresholds.py, Table III)
    thresholds = json.loads((DSG_DATA / "dsg_thresholds.json").read_text())["results"]
    compiled = [(e["lambda2"], e["saving_gauss_block"], e["saving_kin_pot"]) for e in thresholds if e["infidelity"] == 1e-3]
    ax.plot([c[0] for c in compiled], [c[1] for c in compiled], "*", color="k", mfc=C_GAUSS, ms=9, mew=0.6, zorder=5)
    ax.plot([c[0] for c in compiled], [c[2] for c in compiled], "*", color="k", mfc=C_PROG, ms=9, mew=0.6, zorder=5)
    ax.set(xscale="log", yscale="log", xlabel=r"$\lambda_2$ ($E_J=1$)", ylabel="ratio to hybrid", ylim=(1, 90))
    ax.set_xticks(lam, labels=[f"{x:g}" for x in lam])
    ax.set_xticks([], minor=True)
    panel(ax, "(c)", x=0.62)

    h = [Line2D([], [], color=c, marker=mk, lw=0.6, ms=3.2, label=lab) for c, lab, mk in
         [(C_HYBRID, r"hybrid, $H_{\rm SG}|\Delta H$", "o"), (C_HYBRID_RET, "retuned hybrid", "^"),
          (C_GAUSS, r"Gaussian block, $(H_{\rm kin}+H_{\rm bond})|(V_J+\Delta H)$", "D"),
          (C_PROG, r"kinetic-potential, $H_{\rm kin}|(H_{\rm bond}+V_J+\Delta H)$", "s")]]
    h += [Line2D([], [], color="0.55", marker="v", ls=":", ms=3.2, label="harmonic block"),
          Line2D([], [], color="k", marker="*", ls="", ms=7, mfc="0.6", mew=0.5, label="cheapest passing (a), primitive ratio (c)"),
          Line2D([], [], color="0.3", ls=":", label=r"$\kappa=0.5$"), Line2D([], [], color="0.3", ls="-", label=r"$\kappa=1$"),
          Line2D([], [], color="0.3", ls="--", label=r"$\kappa=2$")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=6.6, columnspacing=1.1, handlelength=2.0)
    save(fig, "fig_dsg_resources", preview)


def fig_dsg_subdivisions(preview):
    """Archived r = 10, 20 comparison with s = 1..32 at cutoffs 96/128 (appendix)."""
    agg = json.loads((COMMON_DATA / "analysis_summary.json").read_text())["E4_compiled"]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    style = {
        "H_trig": (C_HYBRID, "hybrid"),
        "H_trig_ret": (C_HYBRID_RET, "retuned hybrid"),
        "G_trig": (C_PROG, "kinetic-potential"),
        "H_SNAP": (C_SNAP, "SNAP+D hybrid"),
    }
    for route, (c, lab) in style.items():
        for r_steps, mk in ((10, "o"), (20, "s")):
            for order, fill in ((1, "white"), (2, c)):
                if route == "H_SNAP" and order == 1:
                    continue
                cfg = sorted([k for k in agg["rows"] if k["route"] == route and k["r"] == r_steps and (k["order"] == order or route == "H_SNAP")],
                             key=lambda k: k["total_programmable_primitives"] - k["primitive_counts"].get("Ry", 0))
                if not cfg:
                    continue
                # archived counts conjugate every trigonometric factor by R_y(+-pi/2); with the qubit basis change
                # applied once per circuit (trig_counts.py) a circuit needs CD + Rz + Gaussian + SNAP + D + 4 primitives
                xs = [k["total_programmable_primitives"] - (k["primitive_counts"].get("Ry", 0) - 4 if route != "H_SNAP" else 0)
                      for k in cfg]
                ys = [k["amplitude"] for k in cfg]
                ax.plot(xs, ys, "-", color=c, lw=0.6, alpha=0.6)
                if route == "H_SNAP":
                    ax.plot(xs, ys, "^" if r_steps == 10 else "v", color=c, mfc=c, ms=3.4, mew=0.7)
                else:
                    ax.plot(xs, ys, mk, color=c, mfc=fill, ms=3.2, mew=0.7)
    ax.axhline(np.sqrt(1e-3), color="0.4", lw=0.7, ls=":")
    ax.set(xscale="log", yscale="log", xlim=(60, 2e4), ylim=(3e-3, 0.35),
           xlabel="ideal programmable primitives", ylabel="state error incl. margins")
    h = [Line2D([], [], color=c, marker="^" if lab.startswith("SNAP") else "o", lw=0.6, ms=3.2, label=lab) for c, lab in style.values()]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=6.5, columnspacing=1.0, handletextpad=0.4)
    save(fig, "fig_dsg_subdivisions", preview)


def fig_dsg_convergence(preview):
    """Reference cutoff convergence for the double sine-Gordon instance (appendix)."""
    pairs = ["32→48", "48→64", "64→96", "96→128", "128→160"]
    state = [0.04253, 0.01676, 0.006886064923459318, 0.0013148548922630218, 2.30e-4]
    # values quoted from double_sine_gordon_data records: stage1 E4_reference, stage1_continuation, sg128/reference, sg160
    fig, ax = plt.subplots(figsize=(COL1, 1.7))
    ax.plot(range(len(pairs)), state, "o-", color=C_TARGET, label="state distance")
    ax.axhline(np.sqrt(1e-3) / 10, color="0.4", lw=0.7, ls=":")
    ax.set(yscale="log", ylabel="adjacent-cutoff change", ylim=(1e-4, 0.1))
    ax.set_xticks(range(len(pairs)), labels=pairs)
    save(fig, "fig_dsg_convergence", preview)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default=None)
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    if args.preview:
        Path(args.preview).mkdir(parents=True, exist_ok=True)
    fig_sg_classical(args.preview)
    fig_sg_polynomial(args.preview)
    fig_duffing_native(args.preview)
    fig_duffing_correction(args.preview)
    fig_duffing_training(args.preview)
    fig_dsg_dynamics(args.preview)
    fig_dsg_resources(args.preview)
    fig_dsg_subdivisions(args.preview)


if __name__ == "__main__":
    main()
