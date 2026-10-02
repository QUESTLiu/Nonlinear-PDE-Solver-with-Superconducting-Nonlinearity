#!/usr/bin/env python3
"""
run_snapd.py -- Section 3.7, SNAP+D compilation of programmable
deformations, with a wall-clock budget.

This is the script that closes the gap the referee will find: across the
whole manuscript, no nontrivial correction has yet been compiled into a
finite-depth circuit with a reported gate count.

The objective is OPERATOR fidelity against a single Trotter block,

    F = |Tr(U_target^dag U(theta, alpha))|^2 / d^2,
    U_target = exp(-i dH_site * dt)

not trajectory fidelity.  Optimising against a trajectory requires the
exact solution to define the loss and gives no guarantee outside the
training horizon and initial states.  An operator-fidelity block is
compiled once and composed for arbitrary time and arbitrary input, which
is what a compilation claim requires.

Targets
-------
disorder          delta * (1 - cos phi)          site-resolved junction disorder
second_harmonic   lam   * (1 - cos 2 phi)        double sine-Gordon
native_cosine     E_J   * (1 - cos phi)          cost of synthesising the
                                                 interaction the hardware
                                                 already has -- the number
                                                 that makes Sec. 3.5 quantitative

Reports operator infidelity against layer count L, and converts it into
a circuit-time budget: L ancilla-mediated SNAP operations per site per
Trotter step at t_SNAP ~ 2 pi / chi, against T2 of the ancilla.

Example
-------
python run_snapd.py --E-J 1.0 --kappa 1.0 --n-cut 20 --target second_harmonic \
    --coupling 0.3 --block-dt 0.05 --layers 1,2,4,6,8,12,16 --n-restarts 8
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

if __package__:
    from sine_gordon_code import sglib as sg, common
else:
    import sglib as sg
    import common


# ---------------------------------------------------------------------

def displacement(a, alpha):
    ad = a.conj().T
    return expm(alpha * ad - np.conj(alpha) * a)


def snap(thetas):
    return np.diag(np.exp(1j * thetas))


def unpack(x, L, n_cut):
    """x -> list of (alpha, thetas) per layer."""
    out = []
    k = 0
    for _ in range(L):
        ar, ai = x[k], x[k + 1]
        k += 2
        th = x[k:k + n_cut]
        k += n_cut
        out.append((ar + 1j * ai, th))
    return out


def build_ansatz(x, L, n_cut, a):
    U = np.eye(n_cut, dtype=complex)
    for alpha, th in unpack(x, L, n_cut):
        U = displacement(a, alpha) @ snap(th) @ U
    return U


def objective_and_gradient(x, U_target, L, n_cut, a):
    """Operator-infidelity objective and analytic reverse-mode gradient.

    The original driver let SciPy finite-difference every parameter.  At
    L=24 and n_cut=20 that means more than 500 circuit evaluations per
    optimizer iteration.  This version differentiates the circuit product
    analytically; only the two displacement derivatives in each layer require
    matrix-exponential Frechet derivatives.
    """
    ad = a.conj().T
    e_ar = ad - a
    e_ai = 1j * (ad + a)
    parsed = unpack(x, L, n_cut)

    gates = []
    d_ar = []
    d_ai = []
    snaps = []
    for alpha, theta in parsed:
        generator = alpha * ad - np.conj(alpha) * a
        D = expm(generator)
        S = snap(theta)
        gates.append(D @ S)
        snaps.append(S)
        d_ar.append(expm_frechet(generator, e_ar, compute_expm=False))
        d_ai.append(expm_frechet(generator, e_ai, compute_expm=False))

    # prefix[l] = G_{l-1} ... G_0
    prefix = [np.eye(n_cut, dtype=complex)]
    for gate in gates:
        prefix.append(gate @ prefix[-1])
    # suffix[l] = G_{L-1} ... G_{l+1}
    suffix = [None] * L
    running = np.eye(n_cut, dtype=complex)
    for layer in range(L - 1, -1, -1):
        suffix[layer] = running
        running = running @ gates[layer]

    U = prefix[-1]
    target_dag = U_target.conj().T
    overlap = np.trace(target_dag @ U)
    loss = float(np.clip(1.0 - abs(overlap) ** 2 / n_cut**2, 0.0, 1.0))
    scale = -2.0 / n_cut**2
    grad = np.empty_like(x, dtype=float)

    cursor = 0
    for layer, (_alpha, theta) in enumerate(parsed):
        # Cyclic trace: Tr(Ut^dag suffix dG prefix) = Tr(B dG).
        B = prefix[layer] @ target_dag @ suffix[layer]
        dc_ar = np.trace(B @ d_ar[layer] @ snaps[layer])
        dc_ai = np.trace(B @ d_ai[layer] @ snaps[layer])
        grad[cursor] = scale * np.real(np.conj(overlap) * dc_ar)
        grad[cursor + 1] = scale * np.real(np.conj(overlap) * dc_ai)
        cursor += 2

        # dG/dtheta_n = D diag(i exp(i theta_n) e_n).  All SNAP-phase
        # derivatives follow from one diagonal extraction.
        D = gates[layer] @ snaps[layer].conj().T
        diag_bd = np.diag(B @ D)
        dc_theta = diag_bd * (1j * np.exp(1j * theta))
        grad[cursor:cursor + n_cut] = (
            scale * np.real(np.conj(overlap) * dc_theta)
        )
        cursor += n_cut
    return loss, grad


def _run_restart(payload):
    (U_target, L, n_cut, a, x0, maxiter, alpha_bound) = payload
    bounds = []
    for _ in range(L):
        bounds.extend([(-alpha_bound, alpha_bound)] * 2)
        bounds.extend([(-np.pi, np.pi)] * n_cut)

    def fun(x):
        return objective_and_gradient(x, U_target, L, n_cut, a)

    result = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=dict(maxiter=maxiter, maxfun=10**6, ftol=1e-14, gtol=1e-9),
    )
    return float(np.clip(result.fun, 0.0, 1.0)), result.x, bool(result.success), (
        result.message.decode() if isinstance(result.message, bytes)
        else str(result.message)
    )


def _checkpoint_signature(args, layers):
    return json.dumps(
        dict(
            target=args.target,
            E_J=args.E_J,
            coupling=args.coupling,
            block_dt=args.block_dt,
            n_cut=args.n_cut,
            layers=layers,
            n_restarts=args.n_restarts,
            maxiter=args.maxiter,
            seed=args.seed,
            alpha_bound=args.alpha_bound,
        ),
        sort_keys=True,
    )


def _write_checkpoint(path, signature, completed_layers, infidelities, vectors):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": np.array(signature),
        "completed_layers": np.asarray(completed_layers, dtype=int),
        "infidelities": np.asarray(infidelities, dtype=float),
    }
    payload.update(vectors)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary = handle.name
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_checkpoint(path, signature):
    if not path.exists():
        return [], [], {}
    with np.load(path, allow_pickle=False) as saved:
        found = str(saved["signature"].item())
        if found != signature:
            raise RuntimeError(
                f"checkpoint {path} belongs to a different parameter set; "
                "use a new --tag or remove the stale checkpoint"
            )
        layers = saved["completed_layers"].astype(int).tolist()
        infidelities = saved["infidelities"].astype(float).tolist()
        vectors = {
            key: saved[key].copy()
            for key in saved.files
            if key.startswith("x_L")
        }
    if len(layers) != len(infidelities):
        raise RuntimeError(f"invalid checkpoint: {path}")
    return layers, infidelities, vectors


def target_potential(kind, lam_phi, E_J, coupling):
    if kind == "disorder":
        return coupling * (1.0 - np.cos(lam_phi))
    if kind == "second_harmonic":
        return coupling * (1.0 - np.cos(2.0 * lam_phi))
    if kind == "native_cosine":
        return E_J * (1.0 - np.cos(lam_phi))
    raise ValueError(f"unknown target: {kind}")


# ---------------------------------------------------------------------

def main():
    p = common.base_parser(__doc__)
    p.add_argument("--target", default="second_harmonic",
                   choices=["disorder", "second_harmonic", "native_cosine"])
    p.add_argument("--coupling", type=float, default=0.3,
                   help="delta or lambda for the deformation")
    p.add_argument("--block-dt", type=float, default=0.05,
                   help="Trotter step the block must implement")
    p.add_argument("--layers", default="1,2,4,6,8,12,16",
                   help="comma-separated SNAP+D layer counts to sweep")
    p.add_argument("--n-restarts", type=int, default=8)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--workers", type=int, default=0,
        help="parallel restart workers; 0 uses LSB_DJOB_NUMPROC or 1")
    p.add_argument(
        "--alpha-bound", type=float, default=3.0,
        help="symmetric bound on displacement quadratures")
    p.add_argument(
        "--resume", action="store_true",
        help="resume completed layer counts from an atomic checkpoint")
    # hardware budget
    p.add_argument("--chi-mhz", type=float, default=1.5,
                   help="dispersive shift chi/2pi in MHz; sets t_SNAP")
    p.add_argument("--t2-anc-us", type=float, default=80.0,
                   help="ancilla T2 in microseconds")
    p.add_argument(
        "--t-displacement-us",
        type=float,
        default=0.0,
        help="duration of one displacement; 0 gives a SNAP-only lower bound",
    )
    p.add_argument(
        "--coherence-fraction",
        type=float,
        default=0.25,
        help="maximum allowed circuit time as a fraction of ancilla T2",
    )
    p.add_argument("--sim-time", type=float, default=5.0,
                   help="simulated time horizon for the budget")
    args = p.parse_args()

    layers = [int(v) for v in args.layers.split(",")]
    args.n_site = 1                       # single-mode compilation
    ctx = common.setup(args, f"snapd_{args.target}")
    basis = ctx["basis"]
    a = sg.annihilation(args.n_cut)

    V = target_potential(args.target, basis.lam_phi, args.E_J, args.coupling)
    # dH is diagonal in the phase eigenbasis; rotate back to Fock
    dH = basis.U_phi @ np.diag(V) @ basis.U_phi.conj().T
    U_target = expm(-1j * dH * args.block_dt)

    norm_dH = float(np.linalg.norm(dH, 2))
    print(f"[snapd] target={args.target} coupling={args.coupling:g} "
          f"block_dt={args.block_dt:g}")
    print(f"[snapd] ||dH||_2 = {norm_dH:.4f}   "
          f"||dH|| * block_dt = {norm_dH*args.block_dt:.4f}", flush=True)

    if args.dry_run:
        return

    if args.n_restarts < 1:
        raise ValueError("--n-restarts must be positive")
    if not 0.0 < args.coherence_fraction <= 1.0:
        raise ValueError("--coherence-fraction must lie in (0, 1]")
    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("LSB_DJOB_NUMPROC", "1"))
    workers = max(1, min(workers, args.n_restarts))
    print(f"[snapd] parallel restart workers = {workers}", flush=True)

    checkpoint = Path(args.outdir).expanduser().resolve() / (
        f"{ctx['tag']}_checkpoint.npz"
    )
    signature = _checkpoint_signature(args, layers)
    if args.resume:
        completed_layers, best_infid, best_x = _load_checkpoint(
            checkpoint, signature
        )
        if completed_layers:
            print(
                f"[snapd] resumed layers {completed_layers} from {checkpoint}",
                flush=True,
            )
    else:
        completed_layers, best_infid, best_x = [], [], {}
    previous_best = None
    previous_L = 0
    if completed_layers:
        previous_L = completed_layers[-1]
        previous_best = best_x[f"x_L{previous_L}"]

    optimization_seconds = []
    for L in layers:
        if L in completed_layers:
            optimization_seconds.append(float("nan"))
            continue
        layer_start = time.perf_counter()
        n_par = L * (args.n_cut + 2)
        best = (np.inf, None)
        starts = []
        for restart in range(args.n_restarts):
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, L, restart])
            )
            x0 = np.concatenate([
                np.concatenate([rng.normal(0, 0.3, 2),
                                rng.normal(0, 0.5, args.n_cut)])
                for _ in range(L)])
            if restart == 0 and previous_best is not None and L > previous_L:
                # Pad the preceding optimum with identity layers.  The first
                # restart therefore cannot be worse solely because L grew.
                x0[:] = 0.0
                x0[:previous_best.size] = previous_best
            starts.append(x0)

        payloads = [
            (U_target, L, args.n_cut, a, x0, args.maxiter, args.alpha_bound)
            for x0 in starts
        ]
        if workers == 1:
            outcomes = map(_run_restart, payloads)
            for restart, outcome in enumerate(outcomes):
                value, x_opt, success, message = outcome
                print(
                    f"[snapd] L={L:3d} restart={restart:2d} "
                    f"infid={value:.4e} success={success} {message}",
                    flush=True,
                )
                if value < best[0]:
                    best = (value, x_opt.copy())
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                future_to_restart = {
                    pool.submit(_run_restart, payload): restart
                    for restart, payload in enumerate(payloads)
                }
                for future in as_completed(future_to_restart):
                    restart = future_to_restart[future]
                    value, x_opt, success, message = future.result()
                    print(
                        f"[snapd] L={L:3d} restart={restart:2d} "
                        f"infid={value:.4e} success={success} {message}",
                        flush=True,
                    )
                    if value < best[0]:
                        best = (value, x_opt.copy())
        if best[1] is None:
            raise RuntimeError(f"all optimizer restarts failed at L={L}")
        best_infid.append(best[0])
        best_x[f"x_L{L}"] = best[1]
        completed_layers.append(L)
        previous_best = best[1]
        previous_L = L
        _write_checkpoint(
            checkpoint,
            signature,
            completed_layers,
            best_infid,
            best_x,
        )
        optimization_seconds.append(time.perf_counter() - layer_start)
        print(f"[snapd] L={L:3d}  params={n_par:4d}  "
              f"operator infidelity = {best[0]:.4e}  "
              f"opt_time={optimization_seconds[-1]:.1f} s", flush=True)

    if completed_layers != layers:
        raise RuntimeError(
            f"checkpoint layer order {completed_layers} does not match {layers}"
        )
    best_infid = np.asarray(best_infid, dtype=float)
    optimization_seconds = np.asarray(optimization_seconds, dtype=float)
    spectral_errors = np.asarray(
        [
            sg.phase_aligned_operator_distance(
                U_target,
                build_ansatz(best_x[f"x_L{L}"], L, args.n_cut, a),
            )
            for L in layers
        ],
        dtype=float,
    )

    # ---- hardware budget -------------------------------------------
    t_snap_us = 1.0 / args.chi_mhz          # ~ 2 pi / chi with chi/2pi in MHz
    t_layer_us = t_snap_us + args.t_displacement_us
    n_trotter = int(np.ceil(args.sim_time / args.block_dt))
    print(f"\n[snapd] --- circuit-time budget "
          f"(chi/2pi = {args.chi_mhz} MHz, t_SNAP = {t_snap_us:.3f} us) ---")
    print(f"[snapd] Trotter steps for T={args.sim_time:g}: {n_trotter}")
    coherence_budget = args.coherence_fraction * args.t2_anc_us
    print(f"[snapd] coherence safety budget = "
          f"{args.coherence_fraction:g} * T2_anc = "
          f"{coherence_budget:g} us")
    if args.t_displacement_us == 0:
        print(
            "[snapd] displacement duration is zero: circuit times are "
            "SNAP-only lower bounds",
            flush=True,
        )
    # The trace-infidelity accumulation is a heuristic coherent proxy.  The
    # spectral-norm column provides the rigorous telescoping bound for repeated
    # standalone blocks.
    budget = []
    trace_proxy = []
    composed_norm_bound = []
    worst_state_infid_bound = []
    for L, infid, spectral_error in zip(
        layers, best_infid, spectral_errors
    ):
        t_circ = L * n_trotter * t_layer_us
        proxy = min(1.0, (n_trotter * np.sqrt(max(infid, 0.0))) ** 2)
        norm_bound = min(2.0, n_trotter * spectral_error)
        state_bound = min(1.0, norm_bound**2)
        budget.append(t_circ)
        trace_proxy.append(proxy)
        composed_norm_bound.append(norm_bound)
        worst_state_infid_bound.append(state_bound)
        flag = "OK" if t_circ < coherence_budget else "EXCEEDS SAFETY BUDGET"
        print(f"[snapd] L={L:3d}  infid/block={infid:.3e}  "
              f"spectral_err={spectral_error:.3e}  "
              f"repeat_norm_bound<={norm_bound:.3e}  "
              f"trace_proxy~{proxy:.3e}  T_circ={t_circ:9.1f} us  "
              f"vs budget={coherence_budget:g} us "
              f"(T2_anc={args.t2_anc_us:g} us)  [{flag}]", flush=True)
    budget = np.array(budget)
    trace_proxy = np.array(trace_proxy)
    composed_norm_bound = np.array(composed_norm_bound)
    worst_state_infid_bound = np.array(worst_state_infid_bound)

    feasible = [(L, i, t) for L, i, t in zip(layers, best_infid, budget)
                if t < coherence_budget]
    if feasible:
        L_f, i_f, t_f = min(feasible, key=lambda z: z[1])
        print(f"\n[snapd] best infidelity within coherence: "
              f"{i_f:.3e} at L={L_f} (T_circ={t_f:.1f} us)", flush=True)
        ctx["params"].update(best_feasible_L=int(L_f),
                             best_feasible_infid=float(i_f))
    else:
        print(
            "\n[snapd] NO tested layer count fits the assumed coherence "
            "safety budget. Report this only from a converged production "
            "optimization, together with the timing assumptions.",
            flush=True,
        )
        ctx["params"]["best_feasible_L"] = -1

    ctx["params"].update(layers=layers, norm_dH=norm_dH,
                         t_snap_us=t_snap_us, n_trotter=n_trotter,
                         t_displacement_us=args.t_displacement_us,
                         t_layer_us=t_layer_us,
                         t2_anc_us=args.t2_anc_us,
                         sim_time=args.sim_time,
                         block_dt=args.block_dt,
                         target=args.target,
                         coupling=args.coupling,
                         coherence_fraction=args.coherence_fraction,
                         coherence_budget_us=coherence_budget,
                         workers=workers, alpha_bound=args.alpha_bound,
                         analytic_gradient=True, checkpoint=str(checkpoint),
                         resumed=bool(args.resume))
    sg.save_run(args.outdir, ctx["tag"], ctx["params"], write_csv=args.csv,
                arrays=dict(
        layers=np.array(layers), infidelity=best_infid, T_circ_us=budget,
        spectral_error=spectral_errors,
        repeated_operator_error_bound=composed_norm_bound,
        worst_state_infidelity_bound=worst_state_infid_bound,
        coherent_trace_infidelity_proxy=trace_proxy,
        optimization_seconds=optimization_seconds,
        lam_phi=basis.lam_phi, V_target=V, **best_x))


if __name__ == "__main__":
    main()
