#!/usr/bin/env python3
"""Section 3.4 -- Classical-field validation for the finite-cutoff sine-Gordon lattice.

The driver compares the quantum expectation values against:

1. the classical discrete sine-Gordon equation;
2. a diagnostic Debye--Waller reference with
   E_J -> E_J exp(-Delta_phi^2/2).

The classical references start from the expectations actually represented by
the finite Fock basis.  The maximum deviation is taken over stored times and
sites.  The Ehrenfest threshold is a user-selected fraction of the global
half peak-to-peak classical oscillation amplitude, and its first crossing is
linearly interpolated between observation times.

This is numerically converged finite-cutoff evolution, not an analytic "exact
evolution."  Use both convergence flags for production data.

For an E_J scan, use ``--tau-max`` so every point covers the same number of
local oscillator time units.  The production array uses phi0=1 and
tau_max=5.2; phi0=3 at T=20 lies near the cosine barrier and is not a
cutoff-efficient crossover benchmark.
"""

from __future__ import annotations

import numpy as np

if __package__:
    from . import common, sglib as sg
    from .native_correction import moment_diagnostics,prepare_moment_diagnostics
else:
    import common
    import sglib as sg
    from native_correction import moment_diagnostics,prepare_moment_diagnostics


def main() -> None:
    parser = common.base_parser(__doc__)
    parser.add_argument("--lambda2", type=float, default=0.0)
    parser.add_argument(
        "--ehrenfest-tol",
        type=float,
        default=0.1,
        help="threshold as a fraction of the half peak-to-peak amplitude",
    )
    parser.add_argument(
        "--propagator", choices=["strang", "krylov"], default="strang"
    )
    parser.add_argument("--krylov-m", type=int, default=30)
    parser.add_argument(
        "--convergence-check",
        action="store_true",
        help="rerun at n_cut-convergence_delta",
    )
    parser.add_argument("--convergence-delta", type=int, default=16)
    parser.add_argument(
        "--time-convergence-check",
        action="store_true",
        help="rerun Strang evolution at dt/2 on the same observation grid",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=-1.0,
        help=(
            "fixed scaled horizon tau_max=Omega_time*T; <=0 uses --T. "
            "Recommended for E_J scans."
        ),
    )
    parser.add_argument("--steps", type=int, default=None,
                        help="exact integration step count at the requested horizon")
    parser.add_argument("--zero-amplitude-scale", type=float, default=1.,
                        help="absolute normalization when the classical amplitude is zero")
    parser.add_argument(
        "--leakage-tol",
        type=float,
        default=1e-5,
        help="warning threshold for maximum top-level population",
    )
    args = parser.parse_args()
    if not 0.0 < args.ehrenfest_tol < 1.0:
        raise ValueError("--ehrenfest-tol must lie in (0,1)")

    tau_requested = None
    if args.tau_max > 0:
        omega_plan = float(args.omega_time) if args.omega_time is not None else float(np.sqrt(args.E_J + 2 * args.kappa))
        if omega_plan <= 0:
            raise ValueError("Omega_time must be positive")
        requested_time = args.tau_max / omega_plan
        planned_steps = max(1, int(round(requested_time / args.dt)))
        args.T = requested_time
        args.dt = args.T / (planned_steps if args.steps is None else args.steps)
        tau_requested = float(args.tau_max)
        print(
            f"[classical] scaled horizon tau_max={args.tau_max:g}: "
            f"T={args.T:g} ({planned_steps} steps)",
            flush=True,
        )
    if args.steps is not None:
        if args.steps < 1:
            raise ValueError("steps must be positive")
        args.dt = args.T / args.steps
    if args.zero_amplitude_scale <= 0:
        raise ValueError("zero-amplitude scale must be positive")

    ctx = common.setup(args, "classical")
    basis = ctx["basis"]
    if args.dry_run:
        return

    psi0, phis0, pis0, offsets, info = sg.make_initial_condition(
        args.init,
        args.n_site,
        args.n_cut,
        ctx["w"],
        args.E_J,
        args.kappa,
        args.phi0,
        args.pi0,
        args.site,
        args.use_offsets,
    )
    ctx["params"].update(info)
    print(f"[classical] requested phi_j(0) = {np.round(phis0, 5)}", flush=True)

    n_steps, stride = common.steps_and_stride(args.T, args.dt, args.n_obs)

    def quantum_trajectory(
        local_basis,
        local_psi0,
        local_offsets,
        dt,
        steps,
        obs_stride,
    ):
        onsite = sg.onsite_vals_with_offsets(
            "native",
            local_basis.lam_phi,
            args.E_J,
            args.n_site,
            local_offsets,
        )
        onsite += args.lambda2 * (1 - np.cos(2 * (local_basis.lam_phi[None, :] + local_offsets[:, None])))
        store = common.new_store()
        basic_observer = common.make_observer(
            local_basis.phi,
            local_basis.pi,
            local_basis.num,
            store,
            local_offsets,
        )
        hamiltonian = sg.HamiltonianApply(local_basis, args.n_site, args.kappa, onsite, args.pbc, offsets=local_offsets)
        prepared=prepare_moment_diagnostics(local_basis,args.E_J,args.lambda2,local_offsets)
        def observer(t, psi):
            basic_observer(t, psi)
            diagnostics = moment_diagnostics(psi, local_basis, args.E_J, args.kappa, args.lambda2, hamiltonian, offsets=local_offsets, pbc=args.pbc,prepared=prepared)
            for key, value in diagnostics.items():
                store.setdefault(key, []).append(value)
        if args.propagator == "strang":
            propagator = sg.SplitOperator(
                local_basis,
                args.n_site,
                args.kappa,
                onsite,
                args.pbc,
                offsets=local_offsets,
            )
            propagator.evolve(
                local_psi0, dt, steps, obs_stride, observer
            )
        else:
            hamiltonian = sg.HamiltonianApply(
                local_basis,
                args.n_site,
                args.kappa,
                onsite,
                args.pbc,
                offsets=local_offsets,
            )
            sg.krylov_evolve(
                hamiltonian,
                local_psi0,
                dt,
                steps,
                args.krylov_m,
                obs_stride,
                observer,
            )
        return common.stack(store)

    quantum = quantum_trajectory(
        basis, psi0, offsets, args.dt, n_steps, stride
    )
    times = quantum["t"]

    # These are the initial data promised by the manuscript definition.
    matched_phis0 = quantum["phi"][0].copy()
    matched_pis0 = quantum["pi"][0].copy()
    initial_representation_error = float(
        max(
            np.max(np.abs(matched_phis0 - phis0)),
            np.max(np.abs(matched_pis0 - pis0)),
        )
    )
    classical_phi, classical_pi = sg.classical_sine_gordon(
        matched_phis0,
        matched_pis0,
        args.E_J,
        args.kappa,
        times,
        args.pbc,
        lambda2=args.lambda2,
    )
    dw_phi, dw_pi = sg.classical_sine_gordon(
        matched_phis0,
        matched_pis0,
        args.E_J,
        args.kappa,
        times,
        args.pbc,
        E_J_eff=ctx["semi"]["E_J_eff"],
        lambda2=args.lambda2,
        lambda2_eff=args.lambda2 * np.exp(-2 * basis.delta_phi_sq),
    )

    site_amplitudes = 0.5 * np.ptp(classical_phi, axis=0)
    oscillation_amplitude = float(np.max(site_amplitudes))
    if oscillation_amplitude == 0:
        oscillation_amplitude = args.zero_amplitude_scale
    threshold = float(args.ehrenfest_tol * oscillation_amplitude)
    dev_classical = np.max(
        np.abs(quantum["phi"] - classical_phi), axis=1
    )
    dev_dw = np.max(np.abs(quantum["phi"] - dw_phi), axis=1)

    def first_crossing(deviation):
        indices = np.flatnonzero(deviation > threshold)
        if indices.size == 0:
            return None,None
        right = int(indices[0])
        if right == 0:
            return float(times[0]),[float(times[0]),float(times[0])]
        left = right - 1
        y0 = float(deviation[left])
        y1 = float(deviation[right])
        if y1 <= y0:
            return float(times[right]),[float(times[left]),float(times[right])]
        fraction = np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0)
        return float(times[left] + fraction * (times[right] - times[left])),[float(times[left]),float(times[right])]

    t_ehrenfest_classical,bracket_classical = first_crossing(dev_classical)
    t_ehrenfest_dw,bracket_dw = first_crossing(dev_dw)
    tau_ehrenfest_classical = (
        None
        if t_ehrenfest_classical is None
        else float(ctx["Omega_time"] * t_ehrenfest_classical)
    )
    tau_ehrenfest_dw = (
        None
        if t_ehrenfest_dw is None
        else float(ctx["Omega_time"] * t_ehrenfest_dw)
    )
    tau_actual = float(ctx["Omega_time"] * times[-1])

    cutoff_convergence = None
    cutoff_compared = None
    if args.convergence_check:
        cutoff_compared = args.n_cut - args.convergence_delta
        if cutoff_compared < 4:
            raise ValueError("n_cut-convergence_delta must be at least 4")
        basis_low = sg.build_mode_basis(cutoff_compared, ctx["w"])
        psi0_low, _, _, offsets_low, _ = sg.make_initial_condition(
            args.init,
            args.n_site,
            cutoff_compared,
            ctx["w"],
            args.E_J,
            args.kappa,
            args.phi0,
            args.pi0,
            args.site,
            args.use_offsets,
        )
        quantum_low = quantum_trajectory(
            basis_low,
            psi0_low,
            offsets_low,
            args.dt,
            n_steps,
            stride,
        )
        if quantum_low["phi"].shape != quantum["phi"].shape:
            raise RuntimeError("cutoff trajectories use different grids")
        cutoff_convergence = float(
            np.max(np.abs(quantum["phi"] - quantum_low["phi"]))
        )

    time_step_convergence = None
    if args.time_convergence_check:
        if args.propagator != "strang":
            raise ValueError(
                "--time-convergence-check applies only to Strang propagation"
            )
        quantum_half = quantum_trajectory(
            basis,
            psi0,
            offsets,
            args.dt / 2.0,
            2 * n_steps,
            2 * stride,
        )
        if quantum_half["phi"].shape != quantum["phi"].shape or not np.allclose(
            quantum_half["t"], times, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("dt and dt/2 runs use different observation grids")
        time_step_convergence = float(
            np.max(np.abs(quantum["phi"] - quantum_half["phi"]))
        )

    delta_phi = float(np.sqrt(ctx["semi"]["delta_phi_sq"]))
    if args.E_J > 0:
        force_renormalization_pct = float(
            100.0 * (1.0 - ctx["semi"]["E_J_eff"] / args.E_J)
        )
    else:
        force_renormalization_pct = 0.0
    max_dev_classical = float(np.max(dev_classical))
    max_dev_dw = float(np.max(dev_dw))
    dw_improvement_pct = (
        float(100.0 * (1.0 - max_dev_dw / max_dev_classical))
        if max_dev_classical > 0
        else 0.0
    )

    def display_time(value, horizon):
        return f"{value:.6g}" if value is not None else f">{horizon:.6g}"

    print(f"[classical] matched phi_j(0)   = {np.round(matched_phis0, 5)}")
    print(f"[classical] oscillation amplitude = {oscillation_amplitude:.6g}")
    print(f"[classical] threshold              = {threshold:.6g}")
    print(f"[classical] Delta phi              = {delta_phi:.6g}")
    print(
        f"[classical] force renormalization = "
        f"{force_renormalization_pct:.4f}%"
    )
    print(
        f"[classical] initial representation error = "
        f"{initial_representation_error:.4e}"
    )
    print(f"[classical] max dev vs classical = {max_dev_classical:.4e}")
    print(f"[classical] max dev vs DW        = {max_dev_dw:.4e}")
    print(
        f"[classical] DW discrepancy improvement = "
        f"{dw_improvement_pct:+.3f}%"
    )
    print(
        f"[classical] t_E (classical) = "
        f"{display_time(t_ehrenfest_classical, times[-1])}"
    )
    print(
        f"[classical] t_E (DW)        = "
        f"{display_time(t_ehrenfest_dw, times[-1])}"
    )
    print(
        f"[classical] tau_E (classical) = "
        f"{display_time(tau_ehrenfest_classical, tau_actual)}"
    )
    print(
        f"[classical] tau_E (DW)        = "
        f"{display_time(tau_ehrenfest_dw, tau_actual)}"
    )
    print(
        f"[classical] max top-level leakage = "
        f"{quantum['leak'].max():.3e}",
        flush=True,
    )
    if cutoff_convergence is not None:
        print(
            f"[classical] cutoff convergence max_t,j "
            f"|phi({args.n_cut})-phi({cutoff_compared})| = "
            f"{cutoff_convergence:.3e}",
            flush=True,
        )
    if time_step_convergence is not None:
        print(
            f"[classical] time convergence max_t,j "
            f"|phi(dt)-phi(dt/2)| = {time_step_convergence:.3e}",
            flush=True,
        )
    if quantum["leak"].max() > args.leakage_tol:
        print(
            f"[classical] WARNING top-level population exceeds "
            f"{args.leakage_tol:g}; "
            "increase --n-cut",
            flush=True,
        )

    ctx["params"].update(
        max_dev_classical=max_dev_classical,
        max_dev_debye_waller=max_dev_dw,
        delta_phi=delta_phi,
        force_renormalization_pct=force_renormalization_pct,
        debye_waller_improvement_pct=dw_improvement_pct,
        initial_representation_error=initial_representation_error,
        cutoff_convergence=cutoff_convergence,
        cutoff_convergence_normalized=None if cutoff_convergence is None else cutoff_convergence/oscillation_amplitude,
        cutoff_compared=cutoff_compared,
        time_step_convergence=time_step_convergence,
        time_step_convergence_normalized=None if time_step_convergence is None else time_step_convergence/oscillation_amplitude,
        ehrenfest_fraction=float(args.ehrenfest_tol),
        ehrenfest_threshold=threshold,
        ehrenfest_crossed_classical=t_ehrenfest_classical is not None,
        ehrenfest_crossed_dw=t_ehrenfest_dw is not None,
        t_ehrenfest_classical=t_ehrenfest_classical,
        t_ehrenfest_dw=t_ehrenfest_dw,
        t_ehrenfest_bracket_classical=bracket_classical,
        t_ehrenfest_bracket_dw=bracket_dw,
        tau_ehrenfest_bracket_classical=None if bracket_classical is None else [ctx["Omega_time"]*t for t in bracket_classical],
        tau_ehrenfest_bracket_dw=None if bracket_dw is None else [ctx["Omega_time"]*t for t in bracket_dw],
        tau_ehrenfest_classical=tau_ehrenfest_classical,
        tau_ehrenfest_dw=tau_ehrenfest_dw,
        tau_max_requested=tau_requested,
        tau_max_actual=tau_actual,
        max_leakage=float(quantum["leak"].max()),
        leakage_tolerance=float(args.leakage_tol),
        oscillation_amplitude=oscillation_amplitude,
        zero_amplitude_scale=args.zero_amplitude_scale,
        deviation_norm="max_over_time_and_sites",
        boundary_condition="periodic" if args.pbc else "open",
    )

    sg.save_run(
        args.outdir,
        ctx["tag"],
        ctx["params"],
        write_csv=args.csv,
        arrays=dict(
            t=times,
            tau=ctx["Omega_time"] * times,
            **{key: value for key, value in quantum.items() if key not in ("t", "phi", "pi", "n", "leak")},
            phi_q=quantum["phi"],
            pi_q=quantum["pi"],
            n_q=quantum["n"],
            leak=quantum["leak"],
            phi_cl=classical_phi,
            pi_cl=classical_pi,
            phi_dw=dw_phi,
            pi_dw=dw_pi,
            dev_cl=dev_classical,
            dev_dw=dev_dw,
            phis0_requested=phis0,
            pis0_requested=pis0,
            phis0_matched=matched_phis0,
            pis0_matched=matched_pis0,
            site_amplitudes=site_amplitudes,
            cutoff_convergence=np.array(
                [np.nan if cutoff_convergence is None else cutoff_convergence]
            ),
            time_step_convergence=np.array(
                [
                    np.nan
                    if time_step_convergence is None
                    else time_step_convergence
                ]
            ),
        ),
    )


if __name__ == "__main__":
    main()
