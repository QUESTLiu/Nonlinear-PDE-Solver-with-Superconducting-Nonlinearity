#!/usr/bin/env python3
"""
run_polydeg.py -- Section 3.5, native cosine versus polynomial synthesis.

Sweeps polynomial DEGREE in two families and reports the infidelity
against native-cosine evolution:

  taylor     -- Taylor truncation of E_J(1-cos phi) about phi = 0
  chebyshev  -- degree-d Chebyshev interpolant on
                [-phi_max, phi_max]

Comparing only low Taylor orders overstates the case for the native
interaction.  The Chebyshev interpolant is a substantially stronger baseline.
It is near-minimax for this analytic function, but the code does not claim to
solve the exact Remez minimax problem.

Two choices of phi_max are reported:
  spectral  -- max |eigenvalue of the truncated phi|; conservative, since
               the operator acts on the whole truncated space
  accessed  -- max_t |<phi>| + 3 Delta_phi from a native reference run;
               optimistic, valid only if the state stays put

The deliverable is d*(phi_0): the degree needed to reach a target
infidelity, as a function of amplitude.  Feed d* into run_snapd.py to
convert it into a non-Gaussian gate count.

Example
-------
python run_polydeg.py --E-J 1.0 --kappa 0.0 --n-site 1 --n-cut 64 \
    --phi0 3.0 --T 1 --dt 1e-3 --degrees 4,6,8,10,12,14,16,20,24
"""

import numpy as np

import sglib as sg
import common


def main():
    p = common.base_parser(__doc__)
    p.add_argument("--degrees", default="4,6,8,10,12,14,16",
                   help="comma-separated even polynomial degrees")
    p.add_argument("--phi-max-mode", default="both",
                   choices=["spectral", "accessed", "both"])
    p.add_argument("--target-infid", type=float, default=1e-3,
                   help="threshold defining the required degree d*")
    p.add_argument(
        "--leakage-tol",
        type=float,
        default=1e-5,
        help="maximum top-level population allowed when defining d*",
    )
    args = p.parse_args()

    degrees = [int(d) for d in args.degrees.split(",")]
    ctx = common.setup(args, "polydeg")
    basis = ctx["basis"]
    if args.dry_run:
        return

    psi0, phis0, pis0, offsets, info = sg.make_initial_condition(
        args.init, args.n_site, args.n_cut, ctx["w"], args.E_J, args.kappa,
        args.phi0, args.pi0, args.site, args.use_offsets)
    ctx["params"].update(info)
    n_steps, stride = common.steps_and_stride(args.T, args.dt, args.n_obs)

    def evolve(onsite, want_traj=False):
        prop = sg.SplitOperator(basis, args.n_site, args.kappa, onsite, args.pbc,
                                offsets=offsets)
        store = common.new_store()
        obs = common.make_observer(ctx["phi_op"], ctx["pi_op"], ctx["num_op"],
                                   store, offsets)
        # keep the wavefunction at every stored time for the fidelity series
        frames = []

        def obs2(t, psi):
            obs(t, psi)
            frames.append(psi.copy())
        prop.evolve(psi0, args.dt, n_steps, stride, obs2)
        return common.stack(store), frames

    native_on = sg.onsite_vals_with_offsets("native", basis.lam_phi,
                                            args.E_J, args.n_site, offsets)
    ref, ref_frames = evolve(native_on)
    print(f"[polydeg] reference done; max leakage {ref['leak'].max():.3e}",
          flush=True)

    # Build the accessed interval from the actual evolving variance rather
    # than assuming that the nonlinear state retains its initial coherent
    # width.  There is no reason to extend the interval beyond the spectrum
    # of the finite-cutoff phi operator: the projected Hamiltonian never
    # evaluates the polynomial there.
    phi_sq_op = basis.phi @ basis.phi
    phi_max_accessed_raw = 0.0
    for state, mean_physical in zip(ref_frames, ref["phi"]):
        for site in range(args.n_site):
            mean_local = float(mean_physical[site] - offsets[site])
            second_local = float(
                sg.expect_mode(state, phi_sq_op, site).real
            )
            variance = max(second_local - mean_local**2, 0.0)
            phi_max_accessed_raw = max(
                phi_max_accessed_raw,
                abs(float(mean_physical[site])) + 3.0 * np.sqrt(variance),
            )
    physical_spectrum = offsets[:, None] + basis.lam_phi[None, :]
    phi_max_spectral = float(np.max(np.abs(physical_spectrum)))
    phi_max_accessed = float(
        min(phi_max_accessed_raw, phi_max_spectral)
    )
    clamp_note = (
        f" (raw 3-sigma bound {phi_max_accessed_raw:.3f}, "
        "clamped to finite spectrum)"
        if phi_max_accessed_raw > phi_max_spectral
        else ""
    )
    print(f"[polydeg] phi_max spectral={phi_max_spectral:.3f}  "
          f"accessed={phi_max_accessed:.3f}{clamp_note}", flush=True)

    families = [("taylor", None)]
    if args.phi_max_mode in ("spectral", "both"):
        families.append(("chebyshev_spectral", phi_max_spectral))
    if args.phi_max_mode in ("accessed", "both"):
        families.append(("chebyshev_accessed", phi_max_accessed))

    results = {}
    for fam, phi_max in families:
        kind = "taylor" if fam == "taylor" else "chebyshev"
        max_infid, fin_infid, max_leak, potential_error = [], [], [], []
        for d in degrees:
            on = sg.onsite_vals_with_offsets(kind, basis.lam_phi, args.E_J,
                                             args.n_site, offsets,
                                             degree=d,
                                             phi_max=phi_max or 1.0)
            approximate, frames = evolve(on)
            series = np.array([sg.infidelity(a, b)
                               for a, b in zip(ref_frames, frames)])
            max_infid.append(series.max())
            fin_infid.append(series[-1])
            max_leak.append(float(approximate["leak"].max()))
            potential_error.append(float(np.max(np.abs(on - native_on))))
            print(f"[polydeg] {fam:22s} d={d:2d}  "
                  f"max_infid={series.max():.4e}  "
                  f"potential_err={potential_error[-1]:.3e}  "
                  f"leak={max_leak[-1]:.3e}", flush=True)
        results[fam + "_max"] = np.array(max_infid)
        results[fam + "_final"] = np.array(fin_infid)
        results[fam + "_leakage"] = np.array(max_leak)
        results[fam + "_potential_error"] = np.array(potential_error)

        below = [
            d
            for d, value, leakage in zip(degrees, max_infid, max_leak)
            if value < args.target_infid and leakage <= args.leakage_tol
        ]
        d_star = below[0] if below else -1
        ctx["params"][f"d_star_{fam}"] = int(d_star)
        ctx["params"][f"max_leakage_{fam}"] = float(max(max_leak))
        if d_star >= 0:
            d_star_index = degrees.index(d_star)
            ctx["params"][f"leakage_at_d_star_{fam}"] = float(
                max_leak[d_star_index]
            )
        else:
            ctx["params"][f"leakage_at_d_star_{fam}"] = None
        print(f"[polydeg] {fam}: d* (infid < {args.target_infid:g}) = {d_star}",
              flush=True)
        if max(max_leak) > args.leakage_tol:
            print(
                f"[polydeg] NOTE {fam} contains degrees with top-level "
                f"population above {args.leakage_tol:g}; those individual "
                "points are marked as cutoff-limited and excluded from d*",
                flush=True,
            )

    ctx["params"].update(phi_max_spectral=phi_max_spectral,
                         phi_max_accessed=phi_max_accessed,
                         phi_max_accessed_raw=phi_max_accessed_raw,
                         degrees=degrees,
                         leakage_tolerance=float(args.leakage_tol),
                         approximation="Chebyshev interpolant (not exact minimax)",
                         max_leakage_reference=float(ref["leak"].max()))
    sg.save_run(args.outdir, ctx["tag"], ctx["params"], write_csv=args.csv,
                arrays=dict(
        degrees=np.array(degrees), t=ref["t"], phi_ref=ref["phi"],
        leak_ref=ref["leak"], lam_phi=basis.lam_phi, **results))


if __name__ == "__main__":
    main()
