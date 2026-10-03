#!/usr/bin/env python3
"""
run_multisite.py -- Section 3.6, multi-site dynamics.

Runs one (N_site, n_cut) point.  Submit as an LSF job array over N_site
to trace the frontier: full state-vector simulation costs
n_cut^N_site complex amplitudes and dies around N_site = 7-8 at
n_cut = 12, whereas a Josephson array scales linearly.

Also runs the classical lattice comparison and a cutoff-convergence
diagnostic, so each point in the scan carries its own validity evidence.

A kink of width ell = sqrt(kappa/E_J) needs N_site >~ 3 ell to be
resolved.  At accessible sizes you are in the strongly discrete regime
where the kink is Peierls-Nabarro pinned -- physically meaningful for
Josephson arrays, but NOT the continuum soliton.  The script reports
sites_per_width so this can be stated honestly.

Example
-------
python run_multisite.py --E-J 1.0 --kappa 4.0 --n-site 6 --n-cut 12 \
    --init kink --T 10 --dt 5e-3
"""

from pathlib import Path
import sys

import numpy as np

if not __package__:
    # The shared SG implementation lives outside this legacy script directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sine_gordon_code import common, sglib as sg


def main():
    p = common.base_parser(__doc__)
    p.add_argument(
        "--convergence-check",
        action="store_true",
        help="rerun at n_cut-2 and report the difference",
    )
    args = p.parse_args()

    ctx = common.setup(args, "multisite")
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
    if "kink_width_sites" in info:
        print(
            f"[multisite] kink width {info['kink_width_sites']:.2f} sites; "
            f"lattice holds {info['sites_per_width']:.2f} widths "
            f"({'resolved' if info['sites_per_width'] > 3 else 'DISCRETE / pinned'})",
            flush=True,
        )
    print(f"[multisite] phi_j(0) = {np.round(phis0, 3)}", flush=True)

    onsite = sg.onsite_vals_with_offsets(
        "native", basis.lam_phi, args.E_J, args.n_site, offsets
    )
    n_steps, stride = common.steps_and_stride(args.T, args.dt, args.n_obs)

    store = common.new_store()
    obs = common.make_observer(
        ctx["phi_op"], ctx["pi_op"], ctx["num_op"], store, offsets
    )
    prop = sg.SplitOperator(
        basis, args.n_site, args.kappa, onsite, args.pbc, offsets=offsets
    )
    psi_final, _ = prop.evolve(psi0, args.dt, n_steps, stride, obs)
    q = common.stack(store)

    matched_phis0 = q["phi"][0].copy()
    matched_pis0 = q["pi"][0].copy()
    cl_phi, _ = sg.classical_sine_gordon(
        matched_phis0, matched_pis0, args.E_J, args.kappa, q["t"], args.pbc
    )
    dw_phi, _ = sg.classical_sine_gordon(
        matched_phis0,
        matched_pis0,
        args.E_J,
        args.kappa,
        q["t"],
        args.pbc,
        E_J_eff=ctx["semi"]["E_J_eff"],
    )
    dev_cl = np.max(np.abs(q["phi"] - cl_phi), axis=1)
    dev_dw = np.max(np.abs(q["phi"] - dw_phi), axis=1)

    print(f"[multisite] max leakage                 = {q['leak'].max():.3e}")
    print(f"[multisite] max <n_j> (displaced basis) = {q['n'].max():.3e}")
    print(f"[multisite] max dev vs class.  = {dev_cl.max():.4e}")
    print(f"[multisite] max dev vs DW      = {dev_dw.max():.4e}", flush=True)
    if q["leak"].max() > 1e-6:
        print(
            "[multisite] WARNING top-level population exceeds 1e-6; increase --n-cut",
            flush=True,
        )

    conv = float("nan")
    if args.convergence_check and args.n_cut > 6:
        nc2 = args.n_cut - 2
        w = ctx["w"]
        b2 = sg.build_mode_basis(nc2, w)
        psi0b, _, _, off2, _ = sg.make_initial_condition(
            args.init,
            args.n_site,
            nc2,
            w,
            args.E_J,
            args.kappa,
            args.phi0,
            args.pi0,
            args.site,
            args.use_offsets,
        )
        on2 = sg.onsite_vals_with_offsets(
            "native", b2.lam_phi, args.E_J, args.n_site, off2
        )
        prop2 = sg.SplitOperator(
            b2, args.n_site, args.kappa, on2, args.pbc, offsets=off2
        )
        # mode_operators returns (phi, pi, n).  The cutoff diagnostic must
        # compare phi at both cutoffs; selecting the second return value would
        # silently compare the high-cutoff field against low-cutoff momentum.
        phi_op2, _, _ = sg.mode_operators(nc2, w)
        store2 = {"t": [], "phi": []}

        def obs2(t, psi):
            store2["t"].append(float(t))
            store2["phi"].append(
                off2
                + np.array(
                    [sg.expect_mode(psi, phi_op2, j).real for j in range(args.n_site)]
                )
            )

        prop2.evolve(psi0b, args.dt, n_steps, stride, obs2)
        phi_b = np.asarray(store2["phi"])
        if phi_b.shape != q["phi"].shape:
            raise RuntimeError("cutoff trajectories were stored on different grids")
        # Compare observables, not overlaps: the Hilbert spaces differ.
        conv = float(np.max(np.abs(q["phi"] - phi_b)))
        print(
            f"[multisite] cutoff convergence max_t,j |phi(n_cut)-phi(n_cut-2)| "
            f"= {conv:.3e}",
            flush=True,
        )

    ctx["params"].update(
        max_leakage=float(q["leak"].max()),
        max_n=float(q["n"].max()),
        max_dev_classical=float(dev_cl.max()),
        max_dev_debye_waller=float(dev_dw.max()),
        cutoff_convergence=None if np.isnan(conv) else conv,
        statevector_amplitudes=int(args.n_cut**args.n_site),
        statevector_gib=float(args.n_cut**args.n_site * 16 / 1024**3),
        dense_working_memory_gib=float(
            sg.estimate_split_memory_gb(args.n_cut, args.n_site)
        ),
        boundary_condition="periodic" if args.pbc else "open",
        initial_representation_error=float(
            max(
                np.max(np.abs(matched_phis0 - phis0)),
                np.max(np.abs(matched_pis0 - pis0)),
            )
        ),
    )

    sg.save_run(
        args.outdir,
        ctx["tag"],
        ctx["params"],
        write_csv=args.csv,
        arrays=dict(
            t=q["t"],
            phi_q=q["phi"],
            pi_q=q["pi"],
            n_q=q["n"],
            leak=q["leak"],
            phi_cl=cl_phi,
            phi_dw=dw_phi,
            dev_cl=dev_cl,
            dev_dw=dev_dw,
            phis0_requested=phis0,
            pis0_requested=pis0,
            phis0_matched=matched_phis0,
            pis0_matched=matched_pis0,
        ),
    )


if __name__ == "__main__":
    main()
