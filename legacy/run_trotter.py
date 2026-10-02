#!/usr/bin/env python3
"""
run_trotter.py -- product-formula scaling check.

Halves dt and confirms the factor-four reduction expected of a
second-order Strang splitting.  The reference is computed with Krylov
propagation, NOT with the same splitting -- an anomalously small Trotter
error is the classic symptom of a reference that shares the splitting it
is supposed to validate.

Reports both fitted exponents: vector error scales as dt^2, while
infidelity (the squared overlap error) scales as dt^4.

Note that this cost is a classical-simulation artifact and does not
appear on hardware: the charging and Josephson terms are simultaneously
present in the circuit and evolve together at zero gate cost.  The check
exists to validate the numerics, not to price the device.

Krylov needs ~m state vectors, so keep this study small: N_site <= 3.

Example
-------
python run_trotter.py --E-J 1.0 --kappa 1.0 --n-site 2 --n-cut 24 \
    --phi0 1.0 --T 1.0 --dts 0.05,0.025,0.0125,0.00625
"""

import numpy as np

import sglib as sg
import common


def main():
    p = common.base_parser(__doc__)
    p.add_argument("--dts", default="0.08,0.04,0.02,0.01,0.005",
                   help="comma-separated Strang steps to test")
    p.add_argument("--ref-dt", type=float, default=-1.0,
                   help="Krylov reference step; -1 uses min(dts)/8")
    p.add_argument("--krylov-m", type=int, default=40)
    args = p.parse_args()

    dts = [float(x) for x in args.dts.split(",")]
    ctx = common.setup(args, "trotter")
    basis = ctx["basis"]

    mem = sg.estimate_memory_gb(args.n_cut, args.n_site,
                                n_vectors=args.krylov_m + 3)
    print(f"[trotter] Krylov memory ~{mem:.2f} GB at m={args.krylov_m}",
          flush=True)
    if mem > 100:
        raise MemoryError("Krylov reference too large; reduce n_cut/n_site/m")
    if args.dry_run:
        return

    psi0, _, _, offsets, info = sg.make_initial_condition(
        args.init, args.n_site, args.n_cut, ctx["w"], args.E_J, args.kappa,
        args.phi0, args.pi0, args.site, args.use_offsets)
    ctx["params"].update(info)
    onsite = sg.onsite_vals_with_offsets("native", basis.lam_phi,
                                         args.E_J, args.n_site, offsets)

    ref_dt = min(dts) / 8.0 if args.ref_dt <= 0 else args.ref_dt
    H = sg.HamiltonianApply(basis, args.n_site, args.kappa, onsite, args.pbc,
                                offsets=offsets)
    n_ref = max(1, int(round(args.T / ref_dt)))
    ref_dt = args.T / n_ref
    print(f"[trotter] Krylov reference: dt={ref_dt:g}, {n_ref} steps",
          flush=True)
    psi_ref, _ = sg.krylov_evolve(H, psi0, ref_dt, n_ref, args.krylov_m)

    # self-consistency: halve the reference step and check it has converged
    psi_ref2, _ = sg.krylov_evolve(H, psi0, ref_dt / 2, 2 * n_ref,
                                   args.krylov_m)
    ref_selfconv = sg.infidelity(psi_ref, psi_ref2)
    print(f"[trotter] reference self-convergence = {ref_selfconv:.3e}",
          flush=True)
    if ref_selfconv > 1e-12:
        print("[trotter] WARNING reference not converged; reduce --ref-dt",
              flush=True)

    # Two error metrics, because they scale differently and confusing them
    # is the easiest way to misread this plot:
    #   vector error  || psi_ref - psi_dt ||  ~ dt^2   (second-order Strang)
    #   infidelity    1 - |<psi_ref|psi_dt>|^2 ~ dt^4  (it is the square)
    # We fit the vector error, and report the infidelity exponent too.
    dts_used, infids, verrs = [], [], []
    for dt_req in dts:
        n_steps = max(1, int(round(args.T / dt_req)))
        dt = args.T / n_steps          # snap so n_steps*dt == T exactly
        prop = sg.SplitOperator(basis, args.n_site, args.kappa, onsite,
                                args.pbc, offsets=offsets)
        psi, _ = prop.evolve(psi0, dt, n_steps)
        ov = np.vdot(psi_ref2.reshape(-1), psi.reshape(-1))
        # remove the global phase before taking the vector difference
        psi_al = psi * np.exp(-1j * np.angle(ov))
        verr = float(np.linalg.norm((psi_ref2 - psi_al).reshape(-1)))
        val = float(np.clip(1.0 - abs(ov) ** 2, 0.0, 1.0))
        dts_used.append(dt)
        infids.append(val)
        verrs.append(verr)
        print(f"[trotter] dt={dt:<10.6g} steps={n_steps:<7d} "
              f"vec_err={verr:.6e}  infid={val:.6e}", flush=True)

    dts_used = np.array(dts_used)
    infids = np.array(infids)
    verrs = np.array(verrs)

    def fit(y, floor):
        good = y > floor
        if good.sum() < 2:
            return float("nan")
        return float(np.polyfit(np.log(dts_used[good]), np.log(y[good]), 1)[0])

    floor_v = max(10 * np.sqrt(abs(ref_selfconv)), 1e-11)
    p_vec = fit(verrs, floor_v)
    p_inf = fit(infids, max(10 * abs(ref_selfconv), 1e-14))
    print(f"[trotter] fitted exponent, vector error = {p_vec:.4f} (expect ~2)")
    print(f"[trotter] fitted exponent, infidelity   = {p_inf:.4f} (expect ~4)",
          flush=True)
    if np.isnan(p_inf):
        print("[trotter] infidelity exponent is unresolved because fewer than "
              "two values are above the numerical-noise floor; use larger "
              "test steps for the production scaling fit.", flush=True)
    if not np.isnan(p_vec) and abs(p_vec - 2.0) > 0.3:
        print("[trotter] WARNING vector-error exponent is not 2. Check that "
              "the largest dt is not outside the asymptotic regime, and that "
              "the reference does not share the splitting.", flush=True)

    ctx["params"].update(exponent_vector=p_vec, exponent_infidelity=p_inf,
                         ref_dt=ref_dt, ref_selfconv=float(ref_selfconv),
                         dts_requested=dts, dts_used=dts_used.tolist())
    sg.save_run(args.outdir, ctx["tag"], ctx["params"], write_csv=args.csv,
                arrays=dict(
        dts=dts_used, infidelities=infids, vector_errors=verrs,
        exponent_vector=np.array([p_vec]),
        exponent_infidelity=np.array([p_inf]),
        ref_selfconv=np.array([ref_selfconv])))


if __name__ == "__main__":
    main()
