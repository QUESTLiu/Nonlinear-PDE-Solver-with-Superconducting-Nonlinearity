"""Bounded grounded-target scan for the two-site double sine-Gordon benchmark.

Run from the repository root with the documented cvdv interpreter:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python double_sine_gordon_code/dsg_grounding_check.py

The scan adds g_sh/2 * (phi_1^2 + phi_2^2) to both native and target
Hamiltonians at g_sh=0.1, without changing the preparation, basis frequency,
horizon, block orders, or acceptance rule. It evaluates r=4,...,9 with s=1
and local cutoffs 128/160. Native blocks use the grounded local Hamiltonian.
Gaussian blocks use the canonical shear decomposition of Appendix J, evaluated
as the merged finite-matrix circuit. Counts come from dsg_step_scan.py.

The sparse-exponential reference is independently compared with a matrix-free
DOP853 solution at cutoff 160 on every sampled time. Their discrepancy is an
empirical solver margin, not a rigorous infinite-dimensional error bound.
The 1200-second limit bounds this one comparison. Existing records are never
overwritten. No trajectory arrays are saved in the compact output record.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye, kron

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common_code.metrics import projective_distance
from common_code.run_pilot import _sg_initial
from double_sine_gordon_code import dsg_step_scan as scan
from double_sine_gordon_code.gaussian_sequence import merged_gaussian_readouts
from sine_gordon_code import native_correction as nc, sglib as sg

G_SH = 0.1
STEPS = tuple(range(4, 10))


def grounded_problem(n, g_sh=G_SH):
    """Add the same spectral quadratic operator to native and target models."""
    problem = scan.build_two_site_problem(n, scan.OMEGA_B, scan.E_J, scan.KAPPA, scan.LAM2)
    phi = problem["basis"].phi
    local = 0.5 * g_sh * (phi @ phi)
    identity = eye(n, dtype=complex, format="csr")
    shunt = kron(csr_matrix(local), identity) + kron(identity, csr_matrix(local))
    problem["H_native"] = (problem["H_native"] + shunt).tocsr()
    problem["H_target"] = (problem["H_target"] + shunt).tocsr()
    problem["native_local_with_bond"] = problem["native_local_with_bond"] + local
    return problem


def shear_matrices(tau, g_sh=G_SH):
    """Appendix J: A=Omega*tan(tau*Omega/2), B=sin(tau*Omega)/Omega.

    K=kappa*[[1,-1],[-1,1]]+g_sh*I has squared normal-mode frequencies
    g_sh and 2*kappa+g_sh. Multiplying the position/momentum/position
    symplectic matrices gives exp(tau*[[0,I],[-K,0]]). The continuous
    zero-mode limits are A=0 and B=tau. The tested steps precede the
    first tangent singularity. These identities concern canonical operators,
    not a group law for the finite Fock matrices used in propagation.
    """
    stiffness = scan.KAPPA * np.array([[1., -1.], [-1., 1.]]) + g_sh * np.eye(2)
    eig, modes = np.linalg.eigh(stiffness)
    frequencies = np.sqrt(eig)
    if np.max(frequencies * abs(tau)) >= np.pi:
        raise ValueError("step is outside the continuous Gaussian shear branch")
    a = frequencies * np.tan(tau * frequencies / 2)
    b = tau * np.sinc(tau * frequencies / np.pi)
    return (modes * a) @ modes.T, (modes * b) @ modes.T


def gaussian_generators(basis, tau, g_sh=G_SH):
    a, b = shear_matrices(tau, g_sh)

    def quadratic(spectrum, matrix):
        x, y = np.meshgrid(spectrum, spectrum, indexing="ij")
        return 0.5 * matrix[0, 0] * x*x + matrix[0, 1] * x*y + 0.5 * matrix[1, 1] * y*y

    return quadratic(basis.lam_phi, a), quadratic(np.linalg.eigvalsh(basis.pi), b)


def trajectory(problem, r, route, order):
    basis, n = problem["basis"], problem["basis"].n_cut
    state = np.zeros((n, n, 2), complex)
    state[:, :, 0] = _sg_initial(basis, 1., 0.)
    peak = nc.populations(state)["boundary_band_population"]

    def observe(value):
        nonlocal peak
        peak = max(peak, nc.populations(value)["boundary_band_population"])

    h = scan.T / r
    if route == "hybrid":
        step = nc.make_step(problem, scan.E_J, scan.KAPPA, scan.LAM2, h,
                            "H_trig", order, 1, observer=observe)

        def readouts():
            current = state
            yield current
            for _ in range(r):
                current = step(current)
                yield current

        values = readouts()
    else:
        _, u_pi = np.linalg.eigh(basis.pi)
        blocks = {m: nc.trig_block(basis.phi, m, h*c, o, 1)
                  for m, c, o in zip((1, 2), (scan.E_J, scan.LAM2), scan.ORDER_CODES[order])}

        def diagonal(value, u, generator):
            rotated = sg._apply_axis(u.conj().T, sg._apply_axis(u.conj().T, value, 0), 1)
            return sg._apply_axis(u, sg._apply_axis(u, np.exp(-1j*generator)[:, :, None] * rotated, 0), 1)

        def potential(value, observer):
            for site in (0, 1):
                for m in (1, 2):
                    value = nc.apply_joint_local(blocks[m], value, site)
                    observer(value)
            return value

        values = merged_gaussian_readouts(state, r, gaussian_generators(basis, h/2),
                                          gaussian_generators(basis, h),
                                          lambda v, q: diagonal(v, basis.U_phi, q),
                                          lambda v, p: diagonal(v, u_pi, p), potential, observe)
    states = np.array(list(values))
    return states, peak


def observables(states, basis, joint=False):
    shape = (basis.n_cut, basis.n_cut, 2) if joint else (basis.n_cut, basis.n_cut)
    rows = [nc.two_site_observables(value.reshape(shape), basis, joint=joint) for value in states]
    return dict(C12=np.array([complex(row["C12_real"], row["C12_imag"]) for row in rows]),
                S2=np.array([row["S2"] for row in rows]))


def solver_check(problem, references):
    """Compare distinct propagation algorithms and Hamiltonian representations."""
    basis, n = problem["basis"], problem["basis"].n_cut
    local = problem["native_local_with_bond"] + problem["correction_local"]

    def rhs(_, flat):
        state = flat.reshape(n, n)
        action = local @ state + state @ local.T - scan.KAPPA * basis.phi @ state @ basis.phi.T
        return -1j * action.ravel()

    times = sorted({float(t) for r in STEPS for t in np.linspace(0, scan.T, r+1)})
    solved = solve_ivp(rhs, (0, scan.T), _sg_initial(basis, 1., 0.).ravel(),
                       method="DOP853", t_eval=times, rtol=1e-11, atol=1e-13)
    if not solved.success:
        raise RuntimeError(solved.message)
    norm_drift = float(np.max(abs(np.linalg.norm(solved.y, axis=0)-1)))
    normalized = solved.y.T / np.linalg.norm(solved.y.T, axis=1)[:, None]
    obs = observables(normalized, basis)
    lookup = {t: i for i, t in enumerate(times)}
    margins = dict(state=0., C12=0., S2=0.)
    for r in STEPS:
        indices = [lookup[float(t)] for t in np.linspace(0, scan.T, r+1)]
        reference = references[n, r]
        margins["state"] = max(margins["state"], max(projective_distance(a, b)
                               for a, b in zip(reference["states"], normalized[indices])))
        for key in ("C12", "S2"):
            margins[key] = max(margins[key], float(np.max(abs(reference[key]-obs[key][indices]))))
    return dict(margins=margins, method="DOP853", cutoff=n, rtol=1e-11, atol=1e-13,
                sampled_times=times, function_evaluations=solved.nfev, norm_drift=norm_drift)


def evaluate(problems, references, solver, r, route, order):
    low, high = scan.CUTS
    outputs = {n: trajectory(problems[n], r, route, order) for n in scan.CUTS}
    states, peak = outputs[low]
    padded = np.zeros_like(outputs[high][0])
    padded[:, :low, :low, :] = states
    padding = max(projective_distance(a, b) for a, b in zip(padded, outputs[high][0]))
    target = np.zeros((r+1, high, high), complex)
    target[:, :low, :low] = references[low, r]["states"].reshape(r+1, low, low)
    ref_margin = max(projective_distance(a, b) for a, b in zip(target, references[high, r]["states"]))
    raw = [nc.reduced_metrics(value, ref)["bures_distance"] for value, ref in zip(states, references[low, r]["states"])]
    state_error = max(raw) + ref_margin + padding + solver["state"]
    margins = dict(reference=float(ref_margin), padding=float(padding), solver=solver["state"])
    errors = dict(state=float(state_error))
    obs = {n: observables(outputs[n][0], problems[n]["basis"], joint=True) for n in scan.CUTS}
    for key in ("C12", "S2"):
        margins[key] = float(np.max(abs(references[low, r][key]-references[high, r][key]))
                             + solver[key] + np.max(abs(obs[low][key]-obs[high][key])))
        errors[key] = float(np.max(abs(obs[low][key]-references[low, r][key])) + margins[key])
    boundary = float(max(peak, outputs[high][1]))
    criteria = dict(state=bool(state_error <= scan.EPS and ref_margin <= scan.EPS/10 and solver["state"] <= scan.EPS/10),
                    C12=bool(errors["C12"] <= .01 and margins["C12"] <= .001),
                    S2=bool(errors["S2"] <= .02 and margins["S2"] <= .005),
                    boundary=bool(boundary <= 1e-6))
    counts, waits = scan.primitive_counts(route, r, order, 1)
    return dict(route=route, r=r, order=order, block_orders=list(scan.ORDER_CODES[order]),
                primitive_counts=counts, total_primitives=sum(counts.values()), native_waits=waits,
                errors=errors, max_bures=float(max(raw)), margins=margins, boundary_population=boundary,
                criteria=criteria, all_criteria=all(criteria.values()))


def independent_checks():
    """Small first-principles Hamiltonian and symplectic oracles, plus the g=0 limit."""
    problem = grounded_problem(6)
    basis = problem["basis"]
    identity = np.eye(6)
    x, y = np.kron(basis.phi, identity), np.kron(identity, basis.phi)
    p, q = np.kron(basis.pi, identity), np.kron(identity, basis.pi)
    native = .5*(p@p+q@q) + .5*scan.KAPPA*(x-y)@(x-y) + .5*G_SH*(x@x+y@y)
    native = native + scan.E_J*(2*np.eye(36)-(expm(1j*x)+expm(-1j*x)+expm(1j*y)+expm(-1j*y))/2)
    target = native + scan.LAM2*(2*np.eye(36)-(expm(2j*x)+expm(-2j*x)+expm(2j*y)+expm(-2j*y))/2)
    h_error = float(np.linalg.norm(target-problem["H_target"].toarray(), 2))
    rng = np.random.default_rng(720)
    state = rng.normal(size=(6, 6, 2)) + 1j*rng.normal(size=(6, 6, 2))
    state /= np.linalg.norm(state)
    step = nc.make_step(problem, scan.E_J, scan.KAPPA, scan.LAM2, .19, "native")
    native_error = float(np.linalg.norm(step(state).reshape(36, 2)-expm(-.19j*native)@state.reshape(36, 2)))
    a, b = shear_matrices(.43)
    eye2, zero = np.eye(2), np.zeros((2, 2))
    position = np.block([[eye2, zero], [-a, eye2]])
    momentum = np.block([[eye2, b], [zero, eye2]])
    stiffness = scan.KAPPA*np.array([[1., -1.], [-1., 1.]])+G_SH*eye2
    symplectic_error = float(np.linalg.norm(position@momentum@position-expm(.43*np.block([[zero, eye2], [-stiffness, zero]]))))
    zero_shunt_error = float(max(np.max(abs(a-b)) for a, b in zip(
        gaussian_generators(basis, .43, 0.), scan.gaussian_generators(basis, .43))))
    result = dict(hamiltonian_error=h_error, native_action_error=native_error,
                  symplectic_error=symplectic_error, zero_shunt_generators_error=zero_shunt_error)
    if max(result.values()) > 1e-11:
        raise AssertionError(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO/"double_sine_gordon_data/dsg_grounding_check.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    paths = [Path(__file__), Path(scan.__file__), Path(nc.__file__), Path(sg.__file__),
             REPO/"double_sine_gordon_code/run_snapd_end_to_end.py", REPO/"double_sine_gordon_code/gaussian_sequence.py",
             REPO/"double_sine_gordon_code/trig_counts.py", REPO/"common_code/metrics.py", REPO/"common_code/run_pilot.py"]
    report = dict(status="running", parameters=dict(g_sh=G_SH, E_J=scan.E_J, kappa=scan.KAPPA, lambda2=scan.LAM2,
                  omega_b=scan.OMEGA_B, T=scan.T, preparation=[1., 0.], cutoffs=list(scan.CUTS), steps=list(STEPS), s=1),
                  environment=scan.environment(), command=sys.argv, wall_limit_seconds=1200,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, rows=[])

    def timeout(*_):
        raise TimeoutError("1200-second comparison limit reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1200)
    try:
        report["independent_checks"] = independent_checks()
        print("independent checks", report["independent_checks"], flush=True)
        problems, references = {}, {}
        for n in scan.CUTS:
            problem = problems[n] = grounded_problem(n)
            initial = _sg_initial(problem["basis"], 1., 0.)
            for r in STEPS:
                tick = time.perf_counter()
                states = scan.exact_trajectory(problem["H_target"], initial, scan.T, r)
                references[n, r] = dict(states=states, **observables(states, problem["basis"]))
                print(f"reference N={n} r={r}: {time.perf_counter()-tick:.2f}s", flush=True)
        report["solver_check"] = solver_check(problems[scan.CUTS[-1]], references)
        print("solver margins", report["solver_check"]["margins"], flush=True)
        # Run the original winning pair first to measure cost, then finish the
        # stated common scan without changing the candidate set after seeing it.
        candidates = [("hybrid", 6, 1), ("gauss_block", 7, 3)]
        candidates += [(route, r, order) for r in STEPS for route, orders in
                       (("hybrid", (1, 2)), ("gauss_block", (1, 2, 3))) for order in orders
                       if (route, r, order) not in candidates]
        for route, r, order in candidates:
            tick = time.perf_counter()
            row = evaluate(problems, references, report["solver_check"]["margins"], r, route, order)
            row["wall_seconds"] = time.perf_counter()-tick
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(row), flush=True)
        report["cheapest_passing"] = {}
        for route in ("hybrid", "gauss_block"):
            passing = [row for row in report["rows"] if row["route"] == route and row["all_criteria"]]
            report["cheapest_passing"][route] = min(passing, key=lambda row: (row["total_primitives"], row["errors"]["state"])) if passing else None
        report["status"] = "completed"
    except TimeoutError as exc:
        report.update(status="incomplete", reason=str(exc))
    except Exception as exc:
        report.update(status="failed", reason=str(exc))
        raise
    finally:
        signal.alarm(0)
        report["wall_seconds"] = time.perf_counter()-started
        args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
