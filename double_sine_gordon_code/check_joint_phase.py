"""Bounded joint-phase comparison for the two-site lambda2=0.3 benchmark.

Run from the repository root in the cvdv environment, with one BLAS thread:
    python double_sine_gordon_code/check_joint_phase.py

The search includes r=6,7,8, degrees K=1,...,10, and fixed or alternating
completion signs for both routes. Actual
Rz/Ry/CD_Z pulse counts identify every candidate that could lower the existing
route minimum (64 hybrid, 209 programmable). Only these competitive candidates
need full propagation. At r=7, one additional circuit per route is selected by
a common single-block error target of 1e-3 to check the complete construction.
There is no fitting, postselection, reset, or hardware-noise claim.

The command has a 600-second wall limit. It saves compact evidence after each
completed circuit and never modifies the existing benchmark records. Reference
trajectories are rebuilt at 128/160 for the required grids with the repository
propagator. The full comparison uses the same empirical acceptance margins.
"""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections import Counter

import numpy as np
from scipy.linalg import expm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common_code.metrics import projective_distance
from common_code.run_pilot import _sg_initial
from double_sine_gordon_code import dsg_step_scan as scan
from double_sine_gordon_code.gaussian_sequence import merged_gaussian_readouts
from double_sine_gordon_code.joint_phase import (
    compile_phase,
    merge_rotations,
    pulse_response,
)
from sine_gordon_code import native_correction as nc, sglib as sg

OUTPUT = REPO / "double_sine_gordon_data/joint_phase_check.json"


def pulse_schedule(compiled, r, pattern="fixed"):
    """Conjugate both site words by Z in every other step when requested.

    Rz(pi) W Rz(-pi)=Z W Z flips both complementary entries. Boundary
    conjugations are absorbed by the same full-SU(2) merging as other gates.
    See arXiv:2609.40078v1, Sec. III.4, following Eq. (29).
    """
    pulses = []
    for step in range(r):
        for _ in range(2):
            flip = pattern == "alternating" and step % 2
            if flip:
                pulses.append(("Rz", -np.pi))
            pulses.extend(compiled["pulses"])
            if flip:
                pulses.append(("Rz", np.pi))
    return pulses


def counts(compiled, route, r, pattern="fixed"):
    """Fuse full SU(2) rotations across word boundaries and system intervals."""
    pulses, _ = merge_rotations(pulse_schedule(compiled, r, pattern))
    result = Counter(gate for gate, _ in pulses)
    result["Gaussian"] = 0 if route == "hybrid" else 6 * r + 9
    return dict(result)


def run(problem, reference, r, route, compiled, pattern="fixed"):
    basis, n = problem["basis"], problem["basis"].phi.shape[0]
    joint = np.zeros((n, n, 2), complex)
    joint[:, :, 0] = _sg_initial(basis, 1.0, 0.0)
    spectrum = basis.lam_phi * (2 if route == "hybrid" else 1)
    response = pulse_response(compiled["pulses"], spectrum)
    potential_calls = 0
    peak = nc.populations(joint)["boundary_band_population"]

    def observe(state):
        nonlocal peak
        peak = max(peak, nc.populations(state)["boundary_band_population"])

    def potential(state, observer):
        nonlocal potential_calls
        block = response
        if pattern == "alternating" and potential_calls % 2:
            block = response * np.array([[1, -1], [-1, 1]])
        potential_calls += 1
        for site in (0, 1):
            rotated = sg._apply_axis(basis.U_phi.conj().T, state, site)
            if site == 0:
                rotated = np.einsum("iab,ijb->ija", block, rotated)
            else:
                rotated = np.einsum("jab,ijb->ija", block, rotated)
            state = sg._apply_axis(basis.U_phi, rotated, site)
            observer(state)
        return state

    h = scan.T / r
    if route == "gauss_block":
        _, up = np.linalg.eigh(basis.pi)

        def diagonal(state, u, generator):
            v = sg._apply_axis(u.conj().T, sg._apply_axis(u.conj().T, state, 0), 1)
            return sg._apply_axis(
                u, sg._apply_axis(u, np.exp(-1j * generator)[:, :, None] * v, 0), 1
            )

        readouts = merged_gaussian_readouts(
            joint,
            r,
            scan.gaussian_generators(basis, h / 2),
            scan.gaussian_generators(basis, h),
            lambda v, q: diagonal(v, basis.U_phi, q),
            lambda v, p: diagonal(v, up, p),
            potential,
            observe,
        )
    else:
        native_steps = {
            duration: nc.make_step(
                problem, scan.E_J, scan.KAPPA, scan.LAM2, duration, "native"
            )
            for duration in (h / 2, h)
        }

        def native(state, duration):
            return native_steps[duration](state)

        def trajectory():
            yield joint
            state = native(joint, h / 2)
            for k in range(1, r + 1):
                observe(state)
                state = potential(state, observe)
                closed = native(state, h / 2)
                observe(closed)
                yield closed
                if k < r:
                    state = native(state, h)

        readouts = trajectory()
    states, rows = [], []
    for k, state in enumerate(readouts):
        metrics = nc.reduced_metrics(state, reference["states"][k])
        obs = nc.two_site_observables(state, basis, joint=True)
        c = complex(obs["C12_real"], obs["C12_imag"])
        rows.append(
            dict(
                bures=metrics["bures_distance"],
                C12=c,
                S2=obs["S2"],
                ancilla=metrics["ancilla_excited_population"],
                purity=metrics["system_purity"],
            )
        )
        states.append(state)
    return np.array(states), rows, peak


def evaluate(problems, refs, r, route, compiled, pattern="fixed"):
    outputs = {
        n: run(problems[n], refs[n, r], r, route, compiled, pattern) for n in scan.CUTS
    }
    low, high = scan.CUTS
    states, rows, peak = outputs[low]
    padded = np.zeros_like(outputs[high][0])
    padded[:, :low, :low, :] = states
    pad = max(
        projective_distance(a.ravel(), b.ravel())
        for a, b in zip(padded, outputs[high][0])
    )
    target = np.zeros((r + 1, high, high), complex)
    target[:, :low, :low] = refs[low, r]["states"].reshape(r + 1, low, low)
    reference = max(
        projective_distance(a.ravel(), b.ravel())
        for a, b in zip(target, refs[high, r]["states"])
    )
    cref = float(np.max(abs(refs[low, r]["C12"] - refs[high, r]["C12"])))
    sref = float(np.max(abs(refs[low, r]["S2"] - refs[high, r]["S2"])))
    uc = (
        cref
        + scan.SOLVER["C12"]
        + max(abs(a["C12"] - b["C12"]) for a, b in zip(rows, outputs[high][1]))
    )
    us = (
        sref
        + scan.SOLVER["S2"]
        + max(abs(a["S2"] - b["S2"]) for a, b in zip(rows, outputs[high][1]))
    )
    _, merge_allowance = merge_rotations(pulse_schedule(compiled, r, pattern))
    error = (
        max(row["bures"] for row in rows)
        + reference
        + scan.SOLVER["state"]
        + pad
        + merge_allowance
    )
    ec = (
        max(abs(row["C12"] - refs[low, r]["C12"][k]) for k, row in enumerate(rows)) + uc
    )
    es = max(abs(row["S2"] - refs[low, r]["S2"][k]) for k, row in enumerate(rows)) + us
    boundary = max(peak, outputs[high][2])
    criteria = dict(
        state=bool(error <= scan.EPS and reference <= scan.EPS / 10),
        C12=bool(ec <= 0.01 and uc <= 0.001),
        S2=bool(es <= 0.02 and us <= 0.005),
        boundary=bool(boundary <= 1e-6),
    )
    primitives = counts(compiled, route, r, pattern)
    return dict(
        route=route,
        r=r,
        degree=compiled["degree"],
        completion_pattern=pattern,
        cutoffs=list(scan.CUTS),
        primitive_counts=primitives,
        total_primitives=sum(primitives.values()),
        native_waits=r + 1 if route == "hybrid" else 0,
        state_error=float(error),
        C12_error=float(ec),
        S2_error=float(es),
        margins=dict(
            reference=float(reference),
            padding=float(pad),
            solver=scan.SOLVER["state"],
            merging=merge_allowance,
            C12=float(uc),
            S2=float(us),
        ),
        boundary_population=float(boundary),
        max_ancilla_excited=max(row["ancilla"] for row in rows),
        min_system_purity=min(row["purity"] for row in rows),
        criteria=criteria,
        all_criteria=all(criteria.values()),
    )


def independent_checks():
    """Check physical pulse matrices and the grounding shear identity independently."""
    sample = compile_phase(0.43, 0.129, 5)
    phi = np.array([[0.2, 0.17, 0], [0.17, -0.3, 0.23], [0, 0.23, 0.6]])
    z = np.diag([1.0, -1.0])
    y = np.array([[0, -1j], [1j, 0]])
    actual = np.eye(6, dtype=complex)
    for gate, angle in sample["pulses"]:
        generator = (
            np.kron(phi, z) * angle
            if gate == "CD_Z"
            else -0.5 * angle * np.kron(np.eye(3), y if gate == "Ry" else z)
        )
        actual = expm(1j * generator) @ actual
    eig, vec = np.linalg.eigh(phi)
    response = pulse_response(sample["pulses"], eig)
    reference = np.einsum("ix,xab,jx->iajb", vec, response, vec.conj()).reshape(6, 6)
    pulse_error = float(np.linalg.norm(actual - reference))
    k = np.array([[1.2, -1.0], [-1.0, 1.2]])
    w, v = np.linalg.eigh(k)
    tau = 0.43
    a = (v * (np.sqrt(w) * np.tan(tau * np.sqrt(w) / 2))) @ v.T
    b = (v * (np.sin(tau * np.sqrt(w)) / np.sqrt(w))) @ v.T
    eye, zero = np.eye(2), np.zeros((2, 2))
    position = np.block([[eye, zero], [-a, eye]])
    momentum = np.block([[eye, b], [zero, eye]])
    exact = expm(tau * np.block([[zero, eye], [-k, zero]]))
    shear_error = float(np.linalg.norm(position @ momentum @ position - exact))
    if pulse_error > 1e-11 or shear_error > 1e-12:
        raise AssertionError((pulse_error, shear_error))
    # Independent dense circuit oracle for the complete boundary merges, with
    # different site signals and arbitrary intervening system-only evolution.
    sys_phi = [np.kron(phi, np.eye(3)), np.kron(np.eye(3), phi)]
    rng = np.random.default_rng(524)
    hermitian = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    system = np.kron(expm(-0.03j * (hermitian + hermitian.conj().T)), np.eye(2))

    def circuit(pulses):
        unitary, queries = np.eye(18, dtype=complex), 0
        for gate, angle in pulses:
            if gate == "CD_Z":
                if queries and queries % (4 * sample["degree"]) == 0:
                    unitary = system @ unitary
                site = (queries // (2 * sample["degree"])) % 2
                generator = angle * np.kron(sys_phi[site], z)
                queries += 1
            else:
                generator = -0.5 * angle * np.kron(np.eye(9), y if gate == "Ry" else z)
            unitary = expm(1j * generator) @ unitary
        return unitary

    merge_error = 0.0
    for pattern in ("fixed", "alternating"):
        raw = pulse_schedule(sample, 3, pattern)
        merged, _ = merge_rotations(raw)
        merge_error = max(
            merge_error, float(np.linalg.norm(circuit(raw) - circuit(merged), 2))
        )
        if len(merged) != sum(counts(sample, "hybrid", 3, pattern).values()):
            raise AssertionError("count does not describe the merged circuit")
    small = scan.build_two_site_problem(
        6, scan.OMEGA_B, scan.E_J, scan.KAPPA, scan.LAM2
    )
    psi = rng.normal(size=(6, 6, 2)) + 1j * rng.normal(size=(6, 6, 2))
    psi /= np.linalg.norm(psi)
    step = nc.make_step(small, scan.E_J, scan.KAPPA, scan.LAM2, 0.19, "native")
    native_exact = expm(-0.19j * small["H_native"].toarray()) @ psi.reshape(36, 2)
    native_error = float(np.linalg.norm(step(psi).reshape(36, 2) - native_exact))
    if merge_error > 1e-11 or native_error > 1e-11:
        raise AssertionError((merge_error, native_error))
    return dict(
        dense_pulse_error=pulse_error,
        grounding_symplectic_error=shear_error,
        whole_circuit_merge_error=merge_error,
        matrix_free_native_error=native_error,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"existing evidence must be kept: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def timeout(*_):
        raise TimeoutError("600-second comparison limit")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    report = dict(
        environment=scan.environment(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        command=" ".join(["python", *sys.argv]),
        limit_seconds=600,
        preparation=[1.0, 0.0],
        steps=[6, 7, 8],
        degrees=list(range(1, 11)),
        completion_patterns=["fixed", "alternating"],
        source_hashes={
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("joint_phase.py"))
        },
        checks=independent_checks(),
        compilers=[],
        rows=[],
        status="running",
    )
    try:
        selected = []
        for route in ("hybrid", "gauss_block"):
            for r in (6, 7, 8):
                exemplar = False
                for degree in range(1, 11):
                    a, b = (
                        (scan.T / r * 0.3, 0.0)
                        if route == "hybrid"
                        else (scan.T / r, scan.T / r * 0.3)
                    )
                    compiled = compile_phase(a, b, degree)
                    accurate = (
                        compiled["diagnostics"]["sampled_paired_error"]
                        + compiled["diagnostics"]["interpolation_allowance"]
                        <= 0.001
                    )
                    example = r == 7 and accurate and not exemplar
                    if example:
                        exemplar = True
                    for pattern in ("fixed", "alternating"):
                        cost = sum(counts(compiled, route, r, pattern).values())
                        competitive = cost < (64 if route == "hybrid" else 209)
                        propagate = competitive or (example and pattern == "fixed")
                        report["compilers"].append(
                            dict(
                                route=route,
                                r=r,
                                total_primitives=cost,
                                completion_pattern=pattern,
                                propagated=propagate,
                                **compiled,
                            )
                        )
                        if propagate:
                            selected.append((route, r, compiled, pattern))
        print(
            "selected",
            [(route, r, c["degree"], p) for route, r, c, p in selected],
            flush=True,
        )
        problems, refs = {}, {}
        for n in scan.CUTS:
            problems[n] = scan.build_two_site_problem(
                n, scan.OMEGA_B, scan.E_J, scan.KAPPA, scan.LAM2
            )
            initial = _sg_initial(problems[n]["basis"], 1.0, 0.0)
            for r in sorted({r for _, r, _, _ in selected}):
                tick = time.perf_counter()
                states = scan.exact_trajectory(
                    problems[n]["H_target"], initial, scan.T, r
                )
                obs = [
                    nc.two_site_observables(v.reshape(n, n), problems[n]["basis"])
                    for v in states
                ]
                refs[n, r] = dict(
                    states=states,
                    C12=np.array([complex(o["C12_real"], o["C12_imag"]) for o in obs]),
                    S2=np.array([o["S2"] for o in obs]),
                )
                print(
                    f"reference N={n} r={r} {time.perf_counter() - tick:.2f}s",
                    flush=True,
                )
        for route, r, compiled, pattern in selected:
            tick = time.perf_counter()
            row = evaluate(problems, refs, r, route, compiled, pattern)
            row["seconds"] = time.perf_counter() - tick
            report["rows"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        report["status"] = "completed"
    except TimeoutError as exc:
        report["status"] = "incomplete"
        report["reason"] = str(exc)
    except Exception as exc:
        report["status"] = "failed"
        report["reason"] = str(exc)
        raise
    finally:
        signal.alarm(0)
        report["wall_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
