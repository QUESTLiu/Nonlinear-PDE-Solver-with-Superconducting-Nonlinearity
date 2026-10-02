"""Small independent scientific checks. No pilot or production scan is run."""

import unittest
from unittest.mock import patch
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy.linalg import expm

from duffing_code import duffing_padded_hybrid_validation as df
from duffing_code.native_comparison import kerr_snap, constructive_pair
from sine_gordon_code import sglib as sg
from sine_gordon_code import native_correction as nc
from double_sine_gordon_code.run_snapd_end_to_end import build_two_site_problem
from common_code.metrics import projective_distance, projective_distances, centered_operator, native_residual_integral, native_verdict, family_action_sum, first_crossing

RECORDS = {}


def independent_ladder(n):
    out = np.zeros((n,n),complex)
    for level in range(1,n):
        out[level-1,level] = np.sqrt(level)
    return out


class RepresentationChecks(unittest.TestCase):
    def test_batch_replay_preserves_existing_job_log(self):
        from common_code.records import ROOT
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            log = output / 'archived_job.log'
            original = b'archived scientific run\n'
            log.write_bytes(original)
            manifest = dict(output_root=str(output), execution_record='fresh_receipt.json',
                            memory_cap_bytes=256 * 1024**2, storage_cap_bytes=1024**2,
                            jobs=[dict(id='archived_job', batch='B0', wall_cap_seconds=5,
                                       command=[sys.executable, '-c', "print('new run')"])])
            path = output / 'manifest.json'
            path.write_text(json.dumps(manifest))
            result = subprocess.run([sys.executable, '-m', 'common_code.run_batch',
                                     str(path), '--batch', 'B0', '--execute'],
                                    cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(log.read_bytes(), original)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FileExistsError', result.stderr)
            self.assertIn(str(log), result.stderr)
            self.assertFalse((output / 'fresh_receipt.json').exists())

    def test_gaussian_boundary_checkpoints_cover_complete_prefixes(self):
        """Dense complete circuits determine every monitored state and boundary maximum."""
        from double_sine_gordon_code.gaussian_sequence import merged_gaussian_readouts
        n, steps, h = 6, 3, .8
        a = independent_ladder(n)
        q = (a + a.conj().T) / np.sqrt(2)
        p = 1j * (a.conj().T - a) / np.sqrt(2)
        q_values, q_basis = np.linalg.eigh(q)
        p_values, p_basis = np.linalg.eigh(p)
        q_coefficient = lambda tau: .5 * np.tan(tau / 2)
        p_coefficient = lambda tau: .5 * np.sin(tau)
        half = (q_coefficient(h / 2) * q_values**2, p_coefficient(h / 2) * p_values**2)
        full = (q_coefficient(h) * q_values**2, p_coefficient(h) * p_values**2)

        def position(state, generator):
            return q_basis @ (np.exp(-1j * generator)[:, None] * (q_basis.conj().T @ state))

        def momentum(state, generator):
            return p_basis @ (np.exp(-1j * generator)[:, None] * (p_basis.conj().T @ state))

        cosine = (expm(1j * q) + expm(-1j * q)) / 2
        sine = (expm(1j * q) - expm(-1j * q)) / (2j)
        y, z = np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])
        blocks = [expm(.2j * (np.kron(cosine, z) + sign * np.kron(sine, y)))
                  for sign in (1, -1)]

        def potential(state, observe):
            for block in blocks:
                state = (block @ state.ravel()).reshape(n, 2)
                observe(state)
            return state

        initial = np.zeros((n, 2), complex)
        initial[0, 0], initial[1, 1], initial[2, 0] = 1., .3j, .2
        initial /= np.linalg.norm(initial)

        def dense_position(coefficient):
            return np.kron(expm(-1j * coefficient * (q @ q)), np.eye(2))

        def dense_momentum(coefficient):
            return np.kron(expm(-1j * coefficient * (p @ p)), np.eye(2))

        def complete_circuit(count):
            state = dense_momentum(p_coefficient(h / 2)) @ dense_position(q_coefficient(h / 2)) @ initial.ravel()
            checkpoints = []
            for j in range(1, count + 1):
                left = q_coefficient(h / 2 if j == 1 else h)
                right = q_coefficient(h / 2 if j == count else h)
                state = dense_position(left + right) @ state
                checkpoints.append(state.copy())
                for block in blocks:
                    state = block @ state
                    checkpoints.append(state.copy())
                state = dense_momentum(p_coefficient(h / 2 if j == count else h)) @ state
            state = dense_position(q_coefficient(h / 2)) @ state
            checkpoints.append(state.copy())
            return checkpoints, state

        for final_only in (False, True):
            with self.subTest(final_only=final_only):
                observed = []
                actual = list(merged_gaussian_readouts(initial, steps, half, full, position,
                              momentum, potential, lambda state: observed.append(state.ravel().copy()),
                              final_only=final_only))
                counts = [steps] if final_only else range(1, steps + 1)
                expected, readouts = [], [initial.ravel()]
                for count in counts:
                    checkpoints, state = complete_circuit(count)
                    expected.extend(checkpoints)
                    readouts.append(state)
                np.testing.assert_allclose(np.array(actual).reshape(-1, 2 * n), readouts, atol=3e-12, rtol=0)
                if final_only:
                    np.testing.assert_allclose(observed, expected, atol=3e-12, rtol=0)
                else:
                    distances = np.linalg.norm(np.array(expected)[:, None, :] - np.array(observed)[None, :, :], axis=2)
                    self.assertLess(float(np.max(np.min(distances, axis=1))), 3e-12)
                    self.assertLess(float(np.max(np.min(distances, axis=0))), 3e-12)
                boundary = lambda states: max(float(np.sum(abs(state.reshape(n, 2)[-4:])**2)) for state in states)
                self.assertAlmostEqual(boundary(observed), boundary(expected), places=12)

    def test_gaussian_error_intervals_and_unresolved_selection(self):
        """Partial trace contracts Bures distance; overlapping bounds cannot select a winner."""
        from double_sine_gordon_code.audit_gaussian_sequence import distance, interval
        from double_sine_gordon_code.gaussian_evidence import passes_at
        rng = np.random.default_rng(4102)
        old = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
        old /= np.linalg.norm(old)
        new = old + .03 * (rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)))
        new /= np.linalg.norm(new)
        target = old[:, 0] / np.linalg.norm(old[:, 0])
        def reduced_bures(state):
            fidelity = sum(abs(np.vdot(target, state[:, q]))**2 for q in (0, 1))
            return np.sqrt(2 - 2 * np.sqrt(fidelity))
        lower, upper = interval(reduced_bures(old), [distance(old, new)])
        self.assertLessEqual(lower, reduced_bures(new))
        self.assertLessEqual(reduced_bures(new), upper)
        row = dict(acceptance=dict(boundary=True, C12=True, S2=True),
                   margins=dict(reference=.0001, solver=0.), state_error=.04,
                   bounds=dict(state_error=[.02, .04]), route='gauss_block', lambda2=.3,
                   preparation=[1., 0.], cutoffs=[128, 160], r=7, order=3, s=1)
        with self.assertRaises(ValueError):
            passes_at(row, np.sqrt(1e-3))
        row['bounds']['state_error'] = [.035, .04]
        self.assertFalse(passes_at(row, np.sqrt(1e-3)))
        row['bounds']['state_error'] = [.02, .025]
        self.assertTrue(passes_at(row, np.sqrt(1e-3)))

    def test_three_site_gaussian_readouts(self):
        """Open-chain modes 0,1,3 give an independent dense interval oracle."""
        from double_sine_gordon_code.dsg_chain import Chain, T, OMEGA_B
        n, r = 3, 3
        chain = Chain(3, n, .3, (.7, .2))
        a = independent_ladder(n)
        q = (a + a.conj().T) / np.sqrt(2 * OMEGA_B)
        p = 1j * np.sqrt(OMEGA_B / 2) * (a.conj().T - a)
        def site_matrix(local, site):
            out = np.ones((1, 1))
            for j in range(3):
                out = np.kron(out, local if j == site else np.eye(n))
            return out
        qs, ps = [[site_matrix(local, j) for j in range(3)] for local in (q, p)]
        modes = [np.ones(3) / np.sqrt(3), np.array([1, 0, -1]) / np.sqrt(2),
                 np.array([1, -2, 1]) / np.sqrt(6)]
        def gaussian(tau):
            Q = np.zeros((n**3, n**3), complex)
            P = Q.copy()
            for w, mode in zip((0, 1, np.sqrt(3)), modes):
                qm, pm = [sum(v * m for v, m in zip(mode, matrices)) for matrices in (qs, ps)]
                aa = .5 * w * np.tan(.5 * w * tau)
                bb = np.sin(w * tau) / (2 * w) if w else tau / 2
                Q += aa * (qm @ qm)
                P += bb * (pm @ pm)
            phi = expm(-1j * Q)
            return np.kron(phi @ expm(-1j * P) @ phi, np.eye(2))
        h = T / r
        x, z = np.array([[0, 1], [1, 0]]), np.diag([1, -1])
        potential = np.eye(2 * n**3, dtype=complex)
        for coordinate in qs:
            for m, c in ((1, 1.), (2, .3)):
                d = expm(.5j * m * np.kron(coordinate, x))
                rot = expm(.5j * h * c * np.kron(np.eye(n**3), z))
                potential = np.exp(-1j * h * c) * (d @ rot @ d.conj().T) @ (d.conj().T @ rot @ d) @ potential
        initial = np.zeros((n, n, n, 2), complex)
        initial[..., 0] = chain.psi0
        actual = list(chain.gaussian_readouts(initial, h, r, 1, 1, lambda state: None))
        half, full = gaussian(h / 2), gaussian(h)
        state = half @ initial.ravel()
        for k in range(1, r + 1):
            state = potential @ state
            np.testing.assert_allclose(actual[k].ravel(), half @ state, atol=3e-12, rtol=0)
            state = full @ state
        self.assertEqual(chain.counts('gauss_block', r, 1, 1)[0]['Gaussian'], 12 * r + 18)

    def test_gaussian_readouts_match_the_counted_interval_sequence(self):
        """Appendix J counts merged intervals, not repeated finite half steps."""
        from double_sine_gordon_code import dsg_step_scan as scan
        n, r, order = 4, 3, 3
        h = scan.T / r
        a = independent_ladder(n)
        q = (a + a.conj().T) / np.sqrt(2 * scan.OMEGA_B)
        p = 1j * np.sqrt(scan.OMEGA_B / 2) * (a.conj().T - a)
        eye = np.eye(n)
        q1, q2 = np.kron(q, eye), np.kron(eye, q)
        p1, p2 = np.kron(p, eye), np.kron(eye, p)
        identity = np.eye(n * n)
        x, z = np.array([[0, 1], [1, 0]]), np.diag([1, -1])

        def gaussian(tau):
            omega = np.sqrt(2 * scan.KAPPA)
            aa = omega * np.tan(omega * tau / 2) / 2
            bb = np.sin(omega * tau) / (2 * omega)
            phi = expm(-.5j * aa * (q1 - q2) @ (q1 - q2))
            momentum = expm(-1j * ((tau / 4 + bb / 2) * (p1 @ p1 + p2 @ p2)
                                  + (tau / 2 - bb) * (p1 @ p2)))
            return np.kron(phi @ momentum @ phi, np.eye(2))

        # Independent full joint matrices implement the Appendix J exponentials.
        potential = np.eye(2 * n * n, dtype=complex)
        for coordinate in (q1, q2):
            for harmonic, coefficient, block_order in ((1, scan.E_J, 2), (2, scan.LAM2, 1)):
                controlled = expm(.5j * harmonic * np.kron(coordinate, x))
                def elementary(v, sign):
                    rot = expm(1j * v * np.kron(identity, z))
                    d = controlled if sign == 1 else controlled.conj().T
                    return d @ rot @ d.conj().T
                u = h * coefficient
                if block_order == 1:
                    block = elementary(u / 2, 1) @ elementary(u / 2, -1)
                else:
                    block = elementary(u / 4, 1) @ elementary(u / 2, -1) @ elementary(u / 4, 1)
                potential = np.exp(-1j * u) * block @ potential

        prep = (.7, .2)
        problem = build_two_site_problem(n, scan.OMEGA_B, scan.E_J, scan.KAPPA, scan.LAM2)
        initial = scan._sg_initial(problem['basis'], *prep)
        target = np.array([expm(-1j * k * h * problem['H_target'].toarray()) @ initial.ravel()
                           for k in range(r + 1)])
        obs = [nc.two_site_observables(state.reshape(n, n), problem['basis']) for state in target]
        reference = dict(states=target, C12=np.array([v['C12_real'] + 1j * v['C12_imag'] for v in obs]),
                         S2=np.array([v['S2'] for v in obs]))
        _, actual, _, _ = scan.run_circuit('gauss_block', n, r, order, 1, prep, reference)
        final_only = list(scan.gauss_block_readouts(problem, actual[0], h, r, order, 1,
                                                   lambda state: None, final_only=True))
        np.testing.assert_allclose(final_only[-1], actual[-1], atol=3e-12, rtol=0)
        half, full = gaussian(h / 2), gaussian(h)
        joint = np.kron(initial.ravel(), [1, 0])
        running = half @ joint
        residuals = []
        for k in range(1, r + 1):
            running = potential @ running
            expected = half @ running
            residuals.append(float(np.linalg.norm(actual[k].ravel() - expected)))
            np.testing.assert_allclose(actual[k].ravel(), expected, atol=3e-12, rtol=0)
            running = full @ running
        counts, _ = scan.primitive_counts('gauss_block', r, order, 1)
        # r+2 position factors and r+1 momentum factors, three gates each.
        self.assertEqual(counts['Gaussian'], 3 * ((r + 2) + (r + 1)))
        RECORDS['merged_gaussian_dense_oracle'] = {'cutoff': n, 'steps': r, 'state_residuals': residuals}

    def test_sg_evaluation_uses_requested_active_space(self):
        from common_code.run_pilot import evaluate_sg
        phases=np.array([[0.,.1,-.2,.3]])
        vector=df.pack_parameters(phases,np.zeros((1,2)))
        problem=build_two_site_problem(8,np.sqrt(3),1.,1.,.3)
        half=expm(-1j*(5.2/np.sqrt(3)/20)*problem["H_native"].toarray())
        local=np.ones(8,dtype=complex);local[:4]=np.exp(1j*phases[0])
        cache={}
        with tempfile.TemporaryDirectory() as directory:
            for i,preparation in enumerate(((1.,0.),(.4,.2))):
                out=Path(directory)/str(i);out.mkdir()
                evaluate_sg(out,vector,1,10,None,[("H_SNAP",2,1)],reference_cache=cache,
                            validation_cutoffs=(8,12),training_dimensions=(8,4,2),preparation=preparation)
                with np.load(out/"E4_H_SNAP_o2_s1_N8.npz") as data:
                    actual=data["joint_step_states"][1,:,:,0].ravel()
                    reported_distance=float(data["bures_distance"][1])
                phi0,pi0=preparation;omega=np.sqrt(3)
                alpha=np.sqrt(omega/2)*phi0+1j*pi0/np.sqrt(2*omega)
                initial=np.kron(df.coherent_state(8,alpha)[0],np.eye(8,dtype=complex)[:,0])
                expected=half@((np.kron(local,local))*(half@initial))
                target=expm(-1j*(5.2/np.sqrt(3)/10)*problem["H_target"].toarray())@initial
                np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0)
                np.testing.assert_allclose(reported_distance,projective_distance(expected,target),atol=2e-12,rtol=0)

    def test_paired_frames_preserve_diagonal_start_and_escape_parity(self):
        n,d,k,layers=12,10,6,4
        a=independent_ladder(n)
        phi=(a+a.conj().T)/np.sqrt(2*np.sqrt(3))
        cosine=(expm(2j*phi)+expm(-2j*phi))/2
        target=expm(-.09j*(np.eye(n)-cosine))
        inputs=np.eye(n,dtype=complex)[:,:k]
        vector=df.paired_frame_parameters(target,d,layers,20260729)
        theta,alpha=df.unpack_parameters(vector,layers,d)
        direct=np.eye(n,dtype=complex)
        for phases,(x,y) in zip(theta,alpha):
            snap=np.ones(n,dtype=complex);snap[:d]=np.exp(1j*phases)
            direct=expm((x+1j*y)*a.conj().T-(x-1j*y)*a)@(snap[:,None]*direct)
        diagonal=np.ones(n,dtype=complex);diagonal[:d]=np.exp(1j*np.angle(np.diag(target)[:d]))
        np.testing.assert_allclose(direct,np.diag(diagonal),atol=3e-14,rtol=0)
        compiler=df.SnapDisplacementCompiler(n,d,layers,inputs,target@inputs,0.)
        loss,gradient=compiler.value_and_grad(vector)
        zero=df.pack_parameters(theta,np.zeros_like(alpha))
        diagonal_loss,diagonal_gradient=compiler.value_and_grad(zero)
        self.assertAlmostEqual(loss,diagonal_loss,places=13)
        self.assertLess(np.linalg.norm(diagonal_gradient),3e-14)
        self.assertGreater(np.linalg.norm(gradient),1e-4)
        index=int(np.argmax(abs(gradient)));epsilon=1e-6
        plus=vector.copy();minus=vector.copy();plus[index]+=epsilon;minus[index]-=epsilon
        derivative=(compiler.value_and_grad(plus)[0]-compiler.value_and_grad(minus)[0])/(2*epsilon)
        np.testing.assert_allclose(gradient[index],derivative,atol=3e-10,rtol=0)
        RECORDS["paired_frame_gradient"]={"zero_norm":float(np.linalg.norm(diagonal_gradient)),"paired_norm":float(np.linalg.norm(gradient)),"finite_difference_error":float(abs(gradient[index]-derivative))}

    def test_free_kinetic_preservation_and_option_scope(self):
        from sine_gordon_code import common
        parser=common.base_parser("free kinetic control")
        args=parser.parse_args(["--E-J","0","--kappa","0","--omega","1.2","--n-cut","8"])
        with contextlib.redirect_stdout(io.StringIO()):
            ctx=common.setup(args,"free")
        self.assertEqual(ctx["Omega_time"],0.)
        self.assertEqual(ctx["params"]["lambda2"],0.)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parser.parse_args(["--lambda2",".3"])
    def test_sandwich_and_finite_ccr(self):
        errors = []
        for n in (8,12,16):
            a = independent_ladder(n+8)
            q = (a+a.T.conj())/np.sqrt(2)
            p = 1j*(a.T.conj()-a)/np.sqrt(2)
            target = (.5*(p@p+q@q)+.1/4*np.linalg.matrix_power(q,4))[:n,:n]
            ops = df.hard_sandwiched_operators(n,1.,.1)
            np.testing.assert_allclose(ops["H_D"], target, atol=3e-12, rtol=0)
            np.testing.assert_allclose(ops["H_D"],ops["H_K"]+ops["lambda"]*(ops["V_2"]+ops["V_4"]),atol=3e-12,rtol=0)
            for key in ("H_D","H_K","V_2","V_4"):
                np.testing.assert_allclose(ops[key],ops[key].conj().T,atol=3e-12,rtol=0)
            an=a[:n,:n]
            expected=np.eye(n);expected[-1,-1]-=n
            np.testing.assert_allclose(an@an.conj().T-an.conj().T@an,expected,atol=1e-14,rtol=0)
            wrong=np.linalg.matrix_power(q[:n,:n],4)
            self.assertGreater(np.linalg.norm(wrong-np.linalg.matrix_power(q,4)[:n,:n]),1)
            errors.append(float(np.linalg.norm(target-ops["H_D"])))
        RECORDS["sandwich_residuals"] = errors

    def test_physical_cosine_and_quadratic_sandwich(self):
        basis=sg.build_mode_basis(8,1.3)
        a=independent_ladder(40)
        q=(a+a.conj().T)/np.sqrt(2*1.3)
        p=1j*np.sqrt(1.3/2)*(a.conj().T-a)
        q2,p2=nc.sandwich_local_operators(basis)
        np.testing.assert_allclose(q2,(q@q)[:8,:8],atol=5e-15,rtol=0)
        np.testing.assert_allclose(p2,(p@p)[:8,:8],atol=5e-15,rtol=0)
        for harmonic in (1,2):
            np.testing.assert_allclose(nc.vertex_sandwich(8,1.3,harmonic),expm(1j*harmonic*q)[:8,:8],atol=3e-13,rtol=0)
            independent=(expm(1j*harmonic*q)+expm(-1j*harmonic*q))/2
            sandwich=nc.cosine_sandwich(8,1.3,harmonic)
            np.testing.assert_allclose(sandwich,independent[:8,:8],atol=3e-13,rtol=0)
        spectral=(basis.U_phi*np.cos(basis.lam_phi))@basis.U_phi.conj().T
        self.assertGreater(np.linalg.norm(spectral-nc.cosine_sandwich(8,1.3)),.1)

    def test_retuning_and_deformed_classical_force(self):
        basis=sg.build_mode_basis(12,1.7)
        c=(basis.U_phi*np.cos(basis.lam_phi))@basis.U_phi.conj().T
        c2=(basis.U_phi*np.cos(2*basis.lam_phi))@basis.U_phi.conj().T
        eye=np.eye(12)
        for lam in (0.,.3,-.2):
            np.testing.assert_allclose(eye-c+lam*(eye-c2),(1+4*lam)*(eye-c)-2*lam*(eye-c)@(eye-c),atol=8e-15,rtol=0)
        times=np.array([0.,1e-5])
        q,p=sg.classical_sine_gordon(np.array([.7,0.]),np.zeros(2),1.,1.,times,False,lambda2=.3)
        force=np.array([-.7-np.sin(.7)-.6*np.sin(1.4),.7])
        np.testing.assert_allclose(p[-1]/times[-1],force,atol=2e-9,rtol=0)

    def test_fixed_variance_driver_time_scale(self):
        from sine_gordon_code import run_classical
        with tempfile.TemporaryDirectory() as directory:
            args=["run_classical","--E-J","8","--kappa","1","--n-site","2","--n-cut","8","--omega",str(np.sqrt(3)),"--omega-time",str(np.sqrt(10)),"--phi0",".1","--tau-max",".2","--steps","16","--n-obs","9","--outdir",directory,"--tag","frequency_check"]
            with patch("sys.argv",args),contextlib.redirect_stdout(io.StringIO()):
                run_classical.main()
            metadata=json.loads(next(Path(directory).glob("*.json")).read_text())
            params=metadata.get("params",metadata)
            self.assertAlmostEqual(params["omega_b"],np.sqrt(3))
            self.assertAlmostEqual(params["Omega_time"],np.sqrt(10))
            self.assertAlmostEqual(params["tau_max_actual"],.2)
            with np.load(next(Path(directory).glob("*.npz"))) as data:
                self.assertAlmostEqual(data["t"][-1],.2/np.sqrt(10))
                np.testing.assert_allclose(data["tau"],np.sqrt(10)*data["t"],atol=1e-15,rtol=0)
                self.assertEqual(data["delta_NG_2"].shape,(9,2))
            RECORDS["fixed_variance_public_driver"]={"omega_b":params["omega_b"],"Omega_time":params["Omega_time"],"tau_actual":params["tau_max_actual"],"steps":16,"cutoff":8}


class ErrorChecks(unittest.TestCase):
    def test_phase_residual_and_three_verdicts(self):
        times=np.linspace(0,1,17)
        psi=np.ones(2)/np.sqrt(2)
        states=np.repeat(psi[:,None],len(times),axis=1)
        zero=native_residual_integral(3*np.eye(2),states,times)
        np.testing.assert_allclose(zero["centered_integral"],0,atol=2e-15,rtol=0)
        record=native_residual_integral(np.diag([.2,-.2]),states,times)
        np.testing.assert_allclose(record["centered_integral"],.2*times,atol=2e-15,rtol=0)
        evolved=np.exp(-1j*np.array([.2,-.2]))*psi
        self.assertLessEqual(projective_distance(psi,evolved),record["centered_integral"][-1])
        np.testing.assert_allclose(centered_operator(np.diag([1.,3.])+7*np.eye(2)),np.diag([-1.,1.]),atol=0,rtol=0)
        self.assertEqual(native_verdict(.01,.001,amplitude_margin=0)["verdict"],"sufficient")
        self.assertEqual(native_verdict(2.,.001)["verdict"],"inconclusive")
        self.assertEqual(native_verdict(2.,.001,observed_infidelity=.1,infidelity_uncertainty=.01)["verdict"],"observed_failure")
        with self.assertRaises(ValueError):
            native_verdict(-1.,.001)
        phases=np.exp(-1j*np.arange(17)*.123)
        same=np.vstack([np.sqrt(.7)*phases,np.sqrt(.3)*phases.conj()])
        np.testing.assert_array_equal(projective_distances(same,same),np.zeros(17))

    def test_action_sum_and_outgoing_error(self):
        x=np.array([1.,0.],complex)
        u=expm(-.13j*nc.X)
        w=expm(-.15j*nc.X)
        errors=family_action_sum(lambda a:w@a,lambda a:u@a,x,5)
        self.assertLessEqual(projective_distance(np.linalg.matrix_power(u,5)@x,np.linalg.matrix_power(w,5)@x),sum(errors)+1e-14)
        action=(w-u)[:,:1]
        self.assertGreater(np.linalg.norm(action),abs(action[0,0]))
        RECORDS["family_bound"]={"action_sum":float(sum(errors)),"actual":projective_distance(np.linalg.matrix_power(u,5)@x,np.linalg.matrix_power(w,5)@x)}

    def test_kerr_replacement_locations(self):
        ops=df.hard_sandwiched_operators(12,1.,.1)
        gates=df.build_fixed_gates(ops,.03)
        psi=np.zeros(12,complex);psi[0]=1/np.sqrt(2);psi[8]=1j/np.sqrt(2)
        _,replacement=kerr_snap(np.diag(ops["H_K"]).real,.03,5)
        exact=expm(-.03j*ops["H_K"])
        np.testing.assert_allclose((replacement-exact)[:,:5],0,atol=1e-14,rtol=0)
        pair=constructive_pair(ops,gates,gates.u4_exact,psi,.03,3,5)
        self.assertEqual(len(pair["locations"]),4)
        self.assertLessEqual(np.linalg.norm(pair["hybrid"][:,-1]-pair["replacement"][:,-1]),pair["replacement_bound"]+1e-13)
        direct=np.linalg.matrix_power(gates.k_half@gates.v2_half@gates.u4_exact@gates.v2_half@gates.k_half,3)@psi
        np.testing.assert_allclose(pair["hybrid"][:,-1],direct,atol=1e-14,rtol=0)

    def test_crossing_bracket_and_censoring(self):
        event=first_crossing([0,1,2],[0,.05,.15],.1)
        self.assertEqual(event["bracket"],[1.,2.])
        self.assertAlmostEqual(event["time"],1.5)
        self.assertIsNone(first_crossing([0,1],[0,.05],.1)["time"])


class CompilerChecks(unittest.TestCase):
    def test_finite_gradient_and_common_phase(self):
        n,d,layers=8,4,2
        inputs=np.zeros((n,3),complex)
        inputs[0,0]=1
        inputs[-1,1]=1
        inputs[1,2]=1/np.sqrt(2);inputs[-2,2]=1j/np.sqrt(2)
        a=independent_ladder(n)
        target=expm(-.07j*(a+a.conj().T)@(a+a.conj().T))@inputs
        compiler=df.SnapDisplacementCompiler(n,d,layers,inputs,target,1e-8)
        rng=np.random.Generator(np.random.PCG64(19))
        vector=df.pack_parameters(rng.normal(0,.2,(layers,d)),rng.normal(0,.2,(layers,2)),.31)
        def independent_loss(v):
            phases=v[:layers*d].reshape(layers,d)
            alphas=v[layers*d:-1].reshape(layers,2)
            y=inputs.copy()
            for phase,xy in zip(phases,alphas):
                diag=np.ones(n,complex);diag[:d]=np.exp(1j*phase)
                alpha=xy[0]+1j*xy[1]
                y=expm(alpha*a.conj().T-alpha.conjugate()*a)@(diag[:,None]*y)
            return np.linalg.norm(np.exp(-1j*v[-1])*y-target)**2/3+1e-8*np.sum(alphas**2)/layers
        value,gradient=compiler.value_and_grad(vector)
        self.assertAlmostEqual(value,independent_loss(vector),places=13)
        errors=[]
        for index in range(vector.size):
            eps=2e-6
            v1,v2=vector.copy(),vector.copy();v1[index]+=eps;v2[index]-=eps
            numeric=(independent_loss(v1)-independent_loss(v2))/(2*eps)
            errors.append(abs(numeric-gradient[index]))
        self.assertLess(max(errors),3e-9)
        for index,limit in ((layers*d,1.5),(0,np.pi),(vector.size-1,np.pi)):
            boundary=vector.copy();boundary[index]=limit
            _,grad=compiler.value_and_grad(boundary)
            h=2e-5;b1=boundary.copy();b2=boundary.copy();b1[index]-=h;b2[index]-=2*h
            fd=(3*independent_loss(boundary)-4*independent_loss(b1)+independent_loss(b2))/(2*h)
            self.assertLess(abs(fd-grad[index]),2e-8)
        RECORDS["finite_gradient_max_absolute_error"]=max(errors)
        basis_inputs=np.eye(4,dtype=complex)[:,:2]
        basis_targets=basis_inputs@np.diag([1.,-1.])
        wrong=df.SnapDisplacementCompiler(4,4,1,basis_inputs,basis_targets,0)
        x=df.pack_parameters(np.zeros((1,4)),np.zeros((1,2)))
        self.assertGreater(wrong.value_and_grad(x)[0],1.)
        aligned=df.SnapDisplacementCompiler(4,4,1,basis_inputs,np.exp(-.3j)*basis_inputs,0)
        x[-1]=.3
        self.assertLess(aligned.value_and_grad(x)[0],1e-28)

    def test_weyl_boundary_falsifier(self):
        a=independent_ladder(8);alpha=.3+.2j
        d,dx,_=df.finite_displacement(alpha,a)
        wrong=(a.conj().T-a+1j*alpha.imag*np.eye(8))@d
        gap=float(np.linalg.norm((wrong-dx)[:,-1]))
        self.assertGreater(gap,.1)
        RECORDS["old_weyl_boundary_derivative_error"]=gap


class TrigonometricChecks(unittest.TestCase):
    def test_joint_conjugation_and_two_ancilla(self):
        for n in (8,12,16):
            basis=sg.build_mode_basis(n,1.3)
            for sign in (-1,1):
                d=expm(.5j*np.kron(basis.phi,nc.X))
                z=np.kron(np.eye(n),nc.Z)
                sigma=d@z@d.conj().T if sign==1 else d.conj().T@z@d
                np.testing.assert_allclose(nc.elementary_trig(basis.phi,1,-.13,sign),expm(-.13j*sigma),atol=5e-14,rtol=0)
            plus=d@z@d.conj().T;minus=d.conj().T@z@d
            cosine=(basis.U_phi*np.cos(basis.lam_phi))@basis.U_phi.conj().T
            np.testing.assert_allclose(plus+minus,2*np.kron(cosine,nc.Z),atol=5e-14,rtol=0)
        n=8;basis=sg.build_mode_basis(n,1.3)
        d=expm(.5j*np.kron(basis.phi,nc.X));sigma=d@np.kron(np.eye(n),nc.Z)@d.conj().T
        hb=np.array([[1,1],[1,-1]])/np.sqrt(2)
        control=np.kron(np.eye(2*n),np.diag([1.,0.]))+np.kron(sigma,np.diag([0.,1.]))
        compute=np.kron(np.eye(2*n),hb)@control@np.kron(np.eye(2*n),hb)
        original=compute@np.kron(np.eye(2*n),expm(.21j*nc.Z))@compute
        # Columns include all states of qubit a, including entangled inputs.
        np.testing.assert_allclose(original[:,::2],np.kron(nc.elementary_trig(basis.phi,1,.21,1),np.array([[1.],[0.]])),atol=5e-14,rtol=0)

    def test_orders_sign_and_schedule(self):
        basis=sg.build_mode_basis(8,1.3)
        y=np.array([[0,-1j],[1j,0]])
        curves=[]
        for order in (1,2):
            errors=[]
            for u in (.04,.02,.01):
                cosine=(basis.U_phi*np.cos(2*basis.lam_phi))@basis.U_phi.conj().T
                target=np.kron(expm(-1j*u*(np.eye(8)-cosine)),np.array([[1.],[0.]]))
                actual=nc.trig_block(basis.phi,2,u,order,1)[:,::2]
                errors.append(float(np.linalg.norm(actual-target,2)))
            self.assertTrue(all(3.8 < errors[i]/errors[i+1] < 4.2 for i in (0,1)) if order==1 else all(7.5 < errors[i]/errors[i+1] < 8.5 for i in (0,1)))
            curves.append(dict(order=order,u=[.04,.02,.01],action_error=errors))
            np.testing.assert_allclose(nc.trig_block(basis.phi,2,0.,order,4),np.eye(16),atol=5e-14,rtol=0)
        pulses=nc.trig_schedule(1,1.3,(-1.2,.3),.02,2,2)
        def execute(schedule):
            out=np.eye(16,dtype=complex)
            for pulse in schedule:
                value=pulse["value"]
                if pulse["gate"]=="CD_Z":
                    op=expm(1j*value*np.kron(independent_ladder(8)+independent_ladder(8).T,nc.Z))
                else:
                    op=np.kron(np.eye(8),expm(-.5j*value*(y if pulse["gate"]=="Ry" else nc.Z)))
                out=op@out
            return np.exp(-.02j*(-1.2+.3))*out
        expected=nc.trig_block(basis.phi,2,.006,2,2)@nc.trig_block(basis.phi,1,-.024,2,2)
        np.testing.assert_allclose(execute(pulses),expected,atol=6e-14,rtol=0)
        np.testing.assert_allclose(execute(nc.merge_pulses(pulses)),expected,atol=6e-14,rtol=0)
        RECORDS["trig_local_order_checks"]=curves
        RECORDS["pulse_counts"]={"unmerged":len(pulses),"merged":len(nc.merge_pulses(pulses))}

    def test_route_generators_and_reduced_metrics(self):
        n=8;problem=build_two_site_problem(n,1.3,1.,1.,.3)
        psi=np.zeros((n,n),complex);psi[0,0]=1/np.sqrt(2);psi[2,1]=1j/np.sqrt(2)
        joint=np.zeros((n,n,2),complex);joint[:,:,0]=psi
        generator=-1j*(problem["H_target"]@psi.ravel()).reshape(n,n)
        errors={}
        for route in ("exact_split","H_SNAP","H_trig","G_trig","H_trig_ret"):
            h=1e-6
            local=expm(-1j*h*problem["correction_local"])
            step=nc.make_step(problem,1.,1.,.3,h,route,order=2,subdivisions=1,local_correction=local)
            derivative=(step(joint)-joint)/h
            err=float(np.linalg.norm(derivative[:,:,0]-generator))
            self.assertLess(err,2e-4)
            self.assertLess(np.linalg.norm(derivative[:,:,1]),2e-5)
            errors[route]=err
        mixed=np.zeros_like(joint);mixed[0,0,0]=1/np.sqrt(2);mixed[1,1,1]=1/np.sqrt(2)
        native_step=nc.make_step(problem,1.,1.,.3,.17,"native")
        independent_native=(expm(-.17j*problem["H_native"].toarray())@mixed.reshape(n*n,2)).reshape(n,n,2)
        np.testing.assert_allclose(native_step(mixed),independent_native,atol=5e-13,rtol=0)
        reference=np.zeros((n,n),complex);reference[0,0]=1
        metrics=nc.reduced_metrics(mixed,reference)
        self.assertAlmostEqual(metrics["infidelity"],.5)
        self.assertAlmostEqual(metrics["system_purity"],.5)
        obs=nc.two_site_observables(mixed,problem["basis"],joint=True)
        self.assertAlmostEqual(obs["S2"],np.log(2))
        vertex=expm(1j*problem["basis"].phi)
        flat=psi.ravel()
        direct=np.vdot(flat,np.kron(vertex,vertex.conj().T)@flat)-np.vdot(flat,np.kron(vertex,np.eye(n))@flat)*np.vdot(flat,np.kron(np.eye(n),vertex.conj().T)@flat)
        pure_obs=nc.two_site_observables(psi,problem["basis"])
        self.assertLess(abs(complex(pure_obs["C12_real"],pure_obs["C12_imag"])-direct),2e-14)
        RECORDS["route_generator_residuals"]=errors


if __name__ == "__main__":
    unittest.main()
