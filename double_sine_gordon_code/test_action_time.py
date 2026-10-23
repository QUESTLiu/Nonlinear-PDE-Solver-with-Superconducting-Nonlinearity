"""Independent identities for the action-based time accounting."""

import math
import unittest

import numpy as np
from scipy.linalg import expm

from double_sine_gordon_code import dsg_action_time as action
from double_sine_gordon_code.dsg_control_tradeoffs import load_rows, timing_point


class ActionTimeTests(unittest.TestCase):
    def test_trigonometric_actions_and_boundary_rotations(self):
        unit = 1 / (2 * math.sqrt(2 * action.OMEGA_B))
        factors = {"hybrid": {1: 16, 2: 24}, "gauss_block": {1: 24, 2: 32, 3: 28}}
        for route, orders in factors.items():
            for order, factor in orders.items():
                for r in (1, 4, 7):
                    row = dict(route=route, r=r, order=order, s=1, lambda2=0.3)
                    pulses = action.trig_pulses(row)
                    total = {k: sum(abs(p["value"]) for p in pulses if p["gate"] == k)
                             for k in ("CD_Z", "Ry", "Rz")}
                    # Each E factor has two displacement endpoints. Adjacent
                    # endpoints add, except the opposite signs at a (2,2)
                    # harmonic boundary. Only global qubit bases remain.
                    self.assertAlmostEqual(total["CD_Z"], factor * r * unit, places=12)
                    potential = 0.3 if route == "hybrid" else 1.3
                    self.assertAlmostEqual(total["Ry"], math.pi + 4 * action.T * potential, places=12)
                    self.assertAlmostEqual(total["Rz"], math.pi, places=12)

    def test_gaussian_canonical_map(self):
        eye, zero = np.eye(2), np.zeros((2, 2))
        J = np.block([[zero, eye], [-eye, zero]])
        for kappa in (0.5, 1.0, 2.0):
            for tau in (0.07, 0.4, 0.9):
                c = action.shear_coefficients(tau, kappa)
                q = np.array([[2*c["X2"], c["XX"]], [c["XX"], 2*c["X2"]]])
                p = np.array([[2*c["P2"], c["PP"]], [c["PP"], 2*c["P2"]]])
                sq = np.block([[eye, zero], [-q, eye]])
                sp = np.block([[eye, p], [zero, eye]])
                G = np.block([[kappa/action.OMEGA_B*np.array([[1, -1], [-1, 1]]), zero],
                              [zero, action.OMEGA_B*eye]])
                np.testing.assert_allclose(sq @ sp @ sq, expm(J @ G * tau), atol=2e-14, rtol=2e-14)

    def test_quadratic_actions_counts_and_limit(self):
        previous = math.inf
        for r in (2, 4, 8, 16, 64):
            pulses = action.gaussian_pulses("gauss_block", r)
            self.assertEqual(len(pulses), 6*r + 9)
            totals = {k: sum(abs(v) for kind, v in pulses if kind == k)
                      for k in ("X2", "P2", "XX", "PP")}
            self.assertAlmostEqual(totals["X2"], totals["XX"], places=12)
            self.assertAlmostEqual(totals["P2"] + totals["PP"], action.OMEGA_B*action.T, places=12)
            current = sum(totals.values())
            local_limit = action.OMEGA_B*action.T + 2*action.T/action.OMEGA_B
            self.assertGreater(current, local_limit)
            self.assertLess(current, previous)
            previous = current
            kp = action.gaussian_pulses("kin_pot", r)
            self.assertEqual(len(kp), 5*r+2)
            self.assertAlmostEqual(sum(abs(v) for _, v in kp), local_limit, places=12)

    def test_fixed_count_limit_recovers_previous_selection(self):
        candidates = action.load_candidates()
        rows = load_rows()
        zero_rates = dict.fromkeys(action.KINDS, 0.0)
        for allowance in action.ALLOWANCES:
            for switch in (0.0, 13.0, 15.0):
                for gaussian in (0.1, 1.0, 3.0):
                    overheads = action.rates(gaussian)
                    new = action.select(candidates, allowance, overheads, zero_rates, switch)
                    old = timing_point(rows, allowance, gaussian, switch)
                    self.assertEqual(new["routes"]["hybrid"]["id"], old["hybrid"])
                    self.assertEqual(new["routes"]["gauss_block"]["id"], old["gauss_block"])
                    same_pool_boundary = (new["routes"]["gauss_block"]["time_without_native"]
                                          - new["routes"]["hybrid"]["time_without_native"])
                    self.assertAlmostEqual(same_pool_boundary, old["critical_native_time"])

    def test_cost_domain_and_candidate_acceptance(self):
        candidates = action.load_candidates()
        for overhead in (-1, math.nan):
            with self.assertRaises(ValueError):
                action.control_time(candidates[0], overhead, action.rates())
        with self.assertRaises(ValueError):
            action.shear_coefficients(math.pi/math.sqrt(2))
        with self.assertRaisesRegex(ValueError, "No accepted hybrid candidate"):
            action.select([], 0.0, 0.0, action.rates())
        for allowance in action.ALLOWANCES:
            chosen = action.select(candidates, allowance, 0.0, action.rates())
            index = {c["id"]: c for c in candidates}
            for result in chosen["routes"].values():
                self.assertTrue(index[result["id"]]["eligible"][str(allowance)])
                self.assertLessEqual(index[result["id"]]["state_error"]+allowance, action.EPS)


if __name__ == "__main__":
    unittest.main()
