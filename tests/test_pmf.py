"""Tests for the discrete-distribution decision layer.

These cover the parts that v1 did not have: an explicit pmf, the fair binary
baseline, bootstrap intervals and capacity allocation from a pmf.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import policy  # noqa: E402


class TestEmpiricalPmf(unittest.TestCase):
    def test_pmf_sums_to_one_and_matches_frequencies(self):
        values = np.array([-1, -1, 0, 1, 1, 1, 2.0])
        pmf = policy.empirical_pmf(values)
        self.assertAlmostEqual(pmf.sum(), 1.0)
        atoms = policy.ATOMS
        self.assertAlmostEqual(pmf[atoms == 1][0], 3 / 7)
        self.assertAlmostEqual(pmf[atoms == 0][0], 1 / 7)

    def test_empty_sample_raises(self):
        with self.assertRaises(ValueError):
            policy.empirical_pmf(np.array([]))


class TestExactExpectedExcess(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(3)
        pmf = rng.dirichlet(np.ones(len(policy.ATOMS)), size=5)
        for reduction in (0, 1, 2):
            got = policy.expected_excess_penalty_pmf(pmf, reduction=reduction)
            tail = np.clip(policy.ATOMS - reduction, 0, None)
            np.testing.assert_allclose(got, pmf @ tail)

    def test_quantile_grid_overstates_a_uniform_atom_distribution(self):
        """The whole reason for the pmf rewrite: 3 points misstate a flat pmf."""
        uniform = np.full(len(policy.ATOMS), 1 / len(policy.ATOMS))
        exact = policy.expected_excess_penalty_pmf(uniform)[0]
        grid = policy.quantile_grid(
            np.array([-2.0]), np.array([0.0]), np.array([2.0])
        )
        approx = policy.expected_excess_penalty(draws=grid)[0]
        self.assertGreater(abs(approx - exact), 1e-6)


class TestPmfPolicies(unittest.TestCase):
    def test_no_delay_means_no_action(self):
        pmf = np.zeros((2, len(policy.ATOMS)))
        pmf[:, policy.ATOMS == 0] = 1.0
        self.assertTrue((policy.policy_expected_cost_pmf(pmf, 200) == policy.IDX_NONE).all())

    def test_short_delay_prefers_express(self):
        pmf = np.zeros((1, len(policy.ATOMS)))
        pmf[0, policy.ATOMS == 3] = 1.0
        self.assertEqual(policy.policy_expected_cost_pmf(pmf, 200)[0], policy.IDX_EXPRESS)

    def test_very_long_delay_prefers_notification(self):
        # 通知：20 + 0.6*200*6 = 740；加急：100 + 200*(6-2) = 900
        atoms = np.arange(-2, 9)
        pmf = np.zeros((1, len(atoms)))
        pmf[0, atoms == 6] = 1.0
        self.assertEqual(policy.policy_expected_cost_pmf(pmf, 200, atoms=atoms)[0], policy.IDX_NOTIFY)

    def test_two_part_pmf_rows_sum_to_one(self):
        pos = policy.empirical_pmf(np.array([1, 2, 2, 3]))
        neg = policy.empirical_pmf(np.array([-1, 0, 0]))
        pmf = policy.two_part_pmf(np.array([0.0, 0.3, 1.0]), pos, neg)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0)

    def test_cost_aware_binary_is_never_worse_than_never_intervening(self):
        pos = policy.empirical_pmf(np.array([1, 2, 2, 3]))
        neg = policy.empirical_pmf(np.array([-1, 0, 0]))
        p_late = np.array([0.1, 0.5, 0.9])
        idx = policy.policy_cost_aware_binary(p_late, pos, neg, 200)
        pmf = policy.two_part_pmf(p_late, pos, neg)
        chosen = policy.realised_cost(idx, np.array([1.0, 2.0, 3.0]), 200)
        never = policy.realised_cost(
            np.full(3, policy.IDX_NONE), np.array([1.0, 2.0, 3.0]), 200
        )
        self.assertTrue((chosen <= never).all())
        self.assertEqual(pmf.shape, (3, len(policy.ATOMS)))

    def test_capacity_pmf_respects_capacity(self):
        rng = np.random.default_rng(0)
        pmf = rng.dirichlet(np.ones(len(policy.ATOMS)), size=100)
        for cap in (0.0, 0.1, 0.5, 1.0):
            idx = policy.policy_capacity_pmf(pmf, 200, cap)
            self.assertLessEqual(int((idx != policy.IDX_NONE).sum()), round(cap * 100) + 1)

    def test_tune_threshold_returns_a_grid_point(self):
        rng = np.random.default_rng(1)
        p_late = rng.uniform(size=200)
        delay = rng.integers(-2, 5, 200).astype(float)
        tau, cost = policy.tune_threshold(p_late, delay, 200)
        self.assertGreaterEqual(tau, 0.05)
        self.assertLessEqual(tau, 0.95)
        self.assertGreaterEqual(cost, 0.0)


class TestBootstrap(unittest.TestCase):
    def test_identical_policies_are_not_significant(self):
        rng = np.random.default_rng(2)
        cost = rng.uniform(0, 300, 500)
        stats = policy.bootstrap_diff(cost, cost, n_boot=300)
        self.assertAlmostEqual(stats["mean_diff"], 0.0)
        self.assertFalse(stats["significant"])

    def test_a_clear_difference_is_significant(self):
        rng = np.random.default_rng(2)
        cost = rng.uniform(0, 300, 500)
        stats = policy.bootstrap_diff(cost + 10, cost, n_boot=300)
        self.assertAlmostEqual(stats["mean_diff"], 10.0)
        self.assertTrue(stats["significant"])


class TestActionMix(unittest.TestCase):
    def test_shares_sum_to_one(self):
        idx = np.array([policy.IDX_NONE, policy.IDX_NOTIFY, policy.IDX_EXPRESS, policy.IDX_EXPRESS])
        mix = policy.action_mix(idx)
        self.assertAlmostEqual(sum(mix.values()), 1.0)
        self.assertAlmostEqual(mix["express"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
