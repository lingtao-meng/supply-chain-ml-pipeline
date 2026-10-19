"""Tests for the budget-constrained allocator.

The interesting bug this guards against: giving every order a single action
throws away the cheap-notify -> express upgrade, so the planner under-spends
and leaves money on the table.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import policy  # noqa: E402


def flat_pmf(atom_values, probabilities):
    pmf = np.zeros(len(policy.ATOMS))
    for value, prob in zip(atom_values, probabilities):  # noqa: B905 (py3.9 compat)
        pmf[int(np.where(policy.ATOMS == value)[0][0])] = prob
    return pmf


STANDARD = flat_pmf([-2, -1, 0, 1, 2], [0.2] * 5)
SECOND = flat_pmf([0, 1, 2, 3, 4], [0.2] * 5)


class TestBudgetPolicy(unittest.TestCase):
    def setUp(self):
        # 40 Standard-style orders (flat -2..2) and 40 Second-style (flat 0..4)
        self.pmf = np.vstack([STANDARD] * 40 + [SECOND] * 40)

    def test_zero_budget_means_no_action(self):
        idx = policy.policy_budget_pmf(self.pmf, 200, budget=0)
        self.assertTrue((idx == policy.IDX_NONE).all())

    def test_large_budget_matches_unconstrained_optimum(self):
        idx = policy.policy_budget_pmf(self.pmf, 200, budget=10_000)
        optimum = policy.policy_expected_cost_pmf(self.pmf, 200)
        np.testing.assert_array_equal(idx, optimum)

    def test_spend_never_exceeds_budget(self):
        costs = np.array([a.cost for a in policy.ACTIONS])
        for budget in (0, 50, 200, 1_000, 5_000):
            idx = policy.policy_budget_pmf(self.pmf, 200, budget)
            self.assertLessEqual(float(costs[idx].sum()), budget + 1e-9, msg=f"budget={budget}")

    def test_tight_budget_uses_notify_before_express(self):
        idx = policy.policy_budget_pmf(self.pmf, 200, budget=20)
        self.assertEqual(int((idx == policy.IDX_NOTIFY).sum()), 1)
        self.assertEqual(int((idx == policy.IDX_EXPRESS).sum()), 0)

    def test_leftover_budget_upgrades_notify_to_express(self):
        # Second Class at $200/day: none = $400, notify = 20+0.6*200*2 = $260,
        # express = 100+200*0.6 = $220.  With $20 the planner notifies; with
        # $100 it can afford the $80 upgrade to express.
        pmf = SECOND.reshape(1, -1)
        self.assertEqual(policy.policy_budget_pmf(pmf, 200, 20)[0], policy.IDX_NOTIFY)
        self.assertEqual(policy.policy_budget_pmf(pmf, 200, 100)[0], policy.IDX_EXPRESS)

    def test_budget_curve_is_non_increasing_in_cost(self):
        curve = policy.budget_curve(
            self.pmf,
            np.array([-2, -1, 0, 1, 2] * 16),
            200,
            budgets=[0, 100, 500, 2_000, 10_000],
        )
        # Realised cost is noisy on a fixed delay vector; the invariant the
        # greedy actually guarantees is that *expected* cost never increases.
        costs = curve["expected_total_cost"].to_numpy()
        self.assertTrue(np.all(np.diff(costs) <= 1e-6), msg=str(costs))
        self.assertTrue((curve["action_spend"] <= curve["budget"] + 1e-6).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
