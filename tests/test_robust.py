"""Tests for the minimax-regret policy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import policy  # noqa: E402


def pmf_for(atom_values):
    pmf = np.zeros(len(policy.ATOMS))
    for value in atom_values:
        pmf[int(np.where(policy.ATOMS == value)[0][0])] = 1 / len(atom_values)
    return pmf


STANDARD = pmf_for([-2, -1, 0, 1, 2])
SECOND = pmf_for([0, 1, 2, 3, 4])


class TestRobustPolicy(unittest.TestCase):
    def setUp(self):
        self.pmf = np.vstack([STANDARD, SECOND])

    def test_single_scenario_matches_expected_cost(self):
        robust = policy.policy_minimax_regret(self.pmf, penalties=[200], express_costs=[100])
        nominal = policy.policy_expected_cost_pmf(self.pmf, 200)
        np.testing.assert_array_equal(robust, nominal)

    def test_worst_case_regret_is_no_worse_than_the_nominal_policy(self):
        penalties, express_costs = [100, 200, 400], [50, 100, 150]
        robust = policy.policy_minimax_regret(self.pmf, penalties, express_costs)
        nominal = policy.policy_expected_cost_pmf(self.pmf, 200)

        costs = policy.scenario_cost_matrix(self.pmf, penalties, express_costs)

        def worst_regret(action_idx):
            chosen = costs[:, np.arange(len(self.pmf)), action_idx]  # (scenarios, orders)
            return (chosen - costs.min(axis=2)).max(axis=0)

        self.assertTrue((worst_regret(robust) <= worst_regret(nominal) + 1e-9).all())

    def test_actions_are_valid(self):
        idx = policy.policy_minimax_regret(self.pmf, [100, 300], [75, 125])
        self.assertTrue(set(np.unique(idx)) <= {policy.IDX_NONE, policy.IDX_NOTIFY, policy.IDX_EXPRESS})

    def test_scenario_matrix_shape(self):
        costs = policy.scenario_cost_matrix(self.pmf, [100, 200], [50, 100, 150])
        self.assertEqual(costs.shape, (6, 2, len(policy.ACTIONS)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
