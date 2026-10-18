"""Unit tests for the policy / cost modules — no data file needed.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import policy  # noqa: E402


class TestCostFunction(unittest.TestCase):
    def test_none_policy_costs_penalty_times_delay(self):
        delay = np.array([-2, -1, 0, 1, 3], dtype=float)
        cost = policy.realised_cost(policy.policy_none(5), delay, penalty=200)
        np.testing.assert_allclose(cost, 200 * np.clip(delay, 0, None))

    def test_actions_trade_fixed_cost_for_reduced_penalty(self):
        """加急适合短延迟（减 2 天就够），提前通知适合长延迟（按比例封顶罚金）。"""
        def cost(action, delay):
            return policy.realised_cost(np.array([action]), np.array([delay]), 200)[0]

        # 延迟 3 天：加急(减2天) 最省 -> 300 vs 通知 380 vs 不管 600
        self.assertLess(cost(policy.IDX_EXPRESS, 3), cost(policy.IDX_NOTIFY, 3))
        self.assertLess(cost(policy.IDX_NOTIFY, 3), cost(policy.IDX_NONE, 3))

        # 延迟 5 天：通知(罚金×0.6) 反而更省 -> 620 vs 加急 700 vs 不管 1000
        self.assertLess(cost(policy.IDX_NOTIFY, 5), cost(policy.IDX_EXPRESS, 5))
        self.assertLess(cost(policy.IDX_EXPRESS, 5), cost(policy.IDX_NONE, 5))

    def test_expected_cost_is_non_negative(self):
        draws = policy.quantile_grid(
            np.zeros(4), np.zeros(4), np.zeros(4)
        )
        exp = policy.expected_cost_per_action(draws, penalty=200)
        self.assertTrue((exp >= 0).all())

    def test_expected_cost_picks_none_when_no_delay_expected(self):
        draws = np.zeros((3, 11))  # 全部 0 延迟
        idx = policy.policy_expected_cost(draws, penalty=200)
        self.assertTrue((idx == policy.IDX_NONE).all())


class TestQuantileGrid(unittest.TestCase):
    def test_grid_is_monotone(self):
        rng = np.random.default_rng(0)
        p10 = rng.normal(0, 1, 50)
        p50 = p10 + 1
        p90 = p50 + 1
        draws = policy.quantile_grid(p10, p50, p90)
        self.assertTrue((np.diff(draws, axis=1) >= -1e-9).all())

    def test_crossing_quantiles_are_repaired(self):
        # p10 > p50 > p90 这种分位交叉要被强制单调
        draws = policy.quantile_grid(
            np.array([3.0]), np.array([1.0]), np.array([2.0])
        )
        self.assertTrue((np.diff(draws, axis=1) >= -1e-9).all())


class TestAllocationPolicies(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.p10 = np.full(200, 0.0)
        self.p50 = rng.normal(1.0, 0.5, 200)
        self.p90 = self.p50 + 2
        self.draws = policy.quantile_grid(self.p10, self.p50, self.p90)
        self.delay = rng.normal(0.5, 1.0, 200)

    def test_capacity_policy_respects_capacity(self):
        for cap in (0.0, 0.1, 0.5, 1.0):
            idx = policy.policy_capacity(self.draws, 200, cap)
            touched = (idx != policy.IDX_NONE).sum()
            self.assertLessEqual(touched, round(cap * len(self.delay)) + 1)

    def test_budget_policy_respects_budget(self):
        for budget in (0, 500, 5_000):
            idx = policy.policy_budget(self.draws, 200, budget)
            spend = sum(
                policy.ACTIONS[a].cost for a in idx if a != policy.IDX_NONE
            )
            self.assertLessEqual(spend, budget + 1e-9)

    def test_oracle_never_worse_than_any_policy(self):
        oracle = policy.realised_cost(
            policy.policy_oracle(self.delay, 200), self.delay, 200
        ).sum()
        for idx in (
            policy.policy_none(len(self.delay)),
            policy.policy_all_express(len(self.delay)),
            policy.policy_expected_cost(self.draws, 200),
        ):
            self.assertGreaterEqual(
                policy.realised_cost(idx, self.delay, 200).sum(), oracle - 1e-6
            )


class TestReporting(unittest.TestCase):
    def test_compare_table_has_one_row_per_policy(self):
        delay = np.array([-1.0, 0.0, 1.0, 2.0])
        policies = {
            "none": policy.policy_none(4),
            "all": policy.policy_all_express(4),
        }
        df = policy.compare(policies, delay, 200)
        self.assertEqual(len(df), 2)
        self.assertIn("cost_gap_vs_best", df.columns)
        self.assertAlmostEqual(df["cost_gap_vs_best"].min(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
