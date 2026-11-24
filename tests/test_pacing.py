"""Tests for the multi-period budget pacing simulation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pacing_simulation import simulate  # noqa: E402
from src import policy  # noqa: E402


def pmf_for(atom_values):
    pmf = np.zeros(len(policy.ATOMS))
    for value in atom_values:
        pmf[int(np.where(policy.ATOMS == value)[0][0])] = 1 / len(atom_values)
    return pmf


class TestPacing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        standard = pmf_for([-2, -1, 0, 1, 2])
        second = pmf_for([0, 1, 2, 3, 4])
        cls.pmf = np.vstack([standard] * 12 + [second] * 12)
        cls.delay = np.array(([-2, -1, 0, 1, 2] * 5)[:24], dtype=float)
        cls.dates = pd.to_datetime(["2024-01-01"] * 6 + ["2024-01-10"] * 6 + ["2024-02-10"] * 12)
        cls.table, cls.periods = simulate(
            cls.pmf, cls.delay, cls.dates, [100, 1_000], penalty=200, actions=policy.ACTIONS
        )

    def test_one_row_per_strategy_plus_the_optimum(self):
        self.assertEqual(len(self.table), 1 + 4 * 2)

    def test_spend_never_exceeds_the_budget(self):
        finite = self.table[np.isfinite(self.table["total_budget"])]
        self.assertTrue((finite["action_spend"] <= finite["total_budget"] + 1e-6).all())

    def test_global_greedy_cost_on_handcrafted_batches(self):
        finite = self.table[np.isfinite(self.table["total_budget"])]
        for budget, group in finite.groupby("total_budget"):
            global_cost = group.loc[group["strategy"] == "global", "expected_total_cost"].iloc[0]
            monthly_cost = group.loc[
                group["strategy"] == "monthly (no carry-over)", "expected_total_cost"
            ].iloc[0]
            self.assertLessEqual(global_cost, monthly_cost + 1e-6, msg=f"budget={budget}")

    def test_unconstrained_row_has_no_budget(self):
        row = self.table.iloc[0]
        self.assertTrue(np.isinf(row["total_budget"]))
        self.assertEqual(row["unspent"], 0.0)


    def test_weekly_release_splits_at_month_boundary_and_keeps_empty_calendar_weeks(self):
        pmf = np.vstack([pmf_for([3])] * 2)
        dates = pd.to_datetime(["2024-01-31", "2024-02-10"])
        _, periods = simulate(pmf, np.array([3, 3]), dates, [200], 200, policy.ACTIONS)
        weekly = periods[periods["strategy"] == "weekly + carry-over"]
        self.assertEqual(list(weekly["period_start"]), ["2024-01-31", "2024-02-01", "2024-02-05"])
        self.assertEqual(list(weekly["period_end"]), ["2024-01-31", "2024-02-04", "2024-02-10"])
        self.assertEqual(list(weekly["test_orders"]), [1, 0, 1])
        np.testing.assert_allclose(weekly["budget_released"], [100, 40, 60])
        np.testing.assert_allclose(weekly["opening_balance"], [0, 0, 40])
        np.testing.assert_allclose(weekly["action_spend"], [100, 0, 100])

    def test_pacing_does_not_borrow_from_future_periods(self):
        for (_, strategy), group in self.periods.groupby(["total_budget", "strategy"], sort=False):
            allowed = group["opening_balance"] + group["budget_released"]
            self.assertTrue((group["action_spend"] <= allowed + 1e-7).all(), msg=strategy)
            if strategy.endswith("+ carry-over"):
                np.testing.assert_allclose(group["opening_balance"].to_numpy()[1:],
                                           group["closing_balance"].to_numpy()[:-1])
            else:
                self.assertTrue((group["opening_balance"] == 0).all(), msg=strategy)

    def test_future_week_scores_and_outcomes_do_not_change_earlier_week_actions(self):
        pmf = np.vstack([pmf_for([4])] * 2)
        dates = pd.to_datetime(["2024-01-01", "2024-01-08"])
        _, baseline = simulate(pmf, np.array([4, 4]), dates, [200], 200, policy.ACTIONS)
        pmf[1] = pmf_for([-2])
        _, changed = simulate(pmf, np.array([-2, -2]), dates, [200], 200, policy.ACTIONS)
        first = (baseline["strategy"] == "weekly + carry-over") & (baseline["period_start"] == "2024-01-01")
        np.testing.assert_array_equal(
            baseline.loc[first, ["orders_notify", "orders_express", "action_spend"]],
            changed.loc[first, ["orders_notify", "orders_express", "action_spend"]],
        )

    def test_no_late_orders_means_recall_is_unavailable(self):
        pmf = np.vstack([pmf_for([-2])] * 2)
        dates = pd.to_datetime(["2024-01-01", "2024-01-08"])
        table, _ = simulate(pmf, np.array([-2, -2]), dates, [100], 200, policy.ACTIONS)
        self.assertTrue(table["recall_late"].isna().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
