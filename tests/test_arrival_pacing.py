"""Checks for chronological decisions, information access, and saved cash ledgers."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.arrival_pacing import decide_on_arrival  # noqa: E402


class TestArrivalPacing(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2024-01-01")
        self.end = pd.Timestamp("2024-01-04")
        self.dates = pd.date_range(self.start, self.end)
        self.fees = np.array([0.0, 20.0, 100.0])

    def test_daily_limit_delays_spend_without_using_future_arrival_counts(self):
        costs = np.tile([800.0, 500.0, 490.0], (4, 1))
        full = decide_on_arrival(costs, self.dates, self.fees, 100, self.start, self.end, daily_release=False)
        paced = decide_on_arrival(costs, self.dates, self.fees, 100, self.start, self.end, daily_release=True)
        self.assertEqual(full["action"].tolist(), [2, 0, 0, 0])
        self.assertEqual(paced["action"].tolist(), [1, 1, 1, 1])
        self.assertEqual(full["limit_reason"].tolist(), ["", "full_budget", "full_budget", "full_budget"])
        self.assertEqual(paced["limit_reason"].tolist(), ["daily_release", "full_budget", "full_budget", "full_budget"])
        self.assertTrue((paced["spent_to_date"] <= paced["released_to_date"]).all())

    def test_changing_future_forecasts_leaves_prefix_unchanged(self):
        earlier_costs = np.array([[125.0, 90.0, 120.0], [240.0, 172.0, 100.0]])
        future_costs = np.array([[800.0, 500.0, 490.0], [0.0, 20.0, 100.0]])
        for daily in (False, True):
            prefix = decide_on_arrival(
                earlier_costs, self.dates[:2], self.fees, 100, self.start, self.end, daily_release=daily
            )
            expanded = decide_on_arrival(
                np.vstack([earlier_costs, future_costs]), self.dates, self.fees, 100, self.start, self.end,
                daily_release=daily,
            )
            pd.testing.assert_frame_equal(prefix, expanded.iloc[:2])

    def test_historical_gate_preserves_budget_for_later_higher_gain(self):
        costs = np.array([[100.0, 85.0, 120.0], [300.0, 230.0, 150.0]])
        baseline = decide_on_arrival(costs, self.dates[:2], self.fees, 100, self.start, self.end, daily_release=False)
        gated = decide_on_arrival(
            costs, self.dates[:2], self.fees, 100, self.start, self.end,
            daily_release=False, min_expected_gain=100.0,
        )
        self.assertEqual(baseline["action"].tolist(), [1, 1])
        self.assertEqual(gated["action"].tolist(), [0, 2])
        self.assertEqual(gated["limit_reason"].tolist(), ["priority_gate", ""])

    def test_equal_expected_cost_prefers_lower_action_spend(self):
        costs = np.array([[70.0, 70.0, 100.0], [300.0, 110.0, 110.0]])
        result = decide_on_arrival(costs, self.dates[:2], self.fees, 200, self.start, self.end, daily_release=False)
        self.assertEqual(result["action"].tolist(), [0, 1])

    def test_rejects_unsorted_order_stream(self):
        costs = np.array([[70.0, 70.0, 100.0], [300.0, 110.0, 110.0]])
        with self.assertRaisesRegex(ValueError, "chronological"):
            decide_on_arrival(costs, self.dates[:2][::-1], self.fees, 200, self.start, self.end, daily_release=False)

    def test_published_cash_ledgers_reconcile_with_budget_and_batch_comparator(self):
        results = ROOT / "results"
        summary = pd.read_csv(results / "arrival_pacing.csv").set_index(["total_budget", "strategy"])
        metadata = json.loads((results / "arrival_pacing_metadata.json").read_text(encoding="utf-8"))
        original = json.loads((results / "pacing_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["source_sha256"], original["source_sha256"])
        self.assertEqual(metadata["test_orders"], original["test_orders"])
        batch = pd.read_csv(results / "pacing.csv").query("strategy == 'global'").set_index("total_budget")
        reference = summary.xs("full-horizon greedy", level="strategy")
        np.testing.assert_allclose(reference["expected_cost"], batch["expected_total_cost"])
        np.testing.assert_allclose(reference["cost"], batch["total_cost"])

        daily = pd.read_csv(results / "arrival_pacing_by_day.csv")
        for keys, days in daily.groupby(["total_budget", "strategy"]):
            record = summary.loc[keys]
            cumulative = days["action_spend"].cumsum()
            np.testing.assert_allclose(cumulative, days["spent_to_date"])
            self.assertTrue((cumulative <= days["released_to_date"] + 1e-7).all())
            self.assertLessEqual(cumulative.iloc[-1], keys[0])
            self.assertAlmostEqual(float(days["action_spend"].sum()), record["action_spend"])
            self.assertAlmostEqual(float(days["cost"].sum()), record["cost"])
            self.assertAlmostEqual(float(days["priority_full_budget"].sum()), record["priority_full_budget"])
            self.assertAlmostEqual(float(days["priority_daily_release"].sum()), record["priority_daily_release"])
            self.assertEqual(int(days["test_orders"].sum()), metadata["test_orders"])


if __name__ == "__main__":
    unittest.main()
