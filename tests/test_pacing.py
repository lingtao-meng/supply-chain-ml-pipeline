"""Tests for the multi-period budget pacing simulation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

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
        cls.delay = np.array([-2, -1, 0, 1, 2] * 6, dtype=float)
        cls.months = np.array(["2024-01"] * 12 + ["2024-02"] * 12)
        cls.table = simulate(
            cls.pmf, cls.delay, cls.months, [100, 1_000], penalty=200, actions=policy.ACTIONS
        )

    def test_one_row_per_strategy_plus_the_optimum(self):
        self.assertEqual(len(self.table), 1 + 3 * 2)

    def test_spend_never_exceeds_the_budget(self):
        finite = self.table[np.isfinite(self.table["total_budget"])]
        self.assertTrue((finite["action_spend"] <= finite["total_budget"] + 1e-6).all())

    def test_global_pacing_is_not_worse_than_monthly_in_expectation(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
