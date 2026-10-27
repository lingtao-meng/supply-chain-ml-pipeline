"""Regression tests for grouped A/B net value and SRM gating."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ab_analysis", ROOT / "scripts" / "ab_analysis.py")
ab = importlib.util.module_from_spec(spec)
sys.modules["ab_analysis"] = ab
spec.loader.exec_module(ab)


class TestGroupedNetValue(unittest.TestCase):
    def test_each_mode_uses_its_own_control(self):
        # First Class: control delay 1 -> treatment 0, worth 200-100=100/order.
        # Same Day: control delay 0 -> treatment -0.05, worth -100/order.
        frame = pd.DataFrame(
            {
                "shipping_mode": ["First Class"] * 200 + ["Same Day"] * 200,
                "delay_days": [1.0] * 100 + [0.0] * 100 + [0.0] * 100 + [-0.05] * 100,
                "treated": [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100,
            }
        )
        _, by_mode = ab.summarise(
            frame,
            break_even={"First Class": 0.5, "Same Day": float("nan")},
            penalty=200.0,
            express_cost=100.0,
            simulated=False,
        )
        first = by_mode.loc[by_mode["shipping_mode"] == "First Class"].iloc[0]
        same = by_mode.loc[by_mode["shipping_mode"] == "Same Day"].iloc[0]
        self.assertAlmostEqual(first["net_value_per_order"], 100.0, delta=5.0)
        self.assertAlmostEqual(same["net_value_per_order"], -100.0, delta=5.0)


class TestSrmGate(unittest.TestCase):
    def test_srm_failure_blocks_rollout(self):
        frame = pd.DataFrame(
            {
                "shipping_mode": ["First Class"] * 600,
                "delay_days": [1.0] * 500 + [0.0] * 100,
                "treated": [0] * 500 + [1] * 100,
            }
        )
        summary, by_mode = ab.summarise(
            frame,
            break_even={"First Class": 0.5},
            penalty=200.0,
            express_cost=100.0,
            simulated=False,
        )
        self.assertFalse(summary["srm"]["ok"])
        self.assertTrue((by_mode["decision"] == "hold (SRM anomaly)").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestNoModeFallback(unittest.TestCase):
    def test_no_shipping_mode_uses_all_control_delays(self):
        # control delay 0.6, effect 1.0, penalty 200, action 100 -> 20/order
        frame = pd.DataFrame(
            {
                "delay_days": [0.6] * 100 + [-0.4] * 100,
                "treated": [0] * 100 + [1] * 100,
            }
        )
        _, by = ab.summarise(frame, break_even={}, penalty=200.0,
                             express_cost=100.0, simulated=False)
        self.assertAlmostEqual(float(by.iloc[0]["net_value_per_order"]), 20.0, delta=2.0)
        self.assertEqual(by.iloc[0]["decision"], "not evaluated (no break-even)")
