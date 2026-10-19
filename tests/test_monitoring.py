"""Tests for PSI monitoring and the retraining recommendation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import monitoring  # noqa: E402


class TestAlertLevels(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(monitoring.alert_level(0.05), "stable")
        self.assertEqual(monitoring.alert_level(0.15), "moderate")
        self.assertEqual(monitoring.alert_level(0.30), "high")
        self.assertEqual(monitoring.alert_level(float("nan")), "unknown")


class TestRecommendation(unittest.TestCase):
    def test_target_drift_means_retrain(self):
        decision = monitoring.recommend(max_feature_psi=0.05, target_psi=0.30)
        self.assertEqual(decision["action"], "retrain")
        self.assertEqual(decision["exit_code"], 2)

    def test_feature_only_drift_means_reweight(self):
        decision = monitoring.recommend(max_feature_psi=0.30, target_psi=0.02)
        self.assertEqual(decision["action"], "reweight")
        self.assertEqual(decision["exit_code"], 1)

    def test_moderate_drift_is_monitored(self):
        self.assertEqual(monitoring.recommend(0.15, 0.15)["action"], "monitor")

    def test_no_drift_is_stable(self):
        self.assertEqual(monitoring.recommend(0.03, 0.02)["action"], "stable")


class TestMonitorFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n = 400
        base = {
            "Shipping Mode": rng.choice(["Standard Class", "Second Class"], n),
            "Market": rng.choice(["Europe", "LATAM"], n),
            "Category Name": rng.choice(["Fishing", "Cleats"], n),
            "Customer Segment": rng.choice(["Consumer", "Corporate"], n),
            "Type": rng.choice(["DEBIT", "TRANSFER"], n),
            "Order Item Total": rng.uniform(10, 500, n),
            "delay_days": rng.integers(-2, 5, n),
        }
        cls.reference = pd.DataFrame(base)
        # current batch: the same delay behaviour, a very different market mix
        shifted = dict(base)
        shifted["Market"] = np.array(["Africa"] * n)
        cls.current = pd.DataFrame(shifted)

    def test_reports_psi_and_a_recommendation(self):
        result = monitoring.monitor_frames(self.reference, self.current)
        self.assertIn("Market", result["features"])
        self.assertEqual(result["alerts"]["Market"], "high")
        self.assertEqual(result["recommendation"], "reweight")
        self.assertEqual(result["exit_code"], 1)

    def test_missing_target_is_tolerated(self):
        current = self.current.drop(columns=["delay_days"])
        result = monitoring.monitor_frames(self.reference, current)
        self.assertIsNone(result["target_psi"])
        self.assertIn(result["recommendation"], {"stable", "monitor", "reweight", "retrain"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
