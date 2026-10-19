"""Tests for the PSI-based drift report."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

import make_synthetic_dataco as fixture  # noqa: E402

from drift_report import build_report, categorical_psi, numeric_psi, psi  # noqa: E402
from src import pipeline  # noqa: E402


class TestPsi(unittest.TestCase):
    def test_identical_distributions_are_zero(self):
        shares = np.array([0.2, 0.3, 0.5])
        self.assertAlmostEqual(psi(shares, shares), 0.0, places=12)

    def test_a_large_shift_is_flagged(self):
        reference = np.array([0.5, 0.5])
        current = np.array([1.0, 0.0])
        self.assertGreater(psi(reference, current), 0.25)

    def test_categorical_psi_aligns_categories(self):
        reference = pd.Series(["A", "A", "B", "C"])
        same = pd.Series(["C", "B", "A", "A"])
        shifted = pd.Series(["A", "A", "A", "A"])
        self.assertAlmostEqual(categorical_psi(reference, same), 0.0, places=12)
        self.assertGreater(categorical_psi(reference, shifted), 0.25)

    def test_numeric_psi_is_zero_for_identical_samples(self):
        reference = pd.Series(np.linspace(-2, 2, 500))
        self.assertAlmostEqual(numeric_psi(reference, reference.copy()), 0.0, places=12)

    def test_numeric_psi_handles_constant_reference(self):
        reference = pd.Series([5.0] * 100)
        current = pd.Series([5.0] * 50)
        # the smoothing epsilon contributes ~1e-8 when sample sizes differ
        self.assertLess(numeric_psi(reference, current), 1e-6)

    def test_numeric_psi_detects_a_shift(self):
        reference = pd.Series(np.zeros(500))
        current = pd.Series(np.ones(500))
        self.assertGreater(numeric_psi(reference, current), 0.25)


class TestDriftReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = fixture.generate_frame(n=900, seed=5)
        frame.to_csv("/tmp/_dataco_drift.csv", index=False, encoding="latin1")
        cls.df = pipeline.load_orders("/tmp/_dataco_drift.csv")
        cls.report, cls.summary = build_report(cls.df, reference_frac=0.6)

    def test_report_has_one_row_per_feature_and_month(self):
        expected = self.df[pipeline.DATE_COL].dt.to_period("M").nunique() * 7
        self.assertEqual(len(self.report), expected)
        self.assertIn("delay_days", set(self.report["feature"]))

    def test_summary_reports_psi_and_a_verdict(self):
        self.assertIn("delay_days", self.summary["max_psi_after_reference"])
        self.assertIn(self.summary["verdict"]["delay_days"], {"stable", "moderate", "high"})
        self.assertGreater(self.summary["reference_rows"], 0)

    def test_psi_is_non_negative(self):
        self.assertTrue((self.report["psi"].dropna() >= -1e-12).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
