"""End-to-end tests for the batch scorer.

Skipped when the model bundle has not been built (``make reproduce``).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

import make_synthetic_dataco as fixture  # noqa: E402

from score_orders import load_bundle, score_frame  # noqa: E402

BUNDLE_PATH = ROOT / "models" / "model_bundle.joblib"
ACTION_COST = {"none": 0.0, "notify": 20.0, "express": 100.0}


@unittest.skipUnless(BUNDLE_PATH.exists(), "model bundle not built (run `make reproduce`)")
class TestScoreOrders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(BUNDLE_PATH)
        cls.frame = fixture.generate_frame(n=40, seed=11)

    def test_output_has_predictions_and_a_decision(self):
        plan = score_frame(self.frame, self.bundle)
        for col in (
            "p_late",
            "p10_delay",
            "p50_delay",
            "p90_delay",
            "recommended_action",
            "expected_saving",
            "expected_cost_none",
            "expected_cost_notify",
            "expected_cost_express",
        ):
            self.assertIn(col, plan.columns)
        self.assertEqual(len(plan), len(self.frame))
        self.assertTrue(plan["recommended_action"].isin(ACTION_COST).all())

    def test_budget_is_respected(self):
        plan = score_frame(self.frame, self.bundle, budget=200)
        spend = plan["recommended_action"].map(ACTION_COST).sum()
        self.assertLessEqual(spend, 200 + 1e-9)
        self.assertGreater(spend, 0)

    def test_higher_penalty_intervenes_more(self):
        cheap = score_frame(self.frame, self.bundle, penalty=10)
        expensive = score_frame(self.frame, self.bundle, penalty=1_000)
        touched_cheap = int((cheap["recommended_action"] != "none").sum())
        touched_expensive = int((expensive["recommended_action"] != "none").sum())
        self.assertGreater(touched_expensive, touched_cheap)


if __name__ == "__main__":
    unittest.main(verbosity=2)
