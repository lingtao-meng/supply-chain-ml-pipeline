"""Tests for the T-learner, Qini curve and cost-based targeting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import uplift  # noqa: E402


def make_experiment(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    treatment = rng.integers(0, 2, n)
    effect = 2.0 * (X[:, 0] > 0) + 1.0 * (X[:, 1] > 0)
    outcome = 1.0 - treatment * effect + rng.normal(0, 0.3, n)
    return X, treatment, outcome, effect


class TestTLearner(unittest.TestCase):
    def test_recovers_the_heterogeneous_effect(self):
        X, treatment, outcome, effect = make_experiment()
        model = uplift.fit_t_learner(X, treatment, outcome)
        cate = model.predict_cate(X)
        self.assertGreater(np.corrcoef(cate, effect)[0, 1], 0.8)
        self.assertAlmostEqual(float(cate.mean()), float(effect.mean()), delta=0.2)


class TestQini(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.effect = rng.uniform(0, 3, 1000)

    def test_endpoints_are_zero(self):
        curve = uplift.qini_curve(self.effect, self.effect)
        self.assertAlmostEqual(curve["qini"].iloc[0], 0.0)
        self.assertAlmostEqual(curve["qini"].iloc[-1], 0.0)

    def test_perfect_ranking_has_auuc_one(self):
        curve = uplift.qini_curve(self.effect, self.effect)
        self.assertAlmostEqual(uplift.auuc(curve), 1.0, places=6)

    def test_random_ranking_is_near_zero(self):
        random_scores = np.random.default_rng(2).uniform(size=len(self.effect))
        curve = uplift.qini_curve(random_scores, self.effect)
        self.assertLess(abs(uplift.auuc(curve)), 0.15)


class TestTargeting(unittest.TestCase):
    def test_selects_the_best_orders_and_computes_net_saving(self):
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        effect = np.ones(10)
        curve = uplift.targeting_curve(
            scores, effect, penalty=200, action_cost=100, fractions=(0.2, 1.0)
        )
        top20 = curve.iloc[0]
        self.assertEqual(int(top20["orders_treated"]), 2)
        self.assertAlmostEqual(float(top20["net_saving"]), 2 * (200 - 100))
        self.assertAlmostEqual(float(curve.iloc[1]["net_saving"]), 10 * (200 - 100))

    def test_negative_net_value_is_reported(self):
        scores = np.ones(10)
        effect = np.full(10, 0.1)  # 200*0.1 = 20 < 100 action cost
        curve = uplift.targeting_curve(scores, effect, penalty=200, action_cost=100,
                                       fractions=(1.0,))
        self.assertLess(float(curve["net_saving"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
