"""Tests for the A/B analysis module and its simulator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import experiment  # noqa: E402


class TestWelchReduction(unittest.TestCase):
    def test_recovers_a_known_reduction(self):
        rng = np.random.default_rng(0)
        control = rng.normal(2.0, 1.5, 2000)
        treatment = rng.normal(1.0, 1.5, 2000)
        estimate = experiment.welch_reduction(control, treatment)
        self.assertAlmostEqual(estimate.reduction, 1.0, delta=0.15)
        self.assertLess(estimate.ci_low, estimate.reduction)
        self.assertGreater(estimate.ci_high, estimate.reduction)
        self.assertLess(estimate.p_value, 1e-6)

    def test_no_effect_is_not_significant(self):
        rng = np.random.default_rng(1)
        estimate = experiment.welch_reduction(rng.normal(1, 1, 1000), rng.normal(1, 1, 1000))
        self.assertGreater(estimate.p_value, 0.01)
        self.assertLess(estimate.ci_low, 0)
        self.assertGreater(estimate.ci_high, 0)

    def test_constant_arms_do_not_crash(self):
        estimate = experiment.welch_reduction(np.zeros(10), np.ones(10))
        self.assertAlmostEqual(estimate.reduction, -1.0)
        self.assertEqual(estimate.ci_low, estimate.ci_high)

    def test_needs_two_observations_per_arm(self):
        with self.assertRaises(ValueError):
            experiment.welch_reduction([1.0], [2.0, 3.0])


class TestSrm(unittest.TestCase):
    def test_balanced_randomisation_passes(self):
        result = experiment.srm_test(1000, 1010)
        self.assertTrue(result["ok"])
        self.assertGreater(result["p_value"], 0.05)

    def test_broken_randomisation_is_flagged(self):
        result = experiment.srm_test(1300, 700)
        self.assertFalse(result["ok"])
        self.assertLess(result["p_value"], 1e-6)


class TestDecision(unittest.TestCase):
    def test_confidence_interval_above_break_even_rolls_out(self):
        self.assertEqual(experiment.rollout_decision(1.2, 1.6, 1.0), "roll out")

    def test_confidence_interval_below_break_even_stops(self):
        self.assertEqual(experiment.rollout_decision(0.1, 0.4, 0.5), "do not roll out")

    def test_straddling_interval_is_inconclusive(self):
        self.assertEqual(experiment.rollout_decision(0.3, 0.8, 0.5), "inconclusive")

    def test_never_paying_intervention(self):
        self.assertEqual(experiment.rollout_decision(0.1, 0.2, float("nan")), "never pays")

    def test_net_value(self):
        self.assertAlmostEqual(experiment.net_value(2.0, 200, 100), 300.0)
        self.assertAlmostEqual(experiment.net_value(0.1, 200, 100), -80.0)


class TestSimulation(unittest.TestCase):
    def test_simulation_recovers_the_effect(self):
        rng = np.random.default_rng(2)
        pools = {
            "Standard Class": rng.normal(0.0, 1.5, 5000),
            "Second Class": rng.normal(2.0, 1.5, 5000),
        }
        frame = experiment.simulate_ab(
            pools, {"Standard Class": 1.5, "Second Class": 2.0}, n_per_arm=600, seed=3, noise=0.3
        )
        self.assertEqual(len(frame), 4 * 600)
        self.assertIn("true_effect", frame.columns)
        by_mode = experiment.analyze_by_group(frame)
        standard = by_mode.loc[by_mode["shipping_mode"] == "Standard Class"].iloc[0]
        self.assertAlmostEqual(standard["reduction"], 1.5, delta=0.25)

    def test_confidence_intervals_have_nominal_coverage(self):
        rng = np.random.default_rng(4)
        pool = rng.normal(0.0, 1.5, 4000)
        covered = 0
        reps = 200
        for rep in range(reps):
            frame = experiment.simulate_ab(
                {"Standard Class": pool}, {"Standard Class": 1.0}, n_per_arm=200, seed=rep, noise=0.5
            )
            estimate = experiment.welch_reduction(
                frame.loc[frame["treated"] == 0, "delay_days"],
                frame.loc[frame["treated"] == 1, "delay_days"],
            )
            covered += int(estimate.ci_low <= 1.0 <= estimate.ci_high)
        self.assertGreater(covered / reps, 0.88)
        self.assertLessEqual(covered / reps, 1.0)

    def test_group_analysis_skips_single_arm_segments(self):
        frame = pd.DataFrame(
            {
                "shipping_mode": ["A"] * 4 + ["B"] * 4,
                "delay_days": [1, 2, 3, 4, 1, 1, 1, 1],
                "treated": [0, 0, 1, 1, 0, 0, 0, 0],
            }
        )
        by_mode = experiment.analyze_by_group(frame)
        self.assertEqual(set(by_mode["shipping_mode"]), {"A"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
