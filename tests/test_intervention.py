"""Tests for the intervention-effect analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from intervention_analysis import (  # noqa: E402
    break_even_reduction,
    misspecification_matrix,
    required_sample_size,
)
from src import policy  # noqa: E402


def pmf_for(atom_values):
    pmf = np.zeros(len(policy.ATOMS))
    for value in atom_values:
        pmf[int(np.where(policy.ATOMS == value)[0][0])] = 1 / len(atom_values)
    return pmf.reshape(1, -1)


FIRST = pmf_for([1])
STANDARD = pmf_for([-2, -1, 0, 1, 2])
SAME_DAY = pmf_for([0])


class TestBreakEven(unittest.TestCase):
    def test_first_class_breaks_even_at_half_a_day(self):
        # express costs $100; the penalty is $200/day and the order is 1 day late
        value = break_even_reduction(FIRST, penalty=200, express_cost=100)
        self.assertAlmostEqual(value, 0.5, places=2)

    def test_same_day_never_breaks_even(self):
        self.assertTrue(np.isnan(break_even_reduction(SAME_DAY, penalty=200, express_cost=100)))

    def test_higher_express_cost_needs_a_bigger_effect(self):
        cheap = break_even_reduction(STANDARD, penalty=200, express_cost=50)
        pricey = break_even_reduction(STANDARD, penalty=200, express_cost=100)
        self.assertLess(cheap, pricey)
        # at $150 the maximum possible benefit ($120) cannot cover the action
        self.assertTrue(np.isnan(break_even_reduction(STANDARD, penalty=200, express_cost=150)))


class TestSampleSize(unittest.TestCase):
    def test_smaller_effects_need_more_orders(self):
        self.assertGreater(required_sample_size(1.5, 0.1), required_sample_size(1.5, 0.5))

    def test_more_noise_needs_more_orders(self):
        self.assertGreater(required_sample_size(3.0, 0.2), required_sample_size(1.5, 0.2))

    def test_zero_effect_is_undefined(self):
        self.assertEqual(required_sample_size(1.5, 0.0), 0)


class TestMisspecification(unittest.TestCase):
    def test_overestimating_the_effect_is_not_harmful_here(self):
        pmf = np.vstack([FIRST] * 3)
        delay = np.array([1.0, 1.0, 1.0])
        table = misspecification_matrix(pmf, delay, assumed_reductions=[0, 2, 4], true_reduction=2)
        by_assumption = table.set_index("assumed_reduction")["total_cost"]
        self.assertLess(by_assumption[2], by_assumption[0])
        self.assertAlmostEqual(by_assumption[2], by_assumption[4])

    def test_output_shape_and_columns(self):
        pmf = np.vstack([STANDARD] * 5)
        delay = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        table = misspecification_matrix(pmf, delay, assumed_reductions=[0, 1, 2], true_reduction=2)
        self.assertEqual(len(table), 3)
        self.assertIn("intervention_rate", table.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
