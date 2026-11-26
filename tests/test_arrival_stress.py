import itertools
import unittest

import numpy as np

from scripts.arrival_stress import exact_allocation, express_break_even


class ArrivalStressTest(unittest.TestCase):
    def test_grouped_integer_optimum_matches_exhaustive_allocation(self):
        expected = np.array([[100, 76, 20], [100, 76, 20], [72, 70, 15], [32, 25, 50]])
        fees = np.array([0, 20, 100])
        orders = np.arange(len(expected))
        for budget in (0, 20, 99, 120, 200):
            chosen = exact_allocation(expected, fees, budget)
            best = min(
                expected[orders, strategy].sum()
                for strategy in itertools.product(range(len(fees)), repeat=len(orders))
                if fees[list(strategy)].sum() <= budget
            )
            self.assertEqual(expected[orders, chosen].sum(), best)

    def test_break_even_checks_fee_against_delay_savings(self):
        threshold, status = express_break_even(np.array([1, 2]), np.array([1, 2]), np.array([0, 20, 100]),
                                               penalty=100, notify_remaining=.6)
        self.assertAlmostEqual(threshold, .8)
        self.assertEqual(status, "threshold")
        threshold, status = express_break_even(np.array([1]), np.array([2]), np.array([0, 20, 100]),
                                               penalty=50, notify_remaining=1)
        self.assertIsNone(threshold)
        self.assertEqual(status, "unattainable")


if __name__ == "__main__":
    unittest.main()
