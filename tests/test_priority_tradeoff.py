import unittest
from fractions import Fraction

from scripts.priority_tradeoff import optimal_ranges


class PriorityTradeoffTest(unittest.TestCase):
    def test_dominated_and_tied_policies(self):
        options = {"a": (10, 3), "b": (11, 3), "c": (15, 0), "d": (10, 3)}
        self.assertEqual(list(optimal_ranges(options)), [
            ("a", 0, Fraction(5, 3)), ("c", Fraction(5, 3), None),
            ("d", 0, Fraction(5, 3))])

    def test_intermediate_policy_has_two_switch_points(self):
        options = {"budget": (10, 5), "balanced": (12, 3), "coverage": (20, 1)}
        self.assertEqual(list(optimal_ranges(options)), [
            ("budget", 0, 1), ("balanced", 1, 4), ("coverage", 4, None)])
