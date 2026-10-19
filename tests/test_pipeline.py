"""Pipeline tests: target definition, leakage, chronological split, encoders.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

import make_synthetic_dataco as fixture  # noqa: E402

from src import pipeline  # noqa: E402

POST_HOC_COLUMNS = {
    "Days for shipping (real)",
    "Delivery Status",
    "Late_delivery_risk",
    "delay_days",
}
FIXTURE_PATH = Path("/tmp/_dataco_test.csv")


def setUpModule() -> None:
    fixture.generate_frame(n=600, seed=7).to_csv(FIXTURE_PATH, index=False, encoding="latin1")


class TestTargetAndLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pipeline.load_orders(str(FIXTURE_PATH))
        cls.X, cls.y = pipeline.build_features(cls.df)

    def test_delay_is_actual_minus_scheduled(self):
        expected = (
            self.df[pipeline.SHIP_COL] - self.df[pipeline.DATE_COL]
        ).dt.days - self.df[pipeline.SCHED_COL]
        np.testing.assert_allclose(self.y, expected.to_numpy())

    def test_features_are_18_columns(self):
        self.assertEqual(len(self.X.columns), 18)

    def test_no_post_hoc_column_leaks_into_features(self):
        for col in POST_HOC_COLUMNS:
            self.assertNotIn(col, self.X.columns)

    def test_target_atoms_are_integers(self):
        self.assertTrue(np.all(self.y == np.round(self.y)))
        self.assertGreaterEqual(self.y.min(), -2)
        self.assertLessEqual(self.y.max(), 4)


class TestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pipeline.load_orders(str(FIXTURE_PATH))
        cls.X, cls.y = pipeline.build_features(cls.df)
        cls.split = pipeline.time_split(cls.X, cls.y)

    def test_split_sizes_are_70_15_15(self):
        n = len(self.X)
        self.assertEqual(len(self.split.y_train), int(n * 0.70))
        self.assertEqual(len(self.split.y_valid), int(n * 0.85) - int(n * 0.70))
        self.assertEqual(len(self.split.y_test), n - int(n * 0.85))

    def test_split_is_chronological(self):
        dates = self.df[pipeline.DATE_COL]
        n_train = len(self.split.y_train)
        self.assertLessEqual(dates.iloc[:n_train].max(), dates.iloc[n_train])
        self.assertEqual(dates.iloc[self.split.test_index[0]], dates.iloc[int(len(self.X) * 0.85)])

    def test_scaler_is_fitted_on_train_only(self):
        train_scaled = self.split.scaler.transform(self.X.iloc[self.split.train_index])
        self.assertTrue(np.allclose(train_scaled.mean(axis=0), 0, atol=1e-8))


class TestEncoders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pipeline.load_orders(str(FIXTURE_PATH))
        cls.encoders = pipeline.fit_encoders(cls.df)

    def test_unseen_category_maps_to_minus_one(self):
        row = self.df.iloc[[0]].copy()
        row["Market"] = "Atlantis"
        features = pipeline.make_features(row, self.encoders)
        self.assertEqual(features["Market_enc"].iloc[0], -1)

    def test_make_features_agrees_with_build_features(self):
        features = pipeline.make_features(self.df, self.encoders)
        built, _ = pipeline.build_features(self.df)
        pd.testing.assert_frame_equal(features, built)


if __name__ == "__main__":
    unittest.main(verbosity=2)
