"""HTTP API tests (skipped when the bundle or FastAPI is unavailable)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # the API extra is optional for the core test suite
    from fastapi.testclient import TestClient

    from api.main import app

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FASTAPI_AVAILABLE = False

BUNDLE_PATH = ROOT / "models" / "model_bundle.joblib"

SAMPLES = [
    {
        "type": "DEBIT",
        "category_name": "Fishing",
        "department_name": "Fan Shop",
        "market": "Europe",
        "shipping_mode": "Same Day",
        "customer_segment": "Consumer",
        "order_date": "2017-12-15",
        "scheduled_days": 0,
        "quantity": 2,
        "unit_price": 120.0,
        "discount_rate": 0.05,
    },
    {
        "type": "TRANSFER",
        "category_name": "Fishing",
        "department_name": "Fan Shop",
        "market": "Europe",
        "shipping_mode": "Second Class",
        "customer_segment": "Corporate",
        "order_date": "2017-12-15",
        "scheduled_days": 2,
        "quantity": 3,
        "unit_price": 80.0,
        "discount_rate": 0.1,
    },
]


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed (requirements-service.txt)")
@unittest.skipUnless(BUNDLE_PATH.exists(), "model bundle not built (run `make reproduce`)")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_reports_a_loaded_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_loaded"])

    def test_model_info_exposes_metrics(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("binary_auc", body["metrics"])
        self.assertIn("penalty", body["params"])

    def test_score_returns_one_prediction_per_order(self):
        response = self.client.post("/score", json={"orders": SAMPLES})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(len(body["orders"]), 2)
        self.assertEqual(body["summary"]["orders"], 2)
        actions = {row["recommended_action"] for row in body["orders"]}
        self.assertTrue(actions <= {"none", "notify", "express"})

    def test_zero_budget_means_no_action(self):
        response = self.client.post("/score", json={"orders": SAMPLES, "budget": 0})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(all(row["recommended_action"] == "none" for row in response.json()["orders"]))

    def test_invalid_order_is_rejected(self):
        response = self.client.post("/score", json={"orders": [{"type": "DEBIT"}]})
        self.assertEqual(response.status_code, 422)

    def test_empty_order_list_is_rejected(self):
        response = self.client.post("/score", json={"orders": []})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
