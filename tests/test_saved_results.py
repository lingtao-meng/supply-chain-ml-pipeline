"""Reject inconsistent saved costs before publishing report summaries."""

import csv
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("saved_results", ROOT / "scripts/verify_results.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class SavedResultsTests(unittest.TestCase):
    def test_saved_tables_reconcile(self):
        verify.build(ROOT)

    def test_changed_per_order_cost_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "results/diagnostics.json", "results/policy_comparison.csv",
                "results/mode_table.csv", "results_asof/diagnostics.json", *verify.PACING_FILES,
            ):
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            path = root / "results/policy_comparison.csv"
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["cost_per_order"] = str(float(rows[0]["cost_per_order"]) + 1)
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "cost per order"):
                verify.build(root)

    def test_pacing_budget_balance_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in verify.PACING_FILES:
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            path = root / "results/pacing_by_period.csv"
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["closing_balance"] = str(float(rows[0]["closing_balance"]) + 100)
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "closing balance"):
                verify.pacing_report(root, 65752, 9863)

    def test_nonfinite_cost_is_rejected(self):
        with self.assertRaises(ValueError):
            verify.close(float("nan"), 1.0, "cost")


if __name__ == "__main__":
    unittest.main()
