#!/usr/bin/env python3
"""Monthly drift report: is the *behaviour* drifting, or just the mix?

The notebook's adversarial validation gets AUC 1.0 on train-vs-test, which
looks alarming.  This script separates the two things that can cause it:

* **composition drift** — the share of markets / categories changes over time;
* **concept drift** — the relationship between features and the delay changes.

PSI (population stability index) is computed monthly against the training
window for a set of categorical and numeric features, including the target.
The usual reading: PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant.

Usage
-----
    python scripts/drift_report.py
    python scripts/drift_report.py --data /tmp/dataco_small.csv --outdir /tmp/drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import monitoring, pipeline  # noqa: E402

CATEGORICAL = monitoring.DEFAULT_CATEGORICAL
NUMERIC = monitoring.DEFAULT_NUMERIC
PSI_MODERATE, PSI_HIGH = monitoring.PSI_MODERATE, monitoring.PSI_HIGH
psi = monitoring.psi
categorical_psi = monitoring.categorical_psi
numeric_psi = monitoring.numeric_psi


def build_report(df: pd.DataFrame, reference_frac: float = 0.70) -> tuple[pd.DataFrame, dict]:
    """Monthly PSI for every tracked feature, plus a machine-readable summary."""
    df = df.sort_values(pipeline.DATE_COL).reset_index(drop=True)
    cut = int(len(df) * reference_frac)
    reference = df.iloc[:cut]
    months = df[pipeline.DATE_COL].dt.to_period("M").astype(str)

    rows = []
    for month, chunk in df.groupby(months, sort=True):
        for feature in CATEGORICAL:
            rows.append(
                {
                    "month": month,
                    "feature": feature,
                    "kind": "categorical",
                    "psi": categorical_psi(reference[feature], chunk[feature]),
                    "n": len(chunk),
                }
            )
        for name, column in NUMERIC.items():
            rows.append(
                {
                    "month": month,
                    "feature": name,
                    "kind": "numeric",
                    "psi": numeric_psi(reference[column], chunk[column]),
                    "n": len(chunk),
                }
            )
    report = pd.DataFrame(rows)

    post_reference = report[report["month"] > str(df[pipeline.DATE_COL].iloc[cut].to_period("M"))]
    peak = (
        post_reference.groupby("feature")["psi"].max().sort_values(ascending=False).to_dict()
    )
    target_peak = float(peak.get("delay_days", float("nan")))
    feature_peak = max(
        (value for feature, value in peak.items() if feature != "delay_days"), default=0.0
    )
    decision = monitoring.recommend(feature_peak, target_peak)
    summary = {
        "reference_rows": int(cut),
        "reference_end": str(df[pipeline.DATE_COL].iloc[cut].date()),
        "months": int(report["month"].nunique()),
        "max_psi_after_reference": {k: float(v) for k, v in peak.items()},
        "verdict": {
            feature: (
                "high" if value > PSI_HIGH else "moderate" if value > PSI_MODERATE else "stable"
            )
            for feature, value in peak.items()
        },
        "recommendation": decision["action"],
        "recommendation_reason": decision["reason"],
    }
    return report, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--reference-frac", type=float, default=0.70)
    args = ap.parse_args()

    outdir, images = Path(args.outdir), Path(args.images)
    outdir.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    df = pipeline.load_orders(str(pipeline.find_data(args.data)))
    report, summary = build_report(df, args.reference_frac)
    report.to_csv(outdir / "drift_report.csv", index=False)
    (outdir / "drift_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"reference: first {args.reference_frac:.0%} of orders, ends {summary['reference_end']}")
    for feature, value in summary["max_psi_after_reference"].items():
        print(f"  {feature:<18s} max PSI {value:7.3f}  ({summary['verdict'][feature]})")
    print(f"  recommendation: {summary['recommendation']} — {summary['recommendation_reason']}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pivot = report.pivot(index="feature", columns="month", values="psi")
        order = list(summary["max_psi_after_reference"])[::-1]
        pivot = pivot.loc[order]

        fig = plt.figure(figsize=(12.5, 5.6))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.55)
        ax = fig.add_subplot(grid[0, :])
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="OrRd", vmin=0, vmax=0.6)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        step = 3
        ax.set_xticks(range(0, len(pivot.columns), step), pivot.columns[::step], rotation=45, ha="right")
        ax.set_title("Monthly PSI vs the training window (composition vs behaviour)")
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, label="PSI")

        month_key = df[pipeline.DATE_COL].dt.to_period("M").astype(str)
        monthly = df.groupby(month_key).agg(
            late_rate=("delay_days", lambda s: float((s > 0).mean())),
            mean_delay=("delay_days", "mean"),
        )
        ax2 = fig.add_subplot(grid[1, 0])
        ax2.plot(range(len(monthly)), monthly["late_rate"], color="#1f6f78", marker="o", markersize=3)
        ax2.set_xticks(range(0, len(monthly), 6), monthly.index[::6], rotation=45, ha="right")
        ax2.set_ylim(0.4, 0.7)
        ax2.set_title("Late rate by month (concept behaviour)")
        ax2.set_ylabel("late rate")

        ax3 = fig.add_subplot(grid[1, 1])
        ax3.plot(range(len(monthly)), monthly["mean_delay"], color="#9b3b3b", marker="o", markersize=3)
        ax3.set_xticks(range(0, len(monthly), 6), monthly.index[::6], rotation=45, ha="right")
        ax3.set_title("Mean delay by month")
        ax3.set_ylabel("days")
        fig.savefig(images / "drift.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {images / 'drift.png'}")
    except Exception as exc:  # pragma: no cover - optional chart
        print(f"chart skipped: {exc}")

    print(f"wrote {outdir / 'drift_report.csv'} and drift_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
