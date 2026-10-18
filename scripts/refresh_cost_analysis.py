#!/usr/bin/env python3
"""Refresh the cost analysis for the late-delivery project.

Why this script exists
----------------------
The original notebook scored each policy with a slightly different cost
formula, so the "do nothing" baseline looked cheaper than it should have.
This script puts every policy on **one** cost function and adds two things
that were missing:

* multi-tier interventions (notify / express) instead of a single $100 action;
* a budget-constrained allocation policy (the LP relaxation of the knapsack),
  which is what a planner actually faces.

Usage
-----
    python scripts/refresh_cost_analysis.py
    python scripts/refresh_cost_analysis.py --no-cache --penalty 200

Outputs (all under ``results/``)
--------------------------------
    policy_comparison.csv     one row per policy, unified cost function
    sensitivity.csv           cost vs. (penalty, express cost)
    capacity_curve.csv        cost vs. expedite capacity
    summary.md                numbers ready to paste into the README
    images/policy_costs.png   bar chart of policy cost
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, policy  # noqa: E402

CACHE = ROOT / "models" / "quantile_cache.joblib"
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"


def fit_or_load(split, no_cache: bool):
    if CACHE.exists() and not no_cache:
        print(f"[cache] {CACHE.relative_to(ROOT)}")
        return joblib.load(CACHE)

    from sklearn.ensemble import GradientBoostingRegressor

    models = {}
    for q in (0.1, 0.5, 0.9):
        t0 = time.time()
        m = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            loss="quantile",
            alpha=q,
            random_state=42,
        )
        m.fit(split.X_train, split.y_train)
        models[q] = m
        print(f"  q={q}: fitted in {time.time() - t0:.0f}s")

    # binary baseline (the "classic" approach this project argues against)
    from sklearn.metrics import roc_auc_score

    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
        clf.fit(split.X_train, (split.y_train > 0).astype(int))
    except Exception as exc:  # pragma: no cover - fallback only
        print(f"  xgboost unavailable ({exc}); using GradientBoostingClassifier")
        from sklearn.ensemble import GradientBoostingClassifier

        clf = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
        clf.fit(split.X_train, (split.y_train > 0).astype(int))

    auc = roc_auc_score((split.y_test > 0).astype(int), clf.predict_proba(split.X_test)[:, 1])
    print(f"  binary AUC (time split) = {auc:.4f}")

    payload = {"quantile": models, "binary": clf, "binary_auc": auc}
    CACHE.parent.mkdir(exist_ok=True)
    joblib.dump(payload, CACHE)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "DataCoSupplyChainDataset.csv"))
    ap.add_argument("--penalty", type=float, default=200.0, help="$ per late day")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    print("== data ==")
    df = pipeline.load_orders(args.data)
    facts = pipeline.describe(df)
    print(json.dumps(facts, indent=2, ensure_ascii=False))

    X, y = pipeline.build_features(df)
    split = pipeline.time_split(X, y)
    print(f"train={len(split.y_train):,} valid={len(split.y_valid):,} test={len(split.y_test):,}")
    print(f"features={len(split.feature_names)}")

    print("== models ==")
    payload = fit_or_load(split, args.no_cache)
    models = payload["quantile"]
    clf = payload["binary"]

    p = {q: m.predict(split.X_test) for q, m in models.items()}
    p10, p50, p90 = p[0.1], p[0.5], p[0.9]
    print(f"quantile coverage: "
          f"P10={(split.y_test <= p10).mean():.1%} "
          f"P50={(split.y_test <= p50).mean():.1%} "
          f"P90={(split.y_test <= p90).mean():.1%}")
    print(f"MAE(P50) = {np.abs(split.y_test - p50).mean():.3f} days")

    draws = policy.quantile_grid(p10, p50, p90)
    delay = split.y_test
    pen = args.penalty

    # ---------------- policies
    n = len(delay)
    pol = {
        "none (do nothing)": policy.policy_none(n),
        "all express": policy.policy_all_express(n),
        "binary (p50>0 -> express)": policy.policy_binary(p50=p50),
        "quantile expected-cost": policy.policy_expected_cost(draws, pen),
        "capacity 25% (best-first)": policy.policy_capacity(draws, pen, 0.25),
        "oracle (hindsight)": policy.policy_oracle(delay, pen),
    }
    comp = policy.compare(pol, delay, pen)
    comp = comp.sort_values("total_cost").reset_index(drop=True)

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    comp.to_csv(RESULTS / "policy_comparison.csv", index=False)
    print("\n" + policy.summarise_table(comp))

    # ---------------- sensitivity
    sens = policy.sensitivity(
        draws, delay, p50,
        penalties=[100, 200, 300, 400, 500],
        express_costs=[50, 75, 100, 125, 150],
    )
    sens.to_csv(RESULTS / "sensitivity.csv", index=False)

    # ---------------- budget curve
    capacities = [0.10, 0.25, 0.50, 0.75, 1.00]
    bc = policy.capacity_curve(draws, delay, pen, capacities)
    bc.to_csv(RESULTS / "capacity_curve.csv", index=False)

    # ---------------- chart
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.2))
        sub = comp[~comp["policy"].str.startswith("oracle")]
        colors = ["#9aa5b1" if "none" in s else "#2f5d8c" for s in sub["policy"]]
        ax.barh(sub["policy"], sub["total_cost"] / 1e3, color=colors)
        ax.set_xlabel("total intervention cost on test set ($k)")
        ax.set_title(f"Unified cost evaluation — {n:,} test orders, penalty ${pen:.0f}/day")
        for y_, v in enumerate(sub["total_cost"] / 1e3):
            ax.text(v + 3, y_, f"${v:,.0f}k", va="center", fontsize=8)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(IMAGES / "policy_costs.png", dpi=150)
        print(f"\nwrote {IMAGES / 'policy_costs.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"chart skipped: {exc}")

    # ---------------- summary
    get = lambda name: comp.loc[comp["policy"] == name].iloc[0]  # noqa: E731
    none_c = get("none (do nothing)")
    all_c = get("all express")
    bin_c = get("binary (p50>0 -> express)")
    our_c = get("quantile expected-cost")
    orc = get("oracle (hindsight)")

    lines = [
        "# Cost analysis — refreshed numbers",
        "",
        f"Generated by `scripts/refresh_cost_analysis.py` · penalty ${pen:.0f}/late-day · "
        f"{n:,} test orders (chronological split).",
        "",
        "| metric | value |",
        "|---|---|",
        f"| orders (full dataset) | {facts['orders']:,} |",
        f"| late rate | {facts['late_rate']:.1%} |",
        f"| total late-days | {facts['total_delay_days']:,.0f} |",
        f"| binary baseline AUC | {payload['binary_auc']:.4f} |",
        f"| MAE (P50, days) | {np.abs(delay - p50).mean():.3f} |",
        f"| P10/P50/P90 coverage | {(delay <= p10).mean():.1%} / {(delay <= p50).mean():.1%} / {(delay <= p90).mean():.1%} |",
        "",
        "## Policy comparison (one cost function for every policy)",
        "",
        policy.summarise_table(comp),
        "",
        "## Headline deltas",
        "",
        f"- vs **do nothing**: ${none_c['total_cost']:,.0f} → ${our_c['total_cost']:,.0f} "
        f"({1 - our_c['total_cost'] / none_c['total_cost']:.1%} lower)",
        f"- vs **binary threshold**: ${bin_c['total_cost']:,.0f} → ${our_c['total_cost']:,.0f} "
        f"({1 - our_c['total_cost'] / bin_c['total_cost']:.1%} lower)",
        f"- vs **all express**: ${all_c['total_cost']:,.0f} → ${our_c['total_cost']:,.0f} "
        f"({1 - our_c['total_cost'] / all_c['total_cost']:.1%} lower)",
        f"- gap to hindsight oracle: {our_c['total_cost'] / orc['total_cost'] - 1:.1%}",
        "",
        "## Capacity-constrained allocation (expedite only k% of orders)",
        "",
        bc[["capacity", "total_cost", "cost_per_order", "intervention_rate",
            "recall_late", "precision", "marginal_saving_per_pp"]]
        .to_string(index=False, float_format=lambda v: f"{v:,.4f}"),
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
