#!/usr/bin/env python3
"""How much does the decision depend on the assumed intervention effect?

DataCo has no treatment/control assignment, so the causal effect of "express"
cannot be estimated from it — the project *assumes* express removes two days
and notify multiplies the penalty by 0.6.  This script quantifies that
assumption instead of hiding it:

1. **Break-even effect** — for each shipping mode, how many days must express
   remove before it beats doing nothing?
2. **Cost of a wrong assumption** — choose actions assuming one reduction,
   evaluate with the true reduction.
3. **A/B sample size** — how many orders per arm are needed to estimate a
   delay reduction of a given size?

Usage::

    python scripts/intervention_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, policy  # noqa: E402

PENALTY = 200.0
EXPRESS_COST = 100.0
TRUE_REDUCTION = 2.0


def expected_excess(pmf: np.ndarray, reduction: float, atoms=policy.ATOMS) -> np.ndarray:
    return policy.expected_excess_penalty_pmf(pmf, atoms, reduction)


break_even_reduction = policy.break_even_reduction


def required_sample_size(
    sigma: float, effect: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Orders per arm for a two-sided two-sample test of a mean difference."""
    from scipy.stats import norm

    if effect <= 0:
        return 0
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    return int(np.ceil(2 * (z_alpha + z_beta) ** 2 * sigma**2 / effect**2))


def misspecification_matrix(
    pmf: np.ndarray,
    delay: np.ndarray,
    assumed_reductions,
    true_reduction: float,
    penalty: float = PENALTY,
    express_cost: float = EXPRESS_COST,
) -> pd.DataFrame:
    """Cost when the policy assumes ``r_a`` but the truth is ``true_reduction``."""
    rows = []
    for assumed in assumed_reductions:
        assumed_actions = (
            policy.NONE,
            policy.NOTIFY,
            policy.Action("express", cost=express_cost, reduction=float(assumed)),
        )
        idx = policy.policy_expected_cost_pmf(pmf, penalty, actions=assumed_actions)
        truth_actions = (
            policy.NONE,
            policy.NOTIFY,
            policy.Action("express", cost=express_cost, reduction=float(true_reduction)),
        )
        cost = policy.realised_cost(idx, delay, penalty, truth_actions)
        rows.append(
            {
                "assumed_reduction": float(assumed),
                "true_reduction": float(true_reduction),
                "total_cost": float(cost.sum()),
                "cost_per_order": float(cost.mean()),
                "intervention_rate": float((idx != policy.IDX_NONE).mean()),
                "express_rate": float((idx == policy.IDX_EXPRESS).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--penalty", type=float, default=PENALTY)
    args = ap.parse_args()

    outdir, images = Path(args.outdir), Path(args.images)
    outdir.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    df = pipeline.to_order_level(pipeline.load_orders(str(pipeline.find_data(args.data))))
    from src import scoring

    bundle = scoring.load_bundle()
    X, y = pipeline.build_features(df, bundle["encoders"])
    split = pipeline.time_split(X, y)
    mode = df.loc[split.test_index, "Shipping Mode"].to_numpy()
    pmf = policy.pmf_matrix(mode, bundle["mode_pmf"], bundle["mode_pmf_default"])
    delay = split.y_test

    # 1. break-even reduction per shipping mode
    break_rows = []
    for name, row_pmf in bundle["mode_pmf"].items():
        base = float(expected_excess(np.atleast_2d(row_pmf), 0.0)[0])
        break_rows.append(
            {
                "shipping_mode": name,
                "expected_excess_days": base,
                "break_even_reduction": break_even_reduction(
                    np.atleast_2d(row_pmf), args.penalty, EXPRESS_COST
                ),
            }
        )
    breaks = pd.DataFrame(break_rows)
    breaks.to_csv(outdir / "intervention_break_even.csv", index=False)
    print(f"break-even express reduction at ${args.penalty:.0f}/day and ${EXPRESS_COST:.0f} express:")
    for _, row in breaks.iterrows():
        value = row["break_even_reduction"]
        print(f"  {row['shipping_mode']:<16s} E[delay+]={row['expected_excess_days']:.2f}d  "
              f"break-even={value:.2f}d" if np.isfinite(value) else
              f"  {row['shipping_mode']:<16s} express never pays for itself")

    # 2. cost of assuming the wrong effect
    assumed_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    miss = misspecification_matrix(pmf, delay, assumed_grid, TRUE_REDUCTION, args.penalty)
    miss.to_csv(outdir / "intervention_misspecification.csv", index=False)
    print("\ncost if the assumed express reduction is wrong (truth = 2 days):")
    for _, row in miss.iterrows():
        print(f"  assume {row['assumed_reduction']:.0f}d -> ${row['total_cost']:,.0f} "
              f"({row['intervention_rate']:.1%} touched, {row['express_rate']:.1%} express)")

    # 3. A/B sample size
    sigma = float(np.std(delay, ddof=1))
    effects = [0.1, 0.2, 0.3, 0.5, 1.0]
    power_rows = [
        {
            "effect_days": effect,
            "orders_per_arm": required_sample_size(sigma, effect),
        }
        for effect in effects
    ]
    power = pd.DataFrame(power_rows)
    power.to_csv(outdir / "ab_sample_size.csv", index=False)
    print(f"\nper-order delay SD = {sigma:.3f} days; orders per arm for 80% power at alpha=0.05:")
    for _, row in power.iterrows():
        print(f"  detect {row['effect_days']:.1f}d -> {int(row['orders_per_arm']):,} per arm")

    summary = {
        "penalty": args.penalty,
        "express_cost": EXPRESS_COST,
        "true_reduction_assumed": TRUE_REDUCTION,
        "break_even": breaks.to_dict(orient="records"),
        "misspecification": miss.to_dict(orient="records"),
        "delay_sd": sigma,
        "sample_size": power.to_dict(orient="records"),
    }
    (outdir / "intervention_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        axes[0].plot(miss["assumed_reduction"], miss["total_cost"] / 1e3, marker="o", color="#1f6f78")
        axes[0].axvline(TRUE_REDUCTION, color="#9b3b3b", linestyle="--", linewidth=1,
                        label="true effect (2d)")
        axes[0].set_xlabel("assumed express reduction (days)")
        axes[0].set_ylabel("total cost (USD k)")
        axes[0].set_title("Cost of assuming the wrong intervention effect")
        axes[0].legend(fontsize=8)

        axes[1].plot(power["effect_days"], power["orders_per_arm"], marker="o", color="#d99a2b")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("delay reduction to detect (days)")
        axes[1].set_ylabel("orders per arm (log)")
        axes[1].set_title("A/B sample size (80% power, alpha 0.05)")
        fig.tight_layout()
        fig.savefig(images / "intervention.png", dpi=150)
        plt.close(fig)
        print(f"wrote {images / 'intervention.png'}")
    except Exception as exc:  # pragma: no cover - optional chart
        print(f"chart skipped: {exc}")

    print(f"\nwrote {outdir / 'intervention_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
