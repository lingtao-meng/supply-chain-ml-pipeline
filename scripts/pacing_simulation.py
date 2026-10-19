#!/usr/bin/env python3
"""Multi-period budget pacing: what does myopia cost?

The static budget analysis assumes the planner sees all orders at once.  In
reality orders arrive over time and the budget is released monthly:

* **global**    — one budget for the whole horizon (upper bound, needs foresight);
* **monthly**   — 1/N of the budget each month, no carry-over;
* **carry-over** — unspent budget rolls into the next month (still no foresight).

The gap between them is the price of pacing, and it tells you whether to spend
early or hold budget back.

Usage::

    python scripts/pacing_simulation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, policy, scoring  # noqa: E402

BUDGETS = [500_000, 1_000_000, 1_500_000]


def month_keys(dates: pd.Series) -> np.ndarray:
    return dates.dt.to_period("M").astype(str).to_numpy()


def action_spend(action_idx: np.ndarray, actions) -> float:
    costs = np.array([a.cost for a in actions])
    return float(costs[action_idx].sum())


def allocate(pmf, penalty, budget, actions) -> np.ndarray:
    return policy.policy_budget_pmf(pmf, penalty, budget, actions=actions)


def simulate(
    pmf: np.ndarray,
    delay: np.ndarray,
    months: np.ndarray,
    budgets,
    penalty: float,
    actions,
) -> pd.DataFrame:
    """Run global / monthly / carry-over pacing for every total budget."""
    rows = []
    order = np.argsort(months, kind="stable")
    pmf, delay, months = pmf[order], delay[order], months[order]
    unique_months = np.unique(months)
    n = len(delay)
    exp_cost = policy.expected_cost_from_pmf(pmf, penalty, actions=actions)

    def expected_total(idx: np.ndarray) -> float:
        return float(exp_cost[np.arange(n), idx].sum())

    unconstrained = policy.policy_expected_cost_pmf(pmf, penalty, actions=actions)
    rows.append(
        {
            "total_budget": np.inf,
            "strategy": "unconstrained optimum",
            "total_cost": float(policy.realised_cost(unconstrained, delay, penalty, actions).sum()),
            "expected_total_cost": expected_total(unconstrained),
            "action_spend": action_spend(unconstrained, actions),
            "unspent": 0.0,
            "intervention_rate": float((unconstrained != policy.IDX_NONE).mean()),
            "recall_late": float(((unconstrained != policy.IDX_NONE) & (delay > 0)).sum() / (delay > 0).sum()),
        }
    )

    for budget in budgets:
        # 1. global allocation (needs foresight, but is the natural upper bound)
        idx_global = allocate(pmf, penalty, budget, actions)
        rows.append(
            {
                "total_budget": budget,
                "strategy": "global",
                "total_cost": float(policy.realised_cost(idx_global, delay, penalty, actions).sum()),
                "expected_total_cost": expected_total(idx_global),
                "action_spend": action_spend(idx_global, actions),
                "unspent": float(budget - action_spend(idx_global, actions)),
                "intervention_rate": float((idx_global != policy.IDX_NONE).mean()),
                "recall_late": float(((idx_global != policy.IDX_NONE) & (delay > 0)).sum() / (delay > 0).sum()),
            }
        )

        # 2. monthly budget, no carry-over
        monthly_budget = budget / len(unique_months)
        idx_monthly = np.full(n, policy.IDX_NONE)
        for month in unique_months:
            mask = months == month
            idx_monthly[mask] = allocate(pmf[mask], penalty, monthly_budget, actions)
        spend_monthly = action_spend(idx_monthly, actions)
        rows.append(
            {
                "total_budget": budget,
                "strategy": "monthly (no carry-over)",
                "total_cost": float(policy.realised_cost(idx_monthly, delay, penalty, actions).sum()),
                "expected_total_cost": expected_total(idx_monthly),
                "action_spend": spend_monthly,
                "unspent": float(budget - spend_monthly),
                "intervention_rate": float((idx_monthly != policy.IDX_NONE).mean()),
                "recall_late": float(((idx_monthly != policy.IDX_NONE) & (delay > 0)).sum() / (delay > 0).sum()),
            }
        )

        # 3. monthly with carry-over: unspent budget rolls into the next month
        leftover = 0.0
        idx_carry = np.full(n, policy.IDX_NONE)
        for month in unique_months:
            mask = months == month
            available = min(budget, monthly_budget + leftover)
            chosen = allocate(pmf[mask], penalty, available, actions)
            idx_carry[mask] = chosen
            leftover = available - action_spend(chosen, actions)
        spend_carry = action_spend(idx_carry, actions)
        rows.append(
            {
                "total_budget": budget,
                "strategy": "monthly + carry-over",
                "total_cost": float(policy.realised_cost(idx_carry, delay, penalty, actions).sum()),
                "expected_total_cost": expected_total(idx_carry),
                "action_spend": spend_carry,
                "unspent": float(budget - spend_carry),
                "intervention_rate": float((idx_carry != policy.IDX_NONE).mean()),
                "recall_late": float(((idx_carry != policy.IDX_NONE) & (delay > 0)).sum() / (delay > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--penalty", type=float, default=200.0)
    args = ap.parse_args()

    outdir, images = Path(args.outdir), Path(args.images)
    outdir.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    df = pipeline.load_orders(str(pipeline.find_data(args.data)))
    bundle = scoring.load_bundle()
    X, y = pipeline.build_features(df, bundle["encoders"])
    split = pipeline.time_split(X, y)
    mode = df.loc[split.test_index, "Shipping Mode"].to_numpy()
    pmf = policy.pmf_matrix(mode, bundle["mode_pmf"], bundle["mode_pmf_default"])
    delay = split.y_test
    months = month_keys(df.loc[split.test_index, pipeline.DATE_COL])
    actions = scoring.actions_from_bundle(bundle)

    table = simulate(pmf, delay, months, BUDGETS, args.penalty, actions)
    table.to_csv(outdir / "pacing.csv", index=False)

    print(f"{len(np.unique(months))} months, {len(delay):,} test orders, penalty ${args.penalty:.0f}/day")
    view = table.copy()
    view["total_budget"] = view["total_budget"].map(lambda v: "none" if np.isinf(v) else f"${v:,.0f}")
    for col in ("total_cost", "action_spend", "unspent"):
        view[col] = view[col].map(lambda v: f"${v:,.0f}")
    for col in ("intervention_rate", "recall_late"):
        view[col] = view[col].map(lambda v: f"{v:.1%}")
    print(view.to_string(index=False))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        finite = table[np.isfinite(table["total_budget"])]
        for strategy, group in finite.groupby("strategy", sort=False):
            axes[0].plot(group["total_budget"] / 1e3, group["total_cost"] / 1e3, marker="o", label=strategy)
            axes[1].plot(group["total_budget"] / 1e3, group["unspent"] / 1e3, marker="o", label=strategy)
        axes[0].set_xlabel("total budget (USD k)")
        axes[0].set_ylabel("total cost (USD k)")
        axes[0].set_title("Pacing cost: global vs monthly")
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel("total budget (USD k)")
        axes[1].set_ylabel("unspent budget (USD k)")
        axes[1].set_title("Budget left on the table")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(images / "pacing.png", dpi=150)
        plt.close(fig)
        print(f"wrote {images / 'pacing.png'}")
    except Exception as exc:  # pragma: no cover - optional chart
        print(f"chart skipped: {exc}")

    print(f"wrote {outdir / 'pacing.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
