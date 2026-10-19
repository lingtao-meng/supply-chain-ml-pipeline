#!/usr/bin/env python3
"""Score a CSV of orders and write an intervention plan.

This is the batch counterpart of the Streamlit demo and the HTTP API: same
model bundle, same cost function, but you feed it a file and get a decision
column back.

Usage
-----
    python scripts/score_orders.py --input orders.csv --out plan.csv
    python scripts/score_orders.py --input orders.csv --out plan.csv --budget 500000

The input must be DataCo-shaped (the columns listed in
``src.scoring.REQUIRED_COLUMNS``).  The output is the input plus the predicted
quantiles, P(late), the expected cost of each action and the recommended one.

``--budget`` allocates a fixed intervention budget with the same incremental
greedy rule as ``src.policy.policy_budget_pmf``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import scoring  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CSV of orders (DataCo columns)")
    ap.add_argument("--out", default="decisions.csv", help="where to write the plan")
    ap.add_argument("--bundle", default=str(scoring.DEFAULT_BUNDLE))
    ap.add_argument("--penalty", type=float, default=None, help="$ per late day")
    ap.add_argument("--budget", type=float, default=None, help="cap on intervention spend")
    ap.add_argument("--use-history", action="store_true", help="use per-mode history instead of the ML pmf")
    args = ap.parse_args()

    bundle = scoring.load_bundle(args.bundle)
    frame = pd.read_csv(args.input, encoding="latin1")
    plan = scoring.score_frame(
        frame,
        bundle,
        penalty=args.penalty,
        budget=args.budget,
        use_history=args.use_history,
    )
    plan.to_csv(args.out, index=False)

    counts = plan["recommended_action"].value_counts()
    penalty = float(bundle["params"]["penalty"] if args.penalty is None else args.penalty)
    cost_by_name = {a.name: a.cost for a in scoring.actions_from_bundle(bundle)}
    spend = float(
        sum(cost_by_name.get(str(name), 0.0) * int(count) for name, count in counts.items())
    )
    print(f"scored {len(plan):,} orders -> {args.out}")
    print(f"  action mix: {counts.to_dict()}")
    print(f"  action spend: ${spend:,.0f} (budget {'none' if args.budget is None else f'${args.budget:,.0f}'})")
    print(f"  expected saving vs no action: ${plan['expected_saving'].sum():,.0f} "
          f"(penalty ${penalty:,.0f}/day)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
