#!/usr/bin/env python3
"""Score a CSV of orders and write an intervention plan.

This is the batch counterpart of the Streamlit demo: same model bundle, same
cost function, but you feed it a file and get a decision column back.

Usage
-----
    python scripts/score_orders.py --input orders.csv --out plan.csv
    python scripts/score_orders.py --input orders.csv --out plan.csv --budget 500000

The input must be DataCo-shaped (the columns listed in ``REQUIRED_COLUMNS``).
The output is the input plus the predicted quantiles, P(late), the expected
cost of each action and the recommended action.

``--budget`` allocates a fixed intervention budget with the same incremental
greedy rule as ``src.policy.policy_budget_pmf``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, policy  # noqa: E402

DEFAULT_BUNDLE = ROOT / "models" / "model_bundle.joblib"
REQUIRED_COLUMNS = [
    *pipeline.CAT_COLS,
    pipeline.DATE_COL,
    pipeline.SCHED_COL,
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "Order Item Total",
]


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it first with: python scripts/refresh_cost_analysis.py"
        )
    return joblib.load(path)


def pmf_from_bundle(bundle: dict, features: np.ndarray) -> np.ndarray:
    """7-atom pmf for every row (model probabilities aligned to ``ATOMS``)."""
    proba = bundle["pmf_model"].predict_proba(features)
    atoms = np.asarray(bundle["atoms"])
    pmf = np.zeros((len(features), len(atoms)))
    for j, cls in enumerate(bundle["pmf_model"].classes_):
        hits = np.where(atoms == int(cls))[0]
        if hits.size:
            pmf[:, hits[0]] = proba[:, j]
    return pmf


def actions_from_bundle(bundle: dict) -> tuple[policy.Action, ...]:
    return tuple(policy.Action(**spec) for spec in bundle["params"]["actions"])


def score_frame(
    frame: pd.DataFrame,
    bundle: dict,
    penalty: float | None = None,
    budget: float | None = None,
    use_history: bool = False,
) -> pd.DataFrame:
    """Return ``frame`` plus predictions and the recommended action."""
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(frame[pipeline.DATE_COL]):
        frame = frame.copy()
        frame[pipeline.DATE_COL] = pd.to_datetime(frame[pipeline.DATE_COL])

    encoders = bundle["encoders"]
    raw = pipeline.make_features(frame, encoders)
    features = bundle["scaler"].transform(raw)

    if use_history:
        table, default = bundle["mode_pmf"], bundle["mode_pmf_default"]
        pmf = np.vstack([table.get(mode, default) for mode in frame["Shipping Mode"]])
    else:
        pmf = pmf_from_bundle(bundle, features)

    actions = actions_from_bundle(bundle)
    penalty = float(bundle["params"]["penalty"] if penalty is None else penalty)
    exp_cost = policy.expected_cost_from_pmf(pmf, penalty, actions=actions)
    if budget is not None:
        chosen = policy.policy_budget_pmf(pmf, penalty, budget, actions=actions)
    else:
        chosen = exp_cost.argmin(axis=1)

    quantiles = {
        q: np.clip(model.predict(features), -5, 10) for q, model in bundle["quantile"].items()
    }

    out = frame.copy()
    out["p_late"] = bundle["binary"].predict_proba(features)[:, 1]
    out["p10_delay"] = quantiles[0.1]
    out["p50_delay"] = quantiles[0.5]
    out["p90_delay"] = quantiles[0.9]
    for i, action in enumerate(actions):
        out[f"expected_cost_{action.name}"] = exp_cost[:, i]
    out["recommended_action"] = [actions[i].name for i in chosen]
    out["expected_saving"] = exp_cost[:, policy.IDX_NONE] - exp_cost[np.arange(len(out)), chosen]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CSV of orders (DataCo columns)")
    ap.add_argument("--out", default="decisions.csv", help="where to write the plan")
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--penalty", type=float, default=None, help="$ per late day")
    ap.add_argument("--budget", type=float, default=None, help="cap on intervention spend")
    ap.add_argument("--use-history", action="store_true", help="use per-mode history instead of the ML pmf")
    args = ap.parse_args()

    bundle = load_bundle(Path(args.bundle))
    frame = pd.read_csv(args.input, encoding="latin1")
    plan = score_frame(
        frame,
        bundle,
        penalty=args.penalty,
        budget=args.budget,
        use_history=args.use_history,
    )
    plan.to_csv(args.out, index=False)

    counts = plan["recommended_action"].value_counts()
    penalty = float(bundle["params"]["penalty"] if args.penalty is None else args.penalty)
    cost_by_name = {a.name: a.cost for a in actions_from_bundle(bundle)}
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
