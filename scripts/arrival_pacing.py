#!/usr/bin/env python3
"""Replay irrevocable budget decisions in timestamp and order-ID order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pacing_simulation import BUDGETS, load_replay_cohorts, source_sha256  # noqa: E402
from src import pipeline, policy  # noqa: E402

START = "2017-09-05"
END = "2018-01-31"
EPS = 1e-9


def decide_on_arrival(
    expected_cost: np.ndarray,
    timestamps: pd.DatetimeIndex,
    fees: np.ndarray,
    budget: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    daily_release: bool,
    min_expected_gain: float = 0.0,
) -> pd.DataFrame:
    """Decide from one order's forecasts and cash; outcomes are not inputs."""
    elapsed_days = (timestamps.normalize() - start).days.to_numpy() + 1
    horizon_days = (end - start).days + 1
    if (
        not len(timestamps) or len(timestamps) != len(expected_cost)
        or not timestamps.is_monotonic_increasing or elapsed_days.min() < 1
        or elapsed_days.max() > horizon_days
    ):
        raise ValueError("Expected a chronological test cohort entirely within the declared calendar")
    spend = 0.0
    rows = []
    for costs, day in zip(expected_cost, elapsed_days, strict=True):
        released = budget * day / horizon_days if daily_release else budget
        room = released - spend
        order = np.lexsort((fees, costs))
        preferred = order[0]
        chosen = order[fees[order] <= room + EPS][0]
        reason = ""
        if costs[policy.IDX_NONE] - costs[preferred] < min_expected_gain - EPS:
            chosen = policy.IDX_NONE
            reason = "priority_gate"
        elif preferred != chosen:
            reason = "full_budget" if fees[preferred] > budget - spend + EPS else "daily_release"
        spend += fees[chosen]
        rows.append((chosen, preferred, reason, room, released, spend))
    return pd.DataFrame(
        rows, columns=("action", "preferred", "limit_reason", "cash_before", "released_to_date", "spent_to_date")
    )


def summarize_decisions(
    decisions: pd.DataFrame,
    delay: np.ndarray,
    expected: np.ndarray,
    high_priority: np.ndarray,
    budget: float,
    penalty: float,
    strategy: str,
) -> tuple[dict, pd.DataFrame]:
    """Apply observed outcomes only after the actions have been selected."""
    fees = np.array([a.cost for a in policy.ACTIONS])
    selected = decisions["action"].to_numpy()
    days = decisions["day"]
    out = decisions.assign(
        test_orders=1,
        priority_orders=high_priority.astype(int),
        orders_notify=(selected == policy.IDX_NOTIFY).astype(int),
        orders_express=(selected == policy.IDX_EXPRESS).astype(int),
        priority_no_action=(high_priority & (selected == policy.IDX_NONE)).astype(int),
        priority_full_budget=(high_priority & decisions["limit_reason"].eq("full_budget")).astype(int),
        priority_daily_release=(high_priority & decisions["limit_reason"].eq("daily_release")).astype(int),
        action_spend=fees[selected],
        cost=policy.realised_cost(selected, delay, penalty),
        expected_cost=expected[np.arange(len(selected)), selected],
    )
    measures = [
        "test_orders", "priority_orders", "orders_notify", "orders_express", "priority_no_action",
        "priority_full_budget", "priority_daily_release", "action_spend", "cost", "expected_cost",
    ]
    sums = out[measures].sum()
    result = {name: int(sums[name]) for name in measures[:-3]}
    result.update({name: float(sums[name]) for name in ("action_spend", "cost", "expected_cost")})
    result.update(
        total_budget=budget,
        strategy=strategy,
        unspent=budget - result["action_spend"],
        first_full_budget_priority_limit=days[out["priority_full_budget"].eq(1)].min(),
    )
    online = strategy != "full-horizon greedy"
    if not online:
        for metric in ("priority_full_budget", "priority_daily_release"):
            result[metric] = np.nan
        return result, pd.DataFrame()
    totals = out.groupby("day", sort=True)[measures].sum()
    totals[["released_to_date", "spent_to_date"]] = out.groupby("day")[
        ["released_to_date", "spent_to_date"]
    ].last()
    totals = totals.reset_index().assign(total_budget=budget, strategy=strategy)
    return result, totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    parser.add_argument("--start", default=START, help="first budget calendar day (YYYY-MM-DD)")
    parser.add_argument("--end", default=END, help="last budget calendar day (YYYY-MM-DD)")
    parser.add_argument("--penalty", type=float, default=200.0)
    parser.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    args = parser.parse_args()
    if any(not np.isfinite(x) or x < 0 for x in [args.penalty, *args.budgets]):
        parser.error("Penalty and budgets must be finite and nonnegative")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start > end or start.normalize() != start or end.normalize() != end:
        parser.error("Use a nonempty calendar with dates at midnight")

    data_path = pipeline.find_data(args.data)
    _, train, test, _, cutoff = load_replay_cohorts(data_path)
    test = test.sort_values([pipeline.DATE_COL, "Order Id"])
    timestamps = pd.DatetimeIndex(test[pipeline.DATE_COL])
    delay = test["delay_days"].to_numpy()
    fees = np.array([action.cost for action in policy.ACTIONS])
    mode = "Shipping Mode"
    lookup, default = policy.pmf_table(train[mode], train["delay_days"])
    train_expected = policy.expected_cost_from_pmf(policy.pmf_matrix(train[mode], lookup, default), args.penalty)
    train_gain = train_expected[:, policy.IDX_NONE] - train_expected.min(axis=1)
    # Fix the threshold on training data before any test orders are scored.
    threshold = float(np.quantile(train_gain, 0.75, method="higher"))
    pmf = policy.pmf_matrix(test[mode], lookup, default)
    expected = policy.expected_cost_from_pmf(pmf, args.penalty)
    gain = expected[:, policy.IDX_NONE] - expected.min(axis=1)
    high_priority = (gain > EPS) & (gain >= threshold - EPS)

    results, daily_rows = [], []
    for budget in sorted(set(args.budgets)):
        for label, daily_release, minimum_gain in (
            ("first-come", False, 0.0),
            ("historical threshold", False, threshold),
            ("daily pacing", True, 0.0),
        ):
            decisions = decide_on_arrival(
                expected, timestamps, fees, budget, start, end,
                daily_release=daily_release, min_expected_gain=minimum_gain,
            )
            decisions["day"] = timestamps.normalize()
            summary, days = summarize_decisions(decisions, delay, expected, high_priority, budget, args.penalty, label)
            results.append(summary)
            daily_rows.append(days)
        selected = policy.policy_budget_pmf(pmf, args.penalty, budget)
        hindsight = pd.DataFrame({"action": selected, "day": timestamps.normalize(), "limit_reason": ""})
        summary, _ = summarize_decisions(
            hindsight, delay, expected, high_priority, budget, args.penalty, "full-horizon greedy"
        )
        results.append(summary)

    result = pd.DataFrame(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.outdir / "arrival_pacing.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(args.outdir / "arrival_pacing_by_day.csv", index=False)
    meta = {
        "source_sha256": source_sha256(data_path),
        "train_orders": len(train),
        "test_orders": len(test),
        "train_label_cutoff_exclusive": cutoff.isoformat(),
        "calendar_start": start.date().isoformat(),
        "calendar_end": end.date().isoformat(),
        "priority_train_quantile": 0.75,
        "priority_expected_gain_threshold": threshold,
        "priority_test_orders": int(high_priority.sum()),
        "penalty_per_day": args.penalty,
    }
    (args.outdir / "arrival_pacing_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
