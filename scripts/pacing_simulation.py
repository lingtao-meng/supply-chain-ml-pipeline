#!/usr/bin/env python3
"""Replay order-level intervention budgets on DataCo.

Every strategy ranks a batch using training-period shipping-mode frequencies.
The global comparator sees the entire test horizon, monthly strategies see
one month at a time, and weekly pacing sees only its Mon–Sun batch, split at
month boundaries. The batch comparator is a greedy heuristic with access to
future orders; it is not an optimal bound or a live arrival policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figure_style import format_figure  # noqa: E402
from src import pipeline, policy  # noqa: E402

BUDGETS = (100_000, 200_000, 300_000, 400_000)


def calendar_periods(dates: pd.DatetimeIndex):
    """Allocate each month's share by days within the observed calendar."""
    calendar = pd.Series(pd.date_range(dates.min(), dates.max(), freq="D"))
    months = list(calendar.groupby(calendar.dt.to_period("M"), sort=True))
    monthly, weekly = [], []
    monthly_share = 1.0 / len(months)
    for _, days in months:
        monthly.append((days.iloc[0], days.iloc[-1], monthly_share))
        for _, week_days in days.groupby(days.dt.to_period("W-SUN"), sort=True):
            share = monthly_share * len(week_days) / len(days)
            weekly.append((week_days.iloc[0], week_days.iloc[-1], share))
    return monthly, weekly


def action_spend(action_idx: np.ndarray, actions: tuple[policy.Action, ...]) -> float:
    costs = np.array([action.cost for action in actions])
    return float(costs[action_idx].sum())


def summarize(periods: list[dict], budget: float, strategy: str) -> dict:
    orders = sum(row["test_orders"] for row in periods)
    late = sum(row["late_orders"] for row in periods)
    spend = sum(row["action_spend"] for row in periods)
    return {
        "total_budget": budget,
        "strategy": strategy,
        "total_cost": sum(row["total_cost"] for row in periods),
        "expected_total_cost": sum(row["expected_total_cost"] for row in periods),
        "action_spend": spend,
        "unspent": budget - spend if np.isfinite(budget) else 0.0,
        "intervention_rate": sum(row["orders_touched"] for row in periods) / orders,
        "recall_late": sum(row["late_orders_touched"] for row in periods) / late if late else np.nan,
        "test_orders": orders,
        "late_orders": late,
    }


def simulate(
    pmf: np.ndarray,
    delay: np.ndarray,
    dates: pd.Series | pd.DatetimeIndex,
    budgets,
    penalty: float,
    actions: tuple[policy.Action, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary and chronological accounting for finite-budget strategies."""
    days = pd.DatetimeIndex(dates).normalize()
    if len(days) == 0 or days.hasnans or len(days) != len(delay) or len(days) != len(pmf):
        raise ValueError("Expected equally sized, nonempty PMF, delay and dated order arrays")
    if not np.isfinite(penalty) or penalty < 0 or not np.isfinite(delay).all():
        raise ValueError("Delay labels must be finite and penalty must be finite and nonnegative")
    budgets = sorted(set(budgets))
    if not budgets or any(not np.isfinite(budget) or budget < 0 for budget in budgets):
        raise ValueError("Provide at least one finite and nonnegative budget")

    monthly, weekly = calendar_periods(days)
    horizon = [(days.min(), days.max(), 1.0)]
    expected = policy.expected_cost_from_pmf(pmf, penalty, actions=actions)

    def account(start, end, mask, chosen, grant, opening) -> dict:
        actual_delay = delay[mask]
        late = actual_delay > 0
        touched = chosen != policy.IDX_NONE
        spend = action_spend(chosen, actions)
        return {
            "period_start": start.date().isoformat(),
            "period_end": end.date().isoformat(),
            "budget_released": grant,
            "opening_balance": opening,
            "action_spend": spend,
            "closing_balance": grant + opening - spend,
            "test_orders": len(chosen),
            "late_orders": int(late.sum()),
            "orders_touched": int(touched.sum()),
            "orders_notify": int((chosen == policy.IDX_NOTIFY).sum()),
            "orders_express": int((chosen == policy.IDX_EXPRESS).sum()),
            "late_orders_touched": int((late & touched).sum()),
            "total_cost": float(policy.realised_cost(chosen, actual_delay, penalty, actions).sum()),
            "expected_total_cost": float(expected[mask, chosen].sum()),
        }

    unconstrained = policy.policy_expected_cost_pmf(pmf, penalty, actions=actions)
    all_orders = np.ones(len(days), dtype=bool)
    optimum = account(days.min(), days.max(), all_orders, unconstrained, np.inf, 0.0)
    rows = [summarize([optimum], np.inf, "unconstrained optimum")]
    details = []
    for budget in budgets:
        for strategy, periods, carries in (
            ("global", horizon, False),
            ("monthly (no carry-over)", monthly, False),
            ("monthly + carry-over", monthly, True),
            ("weekly + carry-over", weekly, True),
        ):
            period_rows = []
            balance = 0.0
            for start, end, share in periods:
                mask = (days >= start) & (days <= end)
                grant = budget * share
                chosen = policy.policy_budget_pmf(pmf[mask], penalty, grant + balance, actions=actions)
                row = account(start, end, mask, chosen, grant, balance)
                period_rows.append(row)
                balance = row["closing_balance"] if carries else 0.0
            rows.append(summarize(period_rows, budget, strategy))
            details.extend({"total_budget": budget, "strategy": strategy, **row} for row in period_rows)
    return pd.DataFrame(rows), pd.DataFrame(details)


def load_replay_cohorts(data_path: Path):
    """Apply the same strict label cutoff in both budget replay studies."""
    df = pipeline.to_order_level(pipeline.load_orders(str(data_path)))
    train_end, test_start = int(len(df) * 0.70), int(len(df) * 0.85)
    if not 0 < train_end < test_start < len(df):
        raise ValueError("Too few orders for the 70/15/15 chronological split")
    train = df.iloc[:train_end]
    cutoff = df[pipeline.DATE_COL].iloc[train_end]
    train = train[train[pipeline.SHIP_COL] < cutoff]
    if train.empty:
        raise ValueError("No training labels were available before the training cutoff")
    test = df.iloc[test_start:]
    return df, train, test, train_end, cutoff


def source_sha256(data_path: Path) -> str:
    digest = hashlib.sha256()
    with data_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--penalty", type=float, default=200.0)
    ap.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    args = ap.parse_args()

    outdir, images = Path(args.outdir), Path(args.images)
    outdir.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    data_path = pipeline.find_data(args.data)
    df, train, test, train_end, cutoff = load_replay_cohorts(data_path)
    mode_pmf, default = policy.pmf_table(train["Shipping Mode"].to_numpy(), train["delay_days"].to_numpy())
    pmf = policy.pmf_matrix(test["Shipping Mode"].to_numpy(), mode_pmf, default)

    table, periods = simulate(
        pmf, test["delay_days"].to_numpy(), test[pipeline.DATE_COL], args.budgets, args.penalty, policy.ACTIONS
    )
    table.to_csv(outdir / "pacing.csv", index=False)
    periods.to_csv(outdir / "pacing_by_period.csv", index=False)
    metadata = {
        "source_sha256": source_sha256(data_path),
        "source_orders": len(df),
        "original_train_orders": train_end,
        "train_orders_with_available_labels": len(train),
        "label_cutoff_exclusive": cutoff.isoformat(),
        "test_period_start": test[pipeline.DATE_COL].iloc[0].isoformat(),
        "test_period_end": test[pipeline.DATE_COL].iloc[-1].isoformat(),
        "test_orders": len(test),
        "late_orders": int((test["delay_days"] > 0).sum()),
        "period_penalty_per_day": args.penalty,
    }
    (outdir / "pacing_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"{len(df):,} source orders, {len(train):,} training orders (shipment before {cutoff})")
    print(f"Test horizon: {test[pipeline.DATE_COL].iloc[0]} through {test[pipeline.DATE_COL].iloc[-1]}")

    print(f"Penalty ${args.penalty:.0f}/day")
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

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
        for axis in axes:
            axis.set_prop_cycle(color=["#355d78", "#986c4e", "#525553", "#7a8797"])
        finite = table[np.isfinite(table["total_budget"])]
        no_carry = finite[finite["strategy"] == "monthly (no carry-over)"].set_index("total_budget")
        markers = ("s", "o", "^", "D")
        for (strategy, group), marker in zip(finite.groupby("strategy", sort=False), markers, strict=True):
            budgets = group["total_budget"].to_numpy()
            delta = group["total_cost"].to_numpy() - no_carry.loc[budgets, "total_cost"].to_numpy()
            axes[0].plot(budgets / 1e3, delta / 1e3, marker=marker, label=strategy)
            axes[1].plot(budgets / 1e3, group["unspent"] / 1e3, marker=marker, label=strategy)
        for axis in axes:
            axis.set_xlabel("Total action budget (USD k)")
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Cost difference (USD k)")
        axes[0].set_title("Cost vs monthly without carry-over")
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel("Unspent budget (USD k)")
        axes[1].set_title("Unused action budget")
        format_figure(fig)
        fig.savefig(images / "pacing.png", dpi=260)
        plt.close(fig)
        print(f"wrote {images / 'pacing.png'}")
    except Exception as exc:  # pragma: no cover - optional chart
        print(f"chart skipped: {exc}")

    print(f"wrote {outdir / 'pacing.csv'}")
    print(f"wrote {outdir / 'pacing_by_period.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
