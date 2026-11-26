"""Hold order-arrival decisions fixed, then stress their assumed cost and effectiveness."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, brentq, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.arrival_pacing import END, EPS, START, decide_on_arrival  # noqa: E402
from scripts.pacing_simulation import BUDGETS, load_replay_cohorts, source_sha256  # noqa: E402
from src import pipeline, policy  # noqa: E402

PENALTIES = (100, 200, 300)
NOTIFY_REMAINDERS = (0.60, 0.80, 1.00)
EXPRESS_DAYS = (0.0, 0.5, 1.0, 2.0)
BASELINE_PENALTY = 200
ONLINE = ("first-come", "historical threshold", "daily pacing")


def exact_allocation(expected: np.ndarray, fees: np.ndarray, budget: float) -> np.ndarray:
    """Optimize the assumed forecast cost with all arrivals visible and no outcomes."""
    costs, membership, supplies = np.unique(expected, axis=0, return_inverse=True, return_counts=True)
    paid_actions = len(fees) - 1
    objective = (costs[:, 1:] - costs[:, :1]).ravel()
    limits = np.repeat(np.eye(len(costs)), paid_actions, axis=1)
    limits = np.vstack([limits, np.tile(fees[1:], len(costs))])
    solution = milp(objective, integrality=np.ones_like(objective), bounds=Bounds(0, np.inf),
                    constraints=LinearConstraint(limits, 0, np.r_[supplies, budget]),
                    options={"mip_rel_gap": 0.0})
    if solution.status != 0:
        raise RuntimeError(f"Budget allocation optimum not established: {solution.message}")
    chosen_counts = np.rint(solution.x).astype(int).reshape(len(costs), paid_actions)
    action = np.zeros(len(expected), dtype=int)
    for group, counts in enumerate(chosen_counts):
        # Allocate ties in timestamp order without reading realised outcomes.
        indices = np.flatnonzero(membership == group)
        start = 0
        for code, count in enumerate(counts, start=1):
            action[indices[start:start + count]] = code
            start += count
    observed = expected[np.arange(len(action)), action].sum() - expected[:, 0].sum()
    if (fees[action].sum() > budget + EPS or (chosen_counts.sum(axis=1) > supplies).any()
            or not np.isclose(observed, solution.fun, atol=1e-6, rtol=0)):
        raise AssertionError("Decoded budget allocation does not reproduce the optimum")
    return action


def frozen_actions(expected, pmf, timestamps, fees, budget, threshold):
    active = {}
    for strategy, daily_release, minimum_gain in zip(
        ONLINE, (False, False, True), (0.0, threshold, 0.0), strict=True,
    ):
        rows = decide_on_arrival(expected, timestamps, fees, budget, pd.Timestamp(START), pd.Timestamp(END),
                                 daily_release=daily_release, min_expected_gain=minimum_gain)
        active[strategy] = rows["action"].to_numpy()
    active["full-horizon greedy"] = policy.policy_budget_pmf(pmf, BASELINE_PENALTY, budget)
    active["full-horizon exact"] = exact_allocation(expected, fees, budget)
    active["no action"] = np.zeros(len(expected), dtype=int)
    return active


def express_break_even(delay, action, fees, penalty, notify_remaining) -> tuple[float | None, str]:
    """Find the least express day saving needed for the frozen plan to pay versus no action."""
    express = action == policy.IDX_EXPRESS
    notify = action == policy.IDX_NOTIFY
    late = np.maximum(delay, 0)
    fees_due = fees[action].sum()
    notify_savings = penalty * (1 - notify_remaining) * late[notify].sum()
    express_lateness = late[express]

    def remaining_cost(reduction):
        express_savings = penalty * np.minimum(reduction, express_lateness).sum()
        return fees_due - notify_savings - express_savings

    if remaining_cost(0) <= EPS:
        return 0.0, "profitable_without_express_effect"
    if express_lateness.size == 0 or remaining_cost(express_lateness.max()) > EPS:
        return None, "unattainable"
    return float(brentq(remaining_cost, 0.0, express_lateness.max())), "threshold"


def run(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    _, train, test, _, cutoff = load_replay_cohorts(data_path)
    test = test.sort_values([pipeline.DATE_COL, "Order Id"])
    timestamps = pd.DatetimeIndex(test[pipeline.DATE_COL])
    delay = test["delay_days"].to_numpy()
    mode = "Shipping Mode"
    lookup, default = policy.pmf_table(train[mode], train["delay_days"])
    training_pmf = policy.pmf_matrix(train[mode], lookup, default)
    training = policy.expected_cost_from_pmf(training_pmf, BASELINE_PENALTY)
    threshold = np.quantile(training[:, 0] - training.min(axis=1), .75, method="higher")
    pmf = policy.pmf_matrix(test[mode], lookup, default)
    expected = policy.expected_cost_from_pmf(pmf, BASELINE_PENALTY)
    gain = expected[:, 0] - expected.min(axis=1)
    priority = (gain > EPS) & (gain >= threshold - EPS)
    fees = np.array([action.cost for action in policy.ACTIONS])
    rows, bounds, thresholds = [], [], []
    for budget in BUDGETS:
        actions = frozen_actions(expected, pmf, timestamps, fees, budget, threshold)
        reference = expected[np.arange(len(delay)), actions["full-horizon exact"]].sum()
        for label, action in actions.items():
            expected_cost = expected[np.arange(len(delay)), action].sum()
            spend = fees[action].sum()
            covered = int(np.count_nonzero(priority & (action != policy.IDX_NONE)))
            if spend > budget + EPS or expected_cost + 1e-6 < reference:
                raise AssertionError(f"Invalid budget or objective comparison for {label}")
            bounds.append({"budget": budget, "strategy": label, "expected_cost": expected_cost,
                           "expected_gap_to_exact": max(expected_cost - reference, 0.0),
                           "action_spend": spend, "priority_covered": covered})
            if label in ONLINE:
                for penalty, remainder in product(PENALTIES, NOTIFY_REMAINDERS):
                    minimum, status = express_break_even(delay, action, fees, penalty, remainder)
                    thresholds.append({"budget": budget, "strategy": label, "penalty": penalty,
                                       "notify_remaining": remainder, "break_even_express_days": minimum,
                                       "status": status})
            for penalty, remainder, reduction in product(PENALTIES, NOTIFY_REMAINDERS, EXPRESS_DAYS):
                realised_actions = (policy.NONE, replace(policy.NOTIFY, multiplier=remainder),
                                    replace(policy.EXPRESS, reduction=reduction))
                cost = policy.realised_cost(action, delay, penalty, realised_actions).sum()
                cost_without_action = penalty * np.maximum(delay, 0).sum()
                rows.append({"budget": budget, "penalty": penalty, "notify_remaining": remainder,
                             "express_days": reduction, "strategy": label, "cost": cost,
                             "cost_without_action": cost_without_action, "saving_vs_no_action": cost_without_action - cost})
    scenarios = pd.DataFrame(rows)
    feasible_actions = scenarios["strategy"].isin((*ONLINE, "no action"))
    keys = ["budget", "penalty", "notify_remaining", "express_days"]
    best = scenarios.loc[feasible_actions].groupby(keys)["cost"].min().rename("best_replay_cost")
    scenarios = scenarios.join(best, on=keys)
    scenarios["regret_vs_best_replay"] = scenarios["cost"] - scenarios["best_replay_cost"]
    metadata = {"source_sha256": source_sha256(data_path), "training_cutoff_exclusive": cutoff.isoformat(),
                "train_orders": len(train), "test_orders": len(test), "priority_orders": int(priority.sum()),
                "base_assumptions": {"penalty": BASELINE_PENALTY, "notify_remaining": 0.60, "express_days": 2},
                "stress_settings_per_budget": len(PENALTIES) * len(NOTIFY_REMAINDERS) * len(EXPRESS_DAYS)}
    return scenarios, pd.DataFrame(bounds), pd.DataFrame(thresholds), metadata


def plot_sensitivity(scenarios: pd.DataFrame, thresholds: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    from figure_style import format_figure

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.5), layout="constrained")
    points = scenarios.query("budget == 200000 and penalty == 200 and notify_remaining == 0.60")
    cuts = thresholds.query("penalty == 200 and notify_remaining == 0.60")
    styles = (("#575c60", "s", "--"), ("#8a6147", "^", ":"), ("#315c77", "o", "-"))
    for strategy, (color, marker, linestyle) in zip(ONLINE, styles, strict=True):
        group = points.loc[points.strategy.eq(strategy)]
        left.plot(group.express_days, group.saving_vs_no_action / 1000,
                  ls=linestyle, color=color, marker=marker, label=strategy, ms=5)
        curve = cuts.loc[cuts.strategy.eq(strategy)]
        right.plot(curve.budget / 1000, curve.break_even_express_days,
                   ls=linestyle, color=color, marker=marker, ms=5)
    left.axhline(0, color="#888888", linewidth=.85, linestyle=":")
    left.set(title="A  USD 200,000 action budget", xlabel="Days removed by express",
             ylabel="Savings vs no action (USD thousands)")
    left.legend(fontsize=9)
    right.set(title="B  Break-even effect", xlabel="Action budget (USD thousands)",
              ylabel="Minimum days removed by express")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    format_figure(fig)
    fig.savefig(output, dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "arrival_stress")
    args = parser.parse_args()
    scenarios, bounds, thresholds, metadata = run(pipeline.find_data(args.data))
    deployable = scenarios.loc[scenarios["strategy"].isin((*ONLINE, "no action"))]
    summary = deployable.groupby(["budget", "strategy"], as_index=False).agg(
        settings=("cost", "size"), best_count=("regret_vs_best_replay", lambda x: int(x.eq(0).sum())),
        worst_regret=("regret_vs_best_replay", "max"), worst_saving=("saving_vs_no_action", "min"),
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    for filename, table in (("scenarios", scenarios), ("expected_bounds", bounds),
                            ("break_even", thresholds), ("summary", summary)):
        table.to_csv(args.outdir / f"{filename}.csv", index=False, float_format="%.6f")
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_sensitivity(scenarios, thresholds, args.outdir / "effect_sensitivity.png")
    print(f"Evaluated {len(scenarios)} strategy/scenario rows across {len(BUDGETS)} budgets.")


if __name__ == "__main__":
    main()
