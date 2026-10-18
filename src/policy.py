"""Unified cost evaluation for late-delivery intervention policies.

The key design decision of this module: **every policy is scored with the same
cost function**, so "no intervention", "intervene on everything" and any
model-driven policy are directly comparable.  The earlier notebook version
scored each policy with a different formula, which made the "no intervention"
baseline look cheaper than it really is.

Cost of one order under an action ``a``::

    cost_i = a.cost + penalty * a.multiplier * max(delay_i - a.reduction, 0)

where ``delay_i`` is the *realised* delay in days.  Evaluating on realised
outcomes keeps the comparison honest: the policy only decides *who* gets an
action, the data decides what happens next.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- actions


@dataclass(frozen=True)
class Action:
    """An intervention tier available to the planner."""

    name: str
    cost: float          # $ per order touched
    reduction: float = 0.0   # days of delay removed
    multiplier: float = 1.0  # remaining penalty exposure


NONE = Action("none", cost=0.0)
NOTIFY = Action("notify", cost=20.0, reduction=0.0, multiplier=0.60)
EXPRESS = Action("express", cost=100.0, reduction=2.0)

ACTIONS: tuple[Action, ...] = (NONE, NOTIFY, EXPRESS)
IDX_NONE, IDX_NOTIFY, IDX_EXPRESS = 0, 1, 2


# ---------------------------------------------------------------- expected penalty

_QGRID = np.linspace(0.005, 0.995, 61)


def quantile_grid(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> np.ndarray:
    """Piecewise-linear inverse CDF through (0.1, p10), (0.5, p50), (0.9, p90).

    Tails are extrapolated with the same slope as the adjacent segment, which
    keeps the approximation conservative without assuming a parametric family.
    """
    p10, p50, p90 = (np.asarray(v, dtype=float) for v in (p10, p50, p90))
    # enforce monotone quantiles (quantile regression can cross)
    p50 = np.maximum(p50, p10)
    p90 = np.maximum(p90, p50)

    low_slope = 3.0 * (p50 - p10)   # per unit of probability below 0.1
    high_slope = 3.0 * (p90 - p50)  # per unit of probability above 0.9

    out = np.empty((len(p10), len(_QGRID)))
    u = _QGRID
    lo = u < 0.1
    mid1 = (u >= 0.1) & (u < 0.5)
    mid2 = (u >= 0.5) & (u < 0.9)
    hi = u >= 0.9

    out[:, lo] = p10[:, None] - (0.1 - u[lo])[None, :] * low_slope[:, None]
    out[:, mid1] = p10[:, None] + (u[mid1] - 0.1)[None, :] / 0.4 * (p50 - p10)[:, None]
    out[:, mid2] = p50[:, None] + (u[mid2] - 0.5)[None, :] / 0.4 * (p90 - p50)[:, None]
    out[:, hi] = p90[:, None] + (u[hi] - 0.9)[None, :] * high_slope[:, None]
    return out


def expected_excess_penalty(draws: np.ndarray, reduction: float = 0.0) -> np.ndarray:
    """E[max(D - reduction, 0)] for every row of ``draws``."""
    return np.clip(draws - reduction, 0.0, None).mean(axis=1)


def expected_cost_per_action(
    draws: np.ndarray, penalty: float, actions: tuple[Action, ...] = ACTIONS
) -> np.ndarray:
    """(n_orders, n_actions) matrix of expected cost."""
    cols = [
        a.cost + penalty * a.multiplier * expected_excess_penalty(draws, a.reduction)
        for a in actions
    ]
    return np.column_stack(cols)


# ---------------------------------------------------------------- realised cost


def realised_cost(
    action_idx: np.ndarray,
    delay: np.ndarray,
    penalty: float,
    actions: tuple[Action, ...] = ACTIONS,
) -> np.ndarray:
    """Per-order cost once the actual delay is known."""
    costs = np.empty(len(delay))
    for i, a in enumerate(actions):
        m = action_idx == i
        if not m.any():
            continue
        costs[m] = a.cost + penalty * a.multiplier * np.clip(
            delay[m] - a.reduction, 0.0, None
        )
    return costs


# ---------------------------------------------------------------- policies


def policy_none(n: int, **_kw) -> np.ndarray:
    return np.full(n, IDX_NONE)


def policy_all_express(n: int, **_kw) -> np.ndarray:
    return np.full(n, IDX_EXPRESS)


def policy_binary(p50: np.ndarray, **_kw) -> np.ndarray:
    """Classic baseline: flag anything predicted to be late, send express."""
    return np.where(p50 > 0, IDX_EXPRESS, IDX_NONE).astype(int)


def policy_expected_cost(
    draws: np.ndarray, penalty: float, actions: tuple[Action, ...] = ACTIONS
) -> np.ndarray:
    """Expected-cost-minimising action per order (the newsvendor-style rule)."""
    return expected_cost_per_action(draws, penalty, actions).argmin(axis=1)


def policy_budget(
    draws: np.ndarray,
    penalty: float,
    budget: float,
    actions: tuple[Action, ...] = ACTIONS,
) -> np.ndarray:
    """Greedy budget-constrained allocation.

    Rank every (order, action) pair by expected net benefit per dollar spent,
    then take pairs in that order while the budget lasts.  This is the LP
    relaxation of the 0/1 knapsack, so it is a valid upper bound for the
    integer allocation and a realistic planner heuristic.
    """
    exp_cost = expected_cost_per_action(draws, penalty, actions)  # (n, k)
    base = exp_cost[:, IDX_NONE]
    benefit = base[:, None] - exp_cost                     # (n, k)
    cost = np.broadcast_to(np.array([a.cost for a in actions]), benefit.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cost > 0, benefit / np.maximum(cost, 1e-9), np.inf)

    valid = (benefit > 0) & (cost > 0)
    order_flat = np.repeat(np.arange(len(draws)), len(actions))
    action_flat = np.tile(np.arange(len(actions)), len(draws))
    ratio_flat = ratio.ravel()
    cost_flat = cost.ravel()
    valid_flat = valid.ravel()

    order_flat = order_flat[valid_flat]
    action_flat = action_flat[valid_flat]
    ratio_flat = ratio_flat[valid_flat]
    cost_flat = cost_flat[valid_flat]

    order = np.argsort(-ratio_flat, kind="stable")
    chosen = np.full(len(draws), IDX_NONE, dtype=int)
    taken = np.zeros(len(draws), dtype=bool)
    spent = 0.0
    for i in order:
        o = order_flat[i]
        if taken[o]:
            continue
        c = cost_flat[i]
        if spent + c > budget:
            continue
        taken[o] = True
        chosen[o] = action_flat[i]
        spent += c
    return chosen


def policy_capacity(
    draws: np.ndarray,
    penalty: float,
    capacity: float,
    actions: tuple[Action, ...] = ACTIONS,
) -> np.ndarray:
    """Capacity-constrained allocation: you can only expedite ``capacity`` of orders.

    Rank orders by expected cost saving and take the best ``capacity * n`` — the
    realistic version of "there are only so many trucks".  Orders whose best
    action does not pay for itself are left alone.
    """
    exp_cost = expected_cost_per_action(draws, penalty, actions)
    base = exp_cost[:, IDX_NONE]
    best_action = exp_cost.argmin(axis=1)
    gain = base - exp_cost.min(axis=1)

    k = int(round(capacity * len(draws)))
    order = np.argsort(-gain)[:k]
    out = np.full(len(draws), IDX_NONE, dtype=int)
    keep = order[gain[order] > 0]
    out[keep] = best_action[keep]
    return out


def policy_oracle(
    delay: np.ndarray, penalty: float, actions: tuple[Action, ...] = ACTIONS
) -> np.ndarray:
    """Hindsight optimum — the lower bound on achievable cost."""
    n, k = len(delay), len(actions)
    costs = np.empty((n, k))
    for i, a in enumerate(actions):
        costs[:, i] = a.cost + penalty * a.multiplier * np.clip(
            delay - a.reduction, 0.0, None
        )
    return costs.argmin(axis=1)


# ---------------------------------------------------------------- scoring


def score_policy(
    name: str,
    action_idx: np.ndarray,
    delay: np.ndarray,
    penalty: float,
    budget: float | None = None,
) -> dict:
    cost = realised_cost(action_idx, delay, penalty)
    late = delay > 0
    touched = action_idx != IDX_NONE
    caught = int((touched & late).sum())
    n_touched = int(touched.sum())

    return {
        "policy": name,
        "total_cost": float(cost.sum()),
        "cost_per_order": float(cost.mean()),
        "intervention_rate": n_touched / len(delay),
        "orders_touched": n_touched,
        "recall_late": caught / max(int(late.sum()), 1),
        "precision": caught / max(n_touched, 1),
        "spend": float(budget) if budget is not None else None,
    }


def compare(
    policies: dict[str, np.ndarray],
    delay: np.ndarray,
    penalty: float,
) -> pd.DataFrame:
    rows = [score_policy(n, idx, delay, penalty) for n, idx in policies.items()]
    df = pd.DataFrame(rows)
    best = df["total_cost"].min()
    df["cost_gap_vs_best"] = df["total_cost"] - best
    return df


def sensitivity(
    draws: np.ndarray,
    delay: np.ndarray,
    p50: np.ndarray,
    penalties: list[float],
    express_costs: list[float],
) -> pd.DataFrame:
    """How the policy ranking moves as the two cost parameters change.

    For every (penalty, express cost) pair we recompute four policies and report
    the cost of each — this is the table that tells you *when* the model-driven
    rule is worth the complexity.
    """
    rows = []
    for p in penalties:
        for c in express_costs:
            actions = (NONE, NOTIFY, Action("express", cost=c, reduction=2.0))
            idx_exp = expected_cost_per_action(draws, p, actions).argmin(axis=1)
            idx_bin = np.where(p50 > 0, IDX_EXPRESS, IDX_NONE).astype(int)
            idx_all = np.full(len(delay), IDX_EXPRESS, dtype=int)
            idx_non = np.full(len(delay), IDX_NONE, dtype=int)

            cost = {
                "none": realised_cost(idx_non, delay, p, actions).sum(),
                "all_express": realised_cost(idx_all, delay, p, actions).sum(),
                "binary": realised_cost(idx_bin, delay, p, actions).sum(),
                "expected_cost": realised_cost(idx_exp, delay, p, actions).sum(),
            }
            rows.append(
                {
                    "penalty_per_day": p,
                    "express_cost": c,
                    "cost_none": cost["none"],
                    "cost_all_express": cost["all_express"],
                    "cost_binary": cost["binary"],
                    "cost_expected": cost["expected_cost"],
                    "vs_binary_pct": 1 - cost["expected_cost"] / cost["binary"],
                    "vs_none_pct": 1 - cost["expected_cost"] / cost["none"],
                    "intervention_rate": float((idx_exp != IDX_NONE).mean()),
                }
            )
    return pd.DataFrame(rows)


def capacity_curve(
    draws: np.ndarray,
    delay: np.ndarray,
    penalty: float,
    capacities: list[float],
) -> pd.DataFrame:
    """Cost as a function of how many orders you are able to expedite."""
    rows = []
    for cap in capacities:
        idx = policy_capacity(draws, penalty, cap)
        m = score_policy(f"capacity_{cap:.0%}", idx, delay, penalty)
        m["capacity"] = cap
        rows.append(m)
    df = pd.DataFrame(rows)
    df["marginal_saving_per_pp"] = -df["total_cost"].diff() / (df["capacity"].diff() * 100)
    return df


def summarise_table(df: pd.DataFrame) -> str:
    """Markdown table for the README / results folder."""
    cols = [
        "policy",
        "total_cost",
        "cost_per_order",
        "intervention_rate",
        "recall_late",
        "precision",
    ]
    view = df[cols].copy()
    view["total_cost"] = view["total_cost"].map(lambda v: f"${v:,.0f}")
    view["cost_per_order"] = view["cost_per_order"].map(lambda v: f"${v:,.2f}")
    view["intervention_rate"] = view["intervention_rate"].map(lambda v: f"{v:.1%}")
    view["recall_late"] = view["recall_late"].map(lambda v: f"{v:.1%}")
    view["precision"] = view["precision"].map(lambda v: f"{v:.1%}")
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in view.values)
    return "\n".join([header, sep, body])
