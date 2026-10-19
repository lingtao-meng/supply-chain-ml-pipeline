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

from dataclasses import dataclass

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

#: Support of the delay target in the DataCo dataset (delays are integers -2..4).
ATOMS = np.arange(-2, 5)


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
    penalty,
    actions: tuple[Action, ...] = ACTIONS,
) -> np.ndarray:
    """Per-order cost once the actual delay is known.

    ``penalty`` may be a scalar or one penalty per order (e.g. a late fee that
    scales with order value).  Subsets are masked before the multiplication, so
    an array penalty keeps its alignment.
    """
    delay = np.asarray(delay, dtype=float)
    action_idx = np.asarray(action_idx)
    pen = np.asarray(penalty, dtype=float)
    if pen.ndim and pen.shape[0] != len(delay):
        raise ValueError(f"penalty has {pen.shape[0]} rows but delay has {len(delay)}")

    costs = np.empty(len(delay))
    for i, a in enumerate(actions):
        m = action_idx == i
        if not m.any():
            continue
        p = pen[m] if pen.ndim else pen
        costs[m] = a.cost + p * a.multiplier * np.clip(delay[m] - a.reduction, 0.0, None)
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


def scenario_cost_matrix(
    pmf: np.ndarray,
    penalties,
    express_costs,
    atoms: np.ndarray = ATOMS,
) -> np.ndarray:
    """(n_scenarios, n_orders, n_actions) expected cost under each cost scenario.

    The only thing that varies between scenarios is the daily penalty and the
    price of express; the action menu is always none / notify / express.
    """
    matrices = []
    for penalty in penalties:
        for express_cost in express_costs:
            actions = (NONE, NOTIFY, Action("express", cost=float(express_cost), reduction=2.0))
            matrices.append(expected_cost_from_pmf(pmf, penalty, atoms, actions))
    return np.stack(matrices)


def policy_minimax_regret(
    pmf: np.ndarray,
    penalties,
    express_costs,
    atoms: np.ndarray = ATOMS,
) -> np.ndarray:
    """Pick the action whose worst-case regret over the scenarios is smallest.

    Expected-cost minimisation assumes the planner knows the penalty and the
    express price.  In practice both are negotiated and uncertain, so the
    robust rule asks: *if I am wrong about the economics, how badly can this
    order go?*  Regret is measured against the best action for each scenario.
    """
    costs = scenario_cost_matrix(pmf, penalties, express_costs, atoms)
    best = costs.min(axis=2, keepdims=True)
    regret = costs - best
    worst = regret.max(axis=0)
    return worst.argmin(axis=1)


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


def policy_capacity_pmf(
    pmf: np.ndarray,
    penalty: float,
    capacity: float,
    atoms: np.ndarray = ATOMS,
    actions: tuple[Action, ...] = ACTIONS,
) -> np.ndarray:
    """Capacity-constrained allocation using an exact pmf.

    Same idea as :func:`policy_capacity` but the expected gain comes from the
    full distribution instead of a quantile reconstruction.
    """
    exp_cost = expected_cost_from_pmf(pmf, penalty, atoms, actions)
    gain = exp_cost[:, IDX_NONE] - exp_cost.min(axis=1)
    best_action = exp_cost.argmin(axis=1)
    k = int(round(capacity * len(pmf)))
    order = np.argsort(-gain)[:k]
    out = np.full(len(pmf), IDX_NONE, dtype=int)
    keep = order[gain[order] > 0]
    out[keep] = best_action[keep]
    return out


def policy_budget_pmf(
    pmf: np.ndarray,
    penalty: float,
    budget: float,
    atoms: np.ndarray = ATOMS,
    actions: tuple[Action, ...] = ACTIONS,
) -> np.ndarray:
    """Budget allocation from an exact pmf, as an incremental greedy.

    The naive version ranks (order, action) pairs once and gives each order a
    single action — which throws away upgrades (notify is cheap and captures
    most of the benefit; express is an $80 upgrade on top of it).  Instead,
    start at ``none`` and repeatedly apply the move with the best marginal
    benefit per marginal dollar, among the moves that are currently available:

    * ``none -> notify``       ($20)
    * ``none -> express``      ($100, no notify first)
    * ``notify -> express``    ($80, the upgrade)

    ``budget`` limits action spend, not the penalty bill — exactly how a
    planner's cost centre works.
    """
    import heapq

    exp_cost = expected_cost_from_pmf(pmf, penalty, atoms, actions)
    n = len(pmf)
    current = np.full(n, IDX_NONE, dtype=int)
    remaining = float(budget)

    def push(heap: list, order: int, frm: int, to: int) -> None:
        gain = float(exp_cost[order, frm] - exp_cost[order, to])
        cost = float(actions[to].cost - actions[frm].cost)
        if gain > 0 and cost > 0:
            heapq.heappush(heap, (-gain / cost, order, frm, to))

    heap: list = []
    for order in range(n):
        push(heap, order, IDX_NONE, IDX_NOTIFY)
        push(heap, order, IDX_NONE, IDX_EXPRESS)

    while heap:
        _, order, frm, to = heapq.heappop(heap)
        if current[order] != frm:
            continue  # stale move: the order already moved on
        cost = float(actions[to].cost - actions[frm].cost)
        if cost > remaining + 1e-9:
            continue  # budget only shrinks, so this move never becomes affordable
        current[order] = to
        remaining -= cost
        if to == IDX_NOTIFY:
            push(heap, order, IDX_NOTIFY, IDX_EXPRESS)
    return current


def budget_curve(
    pmf: np.ndarray,
    delay: np.ndarray,
    penalty: float,
    budgets,
    atoms: np.ndarray = ATOMS,
    actions: tuple[Action, ...] = ACTIONS,
) -> pd.DataFrame:
    """Cost and reach as a function of the intervention budget."""
    rows = []
    late = np.asarray(delay) > 0
    exp_cost = expected_cost_from_pmf(pmf, penalty, atoms, actions)
    for budget in budgets:
        idx = policy_budget_pmf(pmf, penalty, budget, atoms, actions)
        cost = realised_cost(idx, delay, penalty, actions)
        touched = idx != IDX_NONE
        expected_total = float(exp_cost[np.arange(len(pmf)), idx].sum())
        spend = float(sum(actions[i].cost for i in idx if i != IDX_NONE))
        rows.append(
            {
                "budget": float(budget),
                "total_cost": float(cost.sum()),
                "expected_total_cost": expected_total,
                "action_spend": spend,
                "orders_touched": int(touched.sum()),
                "orders_notify": int((idx == IDX_NOTIFY).sum()),
                "orders_express": int((idx == IDX_EXPRESS).sum()),
                "intervention_rate": float(touched.mean()),
                "recall_late": float((touched & late).sum() / max(int(late.sum()), 1)),
                "precision": float((touched & late).sum() / max(int(touched.sum()), 1)),
            }
        )
    return pd.DataFrame(rows)


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
        "orders_notify": int((action_idx == IDX_NOTIFY).sum()),
        "orders_express": int((action_idx == IDX_EXPRESS).sum()),
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


# ---------------------------------------------------------------- discrete pmf
# Why this section exists
# -----------------------
# ``quantile_grid`` rebuilds a distribution from three points, which is a poor
# fit for an integer target with a handful of atoms (DataCo delays take only
# the values -2..4).  Working with an explicit pmf fixes two things at once:
#
# * ``E[(D - r)+]`` becomes exact instead of a piecewise-linear approximation;
# * the *shape* of the distribution (flat, skewed, whatever) survives, which is
#   what a decision rule actually cares about.
#
# The functions below are model-agnostic: the pmf can come from a multiclass
# model, from a per-segment lookup, or from a two-part binary model.


def empirical_pmf(values: np.ndarray, atoms: np.ndarray = ATOMS) -> np.ndarray:
    """Normalised histogram of a discrete sample over ``atoms``."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot estimate a pmf from an empty sample")
    return np.array([(values == a).mean() for a in atoms], dtype=float)


def pmf_table(
    groups: np.ndarray, values: np.ndarray, atoms: np.ndarray = ATOMS
) -> tuple[dict, np.ndarray]:
    """Per-group pmfs plus an overall fallback for unseen groups."""
    groups = np.asarray(groups)
    values = np.asarray(values, dtype=float)
    table = {g: empirical_pmf(values[groups == g], atoms) for g in np.unique(groups)}
    return table, empirical_pmf(values, atoms)


def pmf_matrix(groups: np.ndarray, table: dict, default: np.ndarray) -> np.ndarray:
    """Stack the pmf of every row; unseen groups fall back to ``default``."""
    return np.vstack([table.get(g, default) for g in np.asarray(groups)])


def expected_excess_penalty_pmf(
    pmf: np.ndarray, atoms: np.ndarray = ATOMS, reduction: float = 0.0
) -> np.ndarray:
    """``E[max(D - reduction, 0)]`` exactly, from a matrix of pmfs."""
    pmf = np.atleast_2d(np.asarray(pmf, dtype=float))
    tail = np.clip(np.asarray(atoms, dtype=float) - reduction, 0.0, None)
    return pmf @ tail


def expected_cost_from_pmf(
    pmf: np.ndarray,
    penalty: float,
    atoms: np.ndarray = ATOMS,
    actions: tuple[Action, ...] = ACTIONS,
) -> np.ndarray:
    """(n_orders, n_actions) expected cost implied by a pmf."""
    excess = [expected_excess_penalty_pmf(pmf, atoms, a.reduction) for a in actions]
    return np.column_stack(
        [
            a.cost + np.asarray(penalty, dtype=float) * a.multiplier * e
            for a, e in zip(actions, excess)  # noqa: B905 (py3.9 compat)
        ]
    )


def policy_expected_cost_pmf(
    pmf: np.ndarray,
    penalty: float,
    atoms: np.ndarray = ATOMS,
    actions: tuple[Action, ...] = ACTIONS,
) -> np.ndarray:
    """Expected-cost-minimising action when the delay distribution is known."""
    return expected_cost_from_pmf(pmf, penalty, atoms, actions).argmin(axis=1)


def two_part_pmf(
    p_late: np.ndarray, pos_pmf: np.ndarray, neg_pmf: np.ndarray
) -> np.ndarray:
    """Distribution implied by a binary classifier.

    ``P(late)`` comes from the classifier; the delay distribution *within* the
    late and on-time groups is taken from the training window.  Reconstructing
    a pmf this way is what makes the binary baseline comparable: same model,
    same decision rule, less information.
    """
    p = np.asarray(p_late, dtype=float)[:, None]
    pos = np.asarray(pos_pmf, dtype=float)[None, :]
    neg = np.asarray(neg_pmf, dtype=float)[None, :]
    return p * pos + (1.0 - p) * neg


def policy_cost_aware_binary(
    p_late: np.ndarray,
    pos_pmf: np.ndarray,
    neg_pmf: np.ndarray,
    penalty: float,
    atoms: np.ndarray = ATOMS,
    actions: tuple[Action, ...] = ACTIONS,
) -> np.ndarray:
    """Fair binary baseline: keep the classifier, add the same decision layer.

    The original baseline was ``p50 > 0 -> express`` — that conflates a bad
    threshold with a bad model.  This version feeds the classifier's P(late)
    through exactly the same expected-cost rule the quantile policy uses.
    """
    pmf = two_part_pmf(p_late, pos_pmf, neg_pmf)
    return policy_expected_cost_pmf(pmf, penalty, atoms, actions)


def tune_threshold(
    p_late: np.ndarray, delay: np.ndarray, penalty: float, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Pick an express-only probability threshold on validation data.

    Returns ``(tau, validation_cost)``.  This is the *charitable* version of
    the classic baseline: it gets a whole validation set to tune its threshold,
    so a win over it cannot be explained by a badly chosen 0.5 cut-off.
    """
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    p_late = np.asarray(p_late, dtype=float)
    delay = np.asarray(delay, dtype=float)
    best_tau, best_cost = 0.5, np.inf
    for tau in grid:
        idx = np.where(p_late > tau, IDX_EXPRESS, IDX_NONE)
        cost = float(realised_cost(idx, delay, penalty).sum())
        if cost < best_cost:
            best_tau, best_cost = float(tau), cost
    return best_tau, best_cost


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Paired bootstrap of ``mean(a - b)`` with a percentile confidence interval.

    A 3% cost gap on 27k orders is only interesting if the interval excludes
    zero, so every headline delta in the report carries one of these.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }


def action_mix(action_idx: np.ndarray, actions: tuple[Action, ...] = ACTIONS) -> dict:
    """Share of orders sent to each action."""
    idx = np.asarray(action_idx)
    return {a.name: float((idx == i).mean()) for i, a in enumerate(actions)}
