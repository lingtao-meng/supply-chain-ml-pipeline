"""A/B experiment analysis for the intervention effect.

The policy layer needs one number it cannot get from observational data:
**how many days does an intervention actually remove?**  This module is the
small, honest bridge:

* Welch's test for the reduction in delay (treated vs control), with a CI;
* a sample-ratio-mismatch (SRM) check — a broken randomisation invalidates
  everything downstream;
* per-segment effects, because a global average can hide a segment where the
  intervention does nothing;
* a rollout decision that compares the *lower* confidence bound with the
  break-even effect from the cost model.

``simulate_ab`` generates a synthetic experiment from the observed delay
distribution, so the estimator itself can be validated before real data
exists.  It is not a substitute for running the experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class EffectEstimate:
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    reduction: float
    ci_low: float
    ci_high: float
    p_value: float
    relative_reduction: float

    def as_dict(self) -> dict:
        return asdict(self)


def welch_reduction(control, treatment, alpha: float = 0.05) -> EffectEstimate:
    """Reduction in delay (control − treated) with a Welch CI.

    Positive ``reduction`` means the treatment shortened the delay.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("need at least two observations per arm")

    mean_c, mean_t = float(control.mean()), float(treatment.mean())
    var_c, var_t = float(control.var(ddof=1)), float(treatment.var(ddof=1))
    se = np.sqrt(var_c / len(control) + var_t / len(treatment))
    reduction = mean_c - mean_t

    if se == 0:
        return EffectEstimate(
            n_control=len(control),
            n_treatment=len(treatment),
            mean_control=mean_c,
            mean_treatment=mean_t,
            reduction=reduction,
            ci_low=reduction,
            ci_high=reduction,
            p_value=0.0 if reduction != 0 else 1.0,
            relative_reduction=reduction / mean_c if mean_c != 0 else float("nan"),
        )

    df = se**4 / (
        (var_c / len(control)) ** 2 / (len(control) - 1)
        + (var_t / len(treatment)) ** 2 / (len(treatment) - 1)
    )
    t_stat = reduction / se
    p_value = float(2 * stats.t.sf(abs(t_stat), df))
    critical = float(stats.t.ppf(1 - alpha / 2, df))
    return EffectEstimate(
        n_control=len(control),
        n_treatment=len(treatment),
        mean_control=mean_c,
        mean_treatment=mean_t,
        reduction=reduction,
        ci_low=reduction - critical * se,
        ci_high=reduction + critical * se,
        p_value=p_value,
        relative_reduction=reduction / mean_c if mean_c != 0 else float("nan"),
    )


def srm_test(n_control: int, n_treatment: int, expected_share: float = 0.5) -> dict:
    """Sample-ratio-mismatch chi-square test.

    A tiny p-value means the randomisation is broken and the comparison is
    not trustworthy, whatever the effect estimate says.
    """
    total = n_control + n_treatment
    expected = np.array([total * expected_share, total * (1 - expected_share)])
    observed = np.array([n_control, n_treatment], dtype=float)
    if total == 0:
        return {"n_control": n_control, "n_treatment": n_treatment, "p_value": float("nan"), "ok": False}
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p_value = float(stats.chi2.sf(chi2, df=1))
    return {
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "observed_share": float(n_treatment / total),
        "p_value": p_value,
        "ok": bool(p_value > 0.01),
    }


def analyze_by_group(
    frame: pd.DataFrame,
    group: str = "shipping_mode",
    outcome: str = "delay_days",
    treated: str = "treated",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-segment effect estimates; segments with a single arm are skipped."""
    rows = []
    for value, chunk in frame.groupby(group, sort=True):
        control = chunk.loc[chunk[treated] == 0, outcome]
        treatment = chunk.loc[chunk[treated] == 1, outcome]
        if len(control) < 2 or len(treatment) < 2:
            continue
        estimate = welch_reduction(control, treatment, alpha).as_dict()
        estimate[group] = value
        rows.append(estimate)
    columns = [group, *[c for c in EffectEstimate.__dataclass_fields__ if c != group]]
    return pd.DataFrame(rows, columns=columns)


def simulate_ab(
    delay_by_group: dict,
    effect_by_group: dict,
    n_per_arm: int = 1000,
    seed: int = 0,
    noise: float = 0.0,
) -> pd.DataFrame:
    """Generate a synthetic A/B experiment from observed delay pools.

    Control draws from the observed pool; treatment draws from the same pool
    and shifts it by ``effect`` days (plus optional Gaussian noise).  This lets
    us check CI coverage and the decision rule before real data exists.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for group, pool in delay_by_group.items():
        pool = np.asarray(pool, dtype=float)
        effect = float(effect_by_group.get(group, 0.0))
        control = rng.choice(pool, size=n_per_arm, replace=True)
        treated = rng.choice(pool, size=n_per_arm, replace=True) - effect
        if noise > 0:
            treated = treated + rng.normal(0, noise, size=n_per_arm)
        rows.append(
            pd.DataFrame(
                {
                    "shipping_mode": group,
                    "delay_days": control,
                    "treated": 0,
                    "true_effect": 0.0,
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "shipping_mode": group,
                    "delay_days": treated,
                    "treated": 1,
                    "true_effect": effect,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def rollout_decision(ci_low: float, ci_high: float, break_even: float) -> str:
    """Compare the CI of the effect with the break-even effect."""
    if not np.isfinite(break_even):
        return "never pays"
    if ci_low > break_even:
        return "roll out"
    if ci_high < break_even:
        return "do not roll out"
    return "inconclusive"


def net_value(
    reduction: float,
    penalty: float,
    express_cost: float,
    control_delay=None,
) -> float:
    """Expected saving per treated order, with the penalty floored at zero.

    ``penalty * reduction`` is only an *upper bound*: an order that arrives
    early (delay <= 0) already pays nothing, so shortening it further cannot
    save a cent.  Pass the control-arm realised delays to get the honest
    number; without them the function returns that upper bound and says so in
    the docstring, because silently overstating savings is worse than a bound.
    """
    if control_delay is None:
        return float(penalty * reduction - express_cost)
    d = np.asarray(control_delay, dtype=float)
    before = np.maximum(d, 0.0)
    after = np.maximum(d - reduction, 0.0)
    return float(penalty * float((before - after).mean()) - express_cost)
