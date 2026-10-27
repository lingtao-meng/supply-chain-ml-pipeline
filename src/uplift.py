"""Heterogeneous treatment effects (uplift) and targeting.

The policy so far assumes one effect per action: express removes two days for
everyone.  In reality the effect varies by order, and targeting the orders
where it is largest is where the money is.

This module provides the standard toolkit:

* a T-learner (one outcome model per arm) that predicts the conditional
  average treatment effect (CATE);
* the Qini curve and its area (AUUC) — the uplift analogue of the ROC curve;
* a cost-based targeting curve: ``penalty * effect - action_cost`` for the
  top-k orders by each score.

The simulator lives in ``scripts/uplift_analysis.py`` because it needs the
business effect assumptions; the estimators here are model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class UpliftModel:
    """T-learner: separate outcome models for control and treatment."""

    control_model: HistGradientBoostingRegressor
    treatment_model: HistGradientBoostingRegressor

    def predict_cate(self, X) -> np.ndarray:
        """Control outcome minus treated outcome: positive = delay reduction."""
        return self.control_model.predict(X) - self.treatment_model.predict(X)


def fit_t_learner(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    random_state: int = 42,
) -> UpliftModel:
    """Fit one regressor per arm on the same feature matrix."""
    treatment = np.asarray(treatment).astype(int)
    common = dict(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=random_state,
    )
    control = HistGradientBoostingRegressor(**common).fit(X[treatment == 0], outcome[treatment == 0])
    treated = HistGradientBoostingRegressor(**common).fit(X[treatment == 1], outcome[treatment == 1])
    return UpliftModel(control, treated)


def qini_curve(cate: np.ndarray, effect: np.ndarray) -> pd.DataFrame:
    """Cumulative uplift by descending CATE, plus the random/perfect references.

    At every fraction treated, the Qini value is the extra effect captured
    above what a random selection of the same size would capture.
    """
    cate = np.asarray(cate, dtype=float)
    effect = np.asarray(effect, dtype=float)
    n = len(effect)
    total = float(effect.sum())
    fractions = np.linspace(0, 1, n + 1)[1:]

    def curve(order):
        cumulative = np.cumsum(effect[order])
        return cumulative - fractions * total

    model = np.concatenate([[0.0], curve(np.argsort(-cate))])
    perfect = np.concatenate([[0.0], curve(np.argsort(-effect))])
    return pd.DataFrame(
        {
            "fraction": np.concatenate([[0.0], fractions]),
            "qini": model,
            "perfect": perfect,
            "random": 0.0,
        }
    )


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Version-independent trapezoidal rule (numpy 1.x and 2.x)."""
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def auuc(curve: pd.DataFrame) -> float:
    """Normalised area under the Qini curve (1.0 = perfect ranking)."""
    qini_area = _trapezoid(curve["qini"].to_numpy(), curve["fraction"].to_numpy())
    perfect_area = _trapezoid(curve["perfect"].to_numpy(), curve["fraction"].to_numpy())
    if perfect_area <= 0:
        return float("nan")
    return qini_area / perfect_area


def targeting_curve(
    scores: np.ndarray,
    effect: np.ndarray,
    penalty: float,
    action_cost: float,
    base_delay=None,
    fractions=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00),
) -> pd.DataFrame:
    """Net saving when treating the top-k orders by a score.

    ``effect`` is the *true* per-order reduction in days; the curve therefore
    measures the decision quality of the ranking, not of the outcome model.
    """
    scores = np.asarray(scores, dtype=float)
    effect = np.asarray(effect, dtype=float)
    n = len(effect)
    rows = []
    for fraction in fractions:
        k = max(1, int(round(fraction * n)))
        selected = np.argsort(-scores)[:k]
        if base_delay is None:
            # 无对照延迟时只能给上界：早期订单本来就不赔钱，这里却按全额计入
            benefit = float(penalty * effect[selected].sum())
        else:
            d = np.asarray(base_delay, dtype=float)[selected]
            e = effect[selected]
            benefit = float(penalty * (np.maximum(d, 0.0) - np.maximum(d - e, 0.0)).sum())
        cost = float(action_cost * k)
        rows.append(
            {
                "fraction": fraction,
                "orders_treated": k,
                "benefit": benefit,
                "action_cost": cost,
                "net_saving": benefit - cost,
            }
        )
    return pd.DataFrame(rows)
