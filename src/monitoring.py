"""Population-stability monitoring and retraining recommendations.

``drift_report.py`` produces the monthly picture; this module is the reusable
core plus the decision rule a cron job needs:

* PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant;
* **target** drift means the world changed -> retrain;
* **feature-only** drift with a stable target means the mix changed ->
  reweight or split by segment, do not retrain.

The distinction matters: on DataCo the market mix drifts heavily while the
delay distribution is flat, so a naive "feature PSI high -> retrain" alarm
would burn compute for no gain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE = 0.1
PSI_HIGH = 0.25

DEFAULT_CATEGORICAL = ["Shipping Mode", "Market", "Category Name", "Customer Segment", "Type"]
DEFAULT_NUMERIC = {"order_value": "Order Item Total", "delay_days": "delay_days"}
DEFAULT_TARGET = "delay_days"


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    """Population stability index between two count/share vectors."""
    expected = np.clip(np.asarray(expected, dtype=float), eps, None)
    actual = np.clip(np.asarray(actual, dtype=float), eps, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    expected = np.array([(reference == cat).mean() for cat in categories])
    actual = np.array([(current == cat).mean() for cat in categories])
    return psi(expected, actual)


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI on reference-quantile bins; constant references get a tolerance band."""
    reference = reference.dropna().to_numpy(dtype=float)
    current = current.dropna().to_numpy(dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # constant (or near-constant) reference
        centre = float(reference[0])
        spread = max(float(np.std(reference)), 0.5)
        edges = np.array([-np.inf, centre - spread, centre + spread, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    expected = np.histogram(reference, bins=edges)[0]
    actual = np.histogram(current, bins=edges)[0]
    return psi(expected, actual)


def alert_level(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value > PSI_HIGH:
        return "high"
    if value > PSI_MODERATE:
        return "moderate"
    return "stable"


def recommend(max_feature_psi: float, target_psi: float) -> dict:
    """Turn two PSI numbers into an operational action."""
    if np.isfinite(target_psi) and target_psi > PSI_HIGH:
        return {
            "action": "retrain",
            "reason": "target distribution moved (concept drift)",
            "exit_code": 2,
        }
    if np.isfinite(max_feature_psi) and max_feature_psi > PSI_HIGH:
        return {
            "action": "reweight",
            "reason": "feature drift with a stable target (composition drift)",
            "exit_code": 1,
        }
    if max(max_feature_psi, target_psi, 0.0) > PSI_MODERATE:
        return {"action": "monitor", "reason": "moderate drift", "exit_code": 0}
    return {"action": "stable", "reason": "no material drift", "exit_code": 0}


def monitor_frames(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    categorical=None,
    numeric=None,
    target: str = DEFAULT_TARGET,
) -> dict:
    """PSI report + recommendation for two data slices."""
    categorical = DEFAULT_CATEGORICAL if categorical is None else categorical
    numeric = DEFAULT_NUMERIC if numeric is None else numeric

    features = {}
    for feature in categorical:
        if feature in reference.columns and feature in current.columns:
            features[feature] = categorical_psi(reference[feature], current[feature])
    for name, column in numeric.items():
        if column in reference.columns and column in current.columns:
            features[name] = numeric_psi(reference[column], current[column])

    feature_psi = {k: v for k, v in features.items() if k != target}
    max_feature = max(feature_psi.values(), default=0.0)
    target_psi = features.get(target, float("nan"))
    decision = recommend(max_feature, target_psi)
    return {
        "features": {k: float(v) for k, v in features.items()},
        "alerts": {k: alert_level(v) for k, v in features.items()},
        "max_feature_psi": float(max_feature),
        "target_psi": float(target_psi) if np.isfinite(target_psi) else None,
        "recommendation": decision["action"],
        "reason": decision["reason"],
        "exit_code": decision["exit_code"],
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
    }
