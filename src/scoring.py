"""Shared scoring logic for the CLI, the API and the Streamlit app.

Everything that turns raw orders into predictions / actions lives here, so the
three entry points cannot drift apart:

* ``scripts/score_orders.py``  — batch CSV
* ``api/main.py``              — HTTP service
* ``app/app.py``               — interactive demo
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import pipeline, policy

ROOT = Path(__file__).resolve().parents[1]
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


def load_bundle(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else DEFAULT_BUNDLE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it first with: python scripts/refresh_cost_analysis.py"
        )
    return joblib.load(path)


def actions_from_bundle(bundle: dict) -> tuple[policy.Action, ...]:
    return tuple(policy.Action(**spec) for spec in bundle["params"]["actions"])


def pmf_from_model(bundle: dict, features: np.ndarray) -> np.ndarray:
    """7-atom pmf for every row (model probabilities aligned to ``ATOMS``)."""
    proba = bundle["pmf_model"].predict_proba(features)
    atoms = np.asarray(bundle["atoms"])
    pmf = np.zeros((len(features), len(atoms)))
    for j, cls in enumerate(bundle["pmf_model"].classes_):
        hits = np.where(atoms == int(cls))[0]
        if hits.size:
            pmf[:, hits[0]] = proba[:, j]
    return pmf


def prepare_features(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Validate, parse dates, build the 18 features and scale them."""
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(frame[pipeline.DATE_COL]):
        frame = frame.copy()
        frame[pipeline.DATE_COL] = pd.to_datetime(frame[pipeline.DATE_COL])

    raw = pipeline.make_features(frame, bundle["encoders"])
    return bundle["scaler"].transform(raw)


def predict_pmf(frame: pd.DataFrame, bundle: dict, use_history: bool = False) -> np.ndarray:
    """Distribution over delay atoms for every row."""
    if use_history:
        table, default = bundle["mode_pmf"], bundle["mode_pmf_default"]
        return np.vstack([table.get(mode, default) for mode in frame["Shipping Mode"]])
    return pmf_from_model(bundle, prepare_features(frame, bundle))


def cost_table(pmf: np.ndarray, penalty: float, actions: tuple[policy.Action, ...]) -> pd.DataFrame:
    """Expected cost of every action for a single order (used by the app)."""
    atoms = np.asarray(policy.ATOMS, dtype=float)
    rows = []
    for action in actions:
        excess = float(np.clip(atoms - action.reduction, 0.0, None) @ np.asarray(pmf))
        rows.append(
            {
                "action": action.name,
                "fixed_cost": float(action.cost),
                "expected_remaining_delay": excess,
                "expected_total_cost": float(action.cost + penalty * action.multiplier * excess),
            }
        )
    return pd.DataFrame(rows)


def score_frame(
    frame: pd.DataFrame,
    bundle: dict,
    penalty: float | None = None,
    budget: float | None = None,
    use_history: bool = False,
) -> pd.DataFrame:
    """Return ``frame`` plus predictions and the recommended action."""
    features = prepare_features(frame, bundle)
    pmf = (
        np.vstack(
            [
                bundle["mode_pmf"].get(mode, bundle["mode_pmf_default"])
                for mode in frame["Shipping Mode"]
            ]
        )
        if use_history
        else pmf_from_model(bundle, features)
    )

    actions = actions_from_bundle(bundle)
    penalty = float(bundle["params"]["penalty"] if penalty is None else penalty)
    expected_cost = policy.expected_cost_from_pmf(pmf, penalty, actions=actions)
    if budget is not None:
        chosen = policy.policy_budget_pmf(pmf, penalty, budget, actions=actions)
    else:
        chosen = expected_cost.argmin(axis=1)

    quantiles = {
        q: np.clip(model.predict(features), -5, 10) for q, model in bundle["quantile"].items()
    }

    out = frame.copy()
    out["p_late"] = bundle["binary"].predict_proba(features)[:, 1]
    out["p10_delay"] = quantiles[0.1]
    out["p50_delay"] = quantiles[0.5]
    out["p90_delay"] = quantiles[0.9]
    for i, action in enumerate(actions):
        out[f"expected_cost_{action.name}"] = expected_cost[:, i]
    out["recommended_action"] = [actions[i].name for i in chosen]
    out["expected_saving"] = (
        expected_cost[:, policy.IDX_NONE] - expected_cost[np.arange(len(out)), chosen]
    )
    return out
