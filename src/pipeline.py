"""Data pipeline for the DataCo late-delivery project.

Reproduces the feature engineering / time split used in
``notebooks/supply_chain_ml_pipeline.ipynb`` so that the cost-evaluation
scripts work on exactly the same data as the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

CAT_COLS = [
    "Type",
    "Category Name",
    "Department Name",
    "Market",
    "Shipping Mode",
    "Customer Segment",
]

DATE_COL = "order date (DateOrders)"
SHIP_COL = "shipping date (DateOrders)"
SCHED_COL = "Days for shipment (scheduled)"


@dataclass
class SplitData:
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]
    train_index: np.ndarray
    test_index: np.ndarray


def load_orders(path: str) -> pd.DataFrame:
    """Load DataCo and attach the late-delivery target (``delay_days``).

    ``delay_days`` is measured *after* the fact: it is what we are trying to
    predict, never a model input.
    """
    df = pd.read_csv(path, encoding="latin1")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[SHIP_COL] = pd.to_datetime(df[SHIP_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["delay_days"] = (
        (df[SHIP_COL] - df[DATE_COL]).dt.days - df[SCHED_COL]
    )
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Order-time-only features (no leakage from post-shipment fields)."""
    feat = df[CAT_COLS].copy()
    od = df[DATE_COL]

    feat["order_month"] = od.dt.month
    feat["order_dayofweek"] = od.dt.dayofweek
    feat["order_quarter"] = od.dt.quarter
    feat["order_is_weekend"] = (od.dt.dayofweek >= 5).astype(int)
    feat["order_dayofyear"] = od.dt.dayofyear

    qty = df["Order Item Quantity"]
    price = df["Order Item Product Price"]
    sched = df[SCHED_COL]
    disc_rate = df["Order Item Discount Rate"]

    feat["total_value"] = qty * price
    feat["discount_abs"] = df["Order Item Total"] * disc_rate
    feat["value_per_ship_day"] = feat["total_value"] / (sched + 1)
    feat["is_high_discount"] = (disc_rate > 0.1).astype(int)
    feat["is_bulk"] = (qty > 5).astype(int)
    feat["discount_x_value"] = disc_rate * feat["total_value"]
    feat["qty_x_ship"] = qty * sched

    for col in CAT_COLS:
        feat[f"{col}_enc"] = LabelEncoder().fit_transform(feat[col].astype(str))

    feat = feat.drop(columns=CAT_COLS).fillna(0.0)

    y = df["delay_days"].to_numpy(dtype=float)
    return feat, y


def time_split(
    X: pd.DataFrame, y: np.ndarray, train_frac: float = 0.70, valid_frac: float = 0.15
) -> SplitData:
    """Chronological split — train on the past, test on the future."""
    n = len(X)
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + valid_frac))

    X_tr, X_va, X_te = X.iloc[:t_end], X.iloc[t_end:v_end], X.iloc[v_end:]
    y_tr, y_va, y_te = y[:t_end], y[t_end:v_end], y[v_end:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    return SplitData(
        X_train=X_tr_s,
        X_valid=X_va_s,
        X_test=X_te_s,
        y_train=y_tr,
        y_valid=y_va,
        y_test=y_te,
        scaler=scaler,
        feature_names=list(X.columns),
        train_index=np.arange(t_end),
        test_index=np.arange(v_end, n),
    )


def describe(df: pd.DataFrame) -> dict:
    """Small facts worth printing at the top of any report."""
    late = df["delay_days"] > 0
    return {
        "orders": int(len(df)),
        "late_rate": float(late.mean()),
        "mean_delay": float(df["delay_days"].mean()),
        "total_delay_days": float(np.clip(df["delay_days"], 0, None).sum()),
        "date_min": str(df[DATE_COL].min().date()),
        "date_max": str(df[DATE_COL].max().date()),
    }
