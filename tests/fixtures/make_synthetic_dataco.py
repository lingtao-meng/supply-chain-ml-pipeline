"""Generate a tiny DataCo-shaped dataset for CI and unit tests.

The real file is 91 MB and lives on Kaggle, so CI cannot use it.  This
generator reproduces the *structure* that matters: integer delays that are
uniform inside each shipping mode, exactly like the real data, plus enough
categorical variety for the encoders and the per-mode analysis.

Usage::

    python tests/fixtures/make_synthetic_dataco.py /tmp/dataco_small.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# shipping mode -> (scheduled days, possible delays)
MODES = {
    "Same Day": (0, [0]),
    "First Class": (1, [1]),
    "Second Class": (2, [0, 1, 2, 3, 4]),
    "Standard Class": (4, [-2, -1, 0, 1, 2]),
}
MARKETS = ["Europe", "LATAM", "Pacific Asia"]
CATEGORIES = ["Fishing", "Cleats", "Camping & Hiking"]
DEPARTMENTS = ["Apparel", "Fan Shop"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
TYPES = ["DEBIT", "TRANSFER", "CASH"]


def generate_frame(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic DataCo-shaped frame with order / order-line identifiers.

    Orders are generated first and then expanded into 1–3 lines each, so that
    date, shipping mode and delay are constant within an ``Order Id`` — the
    same invariant the real dataset has.  ``to_order_level`` depends on it, so
    the fixture must exercise it (a 1-line-per-order fixture would never catch
    a regression in the aggregation).
    """
    rng = np.random.default_rng(seed)
    n_orders = max(n // 2, 1)
    order_date = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 700, n_orders), unit="D")
    mode = rng.choice(list(MODES), size=n_orders, p=[0.05, 0.15, 0.2, 0.6])
    scheduled = np.array([MODES[m][0] for m in mode])
    delay = np.array([rng.choice(MODES[m][1]) for m in mode])

    orders = pd.DataFrame(
        {
            "Order Id": np.arange(1, n_orders + 1),
            "Type": rng.choice(TYPES, n_orders),
            "Days for shipping (real)": scheduled + delay,
            "Days for shipment (scheduled)": scheduled,
            "Delivery Status": np.where(delay > 0, "Late delivery", "Shipping on time"),
            "Late_delivery_risk": (delay > 0).astype(int),
            "Category Name": rng.choice(CATEGORIES, n_orders),
            "Customer Segment": rng.choice(SEGMENTS, n_orders),
            "Department Name": rng.choice(DEPARTMENTS, n_orders),
            "Market": rng.choice(MARKETS, n_orders),
            "Shipping Mode": mode,
            "order date (DateOrders)": order_date,
            "shipping date (DateOrders)": order_date + pd.to_timedelta(scheduled + delay, unit="D"),
        }
    )

    lines_per_order = rng.integers(1, 4, size=n_orders)  # 1–3 lines
    df = orders.loc[np.repeat(orders.index, lines_per_order)].reset_index(drop=True)
    n_lines = len(df)
    qty = rng.integers(1, 10, n_lines)
    price = np.round(rng.uniform(10, 500, n_lines), 2)
    discount = np.round(rng.uniform(0, 0.3, n_lines), 2)
    df["Order Item Id"] = np.arange(1, n_lines + 1)
    df["Order Item Quantity"] = qty
    df["Order Item Product Price"] = price
    df["Order Item Discount Rate"] = discount
    df["Order Item Total"] = np.round(qty * price * (1 - discount), 2)
    return df.sort_values("order date (DateOrders)").reset_index(drop=True)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dataco_small.csv")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_frame(n=n)
    frame.to_csv(out, index=False, encoding="latin1")
    print(f"wrote {out} ({len(frame):,} rows, {frame.shape[1]} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
