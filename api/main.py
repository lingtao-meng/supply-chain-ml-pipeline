"""HTTP service for the delay-policy model.

Run locally::

    uvicorn api.main:app --reload --port 8000

Endpoints
---------
``GET  /health``      liveness + whether the bundle is loaded
``GET  /model-info``  metrics and cost parameters baked into the bundle
``POST /score``       score one or more orders, with an optional budget cap

The service is a thin wrapper around :mod:`src.scoring`; the CLI, the API and
the Streamlit app therefore share one implementation of the decision rule.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import scoring  # noqa: E402

app = FastAPI(
    title="Supply Chain Delay Policy API",
    version="2.2.0",
    description="Turn a delay distribution into the cheapest intervention.",
)

_bundle: dict | None = None
_bundle_error: str | None = None


def get_bundle() -> dict:
    global _bundle, _bundle_error
    if _bundle is None:
        try:
            _bundle = scoring.load_bundle()
        except FileNotFoundError as exc:
            _bundle_error = str(exc)
            raise HTTPException(status_code=503, detail=_bundle_error) from exc
    return _bundle


class Order(BaseModel):
    """One order, with the DataCo fields the features need."""

    type: str = Field(description="payment type, e.g. DEBIT")
    category_name: str
    department_name: str
    market: str
    shipping_mode: str
    customer_segment: str
    order_date: str = Field(description="ISO date, e.g. 2017-12-15")
    scheduled_days: int = Field(ge=0, le=10, description="promised transit days")
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)
    discount_rate: float = Field(ge=0, le=1)
    order_total: Optional[float] = Field(default=None, ge=0, description="discounted net total; defaults to quantity*unit_price*(1-discount_rate)")


class ScoreRequest(BaseModel):
    orders: list[Order] = Field(min_length=1)
    penalty: Optional[float] = Field(default=None, gt=0, description="$ per late day")
    budget: Optional[float] = Field(default=None, ge=0, description="cap on action spend")
    use_history: bool = False


class OrderScore(BaseModel):
    recommended_action: str
    expected_saving: float
    p_late: float
    p10_delay: float
    p50_delay: float
    p90_delay: float
    expected_cost_none: float
    expected_cost_notify: float
    expected_cost_express: float


class ScoreResponse(BaseModel):
    orders: list[OrderScore]
    summary: dict[str, Any]


def orders_to_frame(orders: list[Order]) -> pd.DataFrame:
    rows = []
    for order in orders:
        # Training uses a discounted net total; keep the API on the same basis.
        total = order.order_total
        if total is None:
            total = order.quantity * order.unit_price * (1.0 - order.discount_rate)
        rows.append(
            {
                "Type": order.type,
                "Category Name": order.category_name,
                "Department Name": order.department_name,
                "Market": order.market,
                "Shipping Mode": order.shipping_mode,
                "Customer Segment": order.customer_segment,
                "order date (DateOrders)": order.order_date,
                "Days for shipment (scheduled)": order.scheduled_days,
                "Order Item Quantity": order.quantity,
                "Order Item Product Price": order.unit_price,
                "Order Item Discount Rate": order.discount_rate,
                "Order Item Total": total,
            }
        )
    return pd.DataFrame(rows)


def summarise(plan: pd.DataFrame, bundle: dict, penalty: float, budget: float | None) -> dict:
    counts = plan["recommended_action"].value_counts().to_dict()
    costs = {action.name: action.cost for action in scoring.actions_from_bundle(bundle)}
    spend = float(sum(costs.get(str(name), 0.0) * count for name, count in counts.items()))
    return {
        "orders": int(len(plan)),
        "action_mix": {str(k): int(v) for k, v in counts.items()},
        "action_spend": spend,
        "budget": budget,
        "expected_saving": float(plan["expected_saving"].sum()),
        "penalty_per_day": penalty,
    }


@app.get("/health")
def health() -> dict:
    try:
        bundle = get_bundle()
    except HTTPException:
        return {"status": "degraded", "model_loaded": False, "detail": _bundle_error}
    return {"status": "ok", "model_loaded": True, "cache_key": bundle.get("cache_key")}


@app.get("/model-info")
def model_info() -> dict:
    bundle = get_bundle()
    return {
        "metrics": bundle.get("metrics", {}),
        "params": bundle.get("params", {}),
        "cache_key": bundle.get("cache_key"),
        "features": bundle.get("feature_names", []),
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    bundle = get_bundle()
    frame = orders_to_frame(request.orders)
    penalty = float(bundle["params"]["penalty"] if request.penalty is None else request.penalty)
    try:
        plan = scoring.score_frame(
            frame,
            bundle,
            penalty=penalty,
            budget=request.budget,
            use_history=request.use_history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    fields = [
        "recommended_action",
        "expected_saving",
        "p_late",
        "p10_delay",
        "p50_delay",
        "p90_delay",
        "expected_cost_none",
        "expected_cost_notify",
        "expected_cost_express",
    ]
    orders = [
        OrderScore(**{key: record[key] for key in fields})
        for record in plan[fields].to_dict(orient="records")
    ]
    return ScoreResponse(
        orders=orders,
        summary=summarise(plan, bundle, penalty, request.budget),
    )
