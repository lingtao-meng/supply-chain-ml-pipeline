"""Streamlit demo: turn a predicted delay distribution into an action.

Run with::

    streamlit run app/app.py

The app uses the artifact exported by ``scripts/refresh_cost_analysis.py``
(``models/model_bundle.joblib``).  There is **no fallback heuristic**: if the
bundle is missing the app tells you how to build it instead of inventing a
probability.  (v1 of this demo multiplied hand-written weights and added
``np.random.normal`` noise — the numbers looked plausible and meant nothing.)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the app must render on servers and in CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, policy  # noqa: E402

BUNDLE_PATH = ROOT / "models" / "model_bundle.joblib"

SHIPPING_MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]
SCHEDULED_DAYS = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 4}
ACTION_LABELS = {
    "none": "不干预",
    "notify": "提前通知客户",
    "express": "加急运输",
}

st.set_page_config(page_title="延迟交付决策", page_icon="📦", layout="wide")


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    import joblib

    return joblib.load(BUNDLE_PATH)


def build_input_row(values: dict) -> pd.DataFrame:
    """One raw row, shaped like the DataCo columns the features need."""
    return pd.DataFrame(
        [
            {
                "Type": values["type"],
                "Category Name": values["category"],
                "Department Name": values["department"],
                "Market": values["market"],
                "Shipping Mode": values["shipping_mode"],
                "Customer Segment": values["segment"],
                pipeline.DATE_COL: pd.Timestamp(values["order_date"]),
                pipeline.SCHED_COL: SCHEDULED_DAYS[values["shipping_mode"]],
                "Order Item Quantity": values["qty"],
                "Order Item Product Price": values["price"],
                "Order Item Discount Rate": values["discount"],
                "Order Item Total": values["qty"] * values["price"],
            }
        ]
    )


def distribution_for(bundle, row: pd.DataFrame, features: np.ndarray, use_model: bool) -> np.ndarray:
    """pmf over the delay atoms, either from the classifier or from history."""
    if use_model:
        proba = bundle["pmf_model"].predict_proba(features)
        atoms = bundle["atoms"]
        pmf = np.zeros(len(atoms))
        for j, cls in enumerate(bundle["pmf_model"].classes_):
            hits = np.where(atoms == int(cls))[0]
            if hits.size:
                pmf[hits[0]] = proba[0, j]
        return pmf
    mode = row["Shipping Mode"].iloc[0]
    return bundle["mode_pmf"].get(mode, bundle["mode_pmf_default"])


def cost_table(pmf: np.ndarray, penalty: float, actions) -> pd.DataFrame:
    atoms = np.asarray(policy.ATOMS, dtype=float)
    rows = []
    for action in actions:
        excess = float(np.clip(atoms - action["reduction"], 0, None) @ pmf)
        rows.append(
            {
                "动作": ACTION_LABELS.get(action["name"], action["name"]),
                "固定成本 ($)": action["cost"],
                "预期剩余延迟 (天)": excess,
                "预期总成本 ($)": action["cost"] + penalty * action["multiplier"] * excess,
            }
        )
    return pd.DataFrame(rows)


bundle = load_bundle()

st.title("📦 延迟交付：从预测到决策")
st.caption(
    "输入一笔订单 → 预测延迟天数分布 → 用同一个成本函数选出最省钱的动作。"
    "所有数字都来自真实训练好的模型，没有随机数。"
)

if bundle is None:
    st.error(
        "找不到 `models/model_bundle.joblib`。\n\n"
        "先生成它（约 3 分钟，首次训练分位模型）：\n\n"
        "```bash\npython scripts/refresh_cost_analysis.py\n```"
    )
    st.stop()

metrics = bundle["metrics"]
enc = bundle["encoders"]

with st.sidebar:
    st.header("订单信息")
    shipping_mode = st.selectbox("运输方式", SHIPPING_MODES, index=3)
    st.caption(
        f"数据里该方式的承诺时效为 **{SCHEDULED_DAYS[shipping_mode]} 天**，"
        "Ship Mode 是唯一有真实信号的字段。"
    )
    market = st.selectbox("市场", list(enc["Market"].classes_))
    category = st.selectbox("品类", list(enc["Category Name"].classes_))
    department = st.selectbox("部门", list(enc["Department Name"].classes_))
    segment = st.selectbox("客户类型", list(enc["Customer Segment"].classes_))
    order_type = st.selectbox("支付方式", list(enc["Type"].classes_))
    qty = st.slider("数量（件）", 1, 20, 3)
    price = st.slider("单价 ($)", 10.0, 2000.0, 150.0, 10.0)
    discount = st.slider("折扣率", 0.0, 0.5, 0.05, 0.01)
    order_date = st.date_input("下单日期", value=date(2017, 12, 15))

    st.divider()
    penalty = st.number_input(
        "延迟罚金 ($/天)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0
    )
    distribution_source = st.radio(
        "分布来源",
        ["ML 多分类模型", "运输方式历史分布"],
        help="两者在测试集上的决策成本几乎相同；历史分布是这张表的理论上限。",
    )

values = {
    "shipping_mode": shipping_mode,
    "market": market,
    "category": category,
    "department": department,
    "segment": segment,
    "type": order_type,
    "qty": qty,
    "price": price,
    "discount": discount,
    "order_date": order_date,
}
row = build_input_row(values)
raw_features = pipeline.make_features(row, enc)
features = bundle["scaler"].transform(raw_features)

quantiles = {
    q: float(np.clip(model.predict(features)[0], -5, 10))
    for q, model in bundle["quantile"].items()
}
p10, p50, p90 = quantiles[0.1], quantiles[0.5], quantiles[0.9]
use_model = distribution_source.startswith("ML")
pmf = distribution_for(bundle, row, features, use_model)
costs = cost_table(pmf, penalty, bundle["params"]["actions"])
best_idx = int(np.argmin(costs["预期总成本 ($)"]))
best_action = bundle["params"]["actions"][best_idx]

col1, col2, col3, col4 = st.columns(4)
col1.metric("P10 延迟", f"{p10:+.2f} 天")
col2.metric("P50 延迟", f"{p50:+.2f} 天")
col3.metric("P90 延迟", f"{p90:+.2f} 天")
col4.metric("建议动作", ACTION_LABELS.get(best_action["name"], best_action["name"]))

no_action_cost = costs.loc[costs["动作"] == ACTION_LABELS["none"], "预期总成本 ($)"].iloc[0]
best_cost = costs["预期总成本 ($)"].iloc[best_idx]
saving = no_action_cost - best_cost
st.markdown(
    f"在 **${penalty:,.0f}/天** 的罚金下，这笔订单的期望成本是 **${best_cost:,.2f}**"
    + (f"，比不干预省 **${saving:,.2f}**。" if saving > 0 else "，不干预已经是最优。")
)

left, right = st.columns([1.15, 1])

with left:
    st.subheader("预测的延迟分布")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    atoms = bundle["atoms"]
    bars = ax.bar(atoms, pmf, color="#1f6f78", width=0.65)
    for b, p in zip(bars, pmf):  # noqa: B905 (py3.9 compat)
        if p > 0.02:
            ax.text(b.get_x() + b.get_width() / 2, p + 0.01, f"{p:.0%}", ha="center", fontsize=8)
    ax.set_xlabel("delay vs promised days (days)")
    ax.set_ylabel("probability")
    ax.set_xticks(atoms)
    ax.set_ylim(0, max(0.35, pmf.max() * 1.2))
    ax.axvline(0, color="#9b3b3b", linestyle="--", linewidth=1)
    ax.set_title("predicted delay distribution (right of dashed line = late)")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"分布来源：{'7 分类模型' if use_model else '运输方式历史频率'} · "
        f"P(迟到) = {pmf[atoms > 0].sum():.1%}"
    )

with right:
    st.subheader("三个动作的期望成本")
    st.dataframe(
        costs.style.format(
            {
                "固定成本 ($)": "${:,.0f}",
                "预期剩余延迟 (天)": "{:.2f}",
                "预期总成本 ($)": "${:,.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("成本 = 固定成本 + 罚金 × 折扣系数 × E[max(延迟 − 减天数, 0)]，与 script 完全同一套公式。")

with st.expander("这个模型能做什么、不能做什么", expanded=False):
    within = metrics.get("within_mode_auc", {})
    st.markdown(
        f"""
**已经做到**

- 时间切分（过去训练、未来测试），18 个下单时点可得特征，无泄露字段
- 二分类 AUC **{metrics['binary_auc']:.4f}**；P50 的 MAE **{metrics['mae_p50']:.2f} 天**
- 用离散分布直接算 `E[(延迟 − 减天数)+]`，不再用三个分位点插值

**不能做到**

- 只用 `Shipping Mode` 查表的 AUC 是 **{metrics['mode_base_rate_auc']:.4f}**，
  与 18 特征模型几乎相同；方式内部（Standard {within.get('Standard Class', float('nan')):.4f}、
  Second {within.get('Second Class', float('nan')):.4f}）基本是 0.50
- 原因：这个公开数据集的运输天数在方式内部近似均匀随机生成，**不存在可用的单笔信号**
- 所以本项目的价值不在“模型更强”，而在“把预测变成可比较的成本决策”，并证明天花板的来源
"""
    )

st.divider()
st.caption(
    "指标与成本表由 `scripts/refresh_cost_analysis.py` 生成 · "
    f"训练缓存 `{bundle['cache_key']}` · "
    "[GitHub 源码](https://github.com/lingtao-meng/supply-chain-ml-pipeline)"
)
