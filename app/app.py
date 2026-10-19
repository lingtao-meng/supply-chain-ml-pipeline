"""Streamlit demo: turn a predicted delay distribution into an action.

Run with::

    streamlit run app/app.py

Two tabs:

* **单笔决策** — one order, the predicted distribution and the cost of each action;
* **批量评分** — upload a CSV, get the same predictions and a downloadable plan.

The app uses the artifact exported by ``scripts/refresh_cost_analysis.py``
(``models/model_bundle.joblib``).  There is **no fallback heuristic**: if the
bundle is missing the app says how to build it instead of inventing a
probability.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the app must render on servers and in CI
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, scoring  # noqa: E402

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
    return scoring.load_bundle(BUNDLE_PATH)


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


def read_upload(uploaded) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded, encoding="latin1")
    except UnicodeDecodeError:
        uploaded.seek(0)
        return pd.read_csv(uploaded)


def render_single(bundle: dict, metrics: dict, enc: dict) -> None:
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
    use_model = distribution_source.startswith("ML")
    plan = scoring.score_frame(row, bundle, penalty=penalty, use_history=not use_model)
    pmf = scoring.predict_pmf(row, bundle, use_history=not use_model)[0]
    actions = scoring.actions_from_bundle(bundle)
    costs = scoring.cost_table(pmf, penalty, actions)

    p10, p50, p90 = (
        float(plan["p10_delay"].iloc[0]),
        float(plan["p50_delay"].iloc[0]),
        float(plan["p90_delay"].iloc[0]),
    )
    best_name = str(plan["recommended_action"].iloc[0])
    best_cost = float(plan.loc[0, f"expected_cost_{best_name}"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P10 延迟", f"{p10:+.2f} 天")
    col2.metric("P50 延迟", f"{p50:+.2f} 天")
    col3.metric("P90 延迟", f"{p90:+.2f} 天")
    col4.metric("建议动作", ACTION_LABELS.get(best_name, best_name))

    no_action_cost = float(plan["expected_cost_none"].iloc[0])
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
        view = costs.rename(
            columns={
                "action": "动作",
                "fixed_cost": "固定成本 ($)",
                "expected_remaining_delay": "预期剩余延迟 (天)",
                "expected_total_cost": "预期总成本 ($)",
            }
        )
        view["动作"] = view["动作"].map(lambda name: ACTION_LABELS.get(name, name))
        st.dataframe(
            view.style.format(
                {
                    "固定成本 ($)": "${:,.0f}",
                    "预期剩余延迟 (天)": "{:.2f}",
                    "预期总成本 ($)": "${:,.2f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("成本 = 固定成本 + 罚金 × 折扣系数 × E[max(延迟 − 减天数, 0)]，与脚本同一套公式。")

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


def render_batch(bundle: dict) -> None:
    st.subheader("批量评分")
    st.markdown(
        "上传 DataCo 格式的订单 CSV，得到每单的 P10/P50/P90、P(迟到)、三种动作的期望成本和建议动作。"
    )

    template = pd.DataFrame(columns=scoring.REQUIRED_COLUMNS).to_csv(index=False).encode("utf-8")
    st.download_button("下载 CSV 模板", template, file_name="orders_template.csv", mime="text/csv")

    uploaded = st.file_uploader("上传订单 CSV", type=["csv"])
    col1, col2 = st.columns(2)
    penalty = col1.number_input(
        "罚金 ($/天)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0
    )
    budget = col2.number_input(
        "干预预算（可选，0 = 不限）", min_value=0.0, max_value=50_000_000.0, value=0.0, step=50_000.0
    )

    if uploaded is None:
        st.info("还没有文件。先用模板准备一版数据，或直接跑 `python scripts/score_orders.py`。")
        return

    try:
        frame = read_upload(uploaded)
        plan = scoring.score_frame(
            frame, bundle, penalty=penalty, budget=budget if budget > 0 else None
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
        st.error(f"评分失败：{exc}")
        return

    counts = plan["recommended_action"].value_counts()
    cost_by_name = {a.name: a.cost for a in scoring.actions_from_bundle(bundle)}
    spend = sum(cost_by_name.get(str(name), 0.0) * int(count) for name, count in counts.items())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("订单数", f"{len(plan):,}")
    c2.metric("建议干预", f"{int((plan['recommended_action'] != 'none').sum()):,}")
    c3.metric("动作花费", f"${spend:,.0f}")
    c4.metric("期望节省", f"${plan['expected_saving'].sum():,.0f}")
    st.caption(
        "动作分布："
        + " · ".join(
            f"{ACTION_LABELS.get(str(name), name)} {int(count):,}"
            for name, count in counts.items()
        )
    )

    preview_cols = [
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
    st.dataframe(plan[preview_cols].head(200), hide_index=True, use_container_width=True)
    st.download_button(
        "下载决策表",
        plan.to_csv(index=False).encode("utf-8"),
        file_name="decisions.csv",
        mime="text/csv",
    )


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

tab_single, tab_batch = st.tabs(["单笔决策", "批量评分"])
with tab_single:
    render_single(bundle, metrics, enc)
with tab_batch:
    render_batch(bundle)

st.divider()
st.caption(
    "指标与成本表由 `scripts/refresh_cost_analysis.py` 生成 · "
    f"训练缓存 `{bundle['cache_key']}` · "
    "[GitHub 源码](https://github.com/lingtao-meng/supply-chain-ml-pipeline)"
)
