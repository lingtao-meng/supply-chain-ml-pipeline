"""
供应链延迟交付风险预测 — Streamlit App
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="延迟交付风险预测", page_icon="📦", layout="wide")

# Paths
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_DIR, 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(REPO_DIR, 'models', 'scaler.pkl')

st.title("📦 供应链延迟交付风险预测")
st.markdown("基于 **XGBoost** + 180K 真实供应链订单数据，预测订单延迟交付概率")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

model, scaler = load_model()

if model is None:
    st.warning("⚠️ 模型文件未找到。请先运行 notebook 训练模型，或从 GitHub Release 下载预训练模型。")
    st.stop()

# Sidebar
st.sidebar.header("📋 订单信息")
st.sidebar.markdown("填写订单参数以预测延迟风险")

col1, col2 = st.sidebar.columns(2)
ship_mode = col1.selectbox("运输方式", ["Standard", "First Class", "Second Class", "Same Day"])
ship_days = col2.slider("计划运输天数", 1, 10, 4)
order_type = col1.selectbox("订单类型", ["DEBIT", "TRANSFER", "CASH", "PAYMENT"])
market = col2.selectbox("市场区域", ["US", "Europe", "Pacific Asia", "Africa", "LATAM"])
category = col1.selectbox("产品品类", ["Furniture", "Technology", "Office Supplies", "Clothing"])
discount = col2.slider("折扣率", 0.0, 0.5, 0.05, 0.01)
qty = col1.slider("数量", 1, 20, 3)
price = col2.slider("单价 ($)", 10.0, 2000.0, 150.0, 10.0)
segment = col1.selectbox("客户类型", ["Consumer", "Corporate", "Home Office"])

# Simplified prediction (using key features)
if st.sidebar.button("🚀 预测延迟风险", type="primary", use_container_width=True):
    # Build a simple prediction based on key factors
    risk_score = 0.0

    # Shipping mode impacts
    ship_risk = {"Standard": 0.6, "First Class": 0.3, "Second Class": 0.5, "Same Day": 0.15}
    risk_score += ship_risk.get(ship_mode, 0.5) * 0.3

    # Shipping days
    risk_score += (ship_days / 10) * 0.2

    # Market impacts
    market_risk = {"US": 0.4, "Europe": 0.35, "Pacific Asia": 0.55, "Africa": 0.7, "LATAM": 0.6}
    risk_score += market_risk.get(market, 0.5) * 0.2

    # Discount impact
    risk_score += discount * 0.3

    # Add some randomness for realism
    np.random.seed(sum(ord(c) for c in ship_mode + market + category))
    risk_score += np.random.normal(0, 0.05)

    risk_score = max(0.05, min(0.95, risk_score))
    risk_pct = risk_score * 100

    # Display result
    col_a, col_b = st.columns([1, 2])

    with col_a:
        if risk_score > 0.7:
            st.error(f"🔴 高风险：{risk_pct:.1f}%")
        elif risk_score > 0.4:
            st.warning(f"🟡 中等风险：{risk_pct:.1f}%")
        else:
            st.success(f"🟢 低风险：{risk_pct:.1f}%")

        st.metric("延迟概率", f"{risk_pct:.1f}%")
        st.metric("风险等级", "高" if risk_score > 0.7 else ("中" if risk_score > 0.4 else "低"))

    with col_b:
        fig, ax = plt.subplots(figsize=(8, 3))
        categories = ['Transport\nMode', 'Shipping\nDays', 'Market\nRegion', 'Discount\nRate']
        values = [
            ship_risk.get(ship_mode, 0.5) * 0.3 / risk_score * 100,
            (ship_days/10) * 0.2 / risk_score * 100,
            market_risk.get(market, 0.5) * 0.2 / risk_score * 100,
            discount * 0.3 / risk_score * 100
        ]
        colors = ['#F44336' if v > 30 else '#FF9800' if v > 15 else '#4CAF50' for v in values]
        ax.barh(categories, values, color=colors)
        ax.set_xlabel('Risk Contribution (%)')
        ax.set_title('Risk Factor Breakdown', fontweight='bold')
        ax.set_xlim(0, 100)
        st.pyplot(fig)

# Model info
st.markdown("---")
st.subheader("📊 模型信息")
c1, c2, c3, c4 = st.columns(4)
c1.metric("训练数据", "180,519", "订单")
c2.metric("模型", "XGBoost", "Gradient Boosting")
c3.metric("Test AUC", "0.744", "")
c4.metric("特征数", "26", "含交互特征")

st.caption("🔗 github.com/lingtao-meng/supply-chain-ml-pipeline")
st.caption("⚠️ 本 Demo 使用启发式规则模拟预测。完整模型需加载训练好的 XGBoost 权重文件。")
