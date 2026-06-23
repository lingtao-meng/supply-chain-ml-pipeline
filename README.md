# 供应链延迟交付风险预测 — 完整机器学习流水线

基于 **DataCo 全球供应链数据集**（180K+ 订单，53个原始特征），构建从数据清洗到业务价值量化的端到端机器学习流水线。

**七大核心发现：** 🔬 对抗验证分布偏移 | 🤖 AutoML天花板 | 🧠 DL vs 树模型 | ⏰ Concept Drift | 🛡️ 数据泄露修复

## 🎯 业务问题

延迟交付是供应链管理中最昂贵的风险之一。本项目在订单发货前预测其延迟概率，为供应链团队提供**主动干预**的决策依据——将事后补救转变为事前预防。

## 🧠 技术流水线

```
原始数据 (180K × 53)
    │
    ├── 数据清洗 ──── 缺失值处理 · 异常值检测 · 数据泄露排查
    ├── 特征工程 ──── 时间特征 · 类别编码 · SMOTE过采样
    ├── 模型对比 ──── LR · RF · XGBoost · LightGBM · MLP (5模型)
    ├── 超参调优 ──── Optuna贝叶斯搜索 (30 trials)
    ├── 可解释性 ──── SHAP全局+局部特征重要性
    └── 业务翻译 ──── 混淆矩阵→成本模型→ROI
```

## 📊 六模型性能对比（时间序列切分）

| 模型 | AUC | F1 | Precision | Recall | 训练时间 |
|------|:--:|:--:|:--:|:--:|:--:|
| Logistic Regression | 0.723 | 0.638 | 0.682 | 0.601 | 0.1s |
| MLP (3层) | 0.736 | 0.654 | 0.700 | 0.614 | 5.8s |
| TabNet (Google DL) | 0.728 | 0.655 | 0.712 | 0.607 | 42s |
| Random Forest | 0.742 | 0.662 | 0.730 | 0.605 | 4.5s |
| LightGBM | 0.741 | 0.662 | 0.721 | 0.611 | 1.4s |
| **XGBoost** | **0.744** | **0.690** | **0.729** | **0.656** | 1.0s |

> 1️⃣ **对抗验证发现严重分布偏移：** 训练了一个分类器来区分「早期订单」和「后期订单」——AUC=**0.91**。说明训练集和测试集的分布差异极大（Category/Market 变化最大）。这解释了为什么模型 AUC 冲不破 0.74：不是模型不够好，是数据在漂移。**这个发现直接把项目从「调参比赛」升级到「工业 ML 的根本挑战」。**
> 2️⃣ **AutoGluon 天花板：** AutoML 最佳 AUC=0.735——我的手调 XGBoost 反而高出 1.3%。
> 3️⃣ **TabNet 没打赢 XGBoost：** 与 NeurIPS 2022 benchmark 一致。

## 🔍 关键预测因子 (SHAP)

| 排名 | 特征 | SHAP重要性 | 业务含义 |
|:--:|------|:--:|------|
| 1 | Shipping Mode | 0.794 | 不同运输方式的可靠性差异显著 |
| 2 | Scheduled Shipping Days | 0.691 | 计划的运输天数是最直接的预测信号 |
| 3 | Order Type | 0.459 | 不同订单类型（采购/销售/退货）的延迟模式不同 |
| 4 | Day of Year | 0.175 | 季节性波动影响物流效率 |
| 5 | Discount Rate | 0.118 | 高折扣订单可能对应低优先级配送 |

## 💰 业务价值

| 指标 | 数值 |
|------|------|
| 测试集上的净收益 | $600 |
| 全量数据预估年收益 | **~$4,000** |
| 相比无模型（全部客户流失）的成本降低 | **37.1%** |
| TP (成功预警) | 9,733 笔 |
| FP (误报成本) | 3,623 笔 → $362,300 |
| FN (漏报损失) | 5,114 笔 → $2,557,000 |

## 🛠 技术栈

- **数据处理:** Pandas, NumPy (缺失值, 异常值, 类型转换)
- **特征工程:** SMOTE过采样, Label Encoding, 时间特征提取
- **模型:** XGBoost · LightGBM · Random Forest · Logistic Regression · MLP
- **调优:** Optuna (TPE贝叶斯采样, 30次搜索)
- **可解释性:** SHAP (TreeExplainer, Summary Plot)
- **评估:** AUC-ROC, F1, Precision, Recall, Confusion Matrix

## 🖥️ Streamlit 在线应用

交互式延迟交付风险预测工具，选择订单参数即可实时预测。

```bash
# 本地运行
cd app && streamlit run app.py
```

## 🚀 快速开始

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna shap imbalanced-learn streamlit

cd notebooks
jupyter notebook supply_chain_ml_pipeline.ipynb
```

## ⚡ 数据泄露防护

本项目在特征工程阶段主动移除了以下会造成数据泄露的特征：
- `Delivery Status` — 直接暴露标签
- `Days for shipping (real)` — 事后才知道
- `Order Status` — 含延迟信息
- `shipping date` — 事后才知道
- `Profit/Sales per customer` — 事后才知道

> 这是工业级ML项目与课程作业的核心区别：课程作业追求高分数，工业项目追求真实可用。

## 📝 License

MIT
