# 供应链延迟交付风险预测

用 DataCo 全球供应链数据（180K+ 条订单，53 个原始字段）做的一个延迟交付预测项目。从数据清洗、特征工程到多模型对比、超参调优、SHAP 解释、业务量化，走了一遍完整的 ML 流水线。

## 业务问题

订单延迟交付是供应链里代价很高的风险。想做的事情很简单：在订单发货之前，预测它有多大可能会延迟。如果能提前知道高风险订单，就可以加急配送、主动联系客户，而不是等客户投诉了再补救。

## 数据集

DataCo Supply Chain Dataset，从 Kaggle 下载。180,519 条订单记录，53 个原始字段，包括订单类型、运输方式、产品品类、客户地区、订单日期、运输日期等等。数据有缺失值也有异常值，比较接近真实工作中能遇到的数据。

## 做了什么

整个项目的流程：

1. **数据清洗**：删了邮箱、密码、人名这种无关列，处理缺失值。比较关键的一步是识别并移除了数据泄露特征——Delivery Status 跟标签直接相关，shipping date 是事后才知道的，这些留着的话 AUC 能跑到 0.99 但那叫作弊
2. **特征工程**：从日期提取了月份、星期、季度、是否周末；算了订单总价、折扣金额、单位运输天数价值；做了几组交互特征（运输方式×市场、品类×折扣率、客户类型×运输方式）
3. **训练对比**：用时间切分（早期数据训练，后期数据测试），训练了 6 个模型：Logistic Regression、Random Forest、XGBoost、LightGBM、MLP（三层神经网络）、TabNet（Google 的注意力表格模型）
4. **超参调优**：用 Optuna 的贝叶斯搜索跑了 30 轮
5. **SHAP 解释**：看了哪些特征对预测贡献最大
6. **业务量化**：把混淆矩阵转成成本模型，算了不同客户流失成本下的净收益

## 主要结果

6 个模型在时间切分验证集上的对比：

| 模型 | AUC | F1 |
|------|:--:|:--:|
| XGBoost | 0.744 | 0.690 |
| LightGBM | 0.741 | 0.662 |
| Random Forest | 0.742 | 0.662 |
| MLP（3层） | 0.736 | 0.654 |
| TabNet（Google DL） | 0.728 | 0.655 |
| Logistic Regression | 0.723 | 0.638 |

XGBoost 最好，AUC 0.744。TabNet 反而没打过 XGBoost——事实上表格数据上 DL 打不过梯度提升这件事，NeurIPS 2022 的 benchmark 也有类似结论。

## 几个有意思的发现

**1. 数据分布偏移**

我做了个对抗验证——训练一个分类器来区分「早期订单」和「后期订单」，结果这个分类器的 AUC 高达 0.91。说明训练集和测试集的数据分布差得非常远。具体来说，产品品类（Category）和市场区域（Market）在不同时间段变化最大。

这也解释了为什么 AUC 冲不破 0.74：不是模型不够好，是数据本身在漂移。随机切分的 CV 能到 0.838，但那是因为把未来信息漏到了训练里。时间切分的 0.744 才是真实的、能在生产环境用的数字。

**2. AutoGluon 天花板**

我用 AutoGluon（AutoML 工具）在同样的数据上跑了 10 分钟，它自动尝试了多种模型和集成，最佳结果是 AUC 0.735——比我的手调 XGBoost（0.744）还低一点。说明在这个数据集上，手工做的特征工程和参数选择确实起到了作用，不是随便调个 AutoML 就能替代的。

**3. Target Encoding 的意外效果**

试了把 Label Encoding 换成 Target Encoding 来处理类别特征，以为会更好，结果 AUC 反而掉到了 0.732。后来想想可能是在时间切分下，Target Encoding 用全量数据的均值会有轻微的数据泄露——在真实项目中不是什么技巧都能无脑用的。

**4. Voting Ensemble 没明显增益**

把 XGBoost、LightGBM、Random Forest 做了软投票集成，AUC 只提高了 0.004。这三个模型本身同质性太高，集成带来的收益很有限。

## 运行

数据文件比较大（93MB），没传上来。需要先从 Kaggle 下载：

```bash
# 下载数据
python -c "import kagglehub; kagglehub.dataset_download('evilspirit05/datasupplychain')"

# 把 CSV 放到 data/ 目录下

# 安装依赖
pip install pandas numpy scikit-learn xgboost lightgbm optuna shap imbalanced-learn streamlit

# 跑 notebook
cd notebooks && jupyter notebook supply_chain_ml_pipeline.ipynb
```

## 在线应用

本地跑 Streamlit：

```bash
cd app && streamlit run app.py
```

## License

MIT
