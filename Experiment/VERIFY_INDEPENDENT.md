# 独立重实现核验（VERIFY_INDEPENDENT）

> 由 `code/verify_independent.py` 生成。不调用 `states.py`：预处理用 sklearn ColumnTransformer/OneHotEncoder 重写，
> 合成行、测试人群、权重、指标全部独立实现；与主网格落盘的单元格（5 折均值）逐状态逐指标比对。

| 单元格 | 状态 | max \|独立 − 主网格\|（5 指标） |
|---|---|---:|
| pima/LogReg/SMOTE/seed3 | CON | 2.2e-16 |
| pima/LogReg/SMOTE/seed3 | LON | 1.1e-16 |
| pima/LogReg/SMOTE/seed3 | LMN | 1.1e-16 |
| pima/LogReg/SMOTE/seed3 | LMB | 0.0e+00 |
| credit-g/RandomForest/SMOTE/seed7 | CON | 0.0e+00 |
| credit-g/RandomForest/SMOTE/seed7 | LON | 0.0e+00 |
| credit-g/RandomForest/SMOTE/seed7 | LMN | 0.0e+00 |
| credit-g/RandomForest/SMOTE/seed7 | LMB | 1.1e-16 |
| credit-g/XGBoost/RandomOverSampler/seed11 | CON | 1.1e-16 |
| credit-g/XGBoost/RandomOverSampler/seed11 | LON | 0.0e+00 |
| credit-g/XGBoost/RandomOverSampler/seed11 | LMN | 1.1e-16 |
| credit-g/XGBoost/RandomOverSampler/seed11 | LMB | 2.2e-16 |
| bank-marketing/DecisionTree/SMOTE/seed5 | CON | 2.2e-16 |
| bank-marketing/DecisionTree/SMOTE/seed5 | LON | 1.1e-16 |
| bank-marketing/DecisionTree/SMOTE/seed5 | LMN | 1.1e-16 |
| bank-marketing/DecisionTree/SMOTE/seed5 | LMB | 2.2e-16 |

**全部比对的最大绝对差：2.2e-16**
