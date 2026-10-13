# 四状态试点报告（PILOT_REPORT）

> 由 `code/pilot_report.py` 生成，勿手改。全部检验**从落盘 CSV 独立重算**，不依赖 `states.py` 的运行时断言。

- 单元格：25（bank-marketing×LogReg×SMOTE 5 种子；bank-marketing×RandomForest×ADASYN 1 种子；bank-marketing×RandomForest×SMOTE 5 种子；credit-g×LogReg×ADASYN 2 种子；credit-g×LogReg×RandomOverSampler 2 种子；credit-g×LogReg×SMOTE 5 种子；credit-g×RandomForest×SMOTE 5 种子）；折：125；每折 8 个状态（4 主 + 4 补充因子态）

## 1. 不变量（独立重算）

| 检验 | 实测 | 通过 |
|---|---|---|
| 逐折 max |T − (L+S+P)|（6 指标） | 0.0e+00 | ✅ |
| 逐折 max |AUC(LMB) − AUC(LMN)|（P 的解析零） | 6.3e-15 | ✅ |
| CON 与 LON 流行率逐位相同 | True | ✅ |
| max |流行率(LMN) − 流行率(CON)|（加权回自然值） | 0.0e+00 | ✅ |
| LMB 流行率全部恰为 0.5 | True | ✅ |
| n_eval：CON = LON 且 LMN = LMB | True | ✅ |
| 混合测试人群正负类计数相等 | True | ✅ |
| CON 与泄漏训练集样本量、类别计数一致（全部折） | True | ✅ |
| 预处理拟合行 = O_tr（全部折） | True | ✅ |
| 泄漏模型逐行预测与批次无关：max 差 | 0.0e+00 | ✅ |
| k_test = n_neg_te − n_pos_te（全部折） | True | ✅ |

### 1.1 合成行计数与修剪（按重采样器）

| 重采样器 | 折数 | k_clean = k_leak（未修剪）折占比 | 修剪折数 | 最大修剪行数（干净侧 / 泄漏侧） |
|---|---:|---:|---:|---|
| ADASYN | 15 | 0% | 15 | 238 / 3 |
| RandomOverSampler | 10 | 100% | 0 | 0 / 0 |
| SMOTE | 100 | 100% | 0 | 0 / 0 |

## 2. 贡献量初读（SMOTE；逐折计算后按 折→种子 取均值；L+S+P ≡ T）

### F1

| 数据集 | 模型 | CON | L | S | P | T | L+S+P−T |
|---|---|---:|---:|---:|---:|---:|---:|
| bank-marketing | LogReg | 0.5613 | +0.0032 | +0.0177 | +0.2739 | +0.2948 | 0e+00 |
| bank-marketing | RandomForest | 0.5953 | +0.0527 | +0.0840 | +0.2051 | +0.3418 | 0e+00 |
| credit-g | LogReg | 0.5894 | +0.0246 | +0.0321 | +0.1117 | +0.1684 | 0e+00 |
| credit-g | RandomForest | 0.5645 | +0.0534 | +0.1322 | +0.0782 | +0.2637 | 0e+00 |

### AUC

| 数据集 | 模型 | CON | L | S | P | T | L+S+P−T |
|---|---|---:|---:|---:|---:|---:|---:|
| bank-marketing | LogReg | 0.9104 | +0.0026 | +0.0122 | -0.0000 | +0.0147 | 0e+00 |
| bank-marketing | RandomForest | 0.9235 | +0.0215 | +0.0401 | -0.0000 | +0.0616 | 0e+00 |
| credit-g | LogReg | 0.7762 | +0.0193 | +0.0289 | +0.0000 | +0.0483 | 0e+00 |
| credit-g | RandomForest | 0.7885 | +0.0295 | +0.0935 | +0.0000 | +0.1230 | 0e+00 |

