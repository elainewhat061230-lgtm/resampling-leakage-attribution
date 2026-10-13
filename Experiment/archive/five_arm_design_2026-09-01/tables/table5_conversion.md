**Table 5. (SMOTE) Comparability of previously published pre-split numbers, by metric and model class.**

| Metric    | Model class          |   Median total inflation |   (1) Prevalence | (3) True leakage   | Comparability                                                          |
|:----------|:---------------------|-------------------------:|-----------------:|:-------------------|:-----------------------------------------------------------------------|
| ROC-AUC   | Linear (LogReg)      |                   0.011  |          -0      | +0.0030 n.e.       | Closest to comparable; residual gap dominated by synthetic test rows   |
| ROC-AUC   | Flexible (DT/RF/XGB) |                   0.0611 |           0      | +0.0225            | Not correctable: leakage and synthetic-row terms both contribute       |
| AUPRC     | Linear (LogReg)      |                   0.158  |           0.1463 | +0.0034 n.e.       | Not correctable: depends on the original class ratio, often unreported |
| AUPRC     | Flexible (DT/RF/XGB) |                   0.2861 |           0.1642 | +0.0473            | Not correctable: depends on the original class ratio, often unreported |
| F1        | Linear (LogReg)      |                   0.1926 |           0.1675 | +0.0032 n.e.       | Not correctable: depends on the original class ratio, often unreported |
| F1        | Flexible (DT/RF/XGB) |                   0.2537 |           0.1006 | +0.0376            | Not correctable: depends on the original class ratio, often unreported |
| Accuracy  | Linear (LogReg)      |                   0.0031 |          -0.0099 | +0.0010 n.e.       | Not correctable: depends on the original class ratio, often unreported |
| Accuracy  | Flexible (DT/RF/XGB) |                   0.042  |          -0.0604 | +0.0061            | Not correctable: depends on the original class ratio, often unreported |
| Precision | Linear (LogReg)      |                   0.2744 |           0.2721 | +0.0019 n.e.       | Not correctable: depends on the original class ratio, often unreported |
| Precision | Flexible (DT/RF/XGB) |                   0.2498 |           0.2155 | +0.0125            | Not correctable: depends on the original class ratio, often unreported |
