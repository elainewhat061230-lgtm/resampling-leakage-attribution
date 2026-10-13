**Table 1. Prior work on pre-split resampling. Three of the four prior studies report only prevalence-insensitive metrics (the Brier score, which does respond to prevalence, is the one partial exception; see Section 1.1) and evaluate the incorrect arm on a resampled test set; the fourth reports both metric families but states neither an ablation nor its test-set composition. No prior design separates the three mechanisms.**

| Study                    | Domain              | Datasets              | Metrics reported                    | Test set in 'incorrect' arm      | Decomposition?     |
|:-------------------------|:--------------------|:----------------------|:------------------------------------|:---------------------------------|:-------------------|
| Vandewiele et al. (2021) | Preterm birth (EHG) | 1 + synthetic         | ROC-AUC only                        | Contains synthetic rows          | No                 |
| Demircioglu (2024)       | Radiomics           | 15                    | AUC, sens., spec., bal. acc., Brier | Affected by resampling           | No                 |
| Kabane (2024)            | Credit-card fraud   | 1                     | Acc., prec., rec., F1, AUC, MCC     | Not stated                       | No                 |
| Roth (2026)              | General tabular     | 2,047                 | Delta AUC only                      | Standard leaky pipeline          | No                 |
| This study               | Credit risk / fraud | 6 + controlled ladder | AUC, AUPRC, F1, acc., prec., rec.   | Five arms isolate each mechanism | Yes (3 mechanisms) |
