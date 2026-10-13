"""生成论文表格 Table 2–5。

同时输出两种格式：
  * `.md`  —— 供导师/学员在仓库内 review
  * `.tsv` —— **直接粘进 Word 即成表格**（IJDNS 只收 Word）

铁律：不硬编码数值，全部从原始 CSV / 真源脚本现算；关键数字同步写 figure_numbers.log。
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import figstyle as fs
from decision import verdict, NOISE_FLOOR
import build_ground_truth as G
import corpus_metrics as CM
import rank_flip as RF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "tables"))
os.makedirs(OUT, exist_ok=True)
MECH = ["m1_prevalence", "m2_synthetic_rows", "m3_true_leakage"]
MECH_LAB = {"m1_prevalence": "(1) Prevalence shift",
            "m2_synthetic_rows": "(2) Synthetic test rows",
            "m3_true_leakage": "(3) True leakage"}


def emit(df: pd.DataFrame, stem: str, caption: str):
    df.to_csv(os.path.join(OUT, stem + ".tsv"), sep="\t", index=False)
    with open(os.path.join(OUT, stem + ".md"), "w") as f:
        f.write("**%s**\n\n" % caption)
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print("  [table] %-28s %d 行 × %d 列" % (stem, len(df), len(df.columns)))
    return df


# ---------------------------------------------------------------- Table 2
def table2_datasets():
    """数据集描述。"""
    dec = pd.read_csv(os.path.join(HERE, "..", "decomposed_cells.csv"))
    raw, _ = G.load_raw("main__*.csv")
    rows = []
    for label, did, pos in C.DATASETS:
        d = raw[raw.dataset == label]
        n = int(d.n_rows.iloc[0]); prev = float(d.prevalence_orig.iloc[0])
        # 自然比例（未截断）从快照式重算：由 cap 模式反推不可靠，故直接标注是否截断
        capped = "yes" if n == C.ROW_CAP else "no"
        rows.append({"Dataset": label, "OpenML ID": did, "Rows (n)": n,
                     "Minority share": "%.4f" % prev,
                     "Imbalance ratio": "1:%.0f" % ((1 - prev) / prev),
                     "Subsampled to cap": capped,
                     "Cells (model x resampler x seed)": int(d.groupby(["model", "resampler", "seed"]).ngroups)})
    return emit(pd.DataFrame(rows), "table2_datasets",
                "Table 2. Datasets. Subsampling preserved the original class ratio "
                "except for creditcard (see Section 4.11).")


# ---------------------------------------------------------------- Table 3
def table3_decomposition():
    """分解主结果：逐模型 × 指标，三机制的中位与 CI。"""
    dec = pd.read_csv(os.path.join(HERE, "..", "decomposed_cells.csv"))
    sm = dec[dec.resampler == "SMOTE"]
    rows = []
    for metric, mlab in [("f1", "F1"), ("auc", "ROC-AUC")]:
        for model in C.MODEL_NAMES:
            g = sm[sm.model == model]
            rec = {"Metric": mlab, "Model": model}
            tot = 0.0
            vals = {}
            for k in MECH:
                v = verdict(g["%s_%s" % (metric, k)].tolist())
                vals[k] = v
                tot += abs(v["median"])
            for k in MECH:
                v = vals[k]
                mark = "" if v["ci_excludes_floor"] else " n.e."
                rec[MECH_LAB[k]] = "%+.4f [%+.4f, %+.4f]%s" % (v["median"], v["lo"], v["hi"], mark)
                rec[MECH_LAB[k] + " share"] = "%.0f%%" % (100 * abs(v["median"]) / tot) if tot else "—"
            rec["Observed A-B"] = "%+.4f" % g["%s_total" % metric].median()
            rows.append(rec)
            fs.logline("   TAB3 %-4s %-13s %s" % (mlab, model,
                       "  ".join("%s=%+.4f" % (k[:6], vals[k]["median"]) for k in MECH)))
    return emit(pd.DataFrame(rows), "table3_decomposition",
                "Table 3. Decomposition of the pre-split resampling gap into three mechanisms "
                "(SMOTE; median over 6 datasets x 30 seeds, bootstrap 95%% CI). "
                "\"n.e.\" = not established: CI intersects the +/-%.3f noise band." % NOISE_FLOOR)


# ---------------------------------------------------------------- Table 4
def table4_rankflip():
    """协议导致的模型排名翻转，与三条基线并列。"""
    rows = []
    for metric, mlab in [("auc", "ROC-AUC"), ("f1", "F1")]:
        R, B = RF.analyse(metric)
        rows.append({"Metric": mlab,
                     "Protocol-induced flip": "%.1f%% (%d/%d)" % (100 * R.flip.mean(), R.flip.sum(), len(R)),
                     "Seed-noise baseline (this study)": "%.1f%% (%d pairs)" % (100 * B.seed_flip.mean(), int(B.n_pairs.sum())),
                     "Seed-noise baseline (Roth 2026)": "%.1f%%" % (100 * RF.ROTH_SEED_FLIP),
                     "Metric-choice flip (Roth 2026)": "%.1f%%" % (100 * RF.ROTH_METRIC_FLIP),
                     "Ratio to own baseline": "%.1fx" % (R.flip.mean() / B.seed_flip.mean())})
        fs.logline("   TAB4 %-8s flip=%.3f  own_base=%.3f  ratio=%.2f"
                   % (mlab, R.flip.mean(), B.seed_flip.mean(), R.flip.mean() / B.seed_flip.mean()))
    t = emit(pd.DataFrame(rows), "table4_rankflip",
             "Table 4. Protocol choice changes which model is selected as best. "
             "Baselines are reported alongside as required by the preregistration.")

    # 附表：胜者分布
    R, _ = RF.analyse("auc")
    wb = R.winner_B.value_counts(); wa = R.winner_A.value_counts()
    n_cells = len(R)
    w = pd.DataFrame({"Model": C.MODEL_NAMES,
                      "Wins under correct protocol (B)": [int(wb.get(m, 0)) for m in C.MODEL_NAMES],
                      "Wins under correct protocol (B), %": ["%.1f%%" % (100 * wb.get(m, 0) / n_cells) for m in C.MODEL_NAMES],
                      "Wins under as-published protocol (A)": [int(wa.get(m, 0)) for m in C.MODEL_NAMES],
                      "Wins under as-published protocol (A), %": ["%.1f%%" % (100 * wa.get(m, 0) / n_cells) for m in C.MODEL_NAMES]})
    emit(w, "table4b_winners",
         "Table 4b. Distribution of the selected best model, ROC-AUC, 540 cells.")
    fs.logline("   TAB4b n_cells=%d  B-winners=%s  A-winners=%s" % (n_cells, dict(wb), dict(wa)))
    for m in C.MODEL_NAMES:
        fs.logline("   TAB4b %-13s B %3d (%.1f%%)  A %3d (%.1f%%)"
                   % (m, wb.get(m, 0), 100 * wb.get(m, 0) / n_cells,
                      wa.get(m, 0), 100 * wa.get(m, 0) / n_cells))
    return t


# ---------------------------------------------------------------- Table 5
def table5_conversion():
    """换算规则：读者对照已发表的划分前数字时，哪些指标可比。"""
    dec = pd.read_csv(os.path.join(HERE, "..", "decomposed_cells.csv"))
    sm = dec[dec.resampler == "SMOTE"]
    rows = []
    for metric, mlab in [("auc", "ROC-AUC"), ("auprc", "AUPRC"),
                         ("f1", "F1"), ("accuracy", "Accuracy"), ("precision", "Precision")]:
        for grp, models in [("Linear (LogReg)", ["LogReg"]),
                            ("Flexible (DT/RF/XGB)", ["DecisionTree", "RandomForest", "XGBoost"])]:
            g = sm[sm.model.isin(models)]
            v1 = verdict(g["%s_m1_prevalence" % metric].tolist())
            v3 = verdict(g["%s_m3_true_leakage" % metric].tolist())
            tot = g["%s_total" % metric].median()
            if abs(v1["median"]) < NOISE_FLOOR and not v3["ci_excludes_floor"]:
                rec = "Closest to comparable; residual gap dominated by synthetic test rows"
            elif abs(v1["median"]) < NOISE_FLOOR:
                rec = "Not correctable: leakage and synthetic-row terms both contribute"
            else:
                rec = "Not correctable: depends on the original class ratio, often unreported"
            rows.append({"Metric": mlab, "Model class": grp,
                         "Median total inflation": "%+.4f" % tot,
                         "(1) Prevalence": "%+.4f" % v1["median"],
                         "(3) True leakage": "%+.4f%s" % (v3["median"], "" if v3["ci_excludes_floor"] else " n.e."),
                         "Comparability": rec})
    return emit(pd.DataFrame(rows), "table5_conversion",
                "Table 5. (SMOTE) Comparability of previously published pre-split numbers, by metric and model class.")


# ---------------------------------------------------------------- Table 1
def table1_related():
    """Related Work 对比表 —— 本文立论所在，手工录入的字段来自各文全文核读。"""
    rows = [
        {"Study": "Vandewiele et al. (2021)", "Domain": "Preterm birth (EHG)",
         "Datasets": "1 + synthetic", "Metrics reported": "ROC-AUC only",
         "Test set in 'incorrect' arm": "Contains synthetic rows", "Decomposition?": "No"},
        {"Study": "Demircioglu (2024)", "Domain": "Radiomics",
         "Datasets": "15", "Metrics reported": "AUC, sens., spec., bal. acc., Brier",
         "Test set in 'incorrect' arm": "Affected by resampling", "Decomposition?": "No"},
        {"Study": "Kabane (2024)", "Domain": "Credit-card fraud",
         "Datasets": "1", "Metrics reported": "Acc., prec., rec., F1, AUC, MCC",
         "Test set in 'incorrect' arm": "Not stated", "Decomposition?": "No"},
        {"Study": "Roth (2026)", "Domain": "General tabular",
         "Datasets": "2,047", "Metrics reported": "Delta AUC only",
         "Test set in 'incorrect' arm": "Standard leaky pipeline", "Decomposition?": "No"},
        {"Study": "This study", "Domain": "Credit risk / fraud",
         "Datasets": "6 + controlled ladder", "Metrics reported": "AUC, AUPRC, F1, acc., prec., rec.",
         "Test set in 'incorrect' arm": "Five arms isolate each mechanism", "Decomposition?": "Yes (3 mechanisms)"},
    ]
    return emit(pd.DataFrame(rows), "table1_related_work",
                "Table 1. Prior work on pre-split resampling. Three of the four prior studies report "
                "only prevalence-insensitive metrics (the Brier score, which does respond to "
                "prevalence, is the one partial exception; see Section 1.1) and evaluate the "
                "incorrect arm on a resampled test set; the fourth reports both metric families "
                "but states neither an ablation nor its test-set composition. No prior design "
                "separates the three mechanisms.")


if __name__ == "__main__":
    fs.logline("\n## TABLES · 由 make_tables.py 现算")
    for fn in (table1_related, table2_datasets, table3_decomposition,
               table4_rankflip, table5_conversion):
        fn()
    print("\n✅ 表格（.md 供 review，.tsv 可直接粘进 Word）已生成于 %s" % OUT)
