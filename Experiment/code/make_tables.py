"""生成论文表格 Table 1–5（含 4b）—— 四状态有序路径分解。

同时输出 `.md`（供 review）与 `.tsv`（可直接粘进 Word）。不硬编码数值；关键数字同步写 figure_numbers.log。
汇总一律为均值（L + S + P = T 在每个聚合层级精确成立），判据 = 均值的自助 95% CI 整体落在 ±0.003 之外。
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import figstyle as fs
from decision import verdict_mean, NOISE_FLOOR
import build_ground_truth as G
import rank_flip as RF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(fs._BASE, "tables")
os.makedirs(OUT, exist_ok=True)
CELLS = G.CELLS
PREV_INSENSITIVE = {"auc", "recall"}      # 类条件指标：P 解析为零


def emit(df: pd.DataFrame, stem: str, caption: str):
    df.to_csv(os.path.join(OUT, stem + ".tsv"), sep="\t", index=False)
    with open(os.path.join(OUT, stem + ".md"), "w") as f:
        f.write("**%s**\n\n" % caption)
        f.write(df.to_markdown(index=False, disable_numparse=True))   # 保留 "+" 号与末尾零，与 .tsv 逐字一致
        f.write("\n")
    print("  [table] %-28s %d 行 × %d 列" % (stem, len(df), len(df.columns)))
    return df


def cell(v):
    return "%+.4f [%+.4f, %+.4f]%s" % (v["mean"], v["lo"], v["hi"], "" if v["ci_excludes_floor"] else " n.e.")


# ---------------------------------------------------------------- Table 1
def table1_related():
    rows = [
        # 各单元格经 2026-09-26 引用终审逐条核对原文（Publication/REFERENCE_AUDIT_2026-09-26.md）
        {"Study": "Vandewiele et al. (2021)", "Domain": "Preterm birth (EHG)", "Datasets": "1 + synthetic",
         "Metrics reported": "ROC-AUC (accuracy for 4 of 11 pre-split reproductions)",
         "Test set in 'incorrect' arm": "Contains synthetic rows",
         "Decomposition?": "No"},
        {"Study": "Demircioglu (2024)", "Domain": "Radiomics", "Datasets": "15 + synthetic",
         "Metrics reported": "AUC, sens., spec., bal. acc., Brier", "Test set in 'incorrect' arm": "Affected by resampling",
         "Decomposition?": "No"},
        {"Study": "Kabane (2024)", "Domain": "Credit-card fraud", "Datasets": "1",
         "Metrics reported": "Acc., prec., rec., F1 (AUC and MCC only in figures quoted from other studies)",
         "Test set in 'incorrect' arm": "Not stated (pre-split figures quoted from earlier studies)",
         "Decomposition?": "No"},
        {"Study": "Roth (2026)", "Domain": "General tabular", "Datasets": "777 (imbalanced subset of 2,047)",
         "Metrics reported": "Delta AUC only",
         "Test set in 'incorrect' arm": "Split taken after oversampling; composition not reported",
         "Decomposition?": "No"},
        {"Study": "This study", "Domain": "Credit risk / fraud", "Datasets": "6 + controlled ladder",
         "Metrics reported": "AUC, AUPRC, F1, acc., prec.",
         "Test set in 'incorrect' arm": "Four ordered states on shared folds (CON, LON, LMN, LMB)",
         "Decomposition?": "Yes: three path-specific contributions summing exactly to the total"},
    ]
    return emit(pd.DataFrame(rows), "table1_related_work",
                "Table 1. Prior work on pre-split resampling. Three of the four prior studies report "
                "almost exclusively prevalence-insensitive metrics (see Section 1 for the exceptions) "
                "and evaluate the incorrect arm on a resampled test set; the fourth reports "
                "prevalence-sensitive metrics but quotes its pre-split figures from earlier studies "
                "instead of running both protocols. No prior design attributes the inflation along a "
                "single path from the correct to the as-published protocol.")


# ---------------------------------------------------------------- Table 2
def table2_datasets():
    raw, _ = G.load("%s__*.csv" % G.MAIN_TAG)
    rows = []
    for label, did, pos in C.DATASETS:
        d = raw[raw.dataset == label]
        if not len(d):
            continue
        n = int(d.n_rows.iloc[0]); prev = float(d.prevalence_orig.iloc[0])
        rows.append({"Dataset": label, "OpenML ID": did, "Rows (n)": n, "Minority share": "%.4f" % prev,
                     "Imbalance ratio": "1:%.0f" % ((1 - prev) / prev),
                     "Subsampled to cap": "yes" if n == C.ROW_CAP else "no",
                     "Cells (model x resampler x seed)": int(d.groupby(["model", "resampler", "seed"]).ngroups)})
    return emit(pd.DataFrame(rows), "table2_datasets",
                "Table 2. Datasets. Subsampling preserved the original class ratio "
                "except for creditcard (see Section 4.11).")


# ---------------------------------------------------------------- Table 3
def table3_decomposition():
    dec = pd.read_csv(CELLS)
    sm = dec[dec.resampler == "SMOTE"]
    rows = []
    for metric, mlab in [("f1", "F1"), ("auc", "ROC-AUC")]:
        for model in C.MODEL_NAMES:
            g = sm[sm.model == model]
            if not len(g):
                continue
            v = {k: verdict_mean(g["%s_%s" % (metric, k)].tolist()) for k in "LSPT"}
            T = v["T"]["mean"]
            rec = {"Metric": mlab, "Model": model, "CON": "%.4f" % g["%s_CON" % metric].mean()}
            for k, lab in (("L", "L (training-side leakage)"), ("S", "S (synthetic-test mixture)"), ("P", "P (prevalence)")):
                rec[lab] = cell(v[k])
                rec[lab.split(" ")[0] + " share"] = "%.0f%%" % (100 * v[k]["mean"] / T) if abs(T) > 1e-12 else "—"
            rec["T = L + S + P"] = cell(v["T"])
            rows.append(rec)
            fs.logline("   TAB3 %-7s %-13s CON=%.4f  L=%+.4f  S=%+.4f  P=%+.4f  T=%+.4f  shares L/S/P=%.1f/%.1f/%.1f%%"
                       % (mlab, model, g["%s_CON" % metric].mean(), v["L"]["mean"], v["S"]["mean"], v["P"]["mean"], T,
                          *[100 * v[k]["mean"] / T if abs(T) > 1e-12 else np.nan for k in "LSP"]))
    n_ds, n_seed = sm.dataset.nunique(), sm.seed.nunique()
    return emit(pd.DataFrame(rows), "table3_decomposition",
                "Table 3. Four-state decomposition of the pre-split resampling gap (SMOTE; mean over %d datasets "
                "x %d seeds with bootstrap 95%% CI of the mean). Shares are signed fractions of T and sum to 100%%. "
                "\"n.e.\" = not established: CI intersects the +/-%.3f noise band." % (n_ds, n_seed, NOISE_FLOOR))


# ---------------------------------------------------------------- Table 4 / 4b
def table4_rankflip():
    rows = []
    for metric, mlab in [("auc", "ROC-AUC"), ("f1", "F1")]:
        R, B = RF.analyse(metric, CELLS)
        rows.append({"Metric": mlab,
                     "Protocol-induced flip (CON vs LMB)": "%.1f%% (%d/%d)" % (100 * R.flip.mean(), R.flip.sum(), len(R)),
                     "Seed-noise baseline (this study)": "%.1f%% (%d pairs)" % (100 * B.seed_flip.mean(), int(B.n_pairs.sum())),
                     # Roth 的 10.4% 只测了 ROC-AUC（LR 对 RF 跨种子）；31% 是六个指标合并的一个数（100 个数据集）
                     "Seed-noise baseline (Roth 2026)": ("%.1f%%" % (100 * RF.ROTH_SEED_FLIP)
                                                         if metric == "auc" else "n/a (ROC-AUC only)"),
                     "Metric-choice flip (Roth 2026)": "%.1f%% (six metrics pooled)" % (100 * RF.ROTH_METRIC_FLIP),
                     "Ratio to own baseline": "%.1fx" % (R.flip.mean() / B.seed_flip.mean())})
        fs.logline("   TAB4 %-8s flip=%.3f  own_base=%.3f  ratio=%.2f  flip_LON=%.3f  flip_LMN=%.3f"
                   % (mlab, R.flip.mean(), B.seed_flip.mean(), R.flip.mean() / B.seed_flip.mean(),
                      R.flip_LON.mean(), R.flip_LMN.mean()))
    t = emit(pd.DataFrame(rows), "table4_rankflip",
             "Table 4. Protocol choice changes which model is selected as best. A flip is counted when the two "
             "protocols share no best model; models tied exactly for best are all counted as best. "
             "Baselines are reported alongside as required by the preregistration. The two values from "
             "Roth (2026) compare two models (logistic regression and a random forest) rather than four: "
             "10.4% is that study's winner-flip rate across random seeds on ROC-AUC, and 31% is the share "
             "of 100 datasets on which the winner changed with the choice among six metrics (AUC, F1, "
             "accuracy, precision, recall and MCC). Neither is directly comparable with the rates of this study.")
    R, _ = RF.analyse("auc", CELLS)
    wc, wl = RF.win_counts(R, "CON"), RF.win_counts(R, "LMB")
    n = len(R)
    w = pd.DataFrame({"Model": C.MODEL_NAMES,
                      "Wins under correct protocol (CON)": [RF.fmt_count(wc.get(m, 0.0)) for m in C.MODEL_NAMES],
                      "Wins under correct protocol (CON), %": ["%.1f%%" % (100 * wc.get(m, 0.0) / n) for m in C.MODEL_NAMES],
                      "Wins under as-published endpoint (LMB)": [RF.fmt_count(wl.get(m, 0.0)) for m in C.MODEL_NAMES],
                      "Wins under as-published endpoint (LMB), %": ["%.1f%%" % (100 * wl.get(m, 0.0) / n) for m in C.MODEL_NAMES]})
    emit(w, "table4b_winners", "Table 4b. Distribution of the selected best model, ROC-AUC, %d cells "
                               "(an exact tie for best would split the win equally)." % n)
    for m in C.MODEL_NAMES:
        fs.logline("   TAB4b %-13s CON %s (%.1f%%)  LMB %s (%.1f%%)"
                   % (m, RF.fmt_count(wc.get(m, 0.0)), 100 * wc.get(m, 0.0) / n,
                      RF.fmt_count(wl.get(m, 0.0)), 100 * wl.get(m, 0.0) / n))
    return t


# ---------------------------------------------------------------- Table 5
def table5_conversion():
    dec = pd.read_csv(CELLS)
    sm = dec[dec.resampler == "SMOTE"]
    rows = []
    for metric, mlab in [("auc", "ROC-AUC"), ("auprc", "AUPRC"), ("f1", "F1"),
                         ("accuracy", "Accuracy"), ("precision", "Precision")]:
        for grp, models in [("Linear (LogReg)", ["LogReg"]),
                            ("Flexible (DT/RF/XGB)", ["DecisionTree", "RandomForest", "XGBoost"])]:
            g = sm[sm.model.isin(models)]
            if not len(g):
                continue
            v = {k: verdict_mean(g["%s_%s" % (metric, k)].tolist()) for k in "LSPT"}
            per_ds = g.groupby("dataset")["%s_T" % metric].mean()
            # 按指标性质分类，而不是按 P 的点估计是否落在噪声带内：accuracy 的 P 在各模型、各顺序间变号，
            # 其均值偶然接近 0 并不意味着该指标与原始类别比例无关
            if metric in PREV_INSENSITIVE and not v["L"]["ci_excludes_floor"]:
                rec = "Closest to comparable: no prevalence term, leakage not established; synthetic test rows drive T"
            elif metric in PREV_INSENSITIVE:
                rec = "Not correctable: training-side leakage and synthetic-test mixture both contribute"
            else:
                rec = "Not correctable: depends on the original class ratio"
            rows.append({"Metric": mlab, "Model class": grp,
                         "Mean total inflation T": "%+.4f" % v["T"]["mean"],
                         "T range across datasets": "%+.4f to %+.4f" % (per_ds.min(), per_ds.max()),
                         "L (training-side leakage)": "%+.4f%s" % (v["L"]["mean"], "" if v["L"]["ci_excludes_floor"] else " n.e."),
                         "S (synthetic-test mixture)": "%+.4f%s" % (v["S"]["mean"], "" if v["S"]["ci_excludes_floor"] else " n.e."),
                         "P (prevalence)": "%+.4f%s" % (v["P"]["mean"], "" if v["P"]["ci_excludes_floor"] else " n.e."),
                         "Comparability": rec})
            fs.logline("   TAB5 %-9s %-20s T=%+.4f [%s..%s]  L=%+.4f  S=%+.4f  P=%+.4f"
                       % (mlab, grp, v["T"]["mean"], "%+.4f" % per_ds.min(), "%+.4f" % per_ds.max(),
                          v["L"]["mean"], v["S"]["mean"], v["P"]["mean"]))
    return emit(pd.DataFrame(rows), "table5_conversion",
                "Table 5. (SMOTE) Comparability of previously published pre-split numbers, by metric and "
                "model class. T = L + S + P exactly; the range column gives the per-dataset mean of T.")


if __name__ == "__main__":
    fs.logline("\n## TABLES · 由 make_tables.py 现算（四状态设计）")
    for fn in (table1_related, table2_datasets, table3_decomposition, table4_rankflip, table5_conversion):
        fn()
    print("\n✅ 表格已生成于 %s" % OUT)
