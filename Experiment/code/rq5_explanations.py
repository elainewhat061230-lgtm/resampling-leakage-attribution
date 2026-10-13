"""RQ5（探索性）：预测性能同档的模型之间，top-k 特征重要性是否一致？

预注册 §1 把本项列为**探索性**、下游小节，且要求引 Rashomon / 分歧问题文献并说明区别：
  * Krishna et al. (2024, TMLR；arXiv 2022) 等研究的是**同一模型上不同解释方法**的分歧
  * Cavus & Biecek (2025, Information Fusion) 研究的是**正确协议下**等精度模型的预测分歧
  * 本节问的是：**在正确协议（B 臂）下，性能同档的模型给出的 top-k 是否一致**

只在 CON（正确协议）下做 —— 本节与泄漏无关，问的是解释本身的稳健性。
2026-09-25（四状态重构）：预处理改为逐折、只在 O_tr 上拟合（states.FoldPreprocessor），与主设计一致；
旧版（全量拟合 scaler）结果存档于 archive/five_arm_design_2026-09-01/rq5_explanations.csv。
重要性度量：**四个模型统一用置换重要性**（测试折、AUC 打分，n_repeats=5）。
〔2026-09-01 修正，EXTERNAL_REVIEW D6 / DEVIATIONS D-005：初版树模型用 impurity
`feature_importances_`、线性模型用置换重要性——凡含 LogReg 的模型对都把
"方法间分歧"混进了"模型间分歧"，恰是 Krishna 等文献研究的那个混杂。
初版结果存档于 `rq5_explanations_v1_mixedmethods.csv`。〕
只用标准量（Jaccard / Kendall tau），不自造指标。
"""
from __future__ import annotations
import os, sys, glob, time, warnings, itertools
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau

import config as C
import figstyle as fs
import run_main_grid as R
from states import FoldPreprocessor

TOPK = 5
SEEDS = list(range(10))
HERE = os.path.dirname(os.path.abspath(__file__))
OUTCSV = os.path.abspath(os.path.join(HERE, "..", "rq5_explanations.csv"))


def importances(model_name, model, Xtr, ytr, Xte, yte, seed):
    """统一置换重要性（测试折、AUC 打分）—— 不按模型类型分叉，避免方法混杂（D6）。"""
    pi = permutation_importance(model, Xte, yte, n_repeats=5, random_state=seed,
                                n_jobs=-1, scoring="roc_auc")
    return np.asarray(pi.importances_mean, dtype=float)


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B)


def run():
    rows = []
    t0 = time.time()
    for label, did, pos in C.DATASETS:
        X, y = R.load_dataset(label, did, pos, C.ROW_CAP)
        for seed in SEEDS:
            tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y)))
            pre = FoldPreprocessor().fit(X, tr)            # 逐折预处理：只在 O_tr 上拟合
            Z_tr, Z_te = pre.transform(X, tr), pre.transform(X, te)
            # CON 协议：重采样器只看 O_tr（k 值按全量标签定，与主网格一致）
            Xtr, ytr = C.make_resampler("SMOTE", seed, y[tr]).fit_resample(Z_tr, y[tr])
            imps, aucs = {}, {}
            for mname in C.MODEL_NAMES:
                m = C.make_model(mname, seed).fit(Xtr, ytr)
                aucs[mname] = roc_auc_score(y[te], m.predict_proba(Z_te)[:, 1])
                imps[mname] = importances(mname, m, Xtr, ytr, Z_te, y[te], seed)
            tops = {k: tuple(np.argsort(-v)[:TOPK]) for k, v in imps.items()}
            for a, b in itertools.combinations(C.MODEL_NAMES, 2):
                rows.append(dict(dataset=label, seed=seed, model_a=a, model_b=b,
                                 auc_a=aucs[a], auc_b=aucs[b],
                                 auc_gap=abs(aucs[a] - aucs[b]),
                                 jaccard=jaccard(tops[a], tops[b]),
                                 kendall=kendalltau(imps[a], imps[b]).correlation,
                                 top1_same=int(tops[a][0] == tops[b][0])))
        print("  %-16s done  (%.0fs)" % (label, time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTCSV, index=False)
    return df


if __name__ == "__main__":
    df = run() if not os.path.exists(OUTCSV) else pd.read_csv(OUTCSV)
    # 注意：OUTCSV 存在即复用（重建链不重算 RQ5）；预处理或协议改动后须先删除 OUTCSV
    fs.logline("\n## RQ5（探索性）· 性能同档模型之间的 top-%d 特征重要性一致性" % TOPK)
    fs.logline("   样本：%d 个模型对 × 数据集 × 种子（%d 数据集 × %d 种子 × %d 对）"
               % (len(df), df.dataset.nunique(), df.seed.nunique(), len(df) // (df.dataset.nunique() * df.seed.nunique())))
    fs.logline("   全体 Jaccard 中位 = %.3f   Kendall tau 中位 = %.3f   top-1 相同比例 = %.1f%%"
               % (df.jaccard.median(), df.kendall.median(), 100 * df.top1_same.mean()))

    # 只看"性能同档"的对：AUC 差小于噪声地板
    close = df[df.auc_gap < 0.003]
    fs.logline("   其中 **性能同档**（|AUC 差| < 0.003）的 %d 对：Jaccard 中位 = %.3f  top-1 相同 = %.1f%%"
               % (len(close), close.jaccard.median(), 100 * close.top1_same.mean()))
    far = df[df.auc_gap >= 0.003]
    fs.logline("   性能不同档的 %d 对：Jaccard 中位 = %.3f  top-1 相同 = %.1f%%"
               % (len(far), far.jaccard.median(), 100 * far.top1_same.mean()))

    fs.logline("   逐数据集（全部对）：")
    for ds, g in df.groupby("dataset"):
        fs.logline("     %-16s Jaccard 中位=%.3f  top-1 相同=%.0f%%  AUC 差中位=%.4f"
                   % (ds, g.jaccard.median(), 100 * g.top1_same.mean(), g.auc_gap.median()))
    fs.logline("   逐模型对（全部数据集）：")
    for (a, b), g in df.groupby(["model_a", "model_b"]):
        fs.logline("     %-13s vs %-13s Jaccard 中位=%.3f  top-1 相同=%.0f%%"
                   % (a, b, g.jaccard.median(), 100 * g.top1_same.mean()))
    print("\n✅ 明细 %s" % OUTCSV)
