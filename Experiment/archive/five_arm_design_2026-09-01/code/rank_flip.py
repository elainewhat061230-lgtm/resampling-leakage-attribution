"""RQ2′：协议选择是否改变"哪个模型最好"的结论？

预注册 §7 要求翻转率必须与三条基线并列：
  1. **本研究自测的种子噪声基线**（必须自测，不得只引 Roth 的 10.4%——他的语料与模型集不同）
  2. Roth (2026) §3.5 报的种子噪声 winner-flip 率 10.4%
  3. Roth (2026) §5.3 报的指标选择 flip 率 31%

定义：
  * **协议翻转**：同一 (数据集, 重采样器, 种子) 下，B 臂（正确协议）的最佳模型
    与 A 臂（照文献协议）的最佳模型不同
  * **种子噪声基线**：同一 (数据集, 重采样器) 下、**同在 B 臂**，两个不同种子的
    最佳模型不同的比例（所有种子对）
"""
from __future__ import annotations
import os, sys, itertools, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import figstyle as fs

ROTH_SEED_FLIP = 0.104          # Roth (2026) §3.5
ROTH_METRIC_FLIP = 0.31         # Roth (2026) §5.3
HERE = os.path.dirname(os.path.abspath(__file__))
DEC = os.path.abspath(os.path.join(HERE, "..", "decomposed_cells.csv"))


def analyse(metric="auc"):
    d = pd.read_csv(DEC)
    col_B, col_A = "%s_B" % metric, "%s_A" % metric
    rows = []
    for (ds, res, seed), g in d.groupby(["dataset", "resampler", "seed"]):
        if g[col_B].isna().any() or len(g) < 2:
            continue
        wB = g.loc[g[col_B].idxmax(), "model"]
        wA = g.loc[g[col_A].idxmax(), "model"]
        gapB = g[col_B].max() - g[col_B].nlargest(2).iloc[-1]     # 第一名与第二名的差
        rows.append(dict(dataset=ds, resampler=res, seed=seed,
                         winner_B=wB, winner_A=wA, flip=int(wB != wA), gapB=gapB))
    R = pd.DataFrame(rows)

    # 自测种子噪声基线：同在 B 臂，换种子是否改变最佳模型
    base = []
    for (ds, res), g in R.groupby(["dataset", "resampler"]):
        ws = g.sort_values("seed").winner_B.tolist()
        pairs = list(itertools.combinations(ws, 2))
        if pairs:
            base.append(dict(dataset=ds, resampler=res,
                             seed_flip=np.mean([a != b for a, b in pairs]),
                             n_pairs=len(pairs)))
    B = pd.DataFrame(base)
    return R, B


if __name__ == "__main__":
    fs.logline("\n## RQ2′ · 协议选择是否改变最佳模型（预注册 §7）")
    for metric in ["auc", "f1"]:
        R, B = analyse(metric)
        overall = R.flip.mean()
        seedbase = B.seed_flip.mean()
        fs.logline("\n### 指标 = %s" % metric.upper())
        fs.logline("   协议翻转率（B 臂胜者 vs A 臂胜者）= %.1f%%  (%d/%d 个 数据集×重采样器×种子)"
                   % (100 * overall, R.flip.sum(), len(R)))
        fs.logline("   【自测】种子噪声基线（同在 B 臂，换种子）= %.1f%%  (%d 个种子对)"
                   % (100 * seedbase, int(B.n_pairs.sum())))
        fs.logline("   Roth (2026) 种子噪声 winner-flip = %.1f%%" % (100 * ROTH_SEED_FLIP))
        fs.logline("   Roth (2026) 指标选择 flip       = %.1f%%" % (100 * ROTH_METRIC_FLIP))
        fs.logline("   → 协议翻转率是自测种子基线的 %.1f 倍" % (overall / seedbase if seedbase else np.nan))

        fs.logline("   逐数据集：")
        for ds, g in R.groupby("dataset"):
            b = B[B.dataset == ds].seed_flip.mean()
            fs.logline("     %-16s 协议翻转 %5.1f%%   种子基线 %5.1f%%   B臂第一二名差 中位=%.4f"
                       % (ds, 100 * g.flip.mean(), 100 * b, g.gapB.median()))

        # 翻转与"性能扎堆"的关系：第一二名差越小越容易翻
        q = R.gapB.quantile([0.25, 0.5, 0.75]).tolist()
        bins = [-np.inf] + q + [np.inf]
        R["gap_bin"] = pd.cut(R.gapB, bins, labels=["Q1(最扎堆)", "Q2", "Q3", "Q4(最分散)"])
        fs.logline("   按 B 臂第一二名差分组的翻转率：")
        for lab, g in R.groupby("gap_bin", observed=True):
            fs.logline("     %-12s 差中位=%.4f  翻转率 %5.1f%%  (n=%d)"
                       % (lab, g.gapB.median(), 100 * g.flip.mean(), len(g)))

        if metric == "auc":
            fs.logline("   A 臂胜者分布：%s" % dict(R.winner_A.value_counts()))
            fs.logline("   B 臂胜者分布：%s" % dict(R.winner_B.value_counts()))
    print("\n✅ 已追加写入 %s" % fs.LOGPATH)
