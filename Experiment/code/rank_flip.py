"""RQ2′：协议选择是否改变"哪个模型最好"的结论？（四状态设计）

  * **协议翻转**：同一 (数据集, 重采样器, 种子) 下，CON（正确协议）与 LMB（照文献协议的受控终点）
    没有共同的最佳模型
  * **路径分步翻转**（补充）：CON 与 LON、LMN 没有共同最佳模型的比例 —— 翻转发生在路径的哪一步
  * **种子噪声基线（自测）**：同一 (数据集, 重采样器)、同在 CON 下，两个不同种子没有共同最佳模型的比例
  * 并列文献基线：Roth (2026) 种子噪声 winner-flip 10.4%、指标选择 flip 31%

口径（2026-09-25 加固）：
  * **并列第一**：指标值精确相等的模型同为"最佳"；两种协议（或两个种子）只要有共同的最佳模型即算一致。
    胜者分布里，并列的 k 个模型各记 1/k 胜。旧实现用 idxmax 按行序取第一个，结果依赖行序。
  * **完整性**：每组必须包含数据中出现的全部模型，且四个主状态上均无缺失。主数据下出现不完整的组即报错；
    试点数据（只用于调试）剔除不完整的组并在日志中说明。
"""
from __future__ import annotations
import os, sys, itertools, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import figstyle as fs
import build_ground_truth as G

ROTH_SEED_FLIP = 0.104          # Roth (2026) §3.5
ROTH_METRIC_FLIP = 0.31         # Roth (2026) §5.3
HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = G.CELLS          # 试点模式下自动指向 pilot_4state/cells_pilot.csv
STATES4 = ("CON", "LON", "LMN", "LMB")


def best_models(g, col):
    """该列取最大值的全部模型（精确相等即并列），按字母序的元组。"""
    v = g[col].to_numpy()
    return tuple(sorted(g["model"].to_numpy()[v == v.max()]))


def agree(a, b):
    """两组最佳模型是否有交集 —— 有共同的最佳模型即算选择一致。"""
    return bool(set(a) & set(b))


def win_counts(R, state):
    """胜者分布：每次比较记 1 胜，并列的 k 个模型各记 1/k。"""
    out = {}
    for w in R["winners_%s" % state]:
        for m in w:
            out[m] = out.get(m, 0.0) + 1.0 / len(w)
    return out


def fmt_count(v):
    return "%d" % round(v) if abs(v - round(v)) < 1e-9 else "%.1f" % v


def analyse(metric="auc", path=CELLS, strict=None):
    """返回 (R, B)：R 每行一次 (数据集, 重采样器, 种子) 比较；B 每行一个 (数据集, 重采样器) 的种子噪声基线。"""
    strict = (not G.PILOT) if strict is None else strict
    d = pd.read_csv(path)
    expected = set(d["model"].unique())
    if len(expected) < 2:
        raise ValueError("rank_flip：数据中只有 %d 个模型，无从比较" % len(expected))
    cols = ["%s_%s" % (metric, s) for s in STATES4]
    rows, incomplete = [], []
    for (ds, res, seed), g in d.groupby(["dataset", "resampler", "seed"]):
        if set(g["model"]) != expected or len(g) != len(expected) or g[cols].isna().any().any():
            incomplete.append("%s/%s/seed%s" % (ds, res, seed))
            continue
        w = {s: best_models(g, "%s_%s" % (metric, s)) for s in STATES4}
        top2 = np.sort(g["%s_CON" % metric].to_numpy())[::-1][:2]
        rows.append(dict(dataset=ds, resampler=res, seed=seed,
                         **{"winners_%s" % s: w[s] for s in STATES4},
                         flip=int(not agree(w["CON"], w["LMB"])),
                         flip_LON=int(not agree(w["CON"], w["LON"])),
                         flip_LMN=int(not agree(w["CON"], w["LMN"])),
                         n_tied_states=int(sum(len(w[s]) > 1 for s in STATES4)),
                         gap=float(top2[0] - top2[1])))
    if incomplete and strict:
        raise ValueError("rank_flip：%d 个 (数据集, 重采样器, 种子) 组缺模型或有缺失值：%s"
                         % (len(incomplete), ", ".join(incomplete[:10])))
    R = pd.DataFrame(rows)
    R.attrs["incomplete"] = incomplete
    base = []
    for (ds, res), g in R.groupby(["dataset", "resampler"]):
        ws = g.sort_values("seed")["winners_CON"].tolist()
        pairs = list(itertools.combinations(ws, 2))
        if pairs:
            base.append(dict(dataset=ds, resampler=res,
                             seed_flip=np.mean([not agree(a, b) for a, b in pairs]), n_pairs=len(pairs)))
    return R, pd.DataFrame(base)


if __name__ == "__main__":
    fs.logline("\n## RQ2′ · 协议选择是否改变最佳模型（CON vs LMB；四状态设计）")
    fs.logline("   口径：精确并列的模型同为最佳；两协议有共同最佳模型即算一致；胜者分布中并列各记 1/k 胜")
    for metric in ["auc", "f1"]:
        R, B = analyse(metric)
        overall, seedbase = R.flip.mean(), B.seed_flip.mean()
        fs.logline("\n### 指标 = %s" % metric.upper())
        if R.attrs.get("incomplete"):
            fs.logline("   （试点数据：剔除 %d 个不完整的组）" % len(R.attrs["incomplete"]))
        fs.logline("   协议翻转率（CON 胜者 vs LMB 胜者）= %.1f%%  (%d/%d 个 数据集×重采样器×种子)"
                   % (100 * overall, R.flip.sum(), len(R)))
        fs.logline("   路径分步：CON→LON 翻转 %.1f%% (%d/%d)  CON→LMN 翻转 %.1f%% (%d/%d)"
                   % (100 * R.flip_LON.mean(), R.flip_LON.sum(), len(R), 100 * R.flip_LMN.mean(), R.flip_LMN.sum(), len(R)))
        tied = R[R.n_tied_states > 0]
        fs.logline("   并列第一（精确相等）：%d 组比较、共 %d 个（组 × 状态）%s"
                   % (len(tied), int(R.n_tied_states.sum()),
                      "：" + "；".join("%s/%s/seed%s" % (r.dataset, r.resampler, r.seed) for r in tied.itertuples())
                      if len(tied) else ""))
        fs.logline("   【自测】种子噪声基线（同在 CON，换种子）= %.1f%%  (%d 个种子对)"
                   % (100 * seedbase, int(B.n_pairs.sum())))
        fs.logline("   Roth (2026) 种子噪声 winner-flip = %.1f%%；指标选择 flip = %.1f%%"
                   % (100 * ROTH_SEED_FLIP, 100 * ROTH_METRIC_FLIP))
        fs.logline("   → 协议翻转率是自测种子基线的 %.1f 倍" % (overall / seedbase if seedbase else np.nan))
        fs.logline("   逐数据集：")
        for ds, g in R.groupby("dataset"):
            b = B[B.dataset == ds].seed_flip.mean()
            fs.logline("     %-16s 协议翻转 %5.1f%%   种子基线 %5.1f%%   CON 第一二名差 中位=%.4f"
                       % (ds, 100 * g.flip.mean(), 100 * b, g.gap.median()))
        q = R.gap.quantile([0.25, 0.5, 0.75]).tolist()
        R["gap_bin"] = pd.cut(R.gap, [-np.inf] + q + [np.inf], labels=["Q1(最扎堆)", "Q2", "Q3", "Q4(最分散)"])
        fs.logline("   按 CON 第一二名差分组的翻转率：")
        for lab, g in R.groupby("gap_bin", observed=True):
            fs.logline("     %-12s 差中位=%.4f  翻转率 %5.1f%%  (n=%d)" % (lab, g.gap.median(), 100 * g.flip.mean(), len(g)))
        if metric == "auc":
            for s in STATES4:
                vc = win_counts(R, s)
                fs.logline("   %s 胜者分布：%s" % (s, " ".join("%s=%s" % (m, fmt_count(vc.get(m, 0.0))) for m in C.MODEL_NAMES)))
    print("\n✅ 已追加写入 %s" % fs.LOGPATH)
