"""生成论文配图（四状态有序路径分解）。

铁律（KB figure-standards + SOP P3）：
  * 脚本内**不硬编码任何数值**，全部从单元格表 / 原始 CSV 现算
  * 所有标注数字打印到 stdout **并同时写进 figure_numbers.log**（本脚本会先清空该日志）
  * 图内**不放 "Fig. N"** —— 编号交给 caption
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import figstyle as fs
import config as C
from decision import NOISE_FLOOR
import build_ground_truth as G

fs.apply_style()
CONTR = ["L", "S", "P"]
STATES4 = ["CON", "LON", "LMN", "LMB"]

if os.path.exists(fs.LOGPATH):
    os.remove(fs.LOGPATH)
fs.logline("# figure_numbers.log — 由 make_figures.py 现算，图注数字须与本文件一致（四状态设计）")
fs.logline("# 正文宽度 TEXTWIDTH = %.3f in (%.0f pt)，实测自已发表 PDF" % (fs.TEXTWIDTH, fs.TEXTWIDTH * 72))

dec = pd.read_csv(G.CELLS)
sm = dec[dec.resampler == "SMOTE"]
DS_ORDER = [d for d in ["creditcard", "bank-marketing", "credit-default", "adult", "credit-g", "pima"]
            if d in set(sm.dataset)]
DS_SHORT = {"creditcard": "creditcard", "bank-marketing": "bank-mkt", "credit-default": "credit-def",
            "adult": "adult", "credit-g": "credit-g", "pima": "pima"}
MODELS = [m for m in C.MODEL_NAMES if m in set(sm.model)]
MSHORT = {"LogReg": "LR", "DecisionTree": "DT", "RandomForest": "RF", "XGBoost": "XGB"}


def signed_stack(ax, x, parts, width=0.62, labels=True):
    """带符号堆叠：正值自 0 向上、负值自 0 向下；返回 (正向顶, 负向底)。"""
    pos_b = np.zeros(len(x)); neg_b = np.zeros(len(x))
    for k in CONTR:
        v = np.asarray(parts[k], dtype=float)
        vp, vn = np.where(v > 0, v, 0.0), np.where(v < 0, v, 0.0)
        ax.bar(x, vp, width, bottom=pos_b, color=fs.COLORS[k], edgecolor="white", linewidth=0.5,
               label=fs.LABELS[k] if labels else None)
        ax.bar(x, vn, width, bottom=neg_b, color=fs.COLORS[k], edgecolor="white", linewidth=0.5)
        pos_b += vp; neg_b += vn
    return pos_b, neg_b


# ---------------------------------------------------------------- 图 A
def fig_decomposition_by_dataset():
    fs.logline("\n## FIG A · 四状态分解，按数据集（SMOTE；跨模型与种子取均值；T = L + S + P）")
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.45))
    for ax, metric, mlab in zip(axes, ["f1", "auc"], ["F1", "ROC-AUC"]):
        x = np.arange(len(DS_ORDER))
        parts = {k: [sm[sm.dataset == d]["%s_%s" % (metric, k)].mean() for d in DS_ORDER] for k in CONTR}
        T = np.array([sm[sm.dataset == d]["%s_T" % metric].mean() for d in DS_ORDER])
        pos_top, neg_bot = signed_stack(ax, x, parts)
        ax.scatter(x, T, marker="_", s=210, color="0.15", zorder=5, linewidths=1.4, label="T (= L + S + P)")
        ax.set_ylim(min(0.0, neg_bot.min()) * 1.15, max(T.max(), pos_top.max()) * 1.08)   # T 标记不得贴边被截
        for i, d in enumerate(DS_ORDER):
            fs.logline("   %-7s %-16s L=%+.4f  S=%+.4f  P=%+.4f  T=%+.4f  （正文显示 T %+.3f）"
                       % (mlab, d, parts["L"][i], parts["S"][i], parts["P"][i], T[i], T[i]))
        ax.axhline(0, color="0.3", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([DS_SHORT[d] for d in DS_ORDER], rotation=35, ha="right")
        ax.set_ylabel(r"Contribution ($\Delta$ %s)" % mlab)
        ax.set_title(fs.bf(mlab))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, handlelength=1.1, bbox_to_anchor=(0.5, -0.19), borderaxespad=0.0)
    fig.subplots_adjust(wspace=0.34)
    fs.panel_letters(fig, axes)
    return fs.save(fig, "fig_decomposition_by_dataset")


# ---------------------------------------------------------------- 图 B
def fig_composition_by_capacity():
    fs.logline("\n## FIG B · 构成随模型（SMOTE；跨数据集与种子取均值；份额 = 贡献 / T，带符号）")
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.45))
    for ax, metric, mlab in zip(axes, ["f1", "auc"], ["F1", "ROC-AUC"]):
        x = np.arange(len(MODELS))
        parts = {k: [sm[sm.model == m]["%s_%s" % (metric, k)].mean() for m in MODELS] for k in CONTR}
        T = np.array([sm[sm.model == m]["%s_T" % metric].mean() for m in MODELS])
        signed_stack(ax, x, parts)
        ax.scatter(x, T, marker="_", s=210, color="0.15", zorder=5, linewidths=1.4, label="T (= L + S + P)")
        top = max(T.max(), max(np.sum([np.clip(parts[k], 0, None) for k in CONTR], axis=0)))
        ax.set_ylim(min(0, min(np.sum([np.clip(parts[k], None, 0) for k in CONTR], axis=0))) * 1.15, top * 1.12)
        for i, m in enumerate(MODELS):
            b = 0.0
            for k in CONTR:
                v = parts[k][i]
                sh = 100 * v / T[i] if abs(T[i]) > 1e-12 else np.nan
                if v > 0 and v / top > 0.09:
                    ax.text(i, b + v / 2, "%.0f%s" % (sh, fs.PCT), ha="center", va="center",
                            fontsize=fs.ANNOT, color="white")
                if v > 0:
                    b += v
                fs.logline("   %-7s %-13s %s=%+.4f  share=%.1f%%" % (mlab, m, k, v, sh))
            fs.logline("   %-7s %-13s T=%+.4f" % (mlab, m, T[i]))
        ax.axhline(0, color="0.3", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([MSHORT[m] for m in MODELS])
        ax.set_xlabel("Model (nominal flexibility $\\rightarrow$)")
        ax.set_ylabel(r"Contribution ($\Delta$ %s)" % mlab)
        ax.set_title(fs.bf(mlab))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, handlelength=1.1, bbox_to_anchor=(0.5, -0.20), borderaxespad=0.0)
    fig.subplots_adjust(wspace=0.34)
    fs.panel_letters(fig, axes)
    return fs.save(fig, "fig_composition_by_capacity")


# ---------------------------------------------------------------- 图 C
def fig_ladder():
    lad, _ = G.load("ladder-*.csv")
    if lad is None:
        print("  [skip] 无阶梯数据"); return None
    lad = lad.copy()
    lad["dataset"] = lad.dataset + "@r" + lad.ratio.astype(int).astype(str)
    dl = G.cell_table(lad)
    dl["ratio"] = dl.dataset.str.split("@r").str[1].astype(int)
    dl["ds"] = dl.dataset.str.split("@r").str[0]
    fs.logline("\n## FIG C · 受控阶梯：L（训练侧泄漏）vs 不平衡度（ROC-AUC；10 种子均值）")
    dss = [d for d in ["creditcard", "bank-marketing"] if d in set(dl.ds)]
    fig, axes = plt.subplots(1, len(dss), figsize=(fs.TEXTWIDTH, 2.45), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, dss):
        g = dl[dl.ds == ds]
        for m in MODELS:
            gm = g[g.model == m]
            rr = sorted(gm.ratio.unique())
            means = [gm[gm.ratio == r]["auc_L"].mean() for r in rr]
            rho = spearmanr(rr, means).correlation if len(rr) > 2 else np.nan
            ax.plot(rr, means, marker=fs.MARKERS[m], color=fs.COLORS[m], label=m, clip_on=False)
            fs.logline("   %-16s %-13s rho=%+.3f  %s" % (ds, m, rho, " ".join("1:%d=%+.4f" % (r - 1, v) for r, v in zip(rr, means))))
            if len(means) > 1:
                dmin = float(np.min(np.diff(means)))
                fs.logline("   %-16s %-13s 相邻档最小Δ = %+.1e%s" % (ds, m, dmin, "  ← 负档差" if dmin < 0 else ""))
        base = g[g.ratio == 2]["auc_CON"].mean()
        fs.logline("   %-16s 1:1 平衡档 CON AUC 均值 = %.4f （面板标题显示 %.2f）" % (ds, base, base))
        for m in MODELS:
            mb = g[g.model == m]["auc_CON"].mean()
            fs.logline("   %-16s %-13s 全档 CON AUC 均值 = %.4f （正文显示 %.3f）" % (ds, m, mb, mb))
        ax.axhline(NOISE_FLOOR, color="0.55", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 64, 256])
        ax.set_xticklabels(["1:%d" % (r - 1) for r in (2, 4, 8, 16, 64, 256)])
        ax.set_xlabel("Imbalance ratio (minority : majority)")
        ax.set_title("%s\n(balanced-rung CON AUC $=$ %.2f)" % (fs.bf(ds), base), fontsize=fs.FS_BODY)
    axes[0].set_ylabel(r"L, $\Delta$ROC-AUC")
    axes[0].legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
    axes[-1].text(0.98, NOISE_FLOOR, "noise band", fontsize=fs.ANNOT, color="0.45",
                  transform=axes[-1].get_yaxis_transform(), ha="right", va="bottom")
    fs.panel_letters(fig, list(axes))
    fig.subplots_adjust(wspace=0.12)
    return fs.save(fig, "fig_ladder")


# ---------------------------------------------------------------- 图 D
def fig_state_path():
    fs.logline("\n## FIG D · 四状态路径：指标沿 CON → LON → LMN → LMB（SMOTE；跨数据集与种子取均值）")
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.45))
    for ax, metric, mlab in zip(axes, ["f1", "auc"], ["F1", "ROC-AUC"]):
        x = np.arange(4)
        for m in MODELS:
            g = sm[sm.model == m]
            ys = [g["%s_%s" % (metric, s)].mean() for s in STATES4]
            ax.plot(x, ys, marker=fs.MARKERS[m], color=fs.COLORS[m], label=m)
            fs.logline("   %-7s %-13s %s" % (mlab, m, "  ".join("%s=%.4f" % (s, v) for s, v in zip(STATES4, ys))))
        ax.set_xticks(x)
        ax.set_xticklabels(STATES4)
        for xi, k in zip([0.5, 1.5, 2.5], CONTR):          # 步标签：刻度行上、两状态之间
            ax.text(xi, -0.028, r"$\rightarrow$ %s $\rightarrow$" % k, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=fs.FS_TICK, color=fs.COLORS[k])
        ax.set_ylabel(mlab)
        ax.set_title(fs.bf(mlab))
    axes[0].legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
    fs.panel_letters(fig, axes)
    fig.subplots_adjust(wspace=0.28)
    return fs.save(fig, "fig_state_path")


# ---------------------------------------------------------------- 图 E（示意图，无数据数值）
def fig_four_state_design():
    fs.logline("\n## FIG E · 四状态设计示意（纯示意图，无数据数值）")
    fig, ax = plt.subplots(figsize=(fs.TEXTWIDTH, 3.55))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#F2F2F2", ec="0.35", fsz=fs.ANNOT, bold=False):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=2))
        ax.text(x, y, fs.bf(text) if bold else text, ha="center", va="center", fontsize=fsz, zorder=3, linespacing=1.25)

    def arrow(x0, y0, x1, y1, color="0.35"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, shrinkA=1, shrinkB=1), zorder=1)

    box(0.5, 0.945, 0.92, 0.07, "Stratified 5-fold split of the original data, once per seed: "
        "O$_{tr}$ / O$_{te}$ shared by all four states", fc="#E8E8E8")
    box(0.5, 0.855, 0.92, 0.06, "Per-fold preprocessing (encoding, imputation, scaling) fitted on O$_{tr}$ only")
    arrow(0.5, 0.91, 0.5, 0.885)
    box(0.25, 0.725, 0.46, 0.105, "Clean: resampler fitted on O$_{tr}$ $\\rightarrow$ G$_{clean}$\n"
        "train on O$_{tr}$ $+$ G$_{clean}$  $\\Rightarrow$  clean model")
    box(0.75, 0.725, 0.46, 0.105, "Leaky: resampler fitted on O$_{tr}$ $\\cup$ O$_{te}$ $\\rightarrow$ G$_{full}$\n"
        "random split: G$_{test}$ ($n^-_{te} - n^+_{te}$ rows) and G$_{leak}$")
    box(0.75, 0.575, 0.46, 0.10, "train on O$_{tr}$ $+$ G$_{leak}$  $\\Rightarrow$  leaky model\n"
        "(same size and class counts as the clean training set)")
    arrow(0.40, 0.825, 0.28, 0.778); arrow(0.60, 0.825, 0.72, 0.778); arrow(0.75, 0.672, 0.75, 0.626)
    xs = [0.125, 0.375, 0.625, 0.875]
    specs = [("CON", "clean model\nO$_{te}$\nnatural prevalence"),
             ("LON", "leaky model\nO$_{te}$\nnatural prevalence"),
             ("LMN", "leaky model\nO$_{te}$ $+$ G$_{test}$\nreweighted to natural"),
             ("LMB", "leaky model\nO$_{te}$ $+$ G$_{test}$\nunweighted (50/50)")]
    for x, (name, desc) in zip(xs, specs):
        box(x, 0.30, 0.215, 0.19, "%s\n%s" % (fs.bf(name), desc), fsz=fs.ANNOT)
    arrow(0.20, 0.672, 0.13, 0.397); arrow(0.70, 0.524, 0.39, 0.397)
    arrow(0.75, 0.524, 0.63, 0.397); arrow(0.80, 0.524, 0.87, 0.397)
    for x0, x1, k, txt in [(0.234, 0.266, "L", "L: training-side leakage"),
                           (0.484, 0.516, "S", "S: synthetic-test mixture"),
                           (0.734, 0.766, "P", "P: prevalence shift")]:
        arrow(x0, 0.30, x1, 0.30, color=fs.COLORS[k])
        ax.text((x0 + x1) / 2, 0.165, txt, ha="center", va="center", fontsize=fs.ANNOT, color=fs.COLORS[k])
    ax.text(0.5, 0.065, "Each adjacent pair of states differs in exactly one condition.   "
            "T $=$ M(LMB) $-$ M(CON) $=$ L $+$ S $+$ P exactly, in every fold and seed.",
            ha="center", fontsize=fs.ANNOT, color="0.25")
    fs.logline("   （示意图：不含数据数值）")
    return fs.save(fig, "fig_four_state_design")


# ---------------------------------------------------------------- 图 F
def fig_winner_shift():
    import rank_flip as RF
    R_, _ = RF.analyse("auc", G.CELLS)
    wc, wl = RF.win_counts(R_, "CON"), RF.win_counts(R_, "LMB")      # 并列第一各记 1/k 胜
    n = len(R_)
    fs.logline("\n## FIG F · 胜者分布（ROC-AUC，%d 个 数据集×重采样器×种子 单元格；CON vs LMB）" % n)
    fig, ax = plt.subplots(figsize=(fs.TEXTWIDTH * 0.60, 2.35))
    x = np.arange(len(C.MODEL_NAMES)); wdt = 0.38
    cvals = [wc.get(m, 0.0) for m in C.MODEL_NAMES]
    lvals = [wl.get(m, 0.0) for m in C.MODEL_NAMES]
    ax.bar(x - wdt / 2, cvals, wdt, label="Correct protocol (CON)", color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.bar(x + wdt / 2, lvals, wdt, label="As-published endpoint (LMB)", color="#C44E52", edgecolor="white", linewidth=0.5)
    off = 0.012 * max(cvals + lvals)
    for xi, (c, l) in enumerate(zip(cvals, lvals)):
        ax.text(xi - wdt / 2, c + off, RF.fmt_count(c), ha="center", fontsize=fs.ANNOT, color="#4C72B0")
        ax.text(xi + wdt / 2, l + off, RF.fmt_count(l), ha="center", fontsize=fs.ANNOT, color="#C44E52")
        fs.logline("   %-13s CON %s (%.1f%%)  LMB %s (%.1f%%)"
                   % (C.MODEL_NAMES[xi], RF.fmt_count(c), 100 * c / n, RF.fmt_count(l), 100 * l / n))
    ax.set_xticks(x); ax.set_xticklabels([MSHORT[m] for m in C.MODEL_NAMES])
    ax.set_xlabel("Selected best model")
    ax.set_ylabel("Cells won (of %d)" % n)
    ax.set_ylim(0, max(lvals + cvals) * 1.18)
    ax.legend(loc="upper left", handlelength=1.1, borderaxespad=0.3)
    return fs.save(fig, "fig_winner_shift")


if __name__ == "__main__":
    print("生成配图（TEXTWIDTH = %.3f in，usetex=%s）" % (fs.TEXTWIDTH, fs.USE_TEX))
    sizes = {}
    for fn in (fig_decomposition_by_dataset, fig_composition_by_capacity, fig_ladder,
               fig_state_path, fig_four_state_design, fig_winner_shift):
        r = fn()
        if r:
            sizes[fn.__name__] = r
    fs.logline("\n## 图尺寸（英寸 @ %d dpi）" % fs.DPI)
    for k, (w, h) in sizes.items():
        fs.logline("   %-32s %.2f x %.2f in" % (k, w, h))
    print("\n✅ 图与 figure_numbers.log 已生成于 %s" % fs.FIGDIR)
