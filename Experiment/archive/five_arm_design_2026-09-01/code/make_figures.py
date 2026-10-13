"""生成论文配图。

铁律（KB figure-standards + SOP P3）：
  * 脚本内**不硬编码任何数值**，全部从原始 CSV 现算
  * 所有标注数字打印到 stdout **并同时写进 figure_numbers.log**，供审计 diff 图注
  * 图内**不放 "Fig. N"** —— 排版顺序一变内外编号就打架，编号交给 caption
"""
from __future__ import annotations
import os, sys, glob, warnings
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
from decision import verdict, NOISE_FLOOR
import build_ground_truth as G

fs.apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, C.RAW_DIR))
MECH = ["m1_prevalence", "m2_synthetic_rows", "m3_true_leakage"]

if os.path.exists(fs.LOGPATH):
    os.remove(fs.LOGPATH)
fs.logline("# figure_numbers.log — 由 make_figures.py 现算，图注数字须与本文件一致")
fs.logline("# 正文宽度 TEXTWIDTH = %.3f in (%.0f pt)，实测自已发表 PDF" % (fs.TEXTWIDTH, fs.TEXTWIDTH * 72))

dec = pd.read_csv(os.path.join(RAW, "..", "decomposed_cells.csv"))
sm = dec[dec.resampler == "SMOTE"]
DS_ORDER = ["creditcard", "bank-marketing", "credit-default", "adult", "credit-g", "pima"]
DS_SHORT = {"creditcard": "creditcard", "bank-marketing": "bank-mkt",
            "credit-default": "credit-def", "adult": "adult",
            "credit-g": "credit-g", "pima": "pima"}


# ---------------------------------------------------------------- 图 A
def fig_decomposition_by_dataset():
    """三机制构成，按数据集（F1 与 AUC 两个 panel）。"""
    fs.logline("\n## FIG A · 三机制分解，按数据集（SMOTE，跨模型与种子取中位）")
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.45), sharey=False)
    for ax, metric, mlab in zip(axes, ["f1", "auc"], ["F1", "ROC-AUC"]):
        x = np.arange(len(DS_ORDER))
        bottoms_pos = np.zeros(len(DS_ORDER))
        for key in MECH:
            vals = np.array([sm[sm.dataset == d]["%s_%s" % (metric, key)].median() for d in DS_ORDER])
            ax.bar(x, vals, 0.62, bottom=bottoms_pos, label=fs.LABELS[key],
                   color=fs.COLORS[key], edgecolor="white", linewidth=0.5)
            bottoms_pos += vals
            for d, v in zip(DS_ORDER, vals):
                fs.logline("   %-4s %-16s %-20s %+.4f" % (mlab, d, key, v))
        # 观测总差 A−B 的中位：堆叠高度是【三个中位之和】，与观测总差不等（§4.5 非可加性）。
        # 不画这条线，堆叠柱会被读成观测总差 —— 最大处高估 56%（credit-g, AUC）。
        obs = np.array([sm[sm.dataset == d]["%s_total" % metric].median() for d in DS_ORDER])
        ax.scatter(x, obs, marker="_", s=210, color="0.15", zorder=5, linewidths=1.4,
                   label="Observed $A-B$ (median)")
        for d, o, st in zip(DS_ORDER, obs, bottoms_pos):
            fs.logline("   %-4s %-16s %-20s %+.4f  (堆叠和 %+.4f，差 %+.4f"
                       "；正文显示 %+.3f vs %+.3f)"
                       % (mlab, d, "observed_total", o, st, st - o, o, st))
        ax.axhline(0, color="0.3", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([DS_SHORT[d] for d in DS_ORDER], rotation=35, ha="right")
        ax.set_ylabel(r"Decomposition of $\Delta$ %s" % mlab)
        ax.set_title(fs.bf(mlab))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, handlelength=1.1,
               bbox_to_anchor=(0.5, -0.30), borderaxespad=0.0)
    fs.panel_letters(fig, axes)
    fig.subplots_adjust(wspace=0.30)
    return fs.save(fig, "fig_decomposition_by_dataset")


# ---------------------------------------------------------------- 图 B
def fig_composition_by_capacity():
    """机制构成占比随模型容量变化 —— 本文的主结果。"""
    fs.logline("\n## FIG B · 机制构成占比 × 模型容量（SMOTE，跨数据集与种子取中位）")
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.45))
    for ax, metric, mlab in zip(axes, ["f1", "auc"], ["F1", "ROC-AUC"]):
        x = np.arange(len(C.MODEL_NAMES))
        meds = {k: np.array([sm[sm.model == m]["%s_%s" % (metric, k)].median()
                             for m in C.MODEL_NAMES]) for k in MECH}
        tot = np.sum([np.abs(meds[k]) for k in MECH], axis=0)
        bottom = np.zeros(len(C.MODEL_NAMES))
        for key in MECH:
            share = 100 * np.abs(meds[key]) / np.where(tot == 0, np.nan, tot)
            ax.bar(x, share, 0.62, bottom=bottom, label=fs.LABELS[key],
                   color=fs.COLORS[key], edgecolor="white", linewidth=0.5)
            for xi, (s_, v_) in enumerate(zip(share, meds[key])):
                if np.isfinite(s_) and s_ > 7:
                    ax.text(xi, bottom[xi] + s_ / 2, "%.0f%s" % (s_, fs.PCT), ha="center",
                            va="center", fontsize=fs.ANNOT, color="white")
                fs.logline("   %-4s %-13s %-20s median=%+.4f share=%.1f%%"
                           % (mlab, C.MODEL_NAMES[xi], key, v_, s_))
            bottom += np.nan_to_num(share)
        ax.set_xticks(x)
        ax.set_xticklabels(["LR", "DT", "RF", "XGB"])
        ax.set_xlabel("Model (nominal flexibility $\\rightarrow$)")
        ax.set_ylim(0, 100)
        # W7（EXTERNAL_REVIEW）：份额的分母是三机制中位数之和，不是观测 A−B —— 轴标签须如实
        ax.set_ylabel("Share of (1)$+$(2)$+$(3) (%s)" % fs.PCT)
        ax.set_title(fs.bf(mlab))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, handlelength=1.1,
               bbox_to_anchor=(0.5, -0.13), borderaxespad=0.0)
    fs.panel_letters(fig, axes)
    fig.subplots_adjust(wspace=0.30)
    return fs.save(fig, "fig_composition_by_capacity")


# ---------------------------------------------------------------- 图 C
def fig_ladder():
    """受控阶梯：③ 随不平衡度。两个数据集各一 panel（天花板效应对照）。"""
    lad, files = G.load_raw("ladder-*.csv", required=False)
    if lad is None:
        print("  [skip] 无阶梯数据"); return None
    lad = lad.copy(); lad["ratio"] = lad["ratio"].astype(int)
    dl = G.decompose_cells(G.cell_means(lad.assign(dataset=lad.dataset + "@r" + lad.ratio.astype(str))))
    dl["ratio"] = dl.dataset.str.split("@r").str[1].astype(int)
    dl["ds"] = dl.dataset.str.split("@r").str[0]

    fs.logline("\n## FIG C · 受控阶梯：③真泄漏 vs 不平衡度（AUC）")
    dss = ["creditcard", "bank-marketing"]
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.45), sharey=True)
    for ax, ds in zip(axes, dss):
        g = dl[dl.ds == ds]
        for m in C.MODEL_NAMES:
            gm = g[g.model == m]
            rr = sorted(gm.ratio.unique())
            meds = [gm[gm.ratio == r]["auc_m3_true_leakage"].median() for r in rr]
            rho = spearmanr(rr, meds).correlation if len(rr) > 2 else np.nan
            ax.plot(rr, meds, marker=fs.MARKERS[m], color=fs.COLORS[m],
                    label=m, clip_on=False)
            # 档位标签 = 实际 少数:多数 = 1:(r−1)，与 Table 2 口径一致（DEVIATIONS D-007）
            fs.logline("   %-16s %-13s rho=%+.3f  %s" % (ds, m, rho,
                       " ".join("1:%d=%+.4f" % (r - 1, v) for r, v in zip(rr, meds))))
            if len(meds) > 1:
                d_min = float(np.min(np.diff(meds)))
                fs.logline("   %-16s %-13s 相邻档最小Δ = %+.1e%s"
                           % (ds, m, d_min, "  ← 负档差（该曲线非单调不减）" if d_min < 0 else ""))
        base = dl[(dl.ds == ds) & (dl.ratio == 2)]["auc_B"].median()
        fs.logline("   %-16s 1:1 平衡档基线 AUC(B) 中位 = %.4f （面板标题显示 %.2f）" % (ds, base, base))
        for m in C.MODEL_NAMES:
            mb = dl[(dl.ds == ds) & (dl.model == m)]["auc_B"].median()
            fs.logline("   %-16s %-13s 全档 B 臂 AUC 中位 = %.4f （正文显示 %.3f）" % (ds, m, mb, mb))
        ax.axhline(NOISE_FLOOR, color="0.55", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 64, 256])
        ax.set_xticklabels(["1:%d" % (r - 1) for r in (2, 4, 8, 16, 64, 256)])
        ax.set_xlabel("Imbalance ratio (minority : majority)")
        ax.set_title(fs.bf("%s  (baseline AUC $=$ %.2f)" % (ds, base)))
    axes[0].set_ylabel(r"(3) True leakage, $\Delta$AUC")
    axes[0].legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
    axes[1].text(0.98, NOISE_FLOOR, "noise floor", fontsize=fs.ANNOT, color="0.45",
                 transform=axes[1].get_yaxis_transform(), ha="right", va="bottom")
    fs.panel_letters(fig, axes)
    fig.subplots_adjust(wspace=0.12)
    return fs.save(fig, "fig_ladder")


# ---------------------------------------------------------------- 图 D
def fig_additivity():
    """可加性检查：①+②+③ vs 观测总差 A−B。"""
    fs.logline("\n## FIG D · 可加性：机制之和 vs 观测总差")
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.45))
    for ax, metric, mlab in zip(axes, ["f1", "auc"], ["F1", "ROC-AUC"]):
        parts = sum(sm["%s_%s" % (metric, k)] for k in MECH)
        obs = sm["%s_total" % metric]
        for m in C.MODEL_NAMES:
            msk = sm.model == m
            ax.scatter(parts[msk], obs[msk], s=6, alpha=0.45, color=fs.COLORS[m],
                       marker=fs.MARKERS[m], label=m, linewidths=0)
        lo = float(min(parts.min(), obs.min())); hi = float(max(parts.max(), obs.max()))
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.35", lw=0.8, ls="--")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(r"(1) $+$ (2) $+$ (3)")
        ax.set_ylabel(r"Observed total ($A-B$)")
        ax.set_title(fs.bf(mlab))
        gap = (obs - parts)
        fs.logline("   %-4s 可加性缺口 中位=%+.4f  p10=%+.4f  p90=%+.4f  |缺口|>0.01 占 %.1f%%"
                   % (mlab, gap.median(), gap.quantile(.1), gap.quantile(.9),
                      100 * (gap.abs() > 0.01).mean()))
    axes[0].legend(loc="upper left", handlelength=1.0, borderaxespad=0.2, markerscale=1.6)
    fs.panel_letters(fig, axes)
    fig.subplots_adjust(wspace=0.30)
    return fs.save(fig, "fig_additivity")


# ---------------------------------------------------------------- 图 E（示意图，无数据数值）
def fig_five_arm_design():
    """五臂设计示意 —— 纯示意图，不含任何数据数值（应用读者靠 §3.2 的表理解五臂很吃力）。"""
    fs.logline("\n## FIG E · 五臂设计示意（纯示意图，无数据数值）")
    fig, ax = plt.subplots(figsize=(fs.TEXTWIDTH, 3.3))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#F2F2F2", ec="0.35", fsz=fs.ANNOT, bold=False):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc,
                                   edgecolor=ec, linewidth=0.8, zorder=2))
        ax.text(x, y, fs.bf(text) if bold else text, ha="center", va="center",
                fontsize=fsz, zorder=3, linespacing=1.25)
        return x, y, w, h

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.9,
                                    shrinkA=1, shrinkB=1), zorder=1)

    # 顶部：原始数据
    box(0.5, 0.94, 0.34, 0.085, "Original data (imbalanced)", fc="#E8E8E8", bold=True)
    # 分支标题
    ax.text(0.235, 0.845, fs.bf("Correct pipeline"), ha="center", fontsize=fs.FS_TICK)
    ax.text(0.745, 0.845, fs.bf("As-published pipeline"), ha="center", fontsize=fs.FS_TICK)
    # 左支：先划分
    box(0.235, 0.755, 0.30, 0.075, "1. Split (stratified 5-fold CV)")
    box(0.235, 0.625, 0.30, 0.075, "2. Resample the training fold only")
    box(0.235, 0.495, 0.30, 0.075, "3. Train model (clean)")
    arrow(0.42, 0.90, 0.235, 0.795); arrow(0.235, 0.715, 0.235, 0.665)
    arrow(0.235, 0.585, 0.235, 0.535)
    # 右支：先重采样
    box(0.745, 0.755, 0.30, 0.075, "1. Resample the full data (to 1:1)")
    box(0.745, 0.625, 0.30, 0.075, "2. Split (stratified 5-fold CV)")
    box(0.745, 0.495, 0.30, 0.075, "3. Train model (leaky)")
    arrow(0.58, 0.90, 0.745, 0.795); arrow(0.745, 0.715, 0.745, 0.665)
    arrow(0.745, 0.585, 0.745, 0.535)

    # 评估行（右侧四框用小一号字，避免溢出）
    evalrow = [
        (0.125, "B", "original\ntest fold", 0.20, fs.ANNOT),
        (0.345, "E", "same fold,\nreweighted 50/50", 0.20, fs.ANNOT),
        (0.5575, "A", "all test rows\n(orig. $+$ synth.)", 0.115, 6.4),
        (0.6825, "C", "original\nrows only", 0.115, 6.4),
        (0.8075, "C$_2$", "original rows,\nminority $= k$", 0.115, 6.4),
        (0.9325, "D", "orig. majority\n$+$ $k$ synthetic", 0.115, 6.4),
    ]
    for x, name, desc, w, fsz in evalrow:
        box(x, 0.315, w, 0.135, "%s\n%s" % (fs.bf(name), desc), fsz=fsz)
    arrow(0.19, 0.455, 0.125, 0.385); arrow(0.28, 0.455, 0.345, 0.385)
    for x in (0.5575, 0.6825, 0.8075, 0.9325):
        arrow(0.745, 0.455, x, 0.385)

    # 机制公式（配色与其余图一致）
    for x, key, txt in [(0.17, "m1_prevalence", "(1) Prevalence shift $= E - B$"),
                        (0.50, "m2_synthetic_rows", "(2) Synthetic test rows $= D - C_2$"),
                        (0.83, "m3_true_leakage", "(3) True leakage $= C - B$")]:
        ax.text(x, 0.135, txt, ha="center", fontsize=fs.FS_TICK,
                color=fs.COLORS[key], fontweight="normal")
    ax.text(0.5, 0.035, "Each contrast differs in exactly one factor. "
                        "C$_2$ $=$ C whenever fold synthetic $\\geq$ original minority rows.",
            ha="center", fontsize=fs.ANNOT, color="0.30")
    fs.logline("   （示意图：不含数据数值；C2 = 与 D 等量等流行率的 C 下采样版）")
    return fs.save(fig, "fig_five_arm_design")


# ---------------------------------------------------------------- 图 F
def fig_winner_shift():
    """胜者分布：正确协议 vs 照文献协议（ROC-AUC）。Table 4b 的图形化。"""
    import rank_flip as RF
    R_, _ = RF.analyse("auc")
    wb, wa = R_.winner_B.value_counts(), R_.winner_A.value_counts()
    n = len(R_)
    fs.logline("\n## FIG F · 胜者分布（ROC-AUC，%d 个 数据集×重采样器×种子 单元格）" % n)
    fig, ax = plt.subplots(figsize=(fs.TEXTWIDTH * 0.60, 2.35))
    x = np.arange(len(C.MODEL_NAMES)); wdt = 0.38
    bvals = [int(wb.get(m, 0)) for m in C.MODEL_NAMES]
    avals = [int(wa.get(m, 0)) for m in C.MODEL_NAMES]
    ax.bar(x - wdt / 2, bvals, wdt, label="Correct protocol (B)",
           color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.bar(x + wdt / 2, avals, wdt, label="As-published protocol (A)",
           color="#C44E52", edgecolor="white", linewidth=0.5)
    for xi, (b, a) in enumerate(zip(bvals, avals)):
        ax.text(xi - wdt / 2, b + 6, "%d" % b, ha="center", fontsize=fs.ANNOT, color="#4C72B0")
        ax.text(xi + wdt / 2, a + 6, "%d" % a, ha="center", fontsize=fs.ANNOT, color="#C44E52")
        fs.logline("   %-13s B %3d (%.1f%%)  A %3d (%.1f%%)"
                   % (C.MODEL_NAMES[xi], b, 100 * b / n, a, 100 * a / n))
    ax.set_xticks(x); ax.set_xticklabels(["LR", "DT", "RF", "XGB"])
    ax.set_xlabel("Selected best model")
    ax.set_ylabel("Cells won (of %d)" % n)
    ax.set_ylim(0, max(avals + bvals) * 1.18)
    ax.legend(loc="upper left", handlelength=1.1, borderaxespad=0.3)
    return fs.save(fig, "fig_winner_shift")


if __name__ == "__main__":
    print("生成配图（TEXTWIDTH = %.3f in，usetex=%s）" % (fs.TEXTWIDTH, fs.USE_TEX))
    sizes = {}
    for fn in (fig_decomposition_by_dataset, fig_composition_by_capacity,
               fig_ladder, fig_additivity, fig_five_arm_design, fig_winner_shift):
        r = fn()
        if r:
            sizes[fn.__name__] = r
    fs.logline("\n## 图尺寸（英寸 @ %d dpi）" % fs.DPI)
    for k, (w, h) in sizes.items():
        fs.logline("   %-32s %.2f x %.2f in" % (k, w, h))
    print("\n✅ 图与 figure_numbers.log 已生成于 %s" % fs.FIGDIR)
