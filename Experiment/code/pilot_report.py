"""四状态试点报告 —— 从落盘 CSV 独立重算全部不变量（不依赖 states.py 的运行时断言）。

用法：python3 pilot_report.py   → Experiment/pilot_4state/PILOT_REPORT.md
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import states as S

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.abspath(os.path.join(HERE, "..", "pilot_4state"))


def main():
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PD, "raw", "*.csv")))])
    dg = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PD, "raw_diag", "*.csv")))])
    key = ["dataset", "model", "resampler", "seed", "fold"]
    W = raw.pivot_table(index=key, columns="state", values=list(S.METRICS) + ["prevalence", "n_eval", "n_pos", "n_neg"])
    n_folds = len(W)
    L = []
    A = L.append
    A("# 四状态试点报告（PILOT_REPORT）")
    A("")
    A("> 由 `code/pilot_report.py` 生成，勿手改。全部检验**从落盘 CSV 独立重算**，不依赖 `states.py` 的运行时断言。")
    A("")
    A("- 单元格：%d（%s）；折：%d；每折 8 个状态（4 主 + 4 补充因子态）"
      % (raw.groupby(key[:4]).ngroups, "；".join("%s×%s×%s" % k + " %d 种子" % g.seed.nunique()
         for k, g in raw.groupby(["dataset", "model", "resampler"])), n_folds))
    A("")
    A("## 1. 不变量（独立重算）")
    A("")
    rows = []
    tel = max(abs((W[(m, "LMB")] - W[(m, "CON")]) - ((W[(m, "LON")] - W[(m, "CON")]) + (W[(m, "LMN")] - W[(m, "LON")]) + (W[(m, "LMB")] - W[(m, "LMN")]))).max() for m in S.METRICS)
    rows.append(("逐折 max |T − (L+S+P)|（6 指标）", "%.1e" % tel, tel < 1e-12))
    aucP = (W[("auc", "LMB")] - W[("auc", "LMN")]).abs().max()
    rows.append(("逐折 max |AUC(LMB) − AUC(LMN)|（P 的解析零）", "%.1e" % aucP, aucP < 1e-12))
    ok_prev_on = (W[("prevalence", "CON")] == W[("prevalence", "LON")]).all()
    rows.append(("CON 与 LON 流行率逐位相同", str(bool(ok_prev_on)), bool(ok_prev_on)))
    d_prev = (W[("prevalence", "LMN")] - W[("prevalence", "CON")]).abs().max()
    rows.append(("max |流行率(LMN) − 流行率(CON)|（加权回自然值）", "%.1e" % d_prev, d_prev < 1e-12))
    ok_lmb = (W[("prevalence", "LMB")] == 0.5).all()
    rows.append(("LMB 流行率全部恰为 0.5", str(bool(ok_lmb)), bool(ok_lmb)))
    ok_n = ((W[("n_eval", "CON")] == W[("n_eval", "LON")]) & (W[("n_eval", "LMN")] == W[("n_eval", "LMB")])).all()
    rows.append(("n_eval：CON = LON 且 LMN = LMB", str(bool(ok_n)), bool(ok_n)))
    ok_bal = (W[("n_pos", "LMB")] == W[("n_neg", "LMB")]).all()
    rows.append(("混合测试人群正负类计数相等", str(bool(ok_bal)), bool(ok_bal)))
    ok_tm = bool((dg.n_pos_train_clean == dg.n_pos_train_leak).all() and dg.train_match.all())
    rows.append(("CON 与泄漏训练集样本量、类别计数一致（全部折）", str(ok_tm), ok_tm))
    ok_pre = bool(dg.pre_fit_equals_Otr.all() and (dg.pre_fit_n == dg.n_tr).all())
    rows.append(("预处理拟合行 = O_tr（全部折）", str(ok_pre), ok_pre))
    bp = dg.batch_pred_maxdiff.max()
    rows.append(("泄漏模型逐行预测与批次无关：max 差", "%.1e" % bp, bp < 1e-9))
    ok_k = bool((dg.k_test == dg.n_neg_te - dg.n_pos_te).all())
    rows.append(("k_test = n_neg_te − n_pos_te（全部折）", str(ok_k), ok_k))
    A("| 检验 | 实测 | 通过 |")
    A("|---|---|---|")
    for name, val, ok in rows:
        A("| %s | %s | %s |" % (name, val, "✅" if ok else "❌"))
    A("")
    A("### 1.1 合成行计数与修剪（按重采样器）")
    A("")
    A("| 重采样器 | 折数 | k_clean = k_leak（未修剪）折占比 | 修剪折数 | 最大修剪行数（干净侧 / 泄漏侧） |")
    A("|---|---:|---:|---:|---|")
    for r, g in dg.groupby("resampler"):
        eq = (g.k_clean_raw == g.k_leak_raw).mean()
        nt = int(((g.trim_clean > 0) | (g.trim_leak > 0)).sum())
        A("| %s | %d | %.0f%% | %d | %d / %d |" % (r, len(g), 100 * eq, nt, g.trim_clean.max(), g.trim_leak.max()))
    A("")
    A("## 2. 贡献量初读（SMOTE；逐折计算后按 折→种子 取均值；L+S+P ≡ T）")
    A("")
    sm = raw[raw.resampler == "SMOTE"]
    cm = sm.groupby(["dataset", "model", "seed", "state"])[list(S.METRICS)].mean()
    for metric in ["f1", "auc"]:
        A("### %s" % metric.upper())
        A("")
        A("| 数据集 | 模型 | CON | L | S | P | T | L+S+P−T |")
        A("|---|---|---:|---:|---:|---:|---:|---:|")
        for (ds, mo), g in sm.groupby(["dataset", "model"]):
            st = {s: cm.loc[(ds, mo)].xs(s, level="state")[metric].mean() for s in S.MAIN_STATES}
            c = S.contributions(st)
            A("| %s | %s | %.4f | %+.4f | %+.4f | %+.4f | %+.4f | %.0e |"
              % (ds, mo, st["CON"], c["L"], c["S"], c["P"], c["T"], c["L"] + c["S"] + c["P"] - c["T"]))
        A("")
    out = os.path.join(PD, "PILOT_REPORT.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if all(ok for _, _, ok in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
