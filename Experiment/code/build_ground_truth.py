"""真源生成器 —— `RESULTS_GROUND_TRUTH.md` 的唯一来源（四状态有序路径分解）。

KB 铁律：所有见报数字从原始 CSV 现算，不从任何中间稿抄。

聚合约定：每个 (数据集, 模型, 重采样器, 种子) 单元格先对 5 折取均值；跨种子 / 跨单元格一律取**均值**，
并用均值的百分位自助 95% CI（B=10,000，固定种子）判定"确立"（CI 整体落在 ±0.003 之外）。
均值是线性的，因此 L、S、P 的均值在每个聚合层级都恰好加总为 T 的均值；中位数不可加，只作稳健性对照（§9）。

用法：python3 build_ground_truth.py            （主数据：raw/ 与 raw_diag/）
      RLA_DATA=pilot python3 build_ground_truth.py （试点数据，只用于调试本脚本）
"""
from __future__ import annotations
import os, sys, glob, datetime, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import states as S
from decision import (verdict_mean, verdict_median, bootstrap_mean_ci, NOISE_FLOOR,
                      consistency_across_datasets)
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.environ.get("RLA_DATA") == "pilot"
RAW = os.path.abspath(os.path.join(HERE, C.PILOT_RAW_DIR if PILOT else C.RAW_DIR))
DIAG = os.path.abspath(os.path.join(HERE, C.PILOT_DIAG_DIR if PILOT else C.DIAG_DIR))
EXP = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(EXP, "pilot_4state", "GT_PILOT.md") if PILOT else os.path.join(EXP, "RESULTS_GROUND_TRUTH.md")
CELLS = os.path.join(EXP, "pilot_4state", "cells_pilot.csv") if PILOT else os.path.join(EXP, "decomposed_cells.csv")
ARCHIVE = os.path.join(EXP, "archive", "five_arm_design_2026-09-01")
METRICS = list(S.METRICS)
KEYS = ["dataset", "model", "resampler", "seed"]
CLAB = {"L": "L 训练侧泄漏", "S": "S 合成测试行混入", "P": "P 流行率", "T": "T 总虚高"}
MAIN_TAG = "pilot" if PILOT else "main"


def load(pattern, where=RAW, validate=True):
    """读入一组单元格 CSV。validate=True（四状态数据）时逐文件校验完整性，任何一个不完整即中止：
    真源绝不在少于 5 折或缺状态的单元格上静默取均值。归档的旧五臂数据格式不同，以 validate=False 读入。"""
    files = sorted(glob.glob(os.path.join(where, pattern)))
    if not files:
        return None, []
    frames, bad = [], []
    for f in files:
        d = pd.read_csv(f)
        if validate:
            ok = (S.state_frame_complete(d, C.N_SPLITS) if "state" in d.columns
                  else S.diag_frame_complete(d, C.N_SPLITS))
            if not ok:
                bad.append(os.path.basename(f))
        frames.append(d)
    if bad:
        raise RuntimeError("真源中止：%d 个单元格文件不完整（缺折 / 缺状态 / 指标缺失），须重跑：%s"
                           % (len(bad), ", ".join(bad[:10])))
    return pd.concat(frames, ignore_index=True), files


def robust_invariants(dg, label):
    """稳健性网格的运行时不变量实测（诊断文件逐折记录；任一违反在运行时即已中止该单元格）。"""
    return ("%s不变量（%d 折）：逐折 max |T − (L+S+P)| = %.1e；ROC-AUC 上类内重加权的 max 影响 = %.1e；"
            "训练集匹配 %s；预处理拟合行 = O_tr %s；k_test 公式 %s。"
            % (label, len(dg), dg.telescope_maxabs.max(), dg.auc_reweight_maxabs.max(),
               "全部成立" if dg.train_match.all() else "❌ 有违反",
               "全部成立" if dg.pre_fit_equals_Otr.all() else "❌ 有违反",
               "全部成立" if (dg.k_test == dg.n_neg_te - dg.n_pos_te).all() else "❌ 有违反"))


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """折 → 单元格：每状态每指标取 5 折均值；再由四个主状态算 L/S/P/T（线性关系在折均值上依然精确）。"""
    cm = df.groupby(KEYS + ["state"])[METRICS].mean().unstack("state")
    out = pd.DataFrame(index=cm.index)
    for m in METRICS:
        for s in S.STATES:
            if (m, s) in cm.columns:
                out["%s_%s" % (m, s)] = cm[(m, s)]
        c = S.contributions({s: cm[(m, s)] for s in S.MAIN_STATES})
        for k in S.CONTRIB:
            out["%s_%s" % (m, k)] = c[k]
        out["%s_telescope" % m] = c["T"] - (c["L"] + c["S"] + c["P"])
    return out.reset_index()


def fmt(v, flag=True):
    star = "" if (not flag or v["ci_excludes_floor"]) else " (n.e.)"
    return "%+.4f [%+.4f, %+.4f]%s" % (v["mean"], v["lo"], v["hi"], star)


def shares(g, metric):
    T = g["%s_T" % metric].mean()
    return {k: (g["%s_%s" % (metric, k)].mean() / T if abs(T) > 1e-12 else np.nan) for k in "LSP"}


# ---------------------------------------------------------------- 顺序依赖（补充因子态）
FACTORS = ("t", "m", "p")        # 训练 C→L；测试组成 O→M；流行率 N→B


def state_name(t, m, p):
    return ("L" if t else "C") + ("M" if m else "O") + ("B" if p else "N")


def order_decomposition(vals: dict) -> dict:
    """vals: 状态名 → 指标值（8 个因子态）。返回每个因子在 6 种顺序下的贡献。"""
    out = {f: [] for f in FACTORS}
    main = None
    for order in itertools.permutations(FACTORS):
        cur = {"t": 0, "m": 0, "p": 0}
        contrib = {}
        for f in order:
            before = vals[state_name(**cur)]
            cur[f] = 1
            contrib[f] = vals[state_name(**cur)] - before
        for f in FACTORS:
            out[f].append(contrib[f])
        if order == ("t", "m", "p"):
            main = contrib
    return {f: dict(main=main[f], shapley=float(np.mean(out[f])),
                    lo=float(np.min(out[f])), hi=float(np.max(out[f]))) for f in FACTORS}


def main():
    df, files = load("%s__*.csv" % MAIN_TAG)
    if df is None:
        raise SystemExit("raw/ 下没有 %s__*.csv" % MAIN_TAG)
    dg, _ = load("%s__*.csv" % MAIN_TAG, DIAG)
    cells = cell_table(df)
    sm = cells[cells.resampler == "SMOTE"]

    L = []
    A = L.append
    A("# 结果真源（RESULTS_GROUND_TRUTH）—— 四状态有序路径分解" + ("（试点）" if PILOT else ""))
    A("")
    A("> **本文件由 `code/build_ground_truth.py` 生成，请勿手工编辑。** 论文、图注、摘要里的任何数字都必须能在此查到。")
    A("")
    A("- 生成时刻：%s" % datetime.datetime.now().isoformat(timespec="seconds"))
    A("- 预注册 SHA-256：`%s`（数据、模型、重采样器、种子、截断规则沿用；分解设计见 DEVIATIONS D-011）" % C.PREREG_SHA256)
    A("- 原始 CSV：**%d 个状态文件**（%d 行 = 折 × 状态）+ %d 个诊断文件（%d 折）"
      % (len(files), len(df), 0 if dg is None else dg.groupby(KEYS).ngroups, 0 if dg is None else len(dg)))
    A("- 状态：主 CON → LON → LMN → LMB；补充因子态 COB / LOB / CMN / CMB（仅 §8 顺序依赖分析）")
    A("- 贡献：L = LON − CON · S = LMN − LON · P = LMB − LMN · T = LMB − CON = L + S + P")
    A("- 汇总：折均值 → 单元格；跨种子/单元格取**均值**；判据 = 均值的自助 95%% CI（B=10,000）整体落在 ±%.3f 之外" % NOISE_FLOOR)
    runlog = os.path.join(EXP, "full_rerun_4state.log")
    if os.path.exists(runlog) and not PILOT:
        import re as _re
        for mode, mins in _re.findall(r"模式=(\w+) 完成：.*?总耗时 ([\d.]+) 分钟", open(runlog).read()):
            A("- 运行耗时（%s，8 进程并行，无 GPU）：**%s 分钟**" % (mode, mins))
    A("")

    # ================= 0. 数据完整性与不变量 =================
    A("## 0. 数据完整性与不变量")
    A("")
    A("| 数据集 | 行数 | 正类比例 | 单元格数（模型×重采样器×种子） |")
    A("|---|---:|---:|---:|")
    for ds in sorted(df.dataset.unique()):
        d = df[df.dataset == ds]
        A("| %s | %d | %.4f | %d |" % (ds, d.n_rows.iloc[0], d.prevalence_orig.iloc[0],
                                        d.groupby(["model", "resampler", "seed"]).ngroups))
    A("")
    W = df.pivot_table(index=KEYS + ["fold"], columns="state",
                       values=METRICS + ["prevalence", "n_eval", "n_pos", "n_neg"])
    tel = max(float((W[(m, "LMB")] - W[(m, "CON")] - ((W[(m, "LON")] - W[(m, "CON")]) + (W[(m, "LMN")] - W[(m, "LON")])
                                                  + (W[(m, "LMB")] - W[(m, "LMN")]))).abs().max()) for m in METRICS)
    aucP = float((W[("auc", "LMB")] - W[("auc", "LMN")]).abs().max())
    aux0 = max(float((W[("auc", a)] - W[("auc", b)]).abs().max()) for a, b in (("COB", "CON"), ("LOB", "LON"), ("CMB", "CMN")))
    prevN = float((W[("prevalence", "LMN")] - W[("prevalence", "CON")]).abs().max())
    A("**不变量（从状态 CSV 独立重算，另与诊断文件交叉核对；运行时任一违反即中止该单元格）**：")
    A("")
    A("| 检验 | 实测 | 判定 |")
    A("|---|---|---|")
    chk = [
        ("逐折 max \\|T − (L + S + P)\\|（6 指标，%d 折）" % len(W), "%.1e" % tel, tel < 1e-12),
        ("逐折 max \\|AUC(LMB) − AUC(LMN)\\|（P 的解析零）", "%.1e" % aucP, aucP < 1e-12),
        ("补充因子态上类内重加权对 AUC 的 max 影响", "%.1e" % aux0, aux0 < 1e-12),
        ("CON 与 LON 流行率逐位相同", str(bool((W[("prevalence", "CON")] == W[("prevalence", "LON")]).all())),
         bool((W[("prevalence", "CON")] == W[("prevalence", "LON")]).all())),
        ("max \\|流行率(LMN) − 流行率(CON)\\|", "%.1e" % prevN, prevN < 1e-12),
        ("LMB 流行率全部恰为 0.5", str(bool((W[("prevalence", "LMB")] == 0.5).all())), bool((W[("prevalence", "LMB")] == 0.5).all())),
        ("n_eval：CON = LON，LMN = LMB", str(bool(((W[("n_eval", "CON")] == W[("n_eval", "LON")]) & (W[("n_eval", "LMN")] == W[("n_eval", "LMB")])).all())),
         bool(((W[("n_eval", "CON")] == W[("n_eval", "LON")]) & (W[("n_eval", "LMN")] == W[("n_eval", "LMB")])).all())),
    ]
    if dg is not None:
        chk += [
            ("CON 与泄漏训练集样本量、类别计数一致（全部 %d 折）" % len(dg), str(bool(dg.train_match.all() and (dg.n_pos_train_clean == dg.n_pos_train_leak).all())),
             bool(dg.train_match.all() and (dg.n_pos_train_clean == dg.n_pos_train_leak).all())),
            ("预处理拟合行 = O_tr（全部折）", str(bool(dg.pre_fit_equals_Otr.all())), bool(dg.pre_fit_equals_Otr.all())),
            ("k_test = n_neg_te − n_pos_te（全部折）", str(bool((dg.k_test == dg.n_neg_te - dg.n_pos_te).all())),
             bool((dg.k_test == dg.n_neg_te - dg.n_pos_te).all())),
            ("泄漏模型逐行预测与批次无关：max 差", "%.1e" % dg.batch_pred_maxdiff.max(), dg.batch_pred_maxdiff.max() < 1e-9),
            ("诊断文件记录的逐折 max \\|T − (L+S+P)\\|", "%.1e" % dg.telescope_maxabs.max(), dg.telescope_maxabs.max() < 1e-12),
        ]
    for name, val, ok in chk:
        A("| %s | %s | %s |" % (name, val, "✅" if ok else "❌"))
    A("")
    A("单元格层级（折均值）max \\|T − (L+S+P)\\|：%.1e（全部指标）。"
      % max(float(cells["%s_telescope" % m].abs().max()) for m in METRICS))
    A("")
    if dg is not None:
        A("**合成行计数与 ADASYN 修剪**（SMOTE / ROS 的 |G_leak_train| = |G_clean| 由算术保证；ADASYN 生成数只近似目标，")
        A("两侧取较小者、对多出一侧做固定种子均匀随机下采样）：")
        A("")
        A("| 重采样器 | 折数 | 无需修剪的折 | 修剪折占比 | 修剪行数占合成训练行（中位 / 最大） |")
        A("|---|---:|---:|---:|---|")
        for r in C.RESAMPLER_NAMES:
            g = dg[dg.resampler == r]
            if not len(g):
                continue
            trimmed = (g.trim_clean > 0) | (g.trim_leak > 0)
            trim = g.trim_clean + g.trim_leak                  # 每折只有多出的一侧被修剪
            frac = (trim / (g.k_matched + trim))[trimmed]
            A("| %s | %d | %d | %.1f%% | %s |" % (r, len(g), int((~trimmed).sum()), 100 * trimmed.mean(),
                                                  ("%.2f%% / %.2f%%" % (100 * frac.median(), 100 * frac.max())) if trimmed.any() else "—"))
        A("")
        A("**混合测试人群中合成行的占比**（SMOTE；= k_test / (n_te + k_test)，即 LMN/LMB 测试行里合成行的比例）：")
        A("")
        A("| 数据集 | O_te 行数（每折均值） | k_test（每折均值） | 合成行占混合人群 | 合成行占混合人群正类 |")
        A("|---|---:|---:|---:|---:|")
        g0 = dg[dg.resampler == "SMOTE"]
        for ds, g in g0.groupby("dataset"):
            A("| %s | %.0f | %.0f | %.1f%% | %.1f%% |" % (ds, g.n_te.mean(), g.k_test.mean(),
                                                         100 * (g.k_test / (g.n_te + g.k_test)).mean(),
                                                         100 * (g.k_test / (g.n_pos_te + g.k_test)).mean()))
        A("")

    # ================= 1. 主分解 =================
    A("## 1. 主分解（SMOTE；每格 = %d 个种子的均值 [均值的自助 95%% CI]，n.e. = 不确立）" % sm.seed.nunique())
    for metric, mlabel in [("f1", "F1"), ("auc", "ROC-AUC")]:
        A("")
        A("### %s" % mlabel)
        A("")
        A("| 数据集 | 模型 | CON | L 训练侧泄漏 | S 合成测试行混入 | P 流行率 | T 总虚高 |")
        A("|---|---|---:|---|---|---|---|")
        for (ds, model), g in sm.groupby(["dataset", "model"]):
            v = {k: verdict_mean(g["%s_%s" % (metric, k)].tolist()) for k in S.CONTRIB}
            A("| %s | %s | %.4f | %s | %s | %s | %s |" % (ds, model, g["%s_CON" % metric].mean(),
                                                          fmt(v["L"]), fmt(v["S"]), fmt(v["P"]), fmt(v["T"])))

    # ================= 2. 构成随模型 =================
    A("")
    A("## 2. 构成随模型（SMOTE；每模型 %d 数据集 × %d 种子 = %d 单元格的均值；份额 = 贡献 / T，带符号，三者之和恰为 100%%）"
      % (sm.dataset.nunique(), sm.seed.nunique(), len(sm) // max(1, sm.model.nunique())))
    A("")
    A("| 指标 | 模型 | L | S | P | T | L 份额 | S 份额 | P 份额 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for metric in ["f1", "auc", "auprc", "accuracy", "precision", "recall"]:
        for model in C.MODEL_NAMES:
            g = sm[sm.model == model]
            if not len(g):
                continue
            sh = shares(g, metric)
            A("| %s | %s | %+.4f | %+.4f | %+.4f | %+.4f | %s | %s | %s |"
              % (metric.upper(), model, g["%s_L" % metric].mean(), g["%s_S" % metric].mean(),
                 g["%s_P" % metric].mean(), g["%s_T" % metric].mean(),
                 *[("%.0f%%" % (100 * sh[k])) if np.isfinite(sh[k]) else "—" for k in "LSP"]))
    A("")
    A("逐模型判定（SMOTE，每模型 %d 单元格的均值 [CI]）：" % (len(sm) // max(1, sm.model.nunique())))
    A("")
    A("| 指标 | 模型 | L | S | P | T |")
    A("|---|---|---|---|---|---|")
    for metric in ["f1", "auc"]:
        for model in C.MODEL_NAMES:
            g = sm[sm.model == model]
            if not len(g):
                continue
            v = {k: verdict_mean(g["%s_%s" % (metric, k)].tolist()) for k in S.CONTRIB}
            A("| %s | %s | %s | %s | %s | %s |" % (metric.upper(), model, fmt(v["L"]), fmt(v["S"]), fmt(v["P"]), fmt(v["T"])))

    # ================= 3. 跨数据集一致性 =================
    A("")
    A("## 3. 跨数据集一致性（预注册 §7：≥5/6 个数据集各自确立且同向；每数据集合并 %d 模型 × %d 种子）"
      % (sm.model.nunique(), sm.seed.nunique()))
    A("")
    A("| 指标 | 贡献 | 确立数据集数 | 同向数 | 判定 |")
    A("|---|---|---:|---:|---|")
    ds_expected = sorted(dd for dd, _, _ in (C.PILOT["datasets"] if PILOT else C.DATASETS))
    if sorted(sm.dataset.unique()) != ds_expected:
        raise RuntimeError("真源中止：SMOTE 数据集集合 %s 与设计 %s 不符" % (sorted(sm.dataset.unique()), ds_expected))
    for metric in ["f1", "auc"]:
        for k in "LSP":
            vs = [verdict_mean(sm[sm.dataset == dd]["%s_%s" % (metric, k)].tolist()) for dd in ds_expected]
            c = consistency_across_datasets(vs, key="mean", n_total=len(ds_expected),
                                            min_agree=min(5, len(ds_expected)))
            A("| %s | %s | %d | %d | %s |" % (metric.upper(), CLAB[k], c["n_established"], c.get("n_same_sign", 0),
                                              "一致" if c["consistent"] else "不一致"))
    A("")
    A("逐 (数据集, 模型) 的确立计数（SMOTE；每格 %d 种子）：" % sm.seed.nunique())
    A("")
    A("| 指标 | 贡献 | 确立（正） | 确立（负） | 不确立 | LogReg 确立的数据集 |")
    A("|---|---|---:|---:|---:|---|")
    for metric in ["f1", "auc"]:
        for k in "LSP":
            pos = neg = ne = 0
            lr = []
            for (ds, model), g in sm.groupby(["dataset", "model"]):
                v = verdict_mean(g["%s_%s" % (metric, k)].tolist())
                if v["verdict"] == "确立（正）":
                    pos += 1
                elif v["verdict"] == "确立（负）":
                    neg += 1
                else:
                    ne += 1
                if model == "LogReg" and v["ci_excludes_floor"]:
                    lr.append(ds)
            A("| %s | %s | %d | %d | %d | %s |" % (metric.upper(), CLAB[k], pos, neg, ne, ", ".join(lr) or "—"))

    # ================= 4. 重采样器 =================
    A("")
    A("## 4. 重采样器对比（每重采样器合并全部 数据集 × 模型 × 种子；单元格数：%s）"
      % " / ".join("%s %d" % (r, (cells.resampler == r).sum()) for r in C.RESAMPLER_NAMES))
    A("")
    A("| 指标 | 重采样器 | L [CI] | S [CI] | P | T [CI] |")
    A("|---|---|---|---|---:|---|")
    for metric in ["f1", "auc"]:
        for r in C.RESAMPLER_NAMES:
            g = cells[cells.resampler == r]
            if not len(g):
                continue
            v = {k: verdict_mean(g["%s_%s" % (metric, k)].tolist()) for k in S.CONTRIB}
            A("| %s | %s | %s | %s | %+.4f | %s |" % (metric.upper(), r, fmt(v["L"], False), fmt(v["S"], False),
                                                      v["P"]["mean"], fmt(v["T"], False)))
    A("")
    A("> 随机过采样的\"合成\"行是原始少数类行的精确复制品：其 S 衡量的是**复制行**混入测试人群的贡献，")
    A("> 与 SMOTE/ADASYN 的插值行不同源，不与二者并列解释。")
    # 2026-09-26：正文 §4.3.2 对 Roth (2026) 的"机制细化"段引用两个派生比值；按"见报数字由生成器逐字打印"
    # 的纪律在此打印（比值取未圆整的均值之比，再按正文的一位小数显示）。
    mean_of = lambda r, col: cells.loc[cells.resampler == r, col].mean()
    l_ratio = mean_of("RandomOverSampler", "auc_L") / mean_of("SMOTE", "auc_L")
    s_ratio = mean_of("SMOTE", "auc_S") / mean_of("RandomOverSampler", "auc_S")
    A("")
    A("派生比值（ROC-AUC，正文 §4.3.2 引用）：L 随机过采样 / SMOTE = %.4f（正文显示 %.1f 倍）；"
      "S SMOTE / 随机过采样 = %.4f（正文显示 %.1f 倍）。" % (l_ratio, l_ratio, s_ratio, s_ratio))

    # ================= 5. 极端稳健性 =================
    ex, ex_files = load("extreme__*.csv")
    A("")
    A("## 5. 极端稳健性（creditcard 不截断，真实 1:578；4 模型 × SMOTE × 5 种子）")
    A("")
    if ex is None:
        A("*未运行。*")
    else:
        xc = cell_table(ex)
        pan = sm[sm.dataset == "creditcard"]
        A("样本：n=%d，正类比例 %.5f（1:%.0f），%d 个单元格"
          % (ex.n_rows.iloc[0], ex.prevalence_orig.iloc[0],
             (1 - ex.prevalence_orig.iloc[0]) / ex.prevalence_orig.iloc[0], len(xc)))
        A("")
        A("| 模型 | 指标 | 贡献 | 面板 1:40（30 种子） | 不截断 1:578（5 种子） | 判定 |")
        A("|---|---|---|---|---|---|")
        tally = {}
        for model in C.MODEL_NAMES:
            for metric in ["f1", "auc"]:
                for k in ("L", "S", "P") if metric == "f1" else ("L", "S"):
                    a = verdict_mean(pan[pan.model == model]["%s_%s" % (metric, k)].tolist())
                    b = verdict_mean(xc[xc.model == model]["%s_%s" % (metric, k)].tolist())
                    if not a["ci_excludes_floor"] and not b["ci_excludes_floor"]:
                        j = "一致（同判不确立）"
                    elif a["ci_excludes_floor"] != b["ci_excludes_floor"]:
                        j = "一方确立一方不确立"
                    else:
                        j = "同向确立" if np.sign(a["mean"]) == np.sign(b["mean"]) else "**确立且反号**"
                    tally[j] = tally.get(j, 0) + 1
                    A("| %s | %s | %s | %s | %s | %s |" % (model, metric.upper(), k, fmt(a), fmt(b), j))
        A("")
        A("**汇总**（%d 组比较）：%s" % (sum(tally.values()), " · ".join("%s %d 组" % kv for kv in sorted(tally.items()))))
        dx, _ = load("extreme__*.csv", DIAG)
        if dx is not None:
            A("")
            A("不截断数据每折：O_te %.0f 行（正类 %.0f），k_test %.0f 行 → 混合人群正类中合成行占 %.2f%%。"
              % (dx.n_te.mean(), dx.n_pos_te.mean(), dx.k_test.mean(),
                 100 * (dx.k_test / (dx.n_pos_te + dx.k_test)).mean()))
            A("")
            A(robust_invariants(dx, "极端网格"))

    # ================= 6. 受控阶梯 =================
    lad, lad_files = load("ladder-*.csv")
    A("")
    A("## 6. 受控阶梯（固定 400 少数类，只改多数类；SMOTE；10 种子）")
    A("")
    if lad is None:
        A("*未运行。*")
    else:
        lad = lad.copy()
        lad["dataset"] = lad.dataset + "@r" + lad.ratio.astype(int).astype(str)
        dl = cell_table(lad)
        dl["ratio"] = dl.dataset.str.split("@r").str[1].astype(int)
        dl["ds"] = dl.dataset.str.split("@r").str[0]
        ratios = sorted(dl.ratio.unique())
        A("档位标签 = 实际 少数:多数（构造参数 r：多数类 = 400×(r−1)）：%s" % " / ".join("1:%d" % (r - 1) for r in ratios))
        A("")
        A("### 6.1 内建零对照（1:1 平衡档：重采样为空操作，四个状态逐位相同）")
        A("")
        A("| 数据集 | max\\|L\\| | max\\|S\\| | max\\|P\\|（全部 6 指标） | 判定 |")
        A("|---|---:|---:|---:|---|")
        for ds, g in dl[dl.ratio == 2].groupby("ds"):
            mx = {k: max(float(g["%s_%s" % (m, k)].abs().max()) for m in METRICS) for k in "LSP"}
            A("| %s | %.2e | %.2e | %.2e | %s |" % (ds, mx["L"], mx["S"], mx["P"],
                                                   "✅ 逐位为零" if max(mx.values()) == 0 else "❌ 非零"))
        for k, klab in (("L", "L 训练侧泄漏"), ("S", "S 合成测试行混入")):
            A("")
            A("### 6.%d %s 随不平衡度（ROC-AUC，各档 10 种子均值）" % (2 if k == "L" else 3, klab))
            A("")
            A("| 数据集 | 模型 | " + " | ".join("1:%d" % (r - 1) for r in ratios) + " | Spearman ρ | 相邻档最小 Δ | CON AUC（全档均值） |")
            A("|---|---|" + "---:|" * len(ratios) + "---:|---:|---:|")
            for ds, g in dl.groupby("ds"):
                for model in C.MODEL_NAMES:
                    gm = g[g.model == model]
                    rr = sorted(gm.ratio.unique())
                    means = [gm[gm.ratio == r]["auc_%s" % k].mean() for r in rr]
                    cells_ = ["%+.4f" % means[rr.index(r)] if r in rr else "—" for r in ratios]
                    rho = spearmanr(rr, means).correlation if len(rr) > 2 else np.nan
                    dmin = float(np.min(np.diff(means))) if len(means) > 1 else np.nan
                    A("| %s | %s | %s | %+.3f | %+.1e | %.4f |" % (ds, model, " | ".join(cells_), rho, dmin,
                                                                   gm["auc_CON"].mean()))
        A("")
        dld, _ = load("ladder-*.csv", DIAG)
        if dld is not None:
            A("")
            A(robust_invariants(dld, "阶梯网格"))
        bal = dl[dl.ratio == 2].groupby("ds")["auc_CON"].mean()
        A("平衡档（1:1）CON AUC 均值：%s。" % "；".join("%s %.4f" % kv for kv in bal.items()))
        A("")
        A("> **未跑满的档位**：bank-marketing 多数类仅 39,922 行，1:255 档需 400×255 = 102,000 行多数类 → 该档判 insufficient。")

    # ================= 7. 目标期刊语料 =================
    A("")
    A("## 7. 目标期刊语料统计（论文立论依据）")
    A("")
    try:
        import corpus_metrics as CM
        cm = CM.compute()
        A("语料快照 `corpus/ijdns_corpus_dedup_2026-08-30.json`（SHA-256 `%s…`），Crossref + DOAJ 合并去重后 **%d 篇**，"
          "截至 **%s**；机器学习预测类 **%d 篇**。" % (cm["snapshot_sha256"][:16], cm["n_corpus"], cm["snapshot_date"], cm["n_ml"]))
        A("")
        A("| 指标 | 类别 | 篇数 | 占 ML 类 |")
        A("|---|---|---:|---:|")
        for m in cm["metrics"]:
            A("| %s | %s | %d | %.1f%% |" % (m["metric"], m["kind"], m["n"], m["pct"]))
        A("")
        A("- 至少一个对流行率敏感的指标（accuracy / F1 / precision）：**%d 篇（%.1f%%）**；只提 ROC-AUC：**%d 篇（%.1f%%）**"
          % (cm["n_any_sensitive"], cm["pct_any_sensitive"], cm["n_auc_only"], cm["pct_auc_only"]))
    except Exception as e:
        A("*语料统计不可用：%s*" % e)

    # ================= 8. 顺序依赖（补充因子态） =================
    A("")
    A("## 8. 顺序依赖（补充：2×2×2 因子态，6 种分解顺序；SMOTE，每模型 %d 单元格均值）" % (len(sm) // max(1, sm.model.nunique())))
    A("")
    A("主路径 = 预设顺序（训练 → 测试组成 → 流行率）。Shapley = 6 种顺序的平均；[min, max] = 6 种顺序的范围。")
    A("")
    A("| 指标 | 模型 | 因子 | 主路径 | Shapley | 6 种顺序范围 |")
    A("|---|---|---|---:|---:|---|")
    fl = {"t": "训练（L）", "m": "测试组成（S）", "p": "流行率（P）"}
    for metric in ["f1", "auc", "accuracy"]:
        for model in C.MODEL_NAMES:
            g = sm[sm.model == model]
            if not len(g):
                continue
            vals = {s: g["%s_%s" % (metric, s)].mean() for s in S.STATES}
            od = order_decomposition(vals)
            for f in FACTORS:
                o = od[f]
                A("| %s | %s | %s | %+.4f | %+.4f | [%+.4f, %+.4f] |" % (metric.upper(), model, fl[f], o["main"], o["shapley"], o["lo"], o["hi"]))

    # ================= 9. 判据敏感性 =================
    A("")
    A("## 9. 判据敏感性（SMOTE；逐 (数据集, 模型) × 贡献 L/S/P）")
    A("")
    for metric in ["f1", "auc"]:
        n_est = n_lost = n_disagree = n_tot = 0
        lost, dis = [], []
        for (ds, model), g in sm.groupby(["dataset", "model"]):
            for k in "LSP":
                vals = g["%s_%s" % (metric, k)].tolist()
                ci = bootstrap_mean_ci(vals)
                e3 = ci["lo"] > NOISE_FLOOR or ci["hi"] < -NOISE_FLOOR
                e6 = ci["lo"] > 2 * NOISE_FLOOR or ci["hi"] < -2 * NOISE_FLOOR
                med = verdict_median(vals)["ci_excludes_floor"]
                n_tot += 1
                if e3:
                    n_est += 1
                    if not e6:
                        n_lost += 1
                        lost.append("%s/%s/%s" % (ds, model, k))
                if med != e3:
                    n_disagree += 1
                    dis.append("%s/%s/%s" % (ds, model, k))
        A("- **%s**：均值判据确立 %d / %d 项；噪声带加倍到 ±%.3f 后失去确立 **%d** 项%s；"
          "改用中位数判据后判定改变 **%d** 项%s"
          % (metric.upper(), n_est, n_tot, 2 * NOISE_FLOOR, n_lost, ("（" + "、".join(lost) + "）") if lost else "",
             n_disagree, ("（" + "、".join(dis) + "）") if dis else ""))

    # ================= 10. 补充：已归档设计的两项对照 =================
    A("")
    A("## 10. 补充对照（取自已归档的五臂设计原始数据，`archive/five_arm_design_2026-09-01/`）")
    A("")
    old_raw, _ = load("main__*.csv", os.path.join(ARCHIVE, "raw"), validate=False)
    if old_raw is None or PILOT:
        A("*归档数据不可用或试点模式，跳过。*")
    else:
        oB = old_raw[old_raw.arm == "B"].groupby(KEYS)[["auc", "f1"]].mean().reset_index()
        mg = oB.merge(cells[KEYS + ["auc_CON", "f1_CON"]], on=KEYS)
        A("### 10.1 常规管线稳健性：旧 B（全量数据上拟合编码与标准化）vs CON（仅 O_tr 拟合）")
        A("")
        A("同一种子下两者的原始折分相同（StratifiedKFold 只依赖标签）；差别只在预处理的拟合范围。%d 个单元格配对：" % len(mg))
        A("")
        A("| 范围 | 指标 | 均值 (CON − 旧 B) | 中位 | max \\|差\\| |")
        A("|---|---|---:|---:|---:|")
        for scope, g in [("全部", mg)] + [(m, mg[mg.model == m]) for m in C.MODEL_NAMES]:
            for metric in ["auc", "f1"]:
                d = g["%s_CON" % metric] - g[metric]
                A("| %s | %s | %+.4f | %+.4f | %.4f |" % (scope, metric.upper(), d.mean(), d.median(), d.abs().max()))
        A("")
        od_path = os.path.join(ARCHIVE, "decomposed_cells.csv")
        if os.path.exists(od_path):
            od = pd.read_csv(od_path)
            osm = od[od.resampler == "SMOTE"]
            A("### 10.2 旧 D − C₂ 消融：合成少数类 vs 原始少数类的难易（等量、等流行率；旧设计的泄漏模型）")
            A("")
            A("| 模型 | F1 均值 [CI] | AUC 均值 [CI] |")
            A("|---|---|---|")
            for model in C.MODEL_NAMES:
                g = osm[osm.model == model]
                A("| %s | %s | %s |" % (model, fmt(verdict_mean(g["f1_m2_synthetic_rows"].tolist())),
                                        fmt(verdict_mean(g["auc_m2_synthetic_rows"].tolist()))))
            A("")
            A("### 10.3 旧成对对照设计的可加性缺口（取代理由；SMOTE，%d 个单元格）" % len(osm))
            A("")
            A("旧设计三项（① = E − B，② = D − C₂，③ = C − B）各配各的对照、互不处于同一路径；其和与观测总差 A − B 的缺口：")
            A("")
            A("| 指标 | 缺口中位 | \\|缺口\\| > 0.01 的单元格占比 | max \\|缺口\\| |")
            A("|---|---:|---:|---:|")
            for metric in ["f1", "auc"]:
                gap = osm["%s_additivity_gap" % metric]
                A("| %s | %+.4f | %.1f%% | %.4f |" % (metric.upper(), gap.median(), 100 * (gap.abs() > 0.01).mean(), gap.abs().max()))
            A("")
            A("对照：四状态设计的同一量在全部折上 ≤ 1e−12（§0）。")

    A("")
    A("---")
    A("")
    A("**数据来源**：%s %d 个状态 CSV · extreme %d 个 · ladder %d 个" % (MAIN_TAG, len(files), len(ex_files), len(lad_files)))
    open(OUT, "w").write("\n".join(L) + "\n")
    cells.to_csv(CELLS, index=False)
    print("✅ 真源已生成: %s" % OUT)
    print("   %s %d 个 CSV / %d 行 / %d 个单元格" % (MAIN_TAG, len(files), len(df), len(cells)))


if __name__ == "__main__":
    main()
