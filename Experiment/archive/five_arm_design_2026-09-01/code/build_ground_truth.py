"""真源生成器 —— `RESULTS_GROUND_TRUTH.md` 的唯一来源。

KB 铁律：**所有见报数字从原始 CSV 现算，不从任何中间稿抄。**
论文、图注、摘要里的任何数字，都必须能在本文件的输出里查到。

用法：
    python3 build_ground_truth.py
"""
from __future__ import annotations
import os, sys, glob, json, hashlib, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from decision import (verdict, annotate_agreement, NOISE_FLOOR,
                      consistency_across_datasets, bootstrap_median_ci)
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, C.RAW_DIR))
OUT = os.path.abspath(os.path.join(HERE, "..", "RESULTS_GROUND_TRUTH.md"))
METRICS = ["auc", "auprc", "f1", "accuracy", "precision", "recall"]
MECH = [("m1_prevalence", "①流行率"), ("m2_synthetic_rows", "②合成测试行"), ("m3_true_leakage", "③真泄漏")]


def load_raw(pattern="main__*.csv", required=True):
    files = sorted(glob.glob(os.path.join(RAW, pattern)))
    if not files:
        if required:
            raise SystemExit("raw/ 下没有 %s" % pattern)
        return None, []
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df, files


def agreement(a, b):
    """带幅度守卫的一致性判定 —— 两边都不确立即视为一致（同为零效应）。
    ⚠️ 不得直接比 sign(median)：噪声带内的正负号无意义，会造出假分歧。"""
    ae, be = a["ci_excludes_floor"], b["ci_excludes_floor"]
    if not ae and not be:
        return "一致（同判不确立）"
    if ae != be:
        return "一方确立一方不确立"
    return "同向确立" if np.sign(a["median"]) == np.sign(b["median"]) else "**确立且反号**"


def cell_means(df):
    """折 → 单元格：每个 (数据集,模型,重采样器,种子,臂) 取五折均值。"""
    keys = ["dataset", "model", "resampler", "seed", "arm"]
    return df.groupby(keys, as_index=False)[METRICS + ["n_eval", "prevalence"]].mean()


def decompose_cells(cm):
    """按预注册 §5 算三机制。返回每个 (数据集,模型,重采样器,种子) 一行。

    双 C（DEVIATIONS D-004）：② 用 C2（与 D 等量等流行率的 C 下采样版）作对照；
    旧 CSV 无 C2 臂时退回 C —— 凡合成 ≥ 原始的折 C2 ≡ C，退回严格等价。
    """
    idx = ["dataset", "model", "resampler", "seed"]
    wide = cm.pivot_table(index=idx, columns="arm", values=METRICS)
    rows = []
    for key, r in wide.iterrows():
        rec = dict(zip(idx, key))
        for m in METRICS:
            g = lambda a: r.get((m, a), np.nan)
            B, A, Cc, D, E = (g(a) for a in ("B", "A", "C", "D", "E"))
            C2 = g("C2")
            if not np.isfinite(C2):
                C2 = Cc
            rec["%s_B" % m], rec["%s_A" % m] = B, A
            rec["%s_m3_true_leakage" % m] = Cc - B
            rec["%s_m2_synthetic_rows" % m] = D - C2
            rec["%s_m1_prevalence" % m] = E - B
            rec["%s_total" % m] = A - B
            rec["%s_additivity_gap" % m] = (A - B) - ((E - B) + (D - C2) + (Cc - B))
        rows.append(rec)
    return pd.DataFrame(rows)


def fmt_v(v):
    star = "" if v["ci_excludes_floor"] else "  (不确立)"
    vote = "%d/%d" % (v["vote_agree"], v["vote_n"]) if v["vote_applicable"] else "n/a"
    # ⚠️ 单元格内不得出现 "|"，否则会劈开 markdown 表格列
    return "%+.4f [%+.4f, %+.4f]%s · 票%s" % (v["median"], v["lo"], v["hi"], star, vote)


def main():
    df, files = load_raw("main__*.csv")
    cm = cell_means(df)
    dec = decompose_cells(cm)

    L = []
    A = L.append
    A("# 结果真源（RESULTS_GROUND_TRUTH）")
    A("")
    A("> **本文件由 `code/build_ground_truth.py` 生成，请勿手工编辑。**")
    A("> 论文、图注、摘要里的任何数字都必须能在此查到。改完数据重跑本脚本即可。")
    A("")
    A("- 生成时刻：%s" % datetime.datetime.now().isoformat(timespec="seconds"))
    A("- 预注册 SHA-256：`%s`" % C.PREREG_SHA256)
    A("- 原始 CSV：**%d 个文件**，%d 行折级记录" % (len(files), len(df)))
    A("- 判据：自助法 95%% CI（B=10,000）须完全落在噪声带 ±%.3f 之外" % NOISE_FLOOR)
    A("")

    A("## 0. 数据完整性")
    A("")
    A("| 数据集 | 行数 | 原始正类比例 | 模型×重采样器×种子 单元格数 |")
    A("|---|---:|---:|---:|")
    for ds in sorted(df.dataset.unique()):
        d = df[df.dataset == ds]
        A("| %s | %d | %.4f | %d |" % (ds, d.n_rows.iloc[0], d.prevalence_orig.iloc[0],
                                        d.groupby(["model", "resampler", "seed"]).ngroups))
    A("")
    pair = df[df.arm.isin(["C", "C2", "D"])].groupby(
        ["dataset", "model", "resampler", "seed", "arm"])[["n_eval", "prevalence"]].mean()
    ne, pv = pair["n_eval"].unstack(), pair["prevalence"].unstack()
    if "D" in ne.columns and "C" in ne.columns:
        ref_n = ne["C2"].fillna(ne["C"]) if "C2" in ne.columns else ne["C"]
        ref_p = pv["C2"].fillna(pv["C"]) if "C2" in pv.columns else pv["C"]
        dn, dp = (ref_n - ne["D"]).abs(), (ref_p - pv["D"]).abs()
        A("**② 对照对（C2 vs D）等量等流行率核验**（预注册 §5 硬约束；双 C 修正见 DEVIATIONS D-004）：")
        A("n_eval 差 中位 %.1f · 最大 %.1f；prevalence 差 中位 %.4f · 最大 %.4f"
          % (dn.median(), dn.max(), dp.median(), dp.max()))
        A("（旧 CSV 无 C2 臂时以 C 代入 —— 凡合成行 ≥ 原始行的折 C2 ≡ C，代入严格等价）")
        A("")
        # C2 触发统计（折级：C2 的 n_eval < C 的 n_eval 即触发了下采样）
        fl = df[df.arm.isin(["C", "C2"])].pivot_table(
            index=["dataset", "model", "resampler", "seed", "fold"], columns="arm", values="n_eval")
        if "C2" in fl.columns:
            trig_lines = []
            for ds, g in fl.dropna(subset=["C2"]).groupby(level="dataset"):
                t, n = int((g["C2"] < g["C"]).sum()), len(g)
                trig_lines.append("%s %d/%d（%.0f%%）" % (ds, t, n, 100 * t / n))
            A("**C2 下采样触发折**（仅在折内合成少数行 < 原始少数行时触发）：%s；其余数据集无 C2 臂记录（回退 C，数学等价）"
              % " · ".join(trig_lines))
            A("")
        # AUC 上机制① 的解析零：全网格浮点残差上界（正文 §4.1 的"machine precision"依据）
        m1max = dec["auc_m1_prevalence"].abs().max()
        A("**AUC 机制① 解析零核验**：全部 %d 个单元格 max |①_AUC| = %.2e（浮点残差级）" % (len(dec), m1max))
        A("")
    if "order_ok" in df.columns:
        A("**imblearn 保序断言**：通过 %d / %d 条折级记录" % (df.order_ok.sum(), len(df)))
        A("")
    if "mark_source" in df.columns:
        # 原始行识别用哪条判据：数据自带重复行时哈希退化（分不清原件与复制品），
        # 回退到 imblearn 文档化的保序行为。此时保序断言是唯一护栏，故须报出承担比例。
        cell = df.drop_duplicates(["dataset", "model", "resampler", "seed", "tag"])
        cell = cell.assign(crit=np.where(cell.mark_source.astype(str).str.startswith("hash"),
                                         "hash", "order"))
        n_h = int((cell.crit == "hash").sum()); n_o = int((cell.crit == "order").sum())
        A("**原始行识别判据**：哈希 %d 格 / 顺序回退 %d 格（共 %d）。"
          "回退发生在数据自带重复行使哈希退化时（ROS 必然，SMOTE 见下），此时保序断言是唯一护栏。"
          % (n_h, n_o, n_h + n_o))
        sm_cell = cell[cell.resampler == "SMOTE"]
        parts = []
        for ds in sorted(sm_cell.dataset.unique()):
            g = sm_cell[sm_cell.dataset == ds]
            parts.append("%s %d/%d" % (ds, int((g.crit == "order").sum()), len(g)))
        A("- 仅 SMOTE：顺序回退 %d / %d 格 —— %s"
          % (int((sm_cell.crit == "order").sum()), len(sm_cell), " · ".join(parts)))
        A("")

    A("## 1. 三机制分解（主结果）")
    A("")
    n_sm = int((dec.resampler == "SMOTE").sum())
    A("本节与 §2/§3/§9 均为 **SMOTE 口径（%d 个 数据集×模型×种子 单元格）**；三种重采样器的对比见 §4。" % n_sm)
    for metric, mlabel in [("f1", "F1（应用文献主报）"), ("auc", "ROC-AUC（竞品所用）")]:
        A("")
        A("### %s" % mlabel)
        A("")
        A("| 数据集 | 模型 | ①流行率 | ②合成测试行 | ③真泄漏 | 观测总差 A−B | 可加性缺口 |")
        A("|---|---|---|---|---|---:|---:|")
        for (ds, model), g in dec[dec.resampler == "SMOTE"].groupby(["dataset", "model"]):
            cells = []
            for key, _ in MECH:
                v = annotate_agreement(verdict(g["%s_%s" % (metric, key)].tolist()))
                cells.append(fmt_v(v))
            A("| %s | %s | %s | %s | %s | %+.4f | %+.4f |"
              % (ds, model, cells[0], cells[1], cells[2],
                 g["%s_total" % metric].median(), g["%s_additivity_gap" % metric].median()))

    A("")
    A("## 2. 机制构成占比（仅 SMOTE，各机制中位数占三者之和）")
    A("")
    A("| 指标 | 模型 | ①流行率 | ②合成测试行 | ③真泄漏 |")
    A("|---|---|---:|---:|---:|")
    for metric in ["f1", "auc"]:
        for model in C.MODEL_NAMES:
            g = dec[(dec.resampler == "SMOTE") & (dec.model == model)]
            if not len(g):
                continue
            meds = [g["%s_%s" % (metric, k)].median() for k, _ in MECH]
            tot = sum(abs(x) for x in meds)
            share = [(abs(x) / tot * 100 if tot else np.nan) for x in meds]
            A("| %s | %s | %.3f (%.0f%%) | %.3f (%.0f%%) | %.3f (%.0f%%) |"
              % (metric.upper(), model, meds[0], share[0], meds[1], share[1], meds[2], share[2]))

    A("")
    A("## 3. 跨数据集一致性（预注册 §7：≥5/6 各自确立且同向）")
    A("")
    A("| 指标 | 机制 | 确立数据集数 | 同向数 | 判定 |")
    A("|---|---|---:|---:|---|")
    for metric in ["f1", "auc"]:
        for key, label in MECH:
            vs = []
            for ds, g in dec[dec.resampler == "SMOTE"].groupby("dataset"):
                vs.append(verdict(g["%s_%s" % (metric, key)].tolist()))
            c = consistency_across_datasets(vs)
            A("| %s | %s | %d | %d | %s |" % (metric.upper(), label, c["n_established"],
                                              c.get("n_same_sign", 0),
                                              "一致" if c["consistent"] else "不一致"))

    A("")
    A("## 4. 重采样器稳健性（SMOTE vs ROS vs ADASYN，③真泄漏中位）")
    A("")
    A("| 指标 | 重采样器 | 中位 ③ [自助法 95% CI] | 中位总差 A−B | 备注 |")
    A("|---|---|---:|---:|---|")
    for metric in ["f1", "auc"]:
        for r in C.RESAMPLER_NAMES:
            g = dec[dec.resampler == r]
            if not len(g):
                continue
            note = "⚠️ D 臂概念退化（合成行是精确复制品）" if r == "RandomOverSampler" else ""
            ci = bootstrap_median_ci(g["%s_m3_true_leakage" % metric].tolist())
            A("| %s | %s | %+.4f [%+.4f, %+.4f] | %+.4f | %s |"
              % (metric.upper(), r, ci["median"], ci["lo"], ci["hi"],
                 g["%s_total" % metric].median(), note))
    A("")
    tot_disp = {m: [dec[dec.resampler == r]["%s_total" % m].median() for r in C.RESAMPLER_NAMES]
                for m in ["f1", "auc"]}
    A("> 三种重采样器的**总差 A−B 描述性上几乎无法区分**（与 Roth 2026 Exp BA 的报告一致）：")
    A("> 正文引用的圆整显示值 —— F1 %s；AUC %s。" % (
        " / ".join("%.3f" % v for v in tot_disp["f1"]),
        " / ".join("%.3f" % v for v in tot_disp["auc"])))
    A("> 但隔离出的 ③ 相差约 2 倍 —— 总量层面看不见的差别，拆开才可见。")

    # ================= 5. 极端端稳健性检查 =================
    ex, ex_files = load_raw("extreme__*.csv", required=False)
    A("")
    A("## 5. 极端端稳健性检查（creditcard 不截断，真实 1:578）")
    A("")
    if ex is None:
        A("*未运行。*")
    else:
        dex = decompose_cells(cell_means(ex))
        pan = dec[(dec.dataset == "creditcard") & (dec.resampler == "SMOTE")]
        A("样本：n=%d，正类比例 %.5f（1:%.0f），%d 个单元格（%d 模型 × SMOTE × %d 种子）"
          % (ex.n_rows.iloc[0], ex.prevalence_orig.iloc[0],
             (1 - ex.prevalence_orig.iloc[0]) / ex.prevalence_orig.iloc[0],
             dex.groupby(["model", "seed"]).ngroups,
             dex.model.nunique(), dex.seed.nunique()))
        A("")
        A("| 模型 | 指标 | 机制 | 面板 1:40（30 种子） | 极端端 1:578（5 种子） | 判定 |")
        A("|---|---|---|---|---|---|")
        tal = {}
        for model in C.MODEL_NAMES:
            for metric in ["f1", "auc"]:
                for key, label in MECH[1:]:          # ① 在此对比中无信息（面板与极端端同为解析零/流行率项）
                    a = verdict(pan[pan.model == model]["%s_%s" % (metric, key)].tolist())
                    b = verdict(dex[dex.model == model]["%s_%s" % (metric, key)].tolist())
                    j = agreement(a, b)
                    tal[j] = tal.get(j, 0) + 1
                    A("| %s | %s | %s | %s | %s | %s |"
                      % (model, metric.upper(), label, fmt_v(a), fmt_v(b), j))
        A("")
        A("**汇总**：%s" % " · ".join("%s %d 组" % (k, v) for k, v in sorted(tal.items())))
        A("")
        cd = ex[ex.arm.isin(["C", "D"])].groupby("arm").n_eval.mean()
        n_min = ex[ex.arm == "C"].prevalence.mean() * cd.get("C", np.nan)
        A("> ⚠️ **本设计的功率限制（须写进论文限制节）**：D 臂要与 C 等量等流行率，")
        A("> 故能换入的合成行数被**原始少数类数量**卡死。极端端 C 评估 %.0f 行，其中原始少数类仅约 %.0f 行"
          % (cd.get("C", np.nan), n_min))
        A("> （占 %.2f%%）——C 与 D 只差这些行，指标几乎不动。**②在极端不平衡数据上的零值应读作\"测不出\"而非\"不存在\"。**"
          % (100 * n_min / cd.get("C", 1)))

    # ================= 6. 受控阶梯 =================
    lad, lad_files = load_raw("ladder-*.csv", required=False)
    A("")
    A("## 6. 受控阶梯（RQ3：单变量控制，固定少数类只改多数类）")
    A("")
    if lad is None:
        A("*未运行。*")
    else:
        lad = lad.copy()
        lad["ratio"] = lad["ratio"].astype(int)
        dl = decompose_cells(cell_means(lad.assign(dataset=lad.dataset + "@r" + lad.ratio.astype(str))))
        dl["ratio"] = dl.dataset.str.split("@r").str[1].astype(int)
        dl["ds"] = dl.dataset.str.split("@r").str[0]
        A("样本：%d 行折级记录，数据集 %s，档位（少数:多数）%s"
          % (len(lad), ", ".join(sorted(lad.dataset.unique())),
             " / ".join("1:%d" % (r - 1) for r in sorted(lad.ratio.unique()))))
        A("")
        A("> 档位标签为**实际 少数:多数**（与 Table 2 口径一致）。构造参数 r（文件名 `ladder-r{r}`、")
        A("> CSV `ratio` 列）的含义是 多数类 = 400×(r−1)，即 r = 总量/少数类；标签换算见 DEVIATIONS D-007。")
        A("")
        A("### 6.1 内建零对照（1:1 平衡档，构造 400:400，SMOTE 为空操作，③ 必须恒为 0）")
        A("")
        A("| 数据集 | 指标 | max\\|③\\| | 判定 |")
        A("|---|---|---:|---|")
        for ds, g in dl[dl.ratio == 2].groupby("ds"):
            for metric in ["f1", "auc"]:
                mx = g["%s_m3_true_leakage" % metric].abs().max()
                A("| %s | %s | %.2e | %s |" % (ds, metric.upper(), mx,
                                               "✅ 通过" if mx < 1e-9 else "❌ 非零，管道有问题"))
        A("")
        A("### 6.2 ③真泄漏随不平衡度（ROC-AUC，各档中位）")
        A("")
        ratios = sorted(dl.ratio.unique())
        A("| 数据集 | 模型 | " + " | ".join("1:%d" % (r - 1) for r in ratios) + " | Spearman ρ | B 臂 AUC 中位 |")
        A("|---|---|" + "---:|" * len(ratios) + "---:|---:|")
        for ds, g in dl.groupby("ds"):
            for model in C.MODEL_NAMES:
                gm = g[g.model == model]
                rr = sorted(gm.ratio.unique())
                meds = [gm[gm.ratio == r]["auc_m3_true_leakage"].median() for r in rr]
                cells = []
                for r in ratios:
                    cells.append("%+.4f" % meds[rr.index(r)] if r in rr else "—")
                rho = spearmanr(rr, meds).correlation if len(rr) > 2 else np.nan
                baseB = gm["auc_B"].median()
                A("| %s | %s | %s | %+.3f | %.4f |" % (ds, model, " | ".join(cells), rho, baseB))
        A("")
        A("> **未跑满的档位**：bank-marketing 多数类仅 39,922 行，1:255 档构造需 400×255 = 102,000 行多数类")
        A("> → 该档 40 个单元格判 insufficient。这是数据本身的上限，非失败。")

    # ================= 7. 目标期刊语料统计 =================
    A("")
    A("## 7. 目标期刊语料统计（论文 §4.7 的立论依据）")
    A("")
    try:
        import corpus_metrics as CM
        cm = CM.compute()
        A("语料快照 `corpus/ijdns_corpus_dedup_2026-08-30.json`（SHA-256 `%s…`），"
          % cm["snapshot_sha256"][:16])
        A("Crossref + DOAJ 合并去重后 **%d 篇**，截至 **%s**。其中按互斥优先级规则判为"
          % (cm["n_corpus"], cm["snapshot_date"]))
        A("机器学习预测类的 **%d 篇**。" % cm["n_ml"])
        A("")
        A("| 指标 | 类别 | 篇数 | 占 ML 类 |")
        A("|---|---|---:|---:|")
        for m in cm["metrics"]:
            A("| %s | %s | %d | %.1f%% |" % (m["metric"], m["kind"], m["n"], m["pct"]))
        A("")
        A("- 提到**至少一个对流行率敏感**的指标（accuracy / F1 / precision）：**%d 篇（%.1f%%）**"
          % (cm["n_any_sensitive"], cm["pct_any_sensitive"]))
        A("- 只提 ROC-AUC 而不提上述三者：**%d 篇（%.1f%%）**"
          % (cm["n_auc_only"], cm["pct_auc_only"]))
        A("")
        A("> ⚠️ 快照固化于仓库，**不联网重拉** —— 期刊持续出版，重拉会得到不同数字。")
        A("> 引用时须注明\"截至 %s\"。分类规则见 `code/corpus_metrics.py`。" % cm["snapshot_date"])
    except Exception as e:
        A("*语料统计不可用：%s*" % e)

    # ================= 8. Scaler 敏感性核查（D-003）=================
    sens_path = os.path.join(HERE, "..", "scaler_sensitivity.csv")
    A("")
    A("## 8. Scaler 敏感性核查（DEVIATIONS D-003：全量拟合 vs 折内拟合）")
    A("")
    if not os.path.exists(sens_path):
        A("*未运行（`code/scaler_sensitivity.py`）。*")
    else:
        s = pd.read_csv(sens_path)
        A("本研究六臂共用一次全量拟合的 StandardScaler（臂间差分不受其影响；决定理由与代价见 DEVIATIONS D-003）。")
        A("敏感性核查：仅重跑受缩放影响的 LogReg 的 B 臂（树模型对逐特征仿射变换不变），")
        A("对比【全量拟合】与【折内按训练折拟合】两种 scaler 方案，共 %d 个配置（数据集 × 重采样器 × 种子）。" % len(s))
        A("")
        A("| 量（LogReg，B 臂） | max | 中位 |")
        A("|---|---:|---:|")
        A("| 两方案 AUC 差的绝对值 | %.4f | %.4f |" % (s.d_auc.abs().max(), s.d_auc.abs().median()))
        A("| 两方案 F1 差的绝对值 | %.4f | %.4f |" % (s.d_f1.abs().max(), s.d_f1.abs().median()))
        A("")
        A("**复制核验**：敏感性脚本的全量拟合变体与主网格 raw CSV 的 B 臂逐配置比对，")
        A("绝对差 max AUC %.2e · F1 %.2e（浮点级零 = 脚本忠实复刻了主管线）。"
          % (s.chk_auc.abs().max(), s.chk_f1.abs().max()))

    # ================= 9. 判据敏感性（噪声带加倍）=================
    A("")
    A("## 9. 判据敏感性：噪声带 ±0.003 加倍到 ±0.006（限制节依据）")
    A("")
    A("±0.003 是 Roth (2026) 实测的 ΔAUC 种子噪声地板，预注册将其用于全部指标；F1 自身的地板未单独测。")
    A("敏感性：把带宽加倍后，逐 (数据集, 模型) × 机制 的确立判定变化如下（仅 SMOTE）：")
    A("")
    for metric in ["f1", "auc"]:
        n_est = n_lost = 0
        lost = []
        for (ds, model), g in dec[dec.resampler == "SMOTE"].groupby(["dataset", "model"]):
            for key, label in MECH:
                ci = bootstrap_median_ci(g["%s_%s" % (metric, key)].tolist())
                est3 = ci["lo"] > NOISE_FLOOR or ci["hi"] < -NOISE_FLOOR
                est6 = ci["lo"] > 2 * NOISE_FLOOR or ci["hi"] < -2 * NOISE_FLOOR
                if est3:
                    n_est += 1
                    if not est6:
                        n_lost += 1
                        lost.append("%s/%s/%s" % (ds, model, label))
        A("- **%s**：±0.003 下确立 %d 项，加倍后失去确立 **%d** 项%s"
          % (metric.upper(), n_est, n_lost, ("（" + "、".join(lost) + "）") if lost else ""))

    A("")
    A("---")
    A("")
    A("**数据来源**：main %d 个 CSV · extreme %d 个 · ladder %d 个"
      % (len(files), len(ex_files), len(lad_files)))
    open(OUT, "w").write("\n".join(L) + "\n")
    dec.to_csv(os.path.join(RAW, "..", "decomposed_cells.csv"), index=False)
    print("✅ 真源已生成: %s" % OUT)
    print("   main %d 个 CSV / %d 行 / %d 个分解单元格" % (len(files), len(df), len(dec)))


if __name__ == "__main__":
    main()
