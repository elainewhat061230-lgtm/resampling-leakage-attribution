"""骨架验证 —— 只回答三个问题，不产出任何可见报的结果：

  1. D 臂与 E 臂是否真的可计算？（D 曾因单一类别而无定义，E 依赖加权指标）
  2. 合成行识别的两种判据（顺序 / 哈希）是否一致？随机过采样下是否如预期退化？
  3. E 臂的构造是否自洽？—— 内建零对照：把目标流行率设成【原始流行率】时，
     E 必须与 B 逐位相等；不等即构造有误。

⚠️ 本脚本产出的一切数字都是骨架验证用，不可见报。
"""
import json, sys, time, warnings, os
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import (run_five_arms, decompose, mark_original_rows, score,
                  _prevalence_weights, ARMS)

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
import pandas as pd

SEEDS = [11, 23]
t0 = time.time()


def load(did, pos=None, cap=8000, seed=7):
    d = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X = pd.get_dummies(d.data).astype(float).fillna(0).values
    t = d.target.astype(str).str.strip()
    y = (t == pos).astype(int).values if pos else (t == t.value_counts().idxmin()).astype(int).values
    if len(X) > cap:
        rng = np.random.default_rng(seed)
        s = rng.choice(len(X), cap, replace=False)
        X, y = X[s], y[s]
    return X, y


DATASETS = [
    ("Pima diabetes", lambda: load(37, "tested_positive")),
    ("credit default", lambda: load(42477)),
    ("bank-marketing", lambda: load(1461)),
]
MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=3000),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=0,
                                                   n_jobs=-1, min_samples_leaf=5),
}
RESAMPLERS = {
    "SMOTE": lambda: SMOTE(random_state=0),
    "RandomOverSampler": lambda: RandomOverSampler(random_state=0),
}

print("=" * 100)
print("检查 3 —— E 臂构造自洽性（内建零对照：目标流行率 = 原始流行率 时 E 必须 == B）")
print("=" * 100)
X, y = DATASETS[0][1]()
Xs = StandardScaler().fit_transform(X)
ok_all = True
for seed in SEEDS:
    diffs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, y):
        Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xs[tr], y[tr])
        m = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
        p = m.predict_proba(Xs[te])[:, 1]
        yhat = (p >= .5).astype(int)
        base = score(y[te], p, yhat)
        # 目标流行率设成该折的实际流行率 → 权重应退化为均匀
        w = _prevalence_weights(y[te], target=float(y[te].mean()))
        same = score(y[te], p, yhat, sample_weight=w)
        diffs.append(max(abs(base[k] - same[k]) for k in ("auc", "f1", "accuracy", "precision")))
    worst = max(diffs)
    ok = worst < 1e-9
    ok_all &= ok
    print("  seed=%-4d 最大偏差 %.3e   %s" % (seed, worst, "✅ 通过" if ok else "❌ 构造有误"))
if not ok_all:
    print("\n❌ E 臂零对照未通过，后续结果不可信。中止。")
    sys.exit(1)

print()
print("=" * 100)
print("检查 2 —— 合成行识别：顺序判据 vs 哈希判据")
print("=" * 100)
for dname, loader in DATASETS[:2]:
    X, y = loader()
    Xs = StandardScaler().fit_transform(X)
    for rname, mk in RESAMPLERS.items():
        Xr, yr = mk().fit_resample(Xs, y)
        _, rep = mark_original_rows(Xs, y, Xr, yr)
        print("  %-16s %-18s 保序=%-5s  原始行数 顺序=%-6d 哈希=%-6d  一致=%-5s  采用=%s"
              % (dname, rname, rep["order_assumption_holds"], rep["n_orig_by_order"],
                 rep["n_orig_by_hash"], rep["methods_agree"], rep["is_orig_source"]))

print()
print("=" * 100)
print("检查 1 —— D / E 臂可计算性 + 三机制分解（⚠️ 数字仅供验证骨架，不可见报）")
print("=" * 100)
rows = []
for dname, loader in DATASETS:
    X, y = loader()
    for mname, mk_m in MODELS.items():
        for seed in SEEDS:
            res, diag = run_five_arms(X, y, mk_m, lambda: SMOTE(random_state=seed),
                                      seed=seed, n_splits=5)
            empty = diag.get("empty_arms", [])
            dec_auc = decompose(res, "auc")
            dec_f1 = decompose(res, "f1")
            rows.append(dict(dataset=dname, model=mname, seed=seed,
                             empty_arms=",".join(empty) if empty else "",
                             **{("auc_" + k): v for k, v in dec_auc.items() if k != "metric"},
                             **{("f1_" + k): v for k, v in dec_f1.items() if k != "metric"}))
            print("\n  %s / %s / seed=%d   %s" % (dname, mname, seed,
                  ("⚠️ 空臂: " + ",".join(empty)) if empty else "五臂全部可计算 ✅"))
            print("    臂值 (AUC)  B=%.4f  A=%.4f  C=%.4f  D=%.4f  E=%.4f"
                  % tuple(dec_auc[k] for k in ("B_correct", "A_as_published", "C_true_leak",
                                               "D_synth_rows", "E_prevalence_only")))
            print("    机制 (AUC)  ①流行率=%+.4f  ②合成行=%+.4f  ③真泄漏=%+.4f  | A−B=%+.4f  可加性缺口=%+.4f%s"
                  % (dec_auc["m1_prevalence"], dec_auc["m2_synthetic_rows"], dec_auc["m3_true_leakage"],
                     dec_auc["total_observed_A_minus_B"], dec_auc["additivity_gap"],
                     "  [③低于噪声地板]" if dec_auc["below_noise_floor"] else ""))
            print("    机制 (F1 )  ①流行率=%+.4f  ②合成行=%+.4f  ③真泄漏=%+.4f  | A−B=%+.4f  可加性缺口=%+.4f"
                  % (dec_f1["m1_prevalence"], dec_f1["m2_synthetic_rows"], dec_f1["m3_true_leakage"],
                     dec_f1["total_observed_A_minus_B"], dec_f1["additivity_gap"]))
            print("    评估样本量  B=%d  A=%d  C=%d  D=%d"
                  % tuple((res[a] or {}).get("n_eval", -1) for a in ("B", "A", "C", "D")))

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "skeleton_check.csv")
df.to_csv(out, index=False)
print("\n" + "=" * 100)
print("汇总")
print("=" * 100)
print("  运行 %d 组；空臂出现 %d 组" % (len(df), (df.empty_arms != "").sum()))
for m in ("auc", "f1"):
    print("  %s  ①流行率 中位=%+.4f   ②合成行 中位=%+.4f   ③真泄漏 中位=%+.4f"
          % (m.upper().ljust(3), df["%s_m1_prevalence" % m].median(),
             df["%s_m2_synthetic_rows" % m].median(), df["%s_m3_true_leakage" % m].median()))
print("  可加性缺口 AUC 中位=%+.4f  F1 中位=%+.4f  ← 非零是预期内的，须如实报告"
      % (df.auc_additivity_gap.median(), df.f1_additivity_gap.median()))
print("\n  明细已写入 %s" % os.path.normpath(out))
print("  总 CPU %.0fs" % (time.time() - t0))
