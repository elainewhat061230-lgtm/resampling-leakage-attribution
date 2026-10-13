"""验证判据模块 —— 三项检查：
  1. 已知答案的合成数据上，判据给出正确结论（零效应判不确立、大效应判确立）
  2. 真实数据上跑 10 个种子，看两个判据（置信区间 / 24-of-30 投票）是否给同一结论
  3. 关键场景：LogReg 的真泄漏 ≈ +0.004（贴着噪声地板 0.003），两个判据在这里最可能分歧
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision import verdict, annotate_agreement, NOISE_FLOOR
from arms import run_five_arms, decompose

print("=" * 96)
print("检查 1 —— 已知答案的合成数据")
print("=" * 96)
rng = np.random.default_rng(1)
CASES = [
    ("真零效应（噪声 SD=0.004）",        rng.normal(0.000, 0.004, 30), "应判不确立"),
    ("贴地板小效应 +0.004",              rng.normal(0.004, 0.004, 30), "边界情形，看两判据是否分歧"),
    ("中等效应 +0.020",                  rng.normal(0.020, 0.006, 30), "应判确立（正）"),
    ("大效应 +0.090",                    rng.normal(0.090, 0.010, 30), "应判确立（正）"),
    ("负效应 -0.015",                    rng.normal(-0.015, 0.005, 30), "应判确立（负）"),
    ("高方差零效应（SD=0.05）",          rng.normal(0.000, 0.050, 30), "应判不确立"),
]
for name, v, expect in CASES:
    d = annotate_agreement(verdict(v))
    print("  %-26s 中位=%+.4f  CI=[%+.4f, %+.4f]  → %-10s | 投票 %2d/%d %s"
          % (name, d["median"], d["lo"], d["hi"], d["verdict"],
             d["vote_agree"], d["vote_n"],
             ("(旧门槛也过)" if d["legacy_24of30"] else "(旧门槛不过)")))
    print("      预期：%s   两判据一致：%s" % (expect, "是" if d["agrees_with_legacy"] else "**否**"))

print()
print("=" * 96)
print("检查 2/3 —— 真实数据，10 个种子，重点看 LogReg 的真泄漏（贴地板）")
print("=" * 96)
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

d = fetch_openml(data_id=42477, as_frame=True, parser="auto")
X = pd.get_dummies(d.data).astype(float).fillna(0).values
t = d.target.astype(str).str.strip()
y = (t == t.value_counts().idxmin()).astype(int).values
rng2 = np.random.default_rng(7); sel = rng2.choice(len(X), 8000, replace=False)
X, y = X[sel], y[sel]
print("  credit default n=%d 正类=%.3f\n" % (len(y), y.mean()))

MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=3000),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, min_samples_leaf=5),
}
t0 = time.time()
for mname, mk in MODELS.items():
    acc = {"m1_prevalence": [], "m2_synthetic_rows": [], "m3_true_leakage": []}
    for s in range(10):
        res, _ = run_five_arms(X, y, mk, lambda: SMOTE(random_state=s), seed=s, n_splits=5)
        dec = decompose(res, "auc")
        for k in acc:
            acc[k].append(dec[k])
    print("  --- %s (AUC) ---" % mname)
    for k, label in [("m1_prevalence", "①流行率"), ("m2_synthetic_rows", "②合成行"), ("m3_true_leakage", "③真泄漏")]:
        v = annotate_agreement(verdict(acc[k]))
        flag = "" if v["agrees_with_legacy"] else "   ← **两判据分歧**"
        print("    %-8s 中位=%+.4f  CI=[%+.4f, %+.4f]  %-10s | 投票 %2d/%d  旧门槛:%s%s"
              % (label, v["median"], v["lo"], v["hi"], v["verdict"],
                 v["vote_agree"], v["vote_n"], "过" if v["legacy_24of30"] else "不过", flag))
print("\n  噪声地板 = ±%.3f   CPU %.0fs" % (NOISE_FLOOR, time.time() - t0))
