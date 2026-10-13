"""独立重实现核验：不调用 states.py，按规格从头实现四状态，与主网格落盘结果逐状态比对。

独立之处：预处理用 sklearn 的 ColumnTransformer + OneHotEncoder（而非 states.FoldPreprocessor）；
合成行、测试人群、权重、指标全部在本文件内重写。与主网格共享的只有：数据定义（OpenML id、正类、
截断规则）、模型与重采样器配置、以及规格本身规定的随机种子方案（G_full 的置换种子）。

用法：python3 verify_independent.py   → 打印逐单元格比对，并写入 ../VERIFY_INDEPENDENT.md
只选无类别缺失的数据集（OneHotEncoder 会把缺失当作一个类别，与规格"缺失 → 全零"不同）。
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("pima", 37, "tested_positive", "LogReg", "SMOTE", 3),
         ("credit-g", 31, "bad", "RandomForest", "SMOTE", 7),
         ("credit-g", 31, "bad", "XGBoost", "RandomOverSampler", 11),
         ("bank-marketing", 1461, None, "DecisionTree", "SMOTE", 5)]


def load(did, pos):
    d = fetch_openml(data_id=did, as_frame=True, parser="auto")
    t = d.target.astype(str).str.strip()
    y = (t == pos).astype(int).values if pos else (t == t.value_counts().idxmin()).astype(int).values
    X = d.data.copy()
    sel, _ = C.cap_indices(y)
    return X.iloc[sel].reset_index(drop=True), y[sel]


def metrics(y, p, w=None):
    yh = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(y, p, sample_weight=w), auprc=average_precision_score(y, p, sample_weight=w),
                f1=f1_score(y, yh, sample_weight=w), accuracy=accuracy_score(y, yh, sample_weight=w),
                precision=precision_score(y, yh, sample_weight=w, zero_division=0))


def run_case(ds, did, pos, model, res, seed):
    X, y = load(did, pos)
    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object]
    num = [c for c in X.columns if c not in cat]
    out = {s: [] for s in ("CON", "LON", "LMN", "LMB")}
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y)):
        Xtr = X.iloc[tr]
        cats = [sorted(Xtr[c].dropna().astype(str).unique().tolist()) for c in cat]
        ct = ColumnTransformer([("num", FunctionTransformer(lambda a: np.asarray(a, float)), num)] +
                               ([("cat", Pipeline([("str", FunctionTransformer(lambda a: np.asarray(a).astype(str))),
                                                   ("oh", OneHotEncoder(categories=cats, handle_unknown="ignore",
                                                                        sparse_output=False))]), cat)] if cat else []))
        pre = Pipeline([("ct", ct), ("sc", StandardScaler())]).fit(Xtr)
        Ztr, Zte, Zall = pre.transform(Xtr), pre.transform(X.iloc[te]), pre.transform(X)
        ytr, yte = y[tr], y[te]
        Rc = C.make_resampler(res, seed, ytr).fit_resample(Ztr, ytr)
        Gc = Rc[0][len(ytr):]
        Rf = C.make_resampler(res, seed, y).fit_resample(Zall, y)
        Gf = Rf[0][len(y):]
        k_test = int((yte == 0).sum() - (yte == 1).sum())
        perm = np.random.default_rng([seed, fold, 20260925]).permutation(len(Gf))
        Gt, Gl = Gf[perm[:k_test]], Gf[perm[k_test:]]
        assert len(Gc) == len(Gl), "independent: count mismatch (not expected for SMOTE/ROS)"
        mc = C.make_model(model, seed).fit(np.vstack([Ztr, Gc]), np.r_[ytr, np.ones(len(Gc), int)])
        ml = C.make_model(model, seed).fit(np.vstack([Ztr, Gl]), np.r_[ytr, np.ones(len(Gl), int)])
        pc, pl = mc.predict_proba(Zte)[:, 1].astype(float), ml.predict_proba(Zte)[:, 1].astype(float)
        pmix = np.r_[pl, ml.predict_proba(Gt)[:, 1].astype(float)]
        ymix = np.r_[yte, np.ones(k_test, int)]
        pi = yte.mean()
        w = np.where(ymix == 1, pi / (1 - pi) * (ymix == 0).sum() / (ymix == 1).sum(), 1.0)
        out["CON"].append(metrics(yte, pc)); out["LON"].append(metrics(yte, pl))
        out["LMN"].append(metrics(ymix, pmix, w)); out["LMB"].append(metrics(ymix, pmix))
    return {s: pd.DataFrame(v).mean() for s, v in out.items()}


def main():
    L = ["# 独立重实现核验（VERIFY_INDEPENDENT）", "",
         "> 由 `code/verify_independent.py` 生成。不调用 `states.py`：预处理用 sklearn ColumnTransformer/OneHotEncoder 重写，",
         "> 合成行、测试人群、权重、指标全部独立实现；与主网格落盘的单元格（5 折均值）逐状态逐指标比对。", "",
         "| 单元格 | 状态 | max \\|独立 − 主网格\\|（5 指标） |", "|---|---|---:|"]
    worst = 0.0
    for ds, did, pos, model, res, seed in CASES:
        f = os.path.join(HERE, "..", "raw", "main__%s__%s__%s__seed%02d.csv" % (ds, model, res, seed))
        if not os.path.exists(f):
            L.append("| %s/%s/%s/seed%d | — | *主网格该格尚未完成* |" % (ds, model, res, seed)); continue
        ref = pd.read_csv(f).groupby("state")[["auc", "auprc", "f1", "accuracy", "precision"]].mean()
        mine = run_case(ds, did, pos, model, res, seed)
        for s in ("CON", "LON", "LMN", "LMB"):
            d = float((mine[s] - ref.loc[s]).abs().max())
            worst = max(worst, d)
            L.append("| %s/%s/%s/seed%d | %s | %.1e |" % (ds, model, res, seed, s, d))
    L += ["", "**全部比对的最大绝对差：%.1e**" % worst]
    open(os.path.join(HERE, "..", "VERIFY_INDEPENDENT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
