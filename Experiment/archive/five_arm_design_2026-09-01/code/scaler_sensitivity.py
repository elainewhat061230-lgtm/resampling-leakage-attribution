"""Scaler 敏感性核查 —— DEVIATIONS D-003（EXTERNAL_REVIEW D1，路线 A）的配套证据。

背景：主管线在任何划分之前对全量数据拟合一次 StandardScaler，六臂共用
（`arms.py` run_five_arms 开头）。预注册 v2 §2 原文要求"在每个臂内部按该臂的
训练集拟合"，但该规则照字面执行会让泄漏管线的 SMOTE 跑在未标准化特征上，
制造新的几何混杂（详见 DEVIATIONS D-003）。裁定：维持共用全量拟合 + 本脚本
量化其对 arm B 绝对值的影响上界。

设计：
  * 只跑 LogReg —— 树模型对逐特征仿射变换（缩放）不变，不受影响
  * 只跑 B 臂 —— 臂间差分中共用 scaler 完全抵消，受影响的只有 B 的绝对值
    （它喂给 RQ2′ 的胜者判定与 Table 5 的锚点）
  * 变体 common：全量拟合（与主网格逐位一致，作复制核验）
  * 变体 foldfit：每折按训练折拟合 scaler，SMOTE 在折内标准化后的训练折上跑
  * 折划分与主网格完全相同（StratifiedKFold 只依赖 y 与长度，不依赖 X 取值）

输出：`Experiment/scaler_sensitivity.csv`，由 build_ground_truth.py §8 汇总进真源。
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import run_main_grid as R
from arms import score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUTCSV = os.path.abspath(os.path.join(HERE, "..", "scaler_sensitivity.csv"))
MODEL = "LogReg"                     # 唯一受缩放影响的模型


def one_config(job):
    label, did, pos, res, seed = job
    X, y = R.load_dataset(label, did, pos, C.ROW_CAP)
    Xs_common = StandardScaler().fit_transform(X)
    f1c, aucc, f1f, aucf = [], [], [], []
    # 与主网格同款折划分；重采样器工厂同款传全量 y（k_neighbors 由全量少数类数决定）
    for tr, te in StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=seed).split(X, y):
        # --- 变体 common：全量拟合（应与主网格 B 臂逐位一致）---
        Xtr, ytr = C.make_resampler(res, seed, y).fit_resample(Xs_common[tr], y[tr])
        m = C.make_model(MODEL, seed).fit(Xtr, ytr)
        p = m.predict_proba(Xs_common[te])[:, 1]
        rc = score(y[te], p, (p >= 0.5).astype(int))
        f1c.append(rc["f1"]); aucc.append(rc["auc"])
        # --- 变体 foldfit：按训练折拟合 ---
        sc = StandardScaler().fit(X[tr])
        Xtr2, ytr2 = C.make_resampler(res, seed, y).fit_resample(sc.transform(X[tr]), y[tr])
        m2 = C.make_model(MODEL, seed).fit(Xtr2, ytr2)
        p2 = m2.predict_proba(sc.transform(X[te]))[:, 1]
        rf_ = score(y[te], p2, (p2 >= 0.5).astype(int))
        f1f.append(rf_["f1"]); aucf.append(rf_["auc"])
    return dict(dataset=label, resampler=res, seed=seed,
                auc_common=float(np.mean(aucc)), auc_foldfit=float(np.mean(aucf)),
                f1_common=float(np.mean(f1c)), f1_foldfit=float(np.mean(f1f)),
                d_auc=float(np.mean(aucf) - np.mean(aucc)),
                d_f1=float(np.mean(f1f) - np.mean(f1c)))


def main():
    jobs = [(label, did, pos, res, seed)
            for label, did, pos in C.DATASETS
            for res in C.RESAMPLER_NAMES
            for seed in C.SEEDS]
    t0 = time.time()
    n_proc = max(1, min(8, cpu_count() - 2))
    print("scaler 敏感性：%d 配置（%s，B 臂 × 2 变体），并行 %d" % (len(jobs), MODEL, n_proc), flush=True)
    with Pool(n_proc) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one_config, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 30 == 0:
                print("  %d/%d  已用 %.0fs" % (i, len(jobs), time.time() - t0), flush=True)
    s = pd.DataFrame(rows).sort_values(["dataset", "resampler", "seed"]).reset_index(drop=True)

    # ---- 复制核验：common 变体 vs 主网格 raw CSV 的 B 臂（LogReg）----
    import glob
    recs = []
    for f in glob.glob(os.path.join(HERE, C.RAW_DIR, "main__*__%s__*__seed*.csv" % MODEL)):
        d = pd.read_csv(f)
        b = d[d.arm == "B"]
        recs.append(dict(dataset=b.dataset.iloc[0], resampler=b.resampler.iloc[0],
                         seed=int(b.seed.iloc[0]),
                         raw_auc=float(b.auc.mean()), raw_f1=float(b.f1.mean())))
    raw = pd.DataFrame(recs)
    s = s.merge(raw, on=["dataset", "resampler", "seed"], how="left")
    s["chk_auc"] = s.auc_common - s.raw_auc
    s["chk_f1"] = s.f1_common - s.raw_f1
    s.to_csv(OUTCSV, index=False)

    print("\n== 汇总（LogReg，B 臂）==")
    print("  |ΔAUC| max=%.4f  中位=%.4f" % (s.d_auc.abs().max(), s.d_auc.abs().median()))
    print("  |ΔF1|  max=%.4f  中位=%.4f" % (s.d_f1.abs().max(), s.d_f1.abs().median()))
    print("  复制核验 vs raw：max|AUC 差|=%.2e  max|F1 差|=%.2e"
          % (s.chk_auc.abs().max(), s.chk_f1.abs().max()))
    print("  ✅ 写入 %s   （%.1f 分钟）" % (OUTCSV, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
