"""四状态设计的严格不变量测试。运行：python3 -m pytest -q test_states.py

覆盖三类检验：
  1. 正向不变量 —— 独立于 states.py 内部断言，从返回的中间产物重新验算：
     逐折预处理只用 O_tr；CON/LON 同一 O_te；LON/LMN/LMB 同一模型同一预测；LMN 与 LMB 只差权重；
     训练集样本量与类别计数一致；AUC 的 P 为零；T = L + S + P；LMN 加权流行率 = 自然值；LMB = 0.5。
  2. 因果隔离 —— 只扰动 O_te 的特征值：预处理参数、G_clean、干净模型必须逐位不变（干净训练看不见 O_te），
     而全量重采样的 G_full 必须随之改变（泄漏训练看得见 O_te）。
  3. 负对照 —— 故意破坏保序 / 权重，不变量必须报错。能失败的检验才算检验。
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import states as S
import config as C
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


def synth_data(n=600, prev=0.2, seed=0, balanced=False):
    rng = np.random.default_rng(seed)
    n_pos = n // 2 if balanced else int(round(n * prev))
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    rng.shuffle(y)
    X = pd.DataFrame({
        "x1": rng.normal(size=n) + 1.2 * y,
        "x2": rng.normal(size=n) * 3 + 0.5 * y,
        "x3": rng.exponential(size=n),
        "cat_a": pd.Categorical(rng.choice(["u", "v", "w"], size=n)),
        "cat_b": pd.Categorical(np.where(rng.random(n) < 0.1, None,
                                         np.where(y == 1, rng.choice(["p", "q"], n), rng.choice(["q", "r"], n)))),
    })
    return X, y


def run(X, y, model="LogReg", res="SMOTE", seed=0):
    return S.run_four_states(X, y, lambda: C.make_model(model, seed),
                             lambda y_fit: C.make_resampler(res, seed, y_fit), seed=seed, keep_artifacts=True)


def recheck_fold(X, y, a, rec_by_state):
    tr, te, pre = a["tr"], a["te"], a["pre"]
    # 预处理只用 O_tr：拟合位置 = O_tr；词表 = O_tr 中的取值；标准化均值 = O_tr 编码后的列均值
    assert set(pre.fit_positions.tolist()) == set(tr.tolist())
    assert not (set(pre.fit_positions.tolist()) & set(te.tolist()))
    for c in pre.cat_cols:
        assert pre.vocab[c] == sorted(X.iloc[tr][c].dropna().astype(str).unique().tolist())
    np.testing.assert_allclose(pre.scaler.mean_, pre._encode(X.iloc[tr]).mean(axis=0), rtol=0, atol=1e-12)
    # 训练集匹配
    assert len(a["G_clean"]) == len(a["G_leak"])
    # 状态结构
    P = a["pops"]
    assert np.array_equal(P["CON"][0], P["LON"][0]) and np.array_equal(P["CON"][0], te)
    for k in range(3):
        assert np.array_equal(P["LMN"][k], P["LMB"][k])
    assert P["LMB"][3] is None and P["LMN"][3] is not None
    assert P["LON"][4] == P["LMN"][4] == P["LMB"][4]
    np.testing.assert_array_equal(P["LMN"][2][:len(te)], P["LON"][2])
    # 独立重算：LMN 的加权指标（手写权重，不调用 states 的权重函数）
    y_mix, p_mix = P["LMN"][1], P["LMN"][2]
    pi = (y[te] == 1).mean()
    n_pos_mix, n_neg_mix = (y_mix == 1).sum(), (y_mix == 0).sum()
    w = np.where(y_mix == 1, pi * n_neg_mix / ((1 - pi) * n_pos_mix), 1.0)
    assert abs(np.sum(w * (y_mix == 1)) / w.sum() - pi) < 1e-12
    yhat = (p_mix >= 0.5).astype(int)
    assert abs(f1_score(y_mix, yhat, sample_weight=w) - a["res"]["LMN"]["f1"]) < 1e-12
    assert abs(accuracy_score(y_mix, yhat, sample_weight=w) - a["res"]["LMN"]["accuracy"]) < 1e-12
    assert abs(roc_auc_score(y_mix, p_mix) - a["res"]["LMB"]["auc"]) < 1e-12
    assert a["res"]["LMB"]["prevalence"] == 0.5
    # AUC 的 P 为零；T = L + S + P
    r = a["res"]
    assert abs(r["LMB"]["auc"] - r["LMN"]["auc"]) < 1e-12
    for m in S.METRICS:
        c = S.contributions({s: r[s][m] for s in S.MAIN_STATES})
        assert abs(c["T"] - (c["L"] + c["S"] + c["P"])) < 1e-12


@pytest.mark.parametrize("res", ["SMOTE", "RandomOverSampler", "ADASYN"])
@pytest.mark.parametrize("model", ["LogReg", "DecisionTree"])
def test_invariants_all_resamplers(model, res):
    X, y = synth_data()
    recs, diags, arts = run(X, y, model, res)
    assert len(diags) == 5 and len(recs) == 5 * len(S.STATES)
    for a in arts:
        recheck_fold(X, y, a, None)
    for d in diags:
        assert d["train_match"] and d["pre_fit_equals_Otr"]
        assert d["telescope_maxabs"] < 1e-12 and d["auc_reweight_maxabs"] < 1e-12
        assert d["n_pos_train_clean"] == d["n_pos_train_leak"]
        if res in ("SMOTE", "RandomOverSampler"):     # 精确计数：无需修剪
            assert d["trim_clean"] == 0 and d["trim_leak"] == 0 and d["k_clean_raw"] == d["k_leak_raw"]
            assert d["k_clean_raw"] == d["n_neg_tr"] - d["n_pos_tr"]
        assert d["k_test"] == d["n_neg_te"] - d["n_pos_te"]


@pytest.mark.parametrize("model", ["LogReg", "DecisionTree"])
def test_auc_S_is_synthetic_share_times_easiness_gap(model):
    """正文 §4.1 的精确解读：类内等权 ⇒ S_AUC = 合成行占混合正类之比 × (AUC_合成正类 − AUC_原始正类)，
    两个 AUC 都对同一批 O_te 负类、同一个泄漏模型计算。"""
    X, y = synth_data(seed=5)
    _, _, arts = run(X, y, model, "SMOTE", seed=2)
    for a in arts:
        _, ym, pm, _, _ = a["pops"]["LMN"]
        n_te = len(a["te"]); k = len(ym) - n_te
        yte, plon = ym[:n_te], pm[:n_te]
        neg = plon[yte == 0]
        auc_orig = roc_auc_score(yte, plon)
        auc_syn = roc_auc_score(np.r_[np.zeros(len(neg), int), np.ones(k, int)], np.r_[neg, pm[n_te:]])
        share = k / ((yte == 1).sum() + k)
        S_auc = a["res"]["LMN"]["auc"] - a["res"]["LON"]["auc"]
        assert abs(S_auc - share * (auc_syn - auc_orig)) < 1e-12


def test_clean_training_blind_to_test_rows():
    """只扰动 O_te 的特征：干净侧（预处理、G_clean、干净模型）必须逐位不变；泄漏侧必须改变。"""
    X, y = synth_data(seed=3)
    _, _, arts0 = run(X, y, "LogReg", "SMOTE", seed=1)
    a0 = arts0[0]
    X2 = X.copy()
    te = a0["te"]
    X2.loc[te, "x1"] = X2.loc[te, "x1"] + 50.0
    X2.loc[te, "cat_a"] = "w"
    _, _, arts1 = run(X2, y, "LogReg", "SMOTE", seed=1)
    a1 = arts1[0]
    assert np.array_equal(a0["te"], a1["te"])
    np.testing.assert_array_equal(a0["pre"].scaler.mean_, a1["pre"].scaler.mean_)
    np.testing.assert_array_equal(a0["Z_tr"], a1["Z_tr"])
    np.testing.assert_array_equal(a0["G_clean"], a1["G_clean"])
    np.testing.assert_array_equal(a0["m_clean"].coef_, a1["m_clean"].coef_)
    assert not np.array_equal(a0["G_full"], a1["G_full"])      # 泄漏侧看得见 O_te


def test_balanced_data_is_exact_null():
    """数据已平衡：重采样为空操作，四个状态逐位相同，L = S = P = T = 0。"""
    X, y = synth_data(n=400, balanced=True)
    recs, diags, arts = run(X, y, "DecisionTree", "SMOTE")
    df = pd.DataFrame(recs)
    for fold, g in df.groupby("fold"):
        g = g.set_index("state")
        for m in S.METRICS:
            vals = [g.loc[s, m] for s in S.MAIN_STATES]
            assert max(vals) - min(vals) == 0.0, (fold, m, vals)
    assert all(d["k_test"] == 0 and d["k_clean_raw"] == 0 and d["k_full"] == 0 for d in diags)


class _PermutingResampler:
    """负对照：输出前缀被打乱的重采样器（违反保序）。"""
    def __init__(self, base):
        self.base = base
    def fit_resample(self, X, y):
        Xr, yr = self.base.fit_resample(X, y)
        idx = np.arange(len(yr)); idx[[0, 1]] = idx[[1, 0]]
        return Xr[idx], yr[idx]


def test_negative_control_order_violation_is_caught():
    X, y = synth_data(seed=5)
    with pytest.raises(S.InvariantError):
        S.run_four_states(X, y, lambda: C.make_model("LogReg", 0),
                          lambda y_fit: _PermutingResampler(C.make_resampler("SMOTE", 0, y_fit)), seed=0)


def test_negative_control_wrong_weights_are_caught(monkeypatch):
    X, y = synth_data(seed=6)
    real = S._weights_to_prevalence
    monkeypatch.setattr(S, "_weights_to_prevalence", lambda yy, t: real(yy, min(0.99, t * 1.1)))
    with pytest.raises(S.InvariantError):
        run(X, y)


def test_real_small_datasets():
    """真实数据冒烟：pima（纯数值）与 credit-g（13 个类别列，逐折词表）。"""
    import run_main_grid as R
    for label, did, pos in [("pima", 37, "tested_positive"), ("credit-g", 31, "bad")]:
        X, y = R.load_dataset(label, did, pos, C.ROW_CAP)
        recs, diags, arts = run(X, y, "LogReg", "SMOTE", seed=0)
        for a in arts:
            recheck_fold(X, y, a, None)
        if label == "credit-g":
            assert len(arts[0]["pre"].cat_cols) == 13


def test_aggregate_additivity_is_exact():
    """折均值（单元格层级）上 L + S + P = T 依然精确。"""
    X, y = synth_data(seed=7)
    recs, _, _ = run(X, y, "DecisionTree", "SMOTE", seed=2)
    df = pd.DataFrame(recs).groupby("state")[list(S.METRICS)].mean()
    for m in S.METRICS:
        c = S.contributions({s: df.loc[s, m] for s in S.MAIN_STATES})
        assert abs(c["T"] - (c["L"] + c["S"] + c["P"])) < 1e-12


# --------------------------------------------------------------------------- #
# Codex 只读复核后的 6 条加固（2026-09-25）
# --------------------------------------------------------------------------- #
def test_resampler_configured_only_from_fitted_labels():
    """加固 1：重采样器工厂只收到它将要拟合的那份标签——干净侧 = O_tr，泄漏侧 = O_tr ∪ O_te。"""
    from sklearn.model_selection import StratifiedKFold
    X, y = synth_data(seed=7)
    seen = []

    def factory(y_fit):
        seen.append(np.asarray(y_fit).copy())
        return C.make_resampler("SMOTE", 0, y_fit)

    S.run_four_states(X, y, lambda: C.make_model("LogReg", 0), factory, seed=0)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros((len(y), 1)), y))
    assert len(seen) == 2 * len(folds)
    for f, (tr, te) in enumerate(folds):
        np.testing.assert_array_equal(seen[2 * f], y[tr])
        np.testing.assert_array_equal(seen[2 * f + 1], y)


def test_make_resampler_k_follows_its_own_labels():
    y_small = np.array([1] * 4 + [0] * 40)
    assert C.make_resampler("SMOTE", 0, y_small).k_neighbors == 3
    assert C.make_resampler("ADASYN", 0, y_small).n_neighbors == 3
    assert C.make_resampler("SMOTE", 0, np.array([1] * 214 + [0] * 400)).k_neighbors == 5


def test_incomplete_cell_files_are_rejected(tmp_path):
    """加固 2：缺折 / 缺状态 / 指标缺失的单元格文件既不能被续跑跳过，也不能进入真源。"""
    import run_main_grid as R
    import build_ground_truth as G
    X, y = synth_data()
    recs, diags = S.run_four_states(X, y, lambda: C.make_model("LogReg", 0),
                                    lambda y_fit: C.make_resampler("SMOTE", 0, y_fit), seed=0)
    df, dg = pd.DataFrame(recs), pd.DataFrame(diags)
    assert S.state_frame_complete(df, 5) and S.diag_frame_complete(dg, 5)
    assert not S.state_frame_complete(df[df.fold != 4], 5)
    assert not S.state_frame_complete(df[df.state != "LMB"], 5)
    bad = df.copy()
    bad.loc[bad.index[0], "auc"] = np.nan
    assert not S.state_frame_complete(bad, 5)
    assert not S.diag_frame_complete(dg.iloc[:4], 5)

    out, dout = str(tmp_path / "main__a__LogReg__SMOTE__seed00.csv"), str(tmp_path / "diag.csv")
    R.write_atomic(df, out)
    R.write_atomic(dg, dout)
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f]           # 原子写入不留临时文件
    assert R.cell_complete(out, dout)
    df[df.fold != 4].to_csv(out, index=False)                               # 模拟中断留下的半写文件
    assert not R.cell_complete(out, dout)
    with pytest.raises(RuntimeError):
        G.load("main__*.csv", str(tmp_path))


def test_consistency_requires_all_datasets():
    """加固 3：少传一个数据集时必须报错，不能拿 5 个判定冒充 5/6。"""
    from decision import consistency_across_datasets, verdict_mean
    v = verdict_mean([0.01] * 10)
    assert consistency_across_datasets([v] * 6, key="mean")["consistent"]
    with pytest.raises(ValueError):
        consistency_across_datasets([v] * 5, key="mean")


def test_rank_flip_ties_and_completeness(tmp_path):
    """加固 4 + 5：精确并列的模型同为最佳，两协议有共同最佳即不算翻转；胜者分布并列各记 1/k；
    缺模型的组在主数据下报错。"""
    import rank_flip as RF
    rows = []

    def add(seed, con, lmb, models=("M1", "M2", "M3")):
        for m, c, l in zip(models, con, lmb):
            rows.append(dict(dataset="A", resampler="SMOTE", seed=seed, model=m,
                             auc_CON=c, auc_LON=c, auc_LMN=l, auc_LMB=l))

    add(0, (0.90, 0.80, 0.70), (0.95, 0.95, 0.90))      # LMB 上 M1 与 M2 并列第一，CON 胜者 M1 仍在其中
    add(1, (0.90, 0.80, 0.70), (0.80, 0.85, 0.99))      # LMB 胜者换成 M3
    path = str(tmp_path / "cells.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    R, B = RF.analyse("auc", path, strict=True)
    assert R.set_index("seed").flip.to_dict() == {0: 0, 1: 1}
    assert R.set_index("seed").winners_LMB[0] == ("M1", "M2")
    assert RF.win_counts(R, "LMB") == {"M1": 0.5, "M2": 0.5, "M3": 1.0}
    assert B.seed_flip.iloc[0] == 0.0                    # 两个种子的 CON 胜者都是 M1

    add(2, (0.90, 0.80), (0.95, 0.90), models=("M1", "M2"))   # 缺 M3 的组
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError):
        RF.analyse("auc", path, strict=True)
    R2, _ = RF.analyse("auc", path, strict=False)
    assert len(R2) == 2 and len(R2.attrs["incomplete"]) == 1


def test_median_verdict_is_explicitly_named():
    """加固 6：中位数判据只以 verdict_median 之名存在，旧名 verdict 已删除。"""
    import decision
    assert hasattr(decision, "verdict_median") and not hasattr(decision, "verdict")

