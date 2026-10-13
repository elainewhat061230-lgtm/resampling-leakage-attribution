"""五臂评估器 —— 把"划分前重采样造成的指标虚高"拆成三个机制。

臂的定义（B 为基线，其余各隔离一个机制）：

    B  正确      先划分 → 只对训练折重采样 → 在【原始测试折】上评估
    A  照文献    先全量重采样 → 再划分 → 在【全部测试行】上评估          = ①+②+③
    C  真泄漏    先全量重采样 → 再划分 → 在【原始多数类 + 原始少数类】上评估 = ③
    C2 匹配版 C  与 C 同模型同多数类，但原始少数类【下采样到与 D 等量】——
                 专供 ② 的对照使用（见下方"双 C"说明）
    D  合成难度  先全量重采样 → 再划分 → 在【原始多数类 + k 条合成少数类】上评估
                 （k = min(原始少数, 合成少数)）
    E  纯流行率  用 B 臂的干净模型 → 在【原始测试折】上评估，但按权重把
                 有效流行率拉到 50/50                                      = ①

推导：
    ③ = C  − B         真泄漏（测试人群与 B 同分布，只是训练集被污染）
    ② = D − C2         合成少数类行比原始少数类行好预测多少
    ① = E  − B         纯流行率效应（同一个模型、同一批行，只改评估权重）

⚠️ **双 C（2026-09-01，DEVIATIONS D-004）**：当某折的合成少数行 < 原始少数行
   （⇔ 正类占比 > 1/3，六个数据集中仅 pima），把 D 下采样到 k 之后 C 与 D 的
   流行率就不再相等——修正前 `D − C` 在 pima 上混入了约 −0.016 的机械流行率差，
   制造了一个"确立为负"的伪② 。修法是两个对比各配各的对照：
     ③ 用 C（全部原始行，与 B 的测试人群匹配）——修正前后不变；
     ② 用 C2（原始少数类下采样到 k，与 D 严格等量等流行率）。
   凡合成 ≥ 原始的折，C2 ≡ C 逐位相等，故其余五个数据集数字不变。

⚠️ 三者未必可加，A − B ≠ ①+②+③ 是预期内的，须如实报告而非强行凑。

设计约束（来自 W1 闸门与阶段 5 评审，全部为硬条款）：
  * imblearn 保序假设必须有【运行时断言】，且用哈希法交叉核验 —— 这是版本相关行为
  * D 臂不能写成"只在合成行上评估"：SMOTE 只合成少数类，那样是单一类别，AUC 无定义
  * D 臂的合成少数类必须【下采样到与 C 的原始少数类等量】。骨架第一版没做，
    结果 C 流行率 0.22 / D 流行率 0.42，两臂差了两样东西 —— 正是 KB 红旗条款
    "这两臂之间到底改了几样东西"要抓的错误
  * AUC 对【类内均匀】的重新加权是解析恒等的，故 E 臂在 AUC 上必然恰好为 0。
    这是数学恒等式，不是实测结果，报告时必须如此标注
  * 指标必须同时覆盖对流行率敏感（accuracy/F1/precision）与不敏感（AUC/AUPRC）两类
  * 噪声地板沿用 Roth (2026) §3.5：|ΔAUC| < 0.003 视为种子伪影
"""
from __future__ import annotations

import hashlib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score)

NOISE_FLOOR_AUC = 0.003          # Roth (2026) §3.5
ARMS = ("B", "A", "C", "C2", "D", "E")


# --------------------------------------------------------------------------- #
# 合成行识别：不能只信 imblearn 的返回顺序
# --------------------------------------------------------------------------- #
def _row_hashes(X: np.ndarray) -> np.ndarray:
    """逐行内容哈希，用于独立于返回顺序地识别原始行。"""
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    return np.array([hashlib.blake2b(r.tobytes(), digest_size=16).hexdigest() for r in Xc])


def mark_original_rows(X_orig: np.ndarray, y_orig: np.ndarray,
                       X_res: np.ndarray, y_res: np.ndarray) -> tuple[np.ndarray, dict]:
    """返回 is_orig 布尔数组 + 一份核验报告。

    主判据：imblearn 文档化的"原始行在前 n 行"行为（带断言）。
    交叉核验：逐行哈希匹配。两者不一致时以【哈希法】为准并在报告里标红 ——
    因为顺序假设是版本相关的，哈希是内容判据。
    """
    n0 = len(y_orig)
    report: dict = {}

    order_ok = (len(y_res) >= n0
                and np.array_equal(y_res[:n0], y_orig)
                and np.allclose(X_res[:n0], X_orig, rtol=0, atol=0))
    report["order_assumption_holds"] = bool(order_ok)

    by_order = np.zeros(len(y_res), dtype=bool)
    by_order[:n0] = True

    orig_hashes = set(_row_hashes(X_orig).tolist())
    res_hashes = _row_hashes(X_res)
    by_hash = np.array([h in orig_hashes for h in res_hashes])

    report["n_orig_by_order"] = int(by_order.sum())
    report["n_orig_by_hash"] = int(by_hash.sum())
    report["methods_agree"] = bool(np.array_equal(by_order, by_hash))
    # 随机过采样是【精确复制】原始行，哈希法会把复制品也算成原始行 —— 这不是 bug，
    # 但意味着 D 臂对 ROS 退化。调用方需据此解释。
    report["hash_method_degenerate_for_duplication"] = bool(
        report["n_orig_by_hash"] > n0)

    # 运行时断言（HANDOFF §3 硬条款；EXTERNAL_REVIEW D3）：
    # 顺序假设失效且哈希判据又因重复行退化时，C/D 行标签不可信 —— 必须中止，
    # 不能带着错误标签继续跑然后指望事后审计抓到。
    if not order_ok and report["hash_method_degenerate_for_duplication"]:
        raise RuntimeError(
            "imblearn 保序假设失效（order_assumption_holds=False），且哈希判据因重复行退化，"
            "无法可靠识别原始行 —— C/C2/D 臂标签不可信，中止运行。请检查 imblearn 版本行为。")

    is_orig = by_hash if not report["hash_method_degenerate_for_duplication"] else by_order
    report["is_orig_source"] = ("hash" if not report["hash_method_degenerate_for_duplication"]
                                else "order(fallback: duplication detected)")
    if not order_ok:
        report["WARNING"] = "imblearn 未保序 —— 顺序假设失效，已改用哈希法"
    return is_orig, report


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def score(y_true, p_score, y_pred, sample_weight=None) -> dict:
    """同时报对流行率敏感与不敏感两类指标。单一类别时置 nan 而非抛错。"""
    out = {}
    two_class = len(np.unique(y_true)) > 1
    out["auc"] = roc_auc_score(y_true, p_score, sample_weight=sample_weight) if two_class else np.nan
    out["auprc"] = average_precision_score(y_true, p_score, sample_weight=sample_weight) if two_class else np.nan
    out["f1"] = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    out["accuracy"] = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    out["precision"] = precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    out["n_eval"] = int(len(y_true))
    out["prevalence"] = float(np.average(y_true, weights=sample_weight))
    return out


def _prevalence_weights(y, target=0.5) -> np.ndarray:
    """给每行一个权重，使加权后的正类占比等于 target。不改变任何一行的内容。"""
    w = np.ones(len(y), dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return w
    w[y == 1] = target / n_pos
    w[y == 0] = (1.0 - target) / n_neg
    return w * len(y) / w.sum()


# --------------------------------------------------------------------------- #
# 五臂主流程
# --------------------------------------------------------------------------- #
def run_five_arms(X, y, make_model, make_resampler, seed=42, n_splits=5, scale=True):
    """返回 {arm: {metric: mean over folds}} 与一份诊断报告。

    make_model / make_resampler 均为无参工厂函数，每折新建，避免状态泄漏。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    Xs = StandardScaler().fit_transform(X) if scale else X

    diag: dict = {"seed": seed, "n": len(y), "prevalence_orig": float(y.mean())}

    # ---- 泄漏管道：先对全量重采样，再划分 ----
    X_res, y_res = make_resampler().fit_resample(Xs, y)
    is_orig, mark_report = mark_original_rows(Xs, y, X_res, y_res)
    diag["mark_report"] = mark_report
    diag["n_resampled"] = int(len(y_res))
    diag["prevalence_resampled"] = float(y_res.mean())

    folds = {a: [] for a in ARMS}
    fold_records = []          # 预注册 §10 要求：行 = 单折

    # ---- B 臂与 E 臂：干净管道，在原始数据上分折 ----
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xs, y):
        Xtr, ytr = make_resampler().fit_resample(Xs[tr], y[tr])
        m = make_model().fit(Xtr, ytr)
        p = m.predict_proba(Xs[te])[:, 1]
        yhat = (p >= 0.5).astype(int)
        rB = score(y[te], p, yhat)
        folds["B"].append(rB)
        fold_records.append(dict(arm="B", fold=len(folds["B"]) - 1, seed=seed, **rB))
        # E：同一个模型、同一批行，只把评估权重改成 50/50
        rE = score(y[te], p, yhat, sample_weight=_prevalence_weights(y[te], 0.5))
        folds["E"].append(rE)
        fold_records.append(dict(arm="E", fold=len(folds["E"]) - 1, seed=seed, **rE))

    # ---- A / C / D 臂：泄漏管道，在重采样后的数据上分折 ----
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X_res, y_res):
        m = make_model().fit(X_res[tr], y_res[tr])
        p = m.predict_proba(X_res[te])[:, 1]
        yhat = (p >= 0.5).astype(int)
        yte, orig_te = y_res[te], is_orig[te]

        rA = score(yte, p, yhat)
        folds["A"].append(rA)
        fold_records.append(dict(arm="A", fold=len(folds["A"]) - 1, seed=seed, **rA))

        # C：原始多数类 + 原始少数类（即测试折里全部原始行）—— ③ 的对照，与 B 的测试人群匹配
        selC = orig_te
        # D：原始多数类 + k 条合成少数类，k = min(原始少数, 合成少数)
        # C2：原始多数类 + k 条原始少数类 —— ② 的对照，与 D 严格等量等流行率
        #     （双 C 修正，DEVIATIONS D-004：凡合成 ≥ 原始的折 k = 原始数，C2 ≡ C）
        idx_orig_min = np.where(orig_te & (yte == 1))[0]
        idx_synth_min = np.where(~orig_te & (yte == 1))[0]
        k = min(len(idx_orig_min), len(idx_synth_min))
        rng_fold = np.random.default_rng(seed * 1000 + len(folds["A"]))
        # ⚠️ keep_synth 必须先于 keep_orig 抽取：保持与修正前完全相同的 rng 消耗顺序，
        #    使 D 臂在全部数据集上与修正前逐位一致（修正只新增 C2，不改动任何旧臂）
        keep_synth = rng_fold.choice(idx_synth_min, k, replace=False) if k > 0 else np.array([], int)
        if k < len(idx_orig_min):
            keep_orig = rng_fold.choice(idx_orig_min, k, replace=False) if k > 0 else np.array([], int)
        else:
            keep_orig = idx_orig_min
        selD = np.zeros(len(yte), dtype=bool)
        selD[orig_te & (yte == 0)] = True          # 与 C 完全相同的多数类行
        selD[keep_synth] = True
        selC2 = np.zeros(len(yte), dtype=bool)
        selC2[orig_te & (yte == 0)] = True         # 与 C / D 完全相同的多数类行
        selC2[keep_orig] = True
        for arm, sel in (("C", selC), ("C2", selC2), ("D", selD)):
            if sel.sum() >= 10 and len(np.unique(yte[sel])) > 1:
                r = score(yte[sel], p[sel], yhat[sel])
                folds[arm].append(r)
                fold_records.append(dict(arm=arm, fold=len(folds[arm]) - 1, seed=seed, **r))

    results = {}
    for a in ARMS:
        if folds[a]:
            keys = folds[a][0].keys()
            results[a] = {k: float(np.mean([f[k] for f in folds[a]])) for k in keys}
            results[a]["n_folds_used"] = len(folds[a])
        else:
            results[a] = None
            diag.setdefault("empty_arms", []).append(a)
    diag["fold_records"] = fold_records
    return results, diag


def decompose(results: dict, metric="auc") -> dict:
    """把三个机制拆出来。返回 nan 而非抛错，缺臂时如实标注。

    ② 用 C2（与 D 等量等流行率）作对照；无 C2 时（旧结果）退回 C ——
    凡合成 ≥ 原始的折 C2 ≡ C，故退回在那些场景下严格等价。"""
    g = lambda a: (results.get(a) or {}).get(metric, np.nan)
    B, A, C, D, E = (g(a) for a in ("B", "A", "C", "D", "E"))
    C2 = g("C2")
    if not np.isfinite(C2):
        C2 = C
    out = {
        "metric": metric,
        "B_correct": B, "A_as_published": A, "C_true_leak": C,
        "C2_matched": C2,
        "D_synth_rows": D, "E_prevalence_only": E,
        "m3_true_leakage": C - B,
        "m2_synthetic_rows": D - C2,
        "m1_prevalence": E - B,
        "total_observed_A_minus_B": A - B,
    }
    parts = out["m1_prevalence"] + out["m2_synthetic_rows"] + out["m3_true_leakage"]
    out["sum_of_parts"] = parts
    out["additivity_gap"] = out["total_observed_A_minus_B"] - parts
    if metric == "auc":
        out["below_noise_floor"] = bool(abs(out["m3_true_leakage"]) < NOISE_FLOOR_AUC)
        # 解析恒等：类内均匀的重新加权不改变 AUC，故 m1 必然为 0
        out["m1_is_analytic_identity"] = True
    else:
        out["m1_is_analytic_identity"] = False
    return out
