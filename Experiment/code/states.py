"""四状态有序路径分解 —— 评估器、逐折预处理与硬性不变量。

状态按"训练方式 / 测试组成 / 测试流行率"命名：

    CON  Clean–Original–Natural   干净训练；只评原始测试行 O_te；不加权（自然流行率）
    LON  Leaky–Original–Natural   泄漏训练；与 CON 完全相同的 O_te；不加权
    LMN  Leaky–Mixed–Natural      与 LON 同一个已拟合模型；评 O_te + G_test；
                                  类内均匀加权，使加权正类占比 = O_te 的自然正类占比
    LMB  Leaky–Mixed–Balanced     与 LMN 同一模型、同一批测试行、同一组预测；不加权（50/50）

预设分解顺序：clean training → training-side leakage → synthetic-test mixture → prevalence shift

    L = M(LON) − M(CON)   training-side leakage contribution
    S = M(LMN) − M(LON)   synthetic-test mixture contribution
    P = M(LMB) − M(LMN)   prevalence contribution
    T = M(LMB) − M(CON)   total inflation；T = L + S + P 为代数恒等式，逐折校验 |误差| < 1e-12

每个相邻状态只改一个条件：训练集中的合成行是否由"看得见 O_te 的"重采样器生成（CON→LON）；
测试人群是否混入合成行，流行率保持自然值（LON→LMN）；评估权重从自然流行率改为平衡（LMN→LMB）。

构造（每个种子只在原始数据上生成一次 StratifiedKFold，四个状态共享同一 O_tr / O_te）：

  * 逐折预处理（类别词表、缺失填充、StandardScaler）**只在 O_tr 上拟合**，再施加于 O_tr、O_te
    及全量重采样所需的 O_tr ∪ O_te —— 不让预处理泄漏混入重采样泄漏。
  * Clean：仅用 O_tr 拟合重采样器，G_clean = 其生成的合成行；CON 训练集 = O_tr + G_clean。
  * Leaky：用（经 O_tr 拟合的预处理变换后的）O_tr ∪ O_te 拟合全量重采样器，G_full = 其合成行；
    以固定种子、与标签和预测无关的均匀随机置换把 G_full 分成
      G_test（k_test = n_neg_te − n_pos_te 行，使 O_te + G_test 恰为 50/50）与 G_leak_train（其余）。
    LON / LMN / LMB 共用同一个泄漏模型，训练集 = O_tr + G_leak_train。
  * 对 SMOTE 与 RandomOverSampler（目标 1:1 时生成数恰为 n_maj − n_min），
    |G_leak_train| = |G_clean| 由算术保证，无需任何调整。
    ADASYN 的生成数只近似等于目标（内部按密度比取整）：此时两侧取 m = min(|G_clean|, |G_leak_train|)，
    对多出的一侧做固定种子的均匀随机下采样，并逐折记录修剪行数 —— 显式处理，不静默改定义。

补充因子态（不进入主分解，只用于顺序依赖分析；与主状态共用同两个模型、同两批测试行）：

    COB  干净模型 · O_te · 加权到 50/50       LOB  泄漏模型 · O_te · 加权到 50/50
    CMN  干净模型 · 混合 · 加权回自然流行率     CMB  干净模型 · 混合 · 不加权
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score)

MAIN_STATES = ("CON", "LON", "LMN", "LMB")
AUX_STATES = ("COB", "LOB", "CMN", "CMB")
STATES = MAIN_STATES + AUX_STATES
METRICS = ("auc", "auprc", "f1", "accuracy", "precision", "recall")
CONTRIB = ("L", "S", "P", "T")
TOL = 1e-12                     # 代数恒等与解析零的浮点容差
SALT_SPLIT = 20260925           # G_full → G_test / G_leak_train 的置换种子盐
SALT_TRIM = 20260926            # ADASYN 计数匹配修剪的种子盐


class InvariantError(RuntimeError):
    """任一不变量被违反 —— 中止该单元格，不允许带着错误状态继续产出数字。"""


def _require(cond, msg):
    if not bool(cond):
        raise InvariantError(msg)


# --------------------------------------------------------------------------- #
# 逐折预处理：一切参数只从 O_tr 学习
# --------------------------------------------------------------------------- #
def _is_categorical(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object


class FoldPreprocessor:
    """类别列：词表 = O_tr 中出现过的非缺失取值（排序）；未见值与缺失编码为全零指示行
    （与旧管线 pd.get_dummies 对缺失的编码一致）。数值列：缺失填 0（常数，无可学习参数）。
    全部输出列（数值 + 指示列）：StandardScaler 只在 O_tr 上拟合。"""

    def fit(self, X: pd.DataFrame, positions: np.ndarray) -> "FoldPreprocessor":
        positions = np.asarray(positions)
        part = X.iloc[positions]
        self.cat_cols = [c for c in X.columns if _is_categorical(X[c])]
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        self.vocab = {c: sorted(part[c].dropna().astype(str).unique().tolist())
                      for c in self.cat_cols}
        self.scaler = StandardScaler().fit(self._encode(part))
        self.fit_positions = np.array(positions, copy=True)
        self.n_out = int(self.scaler.n_features_in_)
        return self

    def _encode(self, part: pd.DataFrame) -> np.ndarray:
        blocks = []
        if self.num_cols:
            blocks.append(part[self.num_cols].astype(float).fillna(0.0).to_numpy())
        for c in self.cat_cols:
            s = part[c]
            codes = pd.Categorical(s.astype(str).where(s.notna()),
                                   categories=self.vocab[c]).codes
            M = np.zeros((len(part), len(self.vocab[c])))
            ok = codes >= 0
            M[np.flatnonzero(ok), codes[ok]] = 1.0
            blocks.append(M)
        return np.hstack(blocks) if blocks else np.zeros((len(part), 0))

    def transform(self, X: pd.DataFrame, positions=None) -> np.ndarray:
        part = X if positions is None else X.iloc[np.asarray(positions)]
        return self.scaler.transform(self._encode(part))


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def score(y_true, p_score, y_pred, sample_weight=None) -> dict:
    two_class = len(np.unique(y_true)) > 1
    out = {
        "auc": roc_auc_score(y_true, p_score, sample_weight=sample_weight) if two_class else np.nan,
        "auprc": average_precision_score(y_true, p_score, sample_weight=sample_weight) if two_class else np.nan,
        "f1": f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "precision": precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
        "recall": recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0),
    }
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, float)
    out["n_eval"] = int(len(y_true))
    out["n_pos"] = int((y_true == 1).sum())
    out["n_neg"] = int((y_true == 0).sum())
    out["prevalence"] = float(np.sum(w * (y_true == 1)) / np.sum(w))
    out["weighted"] = sample_weight is not None
    return out


# --------------------------------------------------------------------------- #
# 过采样：按 imblearn 的保序行为取合成块，并逐次断言
# --------------------------------------------------------------------------- #
def oversample(resampler, Z: np.ndarray, y: np.ndarray, minority: int) -> tuple[np.ndarray, dict]:
    """返回合成块 G 与诊断。imblearn 的 SMOTE / ADASYN / RandomOverSampler 都把原始行原样
    放在输出前 n 行、合成行追加在后；这里对前缀做逐位断言，失败即中止 —— 合成行的识别
    因此不依赖内容哈希（数据自带重复行时哈希会退化），也不依赖未经校验的顺序假设。"""
    Zr, yr = resampler.fit_resample(Z, y)
    n = len(y)
    _require(Zr.shape[0] >= n and np.array_equal(Zr[:n], Z) and np.array_equal(yr[:n], y),
             "重采样器输出的前 n 行不等于输入（保序假设失效）——无法识别合成行")
    G, gy = np.asarray(Zr[n:], dtype=float), np.asarray(yr[n:])
    _require(np.all(gy == minority), "合成块中出现非少数类标签")
    return G, {"n_in": int(n), "n_synth": int(len(G))}


def _weights_to_prevalence(y: np.ndarray, target: float) -> np.ndarray:
    """类内均匀权重：负类权重 1，正类权重 c，使加权正类占比恰为 target。"""
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    c = (target / (1.0 - target)) * (n_neg / n_pos)
    return np.where(y == 1, c, 1.0)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_four_states(X: pd.DataFrame, y, make_model, make_resampler, seed: int,
                    n_splits: int = 5, keep_artifacts: bool = False):
    """返回 (state_records, fold_diags[, artifacts])。

    make_model()          → 未拟合的模型
    make_resampler(y_fit) → 未拟合的重采样器；只允许用它将要拟合的那份标签来配置（如 k 近邻数），
                            因此干净重采样器的配置只看 O_tr 的标签，泄漏重采样器看 O_tr ∪ O_te 的标签。
    state_records：每折 × 8 个状态一行（4 个主状态 + 4 个补充因子态），含全部指标。
    fold_diags   ：每折一行，含全部不变量的实测值（违反即在此前已抛 InvariantError）。
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    _require(set(np.unique(y).tolist()) == {0, 1}, "标签必须为 0/1 二分类")
    minority = 1      # 约定：正类 = 少数类（平衡数据允许并列，此时重采样为空操作）
    _require(int((y == 1).sum()) <= int((y == 0).sum()), "正类多于负类，违反「正类 = 少数类」约定，须先重编码")

    records, diags, arts = [], [], []
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(np.zeros((len(y), 1)), y)):
        d = {"seed": seed, "fold": fold}

        # ---- 逐折预处理：只在 O_tr 上拟合 ----
        pre = FoldPreprocessor().fit(X, tr)
        fitset = set(pre.fit_positions.tolist())
        _require(fitset <= set(tr.tolist()), "预处理拟合行不是 O_tr 的子集")
        _require(not (fitset & set(te.tolist())), "预处理拟合行与 O_te 相交")
        d["pre_fit_n"] = len(fitset)
        d["pre_fit_equals_Otr"] = fitset == set(tr.tolist())
        d["n_features"] = pre.n_out
        Z_tr, Z_te, Z_all = pre.transform(X, tr), pre.transform(X, te), pre.transform(X)
        y_tr, y_te = y[tr], y[te]
        n_te = len(te)
        n_pos_te, n_neg_te = int((y_te == 1).sum()), int((y_te == 0).sum())
        n_pos_tr, n_neg_tr = int((y_tr == 1).sum()), int((y_tr == 0).sum())
        d.update(n_tr=len(tr), n_te=n_te, n_pos_tr=n_pos_tr, n_neg_tr=n_neg_tr,
                 n_pos_te=n_pos_te, n_neg_te=n_neg_te)

        # ---- Clean：重采样器只看 O_tr ----
        G_clean, dc = oversample(make_resampler(y_tr), Z_tr, y_tr, minority)
        # ---- Leaky：重采样器看 O_tr ∪ O_te（原始行序；预处理仍只在 O_tr 上拟合）----
        G_full, df_ = oversample(make_resampler(y), Z_all, y, minority)
        k_test = n_neg_te - n_pos_te
        _require(k_test >= 0, "测试折少数类多于多数类，k_test < 0")
        _require(len(G_full) >= k_test, "全量重采样的合成行不足以把测试侧补到 50/50")
        perm = np.random.default_rng([seed, fold, SALT_SPLIT]).permutation(len(G_full))
        G_test, G_leak = G_full[perm[:k_test]], G_full[perm[k_test:]]
        k_clean_raw, k_leak_raw = len(G_clean), len(G_leak)
        d.update(k_clean_raw=k_clean_raw, k_full=len(G_full), k_test=k_test, k_leak_raw=k_leak_raw)

        # ---- 训练集计数匹配（SMOTE/ROS 恒等；ADASYN 显式修剪并记录）----
        trim_clean = trim_leak = 0
        if k_clean_raw != k_leak_raw:
            m = min(k_clean_raw, k_leak_raw)
            rng = np.random.default_rng([seed, fold, SALT_TRIM])
            if k_clean_raw > m:
                G_clean = G_clean[np.sort(rng.choice(k_clean_raw, m, replace=False))]
                trim_clean = k_clean_raw - m
            if k_leak_raw > m:
                G_leak = G_leak[np.sort(rng.choice(k_leak_raw, m, replace=False))]
                trim_leak = k_leak_raw - m
        d.update(k_matched=len(G_clean), trim_clean=trim_clean, trim_leak=trim_leak)

        Xc = np.vstack([Z_tr, G_clean]); yc = np.concatenate([y_tr, np.full(len(G_clean), minority)])
        Xl = np.vstack([Z_tr, G_leak]);  yl = np.concatenate([y_tr, np.full(len(G_leak), minority)])
        _require(len(yc) == len(yl) and int(yc.sum()) == int(yl.sum()),
                 "CON 与泄漏训练集的样本量或类别计数不一致")
        d.update(n_train=len(yc), n_pos_train_clean=int(yc.sum()), n_pos_train_leak=int(yl.sum()),
                 train_match=True)

        # ---- 两个模型；全部状态的预测只来自这两个对象 ----
        m_clean = make_model().fit(Xc, yc)
        m_leak = make_model().fit(Xl, yl)
        p_con = m_clean.predict_proba(Z_te)[:, 1].astype(float)
        p_lon = m_leak.predict_proba(Z_te)[:, 1].astype(float)
        if k_test > 0:
            p_lgt = m_leak.predict_proba(G_test)[:, 1].astype(float)
            p_cgt = m_clean.predict_proba(G_test)[:, 1].astype(float)
            joint = m_leak.predict_proba(np.vstack([Z_te, G_test]))[:, 1].astype(float)
        else:
            p_lgt = p_cgt = np.zeros(0)
            joint = p_lon
        _require(all(np.all(np.isfinite(v)) for v in (p_con, p_lon, p_lgt, p_cgt)),
                 "预测概率中出现非有限值")
        p_lmix = np.concatenate([p_lon, p_lgt])
        p_cmix = np.concatenate([p_con, p_cgt])
        d["batch_pred_maxdiff"] = float(np.max(np.abs(joint - p_lmix))) if len(joint) else 0.0
        _require(d["batch_pred_maxdiff"] < 1e-9, "泄漏模型对同一行的预测随批次改变")

        # ---- 测试人群：行 ID（O_te 为原始位置，合成行为 −1−j）、标签、权重 ----
        ids_orig = te.copy()
        ids_mix = np.concatenate([te, -1 - np.arange(k_test)])
        y_mix = np.concatenate([y_te, np.full(k_test, minority)])
        _require(int((y_mix == 1).sum()) == int((y_mix == 0).sum()), "混合测试人群不是精确 50/50")
        pi_te = n_pos_te / n_te
        w_nat_mix = _weights_to_prevalence(y_mix, pi_te)
        w_bal_te = _weights_to_prevalence(y_te, 0.5)

        pops = {   # 状态 → (行 ID, 标签, 预测概率, 权重, 模型身份)
            "CON": (ids_orig, y_te, p_con, None, id(m_clean)),
            "LON": (ids_orig, y_te, p_lon, None, id(m_leak)),
            "LMN": (ids_mix, y_mix, p_lmix, w_nat_mix, id(m_leak)),
            "LMB": (ids_mix, y_mix, p_lmix, None, id(m_leak)),
            "COB": (ids_orig, y_te, p_con, w_bal_te, id(m_clean)),
            "LOB": (ids_orig, y_te, p_lon, w_bal_te, id(m_leak)),
            "CMN": (ids_mix, y_mix, p_cmix, w_nat_mix, id(m_clean)),
            "CMB": (ids_mix, y_mix, p_cmix, None, id(m_clean)),
        }

        # ---- 结构性不变量 ----
        _require(np.array_equal(pops["CON"][0], pops["LON"][0]) and np.array_equal(pops["CON"][1], pops["LON"][1]),
                 "CON 与 LON 的 O_te 行 ID / 顺序 / 标签不一致")
        _require(pops["LON"][4] == pops["LMN"][4] == pops["LMB"][4] and pops["CON"][4] != pops["LON"][4],
                 "LON / LMN / LMB 未共用同一个已拟合的泄漏模型")
        _require(np.array_equal(p_lmix[:n_te], p_lon), "LMN 在 O_te 上的预测不等于 LON 的预测")
        for k in (0, 1, 2):
            _require(np.array_equal(pops["LMN"][k], pops["LMB"][k]),
                     "LMN 与 LMB 的测试行 / 标签 / 预测不完全相同")
        _require(np.array_equal(pops["LMN"][0][:n_te], te), "混合人群的 O_te 块与原始测试行不一致")
        d["lmn_lmb_weights_differ"] = bool(k_test > 0 and not np.allclose(w_nat_mix, 1.0))

        # ---- 评分 ----
        res = {}
        for s in STATES:
            ids, yy, pp, ww, _ = pops[s]
            r = score(yy, pp, (pp >= 0.5).astype(int), ww)
            res[s] = r
            records.append(dict(state=s, fold=fold, seed=seed, **r))
        d["w_pos_LMN"] = float(w_nat_mix[y_mix == 1][0]) if (y_mix == 1).any() else np.nan
        d["prev_CON"], d["prev_LMN"], d["prev_LMB"] = res["CON"]["prevalence"], res["LMN"]["prevalence"], res["LMB"]["prevalence"]
        _require(abs(res["LMN"]["prevalence"] - pi_te) < TOL, "LMN 加权正类占比不等于 O_te 自然流行率")
        _require(res["LMB"]["prevalence"] == 0.5, "LMB 流行率不是 0.5")
        _require(res["CON"]["prevalence"] == res["LON"]["prevalence"] == pi_te, "CON/LON 流行率不等于自然值")

        # ---- 解析零：类内均匀重加权不改变 AUC ----
        auc_gaps = [abs(res[a]["auc"] - res[b]["auc"]) for a, b in
                    (("LMB", "LMN"), ("COB", "CON"), ("LOB", "LON"), ("CMB", "CMN"))]
        d["auc_P_abs"] = auc_gaps[0]
        d["auc_reweight_maxabs"] = float(max(auc_gaps))
        _require(d["auc_reweight_maxabs"] < TOL, "类内均匀重加权改变了 ROC-AUC（应为解析零）")

        # ---- 代数恒等：T = L + S + P ----
        tel = 0.0
        for mtr in METRICS:
            L_ = res["LON"][mtr] - res["CON"][mtr]
            S_ = res["LMN"][mtr] - res["LON"][mtr]
            P_ = res["LMB"][mtr] - res["LMN"][mtr]
            T_ = res["LMB"][mtr] - res["CON"][mtr]
            tel = max(tel, abs(T_ - (L_ + S_ + P_)))
        d["telescope_maxabs"] = tel
        _require(tel < TOL, "逐折 T ≠ L + S + P（超出浮点容差）")

        diags.append(d)
        if keep_artifacts:
            arts.append(dict(tr=tr, te=te, pre=pre, Z_tr=Z_tr, Z_te=Z_te, G_clean=G_clean,
                             G_leak=G_leak, G_test=G_test, G_full=G_full, pops=pops,
                             m_clean=m_clean, m_leak=m_leak, res=res))
    return (records, diags, arts) if keep_artifacts else (records, diags)


def state_frame_complete(df: pd.DataFrame, n_splits: int = 5) -> bool:
    """单元格状态文件是否完整：8 个状态 × n_splits 折各恰好一行，全部指标无缺失。
    续跑判断（run_main_grid）与汇总前校验（build_ground_truth）共用，防止半写文件被当成已完成。"""
    if not {"state", "fold"} <= set(df.columns) or not set(METRICS) <= set(df.columns):
        return False
    if len(df) != n_splits * len(STATES) or set(df["state"]) != set(STATES):
        return False
    for _, g in df.groupby("state"):
        if len(g) != n_splits or set(g["fold"]) != set(range(n_splits)):
            return False
    return not df[list(METRICS)].isna().any().any()


def diag_frame_complete(dg: pd.DataFrame, n_splits: int = 5) -> bool:
    """诊断文件是否完整：n_splits 折各恰好一行。"""
    return "fold" in dg.columns and len(dg) == n_splits and set(dg["fold"]) == set(range(n_splits))


def contributions(state_metric: dict) -> dict:
    """由四个主状态的指标值得到 L / S / P / T（任意聚合层级上都成立的线性关系）。"""
    CON, LON, LMN, LMB = (state_metric[s] for s in MAIN_STATES)
    return {"L": LON - CON, "S": LMN - LON, "P": LMB - LMN, "T": LMB - CON}
