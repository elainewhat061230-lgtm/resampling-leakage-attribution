"""主实验配置 —— 逐条对应 `Ideation/preregistration_v2.md`（SHA-256 见 PREREG_FROZEN.txt）。

2026-09-25：主设计改为四状态有序路径分解（`states.py`）；模型、重采样器、种子、截断规则不变。

配置外置（KB SOP P3 要求）：参数不写死在流程代码里。
本文件的任何改动都等于改预注册，须走 DEVIATIONS.md。
"""
from __future__ import annotations

PREREG_VERSION = "v2"
PREREG_PATH = "preregistration_v2.md"
PREREG_SHA256 = "c2790917d0a930a99773d0d20de491e54dc46d1b2df451a9c3326a8f2e412b87"

# ---- 预注册 §2 数据 ----
ROW_CAP = 20_000                 # 行数上限
MIN_MINORITY = 300               # v2：分层后少数类不得低于此数，否则退回保全少数类
CAP_SEED = 7                     # default_rng(7)

DATASETS = [
    # (标签, OpenML data_id, 正类取值；None = 取少数类)
    ("creditcard",      1597,  None),
    ("bank-marketing",  1461,  None),
    ("adult",           1590,  ">50K"),
    ("credit-default", 42477,  None),
    ("credit-g",          31,  "bad"),
    ("pima",              37,  "tested_positive"),
]

# ---- 预注册 §3 模型（按容量排序）----
MODEL_NAMES = ["LogReg", "DecisionTree", "RandomForest", "XGBoost"]

# ---- 预注册 §4 重采样器 ----
RESAMPLER_NAMES = ["SMOTE", "RandomOverSampler", "ADASYN"]

# ---- 预注册 §7 重复 ----
SEEDS = list(range(30))          # 0..29
N_SPLITS = 5

# ---- 预注册 §2 极端端稳健性检查 ----
EXTREME_CHECK = dict(dataset="creditcard", data_id=1597, cap=None,
                     resamplers=["SMOTE"], seeds=list(range(5)))

# ---- 预注册 §5 受控阶梯 ----
LADDER = dict(datasets=[("creditcard", 1597, None), ("bank-marketing", 1461, None)],
              n_minority=400, ratios=[2, 4, 8, 16, 64, 256],
              resamplers=["SMOTE"], seeds=list(range(10)))

# ---- 四状态设计的试点（2026-09-25 重构；全量重跑前必须先通过）----
PILOT = dict(datasets=[("credit-g", 31, "bad"), ("bank-marketing", 1461, None)],
             models=["LogReg", "RandomForest"], resamplers=["SMOTE"], seeds=list(range(5)),
             # 另用两个重采样器各跑少量格，专门覆盖 ADASYN 计数修剪与 ROS 复制行路径
             extra=[("credit-g", 31, "bad", "LogReg", "RandomOverSampler", [0, 1]),
                    ("credit-g", 31, "bad", "LogReg", "ADASYN", [0, 1]),
                    ("bank-marketing", 1461, None, "RandomForest", "ADASYN", [0])])

# ---- 输出 ----
RAW_DIR = "../raw"               # 每单元格一个状态 CSV（行 = 折 × 状态）
DIAG_DIR = "../raw_diag"         # 每单元格一个诊断 CSV（行 = 折；全部不变量的实测值）
PILOT_RAW_DIR = "../pilot_4state/raw"
PILOT_DIAG_DIR = "../pilot_4state/raw_diag"
LOG_PATH = "../run_main_grid.log"


def make_model(name, seed):
    """预注册 §3 的出厂设置。n_jobs=1：并行放在单元格层面，避免过度订阅。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    if name == "LogReg":
        return LogisticRegression(max_iter=3000)
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=seed, min_samples_leaf=20)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, random_state=seed,
                                      n_jobs=1, min_samples_leaf=5)
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 random_state=seed, n_jobs=1, eval_metric="logloss")
    raise ValueError(name)


def make_resampler(name, seed, y=None):
    """预注册 §4。y = 该重采样器【将要拟合的那份】标签（干净侧只传 O_tr 的标签）；
    少数类不足 6 行时把 k_neighbors 收缩到 n_min − 1（本项目全部网格少数类 ≥ 214 行，从未触发）。"""
    from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
    k = 5
    if y is not None:
        n_min = int(min((y == 0).sum(), (y == 1).sum()))
        k = max(1, min(5, n_min - 1))
    if name == "SMOTE":
        return SMOTE(random_state=seed, k_neighbors=k)
    if name == "RandomOverSampler":
        return RandomOverSampler(random_state=seed)
    if name == "ADASYN":
        return ADASYN(random_state=seed, n_neighbors=k)
    raise ValueError(name)


def cap_indices(y, cap=ROW_CAP, min_min=MIN_MINORITY, seed=CAP_SEED):
    """预注册 v2 §2 的截断规则。

    v1 的规则（无条件保全少数类）会把 adult 的 1:3 翻成 1.4:1 —— 少数类变多数类。
    v2：优先分层保持原始比例；仅当分层后少数类少于下限时才退回保全少数类。
    返回 (选中的行索引, 模式说明)。
    """
    import numpy as np
    n = len(y)
    if n <= cap:
        return np.arange(n), "untruncated"
    rng = np.random.default_rng(seed)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    k_pos = int(round(cap * float(np.mean(y))))
    if k_pos >= min_min and k_pos <= len(pi):
        sel = np.concatenate([rng.choice(pi, k_pos, replace=False),
                              rng.choice(ni, cap - k_pos, replace=False)])
        mode = "stratified"
    else:
        sel = np.concatenate([pi, rng.choice(ni, cap - len(pi), replace=False)])
        mode = "keep_all_minority"
    sel.sort()
    return sel, mode
