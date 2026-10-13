"""效应确立判据 —— 预注册第 7 节的可执行版本。

主判据：**跨种子中位数的自助法 95% 置信区间，必须完全落在噪声带之外。**
    噪声带 = [-0.003, +0.003]（Roth 2026 §3.5 实测的种子噪声地板）
辅判据（并列报告，供直觉参照）：30 次中同号次数。

为什么用置信区间做主判据而不用投票计数：
  * 投票计数只用了每次跑的【符号】，扔掉了幅度信息
  * 24/30 这个门槛看着像"80% 多数"，实际严得多 —— 若真无效应，纯靠运气达到
    24/30 的概率仅 0.07%，比常规 p<0.05 严约 70 倍。这会把真实但幅度小的效应
    （如骨架里 LogReg 的真泄漏 +0.004 AUC）判成"不确立"
  * 置信区间在方法学读者面前更容易辩护

为什么投票计数仍然保留：
  * 对偏态分布稳健，且"30 次里 26 次同向"普通读者一眼就懂
  * 两者结论不一致时，本身就是值得报告的信号
"""
from __future__ import annotations
import numpy as np

NOISE_FLOOR = 0.003          # Roth (2026) §3.5
N_BOOT = 10000
CI_LEVEL = 0.95
BOOT_SEED = 20260830         # 固定，保证可复现


def bootstrap_median_ci(values, n_boot=N_BOOT, level=CI_LEVEL, seed=BOOT_SEED):
    """对跨种子的估计值做自助法，返回中位数及其百分位法置信区间。"""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 3:
        return dict(median=np.nan, lo=np.nan, hi=np.nan, n=len(v))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    meds = np.median(v[idx], axis=1)
    a = (1 - level) / 2
    return dict(median=float(np.median(v)),
                lo=float(np.quantile(meds, a)),
                hi=float(np.quantile(meds, 1 - a)),
                n=int(len(v)))


def vote_count(values, floor=None):
    """同号次数（辅判据）。返回 (同号数, 总数, 方向)。

    ⚠️ 投票判据【完全忽略幅度】：−1e-17 也算一票"负"。在解析零效应（如 AUC 上的
    机制①）上，它会给出 30/30 一致的假象。因此当中位数绝对值低于噪声地板时，
    本函数返回方向 0，调用方须据此把投票标为"不适用"。
    这个缺陷本身就是把置信区间定为主判据的理由之一。
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return 0, 0, 0
    if floor is not None and abs(float(np.median(v))) < floor:
        return 0, len(v), 0          # 幅度低于地板 → 符号无意义
    npos, nneg = int((v > 0).sum()), int((v < 0).sum())
    return (npos, len(v), +1) if npos >= nneg else (nneg, len(v), -1)


def verdict(values, floor=NOISE_FLOOR):
    """按预注册第 7 节判定。

    确立   : 置信区间完全落在噪声带之外（整段 > +floor 或整段 < -floor）
    不确立 : 置信区间与噪声带相交
    """
    ci = bootstrap_median_ci(values)
    agree, n, direction = vote_count(values, floor=floor)
    if not np.isfinite(ci["median"]):
        est = "数据不足"
    elif ci["lo"] > floor:
        est = "确立（正）"
    elif ci["hi"] < -floor:
        est = "确立（负）"
    else:
        est = "不确立"
    return dict(**ci, vote_agree=agree, vote_n=n, vote_direction=direction,
                verdict=est,
                ci_excludes_floor=est.startswith("确立"),
                # 旧的投票门槛也一并算出来，便于在论文里说明两者是否一致
                vote_applicable=bool(direction != 0),
                legacy_24of30=bool(direction != 0 and n and agree / n >= 24 / 30),
                agrees_with_legacy=None)


def annotate_agreement(d):
    """标注主判据与旧投票门槛是否给出同一结论 —— 不一致本身值得报告。
    投票不适用时（幅度低于地板）不判分歧。"""
    d["agrees_with_legacy"] = (True if not d["vote_applicable"]
                               else bool(d["ci_excludes_floor"] == d["legacy_24of30"]))
    return d


def consistency_across_datasets(per_dataset_verdicts, min_agree=5, n_total=6):
    """跨数据集一致性：≥5/6 个数据集方向相同且各自确立。"""
    est = [v for v in per_dataset_verdicts if v["ci_excludes_floor"]]
    if not est:
        return dict(consistent=False, n_established=0, n_same_sign=0, note="无数据集达到确立")
    signs = [1 if v["median"] > 0 else -1 for v in est]
    npos, nneg = signs.count(1), signs.count(-1)
    top = max(npos, nneg)
    return dict(consistent=bool(top >= min_agree), n_established=len(est),
                n_same_sign=top, n_total=n_total,
                direction=(1 if npos >= nneg else -1))
