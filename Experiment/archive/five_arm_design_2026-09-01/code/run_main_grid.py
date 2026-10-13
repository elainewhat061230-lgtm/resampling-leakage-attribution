"""主实验驱动 —— 严格按 `Ideation/preregistration.md` 执行。

用法：
    python3 run_main_grid.py               # 跑主网格
    python3 run_main_grid.py extreme       # 极端端稳健性检查（creditcard 不截断）
    python3 run_main_grid.py ladder        # 受控阶梯

特性：
  * 一配置一 CSV（预注册 §10），行 = 单折
  * **可续跑**：已存在且完整的 CSV 直接跳过，中断后重跑不丢进度
  * 跨单元格多进程并行；模型内部 n_jobs=1，避免过度订阅
  * 启动时校验预注册哈希，不符即拒跑
"""
from __future__ import annotations
import os, sys, time, json, hashlib, warnings, traceback
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from arms import run_five_arms

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, C.RAW_DIR))
os.makedirs(RAW, exist_ok=True)


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.abspath(os.path.join(HERE, C.LOG_PATH)), "a") as f:
        f.write(line + "\n")


def check_prereg():
    p = os.path.abspath(os.path.join(HERE, "..", "..", "Ideation", C.PREREG_PATH))
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != C.PREREG_SHA256:
        raise SystemExit("❌ 预注册文件已变动！\n  期望 %s\n  实际 %s\n"
                         "  主实验拒绝运行。若确需修改预注册，须新建 v2 并在论文中明写 post-hoc。"
                         % (C.PREREG_SHA256, h))
    log("✅ 预注册 %s 哈希校验通过 %s" % (C.PREREG_VERSION, h[:16]))


_CACHE = {}


def load_dataset(label, data_id, pos, cap=C.ROW_CAP):
    key = (data_id, pos, cap)
    if key in _CACHE:
        return _CACHE[key]
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = pd.get_dummies(d.data).astype(float).fillna(0).values
    t = d.target.astype(str).str.strip()
    y = (t == pos).astype(int).values if pos else (t == t.value_counts().idxmin()).astype(int).values
    if cap:
        sel, _mode = C.cap_indices(y, cap=cap)
        X, y = X[sel], y[sel]
    _CACHE[key] = (X, y)
    return X, y


def cell_path(tag, ds, model, res, seed):
    return os.path.join(RAW, "%s__%s__%s__%s__seed%02d.csv" % (tag, ds, model, res, seed))


def run_cell(job):
    tag, ds_label, data_id, pos, cap, model, res, seed, extra = job
    out = cell_path(tag, ds_label, model, res, seed)
    if os.path.exists(out) and os.path.getsize(out) > 200:
        return ("skip", out, 0.0)
    t0 = time.time()
    try:
        X, y = load_dataset(ds_label, data_id, pos, cap)
        if extra:                                   # 阶梯：固定少数类，只改多数类数量
            ratio, n_min = extra["ratio"], extra["n_minority"]
            rng = np.random.default_rng(1000 + seed)
            pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
            if len(pi) < n_min or len(ni) < n_min * (ratio - 1):
                return ("insufficient", out, time.time() - t0)
            sel = np.concatenate([rng.choice(pi, n_min, replace=False),
                                  rng.choice(ni, n_min * (ratio - 1), replace=False)])
            X, y = X[sel], y[sel]
        res_d, diag = run_five_arms(
            X, y,
            lambda: C.make_model(model, seed),
            lambda: C.make_resampler(res, seed, y),
            seed=seed, n_splits=C.N_SPLITS)
        df = pd.DataFrame(diag["fold_records"])
        for k, v in [("tag", tag), ("dataset", ds_label), ("model", model),
                     ("resampler", res), ("n_rows", len(y)),
                     ("prevalence_orig", float(y.mean())),
                     ("order_ok", diag["mark_report"]["order_assumption_holds"]),
                     ("mark_source", diag["mark_report"]["is_orig_source"]),
                     ("ratio", (extra or {}).get("ratio", ""))]:
            df[k] = v
        df.to_csv(out, index=False)
        return ("ok", out, time.time() - t0)
    except Exception:
        with open(out + ".error", "w") as f:
            f.write(traceback.format_exc())
        return ("error", out, time.time() - t0)


def build_jobs(mode):
    jobs = []
    if mode == "main":
        for ds, did, pos in C.DATASETS:
            for m in C.MODEL_NAMES:
                for r in C.RESAMPLER_NAMES:
                    for s in C.SEEDS:
                        jobs.append(("main", ds, did, pos, C.ROW_CAP, m, r, s, None))
    elif mode == "extreme":
        e = C.EXTREME_CHECK
        for m in C.MODEL_NAMES:
            for r in e["resamplers"]:
                for s in e["seeds"]:
                    jobs.append(("extreme", e["dataset"], e["data_id"], None, None, m, r, s, None))
    elif mode == "ladder":
        L = C.LADDER
        for ds, did, pos in L["datasets"]:
            for ratio in L["ratios"]:
                for m in C.MODEL_NAMES:
                    for r in L["resamplers"]:
                        for s in L["seeds"]:
                            jobs.append(("ladder-r%d" % ratio, ds, did, pos, None, m, r, s,
                                         dict(ratio=ratio, n_minority=L["n_minority"])))
    else:
        raise SystemExit("未知模式: %s" % mode)
    return jobs


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    check_prereg()
    jobs = build_jobs(mode)
    done = sum(1 for j in jobs if os.path.exists(cell_path(j[0], j[1], j[5], j[6], j[7])))
    n_proc = max(1, min(8, cpu_count() - 2))
    log("模式=%s  单元格 %d 个（已完成 %d）  并行进程 %d" % (mode, len(jobs), done, n_proc))

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "error": 0, "insufficient": 0}
    with Pool(n_proc) as pool:
        for i, (status, path, dt) in enumerate(pool.imap_unordered(run_cell, jobs, chunksize=1), 1):
            counts[status] += 1
            if i % 25 == 0 or status == "error":
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                eta = (len(jobs) - i) / rate / 60 if rate > 0 else 0
                log("  %d/%d  ok=%d skip=%d err=%d  已用 %.0f 分  预计剩 %.0f 分  %s"
                    % (i, len(jobs), counts["ok"], counts["skip"], counts["error"],
                       el / 60, eta, os.path.basename(path) if status == "error" else ""))
    log("模式=%s 完成：%s  总耗时 %.1f 分钟" % (mode, counts, (time.time() - t0) / 60))
    if counts["error"]:
        log("⚠️ 有 %d 个单元格失败，错误详情见 raw/*.csv.error" % counts["error"])


if __name__ == "__main__":
    main()
