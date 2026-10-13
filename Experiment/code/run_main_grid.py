"""实验驱动 —— 四状态有序路径分解（`states.py`）。

用法：
    python3 run_main_grid.py pilot      # 试点：2 数据集 × {LogReg, RandomForest} × SMOTE × 5 种子（+ ROS/ADASYN 少量格）
    python3 run_main_grid.py            # 主网格 6 × 4 × 3 × 30
    python3 run_main_grid.py extreme    # 极端稳健性（creditcard 不截断，1:578）
    python3 run_main_grid.py ladder     # 受控阶梯（固定 400 少数类，只改多数类）

每个单元格产出两个 CSV：
    raw/      行 = 折 × 状态（4 个主状态 + 4 个补充因子态），全部指标
    raw_diag/ 行 = 折，全部不变量的实测值（任一不变量失败即抛错，该格记 .error，不产出数字）

特性：可续跑（两文件都在且内容完整才跳过；写入先落临时文件再原子改名）；跨单元格多进程并行，模型内部 n_jobs=1；
启动校验预注册哈希。
"""
from __future__ import annotations
import os, sys, time, hashlib, warnings, traceback
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from states import run_four_states, state_frame_complete, diag_frame_complete

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = None
RAW = DIAG = None


def set_dirs(mode):
    global RAW, DIAG
    if mode == "pilot":
        RAW = os.path.abspath(os.path.join(HERE, C.PILOT_RAW_DIR))
        DIAG = os.path.abspath(os.path.join(HERE, C.PILOT_DIAG_DIR))
    else:
        RAW = os.path.abspath(os.path.join(HERE, C.RAW_DIR))
        DIAG = os.path.abspath(os.path.join(HERE, C.DIAG_DIR))
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(DIAG, exist_ok=True)


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.abspath(os.path.join(HERE, C.LOG_PATH)), "a") as f:
        f.write(line + "\n")


def check_prereg():
    p = os.path.abspath(os.path.join(HERE, "..", "..", "Ideation", C.PREREG_PATH))
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != C.PREREG_SHA256:
        raise SystemExit("❌ 预注册文件已变动！期望 %s 实际 %s —— 拒绝运行。" % (C.PREREG_SHA256, h))
    log("✅ 预注册 %s 哈希校验通过 %s" % (C.PREREG_VERSION, h[:16]))


_CACHE = {}


def load_dataset(label, data_id, pos, cap=C.ROW_CAP):
    """返回【未编码】的原始 DataFrame 与 0/1 标签。编码、缺失填充、标准化一律移入逐折预处理
    （states.FoldPreprocessor，只在 O_tr 上拟合）。截断只做行选择（只用标签分层，固定种子）。"""
    key = (data_id, pos, cap)
    if key in _CACHE:
        return _CACHE[key]
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = d.data.copy()
    t = d.target.astype(str).str.strip()
    y = (t == pos).astype(int).values if pos else (t == t.value_counts().idxmin()).astype(int).values
    if cap:
        sel, _mode = C.cap_indices(y, cap=cap)
        X, y = X.iloc[sel].reset_index(drop=True), y[sel]
    _CACHE[key] = (X, y)
    return X, y


def cell_paths(tag, ds, model, res, seed):
    stem = "%s__%s__%s__%s__seed%02d.csv" % (tag, ds, model, res, seed)
    return os.path.join(RAW, stem), os.path.join(DIAG, stem)


def write_atomic(df, path):
    """先写同目录临时文件再原子改名：中断时目标路径上要么是旧的完整文件，要么不存在，不会有半写文件。"""
    tmp = "%s.tmp-%d" % (path, os.getpid())
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def cell_complete(out, dout, n_splits=C.N_SPLITS):
    """续跑判据：两文件都在且内容完整（每状态 n_splits 折、指标无缺失；诊断 n_splits 行）。
    旧判据"文件存在且 > 200 字节"会把中断留下的半写文件当成已完成。"""
    if not (os.path.exists(out) and os.path.exists(dout)):
        return False
    try:
        return state_frame_complete(pd.read_csv(out), n_splits) and diag_frame_complete(pd.read_csv(dout), n_splits)
    except Exception:
        return False


def run_cell(job):
    tag, ds_label, data_id, pos, cap, model, res, seed, extra, raw_dir, diag_dir = job
    stem = "%s__%s__%s__%s__seed%02d.csv" % (tag, ds_label, model, res, seed)
    out, dout = os.path.join(raw_dir, stem), os.path.join(diag_dir, stem)
    if cell_complete(out, dout):
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
            X, y = X.iloc[sel].reset_index(drop=True), y[sel]
        recs, diags = run_four_states(
            X, y,
            lambda: C.make_model(model, seed),
            lambda y_fit: C.make_resampler(res, seed, y_fit),
            seed=seed, n_splits=C.N_SPLITS)
        meta = [("tag", tag), ("dataset", ds_label), ("model", model), ("resampler", res),
                ("n_rows", len(y)), ("prevalence_orig", float(y.mean())),
                ("ratio", (extra or {}).get("ratio", ""))]
        df, dg = pd.DataFrame(recs), pd.DataFrame(diags)
        for k, v in meta:
            df[k] = v
            dg[k] = v
        write_atomic(dg, dout)
        write_atomic(df, out)
        return ("ok", out, time.time() - t0)
    except Exception:
        with open(out + ".error", "w") as f:
            f.write(traceback.format_exc())
        return ("error", out, time.time() - t0)


def build_jobs(mode):
    J = []
    add = lambda *a: J.append(tuple(a) + (RAW, DIAG))
    if mode == "main":
        for ds, did, pos in C.DATASETS:
            for m in C.MODEL_NAMES:
                for r in C.RESAMPLER_NAMES:
                    for s in C.SEEDS:
                        add("main", ds, did, pos, C.ROW_CAP, m, r, s, None)
    elif mode == "pilot":
        P = C.PILOT
        for ds, did, pos in P["datasets"]:
            for m in P["models"]:
                for r in P["resamplers"]:
                    for s in P["seeds"]:
                        add("pilot", ds, did, pos, C.ROW_CAP, m, r, s, None)
        for ds, did, pos, m, r, seeds in P["extra"]:
            for s in seeds:
                add("pilot", ds, did, pos, C.ROW_CAP, m, r, s, None)
    elif mode == "extreme":
        e = C.EXTREME_CHECK
        for m in C.MODEL_NAMES:
            for r in e["resamplers"]:
                for s in e["seeds"]:
                    add("extreme", e["dataset"], e["data_id"], None, None, m, r, s, None)
    elif mode == "ladder":
        L = C.LADDER
        for ds, did, pos in L["datasets"]:
            for ratio in L["ratios"]:
                for m in C.MODEL_NAMES:
                    for r in L["resamplers"]:
                        for s in L["seeds"]:
                            add("ladder-r%d" % ratio, ds, did, pos, None, m, r, s,
                                dict(ratio=ratio, n_minority=L["n_minority"]))
    else:
        raise SystemExit("未知模式: %s" % mode)
    # 大单元格先跑（RF/DT 与大数据集），减少尾部空转
    weight = {"RandomForest": 3, "DecisionTree": 2, "XGBoost": 1, "LogReg": 0}
    J.sort(key=lambda j: (-weight.get(j[5], 0), j[1]))
    return J


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    set_dirs(mode)
    check_prereg()
    jobs = build_jobs(mode)
    n_proc = max(1, min(8, cpu_count() - 2))
    log("模式=%s  单元格 %d 个  并行进程 %d  输出 %s" % (mode, len(jobs), n_proc, RAW))
    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "error": 0, "insufficient": 0}
    with Pool(n_proc) as pool:
        for i, (status, path, dt) in enumerate(pool.imap_unordered(run_cell, jobs, chunksize=1), 1):
            counts[status] += 1
            if i % 25 == 0 or status == "error" or i == len(jobs):
                el = time.time() - t0
                eta = (len(jobs) - i) / (i / el) / 60 if el > 0 else 0
                log("  %d/%d  ok=%d skip=%d err=%d insuff=%d  已用 %.0f 分  预计剩 %.0f 分  %s"
                    % (i, len(jobs), counts["ok"], counts["skip"], counts["error"], counts["insufficient"],
                       el / 60, eta, os.path.basename(path) if status == "error" else ""))
    log("模式=%s 完成：%s  总耗时 %.1f 分钟" % (mode, counts, (time.time() - t0) / 60))
    if counts["error"]:
        log("⚠️ 有 %d 个单元格失败（含不变量违反），详情见 raw/*.csv.error" % counts["error"])
        sys.exit(2)


if __name__ == "__main__":
    main()
