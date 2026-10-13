"""按正确顺序重生成全部派生产物 —— 改完数据/代码后跑这一个脚本即可。

**为什么必须有这个脚本**（2026-09-01 实测踩到）：
  `make_figures.py` 会**删除并重建** `figure_numbers.log`，而 `rq5_explanations.py`
  是**追加**写入同一个日志。因此顺序必须是 figures → rq5；漏跑 rq5 时日志会静默
  丢掉 RQ5 的全部数字，正文里那些数字随即失去真源（审计 A 路会报 major）。
  KB 纪律"派生文档必须脚本生成"——手工按顺序敲四条命令就是手工维护。

用法：
    python3 rebuild_all.py          # 全链条
    python3 rebuild_all.py --check  # 跑完顺带跑审计
"""
from __future__ import annotations
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))

# 顺序有意义，勿调换：
#   1 真源必须最先（图与表都从它/decomposed_cells.csv 取数）
#   2 figures 会清空 figure_numbers.log
#   3 rank_flip / tables / rq5 追加写入该日志，必须在 figures 之后
CHAIN = [
    ("build_ground_truth.py", "真源 RESULTS_GROUND_TRUTH.md"),
    ("make_figures.py",       "图 + figure_numbers.log（会清空该日志）"),
    ("rank_flip.py",          "RQ2′ 翻转率（追加日志）"),
    ("make_tables.py",        "表 1–5（追加日志）"),
    ("rq5_explanations.py",   "RQ5 解释一致性（追加日志）"),
]


def main() -> int:
    t0 = time.time()
    for i, (script, what) in enumerate(CHAIN, 1):
        print("[%d/%d] %-24s %s" % (i, len(CHAIN), script, what), flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            print("✗ %s 失败，链条中止（后续产物会与前面不一致，不要继续）" % script)
            return r.returncode
    print("✅ 全链条完成，耗时 %.1f 秒" % (time.time() - t0))

    if "--check" in sys.argv:
        audit = os.path.join(HERE, "..", "..", "Publication", "audit_all.py")
        print("\n—— 跑定稿前审计 ——", flush=True)
        return subprocess.run([sys.executable, os.path.abspath(audit)]).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
