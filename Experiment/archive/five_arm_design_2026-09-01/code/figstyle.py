"""配图共享样式模块 —— 字体 / dpi / 配色 / 刻度约定的唯一定义处。

所有作图脚本必须 `import figstyle as fs` 并在开头调用 `fs.apply_style()`，
不得各自设 rcParams。

目标期刊 IJDNS 是 **Word 刊**，无 LaTeX class 可编译测宽。正文宽度改由
**已发表 PDF 的实际文本块**量得（poppler `pdftotext -bbox`，四篇样本，
三篇一致于 496 pt）：

    ijdns_2025.7.003  496.1 pt      ijdns_2026.4.022  496.1 pt
    ijdns_2026.4.024  496.2 pt      ijdns_2023.6.003  464.9 pt（旧模板，左边距不同）

    → TEXTWIDTH = 496 pt / 72 = 6.89 in

⚠️ 若按常见的 6.5 in 出图，`width=\\textwidth` 会把图放大 6%，
   标称 8.5 pt 的刻度字实际印成 9.0 pt，图与正文的字号关系就错了。
"""
from __future__ import annotations
import os, shutil, subprocess, tempfile
import matplotlib

TEXTWIDTH = 496.0 / 72.0          # 6.889 in，实测自已发表 PDF
COLWIDTH = TEXTWIDTH              # IJDNS 正文单栏
DPI = 300


def _usetex_works() -> bool:
    """探测必须走【完整导出路径】，不能只编译 DVI。

    两次教训（同一个错犯了两遍，故记在此）：
      1. 初版只跑 `latex t.tex`，DVI 编译通过就判定可用 —— 实际在 matplotlib
         存 PDF 时炸：mathptmx 的 PostScript 字体 zptmcm7y 不在 texlive-basic 里。
         **DVI 能编译 ≠ PDF 能导出。**
      2. 二版改成真存 PDF，但探针只写了 `$x_1$` —— 那用的是数学罗马字体，
         而真正缺的是**数学符号字体**（\Delta 等）。探针仍判通过，图照样炸。
         **探针必须用图里真正会出现的符号，不能用简化样例。**
    """
    PROBE = r"$\Delta$ $\rightarrow$ $-$ $\alpha_1$ \textbf{Ab} 5\%"   # 与正式图同款符号
    if not (shutil.which("latex") and shutil.which("dvipng")):
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with matplotlib.rc_context({
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{amsmath}",
                "font.family": "serif"}):
            with tempfile.TemporaryDirectory() as d:
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.set_xlabel(PROBE)
                ax.set_ylabel(PROBE)
                ax.set_title(PROBE)
                for ext in ("pdf", "png"):        # 两种输出都要能出
                    fig.savefig(os.path.join(d, "t." + ext))
                plt.close(fig)
        return True
    except Exception:
        return False


USE_TEX = _usetex_works()

# ---- LaTeX 特殊字符：定义常量统一处理，避免文字里的 % 炸编译 ----
PCT = r"\%" if USE_TEX else "%"
ENDASH = "--" if USE_TEX else "–"


def bf(text: str) -> str:
    """粗体。usetex 下 fontweight='bold' 会被静默忽略，必须显式包 \\textbf。"""
    return rf"\textbf{{{text}}}" if USE_TEX else text


# ---- 配色：色盲安全，色相/饱和度/明度三维拉开；冗余用 marker 编码 ----
COLORS = {
    # 三个机制
    "m1_prevalence":     "#4C72B0",   # 蓝
    "m2_synthetic_rows": "#DD8452",   # 橙
    "m3_true_leakage":   "#55A868",   # 绿
    # 四个模型（按容量排序，明度递减）
    "LogReg":       "#9BB7D4",
    "DecisionTree": "#5B8FC9",
    "RandomForest": "#2E5E8A",
    "XGBoost":      "#16324F",
    # 重采样器
    "SMOTE": "#4C72B0", "RandomOverSampler": "#C44E52", "ADASYN": "#8172B3",
}
MARKERS = {"LogReg": "o", "DecisionTree": "s", "RandomForest": "^", "XGBoost": "D",
           "SMOTE": "o", "RandomOverSampler": "s", "ADASYN": "^"}

LABELS = {"m1_prevalence": "(1) Prevalence shift",
          "m2_synthetic_rows": "(2) Synthetic test rows",
          "m3_true_leakage": "(3) True leakage"}

# ---- 字号档：在 TEXTWIDTH 下即真实印刷字号 ----
FS_BODY, FS_AXLAB, FS_PANEL, FS_TICK, FS_LEG = 9, 9.5, 10, 8.5, 8.5
ANNOT = 7.5


def apply_style():
    rc = {
        "text.usetex": USE_TEX,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.grid": False,                       # 非 colormap 图不画网格
        "legend.frameon": False,                  # 图例无边框
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,   # 刻度上下左右对称
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "font.size": FS_BODY, "axes.labelsize": FS_AXLAB, "axes.titlesize": FS_PANEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.3, "lines.markersize": 4.0,
    }
    if USE_TEX:
        rc["text.latex.preamble"] = r"\usepackage{mathptmx}\usepackage{amsmath}"
    matplotlib.rcParams.update(rc)


FIGDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
LOGPATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure_numbers.log"))


def save(fig, stem):
    """同时出 PNG + PDF，统一 dpi。图内不放 'Fig. N'，编号交给 caption。"""
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, "%s.%s" % (stem, ext)), dpi=DPI)
    w, h = fig.get_size_inches()
    print("  [saved] %s  %.2f x %.2f in @ %d dpi" % (stem, w, h, DPI))
    return w, h


def logline(msg):
    """所有从数据现算的关键数字同时写进日志，供审计 diff 图注。"""
    print("  " + msg)
    with open(LOGPATH, "a") as f:
        f.write(msg + "\n")


def panel_letters(fig, axes, letters=None):
    """面板序号：字母上边缘贴齐 panel 上边框；水平置于**纵轴刻度值最左端之外**。

    ⚠️ 不能用固定的 axes 坐标偏移（如 -0.02）—— 刻度值宽度随数值位数变化，
    "0.12" 与 "100" 占的宽不同，固定偏移必然压字。**必须量渲染后的刻度值范围。**
    （skill 原文：是"纵轴刻度值最左端"，不是"纵轴标题最左端"。）
    """
    import string
    letters = letters or list(string.ascii_lowercase)
    fig.canvas.draw()                      # 先渲染，才量得到真实 bbox
    out = []
    for ax, L in zip(axes, letters):
        inv = ax.transAxes.inverted()
        xs = []
        for lab in ax.get_yticklabels():
            if not lab.get_text():
                continue
            bb = lab.get_window_extent(renderer=fig.canvas.get_renderer())
            xs.append(inv.transform((bb.x0, 0))[0])
        x = (min(xs) if xs else -0.02) - 0.015     # 刻度值最左端再外移一点
        t = ax.text(x, 1.0, bf("(%s)" % L), transform=ax.transAxes,
                    ha="right", va="top", fontsize=FS_PANEL)
        out.append(t)
    return out
