"""目标期刊语料统计 —— 论文 §4.7 立论所依赖的数字，必须可复现。

背景：选题阶段做过一次"IJDNS 的 ML 论文主报什么指标"的统计，但那是在项目仓库
之外算的。按 KB 铁律，**不在真源里的数字不该出现在任何稿子里**，故在此重算并
写进 `RESULTS_GROUND_TRUTH.md`。

语料快照：`corpus/ijdns_corpus_dedup_2026-08-30.json`
  来源：Crossref `journals/2561-8156/works`（1,040 条）+ DOAJ `search/articles/issn:2561-8156`
       （942 条摘要），按 DOI 合并、按标题去重后 983 篇。拉取日期 2026-08-30。
  ⚠️ 快照固化于仓库内，不重新联网拉取 —— 期刊持续出版，联网重拉会得到不同的数字。
     引用时须注明"截至 2026-08-30"。
"""
from __future__ import annotations
import os, sys, json, re, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.abspath(os.path.join(HERE, "..", "corpus", "ijdns_corpus_dedup_2026-08-30.json"))
SNAPSHOT_DATE = "2026-08-30"

# 形态判定：复现选题期的**完整互斥优先级规则**（顺序即优先级，先命中者胜）。
# ⚠️ 初版只排除了问卷类，得到 107 篇；补全优先级后为 100 篇，与 KB `journals/ijdns.md`
#    记录的"ML 预测类 100 篇"一致。**分类规则必须整套复现，不能只抄其中一条。**
SHAPE_RULES = [
    ("survey-SEM", r"\b(pls-sem|pls sem|structural equation|smartpls|smart-pls|amos|"
                   r"questionnaire|respondents|survey (was|were|of)|"
                   r"sample of \d+ (respondents|participants|students|employees|customers|users)|"
                   r"convenience sampling|purposive sampling|likert)\b"),
    ("MCDM", r"\b(ahp|anp|topsis|dematel|vikor|electre|promethee|best[- ]worst method|"
             r"\bbwm\b|fuzzy ahp|multi[- ]criteria decision|copras|waspas|moora|swara|marcos|edas)\b"),
    ("bibliometric/review", r"\b(bibliometric|scientometric|vosviewer|biblioshiny|systematic (literature )?review|"
                            r"prisma|meta[- ]analysis|co[- ]citation|co[- ]word|scopus database|web of science database)\b"),
    ("DEA/efficiency", r"\b(data envelopment analysis|\bdea\b|malmquist|stochastic frontier|efficiency score)\b"),
    ("ML-predict", r"\b(machine learning|deep learning|neural network|random forest|xgboost|lstm|cnn\b|svm\b|"
                   r"support vector|gradient boosting|classifier|classification accuracy|f1[- ]score|"
                   r"training set|test set|convolutional|transformer|bert|naive bayes|k-means|clustering algorithm|"
                   r"predictive model|prediction model|forecast)\b"),
]

METRICS = [
    ("accuracy",            r"\baccurac(y|ies)\b",                              "prevalence-sensitive"),
    ("F1",                  r"\bf1[- ]?(score|measure)?\b|\bf-?measure\b",       "prevalence-sensitive"),
    ("precision",           r"\bprecision\b",                                    "prevalence-sensitive"),
    ("recall / sensitivity", r"\brecall\b|\bsensitivit(y|ies)\b",               "prevalence-insensitive"),
    ("specificity",         r"\bspecificit(y|ies)\b",                            "prevalence-insensitive"),
    ("ROC-AUC",             r"\bauc\b|\broc\b|area under the (receiver|curve)",  "prevalence-insensitive"),
    ("AUPRC",               r"\bauprc\b|\bpr-auc\b|average precision|precision-recall curve", "prevalence-SENSITIVE"),
    ("balanced accuracy",   r"balanced accurac",                                 "prevalence-insensitive"),
    ("Brier score",         r"\bbrier\b",                                        "prevalence-insensitive"),
    ("MCC",                 r"\bmcc\b|matthews",                                 "prevalence-insensitive"),
]
SENSITIVE = r"\baccurac(y|ies)\b|\bf1[- ]?(score|measure)?\b|\bf-?measure\b|\bprecision\b"
AUC_ONLY = r"\bauc\b|\broc\b"


def _txt(r):
    return (r.get("title", "") + " " + r.get("abstract", "") + " " + " ".join(r.get("keywords") or [])).lower()


def compute():
    rows = json.load(open(SNAPSHOT))
    sha = hashlib.sha256(open(SNAPSHOT, "rb").read()).hexdigest()
    def shape(r):
        t = _txt(r)
        for name, pat in SHAPE_RULES:
            if re.search(pat, t):
                return name
        return "other"
    ml = [r for r in rows if shape(r) == "ML-predict"]
    out = dict(snapshot_sha256=sha, snapshot_date=SNAPSHOT_DATE,
               n_corpus=len(rows), n_ml=len(ml), metrics=[])
    for name, pat, kind in METRICS:
        n = sum(1 for r in ml if re.search(pat, _txt(r)))
        out["metrics"].append(dict(metric=name, kind=kind, n=n,
                                   pct=100.0 * n / len(ml) if ml else 0.0))
    out["n_any_sensitive"] = sum(1 for r in ml if re.search(SENSITIVE, _txt(r)))
    out["n_auc_only"] = sum(1 for r in ml
                            if re.search(AUC_ONLY, _txt(r)) and not re.search(SENSITIVE, _txt(r)))
    out["pct_any_sensitive"] = 100.0 * out["n_any_sensitive"] / len(ml) if ml else 0.0
    out["pct_auc_only"] = 100.0 * out["n_auc_only"] / len(ml) if ml else 0.0
    return out


if __name__ == "__main__":
    r = compute()
    print("语料 %d 篇（快照 %s），ML 预测类 %d 篇" % (r["n_corpus"], r["snapshot_date"], r["n_ml"]))
    for m in r["metrics"]:
        print("  %-22s %-24s %3d  %5.1f%%" % (m["metric"], m["kind"], m["n"], m["pct"]))
    print("  提到至少一个对流行率敏感的指标：%d (%.1f%%)" % (r["n_any_sensitive"], r["pct_any_sensitive"]))
    print("  只提 AUC/ROC 而不提上述三者：  %d (%.1f%%)" % (r["n_auc_only"], r["pct_auc_only"]))
