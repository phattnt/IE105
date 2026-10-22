"""Line -> function aggregation on saved line scores (CPU, no retraining): is the function level weak because of
the max pooling, or because the line scores carry no function-level information?

For every run with test_scores.jsonl (SecVulEval test; PrimeVul saved only function logits, so it is not covered)
the function score is recomputed with several aggregators and each is scored raw and after the length correction
of debias_length.py (residual to E[score | log chars]): ROC-AUC / AUPRC on all 2,606 test functions and the
vulnerable-vs-fixed pair accuracy. `random` and `length_only` baselines come from debias_length.evaluate_set.

    python aggregate_scores.py --data ../../data/secvul_lines --runs results/<run> ... [--out ...]
"""
import argparse
import json
import os

import numpy as np

from debias_length import evaluate_set, load_jsonl

AGG = {}


def agg(name):
    def deco(f):
        AGG[name] = f
        return f
    return deco


@agg("max (hiện tại)")
def _max(s, n):
    return s.max()


@agg("trung bình 3 dòng cao nhất")
def _top3(s, n):
    return np.sort(s)[-3:].mean()


@agg("logsumexp")
def _lse(s, n):
    m = s.max()
    return m + np.log(np.exp(s - m).sum())


@agg("trung bình mọi dòng")
def _mean(s, n):
    return s.mean()


@agg("số dòng > 0")
def _count0(s, n):
    return float((s > 0).sum())


@agg("tỉ lệ dòng > 0")
def _frac0(s, n):
    return float((s > 0).mean())


@agg("max − 0,5·log(số dòng)")
def _max_len(s, n):
    return s.max() - 0.5 * np.log(n)


@agg("max − log(số dòng)")
def _max_len1(s, n):
    return s.max() - np.log(n)


@agg("max − kỳ vọng max của N dòng (Gumbel)")
def _max_gumbel(s, n):
    # expected max of n iid draws from the run's own line-score distribution, approximated by the empirical quantile 1-1/n
    return s.max() - _max_gumbel.q(1 - 1.0 / max(n, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load_jsonl(os.path.join(a.data, "test.jsonl"))
    pairs = [(r["idx"], r["pair_idx"]) for r in rows if r["target"] == 1 and r.get("pair_idx") is not None]

    report = {}
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        scores = {}
        for r in load_jsonl(os.path.join(run, "test_scores.jsonl")):
            s = np.array([x for x in r["line_scores"] if x > -1e8], dtype=float)
            if len(s):
                scores[r["idx"]] = s
        allv = np.concatenate(list(scores.values()))
        _max_gumbel.q = lambda p, allv=allv: np.quantile(allv, p)
        report[name] = {}
        print(f"\n### {name} — SecVulEval test (n={len(scores)})")
        print("| cách gộp | ROC thô | ROC trừ độ dài | AUPRC trừ độ dài | cặp thô | cặp trừ độ dài | ρ độ dài |")
        print("|---|---|---|---|---|---|---|")
        for label, f in AGG.items():
            fn = {i: float(f(s, len(s))) for i, s in scores.items()}
            res, _ = evaluate_set(label, rows, fn, pairs)
            raw, deb = res["raw"], res["debias_same"]
            report[name][label] = {"raw": raw, "debias": deb}
            print(f"| {label} | {raw['roc_auc']:.3f} | {deb['roc_auc']:.3f} | {deb['auprc']:.3f} | {raw['pair_acc']:.1f} | {deb['pair_acc']:.1f} | {raw['spearman_len']:+.2f} |")
        lo = res["length_only"]
        print(f"| (chỉ độ dài, CV) | {lo['roc_auc']:.3f} | – | – | {lo['pair_acc']:.1f} | – | {lo['spearman_len']:+.2f} |")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
