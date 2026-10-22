"""Length-debiased function scoring for exp5 runs (CPU, no torch, no retraining).

The function score of train_lines.py is the max line logit. On SecVulEval that score tracks function length
(shorter = vulnerable, because the fixed version is longer in most pairs); on PrimeVul the vulnerable functions
are the longer ones, so the same habit ranks them backwards (full-test ROC-AUC below 0.5).

This script removes the part of the score that is explained by length alone:
    residual = score - E[score | length]
E[score | length] is a cubic polynomial in log(characters), fitted WITHOUT labels (so fitting it on the scored
set itself leaks nothing). Two variants are reported:
    same   fit on the set being scored
    xfer   fit on the SecVulEval test scores, applied to PrimeVul (cross-dataset: the habit as learned)
Baselines on every set: random (AUPRC = prevalence, ROC 0.5) and length-only (logistic regression on log chars
and log lines, 5-fold cross-validation grouped by project).

    python debias_length.py --data ../../data/secvul_lines --primevul ../../data/primevul \
        --runs results/<run> ... [--out ../../data/debias_length.json]
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def rk(x):
    return np.argsort(np.argsort(x))


def spearman(a, b):
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


def fit_length_curve(length, score, deg=3):
    """E[score | length] as a polynomial in log length; returns a callable."""
    x = np.log(np.maximum(length, 1))
    c = np.polyfit(x, score, deg)
    lo, hi = x.min(), x.max()
    return lambda L: np.polyval(c, np.clip(np.log(np.maximum(L, 1)), lo, hi))


def length_only_cv(y, feats, groups, seed=0):
    """Logistic regression on length features, out-of-fold predictions grouped by project."""
    p = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(feats, y, groups):
        m = LogisticRegression(max_iter=1000).fit(feats[tr], y[tr])
        p[te] = m.predict_proba(feats[te])[:, 1]
    return p


def function_metrics(y, s, length):
    return {"auprc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "spearman_len": spearman(s, length)}


def pair_acc(pairs, s_by_idx):
    """pairs: list of (vul_idx, fix_idx). Ties count half."""
    w = t = 0
    for v, f in pairs:
        if v not in s_by_idx or f not in s_by_idx:
            continue
        a, b = s_by_idx[v], s_by_idx[f]
        t += 1
        w += 1.0 if a > b else 0.5 if a == b else 0.0
    return w / t * 100 if t else None, t


def evaluate_set(name, rows, score_by_idx, pairs, curve_xfer=None):
    """rows: dicts with idx, target, project, func. score_by_idx: raw function score."""
    rows = [r for r in rows if r["idx"] in score_by_idx]
    idx = np.array([r["idx"] for r in rows])
    y = np.array([int(r["target"]) for r in rows])
    chars = np.array([len(r["func"]) for r in rows], dtype=float)
    lines = np.array([r["func"].count("\n") + 1 for r in rows], dtype=float)
    proj = np.array([r["project"] for r in rows])
    raw = np.array([score_by_idx[i] for i in idx], dtype=float)

    out = {"n": int(len(y)), "vulnerable": int(y.sum()), "prevalence": float(y.mean())}
    out["random"] = {"auprc": float(y.mean()), "roc_auc": 0.5, "pair_acc": 50.0}

    feats = np.c_[np.log(chars), np.log(lines)]
    p_len = length_only_cv(y, feats, proj)
    out["length_only"] = function_metrics(y, p_len, chars)
    out["length_only"]["pair_acc"], _ = pair_acc(pairs, dict(zip(idx.tolist(), p_len.tolist())))
    # the explicit rule the datasets reward: shorter member is vulnerable
    out["rule_shorter_is_vul"] = {"pair_acc": pair_acc(pairs, dict(zip(idx.tolist(), (-chars).tolist())))[0]}

    def report(s):
        m = function_metrics(y, s, chars)
        m["pair_acc"], m["pairs_n"] = pair_acc(pairs, dict(zip(idx.tolist(), s.tolist())))
        return m

    out["raw"] = report(raw)
    curve_same = fit_length_curve(chars, raw)
    out["debias_same"] = report(raw - curve_same(chars))
    if curve_xfer is not None:
        out["debias_xfer"] = report(raw - curve_xfer(chars))
    return out, curve_same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="secvul_lines dir (test.jsonl)")
    ap.add_argument("--primevul", required=True, help="dir with primevul_test.jsonl and primevul_test_paired.jsonl")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sv_rows = load_jsonl(os.path.join(a.data, "test.jsonl"))
    sv_pairs = [(r["idx"], r["pair_idx"]) for r in sv_rows if r["target"] == 1 and r.get("pair_idx") is not None]
    pv_rows = load_jsonl(os.path.join(a.primevul, "primevul_test.jsonl"))
    paired = load_jsonl(os.path.join(a.primevul, "primevul_test_paired.jsonl"))
    pv_pairs = []
    for i in range(0, len(paired) - 1, 2):
        x, z = paired[i], paired[i + 1]
        if x["commit_id"] == z["commit_id"] and {x["target"], z["target"]} == {0, 1}:
            v, f = (x, z) if x["target"] == 1 else (z, x)
            pv_pairs.append((v["idx"], f["idx"]))

    results = {}
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        sv_scores = {}
        for r in load_jsonl(os.path.join(run, "test_scores.jsonl")):
            s = [x for x in r["line_scores"] if x > -1e8]
            sv_scores[r["idx"]] = max(s) if s else -1e9
        sv_res, curve = evaluate_set("secvuleval", sv_rows, sv_scores, sv_pairs)
        res = {"secvuleval_test": sv_res}
        pv_file = os.path.join(run, "primevul_test_scores.jsonl")
        if os.path.exists(pv_file):
            pv_scores = {r["idx"]: r["logit"] for r in load_jsonl(pv_file)}
            pv_res, _ = evaluate_set("primevul", pv_rows, pv_scores, pv_pairs, curve_xfer=curve)
            res["primevul_full_test"] = pv_res
        results[name] = res

    # ---- print
    def row(label, m):
        pa = m.get("pair_acc")
        return (f"| {label:28s} | {m.get('auprc', float('nan')):.3f} | {m.get('roc_auc', float('nan')):.3f} | "
                f"{'-' if pa is None else f'{pa:.1f}'} | {m.get('spearman_len', float('nan')):+.2f} |")

    for name, res in results.items():
        for setname, r in res.items():
            print(f"\n### {name} — {setname} (n={r['n']}, lỗi={r['vulnerable']}, tỉ lệ={r['prevalence']:.3f})")
            print("| cách chấm | AUPRC | ROC-AUC | cặp % | Spearman độ dài |")
            print("|---|---|---|---|---|")
            print(row("ngẫu nhiên", r["random"]))
            print(row("chỉ độ dài (CV theo project)", r["length_only"]))
            print(row("quy tắc ngắn hơn = lỗi", {"pair_acc": r["rule_shorter_is_vul"]["pair_acc"]}))
            print(row("mô hình, điểm thô", r["raw"]))
            print(row("mô hình, trừ độ dài (same)", r["debias_same"]))
            if "debias_xfer" in r:
                print(row("mô hình, trừ độ dài (xfer)", r["debias_xfer"]))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
