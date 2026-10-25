"""PrimeVul pairs scored the way the literature scores them (CPU, from saved scores, no retraining).

Pair level (from <run>/primevul_test_scores.jsonl - the 1,128 pair functions are part of primevul_test):
  * P-C / P-V / P-B / P-R  (PrimeVul, Ding et al. 2024): vulnerable flagged & fixed cleared / both flagged / both cleared /
    reversed.  "flagged" = logit > threshold; thresholds reported: 0 (the model's own) and the subset median (flag rate 50%).
  * net = P-C - P-R, and the null model of arXiv 2609.32890: flag every function independently with the model's flag rate p
    -> expected P-C = P-R = p(1-p), net 0.  Bootstrap CI (pairs resampled) for net and for the ranking accuracy.
  * ranking accuracy (vulnerable scored higher, ties half) raw and length-debiased (cubic fit on log chars, label-free).
  * subsets as in 2609.32890: all pairs, length-matched (|chars_v - chars_f| / max < 5%), patch-adds-code (fixed longer),
    patch-removes-code (vulnerable longer).  A length rule (longer = fixed) is the baseline on every subset.
Line level (from <run>/primevul_pair_lines.jsonl written by score_pairs_lines.py), ANVIL-style (arXiv 2408.16028):
  only code lines are ranked (blank and comment lines dropped), vulnerable functions with >= 1 flaw line (lines the fix
  deletes/changes): Top-1/3/5/10, MFR (mean rank of the first flaw line, 1-based), N-MFR (that rank / #code lines),
  line ROC-AUC pooled over all code lines; bootstrap CI over functions; expected values of a random ranking.

    python pair_protocol.py --primevul ../../data/primevul --runs results/<run> ... [--out results/pair_protocol.json]
"""
import argparse
import json
import math
import os

import numpy as np
from sklearn.metrics import roc_auc_score

COMMENT = ("//", "/*", "*", "*/")
B = 2000


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def ci(vals, stat, rng, n=B):
    vals = np.asarray(vals)
    bs = [stat(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def debias(chars, scores):
    x = np.log(np.maximum(chars, 1))
    return scores - np.polyval(np.polyfit(x, scores, 3), x)


def pair_metrics(pv, pf, cv, cf, rng):
    """pv/pf: logits of vulnerable/fixed, cv/cf: chars. -> dict."""
    pv, pf, cv, cf = map(np.asarray, (pv, pf, cv, cf))
    n = len(pv)
    out = {"n": int(n)}
    rank = np.where(pv > pf, 1.0, np.where(pv == pf, 0.5, 0.0))
    out["rank_acc"] = float(rank.mean() * 100)
    out["rank_acc_ci"] = [v * 100 for v in ci(rank, np.mean, rng)]
    allc = np.concatenate([cv, cf]); alls = np.concatenate([pv, pf])
    d = debias(allc, alls)
    dv, df = d[:n], d[n:]
    out["rank_acc_debiased"] = float(np.where(dv > df, 1.0, np.where(dv == df, 0.5, 0.0)).mean() * 100)
    out["length_rule_longer_is_fixed"] = float(np.where(cf > cv, 1.0, np.where(cf == cv, 0.5, 0.0)).mean() * 100)
    for name, t in (("t0", 0.0), ("t_median", float(np.median(alls)))):
        fv, ff = pv > t, pf > t
        pc, pvv, pb, pr = (fv & ~ff), (fv & ff), (~fv & ~ff), (~fv & ff)
        p = float(np.concatenate([fv, ff]).mean())
        net = pc.astype(float) - pr.astype(float)
        out[name] = {"threshold": t, "flag_rate": p * 100,
                     "P-C": float(pc.mean() * 100), "P-V": float(pvv.mean() * 100), "P-B": float(pb.mean() * 100), "P-R": float(pr.mean() * 100),
                     "net": float(net.mean() * 100), "net_ci": [v * 100 for v in ci(net, np.mean, rng)],
                     "null_P-C": p * (1 - p) * 100, "null_net": 0.0}
    return out


def is_code(line):
    s = line.strip()
    return bool(s) and not s.startswith(COMMENT)


def expected_random_topk(n, f, k):
    if f == 0 or n == 0:
        return 0.0
    if k >= n:
        return 1.0
    return 1.0 - math.comb(n - f, k) / math.comb(n, k)


def line_metrics(rows, rng):
    recs, y_all, s_all = [], [], []
    for r in rows:
        if r["target"] != 1:
            continue
        lines = r["func"].split("\n")
        sc = np.asarray(r["line_scores"], float)
        code = [i for i in range(min(len(lines), len(sc))) if is_code(lines[i])]
        fl = [i for i in r["flaw_lines"] if i in set(code)]
        if not fl or not code:
            continue
        order = sorted(code, key=lambda i: (-sc[i], i))
        first = min(order.index(i) for i in fl) + 1          # 1-based rank of the first flaw line among code lines
        recs.append({"first": first, "n": len(code), "f": len(fl),
                     **{f"top{k}": float(first <= k) for k in (1, 3, 5, 10)},
                     **{f"rand_top{k}": expected_random_topk(len(code), len(fl), k) for k in (1, 3, 5, 10)}})
        y_all += [int(i in fl) for i in code]; s_all += [sc[i] for i in code]
    if not recs:
        return None
    out = {"n_functions": len(recs), "n_flaw_lines": int(sum(x["f"] for x in recs)),
           "mfr": float(np.mean([x["first"] for x in recs])), "n_mfr": float(np.mean([x["first"] / x["n"] for x in recs])),
           "rand_mfr": float(np.mean([(x["n"] + 1) / (x["f"] + 1) for x in recs])),
           "line_roc_auc": float(roc_auc_score(y_all, s_all))}
    for k in (1, 3, 5, 10):
        v = np.array([x[f"top{k}"] for x in recs])
        out[f"top{k}"] = float(v.mean() * 100)
        out[f"top{k}_ci"] = [u * 100 for u in ci(v, np.mean, rng)]
        out[f"rand_top{k}"] = float(np.mean([x[f"rand_top{k}"] for x in recs]) * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    paired = load_jsonl(os.path.join(a.primevul, "primevul_test_paired.jsonl"))
    pairs = []
    for i in range(0, len(paired) - 1, 2):
        x, y = paired[i], paired[i + 1]
        if x["commit_id"] == y["commit_id"] and {x["target"], y["target"]} == {0, 1}:
            v, f = (x, y) if x["target"] == 1 else (y, x)
            pairs.append((v["idx"], f["idx"], len(v["func"]), len(f["func"])))
    cv = np.array([p[2] for p in pairs]); cf = np.array([p[3] for p in pairs])
    subsets = {"all": np.ones(len(pairs), bool),
               "length_matched_5pct": np.abs(cv - cf) / np.maximum(cv, cf) < 0.05,
               "patch_adds_code": cf > cv, "patch_removes_code": cv > cf}
    print(f"pairs {len(pairs)}: " + ", ".join(f"{k} {int(m.sum())}" for k, m in subsets.items()))

    report = {}
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        rng = np.random.default_rng(0)
        logit = {r["idx"]: r["logit"] for r in load_jsonl(os.path.join(run, "primevul_test_scores.jsonl"))}
        have = np.array([p[0] in logit and p[1] in logit for p in pairs])     # a smoke run scores only some pairs
        pv = np.array([logit.get(p[0], 0.0) for p in pairs]); pf = np.array([logit.get(p[1], 0.0) for p in pairs])
        rep = {"pairs": {}, "pairs_scored": int(have.sum()), "pairs_total": len(pairs)}
        print(f"\n### {name} — cặp PrimeVul")
        if not have.all():
            print(f"WARNING: {name} has scores for {int(have.sum())}/{len(pairs)} pairs; the other pairs are left out of every row below")
        print("| tập cặp | n | xếp đúng cặp [CI] | sau trừ độ dài | luật độ dài | P-C / P-V / P-B / P-R (ngưỡng 0) | net [CI] | null P-C | P-C @50% gắn cờ | net @50% [CI] | null P-C @50% |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for sname, m in subsets.items():
            m = m & have
            if not m.any():
                continue
            r = pair_metrics(pv[m], pf[m], cv[m], cf[m], rng)
            rep["pairs"][sname] = r
            t0, tm = r["t0"], r["t_median"]
            print(f"| {sname} | {r['n']} | {r['rank_acc']:.1f} [{r['rank_acc_ci'][0]:.1f}; {r['rank_acc_ci'][1]:.1f}] | {r['rank_acc_debiased']:.1f} | "
                  f"{r['length_rule_longer_is_fixed']:.1f} | {t0['P-C']:.1f} / {t0['P-V']:.1f} / {t0['P-B']:.1f} / {t0['P-R']:.1f} | "
                  f"{t0['net']:+.1f} [{t0['net_ci'][0]:+.1f}; {t0['net_ci'][1]:+.1f}] | {t0['null_P-C']:.1f} | {tm['P-C']:.1f} | "
                  f"{tm['net']:+.1f} [{tm['net_ci'][0]:+.1f}; {tm['net_ci'][1]:+.1f}] | {tm['null_P-C']:.1f} |")
        lp = os.path.join(run, "primevul_pair_lines.jsonl")
        if os.path.exists(lp):
            lm = line_metrics(load_jsonl(lp), rng)
            rep["lines_anvil_style"] = lm
            print(f"\n{name} — dòng PrimeVul kiểu ANVIL (chỉ dòng code, n={lm['n_functions']} hàm, {lm['n_flaw_lines']} dòng lỗi)")
            print("| | Top-1 | Top-3 | Top-5 | Top-10 | MFR | N-MFR | ROC-AUC dòng |")
            print("|---|---|---|---|---|---|---|---|")
            print(f"| mô hình | {lm['top1']:.1f} [{lm['top1_ci'][0]:.1f}; {lm['top1_ci'][1]:.1f}] | {lm['top3']:.1f} [{lm['top3_ci'][0]:.1f}; {lm['top3_ci'][1]:.1f}] | "
                  f"{lm['top5']:.1f} [{lm['top5_ci'][0]:.1f}; {lm['top5_ci'][1]:.1f}] | {lm['top10']:.1f} | {lm['mfr']:.1f} | {lm['n_mfr']:.2f} | {lm['line_roc_auc']:.3f} |")
            print(f"| ngẫu nhiên | {lm['rand_top1']:.1f} | {lm['rand_top3']:.1f} | {lm['rand_top5']:.1f} | {lm['rand_top10']:.1f} | {lm['rand_mfr']:.1f} | – | 0.500 |")
        report[name] = rep
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
