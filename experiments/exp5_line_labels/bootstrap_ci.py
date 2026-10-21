"""Bootstrap confidence intervals for the SecVulEval test metrics of one or more exp5 runs (CPU, no torch).

Resamples PROJECTS (the unit of the test split) with replacement, so the interval reflects how much the
numbers depend on which unseen projects happen to be in the test set. For two runs scored on the same
functions it also reports the paired difference (run B - run A) and its interval.

    python bootstrap_ci.py --data <secvul_lines dir> --runs results/A results/B [--n 2000]
Each run dir needs test_scores.jsonl (idx, target, pair_idx, flaw_lines, line_scores).
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np


def norm(s):
    return re.sub(r"\s+", "", s)


def per_function(rows, scores):
    """Per-unit statistics that the metrics are built from (so resampling is cheap)."""
    by = {r["idx"]: r for r in rows}
    out = {}
    for idx, sc in scores.items():
        r = by[idx]
        d = {"project": r["project"]}
        if r["target"] == 1 and r["flaw_lines"]:
            fl = [l for l in r["flaw_lines"] if l < len(sc)]
            order = np.argsort(-np.asarray(sc), kind="stable")
            rank = {int(l): k for k, l in enumerate(order)}
            first = min(rank[l] for l in fl)
            d.update(top1=first < 1, top5=first < 5, top10=first < 10)
        if r["target"] == 1 and r["pair_idx"] in scores:
            fv, ff = max(sc), max(scores[r["pair_idx"]])
            if abs(fv - ff) > 1e-4:
                d["pair"] = fv > ff
            # same-text flaw lines
            f = by[r["pair_idx"]]
            ftext = defaultdict(list)
            for j, t in enumerate(f["func"].split("\n")):
                if norm(t):
                    ftext[norm(t)].append(j)
            vl = r["func"].split("\n")
            wins = []
            for l in r["flaw_lines"]:
                k = norm(vl[l]) if l < len(vl) else ""
                if len(k) >= 6 and k in ftext:
                    j = min(ftext[k], key=lambda x: abs(x - l))
                    a, b = sc[l], scores[r["pair_idx"]][j]
                    wins.append(1.0 if a > b + 1e-4 else (0.5 if abs(a - b) <= 1e-4 else 0.0))
            if wins:
                d["same_text"] = wins
        out[idx] = d
    return out


def metrics(units):
    m = {}
    for k in ("top1", "top5", "top10", "pair"):
        v = [u[k] for u in units if k in u]
        m[k] = 100 * float(np.mean(v)) if v else float("nan")
    st = [w for u in units for w in u.get("same_text", [])]
    m["same_text"] = 100 * float(np.mean(st)) if st else float("nan")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{a.data}/test.jsonl")]
    runs = {}
    for rd in a.runs:
        sc = {json.loads(l)["idx"]: json.loads(l)["line_scores"] for l in open(f"{rd}/test_scores.jsonl")}
        runs[os.path.basename(rd.rstrip("/"))] = per_function(rows, sc)
    projects = sorted({r["project"] for r in rows})
    rng = np.random.default_rng(0)
    res = {}
    names = list(runs)
    boot = {n: [] for n in names}
    for _ in range(a.n):
        pick = rng.choice(projects, size=len(projects), replace=True)
        cnt = defaultdict(int)
        for p in pick:
            cnt[p] += 1
        for n in names:
            units = [u for u in runs[n].values() for _ in range(cnt[u["project"]])]
            boot[n].append(metrics(units))
    for n in names:
        point = metrics(list(runs[n].values()))
        res[n] = {k: {"point": point[k], "ci95": [float(np.nanpercentile([b[k] for b in boot[n]], 2.5)),
                                                  float(np.nanpercentile([b[k] for b in boot[n]], 97.5))]} for k in point}
    if len(names) >= 2:
        base = names[0]
        for n in names[1:]:
            res[f"{n} - {base}"] = {k: {"point": res[n][k]["point"] - res[base][k]["point"],
                                        "ci95": [float(np.nanpercentile([bn[k] - bb[k] for bn, bb in zip(boot[n], boot[base])], q)) for q in (2.5, 97.5)]}
                                    for k in res[base]}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
