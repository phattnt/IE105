"""Score-level ensembles of exp5 runs (CPU, no retraining): does combining models change the function level?

Line scores of each run are z-scored (one mean/std per run over all test lines) and averaged; the function
score is the max of the averaged line scores, exactly as for a single run. On PrimeVul only function logits
were saved, so the ensemble there is the average of z-scored logits (no line metrics). Every combination is
reported raw and after the length correction of debias_length.py (residual to E[score | log chars]), with a
project-level bootstrap of ensemble minus best single member on Top-10 and on the debiased pair accuracy.

    python ensemble_scores.py --data ../../data/secvul_lines --primevul ../../data/primevul \
        --runs results/<run> ... --combos "name=run1+run2" ... [--out ...]
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from debias_length import evaluate_set, fit_length_curve, load_jsonl


def z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def load_run(run):
    sv = {r["idx"]: r for r in load_jsonl(os.path.join(run, "test_scores.jsonl"))}
    pv = {r["idx"]: r["logit"] for r in load_jsonl(os.path.join(run, "primevul_test_scores.jsonl"))}
    return sv, pv


def line_scores_z(sv):
    """z-score all finite line logits of one run with a single mean/std."""
    allv = np.concatenate([[x for x in r["line_scores"] if x > -1e8] for r in sv.values()])
    mu, sd = allv.mean(), allv.std() + 1e-9
    return {i: [(x - mu) / sd if x > -1e8 else -1e9 for x in r["line_scores"]] for i, r in sv.items()}


def combine_lines(members):
    idxs = set.intersection(*(set(m) for m in members))
    out = {}
    for i in idxs:
        arrs = [np.asarray(m[i]) for m in members]
        n = min(len(a) for a in arrs)
        out[i] = np.mean([a[:n] for a in arrs], axis=0).tolist()
    return out


def line_units(rows, lines):
    """Per vulnerable function with flaw lines: project and top-k hits."""
    by = {r["idx"]: r for r in rows}
    units = {}
    for i, sc in lines.items():
        r = by[i]
        if r["target"] != 1 or not r["flaw_lines"]:
            continue
        fl = [l for l in r["flaw_lines"] if l < len(sc)]
        if not fl:
            continue
        order = np.argsort(-np.asarray(sc), kind="stable")
        rank = {int(l): k for k, l in enumerate(order)}
        first = min(rank[l] for l in fl)
        units[i] = {"project": r["project"], "top1": first < 1, "top5": first < 5, "top10": first < 10}
    return units


def pair_units(rows, pairs, score_by_idx):
    by = {r["idx"]: r for r in rows}
    out = {}
    for v, f in pairs:
        if v in score_by_idx and f in score_by_idx:
            a, b = score_by_idx[v], score_by_idx[f]
            out[v] = {"project": by[v]["project"], "pair": 1.0 if a > b else 0.5 if a == b else 0.0}
    return out


def boot_diff(ua, ub, key, n=2000, seed=0):
    """Project-level bootstrap of mean(ua[key]) - mean(ub[key]) over the units both have."""
    common = sorted(set(ua) & set(ub))
    proj = defaultdict(list)
    for i in common:
        proj[ua[i]["project"]].append(i)
    names = list(proj)
    rng = np.random.default_rng(seed)
    A = {i: float(ua[i][key]) for i in common}
    B = {i: float(ub[i][key]) for i in common}
    point = 100 * (np.mean([A[i] for i in common]) - np.mean([B[i] for i in common]))
    diffs = []
    for _ in range(n):
        pick = rng.choice(len(names), len(names), replace=True)
        ids = [i for k in pick for i in proj[names[k]]]
        diffs.append(100 * (np.mean([A[i] for i in ids]) - np.mean([B[i] for i in ids])))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)], "n": len(common)}


def debiased(rows, score_by_idx):
    rows = [r for r in rows if r["idx"] in score_by_idx]
    chars = np.array([len(r["func"]) for r in rows], dtype=float)
    raw = np.array([score_by_idx[r["idx"]] for r in rows])
    curve = fit_length_curve(chars, raw)
    res = raw - curve(chars)
    return {r["idx"]: float(s) for r, s in zip(rows, res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="results/<run> dirs; name = basename")
    ap.add_argument("--combos", nargs="+", required=True, help="label=runA+runB (basenames or unique substrings)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sv_rows = load_jsonl(os.path.join(a.data, "test.jsonl"))
    sv_pairs = [(r["idx"], r["pair_idx"]) for r in sv_rows if r["target"] == 1 and r.get("pair_idx") is not None]
    pv_rows = load_jsonl(os.path.join(a.primevul, "primevul_test.jsonl"))
    paired = load_jsonl(os.path.join(a.primevul, "primevul_test_paired.jsonl"))
    pv_pairs = []
    for i in range(0, len(paired) - 1, 2):
        x, y = paired[i], paired[i + 1]
        if x["commit_id"] == y["commit_id"] and {x["target"], y["target"]} == {0, 1}:
            v, f = (x, y) if x["target"] == 1 else (y, x)
            pv_pairs.append((v["idx"], f["idx"]))

    runs = {}
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        sv, pv = load_run(run)
        runs[name] = {"lines": line_scores_z(sv), "pv": pv}

    def resolve(s):
        hit = [n for n in runs if n == s] or [n for n in runs if s in n]
        assert len(hit) == 1, f"{s} -> {hit}"
        return hit[0]

    systems = {n: ([n], r["lines"], r["pv"]) for n, r in runs.items()}
    for c in a.combos:
        label, members = c.split("=")
        members = [resolve(m) for m in members.split("+")]
        lines = combine_lines([runs[m]["lines"] for m in members])
        idxs = sorted(set.intersection(*(set(runs[m]["pv"]) for m in members)))
        zs = [dict(zip(idxs, z([runs[m]["pv"][i] for i in idxs]))) for m in members]
        pv = {i: float(np.mean([zz[i] for zz in zs])) for i in idxs}
        systems[label] = (members, lines, pv)

    report, units = {}, {}
    for label, (members, lines, pv) in systems.items():
        fn = {i: max([x for x in sc if x > -1e8] or [-1e9]) for i, sc in lines.items()}
        sv_res, _ = evaluate_set("sv", sv_rows, fn, sv_pairs)
        pv_res, _ = evaluate_set("pv", pv_rows, pv, pv_pairs)
        lu = line_units(sv_rows, lines)
        pu_sv = pair_units(sv_rows, sv_pairs, debiased(sv_rows, fn))
        pu_pv = pair_units(pv_rows, pv_pairs, debiased(pv_rows, pv))
        units[label] = (lu, pu_sv, pu_pv)
        report[label] = {
            "members": members,
            "sv_top1": 100 * np.mean([u["top1"] for u in lu.values()]),
            "sv_top5": 100 * np.mean([u["top5"] for u in lu.values()]),
            "sv_top10": 100 * np.mean([u["top10"] for u in lu.values()]),
            "sv_pair_raw": sv_res["raw"]["pair_acc"], "sv_pair_debias": sv_res["debias_same"]["pair_acc"],
            "sv_roc_raw": sv_res["raw"]["roc_auc"], "sv_roc_debias": sv_res["debias_same"]["roc_auc"],
            "sv_spearman_len": sv_res["raw"]["spearman_len"],
            "pv_pair_raw": pv_res["raw"]["pair_acc"], "pv_pair_debias": pv_res["debias_same"]["pair_acc"],
            "pv_roc_raw": pv_res["raw"]["roc_auc"], "pv_roc_debias": pv_res["debias_same"]["roc_auc"],
            "pv_auprc_debias": pv_res["debias_same"]["auprc"],
            "pv_spearman_len": pv_res["raw"]["spearman_len"],
        }

    # bootstrap: ensemble minus its best member on each metric
    for label, (members, _, _) in systems.items():
        if len(members) < 2:
            continue
        r = report[label]
        r["boot"] = {}
        for key, ui, metric in (("top10", 0, "sv_top10"), ("pair", 1, "sv_pair_debias"), ("pair", 2, "pv_pair_debias")):
            best = max(members, key=lambda m: report[m][metric])
            r["boot"][metric] = {"vs": best, **boot_diff(units[label][ui], units[best][ui], key)}

    cols = ["sv_top1", "sv_top5", "sv_top10", "sv_pair_raw", "sv_pair_debias", "sv_roc_debias", "sv_spearman_len",
            "pv_pair_raw", "pv_pair_debias", "pv_roc_raw", "pv_roc_debias", "pv_spearman_len"]
    short = {"sv_top1": "T1", "sv_top5": "T5", "sv_top10": "T10", "sv_pair_raw": "cặp thô", "sv_pair_debias": "cặp trừ ĐD",
             "sv_roc_debias": "ROC trừ ĐD", "sv_spearman_len": "ρ dài", "pv_pair_raw": "PV cặp thô",
             "pv_pair_debias": "PV cặp trừ ĐD", "pv_roc_raw": "PV ROC thô", "pv_roc_debias": "PV ROC trừ ĐD",
             "pv_spearman_len": "PV ρ dài"}
    print("| hệ thống | " + " | ".join(short[c] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, r in report.items():
        cells = [f"{r[c]:.2f}" if abs(r[c]) < 2 else f"{r[c]:.1f}" for c in cols]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\nBootstrap theo project (ghép − thành viên tốt nhất), 2000 lần:")
    for label, r in report.items():
        for metric, b in r.get("boot", {}).items():
            print(f"  {label:10s} {metric:15s} vs {b['vs']:45s} {b['point']:+5.1f} [{b['ci95'][0]:+5.1f}; {b['ci95'][1]:+5.1f}]  n={b['n']}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
