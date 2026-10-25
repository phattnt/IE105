"""Phase 0: commit-level leakage between training data and the test sets, and whether it moves any reported number.

train_lines.py drops SecVulEval train functions whose text equals a PrimeVul test function (and their pair partners);
functions of the same commit with a different text stay in. This script counts what is left:
  secvul_train -> secvul_test       commits shared by the SecVulEval train and test splits
  secvul_train -> primevul_test     kept SecVulEval train functions from a commit of the PrimeVul test pairs
  primevul_train -> primevul_test   PrimeVul's own split (for the PrimeVul-trained pv1)
and rescores, without retraining, each run on the test units that share no commit with its training data:
  SecVulEval test   line Top-1/5/10, pair and same-text (bootstrap_ci.per_function), from test_scores.jsonl
  PrimeVul pairs    pair ranking accuracy from primevul_test_scores.jsonl

    python leak_audit.py --secvul ../../data/secvul_lines --primevul ../../data/primevul \
        --runs results/slide_..._s0_conv results/slide_..._s0_base2 --out results/leak_audit.json
A run trained on PrimeVul (args.data has "primevul" in it) is checked against primevul_train instead.
"""
import argparse
import json
import os

import numpy as np

from bootstrap_ci import metrics, per_function


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def kept_train(rows):
    """the rows train_lines.py actually trains on, as far as the leak filter goes"""
    leak = {r["idx"] for r in rows if r.get("in_primevul_test")}
    leak |= {r["pair_idx"] for r in rows if r["idx"] in leak and r.get("pair_idx") is not None}
    return [r for r in rows if r["idx"] not in leak]


def pv_pairs(paired):
    out = []
    for k in range(0, len(paired) - 1, 2):
        x, y = paired[k], paired[k + 1]
        if x["commit_id"] == y["commit_id"] and {x["target"], y["target"]} == {0, 1}:
            v, f = (x, y) if x["target"] == 1 else (y, x)
            out.append((v["idx"], f["idx"], v["commit_id"], v.get("project", "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secvul", required=True)
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sv = {s: load_jsonl(os.path.join(a.secvul, f"{s}.jsonl")) for s in ("train", "valid", "test")}
    sv_train = kept_train(sv["train"])
    pv_test_pairs = pv_pairs(load_jsonl(os.path.join(a.primevul, "primevul_test_paired.jsonl")))
    pv_train_c = {r["commit_id"] for r in load_jsonl(os.path.join(a.primevul, "primevul_train_paired.jsonl"))}
    pv_train_c |= {r["commit_id"] for r in load_jsonl(os.path.join(a.primevul, "primevul_valid_paired.jsonl"))}
    train_c = {"secvul": {r["commit_id"] for r in sv_train} | {r["commit_id"] for r in sv["valid"]}, "primevul": pv_train_c}
    rep = {"overlap": {}}

    sv_test_c = {r["commit_id"] for r in sv["test"]}
    sh = train_c["secvul"] & sv_test_c
    rep["overlap"]["secvul_train_valid->secvul_test"] = {
        "shared_commits": len(sh), "test_commits": len(sv_test_c),
        "test_functions_in_shared_commits": sum(r["commit_id"] in sh for r in sv["test"]),
        "test_vulnerable_in_shared_commits": sum(r["commit_id"] in sh and r["target"] == 1 for r in sv["test"]),
        "projects": sorted({r["project"] for r in sv["test"] if r["commit_id"] in sh})[:20]}
    pv_test_c = {p[2] for p in pv_test_pairs}
    for name, c in (("secvul_train_valid->primevul_test_pairs", train_c["secvul"]),
                    ("primevul_train_valid->primevul_test_pairs", train_c["primevul"])):
        sh = c & pv_test_c
        rep["overlap"][name] = {"shared_commits": len(sh), "test_pairs": len(pv_test_pairs),
                                "test_pairs_in_shared_commits": sum(p[2] in sh for p in pv_test_pairs),
                                "projects": sorted({p[3] for p in pv_test_pairs if p[2] in sh})}
    rep["overlap"]["secvul_train_kept_functions_from_primevul_test_commits"] = sum(r["commit_id"] in pv_test_c for r in sv_train)
    print(json.dumps(rep["overlap"], indent=1))

    sv_by = {r["idx"]: r for r in sv["test"]}
    for rd in a.runs:
        name = os.path.basename(os.path.normpath(rd))
        args = json.load(open(os.path.join(rd, "results.json"))).get("args", {})
        src = "primevul" if "primevul" in str(args.get("data", "")) else "secvul"
        r = {"trained_on": src}
        tp = os.path.join(rd, "test_scores.jsonl")
        if src == "secvul" and os.path.exists(tp):
            sc = {}
            for x in load_jsonl(tp):
                sc[x["idx"]] = x["line_scores"]
            units = per_function(sv["test"], sc)
            clean = {i: u for i, u in units.items() if sv_by[i]["commit_id"] not in train_c[src]}
            r["secvul_test_all"] = metrics(list(units.values()))
            r["secvul_test_clean"] = metrics(list(clean.values()))
            r["secvul_test_clean"]["n_functions"] = len(clean)
        pp = os.path.join(rd, "primevul_test_scores.jsonl")
        if os.path.exists(pp):
            lg = {x["idx"]: x["logit"] for x in load_jsonl(pp)}
            res = {}
            for sub, keep in (("all", lambda p: True), ("clean", lambda p: p[2] not in train_c[src])):
                d = np.array([lg[p[0]] - lg[p[1]] for p in pv_test_pairs if p[0] in lg and p[1] in lg and keep(p)])
                res[sub] = {"n": len(d), "pair_acc_ties_half": float(100 * (np.mean(d > 0) + 0.5 * np.mean(d == 0)))}
            r["primevul_pairs"] = res
        rep[name] = r
        print(name, json.dumps(r))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
