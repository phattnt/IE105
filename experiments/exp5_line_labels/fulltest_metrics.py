# Function-level metrics on the WHOLE PrimeVul test set (25,911 functions, 695 vulnerable), the way the PrimeVul paper
# reports them: accuracy, precision, recall, F1 (vulnerable = positive class), plus the benign flag rate (false-positive
# rate) and balanced accuracy.
# The LLM scores a stratified sample (see build_llm_inputs.py): P = the test-pair functions (all), V = the other
# vulnerable functions (all), B = a uniform sample of the other benign functions with weight N_B / n_B. Counts are
# weighted, so every number estimates the whole test set; CIs resample within each stratum (B resamples).
# The line model has scores for every function, so its numbers are exact; it is also shown on the same sample as a check.
# usage: python fulltest_metrics.py --manifest inputs/fulltest_manifest.json --line ../results/eval_pv1_best
#          --llm name=<pair-run scores.jsonl>,<rest-run scores.jsonl> [...] --out <json>
import argparse, json
import numpy as np

B = 2000


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def metrics(y, flag, w):
    tp = float(np.sum(w * (y == 1) * flag)); fn = float(np.sum(w * (y == 1) * ~flag))
    fp = float(np.sum(w * (y == 0) * flag)); tn = float(np.sum(w * (y == 0) * ~flag))
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"accuracy": (tp + tn) / (tp + tn + fp + fn) * 100, "precision": prec * 100, "recall": rec * 100, "f1": f1 * 100,
            "benign_flag_rate": fpr * 100, "balanced_accuracy": (rec + 1 - fpr) / 2 * 100,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate(y, s, w, strata, thr, rng):
    flag = s > thr
    out = metrics(y, flag, w)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    bs = []
    for _ in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        bs.append(metrics(y[idx], flag[idx], w[idx]))
    for k in ("accuracy", "precision", "recall", "f1", "benign_flag_rate", "balanced_accuracy"):
        v = [b[k] for b in bs]; out[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def row(name, thr_name, m):
    f = lambda k: f"{m[k]:.1f} [{m[k + '_ci'][0]:.1f}; {m[k + '_ci'][1]:.1f}]"
    return (f"| {name} | {thr_name} | {f('accuracy')} | {f('precision')} | {f('recall')} | {f('f1')} | "
            f"{f('benign_flag_rate')} | {f('balanced_accuracy')} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--line", required=True, help="eval dir with primevul_test_scores.jsonl over the full test")
    ap.add_argument("--llm", nargs="*", default=[], help="name=pairs_scores.jsonl,rest_scores.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    sampled_b, rest_v = set(man["sampled_benign_idx"]), set(man["rest_vulnerable_idx"])
    wB = man["strata"]["B"]["weight"]
    line = {r["idx"]: r for r in load(f"{a.line}/primevul_test_scores.jsonl")}
    assert len(line) == man["n_full"], (len(line), man["n_full"])
    rep = {"n_full": man["n_full"], "n_full_vulnerable": man["n_full_vulnerable"], "rows": {}}
    print(f"full PrimeVul test: {man['n_full']} functions, {man['n_full_vulnerable']} vulnerable "
          f"({man['n_full_vulnerable'] / man['n_full'] * 100:.1f}%); LLM sample: all vulnerable + test-pair benign + "
          f"{len(sampled_b)} other benign x {wB:.2f}")
    print("\n| method | threshold | accuracy [CI] | precision [CI] | recall [CI] | F1 [CI] | benign flagged [CI] | balanced acc. [CI] |")
    print("|---|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)

    # line model: exact on the full test
    ids = sorted(line); y = np.array([line[i]["target"] for i in ids]); s = np.array([line[i]["logit"] for i in ids])
    w = np.ones(len(ids)); st = np.zeros(len(ids), int)
    cands = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 400)))
    best = max(cands, key=lambda t: metrics(y, s > t, w)["f1"])
    for tn, t in (("logit 0", 0.0), ("best F1 on test (oracle)", float(best))):
        m = evaluate(y, s, w, st, t, rng); rep["rows"][f"line_model | {tn}"] = {**m, "threshold": t}; print(row("line model (all 25,911)", tn, m))

    def sample_arrays(scores):
        ids = [i for i in scores if i in line]
        y = np.array([line[i]["target"] for i in ids])
        st = np.array([2 if i in sampled_b else (1 if i in rest_v else 0) for i in ids])
        w = np.where(st == 2, wB, 1.0)
        return ids, y, st, w

    for spec in a.llm:
        name, paths = spec.split("=", 1)
        scores = {}
        for p in paths.split(","):
            scores.update({r["idx"]: r["logit"] for r in load(p)})
        missing_v = man["n_full_vulnerable"] - sum(1 for i in scores if i in line and line[i]["target"] == 1)
        ids, y, st, w = sample_arrays(scores)
        s = np.array([scores[i] for i in ids])
        print(f"| {name}: {len(ids)} scored, vulnerable missing {missing_v}, sampled benign {int((st == 2).sum())} | | | | | | | |")
        for tn, t in (("verdict (score > 0)", 0.0), ("probability >= 95", 1.449)):
            m = evaluate(y, s, w, st, t, rng); rep["rows"][f"{name} | {tn}"] = {**m, "threshold": t}; print(row(name, tn, m))
        ls = np.array([line[i]["logit"] for i in ids])        # line model on the same weighted sample: sanity check
        m = evaluate(y, ls, w, st, float(best), rng); rep["rows"][f"line_model_on_{name}_sample | oracle thr"] = m
        print(row(f"line model on the {name} sample", "best F1 on test (oracle)", m))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
