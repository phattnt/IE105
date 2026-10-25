# Paired comparison of tier-2b LLM arms (and the line model) on the PrimeVul test pairs.
# For every pair (vulnerable v, fixed f) and every arm: net = +1 if only v is flagged (P-C), -1 if only f (P-R), else 0
# (threshold 0 = the LLM's own verdict); rank = 1 / 0.5 / 0 for score(v) > / = / < score(f); debiased rank after
# removing a cubic fit on log(chars) over all scored functions (pair_protocol.debias).
# Differences between arms are paired bootstrap CIs over the same pairs (B resamples).
# Subsets: all, changed (the vulnerable function has >= 1 flaw line), add_only (the fix only adds lines), length_matched (<5%).
# usage: python compare_arms.py --primevul <dir> --line ../results/eval_pv1_best --llm ../results/llm_full_a100 --out <json>
import argparse, json, os
import numpy as np
from pair_protocol import load_jsonl, debias

B = 2000
ARMS = ["plain", "audit", "audit_hint", "audit_rand", "audit_hint_nt"]
CONTRASTS = [("audit_hint", "audit_rand", "effect of the line model's lines vs random lines"),
             ("audit_hint", "audit", "hint vs no hint"),
             ("audit", "plain", "guard-audit prompt vs plain question"),
             ("audit_hint", "plain", "full method vs plain question"),
             ("audit_hint", "audit_hint_nt", "thinking vs no thinking"),
             ("llm_mean", "plain", "mean of thinking arms vs plain (exploratory)"),
             ("plain", "line_model", "plain LLM vs line model (rank only; line model is not thresholded at 0)")]


def per_pair(sv, sf, cv, cf):
    net = np.where((sv > 0) & (sf <= 0), 1.0, np.where((sv <= 0) & (sf > 0), -1.0, 0.0))
    rank = np.where(sv > sf, 1.0, np.where(sv == sf, 0.5, 0.0))
    n = len(sv)
    d = debias(np.concatenate([cv, cf]), np.concatenate([sv, sf]))
    rank_tb = np.where(d[:n] > d[n:], 1.0, np.where(d[:n] == d[n:], 0.5, 0.0))
    # LLM scores take ~20 values, so ~40% of pairs tie exactly. Subtracting a length fit turns every tie into a win for
    # one side decided by length alone (pair_protocol's debiased rank does that). rank_d keeps exact ties at 0.5 and
    # removes the length fit only where the arm itself separates the two functions.
    rank_d = np.where(sv == sf, 0.5, rank_tb)
    return net, rank, rank_d, rank_tb


def boot_ci(x, rng):
    x = np.asarray(x); idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--line", required=True, help="eval dir of the line model (test_scores.jsonl, primevul_test_scores.jsonl), or none")
    ap.add_argument("--llm", required=True, help="pulled LLM run dir with scores/<arm>/primevul_test_scores.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--paired", default="primevul_test_paired.jsonl", help="pair file under --primevul (valid: primevul_valid_paired.jsonl)")
    ap.add_argument("--extra", nargs="*", default=[], help="name=<primevul_test_scores.jsonl> of another run, e.g. audit_8k=...")
    ap.add_argument("--contrast", nargs="*", default=[], help="x:y pairs to add to the contrasts, e.g. audit_8k:audit")
    a = ap.parse_args()
    for c in a.contrast:
        x, y = c.split(":"); CONTRASTS.append((x, y, "extra contrast"))
    paired = load_jsonl(os.path.join(a.primevul, a.paired))
    flaw = {r["idx"]: r.get("flaw_lines") for r in load_jsonl(os.path.join(a.line, "test_scores.jsonl"))} if a.line != "none" else {}
    pairs = []
    for i in range(0, len(paired) - 1, 2):
        x, y = paired[i], paired[i + 1]
        if x["commit_id"] == y["commit_id"] and {x["target"], y["target"]} == {0, 1}:
            v, f = (x, y) if x["target"] == 1 else (y, x)
            pairs.append((v["idx"], f["idx"], len(v["func"]), len(f["func"])))
    cv = np.array([p[2] for p in pairs], float); cf = np.array([p[3] for p in pairs], float)
    subsets = {"all": np.ones(len(pairs), bool)}
    if flaw:                                   # flaw lines are known for the test pairs only
        changed = np.array([bool(flaw.get(p[0])) for p in pairs])
        subsets.update(changed=changed, add_only=~changed)
    subsets["length_matched"] = np.abs(cv - cf) / np.maximum(cv, cf) < 0.05

    scores = {}
    if a.line != "none":
        scores["line_model"] = {r["idx"]: r["logit"] for r in load_jsonl(os.path.join(a.line, "primevul_test_scores.jsonl"))}
    for arm in ARMS:
        p = os.path.join(a.llm, "scores", arm, "primevul_test_scores.jsonl")
        if os.path.exists(p):
            scores[arm] = {r["idx"]: r["logit"] for r in load_jsonl(p)}
    for e in a.extra:
        name, p = e.split("=", 1)
        scores[name] = {r["idx"]: r["logit"] for r in load_jsonl(p)}
    # exploratory (no extra GPU): mean score over the thinking arms present = a cheap self-consistency vote
    think = [k for k in ("plain", "audit", "audit_hint", "audit_rand") if k in scores]
    if len(think) >= 2:
        common = set.intersection(*(set(scores[k]) for k in think))
        scores["llm_mean"] = {i: float(np.mean([scores[k][i] for k in think])) for i in common}
        print("llm_mean = mean of", think)
    have_all = np.array([all(p[0] in s and p[1] in s for s in scores.values()) for p in pairs])
    print(f"pairs {len(pairs)}, scored by every arm {int(have_all.sum())}: " +
          ", ".join(f"{k} {int((m & have_all).sum())}" for k, m in subsets.items()))

    rep = {"arms": {}, "contrasts": {}, "n_pairs": int(have_all.sum())}
    pp = {}
    for arm, s in scores.items():
        sv = np.array([s.get(p[0], 0.0) for p in pairs]); sf = np.array([s.get(p[1], 0.0) for p in pairs])
        pp[arm] = {"net": None, "flag_v": sv > 0, "flag_f": sf > 0}
        net, rank, rank_d, rank_tb = per_pair(sv[have_all], sf[have_all], cv[have_all], cf[have_all])
        full = lambda v: (lambda o: (o.__setitem__(have_all, v), o)[1])(np.full(len(pairs), np.nan))
        pp[arm].update(net=full(net), rank=full(rank), rank_d=full(rank_d), rank_tb=full(rank_tb),
                       tie=full((sv == sf)[have_all].astype(float)))

    print("\n| arm | tập | n | P-C | P-V | P-B | P-R | net [CI] | xếp đúng [CI] | sau trừ độ dài (hòa giữ 0.5) | trừ độ dài kiểu pair_protocol | % cặp hòa | tỉ lệ gắn cờ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in scores:
        rep["arms"][arm] = {}
        for sname, m0 in subsets.items():
            m = m0 & have_all
            if not m.any():
                continue
            fv, ff = pp[arm]["flag_v"][m], pp[arm]["flag_f"][m]
            rng = np.random.default_rng(0)
            r = {"n": int(m.sum()), "P-C": float((fv & ~ff).mean() * 100), "P-V": float((fv & ff).mean() * 100),
                 "P-B": float((~fv & ~ff).mean() * 100), "P-R": float((~fv & ff).mean() * 100),
                 "net": float(pp[arm]["net"][m].mean() * 100), "net_ci": boot_ci(pp[arm]["net"][m], rng),
                 "rank_acc": float(pp[arm]["rank"][m].mean() * 100), "rank_acc_ci": boot_ci(pp[arm]["rank"][m], rng),
                 "rank_acc_debiased": float(pp[arm]["rank_d"][m].mean() * 100),
                 "rank_acc_debiased_tiebreak": float(pp[arm]["rank_tb"][m].mean() * 100), "ties": float(pp[arm]["tie"][m].mean() * 100),
                 "flag_rate": float(np.concatenate([fv, ff]).mean() * 100)}
            rep["arms"][arm][sname] = r
            print(f"| {arm} | {sname} | {r['n']} | {r['P-C']:.1f} | {r['P-V']:.1f} | {r['P-B']:.1f} | {r['P-R']:.1f} | "
                  f"{r['net']:+.1f} [{r['net_ci'][0]:+.1f}; {r['net_ci'][1]:+.1f}] | {r['rank_acc']:.1f} [{r['rank_acc_ci'][0]:.1f}; {r['rank_acc_ci'][1]:.1f}] | "
                  f"{r['rank_acc_debiased']:.1f} | {r['rank_acc_debiased_tiebreak']:.1f} | {r['ties']:.1f} | {r['flag_rate']:.1f} |")

    print("\n| so sánh | ý nghĩa | tập | n | Δnet [CI 95%] | Δxếp đúng [CI] | Δxếp đúng sau trừ độ dài [CI] |")
    print("|---|---|---|---|---|---|---|")
    for x, y, what in CONTRASTS:
        if x not in scores or y not in scores:
            continue
        key = f"{x} - {y}"; rep["contrasts"][key] = {"meaning": what}
        for sname, m0 in subsets.items():
            m = m0 & have_all; rng = np.random.default_rng(1)
            if not m.any():
                continue
            r = {"n": int(m.sum())}
            for k in ("net", "rank", "rank_d"):
                d = pp[x][k][m] - pp[y][k][m]
                r[k] = float(d.mean() * 100); r[k + "_ci"] = boot_ci(d, rng)
            rep["contrasts"][key][sname] = r
            f = lambda k: f"{r[k]:+.1f} [{r[k + '_ci'][0]:+.1f}; {r[k + '_ci'][1]:+.1f}]"
            print(f"| {key} | {what} | {sname} | {r['n']} | {f('net')} | {f('rank')} | {f('rank_d')} |")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, ensure_ascii=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
