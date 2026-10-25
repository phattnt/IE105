# Builds the LLM input sets (llm_audit.py "scores" format: idx, target, func, pair_idx, flaw_lines, line_scores, chars).
#   inputs/valid_pairs.jsonl      the 562 PrimeVul valid pairs (fresh data, never looked at)
#   inputs/fulltest_rest.jsonl    full-test functions NOT in the test pairs: all remaining vulnerable ones + a uniform
#                                 random sample of the remaining benign ones
#   inputs/fulltest_manifest.json strata and weights for estimating precision / recall / F1 / accuracy on the whole test:
#     P = the 1128 test-pair functions (scored separately, weight 1), V = the other vulnerable (all, weight 1),
#     B = the other benign (sample of n_B, weight N_B / n_B)
# usage: python build_llm_inputs.py --primevul ../../data/primevul --n_benign 3000 [--seed 0]
import argparse, json, os, random


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def row(r, pair_idx=None):
    return {"idx": r["idx"], "target": r["target"], "func": r["func"], "pair_idx": pair_idx, "flaw_lines": [],
            "line_scores": [], "chars": len(r["func"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--n_benign", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vp = load(os.path.join(a.primevul, "primevul_valid_paired.jsonl"))
    out = []
    for i in range(0, len(vp) - 1, 2):
        x, y = vp[i], vp[i + 1]
        if x["commit_id"] == y["commit_id"] and {x["target"], y["target"]} == {0, 1}:
            out += [row(x, y["idx"]), row(y, x["idx"])]
    with open(os.path.join(a.out, "valid_pairs.jsonl"), "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"valid_pairs: {len(out)} functions, {len(out) // 2} pairs")

    paired = {r["idx"] for r in load(os.path.join(a.primevul, "primevul_test_paired.jsonl"))}
    full = load(os.path.join(a.primevul, "primevul_test.jsonl"))
    rest_v = [r for r in full if r["idx"] not in paired and r["target"] == 1]
    rest_b = [r for r in full if r["idx"] not in paired and r["target"] == 0]
    rng = random.Random(a.seed)
    samp_b = rng.sample(rest_b, min(a.n_benign, len(rest_b)))
    rows = [row(r) for r in rest_v + samp_b]
    rng.shuffle(rows)
    with open(os.path.join(a.out, "fulltest_rest.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"n_full": len(full), "n_full_vulnerable": sum(r["target"] for r in full),
           "strata": {"P": {"n": len(paired), "sampled": len(paired), "weight": 1.0, "note": "test pairs, scored in the pair run"},
                      "V": {"n": len(rest_v), "sampled": len(rest_v), "weight": 1.0},
                      "B": {"n": len(rest_b), "sampled": len(samp_b), "weight": len(rest_b) / len(samp_b)}},
           "seed": a.seed, "sampled_benign_idx": sorted(r["idx"] for r in samp_b), "rest_vulnerable_idx": sorted(r["idx"] for r in rest_v)}
    json.dump(man, open(os.path.join(a.out, "fulltest_manifest.json"), "w"), indent=1)
    print(f"fulltest_rest: {len(rows)} functions ({len(rest_v)} vulnerable + {len(samp_b)} of {len(rest_b)} benign, "
          f"weight {man['strata']['B']['weight']:.2f})")


if __name__ == "__main__":
    main()
