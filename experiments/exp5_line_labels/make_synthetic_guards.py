"""Synthetic counterfactual pairs for Train 13: remove one guard from a benign PrimeVul function.

For every benign function of primevul_train (valid: primevul_valid) that contains a single-line or 3-line guard
`if (...) return/break/continue/goto;` (same detector as probe_guards.py), one guard is removed - preferring guards
that share an identifier with the next code line (the "victim", i.e. the statement that lost its protection) - and
the result is written as a vulnerable function whose only flaw line is the victim. The untouched function is its
fixed partner (pair_idx both ways). Both versions differ by the guard lines only (1-3 lines), so neither length nor
any other surface feature separates them: the only way to score the pair correctly is to notice that the guard is
missing above the victim.

Output follows the secvul_lines format of prepare_secvuleval.py (train.jsonl, valid.jsonl) and test.jsonl is a copy
of the SecVulEval test split, so train_lines.py scores the stage-1 model on real data at the end of the run.
Functions whose whitespace-stripped text occurs in any SecVulEval split or in primevul_test are skipped.

    python make_synthetic_guards.py --primevul ../../data/primevul --secvul ../../data/secvul_lines \
        --out ../../data/synth_guards [--max_train 60000] [--max_valid 2000]
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import Counter

from probe_guards import find_guards, idents


def norm(s):
    return re.sub(r"\s+", "", s)


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def make_pairs(rows, banned, cap, rng, base_idx):
    stats = Counter()
    out = []
    rng.shuffle(rows)
    for r in rows:
        if r["target"] != 0:
            continue
        stats["benign"] += 1
        lines = r["func"].split("\n")
        if not 5 <= len(lines) <= 300:
            stats["skip_length"] += 1
            continue
        if norm(r["func"]) in banned:
            stats["skip_overlap"] += 1
            continue
        guards = find_guards(lines)
        if not guards:
            stats["no_guard"] += 1
            continue
        stats["with_guard"] += 1
        related = [g for g in guards if set(idents(g[3])) & set(idents(lines[g[2]]))]
        a, b, v, gtext = rng.choice(related or guards)
        stats["related" if related else "unrelated"] += 1
        stats[f"guard_{b - a}_lines"] += 1
        vul_lines = lines[:a] + lines[b:]
        victim = v - (b - a)
        vi, fi = base_idx + 2 * len(out), base_idx + 2 * len(out) + 1
        common = {"project": r.get("project", ""), "commit_id": r.get("commit_id", ""), "cwe": ["SYNTH-guard"],
                  "split": "synthetic", "in_primevul_train": False, "in_primevul_test": False, "guard": gtext.strip()}
        out.append({"idx": vi, "target": 1, "func": "\n".join(vul_lines), "n_lines": len(vul_lines),
                    "flaw_lines": [victim], "pair_idx": fi, **common})
        out.append({"idx": fi, "target": 0, "func": r["func"], "n_lines": len(lines),
                    "flaw_lines": [], "pair_idx": vi, **common})
        if len(out) >= 2 * cap:
            break
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--secvul", required=True, help="secvul_lines dir (overlap filter + test.jsonl copy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_train", type=int, default=60000)
    ap.add_argument("--max_valid", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    banned = set()
    for sp in ("train", "valid", "test"):
        banned |= {norm(r["func"]) for r in load_jsonl(os.path.join(a.secvul, f"{sp}.jsonl"))}
    banned |= {norm(r["func"]) for r in load_jsonl(os.path.join(a.primevul, "primevul_test.jsonl"))}
    print(f"banned texts: {len(banned)}")

    rng = random.Random(a.seed)
    all_stats = {}
    for split, src, cap in (("train", "primevul_train.jsonl", a.max_train), ("valid", "primevul_valid.jsonl", a.max_valid)):
        rows = load_jsonl(os.path.join(a.primevul, src))
        pairs, st = make_pairs(rows, banned, cap, rng, 5_000_000 if split == "train" else 6_000_000)
        with open(os.path.join(a.out, f"{split}.jsonl"), "w", encoding="utf-8") as fh:
            for r in pairs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        st["pairs_written"] = len(pairs) // 2
        all_stats[split] = dict(st)
        print(split, json.dumps(dict(st)))
    shutil.copyfile(os.path.join(a.secvul, "test.jsonl"), os.path.join(a.out, "test.jsonl"))
    with open(os.path.join(a.out, "prepare_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=1)
    print("test.jsonl = SecVulEval test; wrote", a.out)


if __name__ == "__main__":
    main()
