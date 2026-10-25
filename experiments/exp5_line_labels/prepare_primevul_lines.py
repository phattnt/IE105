"""PrimeVul -> line labels from its own vulnerable/fixed pairs, in the secvul_lines format of prepare_secvuleval.py.

train / valid / test come from primevul_{train,valid,test}_paired.jsonl (PrimeVul's own temporal split, so the
test functions are the same 564 pairs that eval_primevul.py scores). Flaw lines = lines of the vulnerable version
that the fix deletes or changes (difflib, the TN1 convention, via eval_primevul.diff_flaw_lines). Pairs whose fix
only adds lines have no flaw line and are kept with flaw_lines = [] (train_lines.py drops them unless a pair loss
is on). --extra_benign N adds N random unpaired benign functions of primevul_train as single target-0 rows (a fixed,
seeded set); --extra_benign_valid N does the same for valid from primevul_valid. A benign function whose
whitespace-normalised text equals any test function is never used. prepare_stats.json reports the commit and text
overlaps between the splits (PrimeVul's own temporal split: 0 commits in common expected).

    python prepare_primevul_lines.py --primevul ../../data/primevul --out ../../data/primevul_lines \
        [--extra_benign 80000 --extra_benign_valid 3000 --seed 0]
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter

from eval_primevul import diff_flaw_lines


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def text_hash(func):
    return hashlib.md5(re.sub(r"\s+", "", func).encode()).hexdigest()


def benign_rows(primevul, split, n, used_idx, banned_hash, seed):
    """n random unpaired benign functions of primevul_<split> (fixed by seed); banned texts are skipped."""
    pool = [r for r in load_jsonl(os.path.join(primevul, f"primevul_{split}.jsonl"))
            if r["target"] == 0 and r["idx"] not in used_idx and text_hash(r["func"]) not in banned_hash]
    pick = random.Random(seed).sample(pool, min(len(pool), n))
    rows = [{"idx": r["idx"], "target": 0, "func": r["func"], "n_lines": r["func"].count("\n") + 1, "flaw_lines": [],
             "pair_idx": None, "project": r.get("project", ""), "commit_id": r["commit_id"], "cwe": [r.get("cwe") or ""],
             "split": split, "in_primevul_train": split == "train", "in_primevul_test": split == "test"} for r in pick]
    return rows, len(pool)


def pair_rows(paired, split):
    out, st = [], Counter()
    for k in range(0, len(paired) - 1, 2):
        v, f = paired[k], paired[k + 1]
        if v["commit_id"] != f["commit_id"] or {v["target"], f["target"]} != {0, 1}:
            st["skipped_misaligned"] += 1
            continue
        if v["target"] == 0:
            v, f = f, v
        fl = diff_flaw_lines(v["func"], f["func"])
        st["pairs"] += 1
        st["add_only_fix"] += int(not fl)
        common = {"project": v.get("project", ""), "commit_id": v["commit_id"], "cwe": [v.get("cwe") or ""], "split": split,
                  "in_primevul_train": split == "train", "in_primevul_test": split == "test"}
        out.append({"idx": v["idx"], "target": 1, "func": v["func"], "n_lines": v["func"].count("\n") + 1,
                    "flaw_lines": fl, "pair_idx": f["idx"], **common})
        out.append({"idx": f["idx"], "target": 0, "func": f["func"], "n_lines": f["func"].count("\n") + 1,
                    "flaw_lines": [], "pair_idx": v["idx"], **common})
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra_benign", type=int, default=0)
    ap.add_argument("--extra_benign_valid", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats, rows_by, st_by = {}, {}, {}
    for split in ("train", "valid", "test"):
        rows_by[split], st_by[split] = pair_rows(load_jsonl(os.path.join(a.primevul, f"primevul_{split}_paired.jsonl")), split)
    test_hash = {text_hash(r["func"]) for r in rows_by["test"]}
    test_hash |= {text_hash(r["func"]) for r in load_jsonl(os.path.join(a.primevul, "primevul_test.jsonl"))}
    for split, n in (("train", a.extra_benign), ("valid", a.extra_benign_valid)):
        if n:
            extra, pool = benign_rows(a.primevul, split, n, {r["idx"] for r in rows_by[split]}, test_hash, a.seed)
            rows_by[split] += extra
            st_by[split]["extra_benign"] = len(extra)
            st_by[split]["benign_pool"] = pool
    H = {s: {text_hash(r["func"]) for r in rows_by[s]} for s in rows_by}
    C = {s: {r["commit_id"] for r in rows_by[s]} for s in rows_by}
    stats["overlap"] = {f"{x}_{y}": {"commits": len(C[x] & C[y]), "same_text": len(H[x] & H[y])}
                        for x, y in (("train", "test"), ("valid", "test"), ("train", "valid"))}
    # strict: nothing that shares a commit (or a text) with the test split is trained or validated on
    for split in ("train", "valid"):
        rows = rows_by[split]
        bad = {r["idx"] for r in rows if text_hash(r["func"]) in H["test"] or r["commit_id"] in C["test"]}
        bad |= {r["pair_idx"] for r in rows if r["idx"] in bad and r["pair_idx"] is not None}
        if bad:
            rows_by[split] = [r for r in rows if r["idx"] not in bad]
            st_by[split]["dropped_sharing_test_commit_or_text"] = len(bad)
    for split, rows in rows_by.items():
        st = st_by[split]
        with open(os.path.join(a.out, f"{split}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {r["idx"]: r for r in rows}
        st["functions"] = len(rows)
        st["vulnerable"] = sum(r["target"] == 1 for r in rows)
        st["vulnerable_with_flaw_lines"] = sum(r["target"] == 1 and bool(r["flaw_lines"]) for r in rows)
        st["benign"] = st["functions"] - st["vulnerable"]
        st["shorter_is_vul_pct"] = round(100 * sum(1 for r in rows if r["target"] == 1 and r["pair_idx"] in by
                                                   and len(r["func"]) < len(by[r["pair_idx"]]["func"])) / max(st["pairs"], 1), 1)
        stats[split] = dict(st)
        print(split, json.dumps(dict(st)))
    print("overlap", json.dumps(stats["overlap"]))
    with open(os.path.join(a.out, "prepare_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
