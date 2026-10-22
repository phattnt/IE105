"""PrimeVul -> line labels from its own vulnerable/fixed pairs, in the secvul_lines format of prepare_secvuleval.py.

train / valid / test come from primevul_{train,valid,test}_paired.jsonl (PrimeVul's own temporal split, so the
test functions are the same 564 pairs that eval_primevul.py scores). Flaw lines = lines of the vulnerable version
that the fix deletes or changes (difflib, the TN1 convention, via eval_primevul.diff_flaw_lines). Pairs whose fix
only adds lines have no flaw line and are kept with flaw_lines = [] (train_lines.py drops them unless a pair loss
is on). --extra_benign N adds N random unpaired benign functions of primevul_train as single target-0 rows.

    python prepare_primevul_lines.py --primevul ../../data/primevul --out ../../data/primevul_lines [--extra_benign 0]
"""
import argparse
import json
import os
import random
from collections import Counter

from eval_primevul import diff_flaw_lines


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


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
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats = {}
    for split in ("train", "valid", "test"):
        rows, st = pair_rows(load_jsonl(os.path.join(a.primevul, f"primevul_{split}_paired.jsonl")), split)
        if split == "train" and a.extra_benign:
            used = {r["idx"] for r in rows}
            ben = [r for r in load_jsonl(os.path.join(a.primevul, "primevul_train.jsonl")) if r["target"] == 0 and r["idx"] not in used]
            for r in random.Random(a.seed).sample(ben, min(len(ben), a.extra_benign)):
                rows.append({"idx": r["idx"], "target": 0, "func": r["func"], "n_lines": r["func"].count("\n") + 1,
                             "flaw_lines": [], "pair_idx": None, "project": r.get("project", ""), "commit_id": r["commit_id"],
                             "cwe": [r.get("cwe") or ""], "split": "train", "in_primevul_train": True, "in_primevul_test": False})
            st["extra_benign"] = a.extra_benign
        with open(os.path.join(a.out, f"{split}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        st["functions"] = len(rows)
        st["shorter_is_vul_pct"] = round(100 * sum(1 for r in rows if r["target"] == 1 and r["pair_idx"] is not None
                                                   and len(r["func"]) < len(next(x["func"] for x in rows if x["idx"] == r["pair_idx"]))) / max(st["pairs"], 1), 1)
        stats[split] = dict(st)
        print(split, json.dumps(dict(st)))
    with open(os.path.join(a.out, "prepare_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
