"""Experiment 5 data: SecVulEval -> function-relative line labels, vulnerable/fixed pairs, project split.

CPU only, needs Internet (HuggingFace + Google Drive for the PrimeVul overlap flags). Output (jsonl, one function per row):
  idx, project, commit_id, cwe, split, target, func, n_lines, flaw_lines (0-based line numbers inside `func`),
  pair_idx (fixed version for a vulnerable row, vulnerable version for a fixed row, else null),
  in_primevul_train / in_primevul_test (same whitespace-normalised text as a PrimeVul function)
commit_message, CVE text and file paths are dropped on purpose: they must never reach a model input.

Split (--split project, default): projects holding more than --big_share of all functions always go to train
(linux, Chrome, tensorflow, Android); the remaining projects are shuffled (seed) and assigned whole to
test / valid until each holds ~10% of the functions; the rest is train. Pairs never straddle splits because
both members share the project. --split commit does the same at commit granularity.
"""
import argparse
import ast
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict

PRIMEVUL_FOLDER = "19iLaNDS0z99N8kB_jBRTmDLehwZBolMY"


def norm(s):
    return re.sub(r"\s+", "", s or "")


def statement_texts(v):
    """changed_statements is a JSON list of [line number in the FILE, statement text]."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    out = []
    for x in v or []:
        t = x[1] if isinstance(x, (list, tuple)) and len(x) == 2 else x
        if isinstance(t, str) and norm(t):
            out.append(t)
    return out


def flaw_lines(func, stmts):
    """Locate each statement in `func` ignoring whitespace; return the 0-based lines it spans."""
    flat, pos = [], []
    for i, ch in enumerate(func):
        if not ch.isspace():
            flat.append(ch)
            pos.append(i)
    flat = "".join(flat)
    starts = [0] + [i + 1 for i, ch in enumerate(func) if ch == "\n"]
    import bisect
    lines, missed = set(), 0
    for s in stmts:
        k = norm(s)
        p = flat.find(k)
        if p < 0:
            missed += 1
            continue
        a, b = pos[p], pos[p + len(k) - 1]
        la, lb = bisect.bisect_right(starts, a) - 1, bisect.bisect_right(starts, b) - 1
        lines.update(range(la, lb + 1))
    return sorted(lines), missed


def parse_list(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:
        return [v] if v else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/secvul_lines" if os.path.isdir("/kaggle/working") else "secvul_lines")
    ap.add_argument("--split", choices=["project", "commit"], default="project")
    ap.add_argument("--big_share", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--primevul_dir", default="/kaggle/tmp/primevul" if os.path.isdir("/kaggle") else "primevul")
    ap.add_argument("--skip_primevul", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    subprocess.run(f"{sys.executable} -m pip install -q datasets gdown", shell=True)
    from datasets import load_dataset

    rows = load_dataset("arag0rn/SecVulEval", split="train").to_list()
    by_idx = {r["idx"]: r for r in rows}
    stats = Counter()
    recs = []
    for r in rows:
        func = r["func_body"] or ""
        vul = bool(r["is_vulnerable"])
        fl, missed = flaw_lines(func, statement_texts(r["changed_statements"])) if vul else ([], 0)
        stats["statements_missed"] += missed
        pair = None
        if vul and r.get("fixed_func_idx") == r.get("fixed_func_idx") and r.get("fixed_func_idx") is not None:
            f = by_idx.get(int(r["fixed_func_idx"]))
            if f is not None and not f["is_vulnerable"]:
                pair = f["idx"]
        recs.append({"idx": r["idx"], "project": r["project"], "commit_id": r["commit_id"],
                     "cwe": parse_list(r.get("cwe_list")), "target": int(vul), "func": func,
                     "n_lines": func.count("\n") + 1, "flaw_lines": fl, "pair_idx": pair})
        stats["vulnerable"] += vul
        stats["vulnerable_with_lines"] += vul and bool(fl)
    rec_by_idx = {x["idx"]: x for x in recs}
    for x in recs:                      # back-link fixed -> vulnerable
        if x["target"] == 1 and x["pair_idx"] is not None:
            rec_by_idx[x["pair_idx"]]["pair_idx"] = x["idx"]
    stats["pairs"] = sum(1 for x in recs if x["target"] == 1 and x["pair_idx"] is not None)

    # ---- split
    unit = (lambda x: x["project"]) if a.split == "project" else (lambda x: x["commit_id"])
    size = Counter(unit(x) for x in recs)
    n = len(recs)
    big = {u for u, c in size.items() if a.split == "project" and c > a.big_share * n}
    rest = sorted(u for u in size if u not in big)
    random.Random(a.seed).shuffle(rest)
    assign, filled = {u: "train" for u in big}, {"test": 0, "valid": 0}
    for u in rest:
        for s in ("test", "valid"):
            if filled[s] < 0.10 * n:
                assign[u] = s
                filled[s] += size[u]
                break
        else:
            assign[u] = "train"
    for x in recs:
        x["split"] = assign[unit(x)]
    stats["big_projects_in_train"] = sorted(big)

    # ---- overlap with PrimeVul
    if not a.skip_primevul:
        os.makedirs(a.primevul_dir, exist_ok=True)
        if not any(f.endswith(".jsonl") for f in os.listdir(a.primevul_dir)):
            subprocess.run(f"gdown -q --folder https://drive.google.com/drive/folders/{PRIMEVUL_FOLDER} -O {a.primevul_dir}", shell=True)
        hashes = {}
        for split in ("train", "valid", "test"):
            p = f"{a.primevul_dir}/primevul_{split}.jsonl"
            if os.path.exists(p):
                hashes[split] = {hashlib.md5(norm(json.loads(l)["func"]).encode()).hexdigest() for l in open(p)}
        for x in recs:
            h = hashlib.md5(norm(x["func"]).encode()).hexdigest()
            x["in_primevul_train"] = h in hashes.get("train", set()) or h in hashes.get("valid", set())
            x["in_primevul_test"] = h in hashes.get("test", set())

    for s in ("train", "valid", "test"):
        part = [x for x in recs if x["split"] == s]
        with open(f"{a.out}/{s}.jsonl", "w") as fh:
            for x in part:
                fh.write(json.dumps(x) + "\n")
        stats[f"{s}_functions"] = len(part)
        stats[f"{s}_vulnerable"] = sum(x["target"] for x in part)
        stats[f"{s}_pairs"] = sum(1 for x in part if x["target"] == 1 and x["pair_idx"] is not None)
        stats[f"{s}_projects"] = len({x["project"] for x in part})
        stats[f"{s}_in_primevul_test"] = sum(bool(x.get("in_primevul_test")) for x in part)
    with open(f"{a.out}/prepare_stats.json", "w") as fh:
        json.dump(dict(stats), fh, indent=1)
    print(json.dumps(dict(stats), indent=1), flush=True)


if __name__ == "__main__":
    main()
