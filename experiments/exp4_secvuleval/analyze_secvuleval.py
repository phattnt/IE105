"""Experiment 4a: what SecVulEval offers for line-level detection, before spending GPU time on it.

CPU only (Kaggle notebook/script with Internet on). Measures:
  1. schema and label balance;
  2. function length in CodeBERT tokens and how many vulnerable statements fall outside the first 510 tokens
     (same A/B/C grouping as experiment 1);
  3. size of the LLM-extracted context fields;
  4. overlap with PrimeVul (whitespace-normalised function hash, and commit id when available) - needed before
     training on one and testing on the other.
Writes secvuleval_stats.json and a few example rows to the working directory.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from collections import Counter

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
STATS = {}


def log(*a):
    print(*a, flush=True)


def step(name):
    def wrap(fn):
        def run(*a, **k):
            log(f"\n===== {name} =====")
            try:
                return fn(*a, **k)
            except Exception:
                traceback.print_exc()
                STATS.setdefault("errors", {})[name] = traceback.format_exc()[-1500:]
        return run
    return wrap


def save():
    with open(f"{OUT}/secvuleval_stats.json", "w") as fh:
        json.dump(STATS, fh, indent=1, default=str)


subprocess.run(f"{sys.executable} -m pip install -q datasets gdown", shell=True)
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ds = load_dataset("arag0rn/SecVulEval", split="train")
rows = ds.to_list()
cols = list(rows[0].keys())


def pick(cands, required=True):
    for c in cands:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in cands:
        for k, v in low.items():
            if c.lower() in k:
                return v
    if required:
        raise KeyError(f"none of {cands} in {cols}")
    return None


@step("schema")
def schema():
    STATS["n_rows"] = len(rows)
    STATS["columns"] = {c: type(rows[0][c]).__name__ for c in cols}
    log(json.dumps(STATS["columns"], indent=1))
    ex = []
    for r in rows[:3]:
        ex.append({c: (str(v)[:600] if not isinstance(v, (int, float, bool)) else v) for c, v in r.items()})
    with open(f"{OUT}/secvuleval_examples.json", "w") as fh:
        json.dump(ex, fh, indent=1, default=str)
    for c in cols:
        log(f"  {c}: {str(rows[0][c])[:300]!r}")
    # first vulnerable example in full for the line-label fields
    for r in rows:
        if any(r.get(k) for k in cols if "statement" in k.lower() or "changed" in k.lower()):
            log("\n--- first row with changed lines/statements ---")
            for c in cols:
                log(f"  {c}: {str(r[c])[:1200]!r}")
            break


schema()
save()
F_CODE = pick(["func_body", "func", "function", "code", "func_before", "source"])
F_LABEL = pick(["is_vulnerable", "target", "label", "vulnerable", "is_vul"], required=False)
F_STMT = pick(["changed_statements", "vuln_statements"], required=False)
F_LINES = pick(["changed_lines"], required=False)
F_COMMIT = pick(["commit_id", "commit", "commit_hash", "fixing_commit"], required=False)
F_PROJ = pick(["project", "repo", "project_name"], required=False)
F_CVE = pick(["cve_id", "cve", "CVE"], required=False)
STATS["fields_used"] = dict(code=F_CODE, label=F_LABEL, statements=F_STMT, lines=F_LINES, commit=F_COMMIT, project=F_PROJ, cve=F_CVE)
log("fields used:", STATS["fields_used"])
ctx_cols = [c for c in cols if any(k in c.lower() for k in ("context", "callee", "caller", "global", "type", "macro", "param", "env"))
            and c not in (F_CODE, F_LABEL, F_STMT, F_LINES)]


def is_vul(r):
    if F_LABEL is None:
        return bool(r.get(F_STMT)) if F_STMT else None
    v = r[F_LABEL]
    return bool(v) if not isinstance(v, str) else v.strip().lower() in ("1", "true", "yes", "vulnerable")


tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")


def norm(s):
    return re.sub(r"\s+", "", s or "")


def as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return [v] if v.strip() else []
    if isinstance(v, dict):
        v = list(v.values())
    out = []
    for x in v:
        if isinstance(x, (list, tuple)):
            out.extend(x)
        elif isinstance(x, dict):
            out.extend(str(y) for y in x.values())
        else:
            out.append(x)
    return out


@step("balance and metadata")
def balance():
    vul = [is_vul(r) for r in rows]
    STATS["n_vulnerable"] = int(sum(bool(v) for v in vul))
    STATS["vulnerable_pct"] = STATS["n_vulnerable"] / len(rows) * 100
    for f in (F_PROJ, F_CVE, F_COMMIT):
        if f:
            STATS[f"n_unique_{f}"] = len({r[f] for r in rows})
    if F_PROJ:
        STATS["top_projects"] = Counter(r[F_PROJ] for r in rows).most_common(15)
    log(json.dumps({k: STATS[k] for k in STATS if k.startswith(("n_", "vulnerable", "top_"))}, indent=1, default=str))


@step("length and visibility of vulnerable statements")
def visibility():
    import numpy as np
    lens, vlens, groups = [], [], Counter()
    no_label, n_stmt, n_found, vis_stmt = 0, 0, 0, 0
    for r in rows:
        code = r[F_CODE] or ""
        enc = tok(code, add_special_tokens=False, return_offsets_mapping=True)
        n = len(enc["input_ids"])
        lens.append(n)
        if not is_vul(r):
            continue
        vlens.append(n)
        cut = enc["offset_mapping"][509][1] if n > 510 else len(code)
        stmts = [str(s) for s in as_list(r.get(F_STMT))] if F_STMT else []
        stmts = [s for s in stmts if norm(s)]
        if not stmts:
            no_label += 1
            continue
        # locate each statement in the function text (whitespace-insensitive) -> character position
        flat, idx = [], []
        for i, ch in enumerate(code):
            if not ch.isspace():
                flat.append(ch); idx.append(i)
        flat = "".join(flat)
        vis = []
        for s in stmts:
            n_stmt += 1
            p = flat.find(norm(s))
            if p < 0:
                continue
            n_found += 1
            vis.append(idx[p] < cut)
        vis_stmt += sum(vis)
        if not vis:
            groups["unlocated"] += 1
        elif n <= 510:
            groups["A_fits"] += 1
        elif any(vis):
            groups["B_truncated_visible"] += 1
        else:
            groups["C_truncated_hidden"] += 1
    a, v = np.array(lens), np.array(vlens)
    STATS["tokens_all"] = {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "gt510_pct": float((a > 510).mean() * 100)}
    STATS["tokens_vulnerable"] = {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                                  **{f"fits_{b}_pct": float((v <= b).mean() * 100) for b in (510, 1022, 2046, 4094)}}
    STATS["vulnerable_without_statement_label"] = no_label
    STATS["statements"] = {"total": n_stmt, "located_in_function_pct": n_found / max(n_stmt, 1) * 100,
                           "visible_in_first_510_tokens_pct_of_located": vis_stmt / max(n_found, 1) * 100}
    STATS["groups_ABC"] = dict(groups)
    log(json.dumps({k: STATS[k] for k in ("tokens_all", "tokens_vulnerable", "vulnerable_without_statement_label", "statements", "groups_ABC")}, indent=1))


@step("context fields")
def context():
    import numpy as np
    STATS["context_columns"] = ctx_cols
    out = {}
    for c in ctx_cols:
        vals = [r[c] for r in rows if is_vul(r)]
        texts = [json.dumps(v) if not isinstance(v, str) else v for v in vals]
        nonempty = [t for t in texts if t and t not in ("[]", "{}", "null", "\"\"")]
        if not nonempty:
            out[c] = {"nonempty_pct": 0.0}
            continue
        n = np.array([len(tok(t[:20000], add_special_tokens=False)["input_ids"]) for t in nonempty[:3000]])
        out[c] = {"nonempty_pct": len(nonempty) / len(texts) * 100, "tokens_median": float(np.median(n)), "tokens_p90": float(np.percentile(n, 90))}
    STATS["context_tokens_vulnerable"] = out
    log(json.dumps(out, indent=1))


@step("overlap with PrimeVul")
def overlap():
    pv_dir = "/kaggle/tmp/primevul" if os.path.isdir("/kaggle") else "./primevul"
    os.makedirs(pv_dir, exist_ok=True)
    if not any(f.endswith(".jsonl") for f in os.listdir(pv_dir)):
        subprocess.run(f"gdown -q --folder https://drive.google.com/drive/folders/19iLaNDS0z99N8kB_jBRTmDLehwZBolMY -O {pv_dir} --remaining-ok", shell=True)
    sv_hash = {hashlib.md5(norm(r[F_CODE]).encode()).hexdigest(): is_vul(r) for r in rows}
    sv_commits = {str(r[F_COMMIT]).strip()[:40] for r in rows} if F_COMMIT else set()
    res = {}
    for split in ("train", "valid", "test", "test_paired"):
        p = f"{pv_dir}/primevul_{split}.jsonl"
        if not os.path.exists(p):
            continue
        hit = hit_vul = n = nv = com = 0
        for line in open(p):
            js = json.loads(line)
            n += 1; nv += js["target"] == 1
            h = hashlib.md5(norm(js["func"]).encode()).hexdigest()
            if h in sv_hash:
                hit += 1; hit_vul += js["target"] == 1
            if sv_commits and str(js.get("commit_id", "")).strip()[:40] in sv_commits:
                com += 1
        res[split] = {"rows": n, "same_function_text": hit, "same_function_text_vulnerable": hit_vul,
                      "vulnerable_rows": nv, "same_commit_rows": com}
    STATS["overlap_with_primevul"] = res
    log(json.dumps(res, indent=1))


balance(); save()
visibility(); save()
context(); save()
overlap(); save()
log("\n===== STATS =====")
log(json.dumps(STATS, indent=1, default=str)[:20000])
