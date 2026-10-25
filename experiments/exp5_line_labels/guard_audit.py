"""Tier 2a of "localise, then audit the guards": a rule-based guard audit on top of the line model's scores (no training,
CPU, seconds). For every function the k highest-scoring code lines are the sink candidates; a sink is "guarded" to the
extent that its identifiers appear in an earlier guard line (if / while / for / assert / ?: / early return ...).
Function scores written per variant to <out>/<variant>/primevul_test_scores.jsonl (idx, target, logit), so
pair_protocol.py can score them exactly like a model:

  max_logit        the line model's own function score (reference)
  topk_mean_p      mean sigmoid of the top-k lines (reference)
  unguarded_max    max over sinks of p_i * (1 - g_i)             g_i = guarded fraction of the sink's identifiers
  unguarded_sum    sum over sinks of p_i * (1 - g_i)
  unguarded_count  number of sinks with g_i < 0.5, ties broken by unguarded_max
  guard_density    - (#guard lines / #code lines), no model at all (does "fewer checks" alone separate the pair?)
  neg_guard_count  - #guard lines, no model (length-confounded on purpose: a control)

Also prints, for the pairs whose fix changes a line, how often the flaw line is unguarded in the vulnerable version but
guarded in the fixed one (text-matched), which is the mechanism the audit relies on.

  python guard_audit.py --scores results/eval_pv1_best/test_scores.jsonl --out results/guard_audit [--k 5]
"""
import argparse
import json
import math
import os
import re

COMMENT = ("//", "/*", "*", "*/")
KEYWORDS = set("""auto break case char const continue default do double else enum extern float for goto if inline int long
register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while bool true false
NULL nullptr new delete this class public private protected template typename namespace using virtual override try catch
throw static_cast dynamic_cast reinterpret_cast const_cast uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t
size_t ssize_t u8 u16 u32 u64 s8 s16 s32 s64 gboolean gint guint gchar guchar gsize gpointer BOOL DWORD WORD BYTE""".split())
GUARD_RE = re.compile(r"\b(if|else\s+if|while|for|switch|assert|ASSERT|BUG_ON|WARN_ON|WARN_ON_ONCE|CHECK|DCHECK|g_assert|"
                      r"g_return_if_fail|g_return_val_if_fail|unlikely|likely|av_assert\d?|Q_ASSERT|RETURN_IF|CHECK_\w+)\b|\?")
IDENT_RE = re.compile(r"(?<![\w.>])([A-Za-z_]\w*)(?!\s*\()")      # identifiers that are not called like functions
MEMBER_RE = re.compile(r"[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)+")


def is_code(line):
    s = line.strip()
    return bool(s) and not s.startswith(COMMENT)


def idents(line):
    """variables (incl. member chains) used in a line, minus keywords/types/numbers."""
    out = set(m.group(0).replace(" ", "") for m in MEMBER_RE.finditer(line))
    out |= {m.group(1) for m in IDENT_RE.finditer(line)}
    return {v for v in out if v.split(".")[0].split("->")[0] not in KEYWORDS and not v[0].isdigit() and len(v) > 1}


def is_guard(line):
    s = line.strip()
    return bool(GUARD_RE.search(s)) or (s.startswith("return") and ("if" in s or "?" in s))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def audit(func, scores, k):
    lines = func.split("\n")
    n = min(len(lines), len(scores))
    code = [i for i in range(n) if is_code(lines[i])]
    guards = [i for i in code if is_guard(lines[i])]
    gid = {i: idents(lines[i]) for i in guards}
    sinks = sorted(code, key=lambda i: -scores[i])[:k]
    rec = {"n_code": len(code), "n_guards": len(guards), "sinks": []}
    for i in sinks:
        v = idents(lines[i])
        if is_guard(lines[i]):               # a guard line itself is not a sink
            v = set()
        covered = {x for x in v if any(x in gid[j] or x.split(".")[0].split("->")[0] in {y.split(".")[0].split("->")[0] for y in gid[j]}
                                       for j in guards if j < i)}
        g = len(covered) / len(v) if v else 1.0
        rec["sinks"].append({"line": i, "p": sigmoid(scores[i]), "logit": scores[i], "n_ident": len(v), "guarded": g})
    return rec


def variants(rec, max_logit):
    s = rec["sinks"]
    un = [x["p"] * (1 - x["guarded"]) for x in s]
    dens = rec["n_guards"] / max(rec["n_code"], 1)
    return {"max_logit": max_logit,
            "topk_mean_p": sum(x["p"] for x in s) / max(len(s), 1),
            "unguarded_max": max(un) if un else 0.0,
            "unguarded_sum": sum(un),
            "unguarded_count": sum(1 for x in s if x["guarded"] < 0.5) + (max(un) if un else 0.0) * 0.001,
            "guard_density": -dens,
            "neg_guard_count": -float(rec["n_guards"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="test_scores.jsonl of a run (func, flaw_lines, line_scores, logit)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.scores, encoding="utf-8")]
    by = {r["idx"]: r for r in rows}
    recs, outs = {}, {}
    for r in rows:
        rec = audit(r["func"], r["line_scores"], a.k)
        recs[r["idx"]] = rec
        outs[r["idx"]] = variants(rec, r["logit"])
    names = list(next(iter(outs.values())).keys())
    for nm in names:
        os.makedirs(f"{a.out}/{nm}", exist_ok=True)
        with open(f"{a.out}/{nm}/primevul_test_scores.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"idx": r["idx"], "target": r["target"], "logit": round(float(outs[r["idx"]][nm]), 6)}) + "\n")
    json.dump({r["idx"]: recs[r["idx"]] for r in rows}, open(f"{a.out}/audit.json", "w"), indent=0)

    # mechanism check on pairs whose fix changes a line: the flaw line in the vulnerable version vs the same line (by text,
    # or the line at the same position when the text changed) in the fixed version
    vul_un, fix_g, both, n_pairs = 0, 0, 0, 0
    sink_hit = 0
    for r in rows:
        if r["target"] != 1 or not r["flaw_lines"] or r["pair_idx"] not in by:
            continue
        f = by[r["pair_idx"]]
        vl, fl = r["func"].split("\n"), f["func"].split("\n")
        rv, rf = audit(r["func"], r["line_scores"], 10 ** 6), audit(f["func"], f["line_scores"], 10 ** 6)
        gv = {x["line"]: x["guarded"] for x in rv["sinks"]}
        gf_text = {}
        for x in rf["sinks"]:
            gf_text.setdefault(fl[x["line"]].strip(), x["guarded"])
        n_pairs += 1
        hit = any(x["line"] in r["flaw_lines"] for x in sorted(rv["sinks"], key=lambda x: -x["p"])[:a.k])
        sink_hit += hit
        i = max(r["flaw_lines"], key=lambda i: r["line_scores"][i] if i < len(r["line_scores"]) else -1e9)
        t = vl[i].strip() if i < len(vl) else ""
        g_v = gv.get(i, 1.0)
        g_f = gf_text.get(t, None)
        if g_f is None:          # the line's text changed in the fix: compare with the fixed line at the same position
            g_f = {x["line"]: x["guarded"] for x in rf["sinks"]}.get(i, 1.0)
        vul_un += g_v < 0.5; fix_g += g_f >= 0.5; both += (g_v < 0.5 and g_f >= 0.5)
    print(f"pairs with a changed line: {n_pairs} | flaw line among the top-{a.k} sinks: {sink_hit / max(n_pairs, 1) * 100:.1f}% | "
          f"flaw line unguarded in vul: {vul_un / max(n_pairs, 1) * 100:.1f}% | its counterpart guarded in fix: {fix_g / max(n_pairs, 1) * 100:.1f}% | "
          f"both (the audit's mechanism fires): {both / max(n_pairs, 1) * 100:.1f}%")
    print("wrote", a.out, names)


if __name__ == "__main__":
    main()
