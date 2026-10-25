"""Phase 0: what kind of fix is each PrimeVul pair, and how often is it "A requires B" with B depending on code outside
the function? Gives the ceiling for the context direction (C3) before any GPU is spent on it.

Per pair (vulnerable v, fixed f), from a line diff with comments and whitespace removed:
  kind     trivial (nothing but comments / whitespace / const / renames changed), add_only (f only inserts lines),
           del_only, modify (lines replaced)
  B types  what the inserted / replaced code of f adds (multi-label, regex heuristics):
           null_check, bounds_check (comparison with a size / length / count / limit), overflow (overflow helpers,
           *_MAX, wider or unsigned types), init (memset / zeroing / = {0} / calloc), release (free / unlock / put /
           unref / close / delete), error_path (return of an error code / goto / ret < 0 checks), lifetime (refcount,
           weak or scoped pointers, std::move), other
  A->B     f adds a check (if / while / ?: / assert / return-on-condition) whose identifiers already occur in the
           unchanged code of v: the precondition A is present in the vulnerable function and only B is missing
  outside  where the knowledge that B is needed has to come from, for the identifiers the added check tests:
           callee  - a value returned by a call (assigned from a call in v, or a call inside the condition): the callee's
                     contract (can it return NULL / an error / a short length?)
           param   - a parameter of the function: what the callers guarantee
           field   - a struct member (a->b, a.b): the type's invariants
           macro   - an ALL_CAPS constant / macro
           local   - nothing of the above: decidable from the function body alone
           A pair is "needs_outside" when any tested identifier is callee or param (types 2 and 3 of the discussion);
           field / macro are reported apart since a model can partly learn them from the code base.

Writes <out>/patch_taxonomy.json (counts per split and per project group) and <out>/patch_taxonomy_examples.md
(random examples per class, for a manual precision check).

    python patch_taxonomy.py --primevul ../../data/primevul --out results/patch_taxonomy
"""
import argparse
import difflib
import json
import os
import random
import re
from collections import Counter, defaultdict

KEYWORDS = set("""if else while for do switch case default return goto break continue sizeof typeof struct union enum
const static volatile extern inline unsigned signed int long short char void float double bool true false NULL nullptr
auto register size_t ssize_t this new delete try catch throw class public private protected virtual override template
typename namespace using operator u8 u16 u32 u64 s8 s16 s32 s64 uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t
int64_t uintptr_t off_t loff_t gfp_t unlikely likely WARN_ON BUG_ON assert DCHECK CHECK""".split())
ID = re.compile(r"[A-Za-z_]\w*")
CHECK = re.compile(r"^\s*(\}?\s*else\s+)?(if|while)\s*\(|\?|^\s*(assert|DCHECK|CHECK|WARN_ON|BUG_ON|BUG_ON_ONCE|"
                   r"WARN_ON_ONCE|RETURN_IF|TF_RETURN_IF|OP_REQUIRES|ensure|EXPECT)\w*\s*\(")
B_TYPES = {
    "null_check": re.compile(r"\bNULL\b|\bnullptr\b|!\s*[A-Za-z_][\w>.\-]*\s*[)&|]|==\s*0\s*\)|IS_ERR|IS_ERR_OR_NULL"),
    "bounds_check": re.compile(r"(<=?|>=?)\s*[^;]*\b(\w*(len|size|count|num|max|min|cap|end|limit|bound|length|nr|idx|"
                               r"index|offset|remaining|avail|space|width|height)\w*)\b|sizeof|ARRAY_SIZE|\.size\(\)|"
                               r"\.length\(\)", re.I),
    "overflow": re.compile(r"overflow|_MAX\b|_MIN\b|\bsize_t\b|\buint\d*_t\b|\bunsigned\b|\bu64\b|\bint64_t\b|checked_|"
                           r"SafeAdd|CheckedNumeric|base::Checked|mul_overflow|add_overflow", re.I),
    "init": re.compile(r"memset|bzero|=\s*\{\s*0?\s*\}|calloc|kzalloc|zalloc|=\s*0\s*;|=\s*NULL\s*;|=\s*nullptr\s*;|"
                       r"\binit\w*\(|explicit_bzero|memzero", re.I),
    "release": re.compile(r"\b\w*(free|unlock|release|put|unref|close|destroy|cleanup|unregister|dispose)\w*\s*\(|"
                          r"\bdelete\b|mutex_unlock|spin_unlock", re.I),
    "error_path": re.compile(r"return\s+-?E[A-Z]+|return\s+-\d|return\s+(false|NULL|nullptr|err|ret|error|status|rc)\b|"
                             r"\bgoto\s+\w+|(ret|err|rc|error|status)\s*(<|!=|==)\s*0|<\s*0\s*\)|\.ok\(\)|!\s*ok\b",
                             re.I),
    "lifetime": re.compile(r"refcount|\bget_\w*\(|\bref\w*\(|kref|scoped_refptr|WeakPtr|unique_ptr|shared_ptr|"
                           r"std::move|Retain|AddRef|base::Unretained|RefPtr|Persistent", re.I),
}


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def strip_comments(code):
    code = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


def norm(line):
    return re.sub(r"\s+", " ", line).strip()


def trivial_norm(code):
    """text with comments, whitespace, const and identifier names removed: equal for both versions => trivial fix"""
    t = re.sub(r"\bconst\b", "", strip_comments(code))
    t = ID.sub(lambda m: m.group(0) if m.group(0) in KEYWORDS else "x", t)
    return re.sub(r"\s+", "", t)


def params(code):
    head = code.split("{", 1)[0]
    m = re.search(r"\((.*)\)", head, re.S)
    out = set()
    if m:
        for p in m.group(1).split(","):
            ids = [i for i in ID.findall(p) if i not in KEYWORDS]
            if ids:
                out.add(ids[-1])
    return out


def call_assigned(lines):
    """identifiers assigned from a function call in these lines: x = foo(...), x = a->b(...)"""
    out = set()
    for l in lines:
        for m in re.finditer(r"([A-Za-z_]\w*)\s*=\s*(\([^)]*\)\s*)?[A-Za-z_][\w:.>\-]*\s*\(", l):
            if m.group(1) not in KEYWORDS:
                out.add(m.group(1))
    return out


def condition_ids(line):
    """identifiers tested by a check line, with their access shape: (name, is_field, is_call)"""
    m = re.search(r"\((.*)\)", line)
    expr = m.group(1) if m else line
    out = []
    for t in re.finditer(r"([A-Za-z_]\w*)(\s*(->|\.)\s*[A-Za-z_]\w*)*(\s*\()?", expr):
        name = t.group(1)
        if name in KEYWORDS:
            continue
        out.append((name, bool(t.group(2)), bool(t.group(4))))
    return out


def classify(v, f):
    a = [norm(x) for x in strip_comments(v).split("\n")]
    b = [norm(x) for x in strip_comments(f).split("\n")]
    a_keep = [x for x in a if x]
    b_keep = [x for x in b if x]
    added, deleted, same = [], [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a_keep, b_keep, autojunk=False).get_opcodes():
        if op == "equal":
            same += a_keep[i1:i2]
        if op in ("replace", "delete"):
            deleted += a_keep[i1:i2]
        if op in ("replace", "insert"):
            added += b_keep[j1:j2]
    r = {"n_added": len(added), "n_deleted": len(deleted)}
    if trivial_norm(v) == trivial_norm(f) or (not added and not deleted):
        r["kind"] = "trivial"
    else:
        r["kind"] = "add_only" if added and not deleted else "del_only" if deleted and not added else "modify"
    # B types: what f's new lines bring that the deleted lines did not already have
    new_text = "\n".join(added)
    old_text = "\n".join(deleted)
    r["b_types"] = sorted(k for k, rx in B_TYPES.items() if rx.search(new_text) and
                          len(rx.findall(new_text)) > len(rx.findall(old_text))) if r["kind"] != "trivial" else []
    if r["kind"] != "trivial" and not r["b_types"]:
        r["b_types"] = ["other"]
    # A->B: a new check over identifiers that the vulnerable version already uses in its unchanged code
    old_lines = set(deleted)
    checks = [l for l in added if CHECK.search(l) and l not in old_lines]
    same_ids = {i for l in same for i in ID.findall(l)} - KEYWORDS
    pv, called = params(v), call_assigned(a_keep)
    tested = [t for l in checks for t in condition_ids(l)]
    tested_a = [t for t in tested if t[0] in same_ids or t[2]]
    r["n_checks"] = len(checks)
    r["a_to_b"] = bool(tested_a)
    src = set()
    for name, field, call in tested_a:
        if call or name in called:
            src.add("callee")
        elif name in pv:
            src.add("param")
        elif field:
            src.add("field")
        elif name.isupper() and len(name) > 1:
            src.add("macro")
        else:
            src.add("local")
    r["outside"] = sorted(src)
    r["needs_outside"] = bool(src & {"callee", "param"})
    r["checks"] = checks[:4]
    return r


def pairs(paired):
    for k in range(0, len(paired) - 1, 2):
        v, f = paired[k], paired[k + 1]
        if v["commit_id"] != f["commit_id"] or {v["target"], f["target"]} != {0, 1}:
            continue
        if v["target"] == 0:
            v, f = f, v
        yield v, f


def summarise(rows):
    n = len(rows)
    pct = lambda c: round(100 * c / max(n, 1), 1)
    kind = Counter(r["kind"] for r in rows)
    bt = Counter(t for r in rows for t in r["b_types"])
    out_src = Counter(t for r in rows if r["a_to_b"] for t in r["outside"])
    ab = sum(r["a_to_b"] for r in rows)
    need = sum(r["needs_outside"] for r in rows)
    return {"pairs": n,
            "kind_pct": {k: pct(c) for k, c in kind.most_common()},
            "b_type_pct": {k: pct(c) for k, c in bt.most_common()},
            "a_to_b_pct": pct(ab),
            "a_to_b_local_only_pct": pct(sum(r["a_to_b"] and r["outside"] == ["local"] for r in rows)),
            "a_to_b_needs_outside_pct": pct(need),
            "a_to_b_field_or_macro_only_pct": pct(sum(r["a_to_b"] and not r["needs_outside"] and
                                                      bool(set(r["outside"]) & {"field", "macro"}) for r in rows)),
            "outside_source_pct_of_a_to_b": {k: round(100 * c / max(ab, 1), 1) for k, c in out_src.most_common()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res, ex = {}, []
    big = {"linux", "Chrome", "tensorflow", "Android", "ImageMagick", "vim", "php-src", "FFmpeg"}
    for split in ("train", "valid", "test"):
        rows = []
        for v, f in pairs(load_jsonl(os.path.join(a.primevul, f"primevul_{split}_paired.jsonl"))):
            r = classify(v["func"], f["func"])
            r.update(idx=v["idx"], project=v.get("project", ""), commit_id=v["commit_id"], cwe=v.get("cwe", ""))
            rows.append(r)
        res[split] = summarise(rows)
        groups = defaultdict(list)
        for r in rows:
            groups[r["project"] if r["project"] in big else "other"].append(r)
        res[split]["by_project"] = {g: {"pairs": len(rs), "a_to_b_pct": summarise(rs)["a_to_b_pct"],
                                        "needs_outside_pct": summarise(rs)["a_to_b_needs_outside_pct"]}
                                    for g, rs in sorted(groups.items(), key=lambda x: -len(x[1]))}
        with open(os.path.join(a.out, f"{split}_pairs.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, json.dumps({k: v for k, v in res[split].items() if k != "by_project"}, ensure_ascii=False))
        if split == "test":
            rng = random.Random(0)
            src = {r["idx"]: (v, f) for v, f in pairs(load_jsonl(os.path.join(a.primevul, "primevul_test_paired.jsonl")))
                   for r in [{"idx": v["idx"]}]}
            for cls, sel in (("needs_outside", lambda r: r["needs_outside"]),
                             ("a_to_b_local_field_macro", lambda r: r["a_to_b"] and not r["needs_outside"]),
                             ("not_a_to_b", lambda r: not r["a_to_b"] and r["kind"] != "trivial"),
                             ("trivial", lambda r: r["kind"] == "trivial")):
                pool = [r for r in rows if sel(r)]
                for r in rng.sample(pool, min(a.examples, len(pool))):
                    v, f = src[r["idx"]]
                    diff = "\n".join(difflib.unified_diff(v["func"].split("\n"), f["func"].split("\n"), lineterm="", n=2))
                    ex.append(f"### {cls} | {r['project']} idx {r['idx']} | kind {r['kind']} | B {r['b_types']} | "
                              f"outside {r['outside']}\n\n```diff\n{diff[:3000]}\n```\n")
    with open(os.path.join(a.out, "patch_taxonomy.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "patch_taxonomy_examples.md"), "w", encoding="utf-8") as fh:
        fh.write("# Patch taxonomy examples (PrimeVul test, random per class, seed 0)\n\n" + "\n".join(ex))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
