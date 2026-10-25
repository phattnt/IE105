"""Phase 2 inputs: cross-function context for PrimeVul pairs, taken from the parent of the fixing commit.

For every pair (vulnerable v, fixed f) the context is built ONCE from the repository at commit^ (the code the
vulnerable function lived in, before any part of the fix), so both versions of a pair get exactly the same context and
only their bodies differ:
  callees  definitions (or prototypes / macros) of the functions the vulnerable version calls
  callers  call sites of the function itself (enclosing function name + a few lines around the call)
searched in a bounded file set: the C/C++ files the commit touches, the headers in their directories and the headers
they #include (one level, resolved against the commit^ tree). No repository is cloned: GitHub's API gives the parent
commit, the changed files and the tree, and raw.githubusercontent.com gives the file contents at commit^.
A control context is built from the same files and the same budget: definitions of functions the vulnerable version
does NOT call, and lines around random other calls ("fake"), so a gain of real over fake context is not "more text".

Stages (each cached under --cache, safe to rerun):
  repos   project -> one GitHub repository containing the commits (search API, 30 requests/min)
  fetch   commit^ sha, changed files, tree, and the bounded file set's contents
  build   writes <out>/ctx_<split>_pairs.jsonl: llm_audit input rows (idx, target, func, pair_idx, flaw_lines,
          line_scores, chars) + context, context_fake, ctx (coverage stats)

    python build_context.py --primevul ../../data/primevul --cache ../../data/ctx --splits valid test --out inputs
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request

SRC_EXT = (".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".inc", ".ipp", ".cu", ".m", ".mm")
KEYWORDS = set("""if else while for do switch case default return goto break continue sizeof typeof alignof decltype
static_cast dynamic_cast reinterpret_cast const_cast defined new delete throw catch try __attribute__ __typeof__
unlikely likely offsetof va_arg va_start va_end assert""".split())
MAX_FILES = 30
MAX_FILE_BYTES = 600_000


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def gh(path, tries=4):
    """GET a GitHub API path with the gh CLI; None on 404/422"""
    for k in range(tries):
        p = subprocess.run(["gh", "api", "-X", "GET", path], capture_output=True, text=True, encoding="utf-8")
        if p.returncode == 0:
            return json.loads(p.stdout)
        err = p.stderr + p.stdout
        if "HTTP 404" in err or "HTTP 422" in err or "HTTP 409" in err:
            return None
        if "rate limit" in err.lower() or "HTTP 403" in err or "HTTP 429" in err:
            time.sleep(65)
            continue
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"gh api {path}: {err[:300]}")


def pairs_of(paired):
    for k in range(0, len(paired) - 1, 2):
        x, y = paired[k], paired[k + 1]
        if x["commit_id"] == y["commit_id"] and {x["target"], y["target"]} == {0, 1}:
            yield (x, y) if x["target"] == 1 else (y, x)


# ---------------------------------------------------------------- repos
def stage_repos(pairs, cache):
    path = os.path.join(cache, "repos.json")
    repos = json.load(open(path)) if os.path.exists(path) else {}
    by_proj = {}
    for v, _ in pairs:
        by_proj.setdefault(v["project"], []).append(v["commit_id"])
    todo = [p for p in by_proj if p not in repos]
    print(f"[repos] {len(by_proj)} projects, {len(todo)} to look up", flush=True)
    for n, proj in enumerate(todo, 1):
        found = None
        for sha in by_proj[proj][:3]:
            r = gh(f"search/commits?q=hash:{sha}&per_page=50")
            time.sleep(2.1)                                         # search API: 30 requests / minute
            items = (r or {}).get("items", [])
            if not items:
                continue
            name = proj.lower().replace("_", "-")
            items.sort(key=lambda it: (it["repository"]["name"].lower() != name, it["repository"].get("fork", True)))
            found = items[0]["repository"]["full_name"]
            break
        repos[proj] = found
        print(f"[repos] {n}/{len(todo)} {proj} -> {found}", flush=True)
        json.dump(repos, open(path, "w"), indent=1)
    return repos


# ---------------------------------------------------------------- fetch
def raw(repo, ref, path, dest):
    if os.path.exists(dest):
        return True
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{urllib.request.quote(path)}"
    for k in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                data = data[:MAX_FILE_BYTES]
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return False


def resolve_includes(text, here, tree_set, tree_by_name):
    out = []
    for m in re.finditer(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', text, re.M):
        inc = m.group(1)
        cands = [os.path.normpath(os.path.join(os.path.dirname(here), inc)).replace("\\", "/"), inc,
                 f"include/{inc}", f"src/{inc}", f"lib/{inc}"]
        hit = next((c for c in cands if c in tree_set), None)
        if hit is None:
            same = [p for p in tree_by_name.get(os.path.basename(inc), []) if p.endswith(inc)]
            hit = same[0] if len(same) == 1 else None
        if hit:
            out.append(hit)
    return out


def stage_fetch_one(repo, sha, cache):
    meta_p = os.path.join(cache, "commits", f"{sha}.json")
    if os.path.exists(meta_p):
        return json.load(open(meta_p))
    c = gh(f"repos/{repo}/commits/{sha}")
    if not c:                                                       # a stale fork: find a repository that has this commit
        r = gh(f"search/commits?q=hash:{sha}&per_page=20")
        time.sleep(2.1)
        items = sorted((r or {}).get("items", []), key=lambda it: it["repository"].get("fork", True))
        if items:
            repo = items[0]["repository"]["full_name"]
            c = gh(f"repos/{repo}/commits/{sha}")
    if not c or not c.get("parents"):
        meta = {"sha": sha, "repo": repo, "ok": False, "why": "commit not found" if not c else "no parent"}
        json.dump(meta, open(meta_p, "w"))
        return meta
    parent = c["parents"][0]["sha"]
    changed = [f["filename"] for f in c.get("files", []) if f["filename"].endswith(SRC_EXT) and f.get("status") != "added"]
    t = gh(f"repos/{repo}/git/trees/{parent}?recursive=1")
    tree = [e["path"] for e in (t or {}).get("tree", []) if e.get("type") == "blob" and e["path"].endswith(SRC_EXT)]
    tree_set, tree_by_name = set(tree), {}
    for p in tree:
        tree_by_name.setdefault(os.path.basename(p), []).append(p)
    fdir = os.path.join(cache, "files", parent)
    files, seen = [], set()

    def add(p):
        if p not in seen and len(files) < MAX_FILES:
            seen.add(p)
            files.append(p)
    for p in changed:
        add(p)
    ok_changed = [p for p in changed if raw(repo, parent, p, os.path.join(fdir, p))]
    for p in ok_changed:                                            # headers they include, then same-directory headers
        text = open(os.path.join(fdir, p), encoding="utf-8", errors="replace").read()
        for h in resolve_includes(text, p, tree_set, tree_by_name):
            add(h)
    for p in ok_changed:
        d = os.path.dirname(p)
        for q in sorted(x for x in tree if os.path.dirname(x) == d and x.endswith((".h", ".hh", ".hpp"))):
            add(q)
    with cf.ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda p: raw(repo, parent, p, os.path.join(fdir, p)), files))
    meta = {"sha": sha, "repo": repo, "ok": True, "parent": parent, "changed": changed,
            "files": [p for p, g in zip(files, got) if g], "tree_truncated": bool((t or {}).get("truncated")),
            "tree_files": len(tree)}
    json.dump(meta, open(meta_p, "w"))
    return meta


def stage_fetch(pairs, repos, cache):
    os.makedirs(os.path.join(cache, "commits"), exist_ok=True)
    commits = sorted({(repos.get(v["project"]), v["commit_id"]) for v, _ in pairs}, key=lambda x: (x[0] or "", x[1]))
    print(f"[fetch] {len(commits)} commits", flush=True)
    out, t0 = {}, time.time()

    def one(rs):
        repo, sha = rs
        if not repo:
            return sha, {"sha": sha, "ok": False, "why": "no repository"}
        try:
            return sha, stage_fetch_one(repo, sha, cache)
        except Exception as e:                                      # keep going; the commit is retried on the next run
            print(f"[fetch] {sha[:10]} {repo}: {e!r}"[:300], flush=True)
            return sha, {"sha": sha, "ok": False, "why": repr(e)[:200]}
    with cf.ThreadPoolExecutor(4) as ex:
        for n, (sha, meta) in enumerate(ex.map(one, commits), 1):
            out[sha] = meta
            if n % 10 == 0 or n == len(commits):
                ok = sum(m.get("ok", False) for m in out.values())
                print(f"[fetch] {n}/{len(commits)} | ok {ok} | {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


# ---------------------------------------------------------------- build
def diff_flaw_lines(vul, fix):
    """eval_primevul.diff_flaw_lines (copied: that module imports torch): lines the fix deletes or changes"""
    import difflib
    a, b = vul.split("\n"), fix.split("\n")
    out = []
    for op, i1, i2, _, _ in difflib.SequenceMatcher(None, [x.strip() for x in a], [x.strip() for x in b],
                                                    autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            out += [i for i in range(i1, i2) if a[i].strip() and not a[i].strip().startswith(("//", "/*", "*", "*/"))]
    return out


def strip_comments_keep_lines(code):
    code = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


def func_name(func):
    head = strip_comments_keep_lines(func).split("{", 1)[0]
    m = re.findall(r"([A-Za-z_~][\w:~]*)\s*\(", head)
    return m[-1].split("::")[-1] if m else None


def callees(func, own):
    body = strip_comments_keep_lines(func)
    body = body.split("{", 1)[1] if "{" in body else body
    body = re.sub(r'"(\\.|[^"\\])*"', '""', body)
    out = []
    for m in re.finditer(r"([A-Za-z_][\w]*)\s*\(", body):
        n = m.group(1)
        if n not in KEYWORDS and n != own and n not in out and not n.isdigit():
            out.append(n)
    return out


def block_end(lines, i, max_lines):
    """index after the brace block that starts at or after line i (bounded)"""
    depth, opened = 0, False
    for j in range(i, min(len(lines), i + 400)):
        s = re.sub(r'"(\\.|[^"\\])*"|\'(\\.|[^\'\\])*\'', "", lines[j])
        depth += s.count("{") - s.count("}")
        opened = opened or "{" in s
        if opened and depth <= 0:
            return j + 1
        if not opened and s.rstrip().endswith(";"):
            return j + 1
    return min(len(lines), i + max_lines)


class Index:
    """definitions, prototypes and macros by name over a set of files"""

    def __init__(self, files):
        self.files, self.defs, self.calls = files, {}, {}
        for path, text in files.items():
            lines = text.split("\n")
            clean = strip_comments_keep_lines(text).split("\n")
            for i, l in enumerate(clean):
                m = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)\(", l)
                if m:
                    j = i
                    while j < len(lines) - 1 and lines[j].rstrip().endswith("\\") and j - i < 25:
                        j += 1
                    self.defs.setdefault(m.group(1), []).append(("macro", path, i, j + 1))
                    continue
                if not l or l[0] in " \t#{}" or l.rstrip().endswith((";", ",")) and "(" not in l:
                    continue
                m = re.match(r"[^=;]*?\b([A-Za-z_~][\w:~]*)\s*\(", l)
                if not m or m.group(1) in KEYWORDS:
                    continue
                name = m.group(1).split("::")[-1]
                tail = "\n".join(clean[i:i + 8])
                cut = re.search(r"[;{]", tail)
                kind = "def" if cut and cut.group(0) == "{" else "proto"
                end = block_end(lines, i, 60) if kind == "def" else i + 1 + (tail[:cut.start()].count("\n") if cut else 0)
                self.defs.setdefault(name, []).append((kind, path, i, max(end, i + 1)))
            for i, l in enumerate(clean):
                for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", l):
                    if l[0:1] in " \t":                                 # a call is indented code, not a definition
                        self.calls.setdefault(m.group(1), []).append((path, i))

    def snippet(self, path, a, b, max_lines):
        lines = self.files[path].split("\n")
        part = lines[a:min(b, a + max_lines)]
        more = b - a > max_lines
        return "\n".join(part) + ("\n    /* ... */" if more else "")

    def enclosing(self, path, i):
        best = None
        for name, ds in self.defs.items():
            for kind, p, a, b in ds:
                if p == path and kind == "def" and a <= i < b and (best is None or a > best[1]):
                    best = (name, a)
        return best[0] if best else None


def render_defs(idx, names, budget, max_lines=30):
    parts, used, found = [], 0, 0
    for n in names:
        ds = idx.defs.get(n)
        if not ds:
            continue
        kind, path, a, b = sorted(ds, key=lambda d: {"def": 0, "macro": 1, "proto": 2}[d[0]])[0]
        s = f"// {kind} {n} ({path})\n" + idx.snippet(path, a, b, max_lines)
        if used + len(s) > budget:
            continue
        parts.append(s)
        used += len(s)
        found += 1
    return parts, used, found


def render_calls(idx, sites, budget, own_def=None, k=3, around=4):
    parts, used = [], 0
    for path, i in sites:
        if own_def and own_def[0] == path and own_def[1] <= i < own_def[2]:
            continue
        caller = idx.enclosing(path, i)
        lines = idx.files[path].split("\n")
        a, b = max(0, i - around), i + around + 1
        if own_def and own_def[0] == path:                          # never show any line of the function itself
            if own_def[1] <= a < own_def[2]:
                a = own_def[2]
            if a < own_def[1] < b:
                b = own_def[1]
        if a > i or b <= i:
            continue
        s = f"// call site in {caller or '?'} ({path}:{i + 1})\n" + "\n".join(lines[a:b])
        if used + len(s) > budget:
            break
        parts.append(s)
        used += len(s)
        if len(parts) >= k:
            break
    return parts, used


def build_pair(v, meta, cache, budget, rng):
    stats = {"commit_ok": bool(meta.get("ok"))}
    if not meta.get("ok"):
        return "", "", stats
    fdir = os.path.join(cache, "files", meta["parent"])
    files = {}
    for p in meta["files"]:
        fp = os.path.join(fdir, p)
        if os.path.exists(fp):
            files[p] = open(fp, encoding="utf-8", errors="replace").read()
    idx = Index(files)
    own = func_name(v["func"])
    own_def = next(((p, a, b) for kind, p, a, b in idx.defs.get(own, []) if kind == "def"), None) if own else None
    called = callees(v["func"], own)
    resolved = [n for n in called if n in idx.defs]
    stats.update(files=len(files), own_found=own_def is not None, callees=len(called), callees_resolved=len(resolved))
    real_defs, used_d, found = render_defs(idx, resolved, int(budget * 0.7))
    sites = idx.calls.get(own, []) if own else []
    real_calls, used_c = render_calls(idx, sites, budget - used_d, own_def)
    stats.update(callee_defs_shown=found, callers_shown=len(real_calls))
    real = "\n\n".join(real_defs + real_calls)
    # control: same files, same budget, functions the vulnerable version does not call, random call sites of others
    others = [n for n, ds in idx.defs.items() if n not in set(called) and n != own and any(d[0] == "def" for d in ds)]
    rng.shuffle(others)
    fake_defs, used_f, _ = render_defs(idx, others, max(used_d, 1), max_lines=30) if used_d else ([], 0, 0)
    other_sites = [s for n, ss in idx.calls.items() if n != own for s in ss]
    rng.shuffle(other_sites)
    fake_calls, _ = render_calls(idx, other_sites, max(len(real) - used_f, 0) if real_calls else 0, own_def,
                                 k=len(real_calls)) if real_calls else ([], 0)
    fake = "\n\n".join(fake_defs + fake_calls)
    stats.update(chars=len(real), fake_chars=len(fake))
    return real, fake, stats


def stage_build(split_pairs, metas, cache, out_dir, budget):
    os.makedirs(out_dir, exist_ok=True)
    for split, pairs in split_pairs.items():
        rows, agg = [], {}
        for v, f in pairs:
            rng = random.Random(int(v["idx"]))
            real, fake, st = build_pair(v, metas.get(v["commit_id"], {}), cache, budget, rng)
            fl = diff_flaw_lines(v["func"], f["func"])
            for r, other in ((v, f), (f, v)):
                rows.append({"idx": r["idx"], "target": r["target"], "func": r["func"], "pair_idx": other["idx"],
                             "flaw_lines": fl if r is v else [], "line_scores": [], "chars": len(r["func"]),
                             "project": v["project"],
                             "commit_id": v["commit_id"], "context": real, "context_fake": fake, "ctx": st})
            # leak check: a line that differs between the two versions must not occur in either context
            nv = {re.sub(r"\s+", "", l) for l in v["func"].split("\n")}
            nf = {re.sub(r"\s+", "", l) for l in f["func"].split("\n")}
            diff = {l for l in nv ^ nf if len(l) > 12}
            ctx_lines = {re.sub(r"\s+", "", l) for l in (real + "\n" + fake).split("\n")}
            st["diff_line_in_context"] = bool(diff & ctx_lines)
            for k, x in st.items():
                agg.setdefault(k, []).append(float(x))
        p = os.path.join(out_dir, f"ctx_{split}_pairs.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = len(pairs)
        summ = {k: round(sum(x) / n, 3) for k, x in agg.items()}
        summ["pairs"] = n
        summ["pairs_with_context"] = sum(1 for r in rows[::2] if r["context"])
        summ["pairs_with_callee_def"] = sum(1 for r in rows[::2] if r["ctx"].get("callee_defs_shown"))
        summ["pairs_with_caller"] = sum(1 for r in rows[::2] if r["ctx"].get("callers_shown"))
        print(f"[build] {split}: {json.dumps(summ)}", flush=True)
        json.dump(summ, open(os.path.join(out_dir, f"ctx_{split}_summary.json"), "w"), indent=1)


def ignore_early_ctrl_c(seconds=20):
    """the app's terminal panel may send a Ctrl+C right after typing the command (as colabctl.py does)"""
    import signal
    t0 = time.time()

    def handler(sig, frame):
        if time.time() - t0 < seconds:
            print(f"(ignored a Ctrl+C in the first {seconds} s; press it again to stop)", flush=True)
            return
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handler)


def main():
    ignore_early_ctrl_c()
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs"))
    ap.add_argument("--stages", nargs="+", default=["repos", "fetch", "build"])
    ap.add_argument("--budget", type=int, default=8000, help="context characters per pair (~2k tokens)")
    ap.add_argument("--limit", type=int, default=None, help="first N pairs per split (smoke)")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    split_pairs = {s: list(pairs_of(load_jsonl(os.path.join(a.primevul, f"primevul_{s}_paired.jsonl"))))[:a.limit]
                   for s in a.splits}
    allp = [p for ps in split_pairs.values() for p in ps]
    repos = stage_repos(allp, a.cache) if "repos" in a.stages else json.load(open(os.path.join(a.cache, "repos.json")))
    metas = stage_fetch(allp, repos, a.cache) if "fetch" in a.stages else {}
    if "build" in a.stages:
        if not metas:
            metas = {v["commit_id"]: json.load(open(os.path.join(a.cache, "commits", f"{v['commit_id']}.json")))
                     for v, _ in allp if os.path.exists(os.path.join(a.cache, "commits", f"{v['commit_id']}.json"))}
        stage_build(split_pairs, metas, a.cache, a.out, a.budget)


if __name__ == "__main__":
    main()
