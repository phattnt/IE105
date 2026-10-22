"""Counterfactual guard probe: does a trained exp5 model react to a guard line being added or removed?

Two edits on SecVulEval test functions, scored with a saved checkpoint (no training):

  insert  (vulnerable functions with flaw lines)  put one line right above the first flaw line:
          G1  related guard    `if (!IDENT) return;`  with IDENT taken from the flaw line
          G2  unrelated guard  `if (error) return;`    (identifier absent from the function)
          N   neutral line     `/* checked */`
          -> change of the flaw line's logit / rank and of the function score, per edit
  delete  (benign functions)  remove an existing single-line or 3-line `if (...) return/break/continue/goto;`
          guard; the victim is the next code line. Control: remove a plain statement line instead.
          -> change of the victim line's logit and of the function score

If the model used guard-sink relations, G1 should lower the flaw line far more than N / G2, and removing a real
guard should raise its victim far more than removing a plain statement.

    python probe_guards.py --ckpt ../../data/colab_ckpt/<run>/ckpt_best.pt --model microsoft/unixcoder-base \
        --max_len 1024 --data ../../data/secvul_lines [--n 250] [--out ../../data/probe_guards_<run>.json]
"""
import argparse
import json
import random
import re
import sys
import time

import numpy as np

import train_lines as tl

KEYWORDS = set("""auto break case char const continue default do double else enum extern float for goto if inline int long
register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while true false
NULL bool class new delete this public private protected template typename namespace using""".split())
GUARD1 = re.compile(r"^\s*if\s*\(.*\)\s*(return\b[^;]*;|break;|continue;|goto\s+\w+;)\s*$")
GUARD_IF = re.compile(r"^\s*if\s*\(.*\)\s*\{?\s*$")
JUMP = re.compile(r"^\s*(return\b[^;]*;|break;|continue;|goto\s+\w+;)\s*$")
STMT = re.compile(r"^\s*[A-Za-z_][\w\->\.\[\]\*\(\)\s,]*=[^=].*;\s*$")   # plain assignment statement


def idents(line):
    out = []
    for m in re.finditer(r"[A-Za-z_]\w*", line):
        w = m.group(0)
        if w in KEYWORDS or line[m.end():m.end() + 1].lstrip().startswith("("):
            continue
        tail = line[m.end():m.end() + 2]
        head = line[max(0, m.start() - 1):m.start()]
        score = 2 if ("->" in tail or tail.startswith("[") or head == "*") else 1
        out.append((score, w))
    out.sort(key=lambda x: -x[0])
    return [w for _, w in out]


def indent_of(line):
    return re.match(r"^\s*", line).group(0)


def next_code_line(lines, j):
    for k in range(j, len(lines)):
        s = lines[k].strip()
        if s and s not in ("{", "}") and not s.startswith("//") and not s.startswith("/*"):
            return k
    return None


def insert_cases(rows, n, rng):
    """-> list of (row, flaw_line, variants{name: (func_text, flaw_line_after_edit)})"""
    cands = [r for r in rows if r["target"] == 1 and r["flaw_lines"]]
    rng.shuffle(cands)
    out = []
    for r in cands:
        lines = r["func"].split("\n")
        fls = sorted(l for l in r["flaw_lines"] if l < len(lines) and lines[l].strip() not in ("", "{", "}"))
        if not fls or len(lines) > 400:
            continue
        t = fls[0]
        ids_ = idents(lines[t])
        if not ids_:
            continue
        ind = indent_of(lines[t])
        edits = {"G1": f"{ind}if (!{ids_[0]}) return;", "G2": f"{ind}if (error) return;", "N": f"{ind}/* checked */"}
        variants = {"orig": ("\n".join(lines), t)}
        for name, new in edits.items():
            variants[name] = ("\n".join(lines[:t] + [new] + lines[t:]), t + 1)
        out.append((r, t, variants))
        if len(out) >= n:
            break
    return out


def find_guards(lines):
    """-> list of (start, end_exclusive, victim_line_index_in_original, guard_text)"""
    out = []
    for j, l in enumerate(lines):
        if GUARD1.match(l):
            v = next_code_line(lines, j + 1)
            if v is not None:
                out.append((j, j + 1, v, l))
        elif GUARD_IF.match(l) and j + 2 < len(lines) and JUMP.match(lines[j + 1]) and lines[j + 2].strip() == "}":
            v = next_code_line(lines, j + 3)
            if v is not None:
                out.append((j, j + 3, v, l))
        elif GUARD_IF.match(l) and not l.rstrip().endswith("{") and j + 1 < len(lines) and JUMP.match(lines[j + 1]):
            v = next_code_line(lines, j + 2)
            if v is not None:
                out.append((j, j + 2, v, l))
    return out


def delete_cases(rows, n, rng):
    """-> list of (row, variants{name: (func_text, victim_line_after_edit)}, related_flag)"""
    cands = [r for r in rows if r["target"] == 0]
    rng.shuffle(cands)
    out = []
    for r in cands:
        lines = r["func"].split("\n")
        if len(lines) > 400:
            continue
        guards = find_guards(lines)
        if not guards:
            continue
        a, b, v, gtext = rng.choice(guards)
        victim_ids = set(idents(lines[v]))
        related = any(w in victim_ids for w in idents(gtext))
        stmts = [j for j, l in enumerate(lines) if STMT.match(l) and not (a <= j < b) and j != v
                 and next_code_line(lines, j + 1) is not None]
        if not stmts:
            continue
        c = rng.choice(stmts)
        cv = next_code_line(lines, c + 1)
        variants = {"orig": ("\n".join(lines), v),
                    "del_guard": ("\n".join(lines[:a] + lines[b:]), v - (b - a)),
                    "del_stmt": ("\n".join(lines[:c] + lines[c + 1:]), cv - 1), "orig_stmt": ("\n".join(lines), cv)}
        out.append((r, variants, related, (b - a)))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_len", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--eval_tokens", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import transformers
    # The checkpoint holds every weight, so build the encoder from its config only (no pretrained download, and
    # no torch.load of a .bin from the Hub, which transformers refuses on torch < 2.6).
    @classmethod
    def from_config_only(cls, name, *args, **kw):
        cfg = transformers.AutoConfig.from_pretrained(name)
        if "modernbert" in name.lower():
            cfg.reference_compile = False     # no Triton on Windows
        return cls.from_config(cfg, **kw)

    transformers.AutoModel.from_pretrained = from_config_only
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    tl.PAD_ID = tok.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tl.make_model(a.model)
    state = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.to(device).eval()
    tl.log("loaded", a.ckpt, "on", device)

    rows = tl.read_split(a.data, "test")
    rng = random.Random(a.seed)
    ins = insert_cases(rows, a.n, rng)
    dele = delete_cases(rows, a.n, rng)
    tl.log(f"insert cases {len(ins)} | delete cases {len(dele)}")

    def score(texts):
        rs = [{"func": t, "n_lines": t.count("\n") + 1, "flaw_lines": []} for t in texts]
        ls, fn = tl.score_rows(model, tok, rs, "slide", a, device)
        return ls, fn

    t0 = time.time()
    # ---- insert
    names = ["orig", "G1", "G2", "N"]
    texts = [v[nm][0] for _, _, v in ins for nm in names]
    ls, fn = score(texts)
    ins_rows = []
    for i, (r, t, v) in enumerate(ins):
        rec = {"idx": r["idx"], "n_lines": r["n_lines"]}
        for k, nm in enumerate(names):
            s, line = ls[i * len(names) + k], v[nm][1]
            rank = int((s > s[line]).sum())
            rec[nm] = {"flaw_logit": float(s[line]), "flaw_rank": rank, "fn": float(fn[i * len(names) + k]),
                       "inserted_logit": None if nm == "orig" else float(s[line - 1])}
        ins_rows.append(rec)
    tl.log(f"insert scored in {time.time() - t0:.0f}s")

    # ---- delete
    t0 = time.time()
    dnames = ["orig", "del_guard", "del_stmt", "orig_stmt"]
    texts = [v[nm][0] for _, v, _, _ in dele for nm in dnames]
    ls, fn = score(texts)
    del_rows = []
    for i, (r, v, related, glen) in enumerate(dele):
        rec = {"idx": r["idx"], "n_lines": r["n_lines"], "related": related, "guard_lines": glen}
        for k, nm in enumerate(dnames):
            s, line = ls[i * len(dnames) + k], v[nm][1]
            rec[nm] = {"victim_logit": float(s[line]), "victim_rank": int((s > s[line]).sum()),
                       "fn": float(fn[i * len(dnames) + k])}
        del_rows.append(rec)
    tl.log(f"delete scored in {time.time() - t0:.0f}s")

    # ---- summary
    def ci(x):
        x = np.asarray(x, float)
        if len(x) < 2:
            return (float("nan"), float("nan"))
        b = [np.mean(np.random.default_rng(s).choice(x, len(x))) for s in range(1000)]
        return tuple(float(q) for q in np.percentile(b, [2.5, 97.5]))

    summ = {"model": a.model, "ckpt": a.ckpt, "n_insert": len(ins_rows), "n_delete": len(del_rows), "insert": {}, "delete": {}}
    print(f"\n## chèn trên dòng lỗi (n={len(ins_rows)}) — thay đổi so với bản gốc")
    print("| chèn | Δ logit dòng lỗi | CI95 | % giảm > 0,5 | Δ hạng dòng lỗi | Δ điểm hàm | Top-10 trước→sau |")
    print("|---|---|---|---|---|---|---|")
    for nm in ["G1", "G2", "N"]:
        d = [r[nm]["flaw_logit"] - r["orig"]["flaw_logit"] for r in ins_rows]
        dr = [r[nm]["flaw_rank"] - r["orig"]["flaw_rank"] for r in ins_rows]
        df = [r[nm]["fn"] - r["orig"]["fn"] for r in ins_rows]
        t10a = np.mean([r["orig"]["flaw_rank"] < 10 for r in ins_rows]) * 100
        t10b = np.mean([r[nm]["flaw_rank"] < 10 for r in ins_rows]) * 100
        lo, hi = ci(d)
        summ["insert"][nm] = {"d_logit": float(np.mean(d)), "ci": [lo, hi], "pct_drop": float(np.mean(np.asarray(d) < -0.5) * 100),
                              "d_rank": float(np.mean(dr)), "d_fn": float(np.mean(df)), "top10_before": float(t10a), "top10_after": float(t10b)}
        print(f"| {nm} | {np.mean(d):+.2f} | [{lo:+.2f}; {hi:+.2f}] | {np.mean(np.asarray(d) < -0.5) * 100:.0f}% | {np.mean(dr):+.1f} | {np.mean(df):+.2f} | {t10a:.0f} → {t10b:.0f} |")
    g1n = [r["G1"]["flaw_logit"] - r["N"]["flaw_logit"] for r in ins_rows]
    lo, hi = ci(g1n)
    summ["insert"]["G1_minus_N"] = {"d_logit": float(np.mean(g1n)), "ci": [lo, hi]}
    print(f"G1 − N (phần riêng của guard liên quan): {np.mean(g1n):+.2f} [{lo:+.2f}; {hi:+.2f}]")

    print(f"\n## xóa trong hàm lành (n={len(del_rows)}) — thay đổi của dòng ngay sau chỗ xóa")
    print("| xóa | Δ logit dòng nạn nhân | CI95 | % tăng > 0,5 | Δ điểm hàm | n |")
    print("|---|---|---|---|---|---|")
    for label, sel in [("guard (tất cả)", del_rows), ("guard liên quan", [r for r in del_rows if r["related"]]),
                       ("guard không liên quan", [r for r in del_rows if not r["related"]])]:
        if not sel:
            continue
        d = [r["del_guard"]["victim_logit"] - r["orig"]["victim_logit"] for r in sel]
        df = [r["del_guard"]["fn"] - r["orig"]["fn"] for r in sel]
        lo, hi = ci(d)
        summ["delete"][label] = {"d_logit": float(np.mean(d)), "ci": [lo, hi], "pct_up": float(np.mean(np.asarray(d) > 0.5) * 100), "d_fn": float(np.mean(df)), "n": len(sel)}
        print(f"| {label} | {np.mean(d):+.2f} | [{lo:+.2f}; {hi:+.2f}] | {np.mean(np.asarray(d) > 0.5) * 100:.0f}% | {np.mean(df):+.2f} | {len(sel)} |")
    d = [r["del_stmt"]["victim_logit"] - r["orig_stmt"]["victim_logit"] for r in del_rows]
    df = [r["del_stmt"]["fn"] - r["orig"]["fn"] for r in del_rows]
    lo, hi = ci(d)
    summ["delete"]["stmt"] = {"d_logit": float(np.mean(d)), "ci": [lo, hi], "pct_up": float(np.mean(np.asarray(d) > 0.5) * 100), "d_fn": float(np.mean(df)), "n": len(del_rows)}
    print(f"| câu lệnh thường (đối chứng) | {np.mean(d):+.2f} | [{lo:+.2f}; {hi:+.2f}] | {np.mean(np.asarray(d) > 0.5) * 100:.0f}% | {np.mean(df):+.2f} | {len(del_rows)} |")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"summary": summ, "insert": ins_rows, "delete": del_rows}, fh, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
