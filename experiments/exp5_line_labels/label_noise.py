"""Spurious ("trivial") flaw-line labels in SecVulEval: detection rule, train/valid filter, statistics (CPU, no torch).

Line labels come from the patch (lines of the vulnerable version that the fix changes or deletes are 1), so some
labels are not the flaw at all: a moved `}`, a blank line, a comment, a lone `;`, an `} else {` that was
re-indented, or every line of a function that the fix rewrote wholesale (hermes idx 16033). Two filters, both off by
default in prepare_secvuleval.py and never applied to test:
  --drop_trivial_flaw       the label of a trivial line becomes 0 (the line stays in the function)
  --max_flaw_density X      a vulnerable function with more than X of its lines labelled is dropped together
                            with its paired fixed version (density measured on the labels as the patch gives them)

Trivial line = after removing comments, the line holds only whitespace, braces, punctuation `( ) [ ] ; ,`, or a bare
`else` between braces. Lines like `return -1;`, `free(p);`, `i++;`, `break;` are never trivial. Comments are found
by scanning the whole function (block comments span lines; string and char literals are skipped), so `*p = 0;` is
code while ` * text` inside a /* */ block is a comment.

Statistics (writes the json + markdown summary + random examples per rule):
    python label_noise.py --data ../../data/secvul_lines --out results
Unit tests: python -m unittest test_label_noise
"""
import argparse
import json
import os
import random
import re
import statistics
from collections import Counter

TRIVIAL_KINDS = ("blank", "comment", "brace", "punct", "else")
_PUNCT = set("()[];,")
_ELSE = re.compile(r"^\}?\s*else\s*\{?$")
# reported in the statistics only, never filtered: short lines that can be the real flaw
_BORDERLINE = [("break/continue", re.compile(r"^(break|continue)\s*;$")),
               ("return;", re.compile(r"^return\s*;$")),
               ("goto/label", re.compile(r"^(goto\s+\w+\s*;|\w+\s*:)$")),
               ("default:", re.compile(r"^default\s*:$")),
               ("#else/#endif", re.compile(r"^#\s*(else|endif)\b"))]


def split_code_comments(func):
    """Per line of `func`: (code text with comments removed, whether the line holds comment text)."""
    code, comment = [[]], [False]
    i, n, state = 0, len(func), None      # state: None, "block", "line", '"' or "'"
    while i < n:
        ch = func[i]
        if ch == "\n":
            if state == "line":
                state = None
            code.append([]); comment.append(False)
            i += 1
            continue
        if state in ("block", "line"):
            if state == "block" and func.startswith("*/", i):
                state = None
                i += 2
            else:
                i += 1
            if not ch.isspace():
                comment[-1] = True
            continue
        if state in ('"', "'"):
            code[-1].append(ch)
            if ch == "\\" and i + 1 < n and func[i + 1] != "\n":
                code[-1].append(func[i + 1])
                i += 2
                continue
            if ch == state:
                state = None
            i += 1
            continue
        if func.startswith("/*", i) or func.startswith("//", i):
            state = "block" if func[i + 1] == "*" else "line"
            comment[-1] = True
            i += 2
            continue
        if ch in "\"'":
            state = ch
        code[-1].append(ch)
        i += 1
    return [("".join(c), m) for c, m in zip(code, comment)]


def trivial_kind(code, has_comment=False):
    """Kind of trivial line ("blank", "comment", "brace", "punct", "else") or None if the line is real code."""
    s = code.strip()
    if not s:
        return "comment" if has_comment else "blank"
    if _ELSE.match(re.sub(r"\s+", " ", s)):
        return "else"
    chars = set(s.replace(" ", "").replace("\t", ""))
    if chars <= _PUNCT | set("{}"):
        return "brace" if chars & set("{}") else "punct"
    return None


def borderline_kind(code):
    s = code.strip()
    return next((name for name, rx in _BORDERLINE if rx.match(s)), None)


def line_kinds(func):
    """trivial_kind of every line of `func` (comments resolved across lines)."""
    return [trivial_kind(c, m) for c, m in split_code_comments(func)]


def is_trivial_line(line):
    """Single-line convenience wrapper (a line read on its own, so no comment context from earlier lines)."""
    return line_kinds(line)[0] is not None


def flaw_density(rec):
    return len(rec["flaw_lines"]) / max(rec["n_lines"], 1)


def drop_trivial_flaws(rec):
    """Return (kept flaw lines, {line: kind} of the labels removed)."""
    kinds = line_kinds(rec["func"])
    removed = {l: kinds[l] for l in rec["flaw_lines"] if l < len(kinds) and kinds[l]}
    return [l for l in rec["flaw_lines"] if l not in removed], removed


def filter_labels(recs, drop_trivial=False, max_density=None):
    """Apply the two filters to one split (train or valid; the caller never passes test). Returns (records, stats).
    A vulnerable function left with no flaw line gets flaw_lines = [] and is then treated exactly like a vulnerable
    function whose patch only adds lines (train_lines.py: dropped in plain mode, line BCE masked in pair mode)."""
    st = Counter()
    drop = set()
    if max_density is not None:
        for x in recs:
            if x["target"] == 1 and x["flaw_lines"] and flaw_density(x) > max_density:
                drop.add(x["idx"])
                if x["pair_idx"] is not None:
                    drop.add(x["pair_idx"])
                st["dense_vulnerable_dropped"] += 1
        st["functions_dropped_by_density"] = len(drop)
    out = []
    for x in recs:
        if x["idx"] in drop:
            continue
        if drop_trivial and x["target"] == 1 and x["flaw_lines"]:
            kept, removed = drop_trivial_flaws(x)
            if removed:
                x = dict(x, flaw_lines=kept)
                st["trivial_labels_removed"] += len(removed)
                st["functions_touched"] += 1
                st["vulnerable_left_without_lines"] += not kept
        out.append(x)
    return out, dict(st)


# ----------------------------------------------------------------------------- statistics
def _summ(v):
    return {"n": len(v), "mean": round(statistics.fmean(v), 2) if v else None,
            "median": statistics.median(v) if v else None}


def split_stats(recs):
    labelled = [x for x in recs if x["target"] == 1 and x["flaw_lines"]]
    kinds_all, border = Counter(), Counter()
    n_flaw = 0
    len_before = {"flaw": [], "clean": [], "clean_nonblank": []}
    len_after = {"flaw": [], "clean": [], "clean_nonblank": []}
    len_after_dens = {"flaw": [], "clean": [], "clean_nonblank": []}
    zero_after, dens = 0, []
    for x in labelled:
        lines = x["func"].split("\n")
        parts = split_code_comments(x["func"])
        kinds = [trivial_kind(c, m) for c, m in parts]
        flaws = set(x["flaw_lines"])
        n_flaw += len(flaws)
        d = flaw_density(x)
        dens.append((d, x))
        kept = {l for l in flaws if not kinds[l]}
        zero_after += not kept
        for l in flaws:
            kinds_all[kinds[l] or "code"] += 1
            if not kinds[l]:
                b = borderline_kind(parts[l][0])
                if b:
                    border[b] += 1
        for tgt, fl in ((len_before, flaws), (len_after, kept)):
            for l, line in enumerate(lines):
                s = len(line.strip())
                if l in fl:
                    tgt["flaw"].append(s)
                else:
                    tgt["clean"].append(s)
                    if s:
                        tgt["clean_nonblank"].append(s)
        if d <= 0.5 and kept:
            for l, line in enumerate(lines):
                s = len(line.strip())
                if l in kept:
                    len_after_dens["flaw"].append(s)
                else:
                    len_after_dens["clean"].append(s)
                    if s:
                        len_after_dens["clean_nonblank"].append(s)
    n_triv = sum(kinds_all[k] for k in TRIVIAL_KINDS)
    dv = sorted(d for d, _ in dens)
    q = lambda p: round(dv[min(len(dv) - 1, int(p * len(dv)))], 4) if dv else None
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0001]
    hist = {f"{lo:.2f}-{min(hi, 1):.2f}": sum(lo <= d < hi for d in dv) for lo, hi in zip(bins, bins[1:])}
    heavy = sorted(dens, key=lambda t: (-t[0], -len(t[1]["flaw_lines"])))[:10]
    return {
        "functions": len(recs),
        "vulnerable": sum(x["target"] for x in recs),
        "vulnerable_with_lines": len(labelled),
        "vulnerable_without_lines_already": sum(x["target"] == 1 and not x["flaw_lines"] for x in recs),
        "flaw_lines": n_flaw,
        "trivial_flaw_lines": n_triv,
        "trivial_pct": round(100 * n_triv / max(n_flaw, 1), 2),
        "kinds": {k: {"n": kinds_all[k], "pct_of_flaw_lines": round(100 * kinds_all[k] / max(n_flaw, 1), 2)}
                  for k in TRIVIAL_KINDS + ("code",)},
        "borderline_not_filtered": {k: {"n": v, "pct_of_flaw_lines": round(100 * v / max(n_flaw, 1), 2)}
                                    for k, v in border.most_common()},
        "density": {"mean": round(statistics.fmean(dv), 4) if dv else None, "median": q(0.5), "p90": q(0.9),
                    "p99": q(0.99), "max": dv[-1] if dv else None, "hist": hist,
                    "n_gt_0.5": sum(d > 0.5 for d in dv),
                    "flaw_lines_in_gt_0.5": sum(len(x["flaw_lines"]) for d, x in dens if d > 0.5)},
        "top10_dense": [{"idx": x["idx"], "project": x["project"], "density": round(d, 4),
                         "flaw_lines": len(x["flaw_lines"]), "n_lines": x["n_lines"]} for d, x in heavy],
        "vulnerable_zero_lines_after_trivial_drop": zero_after,
        "line_chars_before": {k: _summ(v) for k, v in len_before.items()},
        "line_chars_after_trivial_drop": {k: _summ(v) for k, v in len_after.items()},
        "line_chars_after_trivial_drop_and_density_0.5": {k: _summ(v) for k, v in len_after_dens.items()},
    }


def markdown(stats):
    S = ("train", "valid", "test")
    row = lambda name, f: f"| {name} | " + " | ".join(f(stats[s]) for s in S) + " |"
    out = ["# Nhãn dòng thừa trong SecVulEval – thống kê (chưa lọc gì)", "",
           "Sinh bởi `label_noise.py`. \"Dòng rác\" = sau khi bỏ chú thích chỉ còn khoảng trắng / `{` `}` / `( ) [ ] ; ,` / "
           "`else` trần. Độ dài = số ký tự của dòng sau `strip()`. Dòng nhãn 0 = các dòng còn lại của chính những hàm lỗi có nhãn dòng.", "",
           "| | train | valid | test |", "|---|---|---|---|",
           row("Hàm lỗi có nhãn dòng", lambda x: f"{x['vulnerable_with_lines']:,}"),
           row("Dòng nhãn 1", lambda x: f"{x['flaw_lines']:,}"),
           row("**Dòng nhãn 1 là rác**", lambda x: f"**{x['trivial_flaw_lines']:,} ({x['trivial_pct']}%)**")]
    for k, name in (("brace", "`{` / `}` (kể cả `};` `})`)"), ("blank", "trống"), ("comment", "chỉ chú thích"),
                    ("punct", "chỉ dấu câu"), ("else", "`else` trần (`} else {`)")):
        out.append(row(f"&nbsp;&nbsp;{name}", lambda x, k=k: f"{x['kinds'][k]['n']:,} ({x['kinds'][k]['pct_of_flaw_lines']}%)"))
    names = sorted({b for s in S for b in stats[s]["borderline_not_filtered"]})
    for b in names:
        out.append(row(f"&nbsp;&nbsp;*không lọc:* `{b}`", lambda x, b=b: (lambda v: f"{v['n']:,} ({v['pct_of_flaw_lines']}%)" if v else "0")(
            x["borderline_not_filtered"].get(b))))
    out += [row("Mật độ nhãn: trung bình / trung vị / p90", lambda x: f"{x['density']['mean']} / {x['density']['median']} / {x['density']['p90']}"),
            row("Hàm mật độ > 0,5 (số dòng nhãn 1 trong đó)", lambda x: f"{x['density']['n_gt_0.5']} ({x['density']['flaw_lines_in_gt_0.5']:,})"),
            row("Hàm lỗi còn 0 dòng nhãn 1 sau khi bỏ dòng rác", lambda x: f"{x['vulnerable_zero_lines_after_trivial_drop']}"),
            row("Hàm lỗi vốn đã 0 dòng nhãn (bản vá chỉ thêm)", lambda x: f"{x['vulnerable_without_lines_already']}")]
    for key, name in (("line_chars_before", "trước lọc"), ("line_chars_after_trivial_drop", "sau bỏ rác"),
                      ("line_chars_after_trivial_drop_and_density_0.5", "sau bỏ rác + mật độ ≤ 0,5")):
        out.append(row(f"Độ dài dòng nhãn 1 / nhãn 0 / nhãn 0 không trống, TB (trung vị) – {name}",
                       lambda x, key=key: " / ".join(f"{x[key][c]['mean']} ({x[key][c]['median']})" for c in ("flaw", "clean", "clean_nonblank"))))
    out += ["", "Phân bố mật độ (số hàm lỗi có nhãn dòng theo khoảng dòng nhãn 1 / số dòng):", "",
            "| khoảng | train | valid | test |", "|---|---|---|---|"]
    for b in stats["train"]["density"]["hist"]:
        out.append(f"| {b} | " + " | ".join(str(stats[s]["density"]["hist"][b]) for s in S) + " |")
    for s in S:
        out += ["", f"10 hàm nặng nhất – {s}:", "", "| idx | project | mật độ | dòng nhãn 1 / số dòng |", "|---|---|---|---|"]
        out += [f"| {h['idx']} | {h['project']} | {h['density']} | {h['flaw_lines']} / {h['n_lines']} |" for h in stats[s]["top10_dense"]]
    return "\n".join(out) + "\n"


def show(x, removed=(), max_lines=60):
    """Code with line numbers; `>>` = flaw label kept, `x>` = flaw label removed by the trivial rule."""
    fl, out = set(x["flaw_lines"]), []
    lines = x["func"].split("\n")
    for i, line in enumerate(lines[:max_lines]):
        mark = "x>" if i in removed else ">>" if i in fl else "  "
        out.append(f"{mark} {i:4d} | {line}")
    if len(lines) > max_lines:
        out.append(f"   ... ({len(lines) - max_lines} more lines, {sum(l >= max_lines for l in fl)} of them labelled)")
    return "\n".join(out)


def examples(train, k=10, seed=0, max_density=0.5):
    rng = random.Random(seed)
    out = ["# Ví dụ bộ lọc nhãn dòng (train, chọn ngẫu nhiên seed 0)", "",
           "`>>` = dòng nhãn 1 được giữ, `x>` = dòng nhãn 1 bị gỡ nhãn (thành 0) bởi luật dòng rác. Hàm dài chỉ in 60 dòng đầu.", "",
           f"## Luật 1: `--drop_trivial_flaw` ({k} hàm có ít nhất một nhãn bị gỡ)", ""]
    touched = []
    for x in train:
        if x["target"] == 1 and x["flaw_lines"]:
            kept, removed = drop_trivial_flaws(x)
            if removed:
                touched.append((x, removed))
    for x, removed in rng.sample(touched, min(k, len(touched))):
        out += [f"### idx {x['idx']} ({x['project']}) – gỡ {len(removed)}/{len(x['flaw_lines'])} nhãn: "
                + ", ".join(f"dòng {l} ({kd})" for l, kd in sorted(removed.items())), "", "```c", show(x, removed), "```", ""]
    dense = [x for x in train if x["target"] == 1 and x["flaw_lines"] and flaw_density(x) > max_density]
    out += [f"## Luật 2: `--max_flaw_density {max_density}` ({k} hàm bị bỏ cùng hàm cặp; tổng {len(dense)} hàm lỗi)", ""]
    for x in rng.sample(dense, min(k, len(dense))):
        out += [f"### idx {x['idx']} ({x['project']}) – mật độ {flaw_density(x):.2f} ({len(x['flaw_lines'])}/{x['n_lines']} dòng), "
                f"hàm cặp {x['pair_idx']}", "", "```c", show(x), "```", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="secvul_lines dir (unfiltered)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    splits = {s: [json.loads(l) for l in open(f"{a.data}/{s}.jsonl", encoding="utf-8")] for s in ("train", "valid", "test")}
    stats = {s: split_stats(r) for s, r in splits.items()}
    os.makedirs(a.out, exist_ok=True)
    json.dump(stats, open(f"{a.out}/label_noise_stats.json", "w", encoding="utf-8"), indent=1)
    open(f"{a.out}/label_noise_stats.md", "w", encoding="utf-8").write(markdown(stats))
    open(f"{a.out}/label_filter_examples.md", "w", encoding="utf-8").write(examples(splits["train"]))
    print(markdown(stats))


if __name__ == "__main__":
    main()
