"""Clean-label run vs original run on the SAME SecVulEval test (CPU, no torch): one markdown report with the verdict.

Reads results.json / test_scores.jsonl / primevul_results.json of both runs, results/controls.json, and the json
outputs of bootstrap_ci.py (--runs BASE NEW, so the difference is NEW - BASE) and debias_length.py. Adds the
within-function Spearman correlation between line score and line length (characters after strip), averaged over
test functions, with a project bootstrap of the paired difference.

Verdict rule (fixed before the run): "nhãn sạch giúp" only if the Top-10 CI of NEW - BASE excludes 0 on the positive
side AND the length correlation of NEW is not higher than BASE's (point estimate of the difference <= 0, labelled
vulnerable functions). Otherwise: "không có bằng chứng nhiễu nhãn thừa là nút thắt".

    python compare_clean.py --data ../../data/secvul_lines --base results/<_conv> --new results/<_conv_clean> \
        --bootstrap ../../data/bootstrap_conv_vs_clean.json --debias ../../data/debias_conv_clean.json \
        --out results/compare_conv_clean.md
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def length_rho(rows, scores):
    """{idx: Spearman(line score, line chars)} for every function with >= 3 scored lines of varying length."""
    out = {}
    for r in rows:
        sc = scores.get(r["idx"])
        if sc is None:
            continue
        ln = [len(t.strip()) for t in r["func"].split("\n")][:len(sc)]
        s = np.asarray(sc[:len(ln)], float)
        ok = s > -1e8
        if ok.sum() < 3 or len(set(np.asarray(ln)[ok])) < 2 or len(set(s[ok])) < 2:
            continue
        out[r["idx"]] = float(spearmanr(s[ok], np.asarray(ln)[ok]).correlation)
    return out


def rho_table(rows, base_sc, new_sc, n_boot=2000):
    by = {r["idx"]: r for r in rows}
    groups = {"hàm lỗi có nhãn dòng": [r for r in rows if r["target"] == 1 and r["flaw_lines"]], "mọi hàm test": rows}
    out = {}
    for name, rs in groups.items():
        rb, rn = length_rho(rs, base_sc), length_rho(rs, new_sc)
        common = sorted(set(rb) & set(rn))
        per_proj = defaultdict(list)
        for i in common:
            per_proj[by[i]["project"]].append((rb[i], rn[i]))
        projects = sorted(per_proj)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(n_boot):
            v = [d for p in rng.choice(projects, size=len(projects)) for d in per_proj[p]]
            diffs.append(np.mean([n - b for b, n in v]) if v else np.nan)      # new - base
        out[name] = {"n": len(common), "base": float(np.mean([rb[i] for i in common])),
                     "new": float(np.mean([rn[i] for i in common])),
                     "diff": float(np.mean([rn[i] - rb[i] for i in common])),
                     "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="UNFILTERED secvul_lines dir (test.jsonl is the same in both)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--controls", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "controls.json"))
    ap.add_argument("--bootstrap", required=True, help="bootstrap_ci.py --runs BASE NEW --out <this>")
    ap.add_argument("--debias", default=None, help="debias_length.py --runs BASE NEW --out <this>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load_jsonl(f"{a.data}/test.jsonl")
    nb, nn = (os.path.basename(os.path.normpath(p)) for p in (a.base, a.new))
    res = {nb: json.load(open(f"{a.base}/results.json")), nn: json.load(open(f"{a.new}/results.json"))}
    sc = {n: {r["idx"]: r["line_scores"] for r in load_jsonl(f"{p}/test_scores.jsonl")} for n, p in ((nb, a.base), (nn, a.new))}
    for n in sc:
        missing = {r["idx"] for r in rows} - set(sc[n])
        if missing:
            raise SystemExit(f"{n}: test_scores.jsonl misses {len(missing)} test functions - not the same test set")
    ctrl = json.load(open(a.controls))
    boot = json.load(open(a.bootstrap))
    diff = boot[f"{nn} - {nb}"]
    rho = rho_table(rows, sc[nb], sc[nn])
    deb = json.load(open(a.debias, encoding="utf-8")) if a.debias else None
    pv = {n: json.load(open(f"{p}/primevul_results.json")) if os.path.exists(f"{p}/primevul_results.json") else None
          for n, p in ((nb, a.base), (nn, a.new))}

    f1 = lambda v: "chưa đo" if v is None else f"{v:.1f}"
    L = [f"# `{nn}` so với `{nb}` (cùng test SecVulEval)", ""]
    L += ["## Tìm dòng (test SecVulEval)", "",
          "| | Top-1 | Top-5 | Top-10 | IFA (trung vị) | A Top-10 | B Top-10 | C Top-10 |", "|---|---|---|---|---|---|---|---|"]
    entries = [(f"`{nb}`", res[nb]["test"]), (f"`{nn}`", res[nn]["test"])] + \
              [(f"đối chứng: {k}", ctrl[k]) for k in ("length", "anchor", "random")]
    for name, m in entries:
        g = lambda grp, k: m.get(f"lines_{grp}", {}).get(k)
        L.append(f"| {name} | {f1(g('ALL', 'top1'))} | {f1(g('ALL', 'top5'))} | {f1(g('ALL', 'top10'))} | {g('ALL', 'median_ifa')} | "
                 f"{f1(g('A', 'top10'))} | {f1(g('B', 'top10'))} | {f1(g('C', 'top10'))} |")
    L += ["", "đối chứng `length` = dòng dài nhất trước / hồi quy độ dài (`train_lines.py --strategy controls`); test không đổi nên giữ nguyên.",
          "", f"## Bootstrap theo project (`{nn}` − `{nb}`, 2.000 lần)", "", "| chỉ số | chênh | CI 95% |", "|---|---|---|"]
    for k in ("top1", "top5", "top10", "pair", "same_text"):
        L.append(f"| {k} | {diff[k]['point']:+.1f} | [{diff[k]['ci95'][0]:+.1f}; {diff[k]['ci95'][1]:+.1f}] |")
    L += ["", "## Tương quan điểm dòng – độ dài dòng (Spearman trong từng hàm, trung bình)", "",
          f"| nhóm hàm | n | `{nb}` | `{nn}` | chênh (mới − gốc) | CI 95% theo project |", "|---|---|---|---|---|---|"]
    for g, r in rho.items():
        L.append(f"| {g} | {r['n']} | {r['base']:+.3f} | {r['new']:+.3f} | {r['diff']:+.3f} | [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] |")
    L += ["", "## Mức hàm / cặp sau khi trừ độ dài (`debias_length.py`)", ""]
    if deb:
        L += ["| run | tập | cặp thô | cặp trừ độ dài | ROC thô | ROC trừ độ dài |", "|---|---|---|---|---|---|"]
        for n in (nb, nn):
            for s, r in deb.get(n, {}).items():
                p = lambda k, f: r.get(k, {}).get(f)
                fmt = lambda v, d: "–" if v is None else f"{v:.{d}f}"
                L.append(f"| `{n}` | {s} | {fmt(p('raw', 'pair_acc'), 1)} | {fmt(p('debias_same', 'pair_acc'), 1)} | "
                         f"{fmt(p('raw', 'roc_auc'), 3)} | {fmt(p('debias_same', 'roc_auc'), 3)} |")
    else:
        L.append("chưa đo")
    L += ["", "## PrimeVul (`eval_primevul.py`)", "", "| run | Top-10 dòng (A / B / C) | cặp | ROC toàn tập | AUPRC toàn tập |", "|---|---|---|---|---|"]
    for n in (nb, nn):
        p = pv[n]
        if not p:
            L.append(f"| `{n}` | chưa đo | | | |")
            continue
        q = p["pairs_and_lines"]
        L.append(f"| `{n}` | {f1(q['lines_ALL']['top10'])} ({f1(q['lines_A']['top10'])} / {f1(q['lines_B']['top10'])} / "
                 f"{f1(q['lines_C']['top10'])}) | {f1(q.get('pairs', {}).get('ranking_acc_ties_half'))} | "
                 f"{p['full_test']['roc_auc']:.3f} | {p['full_test']['auprc']:.3f} |")
    L.append("\nĐối chứng độ dài trên PrimeVul: chưa đo (việc mở số 1 trong HANDOFF.md).")

    top10_ok = diff["top10"]["ci95"][0] > 0
    rho_ok = rho["hàm lỗi có nhãn dòng"]["diff"] <= 0
    L += ["", "## Kết luận (luật đặt trước khi chạy)", "",
          f"- CI Top-10 không chứa 0 và dương: **{'có' if top10_ok else 'không'}** "
          f"({diff['top10']['point']:+.1f} [{diff['top10']['ci95'][0]:+.1f}; {diff['top10']['ci95'][1]:+.1f}])",
          f"- Tương quan độ dài (hàm lỗi có nhãn dòng) không tăng: **{'có' if rho_ok else 'không'}** "
          f"(chênh {rho['hàm lỗi có nhãn dòng']['diff']:+.3f})", ""]
    if top10_ok and rho_ok:
        L.append("→ **Nhãn sạch giúp** (1 seed).")
    else:
        L.append("→ **Không có bằng chứng nhiễu nhãn thừa là nút thắt.**")
        if diff["top10"]["point"] > 0 and not rho_ok:
            L.append("Top-k tăng nhưng tương quan với độ dài cũng tăng: có thể do lối tắt độ dài mạnh lên, không phải nhờ nhãn sạch.")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
