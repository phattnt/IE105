"""Experiment 2: intra-function context strategies for the original LineVul checkpoint on PrimeVul.

Inference only (no retraining). Same model, same 512-token budget per forward pass, different ways
of choosing WHICH tokens of a long function the model sees:

  head        first 510 tokens (LineVul as published; = experiment 1)
  headtail    first 255 + last 255 tokens
  slide_max   sliding windows of 510 tokens, stride 256, covering the whole function;
              function score = max over windows, line score = max over windows
  slide_mean  same windows, function score = mean over windows
  slice       SySeVR-style heuristic slicing: keep lines with sensitive anchors (memory/string APIs,
              array indexing, pointer access, sizeof, arithmetic on sizes) plus one neighbouring line
              on each side, in original order, then truncate to 510 tokens

For every strategy: pair-wise metrics on the 564 test pairs, attention-based line localization on the
vulnerable members (groups A/B/C are defined by HEAD truncation so the recovery of group C is visible),
and function-level metrics on the full imbalanced test set.

Caveat printed with the results: the checkpoint was trained on head-truncated inputs only, so tail
windows and slices are out of distribution for it. Gains or losses here measure what inference-time
input selection alone can do; retraining with the same strategy is experiment 0/2b.

Runs ON the Colab VM after exp1/setup_vm.py (data under /content/exp1). Imports exp1's model code.
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/content/exp1/code")
import eval_linevul_primevul as e1  # noqa: E402  (model, tokenizer, metrics, labels)

ROOT, DATA, OUT = e1.ROOT, e1.DATA, "/content/exp1/results_exp2"
os.makedirs(OUT, exist_ok=True)
DEV, tok, log = e1.DEV, e1.tok, e1.log
CLS, SEP, PAD, BUDGET = 0, 2, 1, 510
STRIDE = 256

# ----------------------------------------------------------------------------- input strategies
# each returns a list of "views"; a view = (ids[512], offsets[512] into the ORIGINAL function text)


def _wrap(ids, offs):
    ids = [CLS] + list(ids) + [SEP]
    offs = [(0, 0)] + list(offs) + [(0, 0)]
    pad = 512 - len(ids)
    return ids + [PAD] * pad, offs + [(0, 0)] * pad


def full_tokens(func):
    enc = tok(func, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def views_head(func):
    ids, offs = full_tokens(func)
    return [_wrap(ids[:BUDGET], offs[:BUDGET])]


def views_headtail(func):
    ids, offs = full_tokens(func)
    if len(ids) <= BUDGET:
        return [_wrap(ids, offs)]
    h = BUDGET // 2
    t = BUDGET - h
    return [_wrap(ids[:h] + ids[-t:], offs[:h] + offs[-t:])]


def views_slide(func):
    ids, offs = full_tokens(func)
    if len(ids) <= BUDGET:
        return [_wrap(ids, offs)]
    views, start = [], 0
    while True:
        views.append(_wrap(ids[start:start + BUDGET], offs[start:start + BUDGET]))
        if start + BUDGET >= len(ids):
            break
        start += STRIDE
    return views


API = re.compile(r"\b(mem(cpy|move|set|cmp)|str(cpy|ncpy|cat|ncat|len|cmp|ncmp|dup|chr|str|tok)|s?n?printf|s?scanf|gets|fgets|"
                 r"read|write|recv|send|malloc|calloc|realloc|free|alloca|k[mz]alloc|kfree|vmalloc|vfree|"
                 r"copy_(from|to)_user|get_user|put_user|memdup|strlcpy|strlcat|atoi|strtol)\s*\(")
ANCHOR2 = re.compile(r"\[|->|sizeof\s*\(|\*\s*[A-Za-z_]|\+\+|--|<<|>>|\blen\b|\bsize\b|\boffset\b|\bcount\b")


def views_slice(func):
    lines = func.split("\n")
    n = len(lines)
    score = np.zeros(n)
    for i, l in enumerate(lines):
        if API.search(l):
            score[i] = 3
        elif ANCHOR2.search(l):
            score[i] = 2
    for i in range(n):            # neighbours of scored lines
        if score[i] >= 2:
            for j in (i - 1, i + 1):
                if 0 <= j < n and score[j] == 0:
                    score[j] = 1
    score[0] = 4                   # always keep the signature line
    # token count per line (approximate, independent tokenization of each line + newline)
    tl = np.array([len(tok(l + "\n", add_special_tokens=False)["input_ids"]) for l in lines])
    order = sorted(range(n), key=lambda i: (-score[i], i))
    keep, used = [], 0
    for i in order:
        if score[i] == 0:
            break
        if used + tl[i] > BUDGET:
            continue
        keep.append(i)
        used += tl[i]
    keep = sorted(keep)
    # re-tokenize the kept text and map offsets back to the original function
    starts = e1.line_starts(func)
    text = "\n".join(lines[i] for i in keep)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    # char position in `text` -> original char position
    kept_starts = []
    pos = 0
    for i in keep:
        kept_starts.append((pos, starts[i]))
        pos += len(lines[i]) + 1
    ks = np.array([k for k, _ in kept_starts])
    orig = np.array([o for _, o in kept_starts])

    def remap(c):
        j = int(np.searchsorted(ks, c, side="right") - 1)
        return orig[j] + (c - ks[j])

    offs = [(remap(s), remap(s) + (e - s)) if e > 0 else (0, 0) for s, e in enc["offset_mapping"]]
    ids = enc["input_ids"]
    return [_wrap(ids[:BUDGET], offs[:BUDGET])]


STRATEGIES = {"head": views_head, "headtail": views_headtail, "slide": views_slide, "slice": views_slice}


# ----------------------------------------------------------------------------- batched scoring over views
@torch.no_grad()
def score_functions(model, funcs, strategy, want_lines=False, bs=64):
    """Returns {"max": p, "mean": p} per-function probabilities (aggregated over views) and, optionally,
    per-line scores (max over views) plus per-line visibility."""
    make = STRATEGIES[strategy]
    if want_lines:
        bs = 8  # output_attentions keeps [bs, heads, 512, 512] for all 12 layers; 64 overflows a T4
    all_views, owner = [], []
    for fi, f in enumerate(funcs):
        vs = make(f)
        all_views.extend(vs)
        owner.extend([fi] * len(vs))
    log(f"  [{strategy}] {len(funcs)} functions -> {len(all_views)} views")
    probs = np.zeros(len(all_views))
    line_scores = [None] * len(funcs)
    visible = [None] * len(funcs)
    if want_lines:
        for fi, f in enumerate(funcs):
            n = len(e1.line_starts(f))
            line_scores[fi] = np.zeros(n)
            visible[fi] = np.zeros(n, bool)
    for b in range(0, len(all_views), bs):
        chunk = all_views[b:b + bs]
        ids = torch.tensor([v[0] for v in chunk], device=DEV)
        if want_lines:
            prob, atts = model(ids, output_attentions=True)
            a = atts[0].float().sum(1).sum(1).cpu().numpy()  # layer 0: sum heads, sum query rows -> [bs, seq]
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                prob = model(ids)
        probs[b:b + bs] = prob[:, 1].float().cpu().numpy()
        if want_lines:
            for k, (vids, voffs) in enumerate(chunk):
                fi = owner[b + k]
                starts = e1.line_starts(funcs[fi])
                ts = a[k].copy()
                real = [i for i, (s, e) in enumerate(voffs) if e > 0]
                ts[0] = 0.0
                if real:
                    ts[max(real) + 1 if max(real) + 1 < len(ts) else max(real)] = 0.0  # </s>
                per_line = np.zeros(len(starts))
                for i, (s, e) in enumerate(voffs):
                    if e <= 0 or i == 0:
                        continue
                    li = int(np.searchsorted(starts, s, side="right") - 1)
                    per_line[li] += ts[i]
                    visible[fi][li] = True
                line_scores[fi] = np.maximum(line_scores[fi], per_line)  # max over views
        if (b // bs) % 50 == 0:
            log(f"  [{strategy}] views {b}/{len(all_views)}")
    owner = np.array(owner)
    fmax, fmean = np.zeros(len(funcs)), np.zeros(len(funcs))
    for fi in range(len(funcs)):
        p = probs[owner == fi]
        fmax[fi], fmean[fi] = p.max(), p.mean()
    return {"max": fmax, "mean": fmean}, line_scores, visible


# ----------------------------------------------------------------------------- localization with precomputed scores
def localization_from_scores(pairs, probs, line_scores, visible, head_visible):
    recs = []
    for (v, f), p, sc, vis, hv in zip(pairs, probs, line_scores, visible, head_visible):
        flaws = e1.flaw_lines_from_diff(v["func"], f["func"])
        n_lines = len(sc)
        n_tok = e1.n_tokens(v["func"])
        rec = {"idx": v["idx"], "n_lines": n_lines, "n_tokens": n_tok, "truncated": n_tok > BUDGET,
               "n_flaw": len(flaws), "n_flaw_visible": int(sum(hv[i] for i in flaws)),      # visibility under HEAD (group definition)
               "n_flaw_visible_this": int(sum(vis[i] for i in flaws)), "pred_prob": float(p), "true_positive": bool(p > 0.5)}
        if flaws:
            order = sorted(range(n_lines), key=lambda i: (-sc[i], i))
            rank_of = {li: r for r, li in enumerate(order)}
            first = min(rank_of[i] for i in flaws)
            rec["ifa"] = first
            for k in (1, 5, 10):
                rec[f"top{k}"] = first < k
                rec[f"rand_top{k}"] = e1.expected_random_topk(n_lines, len(flaws), k)
        recs.append(rec)
    return recs


def pairs_identical(pairs, strategy):
    make = STRATEGIES[strategy]
    same = 0
    for v, f in pairs:
        a = [x[0] for x in make(v["func"])]
        b = [x[0] for x in make(f["func"])]
        same += a == b
    return same / len(pairs) * 100


# ----------------------------------------------------------------------------- main
def main():
    model = e1.load_model()
    test = e1.read_jsonl(f"{DATA}/primevul_test.jsonl")
    pairs = e1.build_pairs(e1.read_jsonl(f"{DATA}/primevul_test_paired.jsonl"))
    y = np.array([r["target"] for r in test])
    results = {"checkpoint": "12heads_linevul_model.bin (LineVul, trained on Big-Vul, head-truncated)",
               "note": "inference only; strategies other than head are out-of-distribution for this checkpoint",
               "budget_tokens": BUDGET, "stride": STRIDE, "strategies": {}}

    # visibility under HEAD defines the A/B/C groups for every strategy
    _, _, head_vis = score_functions(model, [v["func"] for v, _ in pairs], "head", want_lines=True)

    for strat in STRATEGIES:
        log(f"===== strategy {strat} =====")
        aggs = ("max", "mean") if strat == "slide" else ("max",)   # single-view strategies: max == mean
        pv, ls, vis = score_functions(model, [v["func"] for v, _ in pairs], strat, want_lines=True)
        pf, _, _ = score_functions(model, [f["func"] for _, f in pairs], strat)
        same_pct = pairs_identical(pairs, strat)
        p, _, _ = score_functions(model, [t["func"] for t in test], strat)
        for agg in aggs:
            name = f"{strat}_{agg}" if strat == "slide" else strat
            r = {"pairs": e1.pair_metrics(pairs, pv[agg], pf[agg])}
            r["pairs"]["pairs_identical_after_processing_pct"] = same_pct
            recs = localization_from_scores(pairs, pv[agg], ls, vis, head_vis)
            r["localization"] = e1.summarize_localization(recs)
            flaws_total = sum(x["n_flaw"] for x in recs)
            r["localization"]["flaw_lines_visible_under_this_strategy_pct"] = sum(x["n_flaw_visible_this"] for x in recs) / flaws_total * 100
            with open(f"{OUT}/localization_{name}.jsonl", "w") as fh:
                for x in recs:
                    fh.write(json.dumps(x) + "\n")
            r["full_test"] = e1.function_level(y, p[agg])
            with open(f"{OUT}/full_test_{name}.jsonl", "w") as fh:
                for t, pp in zip(test, p[agg]):
                    fh.write(json.dumps({"idx": t["idx"], "target": t["target"], "prob": float(pp)}) + "\n")
            log(f"[{name}] pairs: {json.dumps(r['pairs'])}")
            log(f"[{name}] localization: {json.dumps(r['localization'])}")
            log(f"[{name}] full test: {json.dumps(r['full_test'])}")
            results["strategies"][name] = r
            with open(f"{OUT}/results.json", "w") as fh:
                json.dump(results, fh, indent=1)
    results["elapsed_s"] = time.time() - e1.T0
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("\n===== RESULTS JSON =====")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
