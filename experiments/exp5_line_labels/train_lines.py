"""Experiment 5: line-level supervision on SecVulEval.

Instead of a function label + attention (LineVul), every line gets its own label from the fix
(changed_statements): lines of the vulnerable version touched by the fix are 1, every other line - of the
vulnerable version, of its fixed version and of unrelated benign functions - is 0. The model:

    encoder (RoBERTa family) -> token states -> mean-pool the tokens of each line -> linear -> line logit
    function score = max line probability over the whole function (all views)
    loss: weighted line BCE (default), or --loss listwise (softmax over each function's lines + lam_bce x BCE);
          --mask_contradict drops fixed-version lines whose text equals a flaw line of the pair from the loss

Strategies (what one 512-token view contains, like exp2/exp3):
  head    first max_len-2 tokens (lines beyond them are never seen, score 0)
  slide   windows of max_len-2 tokens, stride (max_len-2)/2; a line seen by several windows keeps its max
Controls (no GPU): length (logistic regression on log tokens / log lines -> function score),
  anchor (rank lines by memory/string-API and pointer/array anchors from exp2), random.

Evaluation on the test split (projects never seen in training):
  line level   vulnerable functions with >=1 labelled line: Top-1/5/10, IFA, groups A/B/C by head visibility
  function     AUPRC / ROC-AUC (43% prevalence: not comparable with PrimeVul), Spearman with length
  pairs        vulnerable vs its fixed version: share ranked correctly (ties = 0.5), both ways of scoring

Vulnerable functions whose fix only adds lines (no labelled line) are left out of training (their lines are
unknown, not clean) and of line-level metrics, but kept for function-level and pair metrics.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import time

import numpy as np

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def read_split(data, split, limit=None):
    rows = [json.loads(l) for l in open(f"{data}/{split}.jsonl", encoding="utf-8")]
    if limit:
        rng = random.Random(0)
        pairs = [r for r in rows if r["target"] == 1 and r["pair_idx"] is not None]
        pick = {r["idx"] for r in rng.sample(pairs, min(len(pairs), limit // 2))}
        pick |= {r["pair_idx"] for r in rows if r["idx"] in pick}
        rows = [r for r in rows if r["idx"] in pick]
    return rows


def line_starts(func):
    return [0] + [i + 1 for i, ch in enumerate(func) if ch == "\n"]


# ----------------------------------------------------------------------------- metrics shared by all methods
def expected_random_topk(n, f, k):
    if f == 0 or n == 0:
        return 0.0
    if k >= n:
        return 1.0
    return 1.0 - math.comb(n - f, k) / math.comb(n, k)


def rank_metrics(scores, flaws, n_lines):
    order = sorted(range(n_lines), key=lambda i: (-scores[i], i))
    rank_of = {li: r for r, li in enumerate(order)}
    first = min(rank_of[i] for i in flaws if i < n_lines)
    out = {"ifa": first}
    for k in (1, 5, 10):
        out[f"top{k}"] = first < k
        out[f"rand_top{k}"] = expected_random_topk(n_lines, len(flaws), k)
    return out


def evaluate(rows, line_scores, fn_scores, head_visible, n_tokens):
    """line_scores[i]: np.array per line; fn_scores[i]: float; head_visible[i]: bool array per line."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    res = {}
    recs = []
    for r, sc, hv, nt in zip(rows, line_scores, head_visible, n_tokens):
        if r["target"] != 1 or not r["flaw_lines"]:
            continue
        fl = [i for i in r["flaw_lines"] if i < len(sc)]
        if not fl:
            continue
        m = rank_metrics(sc, fl, len(sc))
        m["group"] = "A" if nt <= 510 else ("B" if any(hv[i] for i in fl) else "C")
        recs.append(m)
    for g in ("ALL", "A", "B", "C"):
        rs = [x for x in recs if g == "ALL" or x["group"] == g]
        if not rs:
            continue
        res[f"lines_{g}"] = {"n": len(rs), "median_ifa": float(np.median([x["ifa"] for x in rs])),
                             **{f"top{k}": float(np.mean([x[f"top{k}"] for x in rs]) * 100) for k in (1, 5, 10)},
                             **{f"rand_top{k}": float(np.mean([x[f"rand_top{k}"] for x in rs]) * 100) for k in (1, 5, 10)}}
    y = np.array([r["target"] for r in rows])
    p = np.array(fn_scores, dtype=float)
    nt = np.array(n_tokens)
    rk = lambda a: np.argsort(np.argsort(a))
    res["function"] = {"n": len(y), "vulnerable": int(y.sum()), "auprc": float(average_precision_score(y, p)),
                       "roc_auc": float(roc_auc_score(y, p)), "spearman_with_tokens": float(np.corrcoef(rk(p), rk(nt))[0, 1])}
    by = {r["idx"]: i for i, r in enumerate(rows)}
    pv, pf = [], []
    for r in rows:
        if r["target"] == 1 and r["pair_idx"] in by:
            pv.append(p[by[r["idx"]]]); pf.append(p[by[r["pair_idx"]]])
    pv, pf = np.array(pv), np.array(pf)
    if len(pv):
        tie = np.isclose(pv, pf, rtol=0, atol=1e-4)
        res["pairs"] = {"n": int(len(pv)), "ties": int(tie.sum()),
                        "ranking_acc_ties_half": float((((pv > pf) & ~tie) + 0.5 * tie).mean() * 100),
                        "ranking_acc_non_tied": float((pv > pf)[~tie].mean() * 100) if (~tie).any() else None}
    # length-neutral pair test: a flaw line whose exact text also appears in the fixed version
    # (e.g. kthread_stop(x) that the fix only guards) - same tokens, different context
    same, win = 0, 0.0
    for r in rows:
        if r["target"] != 1 or r["pair_idx"] not in by or not r["flaw_lines"]:
            continue
        f = rows[by[r["pair_idx"]]]
        fl_text = {}
        for j, t in enumerate(f["func"].split("\n")):
            k = re.sub(r"\s+", "", t)
            if k:
                fl_text.setdefault(k, []).append(j)
        vlines = r["func"].split("\n")
        for l in r["flaw_lines"]:
            k = re.sub(r"\s+", "", vlines[l]) if l < len(vlines) else ""
            if len(k) < 6 or k not in fl_text:
                continue
            j = min(fl_text[k], key=lambda x: abs(x - l))
            a, b = line_scores[by[r["idx"]]][l], line_scores[by[r["pair_idx"]]][j]
            same += 1
            win += 1.0 if a > b + 1e-4 else (0.5 if abs(a - b) <= 1e-4 else 0.0)
    res["same_text_flaw_lines"] = {"n": same, "vul_scored_higher_pct": win / same * 100 if same else None}
    return res


# ----------------------------------------------------------------------------- controls
API = re.compile(r"\b(mem(cpy|move|set|cmp)|str(cpy|ncpy|cat|ncat|len|cmp|ncmp|dup|chr|str|tok)|s?n?printf|s?scanf|gets|fgets|"
                 r"read|write|recv|send|malloc|calloc|realloc|free|alloca|k[mz]alloc|kfree|vmalloc|vfree|copy_(from|to)_user|"
                 r"get_user|put_user|memdup|strlcpy|strlcat|atoi|strtol)\s*\(")
ANCHOR2 = re.compile(r"\[|->|sizeof\s*\(|\*\s*[A-Za-z_]|\+\+|--|<<|>>|\blen\b|\bsize\b|\boffset\b|\bcount\b")


def controls(data, out, args):
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    train, test = read_split(data, "train", args.limit), read_split(data, "test", args.limit)
    ntok = lambda rows: [len(tok(r["func"], add_special_tokens=False)["input_ids"]) for r in rows]
    nt_tr, nt_te = ntok(train), ntok(test)
    feats = lambda rows, nt: np.array([[math.log1p(n), math.log1p(r["n_lines"])] for r, n in zip(rows, nt)])
    clf = LogisticRegression().fit(feats(train, nt_tr), [r["target"] for r in train])
    p_len = clf.predict_proba(feats(test, nt_te))[:, 1]
    head_vis = [head_visibility(tok, r["func"], args.max_len) for r in test]
    rng = np.random.default_rng(0)
    results = {}
    for name in ("length", "anchor", "random"):
        ls = []
        for r in test:
            lines = r["func"].split("\n")
            if name == "anchor":
                s = np.array([3.0 * bool(API.search(l)) + 2.0 * bool(ANCHOR2.search(l)) for l in lines])
            elif name == "random":
                s = rng.random(len(lines))
            else:                                   # length control has no line view: longest line first
                s = np.array([len(l.strip()) for l in lines], dtype=float)
            ls.append(s)
        fn = p_len if name == "length" else [float(s.max()) for s in ls]
        results[name] = evaluate(test, ls, fn, head_vis, nt_te)
        log(name, json.dumps(results[name]))
    json.dump(results, open(f"{out}/controls.json", "w"), indent=1)


_GROUP_TOK = None


def group_info(rows):
    """(head_visible per line, n_tokens) under CodeBERT's tokenizer and a 512-token head, as in TN1-TN3."""
    global _GROUP_TOK
    if _GROUP_TOK is None:
        from transformers import AutoTokenizer
        _GROUP_TOK = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    hv = [head_visibility(_GROUP_TOK, r["func"], 512) for r in rows]
    nt = [len(_GROUP_TOK(r["func"], add_special_tokens=False)["input_ids"]) for r in rows]
    return hv, nt


def head_visibility(tok, func, max_len):
    enc = tok(func, add_special_tokens=False, return_offsets_mapping=True)
    starts = line_starts(func)
    vis = np.zeros(len(starts), bool)
    for s, e in enc["offset_mapping"][:max_len - 2]:
        if e > 0:
            vis[np.searchsorted(starts, s, side="right") - 1] = True
    return vis


# ----------------------------------------------------------------------------- model
def contradicting_lines(rows):
    """{idx of a fixed version: its lines whose text (whitespace removed, as in same_text) equals a flaw line of the
    vulnerable partner}. Such a line is 1 in the vulnerable version and 0 in the fixed one although the text is the same
    (39% of SecVulEval train flaw lines, 25/10); --mask_contradict leaves it out of the loss instead of labelling it 0."""
    by = {r["idx"]: r for r in rows}
    out = {}
    for r in rows:
        if r["target"] != 1 or not r["flaw_lines"] or r["pair_idx"] not in by:
            continue
        vl = r["func"].split("\n")
        keys = {re.sub(r"\s+", "", vl[l]) for l in r["flaw_lines"] if l < len(vl)} - {""}
        f = by[r["pair_idx"]]
        hit = {j for j, t in enumerate(f["func"].split("\n")) if re.sub(r"\s+", "", t) in keys}
        if hit:
            out.setdefault(f["idx"], set()).update(hit)
    return out


def build_views(tok, func, flaws, strategy, max_len, ignore=()):
    """-> list of (ids[max_len], tok_line[max_len] with -1 for special/pad, lines_in_view, labels_in_view).
    Labels: 1 flaw, 0 clean, -1 for lines in `ignore` (left out of the loss)."""
    enc = tok(func, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    starts = line_starts(func)
    tline = [int(np.searchsorted(starts, s, side="right") - 1) for s, e in offs]
    budget = max_len - 2
    if strategy == "head" or len(ids) <= budget:
        spans = [(0, min(len(ids), budget))]
    else:
        step = budget // 2
        spans = [(a, min(a + budget, len(ids))) for a in range(0, max(1, len(ids) - budget + step), step)]
        if spans[-1][1] < len(ids):
            spans.append((len(ids) - budget, len(ids)))
    fl, ign = set(flaws), set(ignore)
    views = []
    for a, b in spans:
        vid = [tok.cls_token_id] + ids[a:b] + [tok.sep_token_id]     # padded per batch in collate
        vl = [-1] + tline[a:b] + [-1]
        lines = sorted({l for l in vl if l >= 0})
        views.append((vid, vl, lines, [1 if l in fl else -1 if l in ign else 0 for l in lines]))
    return views


def listwise_loss(z, y, lens, bce_each, C):
    """(listwise, bce, chance) for one batch of views. listwise: per view with a flaw line, -log of the softmax mass on
    its flaw lines (over its labelled lines), summed and divided by the constant C = flaw views per batch on average, so
    each vulnerable function counts once whatever its size or number of flaw lines (a view is the whole function for
    the ~90% that fit one window). A per-batch mean would not: batches are packed by length, so a long view shares its
    batch with 1-3 others and a short one with dozens (debate 25/10). chance: the same quantity for uniform scores,
    log(n_labelled / n_flaw). bce: the weighted line BCE over every labelled line, which keeps scores comparable across
    functions (benign and fixed versions get no listwise term). Lines labelled -1 are in neither."""
    import torch
    V = len(lens)
    vid = torch.repeat_interleave(torch.arange(V, device=z.device), torch.tensor(lens, device=z.device))
    keep, pos = y >= 0, y == 1

    def lse(m):                         # per-view logsumexp of z over the lines in mask m
        v = torch.where(m, z, torch.full_like(z, -1e4))
        mx = torch.full((V,), -1e4, device=z.device).scatter_reduce(0, vid, v, "amax").detach()
        s = torch.zeros(V, device=z.device).index_add(0, vid, torch.exp(v - mx[vid]) * m)
        return mx + torch.log(s.clamp_min(1e-30))
    n_pos = torch.zeros(V, device=z.device).index_add(0, vid, pos.float())
    n_keep = torch.zeros(V, device=z.device).index_add(0, vid, keep.float())
    has = n_pos > 0
    lw = (lse(keep) - lse(pos))[has].sum() / C if has.any() else z.sum() * 0
    chance = float((torch.log(n_keep[has]) - torch.log(n_pos[has])).sum() / C) if has.any() else 0.0
    bce = bce_each(z[keep], y[keep]).mean() if keep.any() else z.sum() * 0
    return lw, bce, chance


def make_model(model_name):
    import torch
    from transformers import AutoModel

    class LineModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            try:
                self.enc = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
            except TypeError:          # ModernBERT and other non-RoBERTa encoders
                self.enc = AutoModel.from_pretrained(model_name)
            h = self.enc.config.hidden_size
            self.drop = torch.nn.Dropout(0.1)
            self.head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Dropout(0.1), torch.nn.Linear(h, 1))

        def forward(self, ids, seg, n_seg, pad_id):
            """seg: [B, L] global line-segment id per token (-1 = ignore). Returns one logit per segment."""
            hs = self.enc(ids, attention_mask=ids.ne(pad_id)).last_hidden_state          # [B, L, H]
            hs = self.drop(hs).reshape(-1, hs.size(-1))
            seg = seg.reshape(-1)
            keep = seg >= 0
            pooled = torch.zeros(n_seg, hs.size(-1), device=hs.device, dtype=hs.dtype)
            pooled.index_add_(0, seg[keep], hs[keep])
            cnt = torch.zeros(n_seg, device=hs.device, dtype=hs.dtype).index_add_(0, seg[keep], torch.ones_like(seg[keep], dtype=hs.dtype))
            pooled = pooled / cnt.clamp(min=1).unsqueeze(1)
            return self.head(pooled.float()).squeeze(-1)

    return LineModel()


PAD_ID = 1
AMP_DTYPE = None   # set from --bf16 in __main__; None = float16 on CUDA


def amp():
    import torch
    return AMP_DTYPE or torch.float16


def collate(batch, device):
    import torch
    L = max(len(v[0]) for v in batch)
    ids = torch.tensor([v[0] + [PAD_ID] * (L - len(v[0])) for v in batch], device=device)
    seg, labels, owners, off = [], [], [], 0
    for v in batch:
        loc = {l: off + j for j, l in enumerate(v[2])}
        seg.append([loc[l] if l >= 0 else -1 for l in v[1]] + [-1] * (L - len(v[1])))
        labels.extend(v[3])
        owners.extend(v[2])
        off += len(v[2])
    return ids, torch.tensor(seg, device=device), off, torch.tensor(labels, device=device, dtype=torch.float), owners


def token_batches(views, budget, shuffle_seed=None):
    """Group view indices into batches of at most `budget` padded tokens; similar lengths together."""
    idx = list(range(len(views)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(idx)
    out, chunk = [], 2000
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: len(views[i][0]))
        cur, cur_max = [], 0
        for i in part:
            m = max(cur_max, len(views[i][0]))
            if cur and m * (len(cur) + 1) > budget:
                out.append(cur); cur, m = [], len(views[i][0])
            cur.append(i); cur_max = m
        if cur:
            out.append(cur)
    if shuffle_seed is not None:
        random.Random(shuffle_seed + 1).shuffle(out)
    return out


def score_rows(model, tok, rows, strategy, args, device):
    """Line LOGITS (float32; unseen lines = -inf) and function logit = max line logit."""
    import torch
    model.eval()
    allv, own = [], []
    for i, r in enumerate(rows):
        for v in build_views(tok, r["func"], r.get("flaw_lines", []), strategy, args.max_len):
            allv.append(v); own.append(i)
    line_scores = [np.full(r["n_lines"], -np.inf) for r in rows]
    with torch.no_grad():
        for bidx in token_batches(allv, args.eval_tokens):
            chunk = [allv[i] for i in bidx]
            ids, seg, n_seg, _, owners = collate(chunk, device)
            with torch.autocast(device.type, dtype=amp(), enabled=device.type == "cuda"):
                z = model(ids, seg, n_seg, tok.pad_token_id).float().cpu().numpy()
            k = 0
            for j, v in zip(bidx, chunk):
                i = own[j]
                for l in v[2]:
                    line_scores[i][l] = max(line_scores[i][l], z[k]); k += 1
    fn = [float(s.max()) if len(s) and np.isfinite(s.max()) else -1e9 for s in line_scores]
    for s in line_scores:          # unseen lines rank after every seen line
        s[~np.isfinite(s)] = -1e9
    return line_scores, fn


def pair_acc_debiased(rows, fs):
    """Pair accuracy (ties half) after removing E[score | log chars] (cubic fit, label-free) from the function scores,
    i.e. the part of the vulnerable-vs-fixed ranking that length alone cannot explain (see debias_length.py)."""
    x = np.log(np.maximum([len(r["func"]) for r in rows], 1))
    s = np.asarray(fs, float)
    ok = s > -1e8
    if ok.sum() < 10:
        return 50.0
    c = np.polyfit(x[ok], s[ok], 3)
    res = {r["idx"]: v - np.polyval(c, xi) for r, v, xi in zip(rows, s, x)}
    w = t = 0
    for r in rows:
        if r["target"] == 1 and r["pair_idx"] in res:
            a, b = res[r["idx"]], res[r["pair_idx"]]
            t += 1
            w += 1.0 if a > b else 0.5 if a == b else 0.0
    return 100 * w / t if t else 50.0


def valid_extra(rows, ls, fs, pos_weight):
    """Extra validation numbers logged at every evaluation point (none of them is used for model selection unless
    --select says so): the weighted line BCE on the seen lines (convergence signal, same loss as training), and the
    function level read the way pair_protocol.py reads it - pair ranking raw / length-debiased / on length-matched
    pairs (|chars diff| / max < 5%) / with the function score = mean of the top-3 line logits, net = P-C - P-R at a 50%
    flag rate over the pair functions, and ROC-AUC over every valid function (benign ones included) raw and debiased."""
    from sklearn.metrics import roc_auc_score
    out = {}
    bce, n = 0.0, 0
    for r, s in zip(rows, ls):
        s = np.asarray(s, float)
        seen = s > -1e8
        if not seen.any():
            continue
        y = np.zeros(len(s)); y[[l for l in r["flaw_lines"] if l < len(s)]] = 1
        z, yy = s[seen], y[seen]
        w = np.where(yy == 1, pos_weight, 1.0)
        bce += float((w * (np.maximum(z, 0) - z * yy + np.log1p(np.exp(-np.abs(z))))).sum()); n += len(z)
    out["line_bce"] = bce / max(n, 1)
    chars = np.log(np.maximum([len(r["func"]) for r in rows], 1))
    s = np.asarray(fs, float)
    ok = s > -1e8
    resid = s - np.polyval(np.polyfit(chars[ok], s[ok], 3), chars) if ok.sum() >= 10 else s * 0
    top3 = np.array([float(np.mean(sorted(np.asarray(x)[np.asarray(x) > -1e8], reverse=True)[:3])) if (np.asarray(x) > -1e8).any() else -1e9 for x in ls])
    by = {r["idx"]: i for i, r in enumerate(rows)}
    pv = [(by[r["idx"]], by[r["pair_idx"]]) for r in rows if r["target"] == 1 and r["pair_idx"] in by]
    if pv:
        iv, iff = np.array([p[0] for p in pv]), np.array([p[1] for p in pv])
        acc = lambda a, b: float(np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0)).mean() * 100)
        cv, cf = np.exp(chars[iv]), np.exp(chars[iff])
        lm = np.abs(cv - cf) / np.maximum(cv, cf) < 0.05
        out["pairs"] = {"n": len(pv), "acc": acc(s[iv], s[iff]), "acc_debiased": acc(resid[iv], resid[iff]),
                        "acc_length_matched": acc(s[iv][lm], s[iff][lm]) if lm.any() else None, "n_length_matched": int(lm.sum()),
                        "acc_top3mean": acc(top3[iv], top3[iff]),
                        "length_rule_longer_is_fixed": acc(cf, cv)}
        t = float(np.median(np.concatenate([s[iv], s[iff]])))
        fv, ff = s[iv] > t, s[iff] > t
        out["pairs"].update({"P-C_at50": float((fv & ~ff).mean() * 100), "P-R_at50": float((~fv & ff).mean() * 100),
                             "net_at50": float(((fv & ~ff).mean() - (~fv & ff).mean()) * 100)})
    y = np.array([r["target"] for r in rows])
    if 0 < y.sum() < len(y):
        out["function"] = {"n": int(len(y)), "vulnerable": int(y.sum()), "roc_auc": float(roc_auc_score(y, s)),
                           "roc_auc_debiased": float(roc_auc_score(y, resid)), "roc_auc_top3mean": float(roc_auc_score(y, top3))}
    return out


# ----------------------------------------------------------------------------- fix-contrastive training
def align_lines(vul, fix):
    """Map each line of the vulnerable version to its counterpart line in the fixed version (difflib on stripped
    lines): equal -> same line; replace -> most similar replacement line; delete -> the fixed line at the junction."""
    import difflib
    a = [x.strip() for x in vul.split("\n")]
    b = [x.strip() for x in fix.split("\n")]
    m = {}
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                m[i1 + k] = j1 + k
        elif op == "replace":
            for i in range(i1, i2):
                m[i] = max(range(j1, j2), key=lambda j: difflib.SequenceMatcher(None, a[i], b[j], autojunk=False).ratio())
        elif op == "delete" and b:
            for i in range(i1, i2):
                m[i] = min(j1, len(b) - 1)
    return m


def pick_views(views, keep_lines, k):
    """At most k views of one function, preferring windows that contain the lines the pair losses need."""
    if len(views) <= k:
        return views
    hit = [v for v in views if keep_lines & set(v[2])]
    rest = [v for v in views if v not in hit]
    return (hit + rest)[:k]


def make_items(train_rows, tok, args):
    """Items = [vulnerable, fixed] pairs (both kept together in one batch) or single functions.
    Each member: {"views", "flaws", "bce" (whether its line labels are known)}; pairs also carry
    "match" = [(flaw line in vul, counterpart line in fix)]."""
    by = {r["idx"]: r for r in train_rows}
    used, items = set(), []
    for r in train_rows:
        if r["target"] != 1 or r["pair_idx"] not in by:
            continue
        f = by[r["pair_idx"]]
        al = align_lines(r["func"], f["func"])
        match = [(l, al[l]) for l in r["flaw_lines"] if l in al]
        vv = build_views(tok, r["func"], r["flaw_lines"], args.strategy, args.max_len)
        fv = build_views(tok, f["func"], [], args.strategy, args.max_len)
        vv = pick_views(vv, {l for l, _ in match} | set(r["flaw_lines"]), args.max_pair_views)
        fv = pick_views(fv, {j for _, j in match}, args.max_pair_views)
        items.append({"members": [{"views": vv, "bce": bool(r["flaw_lines"])}, {"views": fv, "bce": True}], "match": match})
        used |= {r["idx"], f["idx"]}
    for r in train_rows:
        if r["idx"] in used or (r["target"] == 1 and not r["flaw_lines"]):
            continue
        for v in build_views(tok, r["func"], r["flaw_lines"], args.strategy, args.max_len):
            items.append({"members": [{"views": [v], "bce": True}], "match": []})
    return items


def item_batches(items, budget, seed):
    """Pack items so that (number of views) x (longest view) stays under the token budget."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    cost = lambda it: (sum(len(m["views"]) for m in it["members"]), max(len(v[0]) for m in it["members"] for v in m["views"]))
    out, chunk = [], 1000
    for c in range(0, len(order), chunk):
        part = sorted(order[c:c + chunk], key=lambda i: cost(items[i])[1])
        cur, nv, ml = [], 0, 0
        for i in part:
            n, l = cost(items[i])
            if cur and (nv + n) * max(ml, l) > budget:
                out.append(cur); cur, nv, ml = [], 0, 0
            cur.append(i); nv += n; ml = max(ml, l)
        if cur:
            out.append(cur)
    random.Random(seed + 1).shuffle(out)
    return out


def collate_items(items, device):
    """Flatten items into views; return model inputs plus everything the three losses need."""
    import torch
    views, vfunc, bce_ok, func_of = [], [], [], 0
    pairs_fn, line_pairs = [], []
    for it in items:
        ids_f = []
        for m in it["members"]:
            for v in m["views"]:
                views.append(v); vfunc.append(func_of); bce_ok.append(m["bce"])
            ids_f.append(func_of); func_of += 1
        if len(ids_f) == 2:
            pairs_fn.append((ids_f[0], ids_f[1]))
            line_pairs += [((ids_f[0], l), (ids_f[1], j)) for l, j in it["match"]]
    ids, seg, n_seg, y, _ = collate(views, device)
    # per (function, line) unit = max over the windows that contain the line
    ukey, seg_u, mask = {}, [], []
    for v, f, ok in zip(views, vfunc, bce_ok):
        for l in v[2]:
            seg_u.append(ukey.setdefault((f, l), len(ukey)))
            mask.append(1.0 if ok else 0.0)
    u_func = [k[0] for k in sorted(ukey, key=ukey.get)]
    lp = [(ukey[a], ukey[b]) for a, b in line_pairs if a in ukey and b in ukey]
    T = lambda x, dt=torch.long: torch.tensor(x, device=device, dtype=dt)
    return (ids, seg, n_seg, y, T(mask, torch.float), T(seg_u), len(ukey), T(u_func), func_of,
            T([a for a, _ in lp]), T([b for _, b in lp]), T([a for a, _ in pairs_fn]), T([b for _, b in pairs_fn]))


def pair_losses(z, seg_u, n_u, u_func, n_func, lv, lf, fv, ff, margin):
    """Line hinge: flaw line in the vulnerable version must beat its counterpart in the fixed version by `margin`
    (logits). Function hinge: soft-max (logsumexp) over the lines of the vulnerable function must beat the fixed one."""
    import torch
    zu = torch.full((n_u,), -1e4, device=z.device, dtype=z.dtype).scatter_reduce(0, seg_u, z, reduce="amax", include_self=False)
    line = torch.relu(margin - (zu[lv] - zu[lf])).mean() if lv.numel() else zu.sum() * 0
    if fv.numel():
        mx = torch.full((n_func,), -1e4, device=z.device, dtype=z.dtype).scatter_reduce(0, u_func, zu, reduce="amax", include_self=False)
        se = torch.zeros(n_func, device=z.device, dtype=z.dtype).index_add(0, u_func, torch.exp(zu - mx[u_func]))
        F = mx + torch.log(se.clamp(min=1e-12))
        fn = torch.relu(margin - (F[fv] - F[ff])).mean()
    else:
        fn = zu.sum() * 0
    return line, fn


def train(data, out, args):
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    train_all = read_split(data, "train", args.limit)
    if args.extra_train:
        extra = read_split(args.extra_train, "train", args.limit)
        if args.extra_pairs is not None:
            rng = random.Random(args.seed)
            vul = [r for r in extra if r["target"] == 1 and r["pair_idx"] is not None]
            keep = {r["idx"] for r in rng.sample(vul, min(len(vul), args.extra_pairs))}
            keep |= {r["pair_idx"] for r in vul if r["idx"] in keep}
            extra = [r for r in extra if r["idx"] in keep]
        log(f"mixed in {len(extra)} functions from {args.extra_train}")
        train_all += extra
    leak = {r["idx"] for r in train_all if r.get("in_primevul_test")}
    leak |= {r["pair_idx"] for r in train_all if r["idx"] in leak and r["pair_idx"] is not None}
    pair_mode = args.lam_line > 0 or args.lam_fn > 0 or args.pair_batches
    if pair_mode and (args.loss != "bce" or args.mask_contradict):
        raise SystemExit("--loss listwise and --mask_contradict work on single views only (no pair losses)")
    # add-only fixes have no labelled line; pair mode keeps them (function hinge only, line BCE masked)
    train_rows = [r for r in train_all if r["idx"] not in leak and (pair_mode or not (r["target"] == 1 and not r["flaw_lines"]))]
    log(f"dropped {len(leak)} train functions overlapping the PrimeVul test set (incl. pair partners)")
    if args.balance_length:
        by = {r["idx"]: r for r in train_rows}
        longer, rest = [], []
        for r in train_rows:
            if r["target"] == 1 and r["pair_idx"] in by:
                # by characters: in-place edits keep the line count but the fix is still longer in ~75% of pairs
                (longer if len(by[r["pair_idx"]]["func"]) > len(r["func"]) else rest).append(r["idx"])
        drop = set(random.Random(args.seed).sample(longer, max(0, len(longer) - len(rest))))
        drop |= {by[i]["pair_idx"] for i in drop}
        train_rows = [r for r in train_rows if r["idx"] not in drop]
        log(f"balance_length: {len(longer)} pairs with a longer fix (chars) vs {len(rest)} shorter/equal -> dropped {len(drop)} functions")
    valid = read_split(data, "valid", args.limit)
    test = read_split(data, "test", args.limit)
    log(f"train {len(train_rows)} functions{'' if pair_mode else ' (add-only fixes dropped)'} | valid {len(valid)} | test {len(test)} | device {device}")
    # 24/10: tokenising + windowing 87k functions takes ~26 min on the VM's CPU and was redone on every resume; the
    # result is cached in <out>/train_cache.pkl, keyed by everything that changes it (data file, model, window, rows).
    # The pickle is only ever written by this script into its own output dir, so loading it trusts nothing external.
    import hashlib
    import pickle
    key = hashlib.md5(json.dumps([os.path.abspath(f"{data}/train.jsonl"), os.path.getsize(f"{data}/train.jsonl"), args.model, args.max_len,
                                  args.strategy, args.max_pair_views, pair_mode, sorted(r["idx"] for r in train_rows)]
                                 + ([True] if args.mask_contradict else []), default=str).encode()).hexdigest()
    cache = f"{out}/train_cache.pkl"
    cached = None
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as fh:
                cached = pickle.load(fh)
            if cached.get("key") != key:
                log("train cache key differs - rebuilding"); cached = None
        except Exception as e:
            log(f"train cache unreadable ({e!r}) - rebuilding"); cached = None
    t_build = time.time()
    if pair_mode:
        items = cached["items"] if cached else make_items(train_rows, tok, args)
        views = [v for it in items for m in it["members"] if m["bce"] for v in m["views"]]
        log(f"pair mode: {sum(len(it['members']) == 2 for it in items)} vulnerable/fixed pairs, "
            f"{sum(len(it['match']) for it in items)} aligned flaw lines, {sum(len(it['members']) == 1 for it in items)} single views | "
            f"lam_line {args.lam_line} lam_fn {args.lam_fn} margin {args.margin}")
    else:
        ign = contradicting_lines(train_rows) if args.mask_contradict else {}
        if args.mask_contradict:
            log(f"mask_contradict: {sum(map(len, ign.values()))} lines in {len(ign)} fixed versions left out of the loss "
                f"(same text as a flaw line of the vulnerable partner)")
        views = cached["views"] if cached else [v for r in train_rows for v in build_views(tok, r["func"], r["flaw_lines"], args.strategy,
                                                                                            args.max_len, ign.get(r["idx"], ()))]
    log(f"training views {'loaded from cache' if cached else 'built'} in {time.time() - t_build:.0f}s")
    if not cached:
        try:
            with open(cache + ".tmp", "wb") as fh:
                pickle.dump({"key": key, "items": items if pair_mode else None, "views": None if pair_mode else views}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(cache + ".tmp", cache)
            log(f"train cache written ({os.path.getsize(cache) / 2**20:.0f} MB)")
        except Exception as e:
            log(f"train cache not written: {e!r}")
    n_pos = sum(v[3].count(1) for v in views); n_neg = sum(v[3].count(0) for v in views)
    pos_weight = min(args.max_pos_weight, n_neg / max(n_pos, 1))
    log(f"{len(views)} training views | lines: {n_pos} flaw / {n_neg} clean / {sum(v[3].count(-1) for v in views)} masked | "
        f"pos_weight {pos_weight:.1f} | loss {args.loss}" + (f" + {args.lam_bce} x bce" if args.loss == "listwise" else ""))
    valid_groups = group_info(valid)
    model = make_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    n_batches = len(item_batches(items, args.train_tokens, 0)) if pair_mode else len(token_batches(views, args.train_tokens, 0))
    steps = n_batches * args.epochs
    lw_C = max(1e-6, sum(1 for v in views if 1 in v[3]) / max(n_batches, 1))      # fixed from epoch 0's packing
    if args.loss == "listwise":
        log(f"listwise: {sum(1 for v in views if 1 in v[3])} views with a flaw line, C = {lw_C:.2f} per batch")
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and amp() == torch.float16)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    lossf_each = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
    ck_last, ck_best = f"{out}/ckpt_last.pt", f"{out}/ckpt_best.pt"
    start_ep, start_b, step, curve, best = 0, 0, 0, [], {"score": -1, "epoch": -1}
    n_eval = max(1, int(round(1 / args.eval_every)))          # evaluation points per epoch
    eval_at = sorted({min(n_batches, max(1, round(n_batches * k / n_eval))) for k in range(1, n_eval + 1)})
    log(f"{n_batches} batches/epoch, {steps} steps total, warmup {int(0.05 * steps)} | valid at batches {eval_at} of every epoch "
        f"| select {args.select} | patience {args.patience} epochs | amp {amp()} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    train_log = open(f"{out}/train_log.jsonl", "a")

    def validate(ep_f):
        """Score valid, log every metric, update best / curve.json / plots. ep_f = epochs completed (float)."""
        t0 = time.time()
        ls, fs = score_rows(model, tok, valid, args.strategy, args, device)
        m = evaluate(valid, ls, fs, *valid_groups)
        m["pairs_debiased"] = pair_acc_debiased(valid, fs)
        m["extra"] = valid_extra(valid, ls, fs, pos_weight)
        la = m.get("lines_ALL", {})
        if args.select == "top5_pair_debias":
            score = la.get("top5", 0) + m["pairs_debiased"]
        elif args.select == "top5_top10":
            score = la.get("top5", 0) + la.get("top10", 0)
        else:
            score = la.get("top5", 0) + (m["same_text_flaw_lines"]["vul_scored_higher_pct"] or 50)
        curve.append({"epoch": ep_f, "step": step, "valid": m, "select_score": score, "elapsed_min": (time.time() - T0) / 60})
        ex, px, fx = m["extra"], m["extra"].get("pairs", {}), m["extra"].get("function", {})
        log(f"valid ep {ep_f:.2f} step {step} ({time.time() - t0:.0f}s): lines Top-1/5/10 {la.get('top1', 0):.1f}/{la.get('top5', 0):.1f}/{la.get('top10', 0):.1f} "
            f"IFA {la.get('median_ifa', 0):.0f} | line BCE {ex['line_bce']:.4f} | pairs acc {px.get('acc', 0):.1f} debiased {px.get('acc_debiased', 0):.1f} "
            f"len-matched {px.get('acc_length_matched') or 0:.1f} top3mean {px.get('acc_top3mean', 0):.1f} net@50 {px.get('net_at50', 0):+.1f} "
            f"| fn ROC {fx.get('roc_auc', 0):.3f} debiased {fx.get('roc_auc_debiased', 0):.3f} | select {score:.1f} (best {best['score']:.1f} @ ep {best['epoch']})")
        log(f"valid ep {ep_f:.2f} full: {json.dumps(m)}")
        if score > best["score"]:
            best.update({"score": score, "epoch": ep_f, "step": step})
            torch.save({k: v.half() for k, v in model.state_dict().items()}, ck_best + ".tmp")
            os.replace(ck_best + ".tmp", ck_best)     # a kill mid-save never leaves a truncated best checkpoint
            log(f"saved best epoch {ep_f:.2f}")
        json.dump({"args": vars(args), "best": best, "curve": curve}, open(f"{out}/curve.json", "w"), indent=1)
        try:
            import plot_curve
            plot_curve.plot_run(out)
        except Exception as e:       # plotting must never stop a run
            log(f"plot skipped: {e!r}")

    def log_step(ep, bi, nb, loss, parts, t_ep, seen):
        lr = sched.get_last_lr()[0]
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        rate = seen / max(time.time() - t_ep, 1e-6)
        left = (steps - step) * (time.time() - t_ep) / max(bi + 1 - (start_b if ep == start_ep else 0), 1) / 60
        log(f"  ep {ep} b {bi + 1}/{nb} step {step}/{steps} loss {loss:.4f}{parts} | lr {lr:.2e} | {rate:.1f} items/s | mem {mem:.1f} GB | ETA {left:.0f} min")
        train_log.write(json.dumps({"step": step, "epoch": ep + (bi + 1) / nb, "loss": loss, "lr": lr, "items_per_s": rate, "mem_gb": mem,
                                    "elapsed_min": (time.time() - T0) / 60}) + "\n"); train_log.flush()
    if args.init and not os.path.exists(ck_last):
        st = torch.load(args.init, map_location="cpu")
        st = st["model"] if "model" in st else st
        model.load_state_dict({k: v.float() for k, v in st.items()})
        log(f"initialised from {args.init}")
    if os.path.exists(ck_last):
        st = torch.load(ck_last, map_location="cpu")
        model.load_state_dict({k: v.float() for k, v in st["model"].items()})
        start_ep, start_b, step, curve, best = st["epoch"], st["b"], st["step"], st["curve"], st["best"]
        for _ in range(step):
            sched.step()
        log(f"RESUMED epoch {start_ep} batch {start_b} step {step}")

    def save_last(ep, b):
        torch.save({"model": {k: v.half() for k, v in model.state_dict().items()}, "epoch": ep, "b": b, "step": step,
                    "curve": curve, "best": best}, ck_last + ".tmp")
        os.replace(ck_last + ".tmp", ck_last)
        log(f"saved checkpoint epoch {ep} batch {b} step {step}")

    for ep in range(start_ep, args.epochs):
        batches = item_batches(items, args.train_tokens, args.seed * 1000 + ep) if pair_mode else token_batches(views, args.train_tokens, args.seed * 1000 + ep)
        model.train()
        t_ep, run, seen, run_parts, n_run = time.time(), 0.0, 0, np.zeros(4), 0
        first = start_b if ep == start_ep else 0
        # the packing differs per epoch (shuffle), so the evaluation points are recomputed from this epoch's batch count
        eval_at = sorted({min(len(batches), max(1, round(len(batches) * k / n_eval))) for k in range(1, n_eval + 1)})
        log(f"epoch {ep}: {len(batches)} batches, starting at batch {first}, valid at batches {eval_at}")
        for bi in range(first, len(batches)):
            if pair_mode:
                (ids, seg, n_seg, y, mask, seg_u, n_u, u_func, n_func, lv, lf, fv, ff) = collate_items([items[i] for i in batches[bi]], device)
            else:
                ids, seg, n_seg, y, _ = collate([views[i] for i in batches[bi]], device)
                lens = [len(views[i][2]) for i in batches[bi]]
            seen += len(batches[bi])
            if y.numel() == 0:
                continue
            with torch.autocast(device.type, dtype=amp(), enabled=device.type == "cuda"):
                logits = model(ids, seg, n_seg, tok.pad_token_id)
            if pair_mode:
                z = logits.float()
                bce = (lossf_each(z, y) * mask).sum() / mask.sum().clamp(min=1)
                l_line, l_fn = pair_losses(z, seg_u, n_u, u_func, n_func, lv, lf, fv, ff, args.margin)
                loss = bce + args.lam_line * l_line + args.lam_fn * l_fn
                run_parts += [bce.item(), l_line.item(), l_fn.item(), 0.0]
            elif args.loss == "listwise":
                lw, bce, chance = listwise_loss(logits.float(), y, lens, lossf_each, lw_C)
                loss = lw + args.lam_bce * bce
                run_parts += [bce.item(), lw.item(), chance, 0.0]
            else:
                keep = y >= 0                                   # all lines unless --mask_contradict
                loss = lossf(logits.float()[keep], y[keep])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); step += 1; n_run += 1
            run_parts[3] += float(gnorm)
            if step % args.log_every == 0:
                # listwise vs its chance level (uniform scores): kill the run if it is not >10% below chance by ~0.5 epoch
                parts = (f" (bce {run_parts[0] / n_run:.3f} line-hinge {run_parts[1] / n_run:.3f} fn-hinge {run_parts[2] / n_run:.3f})" if pair_mode
                         else f" (listwise {run_parts[1] / n_run:.3f} chance {run_parts[2] / n_run:.3f} bce {run_parts[0] / n_run:.3f} "
                              f"gnorm {run_parts[3] / n_run:.2f})" if args.loss == "listwise" else "")
                log_step(ep, bi, len(batches), run / n_run, parts, t_ep, seen)
                run, run_parts, n_run = 0.0, np.zeros(4), 0
            if step % args.ckpt_every == 0:
                save_last(ep, bi + 1)
            if (bi + 1) in eval_at:
                ep_f = ep + (bi + 1) / len(batches)
                if not any(abs(c["epoch"] - ep_f) < 1e-6 for c in curve):      # not already done before a resume
                    validate(ep_f)
                    model.train()
                    if (bi + 1) < len(batches):
                        save_last(ep, bi + 1)
        save_last(ep + 1, 0)
        if args.patience and (ep + 1) - best["epoch"] >= args.patience:
            log(f"early stop: no improvement for {args.patience} epochs (best epoch {best['epoch']})")
            break
    train_log.close()
    if os.path.exists(ck_best):
        model.load_state_dict({k: v.float() for k, v in torch.load(ck_best, map_location="cpu").items()})
    ls, fs = score_rows(model, tok, test, args.strategy, args, device)
    hv, nt = group_info(test)
    res = {"experiment": f"exp5 line-level supervision on {os.path.basename(os.path.normpath(data))}" + (" + fix-contrastive losses" if pair_mode else ""),
           "args": vars(args), "best_epoch": best["epoch"], "valid_curve": curve, "test": evaluate(test, ls, fs, hv, nt)}
    res["test"]["extra"] = valid_extra(test, ls, fs, pos_weight)
    json.dump(res, open(f"{out}/results.json", "w"), indent=1)
    with open(f"{out}/test_scores.jsonl", "w") as fh:
        for r, s, f in zip(test, ls, fs):
            fh.write(json.dumps({"idx": r["idx"], "target": r["target"], "pair_idx": r["pair_idx"], "chars": len(r["func"]), "func": r["func"],
                                 "flaw_lines": r["flaw_lines"], "logit": round(float(f), 4), "line_scores": [round(float(x), 4) for x in s]}) + "\n")
    os.remove(ck_last) if os.path.exists(ck_last) else None
    log("test:", json.dumps(res["test"]))
    try:
        import plot_curve
        plot_curve.plot_run(out)
    except Exception as e:
        log(f"plot skipped: {e!r}")
    log("DONE")


def find_data(d):
    if d and os.path.exists(f"{d}/train.jsonl"):
        return d
    hits = glob.glob("/kaggle/input/**/train.jsonl", recursive=True) + glob.glob("**/secvul_lines/train.jsonl", recursive=True)
    if not hits:
        raise SystemExit("no prepared data (train.jsonl) found; run prepare_secvuleval.py first")
    return os.path.dirname(hits[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["head", "slide", "controls"], default="head")
    ap.add_argument("--model", default="neulab/codebert-c")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_tokens", type=int, default=8192, help="padded tokens per training batch (16 x 512)")
    ap.add_argument("--eval_tokens", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pos_weight", type=float, default=20.0)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: keep this many functions per split (pairs)")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (Ampere/Ada GPUs such as L4, A100)")
    ap.add_argument("--lam_line", type=float, default=0.0, help="weight of the line-level vulnerable-vs-fixed hinge")
    ap.add_argument("--lam_fn", type=float, default=0.0, help="weight of the function-level vulnerable-vs-fixed hinge")
    ap.add_argument("--margin", type=float, default=1.0, help="hinge margin in logits")
    ap.add_argument("--pair_batches", action="store_true", help="pair batching without pair losses (fair BCE-only control)")
    ap.add_argument("--loss", choices=["bce", "listwise"], default="bce",
                    help="listwise: softmax over each function's lines, -log mass on its flaw lines, + lam_bce x line BCE")
    ap.add_argument("--lam_bce", type=float, default=0.3, help="weight of the line BCE next to the listwise loss")
    ap.add_argument("--mask_contradict", action="store_true",
                    help="leave fixed-version lines with the same text as a flaw line of the pair out of the loss")
    ap.add_argument("--max_pair_views", type=int, default=4, help="max windows per function inside a pair item")
    ap.add_argument("--tag", default="", help="suffix for the run name")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many epochs without a better valid score (0 = off)")
    ap.add_argument("--init", default=None, help="checkpoint (ckpt_best.pt / ckpt_last.pt) to start from, e.g. a stage-1 model")
    ap.add_argument("--eval_every", type=float, default=1.0, help="validate every this fraction of an epoch (0.5 = twice per epoch)")
    ap.add_argument("--log_every", type=int, default=100, help="log the training loss every this many steps")
    ap.add_argument("--select", choices=["top5_same_text", "top5_pair_debias", "top5_top10"], default="top5_same_text",
                    help="epoch selection score on valid: Top-5 + same-text %% (old, biased by one function) or "
                         "Top-5 + length-debiased pair accuracy")
    ap.add_argument("--balance_length", action="store_true",
                    help="drop pairs whose fixed version is longer until they are as many as the shorter/equal ones")
    ap.add_argument("--extra_train", default=None, help="second data dir whose train.jsonl is mixed into training (e.g. synth_guards)")
    ap.add_argument("--extra_pairs", type=int, default=None, help="keep this many pairs of --extra_train (seed-fixed); default all")
    a = ap.parse_args()
    data = find_data(a.data)
    if a.strategy != "controls":
        from transformers import AutoTokenizer
        PAD_ID = AutoTokenizer.from_pretrained(a.model).pad_token_id
        if a.bf16:
            import torch
            AMP_DTYPE = torch.bfloat16
    name = a.strategy if a.strategy == "controls" else f"{a.strategy}_{a.model.split('/')[-1]}_{a.max_len}_s{a.seed}{a.tag}"
    out = a.out or (f"/kaggle/working/exp5/{name}" if os.path.isdir("/kaggle/working") else f"exp5_out/{name}")
    os.makedirs(out, exist_ok=True)
    log("data", data, "-> out", out)
    if a.strategy == "controls":
        controls(data, out, a)
    else:
        train(data, out, a)
