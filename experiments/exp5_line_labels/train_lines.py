"""Experiment 5: line-level supervision on SecVulEval.

Instead of a function label + attention (LineVul), every line gets its own label from the fix
(changed_statements): lines of the vulnerable version touched by the fix are 1, every other line - of the
vulnerable version, of its fixed version and of unrelated benign functions - is 0. The model:

    encoder (RoBERTa family) -> token states -> mean-pool the tokens of each line -> linear -> line logit
    function score = max line probability over the whole function (all views)

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
    rows = [json.loads(l) for l in open(f"{data}/{split}.jsonl")]
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
def build_views(tok, func, flaws, strategy, max_len):
    """-> list of (ids[max_len], tok_line[max_len] with -1 for special/pad, lines_in_view, labels_in_view)."""
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
    fl = set(flaws)
    views = []
    for a, b in spans:
        vid = [tok.cls_token_id] + ids[a:b] + [tok.sep_token_id]     # padded per batch in collate
        vl = [-1] + tline[a:b] + [-1]
        lines = sorted({l for l in vl if l >= 0})
        views.append((vid, vl, lines, [int(l in fl) for l in lines]))
    return views


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
    leak = {r["idx"] for r in train_all if r.get("in_primevul_test")}
    leak |= {r["pair_idx"] for r in train_all if r["idx"] in leak and r["pair_idx"] is not None}
    pair_mode = args.lam_line > 0 or args.lam_fn > 0 or args.pair_batches
    # add-only fixes have no labelled line; pair mode keeps them (function hinge only, line BCE masked)
    train_rows = [r for r in train_all if r["idx"] not in leak and (pair_mode or not (r["target"] == 1 and not r["flaw_lines"]))]
    log(f"dropped {len(leak)} train functions overlapping the PrimeVul test set (incl. pair partners)")
    valid = read_split(data, "valid", args.limit)
    test = read_split(data, "test", args.limit)
    log(f"train {len(train_rows)} functions{'' if pair_mode else ' (add-only fixes dropped)'} | valid {len(valid)} | test {len(test)} | device {device}")
    if pair_mode:
        items = make_items(train_rows, tok, args)
        views = [v for it in items for m in it["members"] if m["bce"] for v in m["views"]]
        log(f"pair mode: {sum(len(it['members']) == 2 for it in items)} vulnerable/fixed pairs, "
            f"{sum(len(it['match']) for it in items)} aligned flaw lines, {sum(len(it['members']) == 1 for it in items)} single views | "
            f"lam_line {args.lam_line} lam_fn {args.lam_fn} margin {args.margin}")
    else:
        views = [v for r in train_rows for v in build_views(tok, r["func"], r["flaw_lines"], args.strategy, args.max_len)]
    n_pos = sum(sum(v[3]) for v in views); n_neg = sum(len(v[3]) for v in views) - n_pos
    pos_weight = min(args.max_pos_weight, n_neg / max(n_pos, 1))
    log(f"{len(views)} training views | lines: {n_pos} flaw / {n_neg} clean | pos_weight {pos_weight:.1f}")
    valid_groups = group_info(valid)
    model = make_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    n_batches = len(item_batches(items, args.train_tokens, 0)) if pair_mode else len(token_batches(views, args.train_tokens, 0))
    steps = n_batches * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and amp() == torch.float16)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    lossf_each = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
    ck_last, ck_best = f"{out}/ckpt_last.pt", f"{out}/ckpt_best.pt"
    start_ep, start_b, step, curve, best = 0, 0, 0, [], {"score": -1, "epoch": -1}
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
        t_ep, run, seen, run_parts = time.time(), 0.0, 0, np.zeros(3)
        first = start_b if ep == start_ep else 0
        for bi in range(first, len(batches)):
            if pair_mode:
                (ids, seg, n_seg, y, mask, seg_u, n_u, u_func, n_func, lv, lf, fv, ff) = collate_items([items[i] for i in batches[bi]], device)
            else:
                ids, seg, n_seg, y, _ = collate([views[i] for i in batches[bi]], device)
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
                run_parts += [bce.item(), l_line.item(), l_fn.item()]
            else:
                loss = lossf(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); step += 1
            if step % 100 == 0:
                parts = f" (bce {run_parts[0] / 100:.3f} line-hinge {run_parts[1] / 100:.3f} fn-hinge {run_parts[2] / 100:.3f})" if pair_mode else ""
                log(f"  ep {ep} step {step}/{steps} loss {run / 100:.4f}{parts} | {seen / (time.time() - t_ep):.1f} items/s")
                run, run_parts = 0.0, np.zeros(3)
            if step % args.ckpt_every == 0:
                save_last(ep, bi + 1)
        ls, fs = score_rows(model, tok, valid, args.strategy, args, device)
        m = evaluate(valid, ls, fs, *valid_groups)
        score = m.get("lines_ALL", {}).get("top5", 0) + (m["same_text_flaw_lines"]["vul_scored_higher_pct"] or 50)
        curve.append({"epoch": ep, "valid": m, "select_score": score})
        log(f"valid ep {ep}: {json.dumps(m)}")
        if score > best["score"]:
            best = {"score": score, "epoch": ep}
            torch.save({k: v.half() for k, v in model.state_dict().items()}, ck_best)
            log(f"saved best epoch {ep}")
        save_last(ep + 1, 0)
        if args.patience and ep - best["epoch"] >= args.patience:
            log(f"early stop: no improvement for {args.patience} epochs (best epoch {best['epoch']})")
            break
    if os.path.exists(ck_best):
        model.load_state_dict({k: v.float() for k, v in torch.load(ck_best, map_location="cpu").items()})
    ls, fs = score_rows(model, tok, test, args.strategy, args, device)
    hv, nt = group_info(test)
    res = {"experiment": "exp5 line-level supervision on SecVulEval" + (" + fix-contrastive losses" if pair_mode else ""), "args": vars(args), "best_epoch": best["epoch"],
           "valid_curve": curve, "test": evaluate(test, ls, fs, hv, nt)}
    json.dump(res, open(f"{out}/results.json", "w"), indent=1)
    with open(f"{out}/test_scores.jsonl", "w") as fh:
        for r, s in zip(test, ls):
            fh.write(json.dumps({"idx": r["idx"], "target": r["target"], "pair_idx": r["pair_idx"],
                                 "flaw_lines": r["flaw_lines"], "line_scores": [round(float(x), 4) for x in s]}) + "\n")
    os.remove(ck_last) if os.path.exists(ck_last) else None
    log("test:", json.dumps(res["test"]))
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
    ap.add_argument("--max_pair_views", type=int, default=4, help="max windows per function inside a pair item")
    ap.add_argument("--tag", default="", help="suffix for the run name")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many epochs without a better valid score (0 = off)")
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
