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
        tie = np.isclose(pv, pf, atol=1e-6)
        res["pairs"] = {"n": int(len(pv)), "ties": int(tie.sum()),
                        "ranking_acc_ties_half": float((((pv > pf) & ~tie) + 0.5 * tie).mean() * 100),
                        "ranking_acc_non_tied": float((pv > pf)[~tie].mean() * 100) if (~tie).any() else None}
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
        vid = [tok.cls_token_id] + ids[a:b] + [tok.sep_token_id]
        vl = [-1] + tline[a:b] + [-1]
        pad = max_len - len(vid)
        vid += [tok.pad_token_id] * pad
        vl += [-1] * pad
        lines = sorted({l for l in vl if l >= 0})
        views.append((vid, vl, lines, [int(l in fl) for l in lines]))
    return views


def make_model(model_name):
    import torch
    from transformers import AutoModel

    class LineModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
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


def collate(batch, device):
    import torch
    ids = torch.tensor([v[0] for v in batch], device=device)
    seg, labels, owners, off = [], [], [], 0
    for v in batch:
        loc = {l: off + j for j, l in enumerate(v[2])}
        seg.append([loc[l] if l >= 0 else -1 for l in v[1]])
        labels.extend(v[3])
        owners.extend(v[2])
        off += len(v[2])
    return ids, torch.tensor(seg, device=device), off, torch.tensor(labels, device=device, dtype=torch.float), owners


def score_rows(model, tok, rows, strategy, args, device):
    import torch
    model.eval()
    allv, own = [], []
    for i, r in enumerate(rows):
        for v in build_views(tok, r["func"], r["flaw_lines"], strategy, args.max_len):
            allv.append(v); own.append(i)
    line_scores = [np.zeros(r["n_lines"]) for r in rows]
    with torch.no_grad():
        for b in range(0, len(allv), args.eval_bs):
            chunk = allv[b:b + args.eval_bs]
            ids, seg, n_seg, _, owners = collate(chunk, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                p = torch.sigmoid(model(ids, seg, n_seg, tok.pad_token_id)).float().cpu().numpy()
            k = 0
            for j, v in enumerate(chunk):
                i = own[b + j]
                for l in v[2]:
                    line_scores[i][l] = max(line_scores[i][l], p[k]); k += 1
    return line_scores, [float(s.max()) if len(s) else 0.0 for s in line_scores]


def train(data, out, args):
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    train_rows = [r for r in read_split(data, "train", args.limit) if not (r["target"] == 1 and not r["flaw_lines"])]
    valid = read_split(data, "valid", args.limit)
    test = read_split(data, "test", args.limit)
    log(f"train {len(train_rows)} functions (add-only fixes dropped) | valid {len(valid)} | test {len(test)} | device {device}")
    views = [v for r in train_rows for v in build_views(tok, r["func"], r["flaw_lines"], args.strategy, args.max_len)]
    n_pos = sum(sum(v[3]) for v in views); n_neg = sum(len(v[3]) for v in views) - n_pos
    pos_weight = min(args.max_pos_weight, n_neg / max(n_pos, 1))
    log(f"{len(views)} training views | lines: {n_pos} flaw / {n_neg} clean | pos_weight {pos_weight:.1f}")
    model = make_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    steps = math.ceil(len(views) / args.bs) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
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

    for ep in range(start_ep, args.epochs):
        order = list(range(len(views)))
        random.Random(args.seed * 1000 + ep).shuffle(order)
        model.train()
        t_ep, run = time.time(), 0.0
        first = start_b if ep == start_ep else 0
        for bi, b in enumerate(range(first * args.bs, len(order), args.bs), start=first):
            ids, seg, n_seg, y, _ = collate([views[i] for i in order[b:b + args.bs]], device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(ids, seg, n_seg, tok.pad_token_id)
            loss = lossf(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); step += 1
            if step % 100 == 0:
                log(f"  ep {ep} step {step}/{steps} loss {run / 100:.4f} | {(b - first * args.bs + args.bs) / (time.time() - t_ep):.1f} views/s")
                run = 0.0
            if step % args.ckpt_every == 0:
                save_last(ep, bi + 1)
        ls, fs = score_rows(model, tok, valid, args.strategy, args, device)
        hv = [head_visibility(tok, r["func"], args.max_len) for r in valid]
        nt = [len(tok(r["func"], add_special_tokens=False)["input_ids"]) for r in valid]
        m = evaluate(valid, ls, fs, hv, nt)
        score = m.get("lines_ALL", {}).get("top5", 0) + m["function"]["auprc"] * 100
        curve.append({"epoch": ep, "valid": m, "select_score": score})
        log(f"valid ep {ep}: {json.dumps(m)}")
        if score > best["score"]:
            best = {"score": score, "epoch": ep}
            torch.save({k: v.half() for k, v in model.state_dict().items()}, ck_best)
            log(f"saved best epoch {ep}")
        save_last(ep + 1, 0)
    if os.path.exists(ck_best):
        model.load_state_dict({k: v.float() for k, v in torch.load(ck_best, map_location="cpu").items()})
    ls, fs = score_rows(model, tok, test, args.strategy, args, device)
    hv = [head_visibility(tok, r["func"], args.max_len) for r in test]
    nt = [len(tok(r["func"], add_special_tokens=False)["input_ids"]) for r in test]
    res = {"experiment": "exp5 line-level supervision on SecVulEval", "args": vars(args), "best_epoch": best["epoch"],
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
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pos_weight", type=float, default=20.0)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: keep this many functions per split (pairs)")
    a = ap.parse_args()
    data = find_data(a.data)
    name = a.strategy if a.strategy == "controls" else f"{a.strategy}_{a.model.split('/')[-1]}_{a.max_len}_s{a.seed}"
    out = a.out or (f"/kaggle/working/exp5/{name}" if os.path.isdir("/kaggle/working") else f"exp5_out/{name}")
    os.makedirs(out, exist_ok=True)
    log("data", data, "-> out", out)
    if a.strategy == "controls":
        controls(data, out, a)
    else:
        train(data, out, a)
