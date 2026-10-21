"""Experiment 5 on TPU (PyTorch/XLA): the same line-level model, data and metrics as train_lines.py, rewritten
for static shapes, which XLA needs (every new tensor shape triggers a recompilation of the whole graph).

Static shapes: views go into length buckets (default 512 / 1024 / 2048 tokens); every batch of a bucket has the
same number of views (budget // bucket) and the same number of line slots per view (bucket // 2); short batches
are filled with empty dummy views. Line pooling scatters token states into a fixed [views * slots + 1, H] buffer
(the last row collects ignored tokens), so the graph never changes shape. bf16 autocast, AdamW, linear warmup,
gradient checkpointing for buckets >= 1024 tokens.

Logging (stdout + <out>/train.log + <out>/progress.json), every --log_every steps: loss, steps/s, views/s,
tokens/s, number of XLA compilations so far, ETA; each epoch: full validation metrics. --bench N runs N training
steps, prints timing/compilation statistics and exits (used to size the real run).
After training: test metrics on SecVulEval and the PrimeVul protocol of TN1-TN3 (eval_primevul.primevul_rows).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lines as tl  # noqa: E402

T0 = time.time()
LOG = None
CLS_ID, SEP_ID = 0, 2


def log(*a):
    msg = f"[{time.time() - T0:8.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(msg + "\n")


def compiles():
    try:
        import torch_xla.debug.metrics as met
        d = met.metric_data("CompileTime")
        return int(d[0]) if d else 0
    except Exception:
        return -1


# ----------------------------------------------------------------------------- static batches
def bucket_of(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


def make_batches(views, buckets, budget, seed=None):
    """-> list of (bucket_len, [view indices]) with exactly budget // bucket_len slots per batch (padded later)."""
    by = {b: [] for b in buckets}
    for i, v in enumerate(views):
        by[bucket_of(len(v[0]), buckets)].append(i)
    out = []
    for b, idx in by.items():
        if seed is not None:
            random.Random(seed + b).shuffle(idx)
        n = max(1, budget // b)
        out += [(b, idx[k:k + n]) for k in range(0, len(idx), n)]
    if seed is not None:
        random.Random(seed).shuffle(out)
    return out


def static_tensors(views, L, n_views, pad_id):
    """ids [n_views, L]; seg [n_views, L] into a buffer of n_views*S rows (+1 dump row); labels/mask [n_views*S]."""
    S = L // 2
    dump = n_views * S
    ids = np.full((n_views, L), pad_id, dtype=np.int64)
    ids[:, 0], ids[:, 1] = CLS_ID, SEP_ID          # filler views: <cls><sep> so attention never sees an all-masked row
    seg = np.full((n_views, L), dump, dtype=np.int64)
    lab = np.zeros(n_views * S, dtype=np.float32)
    msk = np.zeros(n_views * S, dtype=np.float32)
    owners = []                      # (view position, slot) -> line number, for scoring
    for b, v in enumerate(views):
        vid, vl, lines, labels = v
        vid, vl = vid[:L], vl[:L]
        ids[b, :len(vid)] = vid
        slot = {l: j for j, l in enumerate(lines[:S])}
        for t, l in enumerate(vl):
            if l >= 0 and l in slot:
                seg[b, t] = b * S + slot[l]
        for j, (l, y) in enumerate(zip(lines[:S], labels[:S])):
            lab[b * S + j] = y
            msk[b * S + j] = 1.0
            owners.append((b, j, l))
    return ids, seg, lab, msk, owners, S


def make_model(name, grad_ckpt):
    import torch
    from transformers import AutoModel

    class LineModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            kw = {"attn_implementation": "sdpa"}
            try:
                self.enc = AutoModel.from_pretrained(name, add_pooling_layer=False, **kw)
            except TypeError:
                self.enc = AutoModel.from_pretrained(name, **kw)
            if grad_ckpt:
                self.enc.gradient_checkpointing_enable()
            h = self.enc.config.hidden_size
            self.drop = torch.nn.Dropout(0.1)
            self.head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Dropout(0.1), torch.nn.Linear(h, 1))

        def forward(self, ids, seg, n_rows, pad_id):
            hs = self.enc(input_ids=ids, attention_mask=(ids != pad_id).long()).last_hidden_state
            hs = self.drop(hs).reshape(-1, hs.size(-1)).float()
            seg = seg.reshape(-1)
            pooled = torch.zeros(n_rows + 1, hs.size(-1), device=hs.device).index_add(0, seg, hs)
            cnt = torch.zeros(n_rows + 1, device=hs.device).index_add(0, seg, torch.ones_like(seg, dtype=torch.float32))
            pooled = pooled[:n_rows] / cnt[:n_rows].clamp(min=1).unsqueeze(1)
            return self.head(pooled).squeeze(-1)

    return LineModel()


def score_rows_xla(model, tok, rows, strategy, args, dev):
    """Same contract as train_lines.score_rows: per-line logits (unseen = -1e9) and function logit = max."""
    import torch
    import torch_xla.core.xla_model as xm
    model.eval()
    allv, own = [], []
    for i, r in enumerate(rows):
        for v in tl.build_views(tok, r["func"], r.get("flaw_lines", []), strategy, args.max_len):
            allv.append(v); own.append(i)
    line_scores = [np.full(r["n_lines"], -np.inf) for r in rows]
    batches = make_batches(allv, args.buckets, args.eval_tokens)
    t0 = time.time()
    with torch.no_grad():
        for k, (L, bidx) in enumerate(batches):
            n_views = max(1, args.eval_tokens // L)
            ids, seg, _, msk, owners, S = static_tensors([allv[i] for i in bidx], L, n_views, tok.pad_token_id)
            with torch.autocast("xla", dtype=torch.bfloat16, enabled=args.bf16):
                z = model(torch.from_numpy(ids).to(dev), torch.from_numpy(seg).to(dev), n_views * S, tok.pad_token_id)
            xm.mark_step()
            z = z.float().cpu().numpy()
            for b, j, l in owners:
                i = own[bidx[b]]
                line_scores[i][l] = max(line_scores[i][l], z[b * S + j])
            if k % 200 == 0:
                log(f"    scoring {k}/{len(batches)} batches ({len(allv)} views) | compiles {compiles()} | {time.time() - t0:.0f}s")
    fn = [float(s.max()) if len(s) and np.isfinite(s.max()) else -1e9 for s in line_scores]
    for s in line_scores:
        s[~np.isfinite(s)] = -1e9
    return line_scores, fn


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["head", "slide"], default="slide")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--buckets", default="512,1024,2048")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_tokens", type=int, default=8192)
    ap.add_argument("--eval_tokens", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pos_weight", type=float, default=20.0)
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--bench", type=int, default=0, help="run N training steps, report speed/compilations, exit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--primevul", default="/kaggle/tmp/primevul")
    ap.add_argument("--skip_primevul", action="store_true")
    a = ap.parse_args()
    a.buckets = sorted(int(x) for x in a.buckets.split(",") if int(x) <= a.max_len)
    a.bf16 = not a.no_bf16
    name = f"xla_{a.strategy}_{a.model.split('/')[-1]}_{a.max_len}_s{a.seed}" + ("_bench" if a.bench else "")
    out = a.out or f"/kaggle/working/exp5/{name}"
    os.makedirs(out, exist_ok=True)
    LOG = f"{out}/train.log"

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    dev = xm.xla_device()
    log(f"torch {torch.__version__} | torch_xla {torch_xla.__version__} | device {dev} | {xm.xla_device_hw(dev)} | args {vars(a)}")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    data = tl.find_data(a.data)
    tok = AutoTokenizer.from_pretrained(a.model)
    tl.PAD_ID = tok.pad_token_id
    global CLS_ID, SEP_ID
    CLS_ID, SEP_ID = tok.cls_token_id, tok.sep_token_id

    train_all = tl.read_split(data, "train", a.limit)
    leak = {r["idx"] for r in train_all if r.get("in_primevul_test")}
    leak |= {r["pair_idx"] for r in train_all if r["idx"] in leak and r["pair_idx"] is not None}
    train_rows = [r for r in train_all if r["idx"] not in leak and not (r["target"] == 1 and not r["flaw_lines"])]
    valid, test = tl.read_split(data, "valid", a.limit), tl.read_split(data, "test", a.limit)
    log(f"train {len(train_rows)} functions (dropped {len(leak)} overlapping PrimeVul test, and add-only fixes) | valid {len(valid)} | test {len(test)}")
    views = [v for r in train_rows for v in tl.build_views(tok, r["func"], r["flaw_lines"], a.strategy, a.max_len)]
    n_pos = sum(sum(v[3]) for v in views); n_neg = sum(len(v[3]) for v in views) - n_pos
    pos_weight = min(a.max_pos_weight, n_neg / max(n_pos, 1))
    lens = np.array([len(v[0]) for v in views])
    log(f"{len(views)} views | tokens/view median {np.median(lens):.0f} p90 {np.percentile(lens, 90):.0f} | "
        f"per bucket {[(b, int(sum(bucket_of(n, a.buckets) == b for n in lens))) for b in a.buckets]} | "
        f"lines {n_pos} flaw / {n_neg} clean | pos_weight {pos_weight:.1f}")

    model = make_model(a.model, grad_ckpt=not a.no_grad_ckpt and a.max_len >= 1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, eps=1e-8, weight_decay=0.01)
    n_batches = len(make_batches(views, a.buckets, a.train_tokens, 0))
    steps = a.bench or n_batches * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    pw = torch.tensor(pos_weight, device=dev)
    valid_groups = tl.group_info(valid)
    curve, best, step = [], {"score": -1, "epoch": -1}, 0
    log(f"{n_batches} batches/epoch | {steps} steps total")

    def train_step(L, bidx):
        n_views = max(1, a.train_tokens // L)
        ids, seg, lab, msk, _, S = static_tensors([views[i] for i in bidx], L, n_views, tok.pad_token_id)
        ids, seg, lab, msk = (torch.from_numpy(x).to(dev) for x in (ids, seg, lab, msk))
        with torch.autocast("xla", dtype=torch.bfloat16, enabled=a.bf16):
            z = model(ids, seg, n_views * S, tok.pad_token_id)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z.float(), lab, pos_weight=pw, reduction="none")
        loss = (loss * msk).sum() / msk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xm.optimizer_step(opt)
        sched.step()
        return loss

    for ep in range(a.epochs):
        batches = make_batches(views, a.buckets, a.train_tokens, a.seed * 1000 + ep)
        model.train()
        t_ep, t_win, seen, toks, losses = time.time(), time.time(), 0, 0, []
        for bi, (L, bidx) in enumerate(batches):
            loss = train_step(L, bidx)
            step += 1
            seen += len(bidx); toks += sum(len(views[i][0]) for i in bidx)
            losses.append(loss)
            if step % a.log_every == 0 or step <= 5:
                xm.mark_step()
                lv = float(torch.stack(losses).mean().cpu()); losses = []
                el = time.time() - t_ep
                rate = (bi + 1) / el
                eta = (len(batches) - bi - 1) / max(rate, 1e-9) + (a.epochs - ep - 1) * len(batches) / max(rate, 1e-9)
                log(f"  ep {ep} batch {bi + 1}/{len(batches)} step {step}/{steps} | loss {lv:.4f} | {rate:.2f} steps/s "
                    f"{seen / el:.1f} views/s {toks / el:.0f} tok/s | bucket {L} | compiles {compiles()} | lr {sched.get_last_lr()[0]:.2e} | ETA {eta / 60:.0f} min")
                json.dump({"epoch": ep, "batch": bi + 1, "batches": len(batches), "step": step, "steps": steps, "loss": lv,
                           "views_per_s": seen / el, "compiles": compiles(), "eta_min": eta / 60, "elapsed_min": (time.time() - T0) / 60},
                          open(f"{out}/progress.json", "w"))
            if a.bench and step >= a.bench:
                xm.mark_step()
                log(f"BENCH done: {step} steps in {time.time() - t_ep:.0f}s, compiles {compiles()}; "
                    f"estimated epoch {len(make_batches(views, a.buckets, a.train_tokens, 0)) / ((bi + 1) / (time.time() - t_ep)) / 60:.0f} min "
                    f"(includes first-time compilation; real rate is the later log lines)")
                try:
                    import torch_xla.debug.metrics as met
                    log(met.short_metrics_report())
                except Exception as e:
                    log("metrics report unavailable:", e)
                return
        xm.mark_step()
        log(f"epoch {ep} trained in {(time.time() - t_ep) / 60:.1f} min; validating")
        ls, fs = score_rows_xla(model, tok, valid, a.strategy, a, dev)
        m = tl.evaluate(valid, ls, fs, *valid_groups)
        score = m.get("lines_ALL", {}).get("top5", 0) + (m["same_text_flaw_lines"]["vul_scored_higher_pct"] or 50)
        curve.append({"epoch": ep, "valid": m, "select_score": score})
        log(f"VALID ep {ep}: lines_ALL {json.dumps(m.get('lines_ALL'))} | function {json.dumps(m['function'])} | "
            f"pairs {json.dumps(m.get('pairs'))} | same_text {json.dumps(m['same_text_flaw_lines'])}")
        if score > best["score"]:
            best = {"score": score, "epoch": ep}
            xm.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, f"{out}/ckpt_best.pt")
            log(f"saved best epoch {ep}")
    model.load_state_dict({k: v.float() for k, v in torch.load(f"{out}/ckpt_best.pt", map_location="cpu").items()})
    model.to(dev)
    log("scoring SecVulEval test")
    ls, fs = score_rows_xla(model, tok, test, a.strategy, a, dev)
    res = {"experiment": "exp5 line-level supervision on SecVulEval (TPU)", "args": vars(a), "best_epoch": best["epoch"],
           "valid_curve": curve, "test": tl.evaluate(test, ls, fs, *tl.group_info(test))}
    json.dump(res, open(f"{out}/results.json", "w"), indent=1)
    log("TEST SecVulEval:", json.dumps(res["test"]))
    if not a.skip_primevul:
        import eval_primevul as ep_
        pair_rows, test_rows = ep_.primevul_rows(a.primevul, 10 if a.limit else None)
        log(f"scoring PrimeVul: {len(pair_rows) // 2} pairs, {len(test_rows)} test functions")
        pres = {"trained_on": "SecVulEval (exp5, TPU)", "model": a.model, "strategy": a.strategy, "max_len": a.max_len}
        ls, fs = score_rows_xla(model, tok, pair_rows, a.strategy, a, dev)
        pres["pairs_and_lines"] = tl.evaluate(pair_rows, ls, fs, *tl.group_info(pair_rows))
        log("PRIMEVUL paired:", json.dumps(pres["pairs_and_lines"]))
        ls, fs = score_rows_xla(model, tok, test_rows, a.strategy, a, dev)
        pres["full_test"] = tl.evaluate(test_rows, ls, fs, *tl.group_info(test_rows))["function"]
        log("PRIMEVUL full test:", json.dumps(pres["full_test"]))
        json.dump(pres, open(f"{out}/primevul_results.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
