"""Experiment 3 (pilot): retrain LineVul (CodeBERT + LineVul head) on a small PrimeVul subset, once per
input strategy, and evaluate every model with the SAME protocol as experiments 1/2.

Why a pilot: the published checkpoint was trained on Big-Vul and does not transfer to PrimeVul (exp1/exp2),
so strategies can only be compared after training on PrimeVul. The full split (184k functions x 10 epochs)
is ~45 T4-hours; this subset answers "does training WITH the strategy change function-level behaviour?"
in ~1-2 T4-hours per model.

Subset (seed-fixed): every vulnerable function of primevul_train (5,574) + `benign_ratio` x as many benign
functions sampled at random. Validation: same recipe on primevul_valid; used to pick the epoch and the
decision threshold (never the test set). Test: full primevul_test (25,911) + 564 pairs, unchanged.

Strategies (training views):
  head    first 510 tokens                      (LineVul as published)
  slide   510-token windows, stride 256; each window is a training sample inheriting the function label
          (LLM4AVL-style); at most `max_windows` random windows per function per epoch; inference = max over
          all windows (as exp2 slide_max)
  slice   anchor-based slice from exp2 (one view)
  length  control: logistic regression on [log tokens, log lines] - no code content at all

Runs ON the Colab VM after exp1/setup_vm.py; imports exp1 and exp2 modules from /content/exp1/code.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve
from transformers import RobertaConfig, RobertaModel, get_linear_schedule_with_warmup

sys.path.insert(0, "/content/exp1/code")
import eval_linevul_primevul as e1  # noqa: E402
import eval_context_strategies as e2  # noqa: E402

ROOT = "/content/exp3"
DATA, DEV, tok = e1.DATA, e1.DEV, e1.tok
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- data
def subset(rows, benign_ratio, seed):
    vul = [r for r in rows if r["target"] == 1]
    ben = [r for r in rows if r["target"] == 0]
    rng = random.Random(seed)
    ben = rng.sample(ben, min(len(ben), benign_ratio * len(vul)))
    out = vul + ben
    rng.shuffle(out)
    return out


def training_views(rows, strategy, max_windows, rng):
    """List of (ids[512], label) for one epoch."""
    out = []
    for r in rows:
        if strategy == "slide":
            vs = e2.views_slide(r["func"])
            if len(vs) > max_windows:
                vs = rng.sample(vs, max_windows)
        else:
            vs = e2.STRATEGIES[strategy](r["func"])
        out.extend((v[0], r["target"]) for v in vs)
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------------- model
def fresh_model():
    cfg = RobertaConfig.from_pretrained(e1.MODEL_NAME)
    cfg._attn_implementation = "eager"
    model = e1.LineVul(cfg)
    pre = RobertaModel.from_pretrained(e1.MODEL_NAME, add_pooling_layer=False)
    missing, unexpected = model.roberta.load_state_dict(pre.state_dict(), strict=False)
    log("codebert-base loaded | missing:", [m for m in missing if "position_ids" not in m], "| unexpected:", unexpected)
    return model.to(DEV)


def thresholds_from_valid(y, p):
    """Decision thresholds chosen on the validation subset: best-F1, and the VD-S threshold (FPR <= 0.5%)."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    best = int(np.nanargmax(f1[:-1]))
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= 0.005)[0]
    return {"f1_threshold": float(th[best]), "valid_f1_at_best": float(f1[best]),
            "vds_threshold": float(thr[ok[-1]] if len(ok) else thr[0])}


def eval_with_thresholds(y, p, thr):
    out = e1.function_level(y, p)  # threshold 0.5 + oracle VD-S, as before
    pred = (p >= thr["f1_threshold"]).astype(int)
    out["at_valid_f1_threshold"] = {"threshold": thr["f1_threshold"], "f1": float(f1_score(y, pred, zero_division=0)),
                                    "flag_rate": float(pred.mean())}
    pred = (p >= thr["vds_threshold"]).astype(int)
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    out["vd_s_valid_threshold"] = {"threshold": thr["vds_threshold"], "vd_s": fn / max(fn + tp, 1), "actual_fpr": fp / max(fp + tn, 1)}
    return out


# ----------------------------------------------------------------------------- training
def train(strategy, args):
    name = strategy
    out_dir = f"{ROOT}/results/{name}"
    ckpt_dir = f"{ROOT}/ckpt"
    os.makedirs(out_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    train_rows = subset(e1.read_jsonl(f"{DATA}/primevul_train.jsonl"), args.benign_ratio, args.seed)
    valid_rows = subset(e1.read_jsonl(f"{DATA}/primevul_valid.jsonl"), args.benign_ratio, args.seed)
    yv = np.array([r["target"] for r in valid_rows])
    log(f"train subset {len(train_rows)} ({sum(r['target'] for r in train_rows)} vulnerable) | valid subset {len(valid_rows)}")

    model = fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    n_views_epoch = len(training_views(train_rows[:200], strategy, args.max_windows, random.Random(0))) / 200 * len(train_rows)
    total_steps = int(math.ceil(n_views_epoch / args.bs)) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, 0, total_steps)
    scaler = torch.amp.GradScaler("cuda")
    curve, best = [], {"auprc": -1}
    step = 0
    for ep in range(args.epochs):
        views = training_views(train_rows, strategy, args.max_windows, rng)
        log(f"epoch {ep}: {len(views)} training views")
        model.train()
        run_loss, t_ep = 0.0, time.time()
        for b in range(0, len(views), args.bs):
            chunk = views[b:b + args.bs]
            ids = torch.tensor([c[0] for c in chunk], device=DEV)
            lab = torch.tensor([c[1] for c in chunk], device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                prob = model(ids)
            loss = torch.nn.functional.nll_loss(torch.log(prob.float() + 1e-10), lab)  # LineVul: CE on softmax probs
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item(); step += 1
            if step % 100 == 0:
                log(f"  ep {ep} step {step}/{total_steps} loss {run_loss / 100:.4f} | {(b + len(chunk)) / (time.time() - t_ep):.1f} samples/s")
                run_loss = 0.0
        # validation (threshold-free) -> pick epoch
        model.eval()
        pv, _, _ = e2.score_functions(model, [r["func"] for r in valid_rows], strategy)
        pv = pv["max"]
        thr = thresholds_from_valid(yv, pv)
        m = e1.function_level(yv, pv)
        rec = {"epoch": ep, "valid_auprc": m["auprc"], "valid_roc_auc": m["roc_auc"], "valid_f1_at_0.5": m["f1"], **thr}
        curve.append(rec); log("valid:", json.dumps(rec))
        if m["auprc"] > best["auprc"]:
            best = {"auprc": m["auprc"], "epoch": ep, "thresholds": thr}
            torch.save(model.state_dict(), f"{ckpt_dir}/{name}_best.bin")
        with open(f"{out_dir}/valid_curve.json", "w") as fh:
            json.dump(curve, fh, indent=1)
    log(f"best epoch {best['epoch']} (valid AUPRC {best['auprc']:.4f})")
    model.load_state_dict(torch.load(f"{ckpt_dir}/{name}_best.bin", map_location=DEV))
    model.eval()
    evaluate(model, strategy, name, best, out_dir, args)


# ----------------------------------------------------------------------------- evaluation (same protocol as exp2)
def evaluate(model, strategy, name, best, out_dir, args):
    test = e1.read_jsonl(f"{DATA}/primevul_test.jsonl")
    pairs = e1.build_pairs(e1.read_jsonl(f"{DATA}/primevul_test_paired.jsonl"))
    y = np.array([r["target"] for r in test])
    thr = best["thresholds"]
    res = {"experiment": "exp3 pilot retrain", "strategy": strategy, "args": vars(args), "best_epoch": best["epoch"],
           "valid_thresholds": thr}
    _, _, head_vis = e2.score_functions(model, [v["func"] for v, _ in pairs], "head", want_lines=True)
    pv, ls, vis = e2.score_functions(model, [v["func"] for v, _ in pairs], strategy, want_lines=True)
    pf, _, _ = e2.score_functions(model, [f["func"] for _, f in pairs], strategy)
    pv, pf = pv["max"], pf["max"]
    res["pairs"] = e1.pair_metrics(pairs, pv, pf)
    res["pairs"]["pairs_identical_after_processing_pct"] = e2.pairs_identical(pairs, strategy)
    # pair outcomes also at the validation-chosen threshold
    t = thr["f1_threshold"]; cv, cf = pv >= t, pf >= t
    res["pairs"]["at_valid_f1_threshold"] = {"P-C": int((cv & ~cf).sum()), "P-V": int((cv & cf).sum()), "P-B": int((~cv & ~cf).sum()), "P-R": int((~cv & cf).sum())}
    recs = e2.localization_from_scores(pairs, pv, ls, vis, head_vis)
    res["localization"] = e1.summarize_localization(recs)
    res["localization"]["flaw_lines_visible_under_this_strategy_pct"] = sum(x["n_flaw_visible_this"] for x in recs) / sum(x["n_flaw"] for x in recs) * 100
    with open(f"{out_dir}/localization.jsonl", "w") as fh:
        for x in recs:
            fh.write(json.dumps(x) + "\n")
    p, _, _ = e2.score_functions(model, [r["func"] for r in test], strategy)
    p = p["max"]
    res["full_test"] = eval_with_thresholds(y, p, thr)
    with open(f"{out_dir}/full_test.jsonl", "w") as fh:
        for r, pp in zip(test, p):
            fh.write(json.dumps({"idx": r["idx"], "target": r["target"], "prob": float(pp)}) + "\n")
    with open(f"{out_dir}/pair_predictions.jsonl", "w") as fh:
        for (v, fx), a, b in zip(pairs, pv, pf):
            fh.write(json.dumps({"idx_vuln": v["idx"], "idx_fixed": fx["idx"], "prob_vuln": float(a), "prob_fixed": float(b)}) + "\n")
    res["elapsed_s"] = time.time() - T0
    with open(f"{out_dir}/results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"[{name}] pairs: {json.dumps(res['pairs'])}")
    log(f"[{name}] full test: {json.dumps(res['full_test'])}")
    log(f"[{name}] localization: {json.dumps(res['localization'])}")
    log(f"[{name}] DONE")


# ----------------------------------------------------------------------------- length-only control
def length_control(args):
    name = "length"
    out_dir = f"{ROOT}/results/{name}"
    os.makedirs(out_dir, exist_ok=True)
    feats = lambda rows: np.array([[math.log1p(e1.n_tokens(r["func"])), math.log1p(len(e1.line_starts(r["func"])))] for r in rows])
    train_rows = subset(e1.read_jsonl(f"{DATA}/primevul_train.jsonl"), args.benign_ratio, args.seed)
    valid_rows = subset(e1.read_jsonl(f"{DATA}/primevul_valid.jsonl"), args.benign_ratio, args.seed)
    clf = LogisticRegression().fit(feats(train_rows), [r["target"] for r in train_rows])
    yv = np.array([r["target"] for r in valid_rows])
    thr = thresholds_from_valid(yv, clf.predict_proba(feats(valid_rows))[:, 1])
    test = e1.read_jsonl(f"{DATA}/primevul_test.jsonl")
    pairs = e1.build_pairs(e1.read_jsonl(f"{DATA}/primevul_test_paired.jsonl"))
    y = np.array([r["target"] for r in test])
    pv = clf.predict_proba(feats([v for v, _ in pairs]))[:, 1]
    pf = clf.predict_proba(feats([f for _, f in pairs]))[:, 1]
    p = clf.predict_proba(feats(test))[:, 1]
    res = {"experiment": "exp3 pilot: length-only control (logistic regression on log tokens, log lines)",
           "coef": clf.coef_.tolist(), "valid_thresholds": thr, "pairs": e1.pair_metrics(pairs, pv, pf),
           "full_test": eval_with_thresholds(y, p, thr)}
    t = thr["f1_threshold"]; cv, cf = pv >= t, pf >= t
    res["pairs"]["at_valid_f1_threshold"] = {"P-C": int((cv & ~cf).sum()), "P-V": int((cv & cf).sum()), "P-B": int((~cv & ~cf).sum()), "P-R": int((~cv & cf).sum())}
    with open(f"{out_dir}/results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"[{name}] pairs: {json.dumps(res['pairs'])}")
    log(f"[{name}] full test: {json.dumps(res['full_test'])}")
    log(f"[{name}] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["head", "slide", "slice", "length"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benign_ratio", type=int, default=5)
    ap.add_argument("--max_windows", type=int, default=4)
    a = ap.parse_args()
    if a.strategy == "length":
        length_control(a)
    else:
        train(a.strategy, a)
