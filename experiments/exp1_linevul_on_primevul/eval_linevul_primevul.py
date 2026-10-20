"""Experiment 1: the original LineVul checkpoint (trained on Big-Vul) evaluated on PrimeVul.

Runs ON the Colab VM after setup_vm.py. Everything is inference only.

What it measures
  A. Truncation statistics (how much of PrimeVul falls outside LineVul's 512-token window).
  B. Function-level detection on the full imbalanced test set (F1 @0.5, AUPRC, ROC-AUC, VD-S).
  C. Pair-wise metrics on the paired test set (P-C / P-V / P-B / P-R, ranking accuracy,
     length-rule baseline, pairs identical after truncation).
  D. Line-level localization with LineVul's attention scoring, on vulnerable pair members whose
     flaw lines are derived from the diff against the patched version, stratified by whether the
     flaw lines are visible inside the 512-token window.

Nothing from the fixed version, CVE text or commit message is given to the model.
"""
import difflib
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

ROOT = "/content/exp1"
DATA = f"{ROOT}/primevul"
CKPT = f"{ROOT}/ckpt/12heads_linevul_model.bin"
OUT = f"{ROOT}/results"
os.makedirs(OUT, exist_ok=True)
MODEL_NAME = "microsoft/codebert-base"
BLOCK = 512
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


# ----------------------------------------------------------------------------- model
class LineVulHead(nn.Module):
    """Same head as linevul_model.py: dense -> tanh -> out_proj on the <s> token."""

    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.out_proj = nn.Linear(cfg.hidden_size, 2)

    def forward(self, h):
        x = h[:, 0, :]
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class LineVul(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.roberta = RobertaModel(cfg, add_pooling_layer=False)
        self.classifier = LineVulHead(cfg)

    def forward(self, input_ids, output_attentions=False):
        out = self.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
        logits = self.classifier(out.last_hidden_state)
        prob = torch.softmax(logits, dim=-1)
        return (prob, out.attentions) if output_attentions else prob


def load_model():
    cfg = RobertaConfig.from_pretrained(MODEL_NAME, output_attentions=False)
    cfg._attn_implementation = "eager"  # needed to get attention weights back from transformers 5.x
    model = LineVul(cfg)
    sd = torch.load(CKPT, map_location="cpu")
    remap = {}
    for k, v in sd.items():
        if k.startswith("encoder.roberta."):
            remap["roberta." + k[len("encoder.roberta."):]] = v
        elif k.startswith("classifier."):
            remap[k] = v
    missing, unexpected = model.load_state_dict(remap, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    unexpected = [u for u in unexpected if "position_ids" not in u and "pooler" not in u]
    log("checkpoint keys:", len(sd), "-> loaded", len(remap), "| missing:", missing, "| unexpected:", unexpected)
    assert not missing, "checkpoint did not load cleanly"
    return model.to(DEV).eval()


# ----------------------------------------------------------------------------- tokenization
tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(func):
    """Mirror LineVul: tokens[:510] wrapped in <s> </s>, padded to 512. Also return char offsets."""
    enc = tok(func, truncation=True, max_length=BLOCK, padding="max_length", return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def n_tokens(func):
    return len(tok(func, add_special_tokens=False)["input_ids"])


def line_starts(func):
    starts = [0]
    for i, ch in enumerate(func):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def visible_char_end(offsets):
    """Char index just past the last real (non-special, non-pad) token."""
    ends = [e for (s, e) in offsets if e > 0]
    return max(ends) if ends else 0


# ----------------------------------------------------------------------------- A. truncation statistics
def length_stats(name, rows):
    lens = np.array([n_tokens(r["func"]) for r in rows])
    vul = np.array([r["target"] == 1 for r in rows])
    res = {}
    for label, mask in (("all", np.ones_like(vul, bool)), ("vulnerable", vul), ("benign", ~vul)):
        if mask.sum() == 0:
            continue
        l = lens[mask]
        res[label] = {"n": int(mask.sum()), "median_tokens": float(np.median(l)),
                      "pct_gt_510": float((l > 510).mean() * 100), "pct_gt_1024": float((l > 1024).mean() * 100),
                      "pct_gt_2048": float((l > 2048).mean() * 100)}
    log(f"length stats {name}: {json.dumps(res)}")
    return res, lens


# ----------------------------------------------------------------------------- B. batched inference
@torch.no_grad()
def predict_probs(model, funcs, bs=64):
    probs = []
    for i in range(0, len(funcs), bs):
        ids = torch.tensor([encode(f)[0] for f in funcs[i:i + bs]], device=DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            p = model(ids)
        probs.append(p[:, 1].float().cpu().numpy())
        if (i // bs) % 50 == 0:
            log(f"  inference {i}/{len(funcs)}")
    return np.concatenate(probs)


def vd_score(y, p, target_fpr=0.005):
    """PrimeVul's VD-S: FNR at the largest threshold with FPR <= target (threshold picked on this set, as in
    the official calc_vd_score.py — i.e. an oracle threshold)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= target_fpr)[0]
    idx = ok[-1] if len(ok) else int(np.abs(fpr - target_fpr).argmin())
    t = thr[idx]
    pred = (p >= t).astype(int)
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return {"vd_s": fn / (fn + tp) if fn + tp else 0.0, "threshold": float(t), "actual_fpr": float(fpr[idx])}


def function_level(y, p):
    pred = (p > 0.5).astype(int)
    return {"n": int(len(y)), "n_vulnerable": int(y.sum()), "threshold": 0.5,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "flag_rate": float(pred.mean()),
            "auprc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "vd_s_oracle_threshold": vd_score(y, p)}


# ----------------------------------------------------------------------------- C. pairs
def build_pairs(rows):
    """Paired files list each pair as consecutive rows sharing commit_id (vulnerable then patched)."""
    pairs, i = [], 0
    while i + 1 < len(rows):
        a, b = rows[i], rows[i + 1]
        if a["commit_id"] == b["commit_id"] and {a["target"], b["target"]} == {0, 1}:
            v, f = (a, b) if a["target"] == 1 else (b, a)
            pairs.append((v, f))
            i += 2
        else:
            i += 1
    return pairs


def pair_metrics(pairs, pv, pf):
    """pv/pf: probability of 'vulnerable' for the vulnerable / fixed member."""
    cv, cf = pv > 0.5, pf > 0.5
    n = len(pairs)
    pc = int((cv & ~cf).sum()); pvv = int((cv & cf).sum()); pb = int((~cv & ~cf).sum()); pr = int((~cv & cf).sum())
    rank = float(((pv > pf) + 0.5 * (pv == pf)).mean())
    len_char = np.array([len(v["func"]) < len(f["func"]) for v, f in pairs])
    len_tok = np.array([n_tokens(v["func"]) < n_tokens(f["func"]) for v, f in pairs])
    same_trunc = np.array([encode(v["func"])[0] == encode(f["func"])[0] for v, f in pairs])
    return {"n_pairs": n,
            "P-C": pc, "P-V": pvv, "P-B": pb, "P-R": pr,
            "P-C_pct": pc / n * 100, "P-V_pct": pvv / n * 100, "P-B_pct": pb / n * 100, "P-R_pct": pr / n * 100,
            "net_PC_minus_PR_pct": (pc - pr) / n * 100,
            "ranking_accuracy_pct": rank * 100,
            "f1_on_pairs_balanced": float(f1_score(np.r_[np.ones(n), np.zeros(n)], np.r_[cv, cf].astype(int))),
            "length_rule_shorter_is_vulnerable_pct": {"chars": float(len_char.mean() * 100), "tokens": float(len_tok.mean() * 100)},
            "pairs_identical_after_truncation_pct": float(same_trunc.mean() * 100),
            "random_guess_expected_PC_pct": 25.0}


# ----------------------------------------------------------------------------- D. line labels + localization
COMMENT_PREFIXES = ("//", "/*", "*", "*/")


def flaw_lines_from_diff(vuln, fixed):
    """Lines of the vulnerable function that are deleted or replaced in the patch (Big-Vul convention),
    excluding blank and comment-only lines. Returns 0-based line indices into vuln.split('\\n')."""
    vl, fl = vuln.split("\n"), fixed.split("\n")
    vs, fs = [l.strip() for l in vl], [l.strip() for l in fl]
    sm = difflib.SequenceMatcher(None, vs, fs, autojunk=False)
    idx = []
    for op, i1, i2, _, _ in sm.get_opcodes():
        if op in ("delete", "replace"):
            idx.extend(range(i1, i2))
    return [i for i in idx if vs[i] and not vs[i].startswith(COMMENT_PREFIXES)]


@torch.no_grad()
def linevul_line_scores(model, func):
    """LineVul's 'attention' reasoning method, reproduced from linevul_main.py:
    attention of the FIRST layer, summed over heads, column-summed (attention each token receives),
    special tokens zeroed, then summed per source line (mapped through char offsets)."""
    ids, offsets = encode(func)
    ids_t = torch.tensor([ids], device=DEV)
    _, atts = model(ids_t, output_attentions=True)
    a = atts[0][0].float()            # layer 0, batch 0 -> [heads, seq, seq]
    tok_score = a.sum(0).sum(0)       # sum over heads, then over query rows -> attention received per token
    tok_score = tok_score.cpu().numpy()
    real = [i for i, (s, e) in enumerate(offsets) if e > 0]
    tok_score[0] = 0.0
    if real:
        tok_score[max(real)] = 0.0    # </s> is the token after the last real token; LineVul zeroes the last non-zero
    starts = line_starts(func)
    n_lines = len(starts)
    scores = np.zeros(n_lines)
    for i, (s, e) in enumerate(offsets):
        if e <= 0 or i == 0:
            continue
        li = int(np.searchsorted(starts, s, side="right") - 1)
        scores[li] += tok_score[i]
    vis_end = visible_char_end(offsets)
    visible = np.array([(starts[i] < vis_end) for i in range(n_lines)])
    return scores, visible


def expected_random_topk(n, f, k):
    """P(at least one of f flaw lines in a random top-k of n lines)."""
    if f == 0 or n == 0:
        return 0.0
    k = min(k, n)
    return 1.0 - math.comb(n - f, k) / math.comb(n, k) if n - f >= k else 1.0


def localization(model, pairs, pv):
    recs = []
    for (v, f), p in zip(pairs, pv):
        flaws = flaw_lines_from_diff(v["func"], f["func"])
        scores, visible = linevul_line_scores(model, v["func"])
        n_lines = len(scores)
        n_tok = n_tokens(v["func"])
        truncated = n_tok > 510
        fv = [i for i in flaws if visible[i]]
        rec = {"idx": v["idx"], "commit_id": v["commit_id"], "n_lines": n_lines, "n_tokens": n_tok,
               "truncated": truncated, "n_flaw": len(flaws), "n_flaw_visible": len(fv),
               "pred_prob": float(p), "true_positive": bool(p > 0.5)}
        if flaws:
            # invisible lines keep score 0 and therefore rank after every visible line; ties broken by position
            order = sorted(range(n_lines), key=lambda i: (-scores[i], i))
            rank_of = {li: r for r, li in enumerate(order)}
            first = min(rank_of[i] for i in flaws)
            rec["ifa"] = first                      # clean lines inspected before the first flaw line
            for k in (1, 5, 10):
                rec[f"top{k}"] = first < k
                rec[f"rand_top{k}"] = expected_random_topk(n_lines, len(flaws), k)
        recs.append(rec)
    return recs


def summarize_localization(recs):
    def agg(rs, label):
        rs = [r for r in rs if r["n_flaw"] > 0]
        if not rs:
            return {"n": 0}
        out = {"n": len(rs), "median_ifa": float(np.median([r["ifa"] for r in rs])), "mean_ifa": float(np.mean([r["ifa"] for r in rs]))}
        for k in (1, 5, 10):
            out[f"top{k}_acc_pct"] = float(np.mean([r[f"top{k}"] for r in rs]) * 100)
            out[f"random_top{k}_pct"] = float(np.mean([r[f"rand_top{k}"] for r in rs]) * 100)
        return out

    groups = {
        "ALL (vulnerable members with >=1 flaw line)": recs,
        "A: fits in 512 tokens": [r for r in recs if not r["truncated"]],
        "B: truncated, >=1 flaw line visible": [r for r in recs if r["truncated"] and r["n_flaw_visible"] > 0],
        "C: truncated, no flaw line visible": [r for r in recs if r["truncated"] and r["n_flaw_visible"] == 0],
        "TP only (pred>0.5), as in LineVul's RQ2": [r for r in recs if r["true_positive"]],
    }
    summary = {g: agg(rs, g) for g, rs in groups.items()}
    n_all = len(recs)
    flaws_total = sum(r["n_flaw"] for r in recs)
    flaws_vis = sum(r["n_flaw_visible"] for r in recs)
    summary["label_coverage"] = {
        "vulnerable_members": n_all,
        "with_at_least_one_flaw_line": sum(1 for r in recs if r["n_flaw"] > 0),
        "fix_adds_lines_only_no_flaw_line_pct": sum(1 for r in recs if r["n_flaw"] == 0) / n_all * 100,
        "flaw_lines_total": flaws_total,
        "flaw_lines_visible_in_512_pct": flaws_vis / flaws_total * 100 if flaws_total else None,
        "functions_truncated_pct": np.mean([r["truncated"] for r in recs]) * 100,
        "functions_truncated_and_all_flaw_lines_hidden_pct": sum(1 for r in recs if r["n_flaw"] > 0 and r["n_flaw_visible"] == 0) / n_all * 100,
        "mean_lines_per_function": float(np.mean([r["n_lines"] for r in recs])),
    }
    return summary


# ----------------------------------------------------------------------------- main
def main():
    results = {"checkpoint": "12heads_linevul_model.bin (LineVul, trained on Big-Vul)", "data": "PrimeVul original release",
               "device": str(DEV), "torch": torch.__version__}
    model = load_model()

    test = read_jsonl(f"{DATA}/primevul_test.jsonl")
    paired = read_jsonl(f"{DATA}/primevul_test_paired.jsonl")
    pairs = build_pairs(paired)
    log(f"test rows {len(test)} | paired rows {len(paired)} -> {len(pairs)} pairs")

    # A. truncation statistics (test + paired + vulnerable-only over all splits)
    results["length_stats"] = {}
    results["length_stats"]["test"], _ = length_stats("test", test)
    results["length_stats"]["test_paired"], _ = length_stats("test_paired", paired)
    vul_all = [r for split in ("train", "valid", "test") for r in read_jsonl(f"{DATA}/primevul_{split}.jsonl") if r["target"] == 1]
    results["length_stats"]["vulnerable_all_splits"], _ = length_stats("vulnerable_all_splits", vul_all)

    # C. pairs first (small), so partial results exist early
    pv = predict_probs(model, [v["func"] for v, _ in pairs])
    pf = predict_probs(model, [f["func"] for _, f in pairs])
    results["pairs"] = pair_metrics(pairs, pv, pf)
    log("pairs:", json.dumps(results["pairs"]))

    # D. localization on vulnerable pair members
    recs = localization(model, pairs, pv)
    results["localization"] = summarize_localization(recs)
    log("localization:", json.dumps(results["localization"], indent=1))
    with open(f"{OUT}/localization_records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    # B. full imbalanced test set
    y = np.array([r["target"] for r in test])
    p = predict_probs(model, [r["func"] for r in test])
    results["full_test"] = function_level(y, p)
    log("full test:", json.dumps(results["full_test"]))
    with open(f"{OUT}/full_test_predictions.jsonl", "w") as f:
        for r, pp in zip(test, p):
            f.write(json.dumps({"idx": r["idx"], "target": r["target"], "prob": float(pp)}) + "\n")
    with open(f"{OUT}/pair_predictions.jsonl", "w") as f:
        for (v, fx), a, b in zip(pairs, pv, pf):
            f.write(json.dumps({"commit_id": v["commit_id"], "idx_vuln": v["idx"], "idx_fixed": fx["idx"], "prob_vuln": float(a), "prob_fixed": float(b)}) + "\n")

    results["elapsed_s"] = time.time() - T0
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\n===== RESULTS JSON =====")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
