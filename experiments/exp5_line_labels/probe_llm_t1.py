"""Plan v3 Tier 1: Frozen LLM Representation Probing & 5-Gate Anti-Shortcut Audit.

Extracts 4-layer (last_token, mean_pool) hidden states + zero-shot log P(Yes)-log P(No)
from a frozen decoder LLM (e.g. Qwen/Qwen2.5-Coder-7B-Instruct) in a single memory-safe
forward pass (bf16, sdpa, use_cache=False, pooled in hooks), then trains and evaluates a
pre-registered suite of probes and anti-shortcut controls strictly on PrimeVul temporal
splits with zero commit/text leakage into test.

All pure-Python data/prompt/split helpers are importable without torch/numpy so unit
tests can run locally on CPU-only machines.
"""
import argparse
import difflib
import hashlib
import json
import math
import os
import random
import re
import time

COMMENT = ("//", "/*", "*", "*/")

PROBE_PREFIX = (
    "Determine whether the following C/C++ function contains a security vulnerability "
    "(e.g., buffer overflow, out-of-bounds read/write, integer overflow, "
    "null/dangling pointer dereference, use-after-free, double-free, "
    "missing bounds/validation check).\n\n```c\n"
)
PROBE_SUFFIX = "\n```\n\nQuestion: Is this C/C++ function vulnerable? Answer Yes or No.\nAnswer:"


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def text_hash(func):
    return hashlib.md5(re.sub(r"\s+", "", func or "").encode("utf-8", "replace")).hexdigest()


def diff_flaw_lines_pure(vul, fix):
    """Lines of vulnerable code deleted or replaced by fix (ignoring blank/comment lines)."""
    a, b = (vul or "").split("\n"), (fix or "").split("\n")
    out = []
    sm = difflib.SequenceMatcher(None, [x.strip() for x in a], [x.strip() for x in b], autojunk=False)
    for op, i1, i2, _, _ in sm.get_opcodes():
        if op in ("delete", "replace"):
            out.extend(i for i in range(i1, i2) if a[i].strip() and not a[i].strip().startswith(COMMENT))
    return out


def build_probe_input_ids(func, tok, max_len=2048):
    """Build tokenized prompt preserving PROBE_SUFFIX verbatim even when code is truncated.

    Returns (input_ids, was_cut).
    """
    prefix_ids = tok.encode(PROBE_PREFIX, add_special_tokens=True)
    suffix_ids = tok.encode(PROBE_SUFFIX, add_special_tokens=False)
    room = max_len - len(prefix_ids) - len(suffix_ids)
    if room < 16:
        raise ValueError(f"max_len={max_len} too small for prompt wrapper ({len(prefix_ids)+len(suffix_ids)} tok)")
    code_ids = tok.encode(func or "", add_special_tokens=False)
    cut = len(code_ids) > room
    if cut:
        code_ids = code_ids[:room]
    return prefix_ids + code_ids + suffix_ids, cut


def build_probe_prompt(func, max_chars=None):
    """Text helper preserving PROBE_SUFFIX when truncated by char limit (used for TF-IDF/tests)."""
    body = func or ""
    cut = False
    if max_chars is not None:
        room = max_chars - len(PROBE_PREFIX) - len(PROBE_SUFFIX)
        if room > 0 and len(body) > room:
            body = body[:room]
            cut = True
    return PROBE_PREFIX + body + PROBE_SUFFIX, cut


def load_paired_split(path, with_flaw_lines=True):
    """Load PrimeVul paired jsonl into aligned (vulnerable, fixed) pair records."""
    rows = load_jsonl(path)
    pairs = []
    skipped = 0
    for i in range(0, len(rows) - 1, 2):
        a, b = rows[i], rows[i + 1]
        if a.get("commit_id") != b.get("commit_id") or {a.get("target"), b.get("target")} != {0, 1}:
            skipped += 1
            continue
        v, f = (a, b) if a["target"] == 1 else (b, a)
        fl = diff_flaw_lines_pure(v["func"], f["func"]) if with_flaw_lines else []
        pairs.append({
            "v_idx": v["idx"],
            "f_idx": f["idx"],
            "v_func": v["func"],
            "f_func": f["func"],
            "cv": len(v["func"]),
            "cf": len(f["func"]),
            "v_lines": v["func"].count("\n") + 1,
            "f_lines": f["func"].count("\n") + 1,
            "commit_id": v["commit_id"],
            "project": v.get("project") or "",
            "cwe": v.get("cwe") or "",
            "flaw_lines": fl,
        })
    return pairs, skipped


def make_clean_splits(primevul_dir, manifest_path=None, extra_benign=2000, seed=42, test_scores_path=None):
    """Build strictly leak-free PrimeVul train/valid/test pair splits, extra train benigns, and fulltest sample.

    Drops any train/valid pair or extra benign that shares commit_id OR whitespace-normalized MD5 text_hash
    with ANY function in primevul_test.jsonl or primevul_test_paired.jsonl.
    """
    train_raw, train_skip = load_paired_split(os.path.join(primevul_dir, "primevul_train_paired.jsonl"), with_flaw_lines=True)
    valid_raw, valid_skip = load_paired_split(os.path.join(primevul_dir, "primevul_valid_paired.jsonl"), with_flaw_lines=True)
    test_pairs, test_skip = load_paired_split(os.path.join(primevul_dir, "primevul_test_paired.jsonl"), with_flaw_lines=True)

    if test_scores_path and os.path.exists(test_scores_path):
        ts_flaw = {r["idx"]: list(r.get("flaw_lines") or []) for r in load_jsonl(test_scores_path)}
        for p in test_pairs:
            if p["v_idx"] in ts_flaw:
                p["flaw_lines"] = ts_flaw[p["v_idx"]]

    test_full_path = os.path.join(primevul_dir, "primevul_test.jsonl")
    test_full_rows = load_jsonl(test_full_path) if os.path.exists(test_full_path) else []

    test_commits = {p["commit_id"] for p in test_pairs} | {r["commit_id"] for r in test_full_rows if r.get("commit_id")}
    test_hashes = (
        {text_hash(p["v_func"]) for p in test_pairs}
        | {text_hash(p["f_func"]) for p in test_pairs}
        | {text_hash(r["func"]) for r in test_full_rows}
    )

    def _filter_pairs(pairs):
        clean, drop_commit, drop_hash = [], 0, 0
        for p in pairs:
            if p["commit_id"] in test_commits:
                drop_commit += 1
                continue
            if text_hash(p["v_func"]) in test_hashes or text_hash(p["f_func"]) in test_hashes:
                drop_hash += 1
                continue
            clean.append(p)
        return clean, {"raw": len(pairs), "clean": len(clean), "dropped_commit": drop_commit, "dropped_hash": drop_hash}

    train_pairs, train_filter_stats = _filter_pairs(train_raw)
    valid_pairs, valid_filter_stats = _filter_pairs(valid_raw)

    # Exclude ALL raw paired indices (including any leaked/dropped pairs) and leaked commits from extra benign pool
    used_train_idx = {p["v_idx"] for p in train_raw + valid_raw + test_pairs} | {
        p["f_idx"] for p in train_raw + valid_raw + test_pairs
    }
    leaked_pair_commits = {
        p["commit_id"] for p in train_raw + valid_raw
        if p.get("commit_id") and (text_hash(p["v_func"]) in test_hashes or text_hash(p["f_func"]) in test_hashes)
    }
    banned_commits = test_commits | leaked_pair_commits

    benign_pool = []
    train_full_path = os.path.join(primevul_dir, "primevul_train.jsonl")
    if extra_benign > 0 and os.path.exists(train_full_path):
        for r in load_jsonl(train_full_path):
            if (
                r.get("target") == 0
                and r["idx"] not in used_train_idx
                and r.get("commit_id") not in banned_commits
                and text_hash(r.get("func", "")) not in test_hashes
            ):
                benign_pool.append({
                    "idx": r["idx"],
                    "target": 0,
                    "func": r["func"],
                    "chars": len(r["func"]),
                    "n_lines": r["func"].count("\n") + 1,
                    "commit_id": r.get("commit_id") or "",
                    "project": r.get("project") or "",
                })
    rng = random.Random(seed)
    sampled_train_benign = rng.sample(benign_pool, min(len(benign_pool), extra_benign)) if benign_pool else []

    train_projects = {p["project"] for p in train_pairs if p.get("project")}
    train_projects |= {r["project"] for r in sampled_train_benign if r.get("project")}
    for p in test_pairs:
        p["seen_project"] = bool(p.get("project") and p["project"] in train_projects)

    fulltest_rows = {}
    manifest = None
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as mfh:
            manifest = json.load(mfh)
        need_full = (
            {p["v_idx"] for p in test_pairs}
            | {p["f_idx"] for p in test_pairs}
            | set(manifest.get("rest_vulnerable_idx", []))
            | set(manifest.get("sampled_benign_idx", []))
        )
        for r in test_full_rows:
            if r["idx"] in need_full:
                fulltest_rows[r["idx"]] = {
                    "idx": r["idx"],
                    "target": r["target"],
                    "func": r["func"],
                    "chars": len(r["func"]),
                    "n_lines": r["func"].count("\n") + 1,
                    "project": r.get("project") or "",
                }
        for p in test_pairs:
            if p["v_idx"] not in fulltest_rows:
                fulltest_rows[p["v_idx"]] = {
                    "idx": p["v_idx"], "target": 1, "func": p["v_func"],
                    "chars": p["cv"], "n_lines": p["v_lines"], "project": p["project"],
                }
            if p["f_idx"] not in fulltest_rows:
                fulltest_rows[p["f_idx"]] = {
                    "idx": p["f_idx"], "target": 0, "func": p["f_func"],
                    "chars": p["cf"], "n_lines": p["f_lines"], "project": p["project"],
                }

    return {
        "train_pairs": train_pairs,
        "valid_pairs": valid_pairs,
        "test_pairs": test_pairs,
        "extra_train_benign": sampled_train_benign,
        "fulltest_rows": fulltest_rows,
        "manifest": manifest,
        "stats": {
            "skipped_misaligned": {"train": train_skip, "valid": valid_skip, "test": test_skip},
            "train_filter": train_filter_stats,
            "valid_filter": valid_filter_stats,
            "extra_benign_pool": len(benign_pool),
            "extra_benign_sampled": len(sampled_train_benign),
            "test_pairs": len(test_pairs),
            "test_seen_projects": sum(1 for p in test_pairs if p["seen_project"]),
            "test_unseen_projects": sum(1 for p in test_pairs if not p["seen_project"]),
        },
    }


def _debias_cubic(chars, scores):
    import numpy as np
    x = np.log(np.maximum(np.asarray(chars, dtype=np.float64), 1.0))
    s = np.asarray(scores, dtype=np.float64)
    if np.std(x) < 1e-12:
        return s - np.mean(s)
    coef = np.polyfit(x, s, 3)
    return s - np.polyval(coef, x)


def _rank_acc_and_debiased(sv, sf, cv, cf, weights=None):
    import numpy as np
    sv, sf, cv, cf = map(lambda a: np.asarray(a, dtype=np.float64), (sv, sf, cv, cf))
    rank = np.where(sv > sf, 1.0, np.where(sv == sf, 0.5, 0.0))
    allc = np.concatenate([cv, cf])
    alls = np.concatenate([sv, sf])
    d = _debias_cubic(allc, alls)
    n = len(sv)
    dv, df = d[:n], d[n:]
    rank_tb = np.where(dv > df, 1.0, np.where(dv == df, 0.5, 0.0))
    rank_deb = np.where(sv == sf, 0.5, rank_tb)
    if weights is None:
        return float(rank.mean() * 100.0), float(rank_deb.mean() * 100.0), float((sv == sf).mean() * 100.0)
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.maximum(w.sum(), 1e-12)
    return float((w * rank).sum() * 100.0), float((w * rank_deb).sum() * 100.0), float((w * (sv == sf)).sum() * 100.0)


def compute_pair_eval(sv, sf, cv, cf, masks, seed=0, B=2000):
    """Compute comprehensive paired + anti-shortcut metrics across all subsets."""
    import numpy as np
    from scipy.stats import spearmanr

    sv, sf, cv, cf = map(lambda a: np.asarray(a, dtype=np.float64), (sv, sf, cv, cf))
    rng = np.random.default_rng(seed)

    def _subset_metrics(idx_mask, balanced_50_50=False):
        m = np.asarray(idx_mask, dtype=bool)
        n = int(m.sum())
        if n == 0:
            return {"n": 0}
        sv_m, sf_m, cv_m, cf_m = sv[m], sf[m], cv[m], cf[m]
        weights = None
        if balanced_50_50:
            add_m = cf_m > cv_m
            rem_m = cv_m > cf_m
            if add_m.any() and rem_m.any():
                weights = np.where(add_m, 0.5 / add_m.sum(), np.where(rem_m, 0.5 / rem_m.sum(), 0.0))
        ra, rdeb, ties = _rank_acc_and_debiased(sv_m, sf_m, cv_m, cf_m, weights=weights)
        len_rule = float(np.where(cf_m > cv_m, 1.0, np.where(cf_m == cv_m, 0.5, 0.0)).mean() * 100.0)

        rank_arr = np.where(sv_m > sf_m, 1.0, np.where(sv_m == sf_m, 0.5, 0.0))
        alls = np.concatenate([sv_m, sf_m])
        allc = np.concatenate([cv_m, cf_m])
        sp_corr = float(spearmanr(alls, np.log(np.maximum(allc, 1.0))).statistic)
        if math.isnan(sp_corr):
            sp_corr = 0.0

        def _thr_stats(t):
            fv, ff = sv_m > t, sf_m > t
            pc = (fv & ~ff).astype(float)
            pv = (fv & ff).astype(float)
            pb = (~fv & ~ff).astype(float)
            pr = (~fv & ff).astype(float)
            net_arr = pc - pr
            if weights is not None:
                pc_v, pv_v, pb_v, pr_v = (weights * pc).sum() * 100, (weights * pv).sum() * 100, (weights * pb).sum() * 100, (weights * pr).sum() * 100
                net_v = pc_v - pr_v
                flag_r = float((np.concatenate([weights, weights]) * np.concatenate([fv, ff])).sum() * 50.0)
                return {
                    "threshold": float(t), "flag_rate": round(flag_r, 2),
                    "P-C": round(float(pc_v), 2), "P-V": round(float(pv_v), 2),
                    "P-B": round(float(pb_v), 2), "P-R": round(float(pr_v), 2),
                    "net": round(float(net_v), 2),
                }
            bs_net = []
            bs_ra = []
            for _ in range(B):
                bi = rng.integers(0, n, n)
                bs_net.append(float(net_arr[bi].mean() * 100.0))
                bs_ra.append(float(rank_arr[bi].mean() * 100.0))
            return {
                "threshold": float(t),
                "flag_rate": round(float(np.concatenate([fv, ff]).mean() * 100.0), 2),
                "P-C": round(float(pc.mean() * 100.0), 2),
                "P-V": round(float(pv.mean() * 100.0), 2),
                "P-B": round(float(pb.mean() * 100.0), 2),
                "P-R": round(float(pr.mean() * 100.0), 2),
                "net": round(float(net_arr.mean() * 100.0), 2),
                "net_ci": [round(float(np.percentile(bs_net, 2.5)), 2), round(float(np.percentile(bs_net, 97.5)), 2)],
                "rank_ci": [round(float(np.percentile(bs_ra, 2.5)), 2), round(float(np.percentile(bs_ra, 97.5)), 2)],
            }

        t0_st = _thr_stats(0.0)
        tm_st = _thr_stats(float(np.median(alls)))
        return {
            "n": n,
            "rank_acc": round(ra, 2),
            "rank_acc_ci": t0_st.get("rank_ci"),
            "rank_acc_debiased": round(rdeb, 2),
            "ties_pct": round(ties, 2),
            "length_rule_longer_is_fixed": round(len_rule, 2),
            "spearman_log_chars": round(sp_corr, 4),
            "t0": {k: v for k, v in t0_st.items() if k != "rank_ci"},
            "t_median": {k: v for k, v in tm_st.items() if k != "rank_ci"},
        }

    out = {}
    for name, mask in masks.items():
        if name == "length_matched_balanced":
            out[name] = _subset_metrics(mask, balanced_50_50=True)
        else:
            out[name] = _subset_metrics(mask, balanced_50_50=False)
    return out


def compute_fulltest_eval(scores_by_idx, fulltest_rows, manifest):
    """Compute weighted ROC-AUC, length-debiased ROC-AUC, AUPRC, and VD-S (FNR at <=0.5% FPR)."""
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    sampled_b = set(manifest.get("sampled_benign_idx", []))
    wB = float(manifest["strata"]["B"]["weight"])
    ids = [i for i in sorted(fulltest_rows) if i in scores_by_idx]
    y = np.array([fulltest_rows[i]["target"] for i in ids], dtype=np.int64)
    s = np.array([scores_by_idx[i] for i in ids], dtype=np.float64)
    chars = np.array([fulltest_rows[i]["chars"] for i in ids], dtype=np.float64)
    w = np.array([wB if i in sampled_b else 1.0 for i in ids], dtype=np.float64)

    roc = float(roc_auc_score(y, s, sample_weight=w))
    auprc = float(average_precision_score(y, s, sample_weight=w))

    logc = np.log(np.maximum(chars, 1.0))
    V = np.vander(logc, 4)
    coef, *_ = np.linalg.lstsq(V * np.sqrt(w)[:, None], s * np.sqrt(w), rcond=None)
    s_deb = s - V @ coef
    roc_deb = float(roc_auc_score(y, s_deb, sample_weight=w))

    order = np.argsort(-s, kind="mergesort")
    s_ord, y_ord, w_ord = s[order], y[order], w[order]
    tp_cum = np.cumsum(w_ord * (y_ord == 1))
    fp_cum = np.cumsum(w_ord * (y_ord == 0))
    tot_pos = max(float((w * (y == 1)).sum()), 1e-12)
    tot_neg = max(float((w * (y == 0)).sum()), 1e-12)
    end_of_group = np.r_[np.diff(s_ord) != 0, True]
    tpr = tp_cum[end_of_group] / tot_pos
    fpr = fp_cum[end_of_group] / tot_neg
    valid = fpr <= 0.005
    if valid.any():
        best_recall = float(tpr[valid].max())
        idx_best = int(np.flatnonzero(valid)[np.argmax(tpr[valid])])
        actual_fpr = float(fpr[idx_best])
    else:
        best_recall = 0.0
        actual_fpr = 0.0
    vds_fnr_pct = (1.0 - best_recall) * 100.0

    sp_full = float(spearmanr(s, logc).statistic)
    if math.isnan(sp_full):
        sp_full = 0.0
    prevalence = tot_pos / (tot_pos + tot_neg)

    return {
        "n_scored": len(ids),
        "weighted_prevalence": round(prevalence, 5),
        "roc_auc": round(roc, 4),
        "debiased_roc_auc": round(roc_deb, 4),
        "auprc": round(auprc, 4),
        "vds_fnr_at_05pct_fpr": round(vds_fnr_pct, 2),
        "recall_at_05pct_fpr": round(best_recall * 100.0, 2),
        "actual_fpr_pct": round(actual_fpr * 100.0, 3),
        "spearman_log_chars": round(sp_full, 4),
    }


def _layer_indices(num_layers):
    idx = sorted({
        max(0, min(num_layers - 1, round(num_layers * 0.50) - 1)),
        max(0, min(num_layers - 1, round(num_layers * 0.72) - 1)),
        max(0, min(num_layers - 1, round(num_layers * 0.86) - 1)),
        num_layers - 1,
    })
    return idx


def extract_features(items, model_id, max_len, cache_npz=None):
    """Single-pass frozen LLM extraction on GPU with exact right-padding masking & hook pooling.

    items: dict[idx] -> {"func": str, ...}
    Returns dict with keys: idx, last, mean, zero_shot, tok_len, cut, chars, n_lines, layers, telemetry.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cache_npz and os.path.exists(cache_npz):
        print(f"[extract] Loading cached features from {cache_npz}", flush=True)
        z = np.load(cache_npz, allow_pickle=False)
        return {
            "idx": z["idx"],
            "last": z["last"],
            "mean": z["mean"],
            "zero_shot": z["zero_shot"],
            "tok_len": z["tok_len"],
            "cut": z["cut"],
            "chars": z["chars"],
            "n_lines": z["n_lines"],
            "layers": [int(x) for x in z["layers"]],
            "telemetry": json.loads(str(z["telemetry"])),
        }

    t0 = time.time()
    print(f"[extract] Loading tokenizer & model: {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    yes_variants = [tok.encode(s, add_special_tokens=False) for s in (" Yes", "Yes")]
    no_variants = [tok.encode(s, add_special_tokens=False) for s in (" No", "No")]
    yes_ids = sorted({ids[0] for ids in yes_variants if len(ids) == 1})
    no_ids = sorted({ids[0] for ids in no_variants if len(ids) == 1})
    if not yes_ids or not no_ids:
        raise RuntimeError(f"Expected single-token Yes/No IDs, got yes={yes_variants}, no={no_variants}")
    print(
        f"[extract] Yes token IDs={[(i, tok.decode([i])) for i in yes_ids]} | "
        f"No token IDs={[(i, tok.decode([i])) for i in no_ids]}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    num_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    layers = _layer_indices(num_layers)
    print(f"[extract] Model loaded: num_layers={num_layers}, d={hidden_size}, hooks={layers}", flush=True)

    # Tokenize all unique functions
    all_ids_list = []
    idx_order = sorted(items.keys())
    for idx in idx_order:
        r = items[idx]
        inp_ids, was_cut = build_probe_input_ids(r["func"], tok, max_len=max_len)
        all_ids_list.append({
            "idx": idx,
            "input_ids": inp_ids,
            "tok_len": len(inp_ids),
            "cut": int(was_cut),
            "chars": len(r["func"]),
            "n_lines": r["func"].count("\n") + 1,
        })

    # Sort ascending by tok_len for minimal padding waste
    sorted_items = sorted(all_ids_list, key=lambda x: x["tok_len"])
    n_total = len(sorted_items)
    n_layers = len(layers)

    out_last = np.empty((n_total, n_layers, hidden_size), dtype=np.float16)
    out_mean = np.empty((n_total, n_layers, hidden_size), dtype=np.float16)
    out_zs = np.empty((n_total,), dtype=np.float32)

    # Sanity check equivalence of model.model + lm_head on the shortest sequence
    with torch.inference_mode():
        sample_ids = torch.tensor([sorted_items[0]["input_ids"]], device="cuda", dtype=torch.long)
        sample_mask = torch.ones_like(sample_ids)
        full_out = model(input_ids=sample_ids, attention_mask=sample_mask, use_cache=False)
        base_out = model.model(input_ids=sample_ids, attention_mask=sample_mask, use_cache=False)
        head_logits = model.lm_head(base_out.last_hidden_state[:, -1:, :])[:, 0, :]
        max_diff = float((full_out.logits[:, -1, :] - head_logits).abs().max().item())
        top5 = torch.topk(head_logits[0], 5)
        top5_desc = [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top5.values, top5.indices)]
        del full_out, base_out, head_logits
    print(f"[extract] Head equivalence max_diff={max_diff:.2e} | sample0 top5={top5_desc}", flush=True)

    # Register hooks on model.model.layers[l] capturing ONLY pooled [B, d] vectors
    captured = {}
    batch_state = {"mask": None, "last_idx": None}

    def _make_hook(li):
        def _hook(_mod, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            m = batch_state["mask"]
            li_idx = batch_state["last_idx"]
            b_ar = torch.arange(h.shape[0], device=h.device)
            last_vec = h[b_ar, li_idx].float()
            mf = m.unsqueeze(-1).float()
            mean_vec = (h.float() * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
            captured[li] = (last_vec.cpu().numpy().astype(np.float16), mean_vec.cpu().numpy().astype(np.float16))
        return _hook

    handles = [model.model.layers[l].register_forward_hook(_make_hook(l)) for l in layers]
    torch.cuda.reset_peak_memory_stats()

    yes_t = torch.tensor(yes_ids, device="cuda", dtype=torch.long)
    no_t = torch.tensor(no_ids, device="cuda", dtype=torch.long)

    done = 0
    total_toks = 0
    t_loop = time.time()
    try:
        while done < n_total:
            cur_len = sorted_items[done]["tok_len"]
            bs = 4 if cur_len <= 512 else (2 if cur_len <= 1024 else 1)
            end = min(done + bs, n_total)
            batch = sorted_items[done:end]

            def _run_batch(b_items):
                max_b = max(x["tok_len"] for x in b_items)
                inp = torch.full((len(b_items), max_b), pad_id, dtype=torch.long, device="cuda")
                mask = torch.zeros((len(b_items), max_b), dtype=torch.long, device="cuda")
                last_idx = torch.zeros((len(b_items),), dtype=torch.long, device="cuda")
                for bi, x in enumerate(b_items):
                    tl = x["tok_len"]
                    inp[bi, :tl] = torch.tensor(x["input_ids"], dtype=torch.long, device="cuda")
                    mask[bi, :tl] = 1
                    last_idx[bi] = tl - 1
                batch_state["mask"] = mask
                batch_state["last_idx"] = last_idx
                captured.clear()
                with torch.inference_mode():
                    base_out = model.model(input_ids=inp, attention_mask=mask, use_cache=False)
                    b_ar = torch.arange(len(b_items), device="cuda")
                    last_normed = base_out.last_hidden_state[b_ar, last_idx].unsqueeze(1)
                    logits = model.lm_head(last_normed)[:, 0, :].float()
                    lp = torch.log_softmax(logits, dim=-1)
                    lp_yes = torch.logsumexp(lp[:, yes_t], dim=-1)
                    lp_no = torch.logsumexp(lp[:, no_t], dim=-1)
                    zs_vals = (lp_yes - lp_no).cpu().numpy().astype(np.float32)
                return zs_vals

            try:
                zs_vals = _run_batch(batch)
                for j, li in enumerate(layers):
                    lv, mv = captured[li]
                    out_last[done:end, j, :] = lv
                    out_mean[done:end, j, :] = mv
                out_zs[done:end] = zs_vals
                total_toks += sum(x["tok_len"] for x in batch)
                done = end
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise
                for bi in range(len(batch)):
                    zs1 = _run_batch([batch[bi]])
                    for j, li in enumerate(layers):
                        lv, mv = captured[li]
                        out_last[done + bi:done + bi + 1, j, :] = lv
                        out_mean[done + bi:done + bi + 1, j, :] = mv
                    out_zs[done + bi:done + bi + 1] = zs1
                    total_toks += batch[bi]["tok_len"]
                done = end

            if done % 500 < bs or done == n_total:
                el = max(time.time() - t_loop, 1e-6)
                eta = (n_total - done) * el / max(done, 1)
                print(
                    f"[extract] {done}/{n_total} | {done/el:.1f} fn/s | {total_toks/el:.0f} tok/s | "
                    f"peakVRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB | ETA {eta/60:.1f} min",
                    flush=True,
                )
    finally:
        for h in handles:
            h.remove()

    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    extract_secs = round(time.time() - t0, 1)
    del model
    torch.cuda.empty_cache()

    telemetry = {
        "model": model_id,
        "max_len": max_len,
        "n_unique": n_total,
        "n_cut": sum(x["cut"] for x in sorted_items),
        "cut_pct": round(100.0 * sum(x["cut"] for x in sorted_items) / max(n_total, 1), 2),
        "layers": layers,
        "peak_vram_gb": peak_vram_gb,
        "extract_secs": extract_secs,
        "head_equiv_max_diff": round(max_diff, 6),
    }

    res = {
        "idx": np.array([x["idx"] for x in sorted_items], dtype=np.int64),
        "last": out_last,
        "mean": out_mean,
        "zero_shot": out_zs,
        "tok_len": np.array([x["tok_len"] for x in sorted_items], dtype=np.int32),
        "cut": np.array([x["cut"] for x in sorted_items], dtype=np.int8),
        "chars": np.array([x["chars"] for x in sorted_items], dtype=np.int32),
        "n_lines": np.array([x["n_lines"] for x in sorted_items], dtype=np.int32),
        "layers": layers,
        "telemetry": telemetry,
    }
    if cache_npz:
        os.makedirs(os.path.dirname(cache_npz), exist_ok=True)
        np.savez(
            cache_npz,
            idx=res["idx"], last=res["last"], mean=res["mean"], zero_shot=res["zero_shot"],
            tok_len=res["tok_len"], cut=res["cut"], chars=res["chars"], n_lines=res["n_lines"],
            layers=np.array(layers, dtype=np.int32), telemetry=np.array(json.dumps(telemetry)),
        )
        print(f"[extract] Saved cached embeddings to {cache_npz}", flush=True)
    return res


def _fit_and_eval_probes(splits, feats, out_dir, manual40_path=None, seed=42):
    """Run all 9 models (baselines + frozen probes + debiased probes + MLP), select strictly on valid."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn

    idx_to_pos = {int(i): pos for pos, i in enumerate(feats["idx"])}
    train_pairs = splits["train_pairs"]
    valid_pairs = splits["valid_pairs"]
    test_pairs = splits["test_pairs"]
    extra_benign = splits["extra_train_benign"]
    fulltest_rows = splits["fulltest_rows"]
    manifest = splits["manifest"]

    # Build index arrays into feats
    tr_v = np.array([idx_to_pos[p["v_idx"]] for p in train_pairs], dtype=np.int64)
    tr_f = np.array([idx_to_pos[p["f_idx"]] for p in train_pairs], dtype=np.int64)
    va_v = np.array([idx_to_pos[p["v_idx"]] for p in valid_pairs], dtype=np.int64)
    va_f = np.array([idx_to_pos[p["f_idx"]] for p in valid_pairs], dtype=np.int64)
    te_v = np.array([idx_to_pos[p["v_idx"]] for p in test_pairs], dtype=np.int64)
    te_f = np.array([idx_to_pos[p["f_idx"]] for p in test_pairs], dtype=np.int64)
    tr_eb = np.array([idx_to_pos[r["idx"]] for r in extra_benign], dtype=np.int64)

    cv_tr = np.array([p["cv"] for p in train_pairs], dtype=np.float64)
    cf_tr = np.array([p["cf"] for p in train_pairs], dtype=np.float64)
    cv_va = np.array([p["cv"] for p in valid_pairs], dtype=np.float64)
    cf_va = np.array([p["cf"] for p in valid_pairs], dtype=np.float64)
    cv_te = np.array([p["cv"] for p in test_pairs], dtype=np.float64)
    cf_te = np.array([p["cf"] for p in test_pairs], dtype=np.float64)

    # Build valid masks for model selection
    def _balanced_pair_weights(cv_arr, cf_arr):
        add_m = cf_arr > cv_arr
        rem_m = cv_arr > cf_arr
        eq_m = ~(add_m | rem_m)
        w = np.ones(len(cv_arr), dtype=np.float64)
        if add_m.any() and rem_m.any():
            n_tot = float(add_m.sum() + rem_m.sum())
            w[add_m] = 0.5 * n_tot / add_m.sum()
            w[rem_m] = 0.5 * n_tot / rem_m.sum()
            w[eq_m] = 1.0
        return w

    va_weights = _balanced_pair_weights(cv_va, cf_va)
    tr_weights = _balanced_pair_weights(cv_tr, cf_tr)

    def _valid_balanced_score(sc_vec):
        sv, sf = sc_vec[va_v], sc_vec[va_f]
        ra, _, _ = _rank_acc_and_debiased(sv, sf, cv_va, cf_va, weights=va_weights)
        return ra

    def _valid_all_score(sc_vec):
        sv, sf = sc_vec[va_v], sc_vec[va_f]
        ra, _, _ = _rank_acc_and_debiased(sv, sf, cv_va, cf_va, weights=None)
        return ra

    # Build comprehensive test masks
    changed_m = np.array([bool(p["flaw_lines"]) for p in test_pairs])
    seen_m = np.array([bool(p["seen_project"]) for p in test_pairs])
    masks = {
        "all": np.ones(len(test_pairs), dtype=bool),
        "length_matched_5pct": np.abs(cv_te - cf_te) / np.maximum(cv_te, cf_te) < 0.05,
        "length_matched_balanced": np.ones(len(test_pairs), dtype=bool),
        "patch_removes_code": cv_te > cf_te,
        "patch_adds_code": cf_te > cv_te,
        "seen_projects": seen_m,
        "unseen_projects": ~seen_m,
        "changed": changed_m,
        "add_only": ~changed_m,
    }
    if manual40_path and os.path.exists(manual40_path):
        man40 = json.load(open(manual40_path, encoding="utf-8"))
        labels40 = man40.get("labels", [])
        # manual_labels_test40.json sampled pairs with random.Random(1).sample(..., 40) over file-ordered pairs
        samp = random.Random(1).sample(range(len(test_pairs)), min(len(test_pairs), len(labels40)))
        matches = sum(
            1 for x in labels40
            if x["s"] < len(samp) and (not x.get("project") or test_pairs[samp[x["s"]]]["project"] == x["project"])
        )
        if labels40 and matches == len(labels40):
            local_idx = {samp[x["s"]] for x in labels40 if x.get("a_to_b") is True and x.get("outside") is False}
            out_idx = {samp[x["s"]] for x in labels40 if x.get("a_to_b") is True and x.get("outside") is True}
            not_ab_idx = {samp[x["s"]] for x in labels40 if x.get("a_to_b") is False}
            masks["manual40_local"] = np.array([i in local_idx for i in range(len(test_pairs))])
            masks["manual40_outside"] = np.array([i in out_idx for i in range(len(test_pairs))])
            masks["manual40_not_a_to_b"] = np.array([i in not_ab_idx for i in range(len(test_pairs))])
            print(f"[eval] Matched all {matches}/{len(labels40)} manual_labels_test40 pairs by project", flush=True)
        else:
            print(f"[eval] Skipping manual40 breakdown (project match {matches}/{len(labels40)})", flush=True)

    # Length covariates for all unique items
    logc = np.log(np.maximum(feats["chars"], 1.0))
    logt = np.log(np.maximum(feats["tok_len"], 1.0))
    nlines = feats["n_lines"].astype(np.float64)
    len_X = np.column_stack([logc, logt, np.log(np.maximum(nlines, 1.0))])
    len_fit_idx = np.concatenate([tr_v, tr_f, tr_eb])

    def _fit_len_orth(X_raw):
        """Remove linear projection onto [log_chars, log_tokens, log_lines] fitted STRICTLY on train."""
        lx_sc = StandardScaler().fit(len_X[len_fit_idx])
        L_tr = lx_sc.transform(len_X[len_fit_idx])
        L_all = lx_sc.transform(len_X)
        reg = Ridge(alpha=1.0).fit(L_tr, X_raw[len_fit_idx])
        return X_raw - reg.predict(L_all)

    def _record_method(name, sc_all, info, all_results):
        sv_te, sf_te = sc_all[te_v], sc_all[te_f]
        pair_res = compute_pair_eval(sv_te, sf_te, cv_te, cf_te, masks, seed=seed, B=2000)
        full_res = {}
        if manifest and fulltest_rows:
            sc_map = {idx: float(sc_all[idx_to_pos[idx]]) for idx in fulltest_rows if idx in idx_to_pos}
            full_res = compute_fulltest_eval(sc_map, fulltest_rows, manifest)
        all_results[name] = {"config": info, "pairs": pair_res, "fulltest": full_res}

        # Write pair_protocol.py-compatible score files
        sc_dir = os.path.join(out_dir, "scores", name)
        os.makedirs(sc_dir, exist_ok=True)
        with open(os.path.join(sc_dir, "primevul_test_scores.jsonl"), "w", encoding="utf-8") as fh:
            for p, sv_1, sf_1 in zip(test_pairs, sv_te, sf_te):
                fh.write(json.dumps({"idx": p["v_idx"], "target": 1, "logit": float(sv_1)}) + "\n")
                fh.write(json.dumps({"idx": p["f_idx"], "target": 0, "logit": float(sf_1)}) + "\n")
        if fulltest_rows:
            with open(os.path.join(sc_dir, "fulltest_scores.jsonl"), "w", encoding="utf-8") as fh:
                for idx, row in fulltest_rows.items():
                    fh.write(json.dumps({"idx": idx, "target": row["target"], "logit": float(sc_all[idx_to_pos[idx]])}) + "\n")

    results = {}

    # 1. length_only baseline
    lsc = StandardScaler().fit(len_X[np.concatenate([tr_v, tr_f])])
    X_len_tr = lsc.transform(len_X[np.concatenate([tr_v, tr_f])])
    y_pair_tr = np.concatenate([np.ones(len(tr_v)), np.zeros(len(tr_f))])
    clf_len = LogisticRegression(C=1.0, max_iter=1000).fit(X_len_tr, y_pair_tr)
    sc_len = clf_len.decision_function(lsc.transform(len_X))
    _record_method("length_only", sc_len, {"features": ["log_chars", "log_tokens", "log_lines"]}, results)

    # 2. tfidf_word baseline on train_paired
    tr_texts = [p["v_func"] for p in train_pairs] + [p["f_func"] for p in train_pairs]
    vec = TfidfVectorizer(max_features=20000, token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\s]")
    X_tf_tr = vec.fit_transform(tr_texts)
    best_tf = (-1.0, None, None)
    va_texts = [p["v_func"] for p in valid_pairs] + [p["f_func"] for p in valid_pairs]
    X_tf_va = vec.transform(va_texts)
    for C in (0.1, 1.0, 5.0):
        clf = LogisticRegression(C=C, max_iter=1000).fit(X_tf_tr, y_pair_tr)
        dv = clf.decision_function(X_tf_va[:len(valid_pairs)])
        df = clf.decision_function(X_tf_va[len(valid_pairs):])
        va_ra, _, _ = _rank_acc_and_debiased(dv, df, cv_va, cf_va, weights=va_weights)
        if va_ra > best_tf[0]:
            best_tf = (va_ra, C, clf)
    clf_tf = best_tf[2]
    all_texts_by_pos = [""] * len(feats["idx"])
    all_items_flat = {}
    for p in train_pairs + valid_pairs + test_pairs:
        all_items_flat[p["v_idx"]] = p["v_func"]
        all_items_flat[p["f_idx"]] = p["f_func"]
    for r in extra_benign:
        all_items_flat[r["idx"]] = r["func"]
    for idx, r in fulltest_rows.items():
        all_items_flat[idx] = r["func"]
    for idx, pos in idx_to_pos.items():
        all_texts_by_pos[pos] = all_items_flat[idx]
    sc_tf = clf_tf.decision_function(vec.transform(all_texts_by_pos))
    _record_method("tfidf_word", sc_tf, {"C": best_tf[1], "valid_bal_rank": round(best_tf[0], 2)}, results)

    # 3. zero_shot_yesno
    sc_zs = feats["zero_shot"].astype(np.float64)
    _record_method("zero_shot_yesno", sc_zs, {"threshold_note": "log P(Yes) - log P(No), threshold 0.0"}, results)

    # Layer/Pool sweeps for frozen probes 4..8 + MLP 9
    layers = feats["layers"]
    sweep_methods = {
        "probe_std": {"best_val": -1e9},
        "probe_pair_margin": {"best_val": -1e9},
        "probe_len_balanced": {"best_val": -1e9},
        "probe_len_orth": {"best_val": -1e9},
        "probe_mixed_benign": {"best_val": -1e9},
    }
    layer_sweep_log = []
    C_GRID = (1e-3, 1e-2, 1e-1, 1.0)

    def _fit_pairwise_lr(X_sc, sample_w, C_val):
        X_diff = np.concatenate([X_sc[tr_v] - X_sc[tr_f], X_sc[tr_f] - X_sc[tr_v]], axis=0)
        y_diff = np.concatenate([np.ones(len(tr_v)), np.zeros(len(tr_v))])
        w_diff = np.concatenate([sample_w, sample_w])
        clf = LogisticRegression(C=C_val, fit_intercept=False, max_iter=1000).fit(X_diff, y_diff, sample_weight=w_diff)
        raw_sc = X_sc @ clf.coef_[0]
        med_va = float(np.median(np.concatenate([raw_sc[va_v], raw_sc[va_f]])))
        return raw_sc - med_va, {"C": C_val, "intercept_valid_med": round(-med_va, 4)}

    for pool_name in ("last", "mean"):
        for li_pos, layer_id in enumerate(layers):
            X_raw = feats[pool_name][:, li_pos, :].astype(np.float64)
            scaler_pair = StandardScaler().fit(X_raw[np.concatenate([tr_v, tr_f])])
            X_sc = scaler_pair.transform(X_raw)

            X_orth_raw = _fit_len_orth(X_raw)
            scaler_orth = StandardScaler().fit(X_orth_raw[len_fit_idx])
            X_orth_sc = scaler_orth.transform(X_orth_raw)

            X_mix_fit = StandardScaler().fit(X_raw[len_fit_idx])
            X_mix_sc = X_mix_fit.transform(X_raw)

            # A. probe_std (pointwise on train_paired, select on valid_all_rank_acc)
            for C in C_GRID:
                clf = LogisticRegression(C=C, max_iter=1000).fit(X_sc[np.concatenate([tr_v, tr_f])], y_pair_tr)
                sc = clf.decision_function(X_sc)
                v_all = _valid_all_score(sc)
                v_bal = _valid_balanced_score(sc)
                if v_all > sweep_methods["probe_std"]["best_val"]:
                    sweep_methods["probe_std"] = {"best_val": v_all, "sc": sc, "cfg": {"pool": pool_name, "layer": layer_id, "C": C, "valid_all_rank": round(v_all, 2), "valid_bal_rank": round(v_bal, 2)}}

            # B. probe_pair_margin (unweighted pairwise diff, select on valid_bal)
            for C in C_GRID:
                sc, _ = _fit_pairwise_lr(X_sc, np.ones(len(tr_v)), C)
                v_bal = _valid_balanced_score(sc)
                if v_bal > sweep_methods["probe_pair_margin"]["best_val"]:
                    sweep_methods["probe_pair_margin"] = {"best_val": v_bal, "sc": sc, "cfg": {"pool": pool_name, "layer": layer_id, "C": C, "valid_bal_rank": round(v_bal, 2)}}

            # C. probe_len_balanced (50/50 cf>cv vs cv>cf pairwise diff, select on valid_bal)
            for C in C_GRID:
                sc, _ = _fit_pairwise_lr(X_sc, tr_weights, C)
                v_bal = _valid_balanced_score(sc)
                if v_bal > sweep_methods["probe_len_balanced"]["best_val"]:
                    sweep_methods["probe_len_balanced"] = {"best_val": v_bal, "sc": sc, "cfg": {"pool": pool_name, "layer": layer_id, "C": C, "valid_bal_rank": round(v_bal, 2)}}

            # D. probe_len_orth (Ridge length orthogonalization + balanced pairwise diff)
            best_orth_view = (-1e9, None)
            for C in C_GRID:
                sc, _ = _fit_pairwise_lr(X_orth_sc, tr_weights, C)
                v_bal = _valid_balanced_score(sc)
                if v_bal > best_orth_view[0]:
                    best_orth_view = (v_bal, C)
                if v_bal > sweep_methods["probe_len_orth"]["best_val"]:
                    sweep_methods["probe_len_orth"] = {"best_val": v_bal, "sc": sc, "cfg": {"pool": pool_name, "layer": layer_id, "C": C, "valid_bal_rank": round(v_bal, 2)}}

            # E. probe_mixed_benign (train_paired balanced + 2000 unpaired benigns, pointwise LR, select on valid_bal)
            if len(tr_eb) > 0:
                X_mix_tr = np.concatenate([X_mix_sc[tr_v], X_mix_sc[tr_f], X_mix_sc[tr_eb]], axis=0)
                y_mix_tr = np.concatenate([np.ones(len(tr_v)), np.zeros(len(tr_f)), np.zeros(len(tr_eb))])
                eb_w = float(len(tr_v)) / max(float(len(tr_eb)), 1.0)
                w_mix_tr = np.concatenate([tr_weights, tr_weights, np.full(len(tr_eb), eb_w)])
            else:
                X_mix_tr = X_sc[np.concatenate([tr_v, tr_f])]
                y_mix_tr = y_pair_tr
                w_mix_tr = np.concatenate([tr_weights, tr_weights])
            for C in C_GRID:
                clf = LogisticRegression(C=C, max_iter=1000).fit(X_mix_tr, y_mix_tr, sample_weight=w_mix_tr)
                sc = clf.decision_function(X_mix_sc)
                v_bal = _valid_balanced_score(sc)
                if v_bal > sweep_methods["probe_mixed_benign"]["best_val"]:
                    sweep_methods["probe_mixed_benign"] = {"best_val": v_bal, "sc": sc, "cfg": {"pool": pool_name, "layer": layer_id, "C": C, "extra_benign": len(tr_eb), "valid_bal_rank": round(v_bal, 2)}}

            # Record per-(layer, pool) test diagnostics on best_orth C cleanly for layer sweep curve
            best_c = best_orth_view[1]
            sc_l, _ = _fit_pairwise_lr(X_orth_sc, tr_weights, best_c)
            sp_te = float(np.corrcoef(sc_l[te_v], np.log(np.maximum(cv_te, 1.0)))[0, 1]) if len(te_v) > 1 else 0.0
            ra_all, ra_deb, _ = _rank_acc_and_debiased(sc_l[te_v], sc_l[te_f], cv_te, cf_te)
            rem_m = masks["patch_removes_code"]
            ra_rem = _rank_acc_and_debiased(sc_l[te_v][rem_m], sc_l[te_f][rem_m], cv_te[rem_m], cf_te[rem_m])[0] if rem_m.any() else 0.0
            layer_sweep_log.append({
                "pool": pool_name, "layer": layer_id, "best_C": best_c,
                "valid_bal_rank": round(best_orth_view[0], 2),
                "test_all_rank": round(ra_all, 2),
                "test_deb_rank": round(ra_deb, 2),
                "test_rem_rank": round(ra_rem, 2),
                "test_r_logc": round(sp_te, 4),
            })

    for mname, md in sweep_methods.items():
        _record_method(mname, md["sc"], md["cfg"], results)

    # 9. mlp_len_orth on the best (pool, layer) chosen for probe_len_orth (3 seeds averaged)
    best_cfg = sweep_methods["probe_len_orth"]["cfg"]
    best_pool = best_cfg["pool"]
    best_li_pos = layers.index(best_cfg["layer"])
    X_best_raw = feats[best_pool][:, best_li_pos, :].astype(np.float64)
    X_orth_raw = _fit_len_orth(X_best_raw)
    sc_orth = StandardScaler().fit(X_orth_raw[len_fit_idx])
    X_mlp = sc_orth.transform(X_orth_raw).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt_v = torch.tensor(X_mlp[tr_v], device=device)
    Xt_f = torch.tensor(X_mlp[tr_f], device=device)
    Wt_tr = torch.tensor(tr_weights, dtype=torch.float32, device=device)
    Xt_eb = torch.tensor(X_mlp[tr_eb], device=device) if len(tr_eb) > 0 else None
    X_all_t = torch.tensor(X_mlp, device=device)

    seed_preds = []
    seed_val_scores = []
    for s_off in range(3):
        torch.manual_seed(seed + s_off)
        mlp = nn.Sequential(
            nn.Linear(X_mlp.shape[1], 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 1),
        ).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-2)
        best_state, best_va = None, -1e9
        for ep in range(40):
            mlp.train()
            perm = torch.randperm(len(tr_v), device=device)
            for start in range(0, len(tr_v), 256):
                bi = perm[start:start + 256]
                sv_b = mlp(Xt_v[bi]).squeeze(-1)
                sf_b = mlp(Xt_f[bi]).squeeze(-1)
                w_b = Wt_tr[bi]
                loss_pair = (w_b * torch.nn.functional.softplus(-(sv_b - sf_b))).mean()
                loss_bce = 0.25 * (
                    torch.nn.functional.binary_cross_entropy_with_logits(sv_b, torch.ones_like(sv_b))
                    + torch.nn.functional.binary_cross_entropy_with_logits(sf_b, torch.zeros_like(sf_b))
                )
                loss = loss_pair + loss_bce
                if Xt_eb is not None:
                    eb_i = torch.randint(0, len(Xt_eb), (len(bi),), device=device)
                    se_b = mlp(Xt_eb[eb_i]).squeeze(-1)
                    loss = loss + 0.25 * torch.nn.functional.binary_cross_entropy_with_logits(se_b, torch.zeros_like(se_b))
                opt.zero_grad()
                loss.backward()
                opt.step()
            mlp.eval()
            with torch.no_grad():
                sc_ep = mlp(X_all_t).squeeze(-1).cpu().numpy().astype(np.float64)
            va_bal = _valid_balanced_score(sc_ep)
            if va_bal > best_va:
                best_va = va_bal
                best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
        mlp.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        mlp.eval()
        with torch.no_grad():
            sc_s = mlp(X_all_t).squeeze(-1).cpu().numpy().astype(np.float64)
        seed_preds.append(sc_s)
        seed_val_scores.append(round(best_va, 2))

    sc_mlp_mean = np.mean(seed_preds, axis=0)
    med_mlp_va = float(np.median(np.concatenate([sc_mlp_mean[va_v], sc_mlp_mean[va_f]])))
    _record_method(
        "mlp_len_orth",
        sc_mlp_mean - med_mlp_va,
        {"pool": best_pool, "layer": best_cfg["layer"], "seeds": 3, "valid_bal_ranks": seed_val_scores, "valid_ens_bal_rank": round(_valid_balanced_score(sc_mlp_mean - med_mlp_va), 2)},
        results,
    )

    return results, layer_sweep_log


def format_report_md(out_json):
    """Build concise, verification-ready Markdown summary of T1 results."""
    tel = out_json["telemetry"]
    ss = out_json["split_stats"]
    lines = [
        "# Tier 1 Frozen LLM Probe & Anti-Shortcut Report",
        "",
        f"- **Model:** `{tel['model']}` (`max_len={tel['max_len']}`, layers `{tel['layers']}`)",
        f"- **Unique functions extracted:** `{tel['n_unique']}` (truncated: `{tel['n_cut']}` = `{tel['cut_pct']}%`), Peak VRAM `{tel['peak_vram_gb']} GB`, Time `{tel['extract_secs']}s`, Head equiv max diff `{tel['head_equiv_max_diff']}`",
        f"- **Clean splits:** train `{ss['train_filter']['clean']}` pairs (dropped commit `{ss['train_filter']['dropped_commit']}`, hash `{ss['train_filter']['dropped_hash']}`), valid `{ss['valid_filter']['clean']}` pairs, test `{ss['test_pairs']}` pairs (`seen_projects={ss['test_seen_projects']}`, `unseen_projects={ss['test_unseen_projects']}`), extra train benigns `{ss['extra_benign_sampled']}`",
        "",
        "## 1. 5-Gate Anti-Shortcut Pair Matrix (`primevul_test_paired`, n=564)",
        "",
        "| Method | Best Config | All Net@t0 / Net@med | All Rank / Debias / Ties | Bal Rank | LenMatched5% Rank | **PatchRemoves (n=65) Rank** | Seen / Unseen Rank | Changed / AddOnly Rank | Spearman(log chars) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "length_only", "tfidf_word", "zero_shot_yesno",
        "probe_std", "probe_pair_margin", "probe_len_balanced",
        "probe_len_orth", "probe_mixed_benign", "mlp_len_orth",
    ]
    for name in order:
        if name not in out_json["methods"]:
            continue
        m = out_json["methods"][name]
        cfg = m["config"]
        cfg_s = f"{cfg.get('pool', '-')}/L{cfg.get('layer', '-')}/C={cfg.get('C', '-')}" if "layer" in cfg else json.dumps(cfg)[:28]
        p = m["pairs"]
        pa = p["all"]
        pbal = p["length_matched_balanced"]
        plm = p["length_matched_5pct"]
        prem = p["patch_removes_code"]
        pseen = p["seen_projects"]
        puns = p["unseen_projects"]
        pchg = p["changed"]
        padd = p["add_only"]
        lines.append(
            f"| `{name}` | `{cfg_s}` | {pa['t0']['net']:+.1f}% / {pa['t_median']['net']:+.1f}% | "
            f"{pa['rank_acc']:.1f}% / {pa['rank_acc_debiased']:.1f}% / {pa['ties_pct']:.1f}% | "
            f"{pbal['rank_acc']:.1f}% | {plm['rank_acc']:.1f}% | **{prem['rank_acc']:.1f}%** | "
            f"{pseen['rank_acc']:.1f}% / {puns['rank_acc']:.1f}% | "
            f"{pchg['rank_acc']:.1f}% / {padd['rank_acc']:.1f}% | {pa['spearman_log_chars']:+.3f} |"
        )

    lines.extend([
        "",
        "## 2. Full PrimeVul Test (Weighted Estimate over 25,911 Functions, prevalence = 2.68%)",
        "",
        "| Method | ROC-AUC | **Debiased ROC-AUC** | AUPRC | **VD-S (FNR @ 0.5% FPR)** | Recall @ 0.5% FPR | Actual FPR | Spearman(log chars) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in order:
        ft = out_json["methods"].get(name, {}).get("fulltest")
        if not ft:
            continue
        lines.append(
            f"| `{name}` | {ft['roc_auc']:.4f} | **{ft['debiased_roc_auc']:.4f}** | {ft['auprc']:.4f} | "
            f"**{ft['vds_fnr_at_05pct_fpr']:.2f}%** | {ft['recall_at_05pct_fpr']:.2f}% | {ft['actual_fpr_pct']:.3f}% | {ft['spearman_log_chars']:+.3f} |"
        )

    lines.extend([
        "",
        "## 3. Layer & Pooling Sweep (`probe_len_orth`)",
        "",
        "| Pool | Layer | Best C | Valid Bal Rank | Test All Rank | Test Debias Rank | Test PatchRemoves Rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in out_json.get("layer_sweep", []):
        lines.append(
            f"| `{row['pool']}` | {row['layer']} | {row['best_C']} | {row['valid_bal_rank']:.1f}% | "
            f"{row['test_all_rank']:.1f}% | {row['test_deb_rank']:.1f}% | {row['test_rem_rank']:.1f}% |"
        )
    has_man = any("manual40_local" in out_json["methods"][m]["pairs"] for m in out_json["methods"])
    if has_man:
        lines.extend([
            "",
            "## 4. Manual-40 Taxonomy Stratification (`local` vs `outside` vs `not_a_to_b`)",
            "",
            "| Method | `local` (n=15) Rank / Net@med | `outside` (n=12) Rank / Net@med | `not_a_to_b` (n=11) Rank / Net@med |",
            "|---|---:|---:|---:|",
        ])
        for name in order:
            if name not in out_json["methods"]:
                continue
            pr = out_json["methods"][name]["pairs"]
            loc, out_s, nab = pr.get("manual40_local", {}), pr.get("manual40_outside", {}), pr.get("manual40_not_a_to_b", {})
            if loc.get("n") and out_s.get("n") and nab.get("n"):
                lines.append(
                    f"| `{name}` | {loc['rank_acc']:.1f}% / {loc['t_median']['net']:+.1f}% | "
                    f"{out_s['rank_acc']:.1f}% / {out_s['t_median']['net']:+.1f}% | "
                    f"{nab['rank_acc']:.1f}% / {nab['t_median']['net']:+.1f}% |"
                )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--test_scores", default=None)
    ap.add_argument("--manual40", default=None)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--extra_benign", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_npz", default="/content/exp5/tmp_emb/qwen25_7b_feats.npz")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    splits = make_clean_splits(
        a.primevul, manifest_path=a.manifest, extra_benign=a.extra_benign,
        seed=a.seed, test_scores_path=a.test_scores,
    )
    print("Split stats:", json.dumps(splits["stats"]), flush=True)

    # Collect unique functions across all splits
    unique_items = {}
    for p in splits["train_pairs"] + splits["valid_pairs"] + splits["test_pairs"]:
        unique_items[p["v_idx"]] = {"func": p["v_func"]}
        unique_items[p["f_idx"]] = {"func": p["f_func"]}
    for r in splits["extra_train_benign"]:
        unique_items[r["idx"]] = {"func": r["func"]}
    for idx, r in splits["fulltest_rows"].items():
        unique_items[idx] = {"func": r["func"]}

    feats = extract_features(unique_items, a.model, max_len=a.max_len, cache_npz=a.cache_npz)
    methods_res, layer_sweep = _fit_and_eval_probes(
        splits, feats, a.out, manual40_path=a.manual40, seed=a.seed,
    )

    out_json = {
        "telemetry": feats["telemetry"],
        "split_stats": splits["stats"],
        "methods": methods_res,
        "layer_sweep": layer_sweep,
    }
    with open(os.path.join(a.out, "t1_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out_json, fh, indent=1)
    report_md = format_report_md(out_json)
    with open(os.path.join(a.out, "t1_report.md"), "w", encoding="utf-8") as fh:
        fh.write(report_md)
    print(report_md, flush=True)
    print("T1_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
