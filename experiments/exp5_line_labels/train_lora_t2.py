"""Plan v3 Tier 2: Paired Anti-Shortcut QLoRA Fine-Tuning on Qwen2.5-Coder-7B-Instruct.

Directly targets the two Tier 1 failure modes identified on PrimeVul:
  1. Length-matched extra benign sampling (80% log-chars quantile matched + 20% random)
     combined with 50/50 cf>cv vs cv>cf pair weighting to eliminate both negative
     within-pair and positive cross-function length shortcuts + Guard-Syntax collapse.
  2. Dual-layer Residual Mean + Attention Pooling head (L13 + L27) on 4-bit NF4 QLoRA
     trained jointly on within-pair margin + global vulnerable/benign separation.

All pure-Python data/weighting/mask/report helpers stay at top level (stdlib only) so
unit tests run cleanly without torch/numpy/transformers.
"""
import argparse
import bisect
import json
import math
import os
import random
import shutil
import time

from probe_llm_t1 import (
    _layer_indices,
    _rank_acc_and_debiased,
    build_probe_input_ids,
    compute_fulltest_eval,
    compute_pair_eval,
    load_jsonl,
    make_clean_splits,
    text_hash,
)


def compute_pair_length_weights(pairs, max_weight=4.0):
    """Return per-pair weights giving equal total mass to cf>cv and cv>cf pairs.

    Mean weight across all pairs is normalized to 1.0 and clipped to [0.1, max_weight].
    """
    n = len(pairs)
    if n == 0:
        return []
    add_idx = [i for i, p in enumerate(pairs) if p["cf"] > p["cv"]]
    rem_idx = [i for i, p in enumerate(pairs) if p["cv"] > p["cf"]]
    raw = [1.0] * n
    if add_idx and rem_idx:
        n_asym = float(len(add_idx) + len(rem_idx))
        w_add = min(max_weight, max(0.1, 0.5 * n_asym / len(add_idx)))
        w_rem = min(max_weight, max(0.1, 0.5 * n_asym / len(rem_idx)))
        for i in add_idx:
            raw[i] = w_add
        for i in rem_idx:
            raw[i] = w_rem
    mean_w = sum(raw) / n
    if mean_w > 0:
        raw = [min(max_weight, max(0.1, w / mean_w)) for w in raw]
    return raw


def sample_length_matched_benigns(train_pairs, benign_pool, n_total, matched_frac=0.80, n_bins=20, seed=42):
    """Sample unpaired benign functions matching log(chars) distribution of train_pairs.

    Samples `round(n_total * matched_frac)` via quantile bins of train_pairs log(chars)
    (expanding to nearest non-empty bins if a bin runs out) and the remainder uniformly
    from leftover pool items. Never emits duplicates.
    """
    rng = random.Random(seed)
    pool = list(benign_pool)
    if n_total <= 0 or not pool:
        return [], {"requested": n_total, "selected": 0, "matched": 0, "random": 0}
    target_n = min(int(n_total), len(pool))
    n_match_target = min(target_n, max(0, int(round(target_n * matched_frac))))

    pair_logs = sorted(math.log(max(c, 1)) for p in train_pairs for c in (p["cv"], p["cf"]))
    if not pair_logs or n_match_target == 0:
        chosen = rng.sample(pool, target_n)
        return chosen, {"requested": n_total, "selected": len(chosen), "matched": 0, "random": len(chosen)}

    k = max(1, min(int(n_bins), len(pair_logs)))
    edges = [pair_logs[int(i * (len(pair_logs) - 1) / k)] for i in range(1, k)]

    def _bin_of(chars_val):
        return bisect.bisect_right(edges, math.log(max(chars_val, 1)))

    bins = [[] for _ in range(k)]
    for pos, row in enumerate(pool):
        c = row.get("chars", len(row.get("func", "")))
        bins[_bin_of(c)].append(pos)
    for b in bins:
        rng.shuffle(b)

    # Round-robin quota across the k quantile bins (shuffled bin visit order)
    bin_order = list(range(k))
    rng.shuffle(bin_order)
    used_pos = set()
    matched_pos = []

    for step in range(n_match_target):
        desired = bin_order[step % k]
        picked = None
        for offset in range(k):
            for cand_b in (desired + offset, desired - offset) if offset > 0 else (desired,):
                if 0 <= cand_b < k and bins[cand_b]:
                    picked = bins[cand_b].pop()
                    break
            if picked is not None:
                break
        if picked is None:
            break
        used_pos.add(picked)
        matched_pos.append(picked)

    rem_need = target_n - len(matched_pos)
    leftover = [i for i in range(len(pool)) if i not in used_pos]
    rand_pos = rng.sample(leftover, min(rem_need, len(leftover))) if rem_need > 0 else []
    selected = [pool[i] for i in matched_pos + rand_pos]
    rng.shuffle(selected)

    sel_logs = sorted(math.log(max(r.get("chars", len(r.get("func", ""))), 1)) for r in selected)
    p_med = math.exp(pair_logs[len(pair_logs) // 2])
    s_med = math.exp(sel_logs[len(sel_logs) // 2]) if sel_logs else 0.0
    stats = {
        "requested": n_total,
        "selected": len(selected),
        "matched": len(matched_pos),
        "random": len(rand_pos),
        "pair_median_chars": round(p_med, 1),
        "benign_median_chars": round(s_med, 1),
    }
    return selected, stats


def load_valid_extra_benigns(primevul_dir, valid_pairs, test_pairs, n_sample=1000, seed=42):
    """Load clean unpaired benigns from primevul_valid.jsonl strictly isolated from test."""
    if n_sample <= 0:
        return []
    valid_path = os.path.join(primevul_dir, "primevul_valid.jsonl")
    if not os.path.exists(valid_path):
        return []

    test_full_path = os.path.join(primevul_dir, "primevul_test.jsonl")
    test_full_rows = load_jsonl(test_full_path) if os.path.exists(test_full_path) else []

    banned_idx = {p["v_idx"] for p in valid_pairs + test_pairs} | {p["f_idx"] for p in valid_pairs + test_pairs}
    banned_commits = {p["commit_id"] for p in test_pairs if p.get("commit_id")} | {
        r["commit_id"] for r in test_full_rows if r.get("commit_id")
    }
    banned_hashes = (
        {text_hash(p["v_func"]) for p in test_pairs}
        | {text_hash(p["f_func"]) for p in test_pairs}
        | {text_hash(r.get("func", "")) for r in test_full_rows}
    )

    pool = []
    for r in load_jsonl(valid_path):
        if (
            r.get("target") == 0
            and r["idx"] not in banned_idx
            and r.get("commit_id") not in banned_commits
            and text_hash(r.get("func", "")) not in banned_hashes
        ):
            pool.append({
                "idx": r["idx"],
                "target": 0,
                "func": r["func"],
                "chars": len(r["func"]),
                "n_lines": r["func"].count("\n") + 1,
                "commit_id": r.get("commit_id") or "",
                "project": r.get("project") or "",
            })
    rng = random.Random(seed)
    return rng.sample(pool, min(len(pool), int(n_sample))) if pool else []


def build_manual40_and_taxonomy_masks(test_pairs, manual40_path=None, taxonomy_path=None):
    """Return boolean index lists for manual_labels_test40 (40/40 verified) & patch_taxonomy.

    Uses pure Python lists so masks can be built and unit-tested without numpy.
    """
    n = len(test_pairs)
    masks = {}
    info = {"manual40_matched": 0, "taxonomy_matched": 0}

    if manual40_path and os.path.exists(manual40_path):
        with open(manual40_path, encoding="utf-8") as mfh:
            man40 = json.load(mfh)
        labels40 = man40.get("labels", [])
        # manual_labels_test40.json recorded indices from full shuffle sample(range(N), N)[:40]
        samp = random.Random(1).sample(range(n), n)[:len(labels40)]
        matches = sum(
            1 for x in labels40
            if x["s"] < len(samp) and (not x.get("project") or test_pairs[samp[x["s"]]]["project"] == x["project"])
        )
        info["manual40_matched"] = matches
        if labels40 and matches == len(labels40):
            local_i = {samp[x["s"]] for x in labels40 if x.get("a_to_b") is True and x.get("outside") is False}
            out_i = {samp[x["s"]] for x in labels40 if x.get("a_to_b") is True and x.get("outside") is not False}
            not_ab_i = {samp[x["s"]] for x in labels40 if x.get("a_to_b") is False}
            masks["manual40_local"] = [i in local_i for i in range(n)]
            masks["manual40_outside"] = [i in out_i for i in range(n)]
            masks["manual40_not_a_to_b"] = [i in not_ab_i for i in range(n)]

    if taxonomy_path and os.path.exists(taxonomy_path):
        tax_by_vidx = {r["idx"]: r for r in load_jsonl(taxonomy_path)}
        info["taxonomy_matched"] = sum(1 for p in test_pairs if p["v_idx"] in tax_by_vidx)
        if info["taxonomy_matched"] == n and n > 0:
            masks["tax_local_ab"] = [
                bool(tax_by_vidx[p["v_idx"]].get("a_to_b")) and not bool(tax_by_vidx[p["v_idx"]].get("needs_outside"))
                for p in test_pairs
            ]
            masks["tax_out_ab"] = [
                bool(tax_by_vidx[p["v_idx"]].get("a_to_b")) and bool(tax_by_vidx[p["v_idx"]].get("needs_outside"))
                for p in test_pairs
            ]
            masks["tax_not_a_to_b"] = [not bool(tax_by_vidx[p["v_idx"]].get("a_to_b")) for p in test_pairs]

    return masks, info


def pair_benigns_by_length(pair_lengths, benign_lengths, window=16, seed=42):
    """Match each pair index to a benign index of nearly identical length.

    Uses sorted length alignment + local windowed shuffle so epochs vary without
    introducing length mismatch between pairs and extra benigns. Handles M < N
    or M >= N cleanly.
    """
    n, m = len(pair_lengths), len(benign_lengths)
    if n == 0:
        return []
    if m == 0:
        return [0] * n
    p_sorted = sorted(range(n), key=lambda i: (pair_lengths[i], i))
    b_sorted = sorted(range(m), key=lambda j: (benign_lengths[j], j))
    aligned = [b_sorted[(r * m) // n] for r in range(n)]
    w = max(1, int(window))
    rng = random.Random(seed)
    for start in range(0, n, w):
        chunk = aligned[start:start + w]
        rng.shuffle(chunk)
        aligned[start:start + w] = chunk
    out = [0] * n
    for pos, p_idx in enumerate(p_sorted):
        out[p_idx] = aligned[pos]
    return out


def build_length_bucketed_batches(pair_max_lens, seed=42):
    """Pack pairs into length-homogeneous micro-batches and shuffle micro-batch order.

    Buckets: max_tok_len <= 512 -> k=4 (B=12 seqs), <= 1024 -> k=2 (B=6), > 1024 -> k=1 (B=3).
    """
    if not pair_max_lens:
        return []
    sorted_items = sorted(pair_max_lens, key=lambda x: (x[1], x[0]))
    batches = []
    for pred, k_max in (
        (lambda L: L <= 512, 4),
        (lambda L: 512 < L <= 1024, 2),
        (lambda L: L > 1024, 1),
    ):
        grp = [p_idx for p_idx, ml in sorted_items if pred(ml)]
        for i in range(0, len(grp), k_max):
            batches.append(grp[i:i + k_max])
    rng = random.Random(seed)
    rng.shuffle(batches)
    return batches


def compute_t2_composite_val(val_bal_rank_pct, val_debias_roc):
    """Model selection metric balancing 50/50 within-pair rank and global debiased ROC."""
    return 0.5 * (float(val_bal_rank_pct) / 100.0) + 0.5 * float(val_debias_roc)


def evaluate_t2_gate(pair_res, full_res):
    """Evaluate pre-registered Plan v3 Tier 2 gate thresholds cleanly."""
    pa = pair_res.get("all", {})
    prem = pair_res.get("patch_removes_code", {})
    puns = pair_res.get("unseen_projects", {})
    net_med = pa.get("t_median", {}).get("net", 0.0)
    rem_rank = prem.get("rank_acc", 0.0)
    unseen_rank = puns.get("rank_acc", 0.0)
    ft_deb = full_res.get("debiased_roc_auc", 0.0) if full_res else 0.0
    ft_sp = abs(full_res.get("spearman_log_chars", 1.0)) if full_res else 1.0

    checks = {
        "net_med_ge_14": net_med >= 14.0,
        "patch_removes_ge_58": rem_rank >= 58.0,
        "unseen_projects_ge_65": unseen_rank >= 65.0,
        "fulltest_debias_roc_ge_065": ft_deb >= 0.65,
        "fulltest_abs_spearman_lt_025": ft_sp < 0.25,
    }
    if all(checks.values()):
        verdict = "PASS_T2_GATE"
    elif net_med >= 10.0 or ft_deb >= 0.62:
        verdict = "PARTIAL_BELOW_TARGET"
    else:
        verdict = "FAIL_CODE_ONLY_CEILING"
    return {"verdict": verdict, "checks": checks}


def format_t2_report_md(out_json):
    """Build concise Markdown report for Tier 2 Paired Anti-Shortcut QLoRA."""
    tel = out_json.get("telemetry", {})
    ss = out_json.get("split_stats", {})
    bs = out_json.get("benign_stats", {})
    lines = [
        "# Plan v3 Tier 2 — Paired Anti-Shortcut QLoRA Report",
        "",
        f"- **Model:** `{tel.get('model')}` (4-bit NF4 QLoRA `r={tel.get('lora_r')}`, `alpha={tel.get('lora_alpha')}`, layers `{tel.get('mid_layer')}+{tel.get('last_layer')}`, `max_len={tel.get('max_len')}`)",
        f"- **Telemetry:** trainable params `{tel.get('trainable_params')}`, peak VRAM `{tel.get('peak_vram_gb')} GB`, train time `{tel.get('train_secs')}s`, OOM fallbacks `{tel.get('oom_fallbacks', 0)}`",
        f"- **Clean Splits & Anti-Shortcut Sampling:** train `{ss.get('train_pairs')}` pairs (`cf>cv={ss.get('train_add')}`, `cv>cf={ss.get('train_rem')}`), extra benign `{bs.get('selected')}` (`matched={bs.get('matched')}`, `random={bs.get('random')}`, median chars pair=`{bs.get('pair_median_chars')}` vs benign=`{bs.get('benign_median_chars')}`), valid `{ss.get('valid_pairs')}` pairs + `{ss.get('valid_benign')}` benign",
    ]
    if out_json.get("gate"):
        g = out_json["gate"]
        lines.append(f"- **Pre-Registered Gate Verdict:** **`{g['verdict']}`** (`{json.dumps(g['checks'])}`)")

    if out_json.get("curve"):
        lines.extend([
            "",
            "## 1. Training & Validation Curve",
            "",
            "| Epoch | Pairs Done | Train Loss | Valid Bal Rank | Valid `PatchRemoves` | Valid Debias ROC | Valid Net@med | Composite |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in out_json["curve"]:
            lines.append(
                f"| {row['epoch']:.2f} | {row['pairs_done']} | {row['train_loss']:.4f} | "
                f"{row['val_bal_rank']:.1f}% | {row['val_rem_rank']:.1f}% | "
                f"{row['val_debias_roc']:.4f} | {row['val_net_med']:+.1f}% | {row['val_composite']:.4f} |"
            )

    pr = out_json.get("test_pairs")
    if pr:
        pa = pr["all"]
        pbal = pr["length_matched_balanced"]
        plm = pr["length_matched_5pct"]
        prem = pr["patch_removes_code"]
        pseen = pr["seen_projects"]
        puns = pr["unseen_projects"]
        pchg = pr["changed"]
        padd = pr["add_only"]
        lines.extend([
            "",
            "## 2. 5-Gate Anti-Shortcut Pair Matrix (`primevul_test_paired`)",
            "",
            "| Method | Net@t0 / Net@med | All Rank / Debias / Ties | Bal Rank | LenMatched5% | **PatchRemoves Rank** | Seen / Unseen | Changed / AddOnly | Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| `lora_nohint` | {pa['t0']['net']:+.1f}% / {pa['t_median']['net']:+.1f}% | "
                f"{pa['rank_acc']:.1f}% / {pa['rank_acc_debiased']:.1f}% / {pa['ties_pct']:.1f}% | "
                f"{pbal['rank_acc']:.1f}% | {plm['rank_acc']:.1f}% | **{prem['rank_acc']:.1f}%** | "
                f"{pseen['rank_acc']:.1f}% / {puns['rank_acc']:.1f}% | "
                f"{pchg['rank_acc']:.1f}% / {padd['rank_acc']:.1f}% | {pa['spearman_log_chars']:+.3f} |"
            ),
        ])

    ft = out_json.get("test_fulltest")
    if ft:
        lines.extend([
            "",
            "## 3. Full PrimeVul Test (25,911 Functions Weighted Estimate)",
            "",
            "| Method | ROC-AUC | **Debiased ROC-AUC** | AUPRC | **VD-S (FNR @ 0.5% FPR)** | Recall @ 0.5% FPR | Actual FPR | Spearman(log chars) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| `lora_nohint` | {ft['roc_auc']:.4f} | **{ft['debiased_roc_auc']:.4f}** | {ft['auprc']:.4f} | "
                f"**{ft['vds_fnr_at_05pct_fpr']:.2f}%** | {ft['recall_at_05pct_fpr']:.2f}% | {ft['actual_fpr_pct']:.3f}% | {ft['spearman_log_chars']:+.3f} |"
            ),
        ])

    if pr and any(k in pr for k in ("manual40_local", "tax_local_ab")):
        lines.extend([
            "",
            "## 4. Local vs Outside Stratification (`manual40` & `patch_taxonomy`)",
            "",
            "| Subset | n | Rank Acc | Debias Rank | Net@med |",
            "|---|---:|---:|---:|---:|",
        ])
        for k in ("manual40_local", "manual40_outside", "manual40_not_a_to_b", "tax_local_ab", "tax_out_ab", "tax_not_a_to_b"):
            if k in pr and pr[k].get("n"):
                sk = pr[k]
                lines.append(f"| `{k}` | {sk['n']} | {sk['rank_acc']:.1f}% | {sk['rank_acc_debiased']:.1f}% | {sk['t_median']['net']:+.1f}% |")

    return "\n".join(lines) + "\n"


def _find_decoder_layers(backbone):
    """Locate decoder ModuleList across raw transformers or PEFT-wrapped Qwen2 backbones."""
    candidates = [
        getattr(backbone, "layers", None),
        getattr(getattr(backbone, "model", None), "layers", None),
        getattr(getattr(getattr(backbone, "base_model", None), "model", None), "layers", None),
        getattr(getattr(getattr(getattr(backbone, "base_model", None), "model", None), "model", None), "layers", None),
    ]
    for c in candidates:
        if c is not None and len(c) > 0:
            return c
    raise RuntimeError("Could not locate decoder layers on backbone")


def _compute_triplet_loss(sv, sf, sb, w_vec, margin):
    """Exact combined Tier 2 objective (batched sum over k pairs so grad/grad_accum is k-invariant)."""
    import torch
    import torch.nn.functional as F

    pair_rank = (w_vec * F.softplus(margin - (sv - sf))).sum()
    pair_bce = (
        0.5
        * w_vec
        * 0.5
        * (
            F.binary_cross_entropy_with_logits(sv, torch.ones_like(sv), reduction="none")
            + F.binary_cross_entropy_with_logits(sf, torch.zeros_like(sf), reduction="none")
        )
    ).sum()
    benign_loss = (
        0.75
        * (
            F.binary_cross_entropy_with_logits(sb, torch.zeros_like(sb), reduction="none")
            + 0.5 * F.softplus(margin - (sv - sb))
        )
    ).sum()
    total = pair_rank + pair_bce + benign_loss
    return total, pair_rank, pair_bce, benign_loss


def run_training_and_eval(a):
    """Run Tier 2 QLoRA training, checkpointed resume, valid selection, and full test eval."""
    import numpy as np
    import torch
    import torch.nn as nn
    from peft import (
        LoraConfig,
        get_peft_model,
        load_peft_weights,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    os.makedirs(a.out, exist_ok=True)
    splits = make_clean_splits(
        a.primevul, manifest_path=a.manifest, extra_benign=10_000_000,
        seed=a.seed, test_scores_path=a.test_scores,
    )
    train_pairs = splits["train_pairs"]
    valid_pairs = splits["valid_pairs"]
    test_pairs = splits["test_pairs"]
    benign_pool = splits["extra_train_benign"]
    fulltest_rows = splits["fulltest_rows"]
    manifest = splits["manifest"]

    if a.train_limit > 0:
        train_pairs = train_pairs[:a.train_limit]
    if a.valid_limit > 0:
        valid_pairs = valid_pairs[:a.valid_limit]

    valid_benign = load_valid_extra_benigns(a.primevul, valid_pairs, test_pairs, n_sample=a.valid_benign, seed=a.seed)
    selected_benign, benign_stats = sample_length_matched_benigns(
        train_pairs, benign_pool, n_total=a.extra_benign,
        matched_frac=a.matched_frac, n_bins=20, seed=a.seed,
    )

    # Recompute exact seen_project against actual train pairs + sampled extra benigns
    train_projects = {p["project"] for p in train_pairs if p.get("project")} | {
        r["project"] for r in selected_benign if r.get("project")
    }
    for p in test_pairs:
        p["seen_project"] = bool(p.get("project") and p["project"] in train_projects)

    pair_weights = compute_pair_length_weights(train_pairs, max_weight=4.0)
    print(
        f"[data] train_pairs={len(train_pairs)} extra_benign={len(selected_benign)} "
        f"valid_pairs={len(valid_pairs)} valid_benign={len(valid_benign)} benign_stats={benign_stats}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    # Pre-tokenize all items once on CPU
    tok_cache = {}
    def _tok_item(idx, func):
        if idx not in tok_cache:
            ids, cut = build_probe_input_ids(func, tok, max_len=a.max_len)
            tok_cache[idx] = (ids, int(cut), len(func or ""))
        return tok_cache[idx]

    for p in train_pairs + valid_pairs + (test_pairs if not a.skip_test else []):
        _tok_item(p["v_idx"], p["v_func"])
        _tok_item(p["f_idx"], p["f_func"])
    for r in selected_benign + valid_benign:
        _tok_item(r["idx"], r["func"])
    if not a.skip_test:
        for idx, r in fulltest_rows.items():
            _tok_item(idx, r["func"])

    class DualLayerAttnHead(nn.Module):
        """Float32 two-layer residual mean + attention pooling classification head."""

        def __init__(self, hidden_size):
            super().__init__()
            self.in_norm_mid = nn.LayerNorm(hidden_size)
            self.in_norm_last = nn.LayerNorm(hidden_size)
            self.out_norm_mid = nn.LayerNorm(hidden_size)
            self.out_norm_last = nn.LayerNorm(hidden_size)
            self.q_mid = nn.Linear(hidden_size, 1)
            self.q_last = nn.Linear(hidden_size, 1)
            self.gate_mid = nn.Parameter(torch.tensor(0.05))
            self.gate_last = nn.Parameter(torch.tensor(0.05))
            self.mix_logits = nn.Parameter(torch.tensor([1.0, 0.5]))
            self.drop = nn.Dropout(0.15)
            self.classifier = nn.Linear(hidden_size, 1)
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

        def _pool_branch(self, h, mask, in_norm, out_norm, q_lin, gate):
            hf = h.float()
            mf = mask.to(hf.dtype).unsqueeze(-1)
            mean_n = in_norm((hf * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0))
            hn = in_norm(hf)
            logits = q_lin(hn).squeeze(-1).masked_fill(mask == 0, -1e9)
            w = torch.softmax(logits, dim=-1).unsqueeze(-1)
            attn_v = (hn * w).sum(dim=1)
            return out_norm(mean_n + torch.tanh(gate) * attn_v)

        def forward(self, h_mid, h_last, mask):
            z_mid = self._pool_branch(h_mid, mask, self.in_norm_mid, self.out_norm_mid, self.q_mid, self.gate_mid)
            z_last = self._pool_branch(h_last, mask, self.in_norm_last, self.out_norm_last, self.q_last, self.gate_last)
            alpha = torch.softmax(self.mix_logits, dim=0)
            z = alpha[0] * z_mid + alpha[1] * z_last
            return self.classifier(self.drop(z)).squeeze(-1)

    class Qwen25VulClassifier(nn.Module):
        """4-bit NF4 QLoRA backbone + mid/last forward hook + DualLayerAttnHead."""

        def __init__(self, backbone, head, mid_layer_idx):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.mid_layer_idx = mid_layer_idx
            self._hook_state = {"capturing": False, "mid": None}
            layers = _find_decoder_layers(backbone)

            def _mid_hook(_mod, _inp, output):
                if self._hook_state["capturing"]:
                    self._hook_state["mid"] = output[0] if isinstance(output, tuple) else output

            self._hook_handle = layers[mid_layer_idx].register_forward_hook(_mid_hook)

        def extract_hidden(self, input_ids, attention_mask):
            self._hook_state["mid"] = None
            self._hook_state["capturing"] = True
            try:
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                h_last = out.last_hidden_state
                h_mid = self._hook_state["mid"]
            finally:
                self._hook_state["capturing"] = False
                self._hook_state["mid"] = None
            if h_mid is None:
                raise RuntimeError("Mid-layer hook did not capture hidden state")
            return h_mid, h_last

        def forward(self, input_ids, attention_mask):
            h_mid, h_last = self.extract_hidden(input_ids, attention_mask)
            return self.head(h_mid, h_last, attention_mask)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"[model] Loading 4-bit NF4 base AutoModel: {a.model}", flush=True)
    base_model = AutoModel.from_pretrained(
        a.model,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    num_layers = int(base_model.config.num_hidden_layers)
    hidden_size = int(base_model.config.hidden_size)
    mid_layer = _layer_indices(num_layers)[0]
    last_layer = num_layers - 1

    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_cfg = LoraConfig(
        r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    backbone = get_peft_model(base_model, lora_cfg)
    head = DualLayerAttnHead(hidden_size).to("cuda")
    model = Qwen25VulClassifier(backbone, head, mid_layer)

    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    n_trainable = sum(p.numel() for p in lora_params + head_params)
    print(f"[model] layers={num_layers} d={hidden_size} hook={mid_layer}+{last_layer} trainable={n_trainable:,}", flush=True)

    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}],
        weight_decay=0.01,
    )
    total_pairs_planned = max(1, len(train_pairs) * a.epochs)
    total_opt_steps = max(1, math.ceil(total_pairs_planned / a.grad_accum))
    head_warmup_opt_steps = max(0, math.ceil(a.head_warmup_pairs / a.grad_accum)) if a.head_warmup_pairs > 0 else 0

    def _lr_lora(step):
        if step < head_warmup_opt_steps:
            return 0.0
        lora_tot = max(1, total_opt_steps - head_warmup_opt_steps)
        w = min(lora_tot, max(2, int(0.10 * lora_tot)))
        s = step - head_warmup_opt_steps
        if s < w:
            return float(s + 1) / float(w)
        prog = float(s - w) / max(1.0, float(lora_tot - w))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    def _lr_head(step):
        w = max(1, head_warmup_opt_steps // 2) if head_warmup_opt_steps > 0 else max(1, int(0.08 * total_opt_steps))
        if step < w:
            return float(step + 1) / float(w)
        prog = float(step - w) / max(1.0, float(total_opt_steps - w))
        return max(0.10, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [_lr_lora, _lr_head])
    print(
        f"[sched] total_opt_steps={total_opt_steps} head_warmup_opt_steps={head_warmup_opt_steps} "
        f"lora_lr={a.lr} head_lr={a.head_lr}",
        flush=True,
    )

    def _score_items(idx_list):
        model.eval()
        items = sorted([(idx, tok_cache[idx][0]) for idx in idx_list], key=lambda x: len(x[1]))
        scores = {}
        pos = 0
        with torch.inference_mode():
            while pos < len(items):
                tl0 = len(items[pos][1])
                bs = 4 if tl0 <= 512 else (2 if tl0 <= 1024 else 1)
                batch = items[pos:pos + bs]
                max_b = max(len(x[1]) for x in batch)
                inp = torch.full((len(batch), max_b), pad_id, dtype=torch.long, device="cuda")
                mask = torch.zeros((len(batch), max_b), dtype=torch.long, device="cuda")
                for bi, (_, ids) in enumerate(batch):
                    inp[bi, :len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")
                    mask[bi, :len(ids)] = 1
                try:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        sc = model(inp, mask).float().cpu().tolist()
                    for (idx_b, _), v in zip(batch, sc):
                        scores[idx_b] = float(v)
                    pos += len(batch)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(batch) == 1:
                        raise
                    for idx_b, ids in batch:
                        inp1 = torch.tensor([ids], dtype=torch.long, device="cuda")
                        mask1 = torch.ones_like(inp1)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            scores[idx_b] = float(model(inp1, mask1).float().item())
                    pos += len(batch)
        return scores

    def _eval_valid():
        need = sorted({p["v_idx"] for p in valid_pairs} | {p["f_idx"] for p in valid_pairs} | {r["idx"] for r in valid_benign})
        sc_map = _score_items(need)
        sv = np.array([sc_map[p["v_idx"]] for p in valid_pairs], dtype=np.float64)
        sf = np.array([sc_map[p["f_idx"]] for p in valid_pairs], dtype=np.float64)
        cv = np.array([p["cv"] for p in valid_pairs], dtype=np.float64)
        cf = np.array([p["cf"] for p in valid_pairs], dtype=np.float64)
        vw = np.array(compute_pair_length_weights(valid_pairs, max_weight=4.0), dtype=np.float64)
        bal_rank, _, _ = _rank_acc_and_debiased(sv, sf, cv, cf, weights=vw)
        rem_m = cv > cf
        rem_rank = _rank_acc_and_debiased(sv[rem_m], sf[rem_m], cv[rem_m], cf[rem_m])[0] if rem_m.any() else 0.0
        alls_pair = np.concatenate([sv, sf])
        med = float(np.median(alls_pair))
        net_med = float((((sv > med) & (sf <= med)).astype(float) - ((sv <= med) & (sf > med)).astype(float)).mean() * 100.0)

        # Unpaired + paired valid global debiased ROC-AUC
        all_idx = [p["v_idx"] for p in valid_pairs] + [p["f_idx"] for p in valid_pairs] + [r["idx"] for r in valid_benign]
        y_va = np.array([1] * len(valid_pairs) + [0] * (len(valid_pairs) + len(valid_benign)), dtype=np.int64)
        s_va = np.array([sc_map[i] for i in all_idx], dtype=np.float64)
        c_va = np.array([tok_cache[i][2] for i in all_idx], dtype=np.float64)
        logc = np.log(np.maximum(c_va, 1.0))
        V = np.vander(logc, 4)
        coef, *_ = np.linalg.lstsq(V, s_va, rcond=None)
        s_deb = s_va - V @ coef
        deb_roc = float(roc_auc_score(y_va, s_deb))
        comp = compute_t2_composite_val(bal_rank, deb_roc)
        return {
            "val_bal_rank": round(bal_rank, 2),
            "val_rem_rank": round(rem_rank, 2),
            "val_debias_roc": round(deb_roc, 4),
            "val_net_med": round(net_med, 2),
            "val_composite": round(comp, 4),
        }

    ckpt_root = a.ckpt_dir or a.out
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_last_dir = os.path.join(ckpt_root, "ckpt_last")
    ckpt_best_dir = os.path.join(ckpt_root, "ckpt_best")

    def _save_ckpt(path, state_extra):
        tmp = path + "_tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        model.backbone.save_pretrained(os.path.join(tmp, "adapter"))
        torch.save({
            "head": model.head.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            **state_extra,
        }, os.path.join(tmp, "head_and_state.pt"))
        if os.path.exists(path):
            shutil.rmtree(path)
        os.replace(tmp, path)

    start_ep, start_batch_in_ep, pairs_done_in_ep, global_opt_step = 0, 0, 0, 0
    best_composite = -1e9
    curve = []
    oom_fallbacks = 0

    state_pt = os.path.join(ckpt_last_dir, "head_and_state.pt")
    adapter_dir = os.path.join(ckpt_last_dir, "adapter")
    if (not a.no_resume) and os.path.exists(state_pt) and os.path.isdir(adapter_dir):
        print(f"[resume] Loading checkpoint from {ckpt_last_dir}", flush=True)
        set_peft_model_state_dict(model.backbone, load_peft_weights(adapter_dir, device="cuda"), adapter_name="default")
        st = torch.load(state_pt, map_location="cuda", weights_only=False)
        model.head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_ep = int(st["epoch"])
        start_batch_in_ep = int(st.get("batch_in_ep", st.get("pair_in_ep", 0)))
        pairs_done_in_ep = int(st.get("pairs_done_in_ep", 0))
        global_opt_step = int(st["global_opt_step"])
        best_composite = float(st["best_composite"])
        curve = list(st["curve"])
        oom_fallbacks = int(st.get("oom_fallbacks", 0))

    torch.cuda.reset_peak_memory_stats()
    t_train0 = time.time()
    n_tr = len(train_pairs)
    eval_interval_pairs = max(1, int(round(a.eval_every * n_tr)))
    next_eval_pair = ((start_ep * n_tr + pairs_done_in_ep) // eval_interval_pairs + 1) * eval_interval_pairs

    def _pad_triplet_batch(seq_groups):
        """Pack (3*k) variable-length token ID lists into one right-padded GPU batch."""
        max_l = max(len(s) for s in seq_groups)
        inp = torch.full((len(seq_groups), max_l), pad_id, dtype=torch.long, device="cuda")
        msk = torch.zeros((len(seq_groups), max_l), dtype=torch.long, device="cuda")
        for bi, ids in enumerate(seq_groups):
            inp[bi, :len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")
            msk[bi, :len(ids)] = 1
        return inp, msk

    pair_tok_lens = [max(len(tok_cache[p["v_idx"]][0]), len(tok_cache[p["f_idx"]][0])) for p in train_pairs]
    benign_tok_lens = [len(tok_cache[r["idx"]][0]) for r in selected_benign]

    for ep in range(start_ep, a.epochs):
        benign_for_pair = pair_benigns_by_length(pair_tok_lens, benign_tok_lens, window=16, seed=a.seed + 1000 * ep)
        pair_max_lens = []
        for p_idx in range(n_tr):
            b_idx = benign_for_pair[p_idx] if selected_benign else None
            b_len = benign_tok_lens[b_idx] if b_idx is not None else 0
            pair_max_lens.append((p_idx, max(pair_tok_lens[p_idx], b_len)))
        batches = build_length_bucketed_batches(pair_max_lens, seed=a.seed + 2000 * ep)

        # Track sliding window metrics since last log print
        win_loss = win_pair = win_bce = win_ben = 0.0
        win_pairs = win_toks = 0
        ep_loss_sum = ep_pairs_cnt = 0
        last_grad_norm = 0.0
        opt.zero_grad(set_to_none=True)
        accum_pairs = 0
        ckpt_next_pairs = ((pairs_done_in_ep // a.ckpt_every) + 1) * a.ckpt_every if a.ckpt_every > 0 else 10**9
        next_log_pairs = pairs_done_in_ep + 50
        t_win0 = time.time()
        t_ep0 = time.time()

        b_start = start_batch_in_ep if ep == start_ep else 0
        cur_pairs_ep = pairs_done_in_ep if ep == start_ep else 0
        pairs_at_ep_start = cur_pairs_ep

        def _forward_triplet_group(sub_pairs, head_only):
            k = len(sub_pairs)
            v_seqs = [tok_cache[train_pairs[pi]["v_idx"]][0] for pi in sub_pairs]
            f_seqs = [tok_cache[train_pairs[pi]["f_idx"]][0] for pi in sub_pairs]
            b_seqs = [
                tok_cache[selected_benign[benign_for_pair[pi]]["idx"]][0] if selected_benign
                else tok_cache[train_pairs[pi]["f_idx"]][0]
                for pi in sub_pairs
            ]
            inp, msk = _pad_triplet_batch(v_seqs + f_seqs + b_seqs)
            if head_only:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h_mid, h_last = model.extract_hidden(inp, msk)
                logits = model.head(h_mid.detach(), h_last.detach(), msk)
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(inp, msk)
            sv, sf, sb = logits[:k], logits[k:2 * k], logits[2 * k:]
            w_vec = sv.new_tensor([float(pair_weights[pi]) for pi in sub_pairs])
            loss, lp, lbce, lben = _compute_triplet_loss(sv, sf, sb, w_vec, a.margin)
            return loss, lp, lbce, lben

        def _single_pair_scalar_fallback(pi):
            pair = train_pairs[pi]
            v_ids = tok_cache[pair["v_idx"]][0]
            f_ids = tok_cache[pair["f_idx"]][0]
            b_ids = tok_cache[selected_benign[benign_for_pair[pi]]["idx"]][0] if selected_benign else f_ids

            def _fwd1(ids):
                inp1 = torch.tensor([ids], dtype=torch.long, device="cuda")
                msk1 = torch.ones_like(inp1)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    return model(inp1, msk1)

            with torch.no_grad():
                sv0 = float(_fwd1(v_ids).item())
                sf0 = float(_fwd1(f_ids).item())
                sb0 = float(_fwd1(b_ids).item())
            lv = torch.tensor([sv0], dtype=torch.float32, device="cuda", requires_grad=True)
            lf = torch.tensor([sf0], dtype=torch.float32, device="cuda", requires_grad=True)
            lb = torch.tensor([sb0], dtype=torch.float32, device="cuda", requires_grad=True)
            loss_l, lp, lbce, lben = _compute_triplet_loss(lv, lf, lb, lv.new_tensor([float(pair_weights[pi])]), a.margin)
            gv, gf, gb = torch.autograd.grad(loss_l / float(a.grad_accum), (lv, lf, lb))
            vals = (float(loss_l.item()), float(lp.item()), float(lbce.item()), float(lben.item()))
            (_fwd1(v_ids)[0] * gv[0].detach()).backward()
            (_fwd1(f_ids)[0] * gf[0].detach()).backward()
            (_fwd1(b_ids)[0] * gb[0].detach()).backward()
            return vals

        for step_b in range(b_start, len(batches)):
            mb = batches[step_b]
            k = len(mb)
            head_only = global_opt_step < head_warmup_opt_steps
            model.train()

            for pi in mb:
                bi_idx = benign_for_pair[pi] if selected_benign else None
                b_len = benign_tok_lens[bi_idx] if bi_idx is not None else len(tok_cache[train_pairs[pi]["f_idx"]][0])
                win_toks += len(tok_cache[train_pairs[pi]["v_idx"]][0]) + len(tok_cache[train_pairs[pi]["f_idx"]][0]) + b_len

            try:
                loss, lp, lbce, lben = _forward_triplet_group(mb, head_only)
                (loss / float(a.grad_accum)).backward()
                l_vals = (float(loss.item()), float(lp.item()), float(lbce.item()), float(lben.item()))
            except torch.cuda.OutOfMemoryError:
                oom_fallbacks += 1
                loss = lp = lbce = lben = None
                torch.cuda.empty_cache()
                l_vals = [0.0, 0.0, 0.0, 0.0]
                for pi in mb:
                    try:
                        l1, p1, bce1, ben1 = _forward_triplet_group([pi], head_only)
                        (l1 / float(a.grad_accum)).backward()
                        for j_idx, val_t in enumerate((l1, p1, bce1, ben1)):
                            l_vals[j_idx] += float(val_t.item())
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        l1 = p1 = bce1 = ben1 = None
                        v4 = _single_pair_scalar_fallback(pi)
                        for j_idx in range(4):
                            l_vals[j_idx] += v4[j_idx]

            win_loss += l_vals[0]
            win_pair += l_vals[1]
            win_bce += l_vals[2]
            win_ben += l_vals[3]
            win_pairs += k
            ep_loss_sum += l_vals[0]
            ep_pairs_cnt += k
            accum_pairs += k
            cur_pairs_ep += k
            done_total = ep * n_tr + cur_pairs_ep

            if accum_pairs >= a.grad_accum or step_b == len(batches) - 1:
                active_params = [p for p in (lora_params + head_params) if p.grad is not None]
                if active_params:
                    last_grad_norm = float(torch.nn.utils.clip_grad_norm_(active_params, 1.0).item())
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                accum_pairs = 0
                global_opt_step += 1

            if cur_pairs_ep >= next_log_pairs or step_b == len(batches) - 1:
                dt_win = max(time.time() - t_win0, 1e-6)
                el_ep = max(time.time() - t_ep0, 1e-6)
                rate_ep = (cur_pairs_ep - pairs_at_ep_start) / el_ep
                rem_total = (a.epochs * n_tr) - done_total
                eta_min = rem_total / max(rate_ep, 1e-6) / 60.0
                ph = "HEAD" if head_only else "LORA"
                print(
                    f"[train] ep {done_total/n_tr:.2f}/{a.epochs} ({cur_pairs_ep}/{n_tr}) phase={ph} | "
                    f"loss={win_loss/max(win_pairs,1):.4f} "
                    f"(pair={win_pair/max(win_pairs,1):.3f} bce={win_bce/max(win_pairs,1):.3f} ben={win_ben/max(win_pairs,1):.3f}) "
                    f"gnorm={last_grad_norm:.2f} | "
                    f"{win_pairs/dt_win:.2f} pair/s {win_toks/dt_win:.0f} tok/s | "
                    f"peakVRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB | ETA {eta_min:.1f} min",
                    flush=True,
                )
                win_loss = win_pair = win_bce = win_ben = 0.0
                win_pairs = win_toks = 0
                t_win0 = time.time()
                while next_log_pairs <= cur_pairs_ep:
                    next_log_pairs += 50

            if cur_pairs_ep >= ckpt_next_pairs and accum_pairs == 0 and step_b < len(batches) - 1:
                _save_ckpt(ckpt_last_dir, {
                    "epoch": ep, "batch_in_ep": step_b + 1, "pairs_done_in_ep": cur_pairs_ep,
                    "global_opt_step": global_opt_step, "best_composite": best_composite,
                    "curve": curve, "oom_fallbacks": oom_fallbacks,
                })
                while ckpt_next_pairs <= cur_pairs_ep:
                    ckpt_next_pairs += a.ckpt_every

            if done_total >= next_eval_pair or step_b == len(batches) - 1:
                ev = _eval_valid()
                row = {
                    "epoch": round(done_total / n_tr, 3),
                    "pairs_done": done_total,
                    "train_loss": round(ep_loss_sum / max(ep_pairs_cnt, 1), 4),
                    **ev,
                }
                curve.append(row)
                print(f"[valid] {json.dumps(row)}", flush=True)
                if ev["val_composite"] > best_composite:
                    best_composite = ev["val_composite"]
                    _save_ckpt(ckpt_best_dir, {
                        "epoch": ep, "batch_in_ep": step_b + 1, "pairs_done_in_ep": cur_pairs_ep,
                        "global_opt_step": global_opt_step, "best_composite": best_composite,
                        "curve": curve, "oom_fallbacks": oom_fallbacks,
                    })
                    print(f"[valid] saved best checkpoint composite={best_composite:.4f}", flush=True)
                while next_eval_pair <= done_total:
                    next_eval_pair += eval_interval_pairs

        _save_ckpt(ckpt_last_dir, {
            "epoch": ep + 1, "batch_in_ep": 0, "pairs_done_in_ep": 0,
            "global_opt_step": global_opt_step, "best_composite": best_composite,
            "curve": curve, "oom_fallbacks": oom_fallbacks,
        })

    train_secs = round(time.time() - t_train0, 1)
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    test_pair_res, test_full_res, extra_info, gate = None, None, {}, None
    if not a.skip_test:
        if os.path.isdir(os.path.join(ckpt_best_dir, "adapter")):
            print(f"[test] Reloading best checkpoint from {ckpt_best_dir}", flush=True)
            set_peft_model_state_dict(
                model.backbone,
                load_peft_weights(os.path.join(ckpt_best_dir, "adapter"), device="cuda"),
                adapter_name="default",
            )
            bst = torch.load(os.path.join(ckpt_best_dir, "head_and_state.pt"), map_location="cuda", weights_only=False)
            model.head.load_state_dict(bst["head"])

        test_need = sorted(
            {p["v_idx"] for p in test_pairs}
            | {p["f_idx"] for p in test_pairs}
            | set(fulltest_rows.keys())
        )
        te_scores = _score_items(test_need)

        sv_te = np.array([te_scores[p["v_idx"]] for p in test_pairs], dtype=np.float64)
        sf_te = np.array([te_scores[p["f_idx"]] for p in test_pairs], dtype=np.float64)
        cv_te = np.array([p["cv"] for p in test_pairs], dtype=np.float64)
        cf_te = np.array([p["cf"] for p in test_pairs], dtype=np.float64)
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
        ext_masks, extra_info = build_manual40_and_taxonomy_masks(
            test_pairs, manual40_path=a.manual40, taxonomy_path=a.taxonomy,
        )
        for mk, mv in ext_masks.items():
            masks[mk] = np.array(mv, dtype=bool)
        print(f"[test] Extra stratification info: {extra_info}", flush=True)

        test_pair_res = compute_pair_eval(sv_te, sf_te, cv_te, cf_te, masks, seed=a.seed, B=2000)
        if manifest and fulltest_rows:
            test_full_res = compute_fulltest_eval(te_scores, fulltest_rows, manifest)
        gate = evaluate_t2_gate(test_pair_res, test_full_res)

        sc_dir = os.path.join(a.out, "scores", "lora_nohint")
        os.makedirs(sc_dir, exist_ok=True)
        with open(os.path.join(sc_dir, "primevul_test_scores.jsonl"), "w", encoding="utf-8") as fh:
            for p, sv_1, sf_1 in zip(test_pairs, sv_te, sf_te):
                fh.write(json.dumps({"idx": p["v_idx"], "target": 1, "logit": float(sv_1)}) + "\n")
                fh.write(json.dumps({"idx": p["f_idx"], "target": 0, "logit": float(sf_1)}) + "\n")
        if fulltest_rows:
            with open(os.path.join(sc_dir, "fulltest_scores.jsonl"), "w", encoding="utf-8") as fh:
                for idx, row in fulltest_rows.items():
                    fh.write(json.dumps({"idx": idx, "target": row["target"], "logit": float(te_scores[idx])}) + "\n")

    out_json = {
        "telemetry": {
            "model": a.model, "max_len": a.max_len, "lora_r": a.lora_r, "lora_alpha": a.lora_alpha,
            "mid_layer": mid_layer, "last_layer": last_layer, "trainable_params": n_trainable,
            "peak_vram_gb": peak_vram_gb, "train_secs": train_secs, "oom_fallbacks": oom_fallbacks,
        },
        "split_stats": {
            "train_pairs": len(train_pairs),
            "train_add": sum(1 for p in train_pairs if p["cf"] > p["cv"]),
            "train_rem": sum(1 for p in train_pairs if p["cv"] > p["cf"]),
            "valid_pairs": len(valid_pairs),
            "valid_benign": len(valid_benign),
            "test_pairs": len(test_pairs),
            "test_seen_projects": sum(1 for p in test_pairs if p["seen_project"]),
            "test_unseen_projects": sum(1 for p in test_pairs if not p["seen_project"]),
            **extra_info,
        },
        "benign_stats": benign_stats,
        "curve": curve,
        "best_composite": round(best_composite, 4),
        "test_pairs": test_pair_res,
        "test_fulltest": test_full_res,
        "gate": gate,
    }
    with open(os.path.join(a.out, "t2_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out_json, fh, indent=1)
    report_md = format_t2_report_md(out_json)
    with open(os.path.join(a.out, "t2_report.md"), "w", encoding="utf-8") as fh:
        fh.write(report_md)
    print(report_md, flush=True)
    print("T2_LORA_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--test_scores", default=None)
    ap.add_argument("--manual40", default=None)
    ap.add_argument("--taxonomy", default=None)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--extra_benign", type=int, default=4347)
    ap.add_argument("--matched_frac", type=float, default=0.85)
    ap.add_argument("--valid_benign", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--head_warmup_pairs", type=int, default=400)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--eval_every", type=float, default=0.5)
    ap.add_argument("--ckpt_every", type=int, default=400)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--valid_limit", type=int, default=0)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run_training_and_eval(a)


if __name__ == "__main__":
    main()
