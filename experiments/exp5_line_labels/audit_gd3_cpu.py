"""Plan v3 Phase 3.0 CPU Audit: inspect PrimeVul temporal splits, length skew,
token truncation rates, commit/hash leakage, and seen vs unseen projects before GPU work.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_llm_t1 import (  # noqa: E402
    build_probe_input_ids,
    load_jsonl,
    load_paired_split,
    make_clean_splits,
    text_hash,
)


def _split_pair_stats(pairs):
    n = len(pairs)
    n_changed = sum(1 for p in pairs if p["flaw_lines"])
    n_add_only = n - n_changed
    n_cf_gt_cv = sum(1 for p in pairs if p["cf"] > p["cv"])
    n_cv_gt_cf = sum(1 for p in pairs if p["cv"] > p["cf"])
    n_eq = n - n_cf_gt_cv - n_cv_gt_cf
    lm5 = [p for p in pairs if abs(p["cv"] - p["cf"]) / max(p["cv"], p["cf"], 1) < 0.05]
    lm5_cf_gt_cv = sum(1 for p in lm5 if p["cf"] > p["cv"])
    lm5_cv_gt_cf = sum(1 for p in lm5 if p["cv"] > p["cf"])
    return {
        "n_pairs": n,
        "n_changed": n_changed,
        "n_add_only": n_add_only,
        "n_cf_gt_cv": n_cf_gt_cv,
        "n_cv_gt_cf": n_cv_gt_cf,
        "n_equal_len": n_eq,
        "shorter_is_vul_pct": round(100.0 * (n_cf_gt_cv + 0.5 * n_eq) / max(n, 1), 2),
        "length_matched_5pct_count": len(lm5),
        "length_matched_5pct_cf_gt_cv": lm5_cf_gt_cv,
        "length_matched_5pct_cv_gt_cf": lm5_cv_gt_cf,
        "length_matched_5pct_shorter_is_vul_pct": round(
            100.0 * (lm5_cf_gt_cv + 0.5 * (len(lm5) - lm5_cf_gt_cv - lm5_cv_gt_cf)) / max(len(lm5), 1), 2
        ),
    }


def _token_stats(pairs, tok):
    lens = []
    for p in pairs:
        for fn in (p["v_func"], p["f_func"]):
            ids, _ = build_probe_input_ids(fn, tok, max_len=1_000_000)
            lens.append(len(ids))
    lens.sort()
    n = max(len(lens), 1)
    return {
        "n_functions": len(lens),
        "median_tokens": lens[len(lens) // 2] if lens else 0,
        "p90_tokens": lens[int(0.90 * (len(lens) - 1))] if lens else 0,
        "p95_tokens": lens[int(0.95 * (len(lens) - 1))] if lens else 0,
        "max_tokens": lens[-1] if lens else 0,
        "gt_2048_pct": round(100.0 * sum(1 for x in lens if x > 2048) / n, 2),
        "gt_3072_pct": round(100.0 * sum(1 for x in lens if x > 3072) / n, 2),
        "gt_4096_pct": round(100.0 * sum(1 for x in lens if x > 4096) / n, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--test_scores", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    print(f"[audit_gd3] Loading tokenizer {a.tokenizer}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    # Raw split counts before filtering (for leakage report)
    raw_train, _ = load_paired_split(os.path.join(a.primevul, "primevul_train_paired.jsonl"), with_flaw_lines=False)
    raw_valid, _ = load_paired_split(os.path.join(a.primevul, "primevul_valid_paired.jsonl"), with_flaw_lines=False)
    raw_test, _ = load_paired_split(os.path.join(a.primevul, "primevul_test_paired.jsonl"), with_flaw_lines=False)
    test_full = load_jsonl(os.path.join(a.primevul, "primevul_test.jsonl"))

    test_commits = {p["commit_id"] for p in raw_test} | {r.get("commit_id") for r in test_full if r.get("commit_id")}
    test_hashes = (
        {text_hash(p["v_func"]) for p in raw_test}
        | {text_hash(p["f_func"]) for p in raw_test}
        | {text_hash(r.get("func", "")) for r in test_full}
    )

    def _raw_overlap(pairs):
        return {
            "commit_overlap_with_test": sum(1 for p in pairs if p["commit_id"] in test_commits),
            "hash_overlap_with_test": sum(
                1 for p in pairs if text_hash(p["v_func"]) in test_hashes or text_hash(p["f_func"]) in test_hashes
            ),
        }

    splits = make_clean_splits(
        a.primevul, manifest_path=a.manifest, extra_benign=2000, seed=42, test_scores_path=a.test_scores,
    )

    pair_stats = {
        "train_clean": _split_pair_stats(splits["train_pairs"]),
        "valid_clean": _split_pair_stats(splits["valid_pairs"]),
        "test": _split_pair_stats(splits["test_pairs"]),
    }
    token_stats = {
        "train_clean": _token_stats(splits["train_pairs"], tok),
        "valid_clean": _token_stats(splits["valid_pairs"], tok),
        "test": _token_stats(splits["test_pairs"], tok),
    }

    # Project overlap analysis (seen vs unseen on test)
    tr_proj_cnt = collections.Counter(p["project"] for p in splits["train_pairs"] if p.get("project"))
    te_proj_cnt = collections.Counter(p["project"] for p in splits["test_pairs"] if p.get("project"))
    shared_projs = sorted(set(tr_proj_cnt) & set(te_proj_cnt), key=lambda k: -(te_proj_cnt[k] + tr_proj_cnt[k]))
    seen_test_pairs = [p for p in splits["test_pairs"] if p["seen_project"]]
    unseen_test_pairs = [p for p in splits["test_pairs"] if not p["seen_project"]]

    # Fulltest manifest integrity
    manifest_info = {}
    if splits["manifest"] is not None:
        m = splits["manifest"]
        full_by_id = {r["idx"]: r for r in test_full}
        v_set = set(m.get("rest_vulnerable_idx", []))
        b_set = set(m.get("sampled_benign_idx", []))
        p_set = {p["v_idx"] for p in splits["test_pairs"]} | {p["f_idx"] for p in splits["test_pairs"]}
        manifest_info = {
            "n_full_jsonl": len(test_full),
            "n_full_vulnerable_jsonl": sum(1 for r in test_full if r.get("target") == 1),
            "manifest_P_present": len(p_set & set(full_by_id)),
            "manifest_V_present": len(v_set & set(full_by_id)),
            "manifest_B_present": len(b_set & set(full_by_id)),
            "strata": m.get("strata", {}),
        }

    audit = {
        "split_stats": splits["stats"],
        "raw_overlap_before_filter": {"train": _raw_overlap(raw_train), "valid": _raw_overlap(raw_valid)},
        "pair_stats": pair_stats,
        "token_stats": token_stats,
        "project_overlap": {
            "n_train_projects": len(tr_proj_cnt),
            "n_test_projects": len(te_proj_cnt),
            "n_shared_projects": len(shared_projs),
            "test_pairs_seen_projects": len(seen_test_pairs),
            "test_pairs_unseen_projects": len(unseen_test_pairs),
            "top_shared_projects": [
                {"project": p, "train_pairs": tr_proj_cnt[p], "test_pairs": te_proj_cnt[p]}
                for p in shared_projs[:15]
            ],
        },
        "manifest_integrity": manifest_info,
    }

    with open(os.path.join(a.out, "primevul_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=1)

    md_lines = [
        "# Plan v3 Phase 3.0 — PrimeVul Data & Leakage Audit",
        "",
        "## 1. Split Counts & Length Skew",
        "",
        "| Split | Clean Pairs | Changed / AddOnly | `cf > cv` (fix longer) | `cv > cf` (vul longer) | Shorter=Vul % | LenMatched<5% (n) | LenMatched Shorter=Vul % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sname in ("train_clean", "valid_clean", "test"):
        ps = pair_stats[sname]
        md_lines.append(
            f"| `{sname}` | {ps['n_pairs']} | {ps['n_changed']} / {ps['n_add_only']} | "
            f"{ps['n_cf_gt_cv']} | {ps['n_cv_gt_cf']} | {ps['shorter_is_vul_pct']}% | "
            f"{ps['length_matched_5pct_count']} ({ps['length_matched_5pct_cf_gt_cv']} vs {ps['length_matched_5pct_cv_gt_cf']}) | "
            f"**{ps['length_matched_5pct_shorter_is_vul_pct']}%** |"
        )
    md_lines.extend([
        "",
        "## 2. Token Length Distribution (`Qwen/Qwen2.5-Coder-7B-Instruct`)",
        "",
        "| Split | Functions | Median tok | p90 | p95 | Max | >2048 tok % | >3072 tok % | >4096 tok % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for sname in ("train_clean", "valid_clean", "test"):
        ts = token_stats[sname]
        md_lines.append(
            f"| `{sname}` | {ts['n_functions']} | {ts['median_tokens']} | {ts['p90_tokens']} | {ts['p95_tokens']} | "
            f"{ts['max_tokens']} | {ts['gt_2048_pct']}% | {ts['gt_3072_pct']}% | {ts['gt_4096_pct']}% |"
        )
    po = audit["project_overlap"]
    md_lines.extend([
        "",
        "## 3. Leakage & Project Overlap",
        f"- Raw overlap removed: train `{splits['stats']['train_filter']}`, valid `{splits['stats']['valid_filter']}`",
        f"- Extra train benign pool (clean): `{splits['stats']['extra_benign_pool']}` (sampled `{splits['stats']['extra_benign_sampled']}`)",
        f"- Projects: `{po['n_train_projects']}` train, `{po['n_test_projects']}` test, **`{po['n_shared_projects']}` shared** -> Test `seen_projects = {po['test_pairs_seen_projects']}`, `unseen_projects = {po['test_pairs_unseen_projects']}`",
        "",
        "## 4. Full-Test Manifest Integrity",
        f"- `{json.dumps(manifest_info)}`",
    ])
    md_text = "\n".join(md_lines) + "\n"
    with open(os.path.join(a.out, "primevul_audit.md"), "w", encoding="utf-8") as fh:
        fh.write(md_text)
    print(md_text, flush=True)

    tr_n = pair_stats["train_clean"]["n_pairs"]
    tr_cv = pair_stats["train_clean"]["n_cv_gt_cf"]
    if tr_n >= 1500 and tr_cv >= 150:
        print(f"GD3_0_GATE_PASS: train_clean={tr_n} pairs, train_cv_gt_cf={tr_cv}", flush=True)
    else:
        print(f"GD3_0_GATE_WARN: train_clean={tr_n} pairs, train_cv_gt_cf={tr_cv}", flush=True)


if __name__ == "__main__":
    main()
