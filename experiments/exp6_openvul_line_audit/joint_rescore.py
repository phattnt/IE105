"""Joint Line-Sink Scoring and Confidence Calibration for LLM Audit Runs.

Evaluates how combining Encoder Line Hints (sinks) with LLM confidence thresholds (probability >= 85%)
eliminates OR-gate false alarms (P-V) and improves P-C / Net on PrimeVul test pairs across
both Clean (408 pairs) and All (564 pairs) test sets.
"""
import argparse
import json
import os
import random
import numpy as np


def load_primevul_meta(primevul_dir, leak_audit_path):
    """Loads pair classification: changed vs add_only, length matched, and clean vs leaked."""
    pv_lines_test = os.path.join(primevul_dir, "primevul_lines", "test.jsonl")
    if not os.path.exists(pv_lines_test):
        pv_lines_test = os.path.join(primevul_dir, "test.jsonl")
    
    changed_map = {}
    if os.path.exists(pv_lines_test):
        with open(pv_lines_test, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("target") == 1:
                    changed_map[r["idx"]] = len(r.get("flaw_lines", [])) > 0

    pv_paired = os.path.join(primevul_dir, "primevul_test_paired.jsonl")
    if not os.path.exists(pv_paired):
        pv_paired = os.path.join(primevul_dir, "primevul", "primevul_test_paired.jsonl")
    
    len_matched = set()
    if os.path.exists(pv_paired):
        pairs_dict = {}
        with open(pv_paired, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pairs_dict[r["idx"]] = r
        for idx, r in pairs_dict.items():
            if r.get("target") == 1:
                cands = [p for p in pairs_dict.values() if p.get("target") == 0 and p.get("commit_id") == r.get("commit_id")]
                if cands:
                    f_func = cands[0]["func"]
                    v_func = r["func"]
                    if abs(len(f_func) - len(v_func)) / max(len(v_func), 1) <= 0.05:
                        len_matched.add(idx)

    clean_vuls = set()
    leaked_vuls = set()
    if os.path.exists(leak_audit_path):
        with open(leak_audit_path, "r", encoding="utf-8") as f:
            leak_info = json.load(f)
            clean_vuls = {p[0] for p in leak_info.get("clean_pair_indices", [])}
            leaked_vuls = {p[0] for p in leak_info.get("leaked_pair_indices", [])}

    return changed_map, len_matched, clean_vuls, leaked_vuls


def eval_run(jsonl_path, min_prob=0, require_sink_in_hint=False):
    pairs = {}
    if not os.path.exists(jsonl_path):
        return pairs
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx = r["idx"]
            target = r.get("target")
            pair_idx = r.get("pair_idx")
            parsed = r.get("parsed") or {}
            vul = parsed.get("vulnerable", False)
            prob = parsed.get("probability", 50)
            if prob is None:
                prob = 50

            sinks = parsed.get("sinks") or []
            hint_lines = set(r.get("hint_lines") or [])

            is_vul = vul
            if min_prob > 0 and prob < min_prob:
                is_vul = False
            if require_sink_in_hint and is_vul:
                missing_sinks = [s.get("line") for s in sinks if str(s.get("verdict", "")).lower().startswith("miss")]
                if not any(l in hint_lines for l in missing_sinks):
                    is_vul = False

            pairs[idx] = {"target": target, "pair_idx": pair_idx, "is_vul": is_vul, "prob": prob}
    return pairs


def score_pairs(pairs, subset_filter, n_bootstrap=1000):
    matched_pairs = []
    for idx, d in pairs.items():
        if d["target"] == 1 and d["pair_idx"] in pairs and subset_filter(idx):
            v_vul = d["is_vul"]
            f_vul = pairs[d["pair_idx"]]["is_vul"]
            matched_pairs.append((v_vul, f_vul))

    n = len(matched_pairs)
    if n == 0:
        return {"n": 0}

    def calc_metrics(arr):
        pc = sum(1 for v, f in arr if v and not f)
        pv = sum(1 for v, f in arr if v and f)
        pb = sum(1 for v, f in arr if not v and not f)
        pr = sum(1 for v, f in arr if not v and f)
        total = len(arr)
        return {
            "P-C": pc / total * 100.0,
            "P-V": pv / total * 100.0,
            "P-B": pb / total * 100.0,
            "P-R": pr / total * 100.0,
            "Net": (pc - pr) / total * 100.0,
        }

    res = calc_metrics(matched_pairs)
    res["n"] = n

    rng = random.Random(0)
    boot_nets = []
    for _ in range(n_bootstrap):
        sample = [matched_pairs[rng.randint(0, n - 1)] for _ in range(n)]
        boot_nets.append(calc_metrics(sample)["Net"])

    res["Net_CI95"] = [float(np.percentile(boot_nets, 2.5)), float(np.percentile(boot_nets, 97.5))]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_dir", default="experiments/exp5_line_labels/results/llm_full_a100")
    ap.add_argument("--ctx_dir", default="experiments/exp5_line_labels/results/llm_ctx_test")
    ap.add_argument("--primevul_dir", default="data")
    ap.add_argument("--leak_audit", default="experiments/exp5_line_labels/results/leak_audit_openvul.json")
    ap.add_argument("--out", default="experiments/exp5_line_labels/results/joint_rescore_summary.json")
    ap.add_argument("--md_out", default="experiments/exp5_line_labels/results/joint_rescore_summary.md")
    args = ap.parse_args()

    changed_map, len_matched, clean_vuls, leaked_vuls = load_primevul_meta(args.primevul_dir, args.leak_audit)

    filters = {
        "ALL (564 pairs)": lambda i: True,
        "CLEAN (408 pairs)": lambda i: i in clean_vuls,
        "LEAKED (156 pairs)": lambda i: i in leaked_vuls,
        "CHANGED (357 pairs)": lambda i: changed_map.get(i, False),
        "ADD_ONLY (207 pairs)": lambda i: not changed_map.get(i, False),
        "LENGTH_MATCHED (<5%)": lambda i: i in len_matched,
    }

    arms = {
        "plain (Zero-shot)": eval_run(os.path.join(args.llm_dir, "plain.jsonl"), min_prob=50),
        "audit (Default t0)": eval_run(os.path.join(args.llm_dir, "audit.jsonl"), min_prob=50),
        "audit (Prob >= 85%)": eval_run(os.path.join(args.llm_dir, "audit.jsonl"), min_prob=85),
        "audit_hint (Default t0)": eval_run(os.path.join(args.llm_dir, "audit_hint.jsonl"), min_prob=50),
        "audit_hint (Prob >= 85%)": eval_run(os.path.join(args.llm_dir, "audit_hint.jsonl"), min_prob=85),
        "audit_hint (Prob >= 85% + Sink-in-Hint)": eval_run(os.path.join(args.llm_dir, "audit_hint.jsonl"), min_prob=85, require_sink_in_hint=True),
        "audit_rand (Default t0)": eval_run(os.path.join(args.llm_dir, "audit_rand.jsonl"), min_prob=50),
        "audit_ctx (Default t0)": eval_run(os.path.join(args.ctx_dir, "audit_ctx.jsonl"), min_prob=50),
        "audit_ctxfake (Default t0)": eval_run(os.path.join(args.ctx_dir, "audit_ctxfake.jsonl"), min_prob=50),
    }

    results = {}
    for arm_name, pairs_data in arms.items():
        if not pairs_data:
            continue
        results[arm_name] = {}
        for fname, ffunc in filters.items():
            results[arm_name][fname] = score_pairs(pairs_data, ffunc)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown generation
    md_lines = ["# Báo Cáo Phân Tích Toàn Diện: Lọc Kết Hợp Điểm Dòng & Ngưỡng Tự Tin (Joint Line-Sink Filter)", ""]
    md_lines.append("| Cấu hình / Nhánh thử nghiệm | Tập mẫu | P-C (Đúng cả hai) ↑ | P-V (Gắn cờ cả hai) ↓ | P-B (Bỏ qua cả hai) | P-R (Đảo ngược) | Net [CI 95%] ↑ |")
    md_lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|")

    for arm_name, arm_res in results.items():
        for fname, m in arm_res.items():
            if m.get("n", 0) == 0:
                continue
            ci_str = f"[{m['Net_CI95'][0]:.1f}; {m['Net_CI95'][1]:.1f}]"
            md_lines.append(f"| **{arm_name}** | {fname} | {m['P-C']:.1f}% | {m['P-V']:.1f}% | {m['P-B']:.1f}% | {m['P-R']:.1f}% | **{m['Net']:+.2f}%** {ci_str} |")

    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Saved results to {args.out} and {args.md_out}")


if __name__ == "__main__":
    main()
