"""Phase 0 CPU Audit on Colab (--gpu none):
1. OpenVul-GRPO leakage & prompt audit across Leopo1d HuggingFace models/datasets vs PrimeVul test pairs.
2. NMS (r=3) + Hunk-tolerant (+-2) rescore of eval_pv1_best (changed + add_only) and 6 SecVulEval line models.
3. Verify fixed pair_protocol.py (--clean_json, tie-preserving rank_acc_debiased) on existing LLM scores.
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_diff_labels import diverse_topk_order  # noqa: E402
from eval_primevul import diff_flaw_lines, diff_flaw_lines_anchored  # noqa: E402
import train_lines as tl  # noqa: E402

SHA_RE = re.compile(r"\b[0-9a-f]{40}\b", re.I)
CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\s*\n(.*?)```", re.S)


def norm_ws(s):
    return re.sub(r"\s+", "", s or "")


def md5_norm(s):
    return hashlib.md5(norm_ws(s).encode("utf-8", "replace")).hexdigest()


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def extract_commits_and_hashes(val, commits_out, hashes_out):
    """Recursively scan nested dict/list/str row values for 40-hex commit SHAs and code hashes."""
    if isinstance(val, str):
        for m in SHA_RE.findall(val):
            commits_out.add(m.lower())
        # Code block inside prompt/message
        blocks = CODE_BLOCK_RE.findall(val)
        if blocks:
            for b in blocks:
                if len(b) > 40:
                    hashes_out.add(md5_norm(b))
        elif len(val) > 60 and ("\n" in val or ";" in val or "{" in val):
            hashes_out.add(md5_norm(val))
    elif isinstance(val, dict):
        for k, v in val.items():
            if k in ("commit_id", "commit", "commit_sha", "sha") and isinstance(v, str) and v.strip():
                commits_out.add(v.strip().lower())
            extract_commits_and_hashes(v, commits_out, hashes_out)
    elif isinstance(val, (list, tuple)):
        for it in val:
            extract_commits_and_hashes(it, commits_out, hashes_out)


def audit_openvul(primevul_dir, sft_leak_json, eval_pv1_jsonl, out_dir):
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    models = sorted(m.id for m in api.list_models(author="Leopo1d"))
    datasets_list = sorted(d.id for d in api.list_datasets(author="Leopo1d"))
    print("Leopo1d models:", models, flush=True)
    print("Leopo1d datasets:", datasets_list, flush=True)

    target_model = "Leopo1d/OpenVul-Qwen3-4B-GRPO"
    prompt_info = {
        "leopo1d_models": models,
        "leopo1d_datasets": datasets_list,
        "target_model": target_model,
        "model_exists": target_model in models,
        "dataset_samples": {},
    }
    if target_model in models:
        for fname in ("README.md", "tokenizer_config.json", "config.json"):
            try:
                p = hf_hub_download(target_model, fname)
                txt = open(p, encoding="utf-8", errors="replace").read()
                if fname == "README.md":
                    prompt_info["readme_excerpt"] = txt[:6000]
                elif fname == "tokenizer_config.json":
                    tc = json.loads(txt)
                    prompt_info["chat_template"] = tc.get("chat_template", "")
                    prompt_info["tokenizer_class"] = tc.get("tokenizer_class", "")
                elif fname == "config.json":
                    cfg = json.loads(txt)
                    prompt_info["model_config"] = {
                        k: cfg.get(k) for k in ("architectures", "model_type", "max_position_embeddings", "vocab_size")
                    }
            except Exception as e:
                prompt_info[f"{fname}_error"] = repr(e)

    # Load PrimeVul test pairs
    paired = load_jsonl(os.path.join(primevul_dir, "primevul_test_paired.jsonl"))
    pairs = []
    for k in range(0, len(paired) - 1, 2):
        x, y = paired[k], paired[k + 1]
        if x["commit_id"] == y["commit_id"] and {x["target"], y["target"]} == {0, 1}:
            v, f = (x, y) if x["target"] == 1 else (y, x)
            pairs.append({
                "v_idx": v["idx"],
                "f_idx": f["idx"],
                "commit_id": str(v["commit_id"]).strip().lower(),
                "v_hash": md5_norm(v["func"]),
                "f_hash": md5_norm(f["func"]),
            })
    pv_commits = {p["commit_id"] for p in pairs}
    pv_hashes = {p["v_hash"] for p in pairs} | {p["f_hash"] for p in pairs}

    sft_leak = json.load(open(sft_leak_json, encoding="utf-8"))
    sft_leaked_v = {p[0] for p in sft_leak.get("leaked_pair_indices", [])}

    per_ds = {}
    all_train_commits, all_train_hashes = set(), set()
    rl_train_commits, rl_train_hashes = set(), set()

    for ds_id in datasets_list:
        try:
            cfgs = get_dataset_config_names(ds_id)
        except Exception:
            cfgs = ["default"]
        if not cfgs:
            cfgs = [None]
        for cfg in cfgs:
            try:
                ds_dict = load_dataset(ds_id, cfg) if cfg else load_dataset(ds_id)
            except Exception as e:
                per_ds[f"{ds_id}:{cfg}"] = {"error": repr(e)}
                continue
            for split_name, ds in ds_dict.items():
                key = f"{ds_id}::{cfg}::{split_name}"
                ds_commits, ds_hashes = set(), set()
                n_rows = len(ds)
                samples = []
                for idx_row, row in enumerate(ds):
                    if idx_row < 2:
                        # Truncate long fields for readability
                        samples.append({
                            k: (v[:2500] if isinstance(v, str) else v)
                            for k, v in row.items()
                        })
                    extract_commits_and_hashes(row, ds_commits, ds_hashes)
                c_hits = ds_commits & pv_commits
                h_hits = ds_hashes & pv_hashes
                is_eval_only = "test" in split_name.lower() or "eval" in ds_id.lower()
                is_rl_ds = any(tag in ds_id.lower() for tag in ("rl", "grpo", "dpo", "orpo", "pref", "ppo"))
                if not is_eval_only:
                    all_train_commits |= ds_commits
                    all_train_hashes |= ds_hashes
                    if is_rl_ds:
                        rl_train_commits |= ds_commits
                        rl_train_hashes |= ds_hashes
                per_ds[key] = {
                    "n_rows": n_rows,
                    "columns": list(ds.column_names),
                    "is_eval_only_split": is_eval_only,
                    "is_rl_dataset": is_rl_ds,
                    "commit_hits_in_pv_test": len(c_hits),
                    "code_hash_hits_in_pv_test": len(h_hits),
                }
                if samples:
                    prompt_info["dataset_samples"][key] = samples
                print(f"  {key}: rows={n_rows} eval_only={is_eval_only} rl={is_rl_ds} commit_hits={len(c_hits)} hash_hits={len(h_hits)}", flush=True)

    with open(os.path.join(out_dir, "openvul_grpo_prompt_info.json"), "w", encoding="utf-8") as fh:
        json.dump(prompt_info, fh, indent=1, ensure_ascii=False)

    clean_pairs, leaked_pairs = [], []
    rl_leaked_v, union_leaked_v = set(), set()
    for p in pairs:
        in_rl = (p["commit_id"] in rl_train_commits) or (p["v_hash"] in rl_train_hashes) or (p["f_hash"] in rl_train_hashes)
        in_any_ds = (p["commit_id"] in all_train_commits) or (p["v_hash"] in all_train_hashes) or (p["f_hash"] in all_train_hashes)
        in_sft = p["v_idx"] in sft_leaked_v
        if in_rl:
            rl_leaked_v.add(p["v_idx"])
        if in_sft or in_any_ds:
            union_leaked_v.add(p["v_idx"])
            leaked_pairs.append([p["v_idx"], p["f_idx"]])
        else:
            clean_pairs.append([p["v_idx"], p["f_idx"]])

    leak_report = {
        "total_pairs": len(pairs),
        "sft_leaked_count": len(sft_leaked_v),
        "rl_leaked_count": len(rl_leaked_v),
        "union_leaked_count": len(union_leaked_v),
        "clean_pairs_count": len(clean_pairs),
        "leak_percentage": round(100.0 * len(union_leaked_v) / max(len(pairs), 1), 2),
        "clean_pair_indices": clean_pairs,
        "leaked_pair_indices": leaked_pairs,
        "per_dataset_overlap": per_ds,
    }
    with open(os.path.join(out_dir, "leak_audit_openvul_grpo.json"), "w", encoding="utf-8") as fh:
        json.dump(leak_report, fh, indent=1)
    print(f"Leak audit: total={len(pairs)}, sft={len(sft_leaked_v)}, rl={len(rl_leaked_v)}, union={len(union_leaked_v)}, clean={len(clean_pairs)}", flush=True)

    # Build stratified 50-pair CLEAN smoke subset
    pv1_rows = {r["idx"]: r for r in load_jsonl(eval_pv1_jsonl)}
    clean_changed = [pair for pair in clean_pairs if bool(pv1_rows.get(pair[0], {}).get("flaw_lines"))]
    clean_add_only = [pair for pair in clean_pairs if not bool(pv1_rows.get(pair[0], {}).get("flaw_lines"))]
    target_total = min(50, len(clean_pairs))
    n_chg = round(target_total * len(clean_changed) / max(len(clean_pairs), 1))
    n_add = target_total - n_chg
    rng = random.Random(42)
    pick_chg = rng.sample(clean_changed, min(n_chg, len(clean_changed)))
    pick_add = rng.sample(clean_add_only, min(n_add, len(clean_add_only)))
    picked = pick_chg + pick_add
    rng.shuffle(picked)
    smoke = {
        "seed": 42,
        "n_pairs": len(picked),
        "n_changed": len(pick_chg),
        "n_add_only": len(pick_add),
        "clean_changed_pool": len(clean_changed),
        "clean_add_only_pool": len(clean_add_only),
        "pair_indices": picked,
        "changed_pair_indices": pick_chg,
        "add_only_pair_indices": pick_add,
    }
    with open(os.path.join(out_dir, "smoke50_clean_pairs.json"), "w", encoding="utf-8") as fh:
        json.dump(smoke, fh, indent=1)
    print(f"Smoke50: {len(picked)} pairs (changed={len(pick_chg)}, add_only={len(pick_add)})", flush=True)


def rank_subset(rows, head_vis, n_toks, use_nms=False, nms_radius=3, control=None):
    """Compute exact & hunk+-2 Top-1/3/5/10 and median/mean IFA across ALL and A/B/C."""
    recs = []
    for r, hv, nt in zip(rows, head_vis, n_toks):
        fl = [i for i in (r.get("flaw_lines") or []) if i < r["n_lines"]]
        if not fl:
            continue
        func = r["func"]
        lines = func.split("\n")
        n_lines = min(r["n_lines"], len(lines))
        fl_set = [i for i in fl if i < n_lines]
        if not fl_set:
            continue
        if control == "longest":
            sc = np.array([len(lines[i].strip()) for i in range(n_lines)], dtype=float)
            order = sorted(range(n_lines), key=lambda i: (-sc[i], i))
        elif control == "anchor":
            sc = np.array([3.0 * bool(tl.API.search(lines[i])) + 2.0 * bool(tl.ANCHOR2.search(lines[i])) for i in range(n_lines)], dtype=float)
            order = sorted(range(n_lines), key=lambda i: (-sc[i], i))
        else:
            sc = list(r["line_scores"][:n_lines])
            if use_nms:
                order = diverse_topk_order(func, sc, radius=nms_radius, mask_trivial=True)
            else:
                order = sorted(range(n_lines), key=lambda i: (-sc[i], i))

        rank_of = {li: idx for idx, li in enumerate(order)}
        first_exact = min(rank_of[i] for i in fl_set)
        # First rank where any picked line is within +-2 of any flaw line
        first_hunk2 = next((idx for idx, p in enumerate(order) if any(abs(p - f) <= 2 for f in fl_set)), n_lines)
        grp = "A" if nt <= 510 else ("B" if any(hv[i] for i in fl_set if i < len(hv)) else "C")
        recs.append({
            "group": grp,
            "ifa_exact": first_exact,
            "ifa_hunk2": first_hunk2,
            **{f"exact_top{k}": first_exact < k for k in (1, 3, 5, 10)},
            **{f"hunk2_top{k}": first_hunk2 < k for k in (1, 3, 5, 10)},
        })

    out = {}
    for g in ("ALL", "A", "B", "C"):
        rs = [x for x in recs if g == "ALL" or x["group"] == g]
        if not rs:
            continue
        d = {
            "n": len(rs),
            "median_ifa_exact": float(np.median([x["ifa_exact"] for x in rs])),
            "median_ifa_hunk2": float(np.median([x["ifa_hunk2"] for x in rs])),
            "mean_ifa_exact": round(float(np.mean([x["ifa_exact"] for x in rs])), 2),
        }
        for k in (1, 3, 5, 10):
            d[f"exact_top{k}"] = round(float(np.mean([x[f"exact_top{k}"] for x in rs]) * 100), 2)
            d[f"hunk2_top{k}"] = round(float(np.mean([x[f"hunk2_top{k}"] for x in rs]) * 100), 2)
        out[g] = d
    return out


def rescore_lines(primevul_dir, eval_pv1_jsonl, secvul_runs_dir, out_dir):
    report = {"primevul": {}, "secvuleval": {}}

    # 1. PrimeVul eval_pv1_best (changed, add_only_after, add_only_both, all_after)
    pv1 = load_jsonl(eval_pv1_jsonl)
    paired = load_jsonl(os.path.join(primevul_dir, "primevul_test_paired.jsonl"))
    fix_by_commit = {}
    vul_by_commit = {}
    for k in range(0, len(paired) - 1, 2):
        x, y = paired[k], paired[k + 1]
        if x["commit_id"] == y["commit_id"] and {x["target"], y["target"]} == {0, 1}:
            v, f = (x, y) if x["target"] == 1 else (y, x)
            fix_by_commit[v["idx"]] = f["func"]
            vul_by_commit[v["idx"]] = v["func"]

    vul_rows = []
    for r in pv1:
        if r["target"] == 1:
            rr = dict(r)
            rr["n_lines"] = rr["func"].count("\n") + 1
            vul_rows.append(rr)
    hv_pv, nt_pv = tl.group_info(vul_rows)

    changed_idx = [i for i, r in enumerate(vul_rows) if r.get("flaw_lines")]
    add_only_idx = [i for i, r in enumerate(vul_rows) if not r.get("flaw_lines")]

    def subset_eval(indices, custom_flaws=None):
        sub_r = []
        sub_hv = [hv_pv[i] for i in indices]
        sub_nt = [nt_pv[i] for i in indices]
        for i in indices:
            rr = dict(vul_rows[i])
            if custom_flaws is not None:
                rr["flaw_lines"] = custom_flaws[rr["idx"]]
            sub_r.append(rr)
        return {
            "raw": rank_subset(sub_r, sub_hv, sub_nt, use_nms=False),
            "nms_r3": rank_subset(sub_r, sub_hv, sub_nt, use_nms=True, nms_radius=3),
            "longest": rank_subset(sub_r, sub_hv, sub_nt, control="longest"),
            "anchor": rank_subset(sub_r, sub_hv, sub_nt, control="anchor"),
        }

    flaws_after = {
        vul_rows[i]["idx"]: diff_flaw_lines_anchored(vul_rows[i]["func"], fix_by_commit[vul_rows[i]["idx"]], mode="after")
        for i in range(len(vul_rows)) if vul_rows[i]["idx"] in fix_by_commit
    }
    flaws_both = {
        vul_rows[i]["idx"]: diff_flaw_lines_anchored(vul_rows[i]["func"], fix_by_commit[vul_rows[i]["idx"]], mode="both")
        for i in range(len(vul_rows)) if vul_rows[i]["idx"] in fix_by_commit
    }
    report["primevul"]["changed_357"] = subset_eval(changed_idx)
    report["primevul"]["add_only_after_207"] = subset_eval(add_only_idx, flaws_after)
    report["primevul"]["add_only_both_207"] = subset_eval(add_only_idx, flaws_both)
    report["primevul"]["all_anchored_after_564"] = subset_eval(list(range(len(vul_rows))), flaws_after)

    # 2. SecVulEval saved runs
    if secvul_runs_dir and os.path.isdir(secvul_runs_dir):
        b2_path = os.path.join(secvul_runs_dir, "slide_ModernBERT-base_2048_s0_base2", "test_scores.jsonl")
        func_map = {r["idx"]: r["func"] for r in load_jsonl(b2_path)} if os.path.exists(b2_path) else {}
        base_vul = None
        hv_sv = nt_sv = None
        for run_name in sorted(os.listdir(secvul_runs_dir)):
            tp = os.path.join(secvul_runs_dir, run_name, "test_scores.jsonl")
            if not os.path.exists(tp):
                continue
            rows = load_jsonl(tp)
            v_rows = []
            for r in rows:
                if r["target"] == 1 and r.get("flaw_lines"):
                    rr = dict(r)
                    if "func" not in rr:
                        rr["func"] = func_map[rr["idx"]]
                    rr["n_lines"] = rr["func"].count("\n") + 1
                    v_rows.append(rr)
            if base_vul is None:
                hv_sv, nt_sv = tl.group_info(v_rows)
                base_vul = [r["idx"] for r in v_rows]
                report["secvuleval"]["controls"] = {
                    "longest": rank_subset(v_rows, hv_sv, nt_sv, control="longest"),
                    "anchor": rank_subset(v_rows, hv_sv, nt_sv, control="anchor"),
                }
            # Ensure matching order for shared group_info
            by_id = {r["idx"]: r for r in v_rows}
            ordered_v = [by_id[i] for i in base_vul if i in by_id]
            report["secvuleval"][run_name] = {
                "raw": rank_subset(ordered_v, hv_sv, nt_sv, use_nms=False),
                "nms_r3": rank_subset(ordered_v, hv_sv, nt_sv, use_nms=True, nms_radius=3),
            }

    with open(os.path.join(out_dir, "nms_hunk_rescore.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    lines_md = ["# Phase 0 CPU Rescore: Raw vs NMS (r=3) and Exact vs Hunk(±2)\n"]
    lines_md.append("## PrimeVul `eval_pv1_best`\n")
    lines_md.append("| Subset | Method | n | Exact T1 / T3 / T5 / T10 | Hunk±2 T1 / T3 / T5 / T10 | Med IFA (Ex / H2) |")
    lines_md.append("|---|---|---:|---|---|---|")
    for sub_name, sub_obj in report["primevul"].items():
        for mname in ("raw", "nms_r3", "longest", "anchor"):
            a = sub_obj[mname].get("ALL")
            if not a:
                continue
            lines_md.append(
                f"| {sub_name} | {mname} | {a['n']} | "
                f"{a['exact_top1']:.1f} / {a['exact_top3']:.1f} / {a['exact_top5']:.1f} / {a['exact_top10']:.1f} | "
                f"{a['hunk2_top1']:.1f} / {a['hunk2_top3']:.1f} / {a['hunk2_top5']:.1f} / {a['hunk2_top10']:.1f} | "
                f"{a['median_ifa_exact']:.0f} / {a['median_ifa_hunk2']:.0f} |"
            )

    lines_md.append("\n## SecVulEval Test (1,037 functions)\n")
    lines_md.append("| Run | Mode | Exact T1 / T5 / T10 | Hunk±2 T1 / T5 / T10 | Group C Hunk±2 T5 / T10 |")
    lines_md.append("|---|---|---|---|---|")
    for rname, robj in report["secvuleval"].items():
        for mname in ("raw", "nms_r3", "longest", "anchor"):
            if mname not in robj:
                continue
            a = robj[mname]["ALL"]
            c = robj[mname].get("C", {})
            lines_md.append(
                f"| {rname} | {mname} | {a['exact_top1']:.1f} / {a['exact_top5']:.1f} / {a['exact_top10']:.1f} | "
                f"{a['hunk2_top1']:.1f} / {a['hunk2_top5']:.1f} / {a['hunk2_top10']:.1f} | "
                f"{c.get('hunk2_top5', 0):.1f} / {c.get('hunk2_top10', 0):.1f} |"
            )
    with open(os.path.join(out_dir, "nms_hunk_rescore.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines_md) + "\n")
    print("\n".join(lines_md), flush=True)


def verify_pair_protocol(primevul_dir, llm_scores_root, clean_json, out_dir):
    if not llm_scores_root or not os.path.isdir(llm_scores_root):
        return
    runs = []
    for sub in ("openvul_test", "llm_full_a100"):
        sd = os.path.join(llm_scores_root, sub, "scores")
        if os.path.isdir(sd):
            for arm in sorted(os.listdir(sd)):
                if os.path.exists(os.path.join(sd, arm, "primevul_test_scores.jsonl")):
                    runs.append(os.path.join(sd, arm))
    if not runs:
        return
    out_p = os.path.join(out_dir, "pair_protocol_check.json")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pair_protocol.py"),
        "--primevul", primevul_dir,
        "--clean_json", clean_json,
        "--runs", *runs,
        "--out", out_p,
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--sft_leak", required=True)
    ap.add_argument("--eval_pv1", required=True)
    ap.add_argument("--secvul_runs", default="")
    ap.add_argument("--llm_scores", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    audit_openvul(a.primevul, a.sft_leak, a.eval_pv1, a.out)
    rescore_lines(a.primevul, a.eval_pv1, a.secvul_runs, a.out)
    verify_pair_protocol(
        a.primevul, a.llm_scores, os.path.join(a.out, "leak_audit_openvul_grpo.json"), a.out
    )
    print("GD0_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
