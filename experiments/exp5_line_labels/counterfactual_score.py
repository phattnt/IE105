"""Counterfactual function scoring (idea A): instead of "how dangerous does this line look" (length-driven), ask
"how much does this line's score DROP when a guard is inserted right above it". The drop compares two versions of
the same function, so it cannot depend on function length; it measures "this statement is missing a check".

For each function: candidate lines = the K highest-scoring code lines of the base scoring; for each candidate a
guard `if (!IDENT) return;` (IDENT from the line, else `if (error) return;`) is inserted above it and the function
is re-scored. Function scores reported:
    base_max        max line logit (what train_lines.py does)
    cf_max_drop     max over candidates of (base logit - logit after guard)
    cf_sum_drop     sum of positive drops over candidates
    cf_top1_drop    drop of the single best base line
Evaluated on the PrimeVul test pairs (564) and, with --full N, on all vulnerable test functions plus N random benign
ones; every score raw and length-debiased (debias_length.evaluate_set).

    python counterfactual_score.py --ckpt <ckpt_best.pt> --model answerdotai/ModernBERT-base --max_len 2048 \
        --primevul ../../data/primevul [--k 8] [--full 2000] [--out ...]
"""
import argparse
import json
import os
import random
import time

import numpy as np

import train_lines as tl
from debias_length import evaluate_set, load_jsonl
from probe_guards import idents, indent_of


def load_model(a):
    import torch
    import transformers

    @classmethod
    def from_config_only(cls, name, *args, **kw):
        cfg = transformers.AutoConfig.from_pretrained(name)
        if "modernbert" in name.lower():
            cfg.reference_compile = False
        return cls.from_config(cfg, **kw)

    transformers.AutoModel.from_pretrained = from_config_only
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    tl.PAD_ID = tok.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tl.make_model(a.model)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    st = st["model"] if "model" in st else st
    model.load_state_dict({k: v.float() for k, v in st.items()})
    return model.to(device).eval(), tok, device


def score_texts(model, tok, texts, a, device):
    rs = [{"func": t, "n_lines": t.count("\n") + 1, "flaw_lines": []} for t in texts]
    return tl.score_rows(model, tok, rs, "slide", a, device)[0]


def is_code(line):
    s = line.strip()
    return bool(s) and s not in ("{", "}") and not s.startswith("//") and not s.startswith("/*") and not s.startswith("*")


def counterfactual(model, tok, funcs, a, device):
    """funcs: list of function texts -> list of dicts with base_max and cf_* scores."""
    base = score_texts(model, tok, funcs, a, device)
    variants, owner = [], []          # (func index, candidate line, inserted text)
    for i, (f, s) in enumerate(zip(funcs, base)):
        lines = f.split("\n")
        cand = [l for l in np.argsort(-s) if l < len(lines) and is_code(lines[l])][: a.k]
        for l in cand:
            ids_ = idents(lines[l])
            ind = indent_of(lines[l])
            g = f"{ind}if (!{ids_[0]}) return;" if ids_ else f"{ind}if (error) return;"
            variants.append("\n".join(lines[:l] + [g] + lines[l:]))
            owner.append((i, int(l)))
    tl.log(f"{len(funcs)} functions, {len(variants)} counterfactual variants")
    cf = score_texts(model, tok, variants, a, device)
    drops = [[] for _ in funcs]
    for (i, l), s in zip(owner, cf):
        drops[i].append((l, float(base[i][l] - s[l + 1])))
    out = []
    for i, s in enumerate(base):
        d = drops[i]
        best = int(np.argmax(s)) if len(s) else -1
        top1 = next((v for l, v in d if l == best), 0.0)
        out.append({"base_max": float(s.max()) if len(s) else -1e9,
                    "cf_max_drop": max((v for _, v in d), default=0.0),
                    "cf_sum_drop": float(sum(max(v, 0.0) for _, v in d)),
                    "cf_top1_drop": float(top1),
                    "cf_mix": float(s.max()) + max((v for _, v in d), default=0.0)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_len", type=int, required=True)
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--full", type=int, default=0, help="also score all vulnerable test functions + this many benign")
    ap.add_argument("--eval_tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model, tok, device = load_model(a)
    tl.log("loaded", a.ckpt, "on", device)

    paired = load_jsonl(os.path.join(a.primevul, "primevul_test_paired.jsonl"))
    pairs, prow = [], []
    for i in range(0, len(paired) - 1, 2):
        x, y = paired[i], paired[i + 1]
        if x["commit_id"] == y["commit_id"] and {x["target"], y["target"]} == {0, 1}:
            v, f = (x, y) if x["target"] == 1 else (y, x)
            pairs.append((v["idx"], f["idx"]))
            prow += [v, f]
    sets = {"pairs": prow}
    if a.full:
        test = load_jsonl(os.path.join(a.primevul, "primevul_test.jsonl"))
        vul = [r for r in test if r["target"] == 1]
        ben = random.Random(a.seed).sample([r for r in test if r["target"] == 0], a.full)
        sets["full_sample"] = vul + ben

    report = {}
    for name, rows in sets.items():
        t0 = time.time()
        sc = counterfactual(model, tok, [r["func"] for r in rows], a, device)
        tl.log(f"{name}: scored in {time.time() - t0:.0f}s")
        report[name] = {}
        print(f"\n### {name} (n={len(rows)}, lỗi={sum(r['target'] for r in rows)})")
        print("| điểm hàm | ROC thô | ROC trừ độ dài | AUPRC trừ độ dài | cặp thô | cặp trừ độ dài | ρ độ dài |")
        print("|---|---|---|---|---|---|---|")
        for key in ("base_max", "cf_max_drop", "cf_sum_drop", "cf_top1_drop", "cf_mix"):
            fn = {r["idx"]: s[key] for r, s in zip(rows, sc)}
            res, _ = evaluate_set(key, rows, fn, pairs)
            raw, deb = res["raw"], res["debias_same"]
            report[name][key] = {"raw": raw, "debias": deb}
            pa = lambda m: "-" if m.get("pair_acc") is None else f"{m['pair_acc']:.1f}"
            print(f"| {key} | {raw['roc_auc']:.3f} | {deb['roc_auc']:.3f} | {deb['auprc']:.3f} | {pa(raw)} | {pa(deb)} | {raw['spearman_len']:+.2f} |")
        lo = res["length_only"]
        print(f"| (chỉ độ dài, CV) | {lo['roc_auc']:.3f} | – | – | {pa(lo)} | – | {lo['spearman_len']:+.2f} |")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
