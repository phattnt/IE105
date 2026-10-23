"""Score the PrimeVul test pairs with a train_lines.py checkpoint and SAVE the per-line scores (eval_primevul.py keeps
only the aggregate Top-k), so that pair_protocol.py can compute Top-1/3/5, MFR and line ROC-AUC on CPU.

    python score_pairs_lines.py --ckpt <ckpt_best.pt> --model answerdotai/ModernBERT-base --max_len 2048 \
        --primevul ../../data/primevul --out <run>/primevul_pair_lines.jsonl
"""
import argparse
import json

import train_lines as tl
from counterfactual_score import load_model
from eval_primevul import primevul_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_len", type=int, required=True)
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--eval_tokens", type=int, default=16384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok, device = load_model(a)
    rows, _ = primevul_rows(a.primevul)
    tl.log(f"{len(rows) // 2} pairs")
    ls, fs = tl.score_rows(model, tok, rows, "slide", a, device)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r, s, f in zip(rows, ls, fs):
            fh.write(json.dumps({"idx": r["idx"], "pair_idx": r["pair_idx"], "target": r["target"], "chars": len(r["func"]),
                                 "func": r["func"], "flaw_lines": r["flaw_lines"], "logit": round(f, 4),
                                 "line_scores": [round(float(x), 4) for x in s]}) + "\n")
    tl.log("wrote", a.out)


if __name__ == "__main__":
    main()
