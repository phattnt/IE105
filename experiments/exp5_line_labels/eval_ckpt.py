"""Score one checkpoint of train_lines.py without training: the prepared test split (pairs + flaw lines, same metrics
and files as the end of train_lines.py, so pair_protocol.py can read test_scores.jsonl) and, with --primevul, the full
PrimeVul test set (function level only, as eval_primevul.py). Used to score ckpt_best.pt while the run is still training.

  python eval_ckpt.py --ckpt <run>/ckpt_best.pt --data <prepared dir> --out <dir> --model answerdotai/ModernBERT-base \
      --max_len 2048 --strategy slide --bf16 [--eval_tokens 16384] [--primevul /content/exp5/primevul]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lines as tl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--strategy", choices=["head", "slide"], default="slide")
    ap.add_argument("--eval_tokens", type=int, default=16384)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max_pos_weight", type=float, default=20.0)
    ap.add_argument("--primevul", default=None, help="PrimeVul folder (primevul_test.jsonl): also score the full test set")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tl.PAD_ID = tok.pad_token_id
    if a.bf16:
        tl.AMP_DTYPE = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tl.make_model(a.model).to(device)
    st = torch.load(a.ckpt, map_location="cpu")
    st = st["model"] if "model" in st else st
    model.load_state_dict({k: v.float() for k, v in st.items()})
    model.eval()
    tl.log(f"loaded {a.ckpt} | {a.model} {a.strategy} {a.max_len} | device {device}")

    test = tl.read_split(a.data, "test", a.limit)
    pos = sum(len(set(r["flaw_lines"])) for r in test); neg = sum(r["n_lines"] for r in test) - pos
    pos_weight = min(a.max_pos_weight, neg / max(pos, 1))
    t = time.time()
    ls, fs = tl.score_rows(model, tok, test, a.strategy, a, device)
    hv, nt = tl.group_info(test)
    res = {"ckpt": a.ckpt, "data": a.data, "args": vars(a), "test": tl.evaluate(test, ls, fs, hv, nt)}
    res["test"]["extra"] = tl.valid_extra(test, ls, fs, pos_weight)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
    with open(f"{a.out}/test_scores.jsonl", "w") as fh:
        for r, s, f in zip(test, ls, fs):
            fh.write(json.dumps({"idx": r["idx"], "target": r["target"], "pair_idx": r["pair_idx"], "chars": len(r["func"]), "func": r["func"],
                                 "flaw_lines": r["flaw_lines"], "logit": round(float(f), 4), "line_scores": [round(float(x), 4) for x in s]}) + "\n")
    tl.log(f"test ({len(test)} functions, {time.time() - t:.0f}s):", json.dumps(res["test"]))
    tl.log("DONE test")

    if a.primevul:
        rows = [json.loads(l) for l in open(f"{a.primevul}/primevul_test.jsonl")]
        if a.limit:
            rows = rows[:20 * a.limit]
        full = [{"idx": r["idx"], "target": r["target"], "func": r["func"], "n_lines": r["func"].count("\n") + 1, "flaw_lines": [], "pair_idx": None}
                for r in rows]
        t = time.time()
        ls, fs = tl.score_rows(model, tok, full, a.strategy, a, device)
        hv, nt = tl.group_info(full)
        res_full = tl.evaluate(full, ls, fs, hv, nt)["function"]
        json.dump({"ckpt": a.ckpt, "full_test": res_full}, open(f"{a.out}/primevul_results.json", "w"), indent=1)
        with open(f"{a.out}/primevul_test_scores.jsonl", "w") as fh:
            for r, f in zip(full, fs):
                fh.write(json.dumps({"idx": r["idx"], "target": r["target"], "chars": len(r["func"]), "logit": round(float(f), 4)}) + "\n")
        tl.log(f"full test ({len(full)} functions, {time.time() - t:.0f}s):", json.dumps(res_full))
        tl.log("DONE primevul")


if __name__ == "__main__":
    main()
