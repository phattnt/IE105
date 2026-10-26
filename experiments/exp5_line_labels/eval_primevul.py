"""Score a line-level model trained on SecVulEval (train_lines.py) on PrimeVul, with the same protocol as TN1-TN3.

  function level  full primevul_test (25,911 functions, 695 vulnerable): AUPRC, ROC-AUC, Spearman with length
  pairs           primevul_test_paired (564 vulnerable/fixed pairs): ranking (ties = 0.5, non-tied)
  line level      vulnerable pair members, labels = lines the fix deletes/changes (difflib, as in TN1),
                  Top-1/5/10 and IFA by groups A/B/C (CodeBERT tokenizer, 512-token head)
The training script already drops SecVulEval functions identical to a PrimeVul test function.
"""
import argparse
import difflib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lines as tl  # noqa: E402

PRIMEVUL_FOLDER = "19iLaNDS0z99N8kB_jBRTmDLehwZBolMY"
COMMENT = ("//", "/*", "*", "*/")


def diff_flaw_lines(vul, fix):
    a, b = vul.split("\n"), fix.split("\n")
    out = []
    for op, i1, i2, _, _ in difflib.SequenceMatcher(None, [x.strip() for x in a], [x.strip() for x in b], autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            out += [i for i in range(i1, i2) if a[i].strip() and not a[i].strip().startswith(COMMENT)]
    return out


def diff_flaw_lines_anchored(vul, fix):
    """diff_flaw_lines with insertion-anchor support for add_only patches.
    When a patch only adds code, the omitted check is anchored to the adjacent line in vul."""
    a, b = vul.split("\n"), fix.split("\n")
    out, anchors = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [x.strip() for x in a], [x.strip() for x in b], autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            out += [i for i in range(i1, i2) if a[i].strip() and not a[i].strip().startswith(COMMENT)]
        elif op == "insert":
            cand = i1 - 1 if i1 > 0 else i1
            while cand >= 0 and (not a[cand].strip() or a[cand].strip().startswith(COMMENT)):
                cand -= 1
            if cand < 0:
                cand = i1
                while cand < len(a) and (not a[cand].strip() or a[cand].strip().startswith(COMMENT)):
                    cand += 1
            if 0 <= cand < len(a):
                anchors.append(cand)
    if not out and anchors:
        return sorted(list(set(anchors)))
    return sorted(list(set(out)))


def primevul_rows(primevul_dir, limit=None):
    """-> (pair_rows, test_rows) in train_lines row format; downloads PrimeVul if needed."""
    os.makedirs(primevul_dir, exist_ok=True)
    if not any(f.endswith(".jsonl") for f in os.listdir(primevul_dir)):
        subprocess.run(f"gdown -q --folder https://drive.google.com/drive/folders/{PRIMEVUL_FOLDER} -O {primevul_dir}", shell=True)
    rd = lambda n: [json.loads(l) for l in open(f"{primevul_dir}/primevul_{n}.jsonl")]
    test, paired = rd("test"), rd("test_paired")
    if limit:
        paired, test = paired[:2 * limit], test[:20 * limit]
    pair_rows = []
    for k in range(0, len(paired) - 1, 2):
        v, f = paired[k], paired[k + 1]
        if v["target"] == 0:
            v, f = f, v
        if v["commit_id"] != f["commit_id"] or {v["target"], f["target"]} != {0, 1}:
            continue
        iv, iff = f"v{k}", f"f{k}"
        pair_rows.append({"idx": iv, "target": 1, "func": v["func"], "n_lines": v["func"].count("\n") + 1,
                          "flaw_lines": diff_flaw_lines(v["func"], f["func"]), "pair_idx": iff})
        pair_rows.append({"idx": iff, "target": 0, "func": f["func"], "n_lines": f["func"].count("\n") + 1,
                          "flaw_lines": [], "pair_idx": iv})
    test_rows = [{"idx": r["idx"], "target": r["target"], "func": r["func"], "n_lines": r["func"].count("\n") + 1,
                  "flaw_lines": [], "pair_idx": None} for r in test]
    return pair_rows, test_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output dir of train_lines.py (results.json + ckpt_best.pt)")
    ap.add_argument("--primevul", default="/kaggle/tmp/primevul" if os.path.isdir("/kaggle") else "primevul")
    ap.add_argument("--eval_tokens", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N pairs / N*20 test functions")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    cfg = json.load(open(f"{a.run}/results.json"))["args"]
    os.makedirs(a.primevul, exist_ok=True)
    if not any(f.endswith(".jsonl") for f in os.listdir(a.primevul)):
        subprocess.run(f"gdown -q --folder https://drive.google.com/drive/folders/{PRIMEVUL_FOLDER} -O {a.primevul}", shell=True)
    rd = lambda n: [json.loads(l) for l in open(f"{a.primevul}/primevul_{n}.jsonl")]
    test, paired = rd("test"), rd("test_paired")
    if a.limit:
        paired, test = paired[:2 * a.limit], test[:20 * a.limit]

    pair_rows = []
    for k in range(0, len(paired) - 1, 2):
        v, f = paired[k], paired[k + 1]
        if v["target"] == 0:
            v, f = f, v
        if v["commit_id"] != f["commit_id"] or {v["target"], f["target"]} != {0, 1}:
            continue
        iv, iff = f"v{k}", f"f{k}"
        pair_rows.append({"idx": iv, "target": 1, "func": v["func"], "n_lines": v["func"].count("\n") + 1,
                          "flaw_lines": diff_flaw_lines(v["func"], f["func"]), "pair_idx": iff})
        pair_rows.append({"idx": iff, "target": 0, "func": f["func"], "n_lines": f["func"].count("\n") + 1,
                          "flaw_lines": [], "pair_idx": iv})
    test_rows = [{"idx": r["idx"], "target": r["target"], "func": r["func"], "n_lines": r["func"].count("\n") + 1,
                  "flaw_lines": [], "pair_idx": None} for r in test]
    tl.log(f"PrimeVul: {len(pair_rows) // 2} pairs, {len(test_rows)} test functions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tl.PAD_ID = tok.pad_token_id
    if cfg.get("bf16"):
        tl.AMP_DTYPE = torch.bfloat16
    model = tl.make_model(cfg["model"]).to(device)
    model.load_state_dict({k: v.float() for k, v in torch.load(f"{a.run}/ckpt_best.pt", map_location="cpu").items()})

    class A:  # score_rows reads these attributes
        max_len, eval_tokens = cfg["max_len"], a.eval_tokens
    res = {"trained_on": "SecVulEval (exp5)", "model": cfg["model"], "strategy": cfg["strategy"], "max_len": cfg["max_len"]}
    ls, fs = tl.score_rows(model, tok, pair_rows, cfg["strategy"], A, device)
    res["pairs_and_lines"] = tl.evaluate(pair_rows, ls, fs, *tl.group_info(pair_rows))
    tl.log("paired:", json.dumps(res["pairs_and_lines"]))
    ls, fs = tl.score_rows(model, tok, test_rows, cfg["strategy"], A, device)
    hv, nt = tl.group_info(test_rows)
    res["full_test"] = tl.evaluate(test_rows, ls, fs, hv, nt)["function"]
    tl.log("full test:", json.dumps(res["full_test"]))
    json.dump(res, open(f"{a.run}/primevul_results.json", "w"), indent=1)
    with open(f"{a.run}/primevul_test_scores.jsonl", "w") as fh:
        for r, f in zip(test_rows, fs):
            fh.write(json.dumps({"idx": r["idx"], "target": r["target"], "logit": round(f, 4)}) + "\n")
    tl.log("DONE primevul")


if __name__ == "__main__":
    main()
