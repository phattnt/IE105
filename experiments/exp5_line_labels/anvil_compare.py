# Score the line model on ANVIL's own PrimeVul subset and compute ANVIL's line metrics, for a like-for-like comparison
# with ANVIL Table IV (Top-1/3/5, MFR, N-MFR, line ROC-AUC).
#   1) score: the ANVIL functions located in PrimeVul valid/test (anvil_map3.json) are scored with a checkpoint
#      (train_lines.score_rows, slide 2048). Test-pair functions are also compared with the scores computed on Colab,
#      as a check that this machine reproduces them.
#   2) metrics, on ANVIL's labels (their vulnerable lines) and ANVIL's ranked lines (their sampled + vulnerable lines):
#      Top-k = at least one vulnerable line among the k highest-ranked lines; MFR = mean rank of the first vulnerable
#      line; N-MFR two ways (MFR / mean ranked lines, as ANVIL's numbers imply, and mean of rank / ranked lines);
#      ROC-AUC over all ranked lines of these functions (lines as independent samples).
# usage: python anvil_compare.py score --map anvil_map3.json --ckpt ckpt_best.pt --out results/anvil_subset
#        python anvil_compare.py metrics --map anvil_map3.json --out results/anvil_subset
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "primevul")


def load_funcs():
    f = {}
    for s in ("valid", "test"):
        for l in open(f"{PV}/primevul_{s}.jsonl", encoding="utf-8"):
            r = json.loads(l); f[r["idx"]] = r["func"]
    return f


def score(a):
    import torch
    from transformers import AutoTokenizer
    import train_lines as tl
    m = json.load(open(a.map))
    funcs = load_funcs()
    rows = [{"idx": x["idx"], "target": 1, "func": funcs[x["idx"]], "n_lines": funcs[x["idx"]].count("\n") + 1, "flaw_lines": []} for x in m]
    tok = AutoTokenizer.from_pretrained(a.model); tl.PAD_ID = tok.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tl.make_model(a.model).to(device)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=True); st = st["model"] if "model" in st else st
    model.load_state_dict({k: v.float() for k, v in st.items()}); model.eval()
    print(f"scoring {len(rows)} functions on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)
    args = argparse.Namespace(max_len=a.max_len, eval_tokens=a.eval_tokens)
    ls, fs = tl.score_rows(model, tok, rows, "slide", args, device)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/line_scores.jsonl", "w") as fh:
        for r, s, f in zip(rows, ls, fs):
            fh.write(json.dumps({"idx": r["idx"], "logit": round(float(f), 4), "line_scores": [round(float(x), 4) for x in s]}) + "\n")
    # reproduction check against the Colab scores of the test-pair functions
    ref = {}
    for l in open(a.colab_scores, encoding="utf-8"):
        r = json.loads(l); ref[r["idx"]] = r["line_scores"]
    from scipy.stats import spearmanr
    rho, top1 = [], []
    for r, s in zip(rows, ls):
        if r["idx"] in ref and len(ref[r["idx"]]) == len(s) and len(s) > 2:
            c = np.asarray(ref[r["idx"]]); rho.append(spearmanr(c, s)[0]); top1.append(int(np.argmax(c) == np.argmax(s)))
    if rho:
        print(f"check vs Colab on {len(rho)} test functions: median Spearman {np.median(rho):.4f}, min {np.min(rho):.4f}, same top-1 line {np.mean(top1) * 100:.1f}%", flush=True)
    print("SCORE_DONE", flush=True)


def metrics(a):
    from sklearn.metrics import roc_auc_score
    m = json.load(open(a.map))
    sc = {json.loads(l)["idx"]: json.loads(l)["line_scores"] for l in open(f"{a.out}/line_scores.jsonl")}
    rep = {}
    for subset in ("all", "test", "valid"):
        recs, y, s = [], [], []
        for x in m:
            if subset != "all" and x["split"] != subset:
                continue
            ls = np.asarray(sc[x["idx"]]); ranked = [i for i in x["anvil_scored_lines"] if i < len(ls)]
            vul = set(x["anvil_vul_lines"]) & set(ranked)
            if not vul:
                continue
            order = sorted(ranked, key=lambda i: (-ls[i], i))
            first = min(order.index(i) for i in vul) + 1
            recs.append({"first": first, "n": len(ranked)})
            y += [int(i in vul) for i in ranked]; s += [ls[i] for i in ranked]
        if not recs:
            continue
        f = np.array([r["first"] for r in recs]); n = np.array([r["n"] for r in recs])
        rep[subset] = {"functions": len(recs), "vulnerable_lines": int(sum(y)), "ranked_lines": len(y),
                       **{f"top{k}": float((f <= k).mean() * 100) for k in (1, 3, 5, 10)},
                       "mfr": float(f.mean()), "n_mfr_anvil_style": float(f.mean() / n.mean()), "n_mfr_per_function": float((f / n).mean()),
                       "roc_auc": float(roc_auc_score(y, s) * 100)}
    print("| subset | functions | vul lines / ranked lines | Top-1 | Top-3 | Top-5 | Top-10 | MFR | N-MFR (MFR/mean lines) | N-MFR (mean rank/lines) | ROC-AUC |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in rep.items():
        print(f"| {k} | {r['functions']} | {r['vulnerable_lines']} / {r['ranked_lines']} | {r['top1']:.1f} | {r['top3']:.1f} | {r['top5']:.1f} | {r['top10']:.1f} | "
              f"{r['mfr']:.1f} | {r['n_mfr_anvil_style']:.2f} | {r['n_mfr_per_function']:.2f} | {r['roc_auc']:.1f} |")
    print("| ANVIL (paper, 273 functions) | 273 | 924 / 34,744 | 11.7 | 31.3 | 41.4 | – | 25.0 | 0.18 | – | 61.8 |")
    json.dump(rep, open(f"{a.out}/anvil_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "metrics"])
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--eval_tokens", type=int, default=4096)
    ap.add_argument("--colab_scores", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "eval_pv1_best", "test_scores.jsonl"))
    a = ap.parse_args()
    score(a) if a.cmd == "score" else metrics(a)
