"""Plots for one train_lines.py run, written to <run>/plots/*.png (matplotlib, Agg; safe to call during training).

    train_loss.png      training loss / learning rate / throughput per logged step (train_log.jsonl)
    valid_lines.png     valid Top-1/5/10 and median IFA at every evaluation point, random-ranking baselines dashed
    valid_function.png  valid pair accuracy raw / length-debiased / length-matched / top-3-mean, net = P-C - P-R at a
                        50% flag rate, function ROC-AUC raw / debiased, line BCE (convergence)
    test_summary.png    (after the run) test Top-1/5/10 by group A/B/C vs random, pair accuracies

    python plot_curve.py <run dir> [<run dir> ...]
"""
import json
import os
import sys


def _load(run):
    cur = json.load(open(f"{run}/curve.json")) if os.path.exists(f"{run}/curve.json") else None
    steps = [json.loads(l) for l in open(f"{run}/train_log.jsonl")] if os.path.exists(f"{run}/train_log.jsonl") else []
    res = json.load(open(f"{run}/results.json")) if os.path.exists(f"{run}/results.json") else None
    return cur, steps, res


def plot_run(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cur, steps, res = _load(run)
    pd = f"{run}/plots"
    os.makedirs(pd, exist_ok=True)
    name = os.path.basename(os.path.normpath(run))
    if steps:
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        x = [s["epoch"] for s in steps]
        ax[0].plot(x, [s["loss"] for s in steps], lw=1); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch"); ax[0].set_yscale("log")
        ax[1].plot(x, [s["lr"] for s in steps], lw=1); ax[1].set_title("learning rate"); ax[1].set_xlabel("epoch")
        ax[2].plot(x, [s["items_per_s"] for s in steps], lw=1); ax[2].set_title("items / s"); ax[2].set_xlabel("epoch")
        ax2 = ax[2].twinx(); ax2.plot(x, [s["mem_gb"] for s in steps], lw=1, color="gray"); ax2.set_ylabel("max mem GB")
        fig.suptitle(name); fig.tight_layout(); fig.savefig(f"{pd}/train_loss.png", dpi=110); plt.close(fig)
    if cur and cur["curve"]:
        C = cur["curve"]
        ep = [c["epoch"] for c in C]
        la = [c["valid"].get("lines_ALL", {}) for c in C]
        best = cur.get("best", {}).get("epoch")
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for k, col in (("top1", "C0"), ("top5", "C1"), ("top10", "C2")):
            ax[0].plot(ep, [l.get(k) for l in la], "o-", color=col, label=f"Top-{k[3:]}")
            ax[0].plot(ep, [l.get(f"rand_{k}") for l in la], "--", color=col, lw=0.8)
        if best is not None:
            ax[0].axvline(best, color="k", lw=0.8, ls=":")
        ax[0].set_title("valid lines: Top-k (dashed = random)"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("%"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(ep, [l.get("median_ifa") for l in la], "o-", color="C3"); ax[1].set_title("valid lines: median IFA (lower = better)")
        ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.3)
        fig.suptitle(name); fig.tight_layout(); fig.savefig(f"{pd}/valid_lines.png", dpi=110); plt.close(fig)
        ex = [c["valid"].get("extra", {}) for c in C]
        px = [e.get("pairs", {}) for e in ex]
        fx = [e.get("function", {}) for e in ex]
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        for k, lab in (("acc", "raw"), ("acc_debiased", "length-debiased"), ("acc_length_matched", "length-matched <5%"), ("acc_top3mean", "top-3 mean agg.")):
            ax[0].plot(ep, [p.get(k) for p in px], "o-", label=lab)
        ax[0].axhline(50, color="k", lw=0.8, ls=":"); ax[0].set_title("valid pairs: vulnerable ranked above fixed (%)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[1].plot(ep, [p.get("net_at50") for p in px], "o-", color="C4", label="pairs net = P-C - P-R @50% flagged")
        ax[1].plot(ep, [100 * f["roc_auc"] if "roc_auc" in f else None for f in fx], "s-", color="C5", label="fn ROC-AUC x100 raw")
        ax[1].plot(ep, [100 * f["roc_auc_debiased"] if "roc_auc_debiased" in f else None for f in fx], "s--", color="C5", label="fn ROC-AUC x100 debiased")
        ax[1].axhline(0, color="k", lw=0.8, ls=":"); ax[1].axhline(50, color="gray", lw=0.8, ls=":")
        ax[1].set_title("valid function level"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        ax[2].plot(ep, [e.get("line_bce") for e in ex], "o-", color="C6"); ax[2].set_title("valid line BCE (convergence)"); ax[2].grid(alpha=0.3)
        for a in ax:
            a.set_xlabel("epoch")
            if best is not None:
                a.axvline(best, color="k", lw=0.8, ls=":")
        fig.suptitle(name); fig.tight_layout(); fig.savefig(f"{pd}/valid_function.png", dpi=110); plt.close(fig)
    if res:
        t = res["test"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        groups = [g for g in ("ALL", "A", "B", "C") if f"lines_{g}" in t]
        w, xs = 0.25, range(len(groups))
        for i, k in enumerate(("top1", "top5", "top10")):
            ax[0].bar([x + (i - 1) * w for x in xs], [t[f"lines_{g}"][k] for g in groups], w, label=f"Top-{k[3:]}", color=f"C{i}")
            ax[0].scatter([x + (i - 1) * w for x in xs], [t[f"lines_{g}"][f"rand_{k}"] for g in groups], color="k", s=12, zorder=3)
        ax[0].set_xticks(list(xs)); ax[0].set_xticklabels([f"{g} (n={t[f'lines_{g}']['n']})" for g in groups])
        ax[0].set_title("test lines: Top-k by group (dots = random)"); ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
        px = t.get("extra", {}).get("pairs", {})
        keys = [("acc", "raw"), ("acc_debiased", "debiased"), ("acc_length_matched", "len-matched"), ("acc_top3mean", "top-3 mean"), ("length_rule_longer_is_fixed", "length rule")]
        ax[1].bar([l for _, l in keys], [px.get(k) or 0 for k, _ in keys], color="C1"); ax[1].axhline(50, color="k", lw=0.8, ls=":")
        ax[1].set_title(f"test pairs (n={px.get('n')}): vulnerable above fixed (%)"); ax[1].grid(alpha=0.3, axis="y")
        fig.suptitle(f"{name}  best epoch {res.get('best_epoch')}"); fig.tight_layout(); fig.savefig(f"{pd}/test_summary.png", dpi=110); plt.close(fig)
    return pd


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(plot_run(r))
