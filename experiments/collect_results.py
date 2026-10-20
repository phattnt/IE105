"""Collect every experiment's results.json into experiments/results.csv and experiments/RESULTS.md.

One row per (model, training, input strategy). Run from the repo root after downloading new results:
    python experiments/collect_results.py
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["run", "experiment", "backbone", "training", "strategy", "P-C", "P-V", "P-B", "P-R", "rank_acc_pct",
        "F1_test", "F1_valid_thr", "AUPRC", "ROC_AUC", "VDS_oracle", "VDS_valid", "flag_rate", "top10_ALL", "top10_B", "top10_C",
        "IFA_C_median", "flaw_visible_pct", "source"]
G = {"ALL": "ALL (vulnerable members with >=1 flaw line)", "B": "B: truncated, >=1 flaw line visible", "C": "C: truncated, no flaw line visible"}


def r2(x):
    return None if x is None else round(float(x), 3)


def row(run, exp, backbone, training, strategy, pairs, loc, ft, src, rank=None):
    loc = loc or {}
    ft = ft or {}
    return {"run": run, "experiment": exp, "backbone": backbone, "training": training, "strategy": strategy,
            "P-C": pairs.get("P-C"), "P-V": pairs.get("P-V"), "P-B": pairs.get("P-B"), "P-R": pairs.get("P-R"),
            "rank_acc_pct": r2(rank if rank is not None else pairs.get("ranking_accuracy_pct")),
            "F1_test": r2(ft.get("f1")), "F1_valid_thr": r2((ft.get("at_valid_f1_threshold") or {}).get("f1")),
            "AUPRC": r2(ft.get("auprc")), "ROC_AUC": r2(ft.get("roc_auc")),
            "VDS_oracle": r2((ft.get("vd_s_oracle_threshold") or {}).get("vd_s")), "VDS_valid": r2((ft.get("vd_s_valid_threshold") or {}).get("vd_s")),
            "flag_rate": r2(ft.get("flag_rate")),
            "top10_ALL": r2((loc.get(G["ALL"]) or {}).get("top10_acc_pct")), "top10_B": r2((loc.get(G["B"]) or {}).get("top10_acc_pct")),
            "top10_C": r2((loc.get(G["C"]) or {}).get("top10_acc_pct")), "IFA_C_median": r2((loc.get(G["C"]) or {}).get("median_ifa")),
            "flaw_visible_pct": r2(loc.get("flaw_lines_visible_under_this_strategy_pct", (loc.get("label_coverage") or {}).get("flaw_lines_visible_in_512_pct"))),
            "source": os.path.relpath(src, HERE)}


rows = []
p = f"{HERE}/exp1_linevul_on_primevul/results/results.json"
if os.path.exists(p):
    r = json.load(open(p))
    rows.append(row("exp1", "TN1 LineVul gốc trên PrimeVul", "codebert-base", "Big-Vul (checkpoint gốc)", "head", r["pairs"], r["localization"], r["full_test"], p))
p = f"{HERE}/exp2_context_strategies/results/results.json"
if os.path.exists(p):
    r = json.load(open(p))
    rc = json.load(open(f"{HERE}/exp2_context_strategies/results/ranking_check.json")) if os.path.exists(f"{HERE}/exp2_context_strategies/results/ranking_check.json") else {}
    for name, v in r["strategies"].items():
        rows.append(row(f"exp2/{name}", "TN2 chiến lược ngữ cảnh (chỉ suy luận)", "codebert-base", "Big-Vul (checkpoint gốc)", name,
                        v["pairs"], v["localization"], v["full_test"], p, rank=(rc.get(name) or {}).get("ranking_accuracy_ties_pct")))
for p in sorted(glob.glob(f"{HERE}/exp3_pilot_train/results/*/results.json")):
    r = json.load(open(p))
    name = os.path.basename(os.path.dirname(p))
    backbone = r.get("backbone", "codebert-base" if name != "length" else "—")
    rows.append(row(f"exp3/{name}", "TN3 pilot train trên PrimeVul (tập con 1:5, 3 epoch)", backbone.split("/")[-1],
                    "PrimeVul tập con" if name != "length" else "hồi quy logistic trên độ dài", r.get("strategy", name),
                    r["pairs"], r.get("localization"), r["full_test"], p))

with open(f"{HERE}/results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
with open(f"{HERE}/RESULTS.md", "w") as fh:
    fh.write("# Bảng kết quả tổng hợp (tự sinh bởi `collect_results.py`, không sửa tay)\n\n")
    fh.write("Cặp = 564 cặp test của bản PrimeVul gốc; test = 25.911 hàm; Top-10 theo nhóm A/B/C định nghĩa bởi cắt đầu 512 token; `length` = đối chứng chỉ nhìn độ dài.\n\n")
    show = ["run", "backbone", "training", "strategy", "P-C", "P-V", "P-B", "P-R", "rank_acc_pct", "F1_test", "AUPRC", "ROC_AUC", "VDS_oracle", "top10_ALL", "top10_B", "top10_C", "flaw_visible_pct"]
    fh.write("| " + " | ".join(show) + " |\n|" + "---|" * len(show) + "\n")
    for x in rows:
        fh.write("| " + " | ".join("" if x[c] is None else str(x[c]) for c in show) + " |\n")
print(f"{len(rows)} rows -> results.csv, RESULTS.md")
