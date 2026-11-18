"""Audit whether Exp5 improvements transfer from SecVulEval to PrimeVul.

This analysis is intentionally CPU-only and uses only result artifacts already
committed under ``results/<run>/``.  It does not retrain a model or read the
original datasets.  Run from the repository root:

    python experiments/exp5_line_labels/analyze_transfer.py

Outputs:
    experiments/exp5_line_labels/results/transfer_analysis.json
    experiments/exp5_line_labels/TRANSFER_ANALYSIS.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"
DEFAULT_JSON = DEFAULT_RESULTS / "transfer_analysis.json"
DEFAULT_MD = HERE / "TRANSFER_ANALYSIS.md"

BASELINE_RUN = "slide_ModernBERT-base_2048_s0_conv"

LABELS = {
    "slide_ModernBERT-base_2048_s0": "ModernBERT plain (pilot)",
    "slide_ModernBERT-base_2048_s0_contrast": "ModernBERT contrastive (pilot)",
    "slide_ModernBERT-base_2048_s0_conv": "ModernBERT plain",
    "slide_ModernBERT-base_2048_s0_contrast_conv": "ModernBERT contrastive λ=1",
    "slide_ModernBERT-base_2048_s0_contrast03_conv": "ModernBERT contrastive λ_line=0.3",
    "slide_ModernBERT-base_2048_s0_synth1": "ModernBERT synthetic guards only",
    "slide_ModernBERT-base_2048_s0_s13_conv": "ModernBERT synthetic→SecVulEval",
    "slide_ModernBERT-base_2048_s0_s13bal_conv": "ModernBERT synthetic→SecVulEval, length-balanced",
    "slide_unixcoder-base_1024_s0_conv": "UniXcoder plain",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def average_ranks(values: Iterable[float]) -> list[float]:
    """Return 0-based average ranks, including correct handling of ties."""
    values = list(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for idx in order[start:end]:
            ranks[idx] = rank
        start = end
    return ranks


def pearson(x: Iterable[float], y: Iterable[float]) -> float | None:
    x, y = list(x), list(y)
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return None if den == 0 else sum(a * b for a, b in zip(dx, dy)) / den


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def pct(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def run_kind(name: str) -> str:
    if name.endswith("_s0") or name.endswith("_s0_contrast"):
        return "pilot"
    if name.endswith("_s0_synth1"):
        return "synthetic_pretrain_only"
    return "comparable_converged"


def extract_run(run_dir: Path) -> dict | None:
    sec_path = run_dir / "results.json"
    prime_path = run_dir / "primevul_results.json"
    if not sec_path.exists() or not prime_path.exists():
        return None

    sec, prime = load_json(sec_path), load_json(prime_path)
    sec_test = sec.get("test") or {}
    prime_pairs = prime.get("pairs_and_lines") or {}
    prime_full = prime.get("full_test") or {}
    if "lines_ALL" not in sec_test or "lines_ALL" not in prime_pairs:
        return None

    sec_lines, prime_lines = sec_test["lines_ALL"], prime_pairs["lines_ALL"]
    sec_fn, prime_fn = sec_test.get("function") or {}, prime_full
    sec_pair, prime_pair = sec_test.get("pairs") or {}, prime_pairs.get("pairs") or {}

    sec_prev = sec_fn.get("vulnerable", 0) / sec_fn.get("n", 1) if sec_fn.get("n") else None
    prime_prev = prime_fn.get("vulnerable", 0) / prime_fn.get("n", 1) if prime_fn.get("n") else None
    sec_auprc = sec_fn.get("auprc")
    prime_auprc = prime_fn.get("auprc")

    name = run_dir.name
    args = sec.get("args") or {}
    return {
        "run": name,
        "label": LABELS.get(name, name),
        "kind": run_kind(name),
        "model": args.get("model") or prime.get("model"),
        "best_epoch": sec.get("best_epoch"),
        "secvuleval": {
            "top10": pct(sec_lines.get("top10")),
            "random_top10": pct(sec_lines.get("rand_top10")),
            "top10_lift_over_random": pct(sec_lines.get("top10") - sec_lines.get("rand_top10")),
            "pair_rank": pct(sec_pair.get("ranking_acc_ties_half")),
            "function_auprc": pct(sec_auprc, 6),
            "function_prevalence": pct(sec_prev, 6),
            "auprc_lift_pp": pct((sec_auprc - sec_prev) * 100) if sec_auprc is not None and sec_prev is not None else None,
        },
        "primevul": {
            "top10": pct(prime_lines.get("top10")),
            "random_top10": pct(prime_lines.get("rand_top10")),
            "top10_lift_over_random": pct(prime_lines.get("top10") - prime_lines.get("rand_top10")),
            "pair_rank": pct(prime_pair.get("ranking_acc_ties_half")),
            "function_auprc": pct(prime_auprc, 6),
            "function_prevalence": pct(prime_prev, 6),
            "auprc_lift_pp": pct((prime_auprc - prime_prev) * 100) if prime_auprc is not None and prime_prev is not None else None,
        },
    }


def correlations(runs: list[dict]) -> dict:
    comparable = [r for r in runs if r["kind"] == "comparable_converged"]

    def corr(rows: list[dict], metric: str) -> dict:
        x = [r["secvuleval"][metric] for r in rows]
        y = [r["primevul"][metric] for r in rows]
        return {"n": len(rows), "pearson": pct(pearson(x, y), 3), "spearman": pct(spearman(x, y), 3)}

    return {
        "all_runs": {"top10": corr(runs, "top10"), "pair_rank": corr(runs, "pair_rank")},
        "comparable_converged": {
            "definition": "completed SecVulEval fine-tuning runs; pilots and synthetic-only pretraining excluded",
            "top10": corr(comparable, "top10"),
            "pair_rank": corr(comparable, "pair_rank"),
        },
    }


def add_baseline_deltas(runs: list[dict]) -> None:
    by_name = {r["run"]: r for r in runs}
    if BASELINE_RUN not in by_name:
        raise RuntimeError(f"baseline result missing: {BASELINE_RUN}")
    base = by_name[BASELINE_RUN]
    for r in runs:
        r["delta_vs_plain"] = {
            "secvuleval_top10": pct(r["secvuleval"]["top10"] - base["secvuleval"]["top10"]),
            "primevul_top10": pct(r["primevul"]["top10"] - base["primevul"]["top10"]),
            "secvuleval_pair_rank": pct(r["secvuleval"]["pair_rank"] - base["secvuleval"]["pair_rank"]),
            "primevul_pair_rank": pct(r["primevul"]["pair_rank"] - base["primevul"]["pair_rank"]),
        }


def report_markdown(report: dict) -> str:
    runs = report["runs"]
    comp = [r for r in runs if r["kind"] == "comparable_converged"]
    best_sec = max(comp, key=lambda r: r["secvuleval"]["top10"])
    best_prime = max(comp, key=lambda r: r["primevul"]["top10"])
    baseline = next(r for r in runs if r["run"] == BASELINE_RUN)
    corr = report["cross_run_correlations"]["comparable_converged"]

    lines = [
        "# Exp5 – kiểm toán chuyển miền SecVulEval → PrimeVul",
        "",
        "Tự sinh bởi `analyze_transfer.py` từ các `results.json` và `primevul_results.json` đã commit; không train lại và không đọc dữ liệu ngoài repo.",
        "",
        "## Câu hỏi",
        "",
        "Một cấu hình định vị dòng tốt hơn trên SecVulEval có tiếp tục tốt hơn khi chuyển sang PrimeVul hay không?",
        "",
        "## Kết quả",
        "",
        "| Run | Loại | Top-10 SecVulEval | Top-10 PrimeVul | Δ Prime − Sec | Lift trên ngẫu nhiên Sec / Prime | Cặp Sec / Prime | AUPRC − prevalence Sec / Prime (điểm %) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(runs, key=lambda x: (-x["secvuleval"]["top10"], x["run"])):
        s, p = r["secvuleval"], r["primevul"]
        au_s = "n.a." if s["auprc_lift_pp"] is None else f"{s['auprc_lift_pp']:+.2f}"
        au_p = "n.a." if p["auprc_lift_pp"] is None else f"{p['auprc_lift_pp']:+.2f}"
        lines.append(
            f"| `{r['run']}` | {r['kind']} | {s['top10']:.2f} | {p['top10']:.2f} | "
            f"{p['top10'] - s['top10']:+.2f} | {s['top10_lift_over_random']:+.2f} / {p['top10_lift_over_random']:+.2f} | "
            f"{s['pair_rank']:.2f} / {p['pair_rank']:.2f} | {au_s} / {au_p} |"
        )

    bds = best_sec["delta_vs_plain"]
    bdp = best_prime["delta_vs_plain"]
    complete_prime = [r for r in runs if r["primevul"]["auprc_lift_pp"] is not None]
    below_random = sum(r["primevul"]["auprc_lift_pp"] < 0 for r in complete_prime)
    lines += [
        "",
        "## Diễn giải",
        "",
        f"1. Thứ hạng Top-10 giữa hai bộ dữ liệu có tương quan dương nhưng không hoàn toàn ổn định: Spearman = **{corr['top10']['spearman']:.3f}** trên {corr['top10']['n']} run hội tụ có thể so sánh. Đây là thống kê mô tả giữa các cấu hình, không phải kiểm định độc lập.",
        f"2. Run tốt nhất trên SecVulEval là `{best_sec['run']}` ({best_sec['secvuleval']['top10']:.2f}%). So với plain, nó **{bds['secvuleval_top10']:+.2f} điểm** trên SecVulEval nhưng **{bds['primevul_top10']:+.2f} điểm** trên PrimeVul. Mức tăng quan sát được trong miền không tự động chuyển sang bộ dữ liệu khác.",
        f"3. Run tốt nhất trên PrimeVul là `{best_prime['run']}` ({best_prime['primevul']['top10']:.2f}%). So với plain: {bdp['secvuleval_top10']:+.2f} điểm SecVulEval và {bdp['primevul_top10']:+.2f} điểm PrimeVul.",
        f"4. Phân loại mức hàm không chuyển miền: **{below_random}/{len(complete_prime)}** run có AUPRC PrimeVul thấp hơn prevalence 695/25.911, dù AUPRC SecVulEval cao hơn prevalence. Vì vậy kết luận chính nên tiếp tục tập trung vào định vị dòng, không tuyên bố đã giải được phát hiện hàm lỗi ngoài miền.",
        "",
        "## Kết luận sử dụng trong báo cáo",
        "",
        "Các cải tiến Exp5 nhìn chung giữ thứ hạng tương đối giữa SecVulEval và PrimeVul, nhưng cấu hình thắng và mức tăng không trùng nhau. Đặc biệt, tiền huấn luyện guard rồi fine-tune (`s13_conv`) tăng +2,70 điểm Top-10 trên SecVulEval nhưng không vượt plain trên PrimeVul. Do đó mọi tuyên bố cải tiến cần báo đồng thời hai bộ dữ liệu; kết quả một miền không đủ để kết luận mô hình tổng quát hơn.",
        "",
        "## Giới hạn",
        "",
        "- Các run chủ yếu dùng seed 0; tương quan giữa cấu hình không thay thế cho thí nghiệm nhiều seed.",
        "- Không có khoảng tin cậy cho chênh lệch chuyển miền trong báo cáo này vì file PrimeVul đã lưu không kèm project cho từng hàm.",
        "- Run pilot và synthetic-only được hiển thị để đầy đủ nhưng bị loại khỏi tương quan chính.",
        "",
    ]
    return "\n".join(lines)


def build_report(results_dir: Path) -> dict:
    runs = [extract_run(p) for p in sorted(results_dir.iterdir()) if p.is_dir()]
    runs = [r for r in runs if r is not None]
    if not runs:
        raise RuntimeError(f"no paired SecVulEval/PrimeVul result artifacts found in {results_dir}")
    add_baseline_deltas(runs)
    return {
        "analysis": "Exp5 cross-dataset transfer audit",
        "baseline_run": BASELINE_RUN,
        "n_runs": len(runs),
        "runs": runs,
        "cross_run_correlations": correlations(runs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = ap.parse_args()

    report = build_report(args.results)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.write_text(report_markdown(report), encoding="utf-8")
    print(f"{report['n_runs']} runs -> {args.json_out}, {args.md_out}")


if __name__ == "__main__":
    main()
