# Handoff — state as of 2026-10-22

This is the IE105 course project: locating vulnerable lines in C/C++ functions, with LineVul as the baseline. The user wants explanations in plain, simple Vietnamese.

## Where things are
- **Results so far:** `experiments/RESULTS.md` and `experiments/results.csv` cover exp1–exp3. Exp5 is not in them yet, because `collect_results.py` does not read exp5.
- **Exp5 (current line of work):** `experiments/exp5_line_labels/`
  - `train_lines.py` is the line model. Main flags: `--strategy slide/head/controls`, `--lam_line`, `--lam_fn`, `--pair_batches`, `--patience`, `--bf16`.
  - `eval_primevul.py` runs the PrimeVul eval. `bootstrap_ci.py` computes project-level confidence intervals. `prepare_secvuleval.py` prepares the data.
  - `ANALYSIS.md` holds the conclusions for the converged runs, together with Fable's independent review. **Read this first.**
  - `results/`:
    - `slide_ModernBERT-base_2048_s0_conv`: run A, plain line labels.
    - `…_contrast_conv`: run B, fix-contrastive.
    - Two pilot runs, `controls.json`, and `bootstrap_conv_vs_contrast_conv.json`.
  - `review/` contains Fable's analysis scripts. Their data paths point to an old scratchpad, so edit them before rerunning.
  - `colab/` holds the Colab Pro L4 runner: keepalive, auto-relaunch, chunked checkpoint upload. See its README.
- **Notion:** the "IE105" hub page holds four things:
  - Bảng kết quả thí nghiệm (results table): views Số liệu chính, Theo nhóm and Đầy đủ.
  - Công việc (tasks).
  - Ghi chú (notes). The latest is "Kết quả train hội tụ: nhãn dòng vs học tương phản".
  - Ma trận thí nghiệm (experiment matrix).

## Key facts and constraints
- Model inputs are function code only. Never use commit messages, CVE text, or post-fix code at test time. The fixed version may be used only as a training signal.
- Splits are project-level. Train functions that overlap the PrimeVul test set (679) are dropped.
- Length is a strong shortcut on both datasets, and its direction is opposite on each: in SecVulEval the fixed version is longer, while in PrimeVul the vulnerable function is longer. Always report length controls and length-matched metrics.
- `same_text_flaw_lines` as currently computed is dominated by one hermes function (idx 16033). Use a function-averaged version that drops functions with more than 50% of lines labelled.

## Open next steps (from ANALYSIS.md)
1. Run length and longest-line controls on PrimeVul. CPU only, no training.
2. Fix `same_text` (function-averaged) and add a length-matched pair metric. Add both to `evaluate()` and to the epoch-selection score.
3. Train one extra seed each for A and B on an L4, about 3.5 h. Check whether B's length-matched pair advantage of about +12 pp reproduces.
4. Extend `collect_results.py` to cover exp5.

## Not in the repo (lost with the old container)
- Model checkpoints (300 MB each). Rerun training if they are needed. Per-line test scores are already saved in `test_scores.jsonl`.
- The prepared SecVulEval jsonl. Rebuild it with `prepare_secvuleval.py`; `colab/setup.py` shows the exact command.

## Accounts and compute
- Colab Pro had 184.56 compute units left. The L4 costs 1.54 units per hour. The Colab CLI token was deleted, so log in again.
- One stray Colab CPU runtime may still be open (created from the browser). The user should terminate it in Colab → Runtime → Manage sessions.
- Kaggle is no longer used. The user should expire the old Kaggle API token.
