# Exp5 — converged runs: plain line labels (A) vs fix-contrastive (B)

ModernBERT-base, 2048-token sliding windows, SecVulEval line labels, 4 epochs (linear LR to 0), single seed.
A = `results/slide_ModernBERT-base_2048_s0_conv` (best valid epoch 1), B = `results/slide_ModernBERT-base_2048_s0_contrast_conv` (best valid epoch 2).
Analysis: Claude first, then an independent review by Fable given only the raw files (no conclusions), then key claims re-checked by Claude.

## Headline numbers (SecVulEval test, 142 unseen projects)

| Metric | A | B | longest-line / length control | random |
|---|---|---|---|---|
| Line top-1 | 27.5 | 25.7 | 20.2 | 11.4 |
| Line top-10 | 68.8 | 62.9 | 62.0 | 48.7 |
| Pairs, all (979) | 58.0 | 62.3 | 72.3 (length logistic) | 50 |
| Pairs, length-matched (\|Δchars\| ≤ 20, 344) | 50.7 | 62.6 | – | 50 |
| same_text_flaw_lines (line-weighted, as reported) | 53.5 | 60.3 | – | 50 |
| same_text, function-averaged (362 fns) | 60.7 | 62.9 | – | 50 |
| Spearman(function score, length) | −0.75 | −0.40 | – | – |

PrimeVul (not seen in training): paired lines top-10 A 59.4 / B 49.9; pairs 62.4 / 62.3; full test (25,911 fns) ROC-AUC 0.248 / 0.359, AUPRC 0.017 / 0.020.

## Findings that survived cross-checking

1. **same_text_flaw_lines is not usable as reported.** 2,236 of 3,737 lines come from one hermes function (idx 16033) that was rewritten wholesale, so every line is labelled. Function-averaged, A and B are close (60.7 vs 62.9; Fable's project bootstrap [−4.8, 10.3]). The metric is also half of the epoch-selection score, so selection is partly driven by such functions. → Fix: average per function and exclude functions with flaw density > 0.5.
2. **Line localization is mostly a "long line" effect.** Line logits correlate ~0.6 with line length within a function. A beats the longest-line-first control by ~7 pp (top-1/top-10), B by ~1–5 pp. Position is not a shortcut. Real signal exists but is smaller than the raw numbers suggest.
3. **A's function score is a length detector.** On length-matched pairs A is at chance (50.7); per stratum A gets 66 (fix longer) / 41 (fix shorter). On PrimeVul, where vulnerable functions tend to be longer, this inverts the function-level ranking (ROC 0.25).
4. **B trades localization for length-independent discrimination.**
   - Top-10 −5.9 [−8.5, −3.4] (project bootstrap; McNemar top-5 p = 0.013).
   - Length-matched pairs +11.9, project bootstrap [+3.0, +20.2] (Claude); B vs chance p ≈ 2e-6 (Fable).
   - This is the one contrastive result we can currently defend.
5. **Not robust:** pairs over all data (+4.2 [−1.2, 8.4]) and same_text. Pilot vs converged runs of the same recipe differ by 4–7 pp, about the size of the A–B gaps, so everything rests on a single seed.
6. **No leakage found.** There are no exact-text duplicates and no shared commits between train and test, and the project sets are disjoint.

## Where the two analyses disagreed
- Claude initially read same_text 60.3 vs 53.5 as the strongest evidence for B. **Wrong**: one function dominates it (finding 1).
- Claude called PrimeVul top-10 59.4 "best so far". The number is correct, but no longest-line control has been run on PrimeVul yet, so the gain over length is unknown. Fable estimates it at about 5 pp.

## Next steps
1. Run length controls on PrimeVul: −log(tokens) AUC on the full test, plus length-logistic and longest-line on pairs and lines. CPU, under 30 min.
2. Fix same_text (function-averaged, drop rewrites) and add the length-matched pair metric to `evaluate()` and to the selection score.
3. Run one more seed each for A and B (L4, about 3.5 h). If B's length-matched advantage (~+12) reproduces, it is the paper's main claim. If not, drop all A-vs-B claims.

Fable's scripts: `review/` (paths inside point to the session scratchpad copies of the data). Bootstrap: `bootstrap_ci.py`.
