# Tier 1 Frozen LLM Probe & Anti-Shortcut Report

- **Model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (`max_len=2048`, layers `[13, 19, 23, 27]`)
- **Unique functions extracted:** `15066` (truncated: `1979` = `13.14%`), Peak VRAM `15.53 GB`, Time `3418.1s`, Head equiv max diff `0.03125`
- **Clean splits:** train `4347` pairs (dropped commit `6`, hash `1`), valid `562` pairs, test `564` pairs (`seen_projects=455`, `unseen_projects=109`), extra train benigns `2000`

## 1. 5-Gate Anti-Shortcut Pair Matrix (`primevul_test_paired`, n=564)

| Method | Best Config | All Net@t0 / Net@med | All Rank / Debias / Ties | Bal Rank | LenMatched5% Rank | **PatchRemoves (n=65) Rank** | Seen / Unseen Rank | Changed / AddOnly Rank | Spearman(log chars) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `length_only` | `{"features": ["log_chars", "` | +2.1% / +1.4% | 76.4% / 53.2% / 2.5% | 52.6% | 66.8% | **20.0%** | 75.5% / 80.3% | 72.3% / 83.6% | -0.968 |
| `tfidf_word` | `{"C": 1.0, "valid_bal_rank":` | +5.7% / +3.2% | 73.7% / 69.8% / 4.1% | 65.0% | 67.4% | **53.1%** | 73.1% / 76.2% | 69.6% / 80.7% | -0.398 |
| `zero_shot_yesno` | `{"threshold_note": "log P(Ye` | +0.0% / +1.4% | 51.3% / 61.8% / 9.8% | 55.1% | 53.6% | **60.0%** | 50.9% / 53.2% | 52.1% / 50.0% | +0.520 |
| `probe_std` | `mean/L13/C=0.001` | +10.1% / +10.6% | 77.5% / 76.1% / 5.7% | 66.1% | 70.6% | **50.8%** | 76.7% / 80.7% | 73.0% / 85.3% | -0.273 |
| `probe_pair_margin` | `mean/L19/C=0.01` | +5.8% / +6.0% | 83.0% / 78.4% / 5.7% | 69.5% | 77.2% | **50.8%** | 83.7% / 79.8% | 77.5% / 92.5% | -0.684 |
| `probe_len_balanced` | `mean/L13/C=0.01` | +6.0% / +6.7% | 77.8% / 75.7% / 5.7% | 71.7% | 71.5% | **63.1%** | 77.8% / 78.0% | 73.8% / 84.8% | -0.226 |
| `probe_len_orth` | `mean/L13/C=0.01` | +6.7% / +6.7% | 77.2% / 78.8% / 0.2% | 71.9% | 71.1% | **64.6%** | 76.8% / 78.9% | 74.7% / 81.6% | -0.129 |
| `probe_mixed_benign` | `mean/L13/C=0.01` | +1.4% / +1.8% | 60.3% / 65.2% / 5.7% | 62.4% | 59.6% | **64.6%** | 59.3% / 64.2% | 60.4% / 60.1% | +0.469 |
| `mlp_len_orth` | `mean/L13/C=-` | +5.0% / +5.0% | 72.2% / 75.3% / 0.2% | 71.0% | 67.2% | **69.2%** | 72.6% / 70.6% | 70.7% / 74.9% | +0.265 |

## 2. Full PrimeVul Test (Weighted Estimate over 25,911 Functions, prevalence = 2.68%)

| Method | ROC-AUC | **Debiased ROC-AUC** | AUPRC | **VD-S (FNR @ 0.5% FPR)** | Recall @ 0.5% FPR | Actual FPR | Spearman(log chars) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `length_only` | 0.1952 | **0.5087** | 0.0152 | **100.00%** | 0.00% | 0.049% | -0.984 |
| `tfidf_word` | 0.3230 | **0.4939** | 0.0179 | **100.00%** | 0.00% | 0.049% | -0.539 |
| `zero_shot_yesno` | 0.7636 | **0.5653** | 0.0757 | **97.70%** | 2.30% | 0.227% | +0.669 |
| `probe_std` | 0.3328 | **0.4877** | 0.0188 | **99.71%** | 0.29% | 0.049% | -0.513 |
| `probe_pair_margin` | 0.2433 | **0.4541** | 0.0178 | **99.57%** | 0.43% | 0.448% | -0.777 |
| `probe_len_balanced` | 0.3859 | **0.4687** | 0.0253 | **98.99%** | 1.01% | 0.464% | -0.305 |
| `probe_len_orth` | 0.4303 | **0.4548** | 0.0287 | **98.42%** | 1.58% | 0.419% | -0.135 |
| `probe_mixed_benign` | 0.7811 | **0.5888** | 0.0857 | **97.55%** | 2.45% | 0.439% | +0.722 |
| `mlp_len_orth` | 0.7477 | **0.5825** | 0.0653 | **97.70%** | 2.30% | 0.464% | +0.631 |

## 3. Layer & Pooling Sweep (`probe_len_orth`)

| Pool | Layer | Best C | Valid Bal Rank | Test All Rank | Test Debias Rank | Test PatchRemoves Rank |
|---|---:|---:|---:|---:|---:|---:|
| `last` | 13 | 0.1 | 60.1% | 63.6% | 63.0% | 58.5% |
| `last` | 19 | 0.001 | 61.6% | 67.1% | 66.8% | 56.9% |
| `last` | 23 | 0.01 | 59.2% | 66.9% | 68.5% | 60.0% |
| `last` | 27 | 1.0 | 58.9% | 66.8% | 65.2% | 46.1% |
| `mean` | 13 | 0.01 | 70.2% | 77.2% | 78.8% | 64.6% |
| `mean` | 19 | 0.001 | 66.7% | 74.0% | 75.8% | 64.6% |
| `mean` | 23 | 0.01 | 64.7% | 72.6% | 74.7% | 61.5% |
| `mean` | 27 | 0.1 | 69.6% | 74.4% | 75.3% | 52.3% |
