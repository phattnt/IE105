# Báo Cáo Phân Tích Toàn Diện: Lọc Kết Hợp Điểm Dòng & Ngưỡng Tự Tin (Joint Line-Sink Filter)

| Cấu hình / Nhánh thử nghiệm | Tập mẫu | P-C (Đúng cả hai) ↑ | P-V (Gắn cờ cả hai) ↓ | P-B (Bỏ qua cả hai) | P-R (Đảo ngược) | Net [CI 95%] ↑ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **plain (Zero-shot)** | ALL (564 pairs) | 13.3% | 60.6% | 17.4% | 8.7% | **+4.61%** [0.7; 8.5] |
| **plain (Zero-shot)** | CLEAN (408 pairs) | 12.5% | 58.8% | 18.9% | 9.8% | **+2.70%** [-1.5; 7.1] |
| **plain (Zero-shot)** | LEAKED (156 pairs) | 15.4% | 65.4% | 13.5% | 5.8% | **+9.62%** [2.6; 16.7] |
| **plain (Zero-shot)** | CHANGED (357 pairs) | 14.6% | 61.6% | 16.0% | 7.8% | **+6.72%** [2.0; 11.5] |
| **plain (Zero-shot)** | ADD_ONLY (207 pairs) | 11.1% | 58.9% | 19.8% | 10.1% | **+0.97%** [-5.3; 7.2] |
| **plain (Zero-shot)** | LENGTH_MATCHED (<5%) | 13.4% | 62.8% | 15.2% | 8.5% | **+4.88%** [0.0; 10.1] |
| **audit (Default t0)** | ALL (564 pairs) | 16.8% | 61.0% | 10.6% | 11.5% | **+5.32%** [1.1; 9.8] |
| **audit (Default t0)** | CLEAN (408 pairs) | 15.7% | 61.5% | 10.8% | 12.0% | **+3.68%** [-1.7; 9.1] |
| **audit (Default t0)** | LEAKED (156 pairs) | 19.9% | 59.6% | 10.3% | 10.3% | **+9.62%** [1.9; 17.9] |
| **audit (Default t0)** | CHANGED (357 pairs) | 15.1% | 64.7% | 9.5% | 10.6% | **+4.48%** [-1.1; 9.5] |
| **audit (Default t0)** | ADD_ONLY (207 pairs) | 19.8% | 54.6% | 12.6% | 13.0% | **+6.76%** [-0.5; 14.5] |
| **audit (Default t0)** | LENGTH_MATCHED (<5%) | 14.9% | 60.7% | 11.0% | 13.4% | **+1.52%** [-4.9; 7.3] |
| **audit (Prob >= 85%)** | ALL (564 pairs) | 19.1% | 54.8% | 14.5% | 11.5% | **+7.62%** [3.2; 11.9] |
| **audit (Prob >= 85%)** | CLEAN (408 pairs) | 18.4% | 54.9% | 15.0% | 11.8% | **+6.62%** [1.5; 12.5] |
| **audit (Prob >= 85%)** | LEAKED (156 pairs) | 21.2% | 54.5% | 13.5% | 10.9% | **+10.26%** [1.3; 18.6] |
| **audit (Prob >= 85%)** | CHANGED (357 pairs) | 17.1% | 58.8% | 14.0% | 10.1% | **+7.00%** [1.4; 12.3] |
| **audit (Prob >= 85%)** | ADD_ONLY (207 pairs) | 22.7% | 47.8% | 15.5% | 14.0% | **+8.70%** [0.5; 16.9] |
| **audit (Prob >= 85%)** | LENGTH_MATCHED (<5%) | 17.7% | 53.4% | 15.9% | 13.1% | **+4.57%** [-1.5; 10.7] |
| **audit_hint (Default t0)** | ALL (564 pairs) | 19.7% | 45.9% | 21.1% | 13.3% | **+6.38%** [2.1; 10.6] |
| **audit_hint (Default t0)** | CLEAN (408 pairs) | 18.9% | 46.3% | 21.1% | 13.7% | **+5.15%** [-0.5; 10.3] |
| **audit_hint (Default t0)** | LEAKED (156 pairs) | 21.8% | 44.9% | 21.2% | 12.2% | **+9.62%** [0.0; 18.6] |
| **audit_hint (Default t0)** | CHANGED (357 pairs) | 18.2% | 49.3% | 19.6% | 12.9% | **+5.32%** [-0.6; 11.5] |
| **audit_hint (Default t0)** | ADD_ONLY (207 pairs) | 22.2% | 40.1% | 23.7% | 14.0% | **+8.21%** [0.0; 16.4] |
| **audit_hint (Default t0)** | LENGTH_MATCHED (<5%) | 18.0% | 44.2% | 23.5% | 14.3% | **+3.66%** [-2.1; 9.8] |
| **audit_hint (Prob >= 85%)** | ALL (564 pairs) | 20.7% | 42.6% | 24.5% | 12.2% | **+8.51%** [3.9; 12.9] |
| **audit_hint (Prob >= 85%)** | CLEAN (408 pairs) | 20.3% | 42.9% | 24.0% | 12.7% | **+7.60%** [2.2; 13.0] |
| **audit_hint (Prob >= 85%)** | LEAKED (156 pairs) | 21.8% | 41.7% | 25.6% | 10.9% | **+10.90%** [1.9; 19.9] |
| **audit_hint (Prob >= 85%)** | CHANGED (357 pairs) | 18.8% | 46.5% | 23.0% | 11.8% | **+7.00%** [1.4; 13.2] |
| **audit_hint (Prob >= 85%)** | ADD_ONLY (207 pairs) | 24.2% | 35.7% | 27.1% | 13.0% | **+11.11%** [2.9; 19.3] |
| **audit_hint (Prob >= 85%)** | LENGTH_MATCHED (<5%) | 19.2% | 40.9% | 27.1% | 12.8% | **+6.40%** [0.3; 12.5] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | ALL (564 pairs) | 20.4% | 40.8% | 27.0% | 11.9% | **+8.51%** [4.3; 12.8] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | CLEAN (408 pairs) | 20.3% | 41.2% | 26.0% | 12.5% | **+7.84%** [2.5; 13.0] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | LEAKED (156 pairs) | 20.5% | 39.7% | 29.5% | 10.3% | **+10.26%** [1.3; 18.6] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | CHANGED (357 pairs) | 18.2% | 44.8% | 25.2% | 11.8% | **+6.44%** [0.8; 12.3] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | ADD_ONLY (207 pairs) | 24.2% | 33.8% | 30.0% | 12.1% | **+12.08%** [3.4; 20.3] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | LENGTH_MATCHED (<5%) | 19.2% | 39.0% | 29.6% | 12.2% | **+7.01%** [0.9; 13.1] |
| **audit_rand (Default t0)** | ALL (564 pairs) | 22.7% | 36.9% | 24.3% | 16.1% | **+6.56%** [1.4; 11.5] |
| **audit_rand (Default t0)** | CLEAN (408 pairs) | 23.0% | 36.3% | 24.5% | 16.2% | **+6.86%** [0.5; 12.7] |
| **audit_rand (Default t0)** | LEAKED (156 pairs) | 21.8% | 38.5% | 23.7% | 16.0% | **+5.77%** [-3.8; 15.4] |
| **audit_rand (Default t0)** | CHANGED (357 pairs) | 20.2% | 38.9% | 23.5% | 17.4% | **+2.80%** [-3.4; 9.3] |
| **audit_rand (Default t0)** | ADD_ONLY (207 pairs) | 27.1% | 33.3% | 25.6% | 14.0% | **+13.04%** [4.3; 20.8] |
| **audit_rand (Default t0)** | LENGTH_MATCHED (<5%) | 21.0% | 34.8% | 27.1% | 17.1% | **+3.96%** [-2.4; 10.1] |
| **audit_ctx (Default t0)** | ALL (564 pairs) | 13.5% | 66.8% | 10.6% | 9.0% | **+4.43%** [0.7; 8.3] |
| **audit_ctx (Default t0)** | CLEAN (408 pairs) | 12.5% | 66.2% | 11.8% | 9.6% | **+2.94%** [-2.0; 7.4] |
| **audit_ctx (Default t0)** | LEAKED (156 pairs) | 16.0% | 68.6% | 7.7% | 7.7% | **+8.33%** [0.6; 16.0] |
| **audit_ctx (Default t0)** | CHANGED (357 pairs) | 12.3% | 70.0% | 10.1% | 7.6% | **+4.76%** [0.6; 9.2] |
| **audit_ctx (Default t0)** | ADD_ONLY (207 pairs) | 15.5% | 61.4% | 11.6% | 11.6% | **+3.86%** [-3.4; 11.6] |
| **audit_ctx (Default t0)** | LENGTH_MATCHED (<5%) | 11.9% | 67.7% | 11.0% | 9.5% | **+2.44%** [-2.1; 7.3] |
| **audit_ctxfake (Default t0)** | ALL (564 pairs) | 13.8% | 67.9% | 7.8% | 10.5% | **+3.37%** [-0.5; 7.6] |
| **audit_ctxfake (Default t0)** | CLEAN (408 pairs) | 13.0% | 68.9% | 8.8% | 9.3% | **+3.68%** [-1.0; 8.8] |
| **audit_ctxfake (Default t0)** | LEAKED (156 pairs) | 16.0% | 65.4% | 5.1% | 13.5% | **+2.56%** [-5.8; 11.5] |
| **audit_ctxfake (Default t0)** | CHANGED (357 pairs) | 12.0% | 70.3% | 6.7% | 10.9% | **+1.12%** [-3.9; 5.6] |
| **audit_ctxfake (Default t0)** | ADD_ONLY (207 pairs) | 16.9% | 63.8% | 9.7% | 9.7% | **+7.25%** [0.5; 14.5] |
| **audit_ctxfake (Default t0)** | LENGTH_MATCHED (<5%) | 11.6% | 69.2% | 8.5% | 10.7% | **+0.91%** [-4.0; 5.5] |
