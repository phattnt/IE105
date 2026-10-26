# Báo Cáo Phân Tích Toàn Diện: Lọc Kết Hợp Điểm Dòng & Ngưỡng Tự Tin (Joint Line-Sink Filter)

| Cấu hình / Nhánh thử nghiệm | Tập mẫu | P-C (Đúng cả hai) ↑ | P-V (Gắn cờ cả hai) ↓ | P-B (Bỏ qua cả hai) | P-R (Đảo ngược) | Net [CI 95%] ↑ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **plain (Zero-shot)** | ALL (564 pairs) | 19.5% | 51.1% | 19.1% | 10.3% | **+9.22%** [5.0; 13.5] |
| **plain (Zero-shot)** | CLEAN (408 pairs) | 19.9% | 49.5% | 20.1% | 10.5% | **+9.31%** [3.9; 14.5] |
| **plain (Zero-shot)** | LEAKED (156 pairs) | 18.6% | 55.1% | 16.7% | 9.6% | **+8.97%** [1.3; 17.3] |
| **plain (Zero-shot)** | CHANGED (357 pairs) | 19.0% | 53.8% | 18.2% | 9.0% | **+10.08%** [5.0; 15.4] |
| **plain (Zero-shot)** | ADD_ONLY (207 pairs) | 20.3% | 46.4% | 20.8% | 12.6% | **+7.73%** [0.5; 15.5] |
| **plain (Zero-shot)** | LENGTH_MATCHED (<5%) | 17.1% | 52.4% | 19.2% | 11.3% | **+5.79%** [0.0; 11.6] |
| **audit (Default t0)** | ALL (564 pairs) | 14.2% | 63.8% | 10.8% | 11.2% | **+3.01%** [-0.9; 7.3] |
| **audit (Default t0)** | CLEAN (408 pairs) | 12.3% | 63.7% | 12.3% | 11.8% | **+0.49%** [-4.4; 5.4] |
| **audit (Default t0)** | LEAKED (156 pairs) | 19.2% | 64.1% | 7.1% | 9.6% | **+9.62%** [1.3; 17.3] |
| **audit (Default t0)** | CHANGED (357 pairs) | 12.9% | 67.5% | 9.5% | 10.1% | **+2.80%** [-2.0; 7.6] |
| **audit (Default t0)** | ADD_ONLY (207 pairs) | 16.4% | 57.5% | 13.0% | 13.0% | **+3.38%** [-4.3; 10.1] |
| **audit (Default t0)** | LENGTH_MATCHED (<5%) | 12.2% | 65.5% | 10.4% | 11.9% | **+0.30%** [-5.2; 5.5] |
| **audit (Prob >= 85%)** | ALL (564 pairs) | 14.0% | 8.7% | 66.1% | 11.2% | **+2.84%** [-1.6; 6.7] |
| **audit (Prob >= 85%)** | CLEAN (408 pairs) | 12.5% | 8.1% | 66.7% | 12.7% | **-0.25%** [-5.1; 4.7] |
| **audit (Prob >= 85%)** | LEAKED (156 pairs) | 17.9% | 10.3% | 64.7% | 7.1% | **+10.90%** [3.2; 18.6] |
| **audit (Prob >= 85%)** | CHANGED (357 pairs) | 16.2% | 8.4% | 63.6% | 11.8% | **+4.48%** [-0.6; 10.4] |
| **audit (Prob >= 85%)** | ADD_ONLY (207 pairs) | 10.1% | 9.2% | 70.5% | 10.1% | **+0.00%** [-6.3; 6.3] |
| **audit (Prob >= 85%)** | LENGTH_MATCHED (<5%) | 13.7% | 8.8% | 65.5% | 11.9% | **+1.83%** [-3.7; 7.3] |
| **audit_hint (Default t0)** | ALL (564 pairs) | 18.6% | 53.0% | 17.4% | 11.0% | **+7.62%** [3.0; 12.1] |
| **audit_hint (Default t0)** | CLEAN (408 pairs) | 18.1% | 51.2% | 17.9% | 12.7% | **+5.39%** [0.0; 10.5] |
| **audit_hint (Default t0)** | LEAKED (156 pairs) | 19.9% | 57.7% | 16.0% | 6.4% | **+13.46%** [6.4; 21.8] |
| **audit_hint (Default t0)** | CHANGED (357 pairs) | 15.7% | 58.0% | 15.4% | 10.9% | **+4.76%** [-0.3; 9.8] |
| **audit_hint (Default t0)** | ADD_ONLY (207 pairs) | 23.7% | 44.4% | 20.8% | 11.1% | **+12.56%** [4.3; 20.3] |
| **audit_hint (Default t0)** | LENGTH_MATCHED (<5%) | 14.9% | 54.3% | 19.2% | 11.6% | **+3.35%** [-2.7; 9.1] |
| **audit_hint (Prob >= 85%)** | ALL (564 pairs) | 16.8% | 8.3% | 61.9% | 12.9% | **+3.90%** [-0.2; 8.5] |
| **audit_hint (Prob >= 85%)** | CLEAN (408 pairs) | 16.4% | 7.4% | 62.7% | 13.5% | **+2.94%** [-2.2; 8.1] |
| **audit_hint (Prob >= 85%)** | LEAKED (156 pairs) | 17.9% | 10.9% | 59.6% | 11.5% | **+6.41%** [-1.9; 14.7] |
| **audit_hint (Prob >= 85%)** | CHANGED (357 pairs) | 17.9% | 10.6% | 59.9% | 11.5% | **+6.44%** [0.8; 12.3] |
| **audit_hint (Prob >= 85%)** | ADD_ONLY (207 pairs) | 15.0% | 4.3% | 65.2% | 15.5% | **-0.48%** [-7.7; 7.2] |
| **audit_hint (Prob >= 85%)** | LENGTH_MATCHED (<5%) | 13.4% | 8.8% | 65.2% | 12.5% | **+0.91%** [-4.6; 6.4] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | ALL (564 pairs) | 16.1% | 8.2% | 63.5% | 12.2% | **+3.90%** [-0.4; 8.5] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | CLEAN (408 pairs) | 15.7% | 7.1% | 64.5% | 12.7% | **+2.94%** [-2.2; 8.1] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | LEAKED (156 pairs) | 17.3% | 10.9% | 60.9% | 10.9% | **+6.41%** [-1.3; 15.4] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | CHANGED (357 pairs) | 17.1% | 10.4% | 61.6% | 10.9% | **+6.16%** [0.8; 11.8] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | ADD_ONLY (207 pairs) | 14.5% | 4.3% | 66.7% | 14.5% | **+0.00%** [-7.2; 7.2] |
| **audit_hint (Prob >= 85% + Sink-in-Hint)** | LENGTH_MATCHED (<5%) | 12.8% | 8.5% | 66.8% | 11.9% | **+0.91%** [-4.6; 6.1] |
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
