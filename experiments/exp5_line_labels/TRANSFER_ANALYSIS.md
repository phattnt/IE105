# Exp5 – kiểm toán chuyển miền SecVulEval → PrimeVul

Tự sinh bởi `analyze_transfer.py` từ các `results.json` và `primevul_results.json` đã commit; không train lại và không đọc dữ liệu ngoài repo.

## Câu hỏi

Một cấu hình định vị dòng tốt hơn trên SecVulEval có tiếp tục tốt hơn khi chuyển sang PrimeVul hay không?

## Kết quả

| Run | Loại | Top-10 SecVulEval | Top-10 PrimeVul | Δ Prime − Sec | Lift trên ngẫu nhiên Sec / Prime | Cặp Sec / Prime | AUPRC − prevalence Sec / Prime (điểm %) |
|---|---|---:|---:|---:|---:|---:|---:|
| `slide_ModernBERT-base_2048_s0_s13_conv` | comparable_converged | 71.46 | 58.82 | -12.64 | +22.79 / +22.13 | 62.87 / 64.80 | +2.46 / -0.97 |
| `slide_unixcoder-base_1024_s0_conv` | comparable_converged | 70.01 | 60.78 | -9.23 | +21.34 / +24.09 | 57.46 / 59.93 | +1.49 / -1.02 |
| `slide_ModernBERT-base_2048_s0_conv` | comparable_converged | 68.76 | 59.38 | -9.38 | +20.09 / +22.69 | 57.97 / 62.41 | +2.04 / -0.99 |
| `slide_ModernBERT-base_2048_s0` | pilot | 67.60 | 54.34 | -13.26 | +18.93 / +17.65 | 53.78 / 58.87 | +2.16 / -0.99 |
| `slide_ModernBERT-base_2048_s0_contrast03_conv` | comparable_converged | 66.15 | 50.42 | -15.73 | +17.49 / +13.73 | 55.16 / 60.64 | +2.93 / -0.65 |
| `slide_ModernBERT-base_2048_s0_s13bal_conv` | comparable_converged | 65.77 | 54.34 | -11.43 | +17.10 / +17.65 | 56.03 / 58.33 | +2.89 / -1.03 |
| `slide_ModernBERT-base_2048_s0_synth1` | synthetic_pretrain_only | 64.32 | 51.82 | -12.50 | +15.65 / +15.13 | 56.28 / 58.78 | +2.70 / -0.55 |
| `slide_ModernBERT-base_2048_s0_contrast_conv` | comparable_converged | 62.87 | 49.86 | -13.01 | +14.21 / +13.17 | 62.26 / 62.32 | +3.25 / -0.64 |
| `slide_ModernBERT-base_2048_s0_contrast` | pilot | 59.88 | 49.30 | -10.58 | +11.22 / +12.61 | 68.28 / 71.90 | +3.76 / n.a. |

## Diễn giải

1. Thứ hạng Top-10 giữa hai bộ dữ liệu có tương quan dương nhưng không hoàn toàn ổn định: Spearman = **0.771** trên 6 run hội tụ có thể so sánh. Đây là thống kê mô tả giữa các cấu hình, không phải kiểm định độc lập.
2. Run tốt nhất trên SecVulEval là `slide_ModernBERT-base_2048_s0_s13_conv` (71.46%). So với plain, nó **+2.70 điểm** trên SecVulEval nhưng **-0.56 điểm** trên PrimeVul. Mức tăng quan sát được trong miền không tự động chuyển sang bộ dữ liệu khác.
3. Run tốt nhất trên PrimeVul là `slide_unixcoder-base_1024_s0_conv` (60.78%). So với plain: +1.25 điểm SecVulEval và +1.40 điểm PrimeVul.
4. Phân loại mức hàm không chuyển miền: **8/8** run có AUPRC PrimeVul thấp hơn prevalence 695/25.911, dù AUPRC SecVulEval cao hơn prevalence. Vì vậy kết luận chính nên tiếp tục tập trung vào định vị dòng, không tuyên bố đã giải được phát hiện hàm lỗi ngoài miền.

## Kết luận sử dụng trong báo cáo

Các cải tiến Exp5 nhìn chung giữ thứ hạng tương đối giữa SecVulEval và PrimeVul, nhưng cấu hình thắng và mức tăng không trùng nhau. Đặc biệt, tiền huấn luyện guard rồi fine-tune (`s13_conv`) tăng +2,70 điểm Top-10 trên SecVulEval nhưng không vượt plain trên PrimeVul. Do đó mọi tuyên bố cải tiến cần báo đồng thời hai bộ dữ liệu; kết quả một miền không đủ để kết luận mô hình tổng quát hơn.

## Giới hạn

- Các run chủ yếu dùng seed 0; tương quan giữa cấu hình không thay thế cho thí nghiệm nhiều seed.
- Không có khoảng tin cậy cho chênh lệch chuyển miền trong báo cáo này vì file PrimeVul đã lưu không kèm project cho từng hàm.
- Run pilot và synthetic-only được hiển thị để đầy đủ nhưng bị loại khỏi tương quan chính.
