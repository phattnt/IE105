# Bảng kết quả tổng hợp (tự sinh bởi `collect_results.py`, không sửa tay)

Cặp = 564 cặp test của bản PrimeVul gốc; test = 25.911 hàm; Top-10 theo nhóm A/B/C định nghĩa bởi cắt đầu 512 token; `length` = đối chứng chỉ nhìn độ dài.

| run | backbone | training | strategy | P-C | P-V | P-B | P-R | rank_acc_pct | F1_test | AUPRC | ROC_AUC | VDS_oracle | top10_ALL | top10_B | top10_C | flaw_visible_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp1 | codebert-base | Big-Vul (checkpoint gốc) | head | 7 | 106 | 445 | 6 | 47.252 | 0.11 | 0.065 | 0.73 | 0.983 | 43.978 | 56.557 | 0.0 | 43.351 |
| exp2/head | codebert-base | Big-Vul (checkpoint gốc) | head | 7 | 106 | 445 | 6 | 47.695 | 0.11 | 0.065 | 0.73 | 0.983 | 44.258 | 57.377 | 0.0 | 43.351 |
| exp2/headtail | codebert-base | Big-Vul (checkpoint gốc) | headtail | 2 | 96 | 461 | 5 | 51.064 | 0.102 | 0.055 | 0.679 | 0.991 | 47.059 | 43.443 | 20.455 | 42.352 |
| exp2/slide_max | codebert-base | Big-Vul (checkpoint gốc) | slide_max | 2 | 149 | 396 | 17 | 47.518 | 0.127 | 0.076 | 0.751 | 0.984 | 50.7 | 45.082 | 28.788 | 100.0 |
| exp2/slide_mean | codebert-base | Big-Vul (checkpoint gốc) | slide_mean | 5 | 74 | 479 | 6 | 46.986 | 0.082 | 0.064 | 0.74 | 0.988 | 50.7 | 45.082 | 28.788 | 100.0 |
| exp2/slice | codebert-base | Big-Vul (checkpoint gốc) | slice | 5 | 48 | 495 | 16 | 50.443 | 0.07 | 0.058 | 0.715 | 0.99 | 48.459 | 49.18 | 21.97 | 35.434 |
| exp3/head | codebert-base | PrimeVul tập con | head | 2 | 94 | 462 | 6 | 43.085 | 0.193 | 0.13 | 0.833 | 0.947 | 44.258 | 57.377 | 0.0 | 43.351 |
| exp3/head_codebert-c | codebert-c | PrimeVul tập con | head | 4 | 202 | 347 | 11 | 45.922 | 0.233 | 0.157 | 0.828 | 0.911 | 44.258 | 58.197 | 0.0 | 43.351 |
| exp3/head_unixcoder-base | unixcoder-base | PrimeVul tập con | head | 8 | 223 | 319 | 14 | 41.135 | 0.226 | 0.151 | 0.833 | 0.917 | 46.218 | 50.833 | 0.0 | 53.728 |
| exp3/length | — | hồi quy logistic trên độ dài | length | 1 | 120 | 441 | 2 | 13.652 | 0.199 | 0.13 | 0.813 | 0.941 |  |  |  |  |
| exp3/slice | codebert-base | PrimeVul tập con | slice | 5 | 189 | 359 | 11 | 38.83 | 0.237 | 0.165 | 0.835 | 0.902 | 48.459 | 50.0 | 21.97 | 35.434 |
| exp3/slide | codebert-base | PrimeVul tập con | slide | 11 | 223 | 310 | 20 | 44.681 | 0.223 | 0.151 | 0.838 | 0.915 | 50.98 | 46.721 | 28.03 | 100.0 |
