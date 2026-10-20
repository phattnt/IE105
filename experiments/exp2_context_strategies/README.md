# Thí nghiệm 2 – Cách đưa hàm dài vào LineVul (ngữ cảnh trong hàm)

Cùng checkpoint LineVul gốc, **không train lại**, cùng ngân sách 512 token mỗi lần suy luận.
Chỉ đổi **phần nào của hàm** mô hình được xem:

| Chiến lược | Mô hình xem gì |
|---|---|
| `head` | 510 token đầu (LineVul gốc, = thí nghiệm 1) |
| `headtail` | 255 token đầu + 255 token cuối |
| `slide_max` / `slide_mean` | cửa sổ 510 token, bước 256, phủ toàn bộ hàm; điểm hàm = max / mean qua các cửa sổ; điểm dòng = max |
| `slice` | giữ dòng có "mỏ neo" (API bộ nhớ/chuỗi, chỉ số mảng, con trỏ, `sizeof`, biến kích thước) kèm 1 dòng lân cận, theo thứ tự gốc, rồi cắt ở 510 |

Đo cho từng chiến lược: 4 chỉ số cặp trên 564 cặp, định vị dòng chia nhóm A/B/C
(**nhóm định nghĩa theo cắt đầu**, để thấy nhóm C có được "cứu" không), và mức hàm trên toàn bộ tập test.

## Lưu ý khi đọc kết quả

- Checkpoint được train với input **cắt đầu**, nên cửa sổ cuối hàm và slice là **phân bố lạ** với nó.
  Kết quả ở đây đo *chỉ riêng việc chọn input lúc suy luận* làm được gì; train lại cùng chiến lược là bước sau.
- `slide_max` tăng tỉ lệ báo "có lỗi" theo số cửa sổ (hàm dài → nhiều cửa sổ → max cao hơn), nên
  phải nhìn cả P-V và tỉ lệ báo lỗi, không chỉ P-C. `slide_mean` là đối chứng.
- Slicing với ngân sách 510 token không phủ hết hàm rất dài; `flaw_lines_visible_under_this_strategy_pct` cho biết mỗi chiến lược nhìn thấy bao nhiêu % dòng lỗi.

## Chạy

```bash
colab new -s exp2 --gpu T4
colab exec -s exp2 -f ../exp1_linevul_on_primevul/setup_vm.py
colab upload -s exp2 ../exp1_linevul_on_primevul/eval_linevul_primevul.py /content/exp1/code/eval_linevul_primevul.py
colab upload -s exp2 eval_context_strategies.py /content/exp1/code/eval_context_strategies.py
colab exec -s exp2 -f eval_context_strategies.py
colab download -s exp2 /content/exp1/results_exp2/results.json results/results.json
colab stop -s exp2
```

## Kết quả (T4, chỉ suy luận, ~35 phút)

File: `results/results.json` (tổng hợp), `results/localization_*.jsonl` (từng hàm lỗi trong tập cặp),
`results/full_test_*.jsonl` (xác suất từng hàm trong tập test), `results/ranking_check.json`
(độ chính xác xếp hạng tính lại: cặp có input giống hệt nhau tính là hòa, vì hàm lỗi được chấm ở fp32
để lấy attention còn hàm đã vá ở fp16, nên `ranking_accuracy_pct` trong `results.json` bị nhiễu ở 200 cặp đó).

### Mức hàm – 564 cặp (test_paired)

| | `head` | `headtail` | `slide_max` | `slide_mean` | `slice` |
|---|---|---|---|---|---|
| P-C (cả cặp đúng) | 7 | 2 | 2 | 5 | 5 |
| P-V (cả hai bị báo lỗi) | 106 | 96 | 149 | 74 | 48 |
| P-B (cả hai bị bỏ qua) | 445 | 461 | 396 | 479 | 495 |
| P-R (đảo ngược) | 6 | 5 | 17 | 6 | 16 |
| Độ chính xác xếp hạng (cặp giống hệt = hòa) | 47.7% | 51.1% | 47.5% | 47.0% | 50.4% |
| … chỉ trên cặp phân biệt được | 46.4% (n=364) | 51.6% (n=364) | 47.5% (n=564) | 47.0% (n=564) | 50.7% (n=365) |
| F1 cân bằng trên cặp | 0.286 | 0.257 | 0.343 | 0.219 | 0.156 |
| Cặp giống hệt sau xử lý | 35.5% | 35.5% | 0.0% | 0.0% | 35.3% |
| Tỉ lệ báo lỗi: hàm lỗi bị cắt / không bị cắt | 24.5% / 8.8% | 20.8% / 8.8% | 33.9% / 8.8% | 16.1% / 8.8% | 10.6% / 6.2% |

### Mức hàm – toàn bộ tập test (25.911 hàm, 695 lỗi)

| | `head` | `headtail` | `slide_max` | `slide_mean` | `slice` |
|---|---|---|---|---|---|
| F1 @0.5 | 0.110 | 0.102 | 0.127 | 0.082 | 0.070 |
| Precision / Recall | 0.077 / 0.197 | 0.072 / 0.176 | 0.083 / 0.272 | 0.060 / 0.132 | 0.058 / 0.089 |
| Tỉ lệ báo lỗi | 6.9% | 6.5% | 8.8% | 5.9% | 4.1% |
| AUPRC | 0.065 | 0.055 | 0.076 | 0.064 | 0.058 |
| ROC-AUC | 0.730 | 0.679 | 0.751 | 0.740 | 0.715 |
| VD-S (ngưỡng oracle) | 98.3% | 99.1% | 98.4% | 98.8% | 99.0% |

### Định vị dòng – 357 hàm lỗi có nhãn dòng (nhóm A/B/C định nghĩa theo cắt đầu)

| | `head` | `headtail` | `slide_max` | `slide_mean` | `slice` |
|---|---|---|---|---|---|
| % dòng lỗi mô hình nhìn thấy | 43.4% | 42.4% | 100.0% | 100.0% | 35.4% |
| ALL (n=357) Top-5 / Top-10 | 32.2 / 44.3 | 32.5 / 47.1 | 36.1 / 50.7 | 36.1 / 50.7 | 33.9 / 48.5 |
| ALL IFA trung vị | 14 | 12 | 9 | 9 | 11 |
| A (n=103) Top-5 / Top-10 | 60.2 / 85.4 | 60.2 / 85.4 | 60.2 / 85.4 | 60.2 / 85.4 | 58.3 / 81.6 |
| A IFA trung vị | 3 | 3 | 3 | 3 | 3 |
| B (n=122) Top-5 / Top-10 | 43.4 / 57.4 | 29.5 / 43.4 | 34.4 / 45.1 | 34.4 / 45.1 | 36.1 / 49.2 |
| B IFA trung vị | 7 | 14 | 12 | 12 | 10 |
| C (n=132) Top-5 / Top-10 | 0.0 / 0.0 | 13.6 / 20.5 | 18.9 / 28.8 | 18.9 / 28.8 | 12.9 / 22.0 |
| C IFA trung vị | 79 | 78 | 26 | 26 | 70 |

Ngẫu nhiên (kỳ vọng): ALL Top-5/10 = 22.6 / 36.7; B = 23.9 / 36.9; C = 8.5 / 15.6.

## Đọc kết quả

1. **Mức hàm: không chiến lược nào giúp phân biệt bản lỗi và bản vá.** Độ chính xác xếp hạng trên các cặp
   phân biệt được dao động 46–52% ≈ tung đồng xu với mọi cách chọn input. P-C tốt nhất vẫn là `head` (7/564).
   Xác suất "có lỗi" của mô hình vẫn bị kéo bởi độ dài: hàm lỗi bị cắt được báo lỗi 2–4 lần nhiều hơn hàm
   không bị cắt, ở mọi chiến lược (`slide_max` nặng nhất: 33,9% vs 8,8%, vì nhiều cửa sổ → max cao).
   `slide_max` tăng F1 toàn tập test (0,110 → 0,127) nhưng chỉ nhờ recall (0,20 → 0,27) và kéo P-V (149) lẫn
   P-R (17) lên — tức là báo lỗi nhiều hơn chứ không hiểu hơn.
2. **Mức dòng: nhóm C được "cứu" một phần chỉ bằng cách cho mô hình nhìn thấy dòng.** Với `slide`, 132 hàm
   trước đây Top-10 = 0% (dòng lỗi nằm ngoài 512 token) nay đạt Top-10 28,8% (ngẫu nhiên 15,6%), IFA trung vị
   79 → 26. Tổng thể 357 hàm: Top-10 44,3 → 50,7, IFA 14 → 9. Điều này xảy ra dù checkpoint chưa từng thấy
   cửa sổ giữa/cuối hàm lúc train.
3. **Nhưng nhóm B bị kém đi** (`head` Top-10 57,4 → `slide` 45,1; `headtail` 43,4): khi dòng lỗi vốn đã trong
   512 token đầu, mọi cách thay đổi input đều làm attention lệch khỏi phân bố train. Attention lớp 0 lấy max
   qua các cửa sổ không so sánh được giữa cửa sổ với nhau.
4. **`slice` theo mỏ neo cú pháp chưa đủ tốt:** chỉ phủ 35,4% dòng lỗi (ít hơn cắt đầu 43,4%), nên nhóm B/ALL
   không hơn `head`, dù nhóm C có cải thiện (Top-10 22%). Cần cách chọn dòng dựa trên dữ liệu (học) thay vì regex.
5. `headtail` gần như vô ích: dòng lỗi nhìn thấy không tăng (42,4%), mức hàm xấu hơn (AUC 0,73 → 0,68).

**Kết luận cho hướng nghiên cứu:** vấn đề "mô hình không nhìn thấy dòng lỗi" là có thật và sửa được ở mức
dòng ngay cả khi không train lại; nhưng ở mức hàm, LineVul gốc không học được tín hiệu lỗi thật sự trên PrimeVul,
nên thay đổi input lúc suy luận không cứu được. Bước tiếp theo bắt buộc là **train lại với cùng chiến lược**
(TN0 + TN2-train), có đối chứng độ dài, rồi mới so sánh.
