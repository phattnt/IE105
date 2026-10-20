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
