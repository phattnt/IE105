# BÁO CÁO TỔNG KẾT & KIỂM ĐỊNH ĐỐI CHỨNG: ĐỊNH VỊ DÒNG MODERNBERT VÀ PHÁN QUYẾT MỨC HÀM CỦA LLM TRÊN PRIMEVUL (26/10/2026)

> **Dự án**: IE105 - C/C++ Vulnerability Detection & Line-Level Localization  
> **Thời điểm cập nhật**: 26/10/2026  
> **Nguyên tắc đánh giá**: Chỉ dùng mã nguồn hàm ở thời điểm kiểm thử (code-only), đối chứng trực tiếp với nhánh **gợi ý dòng ngẫu nhiên (`audit_rand`)** và giao thức đã đăng ký trước (`t0`, `t_median` trong [`pair_protocol.json`](experiments/exp5_line_labels/results/llm_full_a100/pair_protocol.json) & [`scores/summary.json`](experiments/exp5_line_labels/results/llm_full_a100/scores/summary.json)), kiểm toán rò rỉ độc lập (`CLEAN` 408 cặp vs `LEAKED` 156 cặp).

---

## 1. KẾT LUẬN KHOA HỌC CỐT LÕI (EXECUTIVE VERDICT)

**Tóm lại: Gợi ý dòng của ModernBERT giúp LLM chỉ đúng dòng lỗi thật, nhưng CHƯA giúp LLM phán đúng hàm nào có lỗi. Các con số "tốt" ở mức hàm không vượt qua được nhánh đối chứng dùng dòng ngẫu nhiên (`audit_rand`).**

Khi kiểm tra nghiêm ngặt trên dữ liệu gốc của **Qwen3.5-9B** ([`llm_full_a100/scores/summary.json`](experiments/exp5_line_labels/results/llm_full_a100/scores/summary.json), [`llm_full_a100/pair_protocol.json`](experiments/exp5_line_labels/results/llm_full_a100/pair_protocol.json)) — nơi có đủ cả 3 nhánh đối chứng: **gợi ý ModernBERT (`audit_hint`)**, **gợi ý dòng ngẫu nhiên (`audit_rand`)**, và **không gợi ý (`audit`)** — cùng dữ liệu **OpenVul-4B** ([`openvul_test/scores/summary.json`](experiments/exp5_line_labels/results/openvul_test/scores/summary.json), [`openvul_test/pair_protocol.json`](experiments/exp5_line_labels/results/openvul_test/pair_protocol.json)), bức tranh thực chất như sau:

### 1.1. Ở mức dòng (Line-Level): Gợi ý của ModernBERT CÓ tác dụng thật

| Chỉ số mức dòng (Qwen3.5-9B, 356 hàm lỗi có nhãn dòng) | `audit` (Không gợi ý) | `audit_rand` (Dòng ngẫu nhiên) | `audit_hint` (ModernBERT) | Ý nghĩa kiểm định |
|---|:---:|:---:|:---:|---|
| **Có ít nhất 1 sink LLM chỉ ra là dòng lỗi thật** (`any_sink_is_a_flaw_line`) | 32.3% | 29.5% | **41.0%** | **+11.5 điểm** so với ngẫu nhiên — bằng chứng sạch nhất cho thấy định vị dòng mang thông tin thật |
| **Sink thiếu chốt chặn trúng dòng lỗi thật** (`a_missing_sink_is_a_flaw_line`) | 22.5% | 15.4% | **24.7%** | **+9.3 điểm** so với ngẫu nhiên |
| **Sink nằm trong các dòng được gợi ý** (`a_missing_sink_is_a_hint_line`) | — | **48.0%** | **68.3%** *(OpenVul: 72.9%)* | Đo **mức độ LLM nghe lời gợi ý**, không đo chất lượng gợi ý (vì dòng ngẫu nhiên cũng được LLM nhận tới 48.0%) |

* **Kết luận mức dòng:** Gợi ý ModernBERT vượt gợi ý ngẫu nhiên **+11.5 điểm** ở tỷ lệ chỉ trúng dòng lỗi thật (`41.0%` vs `29.5%`). Đây là bằng chứng sạch và chắc chắn cho thấy mô hình dòng của chúng ta mang thông tin định vị thật.
* **Lưu ý về con số `68.3%` (Qwen) và `72.9%` (OpenVul) ở chỉ số `a_missing_sink_is_a_hint_line`:** Con số này chủ yếu phản ánh việc LLM có xu hướng bám vào bất kỳ dòng nào được đưa vào đề bài (ngay cả khi đưa dòng ngẫu nhiên ở `audit_rand`, LLM cũng chọn tới `48.0%`). Do đó không dùng con số 72.9% làm bằng chứng về chất lượng gợi ý.

---

### 1.2. Ở mức hàm (Function-Level): Gợi ý dòng KHÔNG vượt qua được dòng ngẫu nhiên (`audit_rand`)

| Nhánh thử nghiệm (Qwen3.5-9B, 564 cặp) | `Net` ở ngưỡng `t0` (Mặc định) | `Net` ở ngưỡng trung vị (`t_median` — Giao thức đăng ký trước) | Xếp hạng theo cặp (`rank_acc`) | Tỷ lệ bị cắt suy nghĩ (`think_forced`) |
|---|:---:|:---:|:---:|:---:|
| **`plain` (Hỏi trực tiếp)** | +4.43% | +6.38% | 54.4% | 66.8% (752/1126) |
| **`audit` (Không gợi ý)** | +5.14% | **+10.99%** | **56.7%** | 94.0% (1058/1126) |
| **`audit_rand` (Gợi ý dòng ngẫu nhiên)** | **+6.56%** | +4.43% | 54.6% | **98.4% (1108/1126)** |
| **`audit_hint` (Gợi ý dòng ModernBERT)** | **+6.56%** | +7.27% | 54.7% | **98.4% (1108/1126)** |

Các phát hiện kiểm định quan trọng ở mức hàm:
1. **Ở ngưỡng mặc định `t0`:** Gợi ý thật (`audit_hint`) và gợi ý ngẫu nhiên (`audit_rand`) cho điểm `Net` **bằng y hệt nhau (`+6.56%`)**, và xếp hạng cặp (`rank_acc`) cũng ngang nhau (`54.7%` vs `54.6%`).
2. **Ở ngưỡng trung vị `t_median` (giao thức đã định trước):** Nhánh mạnh nhất lại là **`audit` không có gợi ý (`Net = +10.99%`, `rank_acc = 56.7%`)**, cao hơn cả `audit_hint` (`+7.27%`).
3. **Về các con số hậu kiểm (Post-hoc filtering):**
   * Con số `Net = +8.51%` của Qwen khi lọc `prob >= 85% + Sink-in-Hint` được chọn **sau khi đã xem kết quả tập test** (post-hoc), không phải ngưỡng đăng ký trước.
   * Trên tập con `ADD_ONLY` (207 cặp), ngay cả khi áp bộ lọc này, nhánh dòng ngẫu nhiên `audit_rand` đạt **`+13.04%`**, cao hơn mức **`+12.08%`** của gợi ý thật `audit_hint`!
   * Khoảng tin cậy Bootstrap 95% rộng khoảng **$\pm 4.5$ đến $\pm 5.5$ điểm**, nên các chênh lệch 1–3 điểm giữa các nhánh hoàn toàn nằm trong vùng nhiễu thống kê.

---

### 1.3. Hai yếu tố gây nhiễu lớn chưa được kiểm soát (Uncontrolled Confounders)

1. **Qwen3.5-9B bị cắt ngang suy nghĩ gần như toàn bộ (`think_forced = 98.4%`):**
   * Ở cả `audit_hint` lẫn `audit_rand`, có tới **1.108 / 1.126 lượt (98.4%)** chạm trần `--think_budget` và bị ép dừng suy nghĩ giữa chừng (`think_forced`). Phán quyết cuối cùng vì thế gần như được đưa ra khi mô hình chưa suy nghĩ xong.
2. **OpenVul-4B thiếu nhánh đối chứng ngẫu nhiên (`audit_rand`) và cũng bị nhiễu bởi tỷ lệ cắt suy nghĩ:**
   * Trên OpenVul-4B (xem Bảng 2 bên dưới), bước tăng trên tập `CLEAN` từ `audit` (`+0.49%`) lên `audit_hint` (`+5.39%`) đi kèm việc tỷ lệ bị cắt suy nghĩ (`think_forced`) giảm từ **`74.2%` (835/1126)** xuống **`61.8%` (696/1126)**.
   * Vì chưa chạy `audit_rand` cho OpenVul-4B, không thể tách bạch được phần tăng này bao nhiêu là do nội dung gợi ý dòng, bao nhiêu là do mô hình bớt bị cắt suy nghĩ hoặc do thay đổi tỷ lệ gắn cờ (`flag_rate` giảm từ `76.7%` xuống `67.8%`).
   * Quan trọng hơn: Cả `audit` (`+3.0%` toàn tập / `+0.49%` Clean) và `audit_hint` (`+7.6%` toàn tập / `+5.39%` Clean) trên OpenVul-4B **đều vẫn thua nhánh `plain` hỏi trực tiếp (`+9.4%` toàn tập / `+9.31%` Clean, nơi `think_forced` chỉ là `40.0%`)**!

---

## 2. BẢNG SỐ LIỆU ĐỐI CHỨNG ĐẦY ĐỦ THEO GIAO THỨC CHUẨN (`pair_protocol.py` & `compare_arms.py`)

### Bảng 2.1: Kết quả Qwen3.5-9B trên 564 cặp PrimeVul ([`llm_full_a100/pair_protocol.json`](experiments/exp5_line_labels/results/llm_full_a100/pair_protocol.json))

| Nhánh | `flag_rate` (`t0`) | `P-C / P-V / P-B / P-R` (`t0`) | `Net @ t0` [CI 95%] | `Net @ t_median` [CI 95%] | `rank_acc` [CI 95%] | `rank_acc` (sau trừ độ dài) | `think_forced` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`plain`** | 73.6% | `13.3 / 60.5 / 17.2 / 8.9` | **+4.43%** `[+0.5; +8.3]` | **+6.38%** `[+1.8; +11.0]` | 54.4% `[51.1; 57.6]` | 49.0% | 66.8% |
| **`audit`** | 76.2% | `16.8 / 61.9 / 9.6 / 11.7` | **+5.14%** `[+0.9; +9.6]` | **+10.99%** `[+6.4; +15.8]` | **56.7%** `[53.5; 60.0]` | 47.0% | 94.0% |
| **`audit_rand`** | 56.6% | `19.0 / 41.0 / 27.7 / 12.4` | **+6.56%** `[+1.6; +11.5]` | **+4.43%** `[-0.2; +9.0]` | 54.6% `[51.2; 58.1]` | 49.6% | **98.4%** |
| **`audit_hint`** | 62.9% | `19.0 / 47.2 / 21.5 / 12.4` | **+6.56%** `[+1.8; +11.3]` | **+7.27%** `[+2.5; +12.1]` | 54.7% `[51.4; 58.0]` | 49.1% | **98.4%** |
| **`audit_hint_nt`** *(tắt think)* | 78.5% | `13.8 / 64.5 / 9.9 / 11.7` | **+2.13%** `[-2.0; +6.2]` | **+4.26%** `[+0.0; +8.7]` | 52.3% `[48.9; 55.6]` | 47.5% | 0% *(no_think)* |

---

### Bảng 2.2: Kết quả OpenVul-4B trên 564 cặp PrimeVul ([`openvul_test/pair_protocol.json`](experiments/exp5_line_labels/results/openvul_test/pair_protocol.json) & [`compare_arms.json`](experiments/exp5_line_labels/results/openvul_test/compare_arms.json))

| Nhánh | Tập mẫu | `flag_rate` (`t0`) | `P-C / P-V / P-B / P-R` (`t0`) | `Net @ t0` [CI 95%] | `Net @ t_median` [CI 95%] | `rank_acc` [CI 95%] | `think_forced` |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`plain`** | ALL (564) | 66.0% | `19.5 / 51.2 / 19.1 / 10.1` | **+9.40%** `[+5.1; +13.8]` | **+4.43%** `[-0.4; +8.9]` | **55.7%** `[52.4; 59.0]` | **40.0%** (450/1126) |
| **`plain`** | CLEAN (408) | 64.7% | `19.9 / 49.5 / 20.1 / 10.5` | **+9.31%** `[+3.9; +14.5]` | — | — | — |
| **`audit`** | ALL (564) | 76.7% | `14.2 / 63.8 / 10.8 / 11.2` | **+3.01%** `[-1.2; +7.3]` | **+0.71%** `[-3.5; +4.8]` | 50.8% `[47.4; 54.1]` | **74.2%** (835/1126) |
| **`audit`** | CLEAN (408) | 75.9% | `12.3 / 63.7 / 12.3 / 11.8` | **+0.49%** `[-4.4; +5.4]` | — | — | — |
| **`audit_hint`** | ALL (564) | 67.8% | `18.6 / 53.0 / 17.4 / 11.0` | **+7.62%** `[+3.2; +12.1]` | **+5.50%** `[+0.5; +10.3]` | 55.1% `[52.0; 58.6]` | **61.8%** (696/1126) |
| **`audit_hint`** | CLEAN (408) | 66.8% | `18.1 / 51.2 / 17.9 / 12.7` | **+5.39%** `[+0.0; +10.5]` | — | — | — |
| **`audit_hint`** | LEAKED (156) | 70.8% | `19.9 / 57.7 / 16.0 / 6.4` | **+13.46%** `[+6.4; +21.8]` | — | — | — |

**Kiểm định chênh lệch trực tiếp trên OpenVul-4B ([`openvul_test/compare_arms.json`](experiments/exp5_line_labels/results/openvul_test/compare_arms.json)):**
* `audit_hint - audit` (ALL 564): $\Delta\text{Net} = +4.6\%$ `[-1.4%; +10.6%]`, $\Delta\text{rank\_acc} = +4.3\%$ `[-0.3%; +9.0%]` *(khoảng tin cậy cắt 0, và bị nhiễu bởi chênh lệch `think_forced` 61.8% vs 74.2%)*.
* `audit - plain` (ALL 564): $\Delta\text{Net} = -6.4\%$ `[-12.2%; -0.5%]`, $\Delta\text{rank\_acc} = -4.9\%$ `[-9.7%; -0.4%]` *(đổi từ prompt `plain` sang `audit` làm giảm điểm có ý nghĩa thống kê)*.
* `audit_hint - plain` (ALL 564): $\Delta\text{Net} = -1.8\%$ `[-8.2%; +4.4%]`, $\Delta\text{rank\_acc} = -0.6\%$ `[-5.1%; +4.3%]` *(không thắng được `plain`)*.

---

## 3. ĐÁNH GIÁ CHUNG: ĐIỀU GÌ NÓI ĐƯỢC VÀ ĐIỀU GÌ CHƯA NÓI ĐƯỢC?

1. **Điều ĐÃ NÓI ĐƯỢC (Có bằng chứng đối chứng sạch):**
   * *"Định vị dòng của ModernBERT hướng sự chú ý của LLM vào dòng lỗi thật tốt hơn hẳn so với gợi ý ngẫu nhiên (`41.0%` vs `29.5%` ở `any_sink_is_a_flaw_line`, và `24.7%` vs `15.4%` ở `a_missing_sink_is_a_flaw_line`)."* $\rightarrow$ Đây là một kết quả nhỏ nhưng thật, vượt qua đối chứng `audit_rand`.
   * *"Kiểm toán rò rỉ dữ liệu phát hiện 156/564 cặp (27.7%) của PrimeVul Paired trùng commit với tập huấn luyện SFT của OpenVul, giải thích vì sao điểm trên tập `LEAKED` (`+13.46%`) cao vọt so với tập `CLEAN` (`+5.39%` ở `audit_hint`, `+9.31%` ở `plain`)."*
2. **Điều CHƯA NÓI ĐƯỢC (Không được phép kết luận từ dữ liệu hiện tại):**
   * **Chưa nói được:** *"Gợi ý dòng cải thiện phát hiện lỗ hổng mức hàm."*
   * Số liệu hiện có ủng hộ điều ngược lại: Ở mức hàm, phần biến động điểm số chủ yếu đến từ việc **đổi cấu trúc prompt** (`plain` vs `audit`), **thay đổi tỷ lệ gắn cờ (`flag_rate`)**, và **tỷ lệ bị cắt suy nghĩ (`think_forced`)**, chứ không phải từ nội dung dòng gợi ý (bằng chứng là `audit_hint` = `audit_rand` = `+6.56%` trên Qwen ở `t0`, và thua `audit` ở `t_median`).

---

## 4. THIẾT KẾ THÍ NGHIỆM TIẾP THEO ĐỂ KIỂM ĐỊNH NGHIÊM NGẶT GIẢ THUYẾT MỨC HÀM

Để trả lời dứt điểm câu hỏi *"Liệu gợi ý dòng của ModernBERT có thực sự giúp cải thiện phán quyết mức hàm khi không bị nhiễu bởi prompt dài và cắt suy nghĩ hay không?"*, lượt chạy tiếp theo trên **OpenVul-4B** cần tuân thủ giao thức đăng ký trước (pre-registered protocol) như sau:

1. **Đầy đủ các nhánh đối chứng (Arms):**
   * `plain`: Hỏi trực tiếp không gợi ý.
   * `plain_hint`: Cùng prompt `plain` gọn nhẹ + gắn $K=3$ dòng gợi ý từ ModernBERT.
   * `plain_rand`: Cùng prompt `plain` gọn nhẹ + gắn $K=3$ dòng code ngẫu nhiên (nhánh đối chứng trực tiếp cho `plain_hint`).
   * `audit_rand`: Bổ sung nhánh đối chứng ngẫu nhiên cho cặp `audit` / `audit_hint` trên OpenVul.
   *(Đã bổ sung sẵn hai nhánh `plain_hint` và `plain_rand` vào [`experiments/exp5_line_labels/llm_audit.py`](experiments/exp5_line_labels/llm_audit.py)).*
2. **Kiểm soát triệt để `think_budget` (Loại bỏ nhiễu `think_forced`):**
   * Đặt `--think_budget` đủ lớn (ví dụ `4096` hoặc `6144` tokens trên vLLM) sao cho tỷ lệ `think_forced` ở mọi nhánh giảm xuống mức tối thiểu ($< 5\%$), đảm bảo mọi phán quyết đều được đưa ra sau khi mô hình tự kết thúc chuỗi suy nghĩ (`think_complete`).
3. **Chốt trước ngưỡng và tiêu chí Qua/Trượt (Pre-registered Pass/Fail Criteria):**
   * Không chọn ngưỡng cắt hay bộ lọc sau khi xem tập test. Chấm cố định trên hai ngưỡng chuẩn của [`pair_protocol.py`](experiments/exp5_line_labels/pair_protocol.py): **`t0` (ngưỡng 0 mặc định của mô hình)** và **`t_median` (50% flag rate)** cùng chỉ số không phụ thuộc ngưỡng **`rank_acc`** trên cả tập `ALL (564)` và `CLEAN (408)`.
   * **Tiêu chí Qua (Pass) ở mức dòng:** `audit_hint` phải vượt `audit_rand` có ý nghĩa thống kê trên `any_sink_is_a_flaw_line` và `a_missing_sink_is_a_flaw_line`.
   * **Tiêu chí Qua (Pass) ở mức hàm:** `plain_hint` phải đồng thời vượt cả **`plain_rand`** (chứng minh chất lượng dòng gợi ý có tác dụng thật chứ không phải hiệu ứng prompt) lẫn **`plain`** (chứng minh thêm gợi ý tốt hơn không gợi ý) trên `Net` (`t0` / `t_median`) và `rank_acc`.
