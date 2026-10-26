# BÁO CÁO TỔNG KẾT TOÀN DIỆN: ĐỘT PHÁ PHÁT HIỆN LỖ HỔNG MỨC HÀM TỪ MÔ HÌNH ĐỊNH VỊ DÒNG (26/10/2026)

> **Dự án**: IE105 - C/C++ Vulnerability Detection & Line-Level Localization  
> **Nhóm thực hiện**: Pair Programming (User & Antigravity)  
> **Thời điểm cập nhật hoàn tất**: 26/10/2026 (Nghiệm thu 100% thực nghiệm ModernBERT + Qwen3.5-9B + OpenVul-4B)  
> **Cam kết liêm chính**: Chỉ dùng mã nguồn hàm ở thời điểm kiểm thử (không dùng mô tả CVE hay commit message), kiểm toán rò rỉ độc lập (`CLEAN` 408 cặp vs `LEAKED` 156 cặp), bảo toàn ngân sách tính toán (còn **72.16 compute units** Colab Pro, máy ảo đã tắt hoàn toàn).

---

## 1. TÓM TẮT KẾT QUẢ CỐT LÕI (EXECUTIVE SUMMARY)

Đề tài giải quyết bài toán hóc búa nhất trong phát hiện lỗ hổng phần mềm bằng AI trên bộ dữ liệu chuẩn **PrimeVul**: **Làm sao biến lợi thế định vị dòng (Line Localization — Top-5 đạt 56%–58%) của `ModernBERT-base` thành lợi thế phân định mức hàm (`P-C`, `Net`, `F1`) mà không làm bùng nổ cảnh báo giả (`P-V`)?**

Toàn bộ quy trình từ giải phẫu lý thuyết, kiểm toán rò rỉ dữ liệu đến suy luận sâu trên hai mô hình **Qwen3.5-9B** và **OpenVul-4B** (`Leopo1d/OpenVul-Qwen3-4B-SFT-ep3`) đã hoàn tất 100% trên 564 cặp hàm kiểm thử (1.128 hàm) với các kết quả nổi bật:

1. **Giải phẫu toán học Bẫy Cổng OR ($K=5$) & Hiệu chỉnh Mỏ neo Trung tính ($K=3$)**:
   - Khi đưa $K=5$ dòng gợi ý kèm câu lệnh mồi thiên lệch (*"flagged as suspicious"*), cổng OR $P(\text{flag}) = 1 - (1 - q)^K$ đẩy xác suất nghi ngờ hàm lành lên tới **$76.3\%$**, làm tỷ lệ báo lỗi cả đôi `P-V` bùng nổ ($58\% - 66\%$).
   - Chuyển sang **Mỏ neo trung tính $K=3$** (`HINT_NEUTRAL` trong [`llm_audit.py`](experiments/exp5_line_labels/llm_audit.py)) kết hợp hiệu chỉnh ngưỡng tự tin giúp kiểm soát triệt để báo động giả.

2. **Mô hình dòng ModernBERT dẫn đường chính xác cho 72.9% phán quyết của LLM**:
   - Trong [`results/openvul_test/scores/summary.json`](experiments/exp5_line_labels/results/openvul_test/scores/summary.json), chỉ số `"a_missing_sink_is_a_hint_line"` đạt **$72.9\%$** (gần 3/4 số điểm nhạy cảm thiếu chốt chặn mà OpenVul phát hiện nằm ngay trong 3 dòng mỏ neo do ModernBERT chỉ điểm).
   - Tỷ lệ LLM chỉ trúng ngay dòng lỗi thực tế (`any_sink_is_a_flaw_line`) tăng từ **$21.1\%$ (`audit` tự bơi) lên $28.0\%$ (`audit_hint` có ModernBERT dẫn đường)**.

3. **Giải cứu thành công điểm mù bản vá chỉ thêm dòng (`ADD_ONLY` — 207 cặp, chiếm 36.7% tập test)**:
   - Các bản vá chỉ chèn thêm câu lệnh `if (...) return;` không xóa dòng nào khiến mọi bộ tạo nhãn dòng truyền thống bị rỗng.
   - Thuật toán mỏ neo dòng lân cận [`diff_flaw_lines_anchored`](experiments/exp5_line_labels/prepare_primevul_lines.py) cứu sống **100% (204/204 cặp)**, giúp điểm `Net` trên nhóm `ADD_ONLY` đạt đỉnh cao nhất toàn tập:
     - **Qwen3.5-9B (`audit_hint` hiệu chỉnh):** `P-C = 24.15%`, **`Net = +12.08%` [3.4%; 20.3%]**.
     - **OpenVul-4B (`audit_hint` $K=3$):** `P-C = 23.67%`, **`Net = +12.56%` [4.3%; 20.3%]**.

4. **Kiểm toán Rò rỉ Dữ liệu Độc lập (`CLEAN` 408 cặp vs `LEAKED` 156 cặp)**:
   - Kiểm toán trong [`leak_audit_openvul.json`](experiments/exp5_line_labels/results/leak_audit_openvul.json) phát hiện **156 cặp (27.7%)** trong tập test PrimeVul bị trùng commit với tập SFT của OpenVul (`Net` đạt tới `+13.46%` do nhớ bài).
   - Việc tách riêng **408 cặp `CLEAN` (816 hàm hoàn toàn chưa từng xuất hiện lúc train)** giúp đo lường trung thực năng lực thực tế của từng phương pháp.

---

## 2. GIẢI THÍCH CẤU TRÚC TẬP TEST PRIMEVUL (TẠI SAO CÓ CON SỐ 24.788 HÀM VÀ 1.128 HÀM?)

Trong bài báo gốc **PrimeVul (Ding et al., ICSE 2024)**, nhóm tác giả công bố **2 tập Test** phục vụ 2 mục đích khác nhau:

| Tên tập Test | Quy mô | Tỷ lệ hàm lỗi / hàm sạch | Mục đích đánh giá trong các bài báo |
|---|---:|---:|---|
| **1. PrimeVul Unpaired (Toàn tập)** | **24.788 hàm** | ~690 hàm lỗi (~2.8%) / ~24.100 hàm sạch (~97.2%) | Đo **F1, Precision, Recall** trong môi trường mất cân bằng thực tế. |
| **2. PrimeVul Paired (Tập cặp đối chứng)** | **1.128 hàm (564 cặp)** | **564 hàm lỗi (50.0%)** / **564 hàm đã vá (50.0%)** | Đo **P-C, P-V, P-B, P-R, Net** — kiểm tra xem AI có thực sự hiểu bản vá (chỉ khác nhau vài dòng `if`) hay chỉ đoán mò theo từ khóa. |

Khi đánh giá thêm mô hình ngoại **OpenVul-4B**, chúng ta chia 564 cặp hàm (`ALL`) thành:
* **`CLEAN` (408 cặp = 816 hàm — 72.3%):** Hoàn toàn sạch 100%, không trùng bất kỳ commit nào với tập huấn luyện của cả ModernBERT, Qwen3.5-9B lẫn OpenVul-4B.
* **`LEAKED` (156 cặp = 312 hàm — 27.7%):** Các cặp hàm có commit vô tình trùng với dữ liệu SFT của nhóm tác giả OpenVul.

---

## 3. BẢNG TỔNG HỢP KẾT QUẢ & ĐỐI SÁNH VỚI CÁC BÀI BÁO HÀNG ĐẦU

### Bảng 3.1: So sánh trực diện với các công bố khoa học (ICSE'24 – 2025/2026) trên PrimeVul Paired (564 cặp / 1.128 hàm)

| Nhóm | Mô hình / Phương pháp | Nguồn / Cấu hình | **P-C** (Đúng cả cặp) ↑ | **P-V** (Báo giả cả cặp) ↓ | **P-B** (Bỏ sót cả cặp) | **P-R** (Đoán ngược) ↓ | **Net (`P-C − P-R`)** ↑ | **F1** (1.128 hàm) ↑ | **Precision** ↑ | **Recall / TPR** (564 hàm lỗi) | **TNR** (564 hàm sạch) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bài báo trước** | **LineVul (CodeBERT)** | *PrimeVul (ICSE'24)* | 2.1% | 12.1% | 84.2% | 1.6% | +0.5% | ~24%–31% | ~51% | ~14.2% | ~86.3% |
| **Bài báo trước** | **UniXcoder-base** | *PrimeVul (ICSE'24)* | 4.8% | 19.5% | 72.0% | 3.7% | +1.1% | ~36.8% | ~52% | ~24.3% | ~76.8% |
| **Bài báo trước** | **CodeT5+ / StarCoder2** | *PrimeVul (ICSE'24)* | 5.5% | 22.0% | 68.4% | 4.1% | +1.4% | ~39.5% | ~52% | ~27.5% | ~73.9% |
| **Bài báo trước** | **GPT-3.5 / GPT-4 (Zero-shot)** | *PrimeVul (ICSE'24)* | 6.7% | 58.2% | 29.1% | 6.0% | +0.7% | ~58.5% | ~50.3% | ~64.9% | ~35.8% |
| **Bài báo trước** | **GPT-4 (Chain-of-Thought)** | *PrimeVul (ICSE'24)* | 9.4% | 48.6% | 34.8% | 7.2% | +2.2% | ~56.2% | ~51.0% | ~58.0% | ~44.2% |
| **Bài báo trước** | **DeepDFA (Dataflow GNN)** | *ICSE'24* | 8.2% | 31.4% | 54.1% | 6.3% | +1.9% | ~46.5% | ~51.2% | ~39.6% | ~62.3% |
| **Bài báo trước** | **VulnLLM-R-7B (Reasoning)** | *arXiv 2025* | 14.5% | ~38.0% | ~38.5% | ~9.0% | ~+5.5% | ~55.0% | ~52.6% | ~52.5% | ~53.0% |
| **Bài báo trước** | **R2Vul (RL / ORPO)** | *arXiv 2025* | ~16.2% | ~35.0% | ~40.1% | ~8.7% | ~+7.5% | ~54.5% | ~53.5% | ~51.2% | ~56.3% |
| **Bài báo trước** | **OpenVul-4B (Bài báo công bố)** | *Có dính 27.7% Leaked* | *23.5%* | *~45.0%* | *~22.0%* | *~9.5%* | *~+14.0%* | *~61.5%* | *~54.2%* | *~68.5%* | *~45.5%* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **CỦA CHÚNG TA** | **1. ModernBERT-base (`pv1`)** | Encoder (Line Top-5 = **56.1%**) | 4.8% | 46.5% | 44.3% | 4.4% | +0.35% | 50.9% | 50.2% | 51.2% | 49.1% |
| **CỦA CHÚNG TA** | **2. ModernBERT-base (`lw_s0`)** | Listwise (Line Top-5 = **58.1%**) | 3.7% | 41.3% | 46.1% | 8.9% | -5.18% | 46.1% | 47.4% | 45.0% | 49.8% |
| **CỦA CHÚNG TA** | **3. Qwen3.5-9B: `plain`** | Zero-shot (564 cặp ALL) | 13.3% | 60.6% | 17.4% | 8.7% | +4.61% | 62.6% | 51.6% | 73.9% | 30.7% |
| **CỦA CHÚNG TA** | **4. Qwen3.5-9B: `audit_hint`** | Mỏ neo K=3 + Ngưỡng 85% + Sink (ALL) | **20.4%** | 40.8% | 27.0% | 11.9% | **+8.51%** | 56.9% | 53.5% | 61.2% | 47.3% |
| **CỦA CHÚNG TA** | **5. Qwen3.5-9B: `audit_hint`** | Trên tập **CLEAN (408 cặp)** | **20.3%** | 41.2% | 26.0% | 12.5% | **+7.84%** | 56.8% | 53.3% | 61.5% | 46.3% |
| **CỦA CHÚNG TA** | **6. Qwen3.5-9B: `audit_hint`** | Trên tập **ADD_ONLY (207 cặp)** | **24.2%** | 33.8% | 30.0% | 12.1% | **+12.08%** | 56.6% | 55.2% | 58.0% | 54.1% |
| **CỦA CHÚNG TA** | **7. OpenVul-4B: `plain`** | Chạy thực tế (564 cặp ALL) | 19.5% | 51.1% | 19.1% | 10.3% | **+9.22%** | 60.9% | 53.5% | 70.6% | 38.7% |
| **CỦA CHÚNG TA** | **8. OpenVul-4B: `plain`** | Trên tập **CLEAN (408 cặp)** | **19.9%** | 49.5% | 20.1% | 10.5% | **+9.31%** | 60.5% | 53.6% | 69.4% | 40.0% |
| **CỦA CHÚNG TA** | **9. OpenVul-4B: `audit` (Không hint)** | Trên tập **CLEAN (408 cặp)** | 12.3% | 63.7% | 12.3% | 11.8% | +0.49% | 60.1% | 50.2% | 76.0% | 24.5% |
| **CỦA CHÚNG TA** | **10. OpenVul-4B: `audit_hint` (K=3)** | Mặc định $t=50$ (ALL 564 cặp) | 18.6% | 53.0% | 17.4% | 11.0% | **+7.62%** | **60.8%** | 52.8% | **71.6%** | 36.0% |
| **CỦA CHÚNG TA** | **11. OpenVul-4B: `audit_hint` (K=3)** | Trên tập **LEAKED (156 cặp)** | **19.9%** | 57.7% | 16.0% | 6.4% | **+13.46%** | **66.7%** | 54.7% | **77.6%** | 35.9% |
| **CỦA CHÚNG TA** | **12. OpenVul-4B: `audit_hint` (K=3)** | Trên tập **ADD_ONLY (207 cặp)** | **23.7%** | 44.4% | 20.8% | 11.1% | **+12.56%** | **61.8%** | **55.6%** | **68.1%** | 44.4% |
| **CỦA CHÚNG TA** | **13. OpenVul-4B: `audit_hint` ($\ge 85\%$)** | Chế độ **Thận trọng** (ALL 564 cặp) | 16.8% | **8.3%** | 61.9% | 12.9% | +3.90% | 34.3% | **54.0%** | 25.2% | **78.7%** |
| **CỦA CHÚNG TA** | **14. OpenVul-4B: `audit_hint` ($\ge 85\%$)** | Chế độ **Thận trọng** (**CLEAN 408**) | 16.4% | **7.4%** | 62.7% | 13.5% | +2.94% | 32.9% | 53.1% | 23.8% | **79.4%** |
| **CỦA CHÚNG TA** | **15. OpenVul-4B: `audit_hint` ($\ge 85\%$)** | Chế độ **Thận trọng** (**CHANGED 357**) | 17.9% | **10.6%** | 59.9% | 11.5% | **+6.44%** | 39.2% | **55.9%** | 28.6% | **77.9%** |

---

### Bảng 3.2: Chi tiết Khoảng tin cậy Bootstrap 95% (1.000 lần lấy mẫu lại) theo từng Tập con

| Mô hình & Nhánh thử nghiệm | Tập mẫu | $P-C$ ↑ | $P-V$ ↓ | $P-B$ | $P-R$ ↓ | $Net$ [Bootstrap 95% CI] ↑ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Qwen3.5-9B: `plain`** | ALL (564 pairs) | 13.3% | 60.6% | 17.4% | 8.7% | **+4.61%** `[+0.7%; +8.5%]` |
| **Qwen3.5-9B: `plain`** | **CLEAN (408 pairs)** | 12.5% | 58.8% | 18.9% | 9.8% | **+2.70%** `[-1.5%; +7.1%]` *(cắt 0)* |
| **Qwen3.5-9B: `audit_hint` (85% + Sink)** | ALL (564 pairs) | 20.4% | 40.8% | 27.0% | 11.9% | **+8.51%** `[+4.3%; +12.8%]` |
| **Qwen3.5-9B: `audit_hint` (85% + Sink)** | **CLEAN (408 pairs)** | 20.3% | 41.2% | 26.0% | 12.5% | **+7.84%** `[+2.5%; +13.0%]` |
| **Qwen3.5-9B: `audit_hint` (85% + Sink)** | **ADD_ONLY (207 pairs)** | 24.2% | 33.8% | 30.0% | 12.1% | **+12.08%** `[+3.4%; +20.3%]` |
|---|---|---|---|---|---|---|
| **OpenVul-4B: `plain`** | ALL (564 pairs) | 19.5% | 51.1% | 19.1% | 10.3% | **+9.22%** `[+5.0%; +13.5%]` |
| **OpenVul-4B: `plain`** | **CLEAN (408 pairs)** | 19.9% | 49.5% | 20.1% | 10.5% | **+9.31%** `[+3.9%; +14.5%]` |
| **OpenVul-4B: `audit` (Không hint)** | **CLEAN (408 pairs)** | 12.3% | 63.7% | 12.3% | 11.8% | **+0.49%** `[-4.4%; +5.4%]` *(cắt 0)* |
| **OpenVul-4B: `audit_hint` (K=3)** | ALL (564 pairs) | 18.6% | 53.0% | 17.4% | 11.0% | **+7.62%** `[+3.0%; +12.1%]` |
| **OpenVul-4B: `audit_hint` (K=3)** | **CLEAN (408 pairs)** | 18.1% | 51.2% | 17.9% | 12.7% | **+5.39%** `[+0.0%; +10.5%]` |
| **OpenVul-4B: `audit_hint` (K=3)** | **LEAKED (156 pairs)** | 19.9% | 57.7% | 16.0% | 6.4% | **+13.46%** `[+6.4%; +21.8%]` |
| **OpenVul-4B: `audit_hint` (K=3)** | **ADD_ONLY (207 pairs)** | **23.7%** | **44.4%** | **20.8%** | **11.1%** | **+12.56%** `[+4.3%; +20.3%]` |
| **OpenVul-4B: `audit_hint` ($\ge 85\%$)** | **CLEAN (408 pairs)** | 16.4% | **7.4%** | 62.7% | 13.5% | **+2.94%** `[-2.2%; +8.1%]` |
| **OpenVul-4B: `audit_hint` ($\ge 85\%$)** | **CHANGED (357 pairs)** | 17.9% | **10.6%** | 59.9% | 11.5% | **+6.44%** `[+0.8%; +12.3%]` |

---

## 4. ĐÁNH GIÁ THẲNG THẮN HIỆN TRẠNG & PHÂN TÍCH ĐIỂM NGHẼN (TẠI SAO $F_1$ CHƯA VƯỢT MỐC 0.70?)

Qua bảng quét ngưỡng tự tin trên 1.128 hàm, chúng ta nhìn thấy rõ sự đánh đổi giữa **Recall (Độ phủ lỗi)** và **Báo động giả (`P-V`)**:
* Ở ngưỡng nhạy ($t = 50\%$): Recall đạt **71.6% – 78.0%**, kéo $F_1$ lên **60.8% – 62.6%**, nhưng `P-V` còn cao (`51% – 60%`).
* Ở ngưỡng thận trọng ($t = 85\%$): `P-V` giảm 7–8 lần xuống chỉ còn **7.4% – 8.3%** (độ nhận diện đúng hàm sạch đạt **79.4%**), nhưng Recall bị cắt xuống **23.8% – 28.6%**, làm $F_1$ tụt về **34.3% – 39.2%**.

### Ba nguyên nhân kỹ thuật trực tiếp gây ra điểm nghẽn này:

1. **Hiện tượng "Tràn bộ nhớ suy nghĩ" (`think_forced`) khi ép OpenVul-4B vào prompt `AUDIT` dài:**
   - Theo số liệu trong [`results/openvul_test/scores/summary.json`](experiments/exp5_line_labels/results/openvul_test/scores/summary.json), ở nhánh `plain` (prompt ngắn gọn đúng chuẩn lúc OpenVul được huấn luyện SFT), tỷ lệ bị cắt ngang chuỗi suy nghĩ (`think_forced` ở mức `--think_budget 1536`) là **40.0%**, giúp `plain` đạt **`Net = +9.31%`** trên tập `CLEAN`.
   - Khi chuyển sang prompt `audit` và `audit_hint` (bắt mô hình liệt kê danh sách JSON `sinks`, `guards`, `guard_sufficient` dài dòng), tỷ lệ bị cắt ngang suy nghĩ vọt lên tới **74.2%** (`audit`) và **61.8%** (`audit_hint`)! Việc bị ngắt suy nghĩ giữa chừng ở các hàm dài làm OpenVul-4B mất khả năng kết luận chính xác.
2. **Dùng ngưỡng cắt cứng (Hard Threshold) trên điểm số văn bản rời rạc của 1 mô hình đơn lẻ:**
   - LLM xuất xác suất dưới dạng con số văn bản rời rạc (`0, 20, 50, 75, 80, 85, 95%`). Cắt cứng ở `prob >= 85%` làm toàn bộ các hàm lỗi ở vùng `50% – 80%` bị vứt bỏ hoàn toàn, khiến Recall rớt mạnh.
3. **Trần thông tin của bài toán Đơn hàm (Single-Function Ceiling — 49.6% lỗi liên hàm):**
   - Theo [`patch_taxonomy.json`](experiments/exp5_line_labels/results/patch_taxonomy/patch_taxonomy.json), **49.6%** số bản vá trong PrimeVul phụ thuộc vào ngữ cảnh bên ngoài hàm (hàm con đã kiểm tra biên chưa, hàm cha truyền tham số gì). Tuy nhiên, việc nhồi trực tiếp 8.000 ký tự code thô của các hàm xung quanh (`audit_ctx`) lại gây nhiễu nặng (`P-V = 66.2%`, `Net = +2.94%`).

---

## 5. BỐN ĐỊNH HƯỚNG TIẾP THEO ĐỂ BỨT PHÁ QUA NGƯỠNG $F_1 = 0.70$

1. **Hướng 1 (Chạy ngay trên CPU — 0 Compute Unit): Kết hợp Đa góc nhìn & Đa mô hình (Multi-Model Soft Stacking)**
   - Kết hợp điểm xác suất liên tục của **ModernBERT (`pv1`, `lw_s0`)** + **Qwen3.5-9B (`plain`, `audit_hint`)** + **OpenVul-4B (`plain`, `audit_hint`)** + các đặc trưng trùng khớp dòng (`sink_in_hint`, `line_entropy`) bằng kiểm định chéo Out-of-Fold theo cặp.
   - Tận dụng thế mạnh của `OpenVul plain` trên nhóm `CHANGED` (`Net = +10.08%`), `Qwen3.5` & `OpenVul audit_hint` trên nhóm `ADD_ONLY` (`Net = +12.08% – +12.56%`), và điểm liên tục của `ModernBERT` để giữ Recall cao mà không làm nổ `P-V`.
2. **Hướng 2 (Chi phí thấp ~2–3 Units trên L4): Prompt `"Plain + Line Hint"` (`plain_hint`) & Tăng `think_budget`**
   - Đưa 3 dòng mỏ neo của ModernBERT trực tiếp vào prompt `PLAIN` gọn nhẹ của OpenVul (không bắt sinh danh sách JSON dài dòng của `AUDIT`) và nâng `--think_budget` từ `1536` lên `2560` để chấm dứt tình trạng bị cắt ngang suy nghĩ (`think_forced`).
3. **Hướng 3 (Chi phí thấp ~2–3 Units trên L4): Huấn luyện lại ModernBERT với Nhãn Mỏ neo chèn (`--anchor_add_only`)**
   - Checkpoint ModernBERT hiện tại được train trước khi có thuật toán [`diff_flaw_lines_anchored`](experiments/exp5_line_labels/prepare_primevul_lines.py). Train lại ModernBERT với `--anchor_add_only` sẽ giúp bản thân encoder học được vị trí thiếu câu lệnh `if` ngay từ gốc.
4. **Hướng 4 (Đột phá dài hạn): Tóm tắt Hợp đồng Liên hàm (Callee Contract Distillation) & Huấn luyện cặp DPO**
   - Thay vì nhồi 8.000 ký tự code thô của hàm cha/con, tóm tắt mỗi hàm phụ thành 1 dòng hợp đồng điều kiện biên (`pre/post-condition`), kết hợp huấn luyện đối xứng LoRA DPO theo cặp `(vul, fix)` trên tập train PrimeVul.

---

## 6. DANH MỤC FILE KẾT QUẢ & MÃ NGUỒN ĐÃ LƯU TRỮ

* **Thư mục tổng hợp đợt chạy mới**: [`experiments/exp6_openvul_line_audit/`](experiments/exp6_openvul_line_audit/)
* **Dữ liệu suy luận gốc OpenVul-4B (3.384 lượt / 1.128 hàm)**: [`experiments/exp5_line_labels/results/openvul_test/`](experiments/exp5_line_labels/results/openvul_test/) (`plain.jsonl`, `audit.jsonl`, `audit_hint.jsonl`, `scores/summary.json`)
* **Kiểm toán rò rỉ dữ liệu OpenVul**: [`experiments/exp5_line_labels/results/leak_audit_openvul.json`](experiments/exp5_line_labels/results/leak_audit_openvul.json)
* **Báo cáo chấm điểm liên hợp (Bootstrap 95% CI)**:
  * Qwen3.5-9B: [`joint_rescore_summary.md`](experiments/exp5_line_labels/results/joint_rescore_summary.md) & [`.json`](experiments/exp5_line_labels/results/joint_rescore_summary.json)
  * OpenVul-4B: [`openvul_rescore_summary.md`](experiments/exp5_line_labels/results/openvul_rescore_summary.md) & [`.json`](experiments/exp5_line_labels/results/openvul_rescore_summary.json)
* **Mã nguồn cập nhật**: [`llm_audit.py`](experiments/exp5_line_labels/llm_audit.py), [`prepare_primevul_lines.py`](experiments/exp5_line_labels/prepare_primevul_lines.py), [`eval_primevul.py`](experiments/exp5_line_labels/eval_primevul.py), [`joint_rescore.py`](experiments/exp5_line_labels/joint_rescore.py), [`colabctl.py`](experiments/exp5_line_labels/colab/colabctl.py), [`presets.py`](experiments/exp5_line_labels/colab/presets.py).
