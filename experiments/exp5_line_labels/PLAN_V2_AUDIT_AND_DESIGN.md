# Đánh Giá Toàn Diện, Phản Biện Plan Cũ & Thiết Kế Thử Nghiệm Mới (Plan v2)

> **Phạm vi**: `experiments/exp5_line_labels` & `experiments/exp6_openvul_line_audit`  
> **Trạng thái tài liệu**: Cập nhật ngày 20/11/2026 — **GĐ 0 (`0.02 CU`) · GĐ 1 Sanity (`0.44 CU`) · GĐ 1A Partial Early-Stop (`1.60 CU`, L4)** đã hoàn tất & đóng hết VM (`0` active, còn **`65.23 CU`**). Xem Mục 5 cho Nhật ký Quyết định & Đề xuất xoay trục Plan v2.2 *(bảo lưu nguyên trạng [`FINAL_REPORT.md`](../../FINAL_REPORT.md) làm mốc lịch sử).*

---

## 1. Nền tảng & hai kết quả đã đứng vững từ Exp1–Exp6

Qua chuỗi thí nghiệm Exp1–Exp6 trên hai bộ dữ liệu C/C++ chuẩn thực tế (**SecVulEval** có nhãn dòng và **PrimeVul** chia theo thời gian, khử trùng lặp, tỉ lệ dương 2.7%, 564 cặp *vulnerable/fixed*), trong điều kiện kiểm thử **code-only** (chỉ dùng mã nguồn hàm, không dùng commit message, mô tả CVE hay bản vá):

1. **Định vị dòng (Line-Level AVL) bằng ModernBERT 2048 mang tín hiệu thật và dẫn đầu mốc mở:**
   - Giải quyết điểm mù cắt đầu 512 token của LineVul (vốn làm 56.6% hàm lỗi PrimeVul có `Top-10 = 0%`).
   - Trên SecVulEval test (`n = 1.037` hàm): đạt `Top-10 = 68.8%–71.5%` (vượt `longest-line` `62.0%` và `random` `48.7%`).
   - Trên PrimeVul (`n = 357` hàm có nhãn dòng): đạt `Top-1/5/10 = 16.8–20.4% / 45.1–50.1% / 58.8–61.6%`, vượt đối chứng `longest-line` (`10.4% / 35.3% / 52.4%`) và vượt mốc công bố của ANVIL (`41.4%` Top-5, line ROC `0.618` so với `0.754` của `s13_conv`).
   - Khi đưa làm gợi ý cho **Qwen3.5-9B**, gợi ý từ ModernBERT kéo tỉ lệ chỉ trúng dòng lỗi thật tăng mạnh so với gợi ý ngẫu nhiên (`audit_rand`):
     - `any_sink_is_a_flaw_line`: **`41.0%` vs `29.5%` (+11.5 điểm %)**
     - `a_missing_sink_is_a_flaw_line`: **`24.7%` vs `15.4%` (+9.3 điểm %)**

2. **Giải phẫu lối tắt độ dài (Length Shortcut) & trần của encoder 150M ở mức hàm:**
   - Hướng tương quan độ dài bị đảo ngược giữa SecVulEval và PrimeVul: trong SecVulEval bản vá dài hơn 74% cặp; trong PrimeVul bản vá dài hơn 87.0% cặp.
   - Mọi encoder 150M đều bám độ dài (Spearman `−0.75` đến `−0.88` khi train trên SecVulEval), khiến ROC trên PrimeVul full test bị đảo ngược (`0.23–0.36`); sau khi trừ độ dài (`debias_length.py`), mọi công thức loss/kiến trúc/gộp điểm đều về mức ngẫu nhiên (`0.49–0.53`). Điều này khớp với tổng hợp ~90 bài trong [`khao_sat_tai_lieu.md`](khao_sat_tai_lieu.md): encoder 125–220M chạm trần ở mức hàm, muốn vượt trần cần LLM suy luận + cơ chế xác minh hoặc ngữ cảnh chọn lọc.

---

## 2. Phản biện định lượng Plan cũ: 5 vấn đề phát hiện từ dữ liệu & mã nguồn

Trước khi chi thêm compute unit, kiểm tra trực tiếp trên các file JSONL gốc (`llm_full_a100`, `b8k_testpairs`, `openvul_test`, `eval_pv1_best`) và code hiện hành cho thấy 5 lỗ hổng nghiêm trọng trong các kế hoạch trước:

### 2.1. Đề xuất nâng `--think_budget` cho prompt `AUDIT` (Mục 4 `FINAL_REPORT.md`) đi ngược dữ liệu `b8k_testpairs`
- [`FINAL_REPORT.md`](../../FINAL_REPORT.md#L112-L113) nhận định `think_forced = 98.4%` là yếu tố nhiễu chính ở Qwen3.5-9B và đề xuất nâng `--think_budget` lên `4096–6144` cho cả `audit_rand`.
- Tuy nhiên, kết quả [`b8k_testpairs`](results/b8k_testpairs/scores/summary.json) (đã chạy Qwen3.5-9B `audit` ở budget 8K trong repo) chứng minh điều ngược lại:
  - `think_complete` tăng lên `58.6%` (`649/1108`, cộng thêm `48` `think_reanswer`), `gen_tokens` trung vị vọt từ `2.602` lên `8.362`.
  - Nhưng **`flag_rate` phình từ `76.5%` → `89.6%`**, `missing_sinks_per_function` tăng `1.99 → 2.94`, `P-V` (gắn cờ cả bản lỗi lẫn bản vá) vọt từ `61.9%` → **`83.1%`**, làm **`Net` giảm từ `+5.1%` xuống `+3.5%`** ([`b8k_testpairs/compare_arms.json`](results/b8k_testpairs/compare_arms.json): `audit_8k - audit = -1.6% [-6.5; +3.3]`).
  - Lọc riêng **`206` cặp mà cả hai bản đều `think_complete`**, `P-V` vẫn là **`84.5%`** và `Net = +2.4%`.
- **Nguyên nhân gốc (OR-gate trap):** Prompt [`AUDIT`](llm_audit.py#L78-L92) ép quy tắc *"vulnerable iff có ít nhất 1 sink thiếu check TRONG HÀM NÀY"*. Vì ~30% bản vá thật trong PrimeVul bảo vệ biến phụ thuộc hợp đồng từ caller/callee ngoài hàm ([`manual_labels_test40.json`](results/patch_taxonomy/manual_labels_test40.json)), càng suy nghĩ lâu theo luật OR-gate, mô hình càng tìm ra thêm sink không có check nội hàm → càng over-flag.
- **Trên OpenVul-4B còn gặp 2 lệch pha nữa:**
  1. Đang chấm [`Leopo1d/OpenVul-Qwen3-4B-SFT-ep3`](colab/presets.py#L176) (checkpoint khởi động nguội SFT) thay vì checkpoint RL chủ lực **`OpenVul-Qwen3-4B-GRPO`** (arXiv:2602.14012).
  2. Ép format JSON `AUDIT` gây lệch phân phối prompt làm tụt điểm có ý nghĩa thống kê (`audit - plain = -6.4%`, CI `[-12.2%; -0.5%]`).
- **Quyết định sửa:** **Bỏ hoàn toàn nhánh `audit_rand` cho OpenVul**; với `plain` (nơi 58.5% mẫu `openvul_test` đã xong trong `2.304` token, `p75 = 2.353`), chỉ cần `--think_budget 3072` là giảm mạnh `think_forced` mà tiết kiệm gần nửa GPU so với `6144`.

### 2.2. Giải phẫu vì sao gợi ý dòng tốt (+11.5 pp) vẫn cho `Net` mức hàm đi ngang
Phân tích phân tầng trực tiếp trên `audit_hint.jsonl` của Qwen3.5-9B (`K=5`) và OpenVul-SFT (`K=3`) chỉ ra 3 cơ chế cụ thể:

1. **77–81% gợi ý trùng nhau giữa hai mặt của cặp:** Trung bình **80.9%** (Qwen `K=5`) và **77.3%** (OpenVul `K=3`) văn bản dòng được ModernBERT gợi ý ở bản lỗi **cũng xuất hiện trong Top-K gợi ý của bản đã vá** (bản vá thêm `if (...) return;` ngay trên sink, giữ nguyên dòng sink). Trong một lượt gọi LLM duy nhất, gắn nhãn cảnh báo lên cùng một sink ở cả hai phiên bản tạo *confirmation bias* kéo `P-V` lên.
2. **Gợi ý trúng có lợi, gợi ý trượt kéo tụt phán quyết:** Phân tầng trên Qwen3.5-9B (`K=5`):
   - Nhóm gợi ý trúng ±2 dòng quanh `flaw_lines` (`hit_hunk2`, `n = 219`): `audit_hint` đạt **`Net = +10.0%`**, vượt `audit_rand` (`+6.4%`), `audit` (`+6.4%`), `plain` (`+6.8%`).
   - Nhóm gợi ý trượt (`miss_changed`, `n = 138`): `audit_hint` tụt xuống **`Net = -0.7%`** (thua `plain +3.6%`, `audit +2.2%`). Do LLM bám theo dòng gợi ý tới `68–73%`, gợi ý trượt ở 39–47% hàm có `flaw_lines` triệt tiêu phần tăng của nhóm trúng.
   - Nhóm `add_only` (`n = 207`, chiếm 36.7% cặp PrimeVul): `eval_pv1_best` bị bỏ qua khi train vì [`prepare_primevul_lines.py`](prepare_primevul_lines.py#L6) gán `flaw_lines = []`, nên trên nhóm này `audit_hint` (`+7.7%`) thua `audit_rand` (`+13.0%`).
3. **Gợi ý thô bị dính chùm nhưng `llm_audit.py` chưa bật NMS:** Có tới **57.4%** hàm ở `K=3` (`324/564`) và **84.8%** ở `K=5` (`478/564`) có ít nhất 2 dòng gợi ý nằm sát nhau trong vòng ≤ 3 dòng. Trong khi đó [`diverse_topk_order(radius=3)`](clean_diff_labels.py#L179-L208) đã có sẵn trong repo; kiểm tra trên `eval_pv1_best/test_scores.jsonl` (`n = 357`) cho thấy chỉ bật NMS đã nâng `hunk±2` recall rõ rệt:
   - `K=1`: `31.7% → 31.7%`
   - `K=3`: **`53.5% → 56.3%` (+2.8 pp)**
   - `K=5`: **`61.9% → 68.3%` (+6.4 pp)**
   - `K=10`: **`72.5% → 82.1%` (+9.6 pp)**

### 2.3. Thiếu lực thống kê (`MDE ≈ 10.5 pp`) & lý do chọn Multi-sample Voting thay cho Post-CoT Logprob
- Tính sai số chuẩn ghép cặp (`SE`) trên đúng **408 cặp `CLEAN`** từ các nhánh đã chạy với 1 rollout (`T = 0.6`):
  - `OpenVul audit_hint − plain (CLEAN)`: `SE = 3.80%` → **MDE (80% power) = `10.7` điểm %**
  - `OpenVul audit_hint − audit (CLEAN)`: `SE = 3.66%` → **MDE = `10.3` điểm %**
  - `Qwen audit_hint − audit_rand (CLEAN)`: `SE = 4.03%` → **MDE = `11.3` điểm %**
- Với `n = 408` và 1 rollout nhiễu cao, hiệu ứng đơn lượt ~4–6 điểm gần như chắc chắn bị chìm trong CI `±7.5%`.
- Để thu hẹp phương sai và xóa tỉ lệ cặp hòa (`35–40%` ở bảng điểm rời rạc `{0, 75, 85, 95}`), **không dùng token logprob sau `</think>`** vì xác suất `P("true")` sau chuỗi suy luận dài thường bão hòa về `~0.999`/`~0.001`. Thay vào đó dùng **Multi-sample Self-Consistency (`--n_samples 3`)**: dữ liệu [`compare_arms.json`](results/openvul_test/compare_arms.json#L347-L367) của chính repo (`llm_mean`) chứng minh gộp nhiều lượt suy luận hạ ties từ `33–35%` xuống `10.6%` (OpenVul) và `4.5%` (Qwen `b8k`), đồng thời nâng `rank_acc` OpenVul `55.7% → 59.2%` và `Net` Qwen `+9.6%`.

### 2.4. Hai lỗi đo lường trong `pair_protocol.py` & sắc thái `CLEAN` vs `LEAKED`
1. **`t_median` không giữ được `flag_rate = 50%` trên điểm rời rạc:** Vì [`pair_protocol.py`](pair_protocol.py#L59-L60) cắt theo `pv > t_median`, các nhánh có nhiều điểm trùng ở trung vị bị lệch `flag_rate` rất mạnh (Qwen `audit` `42.9%`, `audit_hint` `35.4%`, `audit_rand` `31.1%`, `audit_hint_nt` `5.9%`; OpenVul `plain` `29.0%`), khiến `Net @ t_median` không so được ngang hàng.
2. **`rank_acc_debiased` phá hòa theo độ dài:** Trước khi sửa, `pair_protocol.py` trừ đường cong độ dài lên cả 35–40% cặp hòa tuyệt đối (`pv == pf`), biến mọi cặp hòa thành thắng/thua chỉ theo độ dài (`47.0–49.6%`). Đã đồng bộ logic giữ hòa `0.5` từ [`compare_arms.py`](compare_arms.py#L28-L32).
3. **Sắc thái `CLEAN (408)` vs `LEAKED (156)`:** Kiểm tra chéo trên **Qwen3.5-9B** (không train SFT OpenVul) cũng cho thấy `LEAKED` có điểm cao hơn `CLEAN` (`plain`: `+8.3%` vs `+2.9%`), trong khi ở `OpenVul plain` thì `CLEAN (+9.6%)` ngang `LEAKED (+9.0%)`. Do đó bắt buộc tiếp tục tách báo cáo `CLEAN` làm thước đo chính, nhưng diễn giải chính xác rằng tập `LEAKED` vừa có nguy cơ trùng train OpenVul vừa chứa các mẫu dễ hơn nói chung.

### 2.5. Lý do hoãn `pv_lw_clean`/`secvul_lw_clean` và hủy train GRPO từ đầu
- Hai preset chưa commit [`pv_lw_clean` và `secvul_lw_clean`](colab/presets.py#L150-L157) gộp 4–5 thay đổi cùng lúc (`diff_clean + collar_radius=2 + norm_maxmean + listwise`, bỏ 80k benigns) và vướng xung đột cờ `if diff_clean and not anchor_add_only` ở [`prepare_primevul_lines.py`](prepare_primevul_lines.py#L48-L52).
- Kế hoạch train on-policy GRPO từ đầu (`42–64 CU`) vượt trần an toàn trên Colab Pro (`67.29 CU`, không có Background Execution khi tắt máy).

---

## 3. Thiết kế Plan v2 (Triển khai đến hết GĐ 2)

### 3.1. Cơ chế trọng tâm: Kiến trúc 2 pha `ModernBERT-NMS Proposer → Prosecutor + Independent Falsifier`
- **Cơ sở tài liệu:** Tổng hợp ~90 bài ở [`khao_sat_tai_lieu.md`](khao_sat_tai_lieu.md#L132) cho thấy ở chế độ chỉ có mã hàm, **xác minh phản biện (verification / falsification)** là đòn bẩy lớn nhất (AEGIS bỏ tranh luận tụt P-C `122 → 57`; VulTrial bỏ moderator tụt `96 → 32`; ReasonVul bỏ abductive tụt `−15.4`), trong khi bước đề xuất manh mối của AEGIS bằng LLM chỉ đạt `16%` recall (`k=1`).
- **Kết hợp thế mạnh của repo:** Dùng **ModernBERT + NMS (`r=3`, K=3)** làm Proposer (`56.3%` hunk±2 recall), rồi tách đôi vai trò xác minh ở LLM thành 2 lượt chat độc lập:
  1. **Prosecutor (`PROSECUTOR_PROMPT`):** Đọc mã hàm đánh số (+ tối đa 3 dòng ứng viên trung lập đối với `debate_nms` / `debate_rand`, hoặc không gợi ý với `debate_none`), nêu **1 claim duy nhất** `(line, cwe, variable, trigger)` nếu có lỗi tự chứa trong hàm.
  2. **Independent Defense / Falsifier (`DEFENSE_PROMPT`):** Chỉ chạy khi Prosecutor báo `vulnerable: true` (trong một hội thoại hoàn toàn mới, **không thấy `<think>` của Prosecutor**), thực hiện đúng 2 kiểm tra hẹp để bác bỏ dương tính giả trên bản đã vá: (i) có `guard_lines` nào trên đường điều khiển trước `line` đã chặn `trigger` chưa? (ii) `trigger` có đang giả định vô căn cứ về caller/ngữ cảnh ngoài hàm không?

### 3.2. Các nhánh thử nghiệm & đối chứng công bằng (`llm_audit.py`)
Mọi nhánh dùng gợi ý và đối chứng ngẫu nhiên giữ **100% cấu trúc câu chữ giống hệt nhau**, chỉ khác đúng danh sách `K=3` số dòng:
- **Nhóm 1 (Prompt gọn + Multi-sample `n_samples=3`):** `plain`, `plain_hint_nms`, `plain_rand`.
- **Nhóm 2 (2 pha Prosecutor → Falsifier):** `debate_nms`, `debate_rand`, `debate_none`.

### 3.3. Thay đổi mã nguồn đã hoàn tất & kiểm thử trong GĐ 0
- [`colab/colabctl.py`](colab/colabctl.py): Vá tương thích chéo Linux/Windows (`start_new_session=True`, `os.kill(pid, 0)`), xác nhận `presets` và `usage` chạy ổn định trên máy Linux.
- [`pair_protocol.py`](pair_protocol.py): Sửa `rank_acc_debiased` giữ hòa `0.5`, in `ties %` và `flag @med %`, thêm `--clean_json`.
- [`compare_arms.py`](compare_arms.py): Bổ sung đầy đủ các arm mới, contrast cặp tương ứng và `--clean_json`.
- [`llm_audit.py`](llm_audit.py): Bổ sung NMS `diverse_topk_order`, `HINT_PLAIN_NEUTRAL` đối xứng, kiến trúc 2 pha `parse_defense` + `build_defense`, `--n_samples` voting, `--subset_json`, và hook `--plain_template_file` / `--system_override`.
- [`test_llm_audit_v2.py`](test_llm_audit_v2.py): Đạt **23/23 unit tests** (0.019s, stdlib thuần).
- [`audit_gd0_cpu.py`](audit_gd0_cpu.py): Script kiểm toán Colab CPU cho `OpenVul-GRPO`, tạo `smoke50_clean_pairs.json`, rescore NMS + hunk±2 cho PrimeVul + 6 run SecVulEval.

### 3.4. Tiến trình chuyển Plan v2 → Plan v2.1 (Kiến trúc tách vai)

Kết quả Sanity Gate GĐ 1 trên Colab L4 (`results/gd1_sanity/`, `0.44 CU`) cho thấy:
1. **Không chạy chung `debate_*` định dạng JSON lên `OpenVul-Qwen3-4B-GRPO`:** Ở format gốc `HAS_VUL`/`NO_VUL` (`think_budget = 3072`), mô hình hoàn tất `4/4` (`0%` `think_forced`), đạt `P-C` sạch trên cặp `4681/161819`. Ngược lại, khi ép `PROSECUTOR_PROMPT` + `DEFENSE_PROMPT` JSON ở `think_budget = 2048`, `OpenVul-GRPO` dính `75%` `think_forced`, suy diễn sai hàm sạch `161819`, và ở Pha 2 bị *confirmation bias / sycophancy* (`defense_rejected_rate = 0.0%`, lặp lại nguyên văn lý do của Pha 1 và nâng xác suất từ `50 → 75`, `70/75 → 100`).
2. **A100 nhanh hơn và rẻ hơn L4 tính theo mỗi triệu token:** Trên A100 (`5.3 CU/h`), run lịch sử `llm_full_a100` (`Qwen3.5-9B` FP8) đạt **`1.625 gen tok/s`** (`29.7 phút` cho `1.128` hàm), trong khi L4 (`1.54 CU/h`) ở `gd1_sanity` chỉ đạt `81–83 gen tok/s` trên `4` hàm (`batch` nhỏ) và kịch bản batch tối đa cũng chỉ `~300 gen tok/s`.

Do đó, **Plan v2.1** tách thực nghiệm thành:
- **GĐ 1A (preset `gd1_smoke50a`, chạy trên Colab L4 do A100/G4 trả `503`):** Bỏ `debate_*` cùng mô hình trên `OpenVul-GRPO`. Đánh giá sạch 3 nhánh đối xứng chuẩn gốc `HAS_VUL`/`NO_VUL` ([`openvul_grpo_plain_templates.json`](inputs/openvul_grpo_plain_templates.json)) gồm **`plain`**, **`plain_hint_nms`** (`K=3`, `r=3`), **`plain_rand`** (`K=3`), `--n_samples 3`, `--think_budget 3072`, `--workers 24` trên 50 cặp `smoke50_clean_pairs.json`, kèm kiểm định ablation `n=1` vs `n=3` và phân tầng `hit_hunk2` / `miss_changed` / `add_only`.
- **GĐ 1B (Có điều kiện, sau Gate GĐ 1A):** Nếu `OpenVul-GRPO` trên GĐ 1A vẫn có `P-V` cao, chạy **Decoupled Cross-Model Skeptical Falsifier** bằng **`Qwen/Qwen3.5-9B` FP8** chỉ trên các hàm đã bị GĐ 1A gắn nhãn `HAS_VUL` (không phải chạy lại Pha 1).

| GĐ | Hạ tầng | Chi phí thực / dự kiến | Nội dung & Gate quyết định |
|---|---|---|---|
| **GĐ 0** *(Xong)* | Colab CPU (`exp5cpu`, `--gpu none`) | **`0.02 CU`** | Kiểm toán rò rỉ `OpenVul-Qwen3-4B-GRPO` (`396/564` cặp `CLEAN`), trích `smoke50_clean_pairs.json`, phát hiện format `HAS_VUL`/`NO_VUL`, rescore NMS Hunk±2. |
| **GĐ 1 Sanity** *(Xong)* | Colab L4 (`exp5llm`) | **`0.44 CU`** | Gate 2 cặp (`n=4`): `plain` đạt `100%` `think_complete` (`0%` `think_forced`); `debate_nms` trượt Gate (`75%` `think_forced`, `0%` rejected) → dừng và chuyển sang Plan v2.1. |
| **GĐ 1A** *(Dừng sớm có chủ đích)* | Colab L4 (`exp5llm`, `1.94 CU/h`) | **`1.60 CU`** | Chạy được **`51/100` hàm (`153` rollout = `25` cặp `CLEAN` hoàn chỉnh)** nhánh `OpenVul-GRPO plain` trước khi dừng theo tín hiệu định lượng: `think_forced = 35.29%` (> `25%`), `flag_rate = 86.27%` (`72%` cặp hòa), `Net = +4.0%`, `rank_acc = 48.0%` vs `OpenVul-SFT plain` đạt `Net = +16.0%`, `rank_acc = 66.0%` trên cùng 25 cặp. *(Trước đó `A100` & `G4` đều báo `503` với `0.00 CU`)*. |
| **GĐ 1B / Plan v2.2** *(Chờ chốt)* | Colab GPU | ~2–5 CU | **Hướng A:** Cross-Model Falsifier (`Qwen3.5-9B` FP8) trên `results/openvul_test/` có sẵn (`0 CU` Pha 1) hoặc **Hướng B:** Khép kín 3 nhánh đối xứng `plain` / `plain_hint_nms` / `plain_rand` trên `OpenVul-SFT-ep3` (50 cặp `CLEAN`). |

### 3.5. Sổ theo dõi Compute Units (Colab Pro)

| Phiên | Công việc | Hạ tầng | Trạng thái VM | Số dư trước | Số dư sau | Chi phí (CU) |
|---|---|---|---|---:|---:|---:|
| `exp5cpu` | GĐ 0 Audit & Rescore ([`results/gd0_audit/`](results/gd0_audit/)) | CPU (`--gpu none`) | Đã tắt (`0` active) | `67.29` | `67.27` | **`0.02`** |
| `exp5llm` | GĐ 1 Sanity Gate ([`results/gd1_sanity/`](results/gd1_sanity/)) | L4 | Đã tắt (`0` active) | `67.27` | `66.83` | **`0.44`** |
| `exp5llm` | Cấp phát GPU cho GĐ 1A (thử `A100`, rồi `G4`) | A100 / G4 | Thất bại `503 Service Unavailable` | `66.83` | `66.83` | **`0.00`** |
| `exp5llm` | GĐ 1A Partial Run ([`results/gd1_smoke50a_partial/`](results/gd1_smoke50a_partial/)) | L4 (`1.94 CU/h`) | Đã tắt (`0` active) | `66.83` | `65.23` | **`1.60`** |
| `exp5cpu` | Plan v3 GĐ 3.0 Audit ([`results/gd3_0_audit/`](results/gd3_0_audit/)) | CPU (`--gpu none`) | Đã tắt (`0` active) | `65.23` | `65.22` | **`0.01`** |
| `exp5llm` | Plan v3 Tầng 1 Frozen Probe ([`results/gd3_1_probe_qwen25_7b/`](results/gd3_1_probe_qwen25_7b/)) | L4 (`1.94 CU/h`) | Đã tắt (`0` active) | `65.22` | `62.01` | **`3.21`** |
| `exp5llm` | Plan v3 Tầng 2 Sanity V1 ([`results/gd3_2_lora_sanity/`](results/gd3_2_lora_sanity/)) | L4 (`1.94 CU/h`) | Đã tắt (`0` active) | `62.01` | `61.29` | **`0.72`** |
| `exp5llm` | Plan v3 Tầng 2 Sanity V2 ([`results/gd3_2_lora_sanity/`](results/gd3_2_lora_sanity/)) | L4 (`1.94 CU/h`) | Đã tắt (`0` active) | `61.29` | `60.47` | **`0.82`** |

---

## 4. Nhật ký kết quả GĐ 0 & GĐ 1 *(Cập nhật trực tiếp từ các phiên Colab)*

### 4.1. GĐ 0 — Kiểm toán `OpenVul-GRPO` & Rescore NMS/Hunk±2 (`exp5cpu`, chi phí `0.02 CU`)

Đã chạy hoàn tất trên Colab CPU `exp5cpu` (không GPU), kéo kết quả về [`results/gd0_audit/`](results/gd0_audit/) và tắt VM ngay sau khi xong (`67.29 → 67.27 CU`).

#### A. Kiểm toán rò rỉ dữ liệu & phát hiện format prompt gốc của `Leopo1d/OpenVul-Qwen3-4B-GRPO`
Nguồn: [`leak_audit_openvul_grpo.json`](results/gd0_audit/leak_audit_openvul_grpo.json) & [`openvul_grpo_prompt_info.json`](results/gd0_audit/openvul_grpo_prompt_info.json).

- **Trạng thái model:** `Leopo1d/OpenVul-Qwen3-4B-GRPO` tồn tại trên Hugging Face (`Qwen3ForCausalLM`, `max_position_embeddings = 40960`, post-train từ `OpenVul-Qwen3-4B-SFT-ep3`).
- **Kiểm toán rò rỉ trên 564 cặp test PrimeVul:**
  - Trùng tập SFT (`leak_audit_openvul.json`): **`156`** cặp.
  - Trùng thêm ở các tập RL/DPO/ORPO (`OpenVul_Vulnerability_Preference_Dataset_for_DPO/ORPO`, `OpenVul_Vulnerability_Query_Dataset_for_RL` — train/valid đều `0` hit thêm, `15` cặp trùng nằm trong DPO/ORPO): tổng **`15`** cặp ở tập RL.
  - **Hợp nhất rò rỉ (`union_leaked_count`):** **`168 / 564` (29.79%)** → còn lại **`396` cặp hoàn toàn sạch (`CLEAN_GRPO`)**, vượt xa ngưỡng Gate `280` cặp để tiếp tục với `OpenVul-Qwen3-4B-GRPO`.
  - Đã trích xuất tập con phân tầng 50 cặp sạch [`smoke50_clean_pairs.json`](results/gd0_audit/smoke50_clean_pairs.json) (`seed = 42`: **`32` `changed` + `18` `add_only`**) giữ đúng tỉ lệ tập sạch (`255` `changed` / `141` `add_only`).
- **Phát hiện quan trọng về prompt gốc:** `OpenVul-Qwen3-4B-GRPO` được RL trên prompt tự nhiên kết thúc bằng nhãn dòng mới **`HAS_VUL`** hoặc **`NO_VUL`** (cùng System prompt chuyên về CWE và quy tắc *"Do not assume vulnerabilities — only report what is supported by the code and context"*), không phải `Answer with JSON only`. Đã bổ sung hỗ trợ parse `HAS_VUL`/`NO_VUL` vào [`llm_audit.py`](llm_audit.py) và lưu bộ template đối xứng code-only tại [`openvul_grpo_plain_templates.json`](inputs/openvul_grpo_plain_templates.json).

#### B. Kết quả Rescore `Raw` vs `NMS (r=3)` và `Exact` vs `Hunk±2`
Nguồn: [`nms_hunk_rescore.md`](results/gd0_audit/nms_hunk_rescore.md).

- **Trên PrimeVul `eval_pv1_best`:**

| Tập con | Cách xếp hạng | n | Exact T1 / T3 / T5 / T10 | Hunk±2 T1 / T3 / T5 / T10 | Med IFA (Exact / H2) |
|---|---|---:|---|---|---|
| `changed` | `raw` | 357 | 18.2 / 43.1 / 50.1 / 61.6 | 31.4 / 53.5 / 61.6 / 72.5 | 4 / 2 |
| `changed` | **`nms_r3`** | 357 | **18.5** / 37.8 / 46.2 / 61.1 | **31.6 / 56.3 / 68.3 / 82.1** | 6 / **2** |
| `changed` | `longest` (đối chứng) | 357 | 9.8 / 23.8 / 31.4 / 48.7 | 18.8 / 40.1 / 49.6 / 66.1 | 10 / 5 |
| `add_only` (neo `after`) | `raw` | 204 | 9.3 / 20.1 / 30.4 / 46.6 | 18.6 / 32.8 / 48.0 / 62.8 | 11 / 5 |
| `add_only` (neo `after`) | **`nms_r3`** | 204 | 9.3 / **21.1** / 29.9 / 41.7 | **18.6 / 42.2 / 55.9 / 66.2** | 14 / **4** |
| `add_only` (neo `after`) | `longest` (đối chứng) | 204 | 3.4 / 15.2 / 21.6 / 36.3 | 14.7 / 31.4 / 38.7 / 58.8 | 15 / 7 |
| `add_only` (neo `both`) | **`nms_r3`** | 204 | 14.7 / 30.4 / 41.2 / 52.0 | **22.6 / 47.1 / 65.2 / 75.5** | 8 / **3** |

- **Trên SecVulEval test (`n = 1.037`):** Trên cả 6 mô hình (`s0_base2`, `s1_base2`, `s0_lw`, `s0_conv`, `s0_s13_conv`, `unixcoder_1024_s0_conv`), bật `nms_r3` tăng **Hunk±2 Top-5 thêm `+6.0` đến `+9.8` điểm %** (ví dụ `s0_base2`: `66.2% → 74.3%`; `s0_conv`: `63.8% → 73.6%`), và **Hunk±2 Top-10 nhóm C tăng từ `39.4–48.5%` lên `54.5–59.7%`**.

#### C. Kiểm tra `pair_protocol.py` đã sửa lỗi đo lường
- Trên 396 cặp `clean` vs 168 cặp `leaked`, giữ nguyên hòa (`hòa=0.5`) trả `rank_acc_debiased` của `OpenVul-SFT audit_hint` về **`53.5%` (`clean`) / `55.1%` (`leaked`)** (thay vì `44.7%` méo do phá hòa theo độ dài), và cột `flag @med %` cho thấy rõ độ lệch điểm vận hành (`5.9%` ở `audit_hint_nt` vs `27.4–34.8%` ở các nhánh khác).

### 4.2. GĐ 1 Sanity Gate (L4, `0.44 CU`) & GĐ 1A (`gd1_smoke50a`)

#### A. Kết quả Sanity Gate 2 cặp (`results/gd1_sanity/`, `n = 4` hàm: cặp `4681/161819` và `4706/161841`)

| Nhánh | Định dạng & `think_budget` | `answer_rate` (cách parse) | `think_complete` / `think_forced` | `gen_tokens` (`mean` / `med` / `max`) | `flag_rate` | Thời gian & Tốc độ (`workers=8`) | Ghi chú hành vi |
|---|---|---|---|---|---|---|---|
| **`plain`** | Native `HAS_VUL`/`NO_VUL`, `3072` | **`100.0%`** (`4/4`, `tag`) | **`4/4`** (`0%` `think_forced`) | `2703` / `2624` / `3352` | `75.0%` | `2.2 min` (`83 tok/s`) | Bắt đúng **`P-C`** trên `4681/161819` (`161819` trả `NO_VUL`, `prob=0`). |
| **`debate_nms`** | JSON Prosecutor+Defense, `2048` | `100.0%` (`4/4`, `json+json`) | `1/4` / **`3/4` (`75.0%`)** | `3686` / `3939` / `4861` | `100.0%` | `3.0 min` (`81 tok/s`) | **Trượt Gate:** `prosecutor_flag=100%`, **`defense_rejected_rate=0.0%`**; trên `161819` (sạch) Pha 1 suy diễn `prob=50`, Pha 2 chép lại lý do và tăng thành `prob=75` (`confirmed=true`). |

> Nguồn xác minh: [`results/gd1_sanity/scores/summary.json`](results/gd1_sanity/scores/summary.json), [`results/gd1_sanity/plain.jsonl`](results/gd1_sanity/plain.jsonl), [`results/gd1_sanity/debate_nms.jsonl`](results/gd1_sanity/debate_nms.jsonl). Gate tự động dừng trước khi chạy 50 cặp (`SANITY_GATE_FAIL on debate_nms: think_forced=75.0%`) và đã tắt VM sạch (`0` active sessions).

#### B. Kết quả GĐ 1A Partial Run trên Colab L4 — Dừng sớm có chủ đích (`1.60 CU`, [`results/gd1_smoke50a_partial/`](results/gd1_smoke50a_partial/))

- **Hạ tầng & quá trình chạy:** Sau khi `A100` và `G4` đều báo `503 Service Unavailable` (`0.00 CU`), phiên chạy trên **Colab L4** (`23,034 MiB`, `1.94 CU/h`, `--max-num-seqs 32`, `--workers 24`). Đến `50/100`, client báo `errors 0 | 38.8 min | 1.3 fn/min | 225 gen tok/s | ETA 39 min` ([`gd1_smoke50a.log`](results/gd1_smoke50a_partial/gd1_smoke50a.log)), tốc độ vLLM server đạt `278–307 gen tok/s`. Đã dừng có chủ đích ở `51` hàm (`153` rollouts = **`25` cặp `CLEAN` hoàn chỉnh**), chấm điểm và tải toàn bộ log về lưu trữ trước khi tắt VM (`66.83 → 65.23 CU`, `Active assignments: 0`).
- **Thông số kỹ thuật `OpenVul-GRPO plain` (`n = 51` hàm / `153` rollouts, nguồn [`scores/summary.json`](results/gd1_smoke50a_partial/scores/summary.json) & [`partial_eval_vs_sft.json`](results/gd1_smoke50a_partial/partial_eval_vs_sft.json)):**
  - `answer_rate`: **`100.0%`** (`51/51`, `153/153` parse theo `tag`), `errors = 0`, `cut_functions = 0`.
  - Chế độ suy nghĩ: `think_complete = 79 (51.63%)`, `think_reanswer = 20 (13.07%)`, **`think_forced = 54 (35.29%)`** — vượt ngưỡng Gate `<= 25.0%`. Độ dài trung bình `3,508.4` token/rollout (`median = 3,805`, `max = 5,323`).
  - Phân bố gắn cờ: `flag_rate = 86.27%` (`target=0`: `21` bị gắn cờ / `4` sạch; `target=1`: `23` bị gắn cờ / `3` sót). Phân bố `vote_rate` bão hòa nặng ở mức `1.0`: `37` hàm (`3/3`), `7` hàm (`2/3`), `4` hàm (`1/3`), `3` hàm (`0/3`).
- **Đối chiếu ngang hàng (Apples-to-Apples) trên đúng `25` cặp `CLEAN` đã chạy xong:**

| Mô hình / Nhánh *(cùng `n = 25` cặp sạch)* | `flag_rate` | `P-C` | `P-V` | `P-B` | `P-R` | `Net` | `rank_acc` | `ties %` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`OpenVul-GRPO plain (n=3)`** | 86.0% | 8.0% (`2`) | **80.0%** (`20`) | 8.0% (`2`) | 4.0% (`1`) | **+4.0%** | **48.0%** | **72.0%** |
| **`OpenVul-GRPO plain (n=1, s0)`** | 84.0% | 8.0% (`2`) | 76.0% (`19`) | 8.0% (`2`) | 8.0% (`2`) | 0.0% | 50.0% | **84.0%** |
| **`OpenVul-SFT plain`** *(JSON có `probability`)* | **72.0%** | **20.0%** (`5`) | 60.0% (`15`) | 16.0% (`4`) | 4.0% (`1`) | **+16.0%** | **66.0%** | **36.0%** |
| **`Qwen3.5-9B plain`** | 72.0% | 4.0% (`1`) | 68.0% (`17`) | 24.0% (`6`) | 4.0% (`1`) | 0.0% | 58.0% | **20.0%** |
| **`Qwen3.5-9B audit_ctx`** | 76.0% | 16.0% (`4`) | 64.0% (`16`) | 12.0% (`3`) | 8.0% (`2`) | +8.0% | 58.0% | 36.0% |

- **Tham chiếu các baseline có sẵn trên trọn `50` cặp `smoke50_clean_pairs.json` ([`partial_eval_vs_sft.json`](results/gd1_smoke50a_partial/partial_eval_vs_sft.json)):**
  - `OpenVul-SFT plain`: **`Net = +24.0%`** (`P-C=28.0%`, `P-V=54.0%`, `P-B=14.0%`, `P-R=4.0%`), **`rank_acc = 68.0%`**, `ties = 32.0%`, `flag_rate = 70.0%`.
  - `OpenVul-SFT audit_hint`: `Net = +10.0%`, `rank_acc = 53.0%`, `flag_rate = 69.0%`.
  - `Qwen3.5-9B audit_ctx`: `Net = +10.0%` (`P-C=20.0%`, `P-V=62.0%`), `rank_acc = 54.0%`, `flag_rate = 77.0%`.
  - `Qwen3.5-9B` (`plain` / `audit` / `audit_hint`): `Net = 0.0% / +2.0% / 0.0%`, `rank_acc = 58.0% / 60.0% / 52.0%`.
- **Kho lưu trữ log & artifact GĐ 1A:** [`plain.jsonl`](results/gd1_smoke50a_partial/plain.jsonl) · [`scores/summary.json`](results/gd1_smoke50a_partial/scores/summary.json) · [`scores/plain/primevul_test_scores.jsonl`](results/gd1_smoke50a_partial/scores/plain/primevul_test_scores.jsonl) · [`partial_eval_vs_sft.json`](results/gd1_smoke50a_partial/partial_eval_vs_sft.json) · [`gd1_smoke50a.log`](results/gd1_smoke50a_partial/gd1_smoke50a.log) · [`vllm_openvul_grpo.log`](results/gd1_smoke50a_partial/vllm_openvul_grpo.log) · [`mirror_task-468.log`](results/gd1_smoke50a_partial/mirror_task-468.log).

---

## 5. Nhật ký Quyết định (Decision Log) & Đề xuất Hướng Xoay Trục (Plan v2.2)

### 5.1. Dòng thời gian các quyết định (Plan v1 → Plan v2 → Plan v2.1 → Early Stop GĐ 1A)

| Mốc | Kế hoạch tại thời điểm đó | Tín hiệu định lượng kích hoạt thay đổi | Quyết định thực thi & Kết quả |
|---|---|---|---|
| **Baseline 26/10 ([`FINAL_REPORT.md`](../../FINAL_REPORT.md))** | Nâng `--think_budget` `AUDIT` lên `4096–6144` cho Qwen3.5-9B & thêm `audit_rand` cho OpenVul-SFT. | Kiểm định `b8k_testpairs` cho thấy tăng think budget `8K` đẩy `flag_rate` `76.5% → 89.6%` và hạ `Net` `+5.1% → +3.5%` (bẫy *OR-gate*). | Bác bỏ nâng think budget cho `AUDIT`; chuyển sang Plan v2 (NMS `r=3`, prompt gọn, falsifier 2 pha, kiểm toán `OpenVul-GRPO`). |
| **GĐ 0 (20/11, CPU, `0.02 CU`)** | Chạy kiểm toán rò rỉ `OpenVul-GRPO`, NMS rescore và sửa `pair_protocol.py`. | `396/564` cặp `CLEAN` (`>= 280` Gate); phát hiện format gốc `HAS_VUL`/`NO_VUL`; `nms_r3` tăng PrimeVul Hunk±2 T5 `61.6% → 68.3%`. | Qua Gate tiền GĐ 1; cập nhật `llm_audit.py` + `openvul_grpo_plain_templates.json`; chuẩn bị Sanity Gate 2 cặp. |
| **GĐ 1 Sanity (20/11, L4, `0.44 CU`)** | Thử 2 cặp cho `plain` + `debate_nms` (JSON Prosecutor+Defense) trên `OpenVul-GRPO`. | `debate_nms` trượt Gate (`think_forced = 75%`, `defense_rejected_rate = 0%`, lặp lại suy diễn sai trên `161819` nâng `50→75`); `plain` đạt `4/4` `think_complete`. | Bỏ `debate_*` cùng mô hình trên `OpenVul-GRPO`; chuyển sang **Plan v2.1** (GĐ 1A `plain` / `plain_hint_nms` / `plain_rand` với `n=3`). |
| **Cấp phát GPU GĐ 1A (20/11)** | Chạy `gd1_smoke50a` trên `A100` (hoặc `G4`). | Cả `A100` và `G4` đều trả `503 Service Unavailable` từ backend Colab (`0.00 CU`). | Tự động chuyển sang `Colab L4` (`1.94 CU/h`, `--max-num-seqs 32`, `--workers 24`). |
| **GĐ 1A Partial Early-Stop (20/11, L4, `1.60 CU`)** | Chạy đủ 50 cặp × 3 nhánh × 3 rollouts (~3h, ~5.5–6 CU) trên `OpenVul-GRPO`. | Tại `51` hàm (`25` cặp `CLEAN`): **`think_forced = 35.29% > 25%`**, **`flag_rate = 86.27%`**, `37/51` hàm bão hòa `vote_rate = 1.0`, `ties = 72.0%`, `rank_acc = 48.0%`, `Net = +4.0%` (kém xa `OpenVul-SFT plain` `+16.0%` Net / `66.0%` rank trên cùng 25 cặp). | **Dừng sớm có chủ đích**: lưu toàn bộ log + kết quả dở dang vào [`results/gd1_smoke50a_partial/`](results/gd1_smoke50a_partial/), tắt `exp5llm` (`65.23 CU` còn lại, `0` active). |

### 5.2. Chẩn đoán khoa học: Vì sao `OpenVul-Qwen3-4B-GRPO` suy giảm khi kiểm thử `code-only`?
1. **Lệch phân phối từ dữ liệu RL (`Context` vs `Code-only`):** Trong GĐ 0 ([`openvul_grpo_prompt_info.json`](results/gd0_audit/openvul_grpo_prompt_info.json)), prompt huấn luyện của `OpenVul-GRPO` có 2 phần `Context` (includes, macros, called methods) và `Code`. Khi bị cắt toàn bộ `Context` ở giao thức `code-only` trên PrimeVul, chính sách RL có xu hướng báo `HAS_VUL` bất cứ khi nào thấy con trỏ/tham số chưa thấy kiểm tra nội hàm (`flag_rate = 86.27%`, `P-V = 80.0%`).
2. **Mất độ phân giải xếp hạng bởi nhãn nhị phân `HAS_VUL`/`NO_VUL`:** Vì không xuất `probability` liên tục `0–100` như JSON của `OpenVul-SFT`, kết hợp tỷ lệ gắn cờ >86%, phần lớn cả 2 mặt của cặp đều nhận `3/3` phiếu `HAS_VUL` (`ties = 72.0%`), đẩy `rank_acc` xuống dưới mức ngẫu nhiên (`48.0%`).
3. **Ngân sách suy nghĩ `3072` vẫn thiếu cho GRPO:** Ngay cả ở prompt gốc, GRPO nghĩ trung bình `3,508` token/rollout (`35.29%` chạm trần `think_forced`), làm Gate `<= 25%` chắc chắn không đạt.

### 5.3. Hai hướng xoay trục đề xuất cho Plan v2.2 (Ưu tiên tiết kiệm CU & đánh đúng nút thắt)

- **Hướng A (Khuyên dùng số 1 — Chi phí Pha 1 = `0 CU`): Independent Cross-Model Skeptical Falsifier**
  - **Cơ sở:** Trên cùng tập `smoke50_clean_pairs.json`, `OpenVul-SFT plain` đạt `P-C = 28%` (`Net = +24%`), nhưng vẫn bị nghẽn bởi **`P-V = 54%`** (gắn cờ cả bản lỗi lẫn bản vá). Kết quả Pha 1 của `OpenVul-SFT` trên **toàn bộ 564 cặp** (`408` `CLEAN`) đã có sẵn tại [`results/openvul_test/`](results/openvul_test/).
  - **Thiết kế:** Dùng một mô hình suy luận đa dụng độc lập (**`Qwen/Qwen3.5-9B` FP8** hoặc một model không bị RL thiên vị xác nhận lỗi) đóng vai **Skeptical Falsifier** chỉ đọc `(code, claim/reason từ OpenVul-SFT)` cho những hàm đã bị `OpenVul-SFT` gắn cờ (`~66%` số hàm), áp dụng 2 bài kiểm tra hẹp: *(i)* có guard trên đường điều khiển trước dòng nghi vấn chưa; *(ii)* có đang giả định vô căn cứ về caller ngoài hàm không.
  - **Mục tiêu định lượng:** Hạ `P-V` có chọn lọc (bác bỏ dương tính giả trên bản vá nhiều hơn trên bản lỗi) để kéo `Net` và `rank_acc` trên tập `CLEAN` vượt mốc `OpenVul-SFT plain` (`+9.31%`), đồng thời đối chứng trực tiếp xem `audit_hint` (Pha 1 có gợi ý ModernBERT trúng dòng lỗi) khi qua Falsifier có thắng `plain` + Falsifier hay không.

- **Hướng B (Khép kín đối chứng sạch cho mô hình mạnh nhất hiện có): `OpenVul-SFT-ep3` với `plain` vs `plain_hint_nms` vs `plain_rand`**
  - **Cơ sở:** `OpenVul-SFT-ep3` (`Net = +9.31%` trên `408` cặp `CLEAN`; `+24.0%` trên `smoke50`) chưa pernah có nhánh đối chứng `plain_rand` và chưa dùng NMS (`r = 3`) hay prompt đối xứng.
  - **Thiết kế:** Chạy 3 nhánh `plain`, `plain_hint_nms` (`K=3`, `r=3`), `plain_rand` (`K=3`) trên `50` cặp `CLEAN` với `OpenVul-SFT-ep3` (format JSON gốc có `probability`, tránh suy thoái bão hòa kiểu GRPO), kiểm định trực tiếp giả thuyết: *gợi ý dòng ModernBERT-NMS có đẩy `OpenVul-SFT` vượt cả `plain` lẫn `plain_rand` khi loại bỏ hoàn toàn khuôn khổ `AUDIT` OR-gate hay không*.

> **Thứ tự khuyến nghị:** Thử **Hướng B** trên 50 cặp trước (hoặc ghép **Hướng A** trực tiếp trên kết quả `results/openvul_test/` đã lưu để không mất chi phí chạy lại Pha 1). Cả hai đều khả thi thoải mái trong **`65.23 CU`** còn lại.
