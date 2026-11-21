# IE105 — Phát hiện Lỗ hổng & Định vị Dòng Mã Nguồn C/C++ (Code-Only)

Dự án nghiên cứu **Phát hiện lỗ hổng mức hàm (Function-level Vulnerability Detection)** và **Định vị dòng lỗi (Line-level Localization)** cho mã nguồn C/C++ theo điều kiện kiểm thử **strict code-only** (mô hình chỉ nhận mã nguồn hàm, tuyệt đối không dùng commit message, mô tả CVE hay mã nguồn sau bản vá lúc suy luận).

---

## 1. Kiến trúc thư mục & Tài liệu chính

| Thư mục / Tài liệu | Nội dung |
|---|---|
| [`experiments/exp1_linevul_on_primevul/`](experiments/exp1_linevul_on_primevul/) | Tái lập LineVul (CodeBERT 512 token) trên PrimeVul & chẩn đoán điểm mù cắt đầu (`56.6%` hàm lỗi PrimeVul có `flaw_lines` ngoài 512 token). |
| [`experiments/exp2_context_strategies/`](experiments/exp2_context_strategies/) | So sánh các chiến lược đọc hàm dài (`head`, `headtail`, `slice`, `slide_max`, `slide_mean`). |
| [`experiments/exp3_pilot_train/`](experiments/exp3_pilot_train/) | Huấn luyện thử nghiệm trên PrimeVul & phát hiện *length shortcut* ở mức hàm (`length` baseline). |
| [`experiments/exp4_secvuleval/`](experiments/exp4_secvuleval/) | Chuẩn bị & đánh giá trên SecVulEval (chia theo project, có nhãn từng dòng từ bản vá). |
| [`experiments/exp5_line_labels/`](experiments/exp5_line_labels/) | Huấn luyện **ModernBERT 2048 sliding-window** với nhãn từng dòng, chuẩn hóa đề xuất dòng (`diverse_topk_order` NMS), sửa giao thức ghép cặp ([`pair_protocol.py`](experiments/exp5_line_labels/pair_protocol.py), [`compare_arms.py`](experiments/exp5_line_labels/compare_arms.py)), và thử nghiệm **LLM Guard Audit / Verification** ([`llm_audit.py`](experiments/exp5_line_labels/llm_audit.py)). |
| [`experiments/exp6_openvul_line_audit/`](experiments/exp6_openvul_line_audit/) | Thử nghiệm OpenVul-4B kết hợp gợi ý dòng & [`joint_rescore.py`](experiments/exp6_openvul_line_audit/joint_rescore.py). |
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | **Mốc lịch sử (26/10/2026):** Báo cáo kiểm định đối chứng Qwen3.5-9B & OpenVul-4B SFT-ep3 trên PrimeVul (giữ nguyên làm mốc so sánh). |
| [`experiments/exp5_line_labels/PLAN_V2_AUDIT_AND_DESIGN.md`](experiments/exp5_line_labels/PLAN_V2_AUDIT_AND_DESIGN.md) | **Nhật ký Plan v2 / v2.1 (20/11/2026):** Phản biện định lượng plan cũ, kết quả GĐ 0, Sanity Gate GĐ 1, GĐ 1A Partial Early-Stop & Nhật ký Quyết định. |
| [`experiments/exp5_line_labels/PLAN_V3_BREAKTHROUGH.md`](experiments/exp5_line_labels/PLAN_V3_BREAKTHROUGH.md) | **Kế hoạch Đột phá 3 Tầng (Plan v3, đang triển khai):** Kiểm toán GĐ 3.0, Frozen LLM Probing + Anti-Shortcut Audit (`Qwen2.5-Coder-7B`), Paired LoRA & Failure Plan. |
| [`experiments/exp5_line_labels/khao_sat_tai_lieu.md`](experiments/exp5_line_labels/khao_sat_tai_lieu.md) | Tổng hợp ~90 công trình khoa học liên quan (lỗi độ dài, định vị dòng, xác minh phản biện trên LLM). |
| [`experiments/exp5_line_labels/TRANSFER_ANALYSIS.md`](experiments/exp5_line_labels/TRANSFER_ANALYSIS.md) | Kiểm toán chuyển miền SecVulEval → PrimeVul trên các run đã lưu. |
| [`experiments/exp5_line_labels/colab/README.md`](experiments/exp5_line_labels/colab/README.md) | Hướng dẫn chạy Colab CPU/GPU qua `colabctl.py` & `google-colab-cli` trên Linux/Windows. |

---

## 2. Tóm tắt các kết quả đã xác minh

### 2.1. Định vị dòng bằng ModernBERT 2048 + Hậu xử lý NMS (`r = 3`)
Nguồn: [`nms_hunk_rescore.md`](experiments/exp5_line_labels/results/gd0_audit/nms_hunk_rescore.md), [`eval_pv1_best/pair_protocol.json`](experiments/exp5_line_labels/results/eval_pv1_best/pair_protocol.json), [`TRANSFER_ANALYSIS.md`](experiments/exp5_line_labels/TRANSFER_ANALYSIS.md).

- **SecVulEval Test (`n = 1.037` hàm):** ModernBERT-base 2048 (`s0_s13_conv`, `s0_lw`, `s0_base2`, `s0_conv`) đạt Exact Top-10 `68.76–71.46%` (so với `longest-line` `62.0%`). Khi bật NMS (`r = 3`), **Hunk±2 Top-5** tăng thêm `+6.0` đến `+9.8` điểm % (ví dụ `s0_base2`: `66.2% → 74.3%`; `s0_conv`: `63.8% → 73.6%`), **Hunk±2 Top-10** đạt `85.0–87.0%`.
- **PrimeVul Paired (`n = 357` hàm có `flaw_lines`):** `eval_pv1_best` đạt Exact Top-1/3/5/10 = `18.8% / 43.1% / 50.4% / 61.6%` (`line_roc_auc = 0.747`). Khi chấm theo **Hunk±2**, bật **NMS (`r = 3`)** đạt Top-3/5/10 = **`56.3% / 68.3% / 82.1%`** (vượt xa `longest` `40.1% / 49.6% / 66.1%`). Trên nhóm **`add_only`** (`n = 204`, neo dòng `after`), `nms_r3` đạt Hunk±2 Top-5 = **`55.9%`** (vs `longest` `38.7%`).
- **Tác động lên LLM ở mức dòng (Qwen3.5-9B, [`llm_full_a100/scores/summary.json`](experiments/exp5_line_labels/results/llm_full_a100/scores/summary.json)):** Đưa gợi ý dòng của ModernBERT nâng tỷ lệ chỉ trúng dòng lỗi thật (`any_sink_is_a_flaw_line`) từ **`29.5%`** (`audit_rand`) lên **`41.0%`** (`audit_hint`, **+11.5 điểm %**).

### 2.2. Nút thắt ở mức hàm & Phản biện Plan cũ
Nguồn: [`FINAL_REPORT.md`](FINAL_REPORT.md), [`PLAN_V2_AUDIT_AND_DESIGN.md`](experiments/exp5_line_labels/PLAN_V2_AUDIT_AND_DESIGN.md), [`ANALYSIS.md`](experiments/exp5_line_labels/ANALYSIS.md).

- **Encoder 150M chạm trần bởi lối tắt độ dài:** Bản vá dài hơn bản lỗi ở `74%` cặp SecVulEval và **`87.0%`** cặp PrimeVul (`eval_pv1_best/pair_protocol.json`). Sau khi trừ độ dài, mọi encoder 150M mất khả năng phân biệt toàn cục trên PrimeVul (`ROC ≈ 0.49–0.50`).
- **Các mốc LLM trước đây (`564` cặp PrimeVul):**
  - `Qwen3.5-9B` (`llm_full_a100`): `plain` đạt `Net @t0 = +4.43%`, `audit` đạt `Net @t0 = +5.14%` (`+10.99%` `@t_median`), nhưng `audit_hint` (`+6.56%`) hòa `audit_rand` (`+6.56%`) ở `t0` với `think_forced = 98.4%`. Khi mở `think_budget` lên `8K` (`b8k_testpairs`), `flag_rate` phình `76.5% → 89.6%`, kéo `Net` giảm từ `+5.1%` xuống `+3.5%` (bẫy *OR-gate*).
  - `OpenVul-Qwen3-4B-SFT-ep3` (`openvul_test`): `plain` đạt **`Net @t0 = +9.40%`** (`ALL`) / `+9.31%` (`CLEAN 408`), mạnh hơn cả `audit` (`+3.01%`) và `audit_hint` (`+7.62%`), và lúc chạy trước đó chưa có nhánh `audit_rand` đối chứng.

### 2.3. Tiến độ Plan v2 → Plan v3 (20/11/2026)

| Giai đoạn | Kết quả chính đã xác minh | Tài liệu / Artifact |
|---|---|---|
| **GĐ 0 (CPU, `0.02 CU`)** | Kiểm toán rò rỉ `Leopo1d/OpenVul-Qwen3-4B-GRPO`: `168 / 564` cặp trùng (`29.79%`) → **`396` cặp `CLEAN`**. Phát hiện format gốc `HAS_VUL` / `NO_VUL` và tạo `smoke50_clean_pairs.json` (`32 changed + 18 add_only`). Sửa lỗi phá hòa theo độ dài trong `pair_protocol.py`. | [`results/gd0_audit/`](experiments/exp5_line_labels/results/gd0_audit/) |
| **GĐ 1 Sanity (L4, `0.44 CU`)** | `plain` (native `HAS_VUL`/`NO_VUL`, `think_budget=3072`) đạt **`4/4` `think_complete` (`0%` `think_forced`)**, phân biệt đúng `P-C` cặp `4681/161819`. `debate_nms` trượt Gate (`75%` `think_forced`, `0%` bị Defense bác bỏ do confirmation bias cùng mô hình). Chuyển sang Plan v2.1. | [`results/gd1_sanity/`](experiments/exp5_line_labels/results/gd1_sanity/) |
| **GĐ 1A Partial (`results/gd1_smoke50a_partial/`, L4, `1.60 CU`)** | Dừng sớm có chủ đích ở `51` hàm (`25` cặp `CLEAN` hoàn chỉnh, `153` rollouts): `think_forced = 35.29% > 25%`, `flag_rate = 86.27%` (`72%` cặp hòa), `Net = +4.0%`, `rank_acc = 48.0%` (kém `OpenVul-SFT plain` `+16.0%` / `66.0%` trên cùng 25 cặp). Đã lưu đủ log, tắt sạch VM (`65.23 CU` còn lại). | [`PLAN_V2_AUDIT_AND_DESIGN.md`](experiments/exp5_line_labels/PLAN_V2_AUDIT_AND_DESIGN.md) · [`results/gd1_smoke50a_partial/`](experiments/exp5_line_labels/results/gd1_smoke50a_partial/) |
| **Plan v3 GĐ 3.0 (CPU, `0.01 CU`)** | Qua Gate `GD3_0_GATE_PASS` (`65.22 CU` còn lại): loại `7` cặp train trùng test (`6` commit, `1` hash) → **`4.347` cặp `train_clean`** (`637` `cv > cf`) & **`562` cặp `valid_clean`**; `LenMatched<5%` vẫn lệch `82.5%–83.8%`; **`455/564` (80.7%)** cặp test nằm ở `119` `seen_projects` (`109` `unseen_projects`); `173.561` benign ngoài cặp sạch. | [`PLAN_V3_BREAKTHROUGH.md`](experiments/exp5_line_labels/PLAN_V3_BREAKTHROUGH.md) · [`results/gd3_0_audit/`](experiments/exp5_line_labels/results/gd3_0_audit/primevul_audit.md) |
| **Plan v3 Tầng 1 (`gd3_1_probe_qwen25_7b`, L4, `3.21 CU`)** | **PARTIAL PASS (`62.01 CU` còn lại, `0` active VM):** `probe_len_orth` (`mean/L13`) đạt Pair-Gate (`Debias 78.8%`, `PatchRemoves 64.6%`, `Unseen 78.9%`, `|Spearman| 0.129`); `mlp_len_orth` đạt `PatchRemoves 69.2%`; xác nhận Bẫy #1 (`probe_std`/`pair_margin` rớt xuống `50.8%` ở `PatchRemoves`). Tuy nhiên Full-Test Gate trượt (`Debias ROC = 0.4548`; tốt nhất `probe_mixed_benign` `0.5888 < 0.65`, `VD-S = 97.55%`) do lối tắt cú pháp vá (`B_ben > P_vul > P_fix`) và `corr(sv, sf) = 0.974`. Bỏ qua Tầng 1B (`~3.8 CU`), chuyển thẳng **Tầng 2 (Paired Anti-Shortcut QLoRA)**. | [`PLAN_V3_BREAKTHROUGH.md`](experiments/exp5_line_labels/PLAN_V3_BREAKTHROUGH.md) · [`t1_report.md`](experiments/exp5_line_labels/results/gd3_1_probe_qwen25_7b/t1_report.md) |
| **Plan v3 Tầng 2 Sanity V1 & V2 (`gd3_2_lora_sanity`, L4, `1.54 CU`)** | **Học ổn định (`60.47 CU` còn lại, `0` active VM):** Nâng cấp V2 (`B=12/6/3`, `LP-FT` `head_warmup`, zero-init, `lr=2e-5`) đẩy `val_bal_rank` trên `100` cặp từ `45.8% → 56.0%` (`+10.2 pp`), `val_debias_roc 0.5208 → 0.5411`, `Net@med -3.0% → +1.0%`, `peakVRAM 12.49 / 23 GB` (`0` OOM) chỉ sau 13 opt steps. Chẩn đoán 4-bit NF4 chậm (`492 tok/s`) → lộ trình chuyển **`bf16` + 16 tầng trên (`12..27`)** cho bản full `~7–8 CU`. | [`PLAN_V3_BREAKTHROUGH.md`](experiments/exp5_line_labels/PLAN_V3_BREAKTHROUGH.md#L172-L210) · [`t2_report.md`](experiments/exp5_line_labels/results/gd3_2_lora_sanity/t2_report.md) |

- **Kiểm thử tự động cục bộ:** `python3 -m unittest discover -s experiments/exp5_line_labels -p "test_*.py" -v` → **`Ran 35 tests ... OK`** (phủ sạch diff labels, lọc nhiễu, audit 2 pha, Plan v3 Tầng 1 & Tầng 2 trên stdlib thuần).
