# Plan v3 — Đột Phá Mức Hàm Kết Hợp Định Vị Dòng & Kế Hoạch Kiểm Soát Thất Bại

> **Phạm vi**: `experiments/exp5_line_labels` (PrimeVul temporal splits: 564 cặp + full test 25.911 hàm)  
> **Trạng thái tài liệu**: Cập nhật ngày 21/11/2026 — **GĐ 3.0 (`0.01 CU`) · Tầng 1 (`3.21 CU`) · Tầng 2 Sanity V1 & V2 (`1.54 CU`) đã hoàn tất · Còn `60.47 CU`, `0` active VM · Sẵn sàng nâng tốc độ `bf16` + layers `12..27` trước bản Full Tầng 2**. *(Bảo lưu nguyên trạng [`FINAL_REPORT.md`](../../FINAL_REPORT.md) và [`PLAN_V2_AUDIT_AND_DESIGN.md`](PLAN_V2_AUDIT_AND_DESIGN.md).)*

---

## 0. Tóm tắt dễ hiểu: Định hướng, Giải pháp & Hiện trạng

Phần này trình bày ngắn gọn, không dùng thuật ngữ nặng, giúp người đọc nắm ngay **tại sao chọn model hiện tại**, **giải pháp khác gì trước đây**, **gợi ý dòng đứng ở đâu**, và **tình trạng hiện tại**.

### 0.1. Dùng model gì và khác gì với các giai đoạn trước?
- **Model chủ lực hiện tại (Plan v3):** **`Qwen/Qwen2.5-Coder-7B-Instruct`** (7 tỷ tham số, chuyên đọc/hiểu mã nguồn C/C++).

| Giai đoạn / Model | Cỡ | Cách dùng | Điểm mạnh & Nút thắt đã gặp |
|---|---:|---|---|
| **Trước 1:** `ModernBERT-base` | 150M | Phân loại từng dòng & hàm | **Giỏi chỉ vị trí dòng lỗi**, nhưng ở mức hàm toàn bám mẹo *"hàm dài hơn là bản vá"*; khi khử độ dài chỉ đạt ngẫu nhiên (`ROC ≈ 0.50`). |
| **Trước 2:** `Qwen3.5-9B` | 9B | Giữ nguyên model, hỏi đáp JSON | Hay đa nghi (báo lỗi cả hai mặt cặp), sinh suy nghĩ quá dài, dễ dính bẫy OR-gate; `Net` kẹt ở `+4.4%` đến `+5.1%`. |
| **Trước 3:** `OpenVul-Qwen3-4B` *(SFT / GRPO)* | 4B | Model bên ngoài train sẵn, hỏi đáp | Nhóm gốc train kèm ngữ cảnh toàn dự án nên sang chế độ **chỉ có 1 hàm** thì báo lỗi bừa (`86.27%`), ngoài ra ~`30%` trùng với tập test. |
| **Hiện tại (Plan v3):** **`Qwen2.5-Coder-7B-Instruct`** | **7B** | **Tự huấn luyện trực tiếp trên dữ liệu sạch** | Đọc code 1 lượt ra thẳng **điểm số rủi ro liên tục** (không sinh văn bản dài dòng), học trên `4.347` cặp sạch + `4.347` hàm bình thường cùng độ dài. |

### 0.2. Vai trò của gợi ý dòng (không bỏ, tạm giữ để dùng đúng nhịp)
- Định vị dòng bằng **ModernBERT + bộ lọc chống dính chùm NMS (`r=3`)** đang đạt `68.3%` Top-5 và **`82.1%` Top-10** (vùng ±2 dòng quanh lỗi trên PrimeVul). Nút thắt lớn nhất của dự án hiện nay là **phân biệt đúng ở mức hàm**.
- **2 hướng tối ưu tiếp cho định vị dòng (sau khi Tầng 2A mức hàm qua cổng):**
  1. **Miễn phí ngay trong Tầng 2:** Đầu chấm điểm `DualLayerAttnHead` của `Qwen2.5-Coder-7B` tự học trọng số chú ý cho từng token/dòng — có thể xuất ra để so với ModernBERT hoặc kết hợp cả hai.
  2. **Nâng cấp sạch ở Tầng 2B (`~3.8 CU`):** Train lại ModernBERT trên **SecVulEval** (tuyệt đối không đụng `primevul_train`, dùng lọc dòng rác & rank loss) để tạo gợi ý NMS sạch ghép vào `LoRA(NMS)` vs `LoRA(rand)` vs `LoRA(no-hint)`.

### 0.3. Hiện trạng & bước tiếp theo
- **Tầng 1 (giữ nguyên model 7B, chỉ gắn đầu dò ngoài — `3.21 CU`):** Chứng minh model 7B *có nhìn ra tín hiệu lỗi thật* (`78.8%` rank, `64.6%` ngay cả khi bản vá ngắn hơn bản lỗi, `78.9%` dự án chưa từng gặp), nhưng đầu dò ngoài bị lỗi **báo nhầm hàng loạt khi thả chung với 25.000 hàm bình thường** (`Debias ROC = 0.4548`) do bám vào *"cú pháp vừa mới vá"*.
- **Tầng 2 Sanity V1 & V2 (huấn luyện trực tiếp LoRA + thêm hàm bình thường cùng độ dài — `1.54 CU`):** Đã chứng minh pipeline học rất ổn (`val_bal_rank` tăng `+10.2%`, `loss` giảm đều từ `2.55 → 1.94` chỉ trên 200 cặp nháp), bộ nhớ dùng rất nhẹ (`12.49 / 23 GB`, `0` lỗi). **Vướng duy nhất:** dùng nén 4-bit `bitsandbytes` chạy chậm (`~492 tok/s`), nếu giữ nguyên chạy toàn bộ dữ liệu sẽ mất ~12 tiếng (`~24 CU`). **Giải pháp kế tiếp:** bỏ nén 4-bit sang `bf16` thuần và chỉ train 16 tầng trên (`layers 12..27`), rút toàn bộ bản full xuống còn **`~1.5–4 tiếng` (`~7–8 CU`)**.

---

## 1. Bối cảnh & Năm bẫy nhỏ nhặt phải chặn từ đầu

Từ [`PLAN_V2_AUDIT_AND_DESIGN.md`](PLAN_V2_AUDIT_AND_DESIGN.md) và khảo sát [`khao_sat_tai_lieu.md`](khao_sat_tai_lieu.md), zero-shot prompting trên LLM 4B–9B chạm trần quanh `Net ≈ +5% đến +10%`. Khi đối chiếu 0 CU hai mô hình độc lập đã lưu trên `396` cặp `CLEAN` (`OpenVul-SFT plain` + `Qwen3.5-9B audit`), `AND` hạ `P-V` từ `50.3% → 36.6%` và đẩy `P-C` lên `22.5%`, còn `MEAN(4 nhánh)` nâng `rank_acc` lên `58.0%`. Muốn phá trần thực sự bằng học biên quyết định theo cặp / probing, Plan v3 khóa chặt **5 bẫy phương pháp luận**:

1. **Bẫy `length_matched (<5%)` vẫn lệch độ dài `83.8%`:** Trong [`eval_pv1_best/pair_protocol.json`](results/eval_pv1_best/pair_protocol.json#L44-L57), ngay ở tập `<5%` (`n=337`), luật mù *"dài hơn = bản vá"* vẫn đúng **`83.8%`**. Vì vậy phải chấm bắt buộc **`patch_removes_code` (`n=65`, luật độ dài = `0.0%`)**, **`length_matched_balanced` (50/50 trọng số `cf > cv` vs `cv > cf`)**, **`|Spearman(log chars)|`**, và **`full-test debiased ROC-AUC / AUPRC / VD-S @ 0.5% FPR`**.
2. **Bẫy *In-Sample Overfitting* khi tạo gợi ý dòng ModernBERT cho tập `train`:** `eval_pv1_best` được train trên `primevul_train`, nên tuyệt đối không dùng nó chấm lại `primevul_train` để làm hint; nếu cần hint cho train/valid/test ở Tầng 1B hoặc Tầng 2, phải dùng ModernBERT train trên **SecVulEval** (`s0_lw` / `s0_base2`, chưa từng thấy `primevul_train` lẫn `primevul_test`).
3. **Bẫy trùng `project` giữa train và test:** Tách báo cáo bắt buộc **`seen_projects`** vs **`unseen_projects`** trên `test`.
4. **Bẫy *Paired-Only Overfitting*:** Phải kiểm tra trộn hàm lành ngoài cặp (`extra_benign = 2,000` sạch trùng test) và chấm trên full test `25,911` hàm ước lượng qua [`fulltest_manifest.json`](inputs/fulltest_manifest.json).
5. **Bẫy hạ tầng Colab Pro (`L4 23 GB` + không Background Execution):** Thiết kế vừa khít `L4 23 GB VRAM` (`bf16` + `sdpa` + `use_cache=False` + pooling trong forward hooks để tránh cấp phát `[B, L, vocab]`), lưu cache ngoài `--out`, luôn đóng VM bằng `--down` và xác nhận `0` phiên hoạt động.

---

## 2. Lộ trình 3 Tầng & Ma trận Thất bại (Pre-Registered Failure Plan)

Ngân sách khởi đầu Plan v3: **`65.23 CU`** (bảo lưu sàn cứng tối thiểu `~25–30 CU`).

| Giai đoạn | Mục tiêu & Đối chứng | Cổng Qua | Kế hoạch Thất bại (Cổng Sửa 1 vòng / Dừng) |
|---|---|---|---|
| **GĐ 3.0 (CPU, `gd3_0_audit`)** | Kiểm toán `primevul_{train,valid,test}`, `cv > cf`, độ dài token (`Qwen2.5-Coder-7B`), rò rỉ commit/hash, `seen vs unseen` projects | `clean_train_pairs ≥ 1500` và `train_cv_gt_cf ≥ 150` | Nếu `cv > cf` < 150 → bổ sung cặp `cv > cf` sạch từ SecVulEval trước khi cân bằng độ dài |
| **Tầng 1 (`gd3_1_probe_qwen25_7b`, L4)** | Trích 4 tầng `(last, mean)` + zero-shot `Yes/No` từ **`Qwen/Qwen2.5-Coder-7B-Instruct`**, quét 9 mô hình/đối chứng trên Ma trận 5 cổng | `rank_acc_debiased ≥ 65%`, **`patch_removes_code ≥ 60%`**, `|Spearman| < 0.25`, `debiased ROC ≥ 0.65` | **Thắng giả độ dài** (`patch_removes_code < 52%`) → giữ nguyên `probe_len_orth`/`mlp_len_orth`; nếu sau khử độ dài vẫn `< 55%` → chốt thất bại Tầng 1, lên thẳng **Tầng 2 (LoRA)** |
| **Tầng 1B (Conditional)** | Train SecVulEval-ModernBERT → so `NMS-line pooling` vs `last/mean` vs `rand-line` | Chỉ chạy nếu Tầng 1 qua cổng thật | Không tốn `~3.8 CU` train ModernBERT nếu Tầng 1 trượt cổng |
| **Tầng 2 (Paired LoRA Verifier)** | Chạy `LoRA(no-hint)` làm cổng trước trên L4; qua cổng mới chạy 2 đối chứng `LoRA(NMS)` & `LoRA(rand)` | `Net ≥ +14%`, `patch_removes_code ≥ 58%`, `unseen_projects` giữ vững, `debiased ROC ≥ 0.65` | **Overfit project/độ dài** → sửa 1 vòng tăng `extra_benign` + LoRA `r=8`; nếu vẫn `Net ≤ +10%` → xác nhận trần code-only theo `manual40` và chuyển **Tầng 3 (Context)** |
| **Tầng 3 (Repo Context + FT Moderator)** | Đưa `real_ctx` vs `fake_ctx` vs `no_ctx` vào verifier/moderator | `real_ctx > fake_ctx` & `real_ctx > no_ctx` (CI 95% > 0), gain tập trung ở `outside` | Nếu `real_ctx` vẫn không thắng `no_ctx` → chốt giới hạn ngữ cảnh tĩnh 8K và dừng thí nghiệm |

---

## 3. Nhật ký GĐ 3.0 — Kiểm toán Dữ liệu PrimeVul (`exp5cpu`, chi phí `0.01 CU`)

Nguồn xác minh: [`results/gd3_0_audit/primevul_audit.md`](results/gd3_0_audit/primevul_audit.md) & [`primevul_audit.json`](results/gd3_0_audit/primevul_audit.json) (đã tắt `exp5cpu`, còn **`65.22 CU`**, `0` active VMs).

### 3.1. Quy mô cặp sạch & mức độ lệch độ dài

| Tập chia | Cặp sạch | `changed` / `add_only` | `cf > cv` (vá dài hơn) | `cv > cf` (lỗi dài hơn) | Shorter=Vul % | `LenMatched<5%` (`n`) | `LenMatched` Shorter=Vul % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`train_clean`** | **`4.347`** | `2.923 / 1.424` | `3.574` | **`637`** | `83.78%` | `2.596` (`2.073` vs `387`) | **`82.47%`** |
| **`valid_clean`** | **`562`** | `372 / 190` | `474` | `67` | `86.21%` | `315` (`251` vs `43`) | **`83.02%`** |
| **`test`** | **`564`** | `357 / 207` | `482` | `65` | `86.97%` | `337` (`274` vs `46`) | **`83.83%`** |

### 3.2. Phân bố độ dài token (`Qwen/Qwen2.5-Coder-7B-Instruct`)

| Tập chia | Số hàm | Trung vị (tok) | p90 | p95 | Max | `> 2048` tok | `> 3072` tok | `> 4096` tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train_clean` | `8.694` | `695` | `3.286` | `5.406` | `219.136` | `17.37%` | `10.88%` | `7.35%` |
| `valid_clean` | `1.124` | `655` | `2.451` | `4.483` | `23.413` | `15.04%` | `7.92%` | `6.14%` |
| `test` | `1.128` | `724` | `3.325` | `6.438` | `30.202` | `18.26%` | `10.82%` | `8.33%` |

### 3.3. 4 phát hiện trực tiếp củng cố các bẫy của Plan v3
1. **Rò rỉ ẩn trong `primevul_train_paired`:** Đã loại **`7`** cặp `train` trùng với `test` (`6` trùng `commit_id`, `1` trùng `text_hash`), còn sạch **`4.347` cặp `train_clean`** và **`562` cặp `valid_clean`** (`0` trùng). Nguồn benign ngoài cặp sạch là **`173.561` hàm** (trích mẫu `2.000` hàm).
2. **Xác nhận Bẫy 1 (`LenMatched<5%`):** Ngay trong tập `length_matched_5pct`, quy tắc mù độ dài vẫn đúng **`82.47%`–`83.83%`** ở cả 3 split. Rất may `train_clean` có **`637` cặp `cv > cf`** (vượt cổng `≥ 150`), đủ để cân bằng 50/50 trọng số khi train.
3. **Xác nhận Bẫy 3 (`seen_projects` vs `unseen_projects`):** Có **`119` project trùng** giữa `592` project train và `200` project test, khiến **`455/564` (80.7%)** cặp test nằm trong `seen_projects` (ví dụ `linux` 818→75, `tensorflow` 51→72, `vim` 25→38, `php-src` 115→15) và chỉ **`109/564` (19.3%)** thuộc `unseen_projects`.
4. **Full-Test Manifest khớp 100%:** `primevul_test.jsonl` có đủ `25.911` hàm (`695` lỗi), khớp trọn `P = 1.126`, `V = 131`, `B = 2.000` (`wB = 12.327`).
5. **Kết luận Gate GĐ 3.0:** **`GD3_0_GATE_PASS`** (`4.347 ≥ 1.500`, `637 ≥ 150`).

---

## 4. Nhật ký Tầng 1 — Frozen LLM Probing `Qwen/Qwen2.5-Coder-7B-Instruct` (`exp5llm`, chi phí `3.21 CU`)

Nguồn xác minh: [`results/gd3_1_probe_qwen25_7b/t1_report.md`](results/gd3_1_probe_qwen25_7b/t1_report.md), [`t1_results.json`](results/gd3_1_probe_qwen25_7b/t1_results.json), [`pair_protocol_t1.json`](results/gd3_1_probe_qwen25_7b/pair_protocol_t1.json), [`gd3_1_probe_qwen25_7b.log`](results/gd3_1_probe_qwen25_7b/gd3_1_probe_qwen25_7b.log).

- **Thông số trích xuất GPU (`NVIDIA L4 23,034 MiB`, `1.94 CU/h`):**
  - Model: `Qwen/Qwen2.5-Coder-7B-Instruct` (`max_len = 2048`, các tầng hook `[13, 19, 23, 27]`, `d = 3584`, `bf16 + sdpa + use_cache=False`).
  - Tổng hàm duy nhất: **`15.066`** hàm (bị cắt `1.979` hàm = `13.14%`), `peakVRAM = 15.53 GB`, thời gian trích GPU `3.418,1s` (`~3.555 tok/s` cuối vòng extract), kiểm tra tương đương `lm_head` `head_equiv_max_diff = 0.03125`.
  - Cache ẩn lưu cục bộ (đã gitignore) tại `experiments/exp5_line_labels/colab/work/qwen25_7b_feats.npz` (`864.32 MB`, `10` mảng).
  - Đã tắt sạch `exp5llm`: còn **`62.01 CU`** (chi phí Tầng 1: **`3.21 CU`** từ mức `65.22 CU`), `Active assignments: 0`.

### 4.1. Ma trận 5 cổng chống lối tắt độ dài (`primevul_test_paired`, `n = 564`)

| Mô hình / Nhánh | Cấu hình tốt nhất | `All Net@t0 / @med` | `All Rank / Debias / Ties` | `Bal Rank` | `LenMatched5%` | **`PatchRemoves (n=65)` [95% CI]** | `Seen / Unseen` | `Changed / AddOnly` | `Spearman` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `length_only` | `log(chars/tok/lines)` | `+2.1% / +1.4%` | `76.4% / 53.2% / 2.5%` | `52.6%` | `66.8%` | **`20.0%`** `[10.8; 30.8]` | `75.5% / 80.3%` | `72.3% / 83.6%` | `-0.968` |
| `tfidf_word` | `C=1.0` | `+5.7% / +3.2%` | `73.7% / 69.8% / 4.1%` | `65.0%` | `67.4%` | **`53.1%`** `[41.5; 64.6]` | `73.1% / 76.2%` | `69.6% / 80.7%` | `-0.398` |
| `zero_shot_yesno` | `log P(Yes)-log P(No)` | `+0.0% / +1.4%` | `51.3% / 61.8% / 9.8%` | `55.1%` | `53.6%` | **`60.0%`** `[48.5; 70.8]` | `50.9% / 53.2%` | `52.1% / 50.0%` | `+0.520` |
| `probe_std` | `mean/L13/C=0.001` | `+10.1% / +10.6%` | `77.5% / 76.1% / 5.7%` | `66.1%` | `70.6%` | **`50.8%`** `[39.2; 63.1]` | `76.7% / 80.7%` | `73.0% / 85.3%` | `-0.273` |
| `probe_pair_margin` | `mean/L19/C=0.01` | `+5.8% / +6.0%` | `83.0% / 78.4% / 5.7%` | `69.5%` | `77.2%` | **`50.8%`** `[38.5; 63.1]` | `83.7% / 79.8%` | `77.5% / 92.5%` | `-0.684` |
| `probe_len_balanced` | `mean/L13/C=0.01` | `+6.0% / +6.7%` | `77.8% / 75.7% / 5.7%` | `71.7%` | `71.5%` | **`63.1%`** `[52.3; 73.9]` | `77.8% / 78.0%` | `73.8% / 84.8%` | `-0.226` |
| **`probe_len_orth`** | **`mean/L13/C=0.01`** | `+6.7% / +6.7%` | `77.2% / `**`78.8%`**` / 0.2%` | **`71.9%`** | `71.1%` | **`64.6%`** `[53.9; 76.9]` | `76.8% / `**`78.9%`** | `74.7% / 81.6%` | **`-0.129`** |
| `probe_mixed_benign` | `mean/L13/C=0.01` | `+1.4% / +1.8%` | `60.3% / 65.2% / 5.7%` | `62.4%` | `59.6%` | **`64.6%`** `[53.1; 76.2]` | `59.3% / 64.2%` | `60.4% / 60.1%` | `+0.469` |
| **`mlp_len_orth`** | `mean/L13` (3 seeds) | `+5.0% / +5.0%` | `72.2% / 75.3% / 0.2%` | `71.0%` | `67.2%` | **`69.2%`** `[58.5; 80.0]` | `72.6% / 70.6%` | `70.7% / 74.9%` | `+0.265` |
| *Tham chiếu: `eval_pv1_best`* | ModernBERT-base 2048 | `+0.5% / -1.6%` | `47.6% / 55.9%` | — | — | `56.9%` | — | — | — |
| *Tham chiếu: `Qwen3.5-9B`* | `plain` / `audit` | `+4.4%` / `+5.1%` (`+11.0% @med`) | `54.4%` / `56.7%` | — | — | `48.5%` / `55.4%` | — | — | — |
| *Tham chiếu: `OpenVul-SFT`* | `plain` | `+9.4% / +4.4%` | `55.7% / 53.0%` | — | — | `57.7%` | — | — | — |

### 4.2. Full PrimeVul Test (25.911 hàm, trọng số từ `fulltest_manifest.json`, prevalence = `2.68%`) & Phân rã tầng điểm

| Mô hình | `ROC-AUC` | **`Debiased ROC-AUC`** | `AUPRC` | **`VD-S (FNR @ 0.5% FPR)`** | `Recall @ 0.5% FPR` | `Spearman` | `corr(sv, sf)` | Mean `P_vul / U_vul / P_fix / B_ben` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `length_only` | `0.1952` | `0.5087` | `0.0152` | `100.00%` | `0.00%` | `-0.984` | `+0.9922` | `-0.000 / -0.010 / -0.003 / +0.062` |
| `tfidf_word` | `0.3230` | `0.4939` | `0.0179` | `100.00%` | `0.00%` | `-0.539` | `+0.9522` | `-0.034 / -0.041 / -0.057 / +0.065` |
| `zero_shot_yesno` | `0.7636` | `0.5653` | `0.0757` | `97.70%` | `2.30%` | `+0.669` | `+0.9567` | `-0.837 / -0.854 / -0.844 / -1.766` |
| `probe_std` | `0.3328` | `0.4877` | `0.0188` | `99.71%` | `0.29%` | `-0.513` | `+0.8691` | `+0.076 / +0.008 / -0.025 / +0.281` |
| `probe_pair_margin` | `0.2433` | `0.4541` | `0.0178` | `99.57%` | `0.43%` | `-0.777` | `+0.9726` | `-0.256 / -3.189 / -1.819 / +8.466` |
| `probe_len_balanced` | `0.3859` | `0.4687` | `0.0253` | `98.99%` | `1.01%` | `-0.305` | `+0.9673` | `-0.208 / -1.773 / -1.125 / +1.681` |
| `probe_len_orth` | `0.4303` | **`0.4548`** | `0.0287` | `98.42%` | `1.58%` | `-0.135` | `+0.9740` | **`-0.371 / -2.111 / -1.329 / +0.794`** |
| **`probe_mixed_benign`** | **`0.7811`** | **`0.5888`** | **`0.0857`** | **`97.55%`** | **`2.45%`** | **`+0.722`** | `+0.9770` | `-0.666 / -0.585 / -0.724 / -2.135` |
| `mlp_len_orth` | `0.7477` | `0.5825` | `0.0653` | `97.70%` | `2.30%` | `+0.631` | `+0.9346` | `+0.032 / -0.072 / -0.102 / -0.824` |

*(Tham chiếu từ [`eval_pv1_best/primevul_results.json`](results/eval_pv1_best/primevul_results.json): ModernBERT-base 2048 đạt raw full-test `ROC-AUC = 0.7848`, `AUPRC = 0.0845` nhưng `spearman_with_tokens = +0.672` nên `Debiased ROC ≈ 0.50`.)*

### 4.3. Quét Tầng & Pooling (`probe_len_orth`)

| Pool | Tầng | Best `C` | Valid Bal Rank | Test All Rank | Test Debias Rank | Test `PatchRemoves` Rank |
|---|---:|---:|---:|---:|---:|---:|
| `last` | 13 | `0.1` | `60.1%` | `63.6%` | `63.0%` | `58.5%` |
| `last` | 19 | `0.001` | `61.6%` | `67.1%` | `66.8%` | `56.9%` |
| `last` | 23 | `0.01` | `59.2%` | `66.9%` | `68.5%` | `60.0%` |
| `last` | 27 | `1.0` | `58.9%` | `66.8%` | `65.2%` | `46.1%` |
| **`mean`** | **`13`** | **`0.01`** | **`70.2%`** | **`77.2%`** | **`78.8%`** | **`64.6%`** |
| `mean` | 19 | `0.001` | `66.7%` | `74.0%` | `75.8%` | `64.6%` |
| `mean` | 23 | `0.01` | `64.7%` | `72.6%` | `74.7%` | `61.5%` |
| `mean` | 27 | `0.1` | `69.6%` | `74.4%` | `75.3%` | `52.3%` |

### 4.4. Phân tầng `manual40` (40/40 khớp) & `patch_taxonomy` (`564` cặp) — Phơi bày **Guard-Syntax / Patch-Artifact Shortcut**

Do `probe_llm_t1.py` gọi `random.Random(1).sample(range(564), 40)` thay vì `sample(range(564), 564)[:40]`, bước `manual40` trong run GPU bị skip an toàn (`29/40`). Kiểm định bù 0 CU trên máy cục bộ với đúng khóa `sample(range(564), 564)[:40]` (khớp **`40/40`** project) và [`patch_taxonomy/test_pairs.jsonl`](results/patch_taxonomy/test_pairs.jsonl) cho ra kết quả cực kỳ sắc nét (`Rank% / Net@med%`):

| Mô hình | `Man40 local (15)` | `Man40 outside (14)` | `Man40 !a_to_b (11)` | `Tax local_ab (58)` | `Tax out_ab (296)` | `Tax !a_to_b (210)` |
|---|---:|---:|---:|---:|---:|---:|
| `length_only` | `86.7% / +0.0%` | `85.7% / +0.0%` | `86.4% / -9.1%` | `61.2% / -1.7%` | `82.6% / +2.7%` | `71.9% / +0.5%` |
| `tfidf_word` | `66.7% / +0.0%` | `71.4% / +14.3%` | `81.8% / +0.0%` | `77.6% / -1.7%` | `79.7% / +5.1%` | `64.0% / +1.9%` |
| `zero_shot_yesno` | `60.0% / +6.7%` | `57.1% / +0.0%` | `31.8% / +0.0%` | `55.2% / +5.2%` | `51.7% / +1.7%` | `49.8% / +0.0%` |
| `probe_std` | `86.7% / +6.7%` | `78.6% / +14.3%` | `63.6% / -9.1%` | `77.6% / +13.8%` | `82.3% / +12.5%` | `70.7% / +7.1%` |
| `probe_len_balanced` | `60.0% / +6.7%` | `85.7% / +7.1%` | `63.6% / +9.1%` | `77.6% / +1.7%` | `83.6% / +9.1%` | `69.8% / +4.8%` |
| **`probe_len_orth`** | **`53.3%`** `/ +6.7%` | **`85.7%`** `/ +14.3%` | `63.6% / +9.1%` | `70.7% / +8.6%` | **`83.4%`** `/ +8.8%` | `70.2% / +3.3%` |
| `probe_mixed_benign` | `60.0% / +0.0%` | `64.3% / +7.1%` | `63.6% / -9.1%` | `69.0% / +1.7%` | `63.7% / +2.7%` | `53.1% / +0.5%` |
| `mlp_len_orth` | `60.0% / +6.7%` | `78.6% / +7.1%` | `63.6% / -9.1%` | `70.7% / +8.6%` | `78.0% / +4.7%` | `64.5% / +4.3%` |

### 4.5. Phán quyết theo Ma trận Thất bại & 4 phát hiện cốt lõi

1. **Xác nhận Bẫy #1 (Bác bỏ kết quả probing ngây thơ kiểu `arXiv 2609.32890`):** `probe_std` (`50.8%`) và `probe_pair_margin` (`50.8%`) thắng rất cao trên `all` (`77.5% / 83.0%`) và `LenMatched5%` (`70.6% / 77.2%`) nhưng **sụp đổ hoàn toàn về ngẫu nhiên `50.8%` trên `patch_removes_code (n=65)`**. Tầng giữa `mean(L13)` đánh bại `last` ở mọi tầng tới `+10–15 điểm %`.
2. **Qua Cổng Cặp thật trên `probe_len_orth` & `mlp_len_orth`:** Sau khử độ dài + cân bằng `50/50`, `probe_len_orth` (`mean/L13`) đạt **`rank_acc_debiased = 78.8%`** (`≥ 65%`), **`patch_removes_code = 64.6%`** (`[53.9; 76.9]`, `≥ 60%`), **`unseen_projects = 78.9%`** (không overfit tên project!), `|Spearman| = 0.129` (`< 0.25`), và `mlp_len_orth` đẩy `patch_removes_code` lên **`69.2%`** (`[58.5; 80.0]`).
3. **TRƯỢT Cổng Toàn cục (`Full-Test & Net Decision Boundary FAIL`) — Phát hiện *Guard-Syntax / Patch-Artifact Shortcut*:**
   - Trên Full PrimeVul Test (`25.911` hàm), `probe_len_orth` đảo ngược thảm hại xuống **`ROC-AUC = 0.4303` / `Debiased ROC = 0.4548`** (`B_ben = +0.794 > P_vul = -0.371 > P_fix = -1.329`). Ngay cả `probe_mixed_benign` cũng chỉ đạt `Debiased ROC = 0.5888` (`< 0.65`), `VD-S = 97.55%`, và bị hút sang **lối tắt độ dài ngược chiều dương (`Spearman = +0.722`)** vì mẫu benign ngẫu nhiên trong PrimeVul ngắn hơn hẳn hàm lỗi/vá cặp (`median`).
   - Trên cặp, dù xếp hạng đạt `77–79%`, biên quyết định toàn cục vẫn nghẽn ở **`Net@med = +6.7%`** vì phương sai giữa các cặp quá lớn lấn át hiệu số trong cặp (`corr(sv, sf) = 0.9740`, `d_mean / sd(sv) = 0.131`).
   - Đặc biệt, Bảng 4.4 phơi bày rõ: **`probe_len_orth` thắng mạnh nhất ở `Man40 outside = 85.7%` nhưng lại tụt xuống `53.3%` ở `Man40 local`**, chứng tỏ mô hình không thực sự hiểu lỗi bảo mật cục bộ mà đang nhận diện ***"cú pháp phòng thủ mới được vá (guard syntax)"*** ở hàm cố định (`P_fix`), khiến hàm sạch bình thường chưa từng chạm vào guard syntax (`B_ben`) bị đẩy lên cao hơn cả hàm lỗi!
   - **Quyết định theo Ma trận Thất bại:** **PARTIAL PASS → Bỏ qua Tầng 1B** (tiết kiệm `~3.8 CU` vì probe tầng ẩn đóng băng không tạo được biên phân loại toàn cục `Debiased ROC ≥ 0.65`), **tiến thẳng Tầng 2 (`Paired Anti-Shortcut QLoRA`)** trên chính `Qwen/Qwen2.5-Coder-7B-Instruct`.

---

## 5. Giải pháp Tầng 2 — `Paired Anti-Shortcut QLoRA` & Nhật ký Sanity V1 / V2 ([`train_lora_t2.py`](train_lora_t2.py))

Từ 2 thất bại chẩn đoán được ở Tầng 1, Tầng 2 thiết kế lại đồng bộ cả dữ liệu, kiến trúc head, hàm mất mát và lịch trình tối ưu trên `Qwen/Qwen2.5-Coder-7B-Instruct` (preset `gd3_2_lora_sanity` & `gd3_2_lora_nohint` trong [`presets.py`](colab/presets.py)):

### 5.1. Kiến trúc & Cơ chế chống lối tắt
1. **Dữ liệu chống cả 2 chiều độ dài & *Guard-Syntax Collapse*:**
   - Trộn **`4.347` cặp `train_clean`** (trọng số 50/50 giữa `cf > cv` và `cv > cf`, kẹp `[0.1, 4.0]` qua `compute_pair_length_weights`) với **`4.347` hàm `extra_benign`** chọn từ `173.561` benign sạch của `primevul_train`, trong đó **`85%` khớp phân bố `log(chars)` theo 20 quantile bins** (`sample_length_matched_benigns`) và `15%` ngẫu nhiên.
   - Ở mỗi epoch, hàm `pair_benigns_by_length` ghép từng cặp `(v_i, f_i)` với 1 benign `b_i` **cùng chiều dài token** (xáo trộn cục bộ trong cửa sổ `window=16`), sau đó `build_length_bucketed_batches` đóng gói thành các micro-batch theo độ dài (`k = 4` cặp / `B = 12` chuỗi cho `<= 512` tok; `k = 2` / `B = 6` cho `<= 1024`; `k = 1` / `B = 3` cho `> 1024`).
2. **Head hai tầng `L13 + L27` (`DualLayerAttnHead`, `float32`):**
   - Dùng `AutoModel` (bỏ `lm_head` để tiết kiệm VRAM), trích đồng thời tầng giữa `L13` (nơi probe Tầng 1 mạnh nhất) và tầng cuối `L27` qua hook an toàn với gradient checkpointing (`use_reentrant=False`).
   - Mỗi nhánh kết hợp *LayerNorm → Residual Mean Pool + tanh(gate) × Attention Pool*, gộp bằng trọng số softmax học được khởi tạo thiên về `L13` (`[1.0, 0.5]`), và tầng `classifier` cuối khởi tạo bằng `0` (*zero-init*) để triệt tiêu gai gradient bước đầu.
3. **Mục tiêu 3 thành phần (`_compute_triplet_loss`) & lịch trình LP-FT:**
   - `loss = 1.0 * w_i * softplus(m - (sv - sf)) + 0.25 * w_i * (BCE(sv, 1) + BCE(sf, 0)) + 0.75 * (BCE(sb, 0) + 0.5 * softplus(m - (sv - sb)))` (`m = 1.0`).
   - Chạy pha khởi động head riêng **LP-FT** (`head_warmup_pairs`: backbone chạy `no_grad`, chỉ cập nhật head) rồi mới mở cosine LoRA `lr = 2e-5` (head `5e-4`). Hỗ trợ lưu `ckpt_last` ngoài `--out` (`--ckpt_dir`), tự fallback 1-item *analytic scalar-gradient* nếu OOM, và chấm chọn checkpoint tốt nhất bằng `0.5 * val_bal_rank + 0.5 * val_debias_roc`.

### 5.2. Nhật ký đối chứng Sanity V1 vs Sanity V2 (`200` cặp × `1` epoch, `valid 100 cặp + 200 benign`)

Nguồn xác minh: [`results/gd3_2_lora_sanity/t2_report.md`](results/gd3_2_lora_sanity/t2_report.md), [`t2_results.json`](results/gd3_2_lora_sanity/t2_results.json), [`mirror_task-954_v1.log`](results/gd3_2_lora_sanity/mirror_task-954_v1.log), [`mirror_task-1013_v2.log`](results/gd3_2_lora_sanity/mirror_task-1013_v2.log).

| Thông số (`gd3_2_lora_sanity` trên L4) | **Sanity V1 (`0.72 CU`)** | **Sanity V2 (`0.82 CU`)** | Chẩn đoán kỹ thuật |
|---|---|---|---|
| **Chiến lược chạy** | `bs=1` lẻ, `lr=1e-4`, `alpha=32`, head random, không warmup | **Batched `B=12/6/3`**, **`LP-FT` `warmup=64`**, zero-init head, `lr=2e-5`, `alpha=16`, `matched_frac=0.85` | Chỉ 13 opt steps để kiểm tra độ ổn định & tốc độ |
| **Loss cuối cửa sổ** (`50→100→150→200`) | `2.18 → 2.20 → 2.22 → 2.2753` | `2.32 → 2.55 → 2.46 → `**`1.9417`** *(mean `2.3180`)* | V2 giảm đều cả 3 nhánh (`pair 1.006`, `bce 0.357`, `ben 0.579`) |
| **`gnorm` cuối** | `11.99` *(nổ sớm)* | **`6.87`** | Ổn định, hội tụ mượt |
| **`Valid Bal Rank` (`100` cặp)** | `45.8%` | **`56.0%` (`+10.2 pp`)** | Bật lên rõ rệt ngay sau 13 opt steps |
| **`Valid Debias ROC` (`300` hàm)** | `0.5208` | **`0.5411`** | Tăng đồng bộ năng lực tách cặp & hàm lành |
| **`Valid Net@med`** | `-3.0%` | **`+1.0%`** | Đảo chiều từ âm sang dương |
| **`Composite`** | `0.4895` | **`0.5504`** | Vượt trội trên toàn bộ tiêu chí chọn lọc |
| **`peakVRAM` / `OOM fallbacks`** | `10.15 GB` / `0` | **`12.49 GB / 23.03 GB`** / `0` | Dư hơn `10.5 GB` VRAM trên L4 |
| **Tốc độ & Thời gian train** | `0.18 pair/s` (`506 tok/s`), `1.239,6s` | `HEAD 0.54 pair/s` (**`1.599 tok/s`**) / `LORA 0.19 pair/s` (`492 tok/s`), `1.013,9s` | **Chạm nút thắt phần cứng NF4** |
| **Trạng thái VM & Số dư CU** | Tắt sạch `0` active (`62.01 → 61.29`) | Tắt sạch `0` active (`61.29 → 60.47`) | Không mất dữ liệu, 0 session treo |

### 5.3. Giải phẫu điểm nghẽn tốc độ NF4 & Phương án tăng tốc `~3×` cho bản Full
- **Nguyên nhân chậm ở `LORA` (`492 tok/s`):** Khi đối chiếu trực tiếp `bf16` thuần ở Tầng 1 (**`3.555 tok/s`**) với `phase=HEAD` không backprop trên **4-bit NF4 `bitsandbytes`** (**`1.599 tok/s`**), riêng giải nén 4-bit NF4 đã chậm hơn `bf16` tới **`2.22×`**, cộng thêm recompute toàn bộ 28 tầng trong `gradient_checkpointing` khiến pha LORA rơi xuống ~`492 tok/s` → bản full 2 epoch (`4.347` cặp) sẽ tốn `~12,5 giờ` (`~24 CU`), vượt dự toán `6–10 CU`.
- **Kế hoạch tối ưu trước khi chạy bản Full `gd3_2_lora_nohint`:**
  1. **Chuyển từ 4-bit NF4 sang `bf16` LoRA thuần (bỏ `bitsandbytes`):** `AutoModel` không có `lm_head` chỉ chiếm `~14,1 GB` trong `23,03 GB` của L4 (`~18,5–19,5 GB` khi kèm activation checkpoint ở `B=6/3/3`), lấy lại đúng tốc độ ma trận Tensor Core bản địa (`~2,2×` nhanh hơn).
  2. **Chỉ gắn LoRA vào 16 tầng trên (`layers 12..27`), đóng băng hoàn toàn `layers 0..11`:** Tầng 1 đã chứng minh biểu diễn lỗi nổi lên từ `L13` trở lên; giữ nguyên 12 tầng đáy giúp bỏ recompute & backward cho `43%` mạng (`~1,4×` nhanh hơn + giảm rủi ro overfit cú pháp C cơ bản).
  3. **Quy trình chạy an toàn:** Kiểm tốc **3 phút** (`50` cặp, `~0.10 CU`) xác nhận đạt `≥ 1.300 tok/s` & `peakVRAM < 21 GB`, sau đó mới chạy **`gd3_2_lora_nohint`** (thử `A100 ~1,5 giờ` trước, fallback `L4 ~3,5–4 giờ`, tổng **`~7–8 CU`**).
- **Cổng qua `gd3_2_lora_nohint` (`PASS_T2_GATE`):** `Net@med ≥ +14%`, `PatchRemoves ≥ 58%`, `Unseen Projects ≥ 65%`, **`Full-Test Debiased ROC ≥ 0.65`**, `|Spearman| < 0.25`. Đạt cổng → tiến sang **Tầng 2B (`SecVulEval-ModernBERT` + `LoRA(NMS)` vs `LoRA(rand)`)**; nếu sau sửa 1 vòng vẫn `Net ≤ +10%` → chốt trần `code-only` và chuyển thẳng **Tầng 3 (Repo Context)**.

---

## 6. Danh mục Kiểm thử Phần mềm (`35/35` OK) & Sổ Ngân sách CU Tổng hợp

### 6.1. Bộ Unit Tests Cục bộ (`python3 -m unittest discover -s experiments/exp5_line_labels -p "test_*.py" -v`)
Phạm vi xác minh: toàn bộ **35 bài test** dùng stdlib Python thuần để kiểm chứng logic tách nhãn diff, lọc nhiễu, đối chứng prompt 2 pha, chống rò rỉ dữ liệu, cân bằng trọng số, ánh xạ chính xác `40/40` `manual40` & `564/564` `patch_taxonomy`, kiểm tra cú pháp (`ast.parse`) và giao thức báo cáo (`Ran 35 tests in 0.105s — OK`); các vòng chạy PyTorch/CUDA thực tế được xác minh riêng trên Colab L4 qua các phiên Sanity V1 & V2.

| File kiểm thử | Số test | Trạng thái | Phạm vi bảo chứng |
|---|---:|---|---|
| [`test_analyze_transfer.py`](test_analyze_transfer.py) | `3` | `OK` | Xếp hạng tie-aware & tái tạo báo cáo chuyển miền từ JSON thật |
| [`test_clean_diff_labels.py`](test_clean_diff_labels.py) | `5` | `OK` | Lọc dòng diff sạch, chống gộp khối `{}` & `diverse_topk_order` NMS |
| [`test_label_noise.py`](test_label_noise.py) | `8` | `OK` | Nhận diện dòng tầm thường/comment chuỗi & ngưỡng mật độ nhiễu |
| [`test_llm_audit_v2.py`](test_llm_audit_v2.py) | `8` | `OK` | Đối xứng prompt hint/rand, parser JSON/`HAS_VUL`, 2 pha Prosecutor→Defense & `n=3` |
| [`test_plan_v3_t1.py`](test_plan_v3_t1.py) | `3` | `OK` | Cắt ghép giữ nguyên `PROBE_SUFFIX`, lọc rò rỉ commit/hash/benign & `ast.parse` |
| [`test_plan_v3_t2.py`](test_plan_v3_t2.py) | `8` | `OK` | Khớp benign 20 quantile, cân bằng trọng số 50/50, bucket `B=12/6/3`, khớp `40/40` `manual40` & `564/564` `taxonomy`, cổng gate & `ast.parse` |
| **Tổng cộng** | **`35`** | **`OK`** | `0` skipped, `0` failed |

### 6.2. Sổ theo dõi Compute Units tích lũy (Colab Pro)

| Phiên | Công việc | Hạ tầng | Trạng thái VM | Số dư trước | Số dư sau | Chi phí (CU) |
|---|---|---|---|---:|---:|---:|
| `exp5cpu` | Plan v2 GĐ 0 Audit & Rescore ([`results/gd0_audit/`](results/gd0_audit/)) | CPU (`--gpu none`) | Đã tắt (`0` active) | `67.29` | `67.27` | **`0.02`** |
| `exp5llm` | Plan v2 GĐ 1 Sanity Gate ([`results/gd1_sanity/`](results/gd1_sanity/)) | L4 | Đã tắt (`0` active) | `67.27` | `66.83` | **`0.44`** |
| `exp5llm` | Thử cấp phát GPU GĐ 1A (`A100`, `G4`) | A100 / G4 | Lỗi `503 Service Unavailable` | `66.83` | `66.83` | **`0.00`** |
| `exp5llm` | Plan v2.1 GĐ 1A Partial ([`results/gd1_smoke50a_partial/`](results/gd1_smoke50a_partial/)) | L4 (`1.94 CU/h`) | Đã tắt (`0` active) | `66.83` | `65.23` | **`1.60`** |
| `exp5cpu` | Plan v3 GĐ 3.0 Data Audit ([`results/gd3_0_audit/`](results/gd3_0_audit/)) | CPU (`--gpu none`) | Đã tắt (`0` active) | `65.23` | `65.22` | **`0.01`** |
| `exp5llm` | Plan v3 Tầng 1 Frozen Probe ([`results/gd3_1_probe_qwen25_7b/`](results/gd3_1_probe_qwen25_7b/)) | L4 (`1.94 CU/h`) | Đã tắt (`0` active) | `65.22` | `62.01` | **`3.21`** |
| `exp5llm` | Plan v3 Tầng 2 Sanity V1 ([`results/gd3_2_lora_sanity/`](results/gd3_2_lora_sanity/)) | L4 (`1.94 CU/h`) | Đã tắt (`0` active) | `62.01` | `61.29` | **`0.72`** |
| `exp5llm` | Plan v3 Tầng 2 Sanity V2 ([`results/gd3_2_lora_sanity/`](results/gd3_2_lora_sanity/)) | L4 (`1.94 CU/h`) | Đã tắt (`0` active) | `61.29` | **`60.47`** | **`0.82`** |
| **Tổng cộng** | **8 mốc thực nghiệm (toàn bộ log đã lưu)** | — | **`0` active** | **`67.29`** | **`60.47`** | **`6.82`** |
