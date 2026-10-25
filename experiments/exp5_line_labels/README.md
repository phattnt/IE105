# Thí nghiệm 5 – Nhãn từng dòng trên SecVulEval (+ loss tương phản lỗi–sửa)

TN1–TN3 chỉ có nhãn hàm, nên định vị dòng phải dựa vào attention. SecVulEval có `changed_statements` của bản vá,
đủ để gán **nhãn cho từng dòng**: dòng của bản lỗi bị bản vá sửa/xóa là 1, mọi dòng khác (của bản lỗi, của bản đã
vá và của hàm lành không liên quan) là 0. Câu hỏi: *giám sát trực tiếp ở mức dòng có định vị tốt hơn không, và có
phân biệt được bản lỗi với bản đã vá (cùng độ dài, gần như cùng chữ) không?*

## Thiết kế

| | |
|---|---|
| Dữ liệu | SecVulEval → `prepare_secvuleval.py`: nhãn dòng, cặp lỗi/đã vá, chia **theo project** (test là project chưa thấy). Bỏ commit message, CVE, đường dẫn file khỏi input |
| Rò rỉ | hàm train trùng (bỏ khoảng trắng) với một hàm `primevul_test` bị loại cùng hàm cặp của nó (679 hàm) |
| Mô hình | ModernBERT-base, `slide` cửa sổ 2048 token (bước 1023), mean-pool token của mỗi dòng → logit dòng; điểm hàm = max logit dòng |
| Loss | BCE theo dòng (pos_weight ≤ 20); thêm tùy chọn hinge **dòng** (dòng lỗi phải hơn dòng tương ứng ở bản vá `margin` logit, `--lam_line`) và hinge **hàm** (logsumexp các dòng bản lỗi > bản vá, `--lam_fn`) |
| Train | AdamW 2e-5, 8192 token/batch, bf16, 4 epoch, `--patience 2`, chọn epoch trên valid bằng Top-5 + % cùng-chữ |
| Chấm | test SecVulEval: Top-1/5/10, IFA, cặp, **cùng-chữ** (dòng lỗi có chữ y hệt một dòng ở bản vá – không thể ăn may nhờ độ dài); PrimeVul (`eval_primevul.py`): cùng giao thức TN1–TN3 |
| Đối chứng | `--strategy controls` (`results/controls.json`): độ dài (hồi quy logistic trên log token/log dòng; dòng dài nhất trước), mỏ neo TN2, ngẫu nhiên |

## Kết quả đã có (seed 0)

Test SecVulEval (1.037 hàm lỗi có nhãn dòng, 979 cặp) và PrimeVul (357 hàm lỗi có nhãn dòng, 564 cặp, toàn bộ 25.911 hàm).

| Lần chạy | λ dòng / λ hàm | Epoch (tốt nhất) | Top-1 / 5 / 10 | IFA | Cùng-chữ | Cặp | PV Top-10 | PV cặp | PV AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| `s0` | 0 / 0 | 3 (ep2) | 29,5 / 52,2 / 67,6 | 4 | 53,4 | 53,8 | 54,3 | 58,9 | 0,017 |
| `s0_contrast` ¹ | 1 / 1 | 3 (ep0) | 20,7 / 45,5 / 59,9 | 6 | 55,3 | 68,3 | 49,3 | 71,9 | – |
| `s0_conv` | 0 / 0 | 4, pat. 2 (ep1) | 27,5 / 53,6 / **68,8** | 4 | 53,5 | 58,0 | **59,4** | 62,4 | 0,017 |
| `s0_contrast_conv` | 1 / 1 | 4, pat. 2 (ep2) | 25,7 / 50,0 / 62,9 | 5 | **60,3** | 62,3 | 49,9 | 62,3 | 0,020 |
| `s0_contrast03_conv` (22/10) | 0,3 / 1 | 4, pat. 2 (ep3) | 25,2 / 51,6 / 66,2 | 5 | 52,2 | 55,2 | 50,4 | 60,6 | 0,020 |
| `unixcoder_1024_s0_conv` (22/10) ² | 0 / 0 | 4, pat. 2 (ep0) | 28,1 / 55,7 / **70,0** | 3 | 57,5 | 57,5 | **60,8** | 59,9 | 0,017 |
| đối chứng độ dài | – | – | – / – / 62,0 | | – | 72,3 | | | |
| ngẫu nhiên | – | – | – / – / 48,2 | | – | 46,8 | | | |

¹ VM bị thu hồi lúc chấm toàn bộ PrimeVul; `results.json` dựng lại từ log, không có `test_scores.jsonl`.
² `microsoft/unixcoder-base`, cửa sổ 1.024 (bước 511), còn lại y hệt; dừng sớm sau epoch 2, ~1,2 giờ ≈ 2 unit.

**Bootstrap theo project** (`bootstrap_ci.py`, 2.000 lần, `contrast_conv − conv`): Top-10 **−5,9** [−8,5; −3,4],
Top-5 −3,7 [−7,8; −0,7], cùng-chữ **+6,8** [1,2; 13,7], cặp +4,2 [−1,2; 8,4].
→ Loss tương phản có giúp phân biệt bản lỗi/bản vá ở cùng chữ, nhưng với λ = 1 nó kéo tụt định vị dòng rõ rệt
(CI không chứa 0 ở cả hai phía). Hinge đang quá nặng so với BCE. Cả hai đều chưa vượt đối chứng độ dài ở chỉ số cặp,
và AUPRC trên PrimeVul vẫn ở mức ngẫu nhiên (tỉ lệ dương 2,7%).

**Run λ_line 0,3 (22/10, L4, ~2 giờ, 3 unit) – không được cả hai mặt.** Bootstrap `contrast03_conv − conv`: Top-10 −2,6 [−5,2; +0,2],
cặp −3,2 [−8,9; +1,3] (ngang plain); `contrast03_conv − contrast_conv`: Top-10 **+3,3** [0,7; 6,5], cặp **−7,4** [−11,2; −3,7].
Cặp sau khi trừ độ dài (`debias_length.py`): SecVulEval plain 51,0 / λ0,3 55,2 / λ1 60,7; PrimeVul 54,3 / 60,0 / 59,8.
→ Hạ λ chỉ dịch điểm dọc theo một đường đánh đổi (tìm dòng ↔ tín hiệu cặp không phụ thuộc độ dài), không có điểm nào được cả hai với 1 seed.
Epoch được chọn (3) do thước cùng-chữ kéo (xem `ANALYSIS.md` mục 1); valid Top-5 của epoch 1 cao hơn (59,0 vs 58,1), chênh trong vùng nhiễu.

**Run UniXcoder 1.024 (22/10, L4, 1,2 giờ, ~2 unit) – đổi backbone không đổi kết quả.** Câu hỏi: backbone học cấu trúc (AST) có tìm dòng tốt hơn ModernBERT không,
tách khỏi hiệu ứng cửa sổ? Theo nhóm (Top-10, SecVulEval / PrimeVul): A (vừa 512 token) 92,6 / 88,3 vs ModernBERT 91,8 / 88,3 – **bằng nhau**;
B 68,4 / 66,4 vs 63,0 / 63,9; C 32,0 / 34,1 vs 36,8 / 32,6 – B hơn, C kém, bù nhau. Bootstrap `unixcoder − conv`: Top-10 +1,3 [−1,4; +4,2],
Top-5 +2,1 [−1,3; +5,4], cặp −0,1 [−5,5; +4,8] – mọi CI chứa 0. Mức hàm/cặp sau khi trừ độ dài: cặp 50,2 (SecVulEval) / 49,0 (PrimeVul) – UniXcoder
bám độ dài còn nặng hơn (Spearman điểm–độ dài −0,79 vs −0,4), toàn tập PrimeVul ROC 0,236 thô → 0,489 sau trừ. Kết luận với 1 seed: định vị dòng
được quyết bởi nhãn dòng + đọc cả hàm, không phải backbone; UniXcoder không có C trong dữ liệu pretrain nhưng cũng không thua. Không chạy thêm backbone.

**Ghép điểm các mô hình (22/10, `ensemble_scores.py`, 0 unit) – không chạm được mức hàm.** Trung bình điểm dòng đã chuẩn hóa z của
2–4 run (điểm hàm = max như cũ; trên PrimeVul chỉ có logit hàm nên ghép ở mức hàm). Bootstrap theo project, ghép − thành viên tốt nhất:

| Ghép | Top-10 SecVulEval | Cặp trừ độ dài SecVulEval | Cặp trừ độ dài PrimeVul | ROC toàn tập PrimeVul sau trừ |
|---|---|---|---|---|
| ModernBERT + UniXcoder | 69,1 (−0,9 [−2,7; +0,7]) | 50,8 (−0,2 [−3,9; +2,9]) | 54,5 (+1,2 [−2,4; +4,1]) | 0,49 |
| A (plain) + B (λ=1) | 66,5 (−2,2 [−4,0; −0,5]) | 61,2 (+0,5 [−2,6; +3,1]) | 58,8 (+0,6 [−2,2; +3,2]) | 0,50 |
| cả 4 run | 69,9 (−0,1 [−3,6; +2,2]) | 57,8 (−2,9 [−6,1; −0,4]) | 59,8 (+1,1 [−2,2; +4,3]) | 0,49 |

→ Không ghép nào vượt thành viên tốt nhất ở bất kỳ cột nào (mọi CI chứa 0 hoặc âm); ROC toàn tập PrimeVul sau khi trừ độ dài vẫn 0,49–0,50 ở
mọi tổ hợp. Hai backbone sai cùng chỗ (tương quan điểm–độ dài của ghép còn mạnh hơn: −0,85), nên ghép chỉ cộng dồn lối tắt độ dài.
Ghép theo vai trò (B chấm hàm/cặp, A chỉ dòng) chỉ là lấy số tốt nhất của từng cột từ hai mô hình, không tạo tín hiệu mới.
Kết luận: bệnh mức hàm nằm ở dữ liệu dạy (lối tắt độ dài), không chữa được bằng ghép điểm; chỉ còn hướng đổi cách dạy (Train 13).

**Phép thử phản thực guard (22/10, `probe_guards.py`, GPU máy nhà, 0 unit) – mô hình gần như mù quan hệ guard–sink.**
250 hàm lỗi test: chèn ngay trên dòng lỗi một guard liên quan `if (!IDENT) return;` (G1), một guard không liên quan `if (error) return;` (G2),
hoặc một dòng chú thích (N); 250 hàm lành: xóa một guard `if (...) return/break/goto;` có sẵn (nạn nhân = dòng code kế tiếp) so với xóa một câu lệnh thường.
Δ = thay đổi logit của dòng lỗi / dòng nạn nhân (độ lệch chuẩn logit dòng lỗi: UniXcoder 1,5; ModernBERT 2,9).

| Checkpoint | chèn G1 | chèn G2 | chèn N | G1 − N | xóa guard | xóa guard liên quan | xóa câu thường |
|---|---|---|---|---|---|---|---|
| UniXcoder 1024 plain (ep0) | −0,07 | −0,05 | +0,01 | −0,08 [−0,10; −0,07] | +0,07 | +0,10 | +0,04 |
| ModernBERT 2048 λ0,3 (ep3) | +0,01 | **+0,20** | **−0,17** | +0,17 | +0,14 | +0,09 | 0,00 |

→ UniXcoder: có phản ứng đúng chiều nhưng ~0,1 logit (1/15 độ lệch chuẩn), chỉ 3% ca đổi quá 0,5; thực tế không dùng guard.
ModernBERT: phản ứng mạnh hơn (20–40% ca đổi quá 0,5) nhưng **không theo quan hệ**: guard không liên quan làm dòng lỗi *tăng* điểm, chú thích làm *giảm*
(Top-10 58 → 51 chỉ vì thêm một dòng chú thích); xóa guard không liên quan (+0,18) tác động mạnh hơn guard liên quan (+0,09). Tức là nó nhạy với
"mật độ câu kiểm tra quanh đây" và với mọi nhiễu ngữ cảnh, không nhạy với "dòng này có được bảo vệ không". Chẩn đoán ở mục trên được xác nhận:
điểm dòng hiện tại đến từ hình dạng của dòng, không từ quan hệ giữa các dòng → Train 13 (cặp phản thực khác đúng một guard) là đúng thuốc.
Cùng script này, đổi chiều (xóa guard khỏi hàm lành của PrimeVul train) là bộ sinh dữ liệu cho Train 13.

## Train 13 – học sơ bộ bằng cặp phản thực "rút guard" (22–23/10)

`make_synthetic_guards.py`: với mỗi hàm lành của `primevul_train` có một guard `if (...) return/break/continue/goto;` (1–3 dòng), xóa guard đó
→ bản "lỗi giả" với đúng một dòng nhãn 1 = dòng kế tiếp (câu lệnh mất bảo vệ); bản gốc là bản "đã vá". Hai bản chỉ khác 1–3 dòng nên không có
lối tắt độ dài. Loại hàm trùng chữ với SecVulEval/PrimeVul test. 40.000 cặp train (ưu tiên guard chia sẻ biến với dòng nạn nhân: 64%), 2.000 cặp
valid; test = SecVulEval test. Hàng đợi: `synth1` (ModernBERT 2048, λ_line 1, λ_fn 1, 1 epoch) → `s13_conv` (dạy tiếp trên SecVulEval như `s0_conv`,
`--init`) → `s13bal_conv` (như trên + `--balance_length`). Chọn epoch bằng `--select top5_pair_debias`. ~7 unit.

**Giai đoạn 1 (`synth1`, 1 giờ, ~1,5 unit) – chưa thấy một lỗi thật nào:** valid tự tạo Top-1/5/10 74,7 / 91,4 / 95,8, cặp 99,1.
Test SecVulEval: Top-10 **64,3** (A **91,8** = bằng `s0_conv`; B 58,9; C 23,8), cặp 56,3 → **55,4** sau trừ độ dài, tương quan điểm–độ dài **−0,21**
(các run học trên SecVulEval: −0,4 … −0,8). PrimeVul Top-10 dòng 51,8. Phép thử guard (`probe_guards.py`) trên checkpoint này:

| Checkpoint | chèn G1 | chèn G2 | chèn N | xóa guard | xóa guard liên quan | xóa câu thường |
|---|---|---|---|---|---|---|
| `synth1` | **−1,66** (77% ca > 0,5) | −1,44 | +1,21 | **+6,00** (99% ca) | +6,09 | +0,82 |
| (các run trước, để so) | −0,07 … +0,01 | | | +0,07 … +0,14 | | |

→ Mô hình học được quan hệ "có/không có câu kiểm tra ngay trên dòng này" rất mạnh (xóa guard: +6 logit; các run cũ: +0,1). Hai điểm cần nói thật:
(1) G2 ≈ G1 và xóa guard không liên quan ≈ liên quan: nó học "có guard ở trên" chứ chưa phải "guard kiểm tra đúng biến của tôi" – hợp lý vì 36% guard
trong dữ liệu tự tạo không chia sẻ biến; (2) chèn một dòng chú thích làm dòng lỗi *tăng* 1,2 logit: lối tắt của dữ liệu tự tạo – dòng nạn nhân thường
đứng ngay sau một dòng không phải câu lệnh (chú thích, khai báo, dòng trống) vì guard hay nằm đầu khối. Phiên bản 2 của dữ liệu cần thêm cặp
đối chứng "xóa một dòng chú thích/trống" với nhãn 0 và guard giả có biến không liên quan.

**Giai đoạn 2 (`s13_conv`: dạy tiếp trên SecVulEval như `s0_conv`, 1,9 giờ, ~3 unit, dừng sớm, best ep0).** Test SecVulEval: Top-10 **71,5**
(A 92,8 / B **71,7** / C 32,5), Top-5 56,3. Bootstrap `s13_conv − s0_conv`: Top-10 **+2,7 [+0,7; +5,4]**, Top-5 **+2,7 [+0,4; +5,5]** – cải thiện tìm dòng đầu
tiên có CI không chứa 0; cặp +5,1 [−1,2; +9,5] nhưng là nhờ bám độ dài nặng hơn (ρ −0,88): sau trừ độ dài 52,8 (plain 51,0). PrimeVul: Top-10 58,8
(plain 59,4; B 68,0 / C 28,8), cặp 64,8 → 50,5 sau trừ, toàn tập ROC 0,22 → 0,52. Phép thử guard sau giai đoạn 2: chèn G1 −0,12, xóa guard **+0,28**
(giai đoạn 1: −1,66 / +6,0; run cũ: ±0,1) – kỹ năng guard bị dữ liệu thật ghi đè ~95% sau một epoch, chỉ còn dấu vết (guard liên quan +0,34 > không
liên quan +0,23). → Dạy **nối tiếp** không giữ được kỹ năng; cần **trộn** hai loại dữ liệu trong cùng một run (`--extra_train`, run `_mix_conv`).

| | Run gốc `s0_conv` | gđ1 `synth1` | gđ2 `s13_conv` |
|---|---|---|---|
| SecVulEval Top-10 (A / B / C) | 68,8 (91,8 / 63,0 / 36,8) | 64,3 (91,8 / 58,9 / 23,8) | **71,5** (92,8 / 71,7 / 32,5) |
| SecVulEval cặp thô → trừ độ dài; ρ | 58,0 → 51,0; −0,75 | 56,3 → 55,4; −0,21 | 62,9 → 52,8; −0,88 |
| PrimeVul Top-10 (A / B / C) | 59,4 (88,3 / 63,9 / 32,6) | 51,8 (82,5 / 60,7 / 19,7) | 58,8 (86,4 / 68,0 / 28,8) |
| PrimeVul cặp trừ độ dài; toàn tập ROC sau trừ | 53,3; 0,49 | 54,8; 0,48 | 50,5; 0,52 |
| Probe: chèn guard / xóa guard | – | −1,66 / +6,0 | −0,12 / +0,28 |

**Đổi cách gộp dòng → hàm trên điểm đã lưu (23/10, `aggregate_scores.py`, 0 unit) – bệnh không nằm ở cách gộp.** Thử 9 cách gộp
(max, trung bình 3 dòng cao nhất, logsumexp, trung bình, số/tỉ lệ dòng > 0, max trừ log số dòng, max trừ kỳ vọng max của N dòng) trên 4 run
(`s0_conv`, `s0_contrast_conv`, `synth1`, `s13_conv`), SecVulEval test, chấm thô và sau trừ độ dài. ROC toàn tập sau trừ độ dài: **0,50–0,53 ở
mọi cách gộp, mọi run** (đối chứng độ dài 0,507); cặp sau trừ độ dài không cách nào vượt max quá 2 điểm, các cách đếm dòng còn tệ hơn (35–48).
→ Kết luận dứt khoát: **điểm dòng không chứa thông tin mức hàm ngoài độ dài**; gộp kiểu gì cũng không lấy ra được. Mức hàm chỉ có thể khá lên
khi chính điểm dòng mang tín hiệu khác (kỹ năng guard của `synth1` là ứng viên duy nhất đến giờ, và nó bị dạy nối tiếp xóa mất → run `_mix_conv`).

**Run 3 (`s13bal_conv`: như giai đoạn 2 + `--balance_length` theo ký tự, 1,8 giờ, ~3 unit, best ep2) – cân độ dài bằng cách bớt dữ liệu không giúp.**
Cân theo ký tự bỏ 3.285 cặp (6.570 hàm, 35% tập dạy; bản đếm theo dòng không bỏ cặp nào vì sửa tại chỗ giữ số dòng, run đó bị dừng).
Test SecVulEval: Top-10 **65,8** (A 89,9 / B 62,5 / C 27,7) – kém `s0_conv` 3 điểm và `s13_conv` 6 điểm; cặp 56,0 → **51,0** sau trừ độ dài (= `s0_conv`), ρ −0,69.
PrimeVul: Top-10 54,3, cặp 58,3 → 52,9, toàn tập ROC 0,26 → 0,48. Probe: xóa guard +0,49, chèn G1 −0,27 nhưng chèn chú thích cũng −0,29 → chỉ dấu vết.
→ Mô hình vẫn học lối tắt từ 65% dữ liệu còn lại; bớt dữ liệu chỉ làm tìm dòng tụt. Hướng "cân độ dài" đóng.

**Chấm hàm bằng phản thực (23/10, `counterfactual_score.py`, checkpoint `synth1`, Colab T4 ~65 phút, `results/counterfactual_synth1.json`) –
cũng không thoát khỏi độ dài.** Ý tưởng: thay vì hỏi "dòng này trông nguy hiểm cỡ nào" (bám độ dài), hỏi "điểm dòng này *tụt* bao nhiêu khi chèn
`if (!x) return;` ngay trên nó" – so hai bản của cùng một hàm nên lẽ ra không phụ thuộc độ dài. Lấy K=8 dòng cao nhất làm ứng viên, chèn guard
lần lượt, điểm hàm = tụt lớn nhất / tổng tụt / tụt của dòng cao nhất. Chấm trên 564 cặp PrimeVul test và trên mẫu toàn tập (695 hàm lỗi + 2.000 lành).

| Điểm hàm | Cặp: ROC thô → trừ độ dài | Cặp: đúng cặp thô → trừ | Toàn tập: ROC thô → trừ | ρ độ dài (toàn tập) |
|---|---|---|---|---|
| max dòng (cách cũ) | 0,521 → 0,516 | 59,6 → 57,8 | 0,397 → 0,486 | −0,35 |
| tụt lớn nhất khi chèn guard | 0,513 → 0,516 | 54,3 → **60,3** | 0,565 → 0,485 | +0,14 |
| tổng tụt | 0,494 → 0,505 | 44,6 → 51,5 | 0,738 → 0,505 | +0,78 |
| tụt của dòng cao nhất | 0,514 → 0,518 | 52,0 → 57,7 | 0,531 → 0,489 | +0,04 |
| (chỉ độ dài, CV) | 0,516 | 84,2 | 0,793 | +0,99 |

Sau trừ độ dài mọi cách đều về **0,485–0,518** = ngẫu nhiên; "tổng tụt" ROC thô 0,74 chỉ vì hàm dài có nhiều dòng để chèn (ρ +0,78, đối chứng độ dài 0,79).
Đúng cặp 60,3 hơn cách cũ 2,5 điểm, trong sai số của 564 cặp. → Kỹ năng guard của `synth1` có thật ở mức dòng (probe +6 logit) nhưng **không phân biệt
được hàm lỗi thật với hàm lành của PrimeVul**: lỗi thật phần lớn không phải "thiếu một câu kiểm tra ngay trên dòng này". Hướng mức hàm đóng;
kết quả chính của exp5 là định vị dòng (Top-10 SecVulEval 68,8 → 71,5 nhờ học sơ bộ phản thực) và kết quả âm có kiểm soát độ dài ở mức hàm.

**Chấm lại theo giao thức cặp của tài liệu (23/10, `pair_protocol.py`, `results/pair_protocol.json`; mức cặp từ điểm đã lưu, 0 GPU;
mức dòng chấm lại 3 checkpoint trên Colab T4 ~15 phút, `score_pairs_lines.py`).** Giao thức: P-C/P-V/P-B/P-R của bài PrimeVul
(gắn cờ = logit > 0), điểm ròng = P-C − P-R với mô hình null "gắn cờ độc lập theo tỉ lệ p" (P-C = P-R = p(1−p), ròng 0) và tập cặp
cân độ dài (lệch ký tự < 5%, 337/564 cặp) theo arXiv 2609.32890, bootstrap 2.000 lần trên cặp. Lưu ý: ngay trong tập "cân độ dài"
luật "bản dài hơn là bản vá" vẫn đúng 83,8% (toàn bộ: 87,0%), nên cột "sau trừ độ dài" (dư sau khi trừ E[điểm | log ký tự]) vẫn là kiểm soát chính.

| Run | xếp đúng cặp [CI] | sau trừ độ dài | P-C / P-V / P-B / P-R | ròng [CI] | cân độ dài: xếp đúng / sau trừ / P-C / ròng [CI] |
|---|---|---|---|---|---|
| Run gốc `s0_conv` | 62,5 [58,7; 66,2] | 54,7 | 3,7 / 59,6 / 34,0 / 2,7 | +1,1 [−1,1; +3,2] | 55,0 / 50,9 / 2,4 / +0,3 [−2,1; +2,7] |
| So-cặp-mạnh `contrast_conv` | 62,3 [58,3; 66,1] | 58,2 | 8,2 / 47,5 / 42,9 / 1,4 | **+6,7 [+4,3; +9,4]** | 55,9 / 52,5 / 3,3 / +1,8 [−0,6; +4,2] |
| So-cặp-nhẹ `contrast03_conv` | 60,6 [56,7; 64,5] | **59,1** | **8,9** / 56,4 / 31,9 / 2,8 | **+6,0 [+3,2; +8,9]** | 56,7 / **55,9** / **5,6** / **+3,3 [+0,3; +6,2]** |
| UniXcoder `unixcoder_conv` | 59,8 [55,8; 63,5] | 49,6 | 2,5 / 75,5 / 20,6 / 1,4 | +1,1 [−0,5; +2,8] | 50,7 / 47,0 / 1,5 / −0,6 [−2,7; +1,5] |
| Tự-tạo `synth1` | 58,7 [54,6; 62,8] | 55,0 | 4,6 / 64,0 / 28,4 / 3,0 | +1,6 [−0,5; +3,9] | 55,0 / 57,3 / 3,6 / +1,8 [−0,6; +4,5] |
| Tự-tạo → thật `s13_conv` | **64,8 [61,0; 68,5]** | 51,8 | 3,0 / 43,1 / 52,8 / 1,1 | +2,0 [+0,4; +3,7] | 56,4 / 50,4 / 1,5 / −0,3 [−2,4; +1,5] |
| Run 3 `s13bal_conv` | 58,4 [54,5; 62,3] | 52,0 | 3,9 / 71,3 / 23,4 / 1,4 | +2,5 [+0,5; +4,4] | 50,4 / 48,8 / 3,6 / +1,5 [−0,9; +4,2] |
| Luật độ dài (dài hơn = vá) | 87,0 | – | – | – | 83,8 / – / – / – |
| Tài liệu: fine-tune trên PrimeVul (Ding 2024) | – | – | P-C 1,1–3,0 | – | – |
| Tài liệu: GPT-3.5 CoT / GPT-4 CoT (Ding 2024) | – | – | P-C 6,2 / 12,9 | – | – |
| Tài liệu: VulTriage GPT-4o + ngữ cảnh (2026) | – | – | P-C 33,8 (147/435) | – | – |

Đọc: (1) P-C của mọi run (3–9%) thấp hơn hẳn null P-C (≈ 23%) vì mô hình trả lời **giống nhau cho cả hai bản** (P-V + P-B ≈ 90%),
đúng hiện tượng "phán quyết bám phần chữ chung của cặp" trong 2609.32890; so sánh đúng là điểm ròng với null 0. (2) Hai run so-cặp có
điểm ròng dương có ý nghĩa (+6 đến +7, CI không chứa 0) và P-C 8–9%, cao hơn fine-tune trong bài PrimeVul (1–3%) và GPT-3.5 CoT, thấp hơn
GPT-4 CoT; **chỉ so-cặp-nhẹ còn giữ được tín hiệu trên cặp cân độ dài** (ròng +3,3 [+0,3; +6,2], sau trừ độ dài 55,9). Các run còn lại
≈ null. (3) `s13_conv` xếp cặp cao nhất (64,8) nhưng là nhờ độ dài: sau trừ chỉ 51,8, cân độ dài ròng −0,3. → Có một tín hiệu cặp nhỏ nhưng
thật ở run so-cặp, cỡ 1/4 khoảng cách tới GPT-4 CoT; không phải "ngẫu nhiên hoàn toàn", cũng không đủ để dùng.

**Mức dòng trên PrimeVul kiểu ANVIL (arXiv 2408.16028): chỉ xếp dòng code (bỏ dòng trống/chú thích), hàm lỗi có ≥ 1 dòng lỗi
(357 hàm, 1.301 dòng lỗi; ANVIL dùng 273 hàm / 924 dòng với lọc riêng của họ nên không trùng tập hoàn toàn).**

| | Top-1 | Top-3 | Top-5 | Top-10 | MFR ↓ | N-MFR ↓ | ROC-AUC dòng |
|---|---|---|---|---|---|---|---|
| Tự-tạo → thật `s13_conv` | **16,8** [13,2; 20,7] | **32,5** [27,7; 37,5] | **45,1** [40,1; 50,1] | 59,4 | 26,7 | 0,22 | 0,754 |
| So-cặp-nhẹ `contrast03_conv` | 16,8 [12,9; 21,0] | 30,0 [25,5; 34,7] | 39,5 [34,5; 44,3] | 50,4 | 30,0 | 0,24 | **0,767** |
| Tự-tạo `synth1` (0 nhãn thật) | 11,8 [8,4; 15,1] | 25,5 [21,0; 30,0] | 35,0 [30,0; 40,1] | 53,2 | 35,2 | 0,25 | 0,652 |
| Ngẫu nhiên (kỳ vọng) | 7,0 | 17,6 | 26,0 | 41,5 | 46,3 | – | 0,500 |
| ANVIL (không train, LLM điền chỗ trống) | 11,7 | 31,3 | 41,4 | – | 25,0 | 0,18 | 0,618 |
| LineVul dạy trên BigVul (theo ANVIL) | 4,0 | 17,0 | 30,0 | – | 17,6 | – | 0,496 |
| LineVD / LLMAO dạy trên BigVul (theo ANVIL) | 9,2 / 9,2 | 15,4 / 17,3 | 23,1 / 26,2 | – | 38,0 / 49,3 | – | 0,50 / 0,52 |

Đọc: `s13_conv` hơn ANVIL ở Top-1 (+5), Top-5 (+4) và ROC-AUC dòng (0,75 so với 0,62), ngang ở Top-3 và MFR (CI ±4–5 điểm nên Top-3/MFR
không phân biệt được); hơn hẳn LineVul/LineVD/LLMAO dạy trên BigVul. `synth1` chưa thấy nhãn thật nào mà đã ngang ANVIL. Đây là chỗ exp5 có
tiến bộ thật và có mốc tài liệu để so, với dè dặt: tập hàm và định nghĩa dòng lỗi không trùng hoàn toàn với ANVIL.

## Lần chạy kế tiếp (Colab Pro, L4)

Quyết định ngân sách (22/10): chạy **một** run trước, rồi mới quyết định có chạy thêm seed không. Run đã chạy:

| Run | Loss | Seed | Trả lời gì |
|---|---|---|---|
| `slide_ModernBERT-base_2048_s0_contrast03_conv` | contrast nhẹ: `--lam_line 0.3 --lam_fn 1.0 --margin 1` | 0 | λ_line 0,3 có giữ được định vị dòng của plain mà vẫn giữ tín hiệu cặp không phụ thuộc độ dài của λ=1 không |

Chung: `--strategy slide --model answerdotai/ModernBERT-base --max_len 2048 --train_tokens 8192 --bf16 --epochs 4 --patience 2 --ckpt_every 400`.
Ước tính trên L4 (theo log cũ): contrast ~1,45 giờ + chấm PrimeVul ~15 phút + chuẩn bị dữ liệu ~15 phút → **~2 giờ ≈ 3 compute unit**.

Kết quả (22/10): vẫn phải đánh đổi, xem bảng trên. Các seed 1, 2 là preset `conv_s1/2`, `contrast03_s1/2` trong `colab/presets.py`, ~8 giờ ≈ 12 unit,
nhưng chỉ đáng chạy cho cấu hình được chọn làm kết quả chính, không phải cho λ 0,3.

Cách chấm mức hàm/cặp từ giờ dùng `debias_length.py` (trừ phần điểm giải thích được bằng độ dài, không cần train lại):
sau khi trừ, plain mất hết tín hiệu cặp (58 → 51%) còn contrast λ=1 giữ gần nguyên (62 → 61% SecVulEval, 58–60% PrimeVul);
trên toàn tập test PrimeVul mọi run đều về đúng ngẫu nhiên (ROC-AUC 0,49), nên mục tiêu "hàm có lỗi không" trên tập mất cân bằng được gác lại.

Chạy trên Colab: `colab/README.md` (`colabctl`, 25/10). Các run trên là các preset trong `colab/presets.py`
(`conv_s1`, `contrast03_s1`, `s13_s1`, …); vd. `colabctl new --gpu L4`, `colabctl start conv_s1`, `colabctl mirror conv_s1`
(kết quả về `results/<run>/`, checkpoint về `data/colab_ckpt/<run>/`). Tính CI như cũ:

```bash
python bootstrap_ci.py --data ../../data/secvul_lines --runs results/slide_ModernBERT-base_2048_s0_conv results/slide_ModernBERT-base_2048_s1_conv --out ../../data/bootstrap_s0_vs_s1_conv.json
```

## Đợt 24–25/10: baseline 2 seed, kiểm tra dữ liệu, ngữ cảnh liên hàm cho LLM

### Baseline `_base2` (preset `base`, 2 seed)

Công thức `s0_conv` với 3 thay đổi về tối ưu: lịch LR giảm tuyến tính về 0 trong **2 epoch** (trước: 4 epoch + patience, đỉnh valid
luôn rơi vào lúc LR còn gần cực đại), valid 4 lần mỗi epoch, chọn epoch theo Top-5 + Top-10. Dữ liệu, loss, mô hình giữ nguyên.

| SecVulEval test (mới − `s0_conv`, CI 95% bootstrap theo project) | seed 0 | seed 1 |
|---|---|---|
| Top-1 | +0,4 [−2,4; +3,4] | +1,6 [−0,7; +4,0] |
| Top-5 | +2,5 [−0,4; +4,7] | +0,8 [−2,1; +2,6] |
| Top-10 | +0,9 [−1,4; +3,3] | +0,2 [−1,8; +2,1] |
| xếp đúng cặp | +0,5 | −0,2 |
| cùng dòng chữ, bản lỗi điểm cao hơn | −4,0 [−5,6; +0,5] | −5,8 [−7,8; +1,6] |

Đọc: **ngang `s0_conv` với khoảng nửa thời gian GPU**, không phải cải thiện (Top-5 +2,5 của seed 0 không lặp lại). Valid
phẳng từ epoch 1,25; train lâu hơn (run cũ, epoch 3: loss train 0,56, valid giảm) là học thuộc nhãn. Loss train (BCE dòng
có trọng số) chỉ thấp hơn mức "đoán hằng số" (0,93 trên train) khoảng 10–17%, valid BCE (1,6–2,5) còn cao hơn mức đó
(1,10) dù Top-k tốt: mô hình học thứ tự dòng trong hàm, không học xác suất tuyệt đối. PrimeVul Top-1/5/10:
16,8/42,9/58,5 (s0), 16,5/45,1/57,7 (s1), so với 19,3/41,2/59,4 của `s0_conv`. `s13b` (synth1 + công thức này, 2 seed) không
chạy xong và được bỏ: lợi ích dự kiến +1–3 điểm dòng, cùng cỡ với chênh lệch giữa hai seed.

### Kiểm tra dữ liệu (CPU)

| | Kết quả | File |
|---|---|---|
| Loại bản vá, 40 cặp test ngẫu nhiên đọc tay | 29/40 (72%) "có A thiếu B"; **12/40 (30%, CI 18–45%) cần thông tin ngoài hàm** (hợp đồng callee, đảm bảo của caller, ngữ nghĩa giao thức/phần cứng); 15/40 quyết được trong hàm | `results/patch_taxonomy/manual_labels_test40.json` |
| Phân loại tự động bằng regex (`patch_taxonomy.py`) | không dùng được cho câu "cần ngoài hàm": khớp tay 17/40, đếm 52% so với 30% | `results/patch_taxonomy/` |
| Rò rỉ theo commit (`leak_audit.py`) | SecVulEval train/test: 0 commit chung. 21 cặp PrimeVul test chung commit với SecVulEval train, 5 với PrimeVul train; bỏ chúng đi mọi số lệch ≤ 0,6 | `results/leak_audit.json` |
| Đối chứng "dòng dài nhất trước", PrimeVul | Top-1/5/10 10,4/35,3/52,4 (mô hình 17–19/45–50/58–62) | `results/controls_primevul_lines.json` |

### Ngữ cảnh liên hàm cho LLM (giai đoạn 2)

`build_context.py` lấy ngữ cảnh ở **commit cha của bản vá** qua GitHub API (không clone): định nghĩa các hàm được gọi và
chỗ gọi hàm này, trong các file bản vá chạm tới, header cùng thư mục và header chúng include. Hai bản của một cặp dùng
**chung một ngữ cảnh**. Đối chứng `context_fake`: cùng file, cùng độ dài, hàm không được gọi và chỗ gọi hàm khác.
Phủ: 1065/1071 commit, 531/564 cặp test có ngữ cảnh (478 có định nghĩa callee, 340 có chỗ gọi), ~2000 ký tự/cặp.
Qwen3.5-9B (vLLM FP8, A100), `llm_audit.py` nhánh `audit` / `audit_ctx` / `audit_ctxfake`, cùng prompt guard-audit.

Phép so đăng ký trước: net (P-C − P-R) của ngữ cảnh thật − ngữ cảnh giả trên 337 cặp cân độ dài, đạt nếu ≥ +5 và CI không chứa 0.

| 337 cặp cân độ dài | thật − giả | thật − không ngữ cảnh | giả − không ngữ cảnh |
|---|---|---|---|
| ngưỡng 0 (quyết định của mô hình) | +2,1 [−2,9; +7,7] | −0,3 [−6,4; +5,0] | −2,4 |
| gắn cờ 50% | +8,3 [+0,0; +17,5] | +3,6 [−5,3; +13,2] | −4,7 |

**Không đạt.** Ngữ cảnh thật không hơn không có ngữ cảnh; phần hơn ngữ cảnh giả chủ yếu do ngữ cảnh không liên quan làm
LLM kém đi. Tách nhỏ (`results/llm_ctx_test/by_type.json`): ở 149 cặp mà dòng sửa gọi đúng một hàm có định nghĩa trong ngữ
cảnh, thật − không là −2,0 (ngưỡng 0) và +7,4 (50%), CI rộng ±12; không nhóm con nào cho tín hiệu cùng chiều ở cả hai ngưỡng.
Bỏ 7 cặp có dòng thay đổi trùng chữ với code trong ngữ cảnh: không đổi. Giới hạn: ~90% câu trả lời bị cắt ở ngân sách
suy nghĩ 1536 token, ngữ cảnh làm phần phải đọc dài thêm; chưa thử ngân sách lớn hơn.

```bash
python build_context.py --primevul ../../data/primevul --cache ../../data/ctx --splits valid test --out inputs
# VM: colabctl -s exp5a100 new --gpu A100; start vllm; push inputs/ctx_test_pairs.jsonl /content/exp5/upload/...; start llm_ctx_test
python pair_protocol.py --primevul ../../data/primevul --runs results/llm_ctx_test/scores/audit results/llm_ctx_test/scores/audit_ctx results/llm_ctx_test/scores/audit_ctxfake
```
