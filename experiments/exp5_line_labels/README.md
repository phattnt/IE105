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

## Lần chạy kế tiếp (Colab Pro, L4)

Quyết định ngân sách (22/10): chạy **một** run trước, rồi mới quyết định có chạy thêm seed không. Hàng đợi mặc định trong `colab_exp5.py`:

| Run | Loss | Seed | Trả lời gì |
|---|---|---|---|
| `slide_ModernBERT-base_2048_s0_contrast03_conv` | contrast nhẹ: `--lam_line 0.3 --lam_fn 1.0 --margin 1` | 0 | λ_line 0,3 có giữ được định vị dòng của plain mà vẫn giữ tín hiệu cặp không phụ thuộc độ dài của λ=1 không |

Chung: `--strategy slide --model answerdotai/ModernBERT-base --max_len 2048 --train_tokens 8192 --bf16 --epochs 4 --patience 2 --ckpt_every 400`.
Ước tính trên L4 (theo log cũ): contrast ~1,45 giờ + chấm PrimeVul ~15 phút + chuẩn bị dữ liệu ~15 phút → **~2 giờ ≈ 3 compute unit**.

Kết quả (22/10): vẫn phải đánh đổi, xem bảng trên. Các seed 1, 2 vẫn để sẵn trong `colab_exp5.py` (`EXP5_EXTRA_SEEDS=1` hoặc `--only`), ~8 giờ ≈ 12 unit,
nhưng chỉ đáng chạy cho cấu hình được chọn làm kết quả chính, không phải cho λ 0,3.

Cách chấm mức hàm/cặp từ giờ dùng `debias_length.py` (trừ phần điểm giải thích được bằng độ dài, không cần train lại):
sau khi trừ, plain mất hết tín hiệu cặp (58 → 51%) còn contrast λ=1 giữ gần nguyên (62 → 61% SecVulEval, 58–60% PrimeVul);
trên toàn tập test PrimeVul mọi run đều về đúng ngẫu nhiên (ROC-AUC 0,49), nên mục tiêu "hàm có lỗi không" trên tập mất cân bằng được gác lại.

Script chạy trên VM: clone/pull repo (nhánh `main` trên GitHub – **push code trước**), cài gói, tải PrimeVul, sinh
`secvul_lines`, rồi lần lượt `train_lines.py` → `eval_primevul.py`. Run đã có `results.json` không train lại, run đã
có `primevul_results.json` bị bỏ qua, nên sau khi VM reset chỉ cần chạy lại. Nếu có `ckpt_last.pt` trong thư mục
run thì train tiếp từ đó.

```bash
cd experiments/exp5_line_labels
colab new -s exp5 --gpu L4
colab upload -s exp5 colab_exp5.py /content/colab_exp5.py
# chạy tách khỏi CLI (kết nối CLI rớt sau ~20 phút)
echo 'import subprocess, sys; subprocess.Popen([sys.executable, "/content/colab_exp5.py"], stdout=open("/content/exp5_launch.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)' | colab exec -s exp5

# theo dõi: mỗi bước, mỗi lần lưu checkpoint, mỗi epoch valid đều ghi vào queue.log
colab download -s exp5 /content/exp5/logs/queue.log queue.log && tail -20 queue.log
colab ls -s exp5 /content/exp5/logs
colab download -s exp5 /content/exp5/logs/slide_ModernBERT-base_2048_s0_contrast03_conv.log s0_c03.log
# đề phòng VM bị thu hồi: kéo checkpoint về, sau reset upload lại vào /content/exp5/<run>/ckpt_last.pt rồi chạy lại lệnh echo ở trên
colab download -s exp5 /content/exp5/slide_ModernBERT-base_2048_s0_contrast03_conv/ckpt_last.pt ckpt_last.pt
```

Tải kết quả về `results/` (mỗi run: `results.json`, `test_scores.jsonl`, `primevul_results.json`,
`primevul_test_scores.jsonl`, log) rồi tính CI:

```bash
for r in slide_ModernBERT-base_2048_s0_contrast03_conv slide_ModernBERT-base_2048_s1_conv slide_ModernBERT-base_2048_s1_contrast03_conv \
         slide_ModernBERT-base_2048_s2_conv slide_ModernBERT-base_2048_s2_contrast03_conv; do
  mkdir -p results/$r
  for f in results.json test_scores.jsonl primevul_results.json primevul_test_scores.jsonl; do
    colab download -s exp5 /content/exp5/$r/$f results/$r/$f
  done
  colab download -s exp5 /content/exp5/logs/$r.log results/$r/train.log
done
colab stop -s exp5
python bootstrap_ci.py --data ../../data/secvul_lines --runs results/slide_ModernBERT-base_2048_s0_conv results/slide_ModernBERT-base_2048_s0_contrast03_conv --out ../../data/bootstrap_s0_conv_vs_c03.json
```

Ghi chú: `colab exec` có timeout mặc định 30 giây, đủ cho lệnh `Popen` (nó trả về ngay). `python /content/colab_exp5.py --dry` in
các lệnh sẽ chạy; `--only <tên hoặc chuỗi con>` chỉ chạy một phần hàng đợi (vd. `--only s0_contrast03`). Theo README của CLI, bản 0.7.5 chỉ hỗ trợ
chính thức Linux/macOS; trên Windows nếu lệnh nào lỗi thì chạy CLI trong WSL.
