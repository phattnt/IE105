# Tổng kết exp5 đợt 24–25/10: listwise loss, baseline base2, vấn đề tồn đọng

Bản này soạn để đưa lên Notion (database Ghi chú, loại Kết luận). Workspace Notion đã hết free blocks nên chưa tạo được
trang; ticket "Viết tổng kết exp5 đợt 24–25/10" và 6 dòng trong Bảng kết quả thí nghiệm đã có trên Notion.

**Tóm tắt.**

- **baseline mới:** base2, gồm ModernBERT-base 2048, BCE, 2 epoch, 2 seed.
- **lw_s0:** listwise loss cộng 0,3 BCE, 1 seed.
- **Kết quả lw_s0:**
  - Top-5 trên SecVulEval tăng nhẹ: 58,1 so với 56,1 / 54,4.
  - pair acc tụt 5 điểm, nên FAIL theo rule đã đăng ký.
- **Mức function:** cả 3 run vẫn ở mức chance.
- **Pair acc:** sau khi bỏ length shortcut, không model nào phân biệt được vulnerable với fixed.
- **Hướng tiếp theo** (đã họp, chờ duyệt): chấm checkpoint OpenVul, rồi GRPO A/B có line reward. Biên bản họp:
  https://app.notion.com/p/404ef02bd54a8114aa89d2193170aa99

## 1. Kết quả mới

### 1.1 SecVulEval test

- 2.606 function, trong đó 1.061 vulnerable (prevalence 40,7%).
- 1.037 function có flaw line.
- 979 pair.

| Run | Top-1 | Top-5 | Top-10 | median IFA | Top-5 A / B / C | pair acc (raw) | pair acc (debiased) | same_text | AUPRC | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 11,4 | 34,3 | 48,7 | – | – | 50 | 50 | 50 | 0,407 | 0,5 |
| base2 s0 | 27,9 | 56,1 | 69,6 | 3 | 81,3 / 48,8 / 22,9 | 58,5 | 52,2 | 49,5 | 0,431 | 0,524 |
| base2 s1 | 29,1 | 54,4 | 68,9 | 3 | 78,2 / 48,3 / 21,6 | 58,0 | 53,1 | 47,7 | 0,435 | 0,531 |
| **lw_s0** | **30,3** | **58,1** | **70,8** | 3 | **83,2 / 49,1 / 27,7** | 53,7 | 50,1 | 54,8 | 0,427 | 0,518 |

**Bootstrap theo project** (CI95, chênh lệch lw − base2):

| Metric | vs s0 | vs s1 |
| --- | --- | --- |
| Top-1 | +2,4 [−0,5; 5,2] | +1,2 [−1,2; 4,2] |
| Top-5 | +1,9 [−0,9; 5,2] | **+3,7 [1,1; 7,1]** |
| Top-10 | +1,2 [−0,8; 3,7] | +1,8 [−0,2; 4,7] |
| pair acc (non-tied) | **−5,2 [−9,8; −1,6]** | −4,4 [−10,4; 0,0] |
| same_text | +5,3 [−3,0; 7,2] | +7,1 [−3,3; 9,0] |

**Các phân tích thêm** (subagent trong cuộc họp):

- **precision@5 và recall@5:** precision@5 khoảng 22%, recall@5 khoảng 40%, như nhau ở cả 3 run. lw chỉ sắp xếp lại các
  dòng đầu, không tìm thêm được flaw line.
- **Top-1 trên function có đúng 1 flaw line:** lw khoảng 20%, base2 khoảng 18%. Chênh lệch +2,5 / +3,2, CI chứa 0.
- **Length shortcut:** quy tắc "bản dài hơn là bản fixed" đúng 81,2% số pair trong test. Phần lớn mức tụt pair acc của lw là
  do mất shortcut này. Spearman giữa score và số token: base2 −0,81 / −0,73, lw −0,68.
- **Label noise:** bỏ 19% số function test đáng ngờ thì Top-5 chỉ tăng 0,3–0,7 và thứ hạng giữ nguyên. Vậy noise không phải
  trần chính.
- **Contradiction mask:** đã code nhưng bỏ trước khi train, vì:
  - 41% số dòng bị mask là dòng trắng hoặc chỉ có dấu ngoặc (trivial);
  - khoảng 40% là trùng text ngẫu nhiên;
  - label 0 ở bản fixed vốn đúng.

### 1.2 Valid curve (Top-5 trên 1.081 function valid)

| epoch | 0,25 | 0,5 | 0,75 | 1,0 | 1,25 | 1,5 | 1,75 | 2,0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base2 s0 | 56,4 | 55,9 | 61,8 | 63,6 | 64,4 | 64,3 | **65,8** | 64,8 |
| base2 s1 | 55,9 | 58,6 | 64,7 | 64,4 | 64,6 | 63,7 | **66,4** | 64,8 |
| lw_s0 | 61,5 | 60,6 | 62,8 | 67,4 | 67,4 | 65,8 | **67,7** | 66,7 |

lw học nhanh hơn ở epoch đầu: +5 điểm tại 0,25. Đến cuối, khoảng cách còn +1,3 đến +1,9.

### 1.3 Chấm chuyển miền sang PrimeVul (không train lại)

- Dòng: 357 function vulnerable có pair, 564 pair.
- Function: full test gồm 25.911 function, 695 vulnerable, chance AUPRC 0,027.

| Run | Top-1 | Top-5 | Top-10 | Top-10 A / B / C | pair acc | full-test AUPRC | full-test ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 5,9 | 22,6 | 36,7 | – | 50 | 0,027 | 0,5 |
| base2 s0 | 16,8 | 42,9 | 58,5 | 90,3 / 64,8 / 28,0 | 62,7 | 0,0172 | 0,228 |
| base2 s1 | 16,5 | 45,1 | 57,7 | 87,4 / 59,8 / 32,6 | 59,8 | 0,0172 | 0,258 |
| lw_s0 | **20,4** | **45,9** | **61,6** | 90,3 / 66,4 / 34,8 | 54,9 | 0,0217 | 0,309 |

- Trên PrimeVul, ở mức dòng lw tốt nhất trong 3 run.
- Ở mức function thì cả 3 run đều **dưới chance**: ROC-AUC 0,23–0,31. Score đang ngược chiều với nhãn vì tương quan âm với
  độ dài.

### 1.4 Vận hành Colab

Thêm safeguards vào `colabctl.py` và đã test trên một CPU VM:

- `mirror --down`: theo dõi hết queue, chạy `final_pull` để kéo ckpt_best và các file result, rồi release VM.
- `watch`: exit 10 khi gặp JOB_ENDED, SESSION_GONE, KERNEL_UNREACHABLE hoặc IDLE_BILLING.
- `keep --idle/--until`.

lw_s0 chạy trên L4 từ 11:37 đến 13:28, tốn **3,15 unit**. Sau đó VM tự tắt và kéo về đủ file. Số dư còn **84,43 unit**.

## 2. Vấn đề tồn đọng

### Các vấn đề về kết quả

1. **Function detection vẫn ở mức chance.**
   - SecVulEval: AUPRC 0,427–0,435, chance là 0,407; ROC-AUC 0,52–0,53.
   - PrimeVul full test: AUPRC 0,017–0,022, dưới chance 0,027.
   - Encoder mức dòng chưa thay được function detector.
2. **Length shortcut.**
   - Raw pair acc 58 là nhờ độ dài. Sau khi debias, chỉ còn 50,1–53,1.
   - Các cách đã thử trước đây đều không chữa được: balance_length, contrastive, synth guard.
3. **lw FAIL theo rule đã đăng ký.**
   - Top-5 chỉ vượt seed 1 một cách chắc chắn (CI không chứa 0), còn so với seed 0 thì CI chứa 0.
   - Pair acc 53,7 dưới guardrail 55.
   - Giữ lw làm nguồn line hint tiềm năng; không chạy thêm seed.
4. **Precision ở mức dòng thấp.**
   - precision@5 khoảng 22%: trung bình trong 5 dòng đưa ra chỉ có 1,1 dòng đúng.
   - recall@5 khoảng 40%.
5. **Group C yếu** (mọi flaw line đều nằm ngoài 512 token đầu).
   - Top-5 trên SecVulEval chỉ 21,6–27,7; median IFA 20–24.
   - Group A đạt 78–83.
6. **Chưa có metric hunk-tolerant (±2 dòng).** Metric exact-line có thể đang đánh giá thấp cả 3 run. Cần rescoring trên CPU.

### Data leakage

7. **Data leakage giữa SecVulEval và PrimeVul.**
   - 757 trong 2.606 function test SecVulEval (29%) có trong PrimeVul train; 216 function có trong PrimeVul test.
   - Kết quả `pv1` (train trên PrimeVul) có thể đã "thấy" một phần test, cần gắn cờ.
   - OpenVul train trên cả hai bộ, nên phải de-dup theo commit, hash và CVE trước khi chấm.

### Hướng LLM (chưa bắt đầu)

8. **OpenVul chưa được kiểm chứng trên protocol của mình.**
   - Bài báo báo P-C 21,08 trên split riêng.
   - Gate để đi tiếp: P-C ≥ 12 và P-C − P-R > +6.
9. **Chi phí GRPO.**
   - Khoảng 2–3 phút mỗi step, tức 21–32 unit mỗi arm cho 120 step.
   - Trần 40 unit chỉ đủ cho khoảng 12 unit mỗi arm.
   - Fallback là rejection sampling kết hợp SFT.
10. **Line reward chưa được kiểm chứng.** Cần check 40 prompt phía train xem hunk có chứa root cause ≥ 70% hay không.

### Việc dở dang

11. **Chưa commit.** Các thay đổi chưa commit gồm:
    - `colabctl`, `vm.py`, `presets.py`;
    - listwise trong `train_lines.py`;
    - `build_context`, `llm_audit`;
    - các thư mục results.

    README đợt 24–25/10 chưa viết phần listwise và safeguards.
12. **Ticket "Thử model hai đầu ra"** có thể bị thay thế nếu LLM đảm nhận phần phát hiện function.

## 3. Việc tiếp theo (chờ duyệt)

0. Commit, viết README, chấm lại theo hunk-tolerant, làm leakage audit. Không tốn unit; riêng việc tải data OpenVul cần
   duyệt.
1. Phase 1: chạy probe 50 pair trên L4, rồi chấm checkpoint OpenVul SFT-ep3 và GRPO, tối đa 6 unit.
2. Chuẩn bị preset GRPO và periodic checkpoint pull trong colabctl. Không tốn unit.
3. Phase 2:
   - probe tối đa 8 unit;
   - arm A chỉ có verdict reward, arm B có thêm line reward (w = 0,25);
   - tổng cap 40 unit.

## 4. Giải thích từ viết tắt và thuật ngữ

### Model, loss và train

| Từ | Nghĩa |
| --- | --- |
| **base2** | Baseline mới: ModernBERT-base, sliding window 2048 token (stride 1023), BCE trên từng dòng, lr 3e-5, 2 epoch |
| **lw / lw_s0** | Run dùng **l**ist**w**ise loss, seed 0 |
| **s0, s1** | Seed 0 và seed 1 của cùng một cấu hình |
| **BCE** | Binary Cross-Entropy: loss phân loại từng dòng là lỗi hay không (pos_weight 20) |
| **listwise loss** | Loss xếp hạng theo cả danh sách dòng của một view: lse(mọi dòng) − lse(các flaw line), đẩy flaw line lên đầu |
| **lse** | log-sum-exp, dạng "max mềm" của một tập logit |
| **chance (trong log)** | Giá trị listwise loss khi model đoán đều, bằng log(số dòng / số flaw line). Loss phải xuống dưới mức này mới là học được |
| **view** | Một cửa sổ 2048 token cắt từ function |
| **MLP head** | Lớp phân loại nhỏ đặt trên vector của mỗi dòng (mean-pool các token của dòng) |
| **slide_max** | Function score = max line logit trên mọi sliding window |
| **contradiction mask** | Bỏ qua dòng có text trùng một flaw line của bản vulnerable nhưng lại nằm trong bản fixed. Đã bỏ |
| **epoch, valid, test** | Một lượt qua hết tập train; tập chọn checkpoint; tập chấm cuối |
| **ckpt / ckpt_best** | Checkpoint; checkpoint có điểm valid tốt nhất |

### Metric

| Từ | Nghĩa |
| --- | --- |
| **Top-k** (Top-1, 5, 10) | % function vulnerable có ít nhất 1 flaw line nằm trong k dòng được model xếp cao nhất |
| **IFA** | Initial False Alarm: số dòng sạch người review phải đọc trước khi gặp flaw line đầu tiên. Càng thấp càng tốt |
| **Group A / B / C** | **A:** function ≤ 510 token (CodeBERT tokenizer), vừa một cửa sổ 512. **B:** function dài hơn, có ít nhất 1 flaw line trong 512 token đầu. **C:** function dài hơn, mọi flaw line nằm sau 512 token đầu, tức LineVul cắt mất |
| **precision@5 / recall@5** | Trong 5 dòng đầu: tỉ lệ dòng đúng là flaw line / tỉ lệ flaw line được tìm thấy |
| **hunk-tolerant (±2)** | Tính trúng nếu dòng được chọn cách flaw line tối đa 2 dòng. **Hunk** là một cụm dòng liền nhau bị đổi trong diff của bản vá |
| **pair acc** | Trên các pair (vulnerable, fixed) của cùng một function: % pair mà bản vulnerable có score cao hơn. **raw** là số gốc; **debiased** là sau khi loại ảnh hưởng của độ dài |
| **length shortcut** | Model dựa vào độ dài thay vì nội dung, ví dụ bản fixed thường dài hơn vì có thêm guard |
| **same_text** | Với các dòng có text giống hệt nhau ở bản vulnerable và bản fixed: % lần model cho dòng ở bản vulnerable điểm cao hơn. Trên 50 nghĩa là model có dùng ngữ cảnh |
| **AUPRC** | Area Under Precision-Recall Curve, đo khả năng nhận ra function vulnerable. Chance bằng tỉ lệ function vulnerable |
| **ROC-AUC** | Area Under ROC Curve. 0,5 là ngẫu nhiên; dưới 0,5 là score ngược chiều |
| **Spearman** | Tương quan thứ hạng, ở đây là giữa function score và số token |
| **P-C / P-V / P-B / P-R** | Metric pair của PrimeVul. **P-C** (Correct): đoán đúng cả hai bản. **P-V**: đoán cả hai là vulnerable. **P-B**: đoán cả hai là benign. **P-R** (Reversed): đoán ngược cả hai |
| **verdict** | Kết luận của LLM: function có lỗi hay không |
| **line-hit** | LLM đoán vulnerable và có dòng trích nằm trong ±2 dòng so với flaw line |

### Thống kê

| Từ | Nghĩa |
| --- | --- |
| **CI / CI95** | Confidence interval 95%, khoảng chứa giá trị thật với độ tin cậy 95%. Nếu CI của chênh lệch chứa 0 thì chưa kết luận được là khác nhau |
| **bootstrap theo project** | Lấy mẫu lại theo cả project (ví dụ cả linux, cả ffmpeg) để tính CI, vì các function trong cùng project không độc lập |
| **SE** | Standard error, sai số chuẩn |
| **MDE** | Minimum Detectable Effect, chênh lệch nhỏ nhất mà cỡ mẫu đủ để phát hiện |
| **McNemar** | Kiểm định so sánh hai model trên cùng các mẫu, theo cặp |
| **guardrail** | Ngưỡng phụ không được vi phạm dù metric chính tăng |
| **rule đã đăng ký (pre-registered)** | Quy tắc pass/fail chốt trước khi train, để không chọn metric sau khi đã thấy số |

### Dữ liệu

| Từ | Nghĩa |
| --- | --- |
| **SecVulEval / PrimeVul** | Hai dataset C/C++. SecVulEval có nhãn dòng và dùng để train. PrimeVul dùng để chấm chuyển miền, có pair vulnerable/fixed |
| **flaw line** | Dòng được gán là gây lỗi, lấy từ các dòng bị xóa hoặc sửa trong bản vá |
| **label noise** | Nhãn sai, ví dụ gán lỗi cho dòng thực ra không lỗi |
| **leakage / de-dup** | Test bị trùng với dữ liệu train của model; lọc trùng |
| **pv1** | Run encoder train trên PrimeVul (nhãn function) |
| **root cause** | Nguyên nhân gốc của lỗi |

### LLM và RL

| Từ | Nghĩa |
| --- | --- |
| **LLM** | Large Language Model |
| **OpenVul** | Bài arXiv 2602.14012 (Qwen3-4B). Công bố code TRL, data và checkpoint SFT, GRPO |
| **SFT** | Supervised Fine-Tuning: dạy model bắt chước câu trả lời mẫu |
| **RL** | Reinforcement Learning: dạy bằng phần thưởng (reward) |
| **GRPO** | Group Relative Policy Optimization: sinh nhiều rollout cho một prompt, thưởng rollout tốt hơn trung bình nhóm |
| **DPO** | Direct Preference Optimization: học từ cặp câu trả lời tốt/xấu |
| **rollout** | Một câu trả lời model tự sinh trong lúc RL |
| **reward / line reward** | Điểm thưởng; line reward là phần thưởng thêm khi trích đúng flaw line |
| **reward hacking** | Model tìm cách ăn điểm thưởng mà không giỏi lên, ví dụ cái gì cũng đoán vulnerable |
| **rejection sampling / RFT** | Sinh nhiều câu trả lời, giữ các câu đạt chuẩn rồi SFT trên đó (Rejection-sampling Fine-Tuning) |
| **arm A / B / C** | Các nhánh thí nghiệm. **A:** chỉ có verdict reward. **B:** thêm line reward. **C:** thêm line hint từ encoder |
| **line hint** | Top-k dòng của encoder đưa vào prompt LLM làm gợi ý |
| **probe** | Lần chạy ngắn để đo tốc độ và chi phí trước khi chạy thật |
| **LoRA / adapter** | Low-Rank Adaptation: chỉ train một phần trọng số nhỏ gắn thêm (adapter) |
| **TRL** | Thư viện Hugging Face dùng để train SFT, DPO, GRPO |
| **vLLM** | Engine sinh text nhanh cho LLM |
| **HF** | Hugging Face, nơi chứa model và dataset |
| **R1 traces** | Chuỗi suy luận do DeepSeek-R1 sinh ra, dùng làm dữ liệu SFT |

### Vận hành

| Từ | Nghĩa |
| --- | --- |
| **L4 / A100** | GPU trên Colab, giá lần lượt khoảng 1,5 và 5,3 unit/giờ |
| **unit** | Colab compute unit, đơn vị tính tiền GPU |
| **VM** | Máy ảo Colab |
| **colabctl** | Script điều khiển Colab của mình: gồm lệnh `mirror`, `watch`, `keep`, `final_pull` |
| **bf16 / fp16** | Định dạng số 16-bit dùng khi train để nhanh và tiết kiệm bộ nhớ |

## Nguồn

- `results/slide_ModernBERT-base_2048_{s0_base2, s1_base2, s0_lw}/results.json` và các file `primevul_results.json` tương
  ứng.
- `results/cmp_lw_vs_base2/bootstrap_vs_s0.json` và `bootstrap_vs_s1.json`.
