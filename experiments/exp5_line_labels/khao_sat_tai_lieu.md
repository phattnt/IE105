# Khảo sát tài liệu: phát hiện lỗ hổng C/C++ mức hàm và mức dòng (cập nhật 23/10/2026)

Tổng hợp từ ~90 bài (2021–2026), đọc bản HTML/PDF gốc. Mỗi số đều có nguồn ở cuối. Ký hiệu: **P-C/P-V/P-B/P-R** = tỉ lệ cặp lỗi/vá mà mô hình đúng cả hai / gắn cờ cả hai / bỏ cả hai / đảo ngược (bài PrimeVul). **VP-S / ròng** = P-C − P-R. **n/r** = bài không báo.

Vị trí của ta để so: ModernBERT-base 150M, học nhãn dòng SecVulEval, chấm PrimeVul. Mức dòng PrimeVul (357 hàm, chỉ dòng code): Top-1 16,8 / Top-3 32,5 / Top-5 45,1 / Top-10 59,4, MFR 26,7, ROC-AUC dòng 0,754. Mức cặp PrimeVul: P-C 3–9%, ròng tốt nhất +6,0 [+3,2; +8,9] (so-cặp-nhẹ), cặp cân độ dài +3,3 [+0,3; +6,2]; sau trừ độ dài ROC ≈ 0,5.

---

## 0. Năm kết luận chính

1. **Mức hàm từ riêng thân hàm bằng encoder nhỏ đã chạm trần toàn ngành.** Mọi encoder 125–220M được công bố trên cặp PrimeVul đạt P-C 0–6% (Ding 2024: 1,1–3,0; VulTrial 2026: 0,9–1,6; CLEAR 2026: 3,0–6,2). Ta 8,9% là cao nhất của họ này. Không có công thức (class weight, contrastive, cân dữ liệu) nào đưa encoder qua mốc ngẫu nhiên 22,7%.
2. **Những gì nhích được cặp PrimeVul đều là LLM 4B+ có suy luận, và có một trong ba thứ: ngữ cảnh ngoài hàm, bước xác minh/tranh luận, hoặc huấn luyện cách suy luận (SFT/RL).** Mốc sạch tốt nhất: VulTriage 33,8% (GPT-4o), AEGIS 28% (DeepSeek, $0,09/mẫu), VulAgentRL 37,8% (Qwen2.5-Coder-7B SFT+GRPO, split riêng). Phoenix 64,4% là rò rỉ (agent nhìn cả bản vá + CVE + commit).
3. **Mức dòng là chỗ encoder nhỏ dẫn đầu.** Mốc PrimeVul duy nhất đã công bố là ANVIL (Top-1 11,7 / Top-3 31,3 / Top-5 41,4, ROC 0,618); LineVul/LineVD/LLMAO dạy trên BigVul đều ≈ ngẫu nhiên khi sang PrimeVul (ROC 0,50). Ta hơn ANVIL ở Top-1, Top-5, ROC. Prompt LLM thuần (GPT-4o, Claude 3.7) thua encoder fine-tune ở mức dòng.
4. **Độ dài là yếu tố gây nhiễu chưa ai đo đúng cách.** Chỉ một bài (2609.32890, 2026) xem độ dài ở mức cặp: bản vá dài hơn 87,5% cặp, luật "ngắn = lỗi" đúng 0,875–0,91. Chưa bài nào trừ phần độ dài ra khỏi điểm của model rồi tính lại ROC, chưa ai báo tỉ lệ này riêng cho SecVulEval/PrimeVul. Kết quả của ta lấp đúng khoảng trống đó.
5. **Dữ liệu tự tạo chỉ giúp trên bộ bẩn.** Trên PrimeVul, hai bộ sinh mạnh nhất (VulScribeR, AVIATOR) cho LineVul/CodeT5 ≈ 0 điểm; VGX không tái lập được bởi 3 nhóm độc lập. Chưa bài nào train tuần tự tự tạo → thật rồi đo mức dòng; kết quả +2,7 Top-10 của ta là số đầu tiên loại này.

---

## 1. Hướng A: định vị dòng lỗi (line-level localization)

### 1.1 Trên PrimeVul (so sánh được với ta)

| Phương pháp | Train | Tập | Top-1 | Top-3 | Top-5 | Top-10 | MFR | ROC dòng |
|---|---|---|---|---|---|---|---|---|
| **Ta (ModernBERT, tự-tạo → thật)** | SecVulEval | 357 hàm | **16,8** | **32,5** | **45,1** | 59,4 | 26,7 | **0,754** |
| ANVIL (CodeLlama-13B, không train) | – | 273 hàm | 11,7 | 31,3 | 41,4 | n/r | 25,0 | 0,618 |
| LLMAO (CodeGen-16B + adapter) | BigVul | 273 | 9,2 | 17,3 | 26,2 | n/r | 49,3 | 0,518 |
| LineVD (GNN) | BigVul | 273 | 9,2 | 15,4 | 23,1 | n/r | 38,0 | 0,496 |
| LineVul (CodeBERT attention) | BigVul | 455 dòng | 4,0 | 17,0 | 30,0 | n/r | 17,6* | 0,496 |

\* LineVul cắt 512 token nên MFR thấp giả tạo (N-MFR 0,71, tệ nhất). ANVIL lọc 273 hàm theo project, ta dùng 357 hàm với dòng lỗi từ diff; không trùng tập hoàn toàn. Thêm: LineVul/CodeBERT/CodeT5 **train trên PrimeVul** rồi chấm dòng: F1 < 0,01, Detection Alignment ≈ 0 (2510.11202); 89% cảnh báo đúng hàm của LineVul trên PrimeVul có IoU dòng ≤ 0,5 (UntrustVul).

### 1.2 Trên Big-Vul (cùng phân phối, random split, nhiễu nhãn 25–54%; chỉ để tham khảo)

| Phương pháp | Năm | Top-1 | Top-5 | Top-10 | Stmt F1 | Ghi chú |
|---|---|---|---|---|---|---|
| LineVul | 2022 | – | – | 65 | – | tái lập bởi người khác: Top-10 49–74 |
| LineVD | 2022 | 21–61 | 45–81 | 59 | 36,0 | nhãn mở rộng theo phụ thuộc |
| COSTA | 2024 | 42,2 | 59,9 | 64,7 | 49 | |
| LEAP (không nhãn dòng) | 2022 | – | – | 80,6 | – | IFA 1,5 |
| EDAT-MLT (CodeT5 + adversarial + đa nhiệm) | 2025 | – | 68,1 | 81,1 | – | bỏ đa nhiệm: Top-10 −26 |
| LocVul (CodeT5 seq2seq) | 2026 | – | – | 82,8 | – | MRR@10 79,4 |
| Zhang 2404.00287 (Qwen2.5-Coder-7B gán nhãn dòng) | 2025 | – | – | – | 63,8 | GPT-4o one-shot 21,6; LineVul 29,7 |
| MatsVD / DCVD | 2024/26 | 93–94 | 95–98 | – | 86–88 | nghi rò rỉ; cùng VELVET được tái lập 31,8 (COSTA) vs 86,1 (DCVD) |

Ta trên SecVulEval test Top-10 71,5 nằm trong dải LineVul (65) – LocVul (83) của loại "cùng phân phối".

### 1.3 Kỹ thuật tạo được gain ở mức dòng (delta tác giả báo)

- Học nhãn dòng trực tiếp thay vì attention của classifier hàm: LineVD vs IVDetect stmt F1 17,6 → 36,0.
- Sinh/gán nhãn dòng bằng code LLM: LocVul Top-10 71,4 → 82,8; Qwen-7B gán nhãn 63,8 F1 vs prompt GPT-4o 21,6; mở rộng ngữ cảnh cửa sổ trượt +3,1.
- Đa nhiệm hàm/CWE + dòng: EDAT-MLT bỏ đa nhiệm Top-10 −0,26, IFA +6,6.
- Phụ thuộc chương trình (taint/CFG): VulTeller F1 36 → 45,8; COSTA Top-1 31,8 → 42,2.
- Codebook/contrastive: OptiMatch stmt F1 36,9 → 82,1 (Big-Vul); LEAP IFA 3,7 → 1,5.
- Không nhãn, chấm bất thường với ngữ cảnh thích nghi: ANVIL AUC 56,9 → 66,3 (Magma).
- Prompt thuần yếu: Claude 3.7 stmt F1 23,8 (SecVulEval, 300 mẫu); 30B open Top-5 ≈ 12.

---

## 2. Hướng B: mức hàm với encoder / model nhỏ

### 2.1 Trần trên cặp PrimeVul

| Nguồn | Model | Công thức | P-C | P-V | P-B |
|---|---|---|---|---|---|
| Ding 2024 | CodeBERT/UniXcoder/CodeT5/StarCoder2/CodeGen2.5 | CE, 512 tok | 1,1–3,0 | 8–13 | 84–88 |
| Ding 2024 | UniXcoder + class weight 40 | | 0,9–2,1 (F1 21 → 26) | 22,5 | |
| Ding 2024 | UniXcoder + CA-CLR (contrastive) | | 2,3 | | |
| VulTrial 2026 | CodeBERT/UniXcoder/LineVul | full hoặc chỉ cặp | 0–1,6 (chỉ cặp: 0, P-V 435/435) | | |
| CLEAR 2026 | CodeBERT / UniXcoder | cân bằng | 3,0 / 6,2 | 55 / 65 | |
| **Ta** | ModernBERT so-cặp-nhẹ | hinge cặp λ 0,3 | **8,9** | 56,4 | 31,9 |
| Ngẫu nhiên | | | 22,7 | 26,2 | 26,4 |

Accuracy trên cặp cân bằng: encoder 50–57% (ReVD baseline 49,9–51; CPRVul 53–57). Khi mẫu cân bằng **có cả hàm lỗi không ghép cặp** thì lên 73–78 balanced acc (Li 2025, Deininger 2026 "0,765", Ouchebara 2025). Khoảng cách ~20 điểm này chính là cỡ của lối tắt độ dài/hàm-không-cặp, chưa bài nào kiểm soát tường minh.

Hai ràng buộc cứng: **27% cặp PrimeVul trở thành giống hệt nhau sau khi cắt 512 token** (Li 2025, 2507.16887) → encoder 512 token có trần P-C ≈ 73%; nhãn OneFunc đúng 86%.

### 2.2 Công thức pretrain/trung gian có bằng chứng

| Công thức | Model | Delta | Bộ | Ghi chú |
|---|---|---|---|---|
| **PDBERT**: CodeBERT + MLM + dự đoán phụ thuộc điều khiển/dữ liệu (Joern) trên 1,9M hàm C/C++ | 125M | ReVeal +4,1 F1, Big-Vul +4,9, Devign +5,5 acc; **PrimeVul chia theo thời gian +5,2 BA** (78,3 vs 73,1), OOD 2024 +4,3 | random + temporal | MLM thêm một mình chỉ +0,4/+1,5/+3,4; encoder duy nhất có bằng chứng chuyển sang PrimeVul sạch |
| TRACED: UniXcoder + dấu vết thực thi | 125M | ReVeal +3,0 F1 | random | |
| DISCO / CONCORD: contrastive với bug chèn (gồm **xóa if-check nhỏ**) lúc pretrain | 125M | ReVeal +0,6 / +1,3 F1; D2A +3,5 | random | gần nhất với "rút guard" của ta, gain nhỏ |
| VulBERTa: MLM từ đầu trên 2,28M hàm C/C++ | 125M | Devign +2,7; Big-Vul tệ hơn | random | domain MLM không ổn định |
| ContraBERT: MLM + InfoNCE | 125M | Devign +2,1 / +0,5 | random | contrastive không MLM: tệ hơn gốc |
| SVulD: triplet trên UniXcoder | 125M | +0,03 F1; acc trên hàm đã vá 46 → 61% | Big-Vul lọc | dùng bản vá làm negative tường minh: tệ hơn |
| **MiST**: mid-training ~1B token văn bản bảo mật + SFT PrimeVul | Qwen3 8B/32B | P-C +13,5 / +11,2 (8B: 2,8 → 16,3; 32B: 5,2 → 16,4; CoT 21,3) | PrimeVul cặp | decoder có suy luận, vẫn < ngẫu nhiên |
| CPRVul: chưng cất suy luận GPT-4.1 + ngữ cảnh chọn lọc, LoRA | Qwen2.5-Coder 7B/32B | acc cặp 62,7 / 67,8 vs UniXcoder 56,7 | PrimeVul ≤1024 tok | **ngữ cảnh thô làm UniXcoder tụt 56,7 → 54,2** |
| Cửa sổ 512 → 1024 | UniXcoder | Big-Vul +5,9 đến +9,6 F1 | | VulGate |

Kết luận: contrastive cho encoder chỉ +0,5–3 điểm trên bộ bẩn, +0,7 P-C trên PrimeVul. Pretrain trung gian **có mục tiêu cấu trúc** (PDBERT) là thứ duy nhất chuyển được sang bộ sạch.

---

## 3. Hướng C: LLM / agent / RL trên cặp PrimeVul (435 cặp)

Cờ giao thức: **sạch** = chỉ hàm (+ ngữ cảnh repo / retrieval từ tập train); **rò rỉ** = thấy bản vá / commit / CVE / cả cặp lúc chấm.

| # | Phương pháp (năm) | Backbone | Train | P-C | P-R | Ròng | Cờ |
|---|---|---|---|---|---|---|---|
| – | Phoenix (2026, 427 cặp) | Qwen2.5-Coder-14B + Qwen3.5-9B | không | 64,4% | n/r | n/r | **rò rỉ** (agent 1–2 nhìn cả cặp + CVE + commit); bản "Blind" 19,9% |
| – | TREAT (200 cặp, cho xem cả cặp) | GPT-4.1 | không | 90,8 | 7,0 | 83,8 | **rò rỉ** |
| 1 | ReasonVul (2026) | Phi-4 + Llama-4-Scout + DeepSeek-V3 | không | 40,0 | n/r | n/r | sạch* (retrieval ReposVul chưa kiểm trùng) |
| 2 | VulAgentRL (2026, split repo riêng) | Qwen2.5-Coder-7B | SFT (lời giải Opus) + GRPO, tool CPG | 37,8 | n/r | n/r | sạch; Opus 4.7 cùng harness 27,3 |
| 3 | **VulTriage** (2026) | GPT-4o | không | **33,8** (147) | 6,7 (29) | 27,1 | sạch |
| 4 | VulAgent có ngữ cảnh Joern (304 cặp) | GPT-4o | không | 28,7 | 10,7 | 18,0 | sạch; không ngữ cảnh 17,5 |
| 5 | **AEGIS** k=2 (2026) | DeepSeek-V3.1 | không | **28,0** (122) | 14,5 (63) | 13,6 | sạch; $0,093/mẫu |
| 6 | CLEAR (gpt-5-mini) | gpt-5-mini | không (đồ thị từ train) | 27,5 | n/r | 19,6 | sạch |
| 7 | VULPO-4B (ContextVul test, 985 cặp) | Qwen3-4B | SFT + RL | 22,6 | 6,2 | 16,4 | sạch; = DeepSeek-V3.1 22,5 |
| 8 | VulTrial + FT moderator | GPT-4o | SFT 50 cặp | 22,1 (96) | 11,7 | 10,3 | sạch |
| 9 | RASM-Vul | DeepSeek-V3 | không (KB = train) | 21,4 (93) | n/r | n/r | sạch |
| 10 | MiST-32B-DPO-PrimeVul CoT | Qwen3-32B | mid-train + SFT | 21,3 | n/r | n/r | sạch |
| 11 | CLEAR (gpt-4o-mini) | gpt-4o-mini | không | 19,2 | 8,6 | 10,6 | sạch |
| 12 | VulTrial (2025) | GPT-4o | không | 18,6 (81) | 11,5 | 7,1 | sạch; $0,07/cặp |
| 13 | ReVD (2025) | Qwen2.5-Coder-7B | SFT + COPO | ~17,9 | ~2,3 | **16,1** | sạch (SFT dữ liệu từ PrimeVul) |
| 14 | VulInstruct (838 mẫu) | DeepSeek-V3 | không | 17,2 | ~9,8 | 7,4 | sạch* (retrieval NVD) |
| 15 | MAVUL (200 cặp JitVul) | GPT-4o | không | 17,5 | 33,5 | **−16** | sạch |
| 16 | GPT-4 CoT (Ding 2024) | GPT-4 | không | 12,9 | 8,3 | 4,6 | sạch |
| 17 | GPTLens / Ding CoT (GPT-4o) | GPT-4o | không | 10,1 / 9,2 | 6,4 / 6,7 | 3,7 / 2,5 | sạch |
| – | Ngẫu nhiên | | | 22,7 | 24,7 | −2 | |
| – | Encoder fine-tune | 125M–7B | SFT | 0–6 | 0–2 | ≈1 | sạch |

Bài 2609.32890 (68 model, 200 cặp): 37/68 model có điểm ròng **không phân biệt được với null** "gắn cờ độc lập"; F1 bám tỉ lệ gắn cờ (ρ 0,86) chứ không bám P-C (0,16).

### 3.1 Ablation: cái gì tạo gain

| Thành phần | Bằng chứng (delta P-C) |
|---|---|
| **Ngữ cảnh ngoài hàm** (callers/callees/CPG) | VulAgent +11,2 (17,5 → 28,7); MAVUL +8,5; AEGIS recall định vị 16 → 71; JitVul ReAct +4,1. Nhất quán nhất. |
| Có CoT (không phải CoT dài) | GPT-4 two-shot → CoT +7,8; MiST 8B direct → CoT 2,2 → 13,6; nhưng 1k → 32k token chỉ +0,02 (CI chứa 0) |
| **Xác minh / tranh luận** | AEGIS bỏ tranh luận 122 → 57; VulTrial bỏ moderator 96 → 32; ReasonVul bỏ abductive −15,4; ngược lại VulTrial nhiều vòng hơn tệ hơn (96 → 81), MAVUL P-R phình 33,5 |
| Retrieval tri thức | RASM-Vul +6,0; VulInstruct đặc tả vs CVE thô +5,3; CLEAR bỏ đồ thị 19,2 → 8,1. Cảnh báo: KB xây từ test làm Vul-RAG 0,20 → 0,51–0,73 (RRR 2609.37669) |
| SFT cách suy luận | VulAgentRL +19,8 (6,6 → 26,4); VULPO +7,5; MiST-8B 10,3 → 16,3 |
| Thiết kế thưởng RL | VulAgentRL: thưởng verdict +1,3, CWE +1,2, **trích đúng nút bằng chứng +8,9**; VULPO SFT → RL +7,65 |
| Thông tin cặp/đáp án (Phoenix) | 19,9 → 64,4: delta lớn nhất khảo sát, không phải gain phát hiện |

Trần sạch với open ≤ 14B trên 435 cặp: ≈ 18–21% P-C / ≈ 16% ròng (ReVD, MiST-8B); với RL + tool CPG trên split tương đương ≈ 35–38%.

---

## 4. Hướng D: dữ liệu tự tạo và tăng cường

### 4.1 Gain trên bộ thật (train trộn, không tuần tự)

| Phương pháp | Cách sinh | Detector | Test thật | Trước → sau |
|---|---|---|---|---|
| VulGen 2023 | mẫu + CodeT5 định vị, 963 mẫu, **chỉ sửa 1 câu lệnh** (39,6% cặp thật) | Devign | ReVeal | F1 14,5 → 19,1 (thật 963: 17,9; SARD: 16,6) |
| VGX 2024 | 604 mẫu + Transformer, 15k | Devign / LineVul | ReVeal | 16,4 → 20,0; LineVul Top-10 48,8 → 58,3 |
| VGX tái lập bởi 3 nhóm | | LineVul | BigVul | 12,4 → 10,2; 34,7 → 34,0; GraphCodeBERT 46,6 → 55,8 (biến đổi giữ ngữ nghĩa: 65,2) |
| VulScribeR 2025 | GPT-3.5 RAG chèn, 5–15k, 72–90% đúng | LineVul | ReVeal / BigVul | 8,4 → 16,1 / 12,4 → 14,2; **PrimeVul 23,1 → 22,4** |
| AVIATOR 2026 | 13 agent, Cppcheck/ESBMC, 93% đúng, 133 CWE | LineVul / CodeT5 | **PrimeVul** | **23,1 → 23,2 / 22,1 → 22,0 (≈ 0)**; BigVul +4 |
| VISION 2025 | GPT-4o-mini phản thực CWE-20 | GGNN | **test cũng là dữ liệu sinh** | 51,8 → 97,8 (không có bộ thật) |
| Mixup 2025 | nhiễu embedding | LineVul | BigVul | 34,7 → 38,1; **random oversampling 38,5 tốt nhất** |
| Biến đổi giữ ngữ nghĩa 2410.00249 | 6 luật AST, 3,2M | CodeBERT | BigVul | 59,1 → 74,6; UniXcoder/PDBERT ≈ 0 |
| Risse 2024 | 11 biến đổi | 6 PLM | Devign | train cùng loại biến đổi lấy lại 69% mất mát; **khác loại: tệ hơn** |
| VulChecker 2023 | ghép đồ thị Juliet vào project thật | GNN riêng | 19 project CVE | AUC 0,54 → 0,84 (chỉ Juliet → ghép) |

### 4.2 Tuần tự vs trộn, quên

- **Không bài nào** trong phát hiện lỗ hổng train tự tạo → thật tuần tự; tất cả trộn. Kết quả "mất 95% kỹ năng guard" của ta không có đối chứng trong tài liệu.
- Bằng chứng gần nhất (He 2022, bug Python): trộn gần như vô ích (AP 5,2 → 7,3) vì bug tự tạo lấn át; **hai pha + loss contrastive giữ tách biệt**: AP 21,0. Ngược với quan sát của ta, khác biệt là họ fine-tune với mất cân bằng nặng và contrastive.
- Quên trong miền: REPEAT mất 33 F1 trên chunk đầu sau 5 chunk; replay 1% lấy lại 1,6; cumulative ≈ window-only (Dou 2026).
- Mẹo giữ tín hiệu cặp khi fine-tune: SVEN loss chỉ trên dòng thay đổi; "paired scheduling" (bản lỗi và bản vá cùng batch, 2602.14012).
- Mục tiêu pretrain cấu trúc (PDBERT, TRACED) **sống sót** qua fine-tune (+3–5 điểm); contrastive bug chèn (DISCO/CONCORD) chỉ +1.

### 4.3 Kiểu lỗi mà dữ liệu tự tạo phủ, và tỉ lệ trong lỗi thật

- VulGen/VinJ: 18 CWE, chỉ 1 câu lệnh. VGX: 23 CWE, vẫn 1 câu lệnh. AVIATOR: 133 CWE, CWE-190 chỉ 67% thành công. VISION: CWE-20. SVEN: 9 CWE.
- Lỗi thật: "thiếu if và câu lệnh" **38,8%** (147 CVE Linux/Xen/OpenSSH, Barbosa 2019); 43,5% là một sai lầm duy nhất; 64% gói trong một hàm. 39,6% cặp sửa C là sửa 1 câu lệnh (VulGen). 41,2% bản vá chỉ **thêm** dòng (FSE22). **Chưa ai đo** tỉ lệ bản vá đúng là `if (...) return;`.

---

## 5. Lối tắt và chất lượng dữ liệu

### 5.1 Nhãn đúng bao nhiêu phần trăm

| Bộ | Croft ICSE23 (n=70) | DiverseVul (n=50) | Risse ISSTA25 (n=100) | PrimeVul (n=50) | CleanVul |
|---|---|---|---|---|---|
| Big-Vul | 54,3 | 36 (gộp) | 39 | 25,0 | |
| Devign | 80,0 | – | 47 | 24,0 | |
| D2A | 28,6 | – | – | – | |
| DiverseVul | – | 60 | 65 | 60 | |
| CVEFixes / CrossVul | – | 36 (gộp) | – | 51,7 / 47,8 | |
| VulnPatchPairs | – | – | – | 36,0 | |
| SVEN | – | – | – | 94,0 | |
| PrimeVul OneFunc / NVDCheck | – | – | – | 86,0 / 92,0 | |
| CleanVul ngưỡng 3 / 4 | – | – | – | – | 90,6 / 97,3 |

Trùng lặp: Big-Vul 12,7% test có trong train, CVEFixes 18,9%, PrimeVul 0,0%. Lớp "lành" cũng bẩn: 1,5–5,1% hàm "lành" Big-Vul/Devign thực ra có lỗ hổng tiềm ẩn (Croft 2024); Risse dựng được ngữ cảnh gây lỗi cho 82/90 hàm "an toàn".

### 5.2 Mọi thứ tài liệu nói về **độ dài**

1. **2609.32890 (2026)**: bản vá dài hơn **87,5%** cặp (1.000 cặp PrimeVul/ARVO/CVE); luật "ngắn = lỗi" 0,875–0,91; cặp cân độ dài (<5%): độ dài 0,5, tf-idf 0,64, probe activation LLM 0,78; cặp bản vá xóa code: probe về 0,55. Không trừ độ dài khỏi điểm model, không thử encoder.
2. SecVulEval App. C: LLM bắt được hàm lỗi median 14–33 dòng, bỏ sót 64–71 dòng (p ≤ 0,017).
3. SoK 2512.18456: trên MegaVul, recall LLM **tăng** theo bin LOC (p 10⁻⁸). Ngược chiều với 2.
4. LOVA: Top-1 định vị < 5% khi > 300 LOC.
5. Li & Paxson CCS17: 78% commit bảo mật **không xóa dòng nào**, median 7 dòng → cơ chế "vá = thêm guard = dài hơn".
6. Steenhoek ICSE23: hệ số feature `length` ≈ 0 trong hồi quy độ khó (không thử lối tắt).
7. Risse ISSTA25 (bag-of-words): F1 0,86 Big-Vul (SOTA 0,96), 0,62 Devign, 0,11 DiverseVul; **không** nhắc độ dài.
8. CausalVul: chỉ tên biến / tên API (F1 tụt 11–41%); không độ dài.

**Chưa ai**: báo tỉ lệ bản vá dài hơn riêng cho SecVulEval/PrimeVul; tính ROC luật độ dài trên toàn tập; trừ phần độ dài khỏi điểm model rồi tính lại ROC; thử encoder/ensemble/loss khác dưới kiểm soát độ dài; train cân độ dài; đưa "cân bằng độ dài" vào khung chất lượng dữ liệu (Croft 5 thuộc tính, Guo 6 vấn đề, PrimeVul 4 vấn đề đều thiếu).

### 5.3 Giao thức đánh giá chuẩn (hợp nhất)

- Chia theo thời gian, cả commit một phía; bỏ trùng hash chuẩn hóa khoảng trắng xuyên split; báo tỉ lệ trùng.
- Kiểm tay ≥ 50–100 nhãn lỗi; coi lớp lành là "không chắc".
- Cặp: P-C/P-V/P-B/P-R, ròng, null gắn cờ độc lập (P-C = p(1−p)), **cặp cân độ dài** và cặp bản vá xóa code riêng.
- Đối chứng: ngẫu nhiên, đếm từ, tf-idf, **chỉ độ dài**, null.
- Không F1 trên tập lệch; VD-S (FNR tại 0,5% FPR); balanced accuracy trên cặp; nhiều seed (34,9% dự đoán đổi theo seed); bootstrap, chỉ so khi CI hiệu không chứa 0 (0,03 trên 200 cặp là nhiễu).
- Mức dòng: chỉ dòng code; nhãn = dòng bản vá xóa/sửa; Top-k + MFR + N-MFR + kỳ vọng ngẫu nhiên; chia nhóm theo hàm bị cắt.

---

## 6. Ý nghĩa cho đề tài

### 6.1 Vị trí hiện tại

| Mục tiêu | Ta | Mốc tài liệu | Nhận định |
|---|---|---|---|
| Dòng, PrimeVul | Top-5 45,1, ROC 0,754 | ANVIL 41,4 / 0,618; LineVul chuyển 30,0 / 0,496 | **Dẫn đầu** trong mốc đã công bố; cần kiểm trùng commit SecVulEval ↔ PrimeVul trước khi khẳng định |
| Dòng, SecVulEval | Top-10 71,5 | chưa ai báo Top-k; LLM stmt F1 23,8 | Số đầu tiên loại này |
| Hàm/cặp, encoder | P-C 8,9, ròng +6,0 | encoder 0–6 | Trần của họ; có tín hiệu nhỏ thật |
| Hàm/cặp, mọi phương pháp sạch | | VulTriage 33,8, AEGIS 28, VulAgentRL 37,8 | Cần đổi họ model + ngữ cảnh |
| Phân tích độ dài | 82–87%, ROC sau trừ ≈ 0,5 cho mọi model | chỉ 1 bài, không thử encoder | **Khoảng trống thật** |

### 6.2 Ba khoảng trống ta có thể lấp mà chưa ai làm

1. **Phân tích độ dài có kiểm soát trên encoder**: tỉ lệ bản vá dài hơn riêng từng bộ, ROC luật độ dài, trừ độ dài khỏi điểm model, ensemble/loss/aggregation dưới kiểm soát độ dài, tương tác với độ sạch nhãn. Đã có số liệu, chỉ cần viết.
2. **Pretrain phản thực tuần tự → fine-tune thật, đo ở mức dòng và đo mức giữ kỹ năng**: chưa ai làm; +2,7 Top-10 và "mất 95%" là số mới. Có thể bổ sung một run trộn (paired scheduling / loss trên dòng đổi) để đối chiếu với He 2022.
3. **Encoder định vị dòng + LLM xác minh**: AEGIS dùng LLM làm bước tìm manh mối (recall 16% khi k=1); ta có bước này tốt hơn (Top-5 45%). Chưa ai ghép encoder định vị với verifier LLM trên PrimeVul.

### 6.3 Những gì tài liệu khuyên **không** làm tiếp

- Thêm run encoder mức hàm (loss, cân dữ liệu, contrastive): trần 0–9%.
- Trộn ngữ cảnh thô vào encoder: CPRVul cho thấy tụt.
- Dùng trường ngữ cảnh SecVulEval làm đầu vào: rút ra khi đã thấy bản vá/CVE, giống hệt trong cặp (3000/3000), có/không ngữ cảnh tiết lộ nhãn (97% vs 67%).
- So với Phoenix/TREAT/SecCodePRM (93,66, chưa giải thích được).

### 6.4 Nếu còn ngân sách, hướng có căn cứ nhất

| Hướng | Căn cứ | Chi phí | Kỳ vọng |
|---|---|---|---|
| Pretrain trung gian kiểu PDBERT (dự đoán phụ thuộc) cho ModernBERT, rồi fine-tune dòng | PDBERT +5 BA trên PrimeVul thời gian, sống qua fine-tune | ~72 giờ 4×3090 với 1,9M hàm (giảm quy mô được) | mức dòng +; mức hàm vẫn trần |
| Encoder của ta làm "clue" + LLM mở 7–14B xác minh có/không (AEGIS-lite) | AEGIS 122 cặp; bỏ tranh luận còn 57 | vài phiên L4, 0 API | 10–20% P-C, điểm mới |
| Chưng cất suy luận (thầy thấy diff lúc train, trò chỉ thấy hàm) + RL thưởng bằng chứng | VulAgentRL SFT 6,6 → 26,4, grounding +8,9 | API thầy + ~10 giờ L4 | 20–30% P-C |
| Probe tuyến tính trên activation LLM đông cứng, cặp cân độ dài | 2609.32890: 0,78 | 1 phiên T4 | biết LLM "biết" gì hơn encoder |

---

## 7. Nguồn chính

Định vị dòng: LineVul (MSR22) · LineVD 2203.05181 · VELVET 2112.10893 · IVDetect 2106.10478 · VulChecker USENIX23 · SliceLocator 2401.02737 · ANVIL 2408.16028 · LLMAO 2310.01726 · VulTeller ASE23 · 2510.11202 · EDAT-MLT 2506.23534 · DCVD 2605.11015 · LocVul IST 2026 · OptiMatch 2306.06109 · T2L 2510.02389 · Zhang 2404.00287 · LOVA 2410.15288 · WAVES 2509.11312 · COSTA IST24 · LEAP 2209.10414 · UntrustVul 2503.14852.

Encoder mức hàm: PrimeVul 2403.18624 · DiverseVul 2304.00409 · MiST 2609.18496 · VulnScout-C 2603.28309 · CPRVul 2602.06751 · VulGate 2508.16625 · From Lab to Reality 2512.10485 · Open or Frontier MDPI 15(9):623 · Revisiting PLMs 2507.16887 · PDBERT 2402.00657 · TRACED 2306.07487 · DeepDFA 2212.08108 · ContraBERT 2301.09072 · SVulD 2308.11237 · VulBERTa 2205.12424 · VulStyle 2604.26313 · FusionVul 2606.08553.

LLM/agent/RL: VulTrial 2505.10961 · Vul-RAG 2406.11147 · Revisiting Vul-RAG 2606.04739 · RRR 2609.37669 · VulInstruct 2511.04014 · MAVUL 2510.00317 · CLEAR 2608.03134 · AEGIS 2603.20637 · VulTriage 2605.09461 · VulAgent 2509.11523 · VULPO 2511.11896 · VulAgentRL 2607.26656 · ReVD ACL25 · RASM-Vul Electronics 15(3):612 · Phoenix 2604.19012 · ReasonVul 2605.18153 · Protocol effects 2609.32890 · TREAT 2510.17163 · CORRECT 2504.13474 · GRPO-VD 2507.03051 · JitVul 2503.03586 · SOTA2 leaderboard.

Dữ liệu tự tạo: VulGen ICSE23 · VGX 2310.15436 · VinJ FSE24 · FSE22 neural editing · VulScribeR 2408.04125 · AVIATOR 2508.20866 · VISION 2508.18933 · Mixup 2504.15632 · SPT 2410.00249 · Risse 2306.17193 · DISCO 2110.03868 · CONCORD 2306.03234 · He ICML22 2204.10049 · REPEAT 2302.03482 · SFT→RL 2602.14012 · Barbosa 1912.01948 · GraftyVul 2608.27928.

Chất lượng dữ liệu: Croft ICSE23 2301.05456 · Steenhoek ICSE23 2212.08109 · CausalVul 2310.07958 · Top Score 2408.12986 · ReVeal 2009.07235 · CleanVul 2411.17274 · Real-Vul 2407.03093 · Sejfia 2403.03024 · Croft latent 2401.11105 · SecLLMHolmes 2312.12575 · SVEN 2302.05319 · SecVulEval 2505.19828 · SoK 2512.18456 · Li & Paxson CCS17.
