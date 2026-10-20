# Thí nghiệm 3 (pilot) – Train lại LineVul trên PrimeVul theo từng chiến lược ngữ cảnh

TN1/TN2 cho thấy checkpoint LineVul gốc (train trên Big-Vul) không chuyển sang PrimeVul được, nên muốn so sánh
các cách đưa hàm dài vào mô hình thì **phải train trên PrimeVul**. Bản đầy đủ (184k hàm × 10 epoch) tốn ~45 giờ T4;
pilot này dùng tập con để trả lời trước câu hỏi: *train cùng chiến lược thì hành vi mức hàm có đổi không?*

## Thiết kế

| | |
|---|---|
| Train | toàn bộ 5.574 hàm lỗi của `primevul_train` + 5× hàm lành lấy ngẫu nhiên (seed 0) → 33.444 hàm |
| Valid | cùng công thức trên `primevul_valid` (4.194 hàm) – chọn epoch (AUPRC) và ngưỡng; **không đụng test** |
| Test | toàn bộ `primevul_test` (25.911) + 564 cặp, cùng giao thức TN1/TN2 |
| Mô hình | CodeBERT-base + đầu LineVul, AdamW 2e-5, batch 16, 3 epoch, fp16, 1 seed |

Chiến lược train (mỗi mô hình một chiến lược, cùng tập con, cùng seed):

| | Mẫu train | Suy luận |
|---|---|---|
| `head` | 510 token đầu (LineVul chuẩn) | như TN1 |
| `slide` | mỗi cửa sổ 510/256 là một mẫu, **kế thừa nhãn hàm** (LLM4AVL), tối đa 4 cửa sổ ngẫu nhiên/hàm/epoch | max qua mọi cửa sổ |
| `slice` | slice theo mỏ neo của TN2 | như TN2 |
| `length` | đối chứng: hồi quy logistic trên [log token, log dòng] – không nhìn code | – |

Báo cáo: chỉ số cặp (ở ngưỡng 0,5 và ngưỡng chọn trên valid), định vị dòng A/B/C, toàn tập test
(F1@0,5, F1 ở ngưỡng valid, AUPRC, ROC-AUC, VD-S với ngưỡng valid *và* ngưỡng oracle).

## Giới hạn của pilot
- Tỉ lệ lớp train 1:5 khác test 1:36 → F1 tuyệt đối không so với paper được; so **giữa các chiến lược** và với `length`.
- 1 seed; chênh lệch nhỏ (< vài điểm) chưa kết luận được.
- Nhãn cửa sổ kế thừa từ hàm là nhãn nhiễu (cửa sổ không chứa dòng lỗi vẫn gắn "lỗi").

## Chạy
```bash
colab new -s exp3 --gpu T4
colab exec -s exp3 -f ../exp1_linevul_on_primevul/setup_vm.py        # dữ liệu
# tạo /content/exp1/code rồi upload eval_linevul_primevul.py, eval_context_strategies.py, train_pilot.py, run_vm.sh vào đó
# chạy tách khỏi CLI (kết nối CLI rớt sau ~20 phút): subprocess.Popen(["bash","/content/exp1/code/run_vm.sh"], start_new_session=True)
# theo dõi: colab download -s exp3 /content/exp3/logs/<strategy>.log ...
# kết quả: /content/exp3/results/<strategy>/{results.json,localization.jsonl,full_test.jsonl,pair_predictions.jsonl,valid_curve.json}
```
`run_vm.sh` bỏ qua chiến lược đã có `results.json`, nên chạy lại được sau khi VM bị reset.

## Backbone trên Kaggle
Colab free thu hồi máy sau 30–40 phút, nên hai backbone (`neulab/codebert-c`, `microsoft/unixcoder-base`, chiến lược `head`)
chạy trên Kaggle. Notebook mới → Accelerator **GPU T4** (chọn T4 ×2 nếu có), Internet **On**, một cell:
```
!git clone -q https://github.com/phattnt/IE105.git
!python IE105/experiments/exp3_pilot_train/kaggle_backbones.py
```
Bấm **Save Version → Save & Run All** để chạy nền (không cần mở trình duyệt). Có 2 GPU thì hai backbone chạy song song.
Kết quả: tab Output của notebook, `exp3/results/<run>/` (tải về bỏ vào `results/` rồi chạy `python experiments/collect_results.py`).
