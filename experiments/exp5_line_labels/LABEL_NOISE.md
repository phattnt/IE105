# Lọc nhãn dòng thừa (train/valid) – run `_conv_clean`

Nhãn dòng SecVulEval suy từ bản vá nên có nhãn thừa: `}`, dòng trống, chỉ chú thích, `;`, `} else {`, và hàm bị viết lại cả
hàm (mọi dòng nhãn 1). Câu hỏi: gỡ các nhãn đó khỏi train/valid có giúp tìm dòng không? Test **không đổi**.

## Thống kê (chưa lọc) – `label_noise.py`, chi tiết `results/label_noise_stats.md` / `.json`

| | train | valid | test |
|---|---|---|---|
| Dòng nhãn 1 | 42.201 | 8.829 | 7.245 |
| Dòng nhãn 1 là rác | 5.044 (12,0%) | 792 (9,0%) | 1.161 (16,0%) |
| `{`/`}` · chú thích · trống · `else` · dấu câu (%) | 4,8 · 4,2 · 2,1 · 0,8 · 0,06 | 3,1 · 3,1 · 2,0 · 0,6 · 0,19 | 7,2 · 4,6 · 3,1 · 1,1 · 0,03 |
| Hàm mật độ nhãn > 0,5 (dòng nhãn 1 trong đó) | 600 (14.696 = 35%) | 66 (4.303) | 50 (3.574) |
| Hàm lỗi còn 0 nhãn sau bỏ rác | 31 | 2 | 9 |
| Độ dài dòng nhãn 1 / 0, TB ký tự: trước → sau bỏ rác | 34,9 / 24,8 → 37,7 / 24,7 | 39,7 / 23,7 → 42,2 / 23,6 | 33,5 / 25,1 → 37,1 / 25,0 |

Lọc làm dòng nhãn 1 **dài thêm** ~3 ký tự → rủi ro lối tắt độ dài mạnh lên; vì vậy có luật kết luận bên dưới.

## Bộ lọc – `prepare_secvuleval.py` (mặc định tắt, chỉ train + valid)

- `--drop_trivial_flaw`: dòng rác (sau khi bỏ chú thích chỉ còn khoảng trắng, `{ }`, `( ) [ ] ; ,`, hoặc `else` trần) về nhãn 0,
  dòng vẫn ở trong hàm. `break;`, `return;`, `goto`, nhãn, `#endif` **không** lọc (có thể là lỗi thật). Unit test: `test_label_noise.py`.
- `--max_flaw_density 0.5`: bỏ hàm lỗi có > 50% số dòng mang nhãn 1 (đo trên nhãn gốc) cùng hàm cặp đã vá.
- Hàm lỗi bị gỡ hết nhãn → `flaw_lines = []`, xử lý y như hàm "bản vá chỉ thêm dòng": run plain bỏ hàm lỗi đó khỏi train
  (hàm cặp đã vá vẫn giữ); valid giữ cho chỉ số hàm/cặp, bỏ khỏi Top-k.
- Bản lọc (`--drop_trivial_flaw --max_flaw_density 0.5`): train bỏ 936 hàm (600 hàm lỗi dày + cặp), gỡ 1.508 nhãn rác ở 774 hàm,
  31 hàm lỗi hết nhãn; valid bỏ 113 hàm, gỡ 298 nhãn, 2 hàm hết nhãn.
- `test.jsonl` giống hệt từng byte bản không lọc (sha256 `e8adb714…0670`); bản mặc định cũng giống hệt cả 3 file sinh bằng code cũ.
- Ví dụ ngẫu nhiên (seed 0) cho từng luật: `results/label_filter_examples.md`.

## Train (Colab L4, ~2 giờ ≈ 3 unit) – chưa chạy

Entry `_conv_clean` trong `colab_exp5.py`: y hệt run gốc `slide_ModernBERT-base_2048_s0_conv` (`colab/run_inner_conv.py`:
`--lr 3e-5`, 4 epoch, patience 2, seed 0, chọn epoch mặc định `top5_same_text`), chỉ khác `--data`. Không dùng
`--select top5_pair_debias` nên **không cần chạy lại run gốc**. Lưu ý: thước chọn epoch tính trên valid **đã lọc** (bỏ 66 hàm dày như hermes)
nên nó ít bị một hàm chi phối hơn run gốc; đây là một phần của can thiệp.

```bash
# push nhánh, trên Colab dùng nhánh này (colab_exp5.py pull `main`): hoặc merge trước, hoặc trên VM: git -C /content/IE105 checkout label-noise-filter
colab new -s exp5 --gpu L4
colab upload -s exp5 experiments/exp5_line_labels/colab_exp5.py /content/colab_exp5.py
echo 'import subprocess, sys; subprocess.Popen([sys.executable, "/content/colab_exp5.py", "--only", "_conv_clean"], stdout=open("/content/exp5_launch.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)' | colab exec -s exp5
```

Lệnh train mà hàng đợi sẽ chạy (`python colab_exp5.py --dry --only _conv_clean`):

```bash
python train_lines.py --strategy slide --train_tokens 8192 --bf16 --ckpt_every 400 --model answerdotai/ModernBERT-base --max_len 2048 \
  --lam_line 0 --lam_fn 0 --epochs 4 --patience 2 --data /content/exp5/secvul_lines_clean --lr 3e-5 --seed 0 --tag _conv_clean \
  --out /content/exp5/slide_ModernBERT-base_2048_s0_conv_clean
python eval_primevul.py --run /content/exp5/slide_ModernBERT-base_2048_s0_conv_clean --primevul /content/exp5/primevul
```

## Đánh giá (CPU, sau khi tải kết quả về `results/`)

```bash
R=slide_ModernBERT-base_2048_s0_conv_clean
for f in results.json test_scores.jsonl primevul_results.json primevul_test_scores.jsonl; do colab download -s exp5 /content/exp5/$R/$f results/$R/$f; done
colab download -s exp5 /content/exp5/logs/$R.log results/$R/train.log
B=results/slide_ModernBERT-base_2048_s0_conv; N=results/$R
python bootstrap_ci.py --data ../../data/secvul_lines --runs $B $N --out ../../data/bootstrap_conv_vs_clean.json
python debias_length.py --data ../../data/secvul_lines --primevul ../../data/primevul --runs $B $N --out ../../data/debias_conv_clean.json
python compare_clean.py --data ../../data/secvul_lines --base $B --new $N --bootstrap ../../data/bootstrap_conv_vs_clean.json \
  --debias ../../data/debias_conv_clean.json --out results/compare_conv_clean.md
```

`compare_clean.py` gom: Top-1/5/10, IFA, nhóm A/B/C của hai run cạnh các đối chứng (`results/controls.json`: dòng dài nhất trước /
độ dài, mỏ neo, ngẫu nhiên – test không đổi nên giữ nguyên), bootstrap theo project, Spearman điểm dòng – độ dài dòng trong từng hàm
(kèm CI theo project), cặp/ROC sau khi trừ độ dài, PrimeVul. Đối chứng độ dài trên PrimeVul: chưa đo.

**Luật kết luận (đặt trước khi chạy):** chỉ viết "nhãn sạch giúp" khi CI Top-10 (clean − gốc) nằm hẳn trên 0 **và** chênh tương quan độ dài
(hàm lỗi có nhãn dòng, ước lượng điểm) ≤ 0. Ngược lại: "không có bằng chứng nhiễu nhãn thừa là nút thắt"; nếu Top-k tăng mà tương quan cũng
tăng thì ghi rõ có thể do lối tắt độ dài. Đã chạy thử toàn bộ chuỗi với `_s13_conv` làm run giả (khớp số bootstrap trong README).
