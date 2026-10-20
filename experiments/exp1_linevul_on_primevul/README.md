# Thí nghiệm 1 – LineVul (checkpoint gốc) trên PrimeVul

Chạy checkpoint LineVul đã train trên Big-Vul (`12heads_linevul_model.bin`) trên PrimeVul,
**không train lại**, để trả lời:

1. Bao nhiêu phần PrimeVul nằm ngoài cửa sổ 512 token của LineVul?
2. LineVul phát hiện được gì ở mức hàm trên dữ liệu thực tế (toàn bộ tập test và tập cặp)?
3. Định vị dòng bằng attention còn đúng được không, và có khác nhau giữa hàm bị cắt / không bị cắt?

Mô hình **không** được xem bản đã vá, mô tả CVE hay commit message. Bản đã vá chỉ dùng
để dựng nhãn dòng (diff) phục vụ chấm điểm.

## File

| File | Chạy ở đâu | Làm gì |
|---|---|---|
| `setup_vm.py` | máy Colab | tải PrimeVul (bản gốc, Google Drive) và checkpoint LineVul, in cấu trúc dữ liệu |
| `eval_linevul_primevul.py` | máy Colab | toàn bộ đánh giá, ghi `results.json` và các file dự đoán vào `/content/exp1/results/` |
| `results/` | repo | kết quả tải về |

## Cách chạy (Colab CLI)

```bash
pip install google-colab-cli
colab new -s exp1 --gpu T4
colab exec -s exp1 -f setup_vm.py
colab exec -s exp1 -f eval_linevul_primevul.py
colab download -s exp1 /content/exp1/results/results.json results/results.json
colab stop -s exp1
```

## Ghi chú về cách đo

- **Tập cặp test của bản gốc có 564 cặp** (khớp bảng V của paper PrimeVul). Con số 435 cặp
  trong VulTrial/VulTriage là bản họ phát hành lại, nên không so trực tiếp số tuyệt đối.
- **VD-S** tính theo script gốc của PrimeVul: ngưỡng chọn ngay trên tập test (oracle), nên lạc quan.
- **Nhãn dòng** = các dòng của hàm lỗi bị xóa hoặc thay trong bản vá (quy ước Big-Vul), bỏ dòng
  trống và dòng chỉ có comment. Bản vá chỉ thêm dòng thì hàm không có nhãn dòng; tỉ lệ này được báo riêng.
- **Điểm dòng** tái hiện đúng phương pháp "attention" của `linevul_main.py`: attention lớp đầu,
  cộng qua các head, lấy lượng attention mỗi token nhận được, bỏ token đặc biệt, cộng theo dòng.
  Khác một chỗ: token được ánh xạ về dòng bằng vị trí ký tự (offset) thay vì tách chuỗi token,
  để tab không bị coi là xuống dòng như trong code gốc.
- Dòng nằm ngoài 512 token nhận điểm 0 và xếp sau mọi dòng nhìn thấy.
- Kết quả định vị chia theo nhóm: A hàm vừa 512 token · B bị cắt nhưng còn thấy dòng lỗi ·
  C bị cắt và mất hết dòng lỗi. Có baseline ngẫu nhiên cho Top-k.
