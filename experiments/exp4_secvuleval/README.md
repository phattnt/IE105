# Thí nghiệm 4a – SecVulEval dùng được tới đâu cho mức dòng

Phân tích trên máy CPU của Kaggle (`analyze_secvuleval.py`, chạy dạng Kaggle script, Internet bật).
Kết quả: `results/secvuleval_stats.json`, ví dụ dữ liệu: `results/secvuleval_examples.json`.

| | SecVulEval | PrimeVul (để so) |
|---|---|---|
| Hàm / hàm lỗi | 25.440 / 10.998 (43%) | test 25.911 / 695 (2,7%) |
| Project / CVE / commit | 707 / 5.659 / 6.050 | |
| Cặp lỗi/vá (`fixed_func_idx`) | **10.024 cặp**, cùng tên hàm 100%, bản vá dài hơn 74% | 564 cặp test, bản vá dài hơn ~86% |
| Token hàm lỗi: trung vị / p90 | 547 / 2.936 | ~1.040 / 5.500 |
| Hàm lỗi vừa 510 / 1.022 / 2.046 token | 48% / 70% / 85% | 28% / 50% / 70% |
| Hàm lỗi không có câu lệnh nhãn (vá chỉ thêm dòng) | 826 (7,5%) | 37% (nhãn diff tự dựng) |
| Câu lệnh lỗi nằm trong 510 token đầu | 65% | 43% |
| Nhóm A / B / C (theo cắt đầu 512) | 4.904 / 3.461 / **1.807** | 103 / 122 / 132 |

**Định dạng nhãn:** `changed_statements` = danh sách `[số dòng trong FILE, câu lệnh]`; tìm lại được 100% câu lệnh trong thân hàm
(so khớp bỏ khoảng trắng), nên đổi sang vị trí trong hàm được.

**Ngữ cảnh chỉ là tên ký hiệu, không có code.** `context` là dict: `External Function` (có ở 69% hàm lỗi, trung vị 1 tên),
`Type Execution Declaration` 46%, `Globals` 34%, `Function Argument` 32%, `Execution Environment` 5%. Trung vị ~60 token.
Muốn có thân hàm được gọi / định nghĩa kiểu thì phải lấy từ repo (`project_url`, `commit_id`, `filepath` có sẵn).

**Trùng với PrimeVul** (cùng nội dung hàm sau khi bỏ khoảng trắng): train 6.159 hàm (2.087 hàm lỗi), valid 803, test 773 (251 hàm lỗi),
tập cặp test 481/1.128 (203/564 hàm lỗi). Commit của SecVulEval phủ khoảng một nửa số dòng PrimeVul.
→ Không train một bên test bên kia khi chưa lọc trùng. Phần trùng dùng để đối chiếu nhãn diff tự dựng với nhãn của SecVulEval.

**Không bao giờ đưa `commit_message` vào input mô hình.**
