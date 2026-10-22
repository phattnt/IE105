# Nhãn dòng thừa trong SecVulEval – thống kê (chưa lọc gì)

Sinh bởi `label_noise.py`. "Dòng rác" = sau khi bỏ chú thích chỉ còn khoảng trắng / `{` `}` / `( ) [ ] ; ,` / `else` trần. Độ dài = số ký tự của dòng sau `strip()`. Dòng nhãn 0 = các dòng còn lại của chính những hàm lỗi có nhãn dòng.

| | train | valid | test |
|---|---|---|---|
| Hàm lỗi có nhãn dòng | 8,054 | 1,081 | 1,037 |
| Dòng nhãn 1 | 42,201 | 8,829 | 7,245 |
| **Dòng nhãn 1 là rác** | **5,044 (11.95%)** | **792 (8.97%)** | **1,161 (16.02%)** |
| &nbsp;&nbsp;`{` / `}` (kể cả `};` `})`) | 2,040 (4.83%) | 274 (3.1%) | 522 (7.2%) |
| &nbsp;&nbsp;trống | 884 (2.09%) | 175 (1.98%) | 225 (3.11%) |
| &nbsp;&nbsp;chỉ chú thích | 1,758 (4.17%) | 273 (3.09%) | 334 (4.61%) |
| &nbsp;&nbsp;chỉ dấu câu | 24 (0.06%) | 17 (0.19%) | 2 (0.03%) |
| &nbsp;&nbsp;`else` trần (`} else {`) | 338 (0.8%) | 53 (0.6%) | 78 (1.08%) |
| &nbsp;&nbsp;*không lọc:* `#else/#endif` | 327 (0.77%) | 35 (0.4%) | 61 (0.84%) |
| &nbsp;&nbsp;*không lọc:* `break/continue` | 285 (0.68%) | 46 (0.52%) | 35 (0.48%) |
| &nbsp;&nbsp;*không lọc:* `goto/label` | 662 (1.57%) | 31 (0.35%) | 112 (1.55%) |
| &nbsp;&nbsp;*không lọc:* `return;` | 125 (0.3%) | 29 (0.33%) | 10 (0.14%) |
| Mật độ nhãn: trung bình / trung vị / p90 | 0.1495 / 0.0638 / 0.4286 | 0.1365 / 0.06 / 0.3571 | 0.1143 / 0.0435 / 0.3043 |
| Hàm mật độ > 0,5 (số dòng nhãn 1 trong đó) | 600 (14,696) | 66 (4,303) | 50 (3,574) |
| Hàm lỗi còn 0 dòng nhãn 1 sau khi bỏ dòng rác | 31 | 2 | 9 |
| Hàm lỗi vốn đã 0 dòng nhãn (bản vá chỉ thêm) | 790 | 12 | 24 |
| Độ dài dòng nhãn 1 / nhãn 0 / nhãn 0 không trống, TB (trung vị) – trước lọc | 34.86 (32) / 24.83 (20.0) / 28.2 (25.0) | 39.67 (36) / 23.7 (18.0) / 26.47 (22) | 33.54 (30) / 25.12 (20.0) / 27.84 (23) |
| Độ dài dòng nhãn 1 / nhãn 0 / nhãn 0 không trống, TB (trung vị) – sau bỏ rác | 37.65 (34) / 24.74 (20.0) / 28.12 (25.0) | 42.19 (38) / 23.62 (18.0) / 26.41 (22.0) | 37.13 (33.0) / 25.04 (20) / 27.77 (23) |
| Độ dài dòng nhãn 1 / nhãn 0 / nhãn 0 không trống, TB (trung vị) – sau bỏ rác + mật độ ≤ 0,5 | 39.39 (37) / 24.84 (20.0) / 28.19 (25) | 38.87 (35.0) / 23.68 (18.0) / 26.43 (22.0) | 40.18 (35) / 25.13 (20.0) / 27.84 (23.0) |

Phân bố mật độ (số hàm lỗi có nhãn dòng theo khoảng dòng nhãn 1 / số dòng):

| khoảng | train | valid | test |
|---|---|---|---|
| 0.00-0.05 | 3404 | 484 | 555 |
| 0.05-0.10 | 1467 | 196 | 162 |
| 0.10-0.20 | 1316 | 177 | 136 |
| 0.20-0.30 | 634 | 88 | 73 |
| 0.30-0.50 | 510 | 64 | 54 |
| 0.50-0.75 | 469 | 38 | 39 |
| 0.75-1.00 | 254 | 34 | 18 |

10 hàm nặng nhất – train:

| idx | project | mật độ | dòng nhãn 1 / số dòng |
|---|---|---|---|
| 18071 | onnx | 1.0 | 983 / 983 |
| 3510 | krb5 | 1.0 | 835 / 835 |
| 7701 | file | 1.0 | 386 / 386 |
| 6189 | linux | 1.0 | 361 / 361 |
| 12241 | keepalived | 1.0 | 343 / 343 |
| 6860 | libarchive | 1.0 | 257 / 257 |
| 18298 | sudo | 1.0 | 225 / 225 |
| 6156 | linux | 1.0 | 195 / 195 |
| 15386 | frr | 1.0 | 195 / 195 |
| 10744 | libvncserver | 1.0 | 142 / 142 |

10 hàm nặng nhất – valid:

| idx | project | mật độ | dòng nhãn 1 / số dòng |
|---|---|---|---|
| 10163 | wireshark | 1.0 | 106 / 106 |
| 15016 | acrn-hypervisor | 1.0 | 95 / 95 |
| 15154 | typed_ast | 1.0 | 69 / 69 |
| 9150 | opa-fm | 1.0 | 57 / 57 |
| 12072 | libxkbcommon | 1.0 | 57 / 57 |
| 17545 | sgxwallet | 1.0 | 39 / 39 |
| 15013 | acrn-hypervisor | 1.0 | 37 / 37 |
| 10156 | wireshark | 1.0 | 36 / 36 |
| 15019 | acrn-hypervisor | 1.0 | 27 / 27 |
| 15029 | acrn-hypervisor | 1.0 | 24 / 24 |

10 hàm nặng nhất – test:

| idx | project | mật độ | dòng nhãn 1 / số dòng |
|---|---|---|---|
| 16033 | hermes | 1.0 | 2753 / 2753 |
| 10007 | vim | 1.0 | 46 / 46 |
| 15638 | contiki-ng | 1.0 | 38 / 38 |
| 15640 | contiki-ng | 1.0 | 6 / 6 |
| 15643 | contiki-ng | 1.0 | 6 / 6 |
| 9887 | vim | 1.0 | 5 / 5 |
| 8151 | bootstrap-dht | 1.0 | 1 / 1 |
| 8152 | bootstrap-dht | 1.0 | 1 / 1 |
| 9759 | vim | 0.9302 | 160 / 172 |
| 11582 | cantata | 0.8571 | 6 / 7 |
