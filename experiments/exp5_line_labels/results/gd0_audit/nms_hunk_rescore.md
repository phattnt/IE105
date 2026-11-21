# Phase 0 CPU Rescore: Raw vs NMS (r=3) and Exact vs Hunk(±2)

## PrimeVul `eval_pv1_best`

| Subset | Method | n | Exact T1 / T3 / T5 / T10 | Hunk±2 T1 / T3 / T5 / T10 | Med IFA (Ex / H2) |
|---|---|---:|---|---|---|
| changed_357 | raw | 357 | 18.2 / 43.1 / 50.1 / 61.6 | 31.4 / 53.5 / 61.6 / 72.5 | 4 / 2 |
| changed_357 | nms_r3 | 357 | 18.5 / 37.8 / 46.2 / 61.1 | 31.6 / 56.3 / 68.3 / 82.1 | 6 / 2 |
| changed_357 | longest | 357 | 9.8 / 23.8 / 31.4 / 48.7 | 18.8 / 40.1 / 49.6 / 66.1 | 10 / 5 |
| changed_357 | anchor | 357 | 7.0 / 17.9 / 28.6 / 46.5 | 14.0 / 27.4 / 40.9 / 57.1 | 12 / 7 |
| add_only_after_207 | raw | 204 | 9.3 / 20.1 / 30.4 / 46.6 | 18.6 / 32.8 / 48.0 / 62.8 | 11 / 5 |
| add_only_after_207 | nms_r3 | 204 | 9.3 / 21.1 / 29.9 / 41.7 | 18.6 / 42.2 / 55.9 / 66.2 | 14 / 4 |
| add_only_after_207 | longest | 204 | 3.4 / 15.2 / 21.6 / 36.3 | 14.7 / 31.4 / 38.7 / 58.8 | 15 / 7 |
| add_only_after_207 | anchor | 204 | 3.4 / 10.3 / 19.6 / 36.3 | 8.8 / 17.6 / 29.4 / 48.0 | 17 / 10 |
| add_only_both_207 | raw | 204 | 14.7 / 27.4 / 41.2 / 55.9 | 22.6 / 39.2 / 54.4 / 69.6 | 7 / 4 |
| add_only_both_207 | nms_r3 | 204 | 14.7 / 30.4 / 41.2 / 52.0 | 22.6 / 47.1 / 65.2 / 75.5 | 8 / 3 |
| add_only_both_207 | longest | 204 | 8.8 / 22.1 / 29.4 / 47.1 | 19.6 / 36.3 / 45.6 / 63.7 | 10 / 5 |
| add_only_both_207 | anchor | 204 | 6.4 / 17.2 / 27.0 / 44.1 | 12.8 / 24.0 / 36.8 / 56.4 | 12 / 8 |
| all_anchored_after_564 | raw | 561 | 15.0 / 34.8 / 43.0 / 56.1 | 26.7 / 46.0 / 56.7 / 69.0 | 7 / 3 |
| all_anchored_after_564 | nms_r3 | 561 | 15.2 / 31.7 / 40.3 / 54.0 | 26.9 / 51.2 / 63.8 / 76.3 | 8 / 2 |
| all_anchored_after_564 | longest | 561 | 7.5 / 20.7 / 27.8 / 44.2 | 17.3 / 36.9 / 45.6 / 63.5 | 12 / 5 |
| all_anchored_after_564 | anchor | 561 | 5.7 / 15.2 / 25.3 / 42.8 | 12.1 / 23.9 / 36.7 / 53.8 | 14 / 8 |

## SecVulEval Test (1,037 functions)

| Run | Mode | Exact T1 / T5 / T10 | Hunk±2 T1 / T5 / T10 | Group C Hunk±2 T5 / T10 |
|---|---|---|---|---|
| controls | longest | 20.1 / 48.2 / 62.0 | 32.0 / 62.9 / 76.5 | 22.1 / 41.1 |
| controls | anchor | 20.4 / 44.6 / 58.4 | 31.3 / 57.2 / 70.6 | 13.0 / 22.1 |
| slide_ModernBERT-base_2048_s0_base2 | raw | 27.9 / 56.1 / 69.6 | 39.8 / 66.2 / 77.2 | 28.6 / 39.4 |
| slide_ModernBERT-base_2048_s0_base2 | nms_r3 | 28.0 / 58.7 / 72.0 | 39.9 / 74.3 / 85.2 | 36.4 / 56.7 |
| slide_ModernBERT-base_2048_s0_conv | raw | 27.5 / 53.6 / 68.8 | 37.4 / 63.8 / 77.4 | 32.9 / 48.5 |
| slide_ModernBERT-base_2048_s0_conv | nms_r3 | 27.3 / 57.0 / 70.1 | 37.1 / 73.6 / 85.3 | 43.3 / 59.7 |
| slide_ModernBERT-base_2048_s0_lw | raw | 30.3 / 58.0 / 70.8 | 41.1 / 68.2 / 79.6 | 35.5 / 47.2 |
| slide_ModernBERT-base_2048_s0_lw | nms_r3 | 30.1 / 60.1 / 72.5 | 41.0 / 74.2 / 86.5 | 39.0 / 58.4 |
| slide_ModernBERT-base_2048_s0_s13_conv | raw | 28.0 / 56.3 / 71.5 | 39.7 / 67.3 / 79.7 | 26.8 / 41.1 |
| slide_ModernBERT-base_2048_s0_s13_conv | nms_r3 | 28.1 / 59.1 / 72.7 | 39.7 / 73.2 / 87.0 | 32.9 / 57.1 |
| slide_ModernBERT-base_2048_s1_base2 | raw | 29.1 / 54.4 / 69.0 | 39.4 / 65.7 / 78.2 | 32.0 / 45.9 |
| slide_ModernBERT-base_2048_s1_base2 | nms_r3 | 29.4 / 57.5 / 70.8 | 39.6 / 74.1 / 85.0 | 38.1 / 54.5 |
| slide_unixcoder-base_1024_s0_conv | raw | 28.1 / 55.7 / 70.0 | 39.0 / 66.1 / 77.8 | 26.8 / 40.3 |
| slide_unixcoder-base_1024_s0_conv | nms_r3 | 28.2 / 59.1 / 71.3 | 39.0 / 73.9 / 84.4 | 36.8 / 48.9 |
