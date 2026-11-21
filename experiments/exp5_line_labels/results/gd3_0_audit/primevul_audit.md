# Plan v3 Phase 3.0 — PrimeVul Data & Leakage Audit

## 1. Split Counts & Length Skew

| Split | Clean Pairs | Changed / AddOnly | `cf > cv` (fix longer) | `cv > cf` (vul longer) | Shorter=Vul % | LenMatched<5% (n) | LenMatched Shorter=Vul % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train_clean` | 4347 | 2923 / 1424 | 3574 | 637 | 83.78% | 2596 (2073 vs 387) | **82.47%** |
| `valid_clean` | 562 | 372 / 190 | 474 | 67 | 86.21% | 315 (251 vs 43) | **83.02%** |
| `test` | 564 | 357 / 207 | 482 | 65 | 86.97% | 337 (274 vs 46) | **83.83%** |

## 2. Token Length Distribution (`Qwen/Qwen2.5-Coder-7B-Instruct`)

| Split | Functions | Median tok | p90 | p95 | Max | >2048 tok % | >3072 tok % | >4096 tok % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train_clean` | 8694 | 695 | 3286 | 5406 | 219136 | 17.37% | 10.88% | 7.35% |
| `valid_clean` | 1124 | 655 | 2451 | 4483 | 23413 | 15.04% | 7.92% | 6.14% |
| `test` | 1128 | 724 | 3325 | 6438 | 30202 | 18.26% | 10.82% | 8.33% |

## 3. Leakage & Project Overlap
- Raw overlap removed: train `{'raw': 4354, 'clean': 4347, 'dropped_commit': 6, 'dropped_hash': 1}`, valid `{'raw': 562, 'clean': 562, 'dropped_commit': 0, 'dropped_hash': 0}`
- Extra train benign pool (clean): `173561` (sampled `2000`)
- Projects: `592` train, `200` test, **`119` shared** -> Test `seen_projects = 455`, `unseen_projects = 109`

## 4. Full-Test Manifest Integrity
- `{"n_full_jsonl": 25911, "n_full_vulnerable_jsonl": 695, "manifest_P_present": 1126, "manifest_V_present": 131, "manifest_B_present": 2000, "strata": {"P": {"n": 1126, "sampled": 1126, "weight": 1.0, "note": "test pairs, scored in the pair run"}, "V": {"n": 131, "sampled": 131, "weight": 1.0}, "B": {"n": 24654, "sampled": 2000, "weight": 12.327}}}`
