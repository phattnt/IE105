# Plan v3 Tier 2 — Paired Anti-Shortcut QLoRA Report

- **Model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (4-bit NF4 QLoRA `r=16`, `alpha=16`, layers `13+27`, `max_len=2048`)
- **Telemetry:** trainable params `40409607`, peak VRAM `12.49 GB`, train time `1013.9s`, OOM fallbacks `0`
- **Clean Splits & Anti-Shortcut Sampling:** train `200` pairs (`cf>cv=182`, `cv>cf=13`), extra benign `200` (`matched=170`, `random=30`, median chars pair=`2472.0` vs benign=`1864.0`), valid `100` pairs + `200` benign

## 1. Training & Validation Curve

| Epoch | Pairs Done | Train Loss | Valid Bal Rank | Valid `PatchRemoves` | Valid Debias ROC | Valid Net@med | Composite |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 200 | 2.3180 | 56.0% | 46.1% | 0.5411 | +1.0% | 0.5504 |
