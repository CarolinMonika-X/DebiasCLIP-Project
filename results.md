# DebiasCLIP Project - Final Results

## Table II: Main Results

| Method | Accuracy | AUC | DPD | EOD |
| :--- | :--- | :--- | :--- | :--- |
| Zero-shot CLIP (Mode 1) | 0.5355 | 0.4988 | 0.0000 | 0.0000 |
| Adapter-only (Mode 2) | 0.5935 | 0.6382 | 0.1473 | 0.1609 |
| **DebiasCLIP (Mode 3)** | **0.6290** | **0.6847** | **0.1044** | **0.0698** |

## Analysis
- **DPD Reduction:** (0.1473 - 0.1044) / 0.1473 = **29.1% decrease**
- **EOD Reduction:** (0.1609 - 0.0698) / 0.1609 = **56.6% decrease**
- **AUC Improvement:** 0.6382 → 0.6847 (+7.3%)
- **Accuracy Improvement:** 0.5935 → 0.6290 (+6.0%)

## Adapter Baseline Model (Mode 2)
- Adapter-only: 0.5935, 0.6382, 0.1473, 0.1609
- File: adapter_baseline.pth

## DebiasCLIP Model (Mode 3)
- DebiasCLIP: 0.6290, 0.6847, 0.1044, 0.0698
- File: debiasclip_adapter_fair.pth

## Test Set Info
- Test size: 2000 samples
- Evaluation Metric: Binary Classification (Glaucoma)

## Final Results (v2 - 10 Epochs)

| Method | Accuracy | AUC | DPD | EOD |
| :--- | :--- | :--- | :--- | :--- |
| Zero-shot CLIP | 0.5355 | 0.4988 | 0.0000 | 0.0000 |
| Adapter-only | 0.5935 | 0.6382 | 0.1473 | 0.1609 |
| **DebiasCLIP (Ours)** | **0.6575** | **0.7233** | **0.0808** | **0.0936** |