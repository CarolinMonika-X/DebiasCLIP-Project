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



# DebiasCLIP Final Results

| Method | Accuracy | AUC | DPD | EOD |
| :--- | :--- | :--- | :--- | :--- |
| Zero-shot CLIP | 0.5355 | 0.4988 | N/A | N/A |
| Adapter-only | 0.5935 | 0.6382 | 0.1473 | 0.1609 |
| DebiasCLIP V2 (Image-only) | 0.6575 | 0.7233 | 0.0808 | 0.0936 |
| **DebiasCLIP V3 (Dual-Modal)** | **0.7330** | **0.8098** | **0.0720** | **0.0759** |

**Key Achievements:**
- Accuracy improved from 59.35% (baseline) to 73.30%.
- AUC improved from 0.6382 (baseline) to 0.8098.
- DPD reduced by 51.1% (0.1473 -> 0.0720).
- EOD reduced by 52.8% (0.1609 -> 0.0759).

# Novelty: Modality-Specific Debiasing Strategy

| Strategy | Accuracy | AUC | DPD (↓) | EOD (↓) |
| :--- | :--- | :--- | :--- | :--- |
| Standard (Baseline) | 70.10% | 0.7826 | 0.1804 | 0.1193 |
| Down-weight Image (α=0.5) | **70.95%** | **0.7869** | **0.1522** | **0.1001** |
| Down-weight Image (α=0.1) | 68.75% | 0.7742 | **0.1219** | 0.1303 |

**Key Finding:** Down-weighting the biased image features by α=0.5 reduces DPD by **15.6%** and EOD by **16.1%**, while improving accuracy.
##s.
CLIP embedding dimension: 512
>>> MODE 3: DebiasCLIP V4 (Dual-Modal Model) <<<
Loaded debiasclip_adapter_v4.pth successfully.
Model loaded successfully!
Test size: 2000
Evaluating... (Idhu konjam neram aagum)

--- Overall Results (Threshold = 0.5) ---
Accuracy: 0.7710
AUC: 0.8588

--- Optimal Threshold Results ---
Best Threshold: 0.50
Best Accuracy: 0.7710 (Improved from 0.7710)

--- Per Group Results (Optimal Threshold = 0.50) ---
Group 0: Accuracy=0.7862, AUC=0.8370, Positive Rate=0.6478
Group 1: Accuracy=0.8019, AUC=0.8668, Positive Rate=0.6645
Group 2: Accuracy=0.7631, AUC=0.8623, Positive Rate=0.5949

Demographic Parity Difference (DPD): 0.0696

--- EOD Calculation (using Optimal Threshold) ---
FPR Gap: 0.0560
TPR Gap (Equal Opportunity Difference): 0.0489
Equalized Odds Difference (EOD): 0.0560