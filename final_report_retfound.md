# Final Upgraded Evaluation (BERT_L2 found: True, files=['bert_L2_s0.npz', 'bert_L2_s2.npz'])

| Tier | Weights | Val AUC | Test AUC | Test Acc (Youden) | Test Acc (MaxAcc) |
|---|---|---:|---:|---:|---:|
| Image-only: ft_image (3 seeds) | - | 0.8237 | 0.7898 | 0.7105 (0.589) | 0.7105 (0.590) |
| Image-only: RETFound (336px TTA) | - | 0.8453 | 0.8245 | 0.7470 (0.500) | 0.7465 (0.499) |
| Clean Mask-2 (TF-IDF+ft+BERT_L2+RETFound) | (0.25, 0.15, 0.3, 0.3) | 0.8856 | 0.8599 | 0.7730 (0.469) | 0.7755 (0.601) |
| Partially masked (Stack+ft+BERT_L2+RETFound) | (0.35, 0.1, 0.25, 0.3) | 0.8940 | 0.8664 | 0.7855 (0.516) | 0.7850 (0.516) |
| Unmasked upper bound (Stack0+ft+BERT_L0+RETFound) | (0.2, 0.15, 0.35, 0.3) | 0.9260 | 0.9137 | 0.8300 (0.523) | 0.8295 (0.522) |

## Paired Test Bootstrap Deltas (1,000 resamples, 95% CI)

- **RETFound minus ft_image (image-only)**: dAcc = +0.0359 [+0.0160, +0.0550], dAUC = +0.0347 [+0.0207, +0.0484]
- **Clean Mask-2 multimodal minus RETFound**: dAcc = +0.0289 [+0.0130, +0.0440], dAUC = +0.0355 [+0.0243, +0.0468]
- **Unmasked 4-way blend minus OpenCLIP MLP baseline**: dAcc = +0.1029 [+0.0845, +0.1220], dAUC = +0.1057 [+0.0913, +0.1209]

## Per-Race Test Rates: Clean Mask-2 Multimodal (thr=0.601, Wilson 95% CI)

| Race | N | TPR (95% CI) | FNR | FPR (95% CI) |
|---|---:|---|---|---|
| Asian | 169 | 0.776 [0.671, 0.855] | 0.224 | 0.097 [0.052, 0.174] |
| Black | 302 | 0.692 [0.625, 0.751] | 0.308 | 0.149 [0.092, 0.231] |
| White | 1529 | 0.672 [0.637, 0.704] | 0.328 | 0.131 [0.109, 0.156] |

## Per-Race Test Rates: Unmasked Upper Bound (thr=0.522, Wilson 95% CI)

| Race | N | TPR (95% CI) | FNR | FPR (95% CI) |
|---|---:|---|---|---|
| Asian | 169 | 0.934 [0.855, 0.972] | 0.066 | 0.108 [0.059, 0.187] |
| Black | 302 | 0.831 [0.773, 0.876] | 0.169 | 0.228 [0.157, 0.319] |
| White | 1529 | 0.822 [0.792, 0.848] | 0.178 | 0.174 [0.149, 0.202] |