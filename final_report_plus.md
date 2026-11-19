# Stack Plus Evaluation

Text preprocessing mask level: `2`.

All model/weight/threshold choices use validation only. Test inference and
evaluation occur after those choices. Accuracy and race rates use each
model's own validation-Youden threshold (ties nearest 0.5).

Final blend: `w=0.6` on the stack logits and `0.4` on ft_image logits; threshold `0.522374`.

## Validation and test metrics

| Model | Validation AUC | Validation accuracy | Test AUC | Test accuracy | Threshold |
|---|---:|---:|---:|---:|---:|
| OpenCLIP MLP baseline | 0.8298 | 0.7520 | 0.8061 | 0.7285 | 0.530093 |
| Mask-2 TF-IDF alone | 0.7636 | 0.6950 | 0.7434 | 0.6870 | 0.480279 |
| Mask-2 TF-IDF + base-model stack | 0.8595 | 0.7840 | 0.8301 | 0.7550 | 0.572156 |
| ft_image alone | 0.8237 | 0.7610 | 0.7898 | 0.7105 | 0.589410 |
| Final logit-space blend | 0.8793 | 0.8050 | 0.8467 | 0.7615 | 0.522374 |

## Paired test bootstrap deltas

Delta is final blend minus the named reference; identical case indices
are used for both predictions. Intervals are 95% percentile CIs from
1,000 resamples (seed 0).

| Comparison | Metric | Delta | 95% percentile CI |
|---|---|---:|---:|
| final minus stack | auc | +0.0166 | [+0.0099, +0.0241] |
| final minus stack | accuracy | +0.0065 | [-0.0065, +0.0200] |
| final minus stack | dpd | -0.0116 | [-0.0623, +0.0408] |
| final minus stack | eod_max | -0.0099 | [-0.0766, +0.0790] |
| final minus openclip joint | auc | +0.0406 | [+0.0279, +0.0522] |
| final minus openclip joint | accuracy | +0.0330 | [+0.0170, +0.0490] |
| final minus openclip joint | dpd | -0.0132 | [-0.0861, +0.0544] |
| final minus openclip joint | eod_max | -0.0254 | [-0.1225, +0.0905] |

## Per-race test rates: Mask-2 TF-IDF + base-model stack

Validation threshold: `0.572156`. Intervals are Wilson 95% CIs.

| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| Asian | 169 | 0.7368 [0.6282, 0.8227] | 0.2632 [0.1773, 0.3718] | 0.1505 [0.0918, 0.2370] |
| Black | 302 | 0.6965 [0.6298, 0.7559] | 0.3035 [0.2441, 0.3702] | 0.2772 [0.1993, 0.3715] |
| White | 1529 | 0.7257 [0.6924, 0.7566] | 0.2743 [0.2434, 0.3076] | 0.2079 [0.1810, 0.2376] |

## Per-race test rates: Final logit-space blend

Validation threshold: `0.522374`. Intervals are Wilson 95% CIs.

| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| Asian | 169 | 0.8289 [0.7290, 0.8972] | 0.1711 [0.1028, 0.2710] | 0.1505 [0.0918, 0.2370] |
| Black | 302 | 0.7463 [0.6819, 0.8014] | 0.2537 [0.1986, 0.3181] | 0.2673 [0.1907, 0.3610] |
| White | 1529 | 0.7554 [0.7232, 0.7850] | 0.2446 [0.2150, 0.2768] | 0.2421 [0.2135, 0.2732] |
