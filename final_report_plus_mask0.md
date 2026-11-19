# Stack Plus Evaluation

Text preprocessing mask level: `0`.

All model/weight/threshold choices use validation only. Test inference and
evaluation occur after those choices. Accuracy and race rates use each
model's own validation-Youden threshold (ties nearest 0.5).

Final blend: `w=0.6` on the stack logits and `0.4` on ft_image logits; threshold `0.548705`.

## Validation and test metrics

| Model | Validation AUC | Validation accuracy | Test AUC | Test accuracy | Threshold |
|---|---:|---:|---:|---:|---:|
| OpenCLIP MLP baseline | 0.8298 | 0.7520 | 0.8061 | 0.7285 | 0.530093 |
| Mask-0 TF-IDF alone | 0.8750 | 0.7920 | 0.8550 | 0.7625 | 0.494617 |
| Mask-0 TF-IDF + base-model stack | 0.8916 | 0.8080 | 0.8683 | 0.7750 | 0.493392 |
| ft_image alone | 0.8237 | 0.7610 | 0.7898 | 0.7105 | 0.589410 |
| Final logit-space blend | 0.9077 | 0.8270 | 0.8803 | 0.7990 | 0.548705 |

## Paired test bootstrap deltas

Delta is final blend minus the named reference; identical case indices
are used for both predictions. Intervals are 95% percentile CIs from
1,000 resamples (seed 0).

| Comparison | Metric | Delta | 95% percentile CI |
|---|---|---:|---:|
| final minus stack | auc | +0.0119 | [+0.0057, +0.0187] |
| final minus stack | accuracy | +0.0240 | [+0.0105, +0.0370] |
| final minus stack | dpd | +0.0327 | [-0.0166, +0.0777] |
| final minus stack | eod_max | -0.0278 | [-0.0734, +0.0680] |
| final minus openclip joint | auc | +0.0741 | [+0.0599, +0.0878] |
| final minus openclip joint | accuracy | +0.0705 | [+0.0535, +0.0875] |
| final minus openclip joint | dpd | -0.0199 | [-0.0885, +0.0499] |
| final minus openclip joint | eod_max | -0.0432 | [-0.1410, +0.0877] |

## Per-race test rates: Mask-0 TF-IDF + base-model stack

Validation threshold: `0.493392`. Intervals are Wilson 95% CIs.

| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| Asian | 169 | 0.9079 [0.8219, 0.9547] | 0.0921 [0.0453, 0.1781] | 0.1935 [0.1261, 0.2853] |
| Black | 302 | 0.7811 [0.7190, 0.8327] | 0.2189 [0.1673, 0.2810] | 0.2574 [0.1822, 0.3505] |
| White | 1529 | 0.7959 [0.7654, 0.8234] | 0.2041 [0.1766, 0.2346] | 0.2586 [0.2292, 0.2902] |

## Per-race test rates: Final logit-space blend

Validation threshold: `0.548705`. Intervals are Wilson 95% CIs.

| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| Asian | 169 | 0.8553 [0.7591, 0.9172] | 0.1447 [0.0828, 0.2409] | 0.1290 [0.0754, 0.2121] |
| Black | 302 | 0.7562 [0.6924, 0.8104] | 0.2438 [0.1896, 0.3076] | 0.2277 [0.1568, 0.3186] |
| White | 1529 | 0.7784 [0.7471, 0.8068] | 0.2216 [0.1932, 0.2529] | 0.1812 [0.1559, 0.2096] |
