# Stack Plus Evaluation

Text preprocessing mask level: `2`.

All model/weight/threshold choices use validation only. Test inference and
evaluation occur after those choices. Accuracy and race rates use each
model's own validation-Youden threshold (ties nearest 0.5).

Final blend: `w=0.9` on the stack logits and `0.1` on ft_image logits; threshold `0.509925`.

## Validation and test metrics

| Model | Validation AUC | Validation accuracy | Test AUC | Test accuracy | Threshold |
|---|---:|---:|---:|---:|---:|
| OpenCLIP MLP baseline | 0.6141 | 0.6400 | 0.7192 | 0.6800 | 0.689881 |
| Mask-2 TF-IDF alone | 0.4598 | 0.4800 | 0.6120 | 0.5800 | 0.387717 |
| Mask-2 TF-IDF + base-model stack | 0.6305 | 0.6600 | 0.6997 | 0.6200 | 0.502971 |
| ft_image alone | 0.8046 | 0.7800 | 0.9286 | 0.8600 | 0.570757 |
| Final logit-space blend | 0.8128 | 0.8000 | 0.9253 | 0.8800 | 0.509925 |

## Paired test bootstrap deltas

Delta is final blend minus the named reference; identical case indices
are used for both predictions. Intervals are 95% percentile CIs from
1,000 resamples (seed 0).

| Comparison | Metric | Delta | 95% percentile CI |
|---|---|---:|---:|
| final minus stack | auc | +0.2256 | [+0.0984, +0.3668] |
| final minus stack | accuracy | +0.2600 | [+0.1400, +0.3800] |
| final minus stack | dpd | -0.1485 | [-0.4706, +0.2821] |
| final minus stack | eod_max | -0.2667 | [-0.6667, +0.4618] |
| final minus openclip joint | auc | +0.2062 | [+0.0801, +0.3524] |
| final minus openclip joint | accuracy | +0.2000 | [+0.0800, +0.3200] |
| final minus openclip joint | dpd | +0.1162 | [-0.3334, +0.4250] |
| final minus openclip joint | eod_max | -0.1333 | [-0.6667, +0.4274] |

## Per-race test rates: Mask-2 TF-IDF + base-model stack

Validation threshold: `0.502971`. Intervals are Wilson 95% CIs.

| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| Asian | 7 | 0.0000 [0.0000, 0.5615] | 1.0000 [0.4385, 1.0000] | 0.0000 [0.0000, 0.4899] |
| Black | 9 | 0.4000 [0.1176, 0.7693] | 0.6000 [0.2307, 0.8824] | 0.2500 [0.0456, 0.6994] |
| White | 34 | 0.4000 [0.2188, 0.6134] | 0.6000 [0.3866, 0.7812] | 0.0000 [0.0000, 0.2153] |

## Per-race test rates: Final logit-space blend

Validation threshold: `0.509925`. Intervals are Wilson 95% CIs.

| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| Asian | 7 | 0.6667 [0.2077, 0.9385] | 0.3333 [0.0615, 0.7923] | 0.0000 [0.0000, 0.4899] |
| Black | 9 | 0.8000 [0.3755, 0.9638] | 0.2000 [0.0362, 0.6245] | 0.0000 [0.0000, 0.4899] |
| White | 34 | 0.8000 [0.5840, 0.9193] | 0.2000 [0.0807, 0.4160] | 0.0000 [0.0000, 0.2153] |
