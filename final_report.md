# Final Test-Set Report

The evaluation uses a **seeded random 70/10/20 split** of the 10,000-row
FairVLMed train dataset (seed 42; 7,000 train / 1,000 validation /
2,000 test). This is not the official folder split.

The stack decision threshold was selected on validation by Youden's J: `0.493392`. The same threshold is used for both models' accuracy
and per-race rates. No test-set tuning was performed.

## Overall test metrics

| Model | AUC | Accuracy at 0.50 | Accuracy at validation threshold | DPD | EOD_max |
|---|---:|---:|---:|---:|---:|
| OpenCLIP img+txt MLP | 0.8061 | 0.7270 | 0.7260 | 0.1366 | 0.1504 |
| Best stack (validation-selected) | 0.8683 | 0.7750 | 0.7750 | 0.0912 | 0.1268 |

## Paired bootstrap deltas

Each resample draws 2000 test cases with replacement and applies
the identical indices to both models. Delta is stack minus OpenCLIP.
Intervals are 95% percentile CIs from 1,000 resamples (seed 0).

| Metric | Delta | 95% percentile CI |
|---|---:|---:|
| auc | +0.0622 | [+0.0475, +0.0757] |
| accuracy | +0.0490 | [+0.0300, +0.0670] |
| dpd | -0.0454 | [-0.1173, +0.0276] |
| eod_max | -0.0236 | [-0.1498, +0.0911] |

## Per-race test rates

Rates use the validation-chosen threshold. CIs are Wilson 95% intervals.

### OpenCLIP img+txt MLP

| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| Asian | 169 | 0.7895 [0.6850, 0.8660] | 0.2105 [0.1340, 0.3150] | 0.2258 [0.1527, 0.3207] |
| Black | 302 | 0.7363 [0.6714, 0.7924] | 0.2637 [0.2076, 0.3286] | 0.3762 [0.2879, 0.4736] |
| White | 1529 | 0.7365 [0.7036, 0.7669] | 0.2635 [0.2331, 0.2964] | 0.2852 [0.2548, 0.3177] |

### Best stack (validation-selected)

| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| Asian | 169 | 0.9079 [0.8219, 0.9547] | 0.0921 [0.0453, 0.1781] | 0.1935 [0.1261, 0.2853] |
| Black | 302 | 0.7811 [0.7190, 0.8327] | 0.2189 [0.1673, 0.2810] | 0.2574 [0.1822, 0.3505] |
| White | 1529 | 0.7959 [0.7654, 0.8234] | 0.2041 [0.1766, 0.2346] | 0.2586 [0.2292, 0.2902] |
