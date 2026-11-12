# Run reconciliation report

## Conclusion

The reported raw Dual results differ because these files summarize **different
trained models**, not because one run used a different AUC cutoff. The older
run reports AUC **0.8005**, EOD_max **0.1453**,
and Black TPR **0.811**. The later placebo-audit
five-seed ensemble reports AUC **0.8137**, EOD_max
**0.1258**, and Black TPR **0.692**.
Later-minus-older differences are AUC **+0.0133**, EOD_max
**-0.0195**, and Black TPR **-0.119**.

The largest confirmed differences are the diagnostic head (256 hidden with
BatchNorm and 0.2 dropout versus 512 hidden without BatchNorm and 0.1 dropout),
optimizer (AdamW versus Adam), epoch count (30 versus 40), batch size (128
versus 256), weight decay, and seed values. These change the fitted weights,
ranking, and thresholded subgroup decisions. The saved summaries do not permit
isolating one of these changes as the sole cause.

Both stored raw-run summaries use a nominal **0.50** cutoff. The older code uses
`>= 0.50`, while the later code uses `> 0.50`. This can affect only examples
whose score is exactly 0.50 and can alter thresholded rates; it cannot change
AUC. AUC changed by +0.0133, so a cutoff convention alone is ruled
out. No separate post-hoc calibration step is configured; a monotonic
calibration alone would not change AUC either.

## Configuration differences

Settings are read from the producer scripts; the actual later cache path and
QUICK mode are checked against `placebo_audit_output.txt`.

| Field | Older unified benchmark | Later placebo audit | Difference and likely relevance |
|---|---|---|---|
| `feature_normalization` | Cached image and text embeddings are independently L2-normalized; neither benchmark applies an additional StandardScaler. | Cached image and text embeddings are independently L2-normalized; neither benchmark applies an additional StandardScaler. | Same. Same cached normalization; not a cause. |
| `feature_layout` | Concatenate cached image[512] then text[512] features | Concatenate cached image[512] then text[512] features | Same. Same joint feature order/dimension; not a cause. |
| `head_architecture` | Linear(1024,256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.2) -> Linear(256,2) | Linear(1024,512) -> ReLU -> Dropout(0.1) -> Linear(512,2) | Different. Major training change; changes learned scores and subgroup errors. |
| `optimizer` | AdamW | Adam (default weight decay 0) | Different. Training change; contributes to a non-identical fit. |
| `learning_rate` | 0.001 | 0.001 | Same. Same; not a cause by itself. |
| `weight_decay` | 0.0001 | 0.0 | Different. Training change; may affect fit. |
| `batch_size` | 128 | 256 | Different. Training change; affects updates and batch statistics. |
| `epochs` | 30 | 40 | Different. Training change; affects fitted weights and probability calibration. |
| `class_weighting` | None for Raw Dual | None for Raw Dual | Same. Same for raw baseline; not a cause. |
| `seed_list` | [0, 100, 200, 300, 400] | [0, 1, 2, 3, 4] | Different. Different initialization/data-order streams; changes fitted heads. |
| `seed_mapping` | torch.manual_seed(seed * 100); seed in range(5) | Raw ensemble uses first five seeds from raw30: torch.manual_seed(seed), np.random.seed(seed) | Different. Different RNG streams; can alter model weights and minibatch order. |
| `threshold_rule` | probability >= 0.50 | probability > 0.50 | Different. Both use nominal 0.50. `>=` versus `>` only differs for exact ties; cannot explain AUC. |
| `ensemble` | Mean predicted probability over five seed models | Mean predicted probability over raw seeds 0-4; these are the first five of the separate 30-seed raw variability run | Different. Both average five heads, but they are not the same five trained heads. |
| `split_file` | cached_features.pt | cached_features.pt | Same. Both runs use cached_features.pt; not a cause. |
| `split_sizes` | {'train': 7000, 'val': 1000, 'test': 2000} | {'train': 7000, 'val': 1000, 'test': 2000} | Same. Same official split sizes; not a cause. |
| `model_selection` | Validation only; raw baseline uses fixed settings | Raw baseline fixed; fairness settings were selected on validation | Different. Raw model has fixed settings; no test-based tuning found. |
| `numpy_seed` | Not set in raw-head training loop | np.random.seed(seed) per raw-head seed | Different. Later run explicitly seeds NumPy; older raw loop does not. |

## Same-prediction 0.50 replay

**Prediction-vector availability:** Unavailable. Neither result JSON contains a 2,000-element test probability vector; the logs contain aggregate metrics only.

Exact re-scoring of the identical predictions cannot be performed from the saved artifacts. The stored benchmark metrics are already reported at the nominal 0.50 cutoff. No model was retrained here to stand in for the missing
predictions. The existing outputs already record raw metrics at 0.50, and
those recorded fixed-cutoff metrics show the gap. A strictly paired replay of
the *same* score arrays, including a test for exact 0.50 ties, requires the
original saved per-patient test probabilities, which neither JSON contains.

The older per-group raw Black TPR is stored directly in its result JSON. The
later Black TPR is read from the placebo run's fixed-cutoff subgroup log.
These are not the same prediction vector: the raw-head configurations and
seed streams differ.

## Split and leakage checks

- Train/validation/test sizes: {'train': 7000, 'val': 1000, 'test': 2000}.
- Test race counts (Asian/Black/White): {'Asian': 169, 'Black': 302, 'White': 1529}.
- The test split is used only for reported evaluation; tuning is recorded as
  validation-only: **False** test-based selection.
- Cache path in both actual runs: `cached_features.pt`.
- Both runs consume the same cached, independently L2-normalized image/text
  embeddings; neither applies an additional feature standardizer.

## Canonical result choice

`canonical_results.json` selects the later `placebo_audit_results.json`
five-seed ensemble as requested. It is the later run, includes a placebo
audit, has an explicit per-group Wilson-interval log, and has test group
counts matching the official cache. This is a **canonical recorded run**, not
a claim that its head configuration is otherwise identical to the older
benchmark. The original result files are left unchanged.

| Model | AUC | Accuracy | DPD | EOD_max |
|---|---:|---:|---:|---:|
| raw | AUC=0.8137391706203092 | accuracy=0.737 | DPD=0.17167600611309214 | EOD_max=0.12583839029064195 |
| leace | AUC=0.811285461498373 | accuracy=0.731 | DPD=0.12884517418394137 | EOD_max=0.09443202384754606 |
| inlp | AUC=0.8123207606998423 | accuracy=0.7335 | DPD=0.16624867745601313 | EOD_max=0.09613541999361228 |
| parity | AUC=0.813755175245646 | accuracy=0.736 | DPD=0.16244758807163295 | EOD_max=0.1159374001916321 |
| labelcond | AUC=0.5034564989281901 | accuracy=0.5085 | DPD=0.0 | EOD_max=0.0 |
| reweight | AUC=0.7919018596374353 | accuracy=0.716 | DPD=0.13497785963399822 | EOD_max=0.1314808900244863 |
| thresholds | AUC=0.8137391706203092 | accuracy=0.7375 | DPD=0.2140170069360085 | EOD_max=0.1654423506866816 |
