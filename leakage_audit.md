# Test-split leakage audit

Scope: inspected the requested scripts as they exist in this workspace. A
script is marked **uses test** when test labels or metrics influence model,
blend, threshold, epoch, or early-stopping choices. Computing final test metrics
after those choices does not, by itself, count as leakage.

| File | Verdict | Evidence |
|---|---|---|
| `src/train_super_ensemble_90plus.py` | **Not auditable — file missing** | The path does not exist in the workspace, is not tracked at `HEAD`, and has no path history in the available Git refs. No clean/uses-test determination can be made without the script. |
| [`src/triple_super_ensemble.py`](./src/triple_super_ensemble.py) | **clean** | MLP and cross-attention epoch counts are fixed at 35 and 40 (lines 67 and 102). The SVM, MLP, and cross-attention test predictions are generated at lines 42, 75, and 110, but blend weights and threshold are selected by validation accuracy at lines 113–129. The selected settings are then applied to test predictions and reported at lines 136–142. |
| [`src/blend_top_models.py`](./src/blend_top_models.py) | **clean** | The MLP trains for a fixed 35 epochs (line 50); blend weights are fixed at 0.60/0.40 (lines 60–62), not selected from test results. Threshold search uses validation labels at lines 64–69. Test labels are used afterward for final metrics at lines 71–75. |
| [`src/boost_accuracy.py`](./src/boost_accuracy.py) | **clean** | Training epochs are fixed by the function default and loop (lines 65, 81–82); test probabilities are collected for later evaluation at lines 96–100. Thresholds are chosen from validation labels at lines 112–125. Test labels and probabilities are used by the final evaluator at lines 127–151; no test-driven choice is present. |
| [`src/boost_accuracy_v2.py`](./src/boost_accuracy_v2.py) | **clean** | The gated-model training epoch count is fixed by the default of 50 (lines 88, 102); test predictions are collected alongside validation predictions at lines 115–119. Hybrid weights are fixed at 0.65/0.35 for both splits at lines 138–143. Thresholds are selected using validation labels at lines 145–157; test labels are only used in metric computation at lines 159–168. |
| [`src/accuracy_boost.py`](./src/accuracy_boost.py) | **clean** | The residual MLP trains for a fixed 35 epochs (line 80). Ensemble weights are fixed at 0.50/0.30/0.20 for both validation and test predictions (lines 118–120); the reported threshold is fixed at 0.50 (line 125). Test labels are used for the printed final comparison at lines 132–138, not to choose settings. |
| [`src/boost_cross_attention.py`](./src/boost_cross_attention.py) | **uses test** | The script computes test accuracy/AUC at the default cutoff (lines 81–84), then explicitly searches thresholds from 0.40 to 0.60 and maximizes accuracy against `y_test` (lines 86–93). The resulting test-tuned `best_thr` and `best_acc` are reported as “Maximum Tuned Accuracy” at line 99. This is direct test-label leakage. |
| [`src/boost_gbdt.py`](./src/boost_gbdt.py) | **clean** | LightGBM early stopping receives training and validation datasets only (lines 49–58); the resulting `best_iteration` is used for both validation and test predictions (lines 62–63). Threshold search uses validation labels at lines 65–71. Test metrics are computed afterward at lines 73–75. |

## Summary

- **Uses test:** `src/boost_cross_attention.py` — threshold selection maximizes
  accuracy on test labels.
- **Clean among scripts present:** `src/triple_super_ensemble.py`,
  `src/blend_top_models.py`, `src/boost_accuracy.py`,
  `src/boost_accuracy_v2.py`, `src/accuracy_boost.py`, and `src/boost_gbdt.py`.
- **Missing / not auditable:** `src/train_super_ensemble_90plus.py`.

No source scripts were modified during this audit.
