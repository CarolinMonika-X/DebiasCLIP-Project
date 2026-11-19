# Stack and final-report audit

Scope: read-only audit of [`stack_models.py`](./src/stack_models.py) and
[`final_report.py`](./src/final_report.py). Verdicts below describe the code path,
not whether the full stack experiment was run.

## Findings

| # | Question | Yes / No | Evidence |
|---|---|---|---|
| 1 | Is every hyperparameter, meta-model C, and decision threshold chosen on validation only? | **No.** Meta-model C and the stack threshold are validation-selected, but the other settings are fixed rather than selected on validation. | The candidate meta-C values are declared in [`stack_models.py:L36`](./src/stack_models.py#L36), scored on validation at [`stack_models.py:L662`](./src/stack_models.py#L662), [`stack_models.py:L670`](./src/stack_models.py#L670), and [`stack_models.py:L671`](./src/stack_models.py#L671), and selected by validation AUC at [`stack_models.py:L675`](./src/stack_models.py#L675)–[`stack_models.py:L680`](./src/stack_models.py#L680). Youden's threshold is calculated from validation labels/probabilities at [`stack_models.py:L326`](./src/stack_models.py#L326)–[`stack_models.py:L338`](./src/stack_models.py#L338), with the validation inputs passed at [`stack_models.py:L681`](./src/stack_models.py#L681)–[`stack_models.py:L685`](./src/stack_models.py#L685). In contrast, the MLP epochs, batch size, learning rate, and weight decay are fixed at [`stack_models.py:L32`](./src/stack_models.py#L32)–[`stack_models.py:L35`](./src/stack_models.py#L35); TF-IDF settings and text-model `C=1.0` are fixed at [`stack_models.py:L230`](./src/stack_models.py#L230)–[`stack_models.py:L245`](./src/stack_models.py#L245). They are not validation-selected. |
| 2 | Are OOF predictions built on TRAIN only with five-fold stratification? | **Yes.** | The folds use `StratifiedKFold(n_splits=N_FOLDS, ...)` at [`stack_models.py:L404`](./src/stack_models.py#L404), and are generated from `train_labels` only at [`stack_models.py:L406`](./src/stack_models.py#L406)–[`stack_models.py:L408`](./src/stack_models.py#L408). Each held-out prediction is written to the corresponding training OOF indices. TF-IDF is fit on each fold's fit indices and applied to its held-out indices at [`stack_models.py:L414`](./src/stack_models.py#L414)–[`stack_models.py:L423`](./src/stack_models.py#L423); MLPs train on fold-fit rows and predict fold-held-out rows at [`stack_models.py:L427`](./src/stack_models.py#L427)–[`stack_models.py:L441`](./src/stack_models.py#L441). |
| 3 | Is there a refit on TRAIN+VAL only after all selection, and is TEST used exactly once? | **No.** There is no TRAIN+VAL refit. Test predictions and scoring occur after meta-C and threshold selection, but test data are loaded earlier for cache validation/variant discovery, and the final report scores the saved test probabilities again. | Full base models are explicitly fit on full TRAIN and predict VAL at [`stack_models.py:L447`](./src/stack_models.py#L447)–[`stack_models.py:L482`](./src/stack_models.py#L482); that fit is called before the meta-C sweep at [`stack_models.py:L621`](./src/stack_models.py#L621)–[`stack_models.py:L635`](./src/stack_models.py#L635). No subsequent TRAIN+VAL refit appears after selection. The C and threshold selection precedes the test-inference section at [`stack_models.py:L662`](./src/stack_models.py#L662)–[`stack_models.py:L692`](./src/stack_models.py#L692), and predictions/metrics are then produced at [`stack_models.py:L720`](./src/stack_models.py#L720)–[`stack_models.py:L778`](./src/stack_models.py#L778). However, `read_cache` loads each split, including TEST, at [`stack_models.py:L94`](./src/stack_models.py#L94)–[`stack_models.py:L127`](./src/stack_models.py#L127); the report separately loads TEST labels/races at [`final_report.py:L221`](./src/final_report.py#L221)–[`final_report.py:L226`](./src/final_report.py#L226) and computes test metrics/bootstrap at [`final_report.py:L489`](./src/final_report.py#L489)–[`final_report.py:L530`](./src/final_report.py#L530). This is repeated test-set reporting, although neither script uses those test metrics to select C or the threshold. |
| 4 | Are race words removed from TF-IDF, and is race never used as a feature? | **Partly.** Listed race/ethnicity words are removed from TF-IDF and checked against its vocabulary. Race labels are not passed as a separate model feature. But the cached CLIP text embeddings are used without scrubbing in the neural base models, so the code does not establish that race information is absent from every feature representation. | The removal vocabulary/regex is defined at [`stack_models.py:L39`](./src/stack_models.py#L39)–[`stack_models.py:L57`](./src/stack_models.py#L57); the TF-IDF preprocessor removes matching terms and checks the fitted vocabulary at [`stack_models.py:L230`](./src/stack_models.py#L230)–[`stack_models.py:L238`](./src/stack_models.py#L238). MLP inputs are image/text feature matrices, with no race array concatenated, at [`stack_models.py:L341`](./src/stack_models.py#L341)–[`stack_models.py:L363`](./src/stack_models.py#L363); text fitting receives text and glaucoma labels, not race labels, at [`stack_models.py:L226`](./src/stack_models.py#L226)–[`stack_models.py:L245`](./src/stack_models.py#L245). Conversely, cached `txt` features are consumed as-is for joint models at [`stack_models.py:L345`](./src/stack_models.py#L345)–[`stack_models.py:L353`](./src/stack_models.py#L353), with no text-token scrubbing in this file. Race may therefore be encoded implicitly in those embeddings. |
| 5 | Does the report print every variant tried, not only the best? | **No.** | `final_report.py` fixes its report model set to `("openclip_joint", "stack")` at [`final_report.py:L23`](./src/final_report.py#L23), and both Markdown and LaTeX iterate only that tuple at [`final_report.py:L306`](./src/final_report.py#L306), [`final_report.py:L342`](./src/final_report.py#L342), [`final_report.py:L402`](./src/final_report.py#L402), and [`final_report.py:L449`](./src/final_report.py#L449). `stack_models.py` dynamically builds the available base-model list at [`stack_models.py:L612`](./src/stack_models.py#L612)–[`stack_models.py:L615`](./src/stack_models.py#L615) and records base-model results at [`stack_models.py:L760`](./src/stack_models.py#L760)–[`stack_models.py:L769`](./src/stack_models.py#L769), but `final_report.py` does not iterate those results. |

## Seeds

- MLP ensemble seed list: `0, 100, 200, 300, 400`, declared at
  [`stack_models.py:L30`](./src/stack_models.py#L30). The script seeds Python
  `random`, NumPy, Torch, and all CUDA devices in `set_seed` at
  [`stack_models.py:L76`](./src/stack_models.py#L76)–[`stack_models.py:L81`](./src/stack_models.py#L81).
  It calls `set_seed(0)` before data/model setup at
  [`stack_models.py:L534`](./src/stack_models.py#L534), and resets the RNGs per
  MLP seed in `train_mlp_model` at [`stack_models.py:L271`](./src/stack_models.py#L271)–[`stack_models.py:L275`](./src/stack_models.py#L275).
- OOF MLP runs use each listed seed plus `10,000 * (fold + 1)` at
  [`stack_models.py:L427`](./src/stack_models.py#L427)–[`stack_models.py:L435`](./src/stack_models.py#L435).
  Full-TRAIN MLPs use the listed seeds directly at
  [`stack_models.py:L468`](./src/stack_models.py#L468)–[`stack_models.py:L474`](./src/stack_models.py#L474).
- The stratified fold splitter uses `random_state=0` at
  [`stack_models.py:L404`](./src/stack_models.py#L404). Text and meta logistic
  regression use `random_state=SEEDS[0]` at
  [`stack_models.py:L239`](./src/stack_models.py#L239)–[`stack_models.py:L245`](./src/stack_models.py#L245)
  and [`stack_models.py:L662`](./src/stack_models.py#L662)–[`stack_models.py:L668`](./src/stack_models.py#L668).
- The data split is reconstructed with Torch generator seed `42` at
  [`stack_models.py:L187`](./src/stack_models.py#L187)–[`stack_models.py:L191`](./src/stack_models.py#L191).
- The final report's paired bootstrap uses seed `0`, configured at
  [`final_report.py:L24`](./src/final_report.py#L24)–[`final_report.py:L25`](./src/final_report.py#L25)
  and passed to `np.random.default_rng` at
  [`final_report.py:L161`](./src/final_report.py#L161). The report does not train
  models or set model-training RNGs.

## Places TEST could influence a choice

1. **Variant roster/eligibility:** `verify_alignment` checks `labels` and `races`
   for every split, including TEST, at [`stack_models.py:L131`](./src/stack_models.py#L131)–[`stack_models.py:L141`](./src/stack_models.py#L141).
   `find_image_variants` uses that check to include or skip a cache at
   [`stack_models.py:L144`](./src/stack_models.py#L144)–[`stack_models.py:L166`](./src/stack_models.py#L166);
   it is invoked before model specs are built at
   [`stack_models.py:L550`](./src/stack_models.py#L550) and
   [`stack_models.py:L612`](./src/stack_models.py#L612). Thus TEST alignment can
   affect which image variant enters the model roster, even though no test
   performance metric is used for meta-C or threshold selection.
2. **No direct metric-based test selection was found:** meta-C selection is based
   on validation AUC at [`stack_models.py:L670`](./src/stack_models.py#L670)–[`stack_models.py:L680`](./src/stack_models.py#L680),
   and threshold selection receives validation labels/probabilities at
   [`stack_models.py:L681`](./src/stack_models.py#L681)–[`stack_models.py:L685`](./src/stack_models.py#L685).
   Test evaluation follows afterward at [`stack_models.py:L692`](./src/stack_models.py#L692)–[`stack_models.py:L769`](./src/stack_models.py#L769).

