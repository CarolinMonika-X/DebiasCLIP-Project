"""Reconcile the unified benchmark and placebo-audit result artifacts.

This script does not train models or modify either source result. The source
artifacts contain aggregate metrics but omit test probability vectors, so an
exact same-prediction threshold replay is only possible if vectors are later
added to the JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_RESULTS = ROOT / "unified_final_results.json"
PLACEBO_RESULTS = ROOT / "placebo_audit_results.json"
UNIFIED_SCRIPT = ROOT / "src" / "unified_benchmark_final.py"
PLACEBO_SCRIPT = ROOT / "src" / "placebo_audit.py"
PLACEBO_LOG = ROOT / "placebo_audit_output.txt"
CACHE_FILE = ROOT / "cached_features.pt"
REPORT_FILE = ROOT / "reconcile_report.md"
CANONICAL_FILE = ROOT / "canonical_results.json"
CHECKPOINT_FILE = ROOT / ".reconcile_runs_checkpoint.json"
RACE_NAMES = ("Asian", "Black", "White")
EXPECTED_TEST_COUNTS = (169, 302, 1529)
SEED = 0


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_fingerprints() -> dict[str, str]:
    inputs = (
        UNIFIED_RESULTS,
        PLACEBO_RESULTS,
        UNIFIED_SCRIPT,
        PLACEBO_SCRIPT,
        PLACEBO_LOG,
        CACHE_FILE,
    )
    return {str(path.relative_to(ROOT)): sha256(path) for path in inputs}


def verify_split_and_cohort_sanity(
    unified: dict[str, Any],
    placebo: dict[str, Any],
) -> dict[str, Any]:
    assert CACHE_FILE.is_file(), f"Missing required cache: {CACHE_FILE}"
    cache = torch.load(CACHE_FILE, map_location="cpu", weights_only=True)
    sizes = {split: len(cache[split]["labels"]) for split in ("train", "val", "test")}
    assert sizes["train"] == 7000, sizes
    assert sizes["val"] == 1000, sizes
    assert sizes["test"] == 2000, sizes

    race_counts = np.bincount(
        cache["test"]["races"].long().numpy(), minlength=len(RACE_NAMES)
    ).tolist()
    assert tuple(race_counts) == EXPECTED_TEST_COUNTS, race_counts

    stored_cohorts = unified.get("cohort_stats", {})
    assert tuple(
        int(stored_cohorts[name]["n"]) for name in RACE_NAMES
    ) == EXPECTED_TEST_COUNTS, stored_cohorts
    assert "table1" in placebo and "ensemble" in placebo["table1"]

    # Neither artifact may select a model or operating point on the test split.
    # The unified benchmark documents its tuning on validation, and placebo
    # fixes its lambda/k to values selected in that validation benchmark.
    assert "validation_tuning" in unified
    assert "validation" in PLACEBO_SCRIPT.read_text(encoding="utf-8").lower()

    return {
        "split_sizes": sizes,
        "test_race_counts": dict(zip(RACE_NAMES, race_counts)),
        "test_race_count_order": list(RACE_NAMES),
        "selection_split": "validation",
        "test_used_for_selection": False,
    }


def config_snapshots() -> dict[str, dict[str, Any]]:
    """Run settings transcribed from the producing scripts and run log."""
    unified_source = UNIFIED_SCRIPT.read_text(encoding="utf-8")
    placebo_source = PLACEBO_SCRIPT.read_text(encoding="utf-8")
    placebo_log = PLACEBO_LOG.read_text(encoding="utf-8")

    # Guard the explicitly transcribed settings against silent source drift.
    for token in (
        "nn.BatchNorm1d(256)",
        "nn.Dropout(0.2)",
        "seed * 100",
        "torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)",
        "for epoch in range(30)",
        "threshold=0.50",
    ):
        assert token in unified_source, f"Could not verify unified setting: {token}"
    for token in (
        "EPOCHS, BATCH, LR, THR, N_CLS = (4 if QUICK else 40), 256, 1e-3, 0.5, 3",
        "nn.Linear(dim, 512)",
        "nn.Dropout(0.1)",
        "torch.manual_seed(seed)",
        "torch.optim.Adam(m.parameters(), lr=LR)",
        "pred = p > t",
    ):
        assert token in placebo_source, f"Could not verify placebo setting: {token}"
    assert "QUICK=False" in placebo_log
    assert "cache: cached_features.pt" in placebo_log

    common_normalization = (
        "Cached image and text embeddings are independently L2-normalized; "
        "neither benchmark applies an additional StandardScaler."
    )
    return {
        "unified_final": {
            "source_script": str(UNIFIED_SCRIPT.relative_to(ROOT)),
            "feature_normalization": common_normalization,
            "feature_layout": "Concatenate cached image[512] then text[512] features",
            "head_architecture": (
                "Linear(1024,256) -> BatchNorm1d(256) -> ReLU -> "
                "Dropout(0.2) -> Linear(256,2)"
            ),
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 128,
            "epochs": 30,
            "class_weighting": "None for Raw Dual",
            "seed_list": [0, 100, 200, 300, 400],
            "seed_mapping": "torch.manual_seed(seed * 100); seed in range(5)",
            "threshold_rule": "probability >= 0.50",
            "ensemble": "Mean predicted probability over five seed models",
            "split_file": "cached_features.pt",
            "split_sizes": {"train": 7000, "val": 1000, "test": 2000},
            "model_selection": "Validation only; raw baseline uses fixed settings",
            "numpy_seed": "Not set in raw-head training loop",
        },
        "placebo_audit": {
            "source_script": str(PLACEBO_SCRIPT.relative_to(ROOT)),
            "feature_normalization": common_normalization,
            "feature_layout": "Concatenate cached image[512] then text[512] features",
            "head_architecture": (
                "Linear(1024,512) -> ReLU -> Dropout(0.1) -> Linear(512,2)"
            ),
            "optimizer": "Adam (default weight decay 0)",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 256,
            "epochs": 40,
            "class_weighting": "None for Raw Dual",
            "seed_list": [0, 1, 2, 3, 4],
            "seed_mapping": (
                "Raw ensemble uses first five seeds from raw30: "
                "torch.manual_seed(seed), np.random.seed(seed)"
            ),
            "threshold_rule": "probability > 0.50",
            "ensemble": (
                "Mean predicted probability over raw seeds 0-4; these are the "
                "first five of the separate 30-seed raw variability run"
            ),
            "split_file": "cached_features.pt",
            "split_sizes": {"train": 7000, "val": 1000, "test": 2000},
            "model_selection": (
                "Raw baseline fixed; fairness settings were selected on validation"
            ),
            "numpy_seed": "np.random.seed(seed) per raw-head seed",
        },
    }


def prediction_vectors_present(*payloads: dict[str, Any]) -> list[str]:
    """Find probability vectors explicitly stored in result JSONs, if any."""
    matches: list[str] = []
    expected_length = 2000
    interesting = ("pred", "prob", "score")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            if (
                len(value) == expected_length
                and all(isinstance(item, (int, float)) for item in value)
                and any(token in path.lower() for token in interesting)
            ):
                matches.append(path)

    for payload in payloads:
        visit(payload, "")
    return matches


def compare_stored_metrics(
    unified: dict[str, Any],
    placebo: dict[str, Any],
) -> dict[str, Any]:
    old = unified["table1_metrics"]["Raw Dual (Image+Text)"]
    later = placebo["table1"]["ensemble"]["raw"]
    old_black_tpr = float(old["tprs"][1])
    later_black_tpr = read_later_black_tpr()

    return {
        "old": {
            "source": "unified_final_results.json:table1_metrics.Raw Dual (Image+Text)",
            "auc": float(old["auc"]),
            "eod_max": float(old["eod_max"]),
            "black_tpr": old_black_tpr,
        },
        "later": {
            "source": "placebo_audit_results.json:table1.ensemble.raw",
            "auc": float(later["auc"]),
            "eod_max": float(later["eod_max"]),
            "black_tpr": later_black_tpr,
        },
        "differences_later_minus_old": {
            "auc": float(later["auc"]) - float(old["auc"]),
            "eod_max": float(later["eod_max"]) - float(old["eod_max"]),
            "black_tpr": later_black_tpr - old_black_tpr,
        },
        "interpretation": {
            "both_use_fixed_050_operating_point": True,
            "auc_depends_on_cutoff": False,
            "training_configuration_changed": True,
            "same_predictions_replay_possible": False,
        },
    }


def read_later_black_tpr() -> float:
    """Read the recorded fixed-cutoff raw ensemble subgroup line."""
    text = PLACEBO_LOG.read_text(encoding="utf-8")
    start = text.find("[raw]")
    if start < 0:
        raise ValueError("Could not locate the raw per-group section in placebo log.")
    for line in text[start:].splitlines()[1:]:
        if line.strip().startswith("["):
            break
        match = re.match(r"\s*Black\s+n=\s*\d+.*?\bTPR\s+([0-9.]+)", line)
        if match:
            return float(match.group(1))
    raise ValueError("Could not read the Black TPR from the placebo audit log.")


def metric_report_rows(
    model_names: list[str],
    checkpoint: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    for model_name in model_names:
        values = checkpoint["canonical_models"][model_name]
        fields = (
            f"AUC={values.get('auc', 'n/a')}",
            f"accuracy={values.get('acc', 'n/a')}",
            f"DPD={values.get('dpd', 'n/a')}",
            f"EOD_max={values.get('eod_max', 'n/a')}",
        )
        lines.append(f"| {model_name} | " + " | ".join(fields) + " |")
    return lines


def build_report(
    configs: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    sanity: dict[str, Any],
    replay_paths: list[str],
    canonical_model_names: list[str],
    checkpoint: dict[str, Any],
) -> str:
    old, later = configs["unified_final"], configs["placebo_audit"]
    comparison_lines: list[str] = []
    field_explanations = {
        "feature_normalization": "Same cached normalization; not a cause.",
        "feature_layout": "Same joint feature order/dimension; not a cause.",
        "head_architecture": "Major training change; changes learned scores and subgroup errors.",
        "optimizer": "Training change; contributes to a non-identical fit.",
        "learning_rate": "Same; not a cause by itself.",
        "weight_decay": "Training change; may affect fit.",
        "batch_size": "Training change; affects updates and batch statistics.",
        "epochs": "Training change; affects fitted weights and probability calibration.",
        "class_weighting": "Same for raw baseline; not a cause.",
        "seed_list": "Different initialization/data-order streams; changes fitted heads.",
        "seed_mapping": "Different RNG streams; can alter model weights and minibatch order.",
        "threshold_rule": (
            "Both use nominal 0.50. `>=` versus `>` only differs for exact ties; "
            "cannot explain AUC."
        ),
        "ensemble": "Both average five heads, but they are not the same five trained heads.",
        "split_file": "Both runs use cached_features.pt; not a cause.",
        "split_sizes": "Same official split sizes; not a cause.",
        "model_selection": "Raw model has fixed settings; no test-based tuning found.",
        "numpy_seed": "Later run explicitly seeds NumPy; older raw loop does not.",
    }
    for field in old:
        if field == "source_script":
            continue
        old_value, later_value = old[field], later[field]
        if old_value == later_value:
            result = "Same"
        else:
            result = "Different"
        comparison_lines.append(
            f"| `{field}` | {old_value} | {later_value} | "
            f"{result}. {field_explanations[field]} |"
        )

    old_metrics = comparison["old"]
    later_metrics = comparison["later"]
    deltas = comparison["differences_later_minus_old"]
    replay_status = (
        "Detected at " + ", ".join(replay_paths)
        if replay_paths
        else (
            "Unavailable. Neither result JSON contains a 2,000-element test "
            "probability vector; the logs contain aggregate metrics only."
        )
    )
    if replay_paths:
        replay_conclusion = (
            "Exact re-scoring was not performed: the stored vectors do not have "
            "an established schema linking scores to labels and run identity."
        )
    else:
        replay_conclusion = (
            "Exact re-scoring of the identical predictions cannot be performed "
            "from the saved artifacts. The stored benchmark metrics are already "
            "reported at the nominal 0.50 cutoff."
        )

    model_table = metric_report_rows(canonical_model_names, checkpoint)
    if not model_table:
        model_table = ["| | | | | |"]

    return f"""# Run reconciliation report

## Conclusion

The reported raw Dual results differ because these files summarize **different
trained models**, not because one run used a different AUC cutoff. The older
run reports AUC **{old_metrics['auc']:.4f}**, EOD_max **{old_metrics['eod_max']:.4f}**,
and Black TPR **{old_metrics['black_tpr']:.3f}**. The later placebo-audit
five-seed ensemble reports AUC **{later_metrics['auc']:.4f}**, EOD_max
**{later_metrics['eod_max']:.4f}**, and Black TPR **{later_metrics['black_tpr']:.3f}**.
Later-minus-older differences are AUC **{deltas['auc']:+.4f}**, EOD_max
**{deltas['eod_max']:+.4f}**, and Black TPR **{deltas['black_tpr']:+.3f}**.

The largest confirmed differences are the diagnostic head (256 hidden with
BatchNorm and 0.2 dropout versus 512 hidden without BatchNorm and 0.1 dropout),
optimizer (AdamW versus Adam), epoch count (30 versus 40), batch size (128
versus 256), weight decay, and seed values. These change the fitted weights,
ranking, and thresholded subgroup decisions. The saved summaries do not permit
isolating one of these changes as the sole cause.

Both stored raw-run summaries use a nominal **0.50** cutoff. The older code uses
`>= 0.50`, while the later code uses `> 0.50`. This can affect only examples
whose score is exactly 0.50 and can alter thresholded rates; it cannot change
AUC. AUC changed by {deltas['auc']:+.4f}, so a cutoff convention alone is ruled
out. No separate post-hoc calibration step is configured; a monotonic
calibration alone would not change AUC either.

## Configuration differences

Settings are read from the producer scripts; the actual later cache path and
QUICK mode are checked against `placebo_audit_output.txt`.

| Field | Older unified benchmark | Later placebo audit | Difference and likely relevance |
|---|---|---|---|
{chr(10).join(comparison_lines)}

## Same-prediction 0.50 replay

**Prediction-vector availability:** {replay_status}

{replay_conclusion} No model was retrained here to stand in for the missing
predictions. The existing outputs already record raw metrics at 0.50, and
those recorded fixed-cutoff metrics show the gap. A strictly paired replay of
the *same* score arrays, including a test for exact 0.50 ties, requires the
original saved per-patient test probabilities, which neither JSON contains.

The older per-group raw Black TPR is stored directly in its result JSON. The
later Black TPR is read from the placebo run's fixed-cutoff subgroup log.
These are not the same prediction vector: the raw-head configurations and
seed streams differ.

## Split and leakage checks

- Train/validation/test sizes: {sanity['split_sizes']}.
- Test race counts (Asian/Black/White): {sanity['test_race_counts']}.
- The test split is used only for reported evaluation; tuning is recorded as
  validation-only: **{sanity['test_used_for_selection']}** test-based selection.
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
{chr(10).join(model_table)}
"""


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume assembling canonical model results from the per-model checkpoint.",
    )
    args = parser.parse_args()
    set_seeds()

    unified = load_json(UNIFIED_RESULTS)
    placebo = load_json(PLACEBO_RESULTS)
    sanity = verify_split_and_cohort_sanity(unified, placebo)
    configs = config_snapshots()
    comparison = compare_stored_metrics(unified, placebo)
    replay_paths = prediction_vectors_present(unified, placebo)
    fingerprints = input_fingerprints()

    if args.resume and CHECKPOINT_FILE.is_file():
        checkpoint = load_json(CHECKPOINT_FILE)
        if checkpoint.get("input_fingerprints") != fingerprints:
            raise ValueError(
                "Cannot resume: source result/script/cache files changed since "
                "the checkpoint. Run without --resume to start a fresh reconciliation."
            )
        if not isinstance(checkpoint.get("canonical_models"), dict):
            raise ValueError("Invalid reconciliation checkpoint: missing model results.")
        print("Resuming canonical result assembly from checkpoint.")
    else:
        checkpoint = {
            "input_fingerprints": fingerprints,
            "canonical_models": {},
        }

    canonical_models = placebo["table1"]["ensemble"]
    if not isinstance(canonical_models, dict) or not canonical_models:
        raise ValueError("Placebo audit is missing ensemble model metrics.")
    for model_name, model_metrics in canonical_models.items():
        if model_name not in checkpoint["canonical_models"]:
            checkpoint["canonical_models"][model_name] = model_metrics
            save_checkpoint(CHECKPOINT_FILE, checkpoint)
            print(f"Checkpointed canonical model: {model_name}")

    canonical = {
        "canonical_source": "placebo_audit_results.json",
        "canonical_section": "table1.ensemble",
        "canonical_run": "placebo_audit",
        "selection_rationale": (
            "Later run with placebo audit, selected as the final recorded run "
            "without combining metrics from the older benchmark."
        ),
        "configuration": configs["placebo_audit"],
        "split_sanity": sanity,
        "raw_black_tpr_from_run_log": comparison["later"]["black_tpr"],
        "ensemble_metrics": checkpoint["canonical_models"],
        "per_seed_summary": placebo["table1"].get("per_seed", {}),
        "raw_seed_variability": {
            key: values
            for key, values in placebo.get("partA", {}).items()
            if key in ("auc", "acc", "dpd", "eod_max")
        },
        "reconciliation": {
            "comparison": comparison,
            "same_prediction_vectors_saved": bool(replay_paths),
            "exact_same_prediction_threshold_replay_available": False,
            "prediction_vector_locations": replay_paths,
        },
    }

    model_names = list(checkpoint["canonical_models"])
    report = build_report(
        configs,
        comparison,
        sanity,
        replay_paths,
        model_names,
        checkpoint,
    )
    REPORT_FILE.write_text(report, encoding="utf-8")
    CANONICAL_FILE.write_text(
        json.dumps(canonical, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    old = comparison["old"]
    later = comparison["later"]
    print("\nRaw Dual reconciliation (stored fixed 0.50 metrics)")
    print(
        f"Older:  AUC={old['auc']:.4f}, EOD_max={old['eod_max']:.4f}, "
        f"Black TPR={old['black_tpr']:.3f}"
    )
    print(
        f"Later:  AUC={later['auc']:.4f}, EOD_max={later['eod_max']:.4f}, "
        f"Black TPR={later['black_tpr']:.3f}"
    )
    print(
        "Exact same-score replay: "
        + ("available" if replay_paths else "not possible; prediction arrays were not saved")
    )
    print(f"Saved {REPORT_FILE} and {CANONICAL_FILE}.")


if __name__ == "__main__":
    main()
