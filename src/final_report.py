"""Generate the paired test-set report for OpenCLIP and the selected stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cached_features.pt"
DEFAULT_STACK_RESULTS = ROOT / "stack_results.json"
DEFAULT_OUTPUT_MD = ROOT / "final_report.md"
DEFAULT_OUTPUT_TEX = ROOT / "tables" / "final_report.tex"
EXPECTED_SIZES = {"train": 7000, "val": 1000, "test": 2000}
EXPECTED_TEST_RACE_COUNTS = (169, 302, 1529)
RACE_NAMES = ("Asian", "Black", "White")
MODEL_NAMES = ("openclip_joint", "stack")
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
DELTA_METRICS = ("auc", "accuracy", "dpd", "eod_max")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval counts must satisfy 0 <= successes <= total.")
    rate = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = (rate + z_squared / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(rate * (1 - rate) / total + z_squared / (4 * total * total))
        / denominator
    )
    return (
        float(max(0.0, center - margin)),
        float(min(1.0, center + margin)),
    )


def validate_vectors(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> None:
    if labels.ndim != 1 or races.shape != labels.shape:
        raise ValueError("Test labels and races must be aligned one-dimensional arrays.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Test labels must include both glaucoma classes.")
    if set(np.unique(races)) != {0, 1, 2}:
        raise ValueError("Test races must include Asian, Black, and White.")
    for model_name, values in probabilities.items():
        if values.shape != labels.shape:
            raise ValueError(
                f"{model_name} test probabilities have shape {values.shape}; "
                f"expected {labels.shape}."
            )
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{model_name} probabilities must be finite values in [0, 1].")


def binary_metrics(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    positive_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []
    for race in range(len(RACE_NAMES)):
        group = races == race
        positive = group & (labels == 1)
        negative = group & (labels == 0)
        if not positive.any() or not negative.any():
            raise ValueError(
                f"Test sample lacks a glaucoma outcome class for {RACE_NAMES[race]}."
            )
        positive_rates.append(float(predictions[group].mean()))
        tprs.append(float(predictions[positive].mean()))
        fprs.append(float(predictions[negative].mean()))
    tpr_gap = max(tprs) - min(tprs)
    fpr_gap = max(fprs) - min(fprs)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "dpd": float(max(positive_rates) - min(positive_rates)),
        "eod_max": float(max(tpr_gap, fpr_gap)),
        "tpr_gap": float(tpr_gap),
        "fpr_gap": float(fpr_gap),
    }


def per_race_rates(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    predictions = probabilities >= threshold
    report: dict[str, dict[str, Any]] = {}
    for race, name in enumerate(RACE_NAMES):
        group = races == race
        positive = group & (labels == 1)
        negative = group & (labels == 0)
        positive_n = int(positive.sum())
        negative_n = int(negative.sum())
        if positive_n == 0 or negative_n == 0:
            raise ValueError(f"Cannot compute Wilson rates for {name}: empty outcome cell.")
        true_positive = int(np.count_nonzero(predictions & positive))
        false_negative = int(np.count_nonzero(~predictions & positive))
        false_positive = int(np.count_nonzero(predictions & negative))
        report[name] = {
            "n": int(group.sum()),
            "positive_n": positive_n,
            "negative_n": negative_n,
            "tpr": true_positive / positive_n,
            "tpr_wilson_95_ci": list(wilson_interval(true_positive, positive_n)),
            "fnr": false_negative / positive_n,
            "fnr_wilson_95_ci": list(wilson_interval(false_negative, positive_n)),
            "fpr": false_positive / negative_n,
            "fpr_wilson_95_ci": list(wilson_interval(false_positive, negative_n)),
        }
    return report


def paired_bootstrap_deltas(
    labels: np.ndarray,
    races: np.ndarray,
    baseline: np.ndarray,
    stack: np.ndarray,
    threshold: float,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive.")
    point_baseline = binary_metrics(labels, races, baseline, threshold)
    point_stack = binary_metrics(labels, races, stack, threshold)
    point_deltas = {
        "auc": point_stack["auc"] - point_baseline["auc"],
        "accuracy": point_stack["accuracy"] - point_baseline["accuracy"],
        "dpd": point_stack["dpd"] - point_baseline["dpd"],
        "eod_max": point_stack["eod_max"] - point_baseline["eod_max"],
    }

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in DELTA_METRICS}
    attempts = 0
    max_attempts = resamples * 10
    while min(map(len, samples.values())) < resamples and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        bootstrap_labels = labels[indices]
        bootstrap_races = races[indices]
        if len(np.unique(bootstrap_labels)) < 2:
            continue
        try:
            baseline_metrics = binary_metrics(
                bootstrap_labels,
                bootstrap_races,
                baseline[indices],
                threshold,
            )
            stack_metrics = binary_metrics(
                bootstrap_labels,
                bootstrap_races,
                stack[indices],
                threshold,
            )
        except ValueError:
            continue
        for metric in DELTA_METRICS:
            samples[metric].append(stack_metrics[metric] - baseline_metrics[metric])
    if any(len(values) != resamples for values in samples.values()):
        raise RuntimeError(
            f"Only obtained {min(map(len, samples.values()))}/{resamples} valid "
            "paired bootstrap resamples."
        )
    return {
        metric: {
            "delta_stack_minus_openclip": point_deltas[metric],
            "percentile_2_5": float(np.percentile(samples[metric], 2.5)),
            "percentile_97_5": float(np.percentile(samples[metric], 97.5)),
            "resamples": resamples,
            "seed": seed,
        }
        for metric in DELTA_METRICS
    }


def load_inputs(
    cache_path: Path,
    results_path: Path,
    probability_archive_override: Path | None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, np.ndarray]]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Stack results not found: {results_path}. Run src/stack_models.py first."
        )
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    sizes = {split: len(cached[split]["labels"]) for split in EXPECTED_SIZES}
    if sizes != EXPECTED_SIZES:
        raise ValueError(f"Expected seeded 70/10/20 split sizes {EXPECTED_SIZES}, got {sizes}.")
    labels = cached["test"]["labels"].detach().cpu().long().numpy().reshape(-1)
    races = cached["test"]["races"].detach().cpu().long().numpy().reshape(-1)
    race_counts = tuple(int(np.count_nonzero(races == race)) for race in range(3))
    if race_counts != EXPECTED_TEST_RACE_COUNTS:
        raise ValueError(
            f"Unexpected test race counts {race_counts}; expected "
            f"{EXPECTED_TEST_RACE_COUNTS} for the configured split."
        )

    stack_results = json.loads(results_path.read_text(encoding="utf-8"))
    try:
        threshold = float(
            stack_results["base_models"]["stack"]["validation_threshold"]["value"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "stack_results.json lacks the stack validation-chosen threshold."
        ) from error
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid validation-chosen threshold: {threshold}")

    archive_path = probability_archive_override
    if archive_path is None:
        archive_name = (
            stack_results.get("config", {}).get("base_test_probabilities_file")
        )
        if not archive_name:
            raise ValueError(
                "stack_results.json does not specify the base test probability "
                "archive; regenerate it with the updated src/stack_models.py."
            )
        archive_path = results_path.parent / archive_name
    if not archive_path.is_file():
        raise FileNotFoundError(f"Base test probability archive not found: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = set(MODEL_NAMES).difference(archive.files)
        if missing:
            raise ValueError(
                f"{archive_path.name} is missing model probabilities: {sorted(missing)}"
            )
        probabilities = {
            name: np.asarray(archive[name], dtype=np.float64).reshape(-1)
            for name in MODEL_NAMES
        }
    stack_probability_path = (
        results_path.parent
        / stack_results.get("config", {}).get("test_probabilities_file", "stack_test_probs.npy")
    )
    if stack_probability_path.is_file():
        separate_stack = np.load(stack_probability_path, allow_pickle=False).reshape(-1)
        if not np.allclose(separate_stack, probabilities["stack"], atol=1e-6):
            raise ValueError(
                "Stack probabilities in the npy and npz outputs do not match."
            )
    validate_vectors(labels, races, probabilities)
    return labels, races, threshold, probabilities


def render_markdown(
    results: dict[str, Any],
    bootstrap: dict[str, dict[str, float]],
    group_rates: dict[str, dict[str, dict[str, Any]]],
    threshold: float,
    n_test: int,
) -> str:
    model_names = {
        "openclip_joint": "OpenCLIP img+txt MLP",
        "stack": "Best stack (validation-selected)",
    }
    lines = [
        "# Final Test-Set Report",
        "",
        "The evaluation uses a **seeded random 70/10/20 split** of the 10,000-row",
        "FairVLMed train dataset (seed 42; 7,000 train / 1,000 validation /",
        "2,000 test). This is not the official folder split.",
        "",
        f"The stack decision threshold was selected on validation by Youden's J: "
        f"`{threshold:.6f}`. The same threshold is used for both models' accuracy",
        "and per-race rates. No test-set tuning was performed.",
        "",
        "## Overall test metrics",
        "",
        "| Model | AUC | Accuracy at 0.50 | Accuracy at validation threshold | DPD | EOD_max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in MODEL_NAMES:
        metrics = results[key]
        lines.append(
            f"| {model_names[key]} | {metrics['auc']:.4f} | "
            f"{metrics['accuracy_at_0_5']:.4f} | "
            f"{metrics['accuracy_at_val_threshold']:.4f} | "
            f"{metrics['dpd']:.4f} | {metrics['eod_max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap deltas",
            "",
            f"Each resample draws {n_test} test cases with replacement and applies",
            "the identical indices to both models. Delta is stack minus OpenCLIP.",
            "Intervals are 95% percentile CIs from 1,000 resamples (seed 0).",
            "",
            "| Metric | Delta | 95% percentile CI |",
            "|---|---:|---:|",
        ]
    )
    for metric in DELTA_METRICS:
        item = bootstrap[metric]
        lines.append(
            f"| {metric} | {item['delta_stack_minus_openclip']:+.4f} | "
            f"[{item['percentile_2_5']:+.4f}, {item['percentile_97_5']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Per-race test rates",
            "",
            "Rates use the validation-chosen threshold. CIs are Wilson 95% intervals.",
            "",
        ]
    )
    for key in MODEL_NAMES:
        lines.extend(
            [
                f"### {model_names[key]}",
                "",
                "| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name in RACE_NAMES:
            group = group_rates[key][name]
            lines.append(
                f"| {name} | {group['n']} | "
                f"{group['tpr']:.4f} "
                f"[{group['tpr_wilson_95_ci'][0]:.4f}, "
                f"{group['tpr_wilson_95_ci'][1]:.4f}] | "
                f"{group['fnr']:.4f} "
                f"[{group['fnr_wilson_95_ci'][0]:.4f}, "
                f"{group['fnr_wilson_95_ci'][1]:.4f}] | "
                f"{group['fpr']:.4f} "
                f"[{group['fpr_wilson_95_ci'][0]:.4f}, "
                f"{group['fpr_wilson_95_ci'][1]:.4f}] |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for char, replacement in replacements.items():
        value = value.replace(char, replacement)
    return value


def render_latex(
    results: dict[str, Any],
    bootstrap: dict[str, dict[str, float]],
    group_rates: dict[str, dict[str, dict[str, Any]]],
    threshold: float,
) -> str:
    model_names = {
        "openclip_joint": "OpenCLIP img+txt MLP",
        "stack": "Best stack",
    }
    lines = [
        "% Generated by src/final_report.py; do not edit by hand.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Test-set performance on the seeded random 70/10/20 split.}",
        r"\label{tab:final-test-performance}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & AUC & Acc.\@0.50 & Acc.\@val. thr. & DPD & EOD$_{\max}$ \\",
        r"\midrule",
    ]
    for key in MODEL_NAMES:
        metrics = results[key]
        lines.append(
            f"{model_names[key]} & {metrics['auc']:.4f} & "
            f"{metrics['accuracy_at_0_5']:.4f} & "
            f"{metrics['accuracy_at_val_threshold']:.4f} & "
            f"{metrics['dpd']:.4f} & {metrics['eod_max']:.4f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{6}{l}{Paired bootstrap deltas (stack $-$ OpenCLIP; "
            r"1,000 resamples, seed 0)} \\",
            r"Metric & \multicolumn{1}{c}{Delta} & "
            r"\multicolumn{4}{c}{95\% percentile CI} \\",
            r"\midrule",
        ]
    )
    for metric in DELTA_METRICS:
        item = bootstrap[metric]
        lines.append(
            f"{latex_escape(metric)} & "
            f"{item['delta_stack_minus_openclip']:+.4f} & "
            f"\\multicolumn{{4}}{{c}}{{"
            f"[{item['percentile_2_5']:+.4f}, "
            f"{item['percentile_97_5']:+.4f}]}} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\vspace{{0.5em}}\\par\\footnotesize Validation threshold "
            f"(Youden's $J$): {threshold:.6f}. "
            r"Per-race TPR/FNR/FPR and Wilson 95\% CIs are reported in "
            r"Table~\ref{tab:final-test-rates}.",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Per-race test rates at the validation-chosen threshold.}",
            r"\label{tab:final-test-rates}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Model & Race & $n$ & TPR (95\% CI) & FNR (95\% CI) & FPR (95\% CI) \\",
            r"\midrule",
        ]
    )
    for key in MODEL_NAMES:
        for name in RACE_NAMES:
            group = group_rates[key][name]
            lines.append(
                f"{model_names[key]} & {name} & {group['n']} & "
                f"{group['tpr']:.4f} "
                f"[{group['tpr_wilson_95_ci'][0]:.4f}, "
                f"{group['tpr_wilson_95_ci'][1]:.4f}] & "
                f"{group['fnr']:.4f} "
                f"[{group['fnr_wilson_95_ci'][0]:.4f}, "
                f"{group['fnr_wilson_95_ci'][1]:.4f}] & "
                f"{group['fpr']:.4f} "
                f"[{group['fpr_wilson_95_ci'][0]:.4f}, "
                f"{group['fpr_wilson_95_ci'][1]:.4f}] \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--stack-results", type=Path, default=DEFAULT_STACK_RESULTS)
    parser.add_argument("--probability-archive", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
    )
    args = parser.parse_args()

    labels, races, threshold, probabilities = load_inputs(
        args.cache,
        args.stack_results,
        args.probability_archive,
    )
    model_metrics = {
        name: binary_metrics(labels, races, probabilities[name], threshold)
        for name in MODEL_NAMES
    }
    results = {
        name: {
            **metrics,
            "accuracy_at_0_5": float(
                accuracy_score(labels, probabilities[name] >= 0.5)
            ),
            "accuracy_at_val_threshold": metrics["accuracy"],
        }
        for name, metrics in model_metrics.items()
    }
    bootstrap = paired_bootstrap_deltas(
        labels,
        races,
        probabilities["openclip_joint"],
        probabilities["stack"],
        threshold,
        resamples=args.bootstrap_resamples,
        seed=BOOTSTRAP_SEED,
    )
    group_rates = {
        name: per_race_rates(labels, races, probabilities[name], threshold)
        for name in MODEL_NAMES
    }

    markdown = render_markdown(
        results,
        bootstrap,
        group_rates,
        threshold,
        len(labels),
    )
    latex = render_latex(results, bootstrap, group_rates, threshold)
    atomic_write_text(args.output_md, markdown)
    atomic_write_text(args.output_tex, latex)
    print(f"Validation-selected threshold: {threshold:.6f}")
    print(
        "Seeded random 70/10/20 split; paired test bootstrap "
        f"n={args.bootstrap_resamples}, seed={BOOTSTRAP_SEED}."
    )
    for metric, item in bootstrap.items():
        print(
            f"d{metric} (stack - OpenCLIP)={item['delta_stack_minus_openclip']:+.4f} "
            f"[{item['percentile_2_5']:+.4f}, {item['percentile_97_5']:+.4f}]"
        )
    print(
        f"Wrote {args.output_md} and {args.output_tex}; "
        f"stack metadata loaded from {args.stack_results.name}."
    )


if __name__ == "__main__":
    main()
