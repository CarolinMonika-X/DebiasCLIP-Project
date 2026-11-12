"""Paired seed-group audit of Raw Dual versus train-only LEACE-Race."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cached_features.pt"
DEFAULT_JSON = ROOT / "seed_paired_results.json"
DEFAULT_TEX = ROOT / "tables" / "seed_paired.tex"
DEFAULT_CHECKPOINT = ROOT / ".seed_paired_checkpoint.pt"
SPLITS = ("train", "val", "test")
RACE_NAMES = ("Asian", "Black", "White")
EXPECTED_SPLIT_SIZES = {"train": 7000, "val": 1000, "test": 2000}
EXPECTED_TEST_RACE_COUNTS = (169, 302, 1529)
SEED_GROUPS = tuple(tuple(range(start, start + 5)) for start in range(0, 30, 5))

HIDDEN_DIM = 256
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.50
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261111
FAMILY_ENDPOINTS = ("d_auc", "d_eod_max", "d_dpd", "d_worst_group_auc")


class DiagnosticHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class WeightedAUC:
    """Fast weighted AUC for bootstrap multiplicities, with tie correction."""

    def __init__(self, labels: np.ndarray, scores: np.ndarray) -> None:
        self.labels = labels.astype(np.int8, copy=False)
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        self.order = order
        self.block_ids = np.cumsum(
            np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
        ) - 1
        self.block_count = int(self.block_ids[-1]) + 1
        self.positive = self.labels[order] == 1

    def __call__(self, multiplicities: np.ndarray) -> float:
        sorted_weights = multiplicities[self.order].astype(np.float64, copy=False)
        positive_weights = np.bincount(
            self.block_ids,
            weights=sorted_weights * self.positive,
            minlength=self.block_count,
        )
        negative_weights = np.bincount(
            self.block_ids,
            weights=sorted_weights * ~self.positive,
            minlength=self.block_count,
        )
        positive_total = positive_weights.sum()
        negative_total = negative_weights.sum()
        if positive_total == 0 or negative_total == 0:
            raise ValueError("A bootstrap test sample omitted a label class.")
        negative_before = np.cumsum(negative_weights) - negative_weights
        concordant = np.sum(
            positive_weights * (negative_before + 0.5 * negative_weights)
        )
        return float(concordant / (positive_total * negative_total))


class BootstrapMetricCalculator:
    def __init__(
        self, labels: np.ndarray, races: np.ndarray, probabilities: np.ndarray
    ) -> None:
        self.labels = labels
        self.races = races
        self.predictions = probabilities >= THRESHOLD
        self.overall_auc = WeightedAUC(labels, probabilities)
        self.groups: list[tuple[np.ndarray, np.ndarray, WeightedAUC]] = []
        for race in range(len(RACE_NAMES)):
            mask = races == race
            self.groups.append(
                (mask, labels[mask], WeightedAUC(labels[mask], probabilities[mask]))
            )

    def calculate(self, multiplicities: np.ndarray) -> dict[str, float]:
        weights = multiplicities.astype(np.float64, copy=False)
        predictions = self.predictions
        total = weights.sum()
        accuracy = np.sum(weights * (predictions == self.labels)) / total
        positive_rates: list[float] = []
        tprs: list[float] = []
        fprs: list[float] = []
        group_aucs: list[float] = []
        for mask, group_labels, auc_calculator in self.groups:
            group_weights = weights[mask]
            group_predictions = predictions[mask]
            positive_denominator = np.sum(group_weights * (group_labels == 1))
            negative_denominator = np.sum(group_weights * (group_labels == 0))
            population_denominator = group_weights.sum()
            if min(positive_denominator, negative_denominator, population_denominator) <= 0:
                raise ValueError("A bootstrap sample omitted a race or outcome class.")
            positive_rates.append(
                float(np.sum(group_weights * group_predictions) / population_denominator)
            )
            tprs.append(
                float(
                    np.sum(group_weights * group_predictions * (group_labels == 1))
                    / positive_denominator
                )
            )
            fprs.append(
                float(
                    np.sum(group_weights * group_predictions * (group_labels == 0))
                    / negative_denominator
                )
            )
            group_aucs.append(auc_calculator(group_weights))

        tpr_gap = max(tprs) - min(tprs)
        fpr_gap = max(fprs) - min(fprs)
        return {
            "auc": self.overall_auc(weights),
            "accuracy": float(accuracy),
            "dpd": float(max(positive_rates) - min(positive_rates)),
            "eod_max": float(max(tpr_gap, fpr_gap)),
            "worst_group_auc": float(min(group_aucs)),
        }


def parse_groups(value: str) -> list[int]:
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise argparse.ArgumentTypeError(
                "Groups must be comma-separated indexes or ranges, e.g. 0,2-3."
            )
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            raise argparse.ArgumentTypeError(f"Invalid descending group range: {token}")
        selected.update(range(start, end + 1))
    if not selected or min(selected) < 0 or max(selected) >= len(SEED_GROUPS):
        raise argparse.ArgumentTypeError("Group indexes must be from 0 through 5.")
    return sorted(selected)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_features(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise ValueError("Feature cache must contain split dictionaries.")

    features: dict[str, dict[str, torch.Tensor]] = {}
    for split in SPLITS:
        if split not in cache:
            raise ValueError(f"Feature cache is missing split {split!r}.")
        source = cache[split]
        required = {"img", "txt", "labels", "races"}
        if not required.issubset(source):
            raise ValueError(f"{split} cache must contain {sorted(required)}.")
        image = source["img"].float()
        text = source["txt"].float()
        labels = source["labels"].long()
        races = source["races"].long()
        if image.ndim != 2 or image.shape[1] != 512:
            raise ValueError(f"Expected 512-D image features in {split}.")
        if text.ndim != 2 or text.shape[1] != 512:
            raise ValueError(f"Expected 512-D text features in {split}.")
        if len(image) != len(text) or len(image) != len(labels) or len(image) != len(races):
            raise ValueError(f"Feature and label lengths disagree in {split}.")
        features[split] = {
            "joint": torch.cat((image, text), dim=1).contiguous(),
            "labels": labels,
            "races": races,
        }

    sizes = {split: len(features[split]["labels"]) for split in SPLITS}
    assert sizes == EXPECTED_SPLIT_SIZES, sizes
    test_races = features["test"]["races"].numpy()
    test_counts = tuple(
        int(np.count_nonzero(test_races == race)) for race in range(len(RACE_NAMES))
    )
    assert test_counts == EXPECTED_TEST_RACE_COUNTS, test_counts
    for split in SPLITS:
        assert set(torch.unique(features[split]["labels"]).tolist()) == {0, 1}
        assert set(torch.unique(features[split]["races"]).tolist()) == {0, 1, 2}
    return features


def fit_leace_race(
    train_features: np.ndarray, train_races: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the LEACE affine transform using train features and race labels only."""
    mean = train_features.mean(axis=0)
    centered = train_features - mean
    race_one_hot = np.eye(len(RACE_NAMES), dtype=np.float64)[train_races]
    centered_race = race_one_hot - race_one_hot.mean(axis=0)
    covariance_xx = centered.T @ centered / len(centered)
    covariance_xz = centered.T @ centered_race / len(centered)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_xx)
    eigenvalues = np.clip(eigenvalues, 1e-5 * eigenvalues.max(), None)
    whitening = (eigenvectors * eigenvalues**-0.5) @ eigenvectors.T
    dewhitening = (eigenvectors * eigenvalues**0.5) @ eigenvectors.T
    left_vectors, singular_values, _ = np.linalg.svd(
        whitening @ covariance_xz, full_matrices=False
    )
    keep = singular_values > 1e-8 * singular_values.max()
    subspace = left_vectors[:, keep]
    eraser = dewhitening @ (subspace @ subspace.T) @ whitening
    return mean.astype(np.float32), eraser.astype(np.float32)


def apply_leace(
    features: np.ndarray, mean: np.ndarray, eraser: np.ndarray
) -> np.ndarray:
    return (features - (features - mean) @ eraser.T).astype(np.float32)


def train_predict(
    train_features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    model = DiagnosticHead(train_features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    train_x = torch.from_numpy(train_features).float()
    train_y = torch.from_numpy(labels).long()
    n_train = len(train_y)

    for _ in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            indexes = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_x[indexes]), train_y[indexes])
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model.eval()


def predict(model: DiagnosticHead, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(features).float())
        return torch.softmax(logits, dim=1)[:, 1].numpy().astype(np.float64)


def binary_metrics(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    multiplicities: np.ndarray | None = None,
) -> dict[str, float]:
    if multiplicities is None:
        multiplicities = np.ones(len(labels), dtype=np.int64)
    return BootstrapMetricCalculator(labels, races, probabilities).calculate(
        multiplicities
    )


def summarize(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "sign_count": {
            "positive": int(np.count_nonzero(array > 0)),
            "negative": int(np.count_nonzero(array < 0)),
            "zero": int(np.count_nonzero(array == 0)),
        },
    }


def bootstrap_intervals(
    group_predictions: list[dict[str, np.ndarray]],
    labels: np.ndarray,
    races: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_count = len(group_predictions)
    test_count = len(labels)
    distributions = {key: np.empty(draws, dtype=np.float64) for key in FAMILY_ENDPOINTS}
    metric_calculators = [
        {
            method: BootstrapMetricCalculator(labels, races, predictions[method])
            for method in ("raw", "leace")
        }
        for predictions in group_predictions
    ]

    for draw in range(draws):
        sampled_groups = rng.integers(0, group_count, size=group_count)
        paired_deltas: dict[str, list[float]] = {
            key: [] for key in FAMILY_ENDPOINTS
        }
        for group_index in sampled_groups:
            case_indices = rng.integers(0, test_count, size=test_count)
            multiplicities = np.bincount(case_indices, minlength=test_count)
            calculators = metric_calculators[int(group_index)]
            raw_metrics = calculators["raw"].calculate(multiplicities)
            leace_metrics = calculators["leace"].calculate(multiplicities)
            for key, metric in (
                ("d_auc", "auc"),
                ("d_eod_max", "eod_max"),
                ("d_dpd", "dpd"),
                ("d_worst_group_auc", "worst_group_auc"),
            ):
                paired_deltas[key].append(leace_metrics[metric] - raw_metrics[metric])
        for key in FAMILY_ENDPOINTS:
            distributions[key][draw] = float(np.mean(paired_deltas[key]))

    bonferroni_alpha = 0.05 / len(FAMILY_ENDPOINTS)
    intervals: dict[str, Any] = {}
    for key, distribution in distributions.items():
        intervals[key] = {
            "bootstrap_mean": float(distribution.mean()),
            "ci_95_percentile": [
                float(np.quantile(distribution, 0.025)),
                float(np.quantile(distribution, 0.975)),
            ],
            "bonferroni_adjusted_ci": [
                float(np.quantile(distribution, bonferroni_alpha / 2)),
                float(np.quantile(distribution, 1 - bonferroni_alpha / 2)),
            ],
        }
    return {
        "draws": draws,
        "seed": seed,
        "resampling": (
            "For each draw, sample seed groups with replacement; for each "
            "selected group occurrence, sample test cases with replacement. "
            "The same case multiplicities are used for paired Raw/LEACE metrics."
        ),
        "nominal_confidence_level": 0.95,
        "bonferroni_family_size": len(FAMILY_ENDPOINTS),
        "bonferroni_adjusted_confidence_level": 1.0 - bonferroni_alpha,
        "intervals": intervals,
    }


def latex_table(summaries: dict[str, Any], intervals: dict[str, Any]) -> str:
    rows = []
    for key, label in (
        ("d_auc", r"$\Delta$ AUC"),
        ("d_eod_max", r"$\Delta$ EOD$_{\max}$"),
        ("d_dpd", r"$\Delta$ DPD"),
        ("d_worst_group_auc", r"$\Delta$ worst-group AUC"),
    ):
        summary = summaries[key]
        ci95 = intervals[key]["ci_95_percentile"]
        ci_adj = intervals[key]["bonferroni_adjusted_ci"]
        signs = summary["sign_count"]
        rows.append(
            f"{label} & {summary['mean']:+.4f} $\\pm$ {summary['sd']:.4f} "
            f"& [{summary['min']:+.4f}, {summary['max']:+.4f}] "
            f"& {signs['positive']}/{signs['negative']}/{signs['zero']} "
            f"& [{ci95[0]:+.4f}, {ci95[1]:+.4f}] "
            f"& [{ci_adj[0]:+.4f}, {ci_adj[1]:+.4f}] \\\\"
        )
    return (
        "% Paired delta is LEACE-Race minus Raw Dual; disparity improvements are negative.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Paired seed-group audit with two-level bootstrap confidence intervals. "
        "Sign counts are positive/negative/zero across seed groups. "
        "Bonferroni intervals adjust for four endpoints.}\n"
        "\\label{tab:seed-paired-audit}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\hline\n"
        "Endpoint & Mean $\\pm$ SD & Seed-group range & Sign count (+/$-$/0) "
        "& 95\\% CI & Bonferroni CI \\\\\n"
        "\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def print_summary(results: dict[str, Any]) -> None:
    print("\nPaired deltas (LEACE-Race minus Raw Dual; fixed threshold >= 0.50)")
    print(f"{'Endpoint':<22} {'Mean +/- SD':<22} {'Min..Max':<25} {'Signs (+/-/0)'}")
    for name in ("d_auc", "d_eod_max", "d_dpd", "d_worst_group_auc"):
        row = results["paired_delta_summary"][name]
        sign = row["sign_count"]
        print(
            f"{name:<22} {row['mean']:+.4f} +/- {row['sd']:.4f}       "
            f"[{row['min']:+.4f}, {row['max']:+.4f}]    "
            f"{sign['positive']}/{sign['negative']}/{sign['zero']}"
        )
    print("\nTwo-level bootstrap percentile intervals")
    for name, intervals in results["bootstrap"]["intervals"].items():
        print(
            f"{name}: 95% CI [{intervals['ci_95_percentile'][0]:+.4f}, "
            f"{intervals['ci_95_percentile'][1]:+.4f}], "
            f"Bonferroni CI [{intervals['bonferroni_adjusted_ci'][0]:+.4f}, "
            f"{intervals['bonferroni_adjusted_ci'][1]:+.4f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--groups",
        type=parse_groups,
        default=list(range(len(SEED_GROUPS))),
        help="Seed-group indexes 0-5 (e.g. 0,2 or 0-1); default runs all six.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive.")
    selected_groups = list(args.groups)
    if not selected_groups:
        parser.error("At least one seed group must be selected.")

    torch.set_num_threads(min(os.cpu_count() or 1, 8))
    set_seed(args.bootstrap_seed)
    features = load_features(args.cache.resolve())
    input_fingerprint = {
        "cache_sha256": sha256_file(args.cache.resolve()),
        "selected_groups": selected_groups,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "training": {
            "head": "Linear(1024,256)-LayerNorm-GELU-Dropout(0.2)-Linear(256,2)",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "threshold": THRESHOLD,
        },
    }
    if args.resume and args.checkpoint.is_file():
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("input_fingerprint") != input_fingerprint:
            raise ValueError(
                "Checkpoint inputs do not match this run; choose a different "
                "checkpoint path or run without --resume."
            )
        stored_predictions = checkpoint.get("predictions")
        if not isinstance(stored_predictions, dict):
            raise ValueError("Invalid checkpoint: missing prediction dictionary.")
        if not all(torch.is_tensor(value) for value in stored_predictions.values()):
            raise ValueError(
                "Checkpoint prediction format is not weights-only safe; run without "
                "--resume to rebuild it."
            )
        predictions = {
            key: value.numpy().astype(np.float64, copy=False)
            for key, value in stored_predictions.items()
        }
        print(f"Resuming from {args.checkpoint}")
    else:
        predictions = {}
        checkpoint = {
            "input_fingerprint": input_fingerprint,
            "predictions": {},
        }

    started = time.perf_counter()
    train_features = features["train"]["joint"].numpy()
    train_labels = features["train"]["labels"].numpy()
    train_races = features["train"]["races"].numpy()
    test_labels = features["test"]["labels"].numpy()
    test_races = features["test"]["races"].numpy()
    test_features = features["test"]["joint"].numpy()
    race_mean, race_eraser = fit_leace_race(train_features, train_races)
    leace_train = apply_leace(train_features, race_mean, race_eraser)
    leace_test = apply_leace(test_features, race_mean, race_eraser)

    total_models = len(selected_groups) * 10
    completed_models = 0
    for group_index in selected_groups:
        for seed in SEED_GROUPS[group_index]:
            for method, train_x, test_x in (
                ("raw", train_features, test_features),
                ("leace", leace_train, leace_test),
            ):
                key = f"{method}:{seed}"
                if key not in predictions:
                    model = train_predict(train_x, train_labels, seed)
                    predictions[key] = predict(model, test_x)
                    checkpoint["predictions"][key] = torch.from_numpy(
                        predictions[key].copy()
                    )
                    save_checkpoint(args.checkpoint, checkpoint)
                completed_models += 1
                print(
                    f"\rModels complete: {completed_models}/{total_models}",
                    end="",
                    flush=True,
                )
    print()

    group_rows: list[dict[str, Any]] = []
    group_predictions: list[dict[str, np.ndarray]] = []
    metric_names = {
        "auc": "auc",
        "eod_max": "eod_max",
        "dpd": "dpd",
        "worst_group_auc": "worst_group_auc",
    }
    for group_index in selected_groups:
        seeds = SEED_GROUPS[group_index]
        raw_probabilities = np.mean(
            [predictions[f"raw:{seed}"] for seed in seeds], axis=0
        )
        leace_probabilities = np.mean(
            [predictions[f"leace:{seed}"] for seed in seeds], axis=0
        )
        raw_metrics = binary_metrics(test_labels, test_races, raw_probabilities)
        leace_metrics = binary_metrics(test_labels, test_races, leace_probabilities)
        delta = {
            f"d_{metric_name}": leace_metrics[metric] - raw_metrics[metric]
            for metric_name, metric in metric_names.items()
        }
        group_rows.append(
            {
                "group_index": group_index,
                "seeds": list(seeds),
                "raw_metrics": raw_metrics,
                "leace_metrics": leace_metrics,
                "paired_delta": delta,
            }
        )
        group_predictions.append(
            {"raw": raw_probabilities, "leace": leace_probabilities}
        )

    summary_keys = ("d_auc", "d_eod_max", "d_dpd", "d_worst_group_auc")
    paired_summaries = {
        key: summarize([row["paired_delta"][key] for row in group_rows])
        for key in summary_keys
    }
    bootstrap = bootstrap_intervals(
        group_predictions,
        test_labels,
        test_races,
        args.bootstrap_draws,
        args.bootstrap_seed,
    )
    results: dict[str, Any] = {
        "method": "Paired 5-seed Raw Dual versus train-only LEACE-Race",
        "complete_six_group_audit": selected_groups == list(range(6)),
        "selected_groups": selected_groups,
        "group_definitions": [
            {"group_index": index, "seeds": list(SEED_GROUPS[index])}
            for index in selected_groups
        ],
        "delta_convention": "LEACE-Race minus Raw Dual",
        "threshold": {"value": THRESHOLD, "rule": "probability >= threshold"},
        "configuration": input_fingerprint["training"],
        "split_sizes": EXPECTED_SPLIT_SIZES,
        "test_race_counts": dict(zip(RACE_NAMES, EXPECTED_TEST_RACE_COUNTS)),
        "test_used_for_model_selection": False,
        "leace_fit_split": "train",
        "group_results": group_rows,
        "paired_delta_summary": paired_summaries,
        "bootstrap": bootstrap,
        "runtime_seconds": time.perf_counter() - started,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    args.output_tex.write_text(
        latex_table(paired_summaries, bootstrap["intervals"]), encoding="utf-8"
    )
    print_summary(results)
    print(f"\nSaved {args.output_json} and {args.output_tex}.")


if __name__ == "__main__":
    main()
