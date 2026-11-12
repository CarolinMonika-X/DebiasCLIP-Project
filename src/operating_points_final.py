"""Compare canonical Raw Dual and LEACE ensembles at validation-set operating points."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cached_features.pt"
DEFAULT_CANONICAL = ROOT / "canonical_results.json"
DEFAULT_OUTPUT = ROOT / "operating_points_final.json"
DEFAULT_TEX = ROOT / "tables" / "opsens.tex"
DEFAULT_CHECKPOINT = ROOT / ".operating_points_final_checkpoint.json"
SPLITS = ("train", "val", "test")
RACE_NAMES = ("Asian", "Black", "White")
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SIZES = {"train": 7000, "val": 1000, "test": 2000}
EXPECTED_TEST_RACE_COUNTS = (169, 302, 1529)
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
FIXED_THRESHOLD = 0.50
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261112
TORCH_NUM_THREADS = os.cpu_count() or 4
OPERATING_POINTS = (
    "fixed_0.50",
    "validation_youden",
    "matched_specificity_0.80",
    "matched_sensitivity_0.80",
)
ENDPOINTS = ("d_auc", "d_eod_max", "d_dpd")
METRIC_KEYS = ("auc", "eod_max", "dpd")


class CanonicalHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class WeightedAUC:
    """AUC calculator for bootstrap multiplicities, including tied scores."""

    def __init__(self, labels: np.ndarray, scores: np.ndarray) -> None:
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        self.labels = labels.astype(np.int8, copy=False)[order]
        self.order = order
        self.block_ids = np.cumsum(
            np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
        ) - 1
        self.block_count = int(self.block_ids[-1]) + 1

    def __call__(self, multiplicities: np.ndarray) -> float:
        weights = multiplicities[self.order].astype(np.float64, copy=False)
        positive = self.labels == 1
        positive_weights = np.bincount(
            self.block_ids,
            weights=weights * positive,
            minlength=self.block_count,
        )
        negative_weights = np.bincount(
            self.block_ids,
            weights=weights * ~positive,
            minlength=self.block_count,
        )
        positive_total = float(positive_weights.sum())
        negative_total = float(negative_weights.sum())
        if positive_total == 0 or negative_total == 0:
            raise ValueError("AUC bootstrap sample omitted a glaucoma class.")
        negative_before = np.cumsum(negative_weights) - negative_weights
        concordant = np.sum(
            positive_weights * (negative_before + 0.5 * negative_weights)
        )
        return float(concordant / (positive_total * negative_total))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cached features not found: {path}")
    cached = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cached, dict):
        raise ValueError("Cached features must contain split dictionaries.")
    data: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        if split not in cached:
            raise ValueError(f"Cache is missing the {split!r} split.")
        row = cached[split]
        required = {"img", "txt", "labels", "races"}
        if not required.issubset(row):
            raise ValueError(f"{split} cache is missing {sorted(required)}.")
        image, text = row["img"].float(), row["txt"].float()
        if image.ndim != 2 or image.shape[1] != 512:
            raise ValueError(f"Expected 512-D image embeddings in {split}.")
        if text.ndim != 2 or text.shape[1] != 512:
            raise ValueError(f"Expected 512-D text embeddings in {split}.")
        labels, races = row["labels"].long(), row["races"].long()
        if len(image) != len(text) or len(image) != len(labels) or len(image) != len(races):
            raise ValueError(f"Features and labels have different lengths in {split}.")
        features = torch.cat((image, text), dim=1).numpy().astype(
            np.float32, copy=False
        )
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite cached features in {split}.")
        data[split] = {
            "features": features,
            "labels": labels.numpy(),
            "races": races.numpy(),
        }
    sizes = {split: len(data[split]["labels"]) for split in SPLITS}
    assert sizes == EXPECTED_SIZES, sizes
    test_counts = tuple(
        int(np.count_nonzero(data["test"]["races"] == race))
        for race in range(len(RACE_NAMES))
    )
    assert test_counts == EXPECTED_TEST_RACE_COUNTS, test_counts
    for split in SPLITS:
        assert set(np.unique(data[split]["labels"])) == {0, 1}
        assert set(np.unique(data[split]["races"])) == {0, 1, 2}
    return data


def fit_leace(train_features: np.ndarray, train_races: np.ndarray) -> Any:
    mean = train_features.mean(axis=0)
    centered = train_features - mean
    race_indicators = np.eye(len(RACE_NAMES), dtype=np.float64)[train_races]
    centered_races = race_indicators - race_indicators.mean(axis=0)
    covariance = centered.T @ centered / len(train_features)
    cross_covariance = centered.T @ centered_races / len(train_features)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 1e-5 * eigenvalues.max(), None)
    whitening = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    inverse_whitening = (
        eigenvectors @ np.diag(eigenvalues ** 0.5) @ eigenvectors.T
    )
    left_vectors, singular_values, _ = np.linalg.svd(
        whitening @ cross_covariance, full_matrices=False
    )
    directions = left_vectors[:, singular_values > 1e-8 * singular_values.max()]
    eraser = inverse_whitening @ (directions @ directions.T) @ whitening

    def transform(features: np.ndarray) -> np.ndarray:
        return (features - (features - mean) @ eraser.T).astype(np.float32)

    return transform


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> dict[str, list[float]]:
    seed_everything(seed)
    model = CanonicalHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_x = torch.from_numpy(train_features).float()
    train_y = torch.from_numpy(train_labels).long()
    val_x = torch.from_numpy(val_features).float()
    test_x = torch.from_numpy(test_features).float()
    for _ in range(EPOCHS):
        permutation = torch.randperm(len(train_x))
        model.train()
        for start in range(0, len(train_x), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            logits = model(train_x[indices])
            loss = nn.functional.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        val_probabilities = torch.softmax(model(val_x), dim=1)[:, 1].numpy()
        test_probabilities = torch.softmax(model(test_x), dim=1)[:, 1].numpy()
    return {
        "val": val_probabilities.astype(float).tolist(),
        "test": test_probabilities.astype(float).tolist(),
    }


def fit_ensemble(
    name: str,
    data: dict[str, dict[str, np.ndarray]],
    transformed: dict[str, np.ndarray],
    checkpoint: dict[str, Any],
    persist: Any,
) -> tuple[np.ndarray, np.ndarray]:
    entry = checkpoint["models"].setdefault(name, {"seed_predictions": []})
    train = data["train"]
    while len(entry["seed_predictions"]) < len(SEEDS):
        seed_index = len(entry["seed_predictions"])
        prediction = train_and_predict(
            transformed["train"],
            train["labels"],
            transformed["val"],
            transformed["test"],
            SEEDS[seed_index],
        )
        entry["seed_predictions"].append(prediction)
        persist()
        print(
            f"{name}: finished canonical seed {SEEDS[seed_index]} "
            f"({seed_index + 1}/{len(SEEDS)})",
            flush=True,
        )
    val_probability = np.mean(
        [row["val"] for row in entry["seed_predictions"]], axis=0
    )
    test_probability = np.mean(
        [row["test"] for row in entry["seed_predictions"]], axis=0
    )
    return val_probability, test_probability


def threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    unique = np.unique(probabilities.astype(np.float64))
    return np.unique(
        np.r_[
            np.nextafter(unique[0], -np.inf),
            unique,
            np.nextafter(unique[-1], np.inf),
        ]
    )


def operating_thresholds(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, dict[str, float]]:
    thresholds = threshold_candidates(probabilities)
    specificity = np.empty(len(thresholds), dtype=np.float64)
    sensitivity = np.empty(len(thresholds), dtype=np.float64)
    j_values = np.empty(len(thresholds), dtype=np.float64)
    negatives = labels == 0
    positives = labels == 1
    if not negatives.any() or not positives.any():
        raise ValueError("Validation split must include both glaucoma classes.")
    for index, threshold in enumerate(thresholds):
        predictions = probabilities > threshold
        sensitivity[index] = predictions[positives].mean()
        specificity[index] = (~predictions[negatives]).mean()
        j_values[index] = sensitivity[index] + specificity[index] - 1

    max_j = float(j_values.max())
    youden_candidates = np.flatnonzero(np.isclose(j_values, max_j, atol=1e-12))
    youden_index = min(
        youden_candidates,
        key=lambda index: (
            abs(float(thresholds[index]) - FIXED_THRESHOLD),
            -float(thresholds[index]),
        ),
    )
    spec_distances = np.abs(specificity - 0.80)
    best_spec_distance = float(spec_distances.min())
    matched_spec_candidates = np.flatnonzero(
        np.isclose(spec_distances, best_spec_distance, atol=1e-12)
    )
    matched_spec_index = min(
        matched_spec_candidates,
        key=lambda index: (
            -float(sensitivity[index]),
            abs(float(thresholds[index]) - FIXED_THRESHOLD),
            -float(thresholds[index]),
        ),
    )
    sens_distances = np.abs(sensitivity - 0.80)
    best_sens_distance = float(sens_distances.min())
    matched_sens_candidates = np.flatnonzero(
        np.isclose(sens_distances, best_sens_distance, atol=1e-12)
    )
    matched_sens_index = min(
        matched_sens_candidates,
        key=lambda index: (
            -float(specificity[index]),
            abs(float(thresholds[index]) - FIXED_THRESHOLD),
            -float(thresholds[index]),
        ),
    )
    selected = {
        "fixed_0.50": (
            FIXED_THRESHOLD,
            float(np.mean(probabilities > FIXED_THRESHOLD)),
        ),
        "validation_youden": (
            float(thresholds[youden_index]),
            float(j_values[youden_index]),
        ),
        "matched_specificity_0.80": (
            float(thresholds[matched_spec_index]),
            float(specificity[matched_spec_index]),
        ),
        "matched_sensitivity_0.80": (
            float(thresholds[matched_sens_index]),
            float(sensitivity[matched_sens_index]),
        ),
    }
    result: dict[str, dict[str, float]] = {}
    for name, (threshold, objective_value) in selected.items():
        predictions = probabilities > threshold
        val_sensitivity = float(predictions[positives].mean())
        val_specificity = float((~predictions[negatives]).mean())
        row = {
            "threshold": threshold,
            "validation_sensitivity": val_sensitivity,
            "validation_specificity": val_specificity,
        }
        if name == "validation_youden":
            row["validation_youden_j"] = objective_value
        elif name == "matched_specificity_0.80":
            row["target_specificity"] = 0.80
            row["absolute_specificity_target_error"] = abs(objective_value - 0.80)
        elif name == "matched_sensitivity_0.80":
            row["target_sensitivity"] = 0.80
            row["absolute_sensitivity_target_error"] = abs(objective_value - 0.80)
        result[name] = row
    return result


def metric_values(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    multiplicities: np.ndarray | None = None,
) -> dict[str, Any]:
    weights = (
        np.ones(len(labels), dtype=np.float64)
        if multiplicities is None
        else multiplicities.astype(np.float64, copy=False)
    )
    predictions = probabilities > threshold
    positive_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []
    group_values: dict[str, dict[str, float]] = {}
    for race, name in enumerate(RACE_NAMES):
        group = races == race
        positive = group & (labels == 1)
        negative = group & (labels == 0)
        group_total = float(weights[group].sum())
        positive_total = float(weights[positive].sum())
        negative_total = float(weights[negative].sum())
        if group_total == 0 or positive_total == 0 or negative_total == 0:
            raise ValueError(f"Bootstrap sample lacks an outcome in race group {name}.")
        tp = float(np.sum(weights[positive] * predictions[positive]))
        fp = float(np.sum(weights[negative] * predictions[negative]))
        positive_rate = float(np.sum(weights[group] * predictions[group]) / group_total)
        tpr = tp / positive_total
        fpr = fp / negative_total
        positive_rates.append(positive_rate)
        tprs.append(tpr)
        fprs.append(fpr)
        group_values[name] = {
            "positive_rate": positive_rate,
            "tpr": tpr,
            "fpr": fpr,
        }
    tpr_max_index = int(np.argmax(tprs))
    tpr_min_index = int(np.argmin(tprs))
    fpr_max_index = int(np.argmax(fprs))
    fpr_min_index = int(np.argmin(fprs))
    return {
        "auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        "accuracy": float(
            np.sum(weights * (predictions == labels)) / weights.sum()
        ),
        "dpd": float(max(positive_rates) - min(positive_rates)),
        "tpr_gap": float(tprs[tpr_max_index] - tprs[tpr_min_index]),
        "fpr_gap": float(fprs[fpr_max_index] - fprs[fpr_min_index]),
        "eod_max": float(
            max(
                tprs[tpr_max_index] - tprs[tpr_min_index],
                fprs[fpr_max_index] - fprs[fpr_min_index],
            )
        ),
        "group_metrics": group_values,
        "gap_groups": {
            "tpr_max": RACE_NAMES[tpr_max_index],
            "tpr_min": RACE_NAMES[tpr_min_index],
            "fpr_max": RACE_NAMES[fpr_max_index],
            "fpr_min": RACE_NAMES[fpr_min_index],
        },
    }


def percentile_ci(values: np.ndarray, confidence: float) -> list[float]:
    alpha = 1.0 - confidence
    return [
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    ]


def paired_bootstrap(
    labels: np.ndarray,
    races: np.ndarray,
    raw_probabilities: np.ndarray,
    leace_probabilities: np.ndarray,
    raw_cutoffs: dict[str, dict[str, float]],
    leace_cutoffs: dict[str, dict[str, float]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    test_count = len(labels)
    auc_calculators = {
        "raw": WeightedAUC(labels, raw_probabilities),
        "leace": WeightedAUC(labels, leace_probabilities),
    }
    distributions = {
        operating_point: {
            endpoint: np.empty(draws, dtype=np.float64)
            for endpoint in ENDPOINTS
        }
        for operating_point in OPERATING_POINTS
    }
    for draw in range(draws):
        sampled = rng.integers(0, test_count, size=test_count)
        multiplicities = np.bincount(sampled, minlength=test_count)
        raw_auc = auc_calculators["raw"](multiplicities)
        leace_auc = auc_calculators["leace"](multiplicities)
        for operating_point in OPERATING_POINTS:
            raw_threshold = raw_cutoffs[operating_point]["threshold"]
            leace_threshold = leace_cutoffs[operating_point]["threshold"]
            raw_metrics = metric_values(
                labels,
                races,
                raw_probabilities,
                raw_threshold,
                multiplicities,
            )
            leace_metrics = metric_values(
                labels,
                races,
                leace_probabilities,
                leace_threshold,
                multiplicities,
            )
            distributions[operating_point]["d_auc"][draw] = leace_auc - raw_auc
            distributions[operating_point]["d_eod_max"][draw] = (
                leace_metrics["eod_max"] - raw_metrics["eod_max"]
            )
            distributions[operating_point]["d_dpd"][draw] = (
                leace_metrics["dpd"] - raw_metrics["dpd"]
            )

    point_estimates: dict[str, dict[str, float]] = {}
    for operating_point in OPERATING_POINTS:
        raw_threshold = raw_cutoffs[operating_point]["threshold"]
        leace_threshold = leace_cutoffs[operating_point]["threshold"]
        raw_metrics = metric_values(
            labels, races, raw_probabilities, raw_threshold
        )
        leace_metrics = metric_values(
            labels, races, leace_probabilities, leace_threshold
        )
        point_estimates[operating_point] = {
            "d_auc": leace_metrics["auc"] - raw_metrics["auc"],
            "d_eod_max": leace_metrics["eod_max"] - raw_metrics["eod_max"],
            "d_dpd": leace_metrics["dpd"] - raw_metrics["dpd"],
        }
    return {
        "draws": draws,
        "seed": seed,
        "delta_convention": "LEACE minus Raw Dual",
        "resampling": (
            "Paired ordinary bootstrap over test cases; identical case "
            "multiplicities are applied to Raw and LEACE at every operating point. "
            "Validation-selected thresholds remain fixed within each draw."
        ),
        "point_estimates": point_estimates,
        "distributions": distributions,
    }


def bootstrap_p_value(distribution: np.ndarray) -> float:
    lower = int(np.count_nonzero(distribution <= 0))
    upper = int(np.count_nonzero(distribution >= 0))
    return float(min(1.0, 2 * (min(lower, upper) + 1) / (len(distribution) + 1)))


def add_holm_intervals(bootstrap: dict[str, Any]) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    for operating_point in OPERATING_POINTS:
        for endpoint in ENDPOINTS:
            distribution = np.asarray(
                bootstrap["distributions"][operating_point][endpoint],
                dtype=np.float64,
            )
            tests.append(
                {
                    "operating_point": operating_point,
                    "endpoint": endpoint,
                    "point_estimate": bootstrap["point_estimates"][operating_point][
                        endpoint
                    ],
                    "distribution": distribution,
                    "p_value": bootstrap_p_value(distribution),
                    "ci_95_percentile": percentile_ci(distribution, 0.95),
                }
            )
    tests.sort(key=lambda test: (test["p_value"], test["operating_point"], test["endpoint"]))
    family_size = len(tests)
    alpha = 0.05
    running_adjusted_p = 0.0
    first_non_rejection: int | None = None
    results: dict[str, dict[str, Any]] = {}
    for rank, test in enumerate(tests):
        remaining_hypotheses = family_size - rank
        step_alpha = alpha / remaining_hypotheses
        adjusted_p = min(
            1.0,
            (remaining_hypotheses) * test["p_value"],
        )
        running_adjusted_p = max(running_adjusted_p, adjusted_p)
        rejected = (
            first_non_rejection is None and test["p_value"] <= step_alpha
        )
        if not rejected and first_non_rejection is None:
            first_non_rejection = rank
        adjusted_ci = percentile_ci(test["distribution"], 1 - step_alpha)
        if not rejected:
            adjusted_ci = [
                min(adjusted_ci[0], 0.0),
                max(adjusted_ci[1], 0.0),
            ]
        confidence_level = 1 - step_alpha
        key = f"{test['operating_point']}:{test['endpoint']}"
        results[key] = {
            "operating_point": test["operating_point"],
            "endpoint": test["endpoint"],
            "rank_by_unadjusted_p": rank + 1,
            "unadjusted_bootstrap_p_value": test["p_value"],
            "holm_adjusted_p_value": running_adjusted_p,
            "holm_step_alpha": step_alpha,
            "holm_rejects_at_0.05": rejected,
            "ci_95_percentile": test["ci_95_percentile"],
            "holm_adjusted_ci": adjusted_ci,
            "holm_ci_confidence_level": confidence_level,
        }
    return {
        "family_size": family_size,
        "familywise_alpha": alpha,
        "method": (
            "Holm step-down over two-sided paired-bootstrap p-values. Each "
            "ranked test receives a percentile interval at confidence "
            "1 - alpha/(m-rank+1); Holm rejection stops at the first "
            "non-rejection. Non-rejected intervals are widened to include zero."
        ),
        "tests": results,
    }


def group_decomposition(
    labels: np.ndarray,
    races: np.ndarray,
    raw_probabilities: np.ndarray,
    leace_probabilities: np.ndarray,
    raw_cutoffs: dict[str, dict[str, float]],
    leace_cutoffs: dict[str, dict[str, float]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for operating_point in OPERATING_POINTS:
        raw = metric_values(
            labels,
            races,
            raw_probabilities,
            raw_cutoffs[operating_point]["threshold"],
        )
        leace = metric_values(
            labels,
            races,
            leace_probabilities,
            leace_cutoffs[operating_point]["threshold"],
        )
        group_changes = {
            race: {
                metric: leace["group_metrics"][race][metric]
                - raw["group_metrics"][race][metric]
                for metric in ("tpr", "fpr")
            }
            for race in RACE_NAMES
        }
        largest_fpr_shift = max(
            RACE_NAMES,
            key=lambda race: abs(group_changes[race]["fpr"]),
        )
        report[operating_point] = {
            "raw": {
                "tpr_gap": raw["tpr_gap"],
                "fpr_gap": raw["fpr_gap"],
                "gap_groups": raw["gap_groups"],
                "per_group": raw["group_metrics"],
            },
            "leace": {
                "tpr_gap": leace["tpr_gap"],
                "fpr_gap": leace["fpr_gap"],
                "gap_groups": leace["gap_groups"],
                "per_group": leace["group_metrics"],
            },
            "delta_leace_minus_raw": {
                "tpr_gap": leace["tpr_gap"] - raw["tpr_gap"],
                "fpr_gap": leace["fpr_gap"] - raw["fpr_gap"],
                "per_group": group_changes,
            },
            "fpr_gap_change_drivers": {
                "raw_max_fpr_group": raw["gap_groups"]["fpr_max"],
                "raw_min_fpr_group": raw["gap_groups"]["fpr_min"],
                "leace_max_fpr_group": leace["gap_groups"]["fpr_max"],
                "leace_min_fpr_group": leace["gap_groups"]["fpr_min"],
                "largest_absolute_group_fpr_change": largest_fpr_shift,
                "largest_absolute_group_fpr_change_value": group_changes[
                    largest_fpr_shift
                ]["fpr"],
            },
        }
    return report


def latex_table(results: dict[str, Any]) -> str:
    rows: list[str] = []
    holm = results["holm_adjustment"]["tests"]
    for operating_point in OPERATING_POINTS:
        for endpoint in ENDPOINTS:
            summary = results["operating_points"][operating_point]["deltas"][endpoint]
            adjusted = holm[f"{operating_point}:{endpoint}"]
            nominal = summary["ci_95_percentile"]
            simultaneous = adjusted["holm_adjusted_ci"]
            operating_label = operating_point.replace("_", r"\ ")
            endpoint_label = endpoint.replace("_", r"\_")
            rows.append(
                f"{operating_label} "
                f"& {endpoint_label} "
                f"& {summary['point_estimate']:+.4f} "
                f"& [{nominal[0]:+.4f}, {nominal[1]:+.4f}] "
                f"& [{simultaneous[0]:+.4f}, {simultaneous[1]:+.4f}] "
                f"& {adjusted['holm_adjusted_p_value']:.4f} \\\\"
            )

    fpr_rows: list[str] = []
    decomposition = results["group_gap_decomposition"]
    for operating_point in OPERATING_POINTS:
        section = decomposition[operating_point]
        raw, leace = section["raw"], section["leace"]
        drivers = section["fpr_gap_change_drivers"]
        shift_race = drivers["largest_absolute_group_fpr_change"]
        shift_value = drivers["largest_absolute_group_fpr_change_value"]
        operating_label = operating_point.replace("_", r"\ ")
        tpr_gap_change = section["delta_leace_minus_raw"]["tpr_gap"]
        fpr_rows.append(
            f"{operating_label} "
            f"& {raw['tpr_gap']:.4f} & {leace['tpr_gap']:.4f} "
            f"& {tpr_gap_change:+.4f} "
            f"& {raw['fpr_gap']:.4f} & {leace['fpr_gap']:.4f} "
            f"& {section['delta_leace_minus_raw']['fpr_gap']:+.4f} "
            f"& {drivers['raw_max_fpr_group']}/{drivers['raw_min_fpr_group']} "
            f"& {drivers['leace_max_fpr_group']}/{drivers['leace_min_fpr_group']} "
            f"& {shift_race} ({shift_value:+.4f}) \\\\"
        )
    return (
        "% Cutoffs are selected using validation ensemble predictions only.\n"
        "% Deltas are LEACE minus Raw Dual; bootstrap resamples are paired by test case.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Operating-point sensitivity analysis for canonical Raw Dual and "
        "LEACE models. Holm-adjusted intervals use a step-down procedure across "
        "12 endpoint tests.}\n"
        "\\label{tab:operating-points-final}\n"
        "\\begin{tabular}{llrrrr}\n"
        "\\hline\n"
        "Operating point & Endpoint & Delta & Paired bootstrap 95\\% CI "
        "& Holm-adjusted CI & Holm $p$ \\\\\n"
        "\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{TPR and FPR gap decomposition. Group pairs identify "
        "maximum/minimum FPR groups; the final column identifies the largest "
        "absolute group FPR shift.}\n"
        "\\label{tab:operating-points-fpr-decomposition}\n"
        "\\begin{tabular}{lrrrrrrlll}\n"
        "\\hline\n"
        "Operating point & Raw TPR gap & LEACE TPR gap & $\\Delta$ TPR gap "
        "& Raw FPR gap & LEACE FPR gap & $\\Delta$ FPR gap "
        "& Raw FPR max/min & LEACE FPR max/min & Largest group FPR shift \\\\\n"
        "\\hline\n"
        + "\n".join(fpr_rows)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive.")
    if not args.canonical.is_file():
        raise FileNotFoundError(f"Canonical results not found: {args.canonical}")
    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    if canonical.get("canonical_run") != "placebo_audit":
        raise ValueError("Canonical results do not designate placebo_audit as final.")
    canonical_config = canonical["configuration"]
    expected_head = "Linear(1024,512) -> ReLU -> Dropout(0.1) -> Linear(512,2)"
    if canonical_config.get("head_architecture") != expected_head:
        raise ValueError("Canonical model architecture differs from this implementation.")
    if canonical_config.get("seed_list") != list(SEEDS):
        raise ValueError("Canonical ensemble seed list differs from this implementation.")
    if canonical_config.get("epochs") != EPOCHS or canonical_config.get("batch_size") != BATCH_SIZE:
        raise ValueError("Canonical training schedule differs from this implementation.")
    if (
        canonical_config.get("optimizer") != "Adam (default weight decay 0)"
        or canonical_config.get("learning_rate") != LEARNING_RATE
        or canonical_config.get("weight_decay") != 0.0
        or canonical_config.get("threshold_rule") != "probability > 0.50"
    ):
        raise ValueError("Canonical optimizer or threshold convention differs.")
    if canonical_config.get("feature_layout") != "Concatenate cached image[512] then text[512] features":
        raise ValueError("Canonical feature layout differs from this implementation.")

    torch.set_num_threads(TORCH_NUM_THREADS)
    cache_path = args.cache.resolve()
    canonical_path = args.canonical.resolve()
    data = load_data(cache_path)
    fingerprint = {
        "cache_sha256": sha256_file(cache_path),
        "canonical_results_sha256": sha256_file(canonical_path),
        "canonical_run": canonical["canonical_run"],
        "canonical_source": canonical["canonical_source"],
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "torch_num_threads": TORCH_NUM_THREADS,
        "threshold_rule": "probability > threshold",
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    if args.resume and args.checkpoint.is_file():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("input_fingerprint") != fingerprint:
            raise ValueError(
                "Checkpoint inputs or settings do not match this run; use a new "
                "checkpoint path or run without --resume."
            )
        print(f"Resuming from {args.checkpoint}", flush=True)
    else:
        checkpoint = {"input_fingerprint": fingerprint, "models": {}}

    def persist() -> None:
        atomic_write_json(args.checkpoint, checkpoint)

    train, val, test = (data[split] for split in SPLITS)
    raw_features = {split: data[split]["features"] for split in SPLITS}
    leace_transform = fit_leace(train["features"], train["races"])
    leace_features = {
        split: leace_transform(data[split]["features"]) for split in SPLITS
    }
    raw_val, raw_test = fit_ensemble(
        "raw", data, raw_features, checkpoint, persist
    )
    leace_val, leace_test = fit_ensemble(
        "leace", data, leace_features, checkpoint, persist
    )
    raw_cutoffs = operating_thresholds(val["labels"], raw_val)
    leace_cutoffs = operating_thresholds(val["labels"], leace_val)

    bootstrap = paired_bootstrap(
        test["labels"],
        test["races"],
        raw_test,
        leace_test,
        raw_cutoffs,
        leace_cutoffs,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    holm = add_holm_intervals(bootstrap)
    point_metrics: dict[str, Any] = {}
    for operating_point in OPERATING_POINTS:
        raw = metric_values(
            test["labels"],
            test["races"],
            raw_test,
            raw_cutoffs[operating_point]["threshold"],
        )
        leace = metric_values(
            test["labels"],
            test["races"],
            leace_test,
            leace_cutoffs[operating_point]["threshold"],
        )
        point_metrics[operating_point] = {
            "cutoffs": {
                "raw_dual": raw_cutoffs[operating_point],
                "leace": leace_cutoffs[operating_point],
                "selection_split": "validation",
            },
            "test_metrics": {"raw_dual": raw, "leace": leace},
            "deltas": {},
        }
        for endpoint, metric in (
            ("d_auc", "auc"),
            ("d_eod_max", "eod_max"),
            ("d_dpd", "dpd"),
        ):
            distribution = bootstrap["distributions"][operating_point][endpoint]
            point_metrics[operating_point]["deltas"][endpoint] = {
                "point_estimate": leace[metric] - raw[metric],
                "ci_95_percentile": percentile_ci(
                    np.asarray(distribution), 0.95
                ),
                "bootstrap_mean": float(np.mean(distribution)),
                "holm_adjusted_p_value": holm["tests"][
                    f"{operating_point}:{endpoint}"
                ]["holm_adjusted_p_value"],
                "holm_adjusted_ci": holm["tests"][
                    f"{operating_point}:{endpoint}"
                ]["holm_adjusted_ci"],
            }

    decomposition = group_decomposition(
        test["labels"],
        test["races"],
        raw_test,
        leace_test,
        raw_cutoffs,
        leace_cutoffs,
    )
    results: dict[str, Any] = {
        "method": "Canonical Raw Dual versus train-only LEACE operating-point audit",
        "canonical_source": canonical["canonical_source"],
        "canonical_run": canonical["canonical_run"],
        "canonical_configuration": canonical_config,
        "configuration": {
            **fingerprint,
            "test_race_counts": dict(zip(RACE_NAMES, EXPECTED_TEST_RACE_COUNTS)),
            "leace_fit_split": "train only",
            "leace_race_labels_used_for_fit": True,
            "model_selection_split": "validation",
            "threshold_rule": "probability > cutoff",
            "threshold_tie_breaks": {
                "validation_youden": (
                    "Among tied Youden maxima, choose the threshold nearest 0.50, "
                    "then the higher threshold."
                ),
                "matched_specificity_0.80": (
                    "Choose specificity closest to 0.80; ties maximize sensitivity, "
                    "then choose threshold nearest 0.50."
                ),
                "matched_sensitivity_0.80": (
                    "Choose sensitivity closest to 0.80; ties maximize specificity, "
                    "then choose threshold nearest 0.50."
                ),
            },
            "test_used_for_cutoff_selection": False,
        },
        "delta_convention": "LEACE minus Raw Dual",
        "bootstrap": {
            key: value
            for key, value in bootstrap.items()
            if key != "distributions"
        },
        "holm_adjustment": holm,
        "operating_points": point_metrics,
        "group_gap_decomposition": decomposition,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_json, results)
    args.output_tex.write_text(latex_table(results), encoding="utf-8")

    print("\nLEACE minus Raw Dual (validation-selected operating points)")
    for operating_point in OPERATING_POINTS:
        print(
            f"{operating_point}: "
            + ", ".join(
                f"{endpoint}={point_metrics[operating_point]['deltas'][endpoint]['point_estimate']:+.4f} "
                f"95%CI={point_metrics[operating_point]['deltas'][endpoint]['ci_95_percentile']}"
                for endpoint in ENDPOINTS
            )
        )
        drivers = decomposition[operating_point]["fpr_gap_change_drivers"]
        print(
            f"  FPR gap change={decomposition[operating_point]['delta_leace_minus_raw']['fpr_gap']:+.4f}; "
            f"largest group shift: {drivers['largest_absolute_group_fpr_change']} "
            f"({drivers['largest_absolute_group_fpr_change_value']:+.4f})"
        )
    print(f"\nSaved {args.output_json} and {args.output_tex}.")


if __name__ == "__main__":
    main()
