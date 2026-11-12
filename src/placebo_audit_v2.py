"""Placebo null audit with resumable five-seed ensembles and runtime pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cached_features.pt"
DEFAULT_RESULTS = ROOT / "placebo_v2_results.json"
DEFAULT_TABLE = ROOT / "tables" / "placebo_v2.tex"
DEFAULT_CHECKPOINT = ROOT / ".placebo_audit_v2_checkpoint.json"
SPLITS = ("train", "val", "test")
RACE_NAMES = ("Asian", "Black", "White")
EXPECTED_SPLIT_SIZES = {"train": 7000, "val": 1000, "test": 2000}
EXPECTED_TEST_RACE_COUNTS = (169, 302, 1529)

DEFAULT_N = 200
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.50
BASE_SEED = 20261112
PILOT_PLACEBOS = 2
PLACEBO_TYPES = {
    "permuted_race_labels": "Permuted race labels (PRIMARY null)",
    "race_permutation_within_glaucoma": (
        "Race labels permuted within glaucoma strata"
    ),
    "random_2d_grassmannian": "Random 2-D Grassmannian subspace removal",
}
AUDIT_METRICS = (
    "auc",
    "accuracy",
    "dpd",
    "eod_max",
    "tpr_gap",
    "fpr_gap",
    "race_probe_macro_auc",
)
P_VALUE_METRICS = ("eod_max", "dpd", "fpr_gap")


class DiagnosticHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


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


def load_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cached features not found: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise ValueError("Cached features must contain split dictionaries.")

    data: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        if split not in cache:
            raise ValueError(f"Cached features are missing split {split!r}.")
        values = cache[split]
        required = {"img", "txt", "labels", "races"}
        if not required.issubset(values):
            raise ValueError(f"{split} split must contain {sorted(required)}.")
        image = values["img"].float()
        text = values["txt"].float()
        labels = values["labels"].long().numpy()
        races = values["races"].long().numpy()
        if image.ndim != 2 or image.shape[1] != 512:
            raise ValueError(f"Expected 512-D image features in {split}.")
        if text.ndim != 2 or text.shape[1] != 512:
            raise ValueError(f"Expected 512-D text features in {split}.")
        if len(image) != len(text) or len(image) != len(labels) or len(image) != len(races):
            raise ValueError(f"Feature and label lengths disagree in {split}.")
        data[split] = {
            "features": torch.cat((image, text), dim=1).numpy().astype(np.float32),
            "labels": labels,
            "races": races,
        }

    sizes = {split: len(data[split]["labels"]) for split in SPLITS}
    assert sizes == EXPECTED_SPLIT_SIZES, sizes
    counts = tuple(
        int(np.count_nonzero(data["test"]["races"] == race))
        for race in range(len(RACE_NAMES))
    )
    assert counts == EXPECTED_TEST_RACE_COUNTS, counts
    for split in SPLITS:
        assert set(np.unique(data[split]["labels"])) == {0, 1}
        assert set(np.unique(data[split]["races"])) == {0, 1, 2}
    return data


def make_leace_transform(
    train_features: np.ndarray,
    protected_labels: np.ndarray,
    whitening: np.ndarray,
    dewhitening: np.ndarray,
    mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    one_hot = np.eye(len(RACE_NAMES), dtype=np.float64)[protected_labels]
    centered_labels = one_hot - one_hot.mean(axis=0)
    centered_features = train_features - mean
    cross_covariance = centered_features.T @ centered_labels / len(train_features)
    whitened_cross_covariance = whitening @ cross_covariance
    left_vectors, singular_values, _ = np.linalg.svd(
        whitened_cross_covariance, full_matrices=False
    )
    if singular_values.size == 0 or singular_values[0] == 0:
        return np.zeros((train_features.shape[1], 0)), np.zeros((0, train_features.shape[1]))
    keep = singular_values > 1e-8 * singular_values[0]
    directions = left_vectors[:, keep]
    left_projection = whitening @ directions
    right_projection = directions.T @ dewhitening
    return left_projection.astype(np.float32), right_projection.astype(np.float32)


def apply_leace(
    features: np.ndarray,
    mean: np.ndarray,
    left_projection: np.ndarray,
    right_projection: np.ndarray,
) -> np.ndarray:
    if left_projection.shape[1] == 0:
        return features.copy()
    centered = features - mean
    return (features - (centered @ left_projection) @ right_projection).astype(
        np.float32
    )


def fit_covariance_whitening(
    train_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, dtype=np.float64)
    centered = train_features.astype(np.float64) - mean
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = 1e-5 * float(eigenvalues.max())
    eigenvalues = np.clip(eigenvalues, floor, None)
    whitening = (eigenvectors * eigenvalues**-0.5) @ eigenvectors.T
    dewhitening = (eigenvectors * eigenvalues**0.5) @ eigenvectors.T
    return (
        mean.astype(np.float32),
        whitening,
        dewhitening,
    )


def fit_diagnostic_head(
    features: np.ndarray, labels: np.ndarray, seed: int
) -> DiagnosticHead:
    set_seed(seed)
    model = DiagnosticHead(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    train_features = torch.from_numpy(features).float()
    train_labels = torch.from_numpy(labels).long()
    for _ in range(EPOCHS):
        model.train()
        indexes = torch.randperm(len(train_labels))
        for start in range(0, len(train_labels), BATCH_SIZE):
            batch = indexes[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(train_features[batch]), train_labels[batch]
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model.eval()


def predict_ensemble(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    seed_predictions: list[np.ndarray] = []
    test_tensor = torch.from_numpy(test_features).float()
    for seed in SEEDS:
        model = fit_diagnostic_head(train_features, train_labels, seed * 100)
        with torch.no_grad():
            probabilities = torch.softmax(model(test_tensor), dim=1)[:, 1]
        seed_predictions.append(probabilities.numpy())
    return np.mean(seed_predictions, axis=0)


def race_probe_macro_auc(
    train_features: np.ndarray,
    train_races: np.ndarray,
    test_features: np.ndarray,
    test_races: np.ndarray,
) -> float:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    probe = LogisticRegression(C=0.01, max_iter=500, solver="lbfgs")
    probe.fit(train_scaled, train_races)
    probabilities = probe.predict_proba(test_scaled)
    return float(
        roc_auc_score(
            test_races,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    )


def evaluate_predictions(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    race_probe: float,
) -> dict[str, float]:
    predictions = probabilities > THRESHOLD
    positive_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []
    for race in range(len(RACE_NAMES)):
        mask = races == race
        race_labels = labels[mask]
        race_predictions = predictions[mask]
        positive = race_labels == 1
        negative = race_labels == 0
        if not positive.any() or not negative.any():
            raise ValueError(f"Test cohort {RACE_NAMES[race]} lacks an outcome class.")
        positive_rates.append(float(race_predictions.mean()))
        tprs.append(float(race_predictions[positive].mean()))
        fprs.append(float(race_predictions[negative].mean()))
    tpr_gap = max(tprs) - min(tprs)
    fpr_gap = max(fprs) - min(fprs)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(np.mean(predictions == labels)),
        "dpd": float(max(positive_rates) - min(positive_rates)),
        "eod_max": float(max(tpr_gap, fpr_gap)),
        "tpr_gap": float(tpr_gap),
        "fpr_gap": float(fpr_gap),
        "race_probe_macro_auc": float(race_probe),
    }


def create_placebo_features(
    placebo_type: str,
    index: int,
    data: dict[str, dict[str, np.ndarray]],
    mean: np.ndarray,
    whitening: np.ndarray,
    dewhitening: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train = data["train"]["features"]
    train_labels = data["train"]["labels"]
    train_races = data["train"]["races"]
    seed = BASE_SEED + index * 1009 + list(PLACEBO_TYPES).index(placebo_type) * 100_003
    rng = np.random.default_rng(seed)
    info: dict[str, Any] = {"rng_seed": seed}

    if placebo_type in (
        "permuted_race_labels",
        "race_permutation_within_glaucoma",
    ):
        if placebo_type == "permuted_race_labels":
            placebo_races = rng.permutation(train_races)
        else:
            placebo_races = train_races.copy()
            for label in (0, 1):
                stratum = np.flatnonzero(train_labels == label)
                placebo_races[stratum] = rng.permutation(train_races[stratum])
            for label in (0, 1):
                assert np.array_equal(
                    np.bincount(placebo_races[train_labels == label], minlength=3),
                    np.bincount(train_races[train_labels == label], minlength=3),
                )
        left_projection, right_projection = make_leace_transform(
            train, placebo_races, whitening, dewhitening, mean
        )
        placebo_features = {
            split: apply_leace(
                data[split]["features"], mean, left_projection, right_projection
            )
            for split in SPLITS
        }
        info["train_label"] = (
            "Permuted race labels"
            if placebo_type == "permuted_race_labels"
            else "Race labels permuted within glaucoma-label strata"
        )
        info["transformation_rank"] = int(left_projection.shape[1])
        return placebo_features, info

    if placebo_type == "random_2d_grassmannian":
        gaussian = rng.standard_normal((train.shape[1], 2))
        basis, _ = np.linalg.qr(gaussian, mode="reduced")
        placebo_features = {
            split: (
                data[split]["features"]
                - (
                    data[split]["features"] - mean
                )
                @ basis
                @ basis.T
            ).astype(np.float32)
            for split in SPLITS
        }
        info["subspace_dimension"] = 2
        return placebo_features, info
    raise ValueError(f"Unknown placebo type: {placebo_type}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.__getitem__)
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[name])
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def summarize_null(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        raise ValueError("Cannot summarize an empty placebo null.")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "percentile_2_5": float(np.percentile(array, 2.5)),
        "percentile_97_5": float(np.percentile(array, 97.5)),
    }


def build_summary(
    placebo_results: dict[str, list[dict[str, Any]]],
    true_leace: dict[str, float],
    target_n: int,
) -> dict[str, Any]:
    null_summary: dict[str, dict[str, Any]] = {}
    inference: dict[str, Any] = {}
    for placebo_type, records in placebo_results.items():
        null_summary[placebo_type] = {
            metric: summarize_null([float(record[metric]) for record in records])
            for metric in AUDIT_METRICS
        }

    for metric in P_VALUE_METRICS:
        p_values = {
            placebo_type: (
                1
                + sum(float(record[metric]) <= true_leace[metric] for record in records)
            )
            / (len(records) + 1)
            for placebo_type, records in placebo_results.items()
        }
        adjusted = holm_adjust(p_values)
        inference[metric] = {
            "alternative": (
                "The true LEACE disparity is lower than the placebo value "
                "(left-tailed; favorable direction)."
            ),
            "true_leace_value": true_leace[metric],
            "p_value_formula": "(k + 1) / (N + 1), where k = count(null <= true LEACE)",
            "by_placebo_type": {
                placebo_type: {
                    "n": len(placebo_results[placebo_type]),
                    "k_null_le_true": int(
                        sum(
                            float(record[metric]) <= true_leace[metric]
                            for record in placebo_results[placebo_type]
                        )
                    ),
                    "p_value": p_values[placebo_type],
                    "holm_adjusted_p_value": adjusted[placebo_type],
                }
                for placebo_type in placebo_results
            },
            "holm_family": "Three placebo types, adjusted separately for this metric.",
        }
    return {
        "target_n_per_type": target_n,
        "primary_null": "permuted_race_labels",
        "primary_null_description": PLACEBO_TYPES["permuted_race_labels"],
        "null_distributions": null_summary,
        "one_sided_permutation_tests": inference,
    }


def make_latex_table(summary: dict[str, Any]) -> str:
    primary = summary["null_distributions"]["permuted_race_labels"]
    p_values = summary["one_sided_permutation_tests"]
    rows = []
    for metric, label in (
        ("eod_max", r"EOD$_{\max}$"),
        ("dpd", "DPD"),
        ("fpr_gap", "FPR gap"),
    ):
        stats = primary[metric]
        primary_p = p_values[metric]["by_placebo_type"]["permuted_race_labels"]
        rows.append(
            f"{label} & {stats['mean']:.4f} $\\pm$ {stats['sd']:.4f} "
            f"& [{stats['percentile_2_5']:.4f}, {stats['percentile_97_5']:.4f}] "
            f"& {primary_p['p_value']:.4f} "
            f"& {primary_p['holm_adjusted_p_value']:.4f} \\\\"
        )
    return (
        "% Primary null: independently permuted train race labels.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Primary placebo null distribution and one-sided comparison "
        "with true LEACE. Lower disparity is favorable; p-values are Holm-adjusted "
        "over the three placebo types separately for each endpoint.}\n"
        "\\label{tab:placebo-v2}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\hline\n"
        "Metric & Null mean $\\pm$ SD & Null 2.5--97.5\\% & Raw $p$ "
        "& Holm-adjusted $p$ \\\\\n"
        "\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def print_results(results: dict[str, Any]) -> None:
    print("\nTrue LEACE-Race five-seed ensemble")
    print(" | ".join(f"{key}={value:.4f}" for key, value in results["true_leace"].items()))
    print(
        "\nPlacebo summaries (mean +/- SD [2.5th, 97.5th percentile]); "
        "primary null is independently permuted race labels"
    )
    for placebo_type, metrics in results["inference"]["null_distributions"].items():
        primary = " PRIMARY" if placebo_type == "permuted_race_labels" else ""
        print(f"\n{placebo_type}{primary}")
        for metric in AUDIT_METRICS:
            stats = metrics[metric]
            print(
                f"  {metric:<22} {stats['mean']:.4f} +/- {stats['sd']:.4f} "
                f"[{stats['percentile_2_5']:.4f}, {stats['percentile_97_5']:.4f}]"
            )
    print("\nOne-sided permutation p-values and Holm adjustment")
    for metric, result in results["inference"]["one_sided_permutation_tests"].items():
        print(f"  {metric}")
        for name, row in result["by_placebo_type"].items():
            print(
                f"    {name:<38} p={row['p_value']:.4f} "
                f"Holm={row['holm_adjusted_p_value']:.4f}"
            )


def main() -> None:
    global BASE_SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be a positive integer.")

    BASE_SEED = args.seed
    torch.set_num_threads(min(os.cpu_count() or 1, 8))
    set_seed(args.seed)
    cache_path = args.cache.resolve()
    data = load_cache(cache_path)
    train_features = data["train"]["features"]
    train_labels = data["train"]["labels"]
    train_races = data["train"]["races"]
    test_labels = data["test"]["labels"]
    test_races = data["test"]["races"]
    fingerprint = {
        "cache_sha256": sha256_file(cache_path),
        "seed": args.seed,
        "training": {
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "threshold": THRESHOLD,
            "threshold_rule": "probability > threshold",
        },
        "placebo_types": list(PLACEBO_TYPES),
    }

    if args.resume and args.checkpoint.is_file():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("input_fingerprint") != fingerprint:
            raise ValueError(
                "Checkpoint inputs do not match this run. Use the same cache, seed, "
                "and training configuration, or choose another checkpoint path."
            )
        placebo_results = checkpoint["placebo_results"]
        if any(len(records) > args.n for records in placebo_results.values()):
            raise ValueError(
                "Checkpoint contains more completed placebos than --n. "
                "Use the same or a larger --n, or choose another checkpoint."
            )
        true_leace = checkpoint.get("true_leace")
        baseline_seconds = float(checkpoint.get("baseline_seconds", 0.0))
        print(f"Resuming from {args.checkpoint}")
    else:
        placebo_results = {name: [] for name in PLACEBO_TYPES}
        true_leace = None
        baseline_seconds = 0.0
        checkpoint = {
            "input_fingerprint": fingerprint,
            "placebo_results": placebo_results,
            "true_leace": None,
            "baseline_seconds": 0.0,
        }

    mean, whitening, dewhitening = fit_covariance_whitening(train_features)
    if true_leace is None:
        baseline_started = time.perf_counter()
        leace_left, leace_right = make_leace_transform(
            train_features,
            train_races,
            whitening,
            dewhitening,
            mean,
        )
        leace_features = {
            split: apply_leace(
                data[split]["features"], mean, leace_left, leace_right
            )
            for split in SPLITS
        }
        leace_probabilities = predict_ensemble(
            leace_features["train"], train_labels, leace_features["test"]
        )
        leace_probe = race_probe_macro_auc(
            leace_features["train"],
            train_races,
            leace_features["test"],
            test_races,
        )
        true_leace = evaluate_predictions(
            test_labels, test_races, leace_probabilities, leace_probe
        )
        baseline_seconds = time.perf_counter() - baseline_started
        checkpoint["true_leace"] = true_leace
        checkpoint["baseline_seconds"] = baseline_seconds
        atomic_write_json(args.checkpoint, checkpoint)

    def run_placebo(placebo_type: str, index: int) -> None:
        placebo_started = time.perf_counter()
        placebo_features, placebo_info = create_placebo_features(
            placebo_type,
            index,
            data,
            mean,
            whitening,
            dewhitening,
        )
        probabilities = predict_ensemble(
            placebo_features["train"],
            train_labels,
            placebo_features["test"],
        )
        probe = race_probe_macro_auc(
            placebo_features["train"],
            train_races,
            placebo_features["test"],
            test_races,
        )
        record = evaluate_predictions(
            test_labels, test_races, probabilities, probe
        )
        record.update(
            {
                "index": index,
                "duration_seconds": time.perf_counter() - placebo_started,
                "metadata": placebo_info,
            }
        )
        placebo_results[placebo_type].append(record)
        checkpoint["placebo_results"] = placebo_results
        atomic_write_json(args.checkpoint, checkpoint)

    run_started = time.perf_counter()
    print(
        f"Runtime pilot: measuring the first {min(args.n, PILOT_PLACEBOS)} "
        "placebo(s) of each type before the remaining run.",
        flush=True,
    )
    for placebo_type in PLACEBO_TYPES:
        records = placebo_results[placebo_type]
        pilot_target = min(args.n, PILOT_PLACEBOS)
        for index in range(len(records), pilot_target):
            run_placebo(placebo_type, index)
            print(
                f"  Pilot {placebo_type}: "
                f"{len(placebo_results[placebo_type])}/{pilot_target} complete "
                f"({placebo_results[placebo_type][-1]['duration_seconds']:.1f}s)",
                flush=True,
            )

    elapsed_pilot = time.perf_counter() - run_started
    remaining_estimate = 0.0
    estimates_by_type: dict[str, float] = {}
    for placebo_type, records in placebo_results.items():
        pilot_durations = [
            float(record["duration_seconds"])
            for record in records[: min(args.n, PILOT_PLACEBOS)]
        ]
        average = float(np.mean(pilot_durations)) if pilot_durations else 0.0
        remaining_count = max(0, args.n - len(records))
        estimates_by_type[placebo_type] = average
        remaining_estimate += average * remaining_count
    estimated_total = baseline_seconds + elapsed_pilot + remaining_estimate
    print("\nRuntime estimate BEFORE remaining placebo run", flush=True)
    print(
        f"  Baseline five-seed LEACE ensemble: {baseline_seconds / 60:.1f} min",
        flush=True,
    )
    for placebo_type, average in estimates_by_type.items():
        print(f"  {placebo_type}: {average:.1f}s/placebo", flush=True)
    print(
        f"  Estimated remaining time: {remaining_estimate / 3600:.2f} hours; "
        f"estimated total: {estimated_total / 3600:.2f} hours.",
        flush=True,
    )
    print(
        "  The two pilot placebos per type count toward N and are checkpointed.",
        flush=True,
    )
    print("Starting the remaining placebos.", flush=True)

    for placebo_type in PLACEBO_TYPES:
        while len(placebo_results[placebo_type]) < args.n:
            index = len(placebo_results[placebo_type])
            run_placebo(placebo_type, index)
            record = placebo_results[placebo_type][-1]
            print(
                f"  {placebo_type}: {index + 1}/{args.n} "
                f"({record['duration_seconds']:.1f}s)",
                flush=True,
            )

    inference = build_summary(placebo_results, true_leace, args.n)
    results: dict[str, Any] = {
        "method": "Three placebo nulls versus five-seed LEACE-Race ensemble",
        "primary_null": inference["primary_null"],
        "primary_null_description": inference["primary_null_description"],
        "true_leace": true_leace,
        "inference": inference,
        "placebo_results": placebo_results,
        "configuration": {
            **fingerprint["training"],
            "leace_fit_split": "train",
            "placebo_count_per_type": args.n,
            "placebo_types": PLACEBO_TYPES,
            "split_sizes": EXPECTED_SPLIT_SIZES,
            "test_race_counts": dict(zip(RACE_NAMES, EXPECTED_TEST_RACE_COUNTS)),
            "test_used_for_model_selection": False,
            "linear_race_probe": (
                "StandardScaler fit on train; LogisticRegression(C=0.01, "
                "max_iter=500, lbfgs); macro one-vs-rest AUC on test."
            ),
            "runtime_pilot_placebos_per_type": min(args.n, PILOT_PLACEBOS),
        },
        "runtime": {
            "baseline_seconds": baseline_seconds,
            "pilot_elapsed_seconds_this_run": elapsed_pilot,
            "estimated_total_seconds_before_full_run": estimated_total,
            "actual_seconds_this_invocation": time.perf_counter() - run_started,
            "mean_placebo_seconds_from_pilot": estimates_by_type,
        },
        "checkpoint_file": str(args.checkpoint.resolve()),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_json, results)
    args.output_tex.write_text(make_latex_table(inference), encoding="utf-8")
    print_results(results)
    print(f"\nSaved {args.output_json} and {args.output_tex}.")


if __name__ == "__main__":
    main()
