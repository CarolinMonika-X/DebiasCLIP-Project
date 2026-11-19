"""Search validation accuracy under race-fairness constraints using race-scrubbed text."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CACHE = ROOT / "cached_features.pt"
SCRUBBED_CACHE = ROOT / ".text_leakage_scrubbed_features.pt"
RESULTS_PATH = ROOT / "accuracy_fairness_results.json"
CHECKPOINT_PATH = ROOT / ".accuracy_fairness_checkpoint.pt"
SPLITS = ("train", "val", "test")
EXPECTED_SIZES = {"train": 7000, "val": 1000, "test": 2000}
EXPECTED_TEST_RACE_COUNTS = (169, 302, 1529)
RACE_NAMES = ("Asian", "Black", "White")
SEEDS = (0, 100, 200, 300, 400)
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FAIRNESS_TOLERANCE = 1e-12
BLEND_WEIGHTS = (0.25, 0.50, 0.75)


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


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def stratified_indices(labels: np.ndarray, races: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or limit >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    buckets = [
        np.flatnonzero((labels == label) & (races == race))
        for label in (0, 1)
        for race in range(3)
    ]
    if any(len(bucket) == 0 for bucket in buckets):
        raise ValueError("Every race/outcome cell must be present for stratified smoke mode.")
    per_bucket = max(1, limit // len(buckets))
    rng = np.random.default_rng(seed)
    chosen = [
        rng.choice(bucket, size=min(per_bucket, len(bucket)), replace=False)
        for bucket in buckets
    ]
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def load_train_validation(
    feature_cache_path: Path,
    scrubbed_cache_path: Path,
    limit: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    if not feature_cache_path.is_file():
        raise FileNotFoundError(f"OpenCLIP feature cache not found: {feature_cache_path}")
    if not scrubbed_cache_path.is_file():
        raise FileNotFoundError(
            f"Scrubbed OpenCLIP text cache not found: {scrubbed_cache_path}. "
            "Run src/text_leakage_ablation.py first."
        )
    raw = torch.load(feature_cache_path, map_location="cpu", weights_only=True)
    scrubbed = torch.load(scrubbed_cache_path, map_location="cpu", weights_only=True)
    if "encoded_by_split" not in scrubbed:
        raise ValueError("Scrubbed cache is missing encoded_by_split.")
    metadata = scrubbed.get("input_fingerprint", {})
    if metadata.get("split_seed") != 42:
        raise ValueError("Scrubbed text cache is not verified for seed-42 split.")

    data: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val"):
        row = raw[split]
        labels = row["labels"].detach().cpu().long().numpy().reshape(-1)
        races = row["races"].detach().cpu().long().numpy().reshape(-1)
        image = row["img"].detach().cpu().float().numpy()
        text_tensor = scrubbed["encoded_by_split"][split]
        text = (
            text_tensor.detach().cpu().float().numpy()
            if isinstance(text_tensor, torch.Tensor)
            else np.asarray(text_tensor, dtype=np.float32)
        )
        if image.shape != (EXPECTED_SIZES[split], 512):
            raise ValueError(f"Invalid cached image matrix for {split}: {image.shape}.")
        if text.shape != image.shape:
            raise ValueError(f"Invalid scrubbed text matrix for {split}: {text.shape}.")
        if labels.shape != (len(image),) or races.shape != labels.shape:
            raise ValueError(f"Label/race size mismatch for {split}.")
        if not np.isfinite(image).all() or not np.isfinite(text).all():
            raise ValueError(f"Non-finite features detected in {split}.")
        if not np.array_equal(np.unique(labels), np.array([0, 1])):
            raise ValueError(f"Both glaucoma labels are required in {split}.")
        if not np.array_equal(np.unique(races), np.array([0, 1, 2])):
            raise ValueError(f"All race groups are required in {split}.")
        data[split] = {
            "img": image.astype(np.float32, copy=False),
            "txt": text.astype(np.float32, copy=False),
            "labels": labels,
            "races": races,
        }

    if limit:
        for split in ("train", "val"):
            selected = stratified_indices(
                data[split]["labels"],
                data[split]["races"],
                limit,
                seed=SEEDS[0] + (0 if split == "train" else 1),
            )
            data[split] = {key: value[selected] for key, value in data[split].items()}
    return data, raw


def fit_leace(train_features: np.ndarray, train_races: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = train_features.astype(np.float64, copy=False)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    concepts = np.eye(3, dtype=np.float64)[train_races]
    concepts -= concepts.mean(axis=0)
    covariance = centered.T @ centered / len(matrix)
    cross_covariance = centered.T @ concepts / len(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()) * 1e-5, np.finfo(np.float64).eps)
    eigenvalues = np.clip(eigenvalues, floor, None)
    whitening = (eigenvectors * eigenvalues**-0.5) @ eigenvectors.T
    inverse_whitening = (eigenvectors * eigenvalues**0.5) @ eigenvectors.T
    directions, singular_values, _ = np.linalg.svd(
        whitening @ cross_covariance,
        full_matrices=False,
    )
    if singular_values.size:
        directions = directions[:, singular_values > 1e-8 * singular_values.max()]
    else:
        directions = directions[:, :0]
    eraser = inverse_whitening @ (directions @ directions.T) @ whitening
    return mean.astype(np.float32), eraser.astype(np.float32)


def apply_leace(features: np.ndarray, mean: np.ndarray, eraser: np.ndarray) -> np.ndarray:
    values = features.astype(np.float32, copy=False)
    return (values - (values - mean) @ eraser.T).astype(np.float32, copy=False)


def construct_representations(
    data: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray] | None]]:
    raw_features = {
        split: {
            "openclip_image": row["img"],
            "scrubbed_text": row["txt"],
            "scrubbed_joint": np.concatenate((row["img"], row["txt"]), axis=1),
        }
        for split, row in data.items()
    }
    representation_names = ("openclip_image", "scrubbed_text", "scrubbed_joint")
    features: dict[str, dict[str, np.ndarray]] = {}
    transforms: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for name in representation_names:
        features[f"raw_{name}"] = {
            split: raw_features[split][name]
            for split in data
        }
        transforms[f"raw_{name}"] = None
        mean, eraser = fit_leace(
            raw_features["train"][name],
            data["train"]["races"],
        )
        debiased_name = f"leace_{name}"
        features[debiased_name] = {
            split: apply_leace(raw_features[split][name], mean, eraser)
            for split in data
        }
        transforms[debiased_name] = (mean, eraser)
    return features, transforms


def train_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
    epochs: int,
) -> DiagnosticHead:
    set_seed(seed)
    model = DiagnosticHead(train_features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    train_x = torch.from_numpy(np.ascontiguousarray(train_features)).float()
    train_y = torch.from_numpy(np.ascontiguousarray(train_labels)).long()
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_y))
        for start in range(0, len(train_y), BATCH_SIZE):
            indexes = permutation[start : start + BATCH_SIZE]
            logits = model(train_x[indexes])
            loss = nn.functional.cross_entropy(logits, train_y[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def predict(model: DiagnosticHead, features: np.ndarray) -> np.ndarray:
    model.eval()
    matrix = torch.from_numpy(np.ascontiguousarray(features)).float()
    with torch.inference_mode():
        probabilities = torch.softmax(model(matrix), dim=1)[:, 1]
    return probabilities.cpu().numpy()


def state_to_cpu(model: DiagnosticHead) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_from_cpu(feature_dim: int, state: dict[str, torch.Tensor]) -> DiagnosticHead:
    model = DiagnosticHead(feature_dim)
    model.load_state_dict(state)
    return model


def fairness_metrics(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    positive_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []
    for race in range(3):
        group = races == race
        positive = group & (labels == 1)
        negative = group & (labels == 0)
        if not positive.any() or not negative.any():
            raise ValueError(
                f"Validation set lacks an outcome group for {RACE_NAMES[race]}."
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


def best_fair_threshold(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    dpd_cap: float,
    eod_cap: float,
) -> dict[str, float | bool]:
    threshold_values = np.unique(
        np.concatenate(
            (
                probabilities.astype(np.float64),
                np.asarray([np.nextafter(float(np.max(probabilities)), np.inf)]),
            )
        )
    )
    auc = float(roc_auc_score(labels, probabilities))
    overall_negative_n = int(np.count_nonzero(labels == 0))
    grouped_scores = {
        (race, label): np.sort(probabilities[(races == race) & (labels == label)])
        for race in range(3)
        for label in (0, 1)
    }
    group_sizes = {
        race: int(np.count_nonzero(races == race))
        for race in range(3)
    }
    best: dict[str, float | bool] | None = None
    for threshold in threshold_values:
        true_positives = [
            len(grouped_scores[(race, 1)])
            - int(np.searchsorted(grouped_scores[(race, 1)], threshold, side="left"))
            for race in range(3)
        ]
        false_positives = [
            len(grouped_scores[(race, 0)])
            - int(np.searchsorted(grouped_scores[(race, 0)], threshold, side="left"))
            for race in range(3)
        ]
        tprs = [
            true_positives[race] / len(grouped_scores[(race, 1)])
            for race in range(3)
        ]
        fprs = [
            false_positives[race] / len(grouped_scores[(race, 0)])
            for race in range(3)
        ]
        positive_rates = [
            (true_positives[race] + false_positives[race]) / group_sizes[race]
            for race in range(3)
        ]
        tpr_gap = max(tprs) - min(tprs)
        fpr_gap = max(fprs) - min(fprs)
        metrics = {
            "auc": auc,
            "accuracy": (
                sum(true_positives)
                + overall_negative_n
                - sum(false_positives)
            ) / len(labels),
            "dpd": max(positive_rates) - min(positive_rates),
            "eod_max": max(tpr_gap, fpr_gap),
            "tpr_gap": tpr_gap,
            "fpr_gap": fpr_gap,
        }
        if (
            metrics["dpd"] > dpd_cap + FAIRNESS_TOLERANCE
            or metrics["eod_max"] > eod_cap + FAIRNESS_TOLERANCE
        ):
            continue
        candidate = {
            **metrics,
            "threshold": float(threshold),
            "eligible": True,
        }
        if best is None or (
            candidate["accuracy"],
            -candidate["eod_max"],
            -candidate["dpd"],
            -abs(float(candidate["threshold"]) - 0.5),
            -float(candidate["threshold"]),
        ) > (
            best["accuracy"],
            -best["eod_max"],
            -best["dpd"],
            -abs(float(best["threshold"]) - 0.5),
            -float(best["threshold"]),
        ):
            best = candidate
    if best is None:
        raise RuntimeError("Raw baseline fairness caps yielded no eligible threshold.")
    return best


def candidate_specs(base_names: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {"name": name, "components": [name], "weights": [1.0]}
        for name in base_names
    ]
    for first_index, first in enumerate(base_names):
        for second in base_names[first_index + 1 :]:
            for first_weight in BLEND_WEIGHTS:
                weights = [first_weight, 1.0 - first_weight]
                candidates.append(
                    {
                        "name": f"blend_{first}_w{first_weight:.2f}_{second}_w{1-first_weight:.2f}",
                        "components": [first, second],
                        "weights": weights,
                    }
                )
    return candidates


def combine_probabilities(
    candidate: dict[str, Any],
    base_probabilities: dict[str, np.ndarray],
) -> np.ndarray:
    return np.average(
        np.column_stack(
            [base_probabilities[name] for name in candidate["components"]]
        ),
        axis=1,
        weights=candidate["weights"],
    )


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2)) / denominator
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def per_group_test_metrics(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    predicted = probabilities >= threshold
    output: dict[str, dict[str, Any]] = {}
    for race, name in enumerate(RACE_NAMES):
        group = races == race
        positive = group & (labels == 1)
        negative = group & (labels == 0)
        positive_n = int(positive.sum())
        negative_n = int(negative.sum())
        tp = int(np.count_nonzero(predicted & positive))
        fn = int(np.count_nonzero(~predicted & positive))
        fp = int(np.count_nonzero(predicted & negative))
        if not positive_n or not negative_n:
            raise ValueError(f"Test sample lacks a positive or negative {name} case.")
        output[name] = {
            "n": int(group.sum()),
            "positive_n": positive_n,
            "negative_n": negative_n,
            "tpr": tp / positive_n,
            "tpr_wilson_95_ci": list(wilson(tp, positive_n)),
            "fnr": fn / positive_n,
            "fnr_wilson_95_ci": list(wilson(fn, positive_n)),
            "fpr": fp / negative_n,
            "fpr_wilson_95_ci": list(wilson(fp, negative_n)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, default=FEATURE_CACHE)
    parser.add_argument("--scrubbed-cache", type=Path, default=SCRUBBED_CACHE)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Stratified smoke-test row cap per split.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")
    if args.limit and args.limit < 60:
        parser.error("--limit must be zero or at least 60 to preserve all six race/outcome cells.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")

    result_path = args.output_json or (
        ROOT / "accuracy_fairness_smoke_results.json" if args.limit else RESULTS_PATH
    )
    checkpoint_path = args.checkpoint or (
        ROOT / ".accuracy_fairness_smoke_checkpoint.pt" if args.limit else CHECKPOINT_PATH
    )
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    set_seed(SEEDS[0])

    print("Loading train/validation features; test labels and predictions remain unused.", flush=True)
    data, raw_cache = load_train_validation(
        args.feature_cache,
        args.scrubbed_cache,
        args.limit,
    )
    input_fingerprint = {
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "scrubbed_cache_sha256": sha256_file(args.scrubbed_cache),
        "seed_list": list(SEEDS),
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "smoke_limit": args.limit,
        "split_seed": 42,
        "race_fairness_caps": "raw scrubbed-joint validation DPD/EOD_max at threshold 0.50",
    }
    checkpoint: dict[str, Any]
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("input_fingerprint") != input_fingerprint:
            raise ValueError("Resume checkpoint fingerprint does not match the current inputs/settings.")
        print(f"Resuming from {checkpoint_path.name}.", flush=True)
        if "final_result" in checkpoint:
            write_json(result_path, checkpoint["final_result"])
            print(
                "The final test evaluation is already checkpointed; restored "
                f"{result_path.name} without rerunning test inference.",
                flush=True,
            )
            return
    else:
        checkpoint = {"input_fingerprint": input_fingerprint, "base_models": {}}

    representations, transforms = construct_representations(data)
    base_names = list(representations)
    val_labels = data["val"]["labels"]
    val_races = data["val"]["races"]
    val_predictions: dict[str, np.ndarray] = {}
    for base_index, name in enumerate(base_names, start=1):
        saved = checkpoint["base_models"].get(name)
        if saved is not None:
            val_predictions[name] = np.asarray(saved["val_probabilities"], dtype=np.float64)
            print(f"[{base_index}/{len(base_names)}] Resumed {name}.", flush=True)
            continue
        print(f"[{base_index}/{len(base_names)}] Training {name}.", flush=True)
        seed_states: list[dict[str, torch.Tensor]] = []
        seed_val_predictions: list[np.ndarray] = []
        for seed_index, seed in enumerate(SEEDS, start=1):
            model = train_model(
                representations[name]["train"],
                data["train"]["labels"],
                seed,
                args.epochs,
            )
            seed_val_predictions.append(predict(model, representations[name]["val"]))
            seed_states.append(state_to_cpu(model))
            print(f"  seed {seed_index}/{len(SEEDS)} finished.", flush=True)
        mean_validation = np.mean(seed_val_predictions, axis=0)
        checkpoint["base_models"][name] = {
            "feature_dim": int(representations[name]["train"].shape[1]),
            "seed_states": seed_states,
            "val_probabilities": mean_validation.tolist(),
            "leace_transform": (
                None
                if transforms[name] is None
                else {
                    "mean": torch.from_numpy(transforms[name][0]),
                    "eraser": torch.from_numpy(transforms[name][1]),
                }
            ),
        }
        save_checkpoint(checkpoint_path, checkpoint)
        val_predictions[name] = mean_validation

    raw_baseline_name = "raw_scrubbed_joint"
    raw_fixed_metrics = fairness_metrics(
        val_labels,
        val_races,
        val_predictions[raw_baseline_name],
        0.5,
    )
    dpd_cap = raw_fixed_metrics["dpd"]
    eod_cap = raw_fixed_metrics["eod_max"]
    print(
        f"Validation fairness caps from race-scrubbed Raw Dual at 0.50: "
        f"DPD<={dpd_cap:.6f}, EOD_max<={eod_cap:.6f}.",
        flush=True,
    )

    candidates = candidate_specs(base_names)
    candidate_results: list[dict[str, Any]] = []
    for candidate in candidates:
        probabilities = combine_probabilities(candidate, val_predictions)
        selected = best_fair_threshold(
            val_labels,
            val_races,
            probabilities,
            dpd_cap,
            eod_cap,
        )
        candidate_results.append(
            {
                **candidate,
                "validation": selected,
            }
        )
    selected_candidate = max(
        candidate_results,
        key=lambda item: (
            item["validation"]["accuracy"],
            -item["validation"]["eod_max"],
            -item["validation"]["dpd"],
            -abs(item["validation"]["threshold"] - 0.5),
        ),
    )
    baseline_candidate = next(
        row for row in candidate_results if row["name"] == raw_baseline_name
    )
    if (
        baseline_candidate["validation"]["dpd"] > dpd_cap + FAIRNESS_TOLERANCE
        or baseline_candidate["validation"]["eod_max"] > eod_cap + FAIRNESS_TOLERANCE
    ):
        raise AssertionError("Raw Dual must be eligible under its own validation caps.")
    print(
        f"Validation-selected model: {selected_candidate['name']} at threshold "
        f"{selected_candidate['validation']['threshold']:.6f}; accuracy="
        f"{selected_candidate['validation']['accuracy']:.4f}.",
        flush=True,
    )

    # Only now access test labels and produce final test probabilities.
    print("Validation selection complete; running final test evaluation.", flush=True)
    raw_test = raw_cache["test"]
    test_labels = raw_test["labels"].detach().cpu().long().numpy().reshape(-1)
    test_races = raw_test["races"].detach().cpu().long().numpy().reshape(-1)
    test_image = raw_test["img"].detach().cpu().float().numpy()
    scrubbed_cache = torch.load(args.scrubbed_cache, map_location="cpu", weights_only=True)
    test_text_tensor = scrubbed_cache["encoded_by_split"]["test"]
    test_text = (
        test_text_tensor.detach().cpu().float().numpy()
        if isinstance(test_text_tensor, torch.Tensor)
        else np.asarray(test_text_tensor, dtype=np.float32)
    )
    if not args.limit:
        assert len(test_labels) == EXPECTED_SIZES["test"]
        counts = tuple(int(np.count_nonzero(test_races == race)) for race in range(3))
        assert counts == EXPECTED_TEST_RACE_COUNTS, counts
    else:
        selected_test = stratified_indices(
            test_labels,
            test_races,
            args.limit,
            seed=SEEDS[0] + 2,
        )
        test_labels = test_labels[selected_test]
        test_races = test_races[selected_test]
        test_image = test_image[selected_test]
        test_text = test_text[selected_test]
    test_features = {
        "openclip_image": test_image,
        "scrubbed_text": test_text,
        "scrubbed_joint": np.concatenate((test_image, test_text), axis=1),
    }
    model_names_for_test = set(baseline_candidate["components"]) | set(
        selected_candidate["components"]
    )
    test_base_predictions: dict[str, np.ndarray] = {}
    for name in model_names_for_test:
        base_name = name.removeprefix("raw_").removeprefix("leace_")
        features = test_features[base_name]
        saved = checkpoint["base_models"][name]
        transform = saved["leace_transform"]
        if transform is not None:
            features = apply_leace(
                features,
                transform["mean"].numpy(),
                transform["eraser"].numpy(),
            )
        seed_probabilities: list[np.ndarray] = []
        for state in saved["seed_states"]:
            model = state_from_cpu(saved["feature_dim"], state)
            seed_probabilities.append(predict(model, features))
        test_base_predictions[name] = np.mean(seed_probabilities, axis=0)

    baseline_test_probabilities = combine_probabilities(
        baseline_candidate,
        test_base_predictions,
    )
    selected_test_probabilities = combine_probabilities(
        selected_candidate,
        test_base_predictions,
    )
    baseline_threshold = float(baseline_candidate["validation"]["threshold"])
    selected_threshold = float(selected_candidate["validation"]["threshold"])

    def final_metrics(probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
        return {
            **fairness_metrics(test_labels, test_races, probabilities, threshold),
            "threshold_selected_on_validation": threshold,
            "per_race": per_group_test_metrics(
                test_labels,
                test_races,
                probabilities,
                threshold,
            ),
        }

    baseline_test = final_metrics(baseline_test_probabilities, baseline_threshold)
    selected_test = final_metrics(selected_test_probabilities, selected_threshold)
    result = {
        "protocol": {
            "split": "seeded random 70/10/20 split using seed 42; not official folder split",
            "split_sizes": (
                {"train": len(data["train"]["labels"]), "val": len(data["val"]["labels"]), "test": len(test_labels)}
            ),
            "test_used_for_selection": False,
            "test_evaluated_after_validation_selection": True,
            "test_evaluation_scope": (
                "One final evaluation of the Raw Dual reference and validation-selected "
                "candidate; all other candidate comparisons are validation-only."
            ),
            "prediction_features_exclude_race_labels": True,
            "text_features": "OpenCLIP embeddings of race-scrubbed gpt4_summary only",
            "race_used_for": "train-only LEACE transform and validation fairness measurements/constraints",
            "mlp": "Linear(d,256)-LayerNorm-GELU-Dropout(0.2)-Linear(256,2)",
            "seeds": list(SEEDS),
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "fairness_rule": (
                "Maximize validation accuracy subject to validation DPD and "
                "EOD_max not exceeding race-scrubbed Raw Dual at threshold 0.50."
            ),
            "baseline_fairness_caps": {
                "reference_model": raw_baseline_name,
                "reference_threshold": 0.5,
                "dpd": dpd_cap,
                "eod_max": eod_cap,
            },
            "global_threshold_only": True,
            "tested_blend_weights": list(BLEND_WEIGHTS),
            "smoke_limit": args.limit,
        },
        "validation_trials": candidate_results,
        "selected_model": {
            **selected_candidate,
            "validation_accuracy_gain_vs_raw_dual": (
                selected_candidate["validation"]["accuracy"]
                - baseline_candidate["validation"]["accuracy"]
            ),
            "test": selected_test,
        },
        "raw_dual_baseline": {
            **baseline_candidate,
            "fixed_0_50_reference_metrics": raw_fixed_metrics,
            "test": baseline_test,
        },
        "test_delta_selected_minus_raw_dual": {
            metric: selected_test[metric] - baseline_test[metric]
            for metric in ("auc", "accuracy", "dpd", "eod_max")
        },
    }
    checkpoint["final_result"] = result
    save_checkpoint(checkpoint_path, checkpoint)
    write_json(result_path, result)
    print(
        f"Final test: Raw Dual accuracy={baseline_test['accuracy']:.4f}, "
        f"DPD={baseline_test['dpd']:.4f}, EOD_max={baseline_test['eod_max']:.4f}; "
        f"selected={selected_candidate['name']} accuracy={selected_test['accuracy']:.4f}, "
        f"DPD={selected_test['dpd']:.4f}, EOD_max={selected_test['eod_max']:.4f}.",
        flush=True,
    )
    print(f"Saved {result_path}.", flush=True)


if __name__ == "__main__":
    main()
