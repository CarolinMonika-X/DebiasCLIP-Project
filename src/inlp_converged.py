"""Validation-selected, converged INLP comparison against train-only LEACE."""

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
DEFAULT_RESULTS = ROOT / "inlp_converged_results.json"
DEFAULT_TABLE = ROOT / "tables" / "inlp.tex"
DEFAULT_CHECKPOINT = ROOT / ".inlp_converged_checkpoint.json"
SPLITS = ("train", "val", "test")
RACE_NAMES = ("Asian", "Black", "White")
EXPECTED_SPLIT_SIZES = {"train": 7000, "val": 1000, "test": 2000}
EXPECTED_TEST_RACE_COUNTS = (169, 302, 1529)

SEEDS = (0, 100, 200, 300, 400)
MAX_ITERATIONS = 20
CONVERGENCE_AUC = 0.52
THRESHOLD = 0.50
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261112
PROBE_METRICS = (
    "auc",
    "accuracy",
    "dpd",
    "eod_max",
    "tpr_gap",
    "fpr_gap",
    "linear_probe_macro_auc",
    "nonlinear_probe_macro_auc",
)


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


class RaceProbe(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(RACE_NAMES)),
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


def load_data(path: Path) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    if not path.is_file():
        raise FileNotFoundError(f"Cached features not found: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise ValueError("Cached features must contain split dictionaries.")

    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    races: dict[str, np.ndarray] = {}
    for split in SPLITS:
        if split not in cache:
            raise ValueError(f"Cached features are missing split {split!r}.")
        values = cache[split]
        required = {"img", "txt", "labels", "races"}
        if not required.issubset(values):
            raise ValueError(f"{split} split must contain {sorted(required)}.")
        image = values["img"].float()
        text = values["txt"].float()
        if image.ndim != 2 or image.shape[1] != 512:
            raise ValueError(f"Expected 512-D image features in {split}.")
        if text.ndim != 2 or text.shape[1] != 512:
            raise ValueError(f"Expected 512-D text features in {split}.")
        if len(image) != len(text):
            raise ValueError(f"Image and text sample counts differ in {split}.")
        features[split] = torch.cat((image, text), dim=1).numpy().astype(
            np.float32, copy=False
        )
        labels[split] = values["labels"].long().numpy()
        races[split] = values["races"].long().numpy()
        if labels[split].shape != (len(features[split]),):
            raise ValueError(f"Invalid glaucoma label shape in {split}.")
        if races[split].shape != labels[split].shape:
            raise ValueError(f"Race and glaucoma sample counts differ in {split}.")
        if not np.isfinite(features[split]).all():
            raise ValueError(f"Non-finite features found in {split}.")

    sizes = {split: len(labels[split]) for split in SPLITS}
    assert sizes == EXPECTED_SPLIT_SIZES, sizes
    test_counts = tuple(
        int(np.count_nonzero(races["test"] == race))
        for race in range(len(RACE_NAMES))
    )
    assert test_counts == EXPECTED_TEST_RACE_COUNTS, test_counts
    for split in SPLITS:
        assert set(np.unique(labels[split])) == {0, 1}
        assert set(np.unique(races[split])) == {0, 1, 2}
    return features, labels, races


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def fit_leace(
    train_features: np.ndarray, train_races: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    matrix = train_features.astype(np.float64, copy=False)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    concepts = np.eye(len(RACE_NAMES), dtype=np.float64)[train_races]
    centered_concepts = concepts - concepts.mean(axis=0)
    covariance = centered.T @ centered / len(matrix)
    cross_covariance = centered.T @ centered_concepts / len(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(1e-5 * float(eigenvalues.max()), np.finfo(np.float64).eps)
    eigenvalues = np.clip(eigenvalues, floor, None)
    whitening = (eigenvectors * eigenvalues**-0.5) @ eigenvectors.T
    inverse_whitening = (eigenvectors * eigenvalues**0.5) @ eigenvectors.T
    directions, singular_values, _ = np.linalg.svd(
        whitening @ cross_covariance, full_matrices=False
    )
    if singular_values.size:
        directions = directions[:, singular_values > 1e-8 * singular_values.max()]
    else:
        directions = directions[:, :0]
    projection = inverse_whitening @ (directions @ directions.T) @ whitening
    return mean.astype(np.float32), projection.astype(np.float32)


def apply_leace(
    features: np.ndarray, mean: np.ndarray, projection: np.ndarray
) -> np.ndarray:
    values = features.astype(np.float64, copy=False)
    return (values - (values - mean) @ projection.T).astype(np.float32)


def project_with_basis(features: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return features.copy()
    return (features - (features @ basis) @ basis.T).astype(np.float32)


def linear_probe_probabilities(
    train_features: np.ndarray,
    probe_features: np.ndarray,
    train_races: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    probe_scaled = scaler.transform(probe_features)
    classifier = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs")
    classifier.fit(train_scaled, train_races)
    probabilities = classifier.predict_proba(probe_scaled)
    class_order = np.argsort(classifier.classes_)
    return probabilities[:, class_order]


def race_auc_from_probabilities(
    races: np.ndarray, probabilities: np.ndarray
) -> float:
    return float(
        roc_auc_score(
            races,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=np.arange(len(RACE_NAMES)),
        )
    )


def add_inlp_direction(
    basis: np.ndarray,
    projected_classifier: np.ndarray,
    projector: np.ndarray,
) -> np.ndarray:
    _, singular_values, right_vectors = np.linalg.svd(
        projected_classifier, full_matrices=False
    )
    if singular_values.size == 0 or singular_values[0] == 0:
        return basis
    rank = int(np.count_nonzero(singular_values > 1e-8 * singular_values[0]))
    candidates = projector @ right_vectors[:rank].T
    residual = candidates - basis @ (basis.T @ candidates)
    new_vectors, residual_singular_values, _ = np.linalg.svd(
        residual, full_matrices=False
    )
    if residual_singular_values.size == 0:
        return basis
    new_rank = int(
        np.count_nonzero(
            residual_singular_values > 1e-8 * residual_singular_values[0]
        )
    )
    if new_rank == 0:
        return basis
    return np.concatenate((basis, new_vectors[:, :new_rank]), axis=1).astype(
        np.float64
    )


def run_inlp_path(
    features: dict[str, np.ndarray],
    races: dict[str, np.ndarray],
    max_iterations: int,
    checkpoint: dict[str, Any],
    save_checkpoint: Any,
) -> dict[str, Any]:
    dimension = features["train"].shape[1]
    saved = checkpoint.get("inlp_state")
    if saved is not None:
        basis = np.asarray(saved["basis"], dtype=np.float64)
        history = saved["history"]
        snapshots = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in saved["snapshots"].items()
        }
        closest_k = int(saved["closest_k"])
        convergence_k = saved.get("convergence_k")
    else:
        basis = np.zeros((dimension, 0), dtype=np.float64)
        history = []
        snapshots: dict[str, np.ndarray] = {}
        closest_k = 0
        convergence_k = None

    if history and len(history) >= max_iterations:
        if convergence_k is None:
            convergence_k = int(history[-1]["k"])
            snapshots["convergence"] = basis.copy()
        return {
            "history": history,
            "snapshots": snapshots,
            "closest_k": closest_k,
            "convergence_k": convergence_k,
        }
    if (
        convergence_k is not None
        and len(history) >= max(2, int(convergence_k))
    ):
        return {
            "history": history,
            "snapshots": snapshots,
            "closest_k": closest_k,
            "convergence_k": int(convergence_k),
        }

    start_k = len(history) + 1
    for iteration in range(start_k, max_iterations + 1):
        projector = np.eye(dimension, dtype=np.float64) - basis @ basis.T
        train_projected = project_with_basis(features["train"], basis)
        classifier = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
        )
        classifier.fit(train_projected, races["train"])
        basis = add_inlp_direction(basis, classifier.coef_, projector)

        validation_train = project_with_basis(features["train"], basis)
        validation_features = project_with_basis(features["val"], basis)
        validation_probabilities = linear_probe_probabilities(
            validation_train, validation_features, races["train"]
        )
        validation_auc = race_auc_from_probabilities(
            races["val"], validation_probabilities
        )
        history.append(
            {
                "k": iteration,
                "validation_race_probe_macro_auc": validation_auc,
                "removed_rank": int(basis.shape[1]),
            }
        )

        if iteration == 2:
            snapshots["k2"] = basis.copy()
        if not snapshots or abs(validation_auc - 0.5) < abs(
            next(
                item["validation_race_probe_macro_auc"]
                for item in history
                if item["k"] == closest_k
            )
            - 0.5
        ):
            closest_k = iteration
            snapshots["closest"] = basis.copy()
        if convergence_k is None and validation_auc <= CONVERGENCE_AUC:
            convergence_k = iteration
            snapshots["convergence"] = basis.copy()

        checkpoint["inlp_state"] = {
            "basis": basis.tolist(),
            "history": history,
            "snapshots": {
                key: value.tolist() for key, value in snapshots.items()
            },
            "closest_k": closest_k,
            "convergence_k": convergence_k,
        }
        save_checkpoint(checkpoint)
        print(
            f"INLP iteration {iteration}/{max_iterations}: validation race "
            f"macro-AUC={validation_auc:.4f}, removed rank={basis.shape[1]}",
            flush=True,
        )

        if convergence_k is not None and iteration >= max(2, convergence_k):
            break

    if "k2" not in snapshots:
        raise RuntimeError("INLP did not reach k=2; k=2 comparison is unavailable.")
    if convergence_k is None:
        convergence_k = int(history[-1]["k"])
        snapshots["convergence"] = basis.copy()
    if "closest" not in snapshots:
        raise RuntimeError("INLP did not produce a closest-to-chance snapshot.")
    return {
        "history": history,
        "snapshots": snapshots,
        "closest_k": closest_k,
        "convergence_k": convergence_k,
    }


def fit_diagnostic_seed(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> nn.Module:
    set_seed(seed)
    model = DiagnosticHead(train_features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    train_x = torch.from_numpy(train_features).float()
    train_y = torch.from_numpy(train_labels).long()
    for _ in range(EPOCHS):
        model.train()
        permutation = torch.randperm(len(train_y))
        for start in range(0, len(train_y), BATCH_SIZE):
            batch = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model.eval()


def diagnostic_ensemble(
    name: str,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    saved_predictions: dict[str, Any],
    save_checkpoint: Any,
) -> np.ndarray:
    if name not in saved_predictions:
        saved_predictions[name] = []
    predictions: list[list[float]] = saved_predictions[name]
    if len(predictions) > len(SEEDS):
        raise ValueError(f"Checkpoint has an invalid number of seeds for {name}.")
    test_x = torch.from_numpy(test_features).float()
    for seed_index in range(len(predictions), len(SEEDS)):
        seed = SEEDS[seed_index]
        model = fit_diagnostic_seed(train_features, train_labels, seed)
        with torch.no_grad():
            probabilities = torch.softmax(model(test_x), dim=1)[:, 1].numpy()
        predictions.append(probabilities.astype(float).tolist())
        save_checkpoint(None)
        print(
            f"  {name}: completed diagnostic seed {seed_index + 1}/{len(SEEDS)}",
            flush=True,
        )
    return np.mean(np.asarray(predictions, dtype=np.float64), axis=0)


def fit_nonlinear_probe(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    test_features: np.ndarray,
    train_races: np.ndarray,
    validation_races: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_features).astype(np.float32)
    validation_x = scaler.transform(validation_features).astype(np.float32)
    test_x = scaler.transform(test_features).astype(np.float32)
    train_tensor = torch.from_numpy(train_x)
    validation_tensor = torch.from_numpy(validation_x)
    test_tensor = torch.from_numpy(test_x)
    train_y = torch.from_numpy(train_races).long()
    validation_y = torch.from_numpy(validation_races).long()

    set_seed(seed)
    model = RaceProbe(train_features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_function = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(30):
        model.train()
        permutation = torch.randperm(len(train_y))
        for start in range(0, len(train_y), BATCH_SIZE):
            batch = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_tensor[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(validation_tensor), validation_y
            ).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Non-linear probe did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(test_tensor), dim=1).numpy()


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def clinical_metrics(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities >= THRESHOLD
    group_positive_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []
    for race in range(len(RACE_NAMES)):
        mask = races == race
        group_labels = labels[mask]
        group_predictions = predictions[mask]
        positives = group_labels == 1
        negatives = group_labels == 0
        if not positives.any() or not negatives.any():
            raise ValueError(f"Race group {RACE_NAMES[race]} lacks an outcome class.")
        group_positive_rates.append(float(group_predictions.mean()))
        tprs.append(float(group_predictions[positives].mean()))
        fprs.append(float(group_predictions[negatives].mean()))
    tpr_gap = max(tprs) - min(tprs)
    fpr_gap = max(fprs) - min(fprs)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(np.mean(predictions == labels)),
        "dpd": float(max(group_positive_rates) - min(group_positive_rates)),
        "eod_max": float(max(tpr_gap, fpr_gap)),
        "tpr_gap": float(tpr_gap),
        "fpr_gap": float(fpr_gap),
    }


def per_group_metrics(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    predictions = probabilities >= THRESHOLD
    result: dict[str, Any] = {}
    for race, name in enumerate(RACE_NAMES):
        mask = races == race
        positives = mask & (labels == 1)
        negatives = mask & (labels == 0)
        positive_n = int(positives.sum())
        negative_n = int(negatives.sum())
        true_positive = int(np.count_nonzero(predictions & positives))
        false_negative = int(np.count_nonzero(~predictions & positives))
        false_positive = int(np.count_nonzero(predictions & negatives))
        tpr_ci = wilson_interval(true_positive, positive_n)
        fnr_ci = wilson_interval(false_negative, positive_n)
        fpr_ci = wilson_interval(false_positive, negative_n)
        result[name] = {
            "n": int(mask.sum()),
            "positive_n": positive_n,
            "negative_n": negative_n,
            "tpr": true_positive / positive_n,
            "tpr_wilson_95_ci": list(tpr_ci),
            "fnr": false_negative / positive_n,
            "fnr_wilson_95_ci": list(fnr_ci),
            "fpr": false_positive / negative_n,
            "fpr_wilson_95_ci": list(fpr_ci),
        }
    return result


def evaluate_predictions(
    labels: np.ndarray,
    races: np.ndarray,
    glaucoma_probabilities: np.ndarray,
    linear_race_probabilities: np.ndarray,
    nonlinear_race_probabilities: np.ndarray,
) -> dict[str, Any]:
    clinical = clinical_metrics(labels, races, glaucoma_probabilities)
    return {
        **clinical,
        "linear_probe_macro_auc": race_auc_from_probabilities(
            races, linear_race_probabilities
        ),
        "nonlinear_probe_macro_auc": race_auc_from_probabilities(
            races, nonlinear_race_probabilities
        ),
        "per_group": per_group_metrics(labels, races, glaucoma_probabilities),
    }


def bootstrap_deltas(
    labels: np.ndarray,
    races: np.ndarray,
    inlp_predictions: dict[str, dict[str, np.ndarray]],
    leace_predictions: dict[str, np.ndarray],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    n_test = len(labels)
    output: dict[str, Any] = {}
    for method, predictions in inlp_predictions.items():
        rng = np.random.default_rng(seed)
        draws: dict[str, list[float]] = {metric: [] for metric in PROBE_METRICS}
        for _ in range(n_resamples):
            indices = rng.integers(0, n_test, size=n_test)
            current = evaluate_predictions(
                labels[indices],
                races[indices],
                predictions["glaucoma"][indices],
                predictions["linear_race"][indices],
                predictions["nonlinear_race"][indices],
            )
            baseline = evaluate_predictions(
                labels[indices],
                races[indices],
                leace_predictions["glaucoma"][indices],
                leace_predictions["linear_race"][indices],
                leace_predictions["nonlinear_race"][indices],
            )
            for metric in PROBE_METRICS:
                draws[metric].append(current[metric] - baseline[metric])
        output[method] = {
            "delta_convention": "INLP minus LEACE",
            "n_resamples": n_resamples,
            "seed": seed,
            "confidence_level": 0.95,
            "intervals": {
                metric: {
                    "mean_delta": float(np.mean(values)),
                    "percentile_2_5": float(np.percentile(values, 2.5)),
                    "percentile_97_5": float(np.percentile(values, 97.5)),
                }
                for metric, values in draws.items()
            },
        }
    return output


def make_latex(results: dict[str, Any]) -> str:
    rows = []
    for name in results["method_order"]:
        metrics = results["methods"][name]["metrics"]
        rows.append(
            f"{name} & {metrics['auc']:.4f} & {metrics['accuracy']:.4f} "
            f"& {metrics['dpd']:.4f} & {metrics['eod_max']:.4f} "
            f"& {metrics['linear_probe_macro_auc']:.4f} "
            f"& {metrics['nonlinear_probe_macro_auc']:.4f} \\\\"
        )
    return (
        "% Fixed 0.50 threshold; race probe k selected on validation only.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Diagnostic and race-decodability results for validation-selected "
        "INLP projections and LEACE.}\n"
        "\\label{tab:inlp-converged}\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\hline\n"
        "Method & AUC & Accuracy & DPD & EOD$_{\\max}$ & Linear race AUC "
        "& Nonlinear race AUC \\\\\n"
        "\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-k", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if not 2 <= args.max_k <= MAX_ITERATIONS:
        parser.error("--max-k must be between 2 and 20.")
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive.")

    torch.set_num_threads(min(os.cpu_count() or 1, 8))
    set_seed(args.bootstrap_seed)
    cache_path = args.cache.resolve()
    features, labels, races = load_data(cache_path)
    fingerprint = {
        "cache_sha256": sha256_file(cache_path),
        "max_k": args.max_k,
        "convergence_auc": CONVERGENCE_AUC,
        "seeds": list(SEEDS),
        "diagnostic_head": (
            "Linear(1024,256)-LayerNorm-GELU-Dropout(0.2)-Linear(256,2)"
        ),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "threshold": THRESHOLD,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    if args.resume and args.checkpoint.is_file():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("input_fingerprint") != fingerprint:
            raise ValueError(
                "Checkpoint inputs do not match this run. Keep cache and settings "
                "unchanged or use another checkpoint file."
            )
        print(f"Resuming from {args.checkpoint}", flush=True)
    else:
        checkpoint = {
            "input_fingerprint": fingerprint,
            "inlp_state": None,
            "model_predictions": {},
            "probe_predictions": {},
            "method_metadata": {},
        }

    def save_checkpoint(_: Any = None) -> None:
        atomic_write_json(args.checkpoint, checkpoint)

    start_time = time.perf_counter()
    train_features = features["train"]
    validation_features = features["val"]
    test_features = features["test"]
    train_races = races["train"]

    print(
        "Selecting INLP iterations using validation race-probe macro-AUC only "
        f"(target <= {CONVERGENCE_AUC:.2f}, maximum k={args.max_k}).",
        flush=True,
    )
    inlp_path = run_inlp_path(
        features,
        races,
        args.max_k,
        checkpoint,
        save_checkpoint,
    )
    k2 = 2
    k_convergence = int(inlp_path["convergence_k"])
    k_closest = int(inlp_path["closest_k"])
    reached_target = any(
        row["validation_race_probe_macro_auc"] <= CONVERGENCE_AUC
        for row in inlp_path["history"]
    )
    convergence_label = "converged" if reached_target else "maximum"
    method_iterations = {
        "INLP k=2": (k2, "k2"),
        f"INLP k={k_convergence} ({convergence_label})": (
            k_convergence,
            "convergence",
        ),
        f"INLP k={k_closest} (closest to chance)": (k_closest, "closest"),
    }
    leace_mean, leace_projection = fit_leace(train_features, train_races)
    leace_features = {
        split: apply_leace(features[split], leace_mean, leace_projection)
        for split in SPLITS
    }
    transformed_methods: dict[str, dict[str, np.ndarray]] = {
        "LEACE": leace_features
    }
    for name, (_, snapshot_key) in method_iterations.items():
        basis = inlp_path["snapshots"][snapshot_key]
        transformed_methods[name] = {
            split: project_with_basis(features[split], basis)
            for split in SPLITS
        }

    model_predictions = checkpoint["model_predictions"]
    probe_predictions = checkpoint["probe_predictions"]
    evaluated: dict[str, dict[str, Any]] = {}
    bootstrap_inputs: dict[str, dict[str, np.ndarray]] = {}
    leace_bootstrap_inputs: dict[str, np.ndarray] | None = None
    test_labels = labels["test"]
    test_races = races["test"]

    for method_name, transformed in transformed_methods.items():
        glaucoma_probabilities = diagnostic_ensemble(
            method_name,
            transformed["train"],
            transformed["test"],
            labels["train"],
            model_predictions,
            save_checkpoint,
        )
        if method_name not in probe_predictions:
            linear_test = linear_probe_probabilities(
                transformed["train"], transformed["test"], train_races
            )
            nonlinear_test = fit_nonlinear_probe(
                transformed["train"],
                transformed["val"],
                transformed["test"],
                races["train"],
                races["val"],
            )
            probe_predictions[method_name] = {
                "linear_race": linear_test.tolist(),
                "nonlinear_race": nonlinear_test.tolist(),
            }
            save_checkpoint()
            print(f"  {method_name}: completed race probes.", flush=True)
        linear_test = np.asarray(
            probe_predictions[method_name]["linear_race"], dtype=np.float64
        )
        nonlinear_test = np.asarray(
            probe_predictions[method_name]["nonlinear_race"], dtype=np.float64
        )
        method_metrics = evaluate_predictions(
            test_labels,
            test_races,
            glaucoma_probabilities,
            linear_test,
            nonlinear_test,
        )
        evaluated[method_name] = {
            "iteration": (
                None
                if method_name == "LEACE"
                else method_iterations[method_name][0]
            ),
            "validation_race_probe_macro_auc": (
                None
                if method_name == "LEACE"
                else next(
                    row["validation_race_probe_macro_auc"]
                    for row in inlp_path["history"]
                    if row["k"] == method_iterations[method_name][0]
                )
            ),
            "metrics": method_metrics,
            "glaucoma_probabilities": glaucoma_probabilities.tolist(),
            "linear_race_probabilities": linear_test.tolist(),
            "nonlinear_race_probabilities": nonlinear_test.tolist(),
        }
        if method_name == "LEACE":
            leace_bootstrap_inputs = {
                "glaucoma": glaucoma_probabilities,
                "linear_race": linear_test,
                "nonlinear_race": nonlinear_test,
            }
        else:
            bootstrap_inputs[method_name] = {
                "glaucoma": glaucoma_probabilities,
                "linear_race": linear_test,
                "nonlinear_race": nonlinear_test,
            }

    if leace_bootstrap_inputs is None:
        raise RuntimeError("LEACE predictions are unavailable for paired bootstrap.")
    paired_bootstrap = bootstrap_deltas(
        test_labels,
        test_races,
        bootstrap_inputs,
        leace_bootstrap_inputs,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    history = inlp_path["history"]
    results: dict[str, Any] = {
        "method": "Validation-converged multiclass INLP versus LEACE",
        "selection": {
            "criterion": "Linear race-probe macro-AUC on validation <= 0.52",
            "max_k": args.max_k,
            "converged": reached_target,
            "k_at_convergence_or_maximum": k_convergence,
            "k2": 2,
            "k_closest_to_chance": k_closest,
            "closest_to_chance_definition": (
                "Minimum absolute distance from validation macro-AUC 0.50 among "
                "iterations k=1 through the stopping iteration."
            ),
            "validation_trace": history,
            "test_used_for_iteration_selection": False,
        },
        "configuration": {
            **fingerprint,
            "train_split_size": EXPECTED_SPLIT_SIZES["train"],
            "validation_split_size": EXPECTED_SPLIT_SIZES["val"],
            "test_split_size": EXPECTED_SPLIT_SIZES["test"],
            "test_race_counts": dict(zip(RACE_NAMES, EXPECTED_TEST_RACE_COUNTS)),
            "threshold_rule": "probability >= 0.50",
            "leace_fit_split": "train",
            "inlp_classifier": "multiclass LogisticRegression(lbfgs, max_iter=3000)",
            "linear_race_probe": (
                "StandardScaler fit on train; LogisticRegression(C=0.01, "
                "max_iter=1000, lbfgs); macro one-vs-rest AUC."
            ),
            "nonlinear_race_probe": (
                "Linear(1024,256)-BatchNorm-ReLU-Dropout(0.2)-Linear(256,3); "
                "AdamW, train-only fit, validation-loss checkpoint."
            ),
        },
        "method_order": list(evaluated),
        "methods": evaluated,
        "paired_bootstrap_vs_leace": paired_bootstrap,
        "runtime_seconds": time.perf_counter() - start_time,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_json, results)
    args.output_tex.write_text(make_latex(results), encoding="utf-8")

    print("\nFinal test metrics (fixed cutoff 0.50)")
    for method_name in results["method_order"]:
        metrics = results["methods"][method_name]["metrics"]
        print(
            f"{method_name:<38} AUC={metrics['auc']:.4f} "
            f"Acc={metrics['accuracy']:.4f} DPD={metrics['dpd']:.4f} "
            f"EOD_max={metrics['eod_max']:.4f} "
            f"linear-race-AUC={metrics['linear_probe_macro_auc']:.4f} "
            f"nonlinear-race-AUC={metrics['nonlinear_probe_macro_auc']:.4f}"
        )
        for race_name, group in metrics["per_group"].items():
            print(
                f"    {race_name:<8} "
                f"TPR={group['tpr']:.3f} "
                f"CI=[{group['tpr_wilson_95_ci'][0]:.3f},"
                f"{group['tpr_wilson_95_ci'][1]:.3f}] "
                f"FNR={group['fnr']:.3f} "
                f"CI=[{group['fnr_wilson_95_ci'][0]:.3f},"
                f"{group['fnr_wilson_95_ci'][1]:.3f}] "
                f"FPR={group['fpr']:.3f} "
                f"CI=[{group['fpr_wilson_95_ci'][0]:.3f},"
                f"{group['fpr_wilson_95_ci'][1]:.3f}]"
            )
    print("\nPaired bootstrap deltas (INLP minus LEACE; 95% percentile CIs)")
    for method_name, summary in paired_bootstrap.items():
        print(f"  {method_name}")
        for metric, interval in summary["intervals"].items():
            print(
                f"    {metric:<25} {interval['mean_delta']:+.4f} "
                f"[{interval['percentile_2_5']:+.4f}, "
                f"{interval['percentile_97_5']:+.4f}]"
            )
    print(f"\nSaved {args.output_json} and {args.output_tex}.")


if __name__ == "__main__":
    main()
