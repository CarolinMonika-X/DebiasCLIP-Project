"""Compare race erasure and glaucoma performance across feature modalities."""

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cached_features.pt"
RESULTS_PATH = ROOT / "modality_ablation_results.json"
SPLITS = ("train", "val", "test")
RACE_CLASSES = 3
GLAUCOMA_THRESHOLD = 0.5
MLP_SEEDS = range(5)
RACE_CANDIDATES = (0.001, 0.01, 0.1, 1.0)


def load_features() -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Load image, text, and joint features plus labels and race codes."""
    data = torch.load(CACHE_PATH, weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a split dictionary in {CACHE_PATH}.")

    features: dict[str, dict[str, np.ndarray]] = {
        modality: {} for modality in ("image", "text", "joint")
    }
    labels: dict[str, np.ndarray] = {}
    races: dict[str, np.ndarray] = {}

    for split in SPLITS:
        if split not in data:
            raise ValueError(f"Cached features are missing the {split!r} split.")
        values = data[split]
        image = values["img"].float().numpy()
        text = values["txt"].float().numpy()
        y = values["labels"].long().numpy()
        race = values["races"].long().numpy()

        if image.ndim != 2 or image.shape[1] != 512:
            raise ValueError(f"{split} image features must have shape (n, 512).")
        if text.ndim != 2 or text.shape[1] != 512:
            raise ValueError(f"{split} text features must have shape (n, 512).")
        if image.shape[0] != text.shape[0] or y.shape != (image.shape[0],):
            raise ValueError(f"Inconsistent feature or label counts in {split}.")
        if race.shape != y.shape:
            raise ValueError(f"Inconsistent race and label counts in {split}.")
        if not np.isfinite(image).all() or not np.isfinite(text).all():
            raise ValueError(f"Non-finite feature values found in {split}.")

        features["image"][split] = image
        features["text"][split] = text
        features["joint"][split] = np.concatenate((image, text), axis=1)
        labels[split] = y
        races[split] = race

    if not np.array_equal(np.unique(races["train"]), np.arange(RACE_CLASSES)):
        raise ValueError("Training race labels must be encoded as 0, 1, and 2.")
    return features, labels, races


def fit_leace(
    features: np.ndarray, concepts: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a closed-form LEACE transform using only the supplied training data."""
    matrix = features.astype(np.float64, copy=False)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    concept_matrix = np.eye(RACE_CLASSES, dtype=np.float64)[concepts]
    centered_concepts = concept_matrix - concept_matrix.mean(axis=0)

    covariance = centered.T @ centered / len(matrix)
    cross_covariance = centered.T @ centered_concepts / len(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalue_floor = max(
        1e-5 * float(eigenvalues.max()), np.finfo(np.float64).eps
    )
    eigenvalues = np.clip(eigenvalues, eigenvalue_floor, None)
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

    def transform(values: np.ndarray) -> np.ndarray:
        values64 = values.astype(np.float64, copy=False)
        return (values64 - (values64 - mean) @ projection.T).astype(np.float32)

    return transform


def transform_splits(
    features: dict[str, np.ndarray],
    transform: Callable[[np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    return {split: transform(features[split]) for split in SPLITS}


def race_auc(
    features: dict[str, np.ndarray],
    races: dict[str, np.ndarray],
) -> float:
    """Tune a standardized linear race probe on validation and report test AUC."""
    scaler = StandardScaler().fit(features["train"])
    train = scaler.transform(features["train"])
    val = scaler.transform(features["val"])
    test = scaler.transform(features["test"])

    best_auc = -np.inf
    best_c = RACE_CANDIDATES[0]
    for candidate_c in RACE_CANDIDATES:
        classifier = LogisticRegression(C=candidate_c, max_iter=3000)
        classifier.fit(train, races["train"])
        val_probabilities = classifier.predict_proba(val)
        val_auc = roc_auc_score(
            races["val"],
            val_probabilities,
            multi_class="ovr",
            average="macro",
        )
        if val_auc > best_auc:
            best_auc = float(val_auc)
            best_c = candidate_c

    classifier = LogisticRegression(C=best_c, max_iter=3000)
    classifier.fit(train, races["train"])
    test_probabilities = classifier.predict_proba(test)
    return float(
        roc_auc_score(
            races["test"],
            test_probabilities,
            multi_class="ovr",
            average="macro",
        )
    )


def glaucoma_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    races: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities > GLAUCOMA_THRESHOLD
    groups = np.unique(races)
    positive_rates: list[float] = []
    true_positive_rates: list[float] = []
    false_positive_rates: list[float] = []

    for group in groups:
        group_mask = races == group
        group_positive = group_mask & (labels == 1)
        group_negative = group_mask & (labels == 0)
        if not group_positive.any() or not group_negative.any():
            raise ValueError(
                f"Race group {group} needs both label classes to compute EOD."
            )
        positive_rates.append(float(predictions[group_mask].mean()))
        true_positive_rates.append(float(predictions[group_positive].mean()))
        false_positive_rates.append(float(predictions[group_negative].mean()))

    equalized_odds_difference = max(
        max(true_positive_rates) - min(true_positive_rates),
        max(false_positive_rates) - min(false_positive_rates),
    )
    return {
        "accuracy": float((predictions == labels).mean()),
        "auc": float(roc_auc_score(labels, probabilities)),
        "dpd": float(max(positive_rates) - min(positive_rates)),
        "eod": float(equalized_odds_difference),
    }


def glaucoma_probabilities(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> np.ndarray:
    """Train standardized two-layer MLP heads and average test probabilities."""
    scaler = StandardScaler().fit(features["train"])
    train = scaler.transform(features["train"])
    test = scaler.transform(features["test"])
    seed_probabilities: list[np.ndarray] = []

    for seed in MLP_SEEDS:
        classifier = MLPClassifier(
            hidden_layer_sizes=(256,),
            alpha=1e-2,
            early_stopping=True,
            max_iter=200,
            random_state=seed,
        )
        classifier.fit(train, labels["train"])
        positive_class_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
        seed_probabilities.append(
            classifier.predict_proba(test)[:, positive_class_index]
        )

    return np.mean(seed_probabilities, axis=0)


def markdown_table(results: list[dict[str, object]]) -> str:
    headers = (
        "Modality",
        "Pre-Erasure Race AUC",
        "Post-Erasure Race AUC",
        "Glaucoma AUC",
        "Glaucoma Acc",
        "DPD",
        "EOD",
    )
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(
        ("---",) * len(headers)
    ) + " |"]
    for result in results:
        post_metrics = result["glaucoma_post_erasure"]
        rows.append(
            "| "
            + " | ".join(
                (
                    str(result["modality"]),
                    f"{result['race_auc_pre_erasure']:.4f}",
                    f"{result['race_auc_post_erasure']:.4f}",
                    f"{post_metrics['auc']:.4f}",
                    f"{post_metrics['accuracy']:.4f}",
                    f"{post_metrics['dpd']:.4f}",
                    f"{post_metrics['eod']:.4f}",
                )
            )
            + " |"
        )
    return "\n".join(rows)


def main() -> None:
    features, labels, races = load_features()
    results: list[dict[str, object]] = []

    for modality in ("image", "text", "joint"):
        raw_features = features[modality]
        erased_features = transform_splits(
            raw_features,
            fit_leace(raw_features["train"], races["train"]),
        )

        pre_erasure_auc = race_auc(raw_features, races)
        post_erasure_auc = race_auc(erased_features, races)
        pre_erasure_glaucoma = glaucoma_metrics(
            glaucoma_probabilities(raw_features, labels),
            labels["test"],
            races["test"],
        )
        post_erasure_glaucoma = glaucoma_metrics(
            glaucoma_probabilities(erased_features, labels),
            labels["test"],
            races["test"],
        )

        results.append(
            {
                "modality": modality,
                "feature_dim": int(raw_features["train"].shape[1]),
                "race_auc_pre_erasure": pre_erasure_auc,
                "race_auc_post_erasure": post_erasure_auc,
                "glaucoma_pre_erasure": pre_erasure_glaucoma,
                "glaucoma_post_erasure": post_erasure_glaucoma,
                "glaucoma_auc": post_erasure_glaucoma["auc"],
                "glaucoma_accuracy": post_erasure_glaucoma["accuracy"],
                "dpd": post_erasure_glaucoma["dpd"],
                "eod": post_erasure_glaucoma["eod"],
            }
        )
        print(
            f"Completed {modality} modality "
            f"({raw_features['train'].shape[1]} dimensions)."
        )

    table = markdown_table(results)
    output = {
        "race_auc_method": (
            "StandardScaler + LogisticRegression; C selected by validation "
            "macro one-vs-rest AUC; reported on test."
        ),
        "glaucoma_method": (
            "StandardScaler + two-layer MLP (256 hidden units), five-seed "
            "test-probability ensemble."
        ),
        "glaucoma_threshold": GLAUCOMA_THRESHOLD,
        "results": results,
        "markdown_table": table,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("\nGlaucoma metrics in the table are post-erasure.")
    print(table)
    print(f"\nSaved results to {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
