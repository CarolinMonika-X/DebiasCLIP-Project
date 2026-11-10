"""Benchmark INLP, LEACE, and race-label reweighting on cached joint features."""

from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cached_features.pt"
FIGURE_PATH = ROOT / "figures" / "fig_inlp_vs_leace.png"
SPLITS = ("train", "val", "test")
INLP_ITERATIONS = 10
GLAUCOMA_THRESHOLD = 0.5


def load_data() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    data = torch.load(CACHE_PATH, weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected split data in {CACHE_PATH}.")

    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    races: dict[str, np.ndarray] = {}
    for split in SPLITS:
        if split not in data:
            raise ValueError(f"Cached features are missing the {split!r} split.")
        values = data[split]
        image = values["img"].float().numpy()
        text = values["txt"].float().numpy()
        if image.ndim != 2 or text.ndim != 2:
            raise ValueError(f"Expected 2D image and text features in {split}.")
        if image.shape[1] != 512 or text.shape[1] != 512:
            raise ValueError(f"Expected 512-dimensional image and text features in {split}.")
        if image.shape[0] != text.shape[0]:
            raise ValueError(f"Image and text sample counts differ in {split}.")
        features[split] = np.concatenate((image, text), axis=1).astype(
            np.float32, copy=False
        )
        labels[split] = values["labels"].long().numpy()
        races[split] = values["races"].long().numpy()
        if labels[split].shape != (len(features[split]),):
            raise ValueError(f"Invalid glaucoma label shape in {split}.")
        if races[split].shape != labels[split].shape:
            raise ValueError(f"Race and glaucoma label counts differ in {split}.")
        if not np.isfinite(features[split]).all():
            raise ValueError(f"Non-finite joint feature values found in {split}.")

    if not np.array_equal(np.unique(races["train"]), np.unique(races["test"])):
        raise ValueError("Train and test race classes do not match.")
    if not np.array_equal(np.unique(labels["train"]), np.array([0, 1])):
        raise ValueError("Glaucoma labels are expected to be encoded as 0 and 1.")
    return features, labels, races


def fit_leace(
    features: np.ndarray, races: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit the closed-form LEACE projection on the training split."""
    matrix = features.astype(np.float64, copy=False)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    classes = np.unique(races)
    concepts = np.eye(len(classes), dtype=np.float64)[
        np.searchsorted(classes, races)
    ]
    centered_concepts = concepts - concepts.mean(axis=0)

    covariance = centered.T @ centered / len(matrix)
    cross_covariance = centered.T @ centered_concepts / len(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(
        1e-5 * float(eigenvalues.max()),
        np.finfo(np.float64).eps,
    )
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

    def transform(values: np.ndarray) -> np.ndarray:
        values64 = values.astype(np.float64, copy=False)
        return (values64 - (values64 - mean) @ projection.T).astype(np.float32)

    return transform


def fit_reweighting(races: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Balance each observed race-by-label group for the glaucoma classifier."""
    groups, inverse, counts = np.unique(
        np.stack((races, labels), axis=1),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if len(groups) < 2:
        raise ValueError("Race-by-label reweighting requires multiple groups.")
    weights = len(races) / (len(groups) * counts[inverse])
    return weights.astype(np.float64)


def race_auc(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_races: np.ndarray,
    test_races: np.ndarray,
) -> float:
    scaler = StandardScaler().fit(train_features)
    train = scaler.transform(train_features)
    test = scaler.transform(test_features)
    probe = RidgeClassifier(
        alpha=1.0,
        class_weight="balanced",
        solver="lsqr",
    )
    probe.fit(train, train_races)
    scores = probe.decision_function(test)
    classes = probe.classes_
    return float(
        np.mean(
            [
                roc_auc_score(test_races == race, scores[:, index])
                for index, race in enumerate(classes)
            ]
        )
    )


def evaluate_glaucoma(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    test_races: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    scaler = StandardScaler().fit(train_features)
    train = scaler.transform(train_features)
    test = scaler.transform(test_features)
    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(train, train_labels, sample_weight=sample_weight)
    positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    probabilities = classifier.predict_proba(test)[:, positive_index]
    predictions = probabilities > GLAUCOMA_THRESHOLD
    groups = np.unique(test_races)
    positive_rates = [
        float(predictions[test_races == group].mean()) for group in groups
    ]
    return {
        "glaucoma_auc": float(roc_auc_score(test_labels, probabilities)),
        "glaucoma_accuracy": float((predictions == test_labels).mean()),
        "dpd": float(max(positive_rates) - min(positive_rates)),
    }


def evaluate_method(
    name: str,
    iteration: int | None,
    train_features: np.ndarray,
    test_features: np.ndarray,
    labels: dict[str, np.ndarray],
    races: dict[str, np.ndarray],
    sample_weight: np.ndarray | None = None,
    timer_started: float | None = None,
) -> dict[str, float | int | str | None]:
    started = perf_counter() if timer_started is None else timer_started
    residual_auc = race_auc(
        train_features, test_features, races["train"], races["test"]
    )
    diagnostic = evaluate_glaucoma(
        train_features,
        test_features,
        labels["train"],
        labels["test"],
        races["test"],
        sample_weight,
    )
    return {
        "method": name,
        "iteration": iteration,
        "wall_clock_seconds": perf_counter() - started,
        "residual_race_auc": residual_auc,
        **diagnostic,
    }


def run_benchmark() -> list[dict[str, float | int | str | None]]:
    features, labels, races = load_data()
    scaler = StandardScaler().fit(features["train"])
    standardized = {
        split: scaler.transform(features[split]).astype(np.float32)
        for split in SPLITS
    }

    results: list[dict[str, float | int | str | None]] = []
    raw_started = perf_counter()
    results.append(
        evaluate_method(
            "Raw baseline",
            0,
            standardized["train"],
            standardized["test"],
            labels,
            races,
            timer_started=raw_started,
        )
    )

    directions: list[np.ndarray] = []
    basis = np.empty((standardized["train"].shape[1], 0), dtype=np.float64)
    inlp_started = perf_counter()
    print("Fitting INLP with RidgeClassifier (balanced race classes)...")
    for iteration in range(1, INLP_ITERATIONS + 1):
        projected_train = standardized["train"]
        if basis.shape[1]:
            projected_train = projected_train - (
                projected_train @ basis
            ) @ basis.T
        classifier = RidgeClassifier(
            alpha=1.0,
            class_weight="balanced",
            solver="lsqr",
        )
        classifier.fit(projected_train, races["train"])
        new_directions = classifier.coef_.T
        if basis.shape[1]:
            new_directions = new_directions - basis @ (
                basis.T @ new_directions
            )
        directions.append(new_directions)

        direction_matrix = np.concatenate(directions, axis=1)
        basis, singular_values, _ = np.linalg.svd(
            direction_matrix, full_matrices=False
        )
        rank_cutoff = (
            1e-10 * singular_values.max()
            if singular_values.size
            else np.inf
        )
        basis = basis[:, singular_values > rank_cutoff]

        projected_train = standardized["train"] - (
            standardized["train"] @ basis
        ) @ basis.T
        projected_test = standardized["test"] - (
            standardized["test"] @ basis
        ) @ basis.T
        results.append(
            evaluate_method(
                "INLP",
                iteration,
                projected_train,
                projected_test,
                labels,
                races,
                timer_started=inlp_started,
            )
        )
        print(
            f"  k={iteration:2d}: race AUC "
            f"{results[-1]['residual_race_auc']:.4f}, glaucoma AUC "
            f"{results[-1]['glaucoma_auc']:.4f}"
        )

    leace_started = perf_counter()
    leace_transform = fit_leace(features["train"], races["train"])
    leace_train = leace_transform(features["train"])
    leace_test = leace_transform(features["test"])
    results.append(
        evaluate_method(
            "LEACE",
            1,
            leace_train,
            leace_test,
            labels,
            races,
            timer_started=leace_started,
        )
    )

    reweight_started = perf_counter()
    sample_weights = fit_reweighting(races["train"], labels["train"])
    results.append(
        evaluate_method(
            "Sample re-weighting",
            None,
            features["train"],
            features["test"],
            labels,
            races,
            sample_weight=sample_weights,
            timer_started=reweight_started,
        )
    )
    return results


def save_plot(results: list[dict[str, float | int | str | None]]) -> None:
    inlp_results = [
        result
        for result in results
        if result["method"] in ("Raw baseline", "INLP")
    ]
    leace = next(result for result in results if result["method"] == "LEACE")
    reweighted = next(
        result for result in results if result["method"] == "Sample re-weighting"
    )
    inlp_x = np.arange(len(inlp_results))
    leace_x = len(inlp_results)
    reweight_x = leace_x + 1
    ticks = [*range(len(inlp_results)), leace_x, reweight_x]
    tick_labels = [
        "Raw / 0",
        *(str(iteration) for iteration in range(1, INLP_ITERATIONS + 1)),
        "LEACE",
        "Re-weight",
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(
        inlp_x,
        [result["residual_race_auc"] for result in inlp_results],
        marker="o",
        label="INLP",
    )
    axes[0].scatter(
        leace_x,
        leace["residual_race_auc"],
        marker="D",
        s=75,
        label="LEACE (one step)",
    )
    axes[0].scatter(
        reweight_x,
        reweighted["residual_race_auc"],
        marker="s",
        s=75,
        label="Sample re-weighting",
    )
    axes[0].set_title("Residual linear race decodability")
    axes[0].set_ylabel("Macro one-vs-rest ROC AUC")
    axes[0].set_xticks(ticks, tick_labels, rotation=35, ha="right")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        inlp_x,
        [result["glaucoma_auc"] for result in inlp_results],
        marker="o",
        label="INLP",
    )
    axes[1].scatter(
        leace_x,
        leace["glaucoma_auc"],
        marker="D",
        s=75,
        label="LEACE (one step)",
    )
    axes[1].scatter(
        reweight_x,
        reweighted["glaucoma_auc"],
        marker="s",
        s=75,
        label="Sample re-weighting",
    )
    axes[1].set_title("Glaucoma diagnostic performance")
    axes[1].set_ylabel("Test ROC AUC")
    axes[1].set_xticks(ticks, tick_labels, rotation=35, ha="right")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def print_summary(results: list[dict[str, float | int | str | None]]) -> None:
    headers = (
        "Method",
        "k",
        "Wall (s)",
        "Race AUC",
        "Glaucoma AUC",
        "Accuracy",
        "DPD",
    )
    rows = [
        (
            str(result["method"]),
            "-" if result["iteration"] is None else str(result["iteration"]),
            f"{result['wall_clock_seconds']:.2f}",
            f"{result['residual_race_auc']:.4f}",
            f"{result['glaucoma_auc']:.4f}",
            f"{result['glaucoma_accuracy']:.4f}",
            f"{result['dpd']:.4f}",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("\n" + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = run_benchmark()
    save_plot(results)
    print_summary(results)
    print(f"\nSaved comparison plot to {FIGURE_PATH}.")
    print(
        "Note: sample re-weighting changes glaucoma training weights, not the "
        "feature representation; its residual race AUC is therefore evaluated "
        "on the unchanged joint features."
    )


if __name__ == "__main__":
    main()
