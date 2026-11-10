"""Audit raw and LEACE model errors across race-by-sex test cohorts.

The cached feature file does not include sex or age. This script aligns the
local FairVLMed metadata to the cache using the seed-42 split from
``src/ablation_full.py`` and verifies race and glaucoma labels before analysis.
"""

import csv
from pathlib import Path
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cached_features.pt"
METADATA_PATH = ROOT / "data" / "FairVLMed" / "ReadMe" / "data_summary.csv"
FIGURE_PATH = ROOT / "figures" / "fig_intersectional_fairness.png"
SPLITS = ("train", "val", "test")
SPLIT_SEED = 42
SEEDS = range(5)
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
THRESHOLD = 0.5
RACE_NAMES = {0: "Asian", 1: "Black", 2: "White"}
RACE_CODES = {"asian": 0, "black": 1, "white": 2}
GLAUCOMA_CODES = {"no": 0, "yes": 1}
SEX_NAMES = {"female": "Female", "male": "Male"}
AGE_CUTOFF = 65


class Head(nn.Module):
    """The two-layer diagnostic head used by Model C and Model F2."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def load_cache() -> dict[str, dict[str, torch.Tensor]]:
    data = torch.load(CACHE_PATH, weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected split data in {CACHE_PATH}.")
    for split in SPLITS:
        if split not in data:
            raise ValueError(f"Cached features are missing the {split!r} split.")
        values = data[split]
        required = {"img", "txt", "labels", "races"}
        if not required.issubset(values):
            raise ValueError(
                f"The {split} cache must contain {sorted(required)}."
            )
        if values["img"].ndim != 2 or values["img"].shape[1] != 512:
            raise ValueError(f"Expected 512-dimensional image features in {split}.")
        if values["txt"].ndim != 2 or values["txt"].shape[1] != 512:
            raise ValueError(f"Expected 512-dimensional text features in {split}.")
    return data


def load_metadata(
    data: dict[str, dict[str, torch.Tensor]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], str]:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Demographic metadata not found at {METADATA_PATH}. "
            "The cached feature file contains race and labels only; provide "
            "the matching FairVLMed data_summary.csv to audit intersectional groups."
        )

    with METADATA_PATH.open(encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        metadata_fields = ("age", "gender", "race", "glaucoma")
        rows = [
            {field: row.get(field, "") for field in metadata_fields}
            for row in reader
        ]
        fields = set(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"No records found in {METADATA_PATH}.")

    lengths = [len(data[split]["labels"]) for split in SPLITS]
    if sum(lengths) != len(rows):
        raise ValueError(
            "Metadata row count does not match the cached split sizes: "
            f"{len(rows)} metadata rows vs {sum(lengths)} cached rows."
        )

    ordered_indices = torch.utils.data.random_split(
        range(len(rows)),
        lengths,
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    metadata_by_split = [
        [rows[index] for index in subset.indices] for subset in ordered_indices
    ]

    metadata_races: list[np.ndarray] = []
    metadata_labels: list[np.ndarray] = []
    for split, split_rows in zip(SPLITS, metadata_by_split):
        try:
            race = np.asarray(
                [RACE_CODES[row["race"].strip().lower()] for row in split_rows],
                dtype=np.int64,
            )
            labels = np.asarray(
                [
                    GLAUCOMA_CODES[row["glaucoma"].strip().lower()]
                    for row in split_rows
                ],
                dtype=np.int64,
            )
        except (KeyError, AttributeError) as error:
            raise ValueError(
                f"Metadata contains an unsupported race or glaucoma value in {split}."
            ) from error

        cached_races = data[split]["races"].long().numpy()
        cached_labels = data[split]["labels"].long().numpy()
        if not np.array_equal(race, cached_races):
            raise ValueError(
                f"Metadata race order does not match cached {split} rows. "
                "Refusing to assign demographic attributes to the wrong patients."
            )
        if not np.array_equal(labels, cached_labels):
            raise ValueError(
                f"Metadata glaucoma labels do not match cached {split} rows. "
                "Refusing to assign demographic attributes to the wrong patients."
            )
        metadata_races.append(race)
        metadata_labels.append(labels)

    if "gender" in fields and all(
        row.get("gender", "").strip().lower() in SEX_NAMES for row in metadata_by_split[-1]
    ):
        cohort_attribute = "sex"
    elif "age" in fields and all(
        _is_valid_age(row.get("age", "")) for row in metadata_by_split[-1]
    ):
        cohort_attribute = "age"
    else:
        raise ValueError(
            "Metadata has no complete sex labels or valid ages for the test split."
        )

    return (
        metadata_races[-1],
        metadata_labels[-1],
        metadata_by_split[-1],
        cohort_attribute,
    )


def _is_valid_age(value: str) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def make_cohort_labels(
    test_rows: list[dict[str, str]],
    races: np.ndarray,
    cohort_attribute: str,
) -> tuple[np.ndarray, list[str]]:
    if cohort_attribute == "sex":
        secondary = np.asarray(
            [row["gender"].strip().lower() for row in test_rows],
            dtype=object,
        )
        secondary_values = ("female", "male")
        secondary_names = SEX_NAMES
    else:
        ages = np.asarray([float(row["age"]) for row in test_rows])
        secondary = np.where(ages < AGE_CUTOFF, "under_65", "65_plus")
        secondary_values = ("under_65", "65_plus")
        secondary_names = {
            "under_65": f"Age < {AGE_CUTOFF}",
            "65_plus": f"Age {AGE_CUTOFF}+",
        }

    cohort_ids = np.full(len(races), -1, dtype=np.int64)
    cohort_names: list[str] = []
    cohort_id = 0
    for race_code, race_name in RACE_NAMES.items():
        for value in secondary_values:
            mask = (races == race_code) & (secondary == value)
            if not mask.any():
                raise ValueError(
                    f"Expected six non-empty race-by-{cohort_attribute} cohorts; "
                    f"{race_name} x {secondary_names[value]} has no test patients."
                )
            cohort_ids[mask] = cohort_id
            cohort_names.append(f"{race_name} {secondary_names[value]}")
            cohort_id += 1
    if np.any(cohort_ids < 0):
        raise ValueError("Some test patients could not be assigned to a cohort.")
    return cohort_ids, cohort_names


def fit_leace(
    features: np.ndarray, races: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit the closed-form race LEACE transform on training features only."""
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
    eigenvalue_floor = max(
        1e-5 * float(eigenvalues.max()),
        np.finfo(np.float64).eps,
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


def train_and_predict(
    train_features: np.ndarray,
    test_features: np.ndarray,
    labels: np.ndarray,
    name: str,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    train_tensor = torch.from_numpy(train_features).float()
    label_tensor = torch.from_numpy(labels).long()
    test_tensor = torch.from_numpy(test_features).float()

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = Head(train_features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for _ in range(EPOCHS):
            permutation = torch.randperm(len(train_tensor))
            for start in range(0, len(train_tensor), BATCH_SIZE):
                batch = permutation[start : start + BATCH_SIZE]
                loss = nn.functional.cross_entropy(
                    model(train_tensor[batch]),
                    label_tensor[batch],
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(test_tensor), dim=1)[:, 1]
        predictions.append(probabilities.numpy())
        print(f"  {name}: completed seed {seed}")

    return np.mean(predictions, axis=0)


def subgroup_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    cohort_ids: np.ndarray,
    cohort_names: list[str],
) -> list[dict[str, float | int | str | None]]:
    predictions = probabilities > THRESHOLD
    results: list[dict[str, float | int | str | None]] = []
    for cohort_id, cohort_name in enumerate(cohort_names):
        mask = cohort_ids == cohort_id
        group_labels = labels[mask]
        group_predictions = predictions[mask]
        positives = group_labels == 1
        negatives = group_labels == 0
        true_positive = int(np.sum(group_predictions & positives))
        false_negative = int(np.sum(~group_predictions & positives))
        false_positive = int(np.sum(group_predictions & negatives))
        true_negative = int(np.sum(~group_predictions & negatives))
        positive_count = int(positives.sum())
        negative_count = int(negatives.sum())
        auc = (
            float(roc_auc_score(group_labels, probabilities[mask]))
            if positive_count and negative_count
            else None
        )
        results.append(
            {
                "cohort": cohort_name,
                "n": int(mask.sum()),
                "positive_n": positive_count,
                "negative_n": negative_count,
                "tpr": true_positive / positive_count if positive_count else None,
                "fnr": false_negative / positive_count if positive_count else None,
                "fpr": false_positive / negative_count if negative_count else None,
                "accuracy": (
                    (true_positive + true_negative) / int(mask.sum())
                    if mask.any()
                    else None
                ),
                "auc": auc,
            }
        )
    return results


def _finite_values(
    rows: list[dict[str, float | int | str | None]], metric: str
) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if row[metric] is not None and np.isfinite(float(row[metric]))
    ]


def disparity_summary(
    raw: list[dict[str, float | int | str | None]],
    leace: list[dict[str, float | int | str | None]],
) -> dict[str, dict[str, float | bool]]:
    summary: dict[str, dict[str, float | bool]] = {}
    for metric in ("tpr", "fnr", "fpr", "accuracy", "auc"):
        raw_values = _finite_values(raw, metric)
        leace_values = _finite_values(leace, metric)
        raw_gap = max(raw_values) - min(raw_values)
        leace_gap = max(leace_values) - min(leace_values)
        summary[metric] = {
            "raw_max_min_gap": raw_gap,
            "leace_max_min_gap": leace_gap,
            "gap_reduced": leace_gap < raw_gap,
            "change": leace_gap - raw_gap,
        }
    return summary


def print_subgroup_table(
    raw: list[dict[str, float | int | str | None]],
    leace: list[dict[str, float | int | str | None]],
) -> None:
    print("\nIntersectional subgroup test metrics")
    print(
        f"{'Cohort':<22} {'Model':<7} {'n':>5} {'TPR':>7} {'FNR':>7} "
        f"{'FPR':>7} {'Accuracy':>9} {'AUC':>7}"
    )
    print("-" * 79)
    for raw_row, leace_row in zip(raw, leace):
        for model_name, row in (("C Raw", raw_row), ("F2 LEACE", leace_row)):
            print(
                f"{str(row['cohort']):<22} {model_name:<7} {int(row['n']):>5} "
                f"{_format_metric(row['tpr']):>7} "
                f"{_format_metric(row['fnr']):>7} "
                f"{_format_metric(row['fpr']):>7} "
                f"{_format_metric(row['accuracy']):>9} "
                f"{_format_metric(row['auc']):>7}"
            )


def _format_metric(value: float | int | str | None) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def print_disparity_summary(
    summary: dict[str, dict[str, float | bool]],
    raw: list[dict[str, float | int | str | None]],
    leace: list[dict[str, float | int | str | None]],
    cohort_attribute: str,
) -> None:
    print("\nMax-min disparity across intersectional cohorts")
    print(f"{'Metric':<12} {'C Raw':>9} {'F2 LEACE':>10} {'Change':>9} {'Reduced?':>10}")
    print("-" * 55)
    for metric, values in summary.items():
        print(
            f"{metric.upper():<12} {values['raw_max_min_gap']:>9.3f} "
            f"{values['leace_max_min_gap']:>10.3f} {values['change']:>+9.3f} "
            f"{str(values['gap_reduced']):>10}"
        )

    print("\nPer-cohort change (F2 LEACE minus C Raw)")
    for raw_row, leace_row in zip(raw, leace):
        changes = {
            metric: (
                float(leace_row[metric]) - float(raw_row[metric])
                if leace_row[metric] is not None and raw_row[metric] is not None
                else None
            )
            for metric in ("tpr", "fnr", "fpr", "accuracy", "auc")
        }
        compounding = (
            changes["fnr"] is not None
            and changes["fpr"] is not None
            and changes["fnr"] > 0
            and changes["fpr"] > 0
        )
        print(
            f"  {raw_row['cohort']:<22} "
            f"dTPR={_format_delta(changes['tpr'])} "
            f"dFNR={_format_delta(changes['fnr'])} "
            f"dFPR={_format_delta(changes['fpr'])} "
            f"dAcc={_format_delta(changes['accuracy'])} "
            f"dAUC={_format_delta(changes['auc'])} "
            f"compounding_error_penalty={compounding}"
        )

    black_groups = [
        (raw_row, leace_row)
        for raw_row, leace_row in zip(raw, leace)
        if str(raw_row["cohort"]).startswith("Black ")
        and (
            cohort_attribute != "sex"
            or str(raw_row["cohort"]).endswith("Female")
        )
    ]
    for black_raw, black_leace in black_groups:
        black_changes = {
            metric: (
                float(black_leace[metric]) - float(black_raw[metric])
                if black_leace[metric] is not None and black_raw[metric] is not None
                else None
            )
            for metric in ("fnr", "fpr", "accuracy", "auc")
        }
        compounding = (
            black_changes["fnr"] is not None
            and black_changes["fpr"] is not None
            and black_changes["fnr"] > 0
            and black_changes["fpr"] > 0
        )
        print(
            f"\n{black_raw['cohort']} assessment: "
            f"dFNR={_format_delta(black_changes['fnr'])}, "
            f"dFPR={_format_delta(black_changes['fpr'])}, "
            f"dAccuracy={_format_delta(black_changes['accuracy'])}, "
            f"dAUC={_format_delta(black_changes['auc'])}, "
            f"compounding_error_penalty={compounding}. "
            "A compounding error penalty means both FNR and FPR increased."
        )


def _format_delta(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.3f}"


def plot_error_rates(
    raw: list[dict[str, float | int | str | None]],
    leace: list[dict[str, float | int | str | None]],
) -> None:
    cohort_names = [str(row["cohort"]) for row in raw]
    positions = np.arange(len(cohort_names))
    bar_height = 0.18
    offsets = (-1.5, -0.5, 0.5, 1.5)
    series = (
        (raw, "fnr", "C Raw FNR", "#4C78A8"),
        (raw, "fpr", "C Raw FPR", "#9ECAE9"),
        (leace, "fnr", "F2 LEACE FNR", "#E45756"),
        (leace, "fpr", "F2 LEACE FPR", "#F2A0A0"),
    )

    figure, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for (rows, metric, label, color), offset in zip(series, offsets):
        axis.barh(
            positions + offset * bar_height,
            [
                float(row[metric]) if row[metric] is not None else np.nan
                for row in rows
            ],
            height=bar_height,
            color=color,
            label=label,
        )
    axis.set_yticks(positions, cohort_names)
    axis.invert_yaxis()
    axis.set_xlim(left=0)
    axis.set_xlabel("Error rate")
    axis.set_title("Intersectional test error rates: C Raw vs F2 LEACE")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(ncol=2, loc="lower right")
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    data = load_cache()
    image = {
        split: data[split]["img"].float().numpy() for split in SPLITS
    }
    text = {
        split: data[split]["txt"].float().numpy() for split in SPLITS
    }
    labels = {
        split: data[split]["labels"].long().numpy() for split in SPLITS
    }
    races = {
        split: data[split]["races"].long().numpy() for split in SPLITS
    }
    joint = {
        split: np.concatenate((image[split], text[split]), axis=1).astype(
            np.float32
        )
        for split in SPLITS
    }

    print("Aligning demographic metadata with cached feature splits...")
    test_races, test_labels, test_rows, cohort_attribute = load_metadata(data)
    cohort_ids, cohort_names = make_cohort_labels(
        test_rows, test_races, cohort_attribute
    )
    print(
        f"Using six race-by-{cohort_attribute} cohorts; "
        "metadata race and glaucoma labels match all cached splits."
    )

    print("\nFitting Model C (raw joint features), five seeds...")
    raw_probabilities = train_and_predict(
        joint["train"], joint["test"], labels["train"], "Model C"
    )

    print("\nFitting the train-only joint LEACE projection...")
    erase_race = fit_leace(joint["train"], races["train"])
    leace_features = {
        split: erase_race(joint[split]) for split in SPLITS
    }
    print("\nFitting Model F2 (joint LEACE features), five seeds...")
    leace_probabilities = train_and_predict(
        leace_features["train"],
        leace_features["test"],
        labels["train"],
        "Model F2",
    )

    raw_metrics = subgroup_metrics(
        raw_probabilities, test_labels, cohort_ids, cohort_names
    )
    leace_metrics = subgroup_metrics(
        leace_probabilities, test_labels, cohort_ids, cohort_names
    )
    disparities = disparity_summary(raw_metrics, leace_metrics)

    print_subgroup_table(raw_metrics, leace_metrics)
    print_disparity_summary(
        disparities,
        raw_metrics,
        leace_metrics,
        cohort_attribute,
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_error_rates(raw_metrics, leace_metrics)
    print(f"\nSaved clustered error-rate plot to {FIGURE_PATH}.")


if __name__ == "__main__":
    main()
