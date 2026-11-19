"""Compare train-only TF-IDF text models using validation AUC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

import stack_models


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "text_tuning.json"
LOGISTIC_REGRESSION_C_VALUES = (0.3, 1.0, 3.0, 10.0)
CALIBRATION_FOLDS = 5


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_train_validation_texts() -> tuple[
    dict[str, list[str]],
    dict[str, np.ndarray],
]:
    print("Loading only the seed-42 TRAIN and VALIDATION text rows.", flush=True)
    dataset = stack_models.load_dataset(
        "harvardairobotics/FairVLMed",
        "default",
        split="train",
    )
    missing = set((*stack_models.TEXT_FIELDS, "glaucoma")).difference(
        dataset.column_names
    )
    if missing:
        raise ValueError(f"HF train dataset is missing columns: {sorted(missing)}")
    expected_total = sum(stack_models.EXPECTED_SIZES.values())
    if len(dataset) != expected_total:
        raise ValueError(f"Expected {expected_total} HF rows, found {len(dataset)}.")

    subsets = torch.utils.data.random_split(
        range(len(dataset)),
        tuple(stack_models.EXPECTED_SIZES[split] for split in stack_models.SPLITS),
        generator=torch.Generator().manual_seed(42),
    )
    text_by_split: dict[str, list[str]] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    for split, subset_indices in zip(
        stack_models.SPLITS[:2],
        subsets[:2],
        strict=True,
    ):
        subset = dataset.select(list(subset_indices.indices))
        labels_by_split[split] = stack_models.labels_from_hub(subset)
        fields = {
            field: [str(value or "") for value in subset[field]]
            for field in stack_models.TEXT_FIELDS
        }
        text_by_split[split] = {
            field: fields[field] for field in stack_models.TEXT_FIELDS
        }

    for split, labels in labels_by_split.items():
        expected_size = stack_models.EXPECTED_SIZES[split]
        if len(labels) != expected_size:
            raise ValueError(
                f"Unexpected {split} label count: {len(labels)}; "
                f"expected {expected_size}."
            )
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"The {split} split must contain both label classes.")
    return text_by_split, labels_by_split


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=lambda text: stack_models.RACE_WORD_RE.sub(" ", text).lower(),
        ngram_range=(1, 2),
        min_df=3,
        max_features=50_000,
    )


def fit_feature_views(
    texts: dict[str, dict[str, list[str]]],
) -> dict[str, tuple[Any, Any]]:
    train_fields = texts["train"]
    val_fields = texts["val"]

    concatenated_train = stack_models.combine_texts(
        train_fields,
        list(range(len(next(iter(train_fields.values()))))),
    )
    concatenated_val = stack_models.combine_texts(
        val_fields,
        list(range(len(next(iter(val_fields.values()))))),
    )
    concatenated_vectorizer = make_vectorizer()
    concatenated_train_features = concatenated_vectorizer.fit_transform(
        concatenated_train
    )
    concatenated_val_features = concatenated_vectorizer.transform(concatenated_val)

    separate_train_features = []
    separate_val_features = []
    for field in stack_models.TEXT_FIELDS:
        vectorizer = make_vectorizer()
        separate_train_features.append(
            vectorizer.fit_transform(train_fields[field])
        )
        separate_val_features.append(vectorizer.transform(val_fields[field]))
        if any(
            stack_models.RACE_WORD_RE.search(term)
            for term in (vectorizer.vocabulary_ or {})
        ):
            raise AssertionError(
                f"Race words remain in the TF-IDF vocabulary for {field}."
            )
    separate_train = hstack(separate_train_features, format="csr")
    separate_val = hstack(separate_val_features, format="csr")

    if any(
        stack_models.RACE_WORD_RE.search(term)
        for term in (concatenated_vectorizer.vocabulary_ or {})
    ):
        raise AssertionError("Race words remain in the concatenated TF-IDF vocabulary.")
    return {
        "concatenated": (concatenated_train_features, concatenated_val_features),
        "separate_fields": (separate_train, separate_val),
    }


def run_tuning() -> dict[str, Any]:
    texts, labels = load_train_validation_texts()
    feature_views = fit_feature_views(texts)
    train_labels = labels["train"]
    validation_labels = labels["val"]

    variants: list[dict[str, Any]] = []
    for view_name, (train_features, validation_features) in feature_views.items():
        for c_value in LOGISTIC_REGRESSION_C_VALUES:
            classifier = LogisticRegression(
                C=c_value,
                max_iter=3000,
                solver="liblinear",
                random_state=stack_models.SEEDS[0],
            )
            classifier.fit(train_features, train_labels)
            validation_probabilities = classifier.predict_proba(
                validation_features
            )[:, 1]
            variants.append(
                {
                    "name": f"logistic_regression_{view_name}_C_{c_value:g}",
                    "model": "LogisticRegression",
                    "feature_view": view_name,
                    "C": c_value,
                    "validation_auc": float(
                        roc_auc_score(validation_labels, validation_probabilities)
                    ),
                }
            )

        calibrated_svc = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                random_state=stack_models.SEEDS[0],
                max_iter=10_000,
            ),
            method="sigmoid",
            cv=StratifiedKFold(
                n_splits=CALIBRATION_FOLDS,
                shuffle=True,
                random_state=stack_models.SEEDS[0],
            ),
        )
        calibrated_svc.fit(train_features, train_labels)
        validation_probabilities = calibrated_svc.predict_proba(
            validation_features
        )[:, 1]
        variants.append(
            {
                "name": f"calibrated_linear_svc_{view_name}",
                "model": "CalibratedClassifierCV(LinearSVC)",
                "feature_view": view_name,
                "C": 1.0,
                "calibration": {
                    "method": "sigmoid",
                    "folds": CALIBRATION_FOLDS,
                    "selection_data": "TRAIN only",
                },
                "validation_auc": float(
                    roc_auc_score(validation_labels, validation_probabilities)
                ),
            }
        )

    selected = max(
        enumerate(variants),
        key=lambda item: (item[1]["validation_auc"], -item[0]),
    )[1]
    return {
        "selection_metric": "validation ROC AUC",
        "selection_rule": "Highest validation AUC; ties favor earlier listed variant.",
        "training_split_size": len(train_labels),
        "validation_split_size": len(validation_labels),
        "test_split_used": False,
        "text_fields": list(stack_models.TEXT_FIELDS),
        "preprocessing": {
            "race_word_regex": stack_models.RACE_WORD_RE.pattern,
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_features": 50_000,
        },
        "logistic_regression_c_values": list(LOGISTIC_REGRESSION_C_VALUES),
        "variants": variants,
        "selected_variant": selected["name"],
        "selected_validation_auc": selected["validation_auc"],
    }


def main() -> None:
    results = run_tuning()
    for variant in results["variants"]:
        print(
            f"{variant['name']}: validation AUC="
            f"{variant['validation_auc']:.6f}",
            flush=True,
        )
    print(
        f"Selected: {results['selected_variant']} "
        f"(validation AUC={results['selected_validation_auc']:.6f})",
        flush=True,
    )
    atomic_write_json(RESULTS_PATH, results)
    print(f"Saved {RESULTS_PATH.name}.", flush=True)


if __name__ == "__main__":
    main()
