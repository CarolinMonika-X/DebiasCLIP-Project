"""Probe glaucoma-label information in FairVLMed clinical text."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "notes_probe_results.json"
VOCABULARY_PATH = ROOT / "notes_probe_vocabularies.json"
SEED = 42
SPLIT_LENGTHS = (7000, 1000, 2000)
C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
TEXT_FIELDS = ("gpt4_summary", "note")
GLAUCOMA_MAP = {"no": 0, "yes": 1}

RACE_WORDS = (
    r"asian(?:[\s-]+american)?",
    r"black",
    r"white",
    r"caucasian",
    r"african(?:[\s-]+american)?",
    r"hispanic",
    r"latino",
    r"latina",
    r"latinx",
    r"indigenous",
    r"native[\s-]+american",
    r"race",
    r"racial",
    r"ethnic(?:ity)?",
)
RACE_WORD_RE = re.compile(
    r"\b(?:" + "|".join(RACE_WORDS) + r")\b",
    flags=re.IGNORECASE,
)
DIAGNOSIS_FEATURE_RE = re.compile(
    r"\bglaucom(?:a|atous)\b|\bdiagnos\w*\s+(?:with|of)\b",
    flags=re.IGNORECASE,
)

NOTE_PATTERNS = {
    "cup": r"\bcup\b",
    "c/d": r"\bc\s*/\s*d\b",
    "cup-to-disc": r"\bcup[\s-]+to[\s-]+disc\b",
    "IOP": r"\biop\b",
    "intraocular": r"\bintraocular\b",
    "pressure": r"\bpressure\b",
    "visual field": r"\bvisual\s+field\b",
    "RNFL": r"\brnfl\b",
    "OCT": r"\boct\b",
    "thinning": r"\bthinning\b",
    "optic nerve": r"\boptic\s+nerve\b",
    "suspect": r"\bsuspect\w*\b",
    "diagnos": r"\bdiagnos\w*\b",
    "treat": r"\btreat\w*\b",
    "drops": r"\bdrops?\b",
    "surgery": r"\bsurger(?:y|ies)\b",
    "family history": r"\bfamily\s+history\b",
}


def seed_rngs() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def remove_race_words(text: str) -> str:
    return RACE_WORD_RE.sub(" ", text).lower()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def label_array(rows: Any) -> np.ndarray:
    labels: list[int] = []
    for row_index, value in enumerate(rows["glaucoma"]):
        label = as_text(value).lower()
        if label not in GLAUCOMA_MAP:
            raise ValueError(
                f"Unknown glaucoma label {label!r} in selected row {row_index}."
            )
        labels.append(GLAUCOMA_MAP[label])
    return np.asarray(labels, dtype=np.int64)


def collect_subset(
    dataset: Any,
    indices: list[int],
) -> tuple[dict[str, list[str]], np.ndarray]:
    subset = dataset.select(indices)
    texts = {
        field: [as_text(value) for value in subset[field]]
        for field in TEXT_FIELDS
    }
    return texts, label_array(subset)


def make_vectorizer(
    vocabulary: dict[str, int] | None = None,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=remove_race_words,
        ngram_range=(1, 2),
        min_df=3,
        max_features=50_000,
        vocabulary=vocabulary,
    )


def make_classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        max_iter=3000,
        solver="liblinear",
        random_state=SEED,
    )


def binary_auc(labels: np.ndarray, scores: np.ndarray, name: str) -> float:
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{name} must contain both glaucoma classes to compute AUC.")
    return float(roc_auc_score(labels, scores))


def print_training_examples(
    texts: dict[str, list[str]],
    labels: np.ndarray,
    source_indices: list[int],
) -> None:
    rng = np.random.default_rng(SEED)
    print("Random TRAIN text examples (maximum 500 characters each):", flush=True)
    for label, class_name in ((0, "negative"), (1, "positive")):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < 4:
            raise ValueError(f"Need at least four {class_name} training rows.")
        selected = rng.choice(candidates, size=4, replace=False)
        for field in TEXT_FIELDS:
            print(f"  {field}, class={label} ({class_name}):", flush=True)
            for subset_position in selected:
                text = texts[field][int(subset_position)]
                excerpt = re.sub(r"\s+", " ", text)[:500]
                print(
                    f"    source_row={source_indices[int(subset_position)]}: "
                    f"{excerpt}",
                    flush=True,
                )


def pattern_frequencies(
    notes: list[str],
    labels: np.ndarray,
) -> dict[str, dict[str, dict[str, int | float]]]:
    results: dict[str, dict[str, dict[str, int | float]]] = {}
    for name, expression in NOTE_PATTERNS.items():
        pattern = re.compile(expression, flags=re.IGNORECASE)
        matches = np.asarray([bool(pattern.search(note)) for note in notes])
        per_class: dict[str, dict[str, int | float]] = {}
        for label, class_name in ((1, "positive"), (0, "negative")):
            class_rows = labels == label
            denominator = int(class_rows.sum())
            numerator = int((matches & class_rows).sum())
            per_class[class_name] = {
                "count": numerator,
                "total": denominator,
                "fraction": numerator / denominator if denominator else 0.0,
            }
        results[name] = per_class
        positive = per_class["positive"]
        negative = per_class["negative"]
        print(
            f"{name}: positive {positive['count']}/{positive['total']} "
            f"({positive['fraction']:.4f}); negative "
            f"{negative['count']}/{negative['total']} "
            f"({negative['fraction']:.4f})",
            flush=True,
        )
    return results


def top_coefficients(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    count: int = 30,
) -> dict[str, list[dict[str, str | float]]]:
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    positive_indices = np.argsort(coefficients)[-count:][::-1]
    negative_indices = np.argsort(coefficients)[:count]

    def format_terms(indices: np.ndarray) -> list[dict[str, str | float]]:
        return [
            {
                "term": str(feature_names[index]),
                "coefficient": float(coefficients[index]),
            }
            for index in indices
        ]

    return {
        "positive": format_terms(positive_indices),
        "negative": format_terms(negative_indices),
    }


def probe_one_field(
    field: str,
    train_texts: list[str],
    train_labels: np.ndarray,
    val_texts: list[str],
    val_labels: np.ndarray,
) -> tuple[dict[str, Any], TfidfVectorizer, LogisticRegression]:
    print(f"\nFitting TRAIN-only TF-IDF for {field}.", flush=True)
    vectorizer = make_vectorizer()
    train_matrix = vectorizer.fit_transform(train_texts)
    val_matrix = vectorizer.transform(val_texts)
    vocabulary = vectorizer.vocabulary_
    assert vocabulary is not None
    if any(RACE_WORD_RE.search(term) for term in vocabulary):
        raise AssertionError(f"Race-related tokens remain in the {field} vocabulary.")

    candidate_scores: list[dict[str, float]] = []
    candidate_models: list[LogisticRegression] = []
    for c_value in C_VALUES:
        model = make_classifier(c_value)
        model.fit(train_matrix, train_labels)
        val_auc = binary_auc(
            val_labels,
            model.predict_proba(val_matrix)[:, 1],
            f"{field} validation set",
        )
        candidate_scores.append({"C": c_value, "val_auc": val_auc})
        candidate_models.append(model)
        print(f"  C={c_value:g}: validation AUC={val_auc:.6f}", flush=True)

    best_index = max(
        range(len(C_VALUES)),
        key=lambda index: (candidate_scores[index]["val_auc"], -C_VALUES[index]),
    )
    best_c = C_VALUES[best_index]
    best_model = candidate_models[best_index]
    coefficient_terms = top_coefficients(vectorizer, best_model)
    direct_terms = [
        {
            "direction": direction,
            **term,
        }
        for direction in ("positive", "negative")
        for term in coefficient_terms[direction]
        if DIAGNOSIS_FEATURE_RE.search(str(term["term"]))
    ]

    all_features = vectorizer.get_feature_names_out()
    keep_features = [
        str(term)
        for term in all_features
        if not DIAGNOSIS_FEATURE_RE.search(str(term))
    ]
    if not keep_features:
        raise ValueError(f"Removing diagnosis terms left no {field} features.")
    reduced_vocabulary = {term: index for index, term in enumerate(keep_features)}
    reduced_vectorizer = make_vectorizer(vocabulary=reduced_vocabulary)
    reduced_train = reduced_vectorizer.fit_transform(train_texts)
    reduced_val = reduced_vectorizer.transform(val_texts)
    reduced_model = make_classifier(best_c)
    reduced_model.fit(reduced_train, train_labels)
    reduced_val_auc = binary_auc(
        val_labels,
        reduced_model.predict_proba(reduced_val)[:, 1],
        f"{field} validation set without diagnosis terms",
    )

    print(f"  Selected C={best_c:g} by validation AUC.", flush=True)
    print("  Top 30 positive coefficients:", flush=True)
    for item in coefficient_terms["positive"]:
        print(f"    {item['term']}: {item['coefficient']:.6f}", flush=True)
    print("  Top 30 negative coefficients:", flush=True)
    for item in coefficient_terms["negative"]:
        print(f"    {item['term']}: {item['coefficient']:.6f}", flush=True)
    print(
        f"  Top-coefficient direct diagnosis terms: {len(direct_terms)}",
        flush=True,
    )
    for item in direct_terms:
        print(
            f"    {item['direction']}: {item['term']} "
            f"({item['coefficient']:.6f})",
            flush=True,
        )
    print(
        f"  Validation AUC after removing direct diagnosis terms: "
        f"{reduced_val_auc:.6f} "
        f"({len(all_features) - len(keep_features)} vocabulary terms removed).",
        flush=True,
    )

    field_result: dict[str, Any] = {
        "selected_C": best_c,
        "C_selection_metric": "validation ROC AUC; ties favor smaller C",
        "validation_auc_by_C": candidate_scores,
        "validation_auc": candidate_scores[best_index]["val_auc"],
        "vocabulary_size": len(all_features),
        "top_30_coefficients": coefficient_terms,
        "top_direct_diagnosis_terms": direct_terms,
        "diagnosis_term_definition": (
            "TF-IDF features containing glaucoma/glaucomatous, or the direct "
            "diagnosis fragments diagnosed/diagnosis with/of."
        ),
        "validation_auc_without_diagnosis_terms": reduced_val_auc,
        "diagnosis_vocabulary_terms_removed": len(all_features) - len(keep_features),
    }
    return field_result, vectorizer, best_model


def main() -> None:
    seed_rngs()
    print("Loading FairVLMed train dataset and building the seed-42 split.", flush=True)
    dataset = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
    required_columns = {*TEXT_FIELDS, "glaucoma"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(f"FairVLMed dataset is missing columns: {sorted(missing_columns)}")

    n_rows = len(dataset)
    if n_rows != sum(SPLIT_LENGTHS):
        raise ValueError(
            f"Expected {sum(SPLIT_LENGTHS)} FairVLMed rows, found {n_rows}."
        )
    subsets = torch.utils.data.random_split(
        range(n_rows),
        SPLIT_LENGTHS,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_indices, val_indices, test_indices = (
        list(subset.indices) for subset in subsets
    )
    if (len(train_indices), len(val_indices), len(test_indices)) != SPLIT_LENGTHS:
        raise AssertionError("Seed-42 train/validation/test split sizes are incorrect.")

    train_texts, train_labels = collect_subset(dataset, train_indices)
    val_texts, val_labels = collect_subset(dataset, val_indices)
    if len(np.unique(train_labels)) != 2 or len(np.unique(val_labels)) != 2:
        raise ValueError("Both glaucoma classes must be present in train and validation.")

    print_training_examples(train_texts, train_labels, train_indices)
    print("\nTRAIN note keyword frequencies by glaucoma label:", flush=True)
    note_frequencies = pattern_frequencies(train_texts["note"], train_labels)

    results: dict[str, Any] = {
        "seed": SEED,
        "split": {
            "method": "torch.utils.data.random_split over HF train row indices",
            "sizes": {"train": len(train_indices), "val": len(val_indices), "test": len(test_indices)},
            "test_access_deferred_until_final_evaluation": True,
        },
        "feature_config": {
            "fields": list(TEXT_FIELDS),
            "vectorizer": "TF-IDF",
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_features": 50_000,
            "race_words_removed": list(RACE_WORDS),
            "classifier": "binary LogisticRegression, liblinear, max_iter=3000",
            "C_values": list(C_VALUES),
            "fit_policy": "vectorizer and classifiers fit on TRAIN rows only",
            "selection_policy": "choose C by VALIDATION ROC AUC",
        },
        "train_note_keyword_frequencies": note_frequencies,
        "fields": {},
    }
    fitted: dict[str, tuple[TfidfVectorizer, LogisticRegression]] = {}
    for field in TEXT_FIELDS:
        field_result, vectorizer, classifier = probe_one_field(
            field,
            train_texts[field],
            train_labels,
            val_texts[field],
            val_labels,
        )
        results["fields"][field] = field_result
        fitted[field] = (vectorizer, classifier)

    vocabulary_output = {
        field: {
            term: int(index)
            for term, index in sorted(
                vectorizer.vocabulary_.items(),
                key=lambda item: item[1],
            )
        }
        for field, (vectorizer, _) in fitted.items()
    }
    VOCABULARY_PATH.write_text(
        json.dumps(vocabulary_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        "\nFinal evaluation: accessing TEST rows once; no choices will be made from "
        "these results.",
        flush=True,
    )
    test_texts, test_labels = collect_subset(dataset, test_indices)
    test_aucs: dict[str, float] = {}
    for field, (vectorizer, classifier) in fitted.items():
        test_matrix = vectorizer.transform(test_texts[field])
        test_aucs[field] = binary_auc(
            test_labels,
            classifier.predict_proba(test_matrix)[:, 1],
            f"{field} test set",
        )
    results["test_auc"] = test_aucs
    results["test_evaluation"] = {
        "performed_once_after_C_selection": True,
        "models_fit_on": "TRAIN only",
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"TEST AUC (final print only): gpt4_summary={test_aucs['gpt4_summary']:.6f}; "
        f"note={test_aucs['note']:.6f}. "
        f"Saved {RESULTS_PATH.name} and {VOCABULARY_PATH.name}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
