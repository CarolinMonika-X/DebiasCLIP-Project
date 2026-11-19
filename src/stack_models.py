"""Train leakage-safe base models and a validation-selected OOF stack."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
OPENCLIP_CACHE = ROOT / "cached_features.pt"
BIOMEDCLIP_CACHE = ROOT / "cached_features_biomedclip.pt"
RESULTS_PATH = ROOT / "stack_results.json"
STACK_TEST_PROBS_PATH = ROOT / "stack_test_probs.npy"
BASE_TEST_PROBS_PATH = ROOT / "stack_base_test_probs.npz"
SPLITS = ("train", "val", "test")
EXPECTED_SIZES = {"train": 7000, "val": 1000, "test": 2000}
SEEDS = (0, 100, 200, 300, 400)
N_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
META_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
TEXT_C = 1.0
TEXT_FIELDS = ("gpt4_summary", "note")
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
    r"ethnic\w*",
)
RACE_WORD_RE = re.compile(
    r"\b(?:" + "|".join(RACE_WORDS) + r")\b",
    flags=re.IGNORECASE,
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required feature cache was not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Feature cache must contain split dictionaries: {path}")
    data: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        if split not in raw:
            raise ValueError(f"{path.name} is missing the {split!r} split.")
        values = raw[split]
        required = {"img", "labels", "races"}
        if not required.issubset(values):
            raise ValueError(
                f"{path.name}/{split} must contain img, labels, and races."
            )
        image = values["img"].detach().cpu().float().numpy()
        labels = values["labels"].detach().cpu().long().numpy().reshape(-1)
        races = values["races"].detach().cpu().long().numpy().reshape(-1)
        if image.ndim != 2 or image.shape[1] != 512:
            raise ValueError(f"Expected 512-D image features in {path.name}/{split}.")
        if image.shape[0] != len(labels) or labels.shape != races.shape:
            raise ValueError(f"Feature/label/race counts differ in {path.name}/{split}.")
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite image features in {path.name}/{split}.")
        data[split] = {"img": image, "labels": labels, "races": races}
        if "txt" in values:
            text = values["txt"].detach().cpu().float().numpy()
            if text.shape != image.shape or not np.isfinite(text).all():
                raise ValueError(f"Invalid text features in {path.name}/{split}.")
            data[split]["txt"] = text
    sizes = {split: len(data[split]["labels"]) for split in SPLITS}
    if sizes != EXPECTED_SIZES:
        raise ValueError(f"Unexpected split sizes in {path.name}: {sizes}")
    return data


def verify_alignment(
    candidate: dict[str, dict[str, np.ndarray]],
    canonical: dict[str, dict[str, np.ndarray]],
    name: str,
) -> None:
    for split in SPLITS:
        for key in ("labels", "races"):
            if not np.array_equal(candidate[split][key], canonical[split][key]):
                raise ValueError(
                    f"{name} {split} {key} do not align with cached_features.pt."
                )


def find_image_variants(
    canonical: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[str]]:
    variants: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    skipped: list[str] = []
    excluded = {OPENCLIP_CACHE.resolve(), BIOMEDCLIP_CACHE.resolve()}
    for path in sorted(ROOT.glob("cached_features_*.pt")):
        if path.resolve() in excluded:
            continue
        if "old" in path.stem.lower() or "biomedclip" in path.stem.lower():
            skipped.append(f"{path.name}: excluded legacy/non-variant cache")
            continue
        try:
            candidate = read_cache(path)
            if any("txt" in candidate[split] for split in SPLITS):
                skipped.append(f"{path.name}: contains text features; expected image-only")
                continue
            verify_alignment(candidate, canonical, path.name)
        except (ValueError, FileNotFoundError) as error:
            skipped.append(f"{path.name}: {error}")
            continue
        name = path.stem.removeprefix("cached_features_")
        variants[name] = candidate
    return variants, skipped


def labels_from_hub(dataset: Any) -> np.ndarray:
    label_map = {"no": 0, "yes": 1}
    labels: list[int] = []
    for index, value in enumerate(dataset["glaucoma"]):
        label = str(value).lower()
        if label not in label_map:
            raise ValueError(f"Unknown glaucoma label {label!r} at HF row {index}.")
        labels.append(label_map[label])
    return np.asarray(labels, dtype=np.int64)


def get_text_data() -> tuple[dict[str, list[str]], np.ndarray]:
    print("Loading HF text fields for the seed-42 split.", flush=True)
    dataset = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
    missing = set((*TEXT_FIELDS, "glaucoma")).difference(dataset.column_names)
    if missing:
        raise ValueError(f"HF train dataset is missing columns: {sorted(missing)}")
    if len(dataset) != sum(EXPECTED_SIZES.values()):
        raise ValueError(f"Expected 10,000 HF rows, found {len(dataset)}.")
    subsets = torch.utils.data.random_split(
        range(len(dataset)),
        (7000, 1000, 2000),
        generator=torch.Generator().manual_seed(42),
    )
    split_indices = {
        split: list(subset.indices)
        for split, subset in zip(SPLITS, subsets, strict=True)
    }
    labels: dict[str, np.ndarray] = {}
    texts: dict[str, list[str]] = {}
    for split in ("train", "val"):
        subset = dataset.select(split_indices[split])
        labels[split] = labels_from_hub(subset)
        cached_labels = canonical_labels[split]
        if not np.array_equal(cached_labels, labels[split]):
            raise ValueError(
                f"HF row labels do not align with cached_features.pt {split} split."
            )
        selected_fields = {
            field: [str(value or "") for value in subset[field]]
            for field in TEXT_FIELDS
        }
        texts[split] = combine_texts(
            selected_fields,
            list(range(len(split_indices[split]))),
        )
    return texts, labels["train"]


def combine_texts(texts: dict[str, list[str]], indices: list[int]) -> list[str]:
    return [
        "\n".join(texts[field][index] for field in TEXT_FIELDS)
        for index in indices
    ]


def fit_text_model(
    train_texts: list[str],
    train_labels: np.ndarray,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        preprocessor=lambda text: RACE_WORD_RE.sub(" ", text).lower(),
        ngram_range=(1, 2),
        min_df=3,
        max_features=50_000,
    )
    features = vectorizer.fit_transform(train_texts)
    if any(RACE_WORD_RE.search(term) for term in (vectorizer.vocabulary_ or {})):
        raise AssertionError("Race words remain in the TF-IDF vocabulary.")
    classifier = LogisticRegression(
        C=TEXT_C,
        max_iter=3000,
        solver="liblinear",
        random_state=SEEDS[0],
    )
    classifier.fit(features, train_labels)
    return vectorizer, classifier


def predict_text(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    texts: list[str],
) -> np.ndarray:
    return classifier.predict_proba(vectorizer.transform(texts))[:, 1]


def train_mlp_seed(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    predict_features: np.ndarray,
    seed: int,
    epochs: int,
) -> np.ndarray:
    model = train_mlp_model(train_features, train_labels, seed, epochs)
    return predict_mlp_model(model, predict_features)


def train_mlp_model(
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


def predict_mlp_model(
    model: DiagnosticHead,
    features: np.ndarray,
) -> np.ndarray:
    model.eval()
    predict_x = torch.from_numpy(np.ascontiguousarray(features)).float()
    with torch.inference_mode():
        probabilities = torch.softmax(model(predict_x), dim=1)[:, 1]
    return probabilities.cpu().numpy()


def metrics_at_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy_at_0_5": float(accuracy_score(labels, probabilities >= 0.5)),
        "accuracy_at_val_youden": float(
            accuracy_score(labels, probabilities >= threshold)
        ),
    }


def choose_youden_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    false_positive, true_positive, thresholds = roc_curve(labels, probabilities)
    valid = np.isfinite(thresholds)
    if not np.any(valid):
        raise ValueError("Validation ROC curve has no finite decision thresholds.")
    youden = true_positive - false_positive
    best = float(np.max(youden[valid]))
    best_indices = np.flatnonzero(valid & np.isclose(youden, best))
    selected = min(best_indices, key=lambda index: (abs(thresholds[index] - 0.5), thresholds[index]))
    return float(thresholds[selected])


def base_feature_matrices(
    cache: dict[str, dict[str, np.ndarray]],
    feature_type: str,
) -> dict[str, np.ndarray]:
    if feature_type == "joint":
        if any("txt" not in cache[split] for split in SPLITS):
            raise ValueError("Joint base model requires cached text embeddings.")
        return {
            split: np.concatenate(
                (cache[split]["img"], cache[split]["txt"]),
                axis=1,
            ).astype(np.float32, copy=False)
            for split in SPLITS
        }
    if feature_type == "image_plus_openclip_text":
        return {
            split: np.concatenate(
                (cache[split]["img"], canonical_text_features[split]),
                axis=1,
            ).astype(np.float32, copy=False)
            for split in SPLITS
        }
    return {split: cache[split]["img"] for split in SPLITS}


def build_base_model_specs(
    canonical: dict[str, dict[str, np.ndarray]],
    biomedclip: dict[str, dict[str, np.ndarray]],
    variants: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "openclip_joint": {
            "kind": "mlp",
            "features": base_feature_matrices(canonical, "joint"),
            "cache": "cached_features.pt",
            "feature_config": "OpenCLIP img + OpenCLIP txt",
        },
        "biomedclip_joint": {
            "kind": "mlp",
            "features": base_feature_matrices(biomedclip, "joint"),
            "cache": "cached_features_biomedclip.pt",
            "feature_config": "BiomedCLIP img + BiomedCLIP txt",
        },
    }
    for name, cache in variants.items():
        specs[f"image_variant_{name}_plus_openclip_text"] = {
            "kind": "mlp",
            "features": base_feature_matrices(cache, "image_plus_openclip_text"),
            "cache": f"cached_features_{name}.pt",
            "feature_config": f"{name} image + OpenCLIP txt",
        }
    return specs


def cv_oof_predictions(
    name: str,
    spec: dict[str, Any],
    train_labels: np.ndarray,
    train_texts: list[str],
    epochs: int,
) -> np.ndarray:
    print(f"Building five-fold OOF predictions for {name}.", flush=True)
    oof = np.full(len(train_labels), np.nan, dtype=np.float64)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEEDS[0])
    feature_matrices = spec.get("features")
    for fold, (fit_indices, holdout_indices) in enumerate(
        folds.split(np.zeros(len(train_labels)), train_labels)
    ):
        print(
            f"  fold {fold + 1}/{N_FOLDS}: train={len(fit_indices)}, "
            f"OOF={len(holdout_indices)}",
            flush=True,
        )
        if spec["kind"] == "text":
            vectorizer, classifier = fit_text_model(
                [train_texts[index] for index in fit_indices],
                train_labels[fit_indices],
            )
            oof[holdout_indices] = predict_text(
                vectorizer,
                classifier,
                [train_texts[index] for index in holdout_indices],
            )
        else:
            assert feature_matrices is not None
            fold_predictions: list[np.ndarray] = []
            for seed_index, seed in enumerate(SEEDS):
                fold_predictions.append(
                    train_mlp_seed(
                        feature_matrices["train"][fit_indices],
                        train_labels[fit_indices],
                        feature_matrices["train"][holdout_indices],
                        seed + (fold + 1) * 10_000,
                        epochs,
                    )
                )
                print(
                    f"    seed {seed_index + 1}/{len(SEEDS)} complete",
                    flush=True,
                )
            oof[holdout_indices] = np.mean(fold_predictions, axis=0)
    if not np.isfinite(oof).all():
        raise RuntimeError(f"OOF predictions for {name} are incomplete.")
    return oof


def fit_full_base_model(
    name: str,
    spec: dict[str, Any],
    train_labels: np.ndarray,
    train_texts: list[str],
    val_texts: list[str],
    epochs: int,
) -> tuple[dict[str, Any], np.ndarray]:
    print(f"Fitting full-TRAIN base model {name}.", flush=True)
    if spec["kind"] == "text":
        vectorizer, classifier = fit_text_model(train_texts, train_labels)
        val_probabilities = predict_text(vectorizer, classifier, val_texts)
        return {
            "kind": "text",
            "vectorizer": vectorizer,
            "classifier": classifier,
        }, val_probabilities

    feature_matrices = spec["features"]
    val_predictions: list[np.ndarray] = []
    models: list[DiagnosticHead] = []
    for seed_index, seed in enumerate(SEEDS):
        model = train_mlp_model(
            feature_matrices["train"],
            train_labels,
            seed,
            epochs,
        )
        models.append(model)
        val_predictions.append(predict_mlp_model(model, feature_matrices["val"]))
        print(f"  seed {seed_index + 1}/{len(SEEDS)} complete", flush=True)
    return {
        "kind": "mlp",
        "features": feature_matrices,
        "models": models,
    }, np.mean(val_predictions, axis=0)


def predict_full_base_test(
    model: dict[str, Any],
    test_texts: list[str],
) -> np.ndarray:
    if model["kind"] == "text":
        return predict_text(
            model["vectorizer"],
            model["classifier"],
            test_texts,
        )
    features = model["features"]
    predictions = [
        predict_mlp_model(seed_model, features["test"])
        for seed_model in model["models"]
    ]
    return np.mean(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Fixed MLP training epochs; does not use validation or test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional smoke-test row cap per split; zero uses full splits.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")
    results_path = (
        ROOT / "stack_results_smoke.json" if args.limit else RESULTS_PATH
    )
    probabilities_path = (
        ROOT / "stack_test_probs_smoke.npy" if args.limit else STACK_TEST_PROBS_PATH
    )
    base_probabilities_path = (
        ROOT / "stack_base_test_probs_smoke.npz"
        if args.limit
        else BASE_TEST_PROBS_PATH
    )

    set_seed(SEEDS[0])
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    global canonical_labels, canonical_text_features

    print("Loading canonical OpenCLIP and BiomedCLIP feature caches.", flush=True)
    canonical = read_cache(OPENCLIP_CACHE)
    biomedclip = read_cache(BIOMEDCLIP_CACHE)
    verify_alignment(biomedclip, canonical, BIOMEDCLIP_CACHE.name)
    canonical_labels = {
        split: canonical[split]["labels"]
        for split in SPLITS
    }
    canonical_text_features = {
        split: canonical[split]["txt"]
        for split in SPLITS
    }
    variants, skipped_variants = find_image_variants(canonical)
    print(
        f"Found aligned image-only variants: {list(variants)}; "
        f"skipped={skipped_variants}",
        flush=True,
    )

    hub_texts, hub_train_labels = get_text_data()
    train_labels = canonical["train"]["labels"]
    if not np.array_equal(train_labels, hub_train_labels):
        raise ValueError("Text training labels do not match cached OpenCLIP labels.")

    if args.limit:
        rng = np.random.default_rng(SEEDS[0])
        selected: list[int] = []
        for label in (0, 1):
            label_indices = np.flatnonzero(train_labels == label)
            selected.extend(
                rng.choice(
                    label_indices,
                    size=min(args.limit // 2, len(label_indices)),
                    replace=False,
                ).tolist()
            )
        if len(selected) < 10:
            raise ValueError("--limit must allow at least 10 stratified train rows.")
        selected = sorted(selected)
        train_labels = train_labels[selected]
        train_texts = [hub_texts["train"][index] for index in selected]
        for cache in [canonical, biomedclip, *variants.values()]:
            cache["train"] = {
                key: value[selected]
                for key, value in cache["train"].items()
            }
        for split in ("val", "test"):
            count = min(args.limit, len(canonical[split]["labels"]))
            for cache in [canonical, biomedclip, *variants.values()]:
                cache[split] = {
                    key: value[:count]
                    for key, value in cache[split].items()
                }
            canonical_text_features[split] = canonical[split]["txt"]
        canonical_text_features["train"] = canonical["train"]["txt"]
        validation_texts = hub_texts["val"][: min(args.limit, len(hub_texts["val"]))]
    else:
        train_texts = hub_texts["train"]
        validation_texts = hub_texts["val"]

    split_labels = {
        "train": train_labels,
        "val": canonical["val"]["labels"],
        "test": canonical["test"]["labels"],
    }
    if args.limit:
        for split in ("val", "test"):
            split_labels[split] = canonical[split]["labels"]

    text_spec: dict[str, Any] = {
        "kind": "text",
        "cache": "FairVLMed gpt4_summary + note",
        "feature_config": "race-word-filtered TF-IDF of gpt4_summary + note",
    }
    specs = build_base_model_specs(canonical, biomedclip, variants)
    specs["text_tfidf"] = text_spec
    model_specs = list(specs)
    print(f"Base models ({len(model_specs)}): {model_specs}", flush=True)

    oof_columns: list[np.ndarray] = []
    validation_columns: list[np.ndarray] = []
    full_models: dict[str, dict[str, Any]] = {}
    base_results: dict[str, dict[str, Any]] = {}
    for name in model_specs:
        spec = specs[name]
        oof = cv_oof_predictions(
            name,
            spec,
            train_labels,
            train_texts,
            args.epochs,
        )
        train_full_texts = train_texts
        base_validation_texts = validation_texts if spec["kind"] == "text" else []
        full_model, val_probabilities = fit_full_base_model(
            name,
            spec,
            train_labels,
            train_full_texts,
            base_validation_texts,
            args.epochs,
        )
        full_models[name] = full_model
        oof_columns.append(oof)
        validation_columns.append(val_probabilities)
        base_results[name] = {
            "feature_config": spec["feature_config"],
            "training_protocol": (
                "five-fold stratified OOF; 5-seed ensemble per fold; "
                "5-seed full-TRAIN ensemble"
                if spec["kind"] == "mlp"
                else "five-fold stratified OOF; fixed C=1.0 logistic regression"
            ),
        }

    oof_matrix = np.column_stack(oof_columns)
    validation_matrix = np.column_stack(validation_columns)
    if oof_matrix.shape != (len(train_labels), len(model_specs)):
        raise AssertionError("Unexpected OOF matrix shape.")
    if validation_matrix.shape != (len(split_labels["val"]), len(model_specs)):
        raise AssertionError("Unexpected validation matrix shape.")

    meta_c_results: list[dict[str, float]] = []
    meta_models: list[LogisticRegression] = []
    for c_value in META_C_VALUES:
        model = LogisticRegression(
            C=c_value,
            max_iter=3000,
            solver="lbfgs",
            random_state=SEEDS[0],
        )
        model.fit(oof_matrix, train_labels)
        val_probabilities = model.predict_proba(validation_matrix)[:, 1]
        val_auc = float(roc_auc_score(split_labels["val"], val_probabilities))
        meta_models.append(model)
        meta_c_results.append({"C": c_value, "val_auc": val_auc})
        print(f"Meta C={c_value:g}: validation AUC={val_auc:.6f}", flush=True)
    selected_index = max(
        range(len(META_C_VALUES)),
        key=lambda index: (meta_c_results[index]["val_auc"], -META_C_VALUES[index]),
    )
    selected_c = META_C_VALUES[selected_index]
    selected_meta = meta_models[selected_index]
    stack_val_probabilities = selected_meta.predict_proba(validation_matrix)[:, 1]
    val_youden_threshold = choose_youden_threshold(
        split_labels["val"],
        stack_val_probabilities,
    )
    print(
        f"Selected meta C={selected_c:g}; validation-Youden threshold="
        f"{val_youden_threshold:.6f}. Test evaluation starts now.",
        flush=True,
    )

    # Test probabilities are generated only after validation selection is complete.
    test_texts: list[str] | None = None
    if "text_tfidf" in full_models:
        raw_dataset_for_test = load_dataset(
            "harvardairobotics/FairVLMed",
            "default",
            split="train",
        )
        test_indices = list(
            torch.utils.data.random_split(
                range(len(raw_dataset_for_test)),
                (7000, 1000, 2000),
                generator=torch.Generator().manual_seed(42),
            )[2].indices
        )
        test_indices = test_indices[: len(split_labels["test"])]
        test_subset = raw_dataset_for_test.select(test_indices)
        test_labels_from_hub = labels_from_hub(test_subset)
        if not np.array_equal(test_labels_from_hub, split_labels["test"]):
            raise ValueError("HF test row labels do not align with cached features.")
        test_texts = combine_texts(
            {
                field: [str(value or "") for value in test_subset[field]]
                for field in TEXT_FIELDS
            },
            list(range(len(test_indices))),
        )

    test_columns: list[np.ndarray] = []
    for model_index, name in enumerate(model_specs):
        model = full_models[name]
        if model["kind"] == "text":
            if test_texts is None:
                raise RuntimeError("Text test rows were not loaded for final evaluation.")
            test_probabilities = predict_text(
                model["vectorizer"],
                model["classifier"],
                test_texts,
            )
        else:
            test_probabilities = predict_full_base_test(
                model,
                [],
            )
        test_columns.append(test_probabilities)
        base_results[name]["validation"] = metrics_at_thresholds(
            split_labels["val"],
            validation_columns[model_index],
            val_youden_threshold,
        )
        print(f"Completed test inference for {name}.", flush=True)

    test_matrix = np.column_stack(test_columns)
    stack_test_probabilities = selected_meta.predict_proba(test_matrix)[:, 1]
    base_results["stack"] = {
        "feature_config": f"logistic regression over {len(model_specs)} base OOF probabilities",
        "selected_meta_C": selected_c,
        "validation_C_sweep": meta_c_results,
        "validation_threshold": {
            "rule": "Youden J maximized on validation; ties nearest 0.5",
            "value": val_youden_threshold,
        },
        "validation": metrics_at_thresholds(
            split_labels["val"],
            stack_val_probabilities,
            val_youden_threshold,
        ),
    }
    for index, name in enumerate(model_specs):
        base_results[name]["test"] = metrics_at_thresholds(
            split_labels["test"],
            test_columns[index],
            val_youden_threshold,
        )
    base_results["stack"]["test"] = metrics_at_thresholds(
        split_labels["test"],
        stack_test_probabilities,
        val_youden_threshold,
    )
    np.save(probabilities_path, stack_test_probabilities.astype(np.float32))
    np.savez_compressed(
        base_probabilities_path,
        **{
            name: probabilities.astype(np.float32)
            for name, probabilities in zip(model_specs, test_columns, strict=True)
        },
        stack=stack_test_probabilities.astype(np.float32),
    )

    result: dict[str, Any] = {
        "config": {
            "split_sizes": {key: len(value) for key, value in split_labels.items()},
            "folds": N_FOLDS,
            "mlp_seeds": list(SEEDS),
            "mlp_epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "head": "Linear(d,256)-LayerNorm-GELU-Dropout(0.2)-Linear(256,2)",
            "cv": "StratifiedKFold(shuffle=True, random_state=0), train rows only",
            "text_fields": list(TEXT_FIELDS),
            "text_race_words_removed": list(RACE_WORDS),
            "text_vectorizer": {
                "ngram_range": [1, 2],
                "min_df": 3,
                "max_features": 50_000,
                "C": TEXT_C,
                "race_words_removed_in_preprocessor": True,
            },
            "meta_model": "LogisticRegression(lbfgs), fit on OOF probabilities",
            "meta_C_values": list(META_C_VALUES),
            "meta_C_selection": "highest validation AUC; ties favor smaller C",
            "decision_threshold": "validation Youden J; ties nearest 0.5",
            "test_evaluation_after_validation_selection": True,
            "test_probabilities_file": probabilities_path.name,
            "base_test_probabilities_file": base_probabilities_path.name,
            "image_variants": list(variants),
            "skipped_image_variant_caches": skipped_variants,
        },
        "base_models": base_results,
    }
    atomic_write_json(results_path, result)
    print(
        f"Saved {results_path.name}, {probabilities_path.name}, "
        f"and {base_probabilities_path.name}.",
        flush=True,
    )
    for name, values in base_results.items():
        print(
            f"{name}: VAL AUC={values['validation']['auc']:.4f}, "
            f"VAL acc@0.5={values['validation']['accuracy_at_0_5']:.4f}, "
            f"VAL acc@Youden={values['validation']['accuracy_at_val_youden']:.4f}; "
            f"TEST AUC={values['test']['auc']:.4f}, "
            f"TEST acc@0.5={values['test']['accuracy_at_0_5']:.4f}, "
            f"TEST acc@Youden={values['test']['accuracy_at_val_youden']:.4f}",
            flush=True,
        )


canonical_labels: dict[str, np.ndarray] = {}
canonical_text_features: dict[str, np.ndarray] = {}


if __name__ == "__main__":
    main()
