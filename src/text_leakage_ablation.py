"""Audit demographic leakage in text and modality-specific glaucoma features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cached_features.pt"
DEFAULT_FEATURE_CACHE = ROOT / ".text_leakage_scrubbed_features.pt"
DEFAULT_CHECKPOINT = ROOT / ".text_leakage_ablation_checkpoint.json"
DEFAULT_RESULTS = ROOT / "text_ablation_results.json"
DEFAULT_TABLE = ROOT / "tables" / "modality.tex"
DATASET_ID = "harvardairobotics/FairVLMed"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
SPLIT_SEED = 42
SPLITS = ("train", "val", "test")
RACE_NAMES = ("Asian", "Black", "White")
EXPECTED_SIZES = {"train": 7000, "val": 1000, "test": 2000}
EXPECTED_TEST_RACE_COUNTS = (169, 302, 1529)

SEEDS = (0, 100, 200, 300, 400)
HIDDEN_DIM = 256
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.50

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
ENCODE_BATCH_SIZE = 256
RACE_PROBE_SEED = 42
RACE_PROBE_EPOCHS = 30
RACE_PROBE_BATCH_SIZE = 128
RACE_PROBE_LEARNING_RATE = 1e-3
RACE_PROBE_WEIGHT_DECAY = 1e-3

RACE_TERMS = re.compile(
    r"\b(?:"
    r"non[\s-]?hispanic|african[\s-]+american|native[\s-]+american|"
    r"asian(?:[\s-]+american)?|caucasian|hispanic|latino|latina|latinx|"
    r"indigenous|biracial|multiracial|black|white|race|racial|ethnic\w*"
    r")\b",
    re.IGNORECASE,
)
GENDER_TERMS = re.compile(
    r"\b(?:"
    r"female|male|women|woman|men|man|girls?|boys?|"
    r"gentlemen|gentleman|ladies|lady|"
    r"gender|sex|transgender|nonbinary|non[\s-]?binary|"
    r"mrs?\.?|miss|ms\.?|"
    r"mother|mothers|father|fathers|"
    r"she|her|hers|he|him|his"
    r")\b",
    re.IGNORECASE,
)
AGE_TERMS = re.compile(
    r"\b(?:"
    r"\d{1,3}\s*(?:[- ]?\s*years?\s*[- ]?\s*old|"
    r"[- ]?\s*yrs?\.?\s*[- ]?\s*old|"
    r"[- ]?\s*years?\s+of\s+age|"
    r"[- ]?\s*y/?o\b|[- ]?yo\b)"
    r"|(?:aged?|age\s*[:=]?)\s*\d{1,3}"
    r"|(?:elderly|older\s+adult|young\s+adult|middle[\s-]?aged|senior)"
    r"|(?:in\s+)?\d{1,2}\s*[- ]?year\s*[- ]?old"
    r")\b",
    re.IGNORECASE,
)
WHITESPACE = re.compile(r"\s+")
PUNCTUATION_SPACES = re.compile(r"\s+([,.;:!?])")


class DiagnosticHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, 2),
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


def dataset_text_fingerprint(dataset: Any) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        for column in ("note", "gpt4_summary"):
            value = to_text(row.get(column))
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big"))
            digest.update(encoded)
    return digest.hexdigest()


def text_list_fingerprint(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def atomic_save_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_cached_features(
    path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cached features not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("Cached feature file must be a split dictionary.")
    data: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        if split not in raw:
            raise ValueError(f"Cached features are missing split {split!r}.")
        row = raw[split]
        if not {"img", "txt", "labels", "races"}.issubset(row):
            raise ValueError(f"{split} cache requires img, txt, labels, and races.")
        image = row["img"].float().numpy()
        text = row["txt"].float().numpy()
        labels = row["labels"].long().numpy()
        races = row["races"].long().numpy()
        if image.ndim != 2 or image.shape[1] != 512:
            raise ValueError(f"{split} image features must have shape (n, 512).")
        if text.ndim != 2 or text.shape[1] != 512:
            raise ValueError(f"{split} text features must have shape (n, 512).")
        if image.shape[0] != text.shape[0] or labels.shape != (len(image),):
            raise ValueError(f"Feature and label lengths disagree in {split}.")
        if races.shape != labels.shape:
            raise ValueError(f"Race and label lengths disagree in {split}.")
        if not np.isfinite(image).all() or not np.isfinite(text).all():
            raise ValueError(f"Non-finite embedding found in {split}.")
        data[split] = {
            "image": image.astype(np.float32, copy=False),
            "text": text.astype(np.float32, copy=False),
            "joint": np.concatenate((image, text), axis=1).astype(
                np.float32, copy=False
            ),
            "labels": labels,
            "races": races,
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
    return data, raw


def to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def scrub_text(text: str) -> str:
    scrubbed = RACE_TERMS.sub(" ", text)
    scrubbed = GENDER_TERMS.sub(" ", scrubbed)
    scrubbed = AGE_TERMS.sub(" ", scrubbed)
    scrubbed = WHITESPACE.sub(" ", scrubbed)
    scrubbed = PUNCTUATION_SPACES.sub(r"\1", scrubbed)
    return scrubbed.strip()


def scrub_dataset_columns(
    dataset: Any,
) -> tuple[list[str], list[str], dict[str, Any]]:
    required = {"gpt4_summary", "note", "glaucoma", "race"}
    columns = set(dataset.column_names)
    missing = required - columns
    if missing:
        raise ValueError(f"FairVLMed source is missing columns: {sorted(missing)}")
    scrubbed_summaries: list[str] = []
    scrubbed_notes: list[str] = []
    changed_summary = 0
    changed_note = 0
    fields_changed = 0
    changed_cases = 0
    for row in dataset:
        summary = to_text(row["gpt4_summary"])
        note = to_text(row["note"])
        clean_summary = scrub_text(summary)
        clean_note = scrub_text(note)
        summary_changed = clean_summary != summary
        note_changed = clean_note != note
        changed_summary += summary_changed
        changed_note += note_changed
        fields_changed += int(summary_changed) + int(note_changed)
        changed_cases += summary_changed or note_changed
        scrubbed_summaries.append(clean_summary)
        scrubbed_notes.append(clean_note)
    return scrubbed_summaries, scrubbed_notes, {
        "total_cases": len(dataset),
        "cases_with_any_field_changed": int(changed_cases),
        "summary_cases_changed": int(changed_summary),
        "note_cases_changed": int(changed_note),
        "changed_fields_total": int(fields_changed),
        "summary_column": "gpt4_summary",
        "note_column": "note",
    }


def encode_texts(
    texts: list[str],
    batch_size: int,
    device: torch.device,
    model: nn.Module,
    open_clip: Any,
) -> np.ndarray:
    try:
        model = model.to(device).eval()
    except AttributeError as error:
        raise RuntimeError("The configured OpenCLIP encoder is invalid.") from error
    encoded: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = open_clip.tokenize(texts[start : start + batch_size]).to(device)
            embeddings = model.encode_text(tokens).float()
            embeddings = embeddings / embeddings.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            encoded.append(embeddings.cpu())
            print(
                f"  encoded scrubbed summaries {min(start + batch_size, len(texts))}"
                f"/{len(texts)}",
                flush=True,
            )
    if not encoded:
        raise ValueError("No summary text available to encode.")
    result = torch.cat(encoded, dim=0).numpy().astype(np.float32, copy=False)
    if result.shape != (len(texts), 512):
        raise ValueError(f"OpenCLIP returned unexpected text feature shape {result.shape}.")
    return result


def load_or_encode_scrubbed_features(
    args: argparse.Namespace,
    cached: dict[str, dict[str, np.ndarray]],
    cached_tensors: dict[str, Any],
    dataset: Any,
    device: torch.device,
    input_fingerprint: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    summaries, notes, scrub_counts = scrub_dataset_columns(dataset)
    if len(summaries) != sum(EXPECTED_SIZES.values()):
        raise ValueError(
            "The raw FairVLMed row count does not equal the concatenated cache size."
        )

    race_map = {"asian": 0, "black": 1, "white": 2}
    label_map = {"no": 0, "yes": 1}
    try:
        dataset_labels = np.asarray(
            [
                label_map[to_text(row["glaucoma"]).strip().lower()]
                for row in dataset
            ],
            dtype=np.int64,
        )
        dataset_races = np.asarray(
            [race_map[to_text(row["race"]).strip().lower()] for row in dataset],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(f"Unsupported source label value: {error}") from error
    subsets = random_split(
        range(len(dataset)),
        [EXPECTED_SIZES[split] for split in SPLITS],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    split_indices = {
        split: np.asarray(subset.indices, dtype=np.int64)
        for split, subset in zip(SPLITS, subsets)
    }
    for split in SPLITS:
        indexes = split_indices[split]
        assert np.array_equal(
            dataset_labels[indexes], cached[split]["labels"]
        ), f"Source/cache glaucoma-label mismatch in {split}."
        assert np.array_equal(
            dataset_races[indexes], cached[split]["races"]
        ), f"Source/cache race-label mismatch in {split}."

    if args.feature_cache.is_file():
        feature_cache = torch.load(
            args.feature_cache, map_location="cpu", weights_only=True
        )
        if feature_cache.get("input_fingerprint") != input_fingerprint:
            raise ValueError(
                "Scrubbed-feature cache fingerprint mismatch; use another path "
                "or omit --resume."
            )
        encoded = feature_cache["encoded_by_split"]
        counts = feature_cache["scrub_counts"]
        print(f"Loaded scrubbed text embeddings from {args.feature_cache}.", flush=True)
    else:
        encoded = {}
        counts = scrub_counts
    if counts != scrub_counts:
        raise ValueError("Scrubbed text cache contains different scrub counts.")
    partial: dict[str, Any] = {
        "input_fingerprint": input_fingerprint,
        "encoded_by_split": encoded,
        "scrub_counts": counts,
        "split_sizes": EXPECTED_SIZES,
    }
    atomic_save_torch(args.feature_cache, partial)
    missing_splits = [split for split in SPLITS if split not in encoded]
    open_clip = None
    encoder: nn.Module | None = None
    if missing_splits:
        try:
            import open_clip as open_clip_module
        except ImportError as error:
            raise RuntimeError(
                "open_clip_torch is required to re-encode scrubbed summaries."
            ) from error
        open_clip = open_clip_module
        encoder, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        encoder = encoder.to(device).eval()
    for split in SPLITS:
        if split in encoded:
            continue
        if encoder is None or open_clip is None:
            raise RuntimeError("OpenCLIP encoder was not initialized.")
        indexes = split_indices[split]
        scrubbed_embeddings = encode_texts(
            [summaries[index] for index in indexes],
            args.encode_batch_size,
            device,
            encoder,
            open_clip,
        )
        encoded[split] = torch.from_numpy(scrubbed_embeddings)
        partial["encoded_by_split"] = encoded
        atomic_save_torch(args.feature_cache, partial)

    scrubbed: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        text_values = encoded[split]
        if isinstance(text_values, torch.Tensor):
            text = text_values.cpu().numpy().astype(np.float32, copy=False)
        else:
            text = np.asarray(text_values, dtype=np.float32)
        if text.shape != (EXPECTED_SIZES[split], 512):
            raise ValueError(
                f"Scrubbed text features have unexpected shape for {split}: {text.shape}."
            )
        image = cached[split]["image"]
        scrubbed[split] = {
            "image": image,
            "text": text,
            "joint": np.concatenate((image, text), axis=1).astype(
                np.float32, copy=False
            ),
            "labels": cached[split]["labels"],
            "races": cached[split]["races"],
        }
        assert np.isfinite(text).all()
    del cached_tensors
    return scrubbed, counts, {
        "source_dataset": DATASET_ID,
        "source_config": DATASET_CONFIG,
        "source_split": DATASET_SPLIT,
        "split_reconstruction": (
            "torch.utils.data.random_split with seed 42 and lengths "
            "[7000, 1000, 2000], verified against cached labels and races."
        ),
        "text_encoder_input": "scrubbed gpt4_summary only",
        "scrubbed_note_used_for_embedding": False,
        "scrubbed_note_handling": (
            "The note field is scrubbed and counted; the existing cached text "
            "pipeline encodes gpt4_summary only, so note was not concatenated."
        ),
        "scrubbed_summary_sha256": text_list_fingerprint(summaries),
        "scrubbed_note_sha256": text_list_fingerprint(notes),
        "scrub_counts": counts,
    }


def fit_leace(
    train_features: np.ndarray, train_races: np.ndarray
) -> Any:
    matrix = train_features.astype(np.float64, copy=False)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    concepts = np.eye(len(RACE_NAMES), dtype=np.float64)[train_races]
    centered_concepts = concepts - concepts.mean(axis=0)
    covariance = centered.T @ centered / len(matrix)
    cross_covariance = centered.T @ centered_concepts / len(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(
        eigenvalues,
        max(1e-5 * float(eigenvalues.max()), np.finfo(np.float64).eps),
        None,
    )
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

    def transform(features: np.ndarray) -> np.ndarray:
        values = features.astype(np.float64, copy=False)
        return (values - (values - mean) @ projection.T).astype(np.float32)

    return transform


def train_diagnostic_seed(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> dict[str, list[float]]:
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
        indexes = torch.randperm(len(train_y))
        for start in range(0, len(train_y), BATCH_SIZE):
            batch = indexes[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(train_x[batch]), train_y[batch]
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.inference_mode():
        val_probabilities = torch.softmax(
            model(torch.from_numpy(val_features).float()), dim=1
        )[:, 1]
        test_probabilities = torch.softmax(
            model(torch.from_numpy(test_features).float()), dim=1
        )[:, 1]
    return {
        "val": val_probabilities.numpy().astype(float).tolist(),
        "test": test_probabilities.numpy().astype(float).tolist(),
    }


def fit_diagnostic_ensemble(
    name: str,
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    checkpoint: dict[str, Any],
    persist: Any,
) -> dict[str, np.ndarray]:
    model_runs = checkpoint["models"].setdefault(name, [])
    while len(model_runs) < len(SEEDS):
        seed_index = len(model_runs)
        result = train_diagnostic_seed(
            features["train"],
            labels["train"],
            features["val"],
            features["test"],
            SEEDS[seed_index],
        )
        model_runs.append(result)
        persist()
        print(
            f"{name}: finished diagnostic seed {SEEDS[seed_index]} "
            f"({seed_index + 1}/{len(SEEDS)})",
            flush=True,
        )
    return {
        split: np.mean(
            [np.asarray(row[split], dtype=np.float64) for row in model_runs], axis=0
        )
        for split in ("val", "test")
    }


def cached_nonlinear_probe(
    name: str,
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    train_races: np.ndarray,
    val_races: np.ndarray,
    test_races: np.ndarray,
    checkpoint: dict[str, Any],
    persist: Any,
) -> dict[str, Any]:
    saved_probes = checkpoint.setdefault("nonlinear_probes", {})
    if name not in saved_probes:
        saved_probes[name] = nonlinear_race_probe_auc(
            train_features,
            val_features,
            test_features,
            train_races,
            val_races,
            test_races,
        )
        persist()
        print(f"Completed and checkpointed nonlinear race probe: {name}.", flush=True)
    return saved_probes[name]


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    predictions = probabilities >= THRESHOLD
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy_at_0.50": float(np.mean(predictions == labels)),
    }


def race_probe_auc(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_races: np.ndarray,
    test_races: np.ndarray,
) -> float:
    scaler = StandardScaler()
    train = scaler.fit_transform(train_features)
    test = scaler.transform(test_features)
    classifier = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs")
    classifier.fit(train, train_races)
    probabilities = classifier.predict_proba(test)
    order = np.argsort(classifier.classes_)
    return float(
        roc_auc_score(
            test_races,
            probabilities[:, order],
            labels=np.arange(len(RACE_NAMES)),
            multi_class="ovr",
            average="macro",
        )
    )


def nonlinear_race_probe_auc(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    train_races: np.ndarray,
    val_races: np.ndarray,
    test_races: np.ndarray,
) -> dict[str, Any]:
    scaler = StandardScaler()
    train = scaler.fit_transform(train_features).astype(np.float32)
    validation = scaler.transform(val_features).astype(np.float32)
    test = scaler.transform(test_features).astype(np.float32)
    train_x = torch.from_numpy(train)
    validation_x = torch.from_numpy(validation)
    test_x = torch.from_numpy(test)
    train_y = torch.from_numpy(train_races).long()
    validation_y = torch.from_numpy(val_races).long()

    set_seed(RACE_PROBE_SEED)
    model = RaceProbe(train_features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=RACE_PROBE_LEARNING_RATE,
        weight_decay=RACE_PROBE_WEIGHT_DECAY,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    for epoch in range(1, RACE_PROBE_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_y))
        for start in range(0, len(train_y), RACE_PROBE_BATCH_SIZE):
            batch = permutation[start : start + RACE_PROBE_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                nn.functional.cross_entropy(
                    model(validation_x), validation_y
                ).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Race probe did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(test_x), dim=1).numpy()
    class_order = np.arange(len(RACE_NAMES))
    auc = float(
        roc_auc_score(
            test_races,
            probabilities[:, class_order],
            labels=class_order,
            multi_class="ovr",
            average="macro",
        )
    )
    return {
        "macro_auc": auc,
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
    }


def metric_deltas(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in ("original", "scrubbed"):
        raw = classification_metrics(labels, probabilities[f"{condition}_raw"])
        leace = classification_metrics(labels, probabilities[f"{condition}_leace"])
        eod, dpd = fairness_metrics(
            labels,
            races,
            probabilities[f"{condition}_raw"],
        )
        leace_eod, leace_dpd = fairness_metrics(
            labels, races, probabilities[f"{condition}_leace"]
        )
        result[condition] = {
            "raw": {**raw, "eod_max": eod, "dpd": dpd},
            "leace": {**leace, "eod_max": leace_eod, "dpd": leace_dpd},
            "leace_minus_raw": {
                "d_eod_max": leace_eod - eod,
                "d_dpd": leace_dpd - dpd,
            },
        }
    result["scrubbing_change_in_leace_minus_raw"] = {
        key: result["scrubbed"]["leace_minus_raw"][key]
        - result["original"]["leace_minus_raw"][key]
        for key in ("d_eod_max", "d_dpd")
    }
    result["scrubbing_effect_by_method"] = {}
    for method in ("raw", "leace"):
        result["scrubbing_effect_by_method"][method] = {
            "d_auc": (
                result["scrubbed"][method]["auc"]
                - result["original"][method]["auc"]
            ),
            "d_eod_max": (
                result["scrubbed"][method]["eod_max"]
                - result["original"][method]["eod_max"]
            ),
            "d_dpd": (
                result["scrubbed"][method]["dpd"]
                - result["original"][method]["dpd"]
            ),
        }
    return result


def fairness_metrics(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    predictions = probabilities >= THRESHOLD
    positive_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []
    for race in range(len(RACE_NAMES)):
        group = races == race
        positive = group & (labels == 1)
        negative = group & (labels == 0)
        if not positive.any() or not negative.any():
            raise ValueError(f"Race group {RACE_NAMES[race]} lacks an outcome.")
        positive_rates.append(float(predictions[group].mean()))
        tprs.append(float(predictions[positive].mean()))
        fprs.append(float(predictions[negative].mean()))
    eod_max = max(max(tprs) - min(tprs), max(fprs) - min(fprs))
    dpd = max(positive_rates) - min(positive_rates)
    return float(eod_max), float(dpd)


def latex_table(results: dict[str, Any]) -> str:
    rows = []
    for modality in ("image", "text", "joint"):
        row = results["modalities"][modality]
        rows.append(
            f"{modality.title()} only & {row['glaucoma']['auc']:.4f} "
            f"& {row['glaucoma']['accuracy_at_0.50']:.4f} "
            f"& {row['linear_race_probe_macro_auc']:.4f} \\\\"
        )
    probe_changes = results["race_probe_changes"]
    fairness = results["fairness_comparison"]
    scrub_rows = []
    for method, display in (("raw", "Raw Dual"), ("leace", "LEACE-Race")):
        before = fairness["original"][method]
        after = fairness["scrubbed"][method]
        linear = probe_changes["linear_joint_macro_auc"]
        nonlinear = probe_changes["nonlinear_joint_macro_auc"]
        scrub_rows.append(
            f"{display} "
            f"& {linear['original_raw'] if method == 'raw' else linear['original_leace']:.4f} "
            f"& {linear['scrubbed_raw'] if method == 'raw' else linear['scrubbed_leace']:.4f} "
            f"& {nonlinear['original_raw'] if method == 'raw' else nonlinear['original_leace']:.4f} "
            f"& {nonlinear['scrubbed_raw'] if method == 'raw' else nonlinear['scrubbed_leace']:.4f} "
            f"& {before['eod_max']:.4f} & {after['eod_max']:.4f} "
            f"& {before['dpd']:.4f} & {after['dpd']:.4f} \\\\"
        )
    for name in ("Original Raw Dual", "Original LEACE-Race", "Scrubbed Raw Dual", "Scrubbed LEACE-Race"):
        row = results["joint_comparison"][name]
        rows.append(
            f"{name} & {row['glaucoma']['auc']:.4f} "
            f"& {row['glaucoma']['accuracy_at_0.50']:.4f} "
            f"& {row['linear_race_probe_macro_auc']:.4f} \\\\"
        )
    return (
        "% Five-seed MLP diagnostic head; race probes are fit on train and evaluated on test.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Modality and text-leakage ablation. Glaucoma accuracy uses a fixed "
        "0.50 cutoff.}\n"
        "\\label{tab:text-modality-ablation}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\hline\n"
        "Representation & Glaucoma AUC & Glaucoma accuracy & Linear race macro-AUC \\\\\n"
        "\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Effect of demographic-term scrubbing on joint-feature race "
        "decodability and test disparity.}\n"
        "\\label{tab:text-scrubbing-effect}\n"
        "\\begin{tabular}{lrrrrrrrr}\n"
        "\\hline\n"
        "Method & Linear AUC pre & Linear AUC post & Nonlinear AUC pre "
        "& Nonlinear AUC post & EOD pre & EOD post & DPD pre & DPD post \\\\\n"
        "\\hline\n"
        + "\n".join(scrub_rows)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.encode_batch_size < 1:
        parser.error("--encode-batch-size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable.")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(min(os.cpu_count() or 1, 8))
    cache_path = args.cache.resolve()
    data, cached_tensors = load_cached_features(cache_path)
    labels = {split: data[split]["labels"] for split in SPLITS}
    races = {split: data[split]["races"] for split in SPLITS}

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
    )
    if len(dataset) != sum(EXPECTED_SIZES.values()):
        raise ValueError(
            f"Expected 10,000 source rows, found {len(dataset)}; cannot align "
            "the scrubbed text with cached official splits."
        )
    raw_source_config = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "split_seed": SPLIT_SEED,
        "dataset_text_sha256": dataset_text_fingerprint(dataset),
    }
    scrubbed_input_fingerprint = {
        "cache_sha256": sha256_file(cache_path),
        **raw_source_config,
        "regex_revision": 1,
        "text_encoder": CLIP_MODEL,
        "text_pretrained": CLIP_PRETRAINED,
        "encode_batch_size": args.encode_batch_size,
    }
    checkpoint_fingerprint = {
        **scrubbed_input_fingerprint,
        "seeds": list(SEEDS),
        "head": (
            "Linear(d,256)-LayerNorm-GELU-Dropout(0.2)-Linear(256,2)"
        ),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "threshold": THRESHOLD,
        "race_probe_seed": RACE_PROBE_SEED,
    }
    if args.resume and args.checkpoint.is_file():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("input_fingerprint") != checkpoint_fingerprint:
            raise ValueError(
                "Checkpoint fingerprint mismatch. Keep inputs/settings unchanged "
                "or select a new checkpoint file."
            )
        print(f"Resuming from {args.checkpoint}.", flush=True)
    else:
        checkpoint = {
            "input_fingerprint": checkpoint_fingerprint,
            "models": {},
        }

    def persist() -> None:
        atomic_write_json(args.checkpoint, checkpoint)

    started = time.time()
    scrubbed_data, scrub_counts, source_metadata = load_or_encode_scrubbed_features(
        args,
        data,
        cached_tensors,
        dataset,
        device,
        scrubbed_input_fingerprint,
    )

    model_data: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "original": data,
        "scrubbed": scrubbed_data,
    }
    leace_transforms: dict[str, Any] = {}
    for condition in ("original", "scrubbed"):
        transform = fit_leace(
            model_data[condition]["train"]["joint"],
            races["train"],
        )
        leace_transforms[condition] = transform
        for split in SPLITS:
            model_data[condition][split]["leace_joint"] = transform(
                model_data[condition][split]["joint"]
            )

    model_predictions: dict[str, dict[str, np.ndarray]] = {}
    modalities: dict[str, dict[str, Any]] = {}
    for modality in ("image", "text", "joint"):
        predictions = fit_diagnostic_ensemble(
            f"modality_{modality}",
            {
                split: model_data["original"][split][modality]
                for split in SPLITS
            },
            labels,
            checkpoint,
            persist,
        )
        model_predictions[f"original_{modality}"] = predictions
        if modality == "joint":
            model_predictions["original_joint"] = predictions
        modalities[modality] = {
            "feature_dim": int(model_data["original"]["train"][modality].shape[1]),
            "glaucoma": classification_metrics(
                labels["test"], predictions["test"]
            ),
            "linear_race_probe_macro_auc": race_probe_auc(
                model_data["original"]["train"][modality],
                model_data["original"]["test"][modality],
                races["train"],
                races["test"],
            ),
            "validation_glaucoma_auc": float(
                roc_auc_score(labels["val"], predictions["val"])
            ),
            "seed_list": list(SEEDS),
        }
    for condition in ("original", "scrubbed"):
        for representation in ("joint", "leace_joint"):
            key = f"{condition}_{representation}"
            if condition == "original" and representation == "joint":
                continue
            predictions = fit_diagnostic_ensemble(
                key,
                {
                    split: model_data[condition][split][representation]
                    for split in SPLITS
                },
                labels,
                checkpoint,
                persist,
            )
            model_predictions[key] = predictions

    linear_probe_results: dict[str, float] = {}
    nonlinear_probe_results: dict[str, dict[str, Any]] = {}
    for condition in ("original", "scrubbed"):
        for representation in ("joint", "leace_joint"):
            key = f"{condition}_{representation}"
            linear_probe_results[key] = race_probe_auc(
                model_data[condition]["train"][representation],
                model_data[condition]["test"][representation],
                races["train"],
                races["test"],
            )
            nonlinear_probe_results[key] = cached_nonlinear_probe(
                key,
                model_data[condition]["train"][representation],
                model_data[condition]["val"][representation],
                model_data[condition]["test"][representation],
                races["train"],
                races["val"],
                races["test"],
                checkpoint,
                persist,
            )

    joint_comparison: dict[str, Any] = {}
    output_names = {
        "original_joint": "Original Raw Dual",
        "original_leace_joint": "Original LEACE-Race",
        "scrubbed_joint": "Scrubbed Raw Dual",
        "scrubbed_leace_joint": "Scrubbed LEACE-Race",
    }
    for key, name in output_names.items():
        probabilities = model_predictions[key]["test"]
        condition = "scrubbed" if key.startswith("scrubbed") else "original"
        representation = "leace_joint" if "leace" in key else "joint"
        joint_comparison[name] = {
            "glaucoma": {
                **classification_metrics(labels["test"], probabilities),
                "eod_max": fairness_metrics(
                    labels["test"], races["test"], probabilities
                )[0],
                "dpd": fairness_metrics(
                    labels["test"], races["test"], probabilities
                )[1],
            },
            "linear_race_probe_macro_auc": linear_probe_results[
                f"{condition}_{representation}"
            ],
            "nonlinear_race_probe_macro_auc": nonlinear_probe_results[
                f"{condition}_{representation}"
            ]["macro_auc"],
        }
    fairness_comparison = metric_deltas(
        labels["test"],
        races["test"],
        {
            "original_raw": model_predictions["original_joint"]["test"],
            "original_leace": model_predictions["original_leace_joint"]["test"],
            "scrubbed_raw": model_predictions["scrubbed_joint"]["test"],
            "scrubbed_leace": model_predictions["scrubbed_leace_joint"]["test"],
        },
    )
    race_probe_changes = {
        "linear_joint_macro_auc": {
            "original_raw": linear_probe_results["original_joint"],
            "scrubbed_raw": linear_probe_results["scrubbed_joint"],
            "scrubbing_delta": (
                linear_probe_results["scrubbed_joint"]
                - linear_probe_results["original_joint"]
            ),
            "original_leace": linear_probe_results["original_leace_joint"],
            "scrubbed_leace": linear_probe_results["scrubbed_leace_joint"],
            "scrubbing_delta_after_leace": (
                linear_probe_results["scrubbed_leace_joint"]
                - linear_probe_results["original_leace_joint"]
            ),
        },
        "nonlinear_joint_macro_auc": {
            "original_raw": nonlinear_probe_results["original_joint"]["macro_auc"],
            "scrubbed_raw": nonlinear_probe_results["scrubbed_joint"]["macro_auc"],
            "scrubbing_delta": (
                nonlinear_probe_results["scrubbed_joint"]["macro_auc"]
                - nonlinear_probe_results["original_joint"]["macro_auc"]
            ),
            "original_leace": nonlinear_probe_results["original_leace_joint"][
                "macro_auc"
            ],
            "scrubbed_leace": nonlinear_probe_results["scrubbed_leace_joint"][
                "macro_auc"
            ],
            "scrubbing_delta_after_leace": (
                nonlinear_probe_results["scrubbed_leace_joint"]["macro_auc"]
                - nonlinear_probe_results["original_leace_joint"]["macro_auc"]
            ),
        },
    }
    results: dict[str, Any] = {
        "method": "Text leakage and modality ablation on cached FairVLMed splits",
        "configuration": {
            **checkpoint_fingerprint,
            **source_metadata,
            "feature_normalization": (
                "Use cached normalized image/text embeddings as-is; scrubbed "
                "text vectors are L2-normalized OpenCLIP outputs."
            ),
            "diagnostic_head": (
                "Linear(d,256)-LayerNorm-GELU-Dropout(0.2)-Linear(256,2)"
            ),
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "threshold_rule": "probability >= 0.50",
            "model_selection": "Fixed epochs and settings; no test-based selection.",
            "test_used_for_model_selection": False,
            "test_race_counts": dict(zip(RACE_NAMES, EXPECTED_TEST_RACE_COUNTS)),
            "linear_race_probe": (
                "Train StandardScaler and LogisticRegression(C=0.01, max_iter=1000, "
                "lbfgs) on train; evaluate macro one-vs-rest AUC on test."
            ),
            "nonlinear_race_probe": (
                "StandardScaler fit on train; Linear(d,256)-BatchNorm-ReLU-"
                "Dropout(0.2)-Linear(256,3); AdamW(1e-3, 1e-3), 30 epochs; "
                "best validation cross-entropy checkpoint."
            ),
            "leace_fit_split": "train only, using race labels",
        },
        "scrub_counts": scrub_counts,
        "modalities": modalities,
        "joint_comparison": joint_comparison,
        "race_probe_changes": race_probe_changes,
        "nonlinear_race_probe_details": nonlinear_probe_results,
        "fairness_comparison": fairness_comparison,
        "runtime_seconds": time.time() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_json, results)
    args.output_tex.write_text(latex_table(results), encoding="utf-8")

    print("\nModality glaucoma and linear race probe results")
    for modality, row in modalities.items():
        glaucoma = row["glaucoma"]
        print(
            f"{modality:<8} glaucoma AUC={glaucoma['auc']:.4f} "
            f"accuracy={glaucoma['accuracy_at_0.50']:.4f} "
            f"linear race macro-AUC={row['linear_race_probe_macro_auc']:.4f}"
        )
    print(
        "\nScrubbing: changed "
        f"{scrub_counts['cases_with_any_field_changed']}/"
        f"{scrub_counts['total_cases']} cases; summaries "
        f"{scrub_counts['summary_cases_changed']}, notes "
        f"{scrub_counts['note_cases_changed']}."
    )
    print("\nRace probe macro-AUC changes after scrubbing")
    for name, values in race_probe_changes.items():
        print(
            f"{name}: raw {values['original_raw']:.4f} -> "
            f"{values['scrubbed_raw']:.4f} "
            f"(delta {values['scrubbing_delta']:+.4f}); "
            f"LEACE {values['original_leace']:.4f} -> "
            f"{values['scrubbed_leace']:.4f} "
            f"(delta {values['scrubbing_delta_after_leace']:+.4f})"
        )
    print("\nScrubbing change in LEACE-minus-Raw disparity")
    for key, value in fairness_comparison[
        "scrubbing_change_in_leace_minus_raw"
    ].items():
        print(f"{key}: {value:+.4f}")
    print(f"\nSaved {args.output_json} and {args.output_tex}.")


if __name__ == "__main__":
    main()
