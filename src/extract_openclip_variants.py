"""Extract resumable OpenCLIP image features for alternate fundus preprocessing."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import random_split

from dataset import FairVLMedDataset
from preview_preprocessing import (
    apply_clahe,
    disc_centered_panel,
    pad_to_square,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cached_features.pt"
DATASET_ROOT = ROOT / "data" / "FairVLMed" / "Dataset" / "dataset"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
SPLITS = ("train", "val", "test")
SPLIT_SEED = 42
CHUNK_SIZE = 256
FEATURE_DIM = 512
RACE_MAP = {"asian": 0, "black": 1, "white": 2}
GLAUCOMA_MAP = {"no": 0, "yes": 1}


def build_split_indices(n: int) -> dict[str, list[int]]:
    train_size = int(0.7 * n)
    validation_size = int(0.1 * n)
    test_size = n - train_size - validation_size
    subsets = random_split(
        range(n),
        (train_size, validation_size, test_size),
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    return {
        split: list(subset.indices)
        for split, subset in zip(SPLITS, subsets, strict=True)
    }


def labels_and_races(dataset: FairVLMedDataset) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    races: list[int] = []
    for row in range(len(dataset)):
        record = dataset.ds[row]
        glaucoma = str(record["glaucoma"]).lower()
        race = str(record["race"]).lower()
        if glaucoma not in GLAUCOMA_MAP:
            raise ValueError(f"Unknown glaucoma label {glaucoma!r} in dataset row {row}.")
        if race not in RACE_MAP:
            raise ValueError(f"Unknown race label {race!r} in dataset row {row}.")
        labels.append(GLAUCOMA_MAP[glaucoma])
        races.append(RACE_MAP[race])
    return np.asarray(labels, dtype=np.int64), np.asarray(races, dtype=np.int64)


def verify_cached_alignment(
    cached: dict[str, Any],
    labels: np.ndarray,
    races: np.ndarray,
    split_indices: dict[str, list[int]],
) -> None:
    for split in SPLITS:
        expected_labels = labels[split_indices[split]]
        expected_races = races[split_indices[split]]
        if split not in cached:
            raise ValueError(f"Base cache is missing split {split!r}.")
        cached_split = cached[split]
        for key in ("labels", "races"):
            if key not in cached_split:
                raise ValueError(f"Base cache split {split!r} is missing {key!r}.")

        cached_labels = cached_split["labels"].detach().cpu().numpy().reshape(-1)
        cached_races = cached_split["races"].detach().cpu().numpy().reshape(-1)
        if not np.array_equal(cached_labels.astype(np.int64), expected_labels):
            raise ValueError(
                f"Seed-{SPLIT_SEED} split label alignment failed for {split}; "
                "refusing to extract features."
            )
        if not np.array_equal(cached_races.astype(np.int64), expected_races):
            raise ValueError(
                f"Seed-{SPLIT_SEED} split race alignment failed for {split}; "
                "refusing to extract features."
            )
        print(
            f"Verified {split} labels/races alignment "
            f"({len(expected_labels)} rows).",
            flush=True,
        )


def preprocess_image(
    image: Image.Image,
    variant: str,
) -> tuple[Image.Image, bool]:
    if variant == "pad_clahe":
        return pad_to_square(apply_clahe(image)), False

    crop, _, detector_status = disc_centered_panel(image)
    if detector_status.startswith("fallback:"):
        return pad_to_square(apply_clahe(image)), True
    return crop, False


def chunk_path(output_dir: Path, split: str, start: int) -> Path:
    return output_dir / f"{split}_{start:05d}.npz"


def read_valid_chunk(
    path: Path,
    variant: str,
    split: str,
    start: int,
    expected_rows: int,
) -> tuple[np.ndarray, int] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            image_features = archive["img"]
            saved_variant = str(archive["variant"].item())
            saved_split = str(archive["split"].item())
            saved_start = int(archive["row_start"].item())
            saved_fallbacks = int(archive["fallback_count"].item())
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"Could not read existing chunk {path}: {error}") from error

    if (
        saved_variant != variant
        or saved_split != split
        or saved_start != start
        or image_features.shape != (expected_rows, FEATURE_DIM)
    ):
        return None
    if not np.isfinite(image_features).all():
        raise ValueError(f"Existing chunk contains non-finite features: {path}")
    if saved_fallbacks < 0 or saved_fallbacks > expected_rows:
        raise ValueError(f"Existing chunk has invalid fallback count: {path}")
    return image_features.astype(np.float32, copy=False), saved_fallbacks


def save_chunk(
    path: Path,
    variant: str,
    split: str,
    start: int,
    image_features: np.ndarray,
    fallback_count: int,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez(
            file,
            img=image_features.astype(np.float32, copy=False),
            variant=np.asarray(variant),
            split=np.asarray(split),
            row_start=np.asarray(start, dtype=np.int64),
            fallback_count=np.asarray(fallback_count, dtype=np.int64),
        )
    os.replace(temporary_path, path)


def encode_chunk(
    rows: list[int],
    variant: str,
    dataset: FairVLMedDataset,
    preprocess: Any,
    model: torch.nn.Module,
) -> tuple[np.ndarray, int]:
    batch: list[torch.Tensor] = []
    fallback_count = 0
    for row in rows:
        image, used_fallback = preprocess_image(
            dataset._load_image(dataset.ds[row]["filename"]),
            variant,
        )
        fallback_count += int(used_fallback)
        image_tensor = preprocess(image)
        if image_tensor.shape != (3, 224, 224):
            raise ValueError(
                f"OpenCLIP preprocessing returned {tuple(image_tensor.shape)}; "
                "expected (3, 224, 224)."
            )
        batch.append(image_tensor)

    images = torch.stack(batch).to(device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        features = model.encode_image(images)
        features = torch.nn.functional.normalize(features.float(), dim=-1)
    image_features = features.cpu().numpy()
    if image_features.shape != (len(rows), FEATURE_DIM):
        raise ValueError(
            f"Expected image features with shape ({len(rows)}, {FEATURE_DIM}), "
            f"got {image_features.shape}."
        )
    return image_features, fallback_count


def assemble_cache(
    output_path: Path,
    output_dir: Path,
    variant: str,
    split_indices: dict[str, list[int]],
    cached: dict[str, Any],
    limit: int,
) -> dict[str, int]:
    output: dict[str, dict[str, torch.Tensor]] = {}
    fallback_counts: dict[str, int] = {}
    for split in SPLITS:
        n_rows = min(limit, len(split_indices[split])) if limit else len(split_indices[split])
        features: list[np.ndarray] = []
        fallback_count = 0
        for start in range(0, n_rows, CHUNK_SIZE):
            expected_rows = min(CHUNK_SIZE, n_rows - start)
            path = chunk_path(output_dir, split, start)
            valid = read_valid_chunk(path, variant, split, start, expected_rows)
            if valid is None:
                raise FileNotFoundError(
                    f"Missing or incomplete feature chunk for {split} rows "
                    f"{start}:{start + expected_rows}: {path}"
                )
            chunk_features, chunk_fallbacks = valid
            features.append(chunk_features)
            fallback_count += chunk_fallbacks
        if not features:
            image_features = np.empty((0, FEATURE_DIM), dtype=np.float32)
        else:
            image_features = np.concatenate(features, axis=0)
        base_split = cached[split]
        output[split] = {
            "img": torch.from_numpy(image_features.copy()),
            "labels": base_split["labels"][:n_rows].detach().cpu().clone(),
            "races": base_split["races"][:n_rows].detach().cpu().clone(),
        }
        fallback_counts[split] = fallback_count

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output, temporary_path)
    os.replace(temporary_path, output_path)
    return fallback_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("pad_clahe", "disc_crop"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke-test limit per split; zero processes every row.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or a positive number.")

    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    torch.manual_seed(SPLIT_SEED)
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))

    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"Required cached feature file not found: {CACHE_PATH}")
    cached = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    dataset = FairVLMedDataset(split="train", root=str(DATASET_ROOT), strict=True)
    if len(dataset) != 10_000:
        raise ValueError(
            f"Expected 10,000 rows from FairVLMed train split, found {len(dataset)}."
        )

    split_indices = build_split_indices(len(dataset))
    labels, races = labels_and_races(dataset)
    verify_cached_alignment(cached, labels, races, split_indices)

    output_dir = ROOT / f"features_{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"cached_features_{args.variant}"
    if args.limit:
        output_name += "_smoke"
    output_path = ROOT / f"{output_name}.pt"

    print(
        f"Loading {MODEL_NAME}/{PRETRAINED} on CPU in eval mode "
        f"for variant={args.variant}.",
        flush=True,
    )
    model, _, clip_preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device="cpu",
    )
    model = model.to(device="cpu").eval()

    start_time = time.perf_counter()
    processed_cases = 0
    for split in SPLITS:
        rows = split_indices[split]
        n_rows = min(args.limit, len(rows)) if args.limit else len(rows)
        print(f"{split}: extracting {n_rows}/{len(rows)} rows.", flush=True)
        for start in range(0, n_rows, CHUNK_SIZE):
            chunk_rows = rows[start : min(start + CHUNK_SIZE, n_rows)]
            expected_rows = len(chunk_rows)
            path = chunk_path(output_dir, split, start)
            valid = read_valid_chunk(
                path,
                args.variant,
                split,
                start,
                expected_rows,
            )
            if valid is not None:
                _, fallbacks = valid
                processed_cases += expected_rows
                elapsed = time.perf_counter() - start_time
                print(
                    f"  {split} {start + expected_rows}/{n_rows}: resumed "
                    f"{expected_rows} cached cases; total {elapsed / processed_cases:.3f}s/case.",
                    flush=True,
                )
                continue

            chunk_started = time.perf_counter()
            image_features, fallbacks = encode_chunk(
                chunk_rows,
                args.variant,
                dataset,
                clip_preprocess,
                model,
            )
            save_chunk(
                path,
                args.variant,
                split,
                start,
                image_features,
                fallbacks,
            )
            processed_cases += expected_rows
            chunk_elapsed = time.perf_counter() - chunk_started
            elapsed = time.perf_counter() - start_time
            print(
                f"  {split} {start + expected_rows}/{n_rows}: "
                f"{expected_rows} cases in {chunk_elapsed:.1f}s "
                f"({chunk_elapsed / expected_rows:.3f}s/case); "
                f"overall {elapsed / processed_cases:.3f}s/case; "
                f"disc-crop fallbacks in chunk={fallbacks}.",
                flush=True,
            )

    fallback_counts = assemble_cache(
        output_path,
        output_dir,
        args.variant,
        split_indices,
        cached,
        args.limit,
    )
    elapsed = time.perf_counter() - start_time
    print(
        f"Saved {output_path}; splits="
        f"{ {split: len(result['img']) for split, result in torch.load(output_path, map_location='cpu', weights_only=True).items()} }",
        flush=True,
    )
    print(
        f"Disc-crop fallbacks: {sum(fallback_counts.values())} "
        f"(train/val/test={fallback_counts['train']}/"
        f"{fallback_counts['val']}/{fallback_counts['test']}).",
        flush=True,
    )
    print(
        f"Total extracted/resumed cases={processed_cases}; "
        f"elapsed={elapsed:.1f}s; overall={elapsed / max(processed_cases, 1):.3f}s/case.",
        flush=True,
    )


if __name__ == "__main__":
    main()
