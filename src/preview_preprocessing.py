"""Create a visual preprocessing comparison for random training images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw
from torch.utils.data import random_split

from dataset import FairVLMedDataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "preprocessing_preview.png"
TOTAL_SPLIT_LENGTHS = (7000, 1000, 2000)
RACE_NAMES = {0: "Asian", 1: "Black", 2: "White"}
GLAUCOMA_NAMES = {0: "No glaucoma", 1: "Glaucoma"}
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"


def build_preprocess() -> tuple[Any, tuple[float, ...], tuple[float, ...]]:
    pretrained_cfg = open_clip.get_pretrained_cfg(CLIP_MODEL, CLIP_PRETRAINED)
    if not pretrained_cfg:
        raise RuntimeError(
            f"Could not find OpenCLIP preprocessing configuration for "
            f"{CLIP_MODEL}/{CLIP_PRETRAINED}."
        )
    preprocess = open_clip.image_transform(
        image_size=224,
        is_train=False,
        mean=pretrained_cfg["mean"],
        std=pretrained_cfg["std"],
        interpolation=pretrained_cfg["interpolation"],
        resize_mode=pretrained_cfg["resize_mode"],
    )
    return preprocess, tuple(pretrained_cfg["mean"]), tuple(pretrained_cfg["std"])


def display_preprocessed(
    image: Image.Image,
    preprocess: Any,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> np.ndarray:
    tensor = preprocess(image)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype)[:, None, None]
    std_tensor = torch.tensor(std, dtype=tensor.dtype)[:, None, None]
    rgb = (tensor * std_tensor + mean_tensor).clamp(0.0, 1.0)
    return (
        rgb.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    canvas = Image.new("RGB", (side, side), color=(0, 0, 0))
    canvas.paste(image, (left, top))
    return canvas.resize((224, 224), Image.Resampling.BICUBIC)


def apply_clahe(image: Image.Image) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)
    return Image.fromarray(enhanced, mode="L").convert("RGB")


def radial_vessel_support(
    blurred: np.ndarray,
    center_x: float,
    center_y: float,
) -> float:
    height, width = blurred.shape
    y_grid, x_grid = np.mgrid[:height, :width]
    dx = x_grid - center_x
    dy = y_grid - center_y
    radius = np.sqrt(dx * dx + dy * dy)
    inner_radius = 0.045 * width
    outer_radius = 0.22 * width
    annulus = (radius >= inner_radius) & (radius <= outer_radius)
    if not np.any(annulus):
        return 0.0

    background = cv2.GaussianBlur(
        blurred.astype(np.float32),
        (0, 0),
        sigmaX=max(5.0, width * 0.035),
    )
    darkness = np.maximum(background - blurred.astype(np.float32), 0.0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_norm = np.sqrt(grad_x * grad_x + grad_y * grad_y) + 1e-6
    radial_norm = radius + 1e-6

    # Vessel edges have a gradient normal to the vessel; radial vessel segments
    # therefore have gradients approximately perpendicular to the radius.
    normal_dot_radius = np.abs(
        (grad_x * dx + grad_y * dy) / (grad_norm * radial_norm)
    )
    radial_alignment = 1.0 - np.clip(normal_dot_radius, 0.0, 1.0)
    darkness_scale = float(np.percentile(darkness[annulus], 90))
    if darkness_scale <= 1e-6:
        return 0.0
    vesselness = np.clip(darkness / darkness_scale, 0.0, 1.0)
    return float(np.mean((vesselness * radial_alignment)[annulus]))


def find_disc_center(image: Image.Image) -> tuple[int, int, str]:
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    blur_sigma = max(1.5, min(height, width) * 0.006)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)

    border_x = int(round(width * 0.12))
    border_y = int(round(height * 0.12))
    interior = blurred[border_y : height - border_y, border_x : width - border_x]
    if interior.size == 0:
        raise ValueError(f"Image is too small for 12% border exclusion: {gray.shape}")

    candidates: list[dict[str, float]] = []
    image_area = float(height * width)
    for percentile in (99.5, 99.0, 98.5, 98.0):
        threshold = float(np.percentile(interior, percentile))
        binary = np.zeros_like(blurred, dtype=np.uint8)
        binary[blurred >= threshold] = 255
        allowed = np.zeros_like(binary, dtype=np.uint8)
        allowed[border_y : height - border_y, border_x : width - border_x] = 255
        binary = cv2.bitwise_and(binary, allowed)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        for component_id in range(1, count):
            x, y, box_width, box_height, area = (
                int(value) for value in stats[component_id]
            )
            area_fraction = area / image_area
            aspect_ratio = box_width / max(box_height, 1)
            if not (0.00015 <= area_fraction <= 0.035):
                continue
            if not (0.55 <= aspect_ratio <= 1.8):
                continue
            fill = area / max(box_width * box_height, 1)
            if fill < 0.28:
                continue

            center_x, center_y = centroids[component_id]
            distance_from_center = float(
                np.hypot(
                    (center_x - (width - 1) / 2),
                    (center_y - (height - 1) / 2),
                )
            )
            if distance_from_center > 0.33 * min(width, height):
                continue
            region = blurred[y : y + box_height, x : x + box_width]
            candidate_mean = float(region.mean())
            compactness = float(fill * np.exp(-abs(np.log(aspect_ratio))))
            centrality = float(
                np.exp(
                    -np.hypot(
                        (center_x - (width - 1) / 2) / width,
                        (center_y - (height - 1) / 2) / height,
                    )
                    / 0.25
                )
            )
            convergence = radial_vessel_support(blurred, center_x, center_y)
            candidates.append(
                {
                    "x": float(center_x),
                    "y": float(center_y),
                    "brightness": candidate_mean,
                    "compactness": compactness,
                    "centrality": centrality,
                    "convergence": convergence,
                }
            )

    if not candidates:
        center_x = (width - 1) / 2
        center_y = (height - 1) / 2
        radius = int(round(0.33 * min(width, height)))
        x0, x1 = max(border_x, int(center_x - radius)), min(
            width - border_x, int(center_x + radius + 1)
        )
        y0, y1 = max(border_y, int(center_y - radius)), min(
            height - border_y, int(center_y + radius + 1)
        )
        search_area = blurred[y0:y1, x0:x1]
        local_y, local_x = np.unravel_index(
            int(np.argmax(search_area)),
            search_area.shape,
        )
        return (
            int(local_x + x0),
            int(local_y + y0),
            "fallback: brightest point in central search area",
        )

    brightness_values = np.asarray(
        [candidate["brightness"] for candidate in candidates],
        dtype=np.float32,
    )
    convergence_values = np.asarray(
        [candidate["convergence"] for candidate in candidates],
        dtype=np.float32,
    )
    brightness_scale = float(np.ptp(brightness_values))
    convergence_scale = float(np.ptp(convergence_values))
    for candidate in candidates:
        brightness_rank = (
            (candidate["brightness"] - float(brightness_values.min()))
            / brightness_scale
            if brightness_scale > 1e-6
            else 1.0
        )
        convergence_rank = (
            (candidate["convergence"] - float(convergence_values.min()))
            / convergence_scale
            if convergence_scale > 1e-6
            else candidate["convergence"]
        )
        candidate["score"] = (
            0.30 * brightness_rank
            + 0.25 * candidate["compactness"]
            + 0.15 * convergence_rank
            + 0.30 * candidate["centrality"]
        )

    selected = max(candidates, key=lambda candidate: candidate["score"])
    return (
        int(round(selected["x"])),
        int(round(selected["y"])),
        "compact bright blob scored with radial vessel support",
    )


def disc_centered_panel(image: Image.Image) -> tuple[Image.Image, tuple[int, int], str]:
    center_x, center_y, reason = find_disc_center(image)
    width, height = image.size
    side = max(1, int(round(width * 0.35)))
    left = center_x - side // 2
    top = center_y - side // 2
    right = left + side
    bottom = top + side

    crop = Image.new("RGB", (side, side), color=(0, 0, 0))
    intersection = image.crop(
        (
            max(0, left),
            max(0, top),
            min(width, right),
            min(height, bottom),
        )
    )
    crop.paste(intersection, (max(0, -left), max(0, -top)))
    return apply_clahe(crop).resize((224, 224), Image.Resampling.BICUBIC), (
        center_x,
        center_y,
    ), reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8, help="Number of images (1-8).")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.limit <= 8:
        parser.error("--limit must be between 1 and 8.")

    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)

    print("Loading FairVLMed train dataset with src/dataset.py loader logic...", flush=True)
    dataset = FairVLMedDataset(split="train", strict=True)
    n = len(dataset)
    if n != sum(TOTAL_SPLIT_LENGTHS):
        raise ValueError(
            f"Expected {sum(TOTAL_SPLIT_LENGTHS)} train rows for the requested "
            f"70/10/20 split, found {n}."
        )
    split_lengths = (
        TOTAL_SPLIT_LENGTHS[0],
        TOTAL_SPLIT_LENGTHS[1],
        n - TOTAL_SPLIT_LENGTHS[0] - TOTAL_SPLIT_LENGTHS[1],
    )
    train_subset, _, _ = random_split(
        range(n),
        split_lengths,
        generator=torch.Generator().manual_seed(args.split_seed),
    )
    selected_indices = random.Random(args.sample_seed).sample(
        list(train_subset.indices),
        args.limit,
    )
    print(
        f"Selected {len(selected_indices)} rows from the seeded 70/10/20 TRAIN "
        f"subset (split seed={args.split_seed}, sample seed={args.sample_seed}).",
        flush=True,
    )

    preprocess, mean, std = build_preprocess()
    column_titles = (
        "(a) Original + detected center",
        "(b) OpenCLIP default eval",
        "(c) Pad square + resize",
        "(d) CLAHE + pad square",
        "(e) Disc crop + CLAHE",
    )
    figure, axes = plt.subplots(
        nrows=args.limit,
        ncols=len(column_titles),
        figsize=(15, max(3.0 * args.limit, 5.0)),
        squeeze=False,
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=10)

    for row, index in enumerate(selected_indices):
        item = dataset.ds[index]
        image = dataset._load_image(item["filename"])
        clip_image = display_preprocessed(image, preprocess, mean, std)
        padded = pad_to_square(image)
        clahe_padded = pad_to_square(apply_clahe(image))
        disc_crop, center, detection_reason = disc_centered_panel(image)

        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        radius = max(8, round(min(image.size) * 0.012))
        center_x, center_y = center
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            outline=(255, 0, 0),
            width=max(2, radius // 4),
        )
        draw.line(
            (center_x - 2 * radius, center_y, center_x + 2 * radius, center_y),
            fill=(255, 0, 0),
            width=max(2, radius // 4),
        )
        draw.line(
            (center_x, center_y - 2 * radius, center_x, center_y + 2 * radius),
            fill=(255, 0, 0),
            width=max(2, radius // 4),
        )

        panels = (
            np.asarray(marked),
            clip_image,
            np.asarray(padded),
            np.asarray(clahe_padded),
            np.asarray(disc_crop),
        )
        print(
            f"[{row + 1}/{args.limit}] row={index} file={item['filename']} "
            f"shape={image.size} disc_center={center} ({detection_reason})",
            flush=True,
        )
        for column, panel in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axis.axis("off")
        axes[row, 0].set_ylabel(
            f"Train row {index}",
            rotation=0,
            labelpad=55,
            va="center",
            fontsize=9,
        )

    figure.tight_layout()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved preprocessing preview to {output}", flush=True)
    print("No feature extraction or model training was performed.", flush=True)


if __name__ == "__main__":
    main()
