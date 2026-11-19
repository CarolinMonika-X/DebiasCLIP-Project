"""Run the configured stack and validation-selected fine-tuned image blend."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

import final_report
import stack_models


ROOT = Path(__file__).resolve().parents[1]
FT_PROBABILITY_FILES = (
    "ft_probs.npz",
    "ft_probs_s1.npz",
    "ft_probs_s2.npz",
)
WEIGHTS = tuple(float(value) for value in np.round(np.arange(0, 1.01, 0.1), 1))
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
DELTA_METRICS = ("auc", "accuracy", "dpd", "eod_max")
RACE_NAMES = ("Asian", "Black", "White")
BASE_RACE_WORD_RE = stack_models.RACE_WORD_RE
ADDITIONAL_MASK_PATTERN = (
    r"glaucoma\w*|trabecul\w*|laser|slt|surger\w*|surgical|drops?|"
    r"latanoprost|timolol|brimonidine|dorzolamide|travoprost|bimatoprost|"
    r"cosopt|combigan|lumigan|xalatan|rhopressa|netarsudil|medication\w*|"
    r"tube|shunt|iop|pressure|diagnos\w*|suspect\w*|dx|referr\w*|"
    r"evaluat\w*|screen\w*|follow[- ]?up|rule[sd]? out|r/o"
)
MASK_LEVEL_2_RE = re.compile(
    f"(?:{BASE_RACE_WORD_RE.pattern})|(?:{ADDITIONAL_MASK_PATTERN})",
    flags=re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def configure_text_mask(mask_level: int) -> None:
    if mask_level not in (0, 2):
        raise ValueError(f"Unsupported text mask level: {mask_level}")
    stack_models.RACE_WORD_RE = BASE_RACE_WORD_RE


def apply_text_mask(
    texts: list[str],
    mask_level: int,
    split_name: str,
) -> list[str]:
    if mask_level == 0:
        return texts

    masked_texts: list[str] = []
    deleted_sentences = 0
    kept_word_counts: list[int] = []
    for text in texts:
        sentences = SENTENCE_SPLIT_RE.split(text)
        kept_sentences: list[str] = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            if MASK_LEVEL_2_RE.search(sentence):
                deleted_sentences += 1
            else:
                kept_sentences.append(sentence)
        masked_text = " ".join(kept_sentences)
        masked_texts.append(masked_text)
        kept_word_counts.append(len(masked_text.split()))

    print(
        f"Mask level 2 {split_name}: {deleted_sentences} sentences deleted; "
        f"median words kept per case {np.median(kept_word_counts):.0f}.",
        flush=True,
    )
    return masked_texts


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def load_ft_image(
    split: str,
    full_split_size: int,
    selected_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for filename in FT_PROBABILITY_FILES:
        path = ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(f"Fine-tuned probability file not found: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if split not in archive.files:
                raise ValueError(f"{filename} is missing the {split!r} array.")
            values = np.asarray(archive[split], dtype=np.float64).reshape(-1)
        if len(values) != full_split_size:
            raise ValueError(
                f"{filename}/{split} has {len(values)} rows; "
                f"expected {full_split_size} for the seed-42 split."
            )
        values = values[:selected_size]
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{filename}/{split} must contain finite probabilities in [0, 1].")
        predictions.append(values)
    return np.mean(predictions, axis=0)


def logit_blend(
    stack_probabilities: np.ndarray,
    image_probabilities: np.ndarray,
    weight: float,
) -> np.ndarray:
    epsilon = 1e-6
    stack_clipped = np.clip(stack_probabilities, epsilon, 1 - epsilon)
    image_clipped = np.clip(image_probabilities, epsilon, 1 - epsilon)
    stack_logits = np.log(stack_clipped) - np.log1p(-stack_clipped)
    image_logits = np.log(image_clipped) - np.log1p(-image_clipped)
    logits = weight * stack_logits + (1 - weight) * image_logits
    return 1 / (1 + np.exp(-logits))


def model_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, probabilities >= threshold)),
        "accuracy_at_0_5": float(accuracy_score(labels, probabilities >= 0.5)),
    }


def paired_bootstrap_deltas(
    labels: np.ndarray,
    races: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    candidate_threshold: float,
    reference_threshold: float,
) -> dict[str, dict[str, float | int]]:
    candidate_point = final_report.binary_metrics(
        labels, races, candidate, candidate_threshold
    )
    reference_point = final_report.binary_metrics(
        labels, races, reference, reference_threshold
    )
    point_deltas = {
        metric: candidate_point[metric] - reference_point[metric]
        for metric in DELTA_METRICS
    }
    samples: dict[str, list[float]] = {metric: [] for metric in DELTA_METRICS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    attempts = 0
    max_attempts = BOOTSTRAP_RESAMPLES * 10
    while min(map(len, samples.values())) < BOOTSTRAP_RESAMPLES and attempts < max_attempts:
        attempts += 1
        indexes = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indexes]
        sampled_races = races[indexes]
        if len(np.unique(sampled_labels)) < 2:
            continue
        try:
            candidate_metrics = final_report.binary_metrics(
                sampled_labels,
                sampled_races,
                candidate[indexes],
                candidate_threshold,
            )
            reference_metrics = final_report.binary_metrics(
                sampled_labels,
                sampled_races,
                reference[indexes],
                reference_threshold,
            )
        except ValueError:
            continue
        for metric in DELTA_METRICS:
            samples[metric].append(
                candidate_metrics[metric] - reference_metrics[metric]
            )
    valid_count = min(map(len, samples.values()))
    if valid_count != BOOTSTRAP_RESAMPLES:
        raise RuntimeError(
            f"Only obtained {valid_count}/{BOOTSTRAP_RESAMPLES} valid paired "
            "bootstrap resamples."
        )
    return {
        metric: {
            "delta": float(point_deltas[metric]),
            "percentile_2_5": float(np.percentile(samples[metric], 2.5)),
            "percentile_97_5": float(np.percentile(samples[metric], 97.5)),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        }
        for metric in DELTA_METRICS
    }


def load_test_texts(
    expected_labels: np.ndarray,
    mask_level: int,
) -> list[str]:
    dataset = stack_models.load_dataset(
        "harvardairobotics/FairVLMed",
        "default",
        split="train",
    )
    subsets = torch.utils.data.random_split(
        range(len(dataset)),
        (7000, 1000, 2000),
        generator=torch.Generator().manual_seed(42),
    )
    test_indices = list(subsets[2].indices)[: len(expected_labels)]
    subset = dataset.select(test_indices)
    test_labels = stack_models.labels_from_hub(subset)
    if not np.array_equal(test_labels, expected_labels):
        raise ValueError("HF test row labels do not align with cached features.")
    test_fields = {
        field: [str(value or "") for value in subset[field]]
        for field in stack_models.TEXT_FIELDS
    }
    test_texts = stack_models.combine_texts(
        test_fields, list(range(len(test_indices)))
    )
    return apply_text_mask(test_texts, mask_level, "test")


def prepare_data(
    limit: int,
    mask_level: int,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, dict[str, np.ndarray]]],
    list[str],
    list[str],
    np.ndarray,
    list[str],
]:
    stack_models.set_seed(stack_models.SEEDS[0])
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))

    canonical = stack_models.read_cache(stack_models.OPENCLIP_CACHE)
    biomedclip = stack_models.read_cache(stack_models.BIOMEDCLIP_CACHE)
    stack_models.verify_alignment(biomedclip, canonical, stack_models.BIOMEDCLIP_CACHE.name)
    stack_models.canonical_labels = {
        split: canonical[split]["labels"] for split in stack_models.SPLITS
    }
    stack_models.canonical_text_features = {
        split: canonical[split]["txt"] for split in stack_models.SPLITS
    }
    variants, skipped_variants = stack_models.find_image_variants(canonical)
    print(
        f"Found aligned image-only variants: {list(variants)}; "
        f"skipped={skipped_variants}",
        flush=True,
    )

    hub_texts, hub_train_labels = stack_models.get_text_data()
    train_labels = canonical["train"]["labels"]
    if not np.array_equal(train_labels, hub_train_labels):
        raise ValueError("Text training labels do not match cached OpenCLIP labels.")

    if limit:
        rng = np.random.default_rng(stack_models.SEEDS[0])
        selected: list[int] = []
        for label in (0, 1):
            label_indices = np.flatnonzero(train_labels == label)
            selected.extend(
                rng.choice(
                    label_indices,
                    size=min(limit // 2, len(label_indices)),
                    replace=False,
                ).tolist()
            )
        if len(selected) < 10:
            raise ValueError("--limit must allow at least 10 stratified train rows.")
        selected.sort()
        train_labels = train_labels[selected]
        train_texts = [hub_texts["train"][index] for index in selected]
        for cache in [canonical, biomedclip, *variants.values()]:
            cache["train"] = {
                key: value[selected] for key, value in cache["train"].items()
            }
            for split in ("val", "test"):
                cache[split] = {
                    key: value[:limit] for key, value in cache[split].items()
                }
        stack_models.canonical_text_features = {
            split: canonical[split]["txt"] for split in stack_models.SPLITS
        }
        validation_texts = hub_texts["val"][:limit]
    else:
        train_texts = hub_texts["train"]
        validation_texts = hub_texts["val"]

    train_texts = apply_text_mask(train_texts, mask_level, "train")
    validation_texts = apply_text_mask(
        validation_texts, mask_level, "validation"
    )

    return (
        canonical,
        biomedclip,
        variants,
        train_texts,
        validation_texts,
        train_labels,
        skipped_variants,
    )


def run(limit: int, mask_level: int) -> None:
    configure_text_mask(mask_level)
    (
        canonical,
        biomedclip,
        variants,
        train_texts,
        validation_texts,
        train_labels,
        skipped_variants,
    ) = prepare_data(limit, mask_level)
    split_labels = {
        split: canonical[split]["labels"] for split in stack_models.SPLITS
    }
    model_specs = stack_models.build_base_model_specs(
        canonical, biomedclip, variants
    )
    model_specs["text_tfidf"] = {
        "kind": "text",
        "cache": "FairVLMed gpt4_summary + note",
        "feature_config": (
            f"mask-level-{mask_level} TF-IDF of gpt4_summary + note"
        ),
    }
    model_names = list(model_specs)
    print(f"Base models ({len(model_names)}): {model_names}", flush=True)

    oof_columns: list[np.ndarray] = []
    validation_columns: list[np.ndarray] = []
    full_models: dict[str, dict[str, Any]] = {}
    base_results: dict[str, dict[str, Any]] = {}
    for name in model_names:
        spec = model_specs[name]
        oof = stack_models.cv_oof_predictions(
            name, spec, train_labels, train_texts, stack_models.EPOCHS
        )
        full_model, val_probabilities = stack_models.fit_full_base_model(
            name,
            spec,
            train_labels,
            train_texts,
            validation_texts if spec["kind"] == "text" else [],
            stack_models.EPOCHS,
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
    if oof_matrix.shape != (len(train_labels), len(model_names)):
        raise AssertionError("Unexpected OOF matrix shape.")
    if validation_matrix.shape != (len(split_labels["val"]), len(model_names)):
        raise AssertionError("Unexpected validation matrix shape.")

    meta_c_results: list[dict[str, float]] = []
    meta_models: list[stack_models.LogisticRegression] = []
    for c_value in stack_models.META_C_VALUES:
        model = stack_models.LogisticRegression(
            C=c_value,
            max_iter=3000,
            solver="lbfgs",
            random_state=stack_models.SEEDS[0],
        )
        model.fit(oof_matrix, train_labels)
        val_probabilities = model.predict_proba(validation_matrix)[:, 1]
        meta_models.append(model)
        meta_c_results.append(
            {
                "C": c_value,
                "val_auc": float(
                    roc_auc_score(split_labels["val"], val_probabilities)
                ),
            }
        )
    selected_index = max(
        range(len(stack_models.META_C_VALUES)),
        key=lambda index: (
            meta_c_results[index]["val_auc"],
            -stack_models.META_C_VALUES[index],
        ),
    )
    selected_c = stack_models.META_C_VALUES[selected_index]
    selected_meta = meta_models[selected_index]
    stack_val_probabilities = selected_meta.predict_proba(validation_matrix)[:, 1]
    stack_threshold = stack_models.choose_youden_threshold(
        split_labels["val"], stack_val_probabilities
    )

    ft_image_val = load_ft_image("val", 1000, len(split_labels["val"]))
    weight_scores = {
        weight: float(
            roc_auc_score(
                split_labels["val"],
                logit_blend(stack_val_probabilities, ft_image_val, weight),
            )
        )
        for weight in WEIGHTS
    }
    selected_weight = max(WEIGHTS, key=lambda weight: (weight_scores[weight], weight))
    final_val_probabilities = logit_blend(
        stack_val_probabilities, ft_image_val, selected_weight
    )
    final_threshold = stack_models.choose_youden_threshold(
        split_labels["val"], final_val_probabilities
    )

    val_probabilities_by_name = {
        name: probabilities
        for name, probabilities in zip(
            model_names, validation_columns, strict=True
        )
    }
    validation_thresholds = {
        "openclip_joint": stack_models.choose_youden_threshold(
            split_labels["val"], val_probabilities_by_name["openclip_joint"]
        ),
        "text_tfidf": stack_models.choose_youden_threshold(
            split_labels["val"], val_probabilities_by_name["text_tfidf"]
        ),
        "stack": stack_threshold,
        "ft_image": stack_models.choose_youden_threshold(
            split_labels["val"], ft_image_val
        ),
        "final_blend": final_threshold,
    }
    print(
        f"Validation selections complete: stack C={selected_c:g}, "
        f"logit blend w={selected_weight:.1f}, "
        f"final Youden threshold={final_threshold:.6f}. "
        "Test inference starts now.",
        flush=True,
    )

    test_texts = load_test_texts(split_labels["test"], mask_level)
    test_columns: list[np.ndarray] = []
    for name in model_names:
        model = full_models[name]
        if model["kind"] == "text":
            test_probabilities = stack_models.predict_text(
                model["vectorizer"], model["classifier"], test_texts
            )
        else:
            test_probabilities = stack_models.predict_full_base_test(model, [])
        test_columns.append(test_probabilities)
    test_probabilities_by_name = {
        name: probabilities
        for name, probabilities in zip(model_names, test_columns, strict=True)
    }
    stack_test_probabilities = selected_meta.predict_proba(
        np.column_stack(test_columns)
    )[:, 1]
    ft_image_test = load_ft_image("test", 2000, len(split_labels["test"]))
    final_test_probabilities = logit_blend(
        stack_test_probabilities, ft_image_test, selected_weight
    )

    report_probabilities = {
        "openclip_joint": (
            val_probabilities_by_name["openclip_joint"],
            test_probabilities_by_name["openclip_joint"],
        ),
        "text_tfidf": (
            val_probabilities_by_name["text_tfidf"],
            test_probabilities_by_name["text_tfidf"],
        ),
        "stack": (stack_val_probabilities, stack_test_probabilities),
        "ft_image": (ft_image_val, ft_image_test),
        "final_blend": (final_val_probabilities, final_test_probabilities),
    }
    model_labels = {
        "openclip_joint": "OpenCLIP MLP baseline",
        "text_tfidf": f"Mask-{mask_level} TF-IDF alone",
        "stack":         f"Mask-{mask_level} TF-IDF + base-model stack",
        "ft_image": "ft_image alone",
        "final_blend": "Final logit-space blend",
    }
    metrics: dict[str, dict[str, Any]] = {}
    for name, (val_probabilities, test_probabilities) in report_probabilities.items():
        threshold = validation_thresholds[name]
        metrics[name] = {
            "label": model_labels[name],
            "validation_threshold": float(threshold),
            "validation": model_metrics(
                split_labels["val"], val_probabilities, threshold
            ),
            "test": model_metrics(
                split_labels["test"], test_probabilities, threshold
            ),
        }

    for index, name in enumerate(model_names):
        base_results[name]["validation"] = stack_models.metrics_at_thresholds(
            split_labels["val"], validation_columns[index], stack_threshold
        )
        base_results[name]["test"] = stack_models.metrics_at_thresholds(
            split_labels["test"], test_columns[index], stack_threshold
        )
    base_results["stack"] = {
        "feature_config": (
            f"logistic regression over {len(model_names)} base OOF probabilities"
        ),
        "selected_meta_C": selected_c,
        "validation_C_sweep": meta_c_results,
        "validation_threshold": {
            "rule": "Youden J maximized on validation; ties nearest 0.5",
            "value": float(stack_threshold),
        },
        "validation": stack_models.metrics_at_thresholds(
            split_labels["val"], stack_val_probabilities, stack_threshold
        ),
        "test": stack_models.metrics_at_thresholds(
            split_labels["test"], stack_test_probabilities, stack_threshold
        ),
    }

    comparisons: dict[str, dict[str, Any]] = {}
    final_test = report_probabilities["final_blend"][1]
    final_eval_threshold = validation_thresholds["final_blend"]
    for reference_name in ("stack", "openclip_joint"):
        reference_test = report_probabilities[reference_name][1]
        comparisons[f"final_minus_{reference_name}"] = paired_bootstrap_deltas(
            split_labels["test"],
            canonical["test"]["races"],
            final_test,
            reference_test,
            final_eval_threshold,
            validation_thresholds[reference_name],
        )

    race_rates = {
        name: final_report.per_race_rates(
            split_labels["test"],
            canonical["test"]["races"],
            report_probabilities[name][1],
            validation_thresholds[name],
        )
        for name in ("stack", "final_blend")
    }

    mask_suffix = "_mask0" if mask_level == 0 else ""
    smoke_suffix = "_smoke" if limit else ""
    output_suffix = f"{mask_suffix}{smoke_suffix}"
    stack_test_path = ROOT / f"stack_test_probs{output_suffix}.npy"
    stack_val_path = ROOT / f"stack_val_probs{output_suffix}.npy"
    base_test_path = ROOT / f"stack_base_test_probs{output_suffix}.npz"
    plus_test_path = ROOT / f"stack_plus_test_probs{output_suffix}.npy"
    results_path = ROOT / f"stack_plus_results{output_suffix}.json"
    markdown_path = ROOT / f"final_report_plus{output_suffix}.md"
    latex_path = ROOT / "tables" / f"final_report_plus{output_suffix}.tex"

    np.save(stack_test_path, stack_test_probabilities.astype(np.float32))
    np.save(stack_val_path, stack_val_probabilities.astype(np.float32))
    np.save(plus_test_path, final_test_probabilities.astype(np.float32))
    np.savez_compressed(
        base_test_path,
        **{
            name: probabilities.astype(np.float32)
            for name, probabilities in zip(model_names, test_columns, strict=True)
        },
        stack=stack_test_probabilities.astype(np.float32),
    )

    result: dict[str, Any] = {
        "config": {
            "split_sizes": {
                split: len(values) for split, values in split_labels.items()
            },
            "folds": stack_models.N_FOLDS,
            "mlp_seeds": list(stack_models.SEEDS),
            "mlp_epochs": stack_models.EPOCHS,
            "batch_size": stack_models.BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": stack_models.LEARNING_RATE,
            "weight_decay": stack_models.WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "cv": "StratifiedKFold(shuffle=True, random_state=0), train rows only",
            "test_evaluation_after_validation_selection": True,
            "stack_test_probabilities_file": stack_test_path.name,
            "stack_validation_probabilities_file": stack_val_path.name,
            "test_probabilities_file": plus_test_path.name,
            "base_test_probabilities_file": base_test_path.name,
            "fine_tuned_probability_files": list(FT_PROBABILITY_FILES),
            "fine_tuned_ensemble": "arithmetic mean of val/test probabilities",
            "mask_level": mask_level,
            "mask_level_0_regex": BASE_RACE_WORD_RE.pattern,
            "mask_level_2_additional_regex": ADDITIONAL_MASK_PATTERN,
            "text_mask_regex": stack_models.RACE_WORD_RE.pattern,
            "limit": limit,
        },
        "existing_stack": {
            "base_models": base_results,
            "skipped_image_variant_caches": skipped_variants,
        },
        "final_blend_selection": {
            "space": "logit",
            "weight_on_stack": selected_weight,
            "validation_auc_by_weight": [
                {"weight_on_stack": weight, "val_auc": weight_scores[weight]}
                for weight in WEIGHTS
            ],
            "weight_selection_rule": (
                "Highest validation AUC; ties favor larger weight on stack."
            ),
            "threshold": float(final_threshold),
            "threshold_selection_rule": (
                "Validation Youden J; ties nearest 0.5."
            ),
        },
        "models": metrics,
        "paired_bootstrap": comparisons,
        "per_race_test_rates": race_rates,
    }
    markdown = render_markdown(result)
    latex = render_latex(result)
    atomic_write_text(
        results_path,
        json.dumps(result, indent=2, allow_nan=False) + "\n",
    )
    atomic_write_text(markdown_path, markdown)
    atomic_write_text(latex_path, latex)
    print(
        f"Wrote {results_path.name}, {stack_val_path.name}, "
        f"{stack_test_path.name}, {plus_test_path.name}, "
        f"{markdown_path.name}, and {latex_path.name}.",
        flush=True,
    )


def render_markdown(result: dict[str, Any]) -> str:
    selection = result["final_blend_selection"]
    lines = [
        "# Stack Plus Evaluation",
        "",
        f"Text preprocessing mask level: "
        f"`{result['config']['mask_level']}`.",
        "",
        "All model/weight/threshold choices use validation only. Test inference and",
        "evaluation occur after those choices. Accuracy and race rates use each",
        "model's own validation-Youden threshold (ties nearest 0.5).",
        "",
        f"Final blend: `w={selection['weight_on_stack']:.1f}` on the stack logits "
        f"and `{1 - selection['weight_on_stack']:.1f}` on ft_image logits; "
        f"threshold `{selection['threshold']:.6f}`.",
        "",
        "## Validation and test metrics",
        "",
        "| Model | Validation AUC | Validation accuracy | Test AUC | Test accuracy | Threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metrics in result["models"].values():
        lines.append(
            f"| {metrics['label']} | {metrics['validation']['auc']:.4f} | "
            f"{metrics['validation']['accuracy']:.4f} | "
            f"{metrics['test']['auc']:.4f} | "
            f"{metrics['test']['accuracy']:.4f} | "
            f"{metrics['validation_threshold']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired test bootstrap deltas",
            "",
            "Delta is final blend minus the named reference; identical case indices",
            "are used for both predictions. Intervals are 95% percentile CIs from",
            f"{BOOTSTRAP_RESAMPLES:,} resamples (seed {BOOTSTRAP_SEED}).",
            "",
            "| Comparison | Metric | Delta | 95% percentile CI |",
            "|---|---|---:|---:|",
        ]
    )
    for comparison, values in result["paired_bootstrap"].items():
        for metric in DELTA_METRICS:
            item = values[metric]
            lines.append(
                f"| {comparison.replace('_', ' ')} | {metric} | "
                f"{item['delta']:+.4f} | "
                f"[{item['percentile_2_5']:+.4f}, "
                f"{item['percentile_97_5']:+.4f}] |"
            )
    for model_name in ("stack", "final_blend"):
        lines.extend(
            [
                "",
                f"## Per-race test rates: {result['models'][model_name]['label']}",
                "",
                f"Validation threshold: "
                f"`{result['models'][model_name]['validation_threshold']:.6f}`. "
                "Intervals are Wilson 95% CIs.",
                "",
                "| Race | N | TPR (95% CI) | FNR (95% CI) | FPR (95% CI) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for race in RACE_NAMES:
            rates = result["per_race_test_rates"][model_name][race]
            lines.append(
                f"| {race} | {rates['n']} | {rates['tpr']:.4f} "
                f"[{rates['tpr_wilson_95_ci'][0]:.4f}, "
                f"{rates['tpr_wilson_95_ci'][1]:.4f}] | "
                f"{rates['fnr']:.4f} "
                f"[{rates['fnr_wilson_95_ci'][0]:.4f}, "
                f"{rates['fnr_wilson_95_ci'][1]:.4f}] | "
                f"{rates['fpr']:.4f} "
                f"[{rates['fpr_wilson_95_ci'][0]:.4f}, "
                f"{rates['fpr_wilson_95_ci'][1]:.4f}] |"
            )
    return "\n".join(lines).rstrip() + "\n"


def render_latex(result: dict[str, Any]) -> str:
    selection = result["final_blend_selection"]
    lines = [
        "% Generated by src/stack_plus.py; do not edit by hand.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Validation and test performance for Stack Plus.}",
        r"\label{tab:stack-plus-performance}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Val AUC & Val Acc. & Test AUC & Test Acc. & Val. thr. \\",
        r"\midrule",
    ]
    for metrics in result["models"].values():
        lines.append(
            f"{metrics['label']} & {metrics['validation']['auc']:.4f} & "
            f"{metrics['validation']['accuracy']:.4f} & "
            f"{metrics['test']['auc']:.4f} & "
            f"{metrics['test']['accuracy']:.4f} & "
            f"{metrics['validation_threshold']:.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\vspace{{0.5em}}\\par\\footnotesize Logit blend weight on stack: "
            f"{selection['weight_on_stack']:.1f}; validation Youden threshold: "
            f"{selection['threshold']:.6f}.",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Paired test-set bootstrap deltas, final blend minus reference.}",
            r"\label{tab:stack-plus-bootstrap}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Comparison & Metric & Delta & \multicolumn{2}{c}{95\% CI} \\",
            r"\midrule",
        ]
    )
    for comparison, values in result["paired_bootstrap"].items():
        for metric in DELTA_METRICS:
            item = values[metric]
            lines.append(
                f"{comparison.replace('_', ' ')} & {metric} & "
                f"{item['delta']:+.4f} & {item['percentile_2_5']:+.4f} & "
                f"{item['percentile_97_5']:+.4f} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Per-race test rates at each model's validation threshold.}",
            r"\label{tab:stack-plus-race-rates}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Model & Race & $n$ & TPR (95\% CI) & FNR (95\% CI) & FPR (95\% CI) \\",
            r"\midrule",
        ]
    )
    for model_name in ("stack", "final_blend"):
        label = result["models"][model_name]["label"]
        for race in RACE_NAMES:
            rates = result["per_race_test_rates"][model_name][race]
            lines.append(
                f"{label} & {race} & {rates['n']} & "
                f"{rates['tpr']:.4f} "
                f"[{rates['tpr_wilson_95_ci'][0]:.4f}, "
                f"{rates['tpr_wilson_95_ci'][1]:.4f}] & "
                f"{rates['fnr']:.4f} "
                f"[{rates['fnr_wilson_95_ci'][0]:.4f}, "
                f"{rates['fnr_wilson_95_ci'][1]:.4f}] & "
                f"{rates['fpr']:.4f} "
                f"[{rates['fpr_wilson_95_ci'][0]:.4f}, "
                f"{rates['fpr_wilson_95_ci'][1]:.4f}] \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-level",
        type=int,
        choices=(0, 2),
        default=2,
        help="Text masking level: 0 removes race terms; 2 also removes clinical terms.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional smoke-test row cap per split; zero uses full splits.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")
    run(args.limit, args.mask_level)


if __name__ == "__main__":
    main()
