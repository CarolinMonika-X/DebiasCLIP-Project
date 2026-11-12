"""Validation-selected debiasing baselines on cached joint image/text features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cached_features.pt"
DEFAULT_RESULTS = ROOT / "baselines_v2_results.json"
DEFAULT_TABLE = ROOT / "tables" / "baselines.tex"
DEFAULT_CHECKPOINT = ROOT / ".baselines_v2_checkpoint.json"
SPLITS = ("train", "val", "test")
RACE_NAMES = ("Asian", "Black", "White")
EXPECTED_SPLIT_SIZES = {"train": 7000, "val": 1000, "test": 2000}
EXPECTED_TEST_RACE_COUNTS = (169, 302, 1529)

SEEDS = (0, 100, 200, 300, 400)
ADVERSARIAL_LAMBDAS = (0.05, 0.1, 0.5, 1.0, 2.0)
PENALTY_LAMBDAS = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0)
FAILURE_LAMBDA = 5.0
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6
THRESHOLD = 0.50
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261112
PENALTY_LAMBDAS_ALL = (*PENALTY_LAMBDAS, FAILURE_LAMBDA)
METHOD_METRICS = (
    "auc",
    "accuracy",
    "dpd",
    "eod_max",
    "linear_race_probe_macro_auc",
)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


class DiagnosticModel(nn.Module):
    def __init__(self, feature_dim: int, adversarial: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.diagnostic = nn.Linear(256, 2)
        self.adversarial = adversarial
        self.race_classifier = nn.Linear(256, len(RACE_NAMES)) if adversarial else None

    def forward(
        self, features: torch.Tensor, reversal_strength: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.encoder(features)
        race_logits = None
        if self.race_classifier is not None:
            race_logits = self.race_classifier(
                GradientReversal.apply(hidden, reversal_strength)
            )
        return self.diagnostic(hidden), race_logits


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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def load_data(path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cached features not found: {path}")
    cached = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cached, dict):
        raise ValueError("Cached features must contain split dictionaries.")
    data: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        if split not in cached:
            raise ValueError(f"Cached features are missing split {split!r}.")
        row = cached[split]
        if not {"img", "txt", "labels", "races"}.issubset(row):
            raise ValueError(f"{split} split is missing required cache fields.")
        image, text = row["img"].float(), row["txt"].float()
        if image.ndim != 2 or image.shape[1] != 512:
            raise ValueError(f"Expected 512-D image features in {split}.")
        if text.ndim != 2 or text.shape[1] != 512:
            raise ValueError(f"Expected 512-D text features in {split}.")
        if len(image) != len(text):
            raise ValueError(f"Image and text feature lengths differ in {split}.")
        features = torch.cat((image, text), dim=1).numpy().astype(
            np.float32, copy=False
        )
        labels = row["labels"].long().numpy()
        races = row["races"].long().numpy()
        if labels.shape != (len(features),) or races.shape != labels.shape:
            raise ValueError(f"Feature, label, and race lengths differ in {split}.")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features found in {split}.")
        data[split] = {"features": features, "labels": labels, "races": races}
    sizes = {split: len(data[split]["labels"]) for split in SPLITS}
    assert sizes == EXPECTED_SPLIT_SIZES, sizes
    counts = tuple(
        int(np.count_nonzero(data["test"]["races"] == race))
        for race in range(len(RACE_NAMES))
    )
    assert counts == EXPECTED_TEST_RACE_COUNTS, counts
    for split in SPLITS:
        assert set(np.unique(data[split]["labels"])) == {0, 1}
        assert set(np.unique(data[split]["races"])) == {0, 1, 2}
    return data


def linear_probe_macro_auc(
    train_features: np.ndarray,
    probe_features: np.ndarray,
    train_races: np.ndarray,
    probe_races: np.ndarray,
) -> float:
    scaler = StandardScaler()
    train = scaler.fit_transform(train_features)
    probe = scaler.transform(probe_features)
    classifier = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs")
    classifier.fit(train, train_races)
    probabilities = classifier.predict_proba(probe)
    order = np.argsort(classifier.classes_)
    return float(
        roc_auc_score(
            probe_races,
            probabilities[:, order],
            labels=np.arange(len(RACE_NAMES)),
            multi_class="ovr",
            average="macro",
        )
    )


def classification_metrics(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    group_thresholds: dict[int, float] | None = None,
) -> dict[str, float]:
    if group_thresholds is None:
        predictions = probabilities >= THRESHOLD
    else:
        predictions = np.asarray(
            [
                probabilities[index] >= group_thresholds[int(race)]
                for index, race in enumerate(races)
            ],
            dtype=bool,
        )
    positive_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []
    for race in range(len(RACE_NAMES)):
        mask = races == race
        positive = mask & (labels == 1)
        negative = mask & (labels == 0)
        if not positive.any() or not negative.any():
            raise ValueError(f"Race group {RACE_NAMES[race]} lacks an outcome class.")
        positive_rates.append(float(predictions[mask].mean()))
        tprs.append(float(predictions[positive].mean()))
        fprs.append(float(predictions[negative].mean()))
    tpr_gap = max(tprs) - min(tprs)
    fpr_gap = max(fprs) - min(fprs)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(np.mean(predictions == labels)),
        "dpd": float(max(positive_rates) - min(positive_rates)),
        "eod_max": float(max(tpr_gap, fpr_gap)),
        "tpr_gap": float(tpr_gap),
        "fpr_gap": float(fpr_gap),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson intervals require a positive sample count.")
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def per_group_metrics(
    labels: np.ndarray,
    races: np.ndarray,
    probabilities: np.ndarray,
    group_thresholds: dict[int, float] | None = None,
) -> dict[str, Any]:
    if group_thresholds is None:
        predictions = probabilities >= THRESHOLD
    else:
        predictions = np.asarray(
            [
                probabilities[index] >= group_thresholds[int(race)]
                for index, race in enumerate(races)
            ],
            dtype=bool,
        )
    result: dict[str, Any] = {}
    for race, name in enumerate(RACE_NAMES):
        mask = races == race
        positives = mask & (labels == 1)
        negatives = mask & (labels == 0)
        positive_n = int(positives.sum())
        negative_n = int(negatives.sum())
        if positive_n == 0 or negative_n == 0:
            raise ValueError(f"Race group {name} lacks an outcome class.")
        true_positive = int(np.count_nonzero(predictions & positives))
        false_negative = int(np.count_nonzero(~predictions & positives))
        false_positive = int(np.count_nonzero(predictions & negatives))
        result[name] = {
            "n": int(mask.sum()),
            "positive_n": positive_n,
            "negative_n": negative_n,
            "tpr": true_positive / positive_n,
            "tpr_wilson_95_ci": list(wilson_interval(true_positive, positive_n)),
            "fnr": false_negative / positive_n,
            "fnr_wilson_95_ci": list(wilson_interval(false_negative, positive_n)),
            "fpr": false_positive / negative_n,
            "fpr_wilson_95_ci": list(wilson_interval(false_positive, negative_n)),
        }
    return result


def label_conditioned_penalty(
    logits: torch.Tensor, labels: torch.Tensor, races: torch.Tensor
) -> torch.Tensor:
    positive_probability = torch.softmax(logits, dim=1)[:, 1]
    penalty = logits.new_zeros(())
    for label in (0, 1):
        means = [
            positive_probability[(labels == label) & (races == race)].mean()
            for race in range(len(RACE_NAMES))
            if ((labels == label) & (races == race)).any()
        ]
        if len(means) > 1:
            stacked = torch.stack(means)
            penalty = penalty + stacked.max() - stacked.min()
    return penalty


def train_seed(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_races: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_races: np.ndarray,
    test_features: np.ndarray | None,
    seed: int,
    mode: str = "raw",
    strength: float = 0.0,
    sample_weights: np.ndarray | None = None,
    early_stopping: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    adversarial = mode == "adversarial"
    model = DiagnosticModel(train_features.shape[1], adversarial=adversarial)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    train_x = torch.from_numpy(train_features).float()
    train_y = torch.from_numpy(train_labels).long()
    train_r = torch.from_numpy(train_races).long()
    train_w = (
        None if sample_weights is None else torch.from_numpy(sample_weights).float()
    )
    val_x = torch.from_numpy(val_features).float()
    best_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_probabilities: np.ndarray | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_y))
        for start in range(0, len(train_y), BATCH_SIZE):
            indexes = permutation[start : start + BATCH_SIZE]
            logits, race_logits = model(train_x[indexes], strength)
            losses = nn.functional.cross_entropy(
                logits, train_y[indexes], reduction="none"
            )
            disease_loss = (
                (losses * train_w[indexes]).mean()
                if train_w is not None
                else losses.mean()
            )
            if mode == "label_conditioned":
                disease_loss = disease_loss + strength * label_conditioned_penalty(
                    logits, train_y[indexes], train_r[indexes]
                )
            if adversarial:
                if race_logits is None:
                    raise RuntimeError("Adversarial model is missing race logits.")
                loss = disease_loss + nn.functional.cross_entropy(
                    race_logits, train_r[indexes]
                )
            else:
                loss = disease_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_x)
            val_probabilities = torch.softmax(val_logits, dim=1)[:, 1].numpy()
        val_auc = float(roc_auc_score(val_labels, val_probabilities))
        if val_auc > best_auc + 1e-8:
            best_auc = val_auc
            best_epoch = epoch
            best_val_probabilities = val_probabilities.copy()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if early_stopping and stale_epochs >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None or best_val_probabilities is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    test_probabilities: np.ndarray | None = None
    if test_features is not None:
        with torch.no_grad():
            test_logits, _ = model(torch.from_numpy(test_features).float())
            test_probabilities = torch.softmax(test_logits, dim=1)[:, 1].numpy()
    val_metrics = classification_metrics(val_labels, val_races, best_val_probabilities)
    return {
        "val_probabilities": best_val_probabilities.astype(float).tolist(),
        "test_probabilities": (
            None
            if test_probabilities is None
            else test_probabilities.astype(float).tolist()
        ),
        "val_metrics": val_metrics,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
    }


def fit_ensemble(
    name: str,
    data: dict[str, dict[str, np.ndarray]],
    checkpoint: dict[str, Any],
    persist: Any,
    *,
    mode: str = "raw",
    strength: float = 0.0,
    sample_weights: np.ndarray | None = None,
    early_stopping: bool = True,
) -> dict[str, Any]:
    experiments = checkpoint["experiments"]
    entry = experiments.setdefault(name, {"seed_runs": []})
    train, val, test = (data[split] for split in SPLITS)
    while len(entry["seed_runs"]) < len(SEEDS):
        seed_index = len(entry["seed_runs"])
        result = train_seed(
            train["features"],
            train["labels"],
            train["races"],
            val["features"],
            val["labels"],
            val["races"],
            test["features"],
            SEEDS[seed_index],
            mode,
            strength,
            sample_weights,
            early_stopping,
        )
        entry["seed_runs"].append(result)
        persist()
        print(
            f"  {name}: completed seed {seed_index + 1}/{len(SEEDS)} "
            f"(best epoch {result['best_epoch']})",
            flush=True,
        )
    val_probabilities = np.mean(
        [row["val_probabilities"] for row in entry["seed_runs"]], axis=0
    )
    test_probabilities = np.mean(
        [row["test_probabilities"] for row in entry["seed_runs"]], axis=0
    )
    return {
        "val_probabilities": val_probabilities,
        "test_probabilities": test_probabilities,
        "val_metrics": classification_metrics(
            val["labels"], val["races"], val_probabilities
        ),
        "test_metrics": classification_metrics(
            test["labels"], test["races"], test_probabilities
        ),
        "seed_epochs": [int(row["best_epoch"]) for row in entry["seed_runs"]],
        "seed_val_aucs": [float(row["best_val_auc"]) for row in entry["seed_runs"]],
    }


def race_weights(races: np.ndarray) -> np.ndarray:
    counts = np.bincount(races, minlength=len(RACE_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Race reweighting requires all race groups in train.")
    per_sample = 1 / counts[races]
    return (per_sample / per_sample.mean()).astype(np.float32)


def train_fixed_seed(
    train_x: np.ndarray,
    train_y: np.ndarray,
    seed: int,
) -> DiagnosticModel:
    set_seed(seed)
    model = DiagnosticModel(train_x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    features = torch.from_numpy(train_x).float()
    labels = torch.from_numpy(train_y).long()
    for _ in range(EPOCHS):
        model.train()
        permutation = torch.randperm(len(labels))
        for start in range(0, len(labels), BATCH_SIZE):
            indexes = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(features[indexes])[0], labels[indexes]
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model.eval()


def cross_fitted_group_thresholds(
    features: np.ndarray,
    labels: np.ndarray,
    races: np.ndarray,
    checkpoint: dict[str, Any],
    persist: Any,
    seed: int = 0,
) -> tuple[dict[int, float], dict[str, Any]]:
    stratification = races * 2 + labels
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out_of_fold = np.zeros(len(labels), dtype=np.float64)
    fold_sizes: list[int] = []
    fold_predictions_cache = checkpoint.setdefault("cross_fit_seed_predictions", {})
    for fold, (train_indices, held_indices) in enumerate(
        splitter.split(features, stratification)
    ):
        fold_predictions: list[np.ndarray] = []
        for seed_index, model_seed in enumerate(SEEDS):
            cache_key = f"fold_{fold}_seed_{seed_index}"
            if cache_key in fold_predictions_cache:
                held_probabilities = np.asarray(
                    fold_predictions_cache[cache_key], dtype=np.float64
                )
                if len(held_probabilities) != len(held_indices):
                    raise ValueError(
                        f"Cross-fit checkpoint has invalid predictions for {cache_key}."
                    )
            else:
                model = train_fixed_seed(
                    features[train_indices],
                    labels[train_indices],
                    model_seed + fold * 10_000,
                )
                with torch.no_grad():
                    logits, _ = model(torch.from_numpy(features[held_indices]).float())
                    held_probabilities = (
                        torch.softmax(logits, dim=1)[:, 1].numpy()
                    )
                fold_predictions_cache[cache_key] = held_probabilities.tolist()
                persist()
                print(
                    f"  cross-fit: completed fold {fold + 1}/5, "
                    f"seed {seed_index + 1}/{len(SEEDS)}",
                    flush=True,
                )
            fold_predictions.append(held_probabilities)
        out_of_fold[held_indices] = np.mean(fold_predictions, axis=0)
        fold_sizes.append(len(held_indices))

    target_tpr = float(
        np.mean(out_of_fold[labels == 1] >= THRESHOLD)
    )
    thresholds: dict[int, float] = {}
    for race in range(len(RACE_NAMES)):
        positive_scores = out_of_fold[(races == race) & (labels == 1)]
        if len(positive_scores) == 0:
            raise ValueError(f"Race {RACE_NAMES[race]} has no positive train cases.")
        thresholds[race] = float(
            np.quantile(positive_scores, 1.0 - target_tpr, method="higher")
        )
    return thresholds, {
        "thresholds_by_race": {
            RACE_NAMES[race]: value for race, value in thresholds.items()
        },
        "target_train_oof_tpr": target_tpr,
        "fold_sizes": fold_sizes,
        "cross_fit_split": "train only",
        "seeds_per_fold": len(SEEDS),
    }


def method_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    races: np.ndarray,
    probe_auc: float,
    thresholds: dict[int, float] | None = None,
) -> dict[str, float]:
    return {
        **classification_metrics(labels, races, probabilities, thresholds),
        "linear_race_probe_macro_auc": float(probe_auc),
    }


def paired_bootstrap(
    labels: np.ndarray,
    races: np.ndarray,
    predictions: dict[str, np.ndarray],
    raw_probabilities: np.ndarray,
    n_resamples: int,
    seed: int,
    thresholds: dict[int, float],
    raw_probe: float,
    method_probes: dict[str, float],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_test = len(labels)
    intervals: dict[str, Any] = {}
    raw_draws: dict[str, np.ndarray] = {
        key: np.empty(n_resamples, dtype=np.float64) for key in METHOD_METRICS
    }
    method_draws: dict[str, dict[str, np.ndarray]] = {
        name: {
            key: np.empty(n_resamples, dtype=np.float64)
            for key in METHOD_METRICS
        }
        for name in predictions
    }
    for draw in range(n_resamples):
        indices = rng.integers(0, n_test, size=n_test)
        boot_labels = labels[indices]
        boot_races = races[indices]
        raw_metrics = classification_metrics(
            boot_labels, boot_races, raw_probabilities[indices]
        )
        raw_values = {
            **raw_metrics,
            "linear_race_probe_macro_auc": raw_probe,
        }
        for key in METHOD_METRICS:
            raw_draws[key][draw] = raw_values[key]
        for name, probabilities in predictions.items():
            operating_thresholds = (
                thresholds if name == "Cross-fitted group cutoffs" else None
            )
            metrics = classification_metrics(
                boot_labels,
                boot_races,
                probabilities[indices],
                operating_thresholds,
            )
            values = {
                **metrics,
                "linear_race_probe_macro_auc": method_probes[name],
            }
            for key in METHOD_METRICS:
                method_draws[name][key][draw] = values[key]

    for name in predictions:
        intervals[name] = {
            "delta_convention": "method minus Raw Dual",
            "n_resamples": n_resamples,
            "seed": seed,
            "confidence_level": 0.95,
            "metrics": {
                metric: {
                    "mean_delta": float(
                        np.mean(method_draws[name][metric] - raw_draws[metric])
                    ),
                    "percentile_2_5": float(
                        np.percentile(
                            method_draws[name][metric] - raw_draws[metric], 2.5
                        )
                    ),
                    "percentile_97_5": float(
                        np.percentile(
                            method_draws[name][metric] - raw_draws[metric], 97.5
                        )
                    ),
                }
                for metric in METHOD_METRICS
            },
        }
    return intervals


def latex_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    tex_rows = []
    for name, row in rows:
        metrics = row["test_metrics"]
        deltas = row.get("paired_bootstrap_vs_raw", {}).get("metrics", {})
        delta_columns = []
        for metric in ("auc", "accuracy", "dpd", "eod_max"):
            interval = deltas.get(metric)
            if interval is None:
                delta_columns.append("--")
            else:
                delta_columns.append(
                    f"{interval['mean_delta']:+.4f} "
                    f"[{interval['percentile_2_5']:+.4f},"
                    f"{interval['percentile_97_5']:+.4f}]"
                )
        tex_rows.append(
            f"{name} & {metrics['auc']:.4f} & {metrics['accuracy']:.4f} "
            f"& {metrics['dpd']:.4f} & {metrics['eod_max']:.4f} "
            f"& {metrics['linear_race_probe_macro_auc']:.4f} & "
            + " & ".join(delta_columns)
            + r" \\"
        )
    return (
        "% Paired percentile bootstrap compares each method to Raw Dual on identical test resamples.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Five-seed baselines at a fixed 0.50 threshold. Bootstrap columns "
        "show method-minus-Raw mean delta and 95\\% percentile CI.}\n"
        "\\label{tab:baselines-v2}\n"
        "\\begin{tabular}{lrrrrrrrrrr}\n"
        "\\hline\n"
        "Method & AUC & Accuracy & DPD & EOD$_{\\max}$ & Linear race AUC "
        "& $\\Delta$ AUC [95\\% CI] & $\\Delta$ Acc. [95\\% CI] "
        "& $\\Delta$ DPD [95\\% CI] & $\\Delta$ EOD [95\\% CI] \\\\\n"
        "\\hline\n"
        + "\n".join(tex_rows)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive.")
    torch.set_num_threads(min(os.cpu_count() or 1, 8))
    cache_path = args.cache.resolve()
    data = load_data(cache_path)
    fingerprint = {
        "cache_sha256": sha256_file(cache_path),
        "seeds": list(SEEDS),
        "adversarial_lambdas": list(ADVERSARIAL_LAMBDAS),
        "label_conditioned_lambdas": list(PENALTY_LAMBDAS),
        "failure_lambda": FAILURE_LAMBDA,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "threshold": THRESHOLD,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    if args.resume and args.checkpoint.is_file():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("input_fingerprint") != fingerprint:
            raise ValueError(
                "Checkpoint inputs do not match. Keep cache and settings fixed "
                "or choose another checkpoint file."
            )
        print(f"Resuming from {args.checkpoint}", flush=True)
    else:
        checkpoint = {
            "input_fingerprint": fingerprint,
            "experiments": {},
            "cross_fitted_thresholds": None,
        }

    def persist() -> None:
        atomic_write_json(args.checkpoint, checkpoint)

    started = time.perf_counter()
    train, val, test = (data[split] for split in SPLITS)
    raw = fit_ensemble("raw", data, checkpoint, persist)
    raw_auc_floor = raw["val_metrics"]["auc"] - 0.01
    print(
        f"Raw validation AUC={raw['val_metrics']['auc']:.4f}; "
        f"candidate constraint AUC >= {raw_auc_floor:.4f}",
        flush=True,
    )

    adversarial_runs: dict[float, dict[str, Any]] = {}
    for value in ADVERSARIAL_LAMBDAS:
        name = f"adversarial_lambda_{value:g}"
        adversarial_runs[value] = fit_ensemble(
            name,
            data,
            checkpoint,
            persist,
            mode="adversarial",
            strength=value,
        )
        metrics = adversarial_runs[value]["val_metrics"]
        print(
            f"  adversarial lambda={value:g}: val AUC={metrics['auc']:.4f} "
            f"EOD_max={metrics['eod_max']:.4f}",
            flush=True,
        )

    penalty_runs: dict[float, dict[str, Any]] = {}
    for value in PENALTY_LAMBDAS_ALL:
        name = f"label_conditioned_lambda_{value:g}"
        penalty_runs[value] = fit_ensemble(
            name,
            data,
            checkpoint,
            persist,
            mode="label_conditioned",
            strength=value,
        )
        metrics = penalty_runs[value]["val_metrics"]
        print(
            f"  label-conditioned lambda={value:g}: val AUC={metrics['auc']:.4f} "
            f"EOD_max={metrics['eod_max']:.4f}",
            flush=True,
        )

    eligible_adversarial = [
        value
        for value, result in adversarial_runs.items()
        if result["val_metrics"]["auc"] >= raw_auc_floor
    ]
    eligible_penalty = [
        value
        for value in PENALTY_LAMBDAS
        if penalty_runs[value]["val_metrics"]["auc"] >= raw_auc_floor
    ]
    if not eligible_adversarial:
        raise RuntimeError(
            "No adversarial lambda met the validation AUC constraint; "
            "refusing to choose a model outside the prespecified rule."
        )
    if not eligible_penalty:
        raise RuntimeError(
            "No label-conditioned lambda met the validation AUC constraint; "
            "refusing to choose a model outside the prespecified rule."
        )
    selected_adversarial = min(
        eligible_adversarial,
        key=lambda value: (
            adversarial_runs[value]["val_metrics"]["eod_max"],
            -adversarial_runs[value]["val_metrics"]["auc"],
            value,
        ),
    )
    selected_penalty = min(
        eligible_penalty,
        key=lambda value: (
            penalty_runs[value]["val_metrics"]["eod_max"],
            -penalty_runs[value]["val_metrics"]["auc"],
            value,
        ),
    )
    print(
        f"Selected by validation: adversarial lambda={selected_adversarial:g}; "
        f"label-conditioned lambda={selected_penalty:g}",
        flush=True,
    )

    weights = race_weights(train["races"])
    reweighted = fit_ensemble(
        "sample_reweighting",
        data,
        checkpoint,
        persist,
        sample_weights=weights,
    )
    if checkpoint.get("cross_fitted_thresholds") is None:
        thresholds, cutoff_metadata = cross_fitted_group_thresholds(
            train["features"],
            train["labels"],
            train["races"],
            checkpoint,
            persist,
        )
        checkpoint["cross_fitted_thresholds"] = {
            "thresholds": {str(key): value for key, value in thresholds.items()},
            "metadata": cutoff_metadata,
        }
        persist()
    else:
        saved = checkpoint["cross_fitted_thresholds"]
        thresholds = {
            int(key): float(value) for key, value in saved["thresholds"].items()
        }
        cutoff_metadata = saved["metadata"]

    probe_auc = linear_probe_macro_auc(
        train["features"], test["features"], train["races"], test["races"]
    )
    val_probe_auc = linear_probe_macro_auc(
        train["features"], val["features"], train["races"], val["races"]
    )
    selected_methods: dict[str, dict[str, Any]] = {
        "Raw Dual": raw,
        f"Adversarial (lambda={selected_adversarial:g})": adversarial_runs[
            selected_adversarial
        ],
        f"Label-conditioned (lambda={selected_penalty:g})": penalty_runs[
            selected_penalty
        ],
        "Sample reweighting": reweighted,
        "Cross-fitted group cutoffs": raw,
        f"Appendix: label-conditioned lambda={FAILURE_LAMBDA:g}": penalty_runs[
            FAILURE_LAMBDA
        ],
    }
    report_rows: dict[str, Any] = {}
    for name, result in selected_methods.items():
        operating_thresholds = (
            thresholds if name == "Cross-fitted group cutoffs" else None
        )
        test_metrics = method_report(
            result["test_probabilities"],
            test["labels"],
            test["races"],
            probe_auc,
            operating_thresholds,
        )
        validation_metrics = classification_metrics(
            val["labels"],
            val["races"],
            result["val_probabilities"],
            thresholds if name == "Cross-fitted group cutoffs" else None,
        )
        report_rows[name] = {
            "selected_on_validation": name
            in {
                f"Adversarial (lambda={selected_adversarial:g})",
                f"Label-conditioned (lambda={selected_penalty:g})",
            },
            "validation_metrics": {
                **validation_metrics,
                "auc_constraint_floor": raw_auc_floor,
                "auc_constraint_met": validation_metrics["auc"] >= raw_auc_floor,
                "linear_race_probe_macro_auc": val_probe_auc,
            },
            "test_metrics": test_metrics,
            "test_per_group_metrics": per_group_metrics(
                test["labels"],
                test["races"],
                result["test_probabilities"],
                operating_thresholds,
            ),
            "seed_best_epochs": result["seed_epochs"],
            "seed_validation_aucs": result["seed_val_aucs"],
        }
    report_rows["Cross-fitted group cutoffs"]["thresholds"] = cutoff_metadata

    pairwise_predictions = {
        name: selected_methods[name]["test_probabilities"]
        for name in selected_methods
        if name not in ("Raw Dual", "Cross-fitted group cutoffs")
    }
    pairwise_predictions["Cross-fitted group cutoffs"] = raw["test_probabilities"]
    probes = {name: probe_auc for name in pairwise_predictions}
    bootstrap = paired_bootstrap(
        test["labels"],
        test["races"],
        pairwise_predictions,
        raw["test_probabilities"],
        args.bootstrap_resamples,
        args.bootstrap_seed,
        thresholds,
        probe_auc,
        probes,
    )
    for name, result in bootstrap.items():
        report_rows[name]["paired_bootstrap_vs_raw"] = result

    # Lambda=5 is reported separately and is never part of validation selection.
    appendix_name = f"Appendix: label-conditioned lambda={FAILURE_LAMBDA:g}"
    report_rows[appendix_name]["failure_case"] = {
        "is_in_selection_grid": False,
        "validation_auc_constraint_met": (
            report_rows[appendix_name]["validation_metrics"]["auc"]
            >= raw_auc_floor
        ),
        "selection_note": (
            "Reported separately as requested; not eligible for lambda selection."
        ),
    }

    table_order = [
        "Raw Dual",
        f"Adversarial (lambda={selected_adversarial:g})",
        f"Label-conditioned (lambda={selected_penalty:g})",
        "Sample reweighting",
        "Cross-fitted group cutoffs",
        appendix_name,
    ]
    table_contents = [
        (
            name,
            {
                **report_rows[name],
                "paired_bootstrap_vs_raw": (
                    {}
                    if name == "Raw Dual"
                    else report_rows[name]["paired_bootstrap_vs_raw"]
                ),
            },
        )
        for name in table_order
    ]
    results: dict[str, Any] = {
        "method": "Five-seed validation-selected fairness baselines",
        "selection_rule": {
            "objective": "Lowest validation EOD_max among candidates meeting AUC floor.",
            "auc_constraint": "validation AUC >= raw validation AUC - 0.01",
            "raw_validation_auc": raw["val_metrics"]["auc"],
            "minimum_validation_auc": raw_auc_floor,
            "adversarial_lambda_grid": list(ADVERSARIAL_LAMBDAS),
            "selected_adversarial_lambda": selected_adversarial,
            "adversarial_candidates": {
                str(value): {
                    "val_auc": adversarial_runs[value]["val_metrics"]["auc"],
                    "val_eod_max": adversarial_runs[value]["val_metrics"]["eod_max"],
                    "eligible": value in eligible_adversarial,
                }
                for value in ADVERSARIAL_LAMBDAS
            },
            "label_conditioned_lambda_grid": list(PENALTY_LAMBDAS),
            "selected_label_conditioned_lambda": selected_penalty,
            "label_conditioned_candidates": {
                str(value): {
                    "val_auc": penalty_runs[value]["val_metrics"]["auc"],
                    "val_eod_max": penalty_runs[value]["val_metrics"]["eod_max"],
                    "eligible": value in eligible_penalty,
                }
                for value in PENALTY_LAMBDAS
            },
            "lambda_5_appendix": {
                "val_auc": penalty_runs[FAILURE_LAMBDA]["val_metrics"]["auc"],
                "val_eod_max": penalty_runs[FAILURE_LAMBDA]["val_metrics"]["eod_max"],
                "auc_constraint_met": (
                    penalty_runs[FAILURE_LAMBDA]["val_metrics"]["auc"]
                    >= raw_auc_floor
                ),
                "is_in_selection_grid": False,
            },
            "test_used_for_selection": False,
        },
        "configuration": {
            **fingerprint,
            "head": (
                "Linear(1024,256)-LayerNorm-GELU-Dropout(0.2)-Linear(256,2)"
            ),
            "test_race_counts": dict(zip(RACE_NAMES, EXPECTED_TEST_RACE_COUNTS)),
            "linear_race_probe": (
                "StandardScaler fit on train; LogisticRegression(C=0.01, "
                "max_iter=1000, lbfgs); macro one-vs-rest AUC."
            ),
            "cross_fitted_cutoffs": cutoff_metadata,
            "sample_reweighting": (
                "Inverse train race-frequency weights normalized to mean 1."
            ),
            "label_conditioned_penalty": (
                "For glaucoma labels 0 and 1, sum race-wise max-minus-min "
                "mean positive probabilities within each minibatch."
            ),
            "adversarial_method": (
                "Race classifier on hidden representation; gradient reversal "
                "strength lambda; diagnostic CE plus race CE."
            ),
        },
        "table_order": table_order,
        "methods": report_rows,
        "paired_bootstrap_vs_raw": bootstrap,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_json, results)
    args.output_tex.write_text(latex_table(table_contents), encoding="utf-8")

    print("\nTest comparison (fixed 0.50, except cross-fitted cutoffs)")
    for name in table_order:
        row = report_rows[name]
        metrics = row["test_metrics"]
        print(
            f"{name:<46} AUC={metrics['auc']:.4f} "
            f"Acc={metrics['accuracy']:.4f} DPD={metrics['dpd']:.4f} "
            f"EOD_max={metrics['eod_max']:.4f} "
            f"linear-race-AUC={metrics['linear_race_probe_macro_auc']:.4f}"
        )
        if "paired_bootstrap_vs_raw" in row:
            auc_delta = row["paired_bootstrap_vs_raw"]["metrics"]["auc"]
            eod_delta = row["paired_bootstrap_vs_raw"]["metrics"]["eod_max"]
            print(
                f"    paired delta AUC={auc_delta['mean_delta']:+.4f} "
                f"[{auc_delta['percentile_2_5']:+.4f},"
                f"{auc_delta['percentile_97_5']:+.4f}] "
                f"EOD={eod_delta['mean_delta']:+.4f} "
                f"[{eod_delta['percentile_2_5']:+.4f},"
                f"{eod_delta['percentile_97_5']:+.4f}]"
            )
    print(f"\nSaved {args.output_json} and {args.output_tex}.")


if __name__ == "__main__":
    main()
