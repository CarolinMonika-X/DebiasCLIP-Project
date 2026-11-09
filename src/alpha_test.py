"""Compare image-feature scaling using train/validation only for model selection."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from tune_head import Adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "cached_features.pt"
CONFIG_PATH = PROJECT_ROOT / "best_head_config.json"
OUTPUT_PATH = PROJECT_ROOT / "alpha_test_output.txt"
SEEDS = range(5)
ALPHAS = (1.0, 0.5, 0.25)
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 8
BOOTSTRAP_RESAMPLES = 1000
THRESHOLD = 0.5
METRIC_NAMES = ("accuracy", "auc", "dpd", "eod", "worst_group_auc")


def load_data():
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"Cached features not found: {CACHE_PATH}")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Best head configuration not found: {CONFIG_PATH}")

    data = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    if not {"train", "val", "test"}.issubset(data):
        raise ValueError("cached_features.pt must contain train, val, and test splits.")
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    required = {"hidden", "dropout", "weight_decay", "lr"}
    if not required.issubset(config):
        raise ValueError(f"{CONFIG_PATH.name} is missing settings: {sorted(required - config.keys())}")

    image_scaler = StandardScaler().fit(data["train"]["img"].numpy())
    text_scaler = StandardScaler().fit(data["train"]["txt"].numpy())

    def prepare(split, alpha):
        image = image_scaler.transform(split["img"].numpy()) * alpha
        text = text_scaler.transform(split["txt"].numpy())
        features = torch.tensor(np.concatenate((image, text), axis=1), dtype=torch.float32)
        labels = split["labels"].float().reshape(-1)
        races = split["races"].long().reshape(-1)
        return features, labels, races

    return data, config, prepare


def train_one(train_features, train_labels, val_features, val_labels, config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = Adapter(
        in_dim=train_features.shape[1],
        hidden=config["hidden"],
        dropout=config["dropout"],
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_count = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_features), val_labels).item()
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"Seed {seed} training produced no validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def calculate_metrics(probabilities, labels, races):
    predictions = probabilities >= THRESHOLD
    groups = np.unique(races)
    group_rates = [float(predictions[races == group].mean()) for group in groups]
    dpd = max(group_rates) - min(group_rates) if len(group_rates) > 1 else 0.0
    gaps = []
    for label in (0, 1):
        rates = []
        for group in groups:
            mask = (races == group) & (labels == label)
            if mask.any():
                rates.append(float(predictions[mask].mean()))
        if len(rates) > 1:
            gaps.append(max(rates) - min(rates))
    eod = max(gaps) if gaps else 0.0
    group_aucs = [
        float(roc_auc_score(labels[races == group], probabilities[races == group]))
        for group in groups
        if np.unique(labels[races == group]).size == 2
    ]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "dpd": float(dpd),
        "eod": float(eod),
        "worst_group_auc": min(group_aucs) if group_aucs else float("nan"),
    }


def bootstrap_differences(alpha_probabilities, baseline_probabilities, labels, races, seed):
    rng = np.random.default_rng(seed)
    differences = {name: [] for name in METRIC_NAMES}
    valid = 0
    attempts = 0
    max_attempts = BOOTSTRAP_RESAMPLES * 10
    while valid < BOOTSTRAP_RESAMPLES and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        sample_labels = labels[indices]
        if np.unique(sample_labels).size != 2:
            continue
        alpha_metrics = calculate_metrics(
            alpha_probabilities[indices], sample_labels, races[indices]
        )
        baseline_metrics = calculate_metrics(
            baseline_probabilities[indices], sample_labels, races[indices]
        )
        if not np.isfinite(alpha_metrics["worst_group_auc"]) or not np.isfinite(
            baseline_metrics["worst_group_auc"]
        ):
            continue
        for name in METRIC_NAMES:
            differences[name].append(alpha_metrics[name] - baseline_metrics[name])
        valid += 1
    if valid != BOOTSTRAP_RESAMPLES:
        raise RuntimeError(
            f"Only obtained {valid} valid bootstrap samples after {attempts} attempts."
        )
    return {
        name: (
            float(np.mean(values)),
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
        for name, values in differences.items()
    }


def run():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data, config, prepare = load_data()
    print("Selected head configuration:")
    print(json.dumps(config, indent=2, sort_keys=True))

    train_labels = data["train"]["labels"].float().reshape(-1)
    val_labels = data["val"]["labels"].float().reshape(-1)
    test_labels = data["test"]["labels"].numpy().astype(int)
    test_races = data["test"]["races"].numpy()
    per_alpha_metrics = {}
    ensemble_probabilities = {}

    for alpha in ALPHAS:
        train_features, _, _ = prepare(data["train"], alpha)
        val_features, _, _ = prepare(data["val"], alpha)
        test_features, _, _ = prepare(data["test"], alpha)
        seed_probabilities = []
        seed_metrics = []
        for seed in SEEDS:
            model, best_epoch = train_one(
                train_features,
                train_labels,
                val_features,
                val_labels,
                config,
                seed,
            )
            with torch.inference_mode():
                probabilities = torch.sigmoid(model(test_features)).numpy()
            seed_probabilities.append(probabilities)
            seed_metrics.append(calculate_metrics(probabilities, test_labels, test_races))
            print(f"alpha={alpha:g}, seed={seed}, best_epoch={best_epoch}")

        ensemble_probabilities[alpha] = np.mean(seed_probabilities, axis=0)
        per_alpha_metrics[alpha] = {
            name: (
                float(np.mean([metrics[name] for metrics in seed_metrics])),
                float(np.std([metrics[name] for metrics in seed_metrics])),
            )
            for name in METRIC_NAMES
        }

    print(f"\nPer-seed test metrics: mean and population std over {len(SEEDS)} seeds")
    print(f"Decision threshold: {THRESHOLD:.10f}")
    for alpha in ALPHAS:
        print(f"\nalpha={alpha:g}")
        for name in METRIC_NAMES:
            mean, std = per_alpha_metrics[alpha][name]
            print(f"{name}: mean={mean:.10f}, std={std:.10f}")

    baseline = ensemble_probabilities[1.0]
    print(
        f"\nPaired {BOOTSTRAP_RESAMPLES}-resample bootstrap differences "
        "(alpha minus alpha=1.0, seed-ensemble test probabilities)"
    )
    for alpha in ALPHAS:
        result = bootstrap_differences(
            ensemble_probabilities[alpha],
            baseline,
            test_labels,
            test_races,
            seed=2026 + int(alpha * 100),
        )
        print(f"\nalpha={alpha:g} versus alpha=1.0")
        for name in METRIC_NAMES:
            mean, low, high = result[name]
            print(
                f"{name} difference: mean={mean:.10f}, "
                f"95% CI=[{low:.10f}, {high:.10f}]"
            )


if __name__ == "__main__":
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        import sys

        class Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, text):
                for stream in self.streams:
                    stream.write(text)
                    stream.flush()

            def flush(self):
                for stream in self.streams:
                    stream.flush()

        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, output)
        try:
            run()
        finally:
            sys.stdout = original_stdout
