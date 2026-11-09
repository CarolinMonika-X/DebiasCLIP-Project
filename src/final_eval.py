"""Final held-out evaluation for the tuned adapter and retrained model C."""
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
OUTPUT_PATH = PROJECT_ROOT / "final_eval_output.txt"
SEEDS = range(5)
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 8
BOOTSTRAP_RESAMPLES = 1000
THRESHOLD_DEFAULT = 0.5


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

    scaler_img = StandardScaler().fit(data["train"]["img"].numpy())
    scaler_txt = StandardScaler().fit(data["train"]["txt"].numpy())

    def prepare(split):
        image = scaler_img.transform(split["img"].numpy())
        text = scaler_txt.transform(split["txt"].numpy())
        features = torch.tensor(np.concatenate((image, text), axis=1), dtype=torch.float32)
        labels = split["labels"].float().reshape(-1)
        races = split["races"].long().reshape(-1)
        return features, labels, races

    return data, config, prepare


def train_one(train_features, train_labels, val_features, val_labels, settings, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Adapter(
        in_dim=train_features.shape[1],
        hidden=settings["hidden"],
        dropout=settings["dropout"],
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
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
    with torch.inference_mode():
        val_probabilities = torch.sigmoid(model(val_features)).numpy()
    return model, val_probabilities, best_epoch, best_loss


def choose_validation_threshold(labels, probabilities):
    candidates = np.append(
        np.unique(probabilities), np.nextafter(probabilities.max(), np.inf)
    )
    accuracies = np.asarray(
        [accuracy_score(labels, probabilities >= threshold) for threshold in candidates]
    )
    best_accuracy = accuracies.max()
    tied = candidates[accuracies == best_accuracy]
    threshold = float(tied[np.argmin(np.abs(tied - THRESHOLD_DEFAULT))])
    return threshold, float(best_accuracy)


def calculate_metrics(probabilities, labels, races, threshold):
    predictions = probabilities >= threshold
    groups = np.unique(races)
    group_rates = [float(predictions[races == group].mean()) for group in groups]
    dpd = max(group_rates) - min(group_rates) if len(group_rates) > 1 else 0.0

    conditional_gaps = []
    for label in (0, 1):
        rates = []
        for group in groups:
            mask = (races == group) & (labels == label)
            if np.any(mask):
                rates.append(float(predictions[mask].mean()))
        if len(rates) > 1:
            conditional_gaps.append(max(rates) - min(rates))
    eod = max(conditional_gaps) if conditional_gaps else 0.0

    group_aucs = [
        float(roc_auc_score(labels[races == group], probabilities[races == group]))
        for group in groups
        if np.unique(labels[races == group]).size == 2
    ]
    worst_group_auc = min(group_aucs) if group_aucs else float("nan")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "dpd": float(dpd),
        "eod": float(eod),
        "worst_group_auc": float(worst_group_auc),
    }


def bootstrap_auc_difference(candidate, baseline, labels, seed=2026):
    rng = np.random.default_rng(seed)
    differences = []
    attempts = 0
    max_attempts = BOOTSTRAP_RESAMPLES * 10
    while len(differences) < BOOTSTRAP_RESAMPLES and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[indices]).size != 2:
            continue
        differences.append(
            roc_auc_score(labels[indices], candidate[indices])
            - roc_auc_score(labels[indices], baseline[indices])
        )
    if len(differences) != BOOTSTRAP_RESAMPLES:
        raise RuntimeError(
            f"Only obtained {len(differences)} valid bootstrap samples "
            f"after {attempts} attempts."
        )
    return (
        float(np.mean(differences)),
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    )


def run():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data, config, prepare = load_data()
    (train_features, train_labels) = prepare(data["train"])[:2]
    (val_features, val_labels) = prepare(data["val"])[:2]
    val_labels_np = val_labels.numpy().astype(int)

    print("Selected tuned-head configuration:")
    print(json.dumps(config, indent=2, sort_keys=True))

    tuned_models = []
    tuned_val_probabilities = []
    for seed in SEEDS:
        model, val_probabilities, epoch, val_loss = train_one(
            train_features,
            train_labels,
            val_features,
            val_labels,
            config,
            seed,
        )
        tuned_models.append(model)
        tuned_val_probabilities.append(val_probabilities)
        print(f"Tuned head seed {seed}: best_epoch={epoch}, validation_loss={val_loss:.10f}")

    tuned_val_ensemble = np.mean(tuned_val_probabilities, axis=0)
    threshold, val_accuracy = choose_validation_threshold(val_labels_np, tuned_val_ensemble)
    print(f"Validation ensemble accuracy at 0.5: {accuracy_score(val_labels_np, tuned_val_ensemble >= 0.5):.10f}")
    print(f"Validation-selected threshold: {threshold:.10f}")
    print(f"Validation ensemble accuracy at selected threshold: {val_accuracy:.10f}")

    # Model C is the unpenalized dual-modal baseline from ablation_full.py.
    model_c_settings = {
        "hidden": 256,
        "dropout": 0.2,
        "weight_decay": 1e-2,
        "lr": 1e-3,
    }
    baseline_models = []
    for seed in SEEDS:
        model, _, epoch, val_loss = train_one(
            train_features,
            train_labels,
            val_features,
            val_labels,
            model_c_settings,
            seed,
        )
        baseline_models.append(model)
        print(f"Model C seed {seed}: best_epoch={epoch}, validation_loss={val_loss:.10f}")

    test_features, test_labels_tensor, test_races_tensor = prepare(data["test"])
    test_labels = test_labels_tensor.numpy().astype(int)
    test_races = test_races_tensor.numpy()
    with torch.inference_mode():
        tuned_test_probabilities = np.mean(
            [torch.sigmoid(model(test_features)).numpy() for model in tuned_models],
            axis=0,
        )
        baseline_test_probabilities = np.mean(
            [torch.sigmoid(model(test_features)).numpy() for model in baseline_models],
            axis=0,
        )

    print(f"Test examples: {len(test_labels)}")
    print("Tuned head test metrics at threshold 0.5:")
    for name, value in calculate_metrics(
        tuned_test_probabilities, test_labels, test_races, 0.5
    ).items():
        print(f"{name}: {value:.10f}")
    print(f"Tuned head test metrics at validation-selected threshold {threshold:.10f}:")
    for name, value in calculate_metrics(
        tuned_test_probabilities, test_labels, test_races, threshold
    ).items():
        print(f"{name}: {value:.10f}")

    baseline_auc = float(roc_auc_score(test_labels, baseline_test_probabilities))
    tuned_auc = float(roc_auc_score(test_labels, tuned_test_probabilities))
    difference, ci_low, ci_high = bootstrap_auc_difference(
        tuned_test_probabilities,
        baseline_test_probabilities,
        test_labels,
    )
    print("AUC difference against retrained model C (tuned head minus model C):")
    print(f"Tuned-head test AUC: {tuned_auc:.10f}")
    print(f"Model C test AUC: {baseline_auc:.10f}")
    print(f"Observed AUC difference: {tuned_auc - baseline_auc:.10f}")
    print(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES}")
    print(f"Bootstrap mean AUC difference: {difference:.10f}")
    print(f"Bootstrap 95% CI: [{ci_low:.10f}, {ci_high:.10f}]")


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
