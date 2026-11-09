"""Tune a residual adapter using only the train and validation cached splits."""
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "cached_features.pt"
CONFIG_PATH = PROJECT_ROOT / "best_head_config.json"
SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 8
HIDDEN_SIZES = (128, 256, 512)
DROPOUTS = (0.2, 0.3, 0.5)
WEIGHT_DECAYS = (1e-3, 1e-2, 1e-1)
LEARNING_RATES = (3e-4, 1e-3)


class Adapter(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = x + self.alpha * self.mlp(x)
        return self.head(x).squeeze(-1)


def load_splits():
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"Cached features not found: {CACHE_PATH}")
    data = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    if not {"train", "val"}.issubset(data):
        raise ValueError("cached_features.pt must contain train and val splits.")
    scaler_img = StandardScaler().fit(data["train"]["img"].numpy())
    scaler_txt = StandardScaler().fit(data["train"]["txt"].numpy())

    def prepare(split):
        image = scaler_img.transform(split["img"].numpy())
        text = scaler_txt.transform(split["txt"].numpy())
        features = torch.tensor(np.concatenate((image, text), axis=1), dtype=torch.float32)
        labels = split["labels"].float().reshape(-1)
        return features, labels

    return prepare(data["train"]), prepare(data["val"])


def train_config(train_data, val_data, config):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_features, train_labels = train_data
    val_features, val_labels = val_data
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
        generator=torch.Generator().manual_seed(SEED),
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(val_features)).numpy()
    labels = val_labels.numpy().astype(int)
    return {
        **config,
        "validation_auc": float(roc_auc_score(labels, probabilities)),
        "validation_accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "best_epoch": best_epoch,
    }


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_data, val_data = load_splits()
    results = []

    print("hidden,dropout,weight_decay,lr,validation_auc,validation_accuracy")
    for hidden in HIDDEN_SIZES:
        for dropout in DROPOUTS:
            for weight_decay in WEIGHT_DECAYS:
                for lr in LEARNING_RATES:
                    config = {
                        "hidden": hidden,
                        "dropout": dropout,
                        "weight_decay": weight_decay,
                        "lr": lr,
                    }
                    result = train_config(train_data, val_data, config)
                    results.append(result)
                    print(
                        f"{hidden},{dropout},{weight_decay},{lr},"
                        f"{result['validation_auc']:.10f},"
                        f"{result['validation_accuracy']:.10f}"
                    )

    best = max(
        results,
        key=lambda result: (
            result["validation_auc"],
            result["validation_accuracy"],
            -result["hidden"],
            -result["dropout"],
            -result["weight_decay"],
            -result["lr"],
        ),
    )
    output = {
        key: best[key]
        for key in (
            "hidden",
            "dropout",
            "weight_decay",
            "lr",
            "validation_auc",
            "validation_accuracy",
            "best_epoch",
        )
    }
    output["seed"] = SEED
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
