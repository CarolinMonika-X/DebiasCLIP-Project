"""Fine-tune the DebiasCLIP adapter using the fixed cached-feature splits."""
import copy
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import build_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "cached_features.pt"
INITIAL_CHECKPOINT = PROJECT_ROOT / "debiasclip_adapter_v4.pth"
OUTPUT_CHECKPOINT = PROJECT_ROOT / "debiasclip_adapter_v7.pth"
SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
PATIENCE = 7
IMAGE_RESIDUAL_WEIGHT = 0.1


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.append(
        np.unique(probabilities), np.nextafter(probabilities.max(), np.inf)
    )
    accuracies = np.asarray(
        [accuracy_score(labels, probabilities >= threshold) for threshold in candidates]
    )
    best_accuracy = float(accuracies.max())
    tied = candidates[accuracies == best_accuracy]
    best_threshold = float(tied[np.argmin(np.abs(tied - 0.5))])
    return best_threshold, best_accuracy


def make_logits(adapter: nn.Module, features: torch.Tensor) -> torch.Tensor:
    image_features = features[:, :512]
    return adapter(features).squeeze(-1) + IMAGE_RESIDUAL_WEIGHT * image_features.mean(dim=-1)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"Cached features not found: {CACHE_PATH}")
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Initial adapter checkpoint not found: {INITIAL_CHECKPOINT}")

    data = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    if not {"train", "val", "test"}.issubset(data):
        raise ValueError("cached_features.pt must contain train, val, and test splits.")

    train_features = torch.cat((data["train"]["img"], data["train"]["txt"]), dim=1)
    train_labels = data["train"]["labels"].float().reshape(-1)
    val_features = torch.cat((data["val"]["img"], data["val"]["txt"]), dim=1)
    val_labels = data["val"]["labels"].float().reshape(-1)

    adapter = build_adapter()
    initial_state = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=True)
    adapter.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    def validate() -> tuple[float, float, float]:
        adapter.eval()
        with torch.no_grad():
            logits = make_logits(adapter, val_features)
            loss = criterion(logits, val_labels).item()
            probabilities = torch.sigmoid(logits).numpy()
        threshold, accuracy = choose_threshold(val_labels.numpy(), probabilities)
        return loss, threshold, accuracy

    best_loss, best_threshold, best_accuracy = validate()
    best_state = copy.deepcopy(adapter.state_dict())
    best_epoch = 0
    lowest_val_loss = best_loss
    epochs_without_loss_improvement = 0
    print(
        f"Initial adapter: val loss={best_loss:.4f}, "
        f"val accuracy={best_accuracy:.4f}, threshold={best_threshold:.4f}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        adapter.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(make_logits(adapter, features), labels)
            loss.backward()
            optimizer.step()

        val_loss, threshold, accuracy = validate()
        print(
            f"Epoch {epoch:02d}: val loss={val_loss:.4f}, "
            f"val accuracy={accuracy:.4f}, threshold={threshold:.4f}"
        )

        if accuracy > best_accuracy or (
            accuracy == best_accuracy and val_loss < best_loss
        ):
            best_state = copy.deepcopy(adapter.state_dict())
            best_loss = val_loss
            best_threshold = threshold
            best_accuracy = accuracy
            best_epoch = epoch

        if val_loss < lowest_val_loss - 1e-5:
            lowest_val_loss = val_loss
            epochs_without_loss_improvement = 0
        else:
            epochs_without_loss_improvement += 1
            if epochs_without_loss_improvement >= PATIENCE:
                break

    torch.save(
        {
            "adapter_state_dict": best_state,
            "decision_threshold": best_threshold,
            "validation_accuracy": best_accuracy,
            "validation_loss": best_loss,
            "best_epoch": best_epoch,
            "seed": SEED,
            "initial_checkpoint": INITIAL_CHECKPOINT.name,
        },
        OUTPUT_CHECKPOINT,
    )
    print(
        f"Saved {OUTPUT_CHECKPOINT.name}: best epoch={best_epoch}, "
        f"validation accuracy={best_accuracy:.4f}, threshold={best_threshold:.4f}"
    )
    print("The held-out test split was not used for training or model selection.")


if __name__ == "__main__":
    main()
