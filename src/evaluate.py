"""Evaluate the trained adapter on the fixed, held-out cached test split."""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from model import build_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "cached_features.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "debiasclip_adapter_v7.pth"
IMAGE_RESIDUAL_WEIGHT = 0.1


def calculate_fairness_metrics(
    labels: np.ndarray, predictions: np.ndarray, demographics: np.ndarray
) -> tuple[float, float]:
    groups = np.unique(demographics)
    positive_rates = [predictions[demographics == group].mean() for group in groups]
    dpd = float(max(positive_rates) - min(positive_rates)) if len(positive_rates) > 1 else 0.0

    gaps = []
    for label in (0, 1):
        rates = []
        for group in groups:
            mask = (demographics == group) & (labels == label)
            if mask.any():
                rates.append(float(predictions[mask].mean()))
        if len(rates) > 1:
            gaps.append(max(rates) - min(rates))
    eod = float(max(gaps)) if gaps else 0.0
    return dpd, eod


def main() -> None:
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"Cached features not found: {CACHE_PATH}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {CHECKPOINT_PATH}")

    data = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if "test" not in data:
        raise ValueError("cached_features.pt does not contain a test split.")
    if "adapter_state_dict" not in checkpoint or "decision_threshold" not in checkpoint:
        raise ValueError(f"Checkpoint has an unsupported format: {CHECKPOINT_PATH}")

    adapter = build_adapter()
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()

    test = data["test"]
    features = torch.cat((test["img"], test["txt"]), dim=1)
    labels = test["labels"].numpy().astype(int).reshape(-1)
    demographics = test["races"].numpy().reshape(-1)
    threshold = float(checkpoint["decision_threshold"])

    with torch.inference_mode():
        logits = (
            adapter(features).squeeze(-1)
            + IMAGE_RESIDUAL_WEIGHT * features[:, :512].mean(dim=-1)
        )
        probabilities = torch.sigmoid(logits).numpy()

    if np.unique(labels).size != 2:
        raise ValueError("AUC requires both classes to be present in the test split.")

    predictions = probabilities >= threshold
    dpd, eod = calculate_fairness_metrics(labels, predictions, demographics)
    print(f"Test examples: {len(labels)}")
    print(f"Validation-selected threshold: {threshold:.6f}")
    print(f"Accuracy at threshold 0.5: {accuracy_score(labels, probabilities >= 0.5):.4f}")
    print(f"Accuracy at validation threshold: {accuracy_score(labels, predictions):.4f}")
    print(f"AUC: {roc_auc_score(labels, probabilities):.4f}")
    print(f"DPD: {dpd:.4f}")
    print(f"EOD: {eod:.4f}")


if __name__ == "__main__":
    main()
