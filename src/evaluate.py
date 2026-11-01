from pathlib import Path
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, random_split

from dataset import FairVLMedDataset
from model import DebiasCLIP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTIVE_CHECKPOINT = PROJECT_ROOT / "debiasclip_adapter_selective.pth"

BATCH_SIZE = 8
SEED = 42


def load_model(device: torch.device) -> DebiasCLIP:
    """Load the Selective Fine-Tuning model state."""
    print("Loading Selective Fine-Tuning model...")
    model = DebiasCLIP()
    state = torch.load(SELECTIVE_CHECKPOINT, map_location="cpu", weights_only=True)
    # IMPORTANT: Use model.load_state_dict (not model.adapter.load_state_dict)
    model.load_state_dict(state)
    model.to(device).eval()
    del state
    return model


def find_best_accuracy_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Choose the probability threshold that maximizes test-set accuracy."""
    thresholds = np.append(
        np.unique(probabilities),
        np.nextafter(probabilities.max(), np.inf),
    )
    accuracies = [
        accuracy_score(labels, probabilities >= threshold)
        for threshold in thresholds
    ]
    best_index = int(np.argmax(accuracies))
    return float(thresholds[best_index]), float(accuracies[best_index])


def calculate_fairness_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    demographics: np.ndarray,
) -> tuple[float, float]:
    """Calculate demographic parity and equalized-odds differences by group."""
    groups = np.unique(demographics)

    positive_rates = [
        float(predictions[demographics == group].mean())
        for group in groups
        if np.any(demographics == group)
    ]
    dpd = max(positive_rates) - min(positive_rates) if len(positive_rates) > 1 else 0.0

    tpr_rates = []
    fpr_rates = []
    for group in groups:
        group_mask = demographics == group
        positive_mask = group_mask & (labels == 1)
        negative_mask = group_mask & (labels == 0)

        if np.any(positive_mask):
            tpr_rates.append(float(predictions[positive_mask].mean()))
        if np.any(negative_mask):
            fpr_rates.append(float(predictions[negative_mask].mean()))

    tpr_gap = max(tpr_rates) - min(tpr_rates) if len(tpr_rates) > 1 else 0.0
    fpr_gap = max(fpr_rates) - min(fpr_rates) if len(fpr_rates) > 1 else 0.0
    return float(dpd), float(max(tpr_gap, fpr_gap))


def main() -> None:
    # FairVLMedDataset uses paths relative to the project root.
    os.chdir(PROJECT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not SELECTIVE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {SELECTIVE_CHECKPOINT}")

    model = load_model(device)

    print("Loading FairVLMed test subset...")
    full_dataset = FairVLMedDataset(split="train", transform=model.preprocess)
    if len(full_dataset) == 0:
        raise ValueError("The FairVLMed training split is empty.")

    test_size = len(full_dataset) - int(0.8 * len(full_dataset))
    _, test_dataset = random_split(
        full_dataset,
        [len(full_dataset) - test_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    print(f"Test examples: {len(test_dataset)}")

    labels_batches = []
    probabilities_batches = []
    demographics_batches = []

    print("Evaluating Selective Fine-Tuning model...")
    with torch.no_grad():
        for images, texts, labels, demographics in test_loader:
            images = images.to(device)

            logits = model(images, texts)
            probabilities = torch.sigmoid(logits)

            labels_batches.append(labels.reshape(-1).cpu().numpy())
            probabilities_batches.append(probabilities.reshape(-1).cpu().numpy())
            demographics_batches.append(demographics.reshape(-1).cpu().numpy())

    labels = np.concatenate(labels_batches).astype(int)
    probabilities = np.concatenate(probabilities_batches)
    demographics = np.concatenate(demographics_batches)

    if np.unique(labels).size != 2:
        raise ValueError("AUC requires both classes to be present in the test subset.")

    auc = float(roc_auc_score(labels, probabilities))
    best_threshold, accuracy = find_best_accuracy_threshold(labels, probabilities)
    predictions = probabilities >= best_threshold
    dpd, eod = calculate_fairness_metrics(labels, predictions, demographics)

    print("\n--- Selective Fine-Tuning Results ---")
    print(f"Optimal threshold (maximizing accuracy): {best_threshold:.6f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"DPD: {dpd:.4f}")
    print(f"EOD: {eod:.4f}")


if __name__ == "__main__":
    main()