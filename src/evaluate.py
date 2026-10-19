import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score
from dataset import FairVLMedDataset
from model import DebiasCLIP
import numpy as np
import open_clip

# ==========================================================
# EVALUATION MODE SETUP
# 1 = Zero-shot CLIP (Baseline)
# 2 = Adapter-only (No fairness loss) (Baseline)
# 3 = DebiasCLIP V3 (Dual-Modal + Residual)
# ==========================================================
EVALUATION_MODE = 3   
# ==========================================================

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

print("Loading DebiasCLIP model...")
model = DebiasCLIP().to(DEVICE)
model.eval()

# --- Load Weights based on Mode ---
if EVALUATION_MODE == 1:
    print(">>> MODE 1: Zero-shot CLIP (No Adapter Weights) <<<")
    
elif EVALUATION_MODE == 2:
    print(">>> MODE 2: Adapter-only Baseline (No Fairness Loss) <<<")
    try:
        model.adapter.load_state_dict(torch.load("adapter_baseline.pth", map_location=DEVICE))
        print("Loaded adapter_baseline.pth successfully.")
    except FileNotFoundError:
        print("ERROR: 'adapter_baseline.pth' file illa.")
    
elif EVALUATION_MODE == 3:
    print(">>> MODE 3: DebiasCLIP V3 (Dual-Modal Model) <<<")
    try:
        # Ippo pudhu V3 model load pannuvom
        model.adapter.load_state_dict(torch.load("debiasclip_adapter_v3.pth", map_location=DEVICE))
        print("Loaded debiasclip_adapter_v3.pth successfully.")
    except FileNotFoundError:
        print("ERROR: 'debiasclip_adapter_v3.pth' file illa. Mudhalla V3 model train pannunga.")

print("Model loaded successfully!")

# --- Test Dataset Load Pannunga ---
preprocess = model.preprocess
full_dataset = FairVLMedDataset(split='train', transform=preprocess)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
_, test_dataset = random_split(full_dataset, [train_size, test_size])

print(f"Test size: {len(test_dataset)}")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Evaluate Pannunga ---
all_labels = []
all_preds = []
all_demographics = []

print("Evaluating... (Idhu konjam neram aagum)")
with torch.no_grad():
    for images, texts, labels, demographics in test_loader:
        images = images.to(DEVICE)
        
        if EVALUATION_MODE == 1:
            image_features = model.clip_model.encode_image(images)
            text_tokens = open_clip.tokenize(texts).to(DEVICE)
            text_features = model.clip_model.encode_text(text_tokens)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logits = (image_features * text_features).sum(dim=-1) * 100
        
        else:
            logits = model(images, texts)
            
        probs = torch.sigmoid(logits).cpu().numpy()
        
        all_labels.extend(labels.numpy())
        all_preds.extend(probs)
        all_demographics.extend(demographics.numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_demographics = np.array(all_demographics)

# --- Overall Metrics ---
accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
auc = roc_auc_score(all_labels, all_preds)

print(f"\n--- Overall Results ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

# --- Fairness Metrics (Per Demographic Group) ---
group_positive_rates = []
print(f"\n--- Per Group Results ---")
for demo in np.unique(all_demographics):
    mask = all_demographics == demo
    if mask.sum() > 0:
        group_acc = accuracy_score(all_labels[mask], (all_preds[mask] > 0.5).astype(int))
        try:
            group_auc = roc_auc_score(all_labels[mask], all_preds[mask])
        except ValueError:
            group_auc = 0.0
        positive_rate = (all_preds[mask] > 0.5).mean()
        group_positive_rates.append(positive_rate)
        
        print(f"Group {demo}: Accuracy={group_acc:.4f}, AUC={group_auc:.4f}, Positive Rate={positive_rate:.4f}")

# --- DPD Calculation ---
if len(group_positive_rates) > 1:
    dpd = max(group_positive_rates) - min(group_positive_rates)
    print(f"\nDemographic Parity Difference (DPD): {dpd:.4f}")

# --- EOD Calculation ---
print(f"\n--- EOD Calculation ---")
eod_tpr = 0.0
eod_fpr = 0.0

for y_val in [0, 1]:
    group_rates = []
    for demo in np.unique(all_demographics):
        mask = (all_demographics == demo) & (all_labels == y_val)
        if mask.sum() > 0:
            rate = (all_preds[mask] > 0.5).mean()
            group_rates.append(rate)
    
    if len(group_rates) > 1:
        gap = max(group_rates) - min(group_rates)
        if y_val == 1:
            eod_tpr = gap
            print(f"TPR Gap (Equal Opportunity Difference): {gap:.4f}")
        else:
            eod_fpr = gap
            print(f"FPR Gap: {gap:.4f}")

eod = max(eod_tpr, eod_fpr)
print(f"Equalized Odds Difference (EOD): {eod:.4f}")