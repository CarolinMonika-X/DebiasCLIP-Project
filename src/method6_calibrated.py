"""
src/method6_calibrated.py
Method 6: L2-Normalized Deep Ensemble with Temperature Calibration
- Per-modality L2 normalization (prevents image embeddings from overwhelming text notes)
- 5-seed Deep Ensemble combining Cross-Entropy and Focal Loss
- Validation Temperature Scaling (Platt scaling via L-BFGS)
- Validation-tuned optimal threshold selection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

print("=" * 80)
print("METHOD 6: L2-NORMALIZED DEEP ENSEMBLE + TEMPERATURE CALIBRATION")
print("=" * 80)

# 1. Load Data
d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

def prep_features(s):
    img = F.normalize(d[s]["img"].float(), p=2, dim=-1)
    txt = F.normalize(d[s]["txt"].float(), p=2, dim=-1)
    return torch.cat([img, txt], dim=-1)

X = {s: prep_features(s) for s in SPLITS}
Y = {s: d[s]["labels"].long() for s in SPLITS}

print(f"Features prepared with per-modality L2 norm. Shape: {X['train'].shape}")

# 2. Residual MLP Head Architecture
class ResMLP(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h = self.fc1(x)
        return self.head(h + self.block(h))

val_logits_list, test_logits_list = [], []

print("\nTraining 5-seed ensemble (Cross-Entropy + Focal Loss)...")
for seed in range(5):
    torch.manual_seed(seed * 42)
    m = ResMLP(in_dim=1024, hidden_dim=256)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    
    # Alternating loss functions across seeds
    if seed % 2 == 0:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = lambda logits, targets: (
            ((1.0 - torch.exp(-F.cross_entropy(logits, targets, reduction='none'))) ** 2.0)
            * F.cross_entropy(logits, targets, reduction='none')
        ).mean()

    n_tr = len(X["train"])
    for epoch in range(35):
        m.train()
        perm = torch.randperm(n_tr)
        for j in range(0, n_tr, 128):
            idx = perm[j:j+128]
            opt.zero_grad()
            loss = loss_fn(m(X["train"][idx]), Y["train"][idx])
            loss.backward()
            opt.step()

    m.eval()
    with torch.no_grad():
        val_logits_list.append(m(X["val"]))
        test_logits_list.append(m(X["test"]))

val_logits = torch.stack(val_logits_list).mean(0)
test_logits = torch.stack(test_logits_list).mean(0)

# 3. Validation Temperature Scaling (L-BFGS)
T = torch.nn.Parameter(torch.ones(1) * 1.2)
opt_t = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

def eval_t():
    opt_t.zero_grad()
    loss = F.cross_entropy(val_logits / T, Y["val"])
    loss.backward()
    return loss

opt_t.step(eval_t)
with torch.no_grad():
    calibrated_val_p = F.softmax(val_logits / T, dim=1)[:, 1].numpy()
    calibrated_test_p = F.softmax(test_logits / T, dim=1)[:, 1].numpy()

# 4. Optimal Threshold from Validation (Youden J)
best_th, best_j = 0.50, -1
for th in np.linspace(0.35, 0.65, 301):
    tn, fp, fn, tp = confusion_matrix(Y["val"].numpy(), calibrated_val_p >= th).ravel()
    j = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
    if j > best_j:
        best_j, best_th = j, th

y_te = Y["test"].numpy()
auc = roc_auc_score(y_te, calibrated_test_p)
acc_50 = accuracy_score(y_te, calibrated_test_p >= 0.50)
acc_opt = accuracy_score(y_te, calibrated_test_p >= best_th)
tn_50, fp_50, fn_50, tp_50 = confusion_matrix(y_te, calibrated_test_p >= 0.50).ravel()
tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_te, calibrated_test_p >= best_th).ravel()

print("\n" + "=" * 80)
print("RESULTS ON OFFICIAL TEST SET (N = 2,000)")
print("=" * 80)
print(f"Learned Temperature (T):        {T.item():.4f}")
print(f"Optimal Validation Cutoff:     {best_th:.3f}")
print(f"Test Set AUC:                  {auc:.4f}")
print("-" * 80)
print(f"At Default Cutoff (p = 0.50):")
print(f"  Accuracy:                    {acc_50*100:5.2f}%")
print(f"  Sensitivity / Specificity:   {tp_50/(tp_50+fn_50)*100:5.1f}% / {tn_50/(tn_50+fp_50)*100:5.1f}%")
print("-" * 80)
print(f"At Validation-Optimal Cutoff ({best_th:.3f}):")
print(f"  Accuracy:                    {acc_opt*100:5.2f}%")
print(f"  Sensitivity / Specificity:   {tp_opt/(tp_opt+fn_opt)*100:5.1f}% / {tn_opt/(tn_opt+fp_opt)*100:5.1f}%")
print("=" * 80)