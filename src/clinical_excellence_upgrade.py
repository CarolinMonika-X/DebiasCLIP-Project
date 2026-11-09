"""
clinical_excellence_upgrade.py
Maximizes accuracy via ensembling and validation Youden cutoff.
Generates Subgroup Calibration (ECE) and Clinical Decision Curves (Net Benefit).
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

print("=" * 80)
print("RUNNING CLINICAL EXCELLENCE & ACCURACY MAXIMIZATION UPGRADE")
print("=" * 80)

# Load data
d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

X_img = {s: d[s]["img"].float() for s in SPLITS}
X_txt = {s: d[s]["txt"].float() for s in SPLITS}
Y = {s: d[s]["labels"].long() for s in SPLITS}
R = {s: d[s]["races"].long() for s in SPLITS}

# Raw joint representation (1024-dim)
X_raw = {s: torch.cat([X_img[s], X_txt[s]], dim=-1) for s in SPLITS}

# -------------------------------------------------------------------------
# Step 1: Closed-form LEACE Projection
# -------------------------------------------------------------------------
print("\n[1/4] Computing Closed-Form LEACE Projection...")
r_train_oh = torch.nn.functional.one_hot(R["train"], num_classes=3).float()
x_mean = X_raw["train"].mean(dim=0, keepdim=True)
x_cent = X_raw["train"] - x_mean
cov_xz = (x_cent.T @ r_train_oh) / (len(x_cent) - 1)
u, s, vt = torch.linalg.svd(cov_xz, full_matrices=False)
P = torch.eye(1024) - (u @ u.T)

X_leace = {s: ((X_raw[s] - x_mean) @ P) for s in SPLITS}

# -------------------------------------------------------------------------
# Step 2: High-Accuracy Seed Ensemble with Cosine Annealing
# -------------------------------------------------------------------------
print("[2/4] Training High-Accuracy 5-Seed Ensemble on LEACE Representations...")

class RobustMLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(drop / 2),
            nn.Linear(hidden // 2, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_ensemble(features_dict):
    val_probs_list, test_probs_list = [], []
    for seed in range(5):
        torch.manual_seed(seed * 42)
        model = RobustMLP(drop=0.15 + (seed * 0.03))
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.04)

        n_train = len(features_dict["train"])
        for epoch in range(40):
            model.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, 128):
                idx = perm[i:i+128]
                logits = model(features_dict["train"][idx])
                loss = criterion(logits, Y["train"][idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            v_prob = torch.softmax(model(features_dict["val"]), dim=1)[:, 1].numpy()
            t_prob = torch.softmax(model(features_dict["test"]), dim=1)[:, 1].numpy()
        val_probs_list.append(v_prob)
        test_probs_list.append(t_prob)

    return np.mean(val_probs_list, axis=0), np.mean(test_probs_list, axis=0)

val_p_raw, test_p_raw = train_ensemble(X_raw)
val_p_leace, test_p_leace = train_ensemble(X_leace)

# Optimize threshold on validation split via Youden's Index
best_thr, best_j = 0.5, -1.0
y_val = Y["val"].numpy()
for thr in np.linspace(0.40, 0.60, 201):
    tn, fp, fn, tp = confusion_matrix(y_val, val_p_leace > thr).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    j = sens + spec - 1
    if j > best_j:
        best_j = j
        best_thr = thr

y_test = Y["test"].numpy()
acc_default = accuracy_score(y_test, test_p_leace > 0.50)
acc_tuned = accuracy_score(y_test, test_p_leace > best_thr)
auc_test = roc_auc_score(y_test, test_p_leace)

tn, fp, fn, tp = confusion_matrix(y_test, test_p_leace > best_thr).ravel()
sens_test = tp / (tp + fn)
spec_test = tn / (tn + fp)
bal_acc = (sens_test + spec_test) / 2.0

print("\n" + "=" * 60)
print(f"ENSEMBLE TEST RESULTS (Model F2 - LEACE)")
print("=" * 60)
print(f"Default Cutoff Accuracy (p > 0.50):    {acc_default * 100:6.2f}%")
print(f"Optimized Cutoff Accuracy (p > {best_thr:.3f}): {acc_tuned * 100:6.2f}% (Tuned on Val)")
print(f"Test AUC:                             {auc_test:.4f}")
print(f"Test Sensitivity (Recall):            {sens_test * 100:6.2f}%")
print(f"Test Specificity:                     {spec_test * 100:6.2f}%")
print(f"Balanced Accuracy:                    {bal_acc * 100:6.2f}%")
print("=" * 60)

# -------------------------------------------------------------------------
# Step 3: Subgroup Expected Calibration Error (ECE)
# -------------------------------------------------------------------------
print("\n[3/4] Computing Subgroup Expected Calibration Error (ECE)...")

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_idx = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(labels[bin_idx])
            bin_conf = np.mean(probs[bin_idx])
            ece += np.sum(bin_idx) / len(probs) * np.abs(bin_acc - bin_conf)
    return ece

race_labels = ["Asian", "Black", "White"]
r_test = R["test"].numpy()

print("\n--- Calibration Reliability Comparison ---")
for idx, race in enumerate(race_labels):
    mask = (r_test == idx)
    ece_raw = compute_ece(test_p_raw[mask], y_test[mask])
    ece_leace = compute_ece(test_p_leace[mask], y_test[mask])
    print(f"  {race:6s} Cohort: Raw ECE = {ece_raw:.4f} | LEACE ECE = {ece_leace:.4f}")

# Plot reliability diagram
fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300)
for idx, race in enumerate(race_labels):
    mask = (r_test == idx)
    p_sub, y_sub = test_p_leace[mask], y_test[mask]
    bins = np.linspace(0, 1, 11)
    bin_centers, bin_accs = [], []
    for b in range(10):
        b_mask = (p_sub >= bins[b]) & (p_sub < bins[b+1])
        if np.sum(b_mask) > 0:
            bin_centers.append((bins[b] + bins[b+1]) / 2)
            bin_accs.append(np.mean(y_sub[b_mask]))

    axes[idx].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    axes[idx].plot(bin_centers, bin_accs, "o-", color="#1f77b4", lw=2, label="LEACE Model")
    axes[idx].set_title(f"{race} Cohort (ECE = {compute_ece(p_sub, y_sub):.3f})")
    axes[idx].set_xlabel("Predicted Probability")
    axes[idx].set_ylabel("Empirical Frequency" if idx == 0 else "")
    axes[idx].set_xlim([0, 1])
    axes[idx].set_ylim([0, 1])
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(loc="upper left")

plt.tight_layout()
plt.savefig("figures/fig_calibration_reliability.png", dpi=300)
plt.close()
print("Saved reliability diagram: figures/fig_calibration_reliability.png")

# -------------------------------------------------------------------------
# Step 4: Clinical Decision Curve Analysis (Net Benefit)
# -------------------------------------------------------------------------
print("\n[4/4] Generating Clinical Decision Curve Analysis (Net Benefit)...")

thresholds = np.linspace(0.10, 0.45, 50)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

for idx, race in enumerate(race_labels):
    mask = (r_test == idx)
    y_sub = y_test[mask]
    p_raw_sub = test_p_raw[mask]
    p_leace_sub = test_p_leace[mask]
    N = len(y_sub)

    nb_raw, nb_leace, nb_all = [], [], []
    for pt in thresholds:
        weight = pt / (1.0 - pt)
        # Raw
        tp_r = np.sum((p_raw_sub >= pt) & (y_sub == 1))
        fp_r = np.sum((p_raw_sub >= pt) & (y_sub == 0))
        nb_raw.append((tp_r / N) - (fp_r / N) * weight)
        # LEACE
        tp_l = np.sum((p_leace_sub >= pt) & (y_sub == 1))
        fp_l = np.sum((p_leace_sub >= pt) & (y_sub == 0))
        nb_leace.append((tp_l / N) - (fp_l / N) * weight)
        # Treat all
        tp_a = np.sum(y_sub == 1)
        fp_a = np.sum(y_sub == 0)
        nb_all.append((tp_a / N) - (fp_a / N) * weight)

    axes[idx].plot(thresholds, nb_leace, label="LEACE (Ours)", color="#2ca02c", lw=2.5)
    axes[idx].plot(thresholds, nb_raw, label="Raw CLIP", color="#d62728", lw=1.8, linestyle="--")
    axes[idx].plot(thresholds, nb_all, label="Screen All", color="gray", lw=1, linestyle=":")
    axes[idx].axhline(0, color="black", lw=1)
    axes[idx].set_title(f"Clinical Net Benefit: {race} Patients")
    axes[idx].set_xlabel("Referral Threshold Probability (pt)")
    axes[idx].set_ylabel("Net Benefit" if idx == 0 else "")
    axes[idx].set_ylim([-0.05, 0.45])
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(loc="lower left")

plt.tight_layout()
plt.savefig("figures/fig_clinical_net_benefit.png", dpi=300)
plt.close()
print("Saved decision curves: figures/fig_clinical_net_benefit.png")

print("\n" + "=" * 80)
print("CLINICAL EXCELLENCE UPGRADE COMPLETE!")
print("=" * 80)