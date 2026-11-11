"""
src/triple_super_ensemble.py
Triple Super-Ensemble: SVM-RBF + Cross-Attention Transformer + Calibrated ResMLP
Optimizes blend weights and threshold strictly on the VALIDATION split.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC

print("=" * 80)
print("TRIPLE SUPER-ENSEMBLE OPTIMIZATION (VALIDATION TUNED)")
print("=" * 80)

d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

X_img = {s: F.normalize(d[s]["img"].float(), p=2, dim=-1) for s in SPLITS}
X_txt = {s: F.normalize(d[s]["txt"].float(), p=2, dim=-1) for s in SPLITS}
X_joint = {s: torch.cat([X_img[s], X_txt[s]], dim=-1) for s in SPLITS}

Y = {s: d[s]["labels"].long().numpy() for s in SPLITS}
R = {s: d[s]["races"].long().numpy() for s in SPLITS}

# 1. LEACE projection
r_tr_oh = np.eye(3)[R["train"]]
x_mean = X_joint["train"].numpy().mean(axis=0, keepdims=True)
x_cent = X_joint["train"].numpy() - x_mean
cov_xz = (x_cent.T @ r_tr_oh) / (len(x_cent) - 1)
u, _, _ = np.linalg.svd(cov_xz, full_matrices=False)
P_leace = np.eye(1024) - (u[:, :2] @ u[:, :2].T)

X_leace = {s: ((X_joint[s].numpy() - x_mean) @ P_leace).astype(np.float32) for s in SPLITS}

# Model 1: Non-linear SVM-RBF
print("Training Model 1: SVM-RBF...")
svm = SVC(C=1.2, kernel="rbf", probability=True, random_state=42)
svm.fit(X_leace["train"], Y["train"])
p_svm_val = svm.predict_proba(X_leace["val"])[:, 1]
p_svm_te = svm.predict_proba(X_leace["test"])[:, 1]

# Model 2: Residual MLP
print("Training Model 2: Deep Residual MLP (5 seeds)...")
class ResMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.block = nn.Sequential(
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        self.head = nn.Linear(256, 2)
    def forward(self, x):
        h = self.fc1(x)
        return self.head(h + self.block(h))

val_mlp_seeds, te_mlp_seeds = [], []
for seed in range(5):
    torch.manual_seed(seed * 100)
    m = ResMLP()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    xtr_t = torch.tensor(X_leace["train"])
    ytr_t = torch.tensor(Y["train"])
    for _ in range(35):
        opt.zero_grad()
        loss = F.cross_entropy(m(xtr_t), ytr_t)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        val_mlp_seeds.append(F.softmax(m(torch.tensor(X_leace["val"])), dim=1)[:, 1].numpy())
        te_mlp_seeds.append(F.softmax(m(torch.tensor(X_leace["test"])), dim=1)[:, 1].numpy())
p_mlp_val = np.mean(val_mlp_seeds, axis=0)
p_mlp_te = np.mean(te_mlp_seeds, axis=0)

# Model 3: Multimodal Cross-Attention Head
print("Training Model 3: Cross-Attention Fusion Network...")
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj_img = nn.Linear(dim, dim)
        self.proj_txt = nn.Linear(dim, dim)
        self.mha_i2t = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.mha_t2i = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    def forward(self, img, txt):
        qi = self.proj_img(img).unsqueeze(1)
        qt = self.proj_txt(txt).unsqueeze(1)
        ai, _ = self.mha_i2t(qi, qt, qt)
        at, _ = self.mha_t2i(qt, qi, qi)
        fused = torch.cat([(qi + ai).squeeze(1), (qt + at).squeeze(1)], dim=-1)
        return self.classifier(fused)

ca_m = CrossAttentionFusion()
ca_opt = torch.optim.AdamW(ca_m.parameters(), lr=8e-4, weight_decay=1e-3)
for _ in range(40):
    ca_opt.zero_grad()
    loss = F.cross_entropy(ca_m(X_img["train"], X_txt["train"]), torch.tensor(Y["train"]))
    loss.backward()
    ca_opt.step()
ca_m.eval()
with torch.no_grad():
    p_ca_val = F.softmax(ca_m(X_img["val"], X_txt["val"]), dim=1)[:, 1].numpy()
    p_ca_te = F.softmax(ca_m(X_img["test"], X_txt["test"]), dim=1)[:, 1].numpy()

# 4. Search optimal weights strictly on VALIDATION split
print("\nSearching optimal combination weights on validation...")
best_val_acc = 0.0
best_weights = None
best_th = 0.50

for w_svm in np.linspace(0.2, 0.7, 11):
    for w_mlp in np.linspace(0.1, 0.6, 11):
        w_ca = 1.0 - w_svm - w_mlp
        if w_ca < 0:
            continue
        p_val_comb = w_svm * p_svm_val + w_mlp * p_mlp_val + w_ca * p_ca_val
        for th in np.linspace(0.44, 0.56, 25):
            acc = accuracy_score(Y["val"], (p_val_comb >= th).astype(int))
            if acc > best_val_acc:
                best_val_acc = acc
                best_weights = (w_svm, w_mlp, w_ca)
                best_th = th

print(f"Optimal Validation Settings:")
print(f"  Weights: SVM={best_weights[0]:.2f}, MLP={best_weights[1]:.2f}, CrossAttn={best_weights[2]:.2f}")
print(f"  Validation Cutoff: {best_th:.3f}")
print(f"  Validation Accuracy: {best_val_acc * 100:.2f}%")

# 5. Evaluate ONCE on Test Set
p_test_comb = best_weights[0] * p_svm_te + best_weights[1] * p_mlp_te + best_weights[2] * p_ca_te
test_preds = (p_test_comb >= best_th).astype(int)

auc_te = roc_auc_score(Y["test"], p_test_comb)
acc_te = accuracy_score(Y["test"], test_preds)
tn, fp, fn, tp = confusion_matrix(Y["test"], test_preds).ravel()

print("\n" + "=" * 80)
print("FINAL TEST SET OUTCOMES (N = 2,000)")
print("=" * 80)
print(f"Test AUC:                   {auc_te:.4f}")
print(f"Test Accuracy:              {acc_te * 100:5.2f}%")
print(f"Sensitivity (Recall):       {tp / (tp + fn) * 100:5.1f}%")
print(f"Specificity:                {tn / (tn + fp) * 100:5.1f}%")
print("=" * 80)