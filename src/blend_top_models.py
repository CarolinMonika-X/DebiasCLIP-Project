"""
src/blend_top_models.py
Blend predictions from Cross-Attention Fusion and SVM-RBF
"""
import torch, numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC

d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

# Prepare features
X_joint = {s: torch.cat([d[s]["img"].float(), d[s]["txt"].float()], dim=-1) for s in SPLITS}
Y = {s: d[s]["labels"].numpy() for s in SPLITS}
R = {s: d[s]["races"].numpy() for s in SPLITS}

# LEACE projector
r_tr_oh = torch.nn.functional.one_hot(torch.tensor(R["train"]), num_classes=3).float()
x_mean = X_joint["train"].mean(dim=0, keepdim=True)
x_cent = X_joint["train"] - x_mean
cov_xz = (x_cent.T @ r_tr_oh) / (len(x_cent) - 1)
u, _, _ = torch.linalg.svd(cov_xz, full_matrices=False)
P_leace = torch.eye(1024) - (u @ u.T)

X_leace_tr = ((X_joint["train"] - x_mean) @ P_leace).numpy()
X_leace_val = ((X_joint["val"] - x_mean) @ P_leace).numpy()
X_leace_te = ((X_joint["test"] - x_mean) @ P_leace).numpy()

# 1. Fit SVM-RBF
print("Fitting SVM-RBF on LEACE features...")
svm = SVC(C=1.0, kernel="rbf", probability=True, random_state=42)
svm.fit(X_leace_tr, Y["train"])
p_svm_val = svm.predict_proba(X_leace_val)[:, 1]
p_svm_test = svm.predict_proba(X_leace_te)[:, 1]

# 2. Re-compute quick MLP probability
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1024, 256), torch.nn.LayerNorm(256), torch.nn.GELU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(256, 2)
        )
    def forward(self, x): return self.net(x)

m = MLP()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
x_tr_t = torch.tensor(X_leace_tr, dtype=torch.float32)
y_tr_t = torch.tensor(Y["train"], dtype=torch.long)
for ep in range(35):
    opt.zero_grad()
    l = torch.nn.functional.cross_entropy(m(x_tr_t), y_tr_t)
    l.backward()
    opt.step()
m.eval()
with torch.no_grad():
    p_mlp_val = torch.nn.functional.softmax(m(torch.tensor(X_leace_val)), dim=1)[:, 1].numpy()
    p_mlp_test = torch.nn.functional.softmax(m(torch.tensor(X_leace_te)), dim=1)[:, 1].numpy()

# Blend: 60% SVM + 40% MLP
p_blend_val = 0.60 * p_svm_val + 0.40 * p_mlp_val
p_blend_test = 0.60 * p_svm_test + 0.40 * p_mlp_test

# Find best threshold on validation
best_th, best_acc = 0.50, 0
for th in np.linspace(0.40, 0.60, 201):
    acc = accuracy_score(Y["val"], (p_blend_val >= th).astype(int))
    if acc > best_acc:
        best_acc, best_th = acc, th

y_te = Y["test"]
test_preds = (p_blend_test >= best_th).astype(int)
auc = roc_auc_score(y_te, p_blend_test)
acc = accuracy_score(y_te, test_preds)
tn, fp, fn, tp = confusion_matrix(y_te, test_preds).ravel()

print("\n" + "=" * 80)
print("SUPER-ENSEMBLE (SVM-RBF + LEACE-MLP) ON TEST SET (N = 2,000)")
print("=" * 80)
print(f"Optimal Validation Cutoff:  {best_th:.3f}")
print(f"Test AUC:                  {auc:.4f}")
print(f"Test Accuracy:             {acc*100:5.2f}%")
print(f"Sensitivity (Recall):      {tp / (tp + fn) * 100:5.1f}%")
print(f"Specificity:               {tn / (tn + fp) * 100:5.1f}%")
print("=" * 80)