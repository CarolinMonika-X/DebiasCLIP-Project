"""
src/accuracy_boost.py
Accuracy improvement techniques on cached_features.pt:
1. Focal Loss (gamma=2.0)
2. Residual MLP Head with LayerNorm
3. Heterogeneous Ensembling (ResMLP + Ridge + ExtraTrees)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

print("=" * 75)
print("ACCURACY BOOST EXPERIMENTS ON CACHED MULTIMODAL FEATURES")
print("=" * 75)

# 1. Load Data
d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

X_img = {s: d[s]["img"].float() for s in SPLITS}
X_txt = {s: d[s]["txt"].float() for s in SPLITS}
Y = {s: d[s]["labels"].long() for s in SPLITS}
R = {s: d[s]["races"].long() for s in SPLITS}

X_joint = {s: torch.cat([X_img[s], X_txt[s]], dim=-1) for s in SPLITS}

y_val = Y["val"].numpy()
y_test = Y["test"].numpy()

# 2. Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# 3. Residual MLP Head Architecture
class ResidualMLP(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h = self.in_proj(x)
        h = h + self.block1(h)  # Residual connection
        return self.out_head(h)

# 4. Train with Focal Loss and Cosine Annealing
def train_res_mlp(use_focal=True):
    val_probs, test_probs = [], []
    for seed in range(5):
        torch.manual_seed(seed * 100)
        m = ResidualMLP(in_dim=1024, hidden_dim=256)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)
        loss_fn = FocalLoss(gamma=2.0) if use_focal else nn.CrossEntropyLoss()

        n_tr = len(X_joint["train"])
        for epoch in range(35):
            m.train()
            perm = torch.randperm(n_tr)
            for j in range(0, n_tr, 128):
                idx = perm[j:j+128]
                opt.zero_grad()
                l = loss_fn(m(X_joint["train"][idx]), Y["train"][idx])
                l.backward()
                opt.step()
            sched.step()

        m.eval()
        with torch.no_grad():
            val_probs.append(F.softmax(m(X_joint["val"]), dim=1)[:, 1].numpy())
            test_probs.append(F.softmax(m(X_joint["test"]), dim=1)[:, 1].numpy())

    return np.mean(val_probs, axis=0), np.mean(test_probs, axis=0)

print("\n[1/3] Training Baseline MLP with Cross-Entropy Loss...")
val_p_ce, test_p_ce = train_res_mlp(use_focal=False)

print("[2/3] Training Residual MLP with Focal Loss (gamma=2.0)...")
val_p_focal, test_p_focal = train_res_mlp(use_focal=True)

# 5. Tree-based Model & Linear Model for Ensembling
print("[3/3] Fitting Diverse Auxiliary Classifiers for Ensembling...")
# Model B: Tuned Logistic Ridge
ridge = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
ridge.fit(X_joint["train"].numpy(), Y["train"].numpy())
test_p_ridge = ridge.predict_proba(X_joint["test"].numpy())[:, 1]
val_p_ridge = ridge.predict_proba(X_joint["val"].numpy())[:, 1]

# Model C: ExtraTrees Classifier
et = ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
et.fit(X_joint["train"].numpy(), Y["train"].numpy())
test_p_et = et.predict_proba(X_joint["test"].numpy())[:, 1]
val_p_et = et.predict_proba(X_joint["val"].numpy())[:, 1]

# Heterogeneous Ensemble (Weighted Average)
val_p_ens = 0.50 * val_p_focal + 0.30 * val_p_ridge + 0.20 * val_p_et
test_p_ens = 0.50 * test_p_focal + 0.30 * test_p_ridge + 0.20 * test_p_et

# Evaluate all
def evaluate_preds(y_true, y_prob, name):
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_prob >= 0.50)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob >= 0.50).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print(f"{name:<35} | AUC: {auc:.4f} | Acc: {acc*100:5.2f}% | Sens: {sens*100:5.1f}% | Spec: {spec*100:5.1f}%")

print("\n" + "=" * 80)
print("TEST SET ACCURACY COMPARISON (Official N=2,000 Test Set)")
print("=" * 80)
evaluate_preds(y_test, test_p_ce, "1. Standard MLP (Cross-Entropy)")
evaluate_preds(y_test, test_p_focal, "2. Residual MLP (Focal Loss)")
evaluate_preds(y_test, test_p_ridge, "3. Regularized Logistic Ridge")
evaluate_preds(y_test, test_p_et, "4. ExtraTrees Ensemble")
evaluate_preds(y_test, test_p_ens, "5. Heterogeneous Tri-Ensemble")
print("=" * 80)