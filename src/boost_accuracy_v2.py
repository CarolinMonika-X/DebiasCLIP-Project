"""
boost_accuracy_v2.py - High-Performance Diagnostic Head for Frozen CLIP Embeddings.
Techniques:
1. Gated Multimodal Fusion (learns dynamic weighting between vision and clinical notes).
2. Focal Loss (focuses learning on ambiguous/borderline glaucoma margins).
3. Hybrid Ensemble (MLP + Regularized Ridge Classifier).
4. Validation-calibrated decision thresholding.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

print("Loading cached features...")
d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

X = {s: d[s]["img"].float().numpy() for s in SPLITS}
T = {s: d[s]["txt"].float().numpy() for s in SPLITS}
Y = {s: d[s]["labels"].long().numpy() for s in SPLITS}
R = {s: d[s]["races"].long().numpy() for s in SPLITS}

# Preserve natural magnitudes (no L2 destructive normalization)
RAW_DUAL = {s: np.concatenate([X[s], T[s]], axis=1).astype(np.float32) for s in SPLITS}

# Fit LEACE on train
def fit_leace(F, z):
    mu = F.mean(0)
    Fc = F - mu
    Z = np.eye(3)[z]
    Zc = Z - Z.mean(0)
    n = len(F)
    Sxx = Fc.T @ Fc / n
    Sxz = Fc.T @ Zc / n
    ev, V = np.linalg.eigh(Sxx)
    ev = np.clip(ev, 1e-5 * ev.max(), None)
    W = V @ np.diag(ev ** -0.5) @ V.T
    Winv = V @ np.diag(ev ** 0.5) @ V.T
    U, S, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    U = U[:, S > 1e-8 * S.max()]
    E = Winv @ (U @ U.T) @ W
    return lambda A: (A - (A - mu) @ E.T).astype(np.float32)

leace = fit_leace(RAW_DUAL["train"], R["train"])
LEACE_DUAL = {s: leace(RAW_DUAL[s]) for s in SPLITS}

# 1. Gated Multimodal Fusion Module
class GatedMultimodalHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # Gating mechanism learns how much text vs vision to trust per patient
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, dim * 2),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.SiLU(),  # Smoother gradients than ReLU
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        # x is [batch, 1024]
        g = self.gate(x)
        x_gated = x * g + x  # Residual gated connection
        return self.classifier(x_gated)

# 2. Focal Loss for Hard Medical Boundaries
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")
        
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()

def train_gated_model(feats, epochs=50, lr=5e-4):
    val_preds, test_preds = [], []
    criterion = FocalLoss(gamma=1.5)
    
    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        m = GatedMultimodalHead()
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
        
        Ft = torch.from_numpy(feats["train"]).float()
        yt = torch.from_numpy(Y["train"]).long()
        
        for ep in range(epochs):
            m.train()
            perm = torch.randperm(len(Ft))
            for i in range(0, len(Ft), 128):  # Smaller batch size for better generalization
                b = perm[i:i+128]
                loss = criterion(m(Ft[b]), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                
        m.eval()
        with torch.no_grad():
            p_val = torch.softmax(m(torch.from_numpy(feats["val"]).float()), 1)[:, 1].numpy()
            p_test = torch.softmax(m(torch.from_numpy(feats["test"]).float()), 1)[:, 1].numpy()
            val_preds.append(p_val)
            test_preds.append(p_test)
            
    return np.mean(val_preds, axis=0), np.mean(test_preds, axis=0)

print("Training Gated Residual Multimodal Head on Raw Features...")
val_p_gate_raw, test_p_gate_raw = train_gated_model(RAW_DUAL)

print("Training Gated Residual Multimodal Head on LEACE Features...")
val_p_gate_leace, test_p_gate_leace = train_gated_model(LEACE_DUAL)

print("Fitting Regularized Linear Calibrators (Ridge / Logistic)...")
clf_raw = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
clf_raw.fit(RAW_DUAL["train"], Y["train"])
ridge_val_raw = clf_raw.predict_proba(RAW_DUAL["val"])[:, 1]
ridge_test_raw = clf_raw.predict_proba(RAW_DUAL["test"])[:, 1]

clf_leace = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
clf_leace.fit(LEACE_DUAL["train"], Y["train"])
ridge_val_leace = clf_leace.predict_proba(LEACE_DUAL["val"])[:, 1]
ridge_test_leace = clf_leace.predict_proba(LEACE_DUAL["test"])[:, 1]

# 3. Hybrid Ensembles (Gated MLP + Regularized Linear)
val_hybrid_raw = 0.65 * val_p_gate_raw + 0.35 * ridge_val_raw
test_hybrid_raw = 0.65 * test_p_gate_raw + 0.35 * ridge_test_raw

val_hybrid_leace = 0.65 * val_p_gate_leace + 0.35 * ridge_val_leace
test_hybrid_leace = 0.65 * test_p_gate_leace + 0.35 * ridge_test_leace

# Optimize decision threshold on validation set
def optimize_threshold(val_p, y_val):
    best_acc = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.35, 0.65, 100):
        acc = accuracy_score(y_val, val_p > thr)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr

thr_raw = optimize_threshold(val_hybrid_raw, Y["val"])
thr_leace = optimize_threshold(val_hybrid_leace, Y["val"])

def get_metrics(test_p, thr):
    acc = accuracy_score(Y["test"], test_p > thr)
    auc = roc_auc_score(Y["test"], test_p)
    return acc, auc

acc_g_raw, auc_g_raw = get_metrics(test_p_gate_raw, 0.5)
acc_hy_raw, auc_hy_raw = get_metrics(test_hybrid_raw, thr_raw)

acc_g_leace, auc_g_leace = get_metrics(test_p_gate_leace, 0.5)
acc_hy_leace, auc_hy_leace = get_metrics(test_hybrid_leace, thr_leace)

print("\n" + "=" * 80)
print("ADVANCED ACCURACY OPTIMIZATION RESULTS (TEST SET)")
print("=" * 80)
print(f"{'Method / Architecture':48s} | {'Test Acc':10s} | {'Test AUC':10s}")
print("-" * 80)
print(f"{'Baseline MLP (Previous Reference)':48s} | 73.15%     | 0.8094")
print(f"{'Gated Multimodal Fusion Head (Raw, p=0.5)':48s} | {acc_g_raw*100:6.2f}%    | {auc_g_raw:.4f}")
print(f"{'Hybrid Ensemble + Val-Tuned Threshold (Raw)':48s} | {acc_hy_raw*100:6.2f}%    | {auc_hy_raw:.4f}")
print("-" * 80)
print(f"{'Gated Multimodal Fusion Head (LEACE, p=0.5)':48s} | {acc_g_leace*100:6.2f}%    | {auc_g_leace:.4f}")
print(f"{'Hybrid Ensemble + Val-Tuned Threshold (LEACE)':48s} | {acc_hy_leace*100:6.2f}%    | {auc_hy_leace:.4f}")
print("=" * 80)
print(f"Optimal Decision Cutoffs: Raw Hybrid = {thr_raw:.3f} | LEACE Hybrid = {thr_leace:.3f}")