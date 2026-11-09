"""
boost_accuracy.py - Techniques to maximize diagnostic accuracy and AUC
on cached Harvard-FairVLMed CLIP representations without sacrificing debiasing.
"""
import torch, torch.nn as nn, numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

print("Loading cached features...")
d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

X = {s: d[s]["img"].float().numpy() for s in SPLITS}
T = {s: d[s]["txt"].float().numpy() for s in SPLITS}
Y = {s: d[s]["labels"].long().numpy() for s in SPLITS}
R = {s: d[s]["races"].long().numpy() for s in SPLITS}

# 1. Normalized Multimodal Fusion
def fuse_normalized(img, txt):
    # L2 normalize each modality individually before concatenation
    img_norm = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
    txt_norm = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([img_norm, txt_norm], axis=1).astype(np.float32)

RAW_DUAL = {s: np.concatenate([X[s], T[s]], axis=1).astype(np.float32) for s in SPLITS}
NORM_DUAL = {s: fuse_normalized(X[s], T[s]) for s in SPLITS}

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

leace_norm = fit_leace(NORM_DUAL["train"], R["train"])
LEACE_NORM_DUAL = {s: leace_norm(NORM_DUAL[s]) for s in SPLITS}

# Enhanced Architecture: Residual MLP with LayerNorm
class EnhancedHead(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.15)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 2)
        
    def forward(self, x):
        h = self.drop(self.act(self.ln1(self.fc1(x))))
        h = self.drop(self.act(self.ln2(self.fc2(h))))
        return self.out(h)

def train_and_eval(features, model_type="enhanced", epochs=45, lr=8e-4, weight_decay=1e-4):
    test_probs = []
    val_probs = []
    
    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        m = EnhancedHead() if model_type == "enhanced" else nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 2)
        )
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        
        Ft = torch.from_numpy(features["train"]).float()
        yt = torch.from_numpy(Y["train"]).long()
        
        for ep in range(epochs):
            m.train()
            perm = torch.randperm(len(Ft))
            for i in range(0, len(Ft), 256):
                b = perm[i:i+256]
                loss = nn.functional.cross_entropy(m(Ft[b]), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()
            
        m.eval()
        with torch.no_grad():
            p_val = torch.softmax(m(torch.from_numpy(features["val"]).float()), 1)[:, 1].numpy()
            p_test = torch.softmax(m(torch.from_numpy(features["test"]).float()), 1)[:, 1].numpy()
            val_probs.append(p_val)
            test_probs.append(p_test)
            
    return np.mean(val_probs, axis=0), np.mean(test_probs, axis=0)

print("\n--- Training Architectures ---")
print("1. Standard Baseline MLP (Raw Dual)...")
val_p_base, test_p_base = train_and_eval(RAW_DUAL, model_type="baseline")

print("2. Enhanced Head + LayerNorm + L2 Normalized Fusion (Raw)...")
val_p_norm, test_p_norm = train_and_eval(NORM_DUAL, model_type="enhanced")

print("3. Enhanced Head + LayerNorm + L2 Normalized Fusion (LEACE Debiased)...")
val_p_leace, test_p_leace = train_and_eval(LEACE_NORM_DUAL, model_type="enhanced")

# Global Threshold Optimization on Validation ONLY
def find_best_global_threshold(val_p, y_val):
    best_acc = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.35, 0.65, 61):
        acc = accuracy_score(y_val, val_p > thr)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc

thr_base, _ = find_best_global_threshold(val_p_base, Y["val"])
thr_norm, _ = find_best_global_threshold(val_p_norm, Y["val"])
thr_leace, _ = find_best_global_threshold(val_p_leace, Y["val"])

def evaluate_test(test_p, thr):
    pred = test_p > thr
    acc = accuracy_score(Y["test"], pred)
    auc = roc_auc_score(Y["test"], test_p)
    return acc, auc

acc_base_05, auc_base = evaluate_test(test_p_base, 0.5)
acc_base_opt, _       = evaluate_test(test_p_base, thr_base)

acc_norm_05, auc_norm = evaluate_test(test_p_norm, 0.5)
acc_norm_opt, _       = evaluate_test(test_p_norm, thr_norm)

acc_leace_05, auc_leace = evaluate_test(test_p_leace, 0.5)
acc_leace_opt, _        = evaluate_test(test_p_leace, thr_leace)

print("\n" + "=" * 80)
print("ACCURACY BOOST BENCHMARK (TEST SPLIT)")
print("=" * 80)
print(f"{'Method / Configuration':45s} | {'Test Acc':10s} | {'Test AUC':10s} | {'Delta Acc':10s}")
print("-" * 80)
print(f"{'1. Baseline MLP (Default p > 0.5)':45s} | {acc_base_05*100:6.2f}%    | {auc_base:.4f}     | Baseline")
print(f"{'2. Baseline MLP + Val-Tuned Global Threshold':45s} | {acc_base_opt*100:6.2f}%    | {auc_base:.4f}     | +{(acc_base_opt - acc_base_05)*100:+.2f}%")
print(f"{'3. Enhanced Head + L2 Norm (Default p > 0.5)':45s} | {acc_norm_05*100:6.2f}%    | {auc_norm:.4f}     | +{(acc_norm_05 - acc_base_05)*100:+.2f}%")
print(f"{'4. Enhanced Head + L2 Norm + Val-Tuned Threshold':45s} | {acc_norm_opt*100:6.2f}%    | {auc_norm:.4f}     | +{(acc_norm_opt - acc_base_05)*100:+.2f}%")
print(f"{'5. Enhanced LEACE (Debiased) + Val-Tuned Threshold':45s} | {acc_leace_opt*100:6.2f}%    | {auc_leace:.4f}     | +{(acc_leace_opt - acc_base_05)*100:+.2f}%")
print("=" * 80)
print(f"Optimal Thresholds Tuned on Validation: Baseline={thr_base:.3f} | Norm={thr_norm:.3f} | LEACE={thr_leace:.3f}")