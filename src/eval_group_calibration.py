"""
eval_group_calibration.py - Evaluates group-specific threshold calibration.
Selects optimal decision thresholds (theta_0, theta_1, theta_2) on VALIDATION ONLY
to minimize Equalized Odds Disparity (EOD), then evaluates on TEST split.
"""
import sys, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

CACHE = "cached_features.pt"
EPOCHS, BATCH, LR, SEEDS = 40, 256, 1e-3, range(5)
SPLITS = ("train", "val", "test")

print("Loading cached features...")
d = torch.load(CACHE, weights_only=True)
X = {s: d[s]["img"].float().numpy() for s in SPLITS}
T = {s: d[s]["txt"].float().numpy() for s in SPLITS}
Y = {s: d[s]["labels"].long().numpy() for s in SPLITS}
R = {s: d[s]["races"].long().numpy() for s in SPLITS}
cat = lambda a, b: np.concatenate([a, b], 1).astype(np.float32)
DUAL = {s: cat(X[s], T[s]) for s in SPLITS}

# LEACE fit on TRAIN only
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

leace = fit_leace(DUAL["train"], R["train"])
DL = {s: leace(DUAL[s]) for s in SPLITS}

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 2))
    def forward(self, x): return self.net(x)

# Train 5 seeds of F2 (Joint LEACE) and C (Dual Raw)
def get_ens_probs(feats):
    p_val, p_test = [], []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        m = Head()
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        Ft, yt = torch.from_numpy(feats["train"]).float(), torch.from_numpy(Y["train"]).long()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(Ft))
            for i in range(0, len(Ft), BATCH):
                b = perm[i:i+BATCH]
                loss = nn.functional.cross_entropy(m(Ft[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p_val.append(torch.softmax(m(torch.from_numpy(feats["val"]).float()), 1)[:, 1].numpy())
            p_test.append(torch.softmax(m(torch.from_numpy(feats["test"]).float()), 1)[:, 1].numpy())
    return np.mean(p_val, 0), np.mean(p_test, 0)

print("Training C (Raw Dual) and F2 (Joint LEACE)...")
c_val_p, c_test_p = get_ens_probs(DUAL)
f2_val_p, f2_test_p = get_ens_probs(DL)

def compute_rates(y, p, thrs, r):
    pred = np.zeros_like(p, dtype=bool)
    for g in (0, 1, 2):
        mask = (r == g)
        pred[mask] = p[mask] > thrs[g]
    G = (0, 1, 2)
    tpr = [pred[(r == g) & (y == 1)].mean() for g in G]
    fpr = [pred[(r == g) & (y == 0)].mean() for g in G]
    pr  = [pred[r == g].mean() for g in G]
    eod = max(max(tpr) - min(tpr), max(fpr) - min(fpr))
    dpd = max(pr) - min(pr)
    acc = (pred == y).mean()
    return acc, dpd, eod, tpr, fpr

# Optimize thresholds on VALIDATION ONLY
def find_best_thrs(val_p, y_val, r_val):
    best_eod = 999.0
    best_thrs = (0.5, 0.5, 0.5)
    # Search grid around 0.5 on validation
    grid = np.linspace(0.35, 0.65, 13)
    for t0 in grid:
        for t1 in grid:
            for t2 in grid:
                acc, dpd, eod, tpr, fpr = compute_rates(y_val, val_p, (t0, t1, t2), r_val)
                if eod < best_eod and acc > 0.70:
                    best_eod = eod
                    best_thrs = (t0, t1, t2)
    return best_thrs

print("\nTuning group-specific thresholds on VALIDATION set...")
c_thrs = find_best_thrs(c_val_p, Y["val"], R["val"])
f2_thrs = find_best_thrs(f2_val_p, Y["val"], R["val"])

print(f"Optimal Thresholds for C  (Asian, Black, White): {c_thrs[0]:.2f}, {c_thrs[1]:.2f}, {c_thrs[2]:.2f}")
print(f"Optimal Thresholds for F2 (Asian, Black, White): {f2_thrs[0]:.2f}, {f2_thrs[1]:.2f}, {f2_thrs[2]:.2f}")

# Evaluate on TEST split
print("\n" + "=" * 75)
print(f"{'Method & Operating Policy':42s} | {'Test Acc':8s} | {'Test DPD':8s} | {'Test EOD':8s}")
print("-" * 75)

# 1. Global Threshold 0.5
c_acc_def, c_dpd_def, c_eod_def, _, _ = compute_rates(Y["test"], c_test_p, (0.5, 0.5, 0.5), R["test"])
f2_acc_def, f2_dpd_def, f2_eod_def, _, _ = compute_rates(Y["test"], f2_test_p, (0.5, 0.5, 0.5), R["test"])

# 2. Calibrated Thresholds
c_acc_cal, c_dpd_cal, c_eod_cal, _, _ = compute_rates(Y["test"], c_test_p, c_thrs, R["test"])
f2_acc_cal, f2_dpd_cal, f2_eod_cal, _, _ = compute_rates(Y["test"], f2_test_p, f2_thrs, R["test"])

print(f"{'C (Raw Dual, Default Global 0.5)':42s} | {c_acc_def:.4f}   | {c_dpd_def:.4f}   | {c_eod_def:.4f}")
print(f"{'C (Raw Dual, Group-Calibrated on Val)':42s} | {c_acc_cal:.4f}   | {c_dpd_cal:.4f}   | {c_eod_cal:.4f}")
print(f"{'F2 (Joint LEACE, Default Global 0.5)':42s} | {f2_acc_def:.4f}   | {f2_dpd_def:.4f}   | {f2_eod_def:.4f}")
print(f"{'F2 (Joint LEACE + Group-Calibrated)':42s} | {f2_acc_cal:.4f}   | {f2_dpd_cal:.4f}   | {f2_eod_cal:.4f}")
print("=" * 75)

# Paired Bootstrap for F2 Calibrated vs C Default
print("\n=== Paired Bootstrap: (F2 + Calibrated) vs (C Default) on TEST (1000 resamples) ===")
rng = np.random.default_rng(0)
n = len(Y["test"])
diffs = {"acc": [], "dpd": [], "eod": []}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, rb = Y["test"][idx], R["test"][idx]
    acc_base, dpd_base, eod_base, _, _ = compute_rates(yb, c_test_p[idx], (0.5, 0.5, 0.5), rb)
    acc_cal, dpd_cal, eod_cal, _, _ = compute_rates(yb, f2_test_p[idx], f2_thrs, rb)
    diffs["acc"].append(acc_cal - acc_base)
    diffs["dpd"].append(dpd_cal - dpd_base)
    diffs["eod"].append(eod_cal - eod_base)

for k in ["acc", "dpd", "eod"]:
    vals = diffs[k]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    sig = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
    print(f"  Δ{k:4s} | mean: {np.mean(vals):+.4f} | 95% CI: [{lo:+.4f}, {hi:+.4f}] ({sig})")