"""
erasure_primary.py - Rigorous statistical evaluation of LEACE concept erasure.
Evaluates both standard MLP heads and purely Linear heads across 5 seeds.
Calculates 95% and Bonferroni-corrected 99.6% bootstrap confidence intervals.
Outputs complete results to erasure_primary_output.txt.
"""
import io, sys, json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# -------------------------------------------------------------
# 1. Output Tee Setup
# -------------------------------------------------------------
class Tee(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

sys.stdout = Tee("erasure_primary_output.txt")

# -------------------------------------------------------------
# 2. Data Loading & Hyperparameters (Unchanged from erasure_experiments.py)
# -------------------------------------------------------------
CACHE = "cached_features.pt"
K_IMG, K_TXT, K_Y, K_RACE = "img", "txt", "labels", "races"
EPOCHS, BATCH, LR, SEEDS, THR, N_BOOT = 40, 256, 1e-3, range(5), 0.5, 1000
N_CLS = 3
SPLITS = ("train", "val", "test")

print("Loading cached features...")
d = torch.load(CACHE, weights_only=True)
X = {s: d[s][K_IMG].float().numpy() for s in SPLITS}
T = {s: d[s][K_TXT].float().numpy() for s in SPLITS}
Y = {s: d[s][K_Y].long().numpy() for s in SPLITS}
R = {s: d[s][K_RACE].long().numpy() for s in SPLITS}
cat = lambda a, b: np.concatenate([a, b], 1).astype(np.float32)

# -------------------------------------------------------------
# 3. LEACE Implementation (Unchanged, fit on TRAIN only)
# -------------------------------------------------------------
def fit_leace(F, z):
    mu = F.mean(0)
    Fc = F - mu
    Z = np.eye(N_CLS)[z]
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

# -------------------------------------------------------------
# 4. Architectures: Standard MLP Head + Purely Linear Head
# -------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        return self.net(x)

class LinearHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, 2)
    def forward(self, x):
        return self.net(x)

def train_head(F, y, seed, head_cls=Head):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = head_cls(F.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    Ft, yt = torch.from_numpy(F).float(), torch.from_numpy(y).long()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Ft))
        for i in range(0, len(Ft), BATCH):
            b = perm[i:i + BATCH]
            loss = nn.functional.cross_entropy(m(Ft[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()

def predict(m, F):
    with torch.no_grad():
        logits = m(torch.from_numpy(F).float())
        return torch.softmax(logits, 1)[:, 1].numpy()

# -------------------------------------------------------------
# 5. Metrics with TPR Gap and FPR Gap Added
# -------------------------------------------------------------
def metrics(y, r, p):
    pred = p > THR
    G = np.unique(r)
    pr  = [pred[r == g].mean() for g in G]
    tpr = [pred[(r == g) & (y == 1)].mean() for g in G]
    fpr = [pred[(r == g) & (y == 0)].mean() for g in G]
    tpr_gap = float(max(tpr) - min(tpr))
    fpr_gap = float(max(fpr) - min(fpr))
    return dict(
        acc=float((pred == y).mean()),
        auc=float(roc_auc_score(y, p)),
        dpd=float(max(pr) - min(pr)),
        eod=float(max(tpr_gap, fpr_gap)),
        tpr_gap=tpr_gap,
        fpr_gap=fpr_gap,
        w_auc=float(min(roc_auc_score(y[r == g], p[r == g]) for g in G))
    )

def run(name, F, head_cls=Head):
    pt, ms = [], []
    for s in SEEDS:
        m = train_head(F["train"], Y["train"], s, head_cls=head_cls)
        pt.append(predict(m, F["test"]))
        ms.append(metrics(Y["test"], R["test"], pt[-1]))
    agg = {k: (float(np.mean([x[k] for x in ms])), float(np.std([x[k] for x in ms]))) for k in ms[0]}
    print(f"{name:20s} | " + " | ".join(f"{k} {v[0]:.4f}+/-{v[1]:.4f}" for k, v in agg.items()))
    return agg, np.mean(pt, 0)

# -------------------------------------------------------------
# 6. Fit LEACE Representations on TRAIN
# -------------------------------------------------------------
print("Fitting LEACE erasers on TRAIN...")
leace_img = fit_leace(X["train"], R["train"])
XL = {s: leace_img(X[s]) for s in SPLITS}

DUAL = {s: cat(X[s], T[s]) for s in SPLITS}
leace_dual = fit_leace(DUAL["train"], R["train"])
DL = {s: leace_dual(DUAL[s]) for s in SPLITS}

feature_sets = {
    "image raw": X,
    "image LEACE": XL,
    "dual raw": DUAL,
    "dual joint LEACE": DL
}

# -------------------------------------------------------------
# 7. Train 5-Seed Heads for 8 Configurations
# -------------------------------------------------------------
print("\n=== Training 8 Configurations (mean +/- std over 5 seeds) ===")
models_config = [
    ("A",          feature_sets["image raw"],        Head),
    ("A_LEACE",    feature_sets["image LEACE"],      Head),
    ("C",          feature_sets["dual raw"],         Head),
    ("F2",         feature_sets["dual joint LEACE"], Head),
    ("lin_A",       feature_sets["image raw"],        LinearHead),
    ("lin_A_LEACE", feature_sets["image LEACE"],      LinearHead),
    ("lin_C",       feature_sets["dual raw"],         LinearHead),
    ("lin_F2",      feature_sets["dual joint LEACE"], LinearHead),
]

results = {}
ens_preds = {}

for name, feats, h_cls in models_config:
    results[name], ens_preds[name] = run(name, feats, head_cls=h_cls)

# -------------------------------------------------------------
# 8. Pre-specified Paired Bootstrap with 95% and 99.6% CIs
# -------------------------------------------------------------
print("\n=== Pre-specified Paired Bootstrap Comparisons (1000 resamples, seed 0, seed-ensembled) ===")
comparisons = [
    ("(i)   F2 minus C (Dual MLP)",                ens_preds["C"],       ens_preds["F2"]),
    ("(ii)  A_LEACE minus A (Image MLP)",          ens_preds["A"],       ens_preds["A_LEACE"]),
    ("(iii) lin_F2 minus lin_C (Dual Linear)",     ens_preds["lin_C"],   ens_preds["lin_F2"]),
    ("(iv)  lin_A_LEACE minus lin_A (Image Linear)", ens_preds["lin_A"], ens_preds["lin_A_LEACE"]),
]

METRICS_ORDER = ["auc", "dpd", "eod", "w_auc", "tpr_gap", "fpr_gap"]

def paired_bootstrap(pa, pb, y, r, seed=0):
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in METRICS_ORDER}
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        y_b, r_b = y[idx], r[idx]
        try:
            ma = metrics(y_b, r_b, pa[idx])
            mb = metrics(y_b, r_b, pb[idx])
        except ValueError:
            continue
        for k in METRICS_ORDER:
            diffs[k].append(mb[k] - ma[k])
    
    res = {}
    for k in METRICS_ORDER:
        v = diffs[k]
        m = float(np.mean(v))
        lo95, hi95 = np.percentile(v, [2.5, 97.5])
        lo996, hi996 = np.percentile(v, [0.2, 99.8])
        res[k] = {
            "mean": m,
            "ci95": (float(lo95), float(hi95)),
            "sig95": bool(lo95 > 0 or hi95 < 0),
            "ci996": (float(lo996), float(hi996)),
            "sig996": bool(lo996 > 0 or hi996 < 0)
        }
    return res

for label, base_pred, var_pred in comparisons:
    print(f"\n--- {label} ---")
    boot_res = paired_bootstrap(base_pred, var_pred, Y["test"], R["test"], seed=0)
    for k in METRICS_ORDER:
        b = boot_res[k]
        sig95_str = "EXCLUDES 0" if b["sig95"] else "includes 0"
        sig996_str = "EXCLUDES 0" if b["sig996"] else "includes 0"
        print(f"  {k:8s} | mean: {b['mean']:+.4f} | 95% CI: [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] ({sig95_str}) | 99.6% CI: [{b['ci996'][0]:+.4f}, {b['ci996'][1]:+.4f}] ({sig996_str})")

# -------------------------------------------------------------
# 9. Per-Group Table on TEST (Seed-Ensembled, Threshold 0.5)
# -------------------------------------------------------------
print("\n=== Per-Group Table on TEST (seed-ensembled, threshold 0.5) ===")
G_unique = np.unique(R["test"])
target_models = [("A", ens_preds["A"]), ("A_LEACE", ens_preds["A_LEACE"]), 
                 ("C", ens_preds["C"]), ("F2", ens_preds["F2"])]

print(f"{'Model':8s} | {'Group':5s} | {'N':5s} | {'Prevalence':10s} | {'Flagged Rate':12s} | {'TPR':8s} | {'FPR':8s} | {'AUC':8s}")
print("-" * 85)

for m_name, pred_p in target_models:
    pred_bin = pred_p > THR
    for g in G_unique:
        mask = (R["test"] == g)
        n_g = int(mask.sum())
        y_g = Y["test"][mask]
        p_g = pred_p[mask]
        bin_g = pred_bin[mask]
        
        prev = float(y_g.mean())
        flagged = float(bin_g.mean())
        tpr = float(bin_g[y_g == 1].mean()) if (y_g == 1).sum() > 0 else 0.0
        fpr = float(bin_g[y_g == 0].mean()) if (y_g == 0).sum() > 0 else 0.0
        auc = float(roc_auc_score(y_g, p_g)) if len(np.unique(y_g)) > 1 else 0.0
        
        print(f"{m_name:8s} | {g:5d} | {n_g:5d} | {prev:10.4f} | {flagged:12.4f} | {tpr:8.4f} | {fpr:8.4f} | {auc:8.4f}")

print("\nSaved all output to erasure_primary_output.txt. Execution complete.")