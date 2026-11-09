"""
latent_erasure.py - Hidden-Layer Concept Erasure (Latent-LEACE).
Erases linear demographic information from the intermediate hidden activations h,
preventing non-linear race interactions from reaching the final diagnostic decision.
"""
import sys, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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

sys.stdout = Tee("latent_erasure_output.txt")

CACHE = "cached_features.pt"
EPOCHS, BATCH, LR, SEEDS, THR, N_BOOT = 40, 256, 1e-3, range(5), 0.5, 1000
N_CLS = 3
SPLITS = ("train", "val", "test")

print("Loading cached features...")
d = torch.load(CACHE, weights_only=True)
X = {s: d[s]["img"].float().numpy() for s in SPLITS}
T = {s: d[s]["txt"].float().numpy() for s in SPLITS}
Y = {s: d[s]["labels"].long().numpy() for s in SPLITS}
R = {s: d[s]["races"].long().numpy() for s in SPLITS}
cat = lambda a, b: np.concatenate([a, b], 1).astype(np.float32)
DUAL = {s: cat(X[s], T[s]) for s in SPLITS}

# LEACE closed-form eraser (fit on TRAIN only)
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

print("Fitting Input LEACE on TRAIN...")
leace_input = fit_leace(DUAL["train"], R["train"])
DL = {s: leace_input(DUAL[s]) for s in SPLITS}

# -------------------------------------------------------------
# Train Hidden Feature Extractor (Layer 1)
# -------------------------------------------------------------
print("\nExtracting latent representations from intermediate hidden layer (dim=512)...")
torch.manual_seed(0)
l1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1))
opt1 = torch.optim.Adam(l1.parameters(), lr=LR)
Ft, yt = torch.from_numpy(DL["train"]).float(), torch.from_numpy(Y["train"]).long()

# Pretrain Layer 1 with auxiliary projection
aux_head = nn.Linear(512, 2)
opt_full = torch.optim.Adam(list(l1.parameters()) + list(aux_head.parameters()), lr=LR)
for _ in range(EPOCHS):
    perm = torch.randperm(len(Ft))
    for i in range(0, len(Ft), BATCH):
        b = perm[i:i+BATCH]
        loss = nn.functional.cross_entropy(aux_head(l1(Ft[b])), yt[b])
        opt_full.zero_grad()
        loss.backward()
        opt_full.step()

l1.eval()
with torch.no_grad():
    H = {s: l1(torch.from_numpy(DL[s]).float()).numpy() for s in SPLITS}

# -------------------------------------------------------------
# Fit Latent-LEACE on the Hidden Activations (TRAIN only)
# -------------------------------------------------------------
print("Fitting Latent-LEACE eraser on hidden representations H['train']...")
leace_latent = fit_leace(H["train"], R["train"])
HL = {s: leace_latent(H[s]) for s in SPLITS}

# -------------------------------------------------------------
# Probe Non-linear and Linear Leakage on Hidden Activations
# -------------------------------------------------------------
print("\n=== Race Probing on Latent Space (TEST split) ===")
for name, feats in [("Raw Hidden H", H), ("Latent-LEACE Erased HL", HL)]:
    # Linear Probe
    clf_lin = LogisticRegression(max_iter=1000, C=0.1)
    clf_lin.fit(feats["train"], R["train"])
    p_lin = clf_lin.predict_proba(feats["test"])
    pred_lin = np.argmax(p_lin, 1)
    macro_auc_lin = roc_auc_score(R["test"], p_lin, multi_class="ovr")
    bal_acc_lin = balanced_accuracy_score(R["test"], pred_lin)

    # MLP Probe
    clf_mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=0)
    clf_mlp.fit(feats["train"], R["train"])
    p_mlp = clf_mlp.predict_proba(feats["test"])
    pred_mlp = np.argmax(p_mlp, 1)
    macro_auc_mlp = roc_auc_score(R["test"], p_mlp, multi_class="ovr")
    bal_acc_mlp = balanced_accuracy_score(R["test"], pred_mlp)

    print(f"[{name}]")
    print(f"  Linear Probe:  Macro-AUC = {macro_auc_lin:.4f} | Bal-Acc = {bal_acc_lin:.4f} (Chance: AUC=0.500, Acc=0.333)")
    print(f"  MLP Probe:     Macro-AUC = {macro_auc_mlp:.4f} | Bal-Acc = {bal_acc_mlp:.4f}")

# -------------------------------------------------------------
# Train Linear Readout Head on HL Across 5 Seeds
# -------------------------------------------------------------
print("\n=== Training Final Diagnostic Head on Latent-LEACE (5 seeds) ===")
pt_latent = []
ms_latent = []
for s in SEEDS:
    torch.manual_seed(s)
    np.random.seed(s)
    head = nn.Linear(512, 2)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    Ht_train, yt_train = torch.from_numpy(HL["train"]).float(), torch.from_numpy(Y["train"]).long()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Ht_train))
        for i in range(0, len(Ht_train), BATCH):
            b = perm[i:i+BATCH]
            loss = nn.functional.cross_entropy(head(Ht_train[b]), yt_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        probs = torch.softmax(head(torch.from_numpy(HL["test"]).float()), 1)[:, 1].numpy()
        pt_latent.append(probs)
        ms_latent.append(metrics(Y["test"], R["test"], probs))

agg = {k: (float(np.mean([x[k] for x in ms_latent])), float(np.std([x[k] for x in ms_latent]))) for k in ms_latent[0]}
print("Latent-LEACE Dual Model (F3) Test Performance:")
for k, v in agg.items():
    print(f"  {k:8s}: {v[0]:.4f} +/- {v[1]:.4f}")

# -------------------------------------------------------------
# Paired Bootstrap: Latent-LEACE (F3) vs Baseline Raw Dual (C)
# -------------------------------------------------------------
print("\n=== Paired Bootstrap vs Baseline C (1000 resamples, seed 0) ===")
# Train seed-ensembled baseline C predictions
class BaselineHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 2))
    def forward(self, x): return self.net(x)

pt_c = []
for s in SEEDS:
    torch.manual_seed(s)
    np.random.seed(s)
    m = BaselineHead()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    dt, yt = torch.from_numpy(DUAL["train"]).float(), torch.from_numpy(Y["train"]).long()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(dt))
        for i in range(0, len(dt), BATCH):
            b = perm[i:i+BATCH]
            loss = nn.functional.cross_entropy(m(dt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pt_c.append(torch.softmax(m(torch.from_numpy(DUAL["test"]).float()), 1)[:, 1].numpy())

ens_f3 = np.mean(pt_latent, 0)
ens_c = np.mean(pt_c, 0)

rng = np.random.default_rng(0)
n = len(Y["test"])
diffs = {"auc": [], "dpd": [], "eod": [], "w_auc": []}
for _ in range(N_BOOT):
    idx = rng.integers(0, n, n)
    yb, rb = Y["test"][idx], R["test"][idx]
    try:
        mc = metrics(yb, rb, ens_c[idx])
        mf3 = metrics(yb, rb, ens_f3[idx])
    except ValueError:
        continue
    for k in diffs:
        diffs[k].append(mf3[k] - mc[k])

for k in ["auc", "dpd", "eod", "w_auc"]:
    v = diffs[k]
    m_val = float(np.mean(v))
    lo, hi = np.percentile(v, [2.5, 97.5])
    sig = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
    print(f"  Δ{k:6s} | mean: {m_val:+.4f} | 95% CI: [{lo:+.4f}, {hi:+.4f}] ({sig})")

print("\nDone. Results saved to latent_erasure_output.txt.")