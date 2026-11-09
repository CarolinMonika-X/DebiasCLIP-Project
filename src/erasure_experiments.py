"""
erasure_experiments.py - representation-level race erasure (LEACE + INLP).
Read-only on cached_features.pt. Does NOT touch ablation_full.py or ablation_results.json.
Erasers are fit on TRAIN only; INLP depth is chosen on VAL only; TEST is reported.
"""
import json, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

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

# ---------------- LEACE (closed-form linear concept erasure) ----------------
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
    E = Winv @ (U @ U.T) @ W                  # x' = x - E (x - mu)
    return lambda A: (A - (A - mu) @ E.T).astype(np.float32)

# ---------------- INLP (iterative nullspace projection) ----------------
def fit_inlp(F, z, n_iter):
    dim = F.shape[1]
    dirs = []
    P = np.eye(dim)
    for _ in range(n_iter):
        clf = LogisticRegression(max_iter=3000)
        clf.fit(F @ P, z)
        dirs.append(clf.coef_)
        Q, _ = np.linalg.qr(np.concatenate(dirs).T)
        P = np.eye(dim) - Q @ Q.T
    return lambda A: (A @ P).astype(np.float32)

# ---------------- leakage probe ----------------
def race_probe(Ftr, rtr, Fte, rte):
    clf = LogisticRegression(max_iter=3000).fit(Ftr, rtr)
    p = clf.predict_proba(Fte)
    return dict(
        acc=float((p.argmax(1) == rte).mean()),
        macro_auc=float(roc_auc_score(rte, p, multi_class="ovr", average="macro")),
        bal_acc=float(balanced_accuracy_score(rte, p.argmax(1)))
    )

# ---------------- classification head ----------------
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

def train_head(F, y, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Head(F.shape[1])
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

# ---------------- metrics ----------------
def metrics(y, r, p):
    pred = p > THR
    G = np.unique(r)
    pr  = [pred[r == g].mean() for g in G]
    tpr = [pred[(r == g) & (y == 1)].mean() for g in G]
    fpr = [pred[(r == g) & (y == 0)].mean() for g in G]
    return dict(
        acc=float((pred == y).mean()),
        auc=float(roc_auc_score(y, p)),
        dpd=float(max(pr) - min(pr)),
        eod=float(max(max(tpr) - min(tpr), max(fpr) - min(fpr))),
        w_auc=float(min(roc_auc_score(y[r == g], p[r == g]) for g in G))
    )

def run(name, F):
    pt, ms = [], []
    for s in SEEDS:
        m = train_head(F["train"], Y["train"], s)
        pt.append(predict(m, F["test"]))
        ms.append(metrics(Y["test"], R["test"], pt[-1]))
    agg = {k: (float(np.mean([x[k] for x in ms])), float(np.std([x[k] for x in ms]))) for k in ms[0]}
    print(f"{name:28s} " + " | ".join(f"{k} {v[0]:.4f}+/-{v[1]:.4f}" for k, v in agg.items()))
    return agg, np.mean(pt, 0)

def boot(pa, pb, y, r, seed=0):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("auc", "dpd", "eod", "w_auc")}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        try:
            ma, mb = metrics(y[i], r[i], pa[i]), metrics(y[i], r[i], pb[i])
        except ValueError:
            continue
        for k in out:
            out[k].append(mb[k] - ma[k])
    res = {}
    for k, v in out.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        res[k] = dict(mean=float(np.mean(v)), lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0))
        print(f"   {k:6s} mean {res[k]['mean']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {'significant' if res[k]['sig'] else 'NOT significant'}")
    return res

out = {"leakage": {}, "runs": {}, "bootstrap": {}}

# 1) Fit LEACE erasers on TRAIN only
print("\nFitting LEACE erasers...")
leace_img = fit_leace(X["train"], R["train"])
XL = {s: leace_img(X[s]) for s in SPLITS}
DUAL = {s: cat(X[s], T[s]) for s in SPLITS}
leace_dual = fit_leace(DUAL["train"], R["train"])
DL = {s: leace_dual(DUAL[s]) for s in SPLITS}

# 2) INLP depth selection on VALIDATION only
chance = 1.0 / N_CLS
chosen_k = None
print("\n=== INLP depth selection (VALIDATION) ===")
for k in (1, 2, 4, 8, 16):
    f = fit_inlp(X["train"], R["train"], k)
    ba = race_probe(f(X["train"]), R["train"], f(X["val"]), R["val"])["bal_acc"]
    print(f"  k={k:2d} val race bal-acc {ba:.4f} (chance={chance:.4f})")
    if chosen_k is None and ba <= chance + 0.02:
        chosen_k = k
chosen_k = chosen_k or 16
inlp = fit_inlp(X["train"], R["train"], chosen_k)
XI = {s: inlp(X[s]) for s in SPLITS}
print(f"--> INLP depth chosen on validation = {chosen_k}")

# 3) Leakage check on TEST
print("\n=== Race leakage on TEST ===")
for nm, F in [("image raw", X), ("image LEACE", XL), (f"image INLP k={chosen_k}", XI),
              ("dual raw", DUAL), ("dual LEACE", DL)]:
    out["leakage"][nm] = race_probe(F["train"], R["train"], F["test"], R["test"])
    print(f"{nm:22s} {out['leakage'][nm]}")

# 4) Train glaucoma heads (5 seeds, test evaluation)
print("\n=== Glaucoma heads (TEST, mean+/-std over 5 seeds) ===")
cfgs = {
    "A_repro  image raw": X,
    "A_LEACE  image erased": XL,
    "C_repro  dual raw": DUAL,
    "F1 dual, image LEACE": {s: cat(XL[s], T[s]) for s in SPLITS},
    "F2 dual, joint LEACE": DL,
    f"G  dual, image INLP k={chosen_k}": {s: cat(XI[s], T[s]) for s in SPLITS}
}
ens = {}
for nm, F in cfgs.items():
    out["runs"][nm], ens[nm] = run(nm, F)

# 5) Bootstrap comparisons vs C_repro
print("\n=== Bootstrap 95% CI vs C_repro (seed-ensembled) ===")
for nm in list(cfgs)[3:]:
    print(f"\n{nm} minus C_repro:")
    out["bootstrap"][nm] = boot(ens["C_repro  dual raw"], ens[nm], Y["test"], R["test"])

out["inlp_chosen_k"] = chosen_k

with open("erasure_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved erasure_results.json. Done.")