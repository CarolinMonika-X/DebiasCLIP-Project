"""
novelty_experiments.py - three method additions, evaluated against model C with the SAME pipeline
(cached features, 70/10/20 split, 5 seeds, bootstrap CIs).
 
  F  label-conditioned penalty on a CELL-BALANCED penalty batch (equal samples per group x label cell)
       -> targets the failure cause we identified: noisy group means in small mini-batches
  G  worst-group loss (max over groups of group BCE) on the same balanced penalty batch (Group-DRO-style)
  H  post-hoc equal-opportunity thresholds: one threshold per group, chosen on VALIDATION only
 
Baselines inside this script: C (no penalty) and E (label-conditioned penalty on ordinary mini-batches).
Run from the project root (needs cached_features.pt):   python src/novelty_experiments.py
Quick test: set env QUICK=1 (3 epochs, 2 seeds, 100 bootstrap resamples).
"""
import os, copy, time, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
 
QUICK = os.environ.get("QUICK") == "1"
CACHE = "cached_features.pt"
EPOCHS = 3 if QUICK else 40
SEEDS = list(range(2 if QUICK else 5))
N_BOOT = 100 if QUICK else 1000
BATCH, LR, THR = 256, 1e-3, 0.5
LAMS = [0.1, 1, 5]
 
# ---------------- data ----------------
D = torch.load(CACHE)
tr, va, te = D["train"], D["val"], D["test"]
sc = {k: StandardScaler().fit(tr[k].numpy()) for k in ("img", "txt")}
def prep(d):
    x = np.concatenate([sc["img"].transform(d["img"].numpy()), sc["txt"].transform(d["txt"].numpy())], axis=1)
    return torch.tensor(x, dtype=torch.float32), d["labels"].float(), d["races"].long()
Xtr, ytr, rtr = prep(tr); Xva, yva, rva = prep(va); Xte, yte, rte = prep(te)
yv, rv, yt, rt = yva.numpy(), rva.numpy(), yte.numpy(), rte.numpy()
GROUPS = np.unique(rt)
 
# ---------------- model ----------------
class Adapter(nn.Module):
    def __init__(self, d=1024, h=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, d))
        self.alpha = nn.Parameter(torch.tensor(0.5)); self.head = nn.Linear(d, 1)
    def forward(self, x):
        return self.head(x + self.alpha * self.mlp(x)).squeeze(-1)
 
# ---------------- penalties ----------------
def label_cond(lg, y, g):
    p = torch.sigmoid(lg); tot = lg.new_zeros(())
    for lab in (0.0, 1.0):
        m = [p[(g == k) & (y == lab)].mean() for k in torch.unique(g) if ((g == k) & (y == lab)).any()]
        if len(m) > 1:
            m = torch.stack(m); tot = tot + (m.max() - m.min())
    return tot
 
def worst_group_bce(lg, y, g):
    bce = nn.functional.binary_cross_entropy_with_logits
    losses = [bce(lg[g == k], y[g == k]) for k in torch.unique(g)]
    return torch.stack(losses).max()
 
# cell-balanced penalty batches: equal number of samples from each (group, label) cell
cell = (rtr * 2 + ytr.long()).numpy()
CELLS = [np.where(cell == c)[0] for c in np.unique(cell)]
def balanced_batch(rs):
    per = BATCH // len(CELLS)
    idx = np.concatenate([rs.choice(c, per, replace=True) for c in CELLS])
    idx = torch.from_numpy(idx)
    return Xtr[idx], ytr[idx], rtr[idx]
 
# ---------------- metrics ----------------
def metrics(p, y, r, thr=THR):
    t = thr if np.isscalar(thr) else np.asarray(thr)[r]
    pred = (p > t).astype(int); gs = np.unique(r)
    dpd = max(pred[r == g].mean() for g in gs) - min(pred[r == g].mean() for g in gs)
    eod = 0.0
    for lab in (0, 1):
        v = [pred[(r == g) & (y == lab)].mean() for g in gs if ((r == g) & (y == lab)).sum() > 0]
        if len(v) > 1: eod = max(eod, max(v) - min(v))
    aucs = [roc_auc_score(y[r == g], p[r == g]) for g in gs if len(np.unique(y[r == g])) > 1]
    return {"acc": accuracy_score(y, pred), "auc": roc_auc_score(y, p), "dpd": dpd, "eod": eod, "w_auc": min(aucs)}
 
# ---------------- training ----------------
def run(method, lam, seed):
    torch.manual_seed(seed); rs = np.random.RandomState(seed)
    m = Adapter(); opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-2); bce = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr, rtr), batch_size=BATCH, shuffle=True)
    best, state = float("inf"), None
    for _ in range(EPOCHS):
        m.train()
        for xb, yb, rb in loader:
            opt.zero_grad(); lg = m(xb); loss = bce(lg, yb)
            if method == "E" and lam > 0:
                loss = loss + lam * label_cond(lg, yb, rb)
            elif method in ("F", "G") and lam > 0:
                xf, yf, rf = balanced_batch(rs); lf = m(xf)
                loss = loss + lam * (label_cond(lf, yf, rf) if method == "F" else worst_group_bce(lf, yf, rf))
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = bce(m(Xva), yva).item()
        if vl < best: best, state = vl, copy.deepcopy(m.state_dict())
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(Xva)).numpy(), torch.sigmoid(m(Xte)).numpy()
 
def run_seeds(method, lam):
    t0 = time.time(); pv, pt = [], []
    for s in SEEDS:
        a, b = run(method, lam, s); pv.append(a); pt.append(b)
    return pv, pt, (time.time() - t0) / len(SEEDS)
 
def summ(preds, y, r):
    ms = [metrics(p, y, r) for p in preds]
    return {k: (float(np.mean([q[k] for q in ms])), float(np.std([q[k] for q in ms]))) for k in ms[0]}
 
def boot(p1, t1, p2, t2, seed=0):
    rng = np.random.default_rng(seed); out = {k: [] for k in ("auc", "dpd", "eod", "w_auc")}
    t1a = t1 if np.isscalar(t1) else np.asarray(t1); t2a = t2 if np.isscalar(t2) else np.asarray(t2)
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yt), len(yt))
        try:
            a = metrics(p1[i], yt[i], rt[i], t1a); b = metrics(p2[i], yt[i], rt[i], t2a)
        except ValueError:
            continue
        for k in out: out[k].append(a[k] - b[k])
    return {k: (float(np.mean(v)), *map(float, np.percentile(v, [2.5, 97.5]))) for k, v in out.items()}
 
def show(title, r):
    print(title)
    for k, (m, lo, hi) in r.items():
        print(f"   {k:6s} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {'significant' if lo > 0 or hi < 0 else 'NOT significant'}")
 
# ---------------- run everything ----------------
print(f"epochs={EPOCHS} seeds={len(SEEDS)} boot={N_BOOT}")
res, ens = {}, {}
pvC, ptC, secC = run_seeds("C", 0)
valC = summ(pvC, yv, rv)
print("\n=== Baseline C (image+text, no penalty) ===")
print({k: f"{v[0]:.4f}±{v[1]:.4f}" for k, v in summ(ptC, yt, rt).items()}, f"  {secC:.1f}s/run")
ens["C"] = np.mean(ptC, 0); ens_val_C = np.mean(pvC, 0)
 
chosen = {}
for method, name in [("E", "E  label-cond, ordinary batches"), ("F", "F  label-cond, cell-balanced penalty batch"),
                     ("G", "G  worst-group BCE, cell-balanced penalty batch")]:
    print(f"\n=== {name} ===")
    rows = []
    for lam in LAMS:
        pv, pt, sec = run_seeds(method, lam)
        mv, mt = summ(pv, yv, rv), summ(pt, yt, rt)
        rows.append((lam, mv, mt)); ens[(method, lam)] = np.mean(pt, 0)
        print(f"lam={lam:<4} VAL auc {mv['auc'][0]:.4f} eod {mv['eod'][0]:.4f} || TEST auc {mt['auc'][0]:.4f} dpd {mt['dpd'][0]:.4f} "
              f"eod {mt['eod'][0]:.4f}±{mt['eod'][1]:.4f} w_auc {mt['w_auc'][0]:.4f} | {sec:.1f}s/run")
    ok = [r for r in rows if r[1]["auc"][0] >= valC["auc"][0] - 0.01] or rows
    chosen[method] = min(ok, key=lambda r: r[1]["eod"][0])[0]
    print(f"--> lambda chosen on validation: {chosen[method]}")
    res[method] = {str(r[0]): {"val": r[1], "test": r[2]} for r in rows}
 
# ---------------- H: post-hoc equal-opportunity thresholds (chosen on validation) ----------------
tpr_target = float((ens_val_C[yv == 1] > THR).mean())
thr = np.zeros(int(GROUPS.max()) + 1)
for g in GROUPS:
    s = ens_val_C[(rv == g) & (yv == 1)]
    thr[g] = float(np.quantile(s, 1 - tpr_target))
print(f"\n=== H: per-group thresholds on model C (target TPR {tpr_target:.3f}, from validation) ===")
print("thresholds per group:", np.round(thr[GROUPS], 3))
mh = metrics(ens["C"], yt, rt, thr); mc = metrics(ens["C"], yt, rt, THR)
print("C  @0.5      :", {k: round(v, 4) for k, v in mc.items()})
print("C  @group-thr:", {k: round(v, 4) for k, v in mh.items()})
pred = ens["C"] > thr[rt]
for g in GROUPS:
    m = rt == g
    print(f"   group {g}: TPR {pred[m & (yt == 1)].mean():.3f}  FPR {pred[m & (yt == 0)].mean():.3f}  flagged {pred[m].mean():.3f}")
 
# ---------------- bootstrap vs C ----------------
print("\n=== Bootstrap 95% CI of difference vs C (test-set resampling, seed-ensembled predictions) ===")
boots = {}
for method in ("E", "F", "G"):
    t = f"{method} (lam={chosen[method]}) minus C"; boots[t] = boot(ens[(method, chosen[method])], THR, ens["C"], THR); show(t + ":", boots[t])
boots["H minus C"] = boot(ens["C"], thr, ens["C"], THR); show("H (group thresholds) minus C:", boots["H minus C"])
 
json.dump({"C_test": summ(ptC, yt, rt), "sweeps": res, "chosen": chosen, "thresholds": thr.tolist(), "bootstrap": boots},
          open("novelty_results.json", "w"), indent=2, default=float)
print("\nSaved novelty_results.json. Done.")