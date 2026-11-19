import sys, json, os, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

files = sys.argv[1:] or ["ft_probs.npz"]
A = torch.load("cached_features.pt", weights_only=True)
S = ("train", "val", "test")
y = {s: A[s]["labels"].numpy().astype(int) for s in S}
r = {s: A[s]["races"].numpy() for s in S}
X = {s: torch.cat([A[s]["img"].float(), A[s]["txt"].float()], 1) for s in S}

def run(Xtr, ytr, Xva, Xte, seed, EPOCHS=20):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            loss = nn.functional.binary_cross_entropy_with_logits(m(Xtr[idx]).squeeze(1), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(Xva).squeeze(1)).numpy(), torch.sigmoid(m(Xte).squeeze(1)).numpy()

if os.path.exists("baseline_probs.npz"):
    B = np.load("baseline_probs.npz"); pv_m, pt_m = B["val"], B["test"]
else:
    ytr_t = torch.tensor(y["train"]).float()
    pv_m, pt_m = zip(*[run(X["train"], ytr_t, X["val"], X["test"], s) for s in range(5)])
    pv_m, pt_m = np.mean(pv_m, 0), np.mean(pt_m, 0)
    np.savez("baseline_probs.npz", val=pv_m, test=pt_m)

def best_thr(p, yy):
    ts = np.arange(0.30, 0.71, 0.01)
    return float(ts[int(np.argmax([((p >= t) == yy).mean() for t in ts]))])

def best_w(fv):
    ws = np.round(np.arange(0, 1.01, 0.1), 1)
    return float(max(ws, key=lambda w: roc_auc_score(y["val"], w * fv + (1 - w) * pv_m)))

def gaps(p, yy, rr, t):
    yh = (p >= t).astype(int)
    pos = [yh[rr == g].mean() for g in range(3)]
    tpr = [yh[(rr == g) & (yy == 1)].mean() for g in range(3)]
    fpr = [yh[(rr == g) & (yy == 0)].mean() for g in range(3)]
    tg, fg = max(tpr) - min(tpr), max(fpr) - min(fpr)
    return dict(dpd=max(pos) - min(pos), tpr_gap=tg, fpr_gap=fg, eod=max(tg, fg),
                fnr_black=1 - tpr[1], fpr_black=fpr[1], fpr_white=fpr[2], fnr_asian=1 - tpr[0])

def report(name, fv, ft):
    w = best_w(fv)
    pv, pt = w * fv + (1 - w) * pv_m, w * ft + (1 - w) * pt_m
    t = best_thr(pv, y["val"])
    g = gaps(pt, y["test"], r["test"], t)
    print(f"{name:28s} w={w:.1f} thr={t:.2f} | val AUC {roc_auc_score(y['val'], pv):.4f} | test AUC {roc_auc_score(y['test'], pt):.4f} "
          f"acc {((pt >= t) == y['test']).mean():.4f} DPD {g['dpd']:.4f} EOD {g['eod']:.4f}")
    return pv, pt, t, w

F = [np.load(f) for f in files]
for f, d in zip(files, F):
    assert len(d["val"]) == len(y["val"]) and len(d["test"]) == len(y["test"]), f
    print(f"{f}: fine-tuned alone val AUC {roc_auc_score(y['val'], d['val']):.4f} test AUC {roc_auc_score(y['test'], d['test']):.4f}")
tb = best_thr(pv_m, y["val"])
gb = gaps(pt_m, y["test"], r["test"], tb)
print(f"{'text-MLP baseline':28s} thr={tb:.2f} | val AUC {roc_auc_score(y['val'], pv_m):.4f} | test AUC {roc_auc_score(y['test'], pt_m):.4f} "
      f"acc {((pt_m >= tb) == y['test']).mean():.4f} DPD {gb['dpd']:.4f} EOD {gb['eod']:.4f}")
for f, d in zip(files, F):
    report("blend with " + f, d["val"], d["test"])
fv_e, ft_e = np.mean([d["val"] for d in F], 0), np.mean([d["test"] for d in F], 0)
pv, pt, t, w = report(f"blend with ensemble of {len(F)}", fv_e, ft_e)
np.savez("ft_blend_probs.npz", base_val=pv_m, base_test=pt_m, blend_val=pv, blend_test=pt, thr_base=tb, thr_blend=t)

rng = np.random.default_rng(0); yt, rt = y["test"], r["test"]
keys = ["dpd", "tpr_gap", "fpr_gap", "eod", "fnr_black", "fpr_black", "fpr_white", "fnr_asian"]
D = {k: [] for k in keys}; dacc, dauc = [], []
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))
    a, b = gaps(pt[i], yt[i], rt[i], t), gaps(pt_m[i], yt[i], rt[i], tb)
    for k in keys: D[k].append(a[k] - b[k])
    dacc.append(((pt[i] >= t) == yt[i]).mean() - ((pt_m[i] >= tb) == yt[i]).mean())
    dauc.append(roc_auc_score(yt[i], pt[i]) - roc_auc_score(yt[i], pt_m[i]))
print("\nPaired bootstrap (blend minus baseline, 1000 resamples), 95% CI:")
print(f"  dAccuracy {np.mean(dacc):+.4f} {np.percentile(dacc, [2.5, 97.5]).round(4)}   dAUC {np.mean(dauc):+.4f} {np.percentile(dauc, [2.5, 97.5]).round(4)}")
for k in keys:
    print(f"  d{k:10s} {np.mean(D[k]):+.4f} {np.percentile(D[k], [2.5, 97.5]).round(4)}")
json.dump({"files": files, "blend_weight": w, "thr_blend": t, "thr_base": tb,
           "bootstrap": {k: [float(np.mean(D[k])), *np.percentile(D[k], [2.5, 97.5]).tolist()] for k in keys}},
          open("ft_blend2_results.json", "w"), indent=2)
print("Saved ft_blend2_results.json and ft_blend_probs.npz")
