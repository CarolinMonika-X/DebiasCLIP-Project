import json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
A = torch.load("cached_features.pt", weights_only=True)
F = np.load("ft_probs.npz")
S = ("train", "val", "test")
y = {s: A[s]["labels"].numpy().astype(int) for s in S}
r = {s: A[s]["races"].numpy() for s in S}
assert len(F["val"]) == len(y["val"]) and len(F["test"]) == len(y["test"])
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

ytr_t = torch.tensor(y["train"]).float()
pv_m, pt_m = zip(*[run(X["train"], ytr_t, X["val"], X["test"], s) for s in range(5)])
pv_m, pt_m = np.mean(pv_m, 0), np.mean(pt_m, 0)
fv, ft = F["val"], F["test"]

ws = np.round(np.arange(0, 1.01, 0.1), 1)
va = {w: roc_auc_score(y["val"], w * fv + (1 - w) * pv_m) for w in ws}
wb = max(va, key=va.get)
print("blend weight on fine-tuned model chosen on VALIDATION:", wb)

def best_thr(p, yy):
    ts = np.arange(0.30, 0.71, 0.01)
    return float(ts[int(np.argmax([((p >= t) == yy).mean() for t in ts]))])

def rates(p, yy, rr, t):
    yh = (p >= t).astype(int)
    pos = [yh[rr == g].mean() for g in range(3)]
    tpr = [yh[(rr == g) & (yy == 1)].mean() for g in range(3)]
    fpr = [yh[(rr == g) & (yy == 0)].mean() for g in range(3)]
    return max(pos) - min(pos), max(max(tpr) - min(tpr), max(fpr) - min(fpr)), tpr, fpr

cands = {"text-MLP baseline (B32 img+txt)": (pv_m, pt_m),
         "fine-tuned image model only": (fv, ft),
         f"blend (w={wb})": (wb * fv + (1 - wb) * pv_m, wb * ft + (1 - wb) * pt_m)}
res = {}
for name, (pv, pt) in cands.items():
    t = best_thr(pv, y["val"])
    dpd, eod, tpr, fpr = rates(pt, y["test"], r["test"], t)
    res[name] = dict(val_auc=roc_auc_score(y["val"], pv), thr=t, val_acc_at_thr=float(((pv >= t) == y["val"]).mean()),
                     test_auc=roc_auc_score(y["test"], pt), test_acc_05=float(((pt >= .5) == y["test"]).mean()),
                     test_acc_thr=float(((pt >= t) == y["test"]).mean()), dpd=float(dpd), eod=float(eod),
                     tpr_by_race=[float(v) for v in tpr], fpr_by_race=[float(v) for v in fpr])
    x = res[name]
    print(f"{name:34s} val AUC {x['val_auc']:.4f} thr {t:.2f} | test AUC {x['test_auc']:.4f} acc@0.5 {x['test_acc_05']:.4f} acc@thr {x['test_acc_thr']:.4f} DPD {dpd:.4f} EOD {eod:.4f}")

# paired bootstrap on TEST: blend minus baseline
kb, ka = list(cands)[2], list(cands)[0]
tb, ta = res[kb]["thr"], res[ka]["thr"]
rng = np.random.default_rng(0); d_acc, d_auc = [], []
yt = y["test"]; pb, pa = cands[kb][1], cands[ka][1]
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))
    d_acc.append(((pb[i] >= tb) == yt[i]).mean() - ((pa[i] >= ta) == yt[i]).mean())
    d_auc.append(roc_auc_score(yt[i], pb[i]) - roc_auc_score(yt[i], pa[i]))
print("blend minus baseline, 95% bootstrap CI  dAccuracy:", np.percentile(d_acc, [2.5, 97.5]).round(4),
      " dAUC:", np.percentile(d_auc, [2.5, 97.5]).round(4))
json.dump(res, open("ft_blend_results.json", "w"), indent=2)
print("Saved ft_blend_results.json")
