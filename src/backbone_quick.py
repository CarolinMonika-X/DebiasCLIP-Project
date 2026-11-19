import json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

A = torch.load("cached_features.pt", weights_only=True)
B = torch.load("cached_features_biomedclip.pt", weights_only=True)
for s in ("train", "val", "test"):
    assert torch.equal(A[s]["labels"].long(), B[s]["labels"].long())
    assert torch.equal(A[s]["races"].long(), B[s]["races"].long())

f = lambda d, s, k: d[s][k].float()
V = {
    "openclip_img":            lambda s: f(A, s, "img"),
    "openclip_img+txt":        lambda s: torch.cat([f(A, s, "img"), f(A, s, "txt")], 1),
    "biomed_img":              lambda s: f(B, s, "img"),
    "biomed_img+txt":          lambda s: torch.cat([f(B, s, "img"), f(B, s, "txt")], 1),
    "biomed_img+openclip_txt": lambda s: torch.cat([f(B, s, "img"), f(A, s, "txt")], 1),
    "all4":                    lambda s: torch.cat([f(A, s, "img"), f(A, s, "txt"), f(B, s, "img"), f(B, s, "txt")], 1),
}
y = {s: A[s]["labels"].float() for s in ("train", "val", "test")}
r = {s: A[s]["races"].long().numpy() for s in ("train", "val", "test")}
EPOCHS, SEEDS = 20, range(5)

def run(Xtr, ytr, Xva, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6   # train statistics only
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = len(Xtr)
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            loss = nn.functional.binary_cross_entropy_with_logits(m(Xtr[idx]).squeeze(1), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(Xva).squeeze(1)).numpy(), torch.sigmoid(m(Xte).squeeze(1)).numpy()

def fair(p, yy, rr, thr=0.5):
    yh = (p >= thr).astype(int)
    pos = [yh[rr == g].mean() for g in range(3)]
    tpr = [yh[(rr == g) & (yy == 1)].mean() for g in range(3)]
    fpr = [yh[(rr == g) & (yy == 0)].mean() for g in range(3)]
    return max(pos) - min(pos), max(max(tpr) - min(tpr), max(fpr) - min(fpr))

res = {}
for name, fn in V.items():
    Xtr, Xva, Xte = fn("train"), fn("val"), fn("test")
    pv, pt = zip(*[run(Xtr, y["train"], Xva, Xte, s) for s in SEEDS])
    pv, pt = np.mean(pv, 0), np.mean(pt, 0)
    yv, yt = y["val"].numpy(), y["test"].numpy()
    dpd, eod = fair(pt, yt, r["test"])
    res[name] = dict(val_auc=roc_auc_score(yv, pv), val_acc=float(((pv >= .5) == yv).mean()),
                     test_auc=roc_auc_score(yt, pt), test_acc=float(((pt >= .5) == yt).mean()),
                     test_dpd=float(dpd), test_eod_max=float(eod))
    print(f"{name:26s} val AUC {res[name]['val_auc']:.4f} acc {res[name]['val_acc']:.4f} | "
          f"test AUC {res[name]['test_auc']:.4f} acc {res[name]['test_acc']:.4f} DPD {dpd:.4f} EOD {eod:.4f}")

best = max(res, key=lambda k: res[k]["val_auc"])
print("\nBest variant by VALIDATION AUC:", best)
json.dump(res, open("backbone_quick_results.json", "w"), indent=2)
print("Saved backbone_quick_results.json")
