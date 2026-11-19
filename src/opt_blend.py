import os, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score, roc_curve

suffix = "" if sys.argv[1] == "none" else sys.argv[1]
bert_files = sys.argv[2:]
A = torch.load("cached_features.pt", weights_only=True)
yv, yt = A["val"]["labels"].numpy().astype(int), A["test"]["labels"].numpy().astype(int)

sv, st = np.load(f"stack_val_probs{suffix}.npy"), np.load(f"stack_test_probs{suffix}.npy")
F = [np.load(f) for f in ("ft_probs.npz", "ft_probs_s1.npz", "ft_probs_s2.npz") if os.path.exists(f)]
fv, ft = np.mean([d["val"] for d in F], 0), np.mean([d["test"] for d in F], 0)
B = [np.load(f) for f in bert_files if os.path.exists(f)]
bv, bt = np.mean([d["val"] for d in B], 0), np.mean([d["test"] for d in B], 0)

has_rf = os.path.exists("retfound_tta_probs.npz")
if has_rf:
    R = np.load("retfound_tta_probs.npz")
    rv, rt = R["val"], R["test"]

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))

def best_acc_thr(p, yy):
    thrs = np.linspace(0.25, 0.75, 501)
    accs = [((p >= t) == yy).mean() for t in thrs]
    best_a = max(accs)
    cands = [t for t, a in zip(thrs, accs) if abs(a - best_a) < 1e-9]
    return float(min(cands, key=lambda t: abs(t - 0.5))), float(best_a)

def youden_thr(p, yy):
    f, t, th = roc_curve(yy, p)
    return float(th[int(np.argmax(t - f))])

best_auc, bw = -1, None
grid = np.arange(0, 1.01, 0.05)
for w1 in grid:
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        for w3 in np.arange(0, 1.01 - w1 - w2, 0.05):
            w4 = max(0.0, 1.0 - w1 - w2 - w3)
            if not has_rf and w4 > 1e-6:
                continue
            z = w1 * lg(sv) + w2 * lg(fv) + w3 * lg(bv) + (w4 * lg(rv) if has_rf else 0.0)
            a = roc_auc_score(yv, z)
            if a > best_auc + 1e-9:
                best_auc = a
                bw = (round(float(w1), 2), round(float(w2), 2), round(float(w3), 2), round(float(w4), 2))

w1, w2, w3, w4 = bw
pv = sig(w1 * lg(sv) + w2 * lg(fv) + w3 * lg(bv) + (w4 * lg(rv) if has_rf else 0.0))
pt = sig(w1 * lg(st) + w2 * lg(ft) + w3 * lg(bt) + (w4 * lg(rt) if has_rf else 0.0))

ty = youden_thr(pv, yv)
ta, va_acc = best_acc_thr(pv, yv)
print(f"=== Suffix: {suffix or 'Mask-2'} | RETFound included: {has_rf} ===")
print(f"Weights (stack, ft_img, BERT, RETFound): {bw}")
print(f"Val AUC:  {roc_auc_score(yv, pv):.4f} | Test AUC:  {roc_auc_score(yt, pt):.4f}")
print(f"Test Acc @ Val-Youden ({ty:.3f}): {((pt >= ty) == yt).mean():.4f}")
print(f"Test Acc @ Val-MaxAcc ({ta:.3f}): {((pt >= ta) == yt).mean():.4f} (Val Acc: {va_acc:.4f})\n")
