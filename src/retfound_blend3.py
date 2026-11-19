import sys, json, numpy as np, torch
from math import sqrt
from sklearn.metrics import roc_auc_score, roc_curve

suffix = "" if sys.argv[1] == "none" else sys.argv[1]
bert_file = sys.argv[2]
A = torch.load("cached_features.pt", weights_only=True)
yv, yt = A["val"]["labels"].numpy().astype(int), A["test"]["labels"].numpy().astype(int)
rt = A["test"]["races"].numpy()
sv, st = np.load(f"stack_val_probs{suffix}.npy"), np.load(f"stack_test_probs{suffix}.npy")
R = np.load("retfound_tta_probs.npz")
fv, ft = R["val"], R["test"]
B = np.load(bert_file)
bv, bt = B["val"], B["test"]

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))
def thr_of(p, yy):
    f, t, th = roc_curve(yy, p); return float(th[int(np.argmax(t - f))])
def acc(p, yy, t): return float(((p >= t) == yy).mean())

for name, a, b in (("stack", sv, st), ("RETFound 518 TTA", fv, ft), ("fine-tuned BERT", bv, bt)):
    t = thr_of(a, yv)
    print(f"{name:20s} val AUC {roc_auc_score(yv, a):.4f} | test AUC {roc_auc_score(yt, b):.4f} acc@valYouden {acc(b, yt, t):.4f}")

best, bw = -1, None
for w1 in np.arange(0, 1.01, 0.1):
    for w2 in np.arange(0, 1.01 - w1, 0.1):
        w3 = 1 - w1 - w2
        a = roc_auc_score(yv, w1 * lg(sv) + w2 * lg(fv) + w3 * lg(bv))
        if a > best + 1e-9: best, bw = a, (round(float(w1), 1), round(float(w2), 1), round(float(w3), 1))
w1, w2, w3 = bw
pv = sig(w1 * lg(sv) + w2 * lg(fv) + w3 * lg(bv)); pt = sig(w1 * lg(st) + w2 * lg(ft) + w3 * lg(bt))
t = thr_of(pv, yv)
print(f"\nweights chosen on VALIDATION (stack, RETFound, BERT) = {bw}, threshold {t:.3f}")
print(f"FINAL RETFound 3-way blend: val AUC {roc_auc_score(yv, pv):.4f} | test AUC {roc_auc_score(yt, pt):.4f} acc {acc(pt, yt, t):.4f}")

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
yh = (pt >= t).astype(int)
print("\nPer-race test rates, RETFound 3-way blend (Wilson 95% CI)")
for g, nm in enumerate(["Asian", "Black", "White"]):
    kp, npos = int(yh[(rt == g) & (yt == 1)].sum()), int(((rt == g) & (yt == 1)).sum())
    kf, nneg = int(yh[(rt == g) & (yt == 0)].sum()), int(((rt == g) & (yt == 0)).sum())
    a1, a2 = wilson(kp, npos); b1, b2 = wilson(kf, nneg)
    print(f"  {nm:6s} TPR {kp / npos:.3f} [{a1:.3f},{a2:.3f}]  FNR {1 - kp / npos:.3f}  FPR {kf / nneg:.3f} [{b1:.3f},{b2:.3f}]")
