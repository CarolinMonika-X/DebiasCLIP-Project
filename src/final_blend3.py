import sys, json, itertools, numpy as np, torch
from math import sqrt
from sklearn.metrics import roc_auc_score, roc_curve

suffix = sys.argv[1]                      # "" for the sentence-deleted run, "_mask0" for the unmasked run
bert_files = sys.argv[2:]                 # e.g. bert_L2_s0.npz
A = torch.load("cached_features.pt", weights_only=True)
yv, yt = A["val"]["labels"].numpy().astype(int), A["test"]["labels"].numpy().astype(int)
rt = A["test"]["races"].numpy()
sv, st = np.load(f"stack_val_probs{suffix}.npy"), np.load(f"stack_test_probs{suffix}.npy")
F = [np.load(f) for f in ("ft_probs.npz", "ft_probs_s1.npz", "ft_probs_s2.npz")]
fv, ft = np.mean([d["val"] for d in F], 0), np.mean([d["test"] for d in F], 0)
B = [np.load(f) for f in bert_files]
bv, bt = np.mean([d["val"] for d in B], 0), np.mean([d["test"] for d in B], 0)
assert len(sv) == len(fv) == len(bv) == len(yv) and len(st) == len(ft) == len(bt) == len(yt)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))

def thr_of(p, yy):
    f, t, th = roc_curve(yy, p); return float(th[int(np.argmax(t - f))])
def acc(p, yy, t): return float(((p >= t) == yy).mean())
for name, a, b in (("stack", sv, st), ("ft image (3 seeds)", fv, ft), ("fine-tuned BERT", bv, bt)):
    t = thr_of(a, yv)
    print(f"{name:20s} val AUC {roc_auc_score(yv, a):.4f} | test AUC {roc_auc_score(yt, b):.4f} acc@valYouden {acc(b, yt, t):.4f}")

best, bw = -1, None
for w1 in np.arange(0, 1.01, 0.1):
    for w2 in np.arange(0, 1.01 - w1, 0.1):
        w3 = 1 - w1 - w2
        a = roc_auc_score(yv, w1 * lg(sv) + w2 * lg(fv) + w3 * lg(bv))
        if a > best + 1e-9: best, bw = a, (round(w1, 1), round(w2, 1), round(w3, 1))
w1, w2, w3 = bw
pv = sig(w1 * lg(sv) + w2 * lg(fv) + w3 * lg(bv)); pt = sig(w1 * lg(st) + w2 * lg(ft) + w3 * lg(bt))
t = thr_of(pv, yv)
print(f"\nweights chosen on VALIDATION (stack, ft image, BERT) = {bw}, threshold {t:.3f}")
print(f"FINAL 3-way blend: val AUC {roc_auc_score(yv, pv):.4f} | test AUC {roc_auc_score(yt, pt):.4f} acc {acc(pt, yt, t):.4f}")

# previous final = 0.6*stack + 0.4*ft image (as in stack_plus.py)
ppv = sig(0.6 * lg(sv) + 0.4 * lg(fv)); ppt = sig(0.6 * lg(st) + 0.4 * lg(ft)); tp = thr_of(ppv, yv)
print(f"PREVIOUS final (stack+ft): test AUC {roc_auc_score(yt, ppt):.4f} acc {acc(ppt, yt, tp):.4f}")
rng = np.random.default_rng(0); da, du = [], []
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))
    da.append(((pt[i] >= t) == yt[i]).mean() - ((ppt[i] >= tp) == yt[i]).mean())
    du.append(roc_auc_score(yt[i], pt[i]) - roc_auc_score(yt[i], ppt[i]))
print("3-way minus previous final, 95% bootstrap CI  dAccuracy", np.percentile(da, [2.5, 97.5]).round(4), " dAUC", np.percentile(du, [2.5, 97.5]).round(4))

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
yh = (pt >= t).astype(int)
print("\nPer-race test rates, final 3-way blend (Wilson 95% CI)")
for g, nm in enumerate(["Asian", "Black", "White"]):
    kp, npos = int(yh[(rt == g) & (yt == 1)].sum()), int(((rt == g) & (yt == 1)).sum())
    kf, nneg = int(yh[(rt == g) & (yt == 0)].sum()), int(((rt == g) & (yt == 0)).sum())
    a1, a2 = wilson(kp, npos); b1, b2 = wilson(kf, nneg)
    print(f"  {nm:6s} TPR {kp / npos:.3f} [{a1:.3f},{a2:.3f}]  FNR {1 - kp / npos:.3f}  FPR {kf / nneg:.3f} [{b1:.3f},{b2:.3f}]")
json.dump({"weights": bw, "threshold": t, "bert_files": bert_files, "suffix": suffix}, open(f"final_blend3{suffix}.json", "w"))
