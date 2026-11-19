import re, sys, json, numpy as np, torch
from math import sqrt
from datasets import load_dataset
from torch.utils.data import random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 2      # 2 = whole diagnosis/treatment sentences deleted; 0 = race words only
ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
y = np.array([{"no": 0, "yes": 1}[str(v).lower()] for v in ds["glaucoma"]])
n = len(ds); n_tr, n_va = int(0.7 * n), int(0.1 * n); n_te = n - n_tr - n_va
parts = random_split(range(n), [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42))
tr, va, te = (list(p.indices) for p in parts)
A = torch.load("cached_features.pt", weights_only=True)
r = A["test"]["races"].numpy()
for s, idx in (("train", tr), ("val", va), ("test", te)):
    assert np.array_equal(A[s]["labels"].numpy().astype(int), y[idx]), "alignment failed " + s

RACE = re.compile(r"asian(?:[\s-]+american)?|black|white|caucasian|african(?:[\s-]+american)?|hispanic|latino|latina|latinx|"
                  r"indigenous|native[\s-]+american|race|racial|ethnic\w*", re.I)
M2 = re.compile(r"glaucoma\w*|trabecul\w*|laser|slt|surger\w*|surgical|drops?|eye ?drops?|latanoprost|timolol|brimonidine|dorzolamide|"
                r"travoprost|bimatoprost|cosopt|combigan|lumigan|xalatan|rhopressa|netarsudil|medication\w*|tube|shunt|iop|pressure|"
                r"diagnos\w*|suspect\w*|dx|referr\w*|evaluat\w*|screen\w*|follow[- ]?up|rule[sd]? out|r/o", re.I)
SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
def prep(t):
    t = str(t)
    if LEVEL == 2:
        t = " ".join(s for s in SPLIT.split(t) if s.strip() and not M2.search(s))
    return RACE.sub(" ", t)
T = [prep(a) + " " + prep(b) for a, b in zip(ds["gpt4_summary"], ds["note"])]

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000)
Xtr = vec.fit_transform([T[i] for i in tr]); Xva = vec.transform([T[i] for i in va]); Xte = vec.transform([T[i] for i in te])
best, bc, bm = -1, None, None
for C in (0.3, 1.0, 3.0, 10.0):
    m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y[tr])
    a = roc_auc_score(y[va], m.predict_proba(Xva)[:, 1])
    print(f"TF-IDF C={C}: val AUC {a:.4f}")
    if a > best + 1e-9: best, bc, bm = a, C, m
tv, tt = bm.predict_proba(Xva)[:, 1], bm.predict_proba(Xte)[:, 1]
print("chosen C (validation):", bc)

F = [np.load(f) for f in ("ft_probs.npz", "ft_probs_s1.npz", "ft_probs_s2.npz")]
fv, ft = np.mean([d["val"] for d in F], 0), np.mean([d["test"] for d in F], 0)
yv, yt = y[va], y[te]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))
def thr_of(p, yy):
    f, t, th = roc_curve(yy, p); return float(th[int(np.argmax(t - f))])
acc = lambda p, yy, t: float(((p >= t) == yy).mean())

ws = np.round(np.arange(0, 1.01, 0.1), 1)
w = float(max(ws, key=lambda w: roc_auc_score(yv, w * lg(tv) + (1 - w) * lg(fv))))
pv, pt = sig(w * lg(tv) + (1 - w) * lg(fv)), sig(w * lg(tt) + (1 - w) * lg(ft))
print(f"\nMASK LEVEL {LEVEL}: image models see no text; text model sees only the processed notes")
for name, a, b in (("TF-IDF alone", tv, tt), ("ft image (3 seeds) alone", fv, ft), (f"blend (w on text={w})", pv, pt)):
    t = thr_of(a, yv)
    print(f"  {name:26s} val AUC {roc_auc_score(yv, a):.4f} acc {acc(a, yv, t):.4f} | test AUC {roc_auc_score(yt, b):.4f} acc {acc(b, yt, t):.4f}")
t, tf = thr_of(pv, yv), thr_of(fv, yv)
rng = np.random.default_rng(0); da, du = [], []
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))
    da.append(((pt[i] >= t) == yt[i]).mean() - ((ft[i] >= tf) == yt[i]).mean())
    du.append(roc_auc_score(yt[i], pt[i]) - roc_auc_score(yt[i], ft[i]))
print("blend minus ft-image-alone, 95% bootstrap CI  dAccuracy", np.percentile(da, [2.5, 97.5]).round(4), " dAUC", np.percentile(du, [2.5, 97.5]).round(4))

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
yh = (pt >= t).astype(int)
print("Per-race test rates of the blend (Wilson 95% CI)")
for g, nm in enumerate(["Asian", "Black", "White"]):
    kp, npos = int(yh[(r == g) & (yt == 1)].sum()), int(((r == g) & (yt == 1)).sum())
    kf, nneg = int(yh[(r == g) & (yt == 0)].sum()), int(((r == g) & (yt == 0)).sum())
    a1, a2 = wilson(kp, npos); b1, b2 = wilson(kf, nneg)
    print(f"  {nm:6s} TPR {kp / npos:.3f} [{a1:.3f},{a2:.3f}]  FNR {1 - kp / npos:.3f}  FPR {kf / nneg:.3f} [{b1:.3f},{b2:.3f}]")
json.dump({"level": LEVEL, "C": bc, "w_text": w, "thr": t}, open(f"clean_blend_L{LEVEL}.json", "w"))
