import os, re, numpy as np, torch
from datasets import load_dataset
from torch.utils.data import random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

A = torch.load("cached_features.pt", weights_only=True)
yv, yt = A["val"]["labels"].numpy().astype(int), A["test"]["labels"].numpy().astype(int)

R = np.load("retfound_tta_probs.npz")
rv, rt = R["val"], R["test"]
F = [np.load(f) for f in ("ft_probs.npz", "ft_probs_s1.npz", "ft_probs_s2.npz") if os.path.exists(f)]
fv, ft = np.mean([d["val"] for d in F], 0), np.mean([d["test"] for d in F], 0)

def best_acc_thr(p, yy):
    thrs = np.linspace(0.25, 0.75, 501)
    accs = [((p >= t) == yy).mean() for t in thrs]
    best_a = max(accs)
    return float(min([t for t, a in zip(thrs, accs) if abs(a - best_a) < 1e-9], key=lambda t: abs(t - 0.5)))

def youden_thr(p, yy):
    f, t, th = roc_curve(yy, p)
    return float(th[int(np.argmax(t - f))])

for name, pv, pt in (("RETFound (336px TTA)", rv, rt), ("ft_image (3 seeds)", fv, ft)):
    ty, ta = youden_thr(pv, yv), best_acc_thr(pv, yv)
    print(f"{name:22s} | Val AUC {roc_auc_score(yv, pv):.4f} | Test AUC {roc_auc_score(yt, pt):.4f} | Test Acc@Youden {((pt>=ty)==yt).mean():.4f} | Test Acc@MaxAcc {((pt>=ta)==yt).mean():.4f}")

# Clean Mask-2 multimodal (NO unmasked CLIP text)
ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
y = np.array([{"no": 0, "yes": 1}[str(v).lower()] for v in ds["glaucoma"]])
n = len(ds); n_tr, n_va = int(0.7 * n), int(0.1 * n); n_te = n - n_tr - n_va
tr, va, te = (list(p.indices) for p in random_split(range(n), [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42)))
RACE = re.compile(r"asian(?:[\s-]+american)?|black|white|caucasian|african(?:[\s-]+american)?|hispanic|latino|latina|latinx|indigenous|native[\s-]+american|race|racial|ethnic\w*", re.I)
M2 = re.compile(r"glaucoma\w*|trabecul\w*|laser|slt|surger\w*|surgical|drops?|eye ?drops?|latanoprost|timolol|brimonidine|dorzolamide|travoprost|bimatoprost|cosopt|combigan|lumigan|xalatan|rhopressa|netarsudil|medication\w*|tube|shunt|iop|pressure|diagnos\w*|suspect\w*|dx|referr\w*|evaluat\w*|screen\w*|follow[- ]?up|rule[sd]? out|r/o", re.I)
SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
def prep(t):
    t = " ".join(s for s in SPLIT.split(str(t)) if s.strip() and not M2.search(s))
    return RACE.sub(" ", t)
T = [prep(a) + " " + prep(b) for a, b in zip(ds["gpt4_summary"], ds["note"])]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000)
bm = LogisticRegression(C=1.0, max_iter=3000).fit(vec.fit_transform([T[i] for i in tr]), y[tr])
tv, tt = bm.predict_proba(vec.transform([T[i] for i in va]))[:, 1], bm.predict_proba(vec.transform([T[i] for i in te]))[:, 1]
B = [np.load(f) for f in ("bert_L2_s0.npz", "bert_L2_s1.npz", "bert_L2_s2.npz") if os.path.exists(f)]
bv, bt = np.mean([d["val"] for d in B], 0), np.mean([d["test"] for d in B], 0)

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))
best_auc, bw = -1, None
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        for w3 in np.arange(0, 1.01 - w1 - w2, 0.05):
            w4 = max(0.0, 1.0 - w1 - w2 - w3)
            a = roc_auc_score(yv, w1*lg(tv) + w2*lg(fv) + w3*lg(bv) + w4*lg(rv))
            if a > best_auc + 1e-9:
                best_auc, bw = a, (round(float(w1), 2), round(float(w2), 2), round(float(w3), 2), round(float(w4), 2))
w1, w2, w3, w4 = bw
pv = sig(w1*lg(tv) + w2*lg(fv) + w3*lg(bv) + w4*lg(rv))
pt = sig(w1*lg(tt) + w2*lg(ft) + w3*lg(bt) + w4*lg(rt))
ty, ta = youden_thr(pv, yv), best_acc_thr(pv, yv)
print(f"\nCLEAN Mask-2 (TF-IDF + ft_img + BERT_L2 + RETFound) weights={bw}:")
print(f"  Val AUC {roc_auc_score(yv, pv):.4f} | Test AUC {roc_auc_score(yt, pt):.4f}")
print(f"  Test Acc@Youden ({ty:.3f}): {((pt>=ty)==yt).mean():.4f} | Test Acc@MaxAcc ({ta:.3f}): {((pt>=ta)==yt).mean():.4f}")
