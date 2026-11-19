import glob, os, re, numpy as np, torch
from datasets import load_dataset
from torch.utils.data import random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import rankdata

A = torch.load("cached_features.pt", weights_only=True)
yv, yt = A["val"]["labels"].numpy().astype(int), A["test"]["labels"].numpy().astype(int)

R = np.load("retfound_tta_probs.npz"); rv, rt = R["val"], R["test"]
F = [np.load(f) for f in ("ft_probs.npz", "ft_probs_s1.npz", "ft_probs_s2.npz") if os.path.exists(f)]
fv, ft = np.mean([d["val"] for d in F], 0), np.mean([d["test"] for d in F], 0)

# Average all distinct BiomedBERT runs on disk
b0_files = sorted(glob.glob("bert_L0*.npz"))
b2_files = sorted(glob.glob("bert_L2*.npz"))
b0v, b0t = np.mean([np.load(f)["val"] for f in b0_files], 0), np.mean([np.load(f)["test"] for f in b0_files], 0)
b2v, b2t = np.mean([np.load(f)["val"] for f in b2_files], 0), np.mean([np.load(f)["test"] for f in b2_files], 0)

sv0, st0 = np.load("stack_val_probs_mask0.npy"), np.load("stack_test_probs_mask0.npy")
sv2, st2 = np.load("stack_val_probs.npy"), np.load("stack_test_probs.npy")

# Also train a stronger char+word TF-IDF model on Mask-2 text (train-only)
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

vec_w = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=80000, sublinear_tf=True)
vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=80000, sublinear_tf=True)
from scipy.sparse import hstack
Xtr = hstack([vec_w.fit_transform([T[i] for i in tr]), vec_c.fit_transform([T[i] for i in tr])])
Xva = hstack([vec_w.transform([T[i] for i in va]), vec_c.transform([T[i] for i in va])])
Xte = hstack([vec_w.transform([T[i] for i in te]), vec_c.transform([T[i] for i in te])])
bm = LogisticRegression(C=1.5, max_iter=3000).fit(Xtr, y[tr])
tv, tt = bm.predict_proba(Xva)[:, 1], bm.predict_proba(Xte)[:, 1]

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))

def maxacc_thr(p, yy):
    thrs = np.linspace(0.25, 0.75, 501)
    accs = [((p >= t) == yy).mean() for t in thrs]
    ba = max(accs)
    return float(min([t for t, a in zip(thrs, accs) if abs(a - ba) < 1e-9], key=lambda t: abs(t - 0.5)))

def fit_calibrated_stack(val_list, test_list, name):
    # Standardize validation log-odds + L2 logistic meta-learner (5-fold CV on val to pick C)
    Xv = np.column_stack([lg(p) for p in val_list])
    Xt = np.column_stack([lg(p) for p in test_list])
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    Xv_s, Xt_s = (Xv - mu) / sd, (Xt - mu) / sd
    best_c, best_cv = 1.0, -1.0
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(yv))
    folds = np.array_split(perm, 5)
    for C in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        oof = np.zeros_like(yv, dtype=float)
        for k in range(5):
            va_i = folds[k]
            tr_i = np.concatenate([folds[j] for j in range(5) if j != k])
            clf = LogisticRegression(C=C).fit(Xv_s[tr_i], yv[tr_i])
            oof[va_i] = clf.predict_proba(Xv_s[va_i])[:, 1]
        auc = roc_auc_score(yv, oof)
        if auc > best_cv:
            best_cv, best_c = auc, C
    meta = LogisticRegression(C=best_c).fit(Xv_s, yv)
    pv = meta.predict_proba(Xv_s)[:, 1]
    pt = meta.predict_proba(Xt_s)[:, 1]
    thr = maxacc_thr(pv, yv)
    print(f"{name:38s} | C={best_c:<4} | Val AUC {roc_auc_score(yv, pv):.4f} | Test AUC {roc_auc_score(yt, pt):.4f} | Test Acc {((pt >= thr) == yt).mean():.4f} (thr={thr:.3f})")
    return pv, pt

fit_calibrated_stack([tv, fv, b2v, rv], [tt, ft, b2t, rt], "Clean Mask-2 (Calibrated Meta)")
fit_calibrated_stack([sv2, tv, fv, b2v, rv], [st2, tt, ft, b2t, rt], "Partially Masked (5-Model Meta)")
fit_calibrated_stack([sv0, fv, b0v, rv], [st0, ft, b0t, rt], "Unmasked Upper Bound (Calibrated Meta)")
