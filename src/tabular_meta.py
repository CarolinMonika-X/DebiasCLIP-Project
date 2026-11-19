import glob, os, re, numpy as np, pandas as pd, torch
from datasets import load_dataset
from torch.utils.data import random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

A = torch.load("cached_features.pt", weights_only=True)
yv, yt = A["val"]["labels"].numpy().astype(int), A["test"]["labels"].numpy().astype(int)

ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
print("Dataset columns:", ds.column_names)
df = ds.to_pandas()
y = df["glaucoma"].astype(str).str.lower().map({"no": 0, "yes": 1}).values
n = len(df); n_tr, n_va = int(0.7 * n), int(0.1 * n); n_te = n - n_tr - n_va
tr, va, te = (list(p.indices) for p in random_split(range(n), [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42)))

# Non-race clinical metadata columns (exclude id, filename, race, glaucoma, text)
exclude = {"filename", "race", "ethnicity", "glaucoma", "gpt4_summary", "note", "slo_fundus"}
num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() < 30]
print("Using non-race tabular features -> numeric:", num_cols, "| categorical:", cat_cols)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
tab_clf = Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.5, max_iter=2000))]).fit(df.iloc[tr], y[tr])
tab_v, tab_t = tab_clf.predict_proba(df.iloc[va])[:, 1], tab_clf.predict_proba(df.iloc[te])[:, 1]
print(f"Tabular-only (non-race clinical metadata) | Val AUC {roc_auc_score(yv, tab_v):.4f} | Test AUC {roc_auc_score(yt, tab_t):.4f}")

# Load existing predictors
R = np.load("retfound_tta_probs.npz"); rv, rt = R["val"], R["test"]
F = [np.load(f) for f in ("ft_probs.npz", "ft_probs_s1.npz", "ft_probs_s2.npz") if os.path.exists(f)]
fv, ft = np.mean([d["val"] for d in F], 0), np.mean([d["test"] for d in F], 0)
b0_files = sorted(glob.glob("bert_L0_s?.npz"))
b2_files = sorted(glob.glob("bert_L2_s?.npz"))
b0v, b0t = np.mean([np.load(f)["val"] for f in b0_files], 0), np.mean([np.load(f)["test"] for f in b0_files], 0)
b2v, b2t = np.mean([np.load(f)["val"] for f in b2_files], 0), np.mean([np.load(f)["test"] for f in b2_files], 0)
sv0, st0 = np.load("stack_val_probs_mask0.npy"), np.load("stack_test_probs_mask0.npy")
sv2, st2 = np.load("stack_val_probs.npy"), np.load("stack_test_probs.npy")

RACE = re.compile(r"asian(?:[\s-]+american)?|black|white|caucasian|african(?:[\s-]+american)?|hispanic|latino|latina|latinx|indigenous|native[\s-]+american|race|racial|ethnic\w*", re.I)
M2 = re.compile(r"glaucoma\w*|trabecul\w*|laser|slt|surger\w*|surgical|drops?|eye ?drops?|latanoprost|timolol|brimonidine|dorzolamide|travoprost|bimatoprost|cosopt|combigan|lumigan|xalatan|rhopressa|netarsudil|medication\w*|tube|shunt|iop|pressure|diagnos\w*|suspect\w*|dx|referr\w*|evaluat\w*|screen\w*|follow[- ]?up|rule[sd]? out|r/o", re.I)
SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
def prep(t):
    t = " ".join(s for s in SPLIT.split(str(t)) if s.strip() and not M2.search(s))
    return RACE.sub(" ", t)
T = [prep(a) + " " + prep(b) for a, b in zip(df["gpt4_summary"], df["note"])]
vec_w = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=80000, sublinear_tf=True)
vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=80000, sublinear_tf=True)
bm = LogisticRegression(C=1.5, max_iter=3000).fit(hstack([vec_w.fit_transform([T[i] for i in tr]), vec_c.fit_transform([T[i] for i in tr])]), y[tr])
tv = bm.predict_proba(hstack([vec_w.transform([T[i] for i in va]), vec_c.transform([T[i] for i in va])]))[:, 1]
tt = bm.predict_proba(hstack([vec_w.transform([T[i] for i in te]), vec_c.transform([T[i] for i in te])]))[:, 1]

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def maxacc_thr(p, yy):
    thrs = np.linspace(0.25, 0.75, 501); accs = [((p >= t) == yy).mean() for t in thrs]; ba = max(accs)
    return float(min([t for t, a in zip(thrs, accs) if abs(a - ba) < 1e-9], key=lambda t: abs(t - 0.5)))

def fit_meta(val_list, test_list, name):
    Xv = np.column_stack([lg(p) for p in val_list])
    Xt = np.column_stack([lg(p) for p in test_list])
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    Xv_s, Xt_s = (Xv - mu) / sd, (Xt - mu) / sd
    best_c, best_cv = 0.1, -1.0
    folds = np.array_split(np.random.default_rng(42).permutation(len(yv)), 5)
    for C in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        oof = np.zeros_like(yv, dtype=float)
        for k in range(5):
            va_i = folds[k]; tr_i = np.concatenate([folds[j] for j in range(5) if j != k])
            oof[va_i] = LogisticRegression(C=C).fit(Xv_s[tr_i], yv[tr_i]).predict_proba(Xv_s[va_i])[:, 1]
        if roc_auc_score(yv, oof) > best_cv:
            best_cv, best_c = roc_auc_score(yv, oof), C
    meta = LogisticRegression(C=best_c).fit(Xv_s, yv)
    pv, pt = meta.predict_proba(Xv_s)[:, 1], meta.predict_proba(Xt_s)[:, 1]
    thr = maxacc_thr(pv, yv)
    print(f"{name:40s} | C={best_c:<4} | Val AUC {roc_auc_score(yv, pv):.4f} | Test AUC {roc_auc_score(yt, pt):.4f} | Test Acc {((pt >= thr) == yt).mean():.4f} (thr={thr:.3f})")

fit_meta([tab_v, tv, fv, b2v, rv], [tab_t, tt, ft, b2t, rt], "Clean Mask-2 + Tabular Metadata")
fit_meta([tab_v, sv2, tv, fv, b2v, rv], [tab_t, st2, tt, ft, b2t, rt], "Partially Masked + Tabular Metadata")
fit_meta([tab_v, sv0, fv, b0v, rv], [tab_t, st0, ft, b0t, rt], "Unmasked + Tabular Metadata")
