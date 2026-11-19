import re, numpy as np, torch
from datasets import load_dataset
from torch.utils.data import random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
y = np.array([{"no": 0, "yes": 1}[str(v).lower()] for v in ds["glaucoma"]])
n = len(ds); n_tr, n_va = int(0.7 * n), int(0.1 * n); n_te = n - n_tr - n_va
parts = random_split(range(n), [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42))
tr, va = list(parts[0].indices), list(parts[1].indices)

RACE = re.compile(r"asian(?:[\s-]+american)?|black|white|caucasian|african(?:[\s-]+american)?|hispanic|latino|latina|latinx|indigenous|native[\s-]+american|race|racial|ethnic\w*", re.I)
FLAG = re.compile(r"glaucoma\w*|trabecul\w*|laser|slt|surger\w*|surgical|drops?|eye ?drops?|latanoprost|timolol|brimonidine|dorzolamide|travoprost|bimatoprost|cosopt|combigan|lumigan|xalatan|medication\w*|tube|shunt|iop|pressure", re.I)
clean = lambda s: RACE.sub(" ", str(s))
gs = [clean(t) for t in ds["gpt4_summary"]]
nt = [clean(t) for t in ds["note"]]
both = [a + " " + b for a, b in zip(gs, nt)]

for nm, T in (("gpt4_summary", gs), ("note", nt)):
    w = np.array([len(t.split()) for t in T])
    print(f"{nm}: median {np.median(w):.0f} words, 90th pct {np.percentile(w, 90):.0f}, share over 60 words {(w > 60).mean():.1%}")

def run(texts, label, drop=False, first_words=None, show=False):
    T = [" ".join(t.split()[:first_words]) if first_words else t for t in texts]
    if drop:
        T = [FLAG.sub(" ", t) for t in T]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000)
    Xtr = vec.fit_transform([T[i] for i in tr]); Xva = vec.transform([T[i] for i in va])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
    auc = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
    print(f"{label:58s} VAL AUC {auc:.4f}")
    if show:
        names = np.array(vec.get_feature_names_out()); c = clf.coef_[0]; o = np.argsort(c)
        print("   top POSITIVE:", ", ".join(f"{names[i]} ({c[i]:.1f})" for i in o[::-1][:40]))
        print("   top NEGATIVE:", ", ".join(f"{names[i]} ({c[i]:.1f})" for i in o[:40]))
        print()

run(gs, "gpt4_summary only")
run(nt, "note only")
run(both, "both fields (as in the stack)", show=True)
run(both, "both, first 60 words only (about the 77-token CLIP limit)", first_words=60)
run(both, "both, diagnosis/treatment terms removed", drop=True, show=True)
run(both, "both, first 60 words AND diagnosis/treatment terms removed", drop=True, first_words=60)
