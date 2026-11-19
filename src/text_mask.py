import re, numpy as np, torch
from datasets import load_dataset
from torch.utils.data import random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
y = np.array([{"no": 0, "yes": 1}[str(v).lower()] for v in ds["glaucoma"]])
n = len(ds); n_tr, n_va = int(0.7 * n), int(0.1 * n); n_te = n - n_tr - n_va
parts = random_split(range(n), [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42))
tr, va, te = (list(p.indices) for p in parts)

RACE = re.compile(r"asian(?:[\s-]+american)?|black|white|caucasian|african(?:[\s-]+american)?|hispanic|latino|latina|latinx|"
                  r"indigenous|native[\s-]+american|race|racial|ethnic\w*", re.I)
M1 = (r"glaucoma\w*|trabecul\w*|laser|slt|surger\w*|surgical|drops?|eye ?drops?|latanoprost|timolol|brimonidine|dorzolamide|"
      r"travoprost|bimatoprost|cosopt|combigan|lumigan|xalatan|rhopressa|netarsudil|medication\w*|tube|shunt|iop|pressure")
M2 = M1 + r"|diagnos\w*|suspect\w*|dx|referr\w*|evaluat\w*|screen\w*|follow[- ]?up|rule[sd]? out|r/o"
MASKS = {"none": None, "M1 (glaucoma/treatment terms)": re.compile(M1, re.I),
         "M2 (M1 + diagnosis/suspect/evaluation terms)": re.compile(M2, re.I)}
clean = lambda s: RACE.sub(" ", str(s))
gs = [clean(t) for t in ds["gpt4_summary"]]; nt = [clean(t) for t in ds["note"]]
both = [a + " " + b for a, b in zip(gs, nt)]

def run(texts, tname, mname, mask, show=False):
    T = [mask.sub(" ", t) if mask else t for t in texts]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000)
    Xtr = vec.fit_transform([T[i] for i in tr]); Xva = vec.transform([T[i] for i in va]); Xte = vec.transform([T[i] for i in te])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
    pv, pt = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    f, t_, th = roc_curve(y[va], pv); thr = float(th[int(np.argmax(t_ - f))])
    print(f"{tname:13s} {mname:46s} val AUC {roc_auc_score(y[va], pv):.4f} | test AUC {roc_auc_score(y[te], pt):.4f} "
          f"acc@valYouden {((pt >= thr) == y[te]).mean():.4f}")
    if show:
        names = np.array(vec.get_feature_names_out()); c = clf.coef_[0]; o = np.argsort(c)
        print("   top POSITIVE:", ", ".join(f"{names[i]} ({c[i]:.1f})" for i in o[::-1][:40]))
        print("   top NEGATIVE:", ", ".join(f"{names[i]} ({c[i]:.1f})" for i in o[:40]))

print("All variants are reported (none selected on test); threshold from validation Youden.\n")
for tname, T in (("gpt4_summary", gs), ("note", nt), ("both", both)):
    for mname, mask in MASKS.items():
        run(T, tname, mname, mask, show=(tname == "both" and mname.startswith("M2")))
    print()
