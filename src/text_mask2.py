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
SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")

def drop_sentences(text, pat):
    keep = [s for s in SPLIT.split(str(text)) if s.strip() and not pat.search(s)]
    return " ".join(keep)

fields = {"gpt4_summary": [str(t) for t in ds["gpt4_summary"]], "note": [str(t) for t in ds["note"]]}
fields["both"] = [a + " " + b for a, b in zip(fields["gpt4_summary"], fields["note"])]
VARIANTS = {"none (race words removed only)": None,
            "M1 sentences dropped": re.compile(M1, re.I),
            "M2 sentences dropped": re.compile(M2, re.I)}

def run(tname, vname, pat):
    T = [RACE.sub(" ", drop_sentences(t, pat) if pat else t) for t in fields[tname]]
    words = np.array([len(t.split()) for t in T])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000)
    Xtr = vec.fit_transform([T[i] for i in tr]); Xva = vec.transform([T[i] for i in va]); Xte = vec.transform([T[i] for i in te])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
    pv, pt = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    f, t_, th = roc_curve(y[va], pv); thr = float(th[int(np.argmax(t_ - f))])
    print(f"{tname:13s} {vname:32s} median words kept {np.median(words):4.0f} empty {np.mean(words == 0):5.1%} | "
          f"val AUC {roc_auc_score(y[va], pv):.4f} | test AUC {roc_auc_score(y[te], pt):.4f} acc@valYouden {((pt >= thr) == y[te]).mean():.4f}")
    if pat is not None and pat.pattern == M2 and tname == "both":
        names = np.array(vec.get_feature_names_out()); c = clf.coef_[0]; o = np.argsort(c)
        print("   top POSITIVE:", ", ".join(f"{names[i]} ({c[i]:.1f})" for i in o[::-1][:30]))
        print("   top NEGATIVE:", ", ".join(f"{names[i]} ({c[i]:.1f})" for i in o[:30]))

print("Whole sentences containing flagged terms are DELETED (no residue). All variants reported; none chosen on test.\n")
for tname in fields:
    for vname, pat in VARIANTS.items():
        run(tname, vname, pat)
    print()
