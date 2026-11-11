"""
notes_check.py - do the clinical notes mention race, and what can a text-only model decode?
Run from the project root: python src/notes_check.py
Writes notes_check_output.txt. Uses the OFFICIAL split (assigned by the folder each image file sits in).
Regex matches need a human read: words like 'white' or 'black' can have non-race meanings.
"""
import os
import re
import random
import collections
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(".", "data", "FairVLMed", "Dataset", "dataset")
FOLDERS = {"Training": "train", "Validation": "val", "Test": "test"}
PAT = re.compile(r"\b(asian|black|white|african|caucasian|hispanic|latino|latina|race|racial|ethnic\w*)\b", re.I)
LOG = open("notes_check_output.txt", "w", encoding="utf-8")

def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()

ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
split = []
for fn in ds["filename"]:
    hit = [v for k, v in FOLDERS.items() if os.path.exists(os.path.join(ROOT, k, fn))]
    split.append(hit[0] if hit else "missing")
split = np.array(split)
race = np.array([{"asian": 0, "black": 1, "white": 2}[str(v).lower()] for v in ds["race"]])
gl = np.array([1 if str(v).lower() == "yes" else 0 for v in ds["glaucoma"]])
say("official split sizes:", {s: int((split == s).sum()) for s in ("train", "val", "test", "missing")})

cols = {c: [str(t) for t in ds[c]] for c in ("gpt4_summary", "note") if c in ds.column_names}
random.seed(0)
for c, texts in cols.items():
    has = np.array([bool(PAT.search(t)) for t in texts])
    say(f"\n=== column '{c}': notes containing a race/ethnicity word ===")
    for s in ("train", "val", "test"):
        m = split == s
        say(f" {s:5s}: {int(has[m].sum())}/{int(m.sum())} = {100 * has[m].mean():.1f}%")
    for g, nm in enumerate(("Asian", "Black", "White")):
        m = race == g
        say(f" {nm:5s}: {int(has[m].sum())}/{int(m.sum())} = {100 * has[m].mean():.1f}%")
    idx = [i for i in np.where(has)[0]]
    say(" example matched sentences (10 random):")
    for i in random.sample(idx, min(10, len(idx))):
        for sent in re.split(r"(?<=[.!?])\s+", texts[i]):
            if PAT.search(sent):
                say(" -", sent[:200].replace("\n", " "))
                break

t = cols["gpt4_summary"]
has_g = np.array(["glaucoma" in x.lower() for x in t])
say("\n=== the word 'glaucoma' in gpt4_summary by true label ===")
for lab in (0, 1):
    m = gl == lab
    say(f" label {lab}: {int(has_g[m].sum())}/{int(m.sum())} notes contain 'glaucoma' ({100 * has_g[m].mean():.1f}%)")

tr, te = split == "train", split == "test"
vec = TfidfVectorizer(max_features=5000).fit([t[i] for i in np.where(tr)[0]])
Xtr, Xte = vec.transform([t[i] for i in np.where(tr)[0]]), vec.transform([t[i] for i in np.where(te)[0]])
rc = LogisticRegression(max_iter=2000).fit(Xtr, race[tr])
gc = LogisticRegression(max_iter=2000).fit(Xtr, gl[tr])
say("\n=== TF-IDF (top 5000 unigrams) + logistic regression on gpt4_summary, TEST ===")
say(f" race macro-AUC {roc_auc_score(race[te], rc.predict_proba(Xte), multi_class='ovr', average='macro'):.4f} (chance 0.5)")
say(f" glaucoma AUC {roc_auc_score(gl[te], gc.predict_proba(Xte)[:, 1]):.4f}")
say("\nSaved notes_check_output.txt")
LOG.close()