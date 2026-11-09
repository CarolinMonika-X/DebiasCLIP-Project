import re
import numpy as np
import torch
import open_clip
from datasets import load_dataset
from open_clip.tokenizer import SimpleTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split

SEED, MAXTOK = 42, 70
ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
notes = [str(t) for t in ds["gpt4_summary"]]
n = len(notes); n_tr, n_va = int(0.7 * n), int(0.1 * n); n_te = n - n_tr - n_va
parts = random_split(range(n), [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(SEED))

tok = SimpleTokenizer()
ntok = lambda s: len(tok.encode(s))

def split_long(s):
    if ntok(s) <= MAXTOK: return [s]
    pieces, cur = [], []
    for w in s.split():
        cur.append(w)
        if ntok(" ".join(cur)) > MAXTOK:
            if len(cur) > 1: pieces.append(" ".join(cur[:-1])); cur = [w]
    if cur: pieces.append(" ".join(cur))
    return pieces

def chunk(note):
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", note.strip()):
        for piece in split_long(sent):
            cand = (cur + " " + piece).strip()
            if cur and ntok(cand) > MAXTOK: out.append(cur); cur = piece
            else: cur = cand
    if cur: out.append(cur)
    return out or [""]

chunks, owner = [], []
for i, note in enumerate(notes):
    for c in chunk(note): chunks.append(c); owner.append(i)
lens = np.array([ntok(s) for s in notes])
print(f"{n} notes | tokens per note: median {np.median(lens):.0f}, max {lens.max()} | notes over 75 tokens "
      f"(truncated before): {(lens > 75).mean():.1%} | chunks per note: {len(chunks) / n:.2f}")

model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.eval(); embs = []
with torch.no_grad():
    for s in range(0, len(chunks), 256):
        t = model.encode_text(open_clip.tokenize(chunks[s:s + 256]))
        embs.append(t / t.norm(dim=-1, keepdim=True))
        if (s // 256) % 20 == 0: print(f"  encoded {s}/{len(chunks)} chunks")
embs = torch.cat(embs)
note_emb = torch.zeros(n, embs.shape[1]).index_add_(0, torch.tensor(owner), embs)
note_emb = note_emb / note_emb.norm(dim=-1, keepdim=True)

D = torch.load("cached_features.pt")
y_all = np.array([1.0 if str(v).lower() == "yes" else 0.0 for v in ds["glaucoma"]])
r_all = np.array([{"asian": 0, "black": 1, "white": 2}[str(v).lower()] for v in ds["race"]])
out = {}
for name, part in zip(("train", "val", "test"), parts):
    idx = list(part.indices)
    assert np.array_equal(y_all[idx], D[name]["labels"].numpy()) and np.array_equal(r_all[idx], D[name]["races"].numpy()), \
        f"alignment mismatch in {name}: cached_features.pt was built from a different split"
    out[name] = note_emb[idx]
torch.save(out, "cached_text_chunked.pt")
print("saved cached_text_chunked.pt (alignment check passed)")

for label, get in (("truncated (current)", lambda k: D[k]["txt"]), ("chunked (new)", lambda k: out[k])):
    sc = StandardScaler().fit(get("train").numpy())
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(get("train").numpy()), D["train"]["labels"].numpy())
    auc = roc_auc_score(D["test"]["labels"].numpy(), clf.predict_proba(sc.transform(get("test").numpy()))[:, 1])
    print(f"text-only test AUC, {label}: {auc:.4f}")