import os, torch, numpy as np
from collections import Counter
from datasets import load_dataset
ROOT = os.path.join("data","FairVLMed","Dataset","dataset")
F = {"train":"Training","val":"Validation","test":"Test"}
G = {"no":0,"yes":1}; R = {"asian":0,"black":1,"white":2}
ds = load_dataset("harvardairobotics/FairVLMed","default",split="train")
fns = ds["filename"]; use = [str(u).lower() for u in ds["use"]]
y = np.array([G[str(v).lower()] for v in ds["glaucoma"]])
r = np.array([R[str(v).lower()] for v in ds["race"]])
c = torch.load("cached_features.pt", weights_only=True)
print("use column values:", Counter(use))
where = {}
for s, f in F.items():
    for fn in os.listdir(os.path.join(ROOT, f)):
        if fn.endswith(".npz"):
            where[fn] = s
print("filenames unique:", len(set(fns)) == len(fns), "| all found in folders:", all(fn in where for fn in fns))
print("first 3 filenames:", list(fns[:3]))
print("use column vs folder:", Counter((use[i], where.get(fns[i])) for i in range(len(fns))))
for s in F:
    cy, cr = c[s]["labels"].numpy(), c[s]["races"].numpy()
    ids = [i for i in range(len(fns)) if where.get(fns[i]) == s]
    print("\n==", s)
    print("cached: n =", len(cy), "prev =", round(float(cy.mean()), 4), "race counts =", np.bincount(cr, minlength=3).tolist())
    print("folder: n =", len(ids), "prev =", round(float(y[ids].mean()), 4), "race counts =", np.bincount(r[ids], minlength=3).tolist())
    for nm, order in {"dataset_order": ids, "sorted_name": sorted(ids, key=lambda i: fns[i])}.items():
        eq = (cy == y[order]) & (cr == r[order])
        print("  ", nm, "position-match fraction:", round(float(eq.mean()), 4))
    print("   cached first 12 (label,race):", list(zip(cy[:12].tolist(), cr[:12].tolist())))
    print("   folder first 12 (label,race):", list(zip(y[ids][:12].tolist(), r[ids][:12].tolist())))
