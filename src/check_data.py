import os, collections
import numpy as np
from datasets import load_dataset

ROOT = os.path.join(".", "data", "FairVLMed", "Dataset", "dataset")
FOLDERS = ["Training", "Validation", "Test"]
KEYWORDS = ["glaucoma", "optic", "cup", "iop", "pressure"]
names = {}

for split in ["train", "validation", "test"]:
    try:
        ds = load_dataset("harvardairobotics/FairVLMed", "default", split=split)
    except Exception as e:
        print(f"\n=== HF split '{split}': cannot load ({repr(e)[:100]})"); continue
    print(f"\n=== HF split '{split}': {len(ds)} rows | columns: {ds.column_names}")
    for col in ("glaucoma", "race"):
        print(f"  raw {col} values:", collections.Counter(str(v).lower() for v in ds[col]).most_common())
    fns = list(ds["filename"]); names[split] = set(fns)
    where = collections.Counter()
    for fn in fns:
        found = [f for f in FOLDERS if os.path.exists(os.path.join(ROOT, f, fn))]
        where[tuple(found) if found else ("MISSING",)] += 1
    print("  image file found in folder(s):", dict(where))
    bad, shapes = 0, collections.Counter()
    for fn in fns[:300]:
        p = next((os.path.join(ROOT, f, fn) for f in FOLDERS if os.path.exists(os.path.join(ROOT, f, fn))), None)
        try:
            z = np.load(p); k = "slo_fundus" if "slo_fundus" in z.files else z.files[0]
            a = z[k]; shapes[(k, str(a.dtype), a.shape)] += 1
        except Exception:
            bad += 1
    print(f"  first 300 files: {bad} fail to load; (key, dtype, shape):", dict(shapes))
    if split == "train":
        txt = [str(t).lower() for t in ds["gpt4_summary"]]; y = np.array([str(v).lower() == "yes" for v in ds["glaucoma"]])
        for w in KEYWORDS:
            m = np.array([w in t for t in txt])
            if m.any(): print(f"  note contains '{w}': {m.mean():.3f} of notes; glaucoma rate when present {y[m].mean():.3f}, absent {y[~m].mean():.3f}")

keys = list(names)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        print(f"\nfilenames shared between '{keys[i]}' and '{keys[j]}': {len(names[keys[i]] & names[keys[j]])}")