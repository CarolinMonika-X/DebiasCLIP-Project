import os, sys, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split
sys.path.insert(0, "src")
from dataset import FairVLMedDataset

MAXSIDE = 336
G = {"no": 0, "yes": 1}; R = {"asian": 0, "black": 1, "white": 2}
ds = FairVLMedDataset(split="train", strict=True)
assert len(ds) == 10000
parts = random_split(range(10000), [7000, 1000, 2000], generator=torch.Generator().manual_seed(42))
cache = torch.load("cached_features.pt", weights_only=True)
out = Path(os.environ.get("PACK_DIR","colab_pack"))
meta = {"files": {}, "labels": {}, "races": {}}
for s, p in zip(("train", "val", "test"), parts):
    (out / s).mkdir(parents=True, exist_ok=True)
    files, labels, races = [], [], []
    for j, idx in enumerate(p.indices):
        item = ds.ds[idx]
        im = ds._load_image(item["filename"]).convert("L")
        w, h = im.size; sc = MAXSIDE / max(w, h)
        if sc < 1:
            im = im.resize((round(w * sc), round(h * sc)), Image.Resampling.BICUBIC)
        rel = f"{s}/{j:05d}.png"; im.save(out / rel)
        files.append(rel)
        labels.append(G[str(item["glaucoma"]).lower()]); races.append(R[str(item["race"]).lower()])
        if j % 1000 == 0: print(s, j, flush=True)
    labels, races = torch.tensor(labels), torch.tensor(races)
    assert torch.equal(labels, cache[s]["labels"].long()) and torch.equal(races, cache[s]["races"].long()), "ALIGNMENT FAILED " + s
    meta["files"][s] = files; meta["labels"][s] = labels; meta["races"][s] = races
    print(s, "aligned with cached_features.pt")
torch.save(meta, out / "meta.pt")
print("done")

