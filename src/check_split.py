import os, torch
from datasets import load_dataset
root = os.path.join("data", "FairVLMed", "Dataset", "dataset")
for f in ("Training", "Validation", "Test"):
    p = os.path.join(root, f)
    n = len([x for x in os.listdir(p) if x.endswith(".npz")]) if os.path.isdir(p) else "folder missing"
    print(f, "npz files:", n)
print("all_images files:", len(os.listdir("all_images")))
ds = load_dataset("harvardairobotics/FairVLMed", "default")
print(ds)
d = torch.load("cached_features.pt", weights_only=True)
print(type(d))
if isinstance(d, dict):
    for k, v in d.items():
        print(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
