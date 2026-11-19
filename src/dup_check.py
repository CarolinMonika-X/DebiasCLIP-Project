import numpy as np, torch
A = torch.load("cached_features.pt", weights_only=True)
norm = lambda x: torch.nn.functional.normalize(x.float(), dim=1)
tr = norm(A["train"]["img"]); ytr = A["train"]["labels"].long()
for s in ("val", "test"):
    q = norm(A[s]["img"]); yq = A[s]["labels"].long()
    mx, idx = (q @ tr.T).max(1)
    same = (ytr[idx] == yq).float()
    print(f"\n{s}: max cosine similarity to any TRAIN image  median {mx.median():.3f}  90th pct {mx.quantile(0.9):.3f}  99th pct {mx.quantile(0.99):.3f}")
    for thr in (0.99, 0.97, 0.95, 0.90):
        m = mx >= thr
        agree = f"{same[m].mean():.3f}" if m.any() else "n/a"
        print(f"  sim >= {thr}: {int(m.sum()):4d} cases ({m.float().mean():.1%}); label agrees with nearest train case: {agree}")
print("\nIf many cases have very high similarity AND high label agreement, the same eye/visit may appear in train and test.")
