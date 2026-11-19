import json, torch
from math import sqrt
res = json.load(open("ft_blend_results.json"))
A = torch.load("cached_features.pt", weights_only=True)
y, r = A["test"]["labels"].numpy().astype(int), A["test"]["races"].numpy()
names = ["Asian", "Black", "White"]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
for key, v in res.items():
    print("\n", key, "(validation-chosen threshold", v["thr"], ")")
    for g in range(3):
        npos = int(((r == g) & (y == 1)).sum()); nneg = int(((r == g) & (y == 0)).sum())
        tpr, fpr = v["tpr_by_race"][g], v["fpr_by_race"][g]
        lo, hi = wilson(round(tpr * npos), npos); flo, fhi = wilson(round(fpr * nneg), nneg)
        print(f"  {names[g]:6s} n={npos + nneg:4d}  TPR {tpr:.3f} [{lo:.3f},{hi:.3f}]  FNR {1 - tpr:.3f} [{1 - hi:.3f},{1 - lo:.3f}]  FPR {fpr:.3f} [{flo:.3f},{fhi:.3f}]")
