import numpy as np, torch
from math import sqrt
P = np.load("ft_blend_probs.npz")
A = torch.load("cached_features.pt", weights_only=True)
y = A["test"]["labels"].numpy().astype(int); r = A["test"]["races"].numpy()
names = ["Asian", "Black", "White"]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
for label, p, t in [("Text+image MLP", P["base_test"], float(P["thr_base"])), ("Blend (3-seed ens.)", P["blend_test"], float(P["thr_blend"]))]:
    yh = (p >= t).astype(int)
    print(f"\n{label}  (threshold {t:.2f})")
    for g in range(3):
        kp, npos = int(yh[(r == g) & (y == 1)].sum()), int(((r == g) & (y == 1)).sum())
        kf, nneg = int(yh[(r == g) & (y == 0)].sum()), int(((r == g) & (y == 0)).sum())
        tl, th = wilson(kp, npos); fl, fh = wilson(kf, nneg)
        print(f"  {names[g]:6s} N={npos + nneg:4d} TPR {kp / npos:.3f} [{tl:.3f}, {th:.3f}]  FNR {1 - kp / npos:.3f} [{1 - th:.3f}, {1 - tl:.3f}]  FPR {kf / nneg:.3f} [{fl:.3f}, {fh:.3f}]")
