import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

N_CLS, SPLITS = 3, ("train", "val", "test")
d = torch.load("cached_features.pt", weights_only=True)
X = {s: d[s]["img"].float().numpy() for s in SPLITS}
T = {s: d[s]["txt"].float().numpy() for s in SPLITS}
Y = {s: d[s]["labels"].long().numpy() for s in SPLITS}
R = {s: d[s]["races"].long().numpy() for s in SPLITS}

def fit_leace(F, z):
    mu = F.mean(0); Fc = F - mu
    Z = np.eye(N_CLS)[z]; Zc = Z - Z.mean(0); n = len(F)
    Sxx = Fc.T @ Fc / n; Sxz = Fc.T @ Zc / n
    ev, V = np.linalg.eigh(Sxx); ev = np.clip(ev, 1e-5 * ev.max(), None)
    W = V @ np.diag(ev ** -0.5) @ V.T; Winv = V @ np.diag(ev ** 0.5) @ V.T
    U, S, _ = np.linalg.svd(W @ Sxz, full_matrices=False); U = U[:, S > 1e-8 * S.max()]
    E = Winv @ (U @ U.T) @ W
    return lambda A: (A - (A - mu) @ E.T).astype(np.float32)

def score(p, y):
    return dict(macro_auc=float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
                bal_acc=float(balanced_accuracy_score(y, p.argmax(1))))

def probes(F):
    sc = StandardScaler().fit(F["train"]); a, b, c = (sc.transform(F[s]) for s in SPLITS)
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(C=C, max_iter=3000).fit(a, R["train"])
        v = roc_auc_score(R["val"], clf.predict_proba(b), multi_class="ovr", average="macro")
        if best is None or v > best[0]: best = (v, clf, C)
    lin = score(best[1].predict_proba(c), R["test"]); lin["C"] = best[2]
    mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, max_iter=200, random_state=0).fit(a, R["train"])
    nl = score(mlp.predict_proba(c), R["test"])
    gl = LogisticRegression(C=0.1, max_iter=3000).fit(a, Y["train"])
    g_auc = float(roc_auc_score(Y["test"], gl.predict_proba(c)[:, 1]))
    return lin, nl, g_auc

cat = lambda a, b: np.concatenate([a, b], 1).astype(np.float32)
DUAL = {s: cat(X[s], T[s]) for s in SPLITS}
le_img, le_dual = fit_leace(X["train"], R["train"]), fit_leace(DUAL["train"], R["train"])
sets = {"image raw": X, "image LEACE": {s: le_img(X[s]) for s in SPLITS},
        "dual raw": DUAL, "dual LEACE": {s: le_dual(DUAL[s]) for s in SPLITS}}
print("Race decodability on TEST (chance: macro-AUC 0.500, balanced accuracy 0.333) and glaucoma probe AUC\n")
print(f"{'features':14s} | linear probe (macro-AUC / bal-acc) | MLP probe (macro-AUC / bal-acc) | glaucoma AUC")
for nm, F in sets.items():
    lin, nl, g = probes(F)
    print(f"{nm:14s} | {lin['macro_auc']:.4f} / {lin['bal_acc']:.4f} (C={lin['C']}) | {nl['macro_auc']:.4f} / {nl['bal_acc']:.4f} | {g:.4f}")