"""
placebo_audit.py - raw-head variability, placebo null distributions, paired bootstrap, per-group errors.
Standalone: needs only a feature cache with train/val/test -> img, txt, labels, races (the OFFICIAL-split cache).
Nothing here uses the test split for any selection; lambda and INLP depth are FIXED to the values you chose on validation.

Run from the project root:
 python src/placebo_audit.py --cache <your_official_split_cache>.pt
 (quick test first: $env:QUICK="1"; python src/placebo_audit.py --cache <file>.pt )
Outputs: placebo_audit_output.txt and placebo_audit_results.json
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--out", default="placebo_audit")
args = ap.parse_args()

QUICK = os.environ.get("QUICK") == "1"
EPOCHS, BATCH, LR, THR, N_CLS = (4 if QUICK else 40), 256, 1e-3, 0.5, 3
N_SEEDS = 2 if QUICK else 5
N_RAW_SEEDS = 6 if QUICK else 30
N_PLACEBO = int(os.environ.get("N_PLACEBO", 3 if QUICK else 50))
N_BOOT = 100 if QUICK else 1000
LAM_PARITY, LAM_LC, INLP_K = 0.1, 5.0, 2  # chosen on validation in unified_benchmark_final.py
SPLITS = ("train", "val", "test")
torch.set_num_threads(os.cpu_count() or 4)

LOG = open(args.out + "_output.txt", "w", encoding="utf-8")

def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()

d = torch.load(args.cache, weights_only=True)
X = {s: d[s]["img"].float().numpy() for s in SPLITS}
T = {s: d[s]["txt"].float().numpy() for s in SPLITS}
Y = {s: d[s]["labels"].long().numpy() for s in SPLITS}
R = {s: d[s]["races"].long().numpy() for s in SPLITS}
DUAL = {s: np.concatenate([X[s], T[s]], 1).astype(np.float32) for s in SPLITS}

say(f"cache: {args.cache} | QUICK={QUICK} | sizes " + str({s: len(Y[s]) for s in SPLITS}))
for s in SPLITS:
    say(f" {s}: group counts {np.bincount(R[s], minlength=3).tolist()} | glaucoma rate by group "
        + str([round(float(Y[s][R[s] == g].mean()), 3) for g in range(3)]))

# ---------------- erasers (fit on TRAIN only) ----------------
def fit_leace(F, z):
    mu = F.mean(0)
    Fc = F - mu
    Z = np.eye(N_CLS)[z]
    Zc = Z - Z.mean(0)
    n = len(F)
    Sxx = Fc.T @ Fc / n
    Sxz = Fc.T @ Zc / n
    ev, V = np.linalg.eigh(Sxx)
    ev = np.clip(ev, 1e-5 * ev.max(), None)
    W = V @ np.diag(ev ** -0.5) @ V.T
    Winv = V @ np.diag(ev ** 0.5) @ V.T
    U, S, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    U = U[:, S > 1e-8 * S.max()]
    E = Winv @ (U @ U.T) @ W
    return lambda A: (A - (A - mu) @ E.T).astype(np.float32)

def fit_random_subspace(F, rank, rng):
    mu = F.mean(0)
    Q, _ = np.linalg.qr(rng.standard_normal((F.shape[1], rank)))
    return lambda A: (A - (A - mu) @ Q @ Q.T).astype(np.float32)

def fit_inlp(F, z, k):
    dim = F.shape[1]
    P = np.eye(dim)
    dirs = []
    for _ in range(k):
        clf = LogisticRegression(max_iter=3000).fit(F @ P, z)
        _, S, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
        dirs.append(Vt[: int((S > 1e-8 * S.max()).sum())])
        Q, _ = np.linalg.qr(np.concatenate(dirs).T)
        P = np.eye(dim) - Q @ Q.T
    return lambda A: (A @ P).astype(np.float32)

def apply(fn):
    return {s: fn(DUAL[s]) for s in SPLITS}

# ---------------- head + training ----------------
class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 2))

    def forward(self, x):
        return self.net(x)

def group_means(p1, y, r, label):
    out = []
    for g in torch.unique(r):
        m = (r == g) if label is None else ((r == g) & (y == label))
        if m.any():
            out.append(p1[m].mean())
    return out

def penalty(kind, logits, y, r):
    p1 = torch.softmax(logits, 1)[:, 1]
    if kind == "parity":
        m = group_means(p1, y, r, None)
        return torch.stack(m).var() if len(m) > 1 else logits.new_zeros(())
    tot = logits.new_zeros(())
    for lab in (0, 1):
        m = group_means(p1, y, r, lab)
        if len(m) > 1:
            m = torch.stack(m)
            tot = tot + (m.max() - m.min())
    return tot

def train_head(F, y, r, seed, kind=None, lam=0.0, w=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Head(F.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    Ft, yt, rt = torch.from_numpy(F).float(), torch.from_numpy(y).long(), torch.from_numpy(r).long()
    wt = None if w is None else torch.from_numpy(w).float()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Ft))
        for i in range(0, len(Ft), BATCH):
            b = perm[i:i + BATCH]
            lg = m(Ft[b])
            ce = nn.functional.cross_entropy(lg, yt[b], reduction="none")
            loss = (ce * wt[b]).mean() if wt is not None else ce.mean()
            if kind:
                loss = loss + lam * penalty(kind, lg, yt[b], rt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()

def predict(m, F):
    with torch.no_grad():
        return torch.softmax(m(torch.from_numpy(F).float()), 1)[:, 1].numpy()

def run_seeds(F, seeds, **kw):
    return [predict(train_head(F["train"], Y["train"], R["train"], s, **kw), F["test"]) for s in seeds]

# ---------------- metrics ----------------
def metrics(y, r, p, thr=THR):
    t = np.asarray(thr)[r] if np.ndim(thr) else thr
    pred = p > t
    G = np.unique(r)
    pr = [pred[r == g].mean() for g in G]
    tpr = [pred[(r == g) & (y == 1)].mean() for g in G]
    fpr = [pred[(r == g) & (y == 0)].mean() for g in G]
    tg, fg = max(tpr) - min(tpr), max(fpr) - min(fpr)
    return dict(
        acc=float((pred == y).mean()),
        auc=float(roc_auc_score(y, p)),
        sens=float(pred[y == 1].mean()),
        spec=float(1 - pred[y == 0].mean()),
        dpd=float(max(pr) - min(pr)),
        tpr_gap=float(tg),
        fpr_gap=float(fg),
        eod_max=float(max(tg, fg)),
        aod=float((tg + fg) / 2),
        w_auc=float(min(roc_auc_score(y[r == g], p[r == g]) for g in G))
    )

KEYS = ["acc", "auc", "sens", "spec", "dpd", "tpr_gap", "fpr_gap", "eod_max", "aod", "w_auc"]

def agg(ms):
    return {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))) for k in KEYS}

def fmt(a, ks=("auc", "acc", "dpd", "eod_max")):
    return " | ".join(f"{k} {a[k][0]:.4f}±{a[k][1]:.4f}" for k in ks)

def race_probe_auc(F):
    sc = StandardScaler().fit(F["train"])
    clf = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(F["train"]), R["train"])
    return float(roc_auc_score(R["test"], clf.predict_proba(sc.transform(F["test"])), multi_class="ovr", average="macro"))

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return (c - h, c + h)

results = {}
t_start = time.time()

# ================= PART A: raw-head variability =================
say("\n=== PART A: raw dual head, %d seeds (fixed 0.5 cutoff) ===" % N_RAW_SEEDS)
t0 = time.time()
raw30 = run_seeds(DUAL, range(N_RAW_SEEDS))
sec_per_head = (time.time() - t0) / N_RAW_SEEDS
say(f" seconds per trained head: {sec_per_head:.1f}")

est_full_heads = 30 + (50 + 50) * 5 + 6 * 5  # 560 heads
est_total_hours = (est_full_heads * sec_per_head) / 3600.0
say(f" Estimated full runtime: {est_total_hours:.2f} hours")
if est_total_hours > 3.0 and not QUICK:
    say(" >>> ESTIMATED RUNTIME EXCEEDS 3 HOURS. Reducing N_PLACEBO from 50 to 30.")
    N_PLACEBO = 30

ms = [metrics(Y["test"], R["test"], p) for p in raw30]
for k in ("auc", "acc", "dpd", "eod_max"):
    v = [m[k] for m in ms]
    say(f" {k:8s} mean {np.mean(v):.4f} std {np.std(v):.4f} min {np.min(v):.4f} max {np.max(v):.4f}")

say(" disjoint 5-seed ensembles (eod_max, dpd, auc):")
ens_ms = []
for j in range(0, N_RAW_SEEDS - N_SEEDS + 1, N_SEEDS):
    e = metrics(Y["test"], R["test"], np.mean(raw30[j:j + N_SEEDS], 0))
    ens_ms.append(e)
    say(f" seeds {j}-{j + N_SEEDS - 1}: eod_max {e['eod_max']:.4f} dpd {e['dpd']:.4f} auc {e['auc']:.4f}")
results["partA"] = {k: [m[k] for m in ms] for k in KEYS}

# ================= methods (5 seeds each) =================
say("\n=== Training the compared methods (%d seeds each) ===" % N_SEEDS)
seeds = list(range(N_SEEDS))
leace = apply(fit_leace(DUAL["train"], R["train"]))
inlp = apply(fit_inlp(DUAL["train"], R["train"], INLP_K))
cell = R["train"] * 2 + Y["train"]
freq = np.bincount(cell, minlength=6).astype(float)
w_re = (1.0 / freq[cell])
w_re = w_re / w_re.mean()

P = {
    "raw": raw30[:N_SEEDS],
    "leace": run_seeds(leace, seeds),
    "inlp": run_seeds(inlp, seeds),
    "parity": run_seeds(DUAL, seeds, kind="parity", lam=LAM_PARITY),
    "labelcond": run_seeds(DUAL, seeds, kind="labelcond", lam=LAM_LC),
    "reweight": run_seeds(DUAL, seeds, w=w_re)
}
say(" trained: " + ", ".join(P))

# cross-fitted group thresholds (out-of-fold TRAIN predictions)
say(" cross-fitting group thresholds on TRAIN (5-fold)...")
oof = np.zeros(len(Y["train"]))
strat = R["train"] * 2 + Y["train"]
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(DUAL["train"], strat):
    m = train_head(DUAL["train"][a], Y["train"][a], R["train"][a], 0)
    oof[b] = predict(m, DUAL["train"][b])

target = float((oof[Y["train"] == 1] > THR).mean())
thr = np.array([float(np.quantile(oof[(R["train"] == g) & (Y["train"] == 1)], 1 - target)) for g in range(N_CLS)])
say(f" pooled out-of-fold TRAIN TPR at 0.5 = {target:.4f}; group thresholds = {np.round(thr, 3).tolist()}")
P["thresholds"] = P["raw"]
TH = {"thresholds": thr}

def thr_of(name):
    return TH.get(name, THR)

SM = {n: agg([metrics(Y["test"], R["test"], p, thr_of(n)) for p in P[n]]) for n in P}
EN = {n: np.mean(P[n], 0) for n in P}
EM = {n: metrics(Y["test"], R["test"], EN[n], thr_of(n)) for n in P}

say("\n=== Table 1 (test, fixed 0.5 unless thresholds): PER-SEED mean±std | ENSEMBLE value ===")
for n in P:
    say(f" {n:11s} per-seed: {fmt(SM[n])}")
    say(f" {'':11s} ensemble: " + " | ".join(f"{k} {EM[n][k]:.4f}" for k in ("auc", "acc", "dpd", "tpr_gap", "fpr_gap", "eod_max", "w_auc")))
results["table1"] = {"per_seed": SM, "ensemble": EM}

# ================= PART B: placebo null distributions =================
say(f"\n=== PART B: placebo nulls ({N_PLACEBO} each, {N_SEEDS}-seed ensembles, fixed 0.5) ===")
rng = np.random.default_rng(123)
null = {"shuffled_leace": [], "random_subspace": []}
for i in range(N_PLACEBO):
    for name, fn in (("shuffled_leace", lambda: fit_leace(DUAL["train"], rng.permutation(R["train"]))),
                     ("random_subspace", lambda: fit_random_subspace(DUAL["train"], 2, rng))):
        F = apply(fn())
        e = metrics(Y["test"], R["test"], np.mean(run_seeds(F, seeds), 0))
        e["race_probe"] = race_probe_auc(F)
        null[name].append(e)
    if (i + 1) % 5 == 0:
        say(f" placebo {i + 1}/{N_PLACEBO} | {(time.time() - t_start) / 60:.1f} min elapsed")

leace_e = dict(EM["leace"])
leace_e["race_probe"] = race_probe_auc(leace)
raw_probe = race_probe_auc(DUAL)
say(f" raw dual linear race probe macro-AUC {raw_probe:.4f} | LEACE-race {leace_e['race_probe']:.4f}")

for name, lst in null.items():
    say(f" -- {name} (N={len(lst)}) vs LEACE-race --")
    for k in ("auc", "dpd", "eod_max", "race_probe"):
        v = np.array([e[k] for e in lst])
        lv = leace_e[k]
        extra = ""
        if k in ("dpd", "eod_max"):
            extra = f" | one-sided p (placebo <= LEACE) = {(1 + (v <= lv).sum()) / (1 + len(v)):.4f}"
        say(f" {k:10s} null mean {v.mean():.4f} std {v.std():.4f} [2.5%,97.5%] [{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}] "
            f"| LEACE {lv:.4f} | percentile rank {100 * (v <= lv).mean():.1f}%{extra}")
results["partB"] = {"null": null, "leace": leace_e, "raw_probe": raw_probe}

# ================= PART C: paired bootstrap =================
say(f"\n=== PART C: paired bootstrap ({N_BOOT} resamples, same indices for all) ===")
brng = np.random.default_rng(0)
n_te = len(Y["test"])
IDX = [brng.integers(0, n_te, n_te) for _ in range(N_BOOT)]
SEC = ["acc", "dpd", "aod", "tpr_gap", "fpr_gap", "w_auc"]

def boot(a, b):
    out = {k: [] for k in ("auc", "eod_max") + tuple(SEC)}
    for i in IDX:
        y, r = Y["test"][i], R["test"][i]
        try:
            ma = metrics(y, r, EN[a][i], thr_of(a))
            mb = metrics(y, r, EN[b][i], thr_of(b))
        except ValueError:
            continue
        for k in out:
            out[k].append(ma[k] - mb[k])
    return out

results["partC"] = {}
for a, b in [("leace", "raw"), ("leace", "inlp"), ("leace", "thresholds"), ("leace", "parity"),
             ("leace", "labelcond"), ("leace", "reweight"), ("inlp", "raw"), ("parity", "raw"),
             ("labelcond", "raw"), ("reweight", "raw"), ("thresholds", "raw")]:
    o = boot(a, b)
    say(f"\n {a} minus {b}:")
    results["partC"][f"{a}-{b}"] = {}
    for k, v in o.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        blo, bhi = np.percentile(v, [0.4167, 99.5833])
        tag = "PRIMARY" if k in ("auc", "eod_max") else "secondary"
        note = ""
        if k == "auc":
            note = f" | non-inferior (lower 95% bound >= -0.01): {lo >= -0.01}"
        say(f" [{tag:9s}] {k:8s} mean {np.mean(v):+.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}] {'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}"
            + (f" | Bonferroni CI [{blo:+.4f}, {bhi:+.4f}] {'EXCLUDES 0' if blo > 0 or bhi < 0 else 'includes 0'}" if k in SEC else "") + note)
        results["partC"][f"{a}-{b}"][k] = dict(mean=float(np.mean(v)), lo=float(lo), hi=float(hi), blo=float(blo), bhi=float(bhi))

# ================= PART D: per-group clinical errors =================
say("\n=== PART D: per-group errors at the stated thresholds (ensemble predictions) with Wilson 95% CIs ===")
y, r = Y["test"], R["test"]
for n in ("raw", "leace", "inlp", "thresholds"):
    pred = EN[n] > (np.asarray(thr_of(n))[r] if np.ndim(thr_of(n)) else thr_of(n))
    say(f" [{n}]")
    for g, nm in enumerate(("Asian", "Black", "White")):
        m = r == g
        pos, neg = m & (y == 1), m & (y == 0)
        tp, fn_, fp = int(pred[pos].sum()), int((~pred[pos]).sum()), int(pred[neg].sum())
        tl, th = wilson(tp, pos.sum())
        fl, fh = wilson(fn_, pos.sum())
        pl, ph = wilson(fp, neg.sum())
        say(f" {nm:5s} n={int(m.sum()):4d} prev={y[m].mean():.3f} TPR {tp / pos.sum():.3f} [{tl:.3f},{th:.3f}] "
            f"FNR {fn_ / pos.sum():.3f} [{fl:.3f},{fh:.3f}] FPR {fp / neg.sum():.3f} [{pl:.3f},{ph:.3f}] AUC {roc_auc_score(y[m], EN[n][m]):.4f}")

json.dump(results, open(args.out + "_results.json", "w"), indent=2, default=float)
say(f"\nSaved {args.out}_output.txt and {args.out}_results.json | total {(time.time() - t_start) / 60:.1f} min")
LOG.close()