import glob, os, re, numpy as np, torch
from math import sqrt
from datasets import load_dataset
from torch.utils.data import random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

A = torch.load("cached_features.pt", weights_only=True)
yv, yt = A["val"]["labels"].numpy().astype(int), A["test"]["labels"].numpy().astype(int)
rt = A["test"]["races"].numpy()

R = np.load("retfound_tta_probs.npz"); rv, rt_p = R["val"], R["test"]
F = [np.load(f) for f in ("ft_probs.npz", "ft_probs_s1.npz", "ft_probs_s2.npz") if os.path.exists(f)]
fv, ft_p = np.mean([d["val"] for d in F], 0), np.mean([d["test"] for d in F], 0)
B0_files = sorted(glob.glob("bert_L0_s?.npz"))
B2_files = sorted(glob.glob("bert_L2_s?.npz"))
b0v, b0t = np.mean([np.load(f)["val"] for f in B0_files], 0), np.mean([np.load(f)["test"] for f in B0_files], 0)
has_b2 = len(B2_files) > 0
if has_b2:
    b2v, b2t = np.mean([np.load(f)["val"] for f in B2_files], 0), np.mean([np.load(f)["test"] for f in B2_files], 0)

sv0, st0 = np.load("stack_val_probs_mask0.npy"), np.load("stack_test_probs_mask0.npy")
sv2, st2 = np.load("stack_val_probs.npy"), np.load("stack_test_probs.npy")

ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
y = np.array([{"no": 0, "yes": 1}[str(v).lower()] for v in ds["glaucoma"]])
n = len(ds); n_tr, n_va = int(0.7 * n), int(0.1 * n); n_te = n - n_tr - n_va
tr, va, te = (list(p.indices) for p in random_split(range(n), [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42)))
RACE = re.compile(r"asian(?:[\s-]+american)?|black|white|caucasian|african(?:[\s-]+american)?|hispanic|latino|latina|latinx|indigenous|native[\s-]+american|race|racial|ethnic\w*", re.I)
M2 = re.compile(r"glaucoma\w*|trabecul\w*|laser|slt|surger\w*|surgical|drops?|eye ?drops?|latanoprost|timolol|brimonidine|dorzolamide|travoprost|bimatoprost|cosopt|combigan|lumigan|xalatan|rhopressa|netarsudil|medication\w*|tube|shunt|iop|pressure|diagnos\w*|suspect\w*|dx|referr\w*|evaluat\w*|screen\w*|follow[- ]?up|rule[sd]? out|r/o", re.I)
SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
def prep(t):
    t = " ".join(s for s in SPLIT.split(str(t)) if s.strip() and not M2.search(s))
    return RACE.sub(" ", t)
T = [prep(a) + " " + prep(b) for a, b in zip(ds["gpt4_summary"], ds["note"])]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000)
bm = LogisticRegression(C=1.0, max_iter=3000).fit(vec.fit_transform([T[i] for i in tr]), y[tr])
tv, tt = bm.predict_proba(vec.transform([T[i] for i in va]))[:, 1], bm.predict_proba(vec.transform([T[i] for i in te]))[:, 1]

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))
def youden_thr(p, yy):
    f, t, th = roc_curve(yy, p); return float(th[int(np.argmax(t - f))])
def maxacc_thr(p, yy):
    thrs = np.linspace(0.25, 0.75, 501); accs = [((p >= t) == yy).mean() for t in thrs]; ba = max(accs)
    return float(min([t for t, a in zip(thrs, accs) if abs(a - ba) < 1e-9], key=lambda t: abs(t - 0.5)))
def opt4(p1v, p2v, p3v, p4v, p1t, p2t, p3t, p4t, use3=True):
    best, bw = -1, None
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            for w3 in (np.arange(0, 1.01 - w1 - w2, 0.05) if use3 else [0.0]):
                w4 = max(0.0, 1.0 - w1 - w2 - w3)
                a = roc_auc_score(yv, w1*lg(p1v) + w2*lg(p2v) + w3*lg(p3v) + w4*lg(p4v))
                if a > best + 1e-9: best, bw = a, (round(float(w1),2), round(float(w2),2), round(float(w3),2), round(float(w4),2))
    w1, w2, w3, w4 = bw
    return bw, sig(w1*lg(p1v)+w2*lg(p2v)+w3*lg(p3v)+w4*lg(p4v)), sig(w1*lg(p1t)+w2*lg(p2t)+w3*lg(p3t)+w4*lg(p4t))

z = np.full_like(tv, 0.5)
w_clean, pv_clean, pt_clean = opt4(tv, fv, b2v if has_b2 else z, rv, tt, ft_p, b2t if has_b2 else z, rt_p, use3=has_b2)
w_part, pv_part, pt_part = opt4(sv2, fv, b2v if has_b2 else z, rv, st2, ft_p, b2t if has_b2 else z, rt_p, use3=has_b2)
w_full, pv_full, pt_full = opt4(sv0, fv, b0v, rv, st0, ft_p, b0t, rt_p, use3=True)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*sqrt(p*(1-p)/n + z*z/(4*n*n)); return (c-h)/d, (c+h)/d

lines = [f"# Final Upgraded Evaluation (BERT_L2 found: {has_b2}, files={B2_files})\n",
         "| Tier | Weights | Val AUC | Test AUC | Test Acc (Youden) | Test Acc (MaxAcc) |",
         "|---|---|---:|---:|---:|---:|"]
tiers = [
    ("Image-only: ft_image (3 seeds)", "-", fv, ft_p),
    ("Image-only: RETFound (336px TTA)", "-", rv, rt_p),
    ("Clean Mask-2 (TF-IDF+ft+BERT_L2+RETFound)", str(w_clean), pv_clean, pt_clean),
    ("Partially masked (Stack+ft+BERT_L2+RETFound)", str(w_part), pv_part, pt_part),
    ("Unmasked upper bound (Stack0+ft+BERT_L0+RETFound)", str(w_full), pv_full, pt_full),
]
for name, w, pv, pt in tiers:
    ty, ta = youden_thr(pv, yv), maxacc_thr(pv, yv)
    lines.append(f"| {name} | {w} | {roc_auc_score(yv,pv):.4f} | {roc_auc_score(yt,pt):.4f} | {((pt>=ty)==yt).mean():.4f} ({ty:.3f}) | {((pt>=ta)==yt).mean():.4f} ({ta:.3f}) |")

base = np.load("baseline_probs.npz"); bv_base, bt_base = base["val"], base["test"]
rng = np.random.default_rng(0)
def boot_delta(pa_v, pa_t, pb_v, pb_t, label):
    ta, tb = maxacc_thr(pa_v, yv), maxacc_thr(pb_v, yv)
    da, du = [], []
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt))
        da.append(((pa_t[i]>=ta)==yt[i]).mean() - ((pb_t[i]>=tb)==yt[i]).mean())
        du.append(roc_auc_score(yt[i], pa_t[i]) - roc_auc_score(yt[i], pb_t[i]))
    lines.append(f"- **{label}**: dAcc = {np.mean(da):+.4f} [{np.percentile(da,2.5):+.4f}, {np.percentile(da,97.5):+.4f}], dAUC = {np.mean(du):+.4f} [{np.percentile(du,2.5):+.4f}, {np.percentile(du,97.5):+.4f}]")

lines.append("\n## Paired Test Bootstrap Deltas (1,000 resamples, 95% CI)\n")
boot_delta(rv, rt_p, fv, ft_p, "RETFound minus ft_image (image-only)")
boot_delta(pv_clean, pt_clean, rv, rt_p, "Clean Mask-2 multimodal minus RETFound")
boot_delta(pv_full, pt_full, bv_base, bt_base, "Unmasked 4-way blend minus OpenCLIP MLP baseline")

for title, pv, pt in (("Clean Mask-2 Multimodal", pv_clean, pt_clean), ("Unmasked Upper Bound", pv_full, pt_full)):
    t = maxacc_thr(pv, yv); yh = (pt >= t).astype(int)
    lines.append(f"\n## Per-Race Test Rates: {title} (thr={t:.3f}, Wilson 95% CI)\n")
    lines.append("| Race | N | TPR (95% CI) | FNR | FPR (95% CI) |")
    lines.append("|---|---:|---|---|---|")
    for g, nm in enumerate(["Asian", "Black", "White"]):
        kp, npos = int(yh[(rt==g)&(yt==1)].sum()), int(((rt==g)&(yt==1)).sum())
        kf, nneg = int(yh[(rt==g)&(yt==0)].sum()), int(((rt==g)&(yt==0)).sum())
        a1, a2 = wilson(kp, npos); b1, b2 = wilson(kf, nneg)
        lines.append(f"| {nm} | {int((rt==g).sum())} | {kp/npos:.3f} [{a1:.3f}, {a2:.3f}] | {1-kp/npos:.3f} | {kf/nneg:.3f} [{b1:.3f}, {b2:.3f}] |")

report = "\n".join(lines)
open("final_report_retfound.md", "w", encoding="utf-8").write(report)
print(report)

np.savez("final_probs.npz", tv=tv, tt=tt, fv=fv, ft=ft_p, b2v=b2v, b2t=b2t, b0v=b0v, b0t=b0t, rv=rv, rt=rt_p, sv0=sv0, st0=st0, sv2=sv2, st2=st2, pv_clean=pv_clean, pt_clean=pt_clean, pv_full=pv_full, pt_full=pt_full)
