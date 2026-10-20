"""
ablation_full.py  -  single pipeline for ALL numbers in the paper.
 
Rows:
  A  image only, no fairness loss
  B  image only, parity loss (variance of group mean predictions)
  C  image+text, no fairness loss
  D  image+text, parity loss
  E  image+text, label-conditioned worst-group loss
 
Outputs: group facts, leakage check, A-E ablation (5 seeds), lambda sweep
(chosen on VALIDATION), per-group breakdown, bootstrap CIs
(C-A, D-C, E-C, and the validation-chosen-lambda versions), JSON dump.
 
Run from project root:
  python src/ablation_full.py | Tee-Object -FilePath final_output.txt
"""
import os, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
 
DEVICE = "cpu"
CACHE = "cached_features.pt"
EPOCHS = 40
BATCH = 256
LR = 1e-3
SEEDS = list(range(5))
LAMBDAS = [0, 0.1, 1, 5, 10]
THR = 0.5
N_BOOT = 1000
 
# ----------------------------------------------------------
# 1. FEATURE CACHING (CLIP is frozen, so do this once)
# ----------------------------------------------------------
def extract_and_cache():
    import open_clip
    from dataset import FairVLMedDataset
    from model import DebiasCLIP
 
    print("Loading CLIP to cache features (one-time)...")
    base = DebiasCLIP().to(DEVICE)
    clip_model, preprocess = base.clip_model, base.preprocess
    clip_model.eval()
 
    dataset = FairVLMedDataset(split='train', transform=preprocess)
    n = len(dataset)
    n_tr, n_va = int(0.7 * n), int(0.1 * n)
    n_te = n - n_tr - n_va
    splits = random_split(dataset, [n_tr, n_va, n_te],
                          generator=torch.Generator().manual_seed(42))
 
    def feats(ds):
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        I, T, Y, R = [], [], [], []
        with torch.no_grad():
            for images, texts, label, race in loader:
                i = clip_model.encode_image(images)
                t = clip_model.encode_text(open_clip.tokenize(list(texts)))
                I.append(i / i.norm(dim=-1, keepdim=True))
                T.append(t / t.norm(dim=-1, keepdim=True))
                Y.append(label); R.append(race)
        return {'img': torch.cat(I), 'txt': torch.cat(T),
                'labels': torch.cat(Y).float(), 'races': torch.cat(R).long()}
 
    out = {'train': feats(splits[0]), 'val': feats(splits[1]), 'test': feats(splits[2])}
    torch.save(out, CACHE)
    return out
 
if os.path.exists(CACHE):
    print(f"Loading cached features from {CACHE}")
    D = torch.load(CACHE)
else:
    D = extract_and_cache()
tr, va, te = D['train'], D['val'], D['test']
 
# ----------------------------------------------------------
# 2. DATA FACTS: group counts and prevalence
# ----------------------------------------------------------
print("\n=== Group counts and glaucoma prevalence ===")
for name, d in [('train', tr), ('val', va), ('test', te)]:
    r, y = d['races'].numpy(), d['labels'].numpy()
    print(f"[{name}] n={len(y)}")
    for g in np.unique(r):
        print(f"   group {g}: n={int((r==g).sum())}  ({(r==g).mean():.3f})  glaucoma rate={y[r==g].mean():.3f}")
maj = np.bincount(te['races'].numpy()).max() / len(te['races'])
print(f"Majority-race accuracy on test: {maj:.4f}")
 
# ----------------------------------------------------------
# 3. LEAKAGE CHECK
# ----------------------------------------------------------
print("\n=== Leakage / signal check (logistic regression on frozen features) ===")
scalers = {k: StandardScaler().fit(tr[k].numpy()) for k in ['img', 'txt']}
leak = {}
for name, key in [("Image", 'img'), ("Text", 'txt')]:
    Xtr = scalers[key].transform(tr[key].numpy())
    Xte = scalers[key].transform(te[key].numpy())
    cy = LogisticRegression(max_iter=3000, C=0.1).fit(Xtr, tr['labels'].numpy())
    cr = LogisticRegression(max_iter=3000, C=0.1).fit(Xtr, tr['races'].numpy())
    auc_y = roc_auc_score(te['labels'].numpy(), cy.predict_proba(Xte)[:, 1])
    pr = cr.predict_proba(Xte)
    auc_r = roc_auc_score(te['races'].numpy(), pr, multi_class='ovr', average='macro')
    bal = balanced_accuracy_score(te['races'].numpy(), pr.argmax(1))
    acc = accuracy_score(te['races'].numpy(), pr.argmax(1))
    leak[name] = {'glaucoma_auc': auc_y, 'race_acc': acc, 'race_macro_auc': auc_r, 'race_bal_acc': bal}
    print(f"{name}-only | glaucoma AUC {auc_y:.4f} | race acc {acc:.4f} "
          f"(majority {maj:.4f}) | race macro-AUC {auc_r:.4f} | race bal-acc {bal:.4f}")
 
# ----------------------------------------------------------
# 4. STANDARDIZED TENSORS
# ----------------------------------------------------------
def prep(d):
    return (torch.tensor(scalers['img'].transform(d['img'].numpy()), dtype=torch.float32),
            torch.tensor(scalers['txt'].transform(d['txt'].numpy()), dtype=torch.float32),
            d['labels'], d['races'])
 
Xi_tr, Xt_tr, y_tr, r_tr = prep(tr)
Xi_va, Xt_va, y_va, r_va = prep(va)
Xi_te, Xt_te, y_te, r_te = prep(te)
 
# ----------------------------------------------------------
# 5. MODEL: residual MLP adapter + linear head
# ----------------------------------------------------------
class Adapter(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                                 nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, in_dim))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.head = nn.Linear(in_dim, 1)
 
    def forward(self, x):
        x = x + self.alpha * self.mlp(x)
        return self.head(x).squeeze(-1)
 
def make_input(xi, xt, use_text):
    return torch.cat([xi, xt], dim=-1) if use_text else xi
 
# ----------------------------------------------------------
# 6. FAIRNESS LOSSES
# ----------------------------------------------------------
def parity_loss(logits, y, g):
    p = torch.sigmoid(logits)
    means = [p[g == k].mean() for k in torch.unique(g)]
    if len(means) < 2:
        return logits.new_zeros(())
    return torch.stack(means).var()
 
def label_cond_loss(logits, y, g):
    p = torch.sigmoid(logits)
    total = logits.new_zeros(())
    for lab in (0.0, 1.0):
        means = []
        for k in torch.unique(g):
            m = (g == k) & (y == lab)
            if m.sum() > 0:
                means.append(p[m].mean())
        if len(means) > 1:
            m = torch.stack(means)
            total = total + (m.max() - m.min())
    return total
 
LOSSES = {'none': None, 'parity': parity_loss, 'labelcond': label_cond_loss}
 
# ----------------------------------------------------------
# 7. METRICS
# ----------------------------------------------------------
def metrics(p, y, r, thr=THR):
    pred = (p > thr).astype(int)
    gs = np.unique(r)
    rates = [pred[r == g].mean() for g in gs]
    dpd = max(rates) - min(rates)
    eod = 0.0
    for lab in (0, 1):
        v = [pred[(r == g) & (y == lab)].mean() for g in gs if ((r == g) & (y == lab)).sum() > 0]
        if len(v) > 1:
            eod = max(eod, max(v) - min(v))
    aucs, accs = [], []
    for g in gs:
        m = r == g
        accs.append(accuracy_score(y[m], pred[m]))
        if len(np.unique(y[m])) > 1:
            aucs.append(roc_auc_score(y[m], p[m]))
    return {'acc': accuracy_score(y, pred), 'auc': roc_auc_score(y, p),
            'dpd': dpd, 'eod': eod,
            'w_auc': min(aucs) if aucs else float('nan'), 'w_acc': min(accs)}
 
# ----------------------------------------------------------
# 8. TRAINING
# ----------------------------------------------------------
def run(use_text, loss_name, lam, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    in_dim = 1024 if use_text else 512
    model = Adapter(in_dim)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    bce = nn.BCEWithLogitsLoss()
    fair = LOSSES[loss_name]
    loader = DataLoader(TensorDataset(Xi_tr, Xt_tr, y_tr, r_tr),
                        batch_size=BATCH, shuffle=True)
    Xva = make_input(Xi_va, Xt_va, use_text)
    Xte = make_input(Xi_te, Xt_te, use_text)
    best, state = float('inf'), None
    for _ in range(EPOCHS):
        model.train()
        for xi, xt, y, r in loader:
            opt.zero_grad()
            lg = model(make_input(xi, xt, use_text))
            loss = bce(lg, y)
            if fair is not None and lam > 0:
                loss = loss + lam * fair(lg, y, r)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = bce(model(Xva), y_va).item()
        if vl < best:
            best, state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(Xva)).numpy()
        p_te = torch.sigmoid(model(Xte)).numpy()
    n_params = sum(p.numel() for p in model.parameters())
    return p_va, p_te, n_params
 
def run_seeds(use_text, loss_name, lam):
    pv, pt = [], []
    for s in SEEDS:
        a, b, n = run(use_text, loss_name, lam, s)
        pv.append(a); pt.append(b)
    return pv, pt, n
 
def summarize(preds, y, r):
    ms = [metrics(p, y, r) for p in preds]
    return {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))) for k in ms[0]}
 
yte, rte = y_te.numpy(), r_te.numpy()
yva, rva = y_va.numpy(), r_va.numpy()
 
# ----------------------------------------------------------
# 9. MAIN ABLATION A-E (lambda = 0.1)
# ----------------------------------------------------------
print("\n=== Ablation (mean ± std over 5 seeds, threshold 0.5, TEST set) ===")
configs = [("A  Image, none", False, 'none', 0),
           ("B  Image, parity 0.1", False, 'parity', 0.1),
           ("C  Dual,  none", True, 'none', 0),
           ("D  Dual,  parity 0.1", True, 'parity', 0.1),
           ("E  Dual,  label-cond 0.1", True, 'labelcond', 0.1)]
A_N, C_N = "A  Image, none", "C  Dual,  none"
D_N, E_N = "D  Dual,  parity 0.1", "E  Dual,  label-cond 0.1"
results, ens = {}, {}
for name, ut, ln, lam in configs:
    pv, pt, n_params = run_seeds(ut, ln, lam)
    s = summarize(pt, yte, rte)
    results[name] = s
    ens[name] = np.mean(pt, axis=0)
    print(f"{name}  [trainable params: {n_params:,}]")
    print("   " + " | ".join(f"{k} {v[0]:.4f}±{v[1]:.4f}" for k, v in s.items()))
 
# ----------------------------------------------------------
# 10. LAMBDA SWEEP (choose on VALIDATION)
# ----------------------------------------------------------
print("\n=== Lambda sweep (dual-modal) ===")
chosen, sweep_ens, sweep_log = {}, {}, {}
for loss_name in ['parity', 'labelcond']:
    rows = []
    for lam in LAMBDAS:
        ln = loss_name if lam > 0 else 'none'
        pv, pt, _ = run_seeds(True, ln, lam)
        mv = summarize(pv, yva, rva)
        mt = summarize(pt, yte, rte)
        rows.append((lam, mv, mt))
        sweep_ens[(loss_name, lam)] = np.mean(pt, axis=0)
        sweep_log[f"{loss_name}_lam{lam}"] = {'val': mv, 'test': mt}
        print(f"[{loss_name}] lam={lam:<4} VAL auc {mv['auc'][0]:.4f} dpd {mv['dpd'][0]:.4f} eod {mv['eod'][0]:.4f}"
              f" || TEST auc {mt['auc'][0]:.4f} dpd {mt['dpd'][0]:.4f} eod {mt['eod'][0]:.4f}"
              f" w_auc {mt['w_auc'][0]:.4f}")
    base_auc = rows[0][1]['auc'][0]
    ok = [r for r in rows if r[1]['auc'][0] >= base_auc - 0.01]
    best = min(ok, key=lambda r: r[1]['eod'][0])
    chosen[loss_name] = best[0]
    print(f"--> {loss_name}: lambda chosen on validation = {best[0]}")
 
# ----------------------------------------------------------
# 11. BOOTSTRAP HELPERS
# ----------------------------------------------------------
def boot_diff(p1, p2, seed=0):
    """metrics(p1) - metrics(p2), resampling test patients."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ['auc', 'dpd', 'eod', 'w_auc']}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(yte), len(yte))
        try:
            m1 = metrics(p1[idx], yte[idx], rte[idx])
            m2 = metrics(p2[idx], yte[idx], rte[idx])
        except ValueError:
            continue
        for k in out:
            out[k].append(m1[k] - m2[k])
    res = {}
    for k, v in out.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        res[k] = (float(np.nanmean(v)), float(lo), float(hi))
    return res
 
def show_boot(title, res):
    print(title)
    for k, (m, lo, hi) in res.items():
        verdict = "significant" if (lo > 0 or hi < 0) else "NOT significant (CI includes 0)"
        print(f"   {k:6s} mean {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")
 
print("\n=== Bootstrap 95% CI of differences (test-set resampling, 1000x, seed-ensembled preds) ===")
boots = {}
comparisons = [("C minus A  (effect of adding text)", ens[C_N], ens[A_N]),
               ("D(0.1) minus C  (parity loss)", ens[D_N], ens[C_N]),
               ("E(0.1) minus C  (label-conditioned loss)", ens[E_N], ens[C_N])]
if chosen['parity'] > 0:
    comparisons.append((f"D(chosen lam={chosen['parity']}) minus C",
                        sweep_ens[('parity', chosen['parity'])], ens[C_N]))
if chosen['labelcond'] > 0:
    comparisons.append((f"E(chosen lam={chosen['labelcond']}) minus C",
                        sweep_ens[('labelcond', chosen['labelcond'])], ens[C_N]))
for title, p1, p2 in comparisons:
    r = boot_diff(p1, p2)
    boots[title] = r
    show_boot(title + ":", r)
 
# ----------------------------------------------------------
# 12. PER-GROUP BREAKDOWN (seed-ensembled, threshold 0.5)
# ----------------------------------------------------------
print("\n=== Per-group breakdown on TEST (seed-ensembled predictions) ===")
prev = [yte[rte == g].mean() for g in np.unique(rte)]
print(f"Prevalence gap across groups (= DPD of a perfect classifier): {max(prev)-min(prev):.3f}")
 
def group_table(name, p):
    pred = p > THR
    print(name)
    for g in np.unique(rte):
        m = rte == g
        pos, neg = m & (yte == 1), m & (yte == 0)
        auc = roc_auc_score(yte[m], p[m]) if len(np.unique(yte[m])) > 1 else float('nan')
        print(f"  group {g}: n={int(m.sum()):4d} prev={yte[m].mean():.3f} pos-rate={pred[m].mean():.3f} "
              f"TPR={pred[pos].mean():.3f} FPR={pred[neg].mean():.3f} AUC={auc:.3f}")
 
group_table("A  Image, none", ens[A_N])
group_table("C  Dual, none", ens[C_N])
if chosen['labelcond'] > 0:
    group_table(f"E  Dual, label-cond (lam={chosen['labelcond']})",
                sweep_ens[('labelcond', chosen['labelcond'])])
 
# ----------------------------------------------------------
# 13. SAVE
# ----------------------------------------------------------
with open("ablation_results.json", "w") as f:
    json.dump({'leakage': leak, 'ablation': results, 'lambda_sweep': sweep_log,
               'chosen_lambda': chosen, 'bootstrap': boots}, f, indent=2, default=float)
print("\nSaved ablation_results.json. Done.")
 