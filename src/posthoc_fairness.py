import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from dataset import FairVLMedDataset
from model import DebiasCLIP
import open_clip
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler

DEVICE = "cpu"
print("Loading cached features...")
if torch.cuda.is_available(): DEVICE = "cuda"

# 1. Load Cached Features (To save time, we use the cached ones)
# If you don't have cached_features.pt, run the extraction block from ablation_full.py first.
D = torch.load("cached_features.pt")
tr, va, te = D['train'], D['val'], D['test']

# 2. Standardize
scaler_img = StandardScaler().fit(tr['img'].numpy())
scaler_txt = StandardScaler().fit(tr['txt'].numpy())

def prep(d):
    xi = torch.tensor(scaler_img.transform(d['img'].numpy()), dtype=torch.float32)
    xt = torch.tensor(scaler_txt.transform(d['txt'].numpy()), dtype=torch.float32)
    return xi, xt, d['labels'], d['races']

Xi_tr, Xt_tr, y_tr, r_tr = prep(tr)
Xi_va, Xt_va, y_va, r_va = prep(va)
Xi_te, Xt_te, y_te, r_te = prep(te)

# 3. Define the Model (Exact Adapter from your main V3)
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

def train_model():
    model = Adapter(1024)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    bce = nn.BCEWithLogitsLoss()
    loader = DataLoader(torch.utils.data.TensorDataset(Xi_tr, Xt_tr, y_tr, r_tr), batch_size=256, shuffle=True)
    
    Xva = torch.cat([Xi_va, Xt_va], dim=-1)
    best_loss, best_state = float('inf'), None
    
    for epoch in range(40):
        model.train()
        for xi, xt, y, r in loader:
            opt.zero_grad()
            logits = model(torch.cat([xi, xt], dim=-1))
            loss = bce(logits, y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = bce(model(Xva), y_va).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            
    model.load_state_dict(best_state)
    return model

print("Training model for post-hoc analysis...")
model = train_model()
model.eval()

with torch.no_grad():
    p_va = torch.sigmoid(model(torch.cat([Xi_va, Xt_va], dim=-1))).numpy()
    p_te = torch.sigmoid(model(torch.cat([Xi_te, Xt_te], dim=-1))).numpy()

yva, rva = y_va.numpy(), r_va.numpy()
yte, rte = y_te.numpy(), r_te.numpy()

# ==========================================================
# POST-HOC THRESHOLD OPTIMIZATION
# ==========================================================
print("\n--- Finding Optimal Thresholds on Validation Set ---")
overall_tpr = (p_va[yva == 1] > 0.5).mean()
thresholds = {g: 0.5 for g in np.unique(rva)} # Default

for g in np.unique(rva):
    mask_pos = (rva == g) & (yva == 1)
    if mask_pos.sum() > 0:
        probs = p_va[mask_pos]
        best_t, best_diff = 0.5, 1.0
        for t in np.arange(0.05, 0.95, 0.05):
            tpr = (probs > t).mean()
            diff = abs(tpr - overall_tpr)
            if diff < best_diff:
                best_diff, best_t = diff, t
        thresholds[g] = best_t
        print(f"Group {g}: Best Threshold = {best_t:.2f}")

# ==========================================================
# EVALUATE ON TEST SET (Before vs After)
# ==========================================================
def calc_metrics(p, y, r, custom_thresholds=None):
    if custom_thresholds:
        pred = np.zeros_like(p, dtype=int)
        for g in np.unique(r):
            mask = r == g
            t = custom_thresholds.get(g, 0.5)
            pred[mask] = (p[mask] > t).astype(int)
    else:
        pred = (p > 0.5).astype(int)
        
    # DPD
    rates = [pred[r == g].mean() for g in np.unique(r)]
    dpd = max(rates) - min(rates)
    
    # EOD
    eod = 0.0
    for lab in [0, 1]:
        tprs = [pred[(r == g) & (y == lab)].mean() for g in np.unique(r) if ((r == g) & (y == lab)).sum() > 0]
        if len(tprs) > 1:
            eod = max(eod, max(tprs) - min(tprs))
            
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, p)
    return acc, auc, dpd, eod

print("\n--- RESULTS ---")
acc_std, auc_std, dpd_std, eod_std = calc_metrics(p_te, yte, rte)
print(f"Standard (Threshold 0.5)  : Acc={acc_std:.4f}, AUC={auc_std:.4f}, DPD={dpd_std:.4f}, EOD={eod_std:.4f}")

acc_opt, auc_opt, dpd_opt, eod_opt = calc_metrics(p_te, yte, rte, thresholds)
print(f"Post-Hoc (Group Thresholds): Acc={acc_opt:.4f}, AUC={auc_opt:.4f}, DPD={dpd_opt:.4f}, EOD={eod_opt:.4f}")