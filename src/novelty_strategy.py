import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Load cached features
D = torch.load("cached_features.pt")
tr, va, te = D['train'], D['val'], D['test']

# Prepare data (Standardize)
from sklearn.preprocessing import StandardScaler
scaler_img = StandardScaler().fit(tr['img'].numpy())
scaler_txt = StandardScaler().fit(tr['txt'].numpy())

def prep(d):
    xi = torch.tensor(scaler_img.transform(d['img'].numpy()), dtype=torch.float32)
    xt = torch.tensor(scaler_txt.transform(d['txt'].numpy()), dtype=torch.float32)
    return xi, xt, d['labels'], d['races']

Xi_tr, Xt_tr, y_tr, r_tr = prep(tr)
Xi_va, Xt_va, y_va, r_va = prep(va)
Xi_te, Xt_te, y_te, r_te = prep(te)

class Adapter(nn.Module):
    def __init__(self, d=1024, h=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, d))
        self.head = nn.Linear(d, 1)
    def forward(self, x):
        return self.head(self.mlp(x)).squeeze(-1)

def train_and_eval(mode, alpha=1.0):
    model = Adapter()
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(20):
        model.train()
        opt.zero_grad()
        
        # --- THE NOVEL STRATEGY ---
        if mode == 'standard':
            # Original: Concatenate both equally
            X_tr = torch.cat([Xi_tr, Xt_tr], dim=-1)
        elif mode == 'downweight_image':
            # Novel: Reduce image weight, increase text weight
            X_tr = torch.cat([Xi_tr * alpha, Xt_tr], dim=-1)
            
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        opt.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        if mode == 'standard':
            X_te = torch.cat([Xi_te, Xt_te], dim=-1)
        else:
            X_te = torch.cat([Xi_te * alpha, Xt_te], dim=-1)
            
        preds = torch.sigmoid(model(X_te)).numpy()
        
    # Metrics
    acc = accuracy_score(y_te.numpy(), (preds > 0.5).astype(int))
    auc = roc_auc_score(y_te.numpy(), preds)
    
    # DPD
    rates = [ (preds[r_te.numpy() == r] > 0.5).mean() for r in np.unique(r_te.numpy()) ]
    dpd = max(rates) - min(rates)
    
    # EOD
    eod = 0.0
    for y_val in [0, 1]:
        tprs = [ (preds[(r_te.numpy() == r) & (y_te.numpy() == y_val)] > 0.5).mean() for r in np.unique(r_te.numpy()) if ((r_te.numpy() == r) & (y_te.numpy() == y_val)).sum() > 0 ]
        if len(tprs) > 1: eod = max(eod, max(tprs) - min(tprs))
        
    return acc, auc, dpd, eod

print("--- Standard Model (Baseline) ---")
acc, auc, dpd, eod = train_and_eval('standard')
print(f"Acc: {acc:.4f}, AUC: {auc:.4f}, DPD: {dpd:.4f}, EOD: {eod:.4f}")

print("\n--- Novel Strategy: Down-weight Image by 0.5 ---")
acc, auc, dpd, eod = train_and_eval('downweight_image', alpha=0.5)
print(f"Acc: {acc:.4f}, AUC: {auc:.4f}, DPD: {dpd:.4f}, EOD: {eod:.4f}")

print("\n--- Novel Strategy: Down-weight Image by 0.1 ---")
acc, auc, dpd, eod = train_and_eval('downweight_image', alpha=0.1)
print(f"Acc: {acc:.4f}, AUC: {auc:.4f}, DPD: {dpd:.4f}, EOD: {eod:.4f}")