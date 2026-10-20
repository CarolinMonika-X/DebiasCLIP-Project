import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dataset import FairVLMedDataset
from model import DebiasCLIP
import open_clip
from torch.utils.data import DataLoader, random_split

DEVICE = "cpu"
print("Loading CLIP Model to cache features...")
base_model = DebiasCLIP().to(DEVICE)
clip_model = base_model.clip_model
preprocess = base_model.preprocess

def extract_features():
    dataset = FairVLMedDataset(split='train', transform=preprocess)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    def get_features(ds):
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        img_feats, txt_feats, labels, races = [], [], [], []
        with torch.no_grad():
            for images, texts, label, race in loader:
                img = clip_model.encode_image(images)
                txt = clip_model.encode_text(open_clip.tokenize(texts))
                img_feats.append(img / img.norm(dim=-1, keepdim=True))
                txt_feats.append(txt / txt.norm(dim=-1, keepdim=True))
                labels.append(label)
                races.append(race)
        return {
            'img': torch.cat(img_feats), 'txt': torch.cat(txt_feats),
            'labels': torch.cat(labels), 'races': torch.cat(races)
        }
    
    return get_features(train_ds), get_features(val_ds), get_features(test_ds)

data_train, data_val, data_test = extract_features()

# ==========================================================
# STEP 3: LEAKAGE CHECK & MAJORITY BASELINE
# ==========================================================
print("\n--- Step 3: Leakage Check ---")
# 1. Text-only Glaucoma
clf_text = LogisticRegression(max_iter=1000).fit(data_train['txt'].numpy(), data_train['labels'].numpy())
auc_text = roc_auc_score(data_test['labels'].numpy(), clf_text.predict_proba(data_test['txt'].numpy())[:, 1])
print(f"1. Text-only AUC (Glaucoma): {auc_text:.4f}")

# 2. Majority Race Baseline
majority_race = np.bincount(data_train['races'].numpy()).argmax()
majority_acc = (data_test['races'].numpy() == majority_race).mean()
print(f"2. Majority Race Baseline Accuracy: {majority_acc:.4f}")

# 3. Text predicting Race
clf_text_race = LogisticRegression(max_iter=1000).fit(data_train['txt'].numpy(), data_train['races'].numpy())
print(f"3. Text-only Acc (Predicting Race): {clf_text_race.score(data_test['txt'].numpy(), data_test['races'].numpy()):.4f}")

# 4. Image predicting Race
clf_img_race = LogisticRegression(max_iter=1000).fit(data_train['img'].numpy(), data_train['races'].numpy())
print(f"4. Image-only Acc (Predicting Race): {clf_img_race.score(data_test['img'].numpy(), data_test['races'].numpy()):.4f}")

# Standardize features for the neural network
scaler_img = StandardScaler().fit(data_train['img'].numpy())
scaler_txt = StandardScaler().fit(data_train['txt'].numpy())

X_img_tr = torch.tensor(scaler_img.transform(data_train['img'].numpy()), dtype=torch.float32)
X_txt_tr = torch.tensor(scaler_txt.transform(data_train['txt'].numpy()), dtype=torch.float32)
X_img_va = torch.tensor(scaler_img.transform(data_val['img'].numpy()), dtype=torch.float32)
X_txt_va = torch.tensor(scaler_txt.transform(data_val['txt'].numpy()), dtype=torch.float32)
X_img_te = torch.tensor(scaler_img.transform(data_test['img'].numpy()), dtype=torch.float32)
X_txt_te = torch.tensor(scaler_txt.transform(data_test['txt'].numpy()), dtype=torch.float32)

y_tr, r_tr = data_train['labels'], data_train['races']
y_va, r_va = data_val['labels'], data_val['races']
y_te, r_te = data_test['labels'], data_test['races']

# ==========================================================
# EXACT MODEL DEFINITION
# ==========================================================
class ExactAdapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, img_feats, txt_feats=None):
        if txt_feats is not None:
            combined = torch.cat([img_feats, txt_feats], dim=-1)
        else:
            combined = torch.cat([img_feats, torch.zeros_like(img_feats)], dim=-1)
        out = self.net(combined)
        logits = out + self.alpha * img_feats.mean(dim=-1, keepdim=True)
        return logits.squeeze(-1)

def fairness_loss(logits, labels, demographics):
    probs = torch.sigmoid(logits)
    unique_demos = torch.unique(demographics)
    total_loss = 0.0
    for label_val in [0.0, 1.0]:
        group_means = []
        for demo in unique_demos:
            mask = (demographics == demo) & (labels == label_val)
            if mask.sum() > 0:
                group_means.append(probs[mask].mean())
        if len(group_means) > 1:
            means_tensor = torch.stack(group_means)
            total_loss += (means_tensor.max() - means_tensor.min())
    return total_loss

def run_experiment(use_text, use_fairness, lam, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ExactAdapter(1024)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # Higher LR for fast cached training
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(50): # 50 epochs ensures convergence
        model.train()
        optimizer.zero_grad()
        logits = model(X_img_tr, X_txt_tr if use_text else None)
        loss = criterion(logits, y_tr)
        
        if use_fairness and lam > 0:
            loss += lam * fairness_loss(logits, y_tr, r_tr)
            
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_logits = model(X_img_va, X_txt_va if use_text else None)
            val_loss = criterion(val_logits, y_va).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_img_te, X_txt_te if use_text else None)
        preds = torch.sigmoid(logits).numpy()
        
        acc = accuracy_score(y_te.numpy(), (preds > 0.5).astype(int))
        auc = roc_auc_score(y_te.numpy(), preds)
        
        rates = [ (preds[r_te.numpy() == r] > 0.5).mean() for r in np.unique(r_te.numpy()) ]
        dpd = max(rates) - min(rates)
        
        eod = 0.0
        for y_val in [0, 1]:
            tprs = [ (preds[(r_te.numpy() == r) & (y_te.numpy() == y_val)] > 0.5).mean() for r in np.unique(r_te.numpy()) if ((r_te.numpy() == r) & (y_te.numpy() == y_val)).sum() > 0 ]
            if len(tprs) > 1: eod = max(eod, max(tprs) - min(tprs))
                
    return acc, auc, dpd, eod

# ==========================================================
# STEPS 1 & 2: ABLATION (5 SEEDS)
# ==========================================================
print("\n--- Ablation Table (Mean ± Std over 5 seeds) ---")
configs = [
    ("A (Image, No Fair)", False, False, 0),
    ("B (Image, Fair 0.1)", False, True, 0.1),
    ("C (Dual, No Fair)", True, False, 0),
    ("D (Dual, Fair 0.1 - Ours)", True, True, 0.1)
]

for name, use_text, use_fair, lam in configs:
    accs, aucs, dpds, eods = [], [], [], []
    for seed in range(5):
        acc, auc, dpd, eod = run_experiment(use_text, use_fair, lam, seed)
        accs.append(acc); aucs.append(auc); dpds.append(dpd); eods.append(eod)
    
    print(f"{name}:")
    print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  DPD: {np.mean(dpds):.4f} ± {np.std(dpds):.4f}")
    print(f"  EOD: {np.mean(eods):.4f} ± {np.std(eods):.4f}\n")