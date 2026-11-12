import torch
from torch.utils.data import DataLoader, random_split
from dataset import FairVLMedDataset
from model import DebiasCLIP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import open_clip
import torch.nn as nn
import torch.optim as optim

DEVICE = "cpu"
print("Loading CLIP Model (Frozen)...")
clip_model = DebiasCLIP().to(DEVICE).clip_model
preprocess = DebiasCLIP().preprocess

def extract_features(split_name):
    print(f"Extracting features for {split_name}...")
    dataset = FairVLMedDataset(split='train', transform=preprocess)
    
    # 70/10/20 split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    if split_name == 'train': ds = train_ds
    elif split_name == 'val': ds = val_ds
    else: ds = test_ds

    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    img_feats, txt_feats, labels, races = [], [], [], []
    
    with torch.no_grad():
        for images, texts, label, race in loader:
            img = clip_model.encode_image(images)
            txt = clip_model.encode_text(open_clip.tokenize(texts))
            img = img / img.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            
            img_feats.append(img)
            txt_feats.append(txt)
            labels.append(label)
            races.append(race)

    return {
        'img': torch.cat(img_feats).numpy(),
        'txt': torch.cat(txt_feats).numpy(),
        'labels': torch.cat(labels).numpy(),
        'races': torch.cat(races).numpy()
    }

# 1. Extract and Save Features
data_train = extract_features('train')
data_test = extract_features('test')
print("Features extracted successfully!")

# ==========================================================
# STEP 2: TEXT LEAKAGE CHECK
# ==========================================================
print("\n--- Step 2: Leakage Check (Logistic Regression) ---")

# Text Only
clf_text = LogisticRegression(max_iter=1000).fit(data_train['txt'], data_train['labels'])
auc_text = roc_auc_score(data_test['labels'], clf_text.predict_proba(data_test['txt'])[:, 1])
print(f"Text-Only Baseline AUC: {auc_text:.4f}")

# Image Only
clf_img = LogisticRegression(max_iter=1000).fit(data_train['img'], data_train['labels'])
auc_img = roc_auc_score(data_test['labels'], clf_img.predict_proba(data_test['img'])[:, 1])
print(f"Image-Only Baseline AUC: {auc_img:.4f}")

# ==========================================================
# STEP 3: FAST ABLATION TABLE
# ==========================================================
print("\n--- Step 3: Ablation Table (Fast Adapter Training) ---")

class FastAdapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_adapter(X_train, y_train, races_train, X_test, y_test, races_test, use_fairness):
    input_dim = X_train.shape[1]
    model = FastAdapter(input_dim)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    races_train_t = torch.tensor(races_train)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    # Train for 100 epochs
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        
        if use_fairness:
            probs = torch.sigmoid(logits)
            unique_races = torch.unique(races_train_t)
            means = [probs[races_train_t == r].mean() for r in unique_races if (races_train_t == r).sum() > 0]
            if len(means) > 1:
                loss += 0.1 * torch.var(torch.stack(means))
        
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_test_t)).numpy()
        acc = accuracy_score(y_test, (preds > 0.5).astype(int))
        auc = roc_auc_score(y_test, preds)
        
        # Calculate DPD on test set
        rates = [ (preds[races_test == r] > 0.5).mean() for r in np.unique(races_test) ]
        dpd = max(rates) - min(rates)
        
    return acc, auc, dpd

# Prepare Data
X_img_train, X_txt_train = data_train['img'], data_train['txt']
X_dual_train = np.concatenate([X_img_train, X_txt_train], axis=1)

X_img_test, X_txt_test = data_test['img'], data_test['txt']
X_dual_test = np.concatenate([X_img_test, X_txt_test], axis=1)

# A: Image, No Fairness
acc, auc, dpd = train_adapter(X_img_train, data_train['labels'], data_train['races'], X_img_test, data_test['labels'], data_test['races'], use_fairness=False)
print(f"A (Image, No Fairness): Acc={acc:.4f}, AUC={auc:.4f}, DPD={dpd:.4f}")

# B: Image, Fairness
acc, auc, dpd = train_adapter(X_img_train, data_train['labels'], data_train['races'], X_img_test, data_test['labels'], data_test['races'], use_fairness=True)
print(f"B (Image, Fairness): Acc={acc:.4f}, AUC={auc:.4f}, DPD={dpd:.4f}")

# C: Dual, No Fairness
acc, auc, dpd = train_adapter(X_dual_train, data_train['labels'], data_train['races'], X_dual_test, data_test['labels'], data_test['races'], use_fairness=False)
print(f"C (Dual, No Fairness): Acc={acc:.4f}, AUC={auc:.4f}, DPD={dpd:.4f}")

# D: Dual, Fairness (Your Model)
acc, auc, dpd = train_adapter(X_dual_train, data_train['labels'], data_train['races'], X_dual_test, data_test['labels'], data_test['races'], use_fairness=True)
print(f"D (Dual, Fairness - Ours): Acc={acc:.4f}, AUC={auc:.4f}, DPD={dpd:.4f}")