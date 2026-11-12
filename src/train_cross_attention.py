"""
src/train_cross_attention.py
Train a Cross-Attention Multimodal Transformer with Manifold MixUp & Label Smoothing.
Learns dynamic interaction between image and clinical text features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

print("=" * 80)
print("TRAINING MULTIMODAL CROSS-ATTENTION FUSION HEAD (WITH MIXUP)")
print("=" * 80)

# 1. Load Data
d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

# Normalize modalities
X_img = {s: F.normalize(d[s]["img"].float(), p=2, dim=-1) for s in SPLITS}
X_txt = {s: F.normalize(d[s]["txt"].float(), p=2, dim=-1) for s in SPLITS}
Y = {s: d[s]["labels"].long() for s in SPLITS}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 2. Cross-Attention Architecture
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        self.img_proj = nn.Linear(dim, dim)
        self.txt_proj = nn.Linear(dim, dim)
        
        # Bi-directional Cross Attention
        self.cross_img2txt = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cross_txt2img = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.ln_img = nn.LayerNorm(dim)
        self.ln_txt = nn.LayerNorm(dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, img_feat, txt_feat):
        # Shape: (B, 1, 512)
        q_img = self.img_proj(img_feat).unsqueeze(1)
        q_txt = self.txt_proj(txt_feat).unsqueeze(1)
        
        # Cross attend
        attn_img, _ = self.cross_img2txt(q_img, q_txt, q_txt)
        attn_txt, _ = self.cross_txt2img(q_txt, q_img, q_img)
        
        out_img = self.ln_img((q_img + attn_img).squeeze(1))
        out_txt = self.ln_txt((q_txt + attn_txt).squeeze(1))
        
        fused = torch.cat([out_img, out_txt], dim=-1)
        return self.classifier(fused)

# 3. Training Loop with MixUp and Cosine Decay
def mixup_data(x1, x2, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x1.size(0)
    index = torch.randperm(batch_size)
    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

test_probs_all, val_probs_all = [], []
epochs = 45

for seed in range(5):
    torch.manual_seed(seed * 2026)
    model = CrossAttentionFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    n_train = len(X_img["train"])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 128):
            idx = perm[i:i+128]
            img_b = X_img["train"][idx].to(device)
            txt_b = X_txt["train"][idx].to(device)
            y_b = Y["train"][idx].to(device)
            
            img_m, txt_m, y_a, y_b_mix, lam = mixup_data(img_b, txt_b, y_b, alpha=0.2)
            
            optimizer.zero_grad()
            logits = model(img_m, txt_m)
            loss = mixup_criterion(criterion, logits, y_a, y_b_mix, lam)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    model.eval()
    with torch.no_grad():
        vp = F.softmax(model(X_img["val"].to(device), X_txt["val"].to(device)), dim=1)[:, 1].cpu().numpy()
        tp = F.softmax(model(X_img["test"].to(device), X_txt["test"].to(device)), dim=1)[:, 1].cpu().numpy()
        val_probs_all.append(vp)
        test_probs_all.append(tp)

ens_val_p = np.mean(val_probs_all, axis=0)
ens_test_p = np.mean(test_probs_all, axis=0)

# Validation threshold search for max accuracy
best_acc, best_th = 0.0, 0.50
for th in np.linspace(0.35, 0.65, 301):
    acc = accuracy_score(Y["val"].numpy(), (ens_val_p >= th).astype(int))
    if acc > best_acc:
        best_acc, best_th = acc, th

y_te = Y["test"].numpy()
test_preds = (ens_test_p >= best_th).astype(int)
auc = roc_auc_score(y_te, ens_test_p)
acc = accuracy_score(y_te, test_preds)
tn, fp, fn, tp = confusion_matrix(y_te, test_preds).ravel()

print("\n" + "=" * 80)
print("CROSS-ATTENTION FUSION RESULTS ON OFFICIAL TEST SET (N = 2,000)")
print("=" * 80)
print(f"Optimal Validation Cutoff:  {best_th:.3f}")
print(f"Test AUC:                  {auc:.4f}")
print(f"Test Accuracy:             {acc * 100:5.2f}%")
print(f"Sensitivity (Recall):      {tp / (tp + fn) * 100:5.1f}%")
print(f"Specificity:               {tn / (tn + fp) * 100:5.1f}%")
print("=" * 80)