"""
src/additional_classifiers.py
Evaluate non-linear tree-based (XGBoost) and kernel (RBF SVM) heads
on Raw Dual vs. LEACE-erased features.
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from xgboost import XGBClassifier

print("=" * 80)
print("EVALUATING ADDITIONAL CLASSIFIERS ON CACHED FEATURES")
print("=" * 80)

# Load cached data
d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

X_img = {s: d[s]["img"].float() for s in SPLITS}
X_txt = {s: d[s]["txt"].float() for s in SPLITS}
Y = {s: d[s]["labels"].numpy() for s in SPLITS}
R = {s: d[s]["races"].numpy() for s in SPLITS}

X_joint = {s: torch.cat([X_img[s], X_txt[s]], dim=-1) for s in SPLITS}

# Compute LEACE projector on Train
r_tr_oh = torch.nn.functional.one_hot(torch.tensor(R["train"]), num_classes=3).float()
x_mean = X_joint["train"].mean(dim=0, keepdim=True)
x_cent = X_joint["train"] - x_mean
cov_xz = (x_cent.T @ r_tr_oh) / (len(x_cent) - 1)
u, _, _ = torch.linalg.svd(cov_xz, full_matrices=False)
P_leace = torch.eye(1024) - (u @ u.T)

X_leace = {s: ((X_joint[s] - x_mean) @ P_leace).numpy() for s in SPLITS}
X_raw = {s: X_joint[s].numpy() for s in SPLITS}

def evaluate_model(clf, name, X_tr, y_tr, X_te, y_te, r_te):
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_te)
    
    auc = roc_auc_score(y_te, probs)
    preds = (probs >= 0.50).astype(int)
    acc = accuracy_score(y_te, preds)
    
    tprs, fprs, sels = [], [], []
    for r_i in range(3):
        m = (r_te == r_i)
        tn, fp, fn, tp = confusion_matrix(y_te[m], preds[m], labels=[0,1]).ravel()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (tn + fp) if (tn + fp) > 0 else 0)
        sels.append((tp + fp) / len(y_te[m]))
        
    eod_max = max(max(tprs) - min(tprs), max(fprs) - min(fprs))
    dpd = max(sels) - min(sels)
    
    print(f"{name:<30} | AUC: {auc:.4f} | Acc: {acc*100:5.2f}% | EOD_max: {eod_max:.4f} | DPD: {dpd:.4f}")

print(f"{'Model & Feature Type':<30} | {'AUC':<7} | {'Accuracy':<9} | {'EOD_max':<8} | {'DPD':<7}")
print("-" * 75)

# 1. XGBoost on Raw and LEACE
xgb_raw = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
evaluate_model(xgb_raw, "XGBoost (Raw Dual)", X_raw["train"], Y["train"], X_raw["test"], Y["test"], R["test"])

xgb_leace = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
evaluate_model(xgb_leace, "XGBoost (LEACE Dual)", X_leace["train"], Y["train"], X_leace["test"], Y["test"], R["test"])

# 2. RBF Support Vector Machine on Raw and LEACE
svm_raw = SVC(C=1.0, kernel="rbf", probability=True, random_state=42)
evaluate_model(svm_raw, "SVM-RBF (Raw Dual)", X_raw["train"], Y["train"], X_raw["test"], Y["test"], R["test"])

svm_leace = SVC(C=1.0, kernel="rbf", probability=True, random_state=42)
evaluate_model(svm_leace, "SVM-RBF (LEACE Dual)", X_leace["train"], Y["train"], X_leace["test"], Y["test"], R["test"])
print("=" * 80)