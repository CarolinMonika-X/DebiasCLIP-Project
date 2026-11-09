"""
boost_gbdt.py - LightGBM with Cross-Modal Interaction Terms.
Extracts higher-order non-linear diagnostic signals from frozen CLIP embeddings.
"""
import torch, numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

print("Loading cached features...")
d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

X = {s: d[s]["img"].float().numpy() for s in SPLITS}
T = {s: d[s]["txt"].float().numpy() for s in SPLITS}
Y = {s: d[s]["labels"].long().numpy() for s in SPLITS}
R = {s: d[s]["races"].long().numpy() for s in SPLITS}

def build_features(x, t):
    # 1. Raw image + text (1024-dim)
    concat = np.concatenate([x, t], axis=1)
    # 2. Element-wise visual-text interaction (top correlated 128 dims via SVD or Hadamard)
    hadamard = x * t
    # 3. Pairwise cosine similarity per patient
    cos_sim = np.sum(x * t, axis=1, keepdims=True) / (
        np.linalg.norm(x, axis=1, keepdims=True) * np.linalg.norm(t, axis=1, keepdims=True) + 1e-8
    )
    return np.concatenate([concat, hadamard[:, :128], cos_sim], axis=1).astype(np.float32)

print("Constructing cross-modal interaction representations...")
FEATS = {s: build_features(X[s], T[s]) for s in SPLITS}

print(f"Feature dimension expanded: 1024 -> {FEATS['train'].shape[1]}")

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42
}

train_data = lgb.Dataset(FEATS["train"], label=Y["train"])
val_data = lgb.Dataset(FEATS["val"], label=Y["val"], reference=train_data)

print("Training tuned LightGBM model with early stopping on validation...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
)

# Predictions
p_val = model.predict(FEATS["val"], num_iteration=model.best_iteration)
p_test = model.predict(FEATS["test"], num_iteration=model.best_iteration)

# Find optimal threshold on validation
best_acc, best_thr = 0.0, 0.5
for thr in np.linspace(0.35, 0.65, 100):
    acc = accuracy_score(Y["val"], p_val > thr)
    if acc > best_acc:
        best_acc = acc
        best_thr = thr

test_acc_05 = accuracy_score(Y["test"], p_test > 0.5)
test_acc_opt = accuracy_score(Y["test"], p_test > best_thr)
test_auc = roc_auc_score(Y["test"], p_test)

print("\n" + "=" * 70)
print("LIGHTGBM + CROSS-MODAL INTERACTION RESULTS (TEST SET)")
print("=" * 70)
print(f"Test Accuracy (Default p > 0.50):       {test_acc_05 * 100:6.2f}%")
print(f"Test Accuracy (Val-Tuned Threshold):    {test_acc_opt * 100:6.2f}% (cutoff = {best_thr:.3f})")
print(f"Test AUC:                               {test_auc:.4f}")
print("=" * 70)