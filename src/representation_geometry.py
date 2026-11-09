"""
representation_geometry.py - Computes quantitative geometry metrics on representation space.
Evaluates Silhouette Score, Pairwise Race Centroid Distances, and Pathology Separation
before and after LEACE on the held-out TEST split.
"""
import torch, numpy as np
from sklearn.metrics import silhouette_score

print("Loading cached features for geometric audit...")
d = torch.load("cached_features.pt", weights_only=True)
SPLIT = "test"
X = d[SPLIT]["img"].float().numpy()
T = d[SPLIT]["txt"].float().numpy()
Y = d[SPLIT]["labels"].long().numpy()
R = d[SPLIT]["races"].long().numpy()
DUAL = np.concatenate([X, T], axis=1).astype(np.float32)

# Load Train for LEACE
X_tr = d["train"]["img"].float().numpy()
T_tr = d["train"]["txt"].float().numpy()
R_tr = d["train"]["races"].long().numpy()
DUAL_tr = np.concatenate([X_tr, T_tr], axis=1).astype(np.float32)

def fit_leace(F, z):
    mu = F.mean(0)
    Fc = F - mu
    Z = np.eye(3)[z]
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

print("Fitting LEACE on TRAIN and projecting TEST...")
leace_fn = fit_leace(DUAL_tr, R_tr)
DUAL_LEACE = leace_fn(DUAL)

def analyze_geometry(name, feats):
    # Subsample 1000 points for fast silhouette calculation
    rng = np.random.RandomState(0)
    sub_idx = rng.choice(len(feats), size=min(1000, len(feats)), replace=False)
    
    sil_race = silhouette_score(feats[sub_idx], R[sub_idx])
    sil_disease = silhouette_score(feats[sub_idx], Y[sub_idx])
    
    # Centroids by Race
    c0 = feats[R == 0].mean(0) # Asian
    c1 = feats[R == 1].mean(0) # Black
    c2 = feats[R == 2].mean(0) # White
    
    d_01 = np.linalg.norm(c0 - c1)
    d_02 = np.linalg.norm(c0 - c2)
    d_12 = np.linalg.norm(c1 - c2)
    mean_race_dist = np.mean([d_01, d_02, d_12])
    
    # Centroids by Disease (Glaucoma vs Healthy)
    c_healthy = feats[Y == 0].mean(0)
    c_glaucoma = feats[Y == 1].mean(0)
    disease_dist = np.linalg.norm(c_glaucoma - c_healthy)
    
    # Disentanglement Ratio: Disease Separation / Demographic Separation
    disentangle_ratio = disease_dist / (mean_race_dist + 1e-8)
    
    return {
        "sil_race": sil_race,
        "sil_disease": sil_disease,
        "mean_race_dist": mean_race_dist,
        "disease_dist": disease_dist,
        "ratio": disentangle_ratio
    }

geom_raw = analyze_geometry("Raw Dual Embeddings", DUAL)
geom_leace = analyze_geometry("LEACE Dual Embeddings", DUAL_LEACE)

print("\n" + "=" * 78)
print("REPRESENTATION GEOMETRY AUDIT (TEST SPLIT)")
print("=" * 78)
print(f"{'Metric':35s} | {'Raw Dual (C)':15s} | {'LEACE (F2)':15s} | {'Delta':10s}")
print("-" * 78)
print(f"{'Demographic Silhouette Score (Race)':35s} | {geom_raw['sil_race']:15.4f} | {geom_leace['sil_race']:15.4f} | {geom_leace['sil_race'] - geom_raw['sil_race']:+10.4f}")
print(f"{'Mean Inter-Race Centroid Distance':35s} | {geom_raw['mean_race_dist']:15.4f} | {geom_leace['mean_race_dist']:15.4f} | {geom_leace['mean_race_dist'] - geom_raw['mean_race_dist']:+10.4f}")
print(f"{'Pathology Centroid Distance (Y=1 vs 0)':35s} | {geom_raw['disease_dist']:15.4f} | {geom_leace['disease_dist']:15.4f} | {geom_leace['disease_dist'] - geom_raw['disease_dist']:+10.4f}")
print(f"{'Disease-to-Demographic Separation Ratio':35s} | {geom_raw['ratio']:15.2f}x| {geom_leace['ratio']:15.2f}x| {geom_leace['ratio'] - geom_raw['ratio']:+10.2f}x")
print("=" * 78)