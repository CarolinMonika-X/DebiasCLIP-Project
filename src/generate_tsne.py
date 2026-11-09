"""
generate_tsne.py - Visualizes Representation Geometry via t-SNE.
Generates 4-panel publication figure:
Top: Colored by Race (Raw vs LEACE) -> Proves demographic manifold collapse.
Bottom: Colored by Glaucoma (Raw vs LEACE) -> Proves pathology preservation.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

print("Loading cached test features for t-SNE visualization...")
d = torch.load("cached_features.pt", weights_only=True)
X_test = d["test"]["img"].float().numpy()
T_test = d["test"]["txt"].float().numpy()
Y_test = d["test"]["labels"].long().numpy()
R_test = d["test"]["races"].long().numpy()
DUAL_test = np.concatenate([X_test, T_test], axis=1).astype(np.float32)

# Fit LEACE on train
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

leace = fit_leace(DUAL_tr, R_tr)
DUAL_leace = leace(DUAL_test)

# Subsample 1200 points for clear visualization
rng = np.random.RandomState(42)
sub_idx = rng.choice(len(DUAL_test), size=min(1200, len(DUAL_test)), replace=False)
raw_sub = DUAL_test[sub_idx]
leace_sub = DUAL_leace[sub_idx]
r_sub = R_test[sub_idx]
y_sub = Y_test[sub_idx]

print("Running t-SNE dimensional reduction (takes ~30-40 seconds)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
emb_raw = tsne.fit_transform(raw_sub)
emb_leace = tsne.fit_transform(leace_sub)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
race_names = ["Asian Ancestry", "Black Ancestry", "White Ancestry"]
race_colors = ["#e74c3c", "#3498db", "#2ecc71"]
disease_names = ["Healthy (No Glaucoma)", "Glaucoma Diagnosed"]
disease_colors = ["#95a5a6", "#9b59b6"]

# (0, 0): Raw colored by Race
for g in (0, 1, 2):
    mask = (r_sub == g)
    axs[0, 0].scatter(emb_raw[mask, 0], emb_raw[mask, 1], c=race_colors[g], label=race_names[g], alpha=0.6, s=18)
axs[0, 0].set_title("(a) Raw Embeddings: Demographic Manifolds (Distinct Clusters)", fontsize=11, fontweight="bold")
axs[0, 0].legend(loc="upper right", markerscale=1.5, fontsize=9)
axs[0, 0].set_xticks([]); axs[0, 0].set_yticks([])

# (0, 1): LEACE colored by Race
for g in (0, 1, 2):
    mask = (r_sub == g)
    axs[0, 1].scatter(emb_leace[mask, 0], emb_leace[mask, 1], c=race_colors[g], label=race_names[g], alpha=0.6, s=18)
axs[0, 1].set_title("(b) LEACE Embeddings: Demographic Collapse (Isotropic Cloud)", fontsize=11, fontweight="bold", color="#27ae60")
axs[0, 1].legend(loc="upper right", markerscale=1.5, fontsize=9)
axs[0, 1].set_xticks([]); axs[0, 1].set_yticks([])

# (1, 0): Raw colored by Pathology
for c in (0, 1):
    mask = (y_sub == c)
    axs[1, 0].scatter(emb_raw[mask, 0], emb_raw[mask, 1], c=disease_colors[c], label=disease_names[c], alpha=0.6, s=18)
axs[1, 0].set_title("(c) Raw Embeddings: Disease Separation", fontsize=11, fontweight="bold")
axs[1, 0].legend(loc="upper right", markerscale=1.5, fontsize=9)
axs[1, 0].set_xticks([]); axs[1, 0].set_yticks([])

# (1, 1): LEACE colored by Pathology
for c in (0, 1):
    mask = (y_sub == c)
    axs[1, 1].scatter(emb_leace[mask, 0], emb_leace[mask, 1], c=disease_colors[c], label=disease_names[c], alpha=0.6, s=18)
axs[1, 1].set_title("(d) LEACE Embeddings: Preserved Disease Separation", fontsize=11, fontweight="bold", color="#27ae60")
axs[1, 1].legend(loc="upper right", markerscale=1.5, fontsize=9)
axs[1, 1].set_xticks([]); axs[1, 1].set_yticks([])

plt.tight_layout()
out_file = OUT / "fig_tsne_manifold.png"
fig.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Successfully generated Figure 3: {out_file}")