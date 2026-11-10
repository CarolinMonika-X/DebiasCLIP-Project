"""
src/unified_benchmark_final.py
Complete, publication-grade benchmark for Harvard-FairVLMed:
- Official split (Train=7000, Val=1000, Test=2000)
- Validation-only tuning: C for probe, lambda for penalties, k for INLP, and operating thresholds
- 9 Evaluation pipelines with standardized 5-seed MLP heads
- Explicit EOD definitions: eod_max = max(tpr_gap, fpr_gap), aod = (tpr_gap + fpr_gap) / 2
- Linear & Non-linear MLP race probes (Macro-AUC & Balanced Accuracy)
- Operating points: Fixed 0.50, Val-Youden, Matched Spec=0.80, Matched Sens=0.80
- 1,000 Paired Bootstrap with 95% CI and 99.17% Bonferroni-corrected CIs
- Outputs saved to unified_final_output.txt and unified_final_results.json
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, balanced_accuracy_score
import json
import time
import math
import sys
import os

# Redirect print to both console and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("unified_final_output.txt")

print("=" * 110)
print("HARVARD-FAIRVLMED UNIFIED BENCHMARK & AUDIT (FINAL RIGOROUS RUN)")
print("Note on Definitions: In earlier scripts, 'EOD' was computed as Average Odds Difference (AOD).")
print("Here, we explicitly separate EOD_max = max(TPR_gap, FPR_gap) and AOD = (TPR_gap + FPR_gap) / 2.")
print("Primary Pre-specified Endpoints: Delta AUC (Non-inferiority margin delta = -0.01) and Delta EOD_max.")
print("=" * 110)

# -----------------------------------------------------------------------------
# 1. Load Cached Features & Verify Official Split
# -----------------------------------------------------------------------------
d = torch.load("cached_features.pt", weights_only=True)
SPLITS = ("train", "val", "test")

X_img = {s: d[s]["img"].float() for s in SPLITS}
X_txt = {s: d[s]["txt"].float() for s in SPLITS}
Y = {s: d[s]["labels"].long() for s in SPLITS}
R = {s: d[s]["races"].long() for s in SPLITS}

X_joint = {s: torch.cat([X_img[s], X_txt[s]], dim=-1) for s in SPLITS}

print(f"\n[Split Confirmation] Train: N={len(Y['train'])}, Val: N={len(Y['val'])}, Test: N={len(Y['test'])}")
print(f"Feature Dimensions: Image={X_img['train'].shape[1]}, Text={X_txt['train'].shape[1]}, Joint={X_joint['train'].shape[1]}")

race_names = ["Asian", "Black", "White"]
for s in SPLITS:
    y_s = Y[s].numpy()
    r_s = R[s].numpy()
    print(f"\n--- {s.upper()} SPLIT COHORT STATISTICS (Total N={len(y_s)}) ---")
    print(f"{'Group':<10} | {'Count (N)':<10} | {'Share (%)':<10} | {'Glaucoma Prevalence (%)':<24}")
    print("-" * 62)
    for i, name in enumerate(race_names):
        n_g = np.sum(r_s == i)
        prev = np.mean(y_s[r_s == i]) * 100
        print(f"{name:<10} | {n_g:<10d} | {n_g/len(r_s)*100:<9.1f}% | {prev:<23.2f}%")
    print(f"{'Overall':<10} | {len(y_s):<10d} | 100.0%    | {np.mean(y_s)*100:<23.2f}%")

# -----------------------------------------------------------------------------
# 2. Race Probes (Linear with Val C-sweep + Non-linear MLP)
# -----------------------------------------------------------------------------
class ProbeMLP(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )
    def forward(self, x):
        return self.net(x)

def audit_race_probes(feat_tr, r_tr, feat_va, r_va, feat_te, r_te):
    # Scale features
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(feat_tr.numpy())
    x_va_s = scaler.transform(feat_va.numpy())
    x_te_s = scaler.transform(feat_te.numpy())
    
    r_va_oh = np.eye(3)[r_va.numpy()]
    r_te_oh = np.eye(3)[r_te.numpy()]
    
    # 1. Linear Probe with validation C sweep
    best_c, best_va_auc = 0.01, -1.0
    for c_cand in [0.001, 0.01, 0.1, 1.0]:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=c_cand)
        clf.fit(x_tr_s, r_tr.numpy())
        va_p = clf.predict_proba(x_va_s)
        auc_v = roc_auc_score(r_va_oh, va_p, multi_class="ovr", average="macro")
        if auc_v > best_va_auc:
            best_va_auc = auc_v
            best_c = c_cand
            
    lin_probe = LogisticRegression(max_iter=1000, solver="lbfgs", C=best_c)
    lin_probe.fit(x_tr_s, r_tr.numpy())
    lin_te_p = lin_probe.predict_proba(x_te_s)
    lin_te_pred = lin_probe.predict(x_te_s)
    lin_auc = roc_auc_score(r_te_oh, lin_te_p, multi_class="ovr", average="macro")
    lin_bacc = balanced_accuracy_score(r_te.numpy(), lin_te_pred)
    
    # 2. MLP Non-linear Probe with Early Stopping
    torch.manual_seed(42)
    mlp = ProbeMLP(in_dim=feat_tr.shape[1])
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_weights = None
    
    x_tr_t = torch.tensor(x_tr_s).float()
    x_va_t = torch.tensor(x_va_s).float()
    x_te_t = torch.tensor(x_te_s).float()
    
    n_tr = len(x_tr_t)
    for epoch in range(30):
        mlp.train()
        perm = torch.randperm(n_tr)
        for j in range(0, n_tr, 128):
            idx = perm[j:j+128]
            opt.zero_grad()
            l = loss_fn(mlp(x_tr_t[idx]), r_tr[idx])
            l.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            v_loss = loss_fn(mlp(x_va_t), r_va).item()
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            
    mlp.load_state_dict(best_weights)
    mlp.eval()
    with torch.no_grad():
        mlp_te_logits = mlp(x_te_t)
        mlp_te_p = torch.softmax(mlp_te_logits, dim=1).numpy()
        mlp_te_pred = torch.argmax(mlp_te_logits, dim=1).numpy()
        
    mlp_auc = roc_auc_score(r_te_oh, mlp_te_p, multi_class="ovr", average="macro")
    mlp_bacc = balanced_accuracy_score(r_te.numpy(), mlp_te_pred)
    
    return {
        "lin_auc": lin_auc, "lin_bacc": lin_bacc, "lin_c": best_c,
        "mlp_auc": mlp_auc, "mlp_bacc": mlp_bacc
    }

# -----------------------------------------------------------------------------
# 3. Fit Erasure Representations (Train Split Only)
# -----------------------------------------------------------------------------
print("\n[Concept Erasure Fitting on Train Split]")
# A. LEACE Joint
r_train_oh = torch.nn.functional.one_hot(R["train"], num_classes=3).float()
x_mean_j = X_joint["train"].mean(dim=0, keepdim=True)
x_cent_j = X_joint["train"] - x_mean_j
cov_xz_j = (x_cent_j.T @ r_train_oh) / (len(x_cent_j) - 1)
u_j, _, _ = torch.linalg.svd(cov_xz_j, full_matrices=False)
P_leace_j = torch.eye(1024) - (u_j @ u_j.T)
X_leace_joint = {s: ((X_joint[s] - x_mean_j) @ P_leace_j) for s in SPLITS}

# B. LEACE Image-Only
x_mean_i = X_img["train"].mean(dim=0, keepdim=True)
x_cent_i = X_img["train"] - x_mean_i
cov_xz_i = (x_cent_i.T @ r_train_oh) / (len(x_cent_i) - 1)
u_i, _, _ = torch.linalg.svd(cov_xz_i, full_matrices=False)
P_leace_i = torch.eye(512) - (u_i @ u_i.T)
X_leace_img = {s: ((X_img[s] - x_mean_i) @ P_leace_i) for s in SPLITS}

# C. Multiclass INLP on Joint (with Validation k Selection)
print("Evaluating INLP candidates k in {1, 2, 5, 10} on Validation split...")
inlp_dict_by_k = {}
P_curr = np.eye(1024)
x_tr_curr = X_joint["train"].clone().numpy()
x_va_curr = X_joint["val"].clone().numpy()
x_te_curr = X_joint["test"].clone().numpy()

for step in range(1, 11):
    clf_inlp = LogisticRegression(max_iter=400, solver="lbfgs", C=0.1)
    clf_inlp.fit(x_tr_curr, R["train"].numpy())
    _, _, vt_inlp = np.linalg.svd(clf_inlp.coef_, full_matrices=False)
    v_step = vt_inlp[0:1, :]
    P_step = np.eye(1024) - (v_step.T @ v_step)
    
    x_tr_curr = x_tr_curr @ P_step
    x_va_curr = x_va_curr @ P_step
    x_te_curr = x_te_curr @ P_step
    
    if step in [1, 2, 5, 10]:
        inlp_dict_by_k[step] = {
            "train": torch.tensor(x_tr_curr).float(),
            "val": torch.tensor(x_va_curr).float(),
            "test": torch.tensor(x_te_curr).float()
        }

# -----------------------------------------------------------------------------
# 4. Standardized Diagnostic MLP Head
# -----------------------------------------------------------------------------
class HeadMLP(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_and_predict(features_dict, penalty_mode=None, penalty_lambda=0.0, sample_weights=None):
    all_seed_val_p, all_seed_te_p = [], []
    for seed in range(5):
        torch.manual_seed(seed * 100)
        m = HeadMLP(in_dim=features_dict["train"].shape[1])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        n_tr = len(features_dict["train"])
        
        for epoch in range(30):
            m.train()
            perm = torch.randperm(n_tr)
            for j in range(0, n_tr, 128):
                idx = perm[j:j+128]
                opt.zero_grad()
                logits = m(features_dict["train"][idx])
                
                if sample_weights is not None:
                    loss = (nn.functional.cross_entropy(logits, Y["train"][idx], reduction="none") * sample_weights[idx]).mean()
                elif penalty_mode == "parity":
                    base_loss = nn.functional.cross_entropy(logits, Y["train"][idx])
                    p_pos = torch.softmax(logits, dim=1)[:, 1]
                    reg = torch.tensor(0.0)
                    for r_idx in range(3):
                        m_r = (R["train"][idx] == r_idx)
                        if m_r.sum() > 0:
                            reg = reg + (p_pos[m_r].mean() - p_pos.mean())**2
                    loss = base_loss + penalty_lambda * reg
                elif penalty_mode == "label_conditioned":
                    base_loss = nn.functional.cross_entropy(logits, Y["train"][idx])
                    p_pos = torch.softmax(logits, dim=1)[:, 1]
                    reg = torch.tensor(0.0)
                    for y_v in [0, 1]:
                        m_y = (Y["train"][idx] == y_v)
                        if m_y.sum() > 0:
                            sub_p = p_pos[m_y]
                            sub_r = R["train"][idx][m_y]
                            for r_idx in range(3):
                                m_yr = (sub_r == r_idx)
                                if m_yr.sum() > 0:
                                    reg = reg + (sub_p[m_yr].mean() - sub_p.mean())**2
                    loss = base_loss + penalty_lambda * reg
                else:
                    loss = nn.functional.cross_entropy(logits, Y["train"][idx])
                    
                loss.backward()
                opt.step()
                
        m.eval()
        with torch.no_grad():
            vp = torch.softmax(m(features_dict["val"]), dim=1)[:, 1].numpy()
            tp = torch.softmax(m(features_dict["test"]), dim=1)[:, 1].numpy()
        all_seed_val_p.append(vp)
        all_seed_te_p.append(tp)
        
    return all_seed_val_p, all_seed_te_p, np.mean(all_seed_val_p, axis=0), np.mean(all_seed_te_p, axis=0)

# Sample weights
counts_r = torch.bincount(R["train"]).float()
w_r = 1.0 / (counts_r + 1e-5)
s_weights = (w_r[R["train"]] / w_r[R["train"]].mean())

# Helper metrics function
def compute_metrics(y_true, y_prob, r_true, threshold=0.50, group_ths=None):
    if group_ths is not None:
        preds = np.zeros_like(y_true)
        for r_i, th in group_ths.items():
            mask = (r_true == r_i)
            preds[mask] = (y_prob[mask] >= th).astype(int)
    else:
        preds = (y_prob >= threshold).astype(int)
        
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, y_prob)
    
    tprs, fprs, sels, g_aucs = [], [], [], []
    for r_i in range(3):
        m = (r_true == r_i)
        tn, fp, fn, tp = confusion_matrix(y_true[m], preds[m], labels=[0,1]).ravel()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (tn + fp) if (tn + fp) > 0 else 0)
        sels.append((tp + fp) / len(y_true[m]))
        g_aucs.append(roc_auc_score(y_true[m], y_prob[m]))
        
    tpr_gap = max(tprs) - min(tprs)
    fpr_gap = max(fprs) - min(fprs)
    dpd = max(sels) - min(sels)
    eod_max = max(tpr_gap, fpr_gap)
    aod = (tpr_gap + fpr_gap) / 2.0
    worst_auc = min(g_aucs)
    
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    return {
        "auc": auc, "acc": acc, "sens": sens, "spec": spec,
        "tpr_gap": tpr_gap, "fpr_gap": fpr_gap,
        "eod_max": eod_max, "aod": aod, "dpd": dpd,
        "worst_auc": worst_auc, "g_aucs": g_aucs, "tprs": tprs, "fprs": fprs, "sels": sels
    }

# -----------------------------------------------------------------------------
# 5. Validation Selection for Penalties, INLP, and Group Thresholds
# -----------------------------------------------------------------------------
y_val = Y["val"].numpy()
r_val = R["val"].numpy()
y_test = Y["test"].numpy()
r_test = R["test"].numpy()

print("\n--- Hyperparameter Selection on Validation Split ---")

# A. Parity Penalty Lambda Sweep
best_lam_p, best_val_dpd = 1.0, float("inf")
p_par_data = {}
for lam in [0.1, 1.0, 5.0, 10.0]:
    _, _, v_p, t_p = train_and_predict(X_joint, penalty_mode="parity", penalty_lambda=lam)
    m_v = compute_metrics(y_val, v_p, r_val, threshold=0.50)
    print(f"  Parity Penalty lambda={lam:4.1f} | Val DPD={m_v['dpd']:.4f} | Val AUC={m_v['auc']:.4f}")
    if m_v["dpd"] < best_val_dpd:
        best_val_dpd = m_v["dpd"]
        best_lam_p = lam
    p_par_data[lam] = (v_p, t_p)

# B. Label-Conditioned Penalty Lambda Sweep
best_lam_l, best_val_eod = 1.0, float("inf")
p_lab_data = {}
for lam in [0.1, 1.0, 5.0]:
    _, _, v_p, t_p = train_and_predict(X_joint, penalty_mode="label_conditioned", penalty_lambda=lam)
    m_v = compute_metrics(y_val, v_p, r_val, threshold=0.50)
    print(f"  Label-Cond Penalty lambda={lam:4.1f} | Val EOD_max={m_v['eod_max']:.4f} | Val AUC={m_v['auc']:.4f}")
    if m_v["eod_max"] < best_val_eod:
        best_val_eod = m_v["eod_max"]
        best_lam_l = lam
    p_lab_data[lam] = (v_p, t_p)

# C. INLP Depth Selection
best_k, best_val_inlp_auc = 10, -1.0
inlp_preds_by_k = {}
for k_cand in [1, 2, 5, 10]:
    _, _, v_p, t_p = train_and_predict(inlp_dict_by_k[k_cand])
    m_v = compute_metrics(y_val, v_p, r_val, threshold=0.50)
    print(f"  INLP k={k_cand:2d} | Val AUC={m_v['auc']:.4f} | Val EOD_max={m_v['eod_max']:.4f}")
    if m_v["auc"] > best_val_inlp_auc:
        best_val_inlp_auc = m_v["auc"]
        best_k = k_cand
    inlp_preds_by_k[k_cand] = (v_p, t_p)

print(f"\n[Validation Optimal Choices] Parity lambda={best_lam_p}, Label-Cond lambda={best_lam_l}, INLP k={best_k}")

# -----------------------------------------------------------------------------
# 6. Execute All 9 Pipelines
# -----------------------------------------------------------------------------
print("\n[Executing All 9 Pipelines across 5 Seeds...]")

pipeline_configs = [
    ("Raw Image-Only", X_img, None, 0.0, None),
    ("Raw Dual (Image+Text)", X_joint, None, 0.0, None),
    ("Sample Reweighting", X_joint, None, 0.0, s_weights),
    (f"Parity Penalty (lam={best_lam_p})", X_joint, "parity", best_lam_p, None),
    (f"Label-Cond Penalty (lam={best_lam_l})", X_joint, "label_conditioned", best_lam_l, None),
    ("Group Thresholds (Hardt)", X_joint, None, 0.0, None),
    (f"INLP (k={best_k})", inlp_dict_by_k[best_k], None, 0.0, None),
    ("LEACE Joint Dual", X_leace_joint, None, 0.0, None),
    ("LEACE Image-Only", X_leace_img, None, 0.0, None),
]

all_models_data = {}
for name, f_dict, p_mode, p_lam, s_w in pipeline_configs:
    v_seeds, t_seeds, v_p, t_p = train_and_predict(f_dict, penalty_mode=p_mode, penalty_lambda=p_lam, sample_weights=s_w)
    all_models_data[name] = {
        "v_seeds": v_seeds, "t_seeds": t_seeds, "v_p": v_p, "t_p": t_p
    }

# Compute Group-Specific Thresholds (Hardt et al.) on Raw Dual Validation
v_p_raw = all_models_data["Raw Dual (Image+Text)"]["v_p"]
tn_pool, fp_pool, fn_pool, tp_pool = confusion_matrix(y_val, (v_p_raw >= 0.50).astype(int), labels=[0,1]).ravel()
target_pool_tpr = tp_pool / (tp_pool + fn_pool)

group_specific_ths = {}
for r_i in range(3):
    m_val = (r_val == r_i)
    best_th, min_diff = 0.50, float("inf")
    for th in np.linspace(0.10, 0.90, 401):
        tn, fp, fn, tp = confusion_matrix(y_val[m_val], (v_p_raw[m_val] >= th).astype(int), labels=[0,1]).ravel()
        sens = tp / (tp + fn)
        if abs(sens - target_pool_tpr) < min_diff:
            min_diff = abs(sens - target_pool_tpr)
            best_th = th
    group_specific_ths[r_i] = best_th

print(f"Group Thresholds (Hardt) matched to pooled Val TPR={target_pool_tpr:.4f}: {group_specific_ths}")

# -----------------------------------------------------------------------------
# 7. Evaluate Race Probes (Linear + MLP)
# -----------------------------------------------------------------------------
print("\nAuditing Representation Race Decodability Probes...")
probes_results = {
    "Raw Dual (Image+Text)": audit_race_probes(X_joint["train"], R["train"], X_joint["val"], R["val"], X_joint["test"], R["test"]),
    "LEACE Joint Dual": audit_race_probes(X_leace_joint["train"], R["train"], X_leace_joint["val"], R["val"], X_leace_joint["test"], R["test"]),
    f"INLP (k={best_k})": audit_race_probes(inlp_dict_by_k[best_k]["train"], R["train"], inlp_dict_by_k[best_k]["val"], R["val"], inlp_dict_by_k[best_k]["test"], R["test"])
}

print(f"\n{'Representation':<25} | {'Lin Macro-AUC':<14} | {'Lin Bal-Acc (%)':<16} | {'MLP Macro-AUC':<14} | {'MLP Bal-Acc (%)':<16}")
print("-" * 90)
for k, pr in probes_results.items():
    print(f"{k:<25} | {pr['lin_auc']:<14.4f} | {pr['lin_bacc']*100:<15.2f}% | {pr['mlp_auc']:<14.4f} | {pr['mlp_bacc']*100:<15.2f}%")

# -----------------------------------------------------------------------------
# 8. Table 1: Standardized Benchmark (p = 0.50)
# -----------------------------------------------------------------------------
print("\n" + "=" * 135)
print("TABLE 1: STANDARDIZED BENCHMARK (OFFICIAL SPLIT, 5 SEEDS, p=0.50, MEAN ± STD OVER SEEDS)")
print("=" * 135)
print(f"{'Method':<28} | {'AUC':<14} | {'Acc (%)':<14} | {'Sens/Spec (%)':<15} | {'TPR Gap':<8} | {'FPR Gap':<8} | {'EOD(max)':<9} | {'AOD':<8} | {'DPD':<8} | {'WorstAUC':<8}")
print("-" * 135)

table1_metrics = {}
for name, data in all_models_data.items():
    g_th = group_specific_ths if "Group Thresholds" in name else None
    
    # Calculate seed mean and std
    seed_aucs, seed_accs = [], []
    for s_idx in range(5):
        s_res = compute_metrics(y_test, data["t_seeds"][s_idx], r_test, threshold=0.50, group_ths=g_th)
        seed_aucs.append(s_res["auc"])
        seed_accs.append(s_res["acc"])
        
    ens_res = compute_metrics(y_test, data["t_p"], r_test, threshold=0.50, group_ths=g_th)
    table1_metrics[name] = ens_res
    
    auc_str = f"{np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.3f}"
    acc_str = f"{np.mean(seed_accs)*100:4.1f}±{np.std(seed_accs)*100:3.1f}%"
    sens_spec = f"{ens_res['sens']*100:4.1f}/{ens_res['spec']*100:4.1f}"
    
    print(f"{name:<28} | {auc_str:<14} | {acc_str:<14} | {sens_spec:<15} | {ens_res['tpr_gap']:<8.4f} | {ens_res['fpr_gap']:<8.4f} | {ens_res['eod_max']:<9.4f} | {ens_res['aod']:<8.4f} | {ens_res['dpd']:<8.4f} | {ens_res['worst_auc']:<8.4f}")

# -----------------------------------------------------------------------------
# 9. Operating Points: Youden & Matched Specificity / Sensitivity
# -----------------------------------------------------------------------------
print("\n" + "=" * 125)
print("TABLE 2: SENSITIVITY CHECK ACROSS OPERATING POINTS (VAL-TUNED YOUDEN & MATCHED OPERATING POINTS)")
print("=" * 125)
print(f"{'Method':<28} | {'Operating Point':<18} | {'Cutoff':<8} | {'AUC':<7} | {'Acc (%)':<8} | {'Sens (%)':<9} | {'Spec (%)':<9} | {'EOD(max)':<9} | {'AOD':<8} | {'DPD':<8}")
print("-" * 125)

op_results = {}
for name, data in all_models_data.items():
    if "Group Thresholds" in name: continue
    vp, tp = data["v_p"], data["t_p"]
    
    # 1. Val-Youden Cutoff
    best_thr, best_j = 0.50, -1.0
    for th in np.linspace(0.20, 0.80, 121):
        tn, fp, fn, tp_c = confusion_matrix(y_val, (vp >= th).astype(int), labels=[0,1]).ravel()
        j = (tp_c / (tp_c + fn)) + (tn / (tn + fp)) - 1
        if j > best_j:
            best_j = j
            best_thr = th
    res_y = compute_metrics(y_test, tp, r_test, threshold=best_thr)
    print(f"{name:<28} | {'Val-Youden':<18} | {best_thr:<8.3f} | {res_y['auc']:<7.4f} | {res_y['acc']*100:<7.2f}% | {res_y['sens']*100:<8.1f}% | {res_y['spec']*100:<8.1f}% | {res_y['eod_max']:<9.4f} | {res_y['aod']:<8.4f} | {res_y['dpd']:<8.4f}")
    
    # 2. Matched Specificity = 0.80 on Validation
    th_sp80 = 0.50
    for th in np.linspace(0.10, 0.90, 201):
        tn, fp, fn, tp_c = confusion_matrix(y_val, (vp >= th).astype(int), labels=[0,1]).ravel()
        if (tn / (tn + fp)) >= 0.80:
            th_sp80 = th
            break
    res_sp = compute_metrics(y_test, tp, r_test, threshold=th_sp80)
    print(f"{name:<28} | {'Matched Spec=0.80':<18} | {th_sp80:<8.3f} | {res_sp['auc']:<7.4f} | {res_sp['acc']*100:<7.2f}% | {res_sp['sens']*100:<8.1f}% | {res_sp['spec']*100:<8.1f}% | {res_sp['eod_max']:<9.4f} | {res_sp['aod']:<8.4f} | {res_sp['dpd']:<8.4f}")

    # 3. Matched Sensitivity = 0.80 on Validation
    th_se80 = 0.50
    for th in np.linspace(0.90, 0.10, 201):
        tn, fp, fn, tp_c = confusion_matrix(y_val, (vp >= th).astype(int), labels=[0,1]).ravel()
        if (tp_c / (tp_c + fn)) >= 0.80:
            th_se80 = th
            break
    res_se = compute_metrics(y_test, tp, r_test, threshold=th_se80)
    print(f"{name:<28} | {'Matched Sens=0.80':<18} | {th_se80:<8.3f} | {res_se['auc']:<7.4f} | {res_se['acc']*100:<7.2f}% | {res_se['sens']*100:<8.1f}% | {res_se['spec']*100:<8.1f}% | {res_se['eod_max']:<9.4f} | {res_se['aod']:<8.4f} | {res_se['dpd']:<8.4f}")
    print("-" * 125)

# -----------------------------------------------------------------------------
# 10. Paired Bootstrap Analysis (1,000 Resamples, Seed 0)
# -----------------------------------------------------------------------------
print("\n" + "=" * 125)
print("TABLE 3: PAIRED BOOTSTRAP (1,000 RESAMPLES, SEED 0): 95% CI AND BONFERRONI 99.17% CI")
print("Primary Endpoints: Delta AUC (Non-inferiority margin delta = -0.01) and Delta EOD_max")
print("Secondary Endpoints: AOD, DPD, Accuracy, TPR Gap, FPR Gap, Worst-group AUC")
print("=" * 125)

def paired_bootstrap(p_a, p_b, name_a, name_b, g_th_a=None, g_th_b=None):
    np.random.seed(0)
    n = len(y_test)
    keys = ["auc", "eod_max", "aod", "dpd", "acc", "tpr_gap", "fpr_gap", "worst_auc"]
    diffs = {k: [] for k in keys}
    
    for _ in range(1000):
        idx = np.random.choice(n, size=n, replace=True)
        y_b, r_b = y_test[idx], r_test[idx]
        res_a = compute_metrics(y_b, p_a[idx], r_b, threshold=0.50, group_ths=g_th_a)
        res_b = compute_metrics(y_b, p_b[idx], r_b, threshold=0.50, group_ths=g_th_b)
        
        for k in keys:
            diffs[k].append(res_a[k] - res_b[k])
            
    print(f"\n--- Comparison: {name_a} MINUS {name_b} ---")
    print(f"{'Endpoint':<28} | {'Mean Diff':<10} | {'95% CI':<24} | {'Bonferroni CI (alpha=0.05/6)':<28} | {'Excludes Zero?'}")
    print("-" * 115)
    
    for k in keys:
        arr = np.array(diffs[k])
        m = np.mean(arr)
        ci95 = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
        # Bonferroni for 6 secondary metrics: 100 - (5/6)% = 99.1667%
        ci_bonf = (np.percentile(arr, 100 * (0.05 / 12)), np.percentile(arr, 100 * (1.0 - 0.05 / 12)))
        
        excl_95 = (ci95[0] > 0 and ci95[1] > 0) or (ci95[0] < 0 and ci95[1] < 0)
        excl_bonf = (ci_bonf[0] > 0 and ci_bonf[1] > 0) or (ci_bonf[0] < 0 and ci_bonf[1] < 0)
        
        tag = "(PRIMARY)" if k in ["auc", "eod_max"] else "(Secondary)"
        lbl = f"{k.upper()} {tag}"
        
        ci95_str = f"[{ci95[0]:+.4f}, {ci95[1]:+.4f}]"
        cibonf_str = f"[{ci_bonf[0]:+.4f}, {ci_bonf[1]:+.4f}]"
        excl_str = f"95%:{'YES' if excl_95 else 'NO'} | Bonf:{'YES' if excl_bonf else 'NO'}"
        
        print(f"{lbl:<28} | {m:<10.4f} | {ci95_str:<24} | {cibonf_str:<28} | {excl_str}")
        
        if k == "auc":
            non_inf = ci95[0] >= -0.01
            print(f"  --> Non-inferiority Check (Lower Bound >= -0.01): {'CONFIRMED (' + str(round(ci95[0], 4)) + ' >= -0.01)' if non_inf else 'FAILED'}")

p_raw = all_models_data["Raw Dual (Image+Text)"]["t_p"]
p_leace = all_models_data["LEACE Joint Dual"]["t_p"]
p_inlp = all_models_data[f"INLP (k={best_k})"]["t_p"]
p_gth = all_models_data["Raw Dual (Image+Text)"]["t_p"]

paired_bootstrap(p_leace, p_raw, "LEACE Joint Dual", "Raw Dual (Image+Text)")
paired_bootstrap(p_leace, p_inlp, "LEACE Joint Dual", f"INLP (k={best_k})")
paired_bootstrap(p_leace, p_gth, "LEACE Joint Dual", "Group Thresholds (Hardt)", g_th_b=group_specific_ths)

# -----------------------------------------------------------------------------
# 11. Appendix: Per-Group Subgroup Metrics with Wilson Intervals
# -----------------------------------------------------------------------------
def wilson_interval(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

print("\n" + "=" * 115)
print("APPENDIX: PER-GROUP DETAILED CLINICAL BREAKDOWN WITH 95% WILSON SCORE INTERVALS")
print("=" * 115)
print(f"{'Group':<8} | {'Model':<15} | {'N':<6} | {'Prev(%)':<8} | {'Flagged Rate':<18} | {'TPR (Recall)':<22} | {'FPR':<22} | {'AUC'}")
print("-" * 115)

for i, g_name in enumerate(race_names):
    m_g = (r_test == i)
    y_g = y_test[m_g]
    n_total = len(y_g)
    n_pos = np.sum(y_g == 1)
    n_neg = np.sum(y_g == 0)
    prev = n_pos / n_total * 100
    
    for m_label, p_arr in [("Raw Dual", p_raw), ("LEACE Dual", p_leace), (f"INLP (k={best_k})", p_inlp)]:
        pred = (p_arr[m_g] >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_g, pred, labels=[0,1]).ravel()
        
        flagged = (tp + fp) / n_total
        fl_ci = wilson_interval(tp + fp, n_total)
        tpr_ci = wilson_interval(tp, n_pos)
        fpr_ci = wilson_interval(fp, n_neg)
        auc_g = roc_auc_score(y_g, p_arr[m_g])
        
        fl_str = f"{flagged:4.3f} [{fl_ci[0]:.3f},{fl_ci[1]:.3f}]"
        tpr_str = f"{tp/n_pos:4.3f} [{tpr_ci[0]:.3f},{tpr_ci[1]:.3f}]"
        fpr_str = f"{fp/n_neg:4.3f} [{fpr_ci[0]:.3f},{fpr_ci[1]:.3f}]"
        
        print(f"{g_name:<8} | {m_label:<15} | {n_total:<6d} | {prev:<7.1f}% | {fl_str:<18} | {tpr_str:<22} | {fpr_str:<22} | {auc_g:.4f}")
    print("-" * 115)

# -----------------------------------------------------------------------------
# 12. Save Results to JSON
# -----------------------------------------------------------------------------
save_payload = {
    "cohort_stats": {name: {"n": int(np.sum(r_test == i)), "prev": float(np.mean(y_test[r_test == i]))} for i, name in enumerate(race_names)},
    "probes": {k: {sub_k: float(sub_v) for sub_k, sub_v in v.items()} for k, v in probes_results.items()},
    "validation_tuning": {"best_parity_lambda": best_lam_p, "best_label_cond_lambda": best_lam_l, "best_inlp_k": best_k},
    "table1_metrics": {k: {sub_k: (float(sub_v) if not isinstance(sub_v, list) else [float(x) for x in sub_v]) for sub_k, sub_v in v.items()} for k, v in table1_metrics.items()}
}

with open("unified_final_results.json", "w", encoding="utf-8") as f:
    json.dump(save_payload, f, indent=2)

print("\nSaved all metrics to unified_final_results.json and unified_final_output.txt.")
print("=" * 110)
print("BENCHMARK EXECUTION COMPLETE.")
print("=" * 110)