"""
make_all_figures.py - ONE script for every figure and table in the paper/slides.
All values come from ablation_results.json (written by ablation_full.py), except the
per-group table in Fig 6, which is copied from the 'Per-group breakdown' block of final_output.txt.

Run from the project root:   python src/make_all_figures.py
Needs only matplotlib + numpy.  Output: ./figures/*.png, results_table.md, results_table.tex
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
J = json.load(open("ablation_results.json"))
AB, BT, SW, LK = J["ablation"], J["bootstrap"], J["lambda_sweep"], J["leakage"]
KEYS = list(AB.keys())
LAB = ["A\nImage", "B\nImage +\nparity", "C\nImage + text", "D\nDual +\nparity", "E\nDual +\nlabel-cond"]
COL = ["#4C72B0", "#8FA9D0", "#DD8452", "#E8B38B", "#55A868"]
GREY, PREV_GAP = "#9a9a9a", 0.216          # PREV_GAP = DPD of a perfect classifier (test prevalence 0.450/0.666/0.484)

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "axes.titleweight": "bold"})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=300, bbox_inches="tight"); plt.close(fig); print("saved", name)

def bars(ax, metric, title, ylim=None):
    m = [AB[k][metric][0] for k in KEYS]; s = [AB[k][metric][1] for k in KEYS]
    b = ax.bar(range(5), m, yerr=s, capsize=4, color=COL)
    top = (ylim[1] - ylim[0]) if ylim else max(m)
    for r, v, e in zip(b, m, s):
        ax.text(r.get_x() + r.get_width() / 2, v + e + 0.01 * top, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(LAB, fontsize=9); ax.set_title(title, fontsize=11)
    if ylim: ax.set_ylim(*ylim)

# ---- Fig 1: AUC ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
bars(ax, "auc", "Test AUC (mean ± std, 5 seeds)", (0.5, 0.85))
ax.axhline(0.5, color="black", ls="--", lw=1); ax.text(4.45, 0.505, "chance", ha="right", fontsize=9)
ax.set_ylabel("AUC (axis starts at chance level)")
save(fig, "fig1_auc.png")

# ---- Fig 2: DPD and EOD --------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
bars(axs[0], "dpd", "DPD (gap in flagged rate)")
axs[0].axhline(PREV_GAP, color="black", ls="--", lw=1)
axs[0].text(4.45, PREV_GAP + 0.004, "perfect classifier = 0.216", ha="right", fontsize=8)
bars(axs[1], "eod", "EOD (gap in TPR/FPR)")
save(fig, "fig2_fairness.png")

# ---- Fig 3: worst-group ---------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
bars(axs[0], "w_auc", "Worst-group AUC", (0.5, 0.85)); bars(axs[1], "w_acc", "Worst-group accuracy", (0.5, 0.8))
save(fig, "fig3_worst_group.png")

# ---- Fig 4: bootstrap forest ---------------------------------------------
bk = list(BT.keys()); names = [k.split("  (")[0] for k in bk]
fig, axs = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True); y = np.arange(len(bk))[::-1]
for ax, (k, t) in zip(axs, [("auc", "ΔAUC"), ("w_auc", "ΔWorst-group AUC"), ("dpd", "ΔDPD"), ("eod", "ΔEOD")]):
    for yi, key in zip(y, bk):
        m, lo, hi = BT[key][k]; sig = lo > 0 or hi < 0
        ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="o", capsize=3, ms=7, color="#c0392b" if sig else GREY)
    ax.axvline(0, color="black", lw=1); ax.set_title(t, fontsize=10.5)
axs[0].set_yticks(y); axs[0].set_yticklabels(names)
fig.suptitle("Bootstrap 95% CI of differences (red = excludes 0; grey = not significant)", fontweight="bold")
save(fig, "fig4_bootstrap_forest.png")

# ---- Fig 5: lambda sweep (test) ------------------------------------------
lams = [0, 0.1, 1, 5, 10]; xs = np.arange(5)
st = {"parity": ("#4C72B0", "Parity penalty"), "labelcond": ("#55A868", "Label-conditioned penalty")}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, (k, t) in zip(axs, [("auc", "Test AUC"), ("eod", "Test EOD"), ("dpd", "Test DPD")]):
    for loss, (c, lab) in st.items():
        m = [SW[f"{loss}_lam{l}"]["test"][k][0] for l in lams]; s = [SW[f"{loss}_lam{l}"]["test"][k][1] for l in lams]
        ax.errorbar(xs + (0.04 if loss == "labelcond" else -0.04), m, yerr=s, marker="o", capsize=3, color=c, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([str(l) for l in lams]); ax.set_xlabel("λ"); ax.set_title(t, fontsize=10.5)
    if k == "dpd": ax.axhline(PREV_GAP, color="black", ls="--", lw=1); ax.text(0.05, PREV_GAP + .004, "perfect classifier 0.216", fontsize=8)
axs[0].legend(fontsize=8)
fig.suptitle("Fairness-loss weight: low DPD/EOD at λ ≥ 5 comes with collapsed AUC", fontweight="bold")
save(fig, "fig5_lambda_sweep.png")

# ---- Fig 6: per-group TPR/FPR with Wilson 95% CI -------------------------
N = np.array([169, 302, 1529]); PREV = np.array([.450, .666, .484]); NPOS, NNEG = PREV * N, N * (1 - PREV)
PG = {"A  Image": [(.461, .183), (.542, .178), (.449, .180)],
      "C  Image + text": [(.671, .226), (.701, .307), (.673, .248)],
      "E  Dual + label-cond (λ=1)": [(.671, .247), (.786, .396), (.709, .307)]}
PC = dict(zip(PG, [COL[0], COL[2], COL[4]]))
def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (i, nn, t) in zip(axs, [(0, NPOS, "TPR (sick patients detected)"), (1, NNEG, "FPR (healthy patients flagged)")]):
    for j, (nm, v) in enumerate(PG.items()):
        for g in range(3):
            p = v[g][i]; lo, hi = wilson(p, nn[g])
            ax.errorbar(g + (j - 1) * .22, p, yerr=[[p - lo], [hi - p]], fmt="o", capsize=3, color=PC[nm], label=nm if g == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Group 0\n(n=169)", "Group 1\n(n=302)", "Group 2\n(n=1529)"]); ax.set_ylim(0, 1); ax.set_title(t, fontsize=10.5)
axs[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Per-group error rates on test (Wilson 95% CI; wide bars = small group)", fontweight="bold")
save(fig, "fig6_pergroup_rates.png")

# ---- Fig 7: feature signal with chance levels ----------------------------
fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
for ax, (k, t, ch) in zip(axs, [("glaucoma_auc", "Predicts glaucoma (AUC)", .5), ("race_macro_auc", "Predicts race (macro-AUC)", .5),
                               ("race_bal_acc", "Predicts race (balanced acc.)", 1 / 3)]):
    v = [LK["Image"][k], LK["Text"][k]]; b = ax.bar(["Image", "Text"], v, color=[COL[0], COL[2]], width=.55)
    ax.axhline(ch, color="black", ls="--", lw=1); ax.text(1.42, ch, "chance", ha="right", va="bottom", fontsize=8)
    for r, x in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, x + .01, f"{x:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1); ax.set_title(t, fontsize=10.5)
fig.suptitle("Signal in frozen CLIP features (logistic probes). Decodable ≠ used by the classifier", fontweight="bold")
save(fig, "fig7_feature_signal.png")

# ---- Tables (markdown + LaTeX) -------------------------------------------
cols = [("acc", "Acc"), ("auc", "AUC"), ("dpd", "DPD"), ("eod", "EOD"), ("w_auc", "Worst-group AUC")]
nice = ["A: Image", "B: Image + parity", "C: Image + text", "D: Image + text + parity", "E: Image + text + label-cond."]
f = lambda k, m: f"{AB[k][m][0]:.3f} ± {AB[k][m][1]:.3f}"
md = ["| Model | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---|" * len(cols)]
tex = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", "Model & " + " & ".join(c for _, c in cols) + " \\\\", "\\midrule"]
for k, n in zip(KEYS, nice):
    md.append(f"| {n} | " + " | ".join(f(k, m) for m, _ in cols) + " |")
    tex.append(n + " & " + " & ".join(f(k, m).replace("±", "$\\pm$") for m, _ in cols) + " \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
open("results_table.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
open("results_table.tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")
print("saved results_table.md, results_table.tex\nDone.")
