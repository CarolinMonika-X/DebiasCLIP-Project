"""
make_all_figures.py - every figure and table for the paper/slides, from saved result files.

Inputs (project root):
  ablation_results.json          written by ablation_full.py (required)
  novelty_results_alpha.json     OPTIONAL, exploratory figure (see schema below)
Outputs: ./figures/*.png, results_table.md, results_table.tex

Run: python src/make_all_figures.py
     python src/make_all_figures.py --results other.json --out my_figs

Optional exploratory file schema (values are [mean, std] or a single number):
  {"configs": {"Baseline (alpha=1)": {"acc": [0.70, 0.01], "auc": [..], "dpd": [..], "eod": [..]},
               "Image x0.5": {...}}}
Without that file the exploratory figure is skipped; nothing is hard-coded.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="ablation_results.json")
ap.add_argument("--novelty", default="novelty_results_alpha.json")
ap.add_argument("--out", default="figures")
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(exist_ok=True)

def load(path, required=False):
    p = Path(path)
    if not p.exists():
        msg = f"{path} not found"
        if required: sys.exit(f"ERROR: {msg}. Run ablation_full.py first.")
        print(f"note: {msg} (skipping what needs it)"); return None
    return json.loads(p.read_text(encoding="utf-8"))

J = load(args.results, required=True)
AB, BT, SW, LK = (J.get(k, {}) for k in ("ablation", "bootstrap", "lambda_sweep", "leakage"))
KEYS = list(AB)

LAB = ["A\nImage", "B\nImage +\nparity", "C\nImage + text", "D\nDual +\nparity", "E\nDual +\nlabel-cond"]
COL = ["#4C72B0", "#8FA9D0", "#DD8452", "#E8B38B", "#55A868"]
GREY, PREV_GAP = "#9a9a9a", 0.216  # PREV_GAP = DPD of a perfect classifier (test prevalence 0.450/0.666/0.484)

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "axes.titleweight": "bold"})

def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight"); plt.close(fig); print("saved", OUT / name)

def ms(v):
    """value -> (mean, std or None); accepts a number or [mean, std]."""
    return (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else (float(v), None)

def bars(ax, vals, labels, title, colors, ylim=None):
    m = [ms(v)[0] for v in vals]; s = [ms(v)[1] or 0 for v in vals]
    b = ax.bar(range(len(m)), m, yerr=s if any(s) else None, capsize=4, color=colors)
    span = (ylim[1] - ylim[0]) if ylim else max(m)
    for r, v, e in zip(b, m, s):
        ax.text(r.get_x() + r.get_width() / 2, v + e + 0.01 * span, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(labels, fontsize=9); ax.set_title(title, fontsize=11)
    if ylim: ax.set_ylim(*ylim)

def ab_bars(ax, metric, title, ylim=None):
    bars(ax, [AB[k][metric] for k in KEYS], LAB[:len(KEYS)], title, COL[:len(KEYS)], ylim)

# ---------------- ablation figures ----------------
if AB:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ab_bars(ax, "auc", "Test AUC (mean ± std, 5 seeds)", (0.5, 0.85))
    ax.axhline(0.5, color="black", ls="--", lw=1); ax.text(4.45, 0.505, "chance", ha="right", fontsize=9)
    ax.set_ylabel("AUC (axis starts at chance level)"); save(fig, "fig1_auc.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ab_bars(axs[0], "dpd", "DPD (gap in flagged rate)")
    axs[0].axhline(PREV_GAP, color="black", ls="--", lw=1)
    axs[0].text(4.45, PREV_GAP + 0.004, "perfect classifier = 0.216", ha="right", fontsize=8)
    ab_bars(axs[1], "eod", "EOD (gap in TPR/FPR)"); save(fig, "fig2_fairness.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ab_bars(axs[0], "w_auc", "Worst-group AUC", (0.5, 0.85)); ab_bars(axs[1], "w_acc", "Worst-group accuracy", (0.5, 0.8))
    save(fig, "fig3_worst_group.png")

# ---------------- bootstrap forest ----------------
if BT:
    keys = list(BT); names = [k.split(" (")[0] for k in keys]; y = np.arange(len(keys))[::-1]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, (k, t) in zip(axs, [("auc", "ΔAUC"), ("w_auc", "ΔWorst-group AUC"), ("dpd", "ΔDPD"), ("eod", "ΔEOD")]):
        for yi, key in zip(y, keys):
            m, lo, hi = BT[key][k]; sig = lo > 0 or hi < 0
            ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="o", capsize=3, ms=7, color="#c0392b" if sig else GREY)
        ax.axvline(0, color="black", lw=1); ax.set_title(t, fontsize=10.5)
    axs[0].set_yticks(y); axs[0].set_yticklabels(names)
    fig.suptitle("Bootstrap 95% CI of differences (red = excludes 0; grey = not significant)", fontweight="bold")
    save(fig, "fig4_bootstrap_forest.png")

# ---------------- lambda sweep ----------------
if SW:
    losses = {"parity": ("#4C72B0", "Parity penalty"), "labelcond": ("#55A868", "Label-conditioned penalty")}
    def lams(loss):
        out = []
        for k in SW:
            if k.startswith(loss + "_lam"): out.append(float(k.split("_lam")[1]))
        return sorted(out)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, (m, t) in zip(axs, [("auc", "Test AUC"), ("eod", "Test EOD"), ("dpd", "Test DPD")]):
        for loss, (c, lab) in losses.items():
            L = lams(loss)
            if not L: continue
            vals = [ms(SW[f"{loss}_lam{l:g}"]["test"][m]) for l in L]; xs = np.arange(len(L))
            ax.errorbar(xs + (0.04 if loss == "labelcond" else -0.04), [v[0] for v in vals],
                        yerr=[v[1] or 0 for v in vals], marker="o", capsize=3, color=c, label=lab)
        ax.set_xticks(xs); ax.set_xticklabels([f"{l:g}" for l in L])
        ax.set_xlabel("λ"); ax.set_title(t, fontsize=10.5)
        if m == "dpd": ax.axhline(PREV_GAP, color="black", ls="--", lw=1); ax.text(0.05, PREV_GAP + .004, "perfect classifier 0.216", fontsize=8)
    axs[0].legend(fontsize=8)
    fig.suptitle("Fairness-loss weight: low DPD/EOD at large λ comes with collapsed AUC", fontweight="bold")
    save(fig, "fig5_lambda_sweep.png")

# ---------------- per-group rates (values copied from final_output_v2.txt) ----------------
N = np.array([169, 302, 1529]); PREV = np.array([.450, .666, .484]); NPOS, NNEG = PREV * N, N * (1 - PREV)
PG = {
    "A  Image": [(0.684, 0.269), (0.711, 0.337), (0.685, 0.274)],  # (TPR, FPR) for groups 0, 1, 2
    "C  Image + text": [(0.750, 0.312), (0.766, 0.366), (0.718, 0.253)],
    "E  Dual + label-cond (λ=1)": [(0.724, 0.226), (0.761, 0.337), (0.732, 0.257)]
}
def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (i, nn, t) in zip(axs, [(0, NPOS, "TPR (sick patients detected)"), (1, NNEG, "FPR (healthy patients flagged)")]):
    for j, (nm, v) in enumerate(PG.items()):
        for g in range(3):
            p = v[g][i]; lo, hi = wilson(p, nn[g])
            ax.errorbar(g + (j - 1) * .22, p, yerr=[[p - lo], [hi - p]], fmt="o", capsize=3, color=[COL[0], COL[2], COL[4]][j], label=nm if g == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"Group {g}\n(n={n})" for g, n in enumerate(N)]); ax.set_ylim(0, 1); ax.set_title(t, fontsize=10.5)
axs[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Per-group error rates on test (Wilson 95% CI; wide bars = small group)", fontweight="bold")
save(fig, "fig6_pergroup_rates.png")

# ---------------- feature signal ----------------
if LK:
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, (k, t, ch) in zip(axs, [("glaucoma_auc", "Predicts glaucoma (AUC)", .5), ("race_macro_auc", "Predicts race (macro-AUC)", .5),
                                    ("race_bal_acc", "Predicts race (balanced acc.)", 1 / 3)]):
        v = [LK["Image"][k], LK["Text"][k]]; b = ax.bar(["Image", "Text"], v, color=[COL[0], COL[2]], width=.55)
        ax.axhline(ch, color="black", ls="--", lw=1); ax.text(1.42, ch, "chance", ha="right", va="bottom", fontsize=8)
        for r, x in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, x + .01, f"{x:.3f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1); ax.set_title(t, fontsize=10.5)
    fig.suptitle("Signal in frozen CLIP features. Decodable ≠ used by the classifier", fontweight="bold")
    save(fig, "fig7_feature_signal.png")

# ---------------- exploratory figure, only from a results file ----------------
NV = load(args.novelty)
if NV and NV.get("configs"):
    cfgs = NV["configs"]; names = list(cfgs); cols = ["#9a9a9a"] + ["#4C72B0"] * (len(names) - 1)
    metrics = [("acc", "Accuracy"), ("dpd", "DPD"), ("eod", "EOD")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2)); single_run = False
    for ax, (m, t) in zip(axs, metrics):
        vals = [cfgs[n][m] for n in names]; single_run |= any(ms(v)[1] is None for v in vals)
        bars(ax, vals, names, t, cols)
        for lbl in ax.get_xticklabels(): lbl.set_rotation(15); lbl.set_ha("right")
    fig.suptitle("Exploratory: down-weighting image features (not part of the main ablation)", fontweight="bold")
    if single_run:
        fig.text(0.5, -0.02, "No error bars: single run, no confidence intervals. Do not claim significance.", ha="center", fontsize=9, color="#c0392b")
    save(fig, "fig8_exploratory_alpha.png")

# ---------------- tables ----------------
if AB:
    cols = [("acc", "Acc"), ("auc", "AUC"), ("dpd", "DPD"), ("eod", "EOD"), ("w_auc", "Worst-group AUC")]
    nice = ["A: Image", "B: Image + parity", "C: Image + text", "D: Image + text + parity", "E: Image + text + label-cond."]
    f = lambda k, m: f"{ms(AB[k][m])[0]:.3f} ± {(ms(AB[k][m])[1] or 0):.3f}"
    md = ["| Model | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---|" * len(cols)]
    tex = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", "Model & " + " & ".join(c for _, c in cols) + " \\\\", "\\midrule"]
    for k, n in zip(KEYS, nice):
        md.append(f"| {n} | " + " | ".join(f(k, m) for m, _ in cols) + " |")
        tex.append(n + " & " + " & ".join(f(k, m).replace("±", "$\\pm$") for m, _ in cols) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    Path("results_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("saved results_table.md, results_table.tex")
print("\nDone.")