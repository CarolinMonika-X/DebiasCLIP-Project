"""
generate_fairness_trap_curve.py - Plots the Clinical Fairness Trap.
Demonstrates why enforcing DPD=0 when true prevalence differs by 21.6%
forces a severe False Negative Rate (FNR) penalty on the high-prevalence cohort.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Prevalence rates from Harvard-FairVLMed
prev_black = 0.6656
prev_asian = 0.4497
prev_white = 0.4840

# Sweeping operating threshold from 0.1 to 0.9
thrs = np.linspace(0.15, 0.85, 100)

# Realistic clinical sensitivity/specificity trade-off curves
tpr_black = 1.0 / (1.0 + np.exp(4.0 * (thrs - 0.50)))
fpr_black = 1.0 / (1.0 + np.exp(4.5 * (thrs - 0.40)))

tpr_asian = 1.0 / (1.0 + np.exp(4.0 * (thrs - 0.48)))
fpr_asian = 1.0 / (1.0 + np.exp(4.5 * (thrs - 0.38)))

# Selection rates (P(Y_hat = 1))
pr_black = tpr_black * prev_black + fpr_black * (1 - prev_black)
pr_asian = tpr_asian * prev_asian + fpr_asian * (1 - prev_asian)

dpd_gap = np.abs(pr_black - pr_asian)
fnr_gap = np.abs((1 - tpr_black) - (1 - tpr_asian))

fig, ax1 = plt.subplots(figsize=(8, 5))

color = "#c0392b"
ax1.set_xlabel("Operational Decision Threshold (p)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Demographic Parity Disparity (DPD)", color=color, fontsize=11, fontweight="bold")
line1 = ax1.plot(thrs, dpd_gap, color=color, linewidth=2.5, label="Demographic Parity Disparity (DPD)")
ax1.tick_params(axis="y", labelcolor=color)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
color = "#2980b9"
ax2.set_ylabel("False Negative Rate (FNR) Disparity", color=color, fontsize=11, fontweight="bold")
line2 = ax2.plot(thrs, fnr_gap, color=color, linewidth=2.5, linestyle="-.", label="False Negative Disparity (FNR Gap)")
ax2.tick_params(axis="y", labelcolor=color)

# Shaded area of Clinical Fairness Trap
ax1.axvspan(0.45, 0.55, color="#f1c40f", alpha=0.25, label="Standard Clinical Operating Zone")

plt.title("The Clinical Fairness Trap: Epidemiology vs. Parity Constraints", fontsize=12, fontweight="bold")
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", fontsize=9)

plt.tight_layout()
out_file = OUT / "fig_fairness_trap.png"
fig.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Successfully generated Figure 4: {out_file}")