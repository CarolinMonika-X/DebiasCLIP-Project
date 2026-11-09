"""
run_all.py - Master reproduction script for DebiasCLIP Project.
Reproduces all primary benchmarks, representation geometry, and figure generation.
"""
import subprocess
import sys

steps = [
    ("Primary Evaluation & Hypothesis Testing", "src/erasure_primary.py"),
    ("Latent & Non-Linear Leakage Audit", "src/latent_erasure_audit.py"),
    ("Representation Geometry Audit", "src/representation_geometry.py"),
    ("Saliency Energy-in-ROI Analysis", "src/counterfactual_controlled.py"),
    ("Generating Figure 1 (Architecture)", "src/generate_architecture_diagram.py"),
    ("Generating Figure 2 (Case Studies)", "src/generate_case_studies.py"),
    ("Generating Figure 3 (t-SNE Geometry)", "src/generate_tsne.py"),
    ("Generating Figure 4 (Fairness Trap)", "src/generate_fairness_trap_curve.py"),
]

print("=" * 80)
print("DEBIASCLIP: REPRODUCING COMPLETE RESEARCH AUDIT")
print("=" * 80)

for desc, script in steps:
    print(f"\n[RUNNING] {desc} ({script})...")
    res = subprocess.run([sys.executable, script], capture_output=False)
    if res.returncode != 0:
        print(f"[ERROR] Step failed: {script}")
        sys.exit(1)

print("\n" + "=" * 80)
print("ALL BENCHMARKS AND FIGURES SUCCESSFULLY REPRODUCED!")
print("Figures saved in figures/ | Primary metrics logged in *.txt")
print("=" * 80)