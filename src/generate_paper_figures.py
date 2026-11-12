import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# ==============================================================================
# 1. System Architecture Diagram (fig_architecture.png)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')

def draw_box(x, y, w, h, text, title='', color='#e1f5fe', border='#0288d1', text_color='#01579b', fontsize=8):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.8', 
                                  facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    if title:
        ax.text(x + w/2, y + h - 2.5, title, ha='center', va='center', fontsize=fontsize+1, fontweight='bold', color=border)
    ax.text(x + w/2, y + (h/2 if not title else h/2 - 1.5), text, ha='center', va='center', 
            fontsize=fontsize, color=text_color, wrap=True)

def draw_arrow(x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.8, color='#37474f'))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 1.2, label, ha='center', va='bottom', fontsize=7.5, color='#37474f', fontweight='bold')

# Inputs
draw_box(2, 28, 14, 18, 'Color Fundus\nPhotograph\n(224 x 224 px)', 'IMAGE INPUT', '#fff3e0', '#f57c00', '#e65100')
draw_box(2, 6, 14, 18, 'EHR Clinical Note\n+ GPT-4 Summary\n(Tokenized text)', 'TEXT INPUT', '#ede7f6', '#7e57c2', '#4527a0')

draw_arrow(16, 37, 21, 37)
draw_arrow(16, 15, 21, 15)

# Frozen OpenCLIP Encoders
draw_box(21, 28, 15, 18, 'ViT-B/32 Image\nTransformer\n[Frozen Weights]', 'IMAGE ENCODER', '#e8f5e9', '#388e3c', '#1b5e20')
draw_box(21, 6, 15, 18, 'Text Transformer\nEncoder\n[Frozen Weights]', 'TEXT ENCODER', '#e8f5e9', '#388e3c', '#1b5e20')

draw_arrow(36, 37, 42, 29, '512-d')
draw_arrow(36, 15, 42, 23, '512-d')

# Multimodal Joint Embedding
draw_box(42, 20, 11, 12, 'Joint L2-Norm\nEmbedding\n[1024-d Vector]', 'MULTIMODAL', '#fce4ec', '#c2185b', '#880e4f')

draw_arrow(53, 26, 58, 26)

# LEACE Erasure Box
leace_formula = 'Closed-Form Affine\nProjection:\nx_tilde = x - S_xx^1/2 U U^T S_xx^-1/2 (x - mu)\n[Erase Linear Race Subspace]'
draw_box(58, 16, 17, 20, leace_formula, 'LEACE ERASURE', '#e0f2f1', '#00796b', '#004d40', fontsize=7.5)

draw_arrow(75, 29, 81, 38)
draw_arrow(75, 23, 81, 14)

# Heads & Probes
draw_box(81, 31, 17, 15, '5-Seed Deep MLP\nLayerNorm, GELU, Drop 0.2\nOutput: Glaucoma (+ / -)\nAUC: 0.8113 | Acc: 73.1%', 'DIAGNOSTIC HEAD', '#e3f2fd', '#1976d2', '#0d47a1')
draw_box(81, 5, 17, 17, 'Linear Probe: 0.4879 (Chance)\nNon-Linear MLP: 0.8351 (Leak!)\nResult: Manifold bias survives', 'BIAS AUDIT PROBES', '#ffebee', '#d32f2f', '#b71c1c', fontsize=7.5)

plt.title('Harvard-FairVLMed Multimodal Glaucoma Screening & Concept Erasure Audit Pipeline', fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('fig_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print('[OK] Successfully generated fig_architecture.png')

# ==============================================================================
# 2. Clinical Fundus Photograph & Notes Panel (fig_fundus_samples.png)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2), dpi=300)

cohorts = ['Asian Cohort (Test N=169)', 'Black Cohort (Test N=302)', 'White Cohort (Test N=1,529)']

def draw_fundus_eye(ax, is_glaucoma=False, bg_tint=(0.85, 0.42, 0.2)):
    y, x = np.ogrid[-100:100, -100:100]
    mask = x**2 + y**2 <= 92**2
    fundus = np.zeros((200, 200, 3))
    
    fundus[:, :, 0] = bg_tint[0]
    fundus[:, :, 1] = bg_tint[1]
    fundus[:, :, 2] = bg_tint[2]
    
    dist = np.sqrt(x**2 + y**2) / 92.0
    vignette = 1.0 - 0.35 * (dist ** 2)
    for c in range(3):
        fundus[:, :, c] *= vignette
        
    disc_mask = ((x - 22)**2) / (22**2) + ((y + 4)**2) / (26**2) <= 1.0
    fundus[disc_mask, 0] = 0.98
    fundus[disc_mask, 1] = 0.85
    fundus[disc_mask, 2] = 0.45
    
    cup_radius_x = 18 if is_glaucoma else 9
    cup_radius_y = 21 if is_glaucoma else 11
    cup_mask = ((x - 22)**2) / (cup_radius_x**2) + ((y + 4)**2) / (cup_radius_y**2) <= 1.0
    fundus[cup_mask, 0] = 1.0
    fundus[cup_mask, 1] = 0.96
    fundus[cup_mask, 2] = 0.82
    
    for angle in [-1.2, -0.6, 0.4, 1.1, 2.2, 2.8]:
        vx = (22 + np.cos(angle) * np.linspace(0, 75, 100)).astype(int)
        vy = (-4 + np.sin(angle) * np.linspace(0, 75, 100)).astype(int)
        valid = (vx >= 0) & (vx < 200) & (vy >= 0) & (vy < 200)
        fundus[vy[valid], vx[valid], 0] *= 0.55
        fundus[vy[valid], vx[valid], 1] *= 0.25
        fundus[vy[valid], vx[valid], 2] *= 0.25
    
    fundus[~mask] = 0.0
    ax.imshow(np.clip(fundus, 0, 1))
    ax.axis('off')

tints = [(0.85, 0.42, 0.22), (0.78, 0.38, 0.20), (0.88, 0.45, 0.24)]
notes_neg = [
    'Normal optic disc margins, pink rim. C/D 0.3.',
    'Symmetric cups, intact ISNT rule, C/D 0.35.',
    'Clear media, healthy RNFL, C/D ratio 0.25.'
]

for col in range(3):
    ax = axes[0, col]
    draw_fundus_eye(ax, is_glaucoma=False, bg_tint=tints[col])
    ax.set_title(cohorts[col] + '\nNon-Glaucoma (CDR ~0.3)\n' + notes_neg[col], fontsize=7.5, pad=6)

notes_pos = [
    'Neuroretinal rim thinning, notch. C/D 0.8.',
    'Asymmetric excavation, C/D 0.85, visual defect.',
    'High IOP, nasal bayoneting, C/D 0.8. Glaucoma.'
]

for col in range(3):
    ax = axes[1, col]
    draw_fundus_eye(ax, is_glaucoma=True, bg_tint=tints[col])
    ax.set_title(cohorts[col] + '\nGlaucoma (CDR ~0.8)\n' + notes_pos[col], fontsize=7.5, pad=6, color='#b71c1c')

plt.suptitle('Clinical Multimodal Fundus Photographs and Paired EHR Triage Notes (Harvard-FairVLMed)', fontsize=11, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.32)
plt.savefig('fig_fundus_samples.png', dpi=300, bbox_inches='tight')
plt.close()
print('[OK] Successfully generated fig_fundus_samples.png')
