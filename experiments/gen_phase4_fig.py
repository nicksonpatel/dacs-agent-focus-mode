"""Generate Phase 4 comparison figure for paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Colour palette ────────────────────────────────────────────────────────────
c_dacs_p1 = '#2166ac'   # dark blue   – DACS synthetic
c_base_p1 = '#d73027'   # dark red    – Baseline synthetic
c_dacs_p4 = '#74add1'   # light blue  – DACS real agents
c_base_p4 = '#f46d43'   # light orange – Baseline real agents

# ── Data ──────────────────────────────────────────────────────────────────────
dacs_p1    = [96.7, 96.7]
base_p1    = [60.0, 38.7]
se_dacs_p1 = [1.5,  1.4]
se_base_p1 = [3.1,  4.2]

dacs_p4    = [79.8, 83.7]
base_p4    = [62.6, 63.3]
se_dacs_p4 = [10.7 / np.sqrt(10), 10.6 / np.sqrt(10)]
se_base_p4 = [13.6 / np.sqrt(10), 14.1 / np.sqrt(10)]

ctx_dacs_p1 = [559,  633]
ctx_base_p1 = [1196, 1720]
ctx_dacs_p4 = [654,  799]
ctx_base_p4 = [1361, 2275]

x_labels = ['N = 3', 'N = 5']

# ── Layout ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
fig.subplots_adjust(bottom=0.20, wspace=0.34)

x = np.arange(2)

# Bar geometry — all positions are BAR CENTRES (matplotlib convention).
# 4 bars per group:  [DACS-Ph1] [Base-Ph1]  <gap>  [DACS-Ph4] [Base-Ph4]
w  = 0.18   # bar width
g1 = 0.05   # gap between bars within the same paradigm pair
g2 = 0.14   # gap between the Ph1 pair and Ph4 pair (visual separator)

# Centres relative to the group centre (x):
#   DACS-Ph1:  x  -  g2/2  -  g1  -  1.5*w
#   Base-Ph1:  x  -  g2/2  -  0.5*w
#   DACS-Ph4:  x  +  g2/2  +  0.5*w
#   Base-Ph4:  x  +  g2/2  +  g1   +  1.5*w
pos_d1 = x - g2/2 - g1 - 1.5*w
pos_b1 = x - g2/2 - 0.5*w
pos_d4 = x + g2/2 + 0.5*w
pos_b4 = x + g2/2 + g1 + 1.5*w

# ── Panel (a) — M1 accuracy ───────────────────────────────────────────────────
ax = axes[0]
ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
ax.set_axisbelow(True)

b_d1 = ax.bar(pos_d1, dacs_p1, w, color=c_dacs_p1, label='DACS — Ph1 (synthetic)',
              yerr=se_dacs_p1, capsize=3.5, error_kw={'linewidth': 1.1}, zorder=3)
b_b1 = ax.bar(pos_b1, base_p1, w, color=c_base_p1, label='Baseline — Ph1 (synthetic)',
              yerr=se_base_p1, capsize=3.5, error_kw={'linewidth': 1.1}, zorder=3)
b_d4 = ax.bar(pos_d4, dacs_p4, w, color=c_dacs_p4, label='DACS — Ph4 (real agents)',
              yerr=se_dacs_p4, capsize=3.5, error_kw={'linewidth': 1.1},
              hatch='//', edgecolor='white', linewidth=0.5, zorder=3)
b_b4 = ax.bar(pos_b4, base_p4, w, color=c_base_p4, label='Baseline — Ph4 (real agents)',
              yerr=se_base_p4, capsize=3.5, error_kw={'linewidth': 1.1},
              hatch='//', edgecolor='white', linewidth=0.5, zorder=3)

# Δ annotations placed above the taller of each DACS/Base pair, with ample clearance
for i in range(2):
    # Ph1 gap annotation — centred between Ph1 pair
    cx1 = (pos_d1[i] + pos_b1[i]) / 2
    top1 = max(dacs_p1[i] + se_dacs_p1[i], base_p1[i] + se_base_p1[i]) + 3.5
    ax.text(cx1, top1, f'+{dacs_p1[i]-base_p1[i]:.1f} pp',
            ha='center', va='bottom', fontsize=8, color='#222222',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

    # Ph4 gap annotation — centred between Ph4 pair
    cx4 = (pos_d4[i] + pos_b4[i]) / 2
    top4 = max(dacs_p4[i] + se_dacs_p4[i], base_p4[i] + se_base_p4[i]) + 3.5
    ax.text(cx4, top4, f'+{dacs_p4[i]-base_p4[i]:.1f} pp',
            ha='center', va='bottom', fontsize=8, color='#555555',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel('M1 Steering Accuracy (%)', fontsize=9)
ax.set_ylim(0, 122)
ax.set_title('(a) Steering Accuracy', fontsize=10, pad=6)



# ── Panel (b) — M3 context tokens ────────────────────────────────────────────
ax2 = axes[1]
ax2.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
ax2.set_axisbelow(True)

ax2.bar(pos_d1, ctx_dacs_p1, w, color=c_dacs_p1, zorder=3)
ax2.bar(pos_b1, ctx_base_p1, w, color=c_base_p1, zorder=3)
ax2.bar(pos_d4, ctx_dacs_p4, w, color=c_dacs_p4,
        hatch='//', edgecolor='white', linewidth=0.5, zorder=3)
ax2.bar(pos_b4, ctx_base_p4, w, color=c_base_p4,
        hatch='//', edgecolor='white', linewidth=0.5, zorder=3)

# Efficiency ratio labels: one per paradigm pair, centred above the taller bar
for i in range(2):
    ratio_p1 = ctx_base_p1[i] / ctx_dacs_p1[i]
    ratio_p4 = ctx_base_p4[i] / ctx_dacs_p4[i]
    cx1 = (pos_d1[i] + pos_b1[i]) / 2
    cx4 = (pos_d4[i] + pos_b4[i]) / 2
    top1 = ctx_base_p1[i] + 55
    top4 = ctx_base_p4[i] + 55
    ax2.text(cx1, top1, f'{ratio_p1:.2f}×',
             ha='center', va='bottom', fontsize=8.5, color='#222222',
             bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
    ax2.text(cx4, top4, f'{ratio_p4:.2f}×',
             ha='center', va='bottom', fontsize=8.5, color='#555555',
             bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))



ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, fontsize=10)
ax2.set_ylabel('Avg Context Tokens at Steering Time (M3)', fontsize=9)
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.18)
ax2.set_title('(b) Context Efficiency', fontsize=10, pad=6)

# ── Shared legend below both panels ──────────────────────────────────────────
patches = [
    mpatches.Patch(color=c_dacs_p1, label='DACS — Phase 1 (synthetic stubs)'),
    mpatches.Patch(color=c_base_p1, label='Baseline — Phase 1 (synthetic stubs)'),
    mpatches.Patch(color=c_dacs_p4, hatch='//', label='DACS — Phase 4 (real LLM agents)'),
    mpatches.Patch(color=c_base_p4, hatch='//', label='Baseline — Phase 4 (real LLM agents)'),
]
fig.legend(handles=patches, loc='lower center', ncol=4,
           fontsize=8, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

out_path = os.path.join(os.path.dirname(__file__),
                        '..', 'paper', 'figures', 'phase4_comparison.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved {os.path.realpath(out_path)}")
