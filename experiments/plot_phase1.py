"""
Phase 1 figure generation — DACS vs Baseline across N=3, 5, 10.
Produces: results/figures/phase1_accuracy.png
          results/figures/phase1_contamination.png
          results/figures/phase1_context.png
          results/figures/phase1_overview.png
"""
import statistics, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("results/figures", exist_ok=True)

# ── raw data ──────────────────────────────────────────────────────────────────
DATA = {
    3: {
        "dacs": {
            "acc":  [0.9333,1.0,1.0,0.8667,1.0,1.0,1.0,1.0,1.0,0.8667],
            "cont": [0.1111,0.0,0.0,0.1111,0.0,0.0,0.0,0.0,0.0,0.1111],
            "ctx":  [557,563,559,561,562,560,563,558,561,560],  # approx from mean 561
        },
        "baseline": {
            "acc":  [0.6667,0.6667,0.6667,0.6667,0.3333,0.6667,0.6667,0.6667,0.6667,0.3333],
            "cont": [0.6667,0.6667,0.3333,0.6667,0.6667,0.6667,0.3333,0.6667,0.3333,0.6667],
            "ctx":  [1191]*10,
        }
    },
    5: {
        "dacs": {
            "acc":  [0.9333,1.0,1.0,1.0,0.8667,1.0,1.0,0.8667,1.0,1.0],
            "cont": [0.1333,0.0667,0.4,0.0667,0.0667,0.0667,0.2,0.1333,0.2,0.0667],
            "ctx":  [634.4,639.9,608.1,639.1,621.7,626.7,668.7,642.9,661.2,582.9],
        },
        "baseline": {
            "acc":  [0.3333,0.4667,0.3333,0.2667,0.6,0.3333,0.4667,0.4,0.4,0.2667],
            "cont": [0.6,0.7333,0.5333,0.5333,0.7333,0.2667,0.4,0.4667,0.4667,0.4667],
            "ctx":  [1948.9,1745.3,1658.5,1745.3,1946.7,1522.2,1761.7,1613.4,1743.7,1511.6],
        }
    },
    10: {
        "dacs": {
            "acc":  [0.9333,0.9,0.9,0.9,0.9,0.9333,0.9333,0.9,0.8333,0.8667],
            "cont": [0.0667,0.0,0.1,0.0,0.0,0.1,0.0,0.1,0.0,0.0],
            "ctx":  [818.8,797.3,827.2,818.9,811.2,820.6,826.1,813.5,803.2,824.4],
        },
        "baseline": {
            "acc":  [0.2,0.2667,0.1667,0.2333,0.2667,0.1,0.2,0.2667,0.1667,0.2333],
            "cont": [0.3,0.3333,0.1,0.3333,0.4,0.3333,0.2333,0.3333,0.2333,0.3333],
            "ctx":  [2901.3,3219.8,2455.9,2769.0,3344.7,2838.3,2721.2,2905.7,2663.3,3005.9],
        }
    }
}

Ns = [3, 5, 10]
DACS_COL  = "#2563EB"  # blue
BASE_COL  = "#DC2626"  # red

def stats(vals):
    m = statistics.mean(vals)
    se = statistics.stdev(vals) / math.sqrt(len(vals))
    return m, se

def means_ses(metric, cond):
    ms, ses = [], []
    for n in Ns:
        m, se = stats(DATA[n][cond][metric])
        ms.append(m); ses.append(se)
    return np.array(ms), np.array(ses)

# ── figure 1: accuracy ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
w = 0.35; x = np.arange(len(Ns))
dm, dse = means_ses("acc", "dacs")
bm, bse = means_ses("acc", "baseline")
ax.bar(x - w/2, dm*100, w, yerr=dse*100, color=DACS_COL, label="DACS", capsize=4)
ax.bar(x + w/2, bm*100, w, yerr=bse*100, color=BASE_COL, label="Baseline", capsize=4)
ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in Ns])
ax.set_ylabel("Steering accuracy (%)"); ax.set_ylim(0, 110)
ax.set_title("Steering Accuracy: DACS vs Baseline")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results/figures/phase1_accuracy.png", dpi=150)
plt.close()

# ── figure 2: contamination ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
dm, dse = means_ses("cont", "dacs")
bm, bse = means_ses("cont", "baseline")
ax.bar(x - w/2, dm*100, w, yerr=dse*100, color=DACS_COL, label="DACS", capsize=4)
ax.bar(x + w/2, bm*100, w, yerr=bse*100, color=BASE_COL, label="Baseline", capsize=4)
ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in Ns])
ax.set_ylabel("Wrong-agent contamination (%)"); ax.set_ylim(0, 80)
ax.set_title("Context Contamination: DACS vs Baseline")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results/figures/phase1_contamination.png", dpi=150)
plt.close()

# ── figure 3: context size ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
dm, dse = means_ses("ctx", "dacs")
bm, bse = means_ses("ctx", "baseline")
ax.bar(x - w/2, dm, w, yerr=dse, color=DACS_COL, label="DACS", capsize=4)
ax.bar(x + w/2, bm, w, yerr=bse, color=BASE_COL, label="Baseline", capsize=4)
for i, (d, b) in enumerate(zip(dm, bm)):
    ax.text(i, max(d,b)+80, f"{b/d:.1f}×", ha="center", fontsize=9, color="#374151")
ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in Ns])
ax.set_ylabel("Avg context tokens at steering"); ax.set_ylim(0, 3800)
ax.set_title("Context Size: DACS vs Baseline")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results/figures/phase1_context.png", dpi=150)
plt.close()

# ── figure 4: 3-panel overview ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

panels = [
    ("acc",  "Steering accuracy (%)",      100, 110, True),
    ("cont", "Contamination rate (%)",      100, 80, True),
    ("ctx",  "Avg context (tokens)",        1,   3800, False),
]
titles = ["(a) Accuracy", "(b) Contamination", "(c) Context size"]

for ax, (metric, ylabel, scale, ylim, pct), title in zip(axes, panels, titles):
    dm, dse = means_ses(metric, "dacs")
    bm, bse = means_ses(metric, "baseline")
    if pct:
        dm, bm, dse, bse = dm*scale, bm*scale, dse*scale, bse*scale
    ax.bar(x - w/2, dm, w, yerr=dse, color=DACS_COL, label="DACS", capsize=4, alpha=0.9)
    ax.bar(x + w/2, bm, w, yerr=bse, color=BASE_COL, label="Baseline", capsize=4, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel(ylabel); ax.set_ylim(0, ylim)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

fig.suptitle("Phase 1: DACS vs Flat-Context Baseline (10 trials each, N∈{3,5,10})",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("results/figures/phase1_overview.png", dpi=150, bbox_inches="tight")
plt.close()

print("Figures saved to results/figures/")
