"""Pareto frontier: token efficiency vs steering accuracy.

Plots all experimental conditions (DACS + baseline × all scenarios) on
a tokens-vs-accuracy scatter and draws the Pareto frontier showing
DACS dominates the efficiency–accuracy tradeoff.

Outputs: paper/figures/pareto_frontier.png

Data sources: hardcoded from Phase 1–3 results (mean per condition).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DACS_COL = "#2563EB"
BASE_COL = "#DC2626"

# ---------------------------------------------------------------------------
# Data: (scenario_label, condition, mean_accuracy, mean_context_tokens)
# All from Phase 1–3 results (10 trials each)
# ---------------------------------------------------------------------------

DATA = [
    # Phase 1: Agent count scaling
    ("s1 N=3",   "dacs",     96.7,  561),
    ("s1 N=3",   "baseline", 60.0,  1191),
    ("s2 N=5",   "dacs",     96.7,  633),
    ("s2 N=5",   "baseline", 38.7,  1720),
    ("s3 N=10",  "dacs",     90.0,  816),
    ("s3 N=10",  "baseline", 21.0,  2882),
    # Phase 2: Agent diversity
    ("s4 Homog",    "dacs",     90.2,  815),
    ("s4 Homog",    "baseline", 52.5,  1869),
    ("s5 Cross",    "dacs",     96.0,  911),
    ("s5 Cross",    "baseline", 37.0,  2643),
    ("s6 Cascade",  "dacs",     94.0,  705),
    ("s6 Cascade",  "baseline", 56.7,  1870),
    # Phase 3: Decision density
    ("s7 D=8",    "dacs",     94.0,  1654),
    ("s7 D=8",    "baseline", 34.8,  5364),
    ("s8 D=15",   "dacs",     98.4,  2755),
    ("s8 D=15",   "baseline", 44.2,  6573),
]


def _pareto_front(points):
    """Return indices of Pareto-optimal points (maximize accuracy, minimize tokens)."""
    pts = np.array(points)  # Nx2: [tokens, accuracy]
    # Sort by tokens ascending
    order = np.argsort(pts[:, 0])
    sorted_pts = pts[order]

    pareto_idx = []
    max_acc = -np.inf
    # Walk from lowest tokens to highest; keep if accuracy improves
    for i, (tok, acc) in enumerate(sorted_pts):
        if acc > max_acc:
            max_acc = acc
            pareto_idx.append(order[i])
    return sorted(pareto_idx, key=lambda i: pts[i, 0])


def plot_pareto():
    """Generate the Pareto frontier figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    dacs_pts = [(d[3], d[2]) for d in DATA if d[1] == "dacs"]
    base_pts = [(d[3], d[2]) for d in DATA if d[1] == "baseline"]
    dacs_labels = [d[0] for d in DATA if d[1] == "dacs"]
    base_labels = [d[0] for d in DATA if d[1] == "baseline"]

    # Scatter
    dacs_tok = [p[0] for p in dacs_pts]
    dacs_acc = [p[1] for p in dacs_pts]
    base_tok = [p[0] for p in base_pts]
    base_acc = [p[1] for p in base_pts]

    ax.scatter(dacs_tok, dacs_acc, c=DACS_COL, marker="o", s=90, zorder=5, label="DACS", edgecolors="white", linewidth=0.5)
    ax.scatter(base_tok, base_acc, c=BASE_COL, marker="s", s=90, zorder=5, label="Baseline", edgecolors="white", linewidth=0.5)

    # Labels
    for (tok, acc), label in zip(dacs_pts, dacs_labels):
        ax.annotate(label, (tok, acc), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color=DACS_COL, alpha=0.8)
    for (tok, acc), label in zip(base_pts, base_labels):
        ax.annotate(label, (tok, acc), textcoords="offset points",
                    xytext=(6, -10), fontsize=7, color=BASE_COL, alpha=0.8)

    # Pareto frontier (all points combined)
    all_pts = list(zip(
        [d[3] for d in DATA],
        [d[2] for d in DATA],
    ))
    pareto_idx = _pareto_front(all_pts)
    pareto_tok = [all_pts[i][0] for i in pareto_idx]
    pareto_acc = [all_pts[i][1] for i in pareto_idx]
    ax.plot(pareto_tok, pareto_acc, "k--", alpha=0.4, linewidth=1.5, label="Pareto frontier")

    # Shade the dominated region
    ax.fill_between(pareto_tok, pareto_acc, 0, alpha=0.05, color="green")

    ax.set_xlabel("Average context tokens (M3)", fontsize=11)
    ax.set_ylabel("Steering accuracy % (M1)", fontsize=11)
    ax.set_title("Token efficiency vs. accuracy: Pareto frontier", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(10, 105)
    ax.set_xlim(0, 7500)

    fig.tight_layout()
    output_dir = os.path.join("paper", "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pareto_frontier.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    plot_pareto()


if __name__ == "__main__":
    main()
