"""
Generate publication figures for Phase 2 and Phase 3 results.
Outputs:
  - paper/figures/phase2_overview.png
  - paper/figures/phase3_overview.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DACS_COL = "#2563EB"
BASE_COL = "#DC2626"


def plot_overview(output_path, title, labels, metrics):
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    panels = [
        ("acc", "Steering accuracy (%)", 110),
        ("cont", "Contamination rate (%)", None),
        ("ctx", "Avg context (tokens)", None),
    ]
    panel_titles = ["(a) Accuracy", "(b) Contamination", "(c) Context size"]

    for ax, (metric_key, ylabel, fixed_ylim), panel_title in zip(axes, panels, panel_titles):
        dacs_mean = np.array(metrics[metric_key]["dacs_mean"])
        dacs_se = np.array(metrics[metric_key]["dacs_se"])
        base_mean = np.array(metrics[metric_key]["base_mean"])
        base_se = np.array(metrics[metric_key]["base_se"])

        ax.bar(x - width / 2, dacs_mean, width, yerr=dacs_se, color=DACS_COL, capsize=4, label="DACS")
        ax.bar(x + width / 2, base_mean, width, yerr=base_se, color=BASE_COL, capsize=4, label="Baseline")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

        if fixed_ylim is not None:
            ax.set_ylim(0, fixed_ylim)
        else:
            ymax = float(max(np.max(dacs_mean + dacs_se), np.max(base_mean + base_se)))
            ax.set_ylim(0, ymax * 1.25)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    output_dir = os.path.join("paper", "figures")
    os.makedirs(output_dir, exist_ok=True)

    phase2_labels = ["s4 (N=3)\nHomogeneous", "s5 (N=5)\nCrossfire", "s6 (N=5)\nCascade"]
    phase2_metrics = {
        "acc": {
            "dacs_mean": [90.2, 96.0, 94.0],
            "dacs_se": [3.5, 0.9, 1.5],
            "base_mean": [52.5, 37.0, 56.7],
            "base_se": [1.7, 1.8, 3.2],
        },
        "cont": {
            "dacs_mean": [0.8, 0.0, 7.3],
            "dacs_se": [0.7, 0.0, 2.2],
            "base_mean": [44.2, 53.0, 28.7],
            "base_se": [3.1, 2.8, 3.4],
        },
        "ctx": {
            "dacs_mean": [815, 911, 705],
            "dacs_se": [10, 6, 10],
            "base_mean": [1869, 2643, 1870],
            "base_se": [49, 33, 38],
        },
    }

    phase3_labels = ["s7 (N=5, D=8)", "s8 (N=3, D=15)"]
    phase3_metrics = {
        "acc": {
            "dacs_mean": [94.0, 98.4],
            "dacs_se": [0.8, 0.5],
            "base_mean": [34.8, 44.2],
            "base_se": [1.8, 1.3],
        },
        "cont": {
            "dacs_mean": [0.2, 0.9],
            "dacs_se": [0.2, 0.9],
            "base_mean": [49.8, 51.6],
            "base_se": [3.3, 2.4],
        },
        "ctx": {
            "dacs_mean": [1654, 2755],
            "dacs_se": [20, 29],
            "base_mean": [5364, 6573],
            "base_se": [98, 153],
        },
    }

    plot_overview(
        os.path.join(output_dir, "phase2_overview.png"),
        "Phase 2: Agent Diversity (10 trials each scenario)",
        phase2_labels,
        phase2_metrics,
    )
    plot_overview(
        os.path.join(output_dir, "phase3_overview.png"),
        "Phase 3: Decision Density Scaling (10 trials each scenario)",
        phase3_labels,
        phase3_metrics,
    )

    print("Saved: paper/figures/phase2_overview.png")
    print("Saved: paper/figures/phase3_overview.png")


if __name__ == "__main__":
    main()
