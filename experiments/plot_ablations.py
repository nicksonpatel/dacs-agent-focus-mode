"""Ablation study figure generation.

Reads results from ablation trials and produces a grouped bar chart comparing
DACS vs. each ablation condition vs. baseline.

Usage
-----
    python -m experiments.plot_ablations [--csv results/ablation_summary.csv]

Outputs: paper/figures/ablations.png

Expected ablation conditions:
  - dacs           (full DACS)
  - no_registry    (Focus without compressed registry)
  - random_focus   (Focus on random agent)
  - flat_ordered   (Flat context, requesting agent first)
  - baseline       (standard flat context)
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _group_by(rows, key_fields):
    """Group rows by (scenario, condition) → list of metric dicts."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        groups[key].append(row)
    return groups


def plot_ablations(csv_path: str, output_path: str | None = None):
    rows = _load_csv(csv_path)
    if not rows:
        print(f"No data in {csv_path}")
        return

    groups = _group_by(rows, ["scenario", "condition"])

    # Figure out which scenarios and conditions are present
    scenarios = sorted({row["scenario"] for row in rows})
    conditions = sorted({row["condition"] for row in rows})

    # Desired display order (if present)
    ORDER = ["dacs", "no_registry", "flat_ordered", "random_focus", "baseline"]
    conditions = [c for c in ORDER if c in conditions] + [
        c for c in conditions if c not in ORDER
    ]

    COLORS = {
        "dacs":         "#2563EB",
        "no_registry":  "#7C3AED",
        "flat_ordered": "#F59E0B",
        "random_focus": "#EF4444",
        "baseline":     "#DC2626",
    }
    LABELS = {
        "dacs":         "DACS (full)",
        "no_registry":  "No registry",
        "flat_ordered": "Flat ordered",
        "random_focus": "Random focus",
        "baseline":     "Baseline (flat)",
    }

    n_scenarios = len(scenarios)
    n_conditions = len(conditions)
    x = np.arange(n_scenarios)
    total_width = 0.8
    bar_width = total_width / n_conditions

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("steering_accuracy", "Steering accuracy (%)", "(a) Accuracy"),
        ("contamination_rate", "Contamination rate (%)", "(b) Contamination"),
    ]):
        ax = axes[ax_idx]
        for i, cond in enumerate(conditions):
            means, ses = [], []
            for scenario in scenarios:
                group = groups.get((scenario, cond), [])
                vals = [float(r[metric]) * 100 for r in group]
                if vals:
                    m = statistics.mean(vals)
                    se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0
                else:
                    m, se = 0, 0
                means.append(m)
                ses.append(se)

            color = COLORS.get(cond, "#6B7280")
            label = LABELS.get(cond, cond)
            offset = (i - n_conditions / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width * 0.9, yerr=ses,
                   color=color, capsize=3, label=label, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        if metric == "steering_accuracy":
            ax.set_ylim(0, 110)

    fig.suptitle("Ablation study: component contributions to DACS", fontsize=12, y=1.02)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join("paper", "figures", "ablations.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ablation results")
    parser.add_argument("--csv", default="results/ablation_summary.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    plot_ablations(args.csv, args.output)


if __name__ == "__main__":
    main()
