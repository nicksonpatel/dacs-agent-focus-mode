"""Analysis and visualisation for the concurrency & interruption experiment.

Reads results_concurrency/concurrency_summary.csv and produces three figures:

Figure 1 — condition_comparison.pdf
    Bar chart: avg_judge_score (with error bars = std) grouped by condition.
    Four bars per scenario: dacs_clean, dacs_concurrent, baseline_clean,
    baseline_concurrent. Primary figure for the paper section.

Figure 2 — score_distribution.pdf
    Box + scatter (strip) plot: distribution of avg_judge_score per condition.
    Shows spread and low-outlier runs. Useful for per-trial variance analysis.

Figure 3 — score_vs_contention.pdf
    Scatter: competing_requests (x) vs avg_judge_score (y), coloured by
    condition (DACS vs baseline). Illustrates whether queue contention
    degrades quality differentially between conditions.

Usage
-----
    python -m experiments_concurrency.analyze
    python -m experiments_concurrency.analyze --csv path/to/custom.csv

Output
------
    results_concurrency/figures/condition_comparison.pdf
    results_concurrency/figures/score_distribution.pdf
    results_concurrency/figures/score_vs_contention.pdf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works in headless environments
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


_DEFAULT_CSV = "results_concurrency/concurrency_summary.csv"
_FIGURES_DIR = Path("results_concurrency/figures")

# Colour palette (DACS = blue family, baseline = orange family)
_CONDITION_COLOURS = {
    "dacs_clean":            "#4878d0",
    "dacs_concurrent":       "#1a3d7a",
    "baseline_clean":        "#ee854a",
    "baseline_concurrent":   "#8b3a00",
}
_CONDITION_LABELS = {
    "dacs_clean":            "DACS clean",
    "dacs_concurrent":       "DACS concurrent",
    "baseline_clean":        "Baseline clean",
    "baseline_concurrent":   "Baseline concurrent",
}
_CONDITION_ORDER = [
    "dacs_clean",
    "dacs_concurrent",
    "baseline_clean",
    "baseline_concurrent",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                row["avg_judge_score"]    = float(row.get("avg_judge_score", 0) or 0)
                row["avg_steering_score"] = float(row.get("avg_steering_score", 0) or 0)
                row["avg_user_score"]     = float(row.get("avg_user_score", 0) or 0)
                row["competing_requests"] = int(row.get("competing_requests", 0) or 0)
                row["steering_accuracy"]  = float(row.get("steering_accuracy", 0) or 0)
                row["inject_count"]       = int(row.get("inject_count", 0) or 0)
                rows.append(row)
            except (ValueError, KeyError):
                continue  # skip malformed rows
    return rows


def _by_condition(rows: list[dict]) -> dict[str, list[float]]:
    """Group avg_judge_score values by condition string."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        cond = row.get("condition", "unknown")
        grouped.setdefault(cond, []).append(row["avg_judge_score"])
    return grouped


def _by_scenario_condition(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Group avg_judge_score by (scenario, condition)."""
    result: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        sc   = row.get("scenario", "unknown")
        cond = row.get("condition", "unknown")
        result.setdefault(sc, {}).setdefault(cond, []).append(row["avg_judge_score"])
    return result


# ---------------------------------------------------------------------------
# Figure 1 — condition_comparison bar chart
# ---------------------------------------------------------------------------

def plot_condition_comparison(rows: list[dict], out_dir: Path) -> None:
    """Bar chart: mean ± std of avg_judge_score per condition, per scenario."""
    grouped = _by_scenario_condition(rows)
    scenarios = sorted(grouped.keys())

    present_conditions = set()
    for sc in grouped.values():
        present_conditions.update(sc.keys())
    conditions = [c for c in _CONDITION_ORDER if c in present_conditions]

    n_sc    = len(scenarios)
    n_cond  = len(conditions)
    x       = np.arange(n_sc)
    bar_w   = 0.8 / n_cond

    fig, ax = plt.subplots(figsize=(max(6, 3 * n_sc), 5))

    for i, cond in enumerate(conditions):
        means = []
        stds  = []
        for sc in scenarios:
            vals = grouped.get(sc, {}).get(cond, [])
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        offset = (i - n_cond / 2 + 0.5) * bar_w
        colour = _CONDITION_COLOURS.get(cond, "grey")
        label  = _CONDITION_LABELS.get(cond, cond)

        ax.bar(
            x + offset, means,
            width=bar_w,
            color=colour,
            label=label,
            yerr=stds,
            capsize=4,
            alpha=0.88,
        )

    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Avg Judge Score (1–10)", fontsize=12)
    ax.set_title("Inline LLM Judge Score by Condition", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 10.5)
    ax.axhline(y=5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "condition_comparison.pdf"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  → saved {path}")


# ---------------------------------------------------------------------------
# Figure 2 — score_distribution box + strip plot
# ---------------------------------------------------------------------------

def plot_score_distribution(rows: list[dict], out_dir: Path) -> None:
    """Box plot (with individual points overlaid) per condition."""
    by_cond = _by_condition(rows)
    present_conditions = [c for c in _CONDITION_ORDER if c in by_cond]

    data   = [by_cond[c] for c in present_conditions]
    labels = [_CONDITION_LABELS.get(c, c) for c in present_conditions]
    colours = [_CONDITION_COLOURS.get(c, "grey") for c in present_conditions]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(present_conditions)), 5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4},
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    # Overlay individual data points (strip)
    rng = np.random.default_rng(42)
    for i, (vals, colour) in enumerate(zip(data, colours), start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=colour,
            s=20,
            alpha=0.7,
            zorder=3,
        )

    ax.set_xticks(range(1, len(present_conditions) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Avg Judge Score (1–10)", fontsize=12)
    ax.set_title("Score Distribution per Condition", fontsize=14)
    ax.set_ylim(0, 10.5)
    ax.grid(axis="y", alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "score_distribution.pdf"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  → saved {path}")


# ---------------------------------------------------------------------------
# Figure 3 — score vs contention scatter
# ---------------------------------------------------------------------------

def plot_score_vs_contention(rows: list[dict], out_dir: Path) -> None:
    """Scatter: competing_requests (x) vs avg_judge_score (y), by condition."""
    present_conditions = sorted({r.get("condition", "unknown") for r in rows})

    fig, ax = plt.subplots(figsize=(7, 5))

    for cond in present_conditions:
        subset = [r for r in rows if r.get("condition") == cond]
        if not subset:
            continue
        xs = [r["competing_requests"] for r in subset]
        ys = [r["avg_judge_score"]    for r in subset]
        ax.scatter(
            xs, ys,
            label=_CONDITION_LABELS.get(cond, cond),
            color=_CONDITION_COLOURS.get(cond, "grey"),
            s=40,
            alpha=0.80,
        )

        # Trend line if we have enough points
        if len(xs) >= 3:
            try:
                coeffs = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs), max(xs), 50)
                ax.plot(
                    x_line,
                    np.polyval(coeffs, x_line),
                    color=_CONDITION_COLOURS.get(cond, "grey"),
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.6,
                )
            except np.linalg.LinAlgError:
                pass

    ax.set_xlabel("Competing Steering Requests (INTERRUPT events)", fontsize=11)
    ax.set_ylabel("Avg Judge Score (1–10)", fontsize=11)
    ax.set_title("Judge Score vs Queue Contention", fontsize=14)
    ax.set_ylim(0, 10.5)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "score_vs_contention.pdf"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  → saved {path}")


# ---------------------------------------------------------------------------
# Bonus: steering score vs user score scatter
# ---------------------------------------------------------------------------

def plot_steering_vs_user_score(rows: list[dict], out_dir: Path) -> None:
    """Scatter: avg_steering_score (x) vs avg_user_score (y) by condition.

    Shows whether DACS maintains both steering quality AND situational
    awareness for user messages simultaneously.
    """
    present_conditions = sorted({r.get("condition", "unknown") for r in rows})

    fig, ax = plt.subplots(figsize=(6, 5))

    for cond in present_conditions:
        subset = [r for r in rows if r.get("condition") == cond]
        if not subset:
            continue
        xs = [r["avg_steering_score"] for r in subset]
        ys = [r["avg_user_score"]     for r in subset]
        ax.scatter(
            xs, ys,
            label=_CONDITION_LABELS.get(cond, cond),
            color=_CONDITION_COLOURS.get(cond, "grey"),
            s=50,
            alpha=0.82,
        )

    # Reference line y=x
    lim_lo = 0
    lim_hi = 10.5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=0.7, alpha=0.4,
            label="y = x")

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.set_xlabel("Avg Steering Score (1–10)", fontsize=11)
    ax.set_ylabel("Avg User Response Score (1–10)", fontsize=11)
    ax.set_title("Steering Quality vs User Awareness", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "steering_vs_user_score.pdf"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  → saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate analysis plots for the concurrency experiment"
    )
    parser.add_argument(
        "--csv",
        default=_DEFAULT_CSV,
        help=f"Path to summary CSV (default: {_DEFAULT_CSV})",
    )
    parser.add_argument(
        "--out",
        default=str(_FIGURES_DIR),
        help="Output directory for figures (default: results_concurrency/figures)",
    )
    args = parser.parse_args()

    csv_path = args.csv
    out_dir  = Path(args.out)

    if not Path(csv_path).exists():
        print(f"[error] CSV not found: {csv_path}")
        print("Run the experiment first: python -m experiments_concurrency.run")
        return

    rows = _load(csv_path)
    if not rows:
        print("[error] CSV is empty or could not be parsed.")
        return

    print(f"Loaded {len(rows)} trial rows from {csv_path}")
    print(f"Generating figures → {out_dir}/")

    plot_condition_comparison(rows, out_dir)
    plot_score_distribution(rows, out_dir)
    plot_score_vs_contention(rows, out_dir)
    plot_steering_vs_user_score(rows, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
