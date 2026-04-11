"""Post-hoc analysis for the DACS WebArena benchmark experiment.

Loads ``results_webarena/summary_webarena.csv`` (M2/M3 metrics) and the
judge CSV files (M1 — steering accuracy), then prints a comparison table
and optionally writes a Markdown report.

Cross-harness comparison:
  - WebArena (this experiment): results_webarena/
  - Real-agent (synthetic tasks): results_real_agent/ or results_real_agent_haiku/
  - Synthetic (Phases 1–3): results/

Usage
-----
    python -m experiments_webarena.analyze
    python -m experiments_webarena.analyze --results-dir results_webarena --report
    python -m experiments_webarena.analyze --compare-real results_real_agent_haiku
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

from experiments_webarena.scenario_defs import WEB_SCENARIOS

_RESULTS_DIR = Path("results_webarena")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_summary(results_dir: Path) -> list[dict]:
    """Load summary_webarena.csv → list of row dicts."""
    path = results_dir / "summary_webarena.csv"
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _load_judge_csv(results_dir: Path, scenario_id: str) -> list[dict]:
    """Load any judge CSV for a scenario (first match wins)."""
    matches = sorted(results_dir.glob(f"judge_results_{scenario_id}_*.csv"))
    if not matches:
        return []
    rows: list[dict] = []
    for path in matches:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                row["keyword_score"] = int(row.get("keyword_score", 0))
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _group_by_condition(rows: list[dict], key: str, cast=float) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        try:
            grouped[row["condition"]].append(cast(row[key]))
        except (KeyError, ValueError):
            pass
    return grouped


def _mean(vals: list) -> float:
    return statistics.mean(vals) if vals else float("nan")


def _fmt(v: float, pct: bool = False) -> str:
    if pct:
        return f"{v:.1%}" if v == v else "—"
    return f"{v:.1f}"   if v == v else "—"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    results_dir: Path = _RESULTS_DIR,
    compare_real_dir: Path | None = None,
    write_report: bool = False,
) -> None:
    summary_rows = _load_summary(results_dir)

    print("\n" + "=" * 70)
    print("  DACS WebArena Benchmark Analysis")
    print("=" * 70)

    # Per-scenario breakdown
    all_lines: list[str] = []
    all_lines.append("# DACS WebArena Benchmark Analysis\n")

    for scenario_id, scenario in WEB_SCENARIOS.items():
        s_rows = [r for r in summary_rows if r.get("scenario") == scenario_id]
        if not s_rows:
            print(f"\n[{scenario_id}] No data found — skipping.")
            continue

        j_rows = _load_judge_csv(results_dir, scenario_id)

        print(f"\n--- {scenario_id}  ({scenario.description[:60]}…) ---")

        # M2 / M3 from summary CSV
        cont_by_cond = _group_by_condition(s_rows, "contamination_rate")
        ctx_by_cond  = _group_by_condition(s_rows, "avg_context_tokens")

        # M1 from judge CSV
        acc_kw_by_cond: dict[str, list]  = defaultdict(list)
        acc_jdg_by_cond: dict[str, list] = defaultdict(list)
        for r in j_rows:
            if r.get("rubric_topic") == "extra":
                continue
            cond = r.get("condition", "")
            acc_kw_by_cond[cond].append(r["keyword_score"])
            acc_jdg_by_cond[cond].append(
                1 if r.get("judge_verdict", "") == "CORRECT" else 0
            )

        conditions = sorted({r["condition"] for r in s_rows})
        header = (
            f"{'Condition':<10} {'Trials':>6}  "
            f"{'Kw Acc':>8}  {'Judge Acc':>9}  "
            f"{'Contam':>8}  {'AvgCtx':>8}"
        )
        print(f"  {header}")
        print("  " + "-" * len(header))

        all_lines.append(f"## {scenario_id}\n")
        all_lines.append(
            "| Condition | Trials | Kw Acc | Judge Acc | Contam | Avg Ctx |"
        )
        all_lines.append(
            "|-----------|--------|--------|-----------|--------|---------|"
        )

        for cond in conditions:
            cond_rows = [r for r in s_rows if r.get("condition") == cond]
            n_trials  = len(cond_rows)
            cont_mean = _mean(cont_by_cond.get(cond, []))
            ctx_mean  = _mean(ctx_by_cond.get(cond, []))
            kw_acc    = _mean(acc_kw_by_cond.get(cond, []))
            jdg_acc   = _mean(acc_jdg_by_cond.get(cond, []))

            print(
                f"  {cond:<10} {n_trials:>6}  "
                f"{_fmt(kw_acc, pct=True):>8}  {_fmt(jdg_acc, pct=True):>9}  "
                f"{_fmt(cont_mean, pct=True):>8}  {_fmt(ctx_mean):>8}"
            )
            all_lines.append(
                f"| {cond} | {n_trials} "
                f"| {_fmt(kw_acc, pct=True)} "
                f"| {_fmt(jdg_acc, pct=True)} "
                f"| {_fmt(cont_mean, pct=True)} "
                f"| {_fmt(ctx_mean)} |"
            )

        all_lines.append("")

        # Context ratio (DACS vs baseline)
        dacs_ctx = _mean(ctx_by_cond.get("dacs", []))
        base_ctx = _mean(ctx_by_cond.get("baseline", []))
        if base_ctx and base_ctx > 0:
            ratio = base_ctx / dacs_ctx
            print(f"\n  Context ratio (baseline / DACS): {ratio:.2f}×")
            all_lines.append(
                f"Context ratio (baseline / DACS): {ratio:.2f}×\n"
            )

    # Optional cross-harness comparison
    if compare_real_dir is not None:
        real_summary = compare_real_dir / "summary_real.csv"
        if real_summary.exists():
            print(f"\n--- Cross-harness comparison: WebArena vs Real-agent ---")
            all_lines.append("\n## Cross-harness Comparison\n")
            all_lines.append(
                "| Harness | Condition | AvgCtx | Contam |"
            )
            all_lines.append("|---------|-----------|--------|--------|")
            with open(real_summary, newline="") as fh:
                real_rows = list(csv.DictReader(fh))
            for label, rows in [("WebArena", summary_rows), ("Real-agent", real_rows)]:
                cont_by_c = _group_by_condition(rows, "contamination_rate")
                ctx_by_c  = _group_by_condition(rows, "avg_context_tokens")
                for cond in sorted({r.get("condition", "") for r in rows}):
                    print(
                        f"  {label:<10} {cond:<10} "
                        f"ctx={_fmt(_mean(ctx_by_c.get(cond, [])))} "
                        f"contam={_fmt(_mean(cont_by_c.get(cond, [])), pct=True)}"
                    )
                    all_lines.append(
                        f"| {label} | {cond} "
                        f"| {_fmt(_mean(ctx_by_c.get(cond, [])))} "
                        f"| {_fmt(_mean(cont_by_c.get(cond, [])), pct=True)} |"
                    )

    if write_report:
        report_path = results_dir / "WEBARENA_RESULTS.md"
        report_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
        print(f"\nReport written to {report_path}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse DACS WebArena benchmark results"
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory containing JSONL logs and summary CSV (default: results_webarena)",
    )
    parser.add_argument(
        "--compare-real", default=None, metavar="DIR",
        help="Path to real-agent results directory for cross-harness comparison",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Write a Markdown report to results_webarena/WEBARENA_RESULTS.md",
    )
    args = parser.parse_args()

    results_dir     = Path(args.results_dir) if args.results_dir else _RESULTS_DIR
    compare_real    = Path(args.compare_real) if args.compare_real else None
    analyze(results_dir, compare_real_dir=compare_real, write_report=args.report)


if __name__ == "__main__":
    main()
