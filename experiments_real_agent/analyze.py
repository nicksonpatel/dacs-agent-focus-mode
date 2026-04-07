"""Analysis and comparison table for the DACS real-agent validation experiment.

Loads:
  - results_real_agent/summary_real.csv     (M2/M3 metrics from run.py)
  - results_real_agent/judge_results.csv    (M1_real from judge.py)
  - results/summary.csv                     (synthetic scenarios for comparison)

Auto-detects which real-agent scenarios have completed results and shows:
  1. Per-scenario four-column table (DACS real / Baseline real / DACS syn / Base syn)
  2. N-scaling summary table when multiple scenarios are available

Synthetic mirror map:
  ra1_n3  →  s1_n3   (N=3)
  ra2_n5  →  s2_n5   (N=5)

Usage
-----
    python -m experiments_real_agent.analyze
    python -m experiments_real_agent.analyze --scenario ra1_n3 ra2_n5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

_REAL_DIR = Path("results_real_agent")
_SYN_DIR  = Path("results")
# May be overridden by --results-dir CLI flag

# Maps real-agent scenario → synthetic mirror scenario
_SYN_MIRROR = {
    "ra1_n3": "s1_n3",
    "ra2_n5": "s2_n5",
}
# Agent count per scenario
_N_AGENTS = {
    "ra1_n3": 3,
    "ra2_n5": 5,
}


# ---------------------------------------------------------------------------
# Welch's t-test (pure stdlib)
# ---------------------------------------------------------------------------

def _welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    n1, n2 = len(a), len(b)
    m1, m2 = statistics.mean(a), statistics.mean(b)
    v1, v2 = statistics.variance(a), statistics.variance(b)
    t_stat = (m1 - m2) / math.sqrt(v1 / n1 + v2 / n2)
    num    = (v1 / n1 + v2 / n2) ** 2
    denom  = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df     = num / denom if denom else 1.0
    p      = _t_sf(abs(t_stat), df) * 2
    return t_stat, p


def _t_sf(t: float, df: float) -> float:
    x = df / (df + t * t)
    return 0.5 * _incbeta(df / 2, 0.5, x)


def _incbeta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (
        math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    return math.exp(lbeta) / a * _lentz_cf(a, b, x)


def _lentz_cf(a: float, b: float, x: float, maxiter: int = 200, eps: float = 1e-10) -> float:
    qab = a + b; qap = a + 1.0; qam = a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d; h = d
    for m in range(1, maxiter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d  = 1.0 + aa * d; c = 1.0 + aa / c
        if abs(d) < 1e-30: d = 1e-30
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d  = 1.0 + aa * d; c = 1.0 + aa / c
        if abs(d) < 1e-30: d = 1e-30
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < eps: break
    return h


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _pct(vals: list[float]) -> str:
    return f"{statistics.mean(vals):.1%}" if vals else "—"

def _std(vals: list[float]) -> str:
    return f"{statistics.stdev(vals):.3f}" if len(vals) >= 2 else "—"

def _tok(vals: list[float]) -> str:
    return f"{statistics.mean(vals):.0f}" if vals else "—"

def _sig(p: float) -> str:
    if math.isnan(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ---------------------------------------------------------------------------
# Per-scenario metrics extraction
# ---------------------------------------------------------------------------

def _judge_m1_by_condition(judge_rows: list[dict], scenario_id: str) -> dict[str, list[float]]:
    """Per-trial M1 accuracy grouped by condition for one scenario."""
    run_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in judge_rows:
        # Accept rows with matching scenario field, or rows whose run_id starts with scenario_id
        row_scenario = row.get("scenario", "")
        run_id = row.get("run_id", "")
        if row_scenario and row_scenario != scenario_id:
            continue
        if not row_scenario and not run_id.startswith(scenario_id):
            continue
        if row.get("rubric_topic", "") == "extra":
            continue
        cond = row.get("condition", "")
        verdict = 1 if row.get("judge_verdict", "").upper() == "CORRECT" else 0
        run_groups[(cond, run_id)].append(verdict)

    result: dict[str, list[float]] = defaultdict(list)
    for (cond, _), verdicts in run_groups.items():
        if verdicts:
            result[cond].append(sum(verdicts) / len(verdicts))
    return dict(result)


def _real_m2_m3_by_condition(
    real_summary: list[dict], scenario_id: str
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    m2: dict[str, list[float]] = defaultdict(list)
    m3: dict[str, list[float]] = defaultdict(list)
    for row in real_summary:
        if row.get("scenario", "") != scenario_id:
            continue
        cond = row.get("condition", "")
        try:
            m2[cond].append(float(row["contamination_rate"]))
            m3[cond].append(float(row["avg_context_tokens"]))
        except (KeyError, ValueError):
            pass
    return dict(m2), dict(m3)


def _syn_metrics_by_condition(
    syn_summary: list[dict], syn_scenario: str
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    m1: dict[str, list[float]] = defaultdict(list)
    m2: dict[str, list[float]] = defaultdict(list)
    m3: dict[str, list[float]] = defaultdict(list)
    for row in syn_summary:
        if row.get("scenario", "") != syn_scenario:
            continue
        cond = row.get("condition", "")
        try:
            m1[cond].append(float(row["steering_accuracy"]))
            m2[cond].append(float(row["contamination_rate"]))
            m3[cond].append(float(row["avg_context_tokens"]))
        except (KeyError, ValueError):
            pass
    return dict(m1), dict(m2), dict(m3)


def _steering_coverage(
    judge_rows: list[dict], scenario_id: str, expected_per_trial: int
) -> dict[str, float]:
    run_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in judge_rows:
        row_scenario = row.get("scenario", "")
        run_id = row.get("run_id", "")
        if row_scenario and row_scenario != scenario_id:
            continue
        if not row_scenario and not run_id.startswith(scenario_id):
            continue
        if row.get("rubric_topic", "") != "extra":
            run_counts[(row["condition"], run_id)] += 1

    by_cond: dict[str, list[float]] = defaultdict(list)
    for (cond, _), count in run_counts.items():
        by_cond[cond].append(min(count / expected_per_trial, 1.0))
    return {cond: statistics.mean(vals) for cond, vals in by_cond.items()}


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_scenario_table(
    scenario_id: str,
    syn_scenario: str,
    m1_by_cond: dict[str, list[float]],
    m2_by_cond: dict[str, list[float]],
    m3_by_cond: dict[str, list[float]],
    syn_m1: dict[str, list[float]],
    syn_m2: dict[str, list[float]],
    syn_m3: dict[str, list[float]],
    coverage: dict[str, float],
) -> None:
    SEP = "-" * 110
    W = 24

    dacs_m1     = m1_by_cond.get("dacs", [])
    baseline_m1 = m1_by_cond.get("baseline", [])
    t_stat, p_val = _welch_t(dacs_m1, baseline_m1)
    n_real = _N_AGENTS.get(scenario_id, "?")

    print(f"\n{SEP}")
    print(f"Scenario: {scenario_id}  (mirrors {syn_scenario}, N={n_real} real LLM agents)")
    print(SEP)

    header = (
        f"{'Metric':<26} "
        f"{'DACS (real)':>{W}} "
        f"{'Baseline (real)':>{W}} "
        f"{'DACS (' + syn_scenario + ' syn)':>{W}} "
        f"{'Base (' + syn_scenario + ' syn)':>{W}}"
    )
    print(header)
    print("-" * len(header))

    def row(name: str, a: str, b: str, c: str, d: str) -> None:
        print(f"{name:<26} {a:>{W}} {b:>{W}} {c:>{W}} {d:>{W}}")

    row("M1 Steering Acc (judge)",
        _pct(dacs_m1), _pct(baseline_m1),
        _pct(syn_m1.get("dacs", [])), _pct(syn_m1.get("baseline", [])))
    row("  (±std)",
        _std(dacs_m1), _std(baseline_m1),
        _std(syn_m1.get("dacs", [])), _std(syn_m1.get("baseline", [])))
    row("M2 Contamination",
        _pct(m2_by_cond.get("dacs", [])), _pct(m2_by_cond.get("baseline", [])),
        _pct(syn_m2.get("dacs", [])), _pct(syn_m2.get("baseline", [])))
    row("M3 Avg Context Tokens",
        _tok(m3_by_cond.get("dacs", [])), _tok(m3_by_cond.get("baseline", [])),
        _tok(syn_m3.get("dacs", [])), _tok(syn_m3.get("baseline", [])))

    cov_d = f"{coverage.get('dacs', 0.):.1%}"
    cov_b = f"{coverage.get('baseline', 0.):.1%}"
    row("Steering Coverage", cov_d, cov_b, "100.0% (fixed)", "100.0% (fixed)")
    row("N trials",
        str(len(dacs_m1)), str(len(baseline_m1)),
        str(len(syn_m1.get("dacs", []))), str(len(syn_m1.get("baseline", []))))

    print()
    print("-" * 80)
    print("Welch's t-test on M1_real DACS vs Baseline")
    if not math.isnan(t_stat):
        sig = _sig(p_val)
        print(f"  t = {t_stat:.3f},  p = {p_val:.4f}  {sig}")
        msg = (
            "DACS significantly outperforms baseline on real agents (p < 0.05)"
            if p_val < 0.05
            else "No significant difference detected (more trials may be needed)"
        )
        print(f"  → {msg}")
    else:
        print("  → Insufficient data for t-test (need ≥2 trials per condition)")
    print("-" * 80)


# ---------------------------------------------------------------------------
# N-scaling summary table (printed only when ≥2 scenarios have data)
# ---------------------------------------------------------------------------

def _print_scaling_table(scaling_rows: list[dict]) -> None:
    SEP = "=" * 100
    print(f"\n{SEP}")
    print("N-SCALING SUMMARY — Real-Agent vs Synthetic")
    print(SEP)
    print(
        f"{'Scenario':<12} {'N':>4} "
        f"{'DACS%':>10} {'Base%':>10} {'Δpp':>8}  "
        f"{'p':>8}  "
        f"{'SynDACS%':>10} {'SynBase%':>10} {'SynΔpp':>8}"
    )
    print("-" * 100)
    for r in scaling_rows:
        d_pct   = r.get("dacs_m1", float("nan"))
        b_pct   = r.get("base_m1", float("nan"))
        delta   = (d_pct - b_pct) * 100 if not math.isnan(d_pct) and not math.isnan(b_pct) else float("nan")
        sd_pct  = r.get("syn_dacs_m1", float("nan"))
        sb_pct  = r.get("syn_base_m1", float("nan"))
        s_delta = (sd_pct - sb_pct) * 100 if not math.isnan(sd_pct) and not math.isnan(sb_pct) else float("nan")
        p = r.get("p_val", float("nan"))
        sig = _sig(p)

        def _f(v: float, fmt: str = ".1%") -> str:
            return f"{v:{fmt}}" if not math.isnan(v) else "—"

        print(
            f"{r['scenario_id']:<12} {r['n']:>4} "
            f"{_f(d_pct):>10} {_f(b_pct):>10} {_f(delta, '+.1f') if not math.isnan(delta) else '—':>8}  "
            f"{_f(p, '.4f') + ' ' + sig:>10}  "
            f"{_f(sd_pct):>10} {_f(sb_pct):>10} {_f(s_delta, '+.1f') if not math.isnan(s_delta) else '—':>8}"
        )
    print(SEP)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _REAL_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", nargs="*",
        help="Scenario IDs to analyse (default: all with results)"
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Path to real-agent results directory (default: results_real_agent)"
    )
    args = parser.parse_args()

    if args.results_dir:
        _REAL_DIR = Path(args.results_dir)

    real_summary = _load_csv(_REAL_DIR / "summary_real.csv")
    # Load judge results from all per-scenario files + legacy combined file
    judge_rows: list[dict] = []
    for csv_path in sorted(_REAL_DIR.glob("judge_results*.csv")):
        judge_rows.extend(_load_csv(csv_path))
    syn_summary  = _load_csv(_SYN_DIR / "summary.csv")

    if not real_summary:
        print(
            f"[ERROR] No data in {_REAL_DIR / 'summary_real.csv'}.\n"
            "Run: python -m experiments_real_agent.run --mode both --trials 10"
        )
        return

    # Determine which scenarios have real data
    available = {row["scenario"] for row in real_summary if row.get("scenario")}
    if args.scenario:
        scenarios = [s for s in args.scenario if s in available]
        missing = [s for s in args.scenario if s not in available]
        for s in missing:
            print(f"[WARN] No results found for scenario '{s}' — skipping.")
    else:
        scenarios = sorted(available, key=lambda s: _N_AGENTS.get(s, 99))

    if not scenarios:
        print("[ERROR] No valid scenarios with results found.")
        return

    print("\n" + "=" * 60)
    print("DACS REAL-AGENT VALIDATION — Analysis Results")
    print("=" * 60)

    scaling_rows: list[dict] = []

    for scenario_id in scenarios:
        syn_scenario = _SYN_MIRROR.get(scenario_id, "")
        from experiments_real_agent.scenario_defs import REAL_SCENARIOS
        scenario_def = REAL_SCENARIOS.get(scenario_id)
        if scenario_def is None:
            print(f"[WARN] No scenario definition for '{scenario_id}' — skipping.")
            continue

        expected_dp = sum(len(s.rubrics) for s in scenario_def.agents)

        m1_by_cond              = _judge_m1_by_condition(judge_rows, scenario_id)
        m2_by_cond, m3_by_cond  = _real_m2_m3_by_condition(real_summary, scenario_id)
        syn_m1, syn_m2, syn_m3  = (
            _syn_metrics_by_condition(syn_summary, syn_scenario)
            if syn_scenario else ({}, {}, {})
        )
        coverage = _steering_coverage(judge_rows, scenario_id, expected_dp)

        _print_scenario_table(
            scenario_id, syn_scenario or "—",
            m1_by_cond, m2_by_cond, m3_by_cond,
            syn_m1, syn_m2, syn_m3,
            coverage,
        )

        dacs_m1     = m1_by_cond.get("dacs", [])
        baseline_m1 = m1_by_cond.get("baseline", [])
        t_stat, p_val = _welch_t(dacs_m1, baseline_m1)

        scaling_rows.append({
            "scenario_id":  scenario_id,
            "n":            _N_AGENTS.get(scenario_id, "?"),
            "dacs_m1":      statistics.mean(dacs_m1) if dacs_m1 else float("nan"),
            "base_m1":      statistics.mean(baseline_m1) if baseline_m1 else float("nan"),
            "p_val":        p_val,
            "syn_dacs_m1":  statistics.mean(syn_m1.get("dacs", [])) if syn_m1.get("dacs") else float("nan"),
            "syn_base_m1":  statistics.mean(syn_m1.get("baseline", [])) if syn_m1.get("baseline") else float("nan"),
        })

    if len(scaling_rows) >= 2:
        _print_scaling_table(scaling_rows)


if __name__ == "__main__":
    main()

