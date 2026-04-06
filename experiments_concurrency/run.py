"""CLI entry point for the concurrency & interruption experiment.

Usage
-----
    python -m experiments_concurrency.run [options]

Options
-------
    --scenario  cc1_n3 cc2_n5 | all          (default: all)
    --trials    N                             (default: 5)
    --mode      dacs | baseline | both        (default: both)
    --inject    clean | concurrent | both     (default: both)
    --model     <model_name>                  (default: DACS_MODEL env or MiniMax-M2.7)
    --budget    <token_count>                 (default: DACS_T env or 204800)
    --parallel  <int>   max concurrent trials (default: 1)

Environment
-----------
    MINIMAX_API_KEY   required
    DACS_MODEL        optional model name override
    DACS_T            optional token budget override

Results
-------
    results_concurrency/<run_id>.jsonl          raw event log per trial
    results_concurrency/concurrency_summary.csv aggregated metrics

Summary CSV columns
-------------------
    run_id, scenario, condition, n_agents, trial, focus_mode, inject,
    steering_accuracy, contamination_rate, avg_context_tokens, p95_context_tokens,
    total_decisions, correct_decisions,
    avg_judge_score, avg_steering_score, avg_user_score,
    min_judge_score, total_judged,
    inject_count, competing_requests
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import fcntl
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from experiments_concurrency.harness import run_concurrent_trial
from experiments_concurrency.scenario_defs import CONCURRENCY_SCENARIOS

load_dotenv()

_DEFAULT_MODEL = os.environ.get("DACS_MODEL", "MiniMax-M2.7")
_DEFAULT_T     = int(os.environ.get("DACS_T", "204800"))

_SUMMARY_PATH  = "results_concurrency/concurrency_summary.csv"
_FIELDNAMES    = [
    "run_id", "scenario", "condition", "n_agents", "trial",
    "focus_mode", "inject",
    # standard keyword metrics (kept for comparison)
    "steering_accuracy", "contamination_rate",
    "avg_context_tokens", "p95_context_tokens",
    "total_decisions", "correct_decisions",
    # inline judge metrics (primary for this experiment)
    "avg_judge_score", "avg_steering_score", "avg_user_score",
    "min_judge_score", "total_judged",
    # concurrency signals
    "inject_count", "competing_requests",
]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

async def run_experiment(
    scenario_ids: list[str],
    n_trials: int,
    modes: list[str],
    inject_modes: list[bool],
    model: str,
    token_budget: int,
    parallel: int,
) -> None:
    Path("results_concurrency").mkdir(exist_ok=True)

    write_header = not Path(_SUMMARY_PATH).exists()
    summary_file = open(_SUMMARY_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(summary_file, fieldnames=_FIELDNAMES, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        summary_file.flush()

    rows: list[dict[str, Any]] = []
    write_lock       = asyncio.Lock()
    trial_semaphore  = asyncio.Semaphore(max(1, parallel))

    async def _run_and_record(
        scenario_id: str,
        mode_name: str,
        inject: bool,
        trial: int,
    ) -> None:
        scenario   = CONCURRENCY_SCENARIOS[scenario_id]
        focus_mode = mode_name == "dacs"
        inj_str    = "concurrent" if inject else "clean"
        run_id     = (
            f"{scenario_id}_{mode_name}_{inj_str}_t{trial:02d}_{uuid.uuid4().hex[:6]}"
        )

        async with trial_semaphore:
            try:
                metrics = await run_concurrent_trial(
                    scenario=scenario,
                    focus_mode=focus_mode,
                    inject=inject,
                    model=model,
                    token_budget=token_budget,
                    run_id=run_id,
                )
            except Exception as exc:  # noqa: BLE001
                Console().print(f"[red]Trial {run_id} failed: {exc}[/red]")
                import traceback
                traceback.print_exc()
                return

        row: dict[str, Any] = {
            "trial":      trial,
            **{k: metrics.get(k, "") for k in _FIELDNAMES if k not in ("trial",)},
        }

        async with write_lock:
            rows.append(row)
            fcntl.flock(summary_file, fcntl.LOCK_EX)
            try:
                writer.writerow(row)
                summary_file.flush()
            finally:
                fcntl.flock(summary_file, fcntl.LOCK_UN)

    tasks: list[asyncio.Task[None]] = []
    for scenario_id in scenario_ids:
        for mode_name in modes:
            for inject in inject_modes:
                for trial in range(1, n_trials + 1):
                    tasks.append(asyncio.create_task(
                        _run_and_record(scenario_id, mode_name, inject, trial)
                    ))

    await asyncio.gather(*tasks)
    summary_file.close()

    # ------------------------------------------------------------------
    # Terminal summary table
    # ------------------------------------------------------------------
    c = Console()
    c.rule("[green]Concurrency Experiment Complete[/green]")
    t = Table(title=f"Results → {_SUMMARY_PATH}", show_lines=True)
    display_cols = [
        "run_id", "condition", "n_agents", "trial",
        "steering_accuracy", "avg_judge_score", "avg_user_score",
        "inject_count", "competing_requests",
    ]
    for col in display_cols:
        t.add_column(col, style="cyan" if "score" in col or "accuracy" in col else "")

    for row in sorted(rows, key=lambda r: (r.get("scenario", ""), r.get("condition", ""), r.get("trial", 0))):
        t.add_row(
            str(row.get("run_id", "")),
            str(row.get("condition", "")),
            str(row.get("n_agents", "")),
            str(row.get("trial", "")),
            f"{float(row.get('steering_accuracy', 0)):.2%}",
            f"{float(row.get('avg_judge_score', 0)):.2f}",
            f"{float(row.get('avg_user_score', 0)):.2f}",
            str(row.get("inject_count", "")),
            str(row.get("competing_requests", "")),
        )
    c.print(t)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DACS concurrency & interruption experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=list(CONCURRENCY_SCENARIOS.keys()) + ["all"],
        default=["all"],
        help="Scenario IDs to run (default: all)",
    )
    parser.add_argument("--trials",  type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["dacs", "baseline", "both"],
        default="both",
        help="LLM condition (default: both)",
    )
    parser.add_argument(
        "--inject",
        choices=["clean", "concurrent", "both"],
        default="both",
        help="Whether to fire user message injections (default: both)",
    )
    parser.add_argument("--model",    default=_DEFAULT_MODEL)
    parser.add_argument("--budget",   type=int, default=_DEFAULT_T)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Max concurrent trials (default: 1 — serial for predictable timing)",
    )
    args = parser.parse_args()

    # Resolve scenario IDs
    if "all" in args.scenario:
        scenario_ids = list(CONCURRENCY_SCENARIOS.keys())
    else:
        scenario_ids = args.scenario

    # Resolve mode + inject
    modes        = ["dacs", "baseline"] if args.mode    == "both" else [args.mode]
    inject_modes = [True, False]         if args.inject == "both" else [args.inject == "concurrent"]

    asyncio.run(
        run_experiment(
            scenario_ids=scenario_ids,
            n_trials=args.trials,
            modes=modes,
            inject_modes=inject_modes,
            model=args.model,
            token_budget=args.budget,
            parallel=args.parallel,
        )
    )


if __name__ == "__main__":
    main()
