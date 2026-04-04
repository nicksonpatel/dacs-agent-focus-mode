"""DACS Experiment Harness.

Runs the full 60-trial experiment: DACS vs baseline × N=3,5,10 × 10 trials each.

Usage
-----
    python -m experiments.run_experiment [--scenario s1_n3] [--trials 10] [--mode both]

Environment
-----------
    MINIMAX_API_KEY  — required
    DACS_MODEL       — optional, default "MiniMax-M2.7"
    DACS_T           — optional, token budget, default 204800

Results
-------
    results/<run_id>.jsonl   — raw event log per trial
    results/summary.csv      — aggregated metrics across all trials
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from agents.code_writer_agent import CodeWriterAgent
from agents.data_processor_agent import DataProcessorAgent
from agents.debugger_agent import DebuggerAgent
from agents.generic_agent import GenericAgent
from agents.long_writer_agent import LongWriterAgent
from agents.research_agent import ResearchAgent
from experiments.metrics import compute_metrics
from experiments.task_suite import SCENARIOS, ScenarioSpec
from src.context_builder import ContextBuilder
from src.logger import Logger
from src.monitor import TerminalMonitor
from src.orchestrator import Orchestrator
from src.protocols import SteeringRequestQueue
from src.registry import RegistryManager

load_dotenv()

_AGENT_CLASSES = {
    "code_writer":    CodeWriterAgent,
    "research":       ResearchAgent,
    "data_processor": DataProcessorAgent,
    "debugger":       DebuggerAgent,
    "long_writer":    LongWriterAgent,
}

_DEFAULT_MODEL = os.environ.get("DACS_MODEL", "MiniMax-M2.7")
_DEFAULT_T     = int(os.environ.get("DACS_T", "204800"))


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

async def run_trial(
    scenario: ScenarioSpec,
    focus_mode: bool,
    model: str,
    token_budget: int,
    run_id: str,
) -> dict[str, Any]:
    log_path = f"results/{run_id}.jsonl"
    logger   = Logger(log_path)
    monitor  = TerminalMonitor(token_budget=token_budget)
    logger.add_sink(monitor.handle)

    logger.log({
        "event":        "RUN_START",
        "run_id":       run_id,
        "condition":    "DACS" if focus_mode else "baseline",
        "scenario":     scenario.scenario_id,
        "n_agents":     len(scenario.agents),
        "focus_mode":   focus_mode,
        "model":        model,
        "token_budget": token_budget,
    })

    registry  = RegistryManager(logger)
    queue     = SteeringRequestQueue(logger)
    cb        = ContextBuilder(token_budget, logger)
    registry.set_context_builder(cb)

    client       = AsyncAnthropic(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url="https://api.minimax.io/anthropic",
    )
    orchestrator = Orchestrator(
        registry=registry,
        queue=queue,
        context_builder=cb,
        llm_client=client,
        model=model,
        token_budget=token_budget,
        focus_mode=focus_mode,
        logger=logger,
    )

    agents: list[BaseAgent] = []
    for spec in scenario.agents:
        if spec.agent_type == "generic":
            agent = GenericAgent(
                agent_id=spec.agent_id,
                task_description=spec.task_description,
                registry=registry,
                queue=queue,
                steps=spec.steps or [],
            )
        else:
            cls   = _AGENT_CLASSES[spec.agent_type]
            agent = cls(
                agent_id=spec.agent_id,
                task_description=spec.task_description,
                registry=registry,
                queue=queue,
            )
        registry.register(spec.agent_id, spec.task_description)
        orchestrator.register_agent(agent)
        agents.append(agent)

    # Run all agents + orchestrator concurrently; stop orchestrator when agents finish
    agent_tasks = [asyncio.create_task(agent.run()) for agent in agents]
    orch_task   = asyncio.create_task(orchestrator.run())

    await asyncio.gather(*agent_tasks)
    orchestrator.stop()
    await orch_task

    logger.log({"event": "RUN_END", "run_id": run_id})
    logger.close()

    return compute_metrics(log_path, scenario)


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

async def run_experiment(
    scenario_ids: list[str],
    n_trials: int,
    modes: list[str],
    model: str,
    token_budget: int,
) -> None:
    Path("results").mkdir(exist_ok=True)
    summary_path = "results/summary.csv"
    fieldnames = [
        "run_id", "scenario", "condition", "n_agents", "trial",
        "steering_accuracy", "contamination_rate",
        "avg_context_tokens", "p95_context_tokens", "user_latency_ms",
        "total_decisions", "correct_decisions",
    ]

    # Write header if file doesn't exist yet
    write_header = not Path(summary_path).exists()
    summary_file = open(summary_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        summary_file.flush()

    rows: list[dict] = []
    for scenario_id in scenario_ids:
        scenario = SCENARIOS[scenario_id]
        for mode_name in modes:
            focus_mode = mode_name == "dacs"
            for trial in range(1, n_trials + 1):
                run_id = f"{scenario_id}_{mode_name}_t{trial:02d}_{uuid.uuid4().hex[:6]}"
                try:
                    metrics = await run_trial(scenario, focus_mode, model, token_budget, run_id)
                except Exception as exc:
                    from rich.console import Console as _Ce
                    _Ce().print(f"[red]Trial {run_id} failed: {exc}[/red]")
                    continue
                row = {
                    "run_id":    run_id,
                    "scenario":  scenario_id,
                    "condition": mode_name,
                    "n_agents":  len(scenario.agents),
                    "trial":     trial,
                    **{k: metrics[k] for k in fieldnames[5:]},
                }
                rows.append(row)
                # Write immediately so progress is preserved on crash
                writer.writerow(row)
                summary_file.flush()

    summary_file.close()

    from rich.console import Console as _C
    from rich.table import Table
    c = _C()
    c.rule("[green]Experiment Complete[/green]")
    t = Table(title=f"Results → {summary_path}", show_lines=True)
    for col in ["run_id", "condition", "n_agents", "trial",
                "steering_accuracy", "contamination_rate", "avg_context_tokens"]:
        t.add_column(col, style="cyan" if "accuracy" in col or "contamination" in col else "")
    for row in rows:
        t.add_row(
            row["run_id"], row["condition"], str(row["n_agents"]), str(row["trial"]),
            f"{row['steering_accuracy']:.2%}", f"{row['contamination_rate']:.2%}",
            f"{row['avg_context_tokens']:.0f}",
        )
    c.print(t)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run DACS experiment")
    parser.add_argument(
        "--scenario", nargs="+",
        default=list(SCENARIOS.keys()),
        help="Scenario ID(s) to run (default: all)",
    )
    parser.add_argument("--trials",  type=int, default=10)
    parser.add_argument(
        "--mode", choices=["dacs", "baseline", "both"], default="both",
    )
    parser.add_argument("--model",  default=_DEFAULT_MODEL)
    parser.add_argument("--budget", type=int, default=_DEFAULT_T)
    args = parser.parse_args()

    modes = ["dacs", "baseline"] if args.mode == "both" else [args.mode]
    asyncio.run(
        run_experiment(
            scenario_ids=args.scenario,
            n_trials=args.trials,
            modes=modes,
            model=args.model,
            token_budget=args.budget,
        )
    )


if __name__ == "__main__":
    main()
