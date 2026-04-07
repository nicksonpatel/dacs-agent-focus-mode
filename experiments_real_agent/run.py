"""DACS real-agent validation experiment runner.

Runs LLMAgent-based trials — each agent calls the LLM to generate its own
work output and autonomously emits steering questions via [[STEER: ...]] markers.
The orchestrator handles these requests identically to the synthetic-agent harness,
enabling a direct real-agent vs synthetic comparison.

Usage
-----
    # OpenRouter (Haiku) — recommended, no MiniMax budget needed
    python -m experiments_real_agent.run --api openrouter --mode dacs   --trials 10
    python -m experiments_real_agent.run --api openrouter --mode both   --trials 10 --scenario ra1_n3 --results-dir results_real_agent_haiku

    # MiniMax (original)
    python -m experiments_real_agent.run --api minimax   --mode both   --trials 10

Environment
-----------
    OpenRouter:  OPENROUTER_API_KEY  (or OR_API_KEY)
    MiniMax:     MINIMAX_API_KEY
    DACS_MODEL   — override model name (skips api-specific defaults)
    DACS_T       — token budget, default 204800

Results
-------
    <results-dir>/<run_id>.jsonl       — per-trial event log
    <results-dir>/summary_real.csv     — aggregated metrics
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import uuid
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agents.llm_agent import LLMAgent
from experiments_real_agent.scenario_defs import REAL_SCENARIOS, RealAgentScenario
from src.context_builder import ContextBuilder
from src.logger import Logger
from src.monitor import TerminalMonitor
from src.openrouter_client import OpenRouterClient
from src.orchestrator import Orchestrator
from src.protocols import SteeringRequestQueue
from src.registry import RegistryManager

load_dotenv()

# Model defaults per API backend
_OPENROUTER_DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
_MINIMAX_DEFAULT_MODEL    = "MiniMax-M2.7"
_DEFAULT_T                = int(os.environ.get("DACS_T", "204800"))
_RESULTS_DIR              = Path("results_real_agent")


def _resolve_api_and_model(api: str, model_override: str | None) -> tuple[str, str, str]:
    """Return (api, api_key, model) based on CLI args and available env vars.

    If ``api`` is ``auto``, prefers OpenRouter when ``OPENROUTER_API_KEY`` is
    set, falls back to MiniMax.
    """
    or_key     = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY", "")
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")

    if api == "auto":
        api = "openrouter" if or_key else "minimax"

    if api == "openrouter":
        if not or_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY (or OR_API_KEY) in .env to use OpenRouter."
            )
        model = model_override or os.environ.get("DACS_MODEL") or _OPENROUTER_DEFAULT_MODEL
        return "openrouter", or_key, model
    else:  # minimax
        if not minimax_key:
            raise RuntimeError("Set MINIMAX_API_KEY in .env to use MiniMax.")
        model = model_override or os.environ.get("DACS_MODEL") or _MINIMAX_DEFAULT_MODEL
        return "minimax", minimax_key, model


def _make_client(
    api: str,
    api_key: str,
    max_concurrent: int = 10,
) -> AsyncAnthropic | OpenRouterClient:
    """Instantiate the appropriate LLM client for the chosen backend."""
    if api == "openrouter":
        return OpenRouterClient(api_key=api_key, max_concurrent=max_concurrent)
    return AsyncAnthropic(
        api_key=api_key,
        base_url="https://api.minimax.io/anthropic",
    )

# Per-agent LLM call parameters forwarded to LLMAgent
_AGENT_MAX_STEPS   = 12
_AGENT_MAX_STEER   = 3   # matches number of rubrics per agent in ra1_n3
_AGENT_MAX_TOKENS  = 800


# ---------------------------------------------------------------------------
# Metrics helpers (inline, no dependency on experiments/metrics.py which
# expects a ScenarioSpec rather than a RealAgentScenario)
# ---------------------------------------------------------------------------

def _compute_real_metrics(log_path: str, scenario: RealAgentScenario) -> dict[str, Any]:
    """Parse JSONL event log and compute M2 / M3 metrics for real-agent trials.

    M1 (steering accuracy) is computed separately by judge.py because the
    keyword matching requires the actual LLM-generated question text.

    Returns:
        contamination_rate  (M2)
        avg_context_tokens  (M3)
        p95_context_tokens  (M3 p95)
        n_steering_responses
        n_agents
    """
    import json, statistics

    agent_ids = [spec.agent_id for spec in scenario.agents]
    events: list[dict] = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    steering_responses = [e for e in events if e.get("event") == "STEERING_RESPONSE"]
    context_built = [
        e for e in events
        if e.get("event") == "CONTEXT_BUILT"
        and e.get("mode") in {"FOCUS", "FLAT"}
    ]

    # M2 — contamination: does a response mention another agent's ID?
    contaminated = 0
    for resp in steering_responses:
        aid  = resp["agent_id"]
        text = resp.get("response_text", "").lower()
        other_ids = [a for a in agent_ids if a != aid]
        if any(oid in text for oid in other_ids):
            contaminated += 1

    context_tokens = [e["token_count"] for e in context_built if "token_count" in e]

    return {
        "contamination_rate":  contaminated / len(steering_responses) if steering_responses else 0.0,
        "avg_context_tokens":  statistics.mean(context_tokens) if context_tokens else 0.0,
        "p95_context_tokens":  (
            sorted(context_tokens)[int(len(context_tokens) * 0.95)]
            if len(context_tokens) >= 2 else (context_tokens[0] if context_tokens else 0)
        ),
        "n_steering_responses": len(steering_responses),
        "n_agents":             len(scenario.agents),
    }


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

async def run_trial(
    scenario: RealAgentScenario,
    focus_mode: bool,
    model: str,
    token_budget: int,
    run_id: str,
    client: "AsyncAnthropic | OpenRouterClient | None" = None,
    results_dir: Path = _RESULTS_DIR,
) -> dict[str, Any]:
    results_dir.mkdir(exist_ok=True)
    log_path = str(results_dir / f"{run_id}.jsonl")

    logger  = Logger(log_path)
    monitor = TerminalMonitor(token_budget=token_budget)
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
        "harness":      "real_agent",
    })

    registry = RegistryManager(logger)
    queue    = SteeringRequestQueue(logger)
    cb       = ContextBuilder(token_budget, logger)
    registry.set_context_builder(cb)

    # Caller is responsible for constructing the client (MiniMax or OpenRouter).
    # Both the orchestrator and all agents share the same client / model so
    # the only variable between DACS and baseline is the context assembly path.
    if client is None:
        # Legacy fallback — use MiniMax if called without explicit client
        client = AsyncAnthropic(
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

    agents: list[LLMAgent] = []
    for spec in scenario.agents:
        agent = LLMAgent(
            agent_id=spec.agent_id,
            task_description=spec.task_description,
            decision_hints=spec.decision_hints,
            client=client,
            model=model,
            registry=registry,
            queue=queue,
            max_steps=_AGENT_MAX_STEPS,
            max_steering_requests=_AGENT_MAX_STEER,
            agent_max_tokens=_AGENT_MAX_TOKENS,
        )
        registry.register(spec.agent_id, spec.task_description)
        orchestrator.register_agent(agent)
        agents.append(agent)

    agent_tasks = [asyncio.create_task(agent.run()) for agent in agents]
    orch_task   = asyncio.create_task(orchestrator.run())

    await asyncio.gather(*agent_tasks)
    orchestrator.stop()
    await orch_task

    logger.log({"event": "RUN_END", "run_id": run_id})
    logger.close()

    return _compute_real_metrics(log_path, scenario)


# ---------------------------------------------------------------------------
# Experiment runner — parallel trials
# ---------------------------------------------------------------------------

async def run_experiment(
    scenario_ids: list[str],
    n_trials: int,
    modes: list[str],
    model: str,
    token_budget: int,
    client: "AsyncAnthropic | OpenRouterClient | None" = None,
    results_dir: Path = _RESULTS_DIR,
    parallel_trials: int = 4,
) -> None:
    """Run all trials, up to ``parallel_trials`` concurrently.

    A single ``client`` instance is shared across every parallel trial.
    For ``OpenRouterClient`` its internal semaphore caps the total number of
    in-flight API calls globally, so parallelism never exceeds the rate limit.

    The CSV writer is guarded by an asyncio.Lock so concurrent coroutines
    don't interleave rows.
    """
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "summary_real.csv"

    fieldnames = [
        "run_id", "scenario", "condition", "n_agents", "trial",
        "contamination_rate", "avg_context_tokens", "p95_context_tokens",
        "n_steering_responses",
    ]
    write_header = not summary_path.exists()
    summary_fh   = open(summary_path, "a", newline="", encoding="utf-8")
    writer       = csv.DictWriter(summary_fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        summary_fh.flush()

    csv_lock = asyncio.Lock()
    rows: list[dict] = []

    # Build the full flat list of (scenario, mode, trial_number) jobs
    jobs: list[tuple] = []
    for scenario_id in scenario_ids:
        for mode_name in modes:
            for trial in range(1, n_trials + 1):
                jobs.append((scenario_id, mode_name, trial))

    # Semaphore that limits how many trials run at the same time
    trial_sem = asyncio.Semaphore(parallel_trials)

    async def _run_one(scenario_id: str, mode_name: str, trial: int) -> None:
        run_id = (
            f"{scenario_id}_{mode_name}_t{trial:02d}_{uuid.uuid4().hex[:6]}"
        )
        async with trial_sem:
            Console().rule(
                f"[cyan]{scenario_id} / {mode_name} / trial {trial}[/cyan]"
            )
            try:
                metrics = await run_trial(
                    REAL_SCENARIOS[scenario_id],
                    mode_name == "dacs",
                    model,
                    token_budget,
                    run_id,
                    client=client,
                    results_dir=results_dir,
                )
            except Exception as exc:
                Console().print(f"[red]Trial {run_id} failed: {exc}[/red]")
                return

        row = {
            "run_id":    run_id,
            "scenario":  scenario_id,
            "condition": mode_name,
            "n_agents":  metrics["n_agents"],
            "trial":     trial,
            **{k: metrics[k] for k in fieldnames[5:]},
        }
        async with csv_lock:
            rows.append(row)
            writer.writerow(row)
            summary_fh.flush()

    await asyncio.gather(*[_run_one(s, m, t) for s, m, t in jobs])

    summary_fh.close()

    # Print summary table
    c = Console()
    c.rule("[green]Real-Agent Experiment Complete[/green]")
    t = Table(title=f"Results → {summary_path}", show_lines=True)
    for col in ["run_id", "condition", "trial",
                "contamination_rate", "avg_context_tokens", "n_steering_responses"]:
        t.add_column(col)
    for row in sorted(rows, key=lambda r: (r["scenario"], r["condition"], r["trial"])):
        t.add_row(
            row["run_id"],
            row["condition"],
            str(row["trial"]),
            f"{row['contamination_rate']:.2%}",
            f"{row['avg_context_tokens']:.0f}",
            str(row["n_steering_responses"]),
        )
    c.print(t)
    c.print(
        "[yellow]Run experiments_real_agent/judge.py next to compute M1 "
        "(steering accuracy).[/yellow]"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DACS real-agent validation experiment"
    )
    parser.add_argument(
        "--scenario", nargs="+",
        default=list(REAL_SCENARIOS.keys()),
        help="Scenario ID(s) to run (default: all)",
    )
    parser.add_argument("--trials",  type=int, default=10)
    parser.add_argument(
        "--mode", choices=["dacs", "baseline", "both"], default="both",
    )
    parser.add_argument(
        "--api", choices=["minimax", "openrouter", "auto"], default="auto",
        help="LLM backend to use (default: auto-detect from env vars, prefer openrouter)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model name (default: anthropic/claude-haiku-4-5 for openrouter, "
             "MiniMax-M2.7 for minimax, or DACS_MODEL env var)",
    )
    parser.add_argument("--budget", type=int, default=_DEFAULT_T)
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory for JSONL logs and summary CSV "
             "(default: results_real_agent_haiku for openrouter, results_real_agent for minimax)",
    )
    parser.add_argument(
        "--parallel-trials", type=int, default=4,
        metavar="N",
        help="Number of trials to run concurrently (default: 4). "
             "All parallel trials share one client and therefore one rate-limit semaphore.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10,
        metavar="N",
        help="Max simultaneous in-flight LLM API calls across ALL parallel trials "
             "(default: 10, ~saturates 3-4 calls/sec at ~3 s Haiku latency). "
             "Only applies to OpenRouter backend.",
    )
    args = parser.parse_args()

    resolved_api, api_key, model = _resolve_api_and_model(args.api, args.model)
    client = _make_client(resolved_api, api_key, max_concurrent=args.max_concurrent)

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(
            "results_real_agent_haiku" if resolved_api == "openrouter"
            else "results_real_agent"
        )

    Console().print(
        f"[bold]API:[/bold] {resolved_api}  "
        f"[bold]Model:[/bold] {model}  "
        f"[bold]Results:[/bold] {results_dir}  "
        f"[bold]Parallel trials:[/bold] {args.parallel_trials}  "
        f"[bold]Max concurrent calls:[/bold] {args.max_concurrent}"
    )

    modes = ["dacs", "baseline"] if args.mode == "both" else [args.mode]
    asyncio.run(
        run_experiment(
            scenario_ids=args.scenario,
            n_trials=args.trials,
            modes=modes,
            model=model,
            token_budget=args.budget,
            client=client,
            results_dir=results_dir,
            parallel_trials=args.parallel_trials,
        )
    )


if __name__ == "__main__":
    main()
