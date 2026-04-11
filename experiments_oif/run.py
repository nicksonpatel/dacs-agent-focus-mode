"""DACS Phase 5 — Orchestrator-Initiated Focus (OIF) experiment runner.

Evaluates T5 user-query routing: with OIF enabled, the orchestrator enters
Focus(aᵢ) before answering a user query that references aᵢ's work.
Without OIF, it answers from the registry summary only.

Conditions
----------
    dacs_oif     — DACS focus mode + OIF T5 routing enabled
    dacs_no_oif  — DACS focus mode, OIF disabled (registry-only user answers)
    baseline     — flat context (no DACS, no OIF)

Usage
-----
    python -m experiments_oif.run --api openrouter --mode all --trials 10
    python -m experiments_oif.run --api openrouter --mode dacs_oif --trials 10 --scenario oif1_n3

Environment
-----------
    OPENROUTER_API_KEY  (or OR_API_KEY)
    MINIMAX_API_KEY
    DACS_MODEL          — override model name
    DACS_T              — token budget (default 204800)

Results
-------
    <results-dir>/<run_id>.jsonl    — per-trial log
    <results-dir>/summary_oif.csv   — aggregated OIF metrics
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
from experiments_concurrency.event_injector import UserInjector
from experiments_oif.scenario_defs import OIF_SCENARIOS, OIFScenario
from src.context_builder import ContextBuilder
from src.logger import Logger
from src.monitor import TerminalMonitor
from src.orchestrator import Orchestrator
from src.protocols import SteeringRequestQueue
from src.registry import RegistryManager

try:
    from src.openrouter_client import OpenRouterClient
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False

load_dotenv()

_OPENROUTER_DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
_MINIMAX_DEFAULT_MODEL    = "MiniMax-M2.7"
_DEFAULT_T                = int(os.environ.get("DACS_T", "204800"))
_RESULTS_DIR              = Path("results_oif")

_AGENT_MAX_STEPS  = 14
_AGENT_MAX_STEER  = 3
_AGENT_MAX_TOKENS = 800

_CONDITIONS = ("dacs_oif", "dacs_no_oif", "baseline")


# ---------------------------------------------------------------------------
# API / model resolution (mirrors experiments_real_agent/run.py)
# ---------------------------------------------------------------------------

def _resolve_api_and_model(api: str, model_override: str | None) -> tuple[str, str, str]:
    or_key      = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY", "")
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    if api == "auto":
        api = "openrouter" if or_key else "minimax"
    if api == "openrouter":
        if not or_key:
            raise RuntimeError("Set OPENROUTER_API_KEY (or OR_API_KEY) in .env.")
        model = model_override or os.environ.get("DACS_MODEL") or _OPENROUTER_DEFAULT_MODEL
        return "openrouter", or_key, model
    else:
        if not minimax_key:
            raise RuntimeError("Set MINIMAX_API_KEY in .env.")
        model = model_override or os.environ.get("DACS_MODEL") or _MINIMAX_DEFAULT_MODEL
        return "minimax", minimax_key, model


def _make_client(api: str, api_key: str):
    if api == "openrouter":
        if not _HAS_OPENROUTER:
            raise RuntimeError("src/openrouter_client.py not found.")
        return OpenRouterClient(api_key=api_key, max_concurrent=10)
    return AsyncAnthropic(
        api_key=api_key,
        base_url="https://api.minimax.io/anthropic",
    )


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_oif_metrics(
    log_path: str,
    scenario: OIFScenario,
) -> dict[str, Any]:
    """Keyword-based M1_T5 (OIF routing accuracy) + M2 contamination + M3 context."""
    import json, statistics

    agent_ids = [a.agent_id for a in scenario.agents]
    query_map = {q.target_agent_id: q for q in scenario.user_queries}

    events: list[dict] = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # --- M1_T5: keyword hit on OIF_USER_QUERY events (dacs_oif condition)
    oif_queries = [e for e in events if e.get("event") == "OIF_USER_QUERY"]
    # Also check USER_RESPONSE events (for dacs_no_oif / baseline)
    user_responses = [e for e in events if e.get("event") == "USER_RESPONSE"]

    keyword_hits = 0
    total_queries = len(scenario.user_queries)

    for q_def in scenario.user_queries:
        # Try OIF_USER_QUERY first (dacs_oif condition)
        for ev in oif_queries:
            if ev.get("agent_id") == q_def.target_agent_id:
                resp = ev.get("response_text", "").lower()
                if any(kw.lower() in resp for kw in q_def.answer_keywords):
                    keyword_hits += 1
                break
        else:
            # Fall back to USER_RESPONSE (no-OIF conditions)
            for ev in user_responses:
                if any(kw.lower() in ev.get("message", "").lower()
                       for kw in q_def.message.lower().split()[:4]):
                    resp = ev.get("response_text", "").lower()
                    if any(kw.lower() in resp for kw in q_def.answer_keywords):
                        keyword_hits += 1
                    break

    # --- M2 contamination (steering responses only)
    steering_responses = [e for e in events if e.get("event") == "STEERING_RESPONSE"]
    contaminated = 0
    for resp_ev in steering_responses:
        aid  = resp_ev.get("agent_id", "")
        text = resp_ev.get("response_text", "").lower()
        if any(oid in text for oid in agent_ids if oid != aid):
            contaminated += 1

    # --- M3 context tokens (focus + OIF focus)
    context_events = [
        e for e in events
        if e.get("event") == "CONTEXT_BUILT"
        and e.get("mode") in {"FOCUS", "FLAT", "FOCUS_OIF_T5"}
    ]
    ctx_tokens = [e["token_count"] for e in context_events if "token_count" in e]

    return {
        "m1_t5_accuracy":       keyword_hits / total_queries if total_queries else 0.0,
        "m1_t5_hits":           keyword_hits,
        "m1_t5_total":          total_queries,
        "contamination_rate":   contaminated / len(steering_responses) if steering_responses else 0.0,
        "avg_context_tokens":   statistics.mean(ctx_tokens) if ctx_tokens else 0.0,
        "n_oif_queries_fired":  len(oif_queries),
        "n_steering_responses": len(steering_responses),
    }


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

async def run_trial(
    scenario: OIFScenario,
    condition: str,
    model: str,
    token_budget: int,
    run_id: str,
    client: Any,
    results_dir: Path = _RESULTS_DIR,
) -> dict[str, Any]:
    results_dir.mkdir(exist_ok=True)
    log_path = str(results_dir / f"{run_id}.jsonl")

    focus_mode = condition in ("dacs_oif", "dacs_no_oif")
    oif_mode   = condition == "dacs_oif"

    logger  = Logger(log_path)
    monitor = TerminalMonitor(token_budget=token_budget)
    logger.add_sink(monitor.handle)

    logger.log({
        "event":        "RUN_START",
        "run_id":       run_id,
        "condition":    condition,
        "scenario":     scenario.scenario_id,
        "n_agents":     len(scenario.agents),
        "focus_mode":   focus_mode,
        "oif_mode":     oif_mode,
        "model":        model,
        "token_budget": token_budget,
        "harness":      "oif",
    })

    registry = RegistryManager(logger)
    queue    = SteeringRequestQueue(logger)
    cb       = ContextBuilder(token_budget, logger)
    registry.set_context_builder(cb)

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
    if oif_mode:
        orchestrator.enable_oif()

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

    # Build injection schedule from scenario user queries
    injection_schedule = [(q.delay_s, q.message) for q in scenario.user_queries]
    injector = UserInjector(
        orchestrator=orchestrator,
        injection_schedule=injection_schedule,
        logger=logger,
    )

    agent_tasks   = [asyncio.create_task(a.run()) for a in agents]
    orch_task     = asyncio.create_task(orchestrator.run())
    inject_task   = asyncio.create_task(injector.run())

    await asyncio.gather(*agent_tasks)
    orchestrator.stop()
    await orch_task
    await inject_task

    logger.log({"event": "RUN_END", "run_id": run_id})
    logger.close()

    metrics = _compute_oif_metrics(log_path, scenario)
    return {
        "run_id":    run_id,
        "condition": condition,
        "scenario":  scenario.scenario_id,
        "log_path":  log_path,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Multi-trial loop
# ---------------------------------------------------------------------------

async def run_condition(
    scenario: OIFScenario,
    condition: str,
    n_trials: int,
    model: str,
    token_budget: int,
    client: Any,
    results_dir: Path,
) -> list[dict[str, Any]]:
    results = []
    for t in range(1, n_trials + 1):
        run_id = f"{scenario.scenario_id}_{condition}_t{t:02d}_{uuid.uuid4().hex[:6]}"
        print(f"  [{condition}] trial {t}/{n_trials}  run_id={run_id}")
        result = await run_trial(
            scenario=scenario,
            condition=condition,
            model=model,
            token_budget=token_budget,
            run_id=run_id,
            client=client,
            results_dir=results_dir,
        )
        results.append(result)
        print(
            f"    m1_t5={result['m1_t5_accuracy']:.3f}  "
            f"contam={result['contamination_rate']:.3f}  "
            f"oif_fired={result['n_oif_queries_fired']}"
        )
    return results


def _write_summary(results: list[dict[str, Any]], results_dir: Path) -> Path:
    path = results_dir / "summary_oif.csv"
    if not results:
        return path
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return path


def _print_summary_table(results: list[dict[str, Any]], console: Console) -> None:
    import statistics

    table = Table(title="Phase 5 OIF Results — T5 User-Query Routing")
    table.add_column("Condition",       style="bold cyan")
    table.add_column("Scenario")
    table.add_column("Trials",          justify="right")
    table.add_column("M1_T5 Acc.",      justify="right", style="green")
    table.add_column("Contam.",         justify="right")
    table.add_column("OIF Fired/trial", justify="right")

    from itertools import groupby
    key = lambda r: (r["condition"], r["scenario"])
    for (cond, scen), group in groupby(sorted(results, key=key), key=key):
        group_l = list(group)
        n = len(group_l)
        m1  = statistics.mean(r["m1_t5_accuracy"]    for r in group_l)
        m2  = statistics.mean(r["contamination_rate"] for r in group_l)
        oif = statistics.mean(r["n_oif_queries_fired"] for r in group_l)
        table.add_row(cond, scen, str(n), f"{m1:.3f}", f"{m2:.3f}", f"{oif:.1f}")

    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DACS Phase 5 OIF experiment runner")
    parser.add_argument("--api",        default="auto",
                        choices=["auto", "openrouter", "minimax"])
    parser.add_argument("--model",      default=None, help="Override model name")
    parser.add_argument("--mode",       default="all",
                        choices=["all", "dacs_oif", "dacs_no_oif", "baseline"])
    parser.add_argument("--scenario",   default="oif1_n3",
                        choices=list(OIF_SCENARIOS))
    parser.add_argument("--trials",     type=int, default=10)
    parser.add_argument("--results-dir", default="results_oif")
    args = parser.parse_args()

    api, api_key, model = _resolve_api_and_model(args.api, args.model)
    client = _make_client(api, api_key)
    scenario = OIF_SCENARIOS[args.scenario]
    results_dir = Path(args.results_dir)
    conditions  = list(_CONDITIONS) if args.mode == "all" else [args.mode]

    console = Console()
    console.print(
        f"\n[bold]DACS Phase 5 OIF Experiment[/bold]  "
        f"scenario={args.scenario}  api={api}  model={model}  "
        f"trials={args.trials}  conditions={conditions}\n"
    )

    all_results: list[dict[str, Any]] = []
    for condition in conditions:
        console.print(f"[yellow]Running condition: {condition}[/yellow]")
        results = asyncio.run(run_condition(
            scenario=scenario,
            condition=condition,
            n_trials=args.trials,
            model=model,
            token_budget=_DEFAULT_T,
            client=client,
            results_dir=results_dir,
        ))
        all_results.extend(results)

    csv_path = _write_summary(all_results, results_dir)
    console.print(f"\n[green]Summary written to {csv_path}[/green]")
    _print_summary_table(all_results, console)


if __name__ == "__main__":
    main()
