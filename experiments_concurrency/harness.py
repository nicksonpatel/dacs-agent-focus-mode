"""Trial harness for the concurrency & interruption experiment.

Wires up: logger → monitor → registry → TrackedQueue → context_builder →
           orchestrator → agents → UserInjector → InlineJudge

Four conditions are produced by combining two independent flags:
  focus_mode  (True = DACS, False = flat-context baseline)
  inject      (True = user messages fired mid-trial, False = clean run)

The condition string stored in summary CSV is therefore one of:
  "dacs_concurrent", "dacs_clean", "baseline_concurrent", "baseline_clean"

TrackedQueue
------------
A thin SteeringRequestQueue subclass that additionally calls
InlineJudge.track_request() when a request is enqueued. This is the only
way to associate request_id → question text without modifying src/, since
the STEERING_REQUEST log event does not include the question field.

Metrics computed
----------------
Beyond the standard fields from experiments.metrics.compute_metrics we
add judge-specific fields computed directly from the JSONL:
    avg_judge_score         — mean of all JUDGE_SCORE events in this trial
    avg_steering_score      — mean of JUDGE_SCORE where event_type_judged="steering"
    avg_user_score          — mean of JUDGE_SCORE where event_type_judged="user"
    min_judge_score         — minimum score across all events
    total_judged            — count of JUDGE_SCORE events
    inject_count            — count of INJECTION events
    competing_requests      — count of INTERRUPT events (queue collisions)
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from agents.generic_agent import GenericAgent
from experiments_concurrency.event_injector import UserInjector
from experiments_concurrency.rubric_judge import InlineJudge
from experiments_concurrency.scenario_defs import ConcurrencyScenario
from src.context_builder import ContextBuilder
from src.logger import Logger
from src.monitor import TerminalMonitor
from src.orchestrator import Orchestrator
from src.protocols import SteeringRequest, SteeringRequestQueue
from src.registry import RegistryManager


# ---------------------------------------------------------------------------
# TrackedQueue — notifies InlineJudge of question text per request_id
# ---------------------------------------------------------------------------

class _TrackedQueue(SteeringRequestQueue):
    """Extends SteeringRequestQueue to forward SteeringRequest objects to
    InlineJudge before they are consumed by the orchestrator."""

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self._judge: InlineJudge | None = None

    def attach_judge(self, judge: InlineJudge) -> None:
        self._judge = judge

    def enqueue(self, request: SteeringRequest) -> None:
        super().enqueue(request)
        if self._judge is not None:
            self._judge.track_request(request)


# ---------------------------------------------------------------------------
# Judge-specific metrics parsed from JSONL
# ---------------------------------------------------------------------------

def _compute_judge_metrics(log_path: str) -> dict[str, Any]:
    """Parse JUDGE_SCORE, INJECTION, and INTERRUPT events from a trial JSONL."""
    all_scores: list[int] = []
    steering_scores: list[int] = []
    user_scores: list[int] = []
    inject_count = 0
    competing_requests = 0

    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = ev.get("event")
                if name == "JUDGE_SCORE":
                    score = ev.get("score", 0)
                    if score > 0:  # score=0 means judge_error
                        all_scores.append(score)
                        if ev.get("event_type_judged") == "steering":
                            steering_scores.append(score)
                        elif ev.get("event_type_judged") == "user":
                            user_scores.append(score)
                elif name == "INJECTION":
                    inject_count += 1
                elif name == "INTERRUPT":
                    competing_requests += 1
    except FileNotFoundError:
        pass

    def _mean(lst: list[int]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "avg_judge_score":    round(_mean(all_scores), 3),
        "avg_steering_score": round(_mean(steering_scores), 3),
        "avg_user_score":     round(_mean(user_scores), 3),
        "min_judge_score":    min(all_scores) if all_scores else 0,
        "total_judged":       len(all_scores),
        "inject_count":       inject_count,
        "competing_requests": competing_requests,
    }


# ---------------------------------------------------------------------------
# Main trial runner
# ---------------------------------------------------------------------------

async def run_concurrent_trial(
    scenario: ConcurrencyScenario,
    focus_mode: bool,
    inject: bool,
    model: str,
    token_budget: int,
    run_id: str,
) -> dict[str, Any]:
    """Run one full trial and return combined metrics dict.

    Parameters
    ----------
    scenario    ConcurrencyScenario with agents + injection schedule.
    focus_mode  True = DACS, False = flat-context baseline.
    inject      True = fire user messages per schedule. False = clean run.
    model       LLM model name.
    token_budget  Token cap T for ContextBuilder.
    run_id      Unique identifier for this trial (used as filename stem).
    """
    log_path = f"results_concurrency/{run_id}.jsonl"
    Path("results_concurrency").mkdir(exist_ok=True)

    logger  = Logger(log_path)
    monitor = TerminalMonitor(token_budget=token_budget)
    logger.add_sink(monitor.handle)

    condition = (
        ("dacs" if focus_mode else "baseline")
        + ("_concurrent" if inject else "_clean")
    )

    logger.log({
        "event":         "RUN_START",
        "run_id":        run_id,
        "condition":     condition,
        "scenario":      scenario.scenario_id,
        "n_agents":      len(scenario.agents),
        "focus_mode":    focus_mode,
        "inject":        inject,
        "model":         model,
        "token_budget":  token_budget,
    })

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------
    registry = RegistryManager(logger)
    queue    = _TrackedQueue(logger)          # TrackedQueue, not plain SteeringRequestQueue
    cb       = ContextBuilder(token_budget, logger)
    registry.set_context_builder(cb)

    client = AsyncAnthropic(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url="https://api.minimax.io/anthropic",
        timeout=120.0,  # 2-min hard timeout per LLM call — prevents indefinite hangs
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

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    agents = []
    agent_task_map: dict[str, str] = {}
    for spec in scenario.agents:
        agent = GenericAgent(
            agent_id=spec.agent_id,
            task_description=spec.task_description,
            registry=registry,
            queue=queue,
            steps=spec.steps or [],
        )
        registry.register(spec.agent_id, spec.task_description)
        orchestrator.register_agent(agent)
        agents.append(agent)
        agent_task_map[spec.agent_id] = spec.task_description

    # ------------------------------------------------------------------
    # InlineJudge
    # ------------------------------------------------------------------
    judge = InlineJudge(
        llm_client=client,
        model=model,
        logger=logger,
        agent_task_descriptions=agent_task_map,
    )
    logger.add_sink(judge.on_event)
    queue.attach_judge(judge)

    # ------------------------------------------------------------------
    # Optional UserInjector
    # ------------------------------------------------------------------
    injector: UserInjector | None = None
    if inject and scenario.user_injection_schedule:
        injector = UserInjector(
            orchestrator=orchestrator,
            injection_schedule=scenario.user_injection_schedule,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Run all tasks concurrently
    # ------------------------------------------------------------------
    agent_tasks  = [asyncio.create_task(a.run()) for a in agents]
    orch_task    = asyncio.create_task(orchestrator.run())
    inject_task  = asyncio.create_task(injector.run()) if injector else None
    judge_task   = asyncio.create_task(judge.run_worker())

    # Wait for all agents to finish
    await asyncio.gather(*agent_tasks)

    # Stop orchestrator and injector (injector may still be sleeping)
    orchestrator.stop()
    if inject_task is not None:
        inject_task.cancel()
        try:
            await inject_task
        except asyncio.CancelledError:
            pass

    await orch_task

    # Drain judge queue then stop
    judge.stop()
    await judge_task

    logger.log({"event": "RUN_END", "run_id": run_id})
    logger.close()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    # Standard keyword-based metrics (count correct_decisions for reference)
    from experiments.metrics import compute_metrics  # noqa: PLC0415
    from experiments.task_suite import ScenarioSpec as _ScenarioSpec  # noqa: PLC0415

    # Build a plain ScenarioSpec so compute_metrics can do keyword matching
    plain_spec = _ScenarioSpec(
        scenario_id=scenario.scenario_id,
        agents=scenario.agents,
    )
    standard = compute_metrics(log_path, plain_spec)
    judge_m   = _compute_judge_metrics(log_path)

    return {
        "run_id":            run_id,
        "scenario":          scenario.scenario_id,
        "condition":         condition,
        "n_agents":          len(scenario.agents),
        "focus_mode":        focus_mode,
        "inject":            inject,
        **standard,
        **judge_m,
    }
