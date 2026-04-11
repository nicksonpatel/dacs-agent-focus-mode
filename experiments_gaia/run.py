"""DACS Phase 5 — GAIA benchmark experiment runner.

Each trial runs N=3 agents concurrently, each assigned a distinct GAIA Level-1
question.  One condition (DACS) uses Registry/Focus mode switching; the other
(baseline) injects all three agents' contexts simultaneously.

The key difference from experiments_real_agent/:
- Agent system prompt is GAIA Q&A style: work toward an answer, ask the
  orchestrator for help on reasoning sub-steps via [[STEER: ...]], then
  emit the final answer via [[ANSWER: <your answer>]].
- Primary metric is exact-match accuracy against GAIA gold answers (computed
  in judge.py), not keyword matching on design decisions.

Usage
-----
    python -m experiments_gaia.run --mode both --trials 5
    python -m experiments_gaia.run --mode dacs --trials 5 --batches gaia_b01_n3 gaia_b02_n3
    python -m experiments_gaia.run --mode both --trials 5 --parallel-trials 2

Environment
-----------
    OPENROUTER_API_KEY   (or OR_API_KEY)
    DACS_MODEL           — override model (default: anthropic/claude-haiku-4-5)
    DACS_T               — token budget (default: 204800)

Results
-------
    results_gaia/<run_id>.jsonl         — per-trial event log
    results_gaia/summary_gaia.csv       — aggregated metrics (M2, M3)
    Run experiments_gaia/judge.py after to compute M1 (exact-match accuracy).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agents.base_agent import BaseAgent
from experiments_gaia.scenario_defs import GAIA_BATCHES, GAIA_SCENARIOS, GAIABatchScenario, GAIAAgentSpec
from src.context_builder import ContextBuilder
from src.logger import Logger
from src.monitor import TerminalMonitor
from src.openrouter_client import OpenRouterClient
from src.orchestrator import Orchestrator
from src.protocols import AgentStatus, SteeringRequestQueue, UrgencyLevel
from src.registry import RegistryManager

load_dotenv()

_OPENROUTER_DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
_MINIMAX_DEFAULT_MODEL    = "MiniMax-M2.7"
_DEFAULT_T                = int(os.environ.get("DACS_T", "204800"))
_RESULTS_DIR              = Path("results_gaia")

# Per-agent LLM call parameters
_AGENT_MAX_STEPS  = 10
_AGENT_MAX_STEER  = 2    # agents ask 1-2 sub-questions; keeps trials short
_AGENT_MAX_TOKENS = 600

# Regexes for parsing agent output
_STEER_RE  = re.compile(r"\[\[STEER:\s*(.*?)\]\]",    re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"\[\[ANSWER:\s*(.*?)\]\]",   re.IGNORECASE | re.DOTALL)
_DONE_RE   = re.compile(r"\[\[DONE\]\]",               re.IGNORECASE)
_MAX_QUESTION_CHARS = 400


# ---------------------------------------------------------------------------
# GAIA-specific LLM agent
# ---------------------------------------------------------------------------

class GAIAAgent(BaseAgent):
    """LLM agent for the GAIA evaluation.

    Unlike the generic LLMAgent used in Phase 4, this agent focuses on
    answering a single factual question and emits [[ANSWER: ...]] when done.
    It asks the orchestrator for help on 1-2 reasoning sub-steps via
    [[STEER: ...]] to ensure steering events actually fire.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        spec: GAIAAgentSpec,
        client: "AsyncAnthropic | OpenRouterClient",
        model: str,
        registry: "RegistryManager",
        queue: "SteeringRequestQueue",
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            task_description=f"Answer GAIA question ({spec.domain}): {spec.question[:60]}",
            registry=registry,
            queue=queue,
        )
        self._client  = client
        self._model   = model
        self._spec    = spec
        self._answer: str = ""   # final answer extracted from [[ANSWER: ...]]
        self._system_prompt = self._build_system_prompt(spec)

    @staticmethod
    def _build_system_prompt(spec: GAIAAgentSpec) -> str:
        return (
            "You are a research agent in a multi-agent orchestration system, "
            "working to answer a factual question.\n\n"
            f"YOUR QUESTION:\n{spec.question}\n\n"
            "INSTRUCTIONS\n"
            "============\n"
            "1. Think step-by-step about how to answer this question.\n"
            "2. You MUST consult the orchestrator on at least one reasoning sub-step "
            "before giving your final answer. Use [[STEER: <your sub-question>]] "
            "to ask the orchestrator for clarification or verification of a key fact "
            "you are uncertain about. Then stop — you will receive guidance before "
            "continuing.\n"
            "3. After receiving guidance, incorporate it and finalize your answer.\n"
            "4. When you have your final answer, emit it on its own line:\n\n"
            "   [[ANSWER: <your concise answer>]]\n\n"
            "Then emit [[DONE]] on the next line.\n\n"
            "Keep answers concise — a single word, number, name, or short phrase.\n"
            "You may use [[STEER: ...]] at most 2 times.\n"
            "Do NOT emit [[ANSWER: ...]] until you have used [[STEER: ...]] at "
            "least once."
        )

    def get_answer(self) -> str:
        """Return the final answer the agent emitted, or '' if none."""
        return self._answer

    async def _execute(self) -> None:
        conversation: list[dict] = [
            {
                "role": "user",
                "content": (
                    "Begin working toward your answer. Show your reasoning. "
                    "Remember to use [[STEER: ...]] to consult the orchestrator on "
                    "a key reasoning sub-step before giving your final answer."
                ),
            }
        ]
        steering_count = 0
        answered = False

        for _step in range(_AGENT_MAX_STEPS):
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=_AGENT_MAX_TOKENS,
                system=self._system_prompt,
                messages=conversation,
            )
            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text = block.text
                    break

            summary_tail = text.strip().replace("\n", " ")[-80:] or "(working)"
            self._push_update(AgentStatus.RUNNING, summary_tail, UrgencyLevel.LOW)

            conversation.append({"role": "assistant", "content": text})

            # Check for [[ANSWER: ...]]
            answer_match = _ANSWER_RE.search(text)
            if answer_match:
                self._answer = answer_match.group(1).strip()
                answered = True

            # Check for [[DONE]] or final answer already extracted
            if _DONE_RE.search(text) or (answered and steering_count > 0):
                break

            # Check for [[STEER: ...]]
            steer_match = _STEER_RE.search(text)
            if steer_match and steering_count < _AGENT_MAX_STEER and not answered:
                raw_question = steer_match.group(1).strip()
                question = raw_question[:_MAX_QUESTION_CHARS]
                steering_count += 1

                steering_resp = await self._request_steering(
                    relevant_context=self._recent_output(k=4),
                    question=question,
                    blocking=True,
                    urgency=UrgencyLevel.MEDIUM,
                )

                guidance_msg = f"Orchestrator guidance: {steering_resp.response_text}"
                conversation.append({"role": "user", "content": guidance_msg})

            await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Metrics from JSONL log
# ---------------------------------------------------------------------------

def _compute_gaia_metrics(log_path: str, scenario: GAIABatchScenario) -> dict[str, Any]:
    """Extract M2 (contamination) and M3 (context tokens) from event log."""
    import statistics

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
        if e.get("event") == "CONTEXT_BUILT" and e.get("mode") in {"FOCUS", "FLAT"}
    ]

    contaminated = 0
    for resp in steering_responses:
        aid  = resp["agent_id"]
        text = resp.get("response_text", "").lower()
        for oid in agent_ids:
            if oid != aid and oid in text:
                contaminated += 1
                break

    context_tokens = [e["token_count"] for e in context_built if "token_count" in e]

    return {
        "contamination_rate":   contaminated / len(steering_responses) if steering_responses else 0.0,
        "avg_context_tokens":   statistics.mean(context_tokens) if context_tokens else 0.0,
        "p95_context_tokens":   (
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
    scenario: GAIABatchScenario,
    focus_mode: bool,
    model: str,
    token_budget: int,
    run_id: str,
    client: "AsyncAnthropic | OpenRouterClient | None" = None,
    results_dir: Path = _RESULTS_DIR,
) -> tuple[dict[str, Any], list[GAIAAgent]]:
    """Run one trial; return (metrics_dict, agents_list).

    The caller may inspect agent.get_answer() after the trial.
    """
    results_dir.mkdir(exist_ok=True)
    log_path = str(results_dir / f"{run_id}.jsonl")

    logger  = Logger(log_path)
    monitor = TerminalMonitor(token_budget=token_budget)
    logger.add_sink(monitor.handle)

    logger.log({
        "event":        "RUN_START",
        "run_id":       run_id,
        "condition":    "dacs" if focus_mode else "baseline",
        "scenario":     scenario.batch_id,
        "n_agents":     len(scenario.agents),
        "focus_mode":   focus_mode,
        "model":        model,
        "token_budget": token_budget,
        "harness":      "gaia",
    })

    registry = RegistryManager(logger)
    queue    = SteeringRequestQueue(logger)
    cb       = ContextBuilder(token_budget, logger)
    registry.set_context_builder(cb)

    if client is None:
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

    agents: list[GAIAAgent] = []
    for spec in scenario.agents:
        agent = GAIAAgent(
            agent_id=spec.agent_id,
            spec=spec,
            client=client,
            model=model,
            registry=registry,
            queue=queue,
        )
        registry.register(spec.agent_id, agent.task_description)
        orchestrator.register_agent(agent)
        agents.append(agent)

    agent_tasks = [asyncio.create_task(a.run()) for a in agents]
    orch_task   = asyncio.create_task(orchestrator.run())

    await asyncio.gather(*agent_tasks)
    orchestrator.stop()
    await orch_task

    # Log agent answers for judge.py to pick up
    for spec, agent in zip(scenario.agents, agents):
        logger.log({
            "event":     "AGENT_ANSWER",
            "agent_id":  spec.agent_id,
            "answer":    agent.get_answer(),
            "gold":      spec.answer,
            "domain":    spec.domain,
            "question":  spec.question,
        })

    logger.log({"event": "RUN_END", "run_id": run_id})
    logger.close()

    return _compute_gaia_metrics(log_path, scenario), agents


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _resolve_api_and_model(api: str, model_override: str | None) -> tuple[str, str, str]:
    or_key      = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY", "")
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    if api == "auto":
        api = "openrouter" if or_key else "minimax"
    if api == "openrouter":
        if not or_key:
            raise RuntimeError("Set OPENROUTER_API_KEY in .env")
        model = model_override or os.environ.get("DACS_MODEL") or _OPENROUTER_DEFAULT_MODEL
        return "openrouter", or_key, model
    else:
        if not minimax_key:
            raise RuntimeError("Set MINIMAX_API_KEY in .env")
        model = model_override or os.environ.get("DACS_MODEL") or _MINIMAX_DEFAULT_MODEL
        return "minimax", minimax_key, model


def _make_client(api: str, api_key: str, max_concurrent: int = 8):
    if api == "openrouter":
        return OpenRouterClient(api_key=api_key, max_concurrent=max_concurrent)
    return AsyncAnthropic(
        api_key=api_key,
        base_url="https://api.minimax.io/anthropic",
    )


async def run_experiment(
    batch_ids: list[str],
    n_trials: int,
    modes: list[str],
    model: str,
    token_budget: int,
    client: Any = None,
    results_dir: Path = _RESULTS_DIR,
    parallel_trials: int = 2,
) -> None:
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "summary_gaia.csv"

    fieldnames = [
        "run_id", "batch_id", "condition", "n_agents", "trial",
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

    jobs: list[tuple] = []
    for batch_id in batch_ids:
        for mode_name in modes:
            for trial in range(1, n_trials + 1):
                jobs.append((batch_id, mode_name, trial))

    trial_sem = asyncio.Semaphore(parallel_trials)

    async def _run_one(batch_id: str, mode_name: str, trial: int) -> None:
        run_id = f"{batch_id}_{mode_name}_t{trial:02d}_{uuid.uuid4().hex[:6]}"
        async with trial_sem:
            Console().rule(f"[cyan]{batch_id} / {mode_name} / trial {trial}[/cyan]")
            try:
                metrics, _ = await run_trial(
                    GAIA_SCENARIOS[batch_id],
                    mode_name == "dacs",
                    model,
                    token_budget,
                    run_id,
                    client=client,
                    results_dir=results_dir,
                )
            except Exception as exc:
                Console().print(f"[red]Trial {run_id} failed: {exc}[/red]")
                import traceback; traceback.print_exc()
                return

        row = {
            "run_id":    run_id,
            "batch_id":  batch_id,
            "condition": mode_name,
            "n_agents":  metrics["n_agents"],
            "trial":     trial,
            "contamination_rate":   metrics["contamination_rate"],
            "avg_context_tokens":   metrics["avg_context_tokens"],
            "p95_context_tokens":   metrics["p95_context_tokens"],
            "n_steering_responses": metrics["n_steering_responses"],
        }
        async with csv_lock:
            rows.append(row)
            writer.writerow(row)
            summary_fh.flush()

    await asyncio.gather(*[_run_one(b, m, t) for b, m, t in jobs])
    summary_fh.close()

    c = Console()
    c.rule("[green]GAIA Experiment Complete[/green]")
    t = Table(title=f"Results → {summary_path}", show_lines=True)
    for col in ["run_id", "condition", "trial", "contamination_rate", "avg_context_tokens"]:
        t.add_column(col)
    for row in sorted(rows, key=lambda r: (r["batch_id"], r["condition"], r["trial"])):
        t.add_row(
            row["run_id"],
            row["condition"],
            str(row["trial"]),
            f"{row['contamination_rate']:.2%}",
            f"{row['avg_context_tokens']:.0f}",
        )
    c.print(t)
    c.print("[yellow]Run experiments_gaia/judge.py next to compute M1 (exact-match accuracy).[/yellow]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    all_batch_ids = [b.batch_id for b in GAIA_BATCHES]
    parser = argparse.ArgumentParser(description="DACS GAIA Phase-5 experiment")
    parser.add_argument("--batches", nargs="+", default=all_batch_ids,
                        help="Batch ID(s) to run (default: all 10)")
    parser.add_argument("--trials",  type=int, default=5)
    parser.add_argument("--mode",    choices=["dacs", "baseline", "both"], default="both")
    parser.add_argument("--api",     choices=["minimax", "openrouter", "auto"], default="auto")
    parser.add_argument("--model",   default=None)
    parser.add_argument("--budget",  type=int, default=_DEFAULT_T)
    parser.add_argument("--results-dir", default=str(_RESULTS_DIR))
    parser.add_argument("--parallel-trials", type=int, default=2)
    parser.add_argument("--max-concurrent",  type=int, default=8)
    args = parser.parse_args()

    resolved_api, api_key, model = _resolve_api_and_model(args.api, args.model)
    client = _make_client(resolved_api, api_key, args.max_concurrent)
    modes  = ["dacs", "baseline"] if args.mode == "both" else [args.mode]

    asyncio.run(run_experiment(
        batch_ids=args.batches,
        n_trials=args.trials,
        modes=modes,
        model=model,
        token_budget=args.budget,
        client=client,
        results_dir=Path(args.results_dir),
        parallel_trials=args.parallel_trials,
    ))


if __name__ == "__main__":
    main()
