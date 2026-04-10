<div align="center">

# DACS — Dynamic Attentional Context Scoping

**Agent-triggered focus sessions for isolated per-agent steering in multi-agent LLM systems**

[![PyPI](https://img.shields.io/pypi/v/dacs-agent?color=blue)](https://pypi.org/project/dacs-agent/)
[![Python](https://img.shields.io/pypi/pyversions/dacs-agent)](https://pypi.org/project/dacs-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

</div>

---

DACS solves a fundamental problem in multi-agent LLM systems: **context pollution**.

When multiple agents share a single orchestrator, their contexts bleed into each other's reasoning — reducing accuracy, wasting tokens, and causing cross-talk. DACS introduces **asymmetric, agent-triggered context isolation**: agents request steering when they have a decision point, and the orchestrator enters an exclusive FOCUS session for that agent while maintaining only compact registry snapshots (≤200 tokens/agent) for all others.

## Benchmark Results

Across **204 trials** over 4 experiment phases (N ∈ {3, 5, 10} agents, decision density D ∈ {1, 8, 15}):

| Metric | DACS | Flat-context Baseline |
|---|---|---|
| Steering accuracy | **90.0 – 98.4 %** | 21.0 – 60.0 % |
| Context contamination | **< 4 %** | 18 – 42 % |
| Context at steering time | 2.12× – 3.53× **smaller** | (full flat context) |
| Context scaling | **~25 tok/agent** | ~820 tok/agent |

All comparisons significant at p < 0.0001 (Welch's t-test).

## Installation

```bash
pip install dacs-agent
```

With the optional Rich terminal monitor:

```bash
pip install "dacs-agent[monitor]"
```

## Quick Start

```python
import asyncio
from dacs import DACSRuntime, BaseAgent, UrgencyLevel


class DataAgent(BaseAgent):
    async def _execute(self):
        self._push_update("Loaded dataset: 50k rows.")

        resp = await self._request_steering(
            context="Dataset has 3 date formats: ISO, US, EU.",
            question="Which format should I normalise to?",
            urgency=UrgencyLevel.HIGH,
        )

        self._push_update(f"Normalising to: {resp.response_text}")
        # ... do the actual work ...
        self._push_update("Normalisation complete.")


async def main():
    async with DACSRuntime(
        model="claude-3-haiku-20240307",
        verbose=True,          # prints a live event feed
    ) as runtime:
        runtime.add_agent(DataAgent(
            agent_id="data",
            task="Normalise date formats in the uploaded dataset",
        ))
        await runtime.run()


asyncio.run(main())
```

## How DACS Works

```
Orchestrator                  Agents
─────────────────────────────────────────────────────────────
REGISTRY mode                 a1, a2, a3 running concurrently
  (≤200 tok/agent)            │
                              a2 hits a decision point
  ◄────── SteeringRequest ─── a2
  FOCUS(a2) mode
  (full context of a2 +       a1 ──heartbeat──► Registry
   compressed R_{a1,a3})      a3 ──heartbeat──► Registry
  LLM call — isolated
  ───── SteeringResponse ────► a2 resumes
  REGISTRY mode               all agents running
```

The orchestrator **never** sees more than one agent's full context at a time.  
High-urgency requests can **interrupt** an active FOCUS session.

## Using StepAgent (no subclassing needed)

For quick prototyping, use the built-in `StepAgent`:

```python
from dacs import DACSRuntime, StepAgent

agent = StepAgent(
    agent_id="researcher",
    task="Find and summarise three papers on RLHF",
    steps=[
        {"summary": "Searching arXiv", "sleep": 0.5},
        {
            "summary": "Found 12 candidates, need to pick 3",
            "question": "Prioritise recency or citation count?",
            "urgency": "MEDIUM",
        },
        {"summary": "Writing summaries"},
    ],
)

async def main():
    async with DACSRuntime(model="claude-3-haiku-20240307") as runtime:
        runtime.add_agent(agent)
        await runtime.run()
```

## Custom API Endpoints

```python
# OpenRouter, Azure, MiniMax, etc.
runtime = DACSRuntime(
    model="anthropic/claude-3-haiku",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-...",
)
```

## Running the Baseline for Comparison

```python
runtime = DACSRuntime(
    model="claude-3-haiku-20240307",
    focus_mode=False,   # flat-context baseline
)
```

## Multi-Agent Example

```python
import asyncio
from dacs import DACSRuntime, BaseAgent, UrgencyLevel


class CodeAgent(BaseAgent):
    async def _execute(self):
        self._push_update("Scaffolded project structure.")
        resp = await self._request_steering(
            context="Auth module: JWT or session cookies?",
            question="Which auth strategy for a public API?",
        )
        self._push_update(f"Auth decision: {resp.response_text}")
        self._push_update("Implemented auth layer.")


class ReviewAgent(BaseAgent):
    async def _execute(self):
        self._push_update("Reviewing PR #42.")
        resp = await self._request_steering(
            context="Function is 300 lines with 3 responsibility areas.",
            question="Suggest refactoring strategy — break into 3 modules or use composition?",
        )
        self._push_update(f"Refactor plan: {resp.response_text}")
        self._push_update("Review complete.")


class DocAgent(BaseAgent):
    async def _execute(self):
        self._push_update("Generating API docs.")
        self._push_update("Docs complete.")


async def main():
    async with DACSRuntime(
        model="claude-3-haiku-20240307",
        verbose=True,
        log_path="run.jsonl",
    ) as runtime:
        runtime.add_agent(CodeAgent(agent_id="coder", task="Implement REST API auth"))
        runtime.add_agent(ReviewAgent(agent_id="reviewer", task="Review PR #42"))
        runtime.add_agent(DocAgent(agent_id="doc", task="Generate API documentation"))
        await runtime.run()


asyncio.run(main())
```

## Low-Level API

You can wire components directly for full control:

```python
from dacs import (
    Logger, RegistryManager, ContextBuilder, SteeringRequestQueue, Orchestrator
)
from anthropic import AsyncAnthropic

logger = Logger("my_run.jsonl")
registry = RegistryManager(logger)
queue = SteeringRequestQueue(logger)
cb = ContextBuilder(token_budget=200_000, logger=logger)
registry.set_context_builder(cb)

orchestrator = Orchestrator(
    registry=registry,
    queue=queue,
    context_builder=cb,
    llm_client=AsyncAnthropic(api_key="..."),
    model="claude-3-haiku-20240307",
    token_budget=200_000,
    logger=logger,
)
```

## Event Log

Every DACS run produces a JSONL log with full observability:

```json
{"event": "RUN_START", "ts": "2024-01-01T00:00:00Z", "focus_mode": true, "n_agents": 3}
{"event": "STEERING_REQUEST", "agent_id": "coder", "urgency": "MEDIUM"}
{"event": "CONTEXT_BUILT", "mode": "FOCUS", "agent_id": "coder", "token_count": 1842}
{"event": "LLM_CALL", "in_tokens": 1842, "out_tokens": 134, "latency_ms": 820}
{"event": "STEERING_RESPONSE", "agent_id": "coder", "context_size_at_time": 1842}
```

Key events: `RUN_START`, `REGISTRY_UPDATE`, `CONTEXT_BUILT`, `STEERING_REQUEST`, `STEERING_RESPONSE`, `LLM_CALL`, `TRANSITION`, `INTERRUPT`, `FOCUS_TIMEOUT`.

## Comparison with Other Frameworks

| | **DACS** | LangGraph | CrewAI | OpenAI Agents SDK |
|---|---|---|---|---|
| Context isolation mechanism | **Agent-triggered, asymmetric** | None | None | None |
| Per-agent context scoping | **FOCUS/REGISTRY modes** | — | — | — |
| Registry (≤200 tok/agent) | ✅ | — | — | — |
| HIGH-urgency interrupts | ✅ | — | — | — |
| Token budget enforcement | **Deterministic (tiktoken)** | Soft | None | Soft |
| Flat-context baseline | ✅ built-in | N/A | N/A | N/A |
| LLM provider | Anthropic (+compatible) | Any | Any | OpenAI + others |

## Citation

```bibtex
@article{patel2025dacs,
  title   = {Dynamic Attentional Context Scoping for Multi-Agent LLM Orchestration},
  author  = {Patel, Nickson},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025},
}
```

## License

MIT — see [LICENSE](LICENSE).
