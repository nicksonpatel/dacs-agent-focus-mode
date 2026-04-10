# Installation & Quick Start

## Requirements

- Python **3.11** or higher
- An Anthropic API key (or any Anthropic-compatible endpoint)

## Install

```bash
pip install dacs-agent
```

Extras:

| Extra | What it adds | Install command |
|---|---|---|
| `monitor` | Rich live terminal event feed | `pip install "dacs-agent[monitor]"` |
| `openai` | OpenAI-compatible client | `pip install "dacs-agent[openai]"` |
| `docs` | MkDocs + Material (for building these docs) | `pip install "dacs-agent[docs]"` |
| `dev` | pytest + pytest-asyncio | `pip install "dacs-agent[dev]"` |

## API key

Set your key in the environment before running:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or pass it directly:

```python
DACSRuntime(model="claude-3-haiku-20240307", api_key="sk-ant-...")
```

## Hello, DACS

The fastest way to get started is with `StepAgent` — no subclassing required:

```python
import asyncio
from dacs import DACSRuntime, StepAgent

agent = StepAgent(
    agent_id="writer",
    task="Write a short blog post about async Python",
    steps=[
        {"summary": "Outlining structure", "sleep": 0.2},
        {
            "summary": "Choosing code examples",
            "question": "Use asyncio.gather or asyncio.TaskGroup examples?",
            "urgency": "MEDIUM",
        },
        {"summary": "Writing draft"},
        {"summary": "Proofreading"},
    ],
)

async def main():
    async with DACSRuntime(
        model="claude-3-haiku-20240307",
        verbose=True,           # live terminal output
        log_path="run.jsonl",   # save full event log
    ) as runtime:
        runtime.add_agent(agent)
        await runtime.run()

asyncio.run(main())
```

## For a custom agent

Subclass `BaseAgent` and implement `_execute()`:

```python
from dacs import BaseAgent, UrgencyLevel

class AnalysisAgent(BaseAgent):
    async def _execute(self):
        # 1. Do some work
        self._push_update("Loaded and cleaned dataset")  # heartbeat

        # 2. Ask the orchestrator for guidance
        resp = await self._request_steering(
            context="Two outlier-removal strategies available: IQR and Z-score.",
            question="Which outlier strategy fits a financial dataset?",
            urgency=UrgencyLevel.HIGH,   # can interrupt another agent's focus session
        )

        # 3. Use the response
        self._push_update(f"Applying: {resp.response_text}")
        # ... rest of work ...
```

Then run it:

```python
async def main():
    async with DACSRuntime(model="claude-3-haiku-20240307") as rt:
        rt.add_agent(AnalysisAgent(agent_id="analyst", task="Analyse Q3 sales data"))
        await rt.run()
```

## Custom endpoints

```python
# OpenRouter
runtime = DACSRuntime(
    model="anthropic/claude-3-haiku",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-...",
)

# MiniMax M2.7 (used in all benchmark experiments)
runtime = DACSRuntime(
    model="MiniMax-M2.7",
    base_url="https://api.minimax.io/anthropic",
    api_key="...",
)
```

## What happens next?

- [Core Concepts](concepts.md) — understand the REGISTRY/FOCUS state machine
- [Tutorials](tutorials/01_hello_dacs.md) — step-by-step worked examples
- [API Reference](api/runtime.md) — full parameter docs
