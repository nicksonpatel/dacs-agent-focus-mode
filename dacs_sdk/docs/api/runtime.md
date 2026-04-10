# DACSRuntime

The high-level entry point for DACS.  Wires all components together in a single context manager.

::: dacs._runtime.DACSRuntime
    options:
      show_source: true
      members:
        - __init__
        - add_agent
        - run
        - ask
        - orchestrator
        - registry

## Example

```python
import asyncio
from dacs import DACSRuntime, StepAgent

async def main():
    async with DACSRuntime(
        model="claude-3-haiku-20240307",
        token_budget=200_000,
        log_path="run.jsonl",
        focus_mode=True,
        verbose=True,
    ) as runtime:
        runtime.add_agent(StepAgent(
            agent_id="a1",
            task="Summarise quarterly earnings report",
            steps=[
                {"summary": "Reading report"},
                {"summary": "Identifying key metrics",
                 "question": "Focus on revenue or margin trends?"},
                {"summary": "Writing summary"},
            ],
        ))
        await runtime.run()

asyncio.run(main())
```

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"claude-3-haiku-20240307"` | LLM model identifier |
| `api_key` | `str \| None` | `None` → env `ANTHROPIC_API_KEY` | Anthropic API key |
| `base_url` | `str \| None` | `None` | Custom endpoint (OpenRouter, MiniMax, etc.) |
| `token_budget` | `int` | `200_000` | Hard token cap on context window |
| `log_path` | `str \| None` | `"dacs_run.jsonl"` | JSONL log path; `None` disables file logging |
| `focus_mode` | `bool` | `True` | `True` = DACS; `False` = flat-context baseline |
| `focus_timeout` | `int` | `60` | Seconds before abandoning an idle FOCUS session |
| `verbose` | `bool` | `False` | Attach `TerminalMonitor` for live output |
