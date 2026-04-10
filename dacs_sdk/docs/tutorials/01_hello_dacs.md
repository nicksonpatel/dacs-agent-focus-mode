# Tutorial 1: Hello, DACS

This tutorial introduces the minimal DACS setup: one agent, one orchestrator, one steering interaction.

## What we'll build

A single agent that:
1. Announces it has started work
2. Asks the orchestrator for a format decision
3. Completes with the chosen format

## Prerequisites

```bash
pip install dacs-agent
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Step 1 — Subclass BaseAgent

```python
# hello_dacs.py
import asyncio
from dacs import DACSRuntime, BaseAgent, UrgencyLevel


class FormatAgent(BaseAgent):
    """Asks the orchestrator which output format to use."""

    async def _execute(self) -> None:
        # Step 1: Push a heartbeat so the registry knows what we're doing
        self._push_update("Processed input data — ready to export.")

        # Step 2: Ask for a decision
        resp = await self._request_steering(
            context=(
                "I have processed 10 000 rows of customer records. "
                "The downstream system supports JSON, CSV, and Parquet."
            ),
            question="Which format should I export to?",
            urgency=UrgencyLevel.MEDIUM,
        )

        # Step 3: Use the response
        chosen_format = resp.response_text
        self._push_update(f"Exporting as: {chosen_format}")
        print(f"\nOrchestrator chose: {chosen_format}")
```

## Step 2 — Run with DACSRuntime

```python
async def main() -> None:
    async with DACSRuntime(
        model="claude-3-haiku-20240307",
        verbose=True,        # prints a live event feed
        log_path="hello.jsonl",  # saves the full JSONL event log
    ) as runtime:
        runtime.add_agent(
            FormatAgent(agent_id="exporter", task="Export processed customer records")
        )
        await runtime.run()


if __name__ == "__main__":
    asyncio.run(main())
```

## Step 3 — Run it

```bash
python hello_dacs.py
```

You'll see output like:

```
2024-01-01 00:00:01  RUN_START               run=a1b2c3d4 model=claude-3-haiku-20240307 focus=on
2024-01-01 00:00:01  REGISTRY_UPDATE         agent=exporter status=RUNNING
2024-01-01 00:00:01  STEERING_REQUEST        agent=exporter urgency=MEDIUM
2024-01-01 00:00:01  CONTEXT_BUILT           mode=FOCUS 1240tok [████░░░░░░░░]
2024-01-01 00:00:02  LLM_CALL                in=1240 out=42 lat=830ms
2024-01-01 00:00:02  STEERING_RESPONSE       agent=exporter ctx=1240tok [████░░░░░░░░]

Orchestrator chose: JSON
```

## What just happened?

1. `DACSRuntime` created the `RegistryManager`, `ContextBuilder`, `SteeringRequestQueue`, and `Orchestrator`.
2. `FormatAgent` ran as an asyncio task alongside the orchestrator.
3. When `_request_steering()` was called, a `SteeringRequest` was enqueued.
4. The orchestrator transitioned `REGISTRY → FOCUS(exporter)`, built a **focused context** (only the exporter's context + task), made one LLM call, then transitioned back `FOCUS → REGISTRY`.
5. The response was delivered to the agent's internal queue and control returned to `_execute()`.

## Using StepAgent instead

No subclassing needed:

```python
from dacs import StepAgent

agent = StepAgent(
    agent_id="exporter",
    task="Export processed customer records",
    steps=[
        {"summary": "Processed input data — ready to export."},
        {
            "summary": "Choosing output format",
            "question": "Which format should I export to: JSON, CSV, or Parquet?",
            "urgency": "MEDIUM",
        },
        {"summary": "Exporting data"},
    ],
)
```

## Next

[Tutorial 2: Three Concurrent Agents →](02_three_agents.md)
