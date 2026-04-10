# Tutorial 3: Custom Agents

This tutorial covers advanced agent patterns: managing state between steps, multiple steering calls, and using `_recent_output()`.

## Pattern 1 — Stateful agent

```python
import asyncio
from dacs import DACSRuntime, BaseAgent, UrgencyLevel


class DataPipelineAgent(BaseAgent):
    """Multi-phase ETL agent with decisions at each phase boundary."""

    def __init__(self, source_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.source_path = source_path
        self.rows_loaded: int = 0
        self.schema_chosen: str = ""
        self.output_path: str = ""

    async def _execute(self) -> None:
        # --- Phase 1: Load ---
        self._push_update(f"Loading data from {self.source_path}")
        await asyncio.sleep(0.1)  # simulate I/O
        self.rows_loaded = 50_000
        self._push_update(f"Loaded {self.rows_loaded:,} rows.")

        # --- Phase 2: Schema decision ---
        resp = await self._request_steering(
            context=(
                f"Loaded {self.rows_loaded:,} rows. "
                "Detected columns: user_id, event_type, timestamp, metadata (JSON blob). "
                "Two schema options: wide (flatten metadata) or narrow (keep JSON blob)."
            ),
            question="Wide or narrow schema for the warehouse table?",
            urgency=UrgencyLevel.HIGH,
        )
        self.schema_chosen = resp.response_text
        self._push_update(f"Schema selected: {self.schema_chosen}.")

        # --- Phase 3: Transform ---
        self._push_update(f"Applying {self.schema_chosen} transform to {self.rows_loaded:,} rows.")
        await asyncio.sleep(0.2)  # simulate transform
        self._push_update("Transform complete. Running validation.")

        # --- Phase 4: Output format ---
        resp2 = await self._request_steering(
            context=self._recent_output(k=3),  # last 3 heartbeats as context
            question="Export as Parquet (smaller, faster queries) or Delta Lake (ACID)?",
        )
        self.output_path = f"/data/output.{resp2.response_text.lower().split()[0]}"
        self._push_update(f"Writing to {self.output_path}.")
        await asyncio.sleep(0.1)
        self._push_update(f"Done. Wrote {self.rows_loaded:,} rows to {self.output_path}.")


async def main() -> None:
    async with DACSRuntime(
        model="claude-3-haiku-20240307",
        log_path="pipeline.jsonl",
    ) as rt:
        rt.add_agent(
            DataPipelineAgent(
                source_path="s3://my-bucket/events/",
                agent_id="etl",
                task="Run ETL pipeline on event stream data",
            )
        )
        await rt.run()


if __name__ == "__main__":
    asyncio.run(main())
```

Key patterns shown:

- **`__init__`** stores agent-specific state; always call `super().__init__(**kwargs)`
- **`_recent_output(k=3)`** collects the last 3 heartbeats as context for the next steering question
- Multiple `_request_steering()` calls in sequence — each gets its own FOCUS session

---

## Pattern 2 — Conditional steering

Only ask for steering when the agent genuinely needs it:

```python
class SmartAgent(BaseAgent):
    async def _execute(self) -> None:
        self._push_update("Analysing input.")
        data_ambiguous = True  # in practice, check the actual data

        if data_ambiguous:
            resp = await self._request_steering(
                context="Data has unexpected null pattern in 'region' column.",
                question="Drop nulls, fill with 'UNKNOWN', or flag for review?",
                urgency=UrgencyLevel.MEDIUM,
            )
            strategy = resp.response_text
        else:
            strategy = "proceed normally"  # no steering needed

        self._push_update(f"Strategy: {strategy}.")
```

This is important — **don't request steering unnecessarily**. Each `_request_steering()` call costs a full LLM round-trip.

---

## Pattern 3 — HIGH-urgency interrupt

Use `urgency=UrgencyLevel.HIGH` for genuinely critical decisions that should interrupt other focus sessions:

```python
class MonitorAgent(BaseAgent):
    """Monitors a live data feed; raises HIGH urgency if anomaly detected."""

    async def _execute(self) -> None:
        for batch_num in range(10):
            self._push_update(f"Processing batch {batch_num}")
            await asyncio.sleep(0.05)

            if batch_num == 5:
                # Anomaly detected — interrupt whatever the orchestrator is doing
                resp = await self._request_steering(
                    context="Batch 5: latency spike detected — 3× normal P99.",
                    question="Halt pipeline and alert on-call, or continue and log?",
                    urgency=UrgencyLevel.HIGH,  # ← will interrupt any active FOCUS session
                    blocking=True,
                )
                self._push_update(f"Anomaly response: {resp.response_text}")

        self._push_update("All batches processed.")
```

!!! warning "Use HIGH urgency sparingly"
    HIGH-urgency requests interrupt another agent's focus session.  Reserve them
    for genuinely time-sensitive decisions.  Use `MEDIUM` for most steering calls.

---

## Pattern 4 — Non-blocking steering

For decisions where the agent can continue on a default path:

```python
class BackgroundAgent(BaseAgent):
    async def _execute(self) -> None:
        self._push_update("Starting background index build.")

        # Non-blocking — agent continues while waiting for response
        resp = await self._request_steering(
            context="Index build started.",
            question="Notify Slack when complete?",
            blocking=False,  # ← agent continues immediately... but actually still awaits
            urgency=UrgencyLevel.LOW,
        )
        notify = "yes" in resp.response_text.lower()
        self._push_update(f"Notify: {notify}.")
        # ... continue index build in real code ...
```

!!! info
    In the current version, `blocking=False` is stored in the `SteeringRequest` metadata
    but the `await` still suspends the coroutine.  Future versions will support
    truly non-blocking fire-and-continue semantics.

---

## Next

[Tutorial 4: Custom Endpoints →](04_custom_endpoints.md)
