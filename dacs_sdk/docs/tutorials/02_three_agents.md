# Tutorial 2: Three Concurrent Agents

This tutorial shows the core DACS value proposition: **multiple agents running concurrently, each getting isolated steering attention**.

## What we'll build

Three agents running simultaneously:

| Agent | Task | Decisions |
|---|---|---|
| `coder` | Implement a REST API | Auth strategy, error handling style |
| `reviewer` | Review a PR | Refactoring approach |
| `doc_writer` | Write API docs | Format choice |

Without DACS (flat baseline), the orchestrator sees all three agents' contexts at every steering call. With DACS, each agent gets a private FOCUS session.

## Step 1 — Define the agents

```python
# three_agents.py
import asyncio
from dacs import DACSRuntime, BaseAgent, UrgencyLevel


class CoderAgent(BaseAgent):
    async def _execute(self) -> None:
        self._push_update("Scaffolded project with FastAPI.")

        resp = await self._request_steering(
            context=(
                "Building a public REST API. "
                "Options: JWT tokens (stateless) or server-side sessions (stateful)."
            ),
            question="Which authentication strategy for a public API?",
            urgency=UrgencyLevel.HIGH,
        )
        auth = resp.response_text
        self._push_update(f"Implementing {auth} auth.")

        resp2 = await self._request_steering(
            context=f"Auth ({auth}) implemented. Writing error handlers.",
            question="Use RFC 7807 problem+json format or plain JSON errors?",
        )
        self._push_update(f"Error format chosen: {resp2.response_text}.")
        self._push_update("Implementation complete.")


class ReviewerAgent(BaseAgent):
    async def _execute(self) -> None:
        self._push_update("Reviewing PR #42: DataProcessor class.")
        await asyncio.sleep(0.3)  # simulate async reading

        resp = await self._request_steering(
            context=(
                "DataProcessor is 280 lines with three distinct concerns: "
                "loading, transformation, and serialisation."
            ),
            question=(
                "Split into three classes or use a single class with composition?"
            ),
        )
        self._push_update(f"Recommended: {resp.response_text}.")
        self._push_update("Review comment posted.")


class DocWriterAgent(BaseAgent):
    async def _execute(self) -> None:
        self._push_update("Generating API reference docs.")
        await asyncio.sleep(0.1)

        resp = await self._request_steering(
            context="Need to pick a documentation format for the REST API.",
            question="Generate OpenAPI 3.1 YAML or Markdown with code examples?",
            urgency=UrgencyLevel.LOW,
        )
        self._push_update(f"Using format: {resp.response_text}.")
        self._push_update("Docs generation complete.")
```

## Step 2 — Run all three concurrently

```python
async def main() -> None:
    async with DACSRuntime(
        model="claude-3-haiku-20240307",
        verbose=True,
        log_path="three_agents.jsonl",
        focus_timeout=30,
    ) as runtime:
        runtime.add_agent(CoderAgent(agent_id="coder", task="Implement REST API with auth"))
        runtime.add_agent(ReviewerAgent(agent_id="reviewer", task="Review PR #42"))
        runtime.add_agent(DocWriterAgent(agent_id="doc", task="Generate API documentation"))
        await runtime.run()


if __name__ == "__main__":
    asyncio.run(main())
```

## Step 3 — Run and observe

```bash
python three_agents.py
```

Watch the terminal output. You'll see:

- All three agents push registry updates as they work
- Each `STEERING_REQUEST` triggers a `TRANSITION: REGISTRY → FOCUS`
- The `CONTEXT_BUILT` event shows the token count for each focus context
- After the response, `TRANSITION: FOCUS → REGISTRY`

Notice how the focus context is **much smaller** than a flat context would be (all three agents combined).

## Step 4 — Compare with the baseline

Change one line:

```python
async with DACSRuntime(
    model="claude-3-haiku-20240307",
    focus_mode=False,   # ← flat-context baseline
    log_path="three_agents_baseline.jsonl",
) as runtime:
```

The `CONTEXT_BUILT` events will now show `mode=FLAT` and a significantly larger token count.

## Understanding the sequence

```
t=0.0s   All agents start concurrently
t=0.0s   coder → STEERING_REQUEST (HIGH, auth question)
          orchestrator: REGISTRY → FOCUS(coder)    ← isolated LLM call
          orchestrator: FOCUS(coder) → REGISTRY
          coder receives response, continues

t=0.1s   doc → STEERING_REQUEST (LOW, format question)
t=0.3s   reviewer → STEERING_REQUEST (MEDIUM, refactoring)
          (queue is processed in urgency order)

t=0.5s   coder → STEERING_REQUEST (MEDIUM, error format)
...
```

## Reading the log

```bash
cat three_agents.jsonl | python -m json.tool | grep '"event":'
```

Key events to look for:

- `"CONTEXT_BUILT"` with `"mode": "FOCUS"` — the isolated context size
- `"INTERRUPT"` — HIGH-urgency preemption of another focus session
- `"REGISTRY_UPDATE"` — agent heartbeats

## Next

[Tutorial 3: Custom Agents →](03_custom_agents.md)
