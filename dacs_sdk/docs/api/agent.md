# BaseAgent & StepAgent

## BaseAgent

Abstract base class for all DACS agents.  Subclass this and implement `_execute()`.

::: dacs._agent.BaseAgent
    options:
      show_source: true
      members:
        - __init__
        - run
        - _execute
        - _request_steering
        - _push_update
        - _recent_output
        - deliver_response

### Minimal subclass

```python
from dacs import BaseAgent, UrgencyLevel

class MyAgent(BaseAgent):
    async def _execute(self) -> None:
        self._push_update("Starting work")

        resp = await self._request_steering(
            context="Context relevant to the decision",
            question="What approach should I take?",
            urgency=UrgencyLevel.MEDIUM,
        )

        self._push_update(f"Proceeding with: {resp.response_text}")
```

### `_push_update` signature

```python
def _push_update(
    self,
    summary: str,                         # required: what happened
    status: AgentStatus = AgentStatus.RUNNING,
    urgency: UrgencyLevel = UrgencyLevel.LOW,
) -> None
```

### `_request_steering` signature

```python
async def _request_steering(
    self,
    context: str,                          # required: text for orchestrator
    question: str,                         # required: the decision needed
    blocking: bool = True,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
) -> SteeringResponse
```

---

## StepAgent

Ready-to-use step-driven agent.  No subclassing needed.

::: dacs._step_agent.StepAgent
    options:
      show_source: true
      members:
        - __init__
        - _execute

### Step dict schema

Each element of the `steps` list is a dict with these keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `summary` | `str` | ✅ | Heartbeat pushed to the registry |
| `question` | `str` | — | If present, emits a SteeringRequest |
| `urgency` | `"LOW" \| "MEDIUM" \| "HIGH"` | — | Default `"MEDIUM"` |
| `sleep` | `float` | — | Seconds to sleep (simulate work) |

### Example

```python
from dacs import StepAgent

agent = StepAgent(
    agent_id="analyst",
    task="Analyse Q3 customer data",
    steps=[
        {"summary": "Loading dataset", "sleep": 0.5},
        {
            "summary": "Choosing analysis method",
            "question": "Use cohort analysis or funnel analysis?",
            "urgency": "HIGH",
        },
        {"summary": "Running analysis"},
        {"summary": "Generating report"},
    ],
)
```
