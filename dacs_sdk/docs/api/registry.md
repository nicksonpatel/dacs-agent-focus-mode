# RegistryManager

Per-agent state store — the DACS registry *R*.

::: dacs._registry.RegistryManager
    options:
      show_source: true
      members:
        - __init__
        - set_context_builder
        - register
        - update
        - get
        - get_all
        - mark_steering_pending
        - mark_steering_complete

## RegistryEntry

::: dacs._protocols.RegistryEntry

## RegistryUpdate

::: dacs._protocols.RegistryUpdate

## Token limits

| Field | Limit |
|---|---|
| `task_description` | 50 tokens |
| `last_output_summary` | 100 tokens |
| Total per-agent entry | ≤ 200 tokens |

Summaries exceeding 100 tokens are automatically truncated and a `REGISTRY_TRUNCATION` event is logged.

## Example

```python
from dacs import Logger, RegistryManager, ContextBuilder, RegistryUpdate, AgentStatus, UrgencyLevel

logger = Logger(None)
cb = ContextBuilder(200_000, logger)
registry = RegistryManager(logger)
registry.set_context_builder(cb)

registry.register("a1", "Summarise the quarterly earnings report")
registry.update("a1", RegistryUpdate(
    agent_id="a1",
    status=AgentStatus.RUNNING,
    last_output_summary="Loaded PDF: 42 pages",
    urgency=UrgencyLevel.LOW,
))

entry = registry.get("a1")
print(entry.status)   # AgentStatus.RUNNING
```
