# Logger

Structured JSONL event log with pluggable sinks.

::: dacs._logger.Logger
    options:
      show_source: true
      members:
        - __init__
        - log
        - add_sink
        - close

## Event schema

Every event is a dict with at least:

| Field | Type | Description |
|---|---|---|
| `event` | `str` | Event type (see table below) |
| `ts` | `str` | ISO 8601 UTC timestamp |
| `run_id` | `str` | Unique run identifier |
| ...other | ... | Event-specific fields |

## Event reference

| Event | Key fields |
|---|---|
| `RUN_START` | `model`, `focus_mode`, `n_agents`, `token_budget` |
| `RUN_END` | — |
| `REGISTRY_UPDATE` | `agent_id`, `status`, `urgency`, `summary_tokens` |
| `REGISTRY_TRUNCATION` | `agent_id`, `original_tokens` |
| `CONTEXT_BUILT` | `mode` (`FOCUS`/`REGISTRY`/`FLAT`), `token_count`, `agent_id` |
| `STEERING_REQUEST` | `agent_id`, `request_id`, `urgency`, `blocking` |
| `STEERING_RESPONSE` | `agent_id`, `request_id`, `context_size_at_time`, `orchestrator_state` |
| `LLM_CALL` | `in_tokens`, `out_tokens`, `latency_ms`, `state` |
| `TRANSITION` | `from_state`, `to_state`, `reason` |
| `INTERRUPT` | `interrupted_agent`, `interrupting_agent`, `urgency` |
| `FOCUS_TIMEOUT` | `agent_id`, `elapsed_s`, `turns` |

## Custom sink example

```python
from dacs import Logger
import sys

logger = Logger("run.jsonl")

# Add a custom sink that prints steering events to stderr
def steering_sink(event: dict) -> None:
    if event["event"] in ("STEERING_REQUEST", "STEERING_RESPONSE"):
        print(f"[STEER] {event}", file=sys.stderr)

logger.add_sink(steering_sink)
```

## Disable file output

```python
# Sinks still work; no file is written
logger = Logger(None)
logger.add_sink(my_sink)
```
