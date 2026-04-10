# TerminalMonitor

Live Rich terminal monitor — displays a colour-coded event feed during a DACS run.

!!! note "Requires the `monitor` extra"
    ```bash
    pip install "dacs-agent[monitor]"
    ```

::: dacs._monitor.TerminalMonitor
    options:
      show_source: true
      members:
        - __init__
        - handle
        - print_summary

## Quick attach

The easiest way to enable the monitor is via `DACSRuntime`:

```python
async with DACSRuntime(
    model="claude-3-haiku-20240307",
    verbose=True,   # ← attaches TerminalMonitor automatically
) as runtime:
    ...
```

## Manual attach

For custom sink configurations:

```python
from dacs import Logger
from dacs._monitor import TerminalMonitor

logger = Logger("run.jsonl")
monitor = TerminalMonitor(token_budget=200_000, width=120)
logger.add_sink(monitor.handle)
```

## Sample output

```
2024-01-01 00:00:01  RUN_START               run=a1b2c3d4 model=claude-3-haiku-20240307 focus=on
2024-01-01 00:00:01  REGISTRY_UPDATE         agent=coder status=RUNNING
2024-01-01 00:00:01  STEERING_REQUEST        agent=coder urgency=HIGH
2024-01-01 00:00:01  CONTEXT_BUILT           mode=FOCUS 1 842tok [████░░░░░░░░]
2024-01-01 00:00:02  LLM_CALL                in=1 842 out=87 lat=830ms
2024-01-01 00:00:02  STEERING_RESPONSE       agent=coder ctx=1 842tok [████░░░░░░░░]
2024-01-01 00:00:02  TRANSITION              FOCUS → REGISTRY (SteeringComplete)
2024-01-01 00:00:02  REGISTRY_UPDATE         agent=reviewer status=RUNNING
```

The token bar `[████░░░░░░░░]` represents usage relative to the `token_budget` (green < 50 %, yellow < 80 %, red ≥ 80 %).
