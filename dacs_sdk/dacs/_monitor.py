"""Rich terminal monitor — plugs into Logger sinks for live event display."""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

_EVENT_STYLES: dict[str, str] = {
    "RUN_START": "bold green",
    "RUN_END": "bold red",
    "STEERING_REQUEST": "yellow",
    "STEERING_RESPONSE": "green",
    "REGISTRY_UPDATE": "cyan",
    "REGISTRY_TRUNCATION": "bold yellow",
    "CONTEXT_BUILT": "blue",
    "LLM_CALL": "magenta",
    "TRANSITION": "bold cyan",
    "INTERRUPT": "bold red",
    "FOCUS_TIMEOUT": "red",
    "USER_REQUEST": "bold white",
    "USER_RESPONSE": "bold white",
}


class TerminalMonitor:
    """Live Rich terminal monitor for DACS events.

    Attach to a :class:`~dacs.Logger` instance via ``logger.add_sink``::

        monitor = TerminalMonitor()
        logger.add_sink(monitor.handle)

    Or pass ``verbose=True`` to :class:`~dacs.DACSRuntime` to attach
    automatically::

        async with DACSRuntime(verbose=True) as runtime:
            ...

    Parameters
    ----------
    token_budget:
        Total token budget used to display a relative context bar.
    width:
        Console width (characters).  ``None`` = auto-detect.
    """

    def __init__(
        self,
        token_budget: int = 200_000,
        width: int | None = None,
    ) -> None:
        if not _RICH_AVAILABLE:
            raise ImportError(
                "TerminalMonitor requires 'rich'. "
                "Install with: pip install dacs-agent[monitor]"
            )
        self._budget = token_budget
        self._console = Console(width=width)
        self._event_count = 0

    def handle(self, event: dict[str, Any]) -> None:
        """Sink callback — call this from ``Logger.add_sink``."""
        etype = event.get("event", "UNKNOWN")
        style = _EVENT_STYLES.get(etype, "white")
        ts = str(event.get("ts", ""))[:19]

        parts: list[str] = [f"[dim]{ts}[/dim]"]
        parts.append(f"[{style}]{etype:<24}[/{style}]")

        # Event-specific details
        if etype == "STEERING_REQUEST":
            parts.append(f"agent=[bold]{event.get('agent_id', '?')}[/bold]")
            parts.append(f"urgency={event.get('urgency', '?')}")
        elif etype == "STEERING_RESPONSE":
            parts.append(f"agent=[bold]{event.get('agent_id', '?')}[/bold]")
            ctx = event.get("context_size_at_time", 0)
            bar = self._token_bar(ctx)
            parts.append(f"ctx={ctx:,}tok {bar}")
        elif etype == "CONTEXT_BUILT":
            mode = event.get("mode", "?")
            ctx = event.get("token_count", 0)
            bar = self._token_bar(ctx)
            parts.append(f"mode={mode} {ctx:,}tok {bar}")
        elif etype == "LLM_CALL":
            parts.append(
                f"in={event.get('in_tokens', 0):,} "
                f"out={event.get('out_tokens', 0):,} "
                f"lat={event.get('latency_ms', 0):.0f}ms"
            )
        elif etype == "TRANSITION":
            parts.append(
                f"{event.get('from_state', '?')} → {event.get('to_state', '?')}"
            )
            reason = event.get("reason", "")
            if reason:
                parts.append(f"[dim]({reason})[/dim]")
        elif etype == "INTERRUPT":
            parts.append(
                f"[bold red]preempt {event.get('preempted_agent_id', '?')} "
                f"→ {event.get('new_agent_id', '?')}[/bold red]"
            )
        elif etype == "REGISTRY_UPDATE":
            parts.append(f"agent=[bold]{event.get('agent_id', '?')}[/bold]")
            parts.append(f"status={event.get('status', '?')}")
        elif etype in ("RUN_START", "RUN_END"):
            run_id = str(event.get("run_id", ""))[:8]
            parts.append(f"run={run_id}")
            if etype == "RUN_START":
                parts.append(f"model={event.get('model', '?')}")
                parts.append(
                    f"focus={'on' if event.get('focus_mode') else 'off'}"
                )

        self._console.print(" ".join(parts))
        self._event_count += 1

    def _token_bar(self, tokens: int, width: int = 12) -> str:
        """Return a small ASCII progress bar relative to the token budget."""
        frac = min(tokens / self._budget, 1.0)
        filled = int(frac * width)
        empty = width - filled
        colour = "green" if frac < 0.5 else "yellow" if frac < 0.8 else "red"
        bar = "█" * filled + "░" * empty
        return f"[{colour}][{bar}][/{colour}]"

    def print_summary(self) -> None:
        """Print a final summary table of observed events."""
        table = Table(title="DACS Run Summary", show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Total events", str(self._event_count))
        self._console.print(table)
