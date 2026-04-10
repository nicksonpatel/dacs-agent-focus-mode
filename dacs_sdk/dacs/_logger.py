"""Structured JSONL event logger with pluggable sinks."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Logger:
    """Append-only JSONL logger.

    Every event is written as a single JSON line. Additional sinks (e.g.
    :class:`~dacs.TerminalMonitor`) can be registered to receive events
    in real time.

    Parameters
    ----------
    path:
        File path to write events to. The parent directory is created
        automatically if it does not exist. Pass ``None`` to disable
        file output (events are still forwarded to sinks).
    """

    def __init__(self, path: str | None = "dacs_run.jsonl") -> None:
        self._sinks: list[Callable[[dict], None]] = []
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115
        else:
            self._f = None

    def add_sink(self, fn: Callable[[dict], None]) -> None:
        """Register a callable that receives every event dict after it is written.

        Example — attach the terminal monitor::

            monitor = TerminalMonitor()
            logger.add_sink(monitor.handle)
        """
        self._sinks.append(fn)

    def log(self, event: dict) -> None:
        """Write *event* to the log file and forward to all sinks."""
        if "ts" not in event:
            event["ts"] = now_iso()
        if self._f is not None:
            self._f.write(json.dumps(event) + "\n")
            self._f.flush()
        for sink in self._sinks:
            sink(event)

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if self._f is not None:
            self._f.close()
