import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Logger:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._sinks: list[Callable[[dict], None]] = []

    def add_sink(self, fn: Callable[[dict], None]) -> None:
        """Register a callable that receives every event dict after it is written."""
        self._sinks.append(fn)

    def log(self, event: dict) -> None:
        if "ts" not in event:
            event["ts"] = now_iso()
        self._f.write(json.dumps(event) + "\n")
        self._f.flush()
        for sink in self._sinks:
            sink(event)

    def close(self) -> None:
        self._f.close()
