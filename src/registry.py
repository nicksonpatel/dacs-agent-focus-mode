from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.protocols import AgentStatus, RegistryEntry, RegistryUpdate, UrgencyLevel

if TYPE_CHECKING:
    from src.context_builder import ContextBuilder
    from src.logger import Logger

_MAX_TASK_TOKENS    = 50
_MAX_SUMMARY_TOKENS = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RegistryManager:
    """Single source of truth for all agent state.

    Thread-safe via asyncio (single-threaded event loop — no explicit lock needed
    when all mutations happen in the same asyncio thread, which they do in this
    harness since agents and orchestrator share one event loop).
    """

    def __init__(self, logger: Logger) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._logger = logger
        self._cb: ContextBuilder | None = None

    def set_context_builder(self, cb: ContextBuilder) -> None:
        """Called after ContextBuilder is created to enable token counting."""
        self._cb = cb

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register(self, agent_id: str, task_description: str) -> None:
        """Called once when an agent is created. task_description is immutable after this."""
        if self._cb and self._cb.count_tokens(task_description) > _MAX_TASK_TOKENS:
            raise ValueError(
                f"task_description for {agent_id!r} is {self._cb.count_tokens(task_description)} tokens "
                f"(limit {_MAX_TASK_TOKENS})"
            )
        self._entries[agent_id] = RegistryEntry(
            agent_id=agent_id,
            task_description=task_description,
            status=AgentStatus.RUNNING,
            last_output_summary="",
            last_updated=_now(),
            pending_steering_request=False,
            urgency=UrgencyLevel.LOW,
        )

    def update(self, agent_id: str, update: RegistryUpdate) -> None:
        """Called by agent on every status change or step completion."""
        if agent_id not in self._entries:
            raise KeyError(f"Agent {agent_id!r} not registered")
        entry = self._entries[agent_id]
        summary = update.last_output_summary
        summary_tokens = 0
        if self._cb:
            toks = self._cb.count_tokens(summary)
            if toks > _MAX_SUMMARY_TOKENS:
                # Truncate to token limit — never silently accept over-budget entries
                encoded = self._cb._enc.encode(summary)
                summary = self._cb._enc.decode(encoded[:_MAX_SUMMARY_TOKENS])
                self._logger.log({
                    "event": "REGISTRY_TRUNCATION",
                    "agent_id": agent_id,
                    "original_tokens": toks,
                })
            summary_tokens = self._cb.count_tokens(summary)
        entry.status = update.status
        entry.last_output_summary = summary
        entry.urgency = update.urgency
        entry.last_updated = _now()
        self._logger.log({
            "event": "REGISTRY_UPDATE",
            "agent_id": agent_id,
            "status": update.status.value,
            "urgency": update.urgency.value,
            "summary_tokens": summary_tokens,
        })

    def get_all(self) -> list[RegistryEntry]:
        """Returns current snapshot of all entries. Used by ContextBuilder."""
        return list(self._entries.values())

    def get(self, agent_id: str) -> RegistryEntry:
        if agent_id not in self._entries:
            raise KeyError(f"Agent {agent_id!r} not registered")
        return self._entries[agent_id]

    def mark_steering_pending(self, agent_id: str) -> None:
        """Called when SteeringRequest is dequeued by Orchestrator."""
        e = self._entries[agent_id]
        e.pending_steering_request = True
        e.status = AgentStatus.WAITING_STEERING
        e.last_updated = _now()

    def mark_steering_complete(self, agent_id: str) -> None:
        """Called after Orchestrator delivers SteeringResponse."""
        e = self._entries[agent_id]
        e.pending_steering_request = False
        e.status = AgentStatus.RUNNING
        e.last_updated = _now()
