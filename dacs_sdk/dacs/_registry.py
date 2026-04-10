"""Per-agent state store — the DACS registry R."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dacs._protocols import AgentStatus, RegistryEntry, RegistryUpdate, UrgencyLevel

if TYPE_CHECKING:
    from dacs._context_builder import ContextBuilder
    from dacs._logger import Logger

_MAX_TASK_TOKENS = 50
_MAX_SUMMARY_TOKENS = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RegistryManager:
    """Single source of truth for all agent state.

    Each agent has a compact :class:`~dacs.RegistryEntry` (≤200 tokens).
    The registry is updated after every agent step (heartbeat) and is the
    data source for building the orchestrator's context in REGISTRY mode.

    Thread safety is guaranteed by the single-threaded asyncio event loop
    that all agents and the orchestrator share.
    """

    def __init__(self, logger: "Logger") -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._logger = logger
        self._cb: ContextBuilder | None = None

    def set_context_builder(self, cb: "ContextBuilder") -> None:
        """Wire in a ContextBuilder to enable token-count enforcement."""
        self._cb = cb

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register(self, agent_id: str, task_description: str) -> None:
        """Register a new agent.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent.
        task_description:
            One-sentence description of the agent's task (≤50 tokens).

        Raises
        ------
        ValueError
            If *task_description* exceeds the 50-token limit.
        """
        if self._cb and self._cb.count_tokens(task_description) > _MAX_TASK_TOKENS:
            tok = self._cb.count_tokens(task_description)
            raise ValueError(
                f"task_description for {agent_id!r} is {tok} tokens "
                f"(limit {_MAX_TASK_TOKENS}). Please shorten it."
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
        """Update an agent's registry entry.

        The *last_output_summary* field is automatically truncated to
        100 tokens if it is too long, and the truncation is logged.
        """
        if agent_id not in self._entries:
            raise KeyError(f"Agent {agent_id!r} not registered")
        entry = self._entries[agent_id]
        summary = update.last_output_summary
        summary_tokens = 0
        if self._cb:
            toks = self._cb.count_tokens(summary)
            if toks > _MAX_SUMMARY_TOKENS:
                encoded = self._cb._enc.encode(summary)
                summary = self._cb._enc.decode(encoded[:_MAX_SUMMARY_TOKENS])
                self._logger.log(
                    {
                        "event": "REGISTRY_TRUNCATION",
                        "agent_id": agent_id,
                        "original_tokens": toks,
                    }
                )
            summary_tokens = self._cb.count_tokens(summary)
        entry.status = update.status
        entry.last_output_summary = summary
        entry.urgency = update.urgency
        entry.last_updated = _now()
        self._logger.log(
            {
                "event": "REGISTRY_UPDATE",
                "agent_id": agent_id,
                "status": update.status.value,
                "urgency": update.urgency.value,
                "summary_tokens": summary_tokens,
            }
        )

    def get_all(self) -> list[RegistryEntry]:
        """Return a snapshot of all registry entries."""
        return list(self._entries.values())

    def get(self, agent_id: str) -> RegistryEntry:
        """Return the registry entry for *agent_id*.

        Raises
        ------
        KeyError
            If the agent is not registered.
        """
        if agent_id not in self._entries:
            raise KeyError(f"Agent {agent_id!r} not registered")
        return self._entries[agent_id]

    # ------------------------------------------------------------------
    # Internal state transitions (called by Orchestrator)
    # ------------------------------------------------------------------

    def mark_steering_pending(self, agent_id: str) -> None:
        e = self._entries[agent_id]
        e.pending_steering_request = True
        e.status = AgentStatus.WAITING_STEERING
        e.last_updated = _now()

    def mark_steering_complete(self, agent_id: str) -> None:
        e = self._entries[agent_id]
        e.pending_steering_request = False
        e.status = AgentStatus.RUNNING
        e.last_updated = _now()
