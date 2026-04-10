"""Dataclasses and enums forming the DACS wire protocol."""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dacs._logger import Logger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentStatus(Enum):
    """Lifecycle state of an agent, tracked in the registry."""

    RUNNING = "RUNNING"
    """Agent is actively processing its task."""

    BLOCKED = "BLOCKED"
    """Agent is blocked on an external dependency (not steering)."""

    WAITING_STEERING = "WAITING_STEERING"
    """Agent has emitted a SteeringRequest and is waiting for a response."""

    COMPLETE = "COMPLETE"
    """Agent has finished its task successfully."""

    FAILED = "FAILED"
    """Agent encountered an unrecoverable error."""


class UrgencyLevel(Enum):
    """Priority level attached to a SteeringRequest.

    - ``HIGH``   — can interrupt an active FOCUS session (preemption protocol).
    - ``MEDIUM`` — queued; agent continues on a default path while waiting.
    - ``LOW``    — queued; processed whenever the orchestrator is idle.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Registry schemas
# ---------------------------------------------------------------------------


@dataclass
class RegistryEntry:
    """Compact per-agent status snapshot stored in the registry (≤200 tokens)."""

    agent_id: str
    task_description: str
    """Short task description. Enforced ≤50 tokens at write time."""
    status: AgentStatus
    last_output_summary: str
    """Latest output summary. Enforced ≤100 tokens at write time."""
    last_updated: str
    """ISO 8601 timestamp of the last update."""
    pending_steering_request: bool
    urgency: UrgencyLevel


@dataclass
class RegistryUpdate:
    """Payload agents push to update their registry entry."""

    agent_id: str
    status: AgentStatus
    last_output_summary: str
    urgency: UrgencyLevel


# ---------------------------------------------------------------------------
# Steering message schemas
# ---------------------------------------------------------------------------


@dataclass
class SteeringRequest:
    """Message emitted by an agent when it needs orchestrator guidance."""

    agent_id: str
    relevant_context: str
    """Recent agent output or decision context assembled by the agent."""
    question: str
    """The specific decision or clarification needed (≤100 tokens recommended)."""
    blocking: bool
    """If True the agent is halted until a response is delivered."""
    urgency: UrgencyLevel
    timestamp: str = field(default_factory=_now)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SteeringResponse:
    """Orchestrator's reply to a SteeringRequest."""

    request_id: str
    """Matches the originating SteeringRequest.request_id."""
    agent_id: str
    response_text: str
    context_size_at_time: int
    """Token count of the orchestrator's context window when the decision was made."""
    orchestrator_state: str
    """'FOCUS' (DACS mode) or 'FLAT' (baseline mode)."""
    timestamp: str = field(default_factory=_now)


@dataclass
class FocusContext:
    """Complete context bundle injected during a FOCUS(aᵢ) session."""

    agent_id: str
    task_description: str
    steering_history: list[dict]
    """Previous steering exchanges: [{"request": {...}, "response": {...}}]."""
    recent_output: str
    """From SteeringRequest.relevant_context."""
    current_request: SteeringRequest


# ---------------------------------------------------------------------------
# Priority queue
# ---------------------------------------------------------------------------


class SteeringRequestQueue:
    """Priority queue for SteeringRequests.

    HIGH urgency requests are placed at the front of the queue, enabling
    preemption of an active FOCUS session.
    """

    def __init__(self, logger: "Logger") -> None:
        self._queue: deque[SteeringRequest] = deque()
        self._logger = logger

    def enqueue(self, request: SteeringRequest) -> None:
        """Add a request to the queue. HIGH urgency requests go to the front."""
        if request.urgency == UrgencyLevel.HIGH:
            self._queue.appendleft(request)
        else:
            self._queue.append(request)
        self._logger.log(
            {
                "event": "STEERING_REQUEST",
                "request_id": request.request_id,
                "agent_id": request.agent_id,
                "urgency": request.urgency.value,
                "blocking": request.blocking,
                "question": request.question,
            }
        )

    def peek(self) -> Optional[SteeringRequest]:
        """Return the next request without removing it."""
        return self._queue[0] if self._queue else None

    def dequeue(self) -> SteeringRequest:
        """Remove and return the next request.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        if not self._queue:
            raise IndexError("dequeue from an empty SteeringRequestQueue")
        return self._queue.popleft()

    def has_high_urgency(self) -> bool:
        """True if the front of the queue is a HIGH urgency request."""
        return bool(self._queue) and self._queue[0].urgency == UrgencyLevel.HIGH

    def __len__(self) -> int:
        """Return the number of pending requests."""
        return len(self._queue)
