from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.logger import Logger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentStatus(Enum):
    RUNNING          = "RUNNING"
    BLOCKED          = "BLOCKED"
    WAITING_STEERING = "WAITING_STEERING"
    COMPLETE         = "COMPLETE"
    FAILED           = "FAILED"


class UrgencyLevel(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


# ---------------------------------------------------------------------------
# Registry schemas
# ---------------------------------------------------------------------------

@dataclass
class RegistryEntry:
    agent_id: str
    task_description: str           # ≤50 tokens enforced at write time
    status: AgentStatus
    last_output_summary: str        # ≤100 tokens enforced at write time
    last_updated: str               # ISO 8601
    pending_steering_request: bool
    urgency: UrgencyLevel


@dataclass
class RegistryUpdate:
    agent_id: str
    status: AgentStatus
    last_output_summary: str
    urgency: UrgencyLevel


# ---------------------------------------------------------------------------
# Steering message schemas
# ---------------------------------------------------------------------------

@dataclass
class SteeringRequest:
    agent_id: str
    relevant_context: str   # assembled by the agent — recent output / decision context
    question: str           # ≤100 tokens; the specific decision needed
    blocking: bool          # True = agent halted waiting for response
    urgency: UrgencyLevel
    timestamp: str = field(default_factory=_now)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SteeringResponse:
    request_id: str              # matches originating SteeringRequest.request_id
    agent_id: str
    response_text: str
    context_size_at_time: int    # token count of context window when decision was made — CRITICAL
    orchestrator_state: str      # "FOCUS" or "FLAT"
    timestamp: str = field(default_factory=_now)


@dataclass
class FocusContext:
    agent_id: str
    task_description: str
    steering_history: list[dict]    # [{"request": {...}, "response": {...}}], capped at K=10
    recent_output: str              # from SteeringRequest.relevant_context
    current_request: SteeringRequest


# ---------------------------------------------------------------------------
# Steering request queue
# ---------------------------------------------------------------------------

class SteeringRequestQueue:
    def __init__(self, logger: Logger) -> None:
        self._queue: deque[SteeringRequest] = deque()
        self._logger = logger

    def enqueue(self, request: SteeringRequest) -> None:
        # HIGH urgency → front of queue (can interrupt active FOCUS session)
        if request.urgency == UrgencyLevel.HIGH:
            self._queue.appendleft(request)
        else:
            self._queue.append(request)
        self._logger.log({
            "event": "STEERING_REQUEST",
            "request_id": request.request_id,
            "agent_id": request.agent_id,
            "urgency": request.urgency.value,
            "blocking": request.blocking,
        })

    def peek(self) -> Optional[SteeringRequest]:
        return self._queue[0] if self._queue else None

    def dequeue(self) -> Optional[SteeringRequest]:
        return self._queue.popleft() if self._queue else None

    def has_high_urgency(self) -> bool:
        return bool(self._queue) and self._queue[0].urgency == UrgencyLevel.HIGH

    def size(self) -> int:
        return len(self._queue)
