"""Tests for enums, dataclasses, and SteeringRequestQueue."""

from __future__ import annotations

import pytest

from dacs._protocols import (
    AgentStatus,
    FocusContext,
    RegistryEntry,
    RegistryUpdate,
    SteeringRequest,
    SteeringRequestQueue,
    SteeringResponse,
    UrgencyLevel,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_agent_status_values() -> None:
    assert AgentStatus.RUNNING.value == "RUNNING"
    assert AgentStatus.COMPLETE.value == "COMPLETE"
    assert len(AgentStatus) == 5


def test_urgency_level_ordering() -> None:
    assert UrgencyLevel.HIGH.value == "HIGH"
    assert UrgencyLevel.LOW.value == "LOW"


# ---------------------------------------------------------------------------
# SteeringRequestQueue
# ---------------------------------------------------------------------------


def _make_request(agent_id: str, urgency: UrgencyLevel) -> SteeringRequest:
    return SteeringRequest(
        agent_id=agent_id,
        relevant_context="ctx",
        question="q?",
        blocking=True,
        urgency=urgency,
    )


class _NullLogger:
    """Minimal logger stub."""

    def log(self, _event: dict) -> None:
        pass


def test_queue_fifo_for_same_urgency() -> None:
    q = SteeringRequestQueue(_NullLogger())
    r1 = _make_request("a1", UrgencyLevel.MEDIUM)
    r2 = _make_request("a2", UrgencyLevel.MEDIUM)
    q.enqueue(r1)
    q.enqueue(r2)
    assert q.dequeue().agent_id == "a1"
    assert q.dequeue().agent_id == "a2"


def test_high_urgency_jumps_queue() -> None:
    q = SteeringRequestQueue(_NullLogger())
    r_low = _make_request("low", UrgencyLevel.LOW)
    r_med = _make_request("med", UrgencyLevel.MEDIUM)
    r_high = _make_request("high", UrgencyLevel.HIGH)
    q.enqueue(r_low)
    q.enqueue(r_med)
    q.enqueue(r_high)
    # HIGH should come first
    assert q.dequeue().agent_id == "high"
    assert q.dequeue().agent_id == "low"
    assert q.dequeue().agent_id == "med"


def test_peek_does_not_remove() -> None:
    q = SteeringRequestQueue(_NullLogger())
    r = _make_request("a1", UrgencyLevel.LOW)
    q.enqueue(r)
    peeked = q.peek()
    assert peeked is not None
    assert peeked.agent_id == "a1"
    assert len(q) == 1


def test_peek_empty_returns_none() -> None:
    q = SteeringRequestQueue(_NullLogger())
    assert q.peek() is None


def test_has_high_urgency() -> None:
    q = SteeringRequestQueue(_NullLogger())
    assert not q.has_high_urgency()
    q.enqueue(_make_request("a1", UrgencyLevel.MEDIUM))
    assert not q.has_high_urgency()
    q.enqueue(_make_request("a2", UrgencyLevel.HIGH))
    assert q.has_high_urgency()


def test_dequeue_empty_raises() -> None:
    q = SteeringRequestQueue(_NullLogger())
    with pytest.raises(IndexError):
        q.dequeue()


def test_request_id_unique() -> None:
    r1 = _make_request("a1", UrgencyLevel.LOW)
    r2 = _make_request("a1", UrgencyLevel.LOW)
    assert r1.request_id != r2.request_id


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


def test_registry_entry_fields() -> None:
    entry = RegistryEntry(
        agent_id="a1",
        task_description="Do something",
        status=AgentStatus.RUNNING,
        last_output_summary="step 1 done",
        last_updated="2024-01-01T00:00:00Z",
        pending_steering_request=False,
        urgency=UrgencyLevel.LOW,
    )
    assert entry.agent_id == "a1"
    assert entry.status == AgentStatus.RUNNING


def test_focus_context_fields() -> None:
    req = _make_request("a1", UrgencyLevel.MEDIUM)
    fc = FocusContext(
        agent_id="a1",
        task_description="Task",
        steering_history=[],
        recent_output="output",
        current_request=req,
    )
    assert fc.agent_id == "a1"
    assert fc.current_request.agent_id == "a1"
