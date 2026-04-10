"""Tests for RegistryManager."""

from __future__ import annotations

import pytest

from dacs._logger import Logger
from dacs._protocols import AgentStatus, RegistryUpdate, UrgencyLevel
from dacs._registry import RegistryManager


def _make_registry() -> RegistryManager:
    return RegistryManager(Logger(None))


def test_register_and_get() -> None:
    reg = _make_registry()
    reg.register("a1", "Do something useful")
    entry = reg.get("a1")
    assert entry.agent_id == "a1"
    assert entry.task_description == "Do something useful"
    assert entry.status == AgentStatus.RUNNING


def test_get_all() -> None:
    reg = _make_registry()
    reg.register("a1", "Task one")
    reg.register("a2", "Task two")
    entries = reg.get_all()
    ids = {e.agent_id for e in entries}
    assert ids == {"a1", "a2"}


def test_update_status() -> None:
    reg = _make_registry()
    reg.register("a1", "Task")
    reg.update(
        "a1",
        RegistryUpdate(
            agent_id="a1",
            status=AgentStatus.COMPLETE,
            last_output_summary="done",
            urgency=UrgencyLevel.LOW,
        ),
    )
    assert reg.get("a1").status == AgentStatus.COMPLETE
    assert reg.get("a1").last_output_summary == "done"


def test_update_unknown_agent_raises() -> None:
    reg = _make_registry()
    with pytest.raises(KeyError):
        reg.update(
            "nonexistent",
            RegistryUpdate(
                agent_id="nonexistent",
                status=AgentStatus.RUNNING,
                last_output_summary="",
                urgency=UrgencyLevel.LOW,
            ),
        )


def test_get_unknown_agent_raises() -> None:
    reg = _make_registry()
    with pytest.raises(KeyError):
        reg.get("ghost")


def test_mark_steering_pending_and_complete() -> None:
    reg = _make_registry()
    reg.register("a1", "Task")
    reg.mark_steering_pending("a1")
    assert reg.get("a1").pending_steering_request is True
    assert reg.get("a1").status == AgentStatus.WAITING_STEERING
    reg.mark_steering_complete("a1")
    assert reg.get("a1").pending_steering_request is False
    assert reg.get("a1").status == AgentStatus.RUNNING


def test_duplicate_register_overwrites() -> None:
    reg = _make_registry()
    reg.register("a1", "First task")
    reg.register("a1", "Second task")
    assert reg.get("a1").task_description == "Second task"
