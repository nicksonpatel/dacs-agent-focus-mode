"""Tests for ContextBuilder."""

from __future__ import annotations

import pytest

from dacs._context_builder import ContextBudgetError, ContextBuilder
from dacs._logger import Logger
from dacs._protocols import (
    FocusContext,
    RegistryEntry,
    SteeringRequest,
    UrgencyLevel,
    AgentStatus,
)
from dacs._registry import RegistryManager


def _builder(budget: int = 200_000) -> ContextBuilder:
    return ContextBuilder(token_budget=budget, logger=Logger(None))


def _entry(agent_id: str, summary: str = "working") -> RegistryEntry:
    return RegistryEntry(
        agent_id=agent_id,
        task_description=f"Task for {agent_id}",
        status=AgentStatus.RUNNING,
        last_output_summary=summary,
        last_updated="2024-01-01T00:00:00Z",
        pending_steering_request=False,
        urgency=UrgencyLevel.LOW,
    )


def _request(agent_id: str) -> SteeringRequest:
    return SteeringRequest(
        agent_id=agent_id,
        relevant_context="some context",
        question="what should I do?",
        blocking=True,
        urgency=UrgencyLevel.MEDIUM,
    )


def _focus(agent_id: str) -> FocusContext:
    return FocusContext(
        agent_id=agent_id,
        task_description=f"Task for {agent_id}",
        steering_history=[],
        recent_output="recent work output",
        current_request=_request(agent_id),
    )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def test_count_tokens_nonempty() -> None:
    cb = _builder()
    n = cb.count_tokens("Hello, world!")
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_empty() -> None:
    cb = _builder()
    assert cb.count_tokens("") == 0


# ---------------------------------------------------------------------------
# build_registry_context
# ---------------------------------------------------------------------------


def test_build_registry_context_returns_str() -> None:
    cb = _builder()
    entries = [_entry("a1"), _entry("a2")]
    result = cb.build_registry_context(entries)
    assert isinstance(result, str)
    assert "a1" in result
    assert "a2" in result


def test_build_registry_context_within_budget() -> None:
    cb = _builder()
    entries = [_entry(f"a{i}") for i in range(10)]
    result = cb.build_registry_context(entries)
    tokens = cb.count_tokens(result)
    assert tokens < 200_000


# ---------------------------------------------------------------------------
# build_focus_context
# ---------------------------------------------------------------------------


def test_build_focus_context_returns_str() -> None:
    cb = _builder()
    focus = _focus("a1")
    registry = [_entry("a1"), _entry("a2"), _entry("a3")]
    result = cb.build_focus_context(focus, registry)
    assert isinstance(result, str)
    assert "a1" in result


def test_focus_context_within_budget() -> None:
    cb = _builder()
    focus = _focus("a1")
    registry = [_entry(f"a{i}") for i in range(5)]
    result = cb.build_focus_context(focus, registry)
    tokens = cb.count_tokens(result)
    assert tokens < 200_000


def test_focus_context_budget_exceeded_raises() -> None:
    tiny_budget = 5  # impossibly small
    cb = _builder(budget=tiny_budget)
    focus = _focus("a1")
    registry = [_entry("a1")]
    with pytest.raises(ContextBudgetError):
        cb.build_focus_context(focus, registry)


# ---------------------------------------------------------------------------
# build_flat_context
# ---------------------------------------------------------------------------


def test_build_flat_context_returns_str() -> None:
    cb = _builder()
    contexts = [_focus("a1"), _focus("a2"), _focus("a3")]
    result = cb.build_flat_context(contexts, _request("a1"))
    assert isinstance(result, str)
    for i in ("a1", "a2", "a3"):
        assert i in result


def test_flat_context_within_budget() -> None:
    cb = _builder()
    contexts = [_focus(f"a{i}") for i in range(5)]
    result = cb.build_flat_context(contexts, _request("a1"))
    tokens = cb.count_tokens(result)
    assert tokens < 200_000
