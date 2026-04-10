"""Integration tests for the full DACS stack (no real LLM calls)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dacs._agent import BaseAgent
from dacs._context_builder import ContextBuilder
from dacs._logger import Logger
from dacs._orchestrator import Orchestrator, OrchestratorState
from dacs._protocols import SteeringRequestQueue, UrgencyLevel
from dacs._registry import RegistryManager
from dacs._step_agent import StepAgent

_TIMEOUT = 10.0  # seconds — guards against infinite loops in tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stack(focus_mode: bool = True):
    """Return a wired (logger, registry, queue, cb, orchestrator) tuple."""
    logger = Logger(None)
    registry = RegistryManager(logger)
    queue = SteeringRequestQueue(logger)
    cb = ContextBuilder(200_000, logger)
    registry.set_context_builder(cb)

    mock_client = MagicMock()
    # Simulate Anthropic response object — type="text" is required by _llm_call
    mock_block = MagicMock()
    mock_block.text = "Use JSON format"
    mock_block.type = "text"
    mock_message = MagicMock()
    mock_message.content = [mock_block]
    mock_message.usage = MagicMock(input_tokens=100, output_tokens=20)
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    orch = Orchestrator(
        registry=registry,
        queue=queue,
        context_builder=cb,
        llm_client=mock_client,
        model="claude-test",
        token_budget=200_000,
        focus_mode=focus_mode,
        focus_timeout=5,
        logger=logger,
    )
    return logger, registry, queue, cb, orch


async def _run(orch: Orchestrator, *agents: BaseAgent, timeout: float = _TIMEOUT) -> None:
    """Run orchestrator + agents concurrently, with a timeout guard."""
    orch_task = asyncio.create_task(orch.run())
    agent_tasks = [asyncio.create_task(a.run()) for a in agents]

    async def _body() -> None:
        await asyncio.gather(*agent_tasks)
        orch.stop()
        await orch_task

    await asyncio.wait_for(_body(), timeout=timeout)


# ---------------------------------------------------------------------------
# Orchestrator state
# ---------------------------------------------------------------------------


def test_orchestrator_initial_state() -> None:
    _, _, _, _, orch = _make_stack()
    assert orch.state == OrchestratorState.REGISTRY
    assert orch.focus_agent_id is None


def test_register_agent() -> None:
    _, registry, _, _, orch = _make_stack()
    registry.register("a1", "Task one")

    class DummyAgent(BaseAgent):
        async def _execute(self):
            pass

    agent = DummyAgent(agent_id="a1", task="Task one")
    orch.register_agent(agent)
    assert "a1" in orch._agents


# ---------------------------------------------------------------------------
# StepAgent without real LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_agent_no_questions_completes() -> None:
    """A StepAgent with no steering questions runs to completion."""
    logger, registry, queue, cb, orch = _make_stack()
    registry.register("a1", "Simple task")

    agent = StepAgent(
        agent_id="a1",
        task="Simple task",
        steps=[
            {"summary": "Step one"},
            {"summary": "Step two"},
            {"summary": "Step three"},
        ],
        registry=registry,
        queue=queue,
    )
    orch.register_agent(agent)
    await _run(orch, agent)

    assert registry.get("a1").status.value == "COMPLETE"


@pytest.mark.asyncio
async def test_step_agent_with_steering() -> None:
    """StepAgent with one steering question gets a response."""
    logger, registry, queue, cb, orch = _make_stack()
    registry.register("writer", "Write a post")

    agent = StepAgent(
        agent_id="writer",
        task="Write a post",
        steps=[
            {"summary": "Researching"},
            {
                "summary": "Choosing examples",
                "question": "Use asyncio.gather or TaskGroup?",
                "urgency": "MEDIUM",
            },
            {"summary": "Writing"},
        ],
        registry=registry,
        queue=queue,
    )
    orch.register_agent(agent)
    await _run(orch, agent)

    assert registry.get("writer").status.value == "COMPLETE"


# ---------------------------------------------------------------------------
# Full concurrent multi-agent run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_agents_concurrent() -> None:
    """Three agents run concurrently, each with one steering question."""
    logger, registry, queue, cb, orch = _make_stack()

    agents = []
    for i in range(3):
        aid = f"a{i+1}"
        registry.register(aid, f"Task {i+1}")
        ag = StepAgent(
            agent_id=aid,
            task=f"Task {i+1}",
            steps=[
                {"summary": f"Working step 1 for {aid}"},
                {"summary": "Decision point", "question": "Which approach?"},
                {"summary": "Final step"},
            ],
            registry=registry,
            queue=queue,
        )
        orch.register_agent(ag)
        agents.append(ag)

    await _run(orch, *agents)

    for a in agents:
        assert registry.get(a.agent_id).status.value == "COMPLETE"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def test_logger_sink() -> None:
    events: list[dict] = []
    logger = Logger(None)
    logger.add_sink(events.append)

    logger.log({"event": "TEST_EVENT", "data": 42})
    assert len(events) == 1
    assert events[0]["event"] == "TEST_EVENT"
    assert events[0]["data"] == 42
    assert "ts" in events[0]


def test_logger_no_file() -> None:
    """Logger(None) should not raise even with events logged."""
    logger = Logger(None)
    logger.log({"event": "RUN_START"})
    logger.close()


# ---------------------------------------------------------------------------
# Baseline mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_mode_runs() -> None:
    """focus_mode=False should complete without error."""
    logger, registry, queue, cb, orch = _make_stack(focus_mode=False)
    registry.register("b1", "Baseline agent")

    agent = StepAgent(
        agent_id="b1",
        task="Baseline agent",
        steps=[
            {"summary": "Step"},
            {"summary": "Decision", "question": "Path A or B?"},
        ],
        registry=registry,
        queue=queue,
    )
    orch.register_agent(agent)
    await _run(orch, agent)

    assert registry.get("b1").status.value == "COMPLETE"


# ---------------------------------------------------------------------------
# Stop before run starts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_before_run_exits_immediately() -> None:
    """If stop() is called before run() starts, run() should return immediately."""
    _, _, _, _, orch = _make_stack()
    orch.stop()
    await asyncio.wait_for(orch.run(), timeout=1.0)

