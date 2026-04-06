"""Quick smoke test for experiments_concurrency module (no LLM calls)."""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, "/Users/nickson/Documents/GitHub/agent-focus-mode")

from src.logger import Logger
from src.protocols import SteeringRequest, UrgencyLevel
from experiments_concurrency.rubric_judge import InlineJudge
from experiments_concurrency.harness import _TrackedQueue
from experiments_concurrency.scenario_defs import CONCURRENCY_SCENARIOS


def test_tracked_queue_routes_to_judge():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp = f.name
    try:
        logger = Logger(tmp)

        class _FakeLLM:
            pass

        judge = InlineJudge(_FakeLLM(), "test", logger, {"a1": "design db schema"})
        queue = _TrackedQueue(logger)
        queue.attach_judge(judge)

        req = SteeringRequest(
            agent_id="a1",
            relevant_context="context",
            question="PostgreSQL or MongoDB?",
            blocking=True,
            urgency=UrgencyLevel.MEDIUM,
        )
        queue.enqueue(req)

        assert req.request_id in judge._pending_questions, "question not tracked"
        assert judge._pending_questions[req.request_id] == "PostgreSQL or MongoDB?"
        print("  test_tracked_queue_routes_to_judge  PASS")
    finally:
        os.unlink(tmp)


def test_judge_enqueues_steering_response():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp = f.name
    try:
        logger = Logger(tmp)

        class _FakeLLM:
            pass

        judge = InlineJudge(_FakeLLM(), "test", logger, {"a1": "task"})
        judge.on_event({
            "event": "STEERING_RESPONSE",
            "request_id": "r1",
            "agent_id": "a1",
            "response_text": "Use PostgreSQL.",
            "context_size_at_time": 400,
            "orchestrator_state": "FOCUS",
        })
        assert judge._queue.qsize() == 1
        print("  test_judge_enqueues_steering_response  PASS")
    finally:
        os.unlink(tmp)


def test_judge_enqueues_user_response():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp = f.name
    try:
        logger = Logger(tmp)

        class _FakeLLM:
            pass

        judge = InlineJudge(_FakeLLM(), "test", logger, {"a1": "task"})
        judge.on_event({
            "event": "USER_RESPONSE",
            "message": "How are things?",
            "response_text": "All agents are progressing.",
        })
        assert judge._queue.qsize() == 1
        enqueued = judge._queue.get_nowait()
        assert enqueued["_type"] == "user"
        print("  test_judge_enqueues_user_response  PASS")
    finally:
        os.unlink(tmp)


def test_judge_stops_cleanly():
    """Judge run_worker() should exit after stop() even with empty queue."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp = f.name
    try:
        logger = Logger(tmp)

        class _FakeLLM:
            pass

        judge = InlineJudge(_FakeLLM(), "test", logger, {})
        judge.stop()

        async def _run():
            await asyncio.wait_for(judge.run_worker(), timeout=2.0)

        asyncio.run(_run())
        print("  test_judge_stops_cleanly  PASS")
    finally:
        os.unlink(tmp)


def test_scenario_structure():
    for sid, sc in CONCURRENCY_SCENARIOS.items():
        for agent in sc.agents:
            steps_with_q = [s for s in (agent.steps or []) if s.get("question")]
            assert len(steps_with_q) == len(agent.decision_points), (
                f"{sid}/{agent.agent_id}: {len(steps_with_q)} questions != "
                f"{len(agent.decision_points)} DPs"
            )
    print("  test_scenario_structure  PASS")


if __name__ == "__main__":
    print("Running smoke tests...")
    test_tracked_queue_routes_to_judge()
    test_judge_enqueues_steering_response()
    test_judge_enqueues_user_response()
    test_judge_stops_cleanly()
    test_scenario_structure()
    print("All smoke tests passed.")
