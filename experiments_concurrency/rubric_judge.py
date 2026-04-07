"""Inline LLM rubric judge for the concurrency & interruption experiment.

InlineJudge runs as a concurrent asyncio worker **during** the trial — not
post-hoc. It attaches to the Logger via a sink callback, enqueues eligible
events as they fire, and calls the LLM for each one to produce a holistic
1–10 quality score.

Two event types are judged
--------------------------
1. STEERING_RESPONSE — orchestrator's answer to an agent steering request.
   Judge prompt: agent task + full question + full response text.

2. USER_RESPONSE — orchestrator's answer to a user-injected message (logged
   by UserInjector directly since handle_user_message() doesn't log this).
   Judge prompt: all agent tasks (situational awareness) + user message +
   full response text.

Score semantics (1–10)
-----------------------
  10  Precise, technically sound, directly addresses the question, zero
      cross-agent confusion.
  7–9  Mostly correct with minor gaps or light hedging.
  4–6  Partially addresses the question or shows mild topic drift.
  1–3  Irrelevant, wrong, or clearly contaminated by another agent's
      vocabulary or domain.

Output: JUDGE_SCORE event written to the **same JSONL** as the trial.
  {
    "event":             "JUDGE_SCORE",
    "request_id":        str | None,     # matches STEERING_RESPONSE / omitted for USER_RESPONSE
    "agent_id":          str | None,     # None for USER_RESPONSE
    "event_type_judged": "steering" | "user",
    "score":             int,            # 1–10
    "reason":            str,            # one-sentence explanation
    "latency_ms":        int,
    "ts":                str,
  }

How to wire up
--------------
    judge = InlineJudge(llm_client, model, logger, agent_task_descriptions)
    # Register sink BEFORE agents start so no events are missed:
    logger.add_sink(judge.on_event)
    # Also pass the queue subclass so questions are tracked (see TrackedQueue
    # in harness.py).
    # Launch as a fourth asyncio task:
    judge_task = asyncio.create_task(judge.run_worker())
    # After trial ends:
    judge.stop()
    await judge_task
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from src.logger import Logger, now_iso

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from src.protocols import SteeringRequest


_STEERING_SYSTEM = """\
You are an expert technical evaluator assessing an AI orchestrator's steering decisions.

Scoring rubric:
  10 — Response is precise, technically sound, and directly addresses the specific \
question without any confusion with other agents' tasks.
   7 — Mostly correct; minor gaps, hedging, or slight imprecision.
   4 — Partially addresses the question, or shows noticeable topic drift.
   1 — Irrelevant, technically wrong, or clearly contaminated by another agent's domain.

Respond with valid JSON only. Example: {"score": 8, "reason": "Correctly recommends sliding window but omits the Redis sorted-set implementation detail."}
"""

_STEERING_USER_TMPL = """\
Agent task: {task}

Steering question: {question}

Orchestrator response:
{response}

Score this response 1–10 using the rubric. Respond with JSON only: {{"score": N, "reason": "one sentence"}}"""


_USER_SYSTEM = """\
You are an expert evaluator assessing an AI orchestrator's situational awareness \
when responding to a human user message during a live multi-agent session.

Scoring rubric:
  10 — Accurately reflects current agent states, directly addresses the user's query, \
no hallucination or wrong agent associations.
   7 — Mostly accurate; minor omissions or light inaccuracy on one agent's state.
   4 — Partially accurate; misses important states or conflates agent tasks.
   1 — Confused, wrong agent associations, or off-topic entirely.

Respond with valid JSON only. Example: {"score": 9, "reason": "Correctly identifies a2 as blocked but does not mention a3's HIGH-urgency request."}
"""

_USER_USER_TMPL = """\
Active agents and their tasks:
{agent_list}

User message: {message}

Orchestrator response:
{response}

Score this response 1–10 using the rubric. Respond with JSON only: {{"score": N, "reason": "one sentence"}}"""


class InlineJudge:
    """Async worker that judges STEERING_RESPONSE and USER_RESPONSE events in real time."""

    def __init__(
        self,
        llm_client: "AsyncAnthropic",
        model: str,
        logger: Logger,
        agent_task_descriptions: dict[str, str],
    ) -> None:
        self._client = llm_client
        self._model = model
        self._logger = logger
        self._agent_tasks = agent_task_descriptions  # {agent_id: task_description}

        # Async queue of dicts ready for judging
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._stop_event = asyncio.Event()

        # Map request_id → question text, populated by track_request() which is
        # called from TrackedQueue.enqueue() before the orchestrator sees the request.
        self._pending_questions: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Called before trial starts — one entry per agent
    # ------------------------------------------------------------------

    def track_request(self, request: "SteeringRequest") -> None:
        """Store the question text keyed by request_id so it's available when
        the STEERING_RESPONSE fires later."""
        self._pending_questions[request.request_id] = request.question

    # ------------------------------------------------------------------
    # Logger sink — called synchronously on every logged event
    # ------------------------------------------------------------------

    def on_event(self, event: dict) -> None:
        """Enqueue events that should be judged. Non-blocking (put_nowait)."""
        ev = event.get("event")
        if ev == "STEERING_RESPONSE":
            self._queue.put_nowait({"_type": "steering", **event})
        elif ev == "USER_RESPONSE":
            self._queue.put_nowait({"_type": "user", **event})

    # ------------------------------------------------------------------
    # Stop signal (call after trial ends)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Async worker — runs as a task throughout the trial
    # ------------------------------------------------------------------

    async def run_worker(self) -> None:
        """Drain the queue until stopped AND queue is empty."""
        while True:
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(0.05)
                continue
            await self._judge_event(event)

    # ------------------------------------------------------------------
    # Judging
    # ------------------------------------------------------------------

    async def _judge_event(self, event: dict) -> None:
        ev_type = event["_type"]
        t0 = time.monotonic()

        if ev_type == "steering":
            score, reason = await self._judge_steering(event)
            request_id = event.get("request_id")
            agent_id = event.get("agent_id")
        else:
            score, reason = await self._judge_user(event)
            request_id = None
            agent_id = None

        latency_ms = round((time.monotonic() - t0) * 1000)

        self._logger.log({
            "event":             "JUDGE_SCORE",
            "request_id":        request_id,
            "agent_id":          agent_id,
            "event_type_judged": ev_type,
            "score":             score,
            "reason":            reason,
            "latency_ms":        latency_ms,
        })

    async def _judge_steering(self, event: dict) -> tuple[int, str]:
        request_id   = event.get("request_id", "")
        agent_id     = event.get("agent_id", "")
        response_text = event.get("response_text", "")
        question      = self._pending_questions.get(request_id, "(question not recorded)")
        task          = self._agent_tasks.get(agent_id, "(task unknown)")

        user_content = _STEERING_USER_TMPL.format(
            task=task,
            question=question,
            response=response_text,
        )
        return await self._llm_judge(_STEERING_SYSTEM, user_content)

    async def _judge_user(self, event: dict) -> tuple[int, str]:
        message   = event.get("message", "")
        response  = event.get("response_text", "")
        agent_lines = "\n".join(
            f"  {aid}: {task}" for aid, task in self._agent_tasks.items()
        )
        user_content = _USER_USER_TMPL.format(
            agent_list=agent_lines,
            message=message,
            response=response,
        )
        return await self._llm_judge(_USER_SYSTEM, user_content)

    async def _llm_judge(self, system: str, user_content: str) -> tuple[int, str]:
        """Call the LLM and parse JSON {score, reason}. Returns (score, reason)."""
        try:
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            # Use block.type == "text" to skip ThinkingBlocks — same pattern
            # as src/orchestrator.py which is proven to work with this model.
            raw = next(
                (block.text for block in msg.content if block.type == "text"), ""
            ).strip()
            if not raw:
                return 0, "judge_error: empty response from model"
            # Strip leading/trailing markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            # Strip <think>…</think> wrapper if present
            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            parsed = json.loads(raw)
            score  = int(parsed["score"])
            reason = str(parsed.get("reason", ""))
            # Clamp to valid range
            score = max(1, min(10, score))
            return score, reason
        except Exception as exc:  # noqa: BLE001
            return 0, f"judge_error: {exc}"
