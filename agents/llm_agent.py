"""Real LLM-driven agent for DACS real-agent validation experiment.

Unlike GenericAgent (which uses scripted step-list stubs), LLMAgent calls the
model to generate actual work output and autonomously decides when it needs
orchestrator guidance by emitting a ``[[STEER: <question>]]`` marker.

This agent is intentionally minimal — it exercises the same
``BaseAgent._request_steering()`` / ``SteeringRequestQueue`` code path as the
synthetic agents, so the orchestrator sees no difference between real and stub
agents.  The only thing that changes is *who* generates the question text:
here it is the LLM, not a hardcoded template.
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

from agents.base_agent import BaseAgent
from src.protocols import AgentStatus, UrgencyLevel

if TYPE_CHECKING:
    from src.protocols import SteeringRequestQueue
    from src.registry import RegistryManager

# Match [[STEER: ...]] — non-greedy, multi-line question text allowed
_STEER_RE = re.compile(r"\[\[STEER:\s*(.*?)\]\]", re.IGNORECASE | re.DOTALL)
_DONE_RE  = re.compile(r"\[\[DONE\]\]", re.IGNORECASE)

# Maximum question length forwarded to the orchestrator (chars → stays well under
# the ≤100-token guidance; ≈400 chars ≈ 100 tokens for typical prose)
_MAX_QUESTION_CHARS = 400


class LLMAgent(BaseAgent):
    """Real LLM-driven agent for the DACS real-agent validation experiment.

    The agent drives an internal conversation loop with the LLM.  When it
    encounters a design decision it emits ``[[STEER: <question>]]``, which is
    caught by ``_execute()``, forwarded to the orchestrator via
    ``BaseAgent._request_steering()``, and the orchestrator's reply is injected
    back into the conversation as a user message before the next LLM turn.

    Args:
        agent_id:              Unique agent identifier.
        task_description:      Task description (≤50 tokens) — written into system
                               prompt and pushed to the registry on heartbeats.
        decision_hints:        Plain-text paragraph naming the *types* of decisions
                               the agent may need guidance on.  Must NOT prescribe
                               correct answers — the LLM must reason to its own
                               conclusions so questions are genuinely open.
        client:                AsyncAnthropic client shared with the orchestrator.
        model:                 Model name for agent LLM calls.
        registry:              Shared RegistryManager instance.
        queue:                 Shared SteeringRequestQueue instance.
        max_steps:             Maximum LLM turns before forced termination.
        max_steering_requests: Maximum [[STEER: ...]] markers the agent may emit.
        agent_max_tokens:      Max tokens per agent LLM response.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        task_description: str,
        decision_hints: str,
        client: AsyncAnthropic,
        model: str,
        registry: "RegistryManager",
        queue: "SteeringRequestQueue",
        max_steps: int = 12,
        max_steering_requests: int = 3,
        agent_max_tokens: int = 800,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            task_description=task_description,
            registry=registry,
            queue=queue,
        )
        self._client      = client
        self._model       = model
        self._max_steps   = max_steps
        self._max_steer   = max_steering_requests
        self._max_tokens  = agent_max_tokens
        self._system_prompt = self._build_system_prompt(
            task_description, decision_hints, max_steering_requests
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(
        task_description: str,
        decision_hints: str,
        max_steering_requests: int,
    ) -> str:
        return (
            "You are an intelligent specialist agent working inside a multi-agent "
            "orchestration system.\n\n"
            f"Your task:\n{task_description}\n\n"
            "Work through this task step-by-step, showing your reasoning and progress "
            "in plain text.  Be concrete and technical.\n\n"
            "REQUIRED DECISION CONSULTATIONS\n"
            "================================\n"
            "You MUST consult the orchestrator on EACH of the following decision "
            "categories before you can mark the task as done.  Do not assume a "
            "default — ask the orchestrator for guidance on every item listed below:\n\n"
            f"{decision_hints}\n\n"
            "Work through these decision points in order.  Each time you reach one, "
            "pause and use [[STEER: ...]] before continuing.\n\n"
            "--- STEERING PROTOCOL ---\n"
            "When you reach a required decision point, emit EXACTLY this token on "
            "its own line (nothing before or after it on that line):\n\n"
            "  [[STEER: <your specific, self-contained question about the decision>]]\n\n"
            "Then STOP immediately — do not write anything after the [[STEER: ...]] "
            "line.  You will receive the orchestrator's guidance before your next turn.\n"
            f"You may use [[STEER: ...]] at most {max_steering_requests} times total.\n\n"
            "After receiving guidance, acknowledge it briefly, then continue your work "
            "incorporating the advice.\n\n"
            "When ALL required decisions have been consulted on and your task is fully "
            "complete, emit EXACTLY this on its own line:\n\n"
            "  [[DONE]]\n\n"
            "Do NOT emit [[DONE]] until you have used [[STEER: ...]] for each of the "
            "required decision categories listed above."
        )

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    async def _execute(self) -> None:
        conversation: list[dict] = [
            {
                "role": "user",
                "content": (
                    "Begin working on your task.  Show your step-by-step reasoning "
                    "and progress.  When you hit a decision point you need guidance "
                    "on, use [[STEER: ...]].  Signal completion with [[DONE]]."
                ),
            }
        ]
        steering_count = 0

        for _step in range(self._max_steps):
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=conversation,
            )
            # Some models (e.g. MiniMax-M2.7) prepend ThinkingBlock objects;
            # extract the first block that actually has a .text attribute.
            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text = block.text
                    break

            # Heartbeat: push last 80 chars as progress summary
            summary_tail = text.strip().replace("\n", " ")[-80:] or "(working)"
            self._push_update(AgentStatus.RUNNING, summary_tail, UrgencyLevel.LOW)

            # Append assistant turn before any branching
            conversation.append({"role": "assistant", "content": text})

            # Termination check
            if _DONE_RE.search(text):
                break

            # Steering check
            steer_match = _STEER_RE.search(text)
            if steer_match and steering_count < self._max_steer:
                raw_question = steer_match.group(1).strip()
                question = raw_question[:_MAX_QUESTION_CHARS]
                steering_count += 1

                steering_resp = await self._request_steering(
                    relevant_context=self._recent_output(k=5),
                    question=question,
                    blocking=True,
                    urgency=UrgencyLevel.MEDIUM,
                )

                # Inject orchestrator guidance as the next user turn
                guidance_msg = f"Orchestrator guidance: {steering_resp.response_text}"
                conversation.append({"role": "user", "content": guidance_msg})

            # Yield to event loop so concurrently running agents and the
            # orchestrator polling loop can make progress
            await asyncio.sleep(0)
