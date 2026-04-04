from __future__ import annotations

import asyncio

from agents.base_agent import BaseAgent
from src.protocols import AgentStatus, UrgencyLevel


_URGENCY_MAP: dict[str, UrgencyLevel] = {
    "LOW":    UrgencyLevel.LOW,
    "MEDIUM": UrgencyLevel.MEDIUM,
    "HIGH":   UrgencyLevel.HIGH,
}


class GenericAgent(BaseAgent):
    """Configurable stub agent driven by caller-supplied step definitions.

    Used for Phase 1+ scenarios where diverse agent tasks require different
    question arcs that can't use the hardcoded specialist agents.

    Each step is a dict with keys:
        summary  (str)               — what the agent is doing / has done
        urgency  (str)               — "LOW" | "MEDIUM" | "HIGH"
        question (str, optional)     — if present, triggers a steering request
        sleep    (float, optional)   — seconds to simulate work (default 0)

    Decision points in the task_suite's AgentSpec must use question_fragment
    values that are substrings of the 'question' fields in these steps.
    """

    def __init__(
        self,
        *,
        steps: list[dict],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._step_defs = steps

    async def _execute(self) -> None:
        for step in self._step_defs:
            summary  = step["summary"]
            urgency  = _URGENCY_MAP[step.get("urgency", "LOW")]
            question = step.get("question", "")
            sleep_s  = float(step.get("sleep", 0))

            self._push_update(AgentStatus.RUNNING, summary, urgency)

            if sleep_s:
                await asyncio.sleep(sleep_s)

            if question:
                response = await self._request_steering(
                    relevant_context=self._recent_output(),
                    question=question,
                    blocking=True,
                    urgency=urgency,
                )
                self._push_update(
                    AgentStatus.RUNNING,
                    f"guidance: {response.response_text[:80]}",
                    UrgencyLevel.LOW,
                )
