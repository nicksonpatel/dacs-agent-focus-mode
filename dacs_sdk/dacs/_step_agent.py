"""Step-driven ready-to-use agent for quick prototyping."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from dacs._agent import BaseAgent
from dacs._protocols import AgentStatus, UrgencyLevel

if TYPE_CHECKING:
    from dacs._protocols import SteeringRequestQueue
    from dacs._registry import RegistryManager

_URGENCY_MAP: dict[str, UrgencyLevel] = {
    "LOW": UrgencyLevel.LOW,
    "MEDIUM": UrgencyLevel.MEDIUM,
    "HIGH": UrgencyLevel.HIGH,
}


class StepAgent(BaseAgent):
    """A step-driven agent configured by a list of step dicts.

    This is the fastest way to create an agent without subclassing.
    Each step is a dict that may contain:

    ``summary`` *(str, required)*
        What the agent is doing at this step.  Pushed as a registry
        heartbeat.
    ``question`` *(str, optional)*
        If present, the agent will emit a SteeringRequest with this text
        and wait for the orchestrator's response before proceeding.
    ``urgency`` *(str, optional)*
        ``"LOW"`` | ``"MEDIUM"`` | ``"HIGH"`` (default ``"MEDIUM"``).
    ``sleep`` *(float, optional)*
        Seconds to sleep before the step, simulating work (default 0).

    Example
    -------
    >>> agent = StepAgent(
    ...     agent_id="writer",
    ...     task="Write a technical blog post about async Python",
    ...     steps=[
    ...         {"summary": "Researching topic", "sleep": 0.5},
    ...         {
    ...             "summary": "Choosing code examples",
    ...             "question": "Should I use asyncio.gather or asyncio.TaskGroup?",
    ...             "urgency": "MEDIUM",
    ...         },
    ...         {"summary": "Writing draft"},
    ...     ],
    ...     registry=...,
    ...     queue=...,
    ... )
    """

    def __init__(
        self,
        *,
        steps: list[dict],
        agent_id: str,
        task: str,
        registry: "RegistryManager | None" = None,
        queue: "SteeringRequestQueue | None" = None,
    ) -> None:
        super().__init__(agent_id=agent_id, task=task, registry=registry, queue=queue)
        self._steps = steps

    async def _execute(self) -> None:
        for step in self._steps:
            summary = step["summary"]
            urgency = _URGENCY_MAP.get(str(step.get("urgency", "MEDIUM")), UrgencyLevel.MEDIUM)
            question = step.get("question", "")
            sleep_s = float(step.get("sleep", 0))

            self._push_update(summary, AgentStatus.RUNNING, urgency)

            if sleep_s:
                await asyncio.sleep(sleep_s)

            if question:
                response = await self._request_steering(
                    context=self._recent_output(),
                    question=question,
                    blocking=True,
                    urgency=urgency,
                )
                self._push_update(
                    f"guidance received: {response.response_text[:80]}",
                    AgentStatus.RUNNING,
                    UrgencyLevel.LOW,
                )
