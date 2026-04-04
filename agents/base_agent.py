from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.protocols import (
    AgentStatus,
    RegistryUpdate,
    SteeringRequest,
    SteeringResponse,
    UrgencyLevel,
)

if TYPE_CHECKING:
    from src.protocols import SteeringRequestQueue
    from src.registry import RegistryManager


class BaseAgent(ABC):
    """Abstract base for all stub agents.

    Agents run as asyncio tasks. Each agent:
      - Pushes RegistryUpdates after every step (event-driven heartbeat)
      - Emits SteeringRequests when it needs orchestrator input
      - Blocks on self._response_queue until the orchestrator delivers a response
    """

    def __init__(
        self,
        agent_id: str,
        task_description: str,
        registry: RegistryManager,
        queue: SteeringRequestQueue,
    ) -> None:
        self.agent_id        = agent_id
        self.task_description = task_description
        self._registry        = registry
        self._queue           = queue
        self._response_queue: asyncio.Queue[SteeringResponse] = asyncio.Queue()
        self._output_history: list[str] = []

    async def deliver_response(self, response: SteeringResponse) -> None:
        """Called by Orchestrator to deliver a steering response."""
        await self._response_queue.put(response)

    async def run(self) -> None:
        """Entry point — run as an asyncio task."""
        await self._execute()
        self._push_update(AgentStatus.COMPLETE, "task complete", UrgencyLevel.LOW)

    @abstractmethod
    async def _execute(self) -> None: ...

    # ------------------------------------------------------------------
    # Helpers used by subclasses
    # ------------------------------------------------------------------

    async def _request_steering(
        self,
        relevant_context: str,
        question: str,
        blocking: bool = True,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    ) -> SteeringResponse:
        """Emit a SteeringRequest and wait for the orchestrator's response."""
        request = SteeringRequest(
            agent_id=self.agent_id,
            relevant_context=relevant_context,
            question=question,
            blocking=blocking,
            urgency=urgency,
        )
        self._push_update(
            AgentStatus.WAITING_STEERING,
            f"waiting: {question[:60]}",
            urgency,
        )
        self._queue.enqueue(request)
        response = await self._response_queue.get()
        return response

    def _push_update(
        self,
        status: AgentStatus,
        summary: str,
        urgency: UrgencyLevel,
    ) -> None:
        """Push a registry heartbeat and record output locally."""
        self._output_history.append(summary)
        self._registry.update(
            self.agent_id,
            RegistryUpdate(
                agent_id=self.agent_id,
                status=status,
                last_output_summary=summary,
                urgency=urgency,
            ),
        )

    def _recent_output(self, k: int = 10) -> str:
        """Return the last k output lines as a single string."""
        return "\n".join(self._output_history[-k:])
