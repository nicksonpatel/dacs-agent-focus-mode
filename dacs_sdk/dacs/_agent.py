"""Abstract base class for DACS agents."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dacs._protocols import (
    AgentStatus,
    RegistryUpdate,
    SteeringRequest,
    SteeringResponse,
    UrgencyLevel,
)

if TYPE_CHECKING:
    from dacs._protocols import SteeringRequestQueue
    from dacs._registry import RegistryManager


class BaseAgent(ABC):
    """Abstract base class for all DACS agents.

    Subclass this and implement :meth:`_execute` to define your agent's
    behaviour.  The agent runs as an asyncio task that shares a single event
    loop with the orchestrator.

    Lifecycle
    ---------
    1. The agent is created and registered with a :class:`~dacs.DACSRuntime`.
    2. ``run()`` is called as an asyncio task.
    3. Inside ``_execute()``, the agent calls :meth:`_push_update` after each
       meaningful step to update its registry entry (heartbeat).
    4. At decision points, the agent calls :meth:`_request_steering` to get
       orchestrator guidance.  The call blocks until the response arrives.
    5. When ``_execute()`` returns, the agent is automatically marked ``COMPLETE``.

    Parameters
    ----------
    agent_id:
        Unique identifier for this agent (e.g. ``"writer"`` or ``"a1"``).
    task:
        One-sentence task description (≤50 tokens).
    registry:
        Shared :class:`~dacs.RegistryManager` instance.  When using
        :class:`~dacs.DACSRuntime` you can omit this — the runtime injects
        it automatically when you call :meth:`~dacs.DACSRuntime.add_agent`.
    queue:
        Shared :class:`~dacs.SteeringRequestQueue` instance.  Same as
        above — omit when using :class:`~dacs.DACSRuntime`.
    """

    def __init__(
        self,
        agent_id: str,
        task: str,
        registry: "RegistryManager | None" = None,
        queue: "SteeringRequestQueue | None" = None,
    ) -> None:
        self.agent_id = agent_id
        self.task = task
        self._registry = registry
        self._queue = queue
        self._response_queue: asyncio.Queue[SteeringResponse] = asyncio.Queue()
        self._output_history: list[str] = []

    async def deliver_response(self, response: SteeringResponse) -> None:
        """Called by :class:`~dacs.Orchestrator` to deliver a steering response."""
        await self._response_queue.put(response)

    async def run(self) -> None:
        """Entry-point called as an asyncio task by :class:`~dacs.DACSRuntime`."""
        await self._execute()
        self._push_update("task complete", AgentStatus.COMPLETE, UrgencyLevel.LOW)

    @abstractmethod
    async def _execute(self) -> None:
        """Implement your agent logic here.

        Call :meth:`_push_update` after each meaningful step and
        :meth:`_request_steering` at decision points.
        """

    # ------------------------------------------------------------------
    # Helpers available in _execute()
    # ------------------------------------------------------------------

    async def _request_steering(
        self,
        context: str,
        question: str,
        blocking: bool = True,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    ) -> SteeringResponse:
        """Emit a :class:`~dacs.SteeringRequest` and wait for the orchestrator.

        The call suspends the agent coroutine until the orchestrator
        delivers a :class:`~dacs.SteeringResponse`.

        Parameters
        ----------
        context:
            Relevant context excerpt for the orchestrator (recent output,
            decision state, etc.).
        question:
            The specific decision or clarification needed.
        blocking:
            If ``True`` the agent has halted and needs the response before
            it can continue.  If ``False`` the agent continues on a default
            path while waiting.
        urgency:
            ``HIGH`` can interrupt an active FOCUS session.

        Returns
        -------
        SteeringResponse
            The orchestrator's decision.
        """
        request = SteeringRequest(
            agent_id=self.agent_id,
            relevant_context=context,
            question=question,
            blocking=blocking,
            urgency=urgency,
        )
        self._push_update(
            f"waiting for steering: {question[:60]}",
            AgentStatus.WAITING_STEERING,
            urgency,
        )
        self._queue.enqueue(request)
        response = await self._response_queue.get()
        return response

    def _push_update(
        self,
        summary: str,
        status: AgentStatus = AgentStatus.RUNNING,
        urgency: UrgencyLevel = UrgencyLevel.LOW,
    ) -> None:
        """Push a heartbeat update to the registry.

        Call this after each meaningful step to keep the orchestrator's
        registry current.

        Parameters
        ----------
        summary:
            Brief description of what was just accomplished (≤100 tokens).
        status:
            New agent status (default: ``RUNNING``).
        urgency:
            Current urgency level (default: ``LOW``).
        """
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
        """Return the last *k* output lines joined as a single string."""
        return "\n".join(self._output_history[-k:])
