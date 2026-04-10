"""DACS orchestrator — the REGISTRY / FOCUS / USER_INTERACT state machine."""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

from dacs._context_builder import SYSTEM_PROMPT, ContextBuilder
from dacs._logger import Logger
from dacs._protocols import (
    FocusContext,
    SteeringRequest,
    SteeringResponse,
    SteeringRequestQueue,
    UrgencyLevel,
)
from dacs._registry import RegistryManager

if TYPE_CHECKING:
    from dacs._agent import BaseAgent

_FOCUS_MAX_TURNS = 3
_POLL_INTERVAL = 0.05  # seconds between queue checks when idle


class OrchestratorState(Enum):
    """Finite states of the DACS orchestrator."""

    REGISTRY = "REGISTRY"
    """Holding lightweight registry summaries; monitoring all agents."""

    FOCUS = "FOCUS"
    """Holding F(aᵢ) + compressed R_{-i}; steering one agent exclusively."""

    USER_INTERACT = "USER_INTERACT"
    """Responding to a user message with registry context."""


class Orchestrator:
    """DACS orchestrator state machine.

    In DACS mode (``focus_mode=True``) the orchestrator cycles between
    ``REGISTRY`` and ``FOCUS(aᵢ)`` states.  In baseline mode the context
    is always flat (all agents concatenated).

    You do not normally instantiate this directly — use :class:`~dacs.DACSRuntime`
    which wires all components for you.

    Parameters
    ----------
    registry:
        Shared :class:`~dacs.RegistryManager`.
    queue:
        Shared :class:`~dacs.SteeringRequestQueue`.
    context_builder:
        Token-counted :class:`~dacs.ContextBuilder`.
    llm_client:
        An ``AsyncAnthropic`` client (or any Anthropic-compatible client).
    model:
        Model identifier (e.g. ``"claude-3-haiku-20240307"``).
    token_budget:
        Hard cap on context window tokens.
    focus_mode:
        ``True`` for DACS, ``False`` for the flat-context baseline.
    focus_timeout:
        Maximum seconds per FOCUS session before the orchestrator times out.
    logger:
        Shared :class:`~dacs.Logger`.
    """

    def __init__(
        self,
        registry: RegistryManager,
        queue: SteeringRequestQueue,
        context_builder: ContextBuilder,
        llm_client: AsyncAnthropic,
        model: str,
        token_budget: int,
        focus_mode: bool = True,
        focus_timeout: int = 60,
        logger: Logger | None = None,
    ) -> None:
        self._registry = registry
        self._queue = queue
        self._cb = context_builder
        self._client = llm_client
        self._model = model
        self._budget = token_budget
        self._focus_mode = focus_mode
        self._focus_timeout = focus_timeout
        self._logger = logger or Logger(None)

        self._state: OrchestratorState = OrchestratorState.REGISTRY
        self._focus_agent_id: str | None = None

        self._agents: dict[str, "BaseAgent"] = {}
        self._steering_history: dict[str, list[dict]] = {}
        self._agent_contexts: dict[str, FocusContext] = {}

        self._running = False
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent: "BaseAgent") -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.agent_id] = agent
        self._steering_history[agent.agent_id] = []

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the orchestrator event loop.

        Runs until :meth:`stop` is called or all agents are done.
        """
        if self._stop_requested:
            return
        self._running = True
        while self._running and not self._stop_requested:
            request = self._queue.peek()
            if request is not None:
                self._queue.dequeue()
                await self._handle_steering(request)
            else:
                await asyncio.sleep(_POLL_INTERVAL)
        self._running = False

    def stop(self) -> None:
        """Signal the event loop to exit after the current steering call."""
        self._stop_requested = True
        self._running = False

    # ------------------------------------------------------------------
    # User message handler
    # ------------------------------------------------------------------

    async def handle_user_message(self, message: str) -> str:
        """Handle a user message in USER_INTERACT state.

        Always responds, even during an active FOCUS session. The current
        state is saved and restored after the user response is sent.

        Returns
        -------
        str
            The orchestrator's response text.
        """
        saved_state = self._state
        saved_focus = self._focus_agent_id
        self._transition(OrchestratorState.USER_INTERACT, None, "user_message")
        context = self._cb.build_registry_context(self._registry.get_all())
        prompt = f"{context}\n\nUser message: {message}"
        response_text, _, _, _ = await self._llm_call(
            context=prompt, state_label="USER_INTERACT", agent_id=None
        )
        self._transition(saved_state, saved_focus, "user_response_sent")
        return response_text

    @property
    def state(self) -> OrchestratorState:
        """Current orchestrator state."""
        return self._state

    @property
    def focus_agent_id(self) -> str | None:
        """Agent ID of the currently focused agent, or ``None``."""
        return self._focus_agent_id

    # ------------------------------------------------------------------
    # Steering dispatch
    # ------------------------------------------------------------------

    async def _handle_steering(self, request: SteeringRequest) -> None:
        self._registry.mark_steering_pending(request.agent_id)
        if self._focus_mode:
            await self._handle_focus(request)
        else:
            await self._handle_flat(request)
        self._registry.mark_steering_complete(request.agent_id)

    async def _handle_focus(self, request: SteeringRequest) -> None:
        self._transition(OrchestratorState.FOCUS, request.agent_id, "SteeringRequest")
        focus = FocusContext(
            agent_id=request.agent_id,
            task_description=self._registry.get(request.agent_id).task_description,
            steering_history=list(self._steering_history[request.agent_id]),
            recent_output=request.relevant_context,
            current_request=request,
        )
        context = self._cb.build_focus_context(focus, self._registry.get_all())
        start = time.monotonic()
        response_text = ""
        context_tokens = 0

        for turn in range(_FOCUS_MAX_TURNS):
            if time.monotonic() - start > self._focus_timeout:
                self._logger.log(
                    {
                        "event": "FOCUS_TIMEOUT",
                        "agent_id": request.agent_id,
                        "elapsed_s": round(time.monotonic() - start, 2),
                        "turns": turn,
                    }
                )
                self._transition(OrchestratorState.REGISTRY, None, "SteeringAbandoned")
                await self._deliver(
                    SteeringResponse(
                        request_id=request.request_id,
                        agent_id=request.agent_id,
                        response_text="[timeout — proceed on default path]",
                        context_size_at_time=context_tokens,
                        orchestrator_state="FOCUS",
                    )
                )
                return

            response_text, p_tok, r_tok, ms = await self._llm_call(
                context=context, state_label="FOCUS", agent_id=request.agent_id
            )
            context_tokens = p_tok

            if self._queue.has_high_urgency():
                next_req = self._queue.peek()
                if next_req and next_req.agent_id != request.agent_id:
                    self._logger.log(
                        {
                            "event": "INTERRUPT",
                            "interrupted_agent": request.agent_id,
                            "interrupting_agent": next_req.agent_id,
                            "urgency": "HIGH",
                            "request_id": next_req.request_id,
                        }
                    )
            break  # single LLM turn per focus session

        steering_resp = SteeringResponse(
            request_id=request.request_id,
            agent_id=request.agent_id,
            response_text=response_text,
            context_size_at_time=context_tokens,
            orchestrator_state="FOCUS",
        )
        self._logger.log(
            {
                "event": "STEERING_RESPONSE",
                "request_id": request.request_id,
                "agent_id": request.agent_id,
                "context_size_at_time": context_tokens,
                "orchestrator_state": "FOCUS",
                "response_text": response_text,
            }
        )
        self._steering_history[request.agent_id].append(
            {
                "request": {"question": request.question, "urgency": request.urgency.value},
                "response": {"response_text": response_text},
            }
        )
        await self._deliver(steering_resp)
        self._transition(OrchestratorState.REGISTRY, None, "SteeringComplete")

    async def _handle_flat(self, request: SteeringRequest) -> None:
        self._agent_contexts[request.agent_id] = FocusContext(
            agent_id=request.agent_id,
            task_description=self._registry.get(request.agent_id).task_description,
            steering_history=list(self._steering_history[request.agent_id]),
            recent_output=request.relevant_context,
            current_request=request,
        )
        all_contexts: list[FocusContext] = []
        for aid in self._agents:
            if aid in self._agent_contexts:
                all_contexts.append(self._agent_contexts[aid])
            else:
                entry = self._registry.get(aid)
                placeholder = SteeringRequest(
                    agent_id=aid,
                    relevant_context="",
                    question="(no steering request submitted yet)",
                    blocking=False,
                    urgency=UrgencyLevel.LOW,
                )
                all_contexts.append(
                    FocusContext(
                        agent_id=aid,
                        task_description=entry.task_description,
                        steering_history=self._steering_history.get(aid, []),
                        recent_output=entry.last_output_summary,
                        current_request=placeholder,
                    )
                )

        context = self._cb.build_flat_context(all_contexts, request)
        response_text, p_tok, r_tok, ms = await self._llm_call(
            context=context, state_label="FLAT", agent_id=request.agent_id
        )
        steering_resp = SteeringResponse(
            request_id=request.request_id,
            agent_id=request.agent_id,
            response_text=response_text,
            context_size_at_time=p_tok,
            orchestrator_state="FLAT",
        )
        self._logger.log(
            {
                "event": "STEERING_RESPONSE",
                "request_id": request.request_id,
                "agent_id": request.agent_id,
                "context_size_at_time": p_tok,
                "orchestrator_state": "FLAT",
                "response_text": response_text,
            }
        )
        self._steering_history[request.agent_id].append(
            {
                "request": {"question": request.question, "urgency": request.urgency.value},
                "response": {"response_text": response_text},
            }
        )
        await self._deliver(steering_resp)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _deliver(self, response: SteeringResponse) -> None:
        agent = self._agents.get(response.agent_id)
        if agent:
            await agent.deliver_response(response)

    def _transition(
        self,
        to: OrchestratorState,
        agent_id: str | None,
        trigger: str,
    ) -> None:
        from_label = self._state.value
        self._state = to
        self._focus_agent_id = agent_id if to == OrchestratorState.FOCUS else None
        self._logger.log(
            {
                "event": "TRANSITION",
                "from": from_label,
                "to": to.value,
                "agent_id": agent_id,
                "trigger": trigger,
            }
        )

    async def _llm_call(
        self,
        context: str,
        state_label: str,
        agent_id: str | None,
    ) -> tuple[str, int, int, int]:
        start = time.monotonic()
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        response_text = next(
            (block.text for block in response.content if block.type == "text"), ""
        )
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens
        self._logger.log(
            {
                "event": "LLM_CALL",
                "state": state_label,
                "agent_id": agent_id,
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "latency_ms": latency_ms,
            }
        )
        return response_text, in_tokens, out_tokens, latency_ms
