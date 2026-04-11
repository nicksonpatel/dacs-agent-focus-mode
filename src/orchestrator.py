from __future__ import annotations

import asyncio
import random
import time
from enum import Enum
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

from src.context_builder import SYSTEM_PROMPT, ContextBuilder
from src.logger import Logger, now_iso
from src.protocols import (
    FocusContext,
    SteeringRequest,
    SteeringResponse,
    SteeringRequestQueue,
    UrgencyLevel,
)
from src.registry import RegistryManager

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class OrchestratorState(Enum):
    REGISTRY      = "REGISTRY"
    FOCUS         = "FOCUS"
    USER_INTERACT = "USER_INTERACT"


_FOCUS_MAX_TURNS = 3   # cap per focus session (in addition to wall-clock timeout)
_POLL_INTERVAL   = 0.05  # seconds between queue checks when idle


class Orchestrator:
    """State machine + LLM call dispatcher.

    DACS mode  (focus_mode=True):  REGISTRY → FOCUS(aᵢ) → REGISTRY
    Baseline   (focus_mode=False): single FLAT state, all agent contexts concatenated
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
        log_path: str = "results/run.jsonl",
        ablation_mode: str | None = None,
    ) -> None:
        self._registry       = registry
        self._queue          = queue
        self._cb             = context_builder
        self._client         = llm_client
        self._model          = model
        self._budget         = token_budget
        self._focus_mode     = focus_mode
        self._focus_timeout  = focus_timeout
        self._logger         = logger or Logger(log_path)
        self._ablation_mode  = ablation_mode   # None | "no_registry" | "random_focus" | "flat_ordered"

        self._state: OrchestratorState   = OrchestratorState.REGISTRY
        self._focus_agent_id: str | None = None

        self._agents:           dict[str, BaseAgent]    = {}
        self._steering_history: dict[str, list[dict]]  = {}   # agent_id → [{request,response}]
        self._agent_contexts:   dict[str, FocusContext] = {}   # for baseline flat mode

        self._running = False

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents[agent.agent_id] = agent
        self._steering_history[agent.agent_id] = []

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        while self._running:
            request = self._queue.peek()
            if request is not None:
                self._queue.dequeue()
                await self._handle_steering(request)
            else:
                await asyncio.sleep(_POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # User message handler
    # ------------------------------------------------------------------

    async def handle_user_message(self, message: str) -> str:
        """Always handled. If in FOCUS, saves state, responds, then resumes."""
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

    # ------------------------------------------------------------------
    # Steering dispatch
    # ------------------------------------------------------------------

    async def _handle_steering(self, request: SteeringRequest) -> None:
        self._registry.mark_steering_pending(request.agent_id)
        if self._ablation_mode == "flat_ordered":
            await self._handle_flat_ordered(request)
        elif self._focus_mode:
            await self._handle_focus(request)
        else:
            await self._handle_flat(request)
        self._registry.mark_steering_complete(request.agent_id)

    async def _handle_focus(self, request: SteeringRequest) -> None:
        # Ablation: random_focus — focus on a random *other* agent's context
        if self._ablation_mode == "random_focus":
            other_ids = [aid for aid in self._agents if aid != request.agent_id]
            focus_id = random.choice(other_ids) if other_ids else request.agent_id
        else:
            focus_id = request.agent_id

        self._transition(OrchestratorState.FOCUS, request.agent_id, "SteeringRequest")
        focus = FocusContext(
            agent_id=focus_id,
            task_description=self._registry.get(focus_id).task_description,
            steering_history=list(self._steering_history.get(focus_id, [])),
            recent_output=request.relevant_context if focus_id == request.agent_id else (
                self._registry.get(focus_id).last_output_summary
            ),
            current_request=request,
        )
        # Ablation: no_registry — omit compressed registry from focus context
        include_registry = self._ablation_mode != "no_registry"
        context = self._cb.build_focus_context(focus, self._registry.get_all(), include_registry=include_registry)
        start = time.monotonic()
        response_text = ""
        context_tokens = 0

        for turn in range(_FOCUS_MAX_TURNS):
            if time.monotonic() - start > self._focus_timeout:
                self._logger.log({
                    "event": "FOCUS_TIMEOUT",
                    "agent_id": request.agent_id,
                    "elapsed_s": round(time.monotonic() - start, 2),
                    "turns": turn,
                })
                self._transition(OrchestratorState.REGISTRY, None, "SteeringAbandoned")
                await self._deliver(SteeringResponse(
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    response_text="[timeout — proceed on default path]",
                    context_size_at_time=context_tokens,
                    orchestrator_state="FOCUS",
                ))
                return

            response_text, p_tok, r_tok, ms = await self._llm_call(
                context=context, state_label="FOCUS", agent_id=request.agent_id
            )
            context_tokens = p_tok

            # Check for HIGH urgency interrupt after every LLM call
            if self._queue.has_high_urgency():
                next_req = self._queue.peek()
                if next_req and next_req.agent_id != request.agent_id:
                    self._logger.log({
                        "event": "INTERRUPT",
                        "interrupted_agent": request.agent_id,
                        "interrupting_agent": next_req.agent_id,
                        "urgency": "HIGH",
                        "request_id": next_req.request_id,
                    })
            # Single LLM turn is the steering response — no multi-turn dialogue
            break

        steering_resp = SteeringResponse(
            request_id=request.request_id,
            agent_id=request.agent_id,
            response_text=response_text,
            context_size_at_time=context_tokens,
            orchestrator_state="FOCUS",
        )
        self._logger.log({
            "event": "STEERING_RESPONSE",
            "request_id": request.request_id,
            "agent_id": request.agent_id,
            "context_size_at_time": context_tokens,
            "orchestrator_state": "FOCUS",
            "response_text": response_text,
        })
        self._steering_history[request.agent_id].append({
            "request": {"question": request.question, "urgency": request.urgency.value},
            "response": {"response_text": response_text},
        })
        await self._deliver(steering_resp)
        self._transition(OrchestratorState.REGISTRY, None, "SteeringComplete")

    async def _handle_flat(self, request: SteeringRequest) -> None:
        """Baseline: concatenate all agent contexts."""
        # Update (or create) the focus context for the requesting agent
        self._agent_contexts[request.agent_id] = FocusContext(
            agent_id=request.agent_id,
            task_description=self._registry.get(request.agent_id).task_description,
            steering_history=list(self._steering_history[request.agent_id]),
            recent_output=request.relevant_context,
            current_request=request,
        )
        # Build full context list for all registered agents
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
                all_contexts.append(FocusContext(
                    agent_id=aid,
                    task_description=entry.task_description,
                    steering_history=self._steering_history.get(aid, []),
                    recent_output=entry.last_output_summary,
                    current_request=placeholder,
                ))

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
        self._logger.log({
            "event": "STEERING_RESPONSE",
            "request_id": request.request_id,
            "agent_id": request.agent_id,
            "context_size_at_time": p_tok,
            "orchestrator_state": "FLAT",
            "response_text": response_text,
        })
        self._steering_history[request.agent_id].append({
            "request": {"question": request.question, "urgency": request.urgency.value},
            "response": {"response_text": response_text},
        })
        await self._deliver(steering_resp)

    async def _handle_flat_ordered(self, request: SteeringRequest) -> None:
        """Ablation: flat context with requesting agent's context placed first."""
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
                all_contexts.append(FocusContext(
                    agent_id=aid,
                    task_description=entry.task_description,
                    steering_history=self._steering_history.get(aid, []),
                    recent_output=entry.last_output_summary,
                    current_request=placeholder,
                ))

        context = self._cb.build_flat_ordered_context(all_contexts, request)
        response_text, p_tok, r_tok, ms = await self._llm_call(
            context=context, state_label="FLAT_ORDERED", agent_id=request.agent_id
        )
        steering_resp = SteeringResponse(
            request_id=request.request_id,
            agent_id=request.agent_id,
            response_text=response_text,
            context_size_at_time=p_tok,
            orchestrator_state="FLAT_ORDERED",
        )
        self._logger.log({
            "event": "STEERING_RESPONSE",
            "request_id": request.request_id,
            "agent_id": request.agent_id,
            "context_size_at_time": p_tok,
            "orchestrator_state": "FLAT_ORDERED",
            "response_text": response_text,
        })
        self._steering_history[request.agent_id].append({
            "request": {"question": request.question, "urgency": request.urgency.value},
            "response": {"response_text": response_text},
        })
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
        self._logger.log({
            "event": "TRANSITION",
            "from": from_label,
            "to": to.value,
            "agent_id": agent_id,
            "trigger": trigger,
        })

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
        latency_ms    = int((time.monotonic() - start) * 1000)
        # Extract text block (skip thinking blocks if model emits them)
        response_text = next(
            (block.text for block in response.content if block.type == "text"), ""
        )
        p_tok = response.usage.input_tokens
        r_tok = response.usage.output_tokens
        self._logger.log({
            "event":           "LLM_CALL",
            "state":           state_label,
            "agent_id":        agent_id,
            "context_tokens":  p_tok,
            "response_tokens": r_tok,
            "latency_ms":      latency_ms,
        })
        return response_text, p_tok, r_tok, latency_ms
