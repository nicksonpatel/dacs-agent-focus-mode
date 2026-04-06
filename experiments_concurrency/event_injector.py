"""UserInjector — fires scheduled user messages at the orchestrator mid-trial.

Simulates a human operator checking in on the system while agents are
actively requesting steering decisions. This is the key element that
differentiates the *_concurrent conditions from the *_clean conditions.

The injector runs as a separate asyncio task alongside the agent tasks and
the orchestrator's run loop. It uses asyncio.sleep for timing so it does
not block or busy-wait.

The orchestrator's handle_user_message() already supports interruption:
it saves the current state (FOCUS or REGISTRY), transitions to USER_INTERACT,
responds, then restores the saved state. No changes to src/ needed.

Events logged
-------------
    INJECTION        {message_fragment, delay_s, actual_ts}
    USER_RESPONSE    {message, response_text}    ← needed by InlineJudge

The USER_RESPONSE event is logged here (not in the orchestrator) because
handle_user_message() returns the response_text but does not log it.
"""
from __future__ import annotations

import asyncio

from src.logger import Logger, now_iso
from src.orchestrator import Orchestrator


class UserInjector:
    """Fires (delay_s, message) pairs from the scenario's injection schedule."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        injection_schedule: list[tuple[float, str]],
        logger: Logger,
    ) -> None:
        self._orchestrator = orchestrator
        self._schedule = injection_schedule
        self._logger = logger
        self._responses: list[dict] = []  # accumulated for post-trial summary

    # ------------------------------------------------------------------
    # Async runner — launch as asyncio.create_task
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Fire scheduled messages in wall-clock order.

        Each (delay_s, message) is fired after sleeping delay_s from the
        moment run() is called (i.e. relative to trial start, not previous
        injection). We sort by delay so out-of-order definitions still work.
        """
        sorted_schedule = sorted(self._schedule, key=lambda x: x[0])
        for delay_s, message in sorted_schedule:
            await asyncio.sleep(delay_s)
            await self._inject(message, delay_s)

    async def _inject(self, message: str, scheduled_delay_s: float) -> None:
        # Log that an injection is happening (before the LLM call)
        self._logger.log({
            "event":          "INJECTION",
            "message_fragment": message[:80],
            "scheduled_delay_s": scheduled_delay_s,
        })

        response_text = await self._orchestrator.handle_user_message(message)

        # Log the full user response so InlineJudge can find and score it
        self._logger.log({
            "event":         "USER_RESPONSE",
            "message":       message,
            "response_text": response_text,
        })

        self._responses.append({
            "message": message,
            "response_text": response_text,
            "scheduled_delay_s": scheduled_delay_s,
            "actual_ts": now_iso(),
        })

    # ------------------------------------------------------------------
    # Accessor (used by harness for metrics)
    # ------------------------------------------------------------------

    @property
    def responses(self) -> list[dict]:
        return list(self._responses)
