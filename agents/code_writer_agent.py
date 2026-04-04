from __future__ import annotations

from agents.base_agent import BaseAgent
from src.protocols import AgentStatus, UrgencyLevel


class CodeWriterAgent(BaseAgent):
    """Simulates incremental code writing with three design decision points.

    Decision points (ground truth for experiment scoring):
      1. insert() implementation style → "recursive"
      2. duplicate value handling      → "ignore duplicates"
      3. to_list() traversal order     → "inorder"
    """

    _STEPS: list[tuple[str, bool, str, UrgencyLevel]] = [
        ("Defined Node class with value, left, right fields", False, "", UrgencyLevel.LOW),
        (
            "Stubbed insert() method, considering iterative vs recursive approaches",
            True,
            "Should insert() be iterative or recursive? Iterative is faster; recursive is cleaner.",
            UrgencyLevel.MEDIUM,
        ),
        ("Implemented insert() per guidance, writing unit tests", False, "", UrgencyLevel.LOW),
        (
            "Tests passing for basic cases; edge case: duplicate values encountered",
            True,
            "How should duplicate values be handled: ignore, raise ValueError, or allow in right subtree?",
            UrgencyLevel.HIGH,
        ),
        ("Handled duplicates per guidance, implementing delete()", False, "", UrgencyLevel.LOW),
        (
            "delete() implemented; need to choose traversal order for to_list()",
            True,
            "Which traversal order for to_list(): inorder (sorted), preorder, or postorder?",
            UrgencyLevel.MEDIUM,
        ),
        ("Implemented to_list() with chosen traversal; all tests passing", False, "", UrgencyLevel.LOW),
    ]

    async def _execute(self) -> None:
        for summary, needs_steering, question, urgency in self._STEPS:
            self._push_update(AgentStatus.RUNNING, summary, urgency)
            if needs_steering:
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
