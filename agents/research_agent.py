from __future__ import annotations

from agents.base_agent import BaseAgent
from src.protocols import AgentStatus, UrgencyLevel


class ResearchAgent(BaseAgent):
    """Simulates literature research and survey writing with three clarification points.

    Decision points (ground truth for experiment scoring):
      1. Conflicting source authority  → "use Vaswani 2017 as authoritative"
      2. Sparse attention scope        → "include Longformer, BigBird, Reformer only"
      3. Citation depth                → "deep (full methodology review)"
    """

    _STEPS: list[tuple[str, bool, str, UrgencyLevel]] = [
        ("Identified 3 primary sources on transformer attention mechanisms", False, "", UrgencyLevel.LOW),
        (
            "Found conflicting complexity claims between Vaswani 2017 and a 2024 survey",
            True,
            "Conflicting claims on attention complexity (O(n²) vs O(n log n)). "
            "Which source should be authoritative for the survey?",
            UrgencyLevel.HIGH,
        ),
        ("Resolved conflict per guidance; expanding search to sparse attention variants", False, "", UrgencyLevel.LOW),
        (
            "Located 12 papers on sparse attention; scope unclear",
            True,
            "Should the survey cover all 12 sparse attention variants or focus on "
            "Longformer, BigBird, and Reformer only?",
            UrgencyLevel.MEDIUM,
        ),
        ("Scoped to specified variants; writing section summaries", False, "", UrgencyLevel.LOW),
        (
            "Section summaries drafted; need citation depth guidance",
            True,
            "Should citations be shallow (abstract + key numbers only) or deep (full methodology review)?",
            UrgencyLevel.MEDIUM,
        ),
        ("Completed all summaries with specified citation depth; compiling final report", False, "", UrgencyLevel.LOW),
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
