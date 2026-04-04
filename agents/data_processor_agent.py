from __future__ import annotations

from agents.base_agent import BaseAgent
from src.protocols import AgentStatus, UrgencyLevel


class DataProcessorAgent(BaseAgent):
    """Simulates CSV data cleaning with three format ambiguity decision points.

    Decision points (ground truth for experiment scoring):
      1. Mixed encoding handling → "coerce to UTF-8"
      2. Null region strategy    → "fill with mode"
      3. Outlier strategy        → "clip to 3-sigma bounds"
    """

    _STEPS: list[tuple[str, bool, str, UrgencyLevel]] = [
        ("Loaded sales CSV: 50,000 rows, 12 columns detected", False, "", UrgencyLevel.LOW),
        (
            "Found mixed encoding in 'description' column: UTF-8 and Latin-1 rows present",
            True,
            "Description column has mixed UTF-8/Latin-1 encoding. "
            "Should I: drop non-UTF-8 rows, coerce all to UTF-8, or preserve encoding as-is?",
            UrgencyLevel.HIGH,
        ),
        ("Encoding resolved per guidance; running null-value audit", False, "", UrgencyLevel.LOW),
        (
            "Null audit complete: 8% nulls in 'region' column, 0.3% in 'amount'",
            True,
            "Region column has 8% null values. Strategy: fill with column mode, "
            "forward-fill by date order, or drop rows with null region?",
            UrgencyLevel.MEDIUM,
        ),
        ("Applied null strategy per guidance; normalizing 'amount' column", False, "", UrgencyLevel.LOW),
        (
            "Amount column has outliers: 0.1% of values exceed 3-sigma bounds",
            True,
            "Amount outliers detected (0.1% of rows exceed 3σ). "
            "Strategy: clip to 3-sigma bounds, drop outlier rows, or keep as-is?",
            UrgencyLevel.MEDIUM,
        ),
        ("Applied outlier strategy per guidance; output written to results/processed.csv", False, "", UrgencyLevel.LOW),
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
