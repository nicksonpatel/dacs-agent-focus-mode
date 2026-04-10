"""Deterministic, token-counted context assembly — the central DACS invariant."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken

from dacs._protocols import AgentStatus, FocusContext, RegistryEntry, SteeringRequest, UrgencyLevel

if TYPE_CHECKING:
    from dacs._logger import Logger

# System prompt injected into every LLM call.
SYSTEM_PROMPT = (
    "You are an orchestrator managing parallel AI agents. "
    "When responding to a steering request, provide a clear, specific decision "
    "or clarification. Reference only the agent and context provided. Be concise."
)


class ContextBudgetError(Exception):
    """Raised when a single agent's focus context alone exceeds the token budget."""


class ContextBuilder:
    """Assembles the orchestrator's prompt before every LLM call.

    The builder enforces a hard token budget ``T`` deterministically.
    Every byte that enters an LLM call is logged with its token count via
    a ``CONTEXT_BUILT`` event, making the context window the primary
    experimental observable.

    Parameters
    ----------
    token_budget:
        Maximum number of tokens that may appear in any single LLM call.
    logger:
        Logger instance to receive ``CONTEXT_BUILT`` events.
    """

    def __init__(self, token_budget: int, logger: "Logger") -> None:
        self._budget = token_budget
        self._logger = logger
        self._enc = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Return the token count for *text* using cl100k_base encoding."""
        return len(self._enc.encode(text))

    def build_focus_context(
        self,
        focus: FocusContext,
        registry: list[RegistryEntry],
    ) -> str:
        """Assemble ``F(aᵢ) + compressed R_{-i}`` for a FOCUS session.

        This is the core DACS context invariant:

        * The focused agent's full context ``F(aᵢ)`` is **never** truncated.
        * All other agents are represented by compressed registry summaries.
        * The total token count is guaranteed to be ≤ ``token_budget``.

        Compression priority when over budget:

        1. Drop ``COMPLETE``/``FAILED`` entries to a 1-line tombstone.
        2. Truncate ``LOW`` urgency entries to task description only.
        3. Truncate ``MEDIUM`` urgency entries to task + status only.
        4. ``HIGH`` urgency and ``WAITING_STEERING`` entries are never truncated.

        Parameters
        ----------
        focus:
            Full context bundle for the agent being steered.
        registry:
            Current snapshot of all registry entries (including ``focus.agent_id``).

        Raises
        ------
        ContextBudgetError
            If ``F(aᵢ)`` alone exceeds ``token_budget``.
        """
        focus_section = _serialize_focus(focus)
        focus_tokens = self.count_tokens(focus_section)
        if focus_tokens > self._budget:
            raise ContextBudgetError(
                f"F({focus.agent_id}) alone is {focus_tokens} tokens, "
                f"which exceeds the token budget of {self._budget}. "
                "Consider reducing steering history or recent_output."
            )
        others = [e for e in registry if e.agent_id != focus.agent_id]
        remaining = self._budget - focus_tokens - 50  # 50-token separator reserve
        registry_section = self._serialize_compressed_registry(others, remaining)
        result = focus_section + "\n\n" + registry_section
        token_count = self.count_tokens(result)
        assert token_count <= self._budget, (
            f"build_focus_context produced {token_count} tokens > budget {self._budget}"
        )
        self._logger.log(
            {
                "event": "CONTEXT_BUILT",
                "mode": "FOCUS",
                "agent_id": focus.agent_id,
                "token_count": token_count,
                "registry_entries": len(others),
                "steering_history_turns": len(focus.steering_history),
            }
        )
        return result

    def build_registry_context(self, registry: list[RegistryEntry]) -> str:
        """Serialize all registry entries for REGISTRY and USER_INTERACT modes."""
        lines = ["=== AGENT REGISTRY ==="]
        for e in registry:
            lines.append(
                f"[{e.agent_id}] status={e.status.value} urgency={e.urgency.value}\n"
                f"  task: {e.task_description}\n"
                f"  last_output: {e.last_output_summary or '(none)'}"
            )
        result = "\n".join(lines)
        token_count = self.count_tokens(result)
        self._logger.log(
            {
                "event": "CONTEXT_BUILT",
                "mode": "REGISTRY",
                "agent_id": None,
                "token_count": token_count,
                "registry_entries": len(registry),
                "steering_history_turns": None,
            }
        )
        return result

    def build_flat_context(
        self,
        all_focus_contexts: list[FocusContext],
        current_request: SteeringRequest,
    ) -> str:
        """Assemble the flat (baseline) context — all agents concatenated.

        This is the comparison condition: every agent's full context
        is injected simultaneously into a single context window.
        """
        header = (
            f"=== FLAT CONTEXT — steering request from {current_request.agent_id} ===\n"
        )
        sections = [header] + [_serialize_focus(fc) for fc in all_focus_contexts]
        result = "\n\n---\n\n".join(sections)
        token_count = self.count_tokens(result)
        assert token_count <= self._budget, (
            f"build_flat_context produced {token_count} tokens > budget {self._budget}"
        )
        self._logger.log(
            {
                "event": "CONTEXT_BUILT",
                "mode": "FLAT",
                "agent_id": current_request.agent_id,
                "token_count": token_count,
                "registry_entries": len(all_focus_contexts),
                "steering_history_turns": None,
            }
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _serialize_compressed_registry(
        self, entries: list[RegistryEntry], budget: int
    ) -> str:
        if not entries:
            return "=== OTHER AGENTS (none) ==="

        def line_full(e: RegistryEntry) -> str:
            if e.status in (AgentStatus.COMPLETE, AgentStatus.FAILED):
                return f"[{e.agent_id}] {e.status.value}"
            return (
                f"[{e.agent_id}] status={e.status.value} urgency={e.urgency.value} | "
                f"task: {e.task_description} | last: {e.last_output_summary}"
            )

        def line_compressed(e: RegistryEntry) -> str:
            if e.status in (AgentStatus.COMPLETE, AgentStatus.FAILED):
                return f"[{e.agent_id}] {e.status.value}"
            if e.urgency == UrgencyLevel.LOW:
                return f"[{e.agent_id}] task: {e.task_description}"
            if e.urgency == UrgencyLevel.MEDIUM:
                return f"[{e.agent_id}] status={e.status.value} | task: {e.task_description}"
            # HIGH / WAITING_STEERING — never truncate
            return line_full(e)

        header = "=== OTHER AGENTS (compressed) ==="
        full_text = header + "\n" + "\n".join(line_full(e) for e in entries)
        if self.count_tokens(full_text) <= budget:
            return full_text
        return header + "\n" + "\n".join(line_compressed(e) for e in entries)


# ---------------------------------------------------------------------------
# Focus context serializer (module-level, shared with tests)
# ---------------------------------------------------------------------------


def _serialize_focus(focus: FocusContext) -> str:
    lines = [
        f"=== FOCUS: Agent {focus.agent_id} ===",
        f"Task: {focus.task_description}",
        "",
    ]
    if focus.steering_history:
        lines.append("Steering history (last exchanges):")
        for i, pair in enumerate(focus.steering_history[-10:], 1):
            q = pair.get("request", {}).get("question", "")
            a = pair.get("response", {}).get("response_text", "")
            lines.append(f"[{i}] Q: {q}")
            lines.append(f"    A: {a}")
        lines.append("")
    if focus.recent_output:
        lines.append(f"Recent output:\n{focus.recent_output}")
        lines.append("")
    req = focus.current_request
    lines.append(f"Current request [{req.urgency.value}]:")
    lines.append(f"Context: {req.relevant_context}")
    lines.append(f"Question: {req.question}")
    return "\n".join(lines)
