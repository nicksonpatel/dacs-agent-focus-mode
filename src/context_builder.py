from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken

from src.protocols import AgentStatus, FocusContext, RegistryEntry, SteeringRequest, UrgencyLevel

if TYPE_CHECKING:
    from src.logger import Logger

# System prompt injected into every LLM call. Counts toward token budget.
SYSTEM_PROMPT = (
    "You are an orchestrator managing parallel AI agents. "
    "When responding to a steering request, provide a clear, specific decision "
    "or clarification. Reference only the agent and context provided. Be concise."
)


class ContextBudgetError(Exception):
    pass


class ContextBuilder:
    """Deterministically assembles the exact token-counted context for every
    orchestrator LLM call. Every byte produced is logged with its token count.
    This is the central experiment variable.
    """

    def __init__(self, token_budget: int, logger: Logger) -> None:
        self._budget = token_budget
        self._logger = logger
        # tiktoken cl100k_base is deterministic, fast, and matches GPT-4o-mini encoding
        self._enc = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Exposed for RegistryManager field enforcement and external use."""
        return len(self._enc.encode(text))

    def build_registry_context(self, registry: list[RegistryEntry]) -> str:
        """Serialize all registry entries. Used in REGISTRY and USER_INTERACT states."""
        lines = ["=== AGENT REGISTRY ==="]
        for e in registry:
            lines.append(
                f"[{e.agent_id}] status={e.status.value} urgency={e.urgency.value}\n"
                f"  task: {e.task_description}\n"
                f"  last_output: {e.last_output_summary or '(none)'}"
            )
        result = "\n".join(lines)
        token_count = self.count_tokens(result)
        self._logger.log({
            "event": "CONTEXT_BUILT",
            "mode": "REGISTRY",
            "agent_id": None,
            "token_count": token_count,
            "registry_entries": len(registry),
            "steering_history_turns": None,
        })
        return result

    def build_focus_context(
        self,
        focus: FocusContext,
        registry: list[RegistryEntry],
        include_registry: bool = True,
    ) -> str:
        """Assemble F(aᵢ) + compressed registry (excluding aᵢ). DACS mode only.

        If *include_registry* is False (``no_registry`` ablation), the
        compressed registry section is omitted entirely — only F(aᵢ) is
        returned.

        Compression priority when over budget:
          1. Drop COMPLETE/FAILED entries to 1-line tombstone
          2. Truncate LOW urgency entries to task_description only
          3. Truncate MEDIUM urgency entries to task+status only
          HIGH urgency + WAITING_STEERING entries are NEVER truncated.

        Raises ContextBudgetError if F(aᵢ) alone exceeds the budget.
        Asserts final count ≤ budget before returning.
        """
        focus_section = _serialize_focus(focus)
        focus_tokens = self.count_tokens(focus_section)
        if focus_tokens > self._budget:
            raise ContextBudgetError(
                f"F({focus.agent_id}) alone is {focus_tokens} tokens, exceeds budget {self._budget}"
            )

        if include_registry:
            others = [e for e in registry if e.agent_id != focus.agent_id]
            # Reserve 50 tokens for the separator between sections
            remaining = self._budget - focus_tokens - 50
            registry_section = self._serialize_compressed_registry(others, remaining)
            result = focus_section + "\n\n" + registry_section
            n_others = len(others)
        else:
            result = focus_section
            n_others = 0

        token_count = self.count_tokens(result)
        assert token_count <= self._budget, (
            f"build_focus_context produced {token_count} tokens > budget {self._budget}"
        )
        self._logger.log({
            "event": "CONTEXT_BUILT",
            "mode": "FOCUS" if include_registry else "FOCUS_NO_REG",
            "agent_id": focus.agent_id,
            "token_count": token_count,
            "registry_entries": n_others,
            "steering_history_turns": len(focus.steering_history),
        })
        return result

    def build_flat_ordered_context(
        self,
        all_focus_contexts: list[FocusContext],
        current_request: SteeringRequest,
    ) -> str:
        """Ablation: flat context with requesting agent placed first.

        Identical to ``build_flat_context`` except the requesting agent's
        context is placed at position 0 (positional bias ablation).
        """
        # Partition: requesting agent first, then all others in original order
        first = [fc for fc in all_focus_contexts if fc.agent_id == current_request.agent_id]
        rest = [fc for fc in all_focus_contexts if fc.agent_id != current_request.agent_id]
        ordered = first + rest

        header = f"=== FLAT ORDERED CONTEXT — steering request from {current_request.agent_id} ===\n"
        sections = [header] + [_serialize_focus(fc) for fc in ordered]
        result = "\n\n---\n\n".join(sections)
        token_count = self.count_tokens(result)
        assert token_count <= self._budget, (
            f"build_flat_ordered_context produced {token_count} tokens > budget {self._budget}"
        )
        self._logger.log({
            "event": "CONTEXT_BUILT",
            "mode": "FLAT_ORDERED",
            "agent_id": current_request.agent_id,
            "token_count": token_count,
            "registry_entries": len(all_focus_contexts),
            "steering_history_turns": None,
        })
        return result

    def build_flat_context(
        self,
        all_focus_contexts: list[FocusContext],
        current_request: SteeringRequest,
    ) -> str:
        """BASELINE mode only. Concatenates all agent contexts into one flat window."""
        header = f"=== FLAT CONTEXT — steering request from {current_request.agent_id} ===\n"
        sections = [header] + [_serialize_focus(fc) for fc in all_focus_contexts]
        result = "\n\n---\n\n".join(sections)
        token_count = self.count_tokens(result)
        assert token_count <= self._budget, (
            f"build_flat_context produced {token_count} tokens > budget {self._budget}"
        )
        self._logger.log({
            "event": "CONTEXT_BUILT",
            "mode": "FLAT",
            "agent_id": current_request.agent_id,
            "token_count": token_count,
            "registry_entries": len(all_focus_contexts),
            "steering_history_turns": None,
        })
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
        # Fall back to compressed representation
        return header + "\n" + "\n".join(line_compressed(e) for e in entries)


# ---------------------------------------------------------------------------
# Module-level serialization (used by both ContextBuilder and tests)
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
