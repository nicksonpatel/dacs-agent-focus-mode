"""Scenario definitions for DACS Phase 5 — Orchestrator-Initiated Focus (OIF) experiment.

Evaluates T5 (user-query routing): with OIF enabled, the orchestrator enters
Focus(aᵢ) before answering a user query whose content matches aᵢ's work.
Without OIF, it answers from the registry only.

Design constraints:
  • Same domain topics as ra1_n3 / s1_n3 for cross-phase comparability.
  • Each scenario has a fixed `user_query_schedule`: (delay_s, query, target_agent_id)
    triples.  `target_agent_id` is the ground-truth answer for the judge:
    "which agent's full context should have been entered to correctly answer this?"
  • Rubric `answer_keywords` name concrete facts that only appear in the focused
    agent's full conversation history, NOT in the ≤100-token registry summary.
    A registry-only (no-OIF) response is unlikely to contain these keywords.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OIFUserQuery:
    """One timed user query injection with OIF ground-truth labelling.

    Attributes:
        delay_s:          Seconds after trial start to fire this query.
        message:          User message text sent to the orchestrator.
        target_agent_id:  The agent whose Focus context should have been entered.
        answer_keywords:  Case-insensitive substrings that a focus-grounded response
                          should contain (≥1 required for keyword hit).
        judge_context:    What a correct, focus-grounded answer looks like —
                          injected into the LLM judge prompt.
    """
    delay_s:         float
    message:         str
    target_agent_id: str
    answer_keywords: list[str]
    judge_context:   str


@dataclass
class OIFAgentSpec:
    """Specification for one LLM agent in an OIF scenario."""
    agent_id:         str
    task_description: str
    decision_hints:   str


@dataclass
class OIFScenario:
    """Top-level OIF scenario container."""
    scenario_id:    str
    description:    str
    agents:         list[OIFAgentSpec]
    user_queries:   list[OIFUserQuery]


# ---------------------------------------------------------------------------
# oif1_n3 — 3 agents, T5 evaluation, same domains as ra1_n3
# ---------------------------------------------------------------------------
#
# a1: BST implementation
# a2: Transformer survey
# a3: CSV cleaning
#
# 3 user queries, each targeting a different agent.
# Queries fire at 8s, 16s, 24s — mid-task for all agents, giving them time
# to generate non-trivial focus context before the queries arrive.

OIF1_N3 = OIFScenario(
    scenario_id="oif1_n3",
    description=(
        "OIF Phase 5 — T5 user-query routing, 3 agents (BST / transformer survey / CSV). "
        "Three user queries fire mid-trial, each targeting a specific agent. "
        "Measures whether Focus-grounded responses contain agent-specific detail "
        "absent from the ≤100-token registry summary."
    ),
    agents=[
        OIFAgentSpec(
            agent_id="a1",
            task_description=(
                "Implement a binary search tree (BST) in Python with insert, "
                "search, and inorder traversal."
            ),
            decision_hints=(
                "- Traversal order: which traversal method should be the default.\n"
                "- Duplicate value handling: what insert should do with existing keys.\n"
                "- Implementation style: recursive vs iterative for core operations."
            ),
        ),
        OIFAgentSpec(
            agent_id="a2",
            task_description=(
                "Survey transformer attention mechanisms and key variants from 2017–2022."
            ),
            decision_hints=(
                "- Primary source: which paper is the foundational transformer reference.\n"
                "- Sparse variants: which long-context attention models to cover.\n"
                "- Conflicting claims: how to resolve disagreements between papers."
            ),
        ),
        OIFAgentSpec(
            agent_id="a3",
            task_description=(
                "Design a CSV data cleaning pipeline handling encoding, nulls, "
                "and type mismatches."
            ),
            decision_hints=(
                "- Encoding detection: how to handle unknown or mixed file encodings.\n"
                "- Null strategy: whether to impute or drop rows with missing values.\n"
                "- Type inference: strict schema vs infer-from-data approach."
            ),
        ),
    ],
    user_queries=[
        OIFUserQuery(
            delay_s=8.0,
            message=(
                "How is the BST implementation going? "
                "What traversal approach is being used and why?"
            ),
            target_agent_id="a1",
            answer_keywords=[
                "inorder", "in-order", "sorted order", "recursive", "iterative",
                "traversal", "duplicate",
            ],
            judge_context=(
                "A focus-grounded answer will reference the specific traversal design "
                "choice (inorder for sorted-order output) and the duplicate-handling "
                "decision being worked on or already advised.  A registry-only answer "
                "will be vague (e.g. 'the BST agent is running') with no implementation "
                "specifics."
            ),
        ),
        OIFUserQuery(
            delay_s=14.0,
            message=(
                "What has the transformer survey found so far? "
                "Which paper is it treating as the primary foundational reference?"
            ),
            target_agent_id="a2",
            answer_keywords=[
                "vaswani", "attention is all you need", "2017", "longformer",
                "bigbird", "reformer", "sparse",
            ],
            judge_context=(
                "A focus-grounded answer names Vaswani et al. 2017 as the foundational "
                "reference and likely mentions the sparse attention variants being "
                "surveyed (Longformer, BigBird, Reformer).  A registry-only answer "
                "will say only that the survey agent is running without paper-specific "
                "detail."
            ),
        ),
        OIFUserQuery(
            delay_s=20.0,
            message=(
                "What encoding or data quality issues has the CSV cleaning agent "
                "encountered so far?"
            ),
            target_agent_id="a3",
            answer_keywords=[
                "encoding", "utf", "null", "missing", "impute", "drop", "schema",
                "type", "inference",
            ],
            judge_context=(
                "A focus-grounded answer describes the specific encoding strategy "
                "(e.g. UTF-8 detection, chardet) and the null-handling decision "
                "(impute vs drop) being worked on.  A registry-only answer provides "
                "only a generic status update."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# oif2_n5 — 5 agents, T5 routing at N=5
# ---------------------------------------------------------------------------
#
# Adds a4 (debugger) and a5 (long-form writer) to oif1_n3 to test routing
# accuracy at N=5.  User queries include one per new agent plus two
# cross-agent decoys to test whether the orchestrator avoids mis-routing.

OIF2_N5 = OIFScenario(
    scenario_id="oif2_n5",
    description=(
        "OIF Phase 5 — T5 user-query routing, 5 agents (BST / transformer / CSV / "
        "debugger / writer).  5 user queries, each targeting a specific agent.  "
        "Tests routing accuracy and discrimination at N=5."
    ),
    agents=[
        OIFAgentSpec(
            agent_id="a1",
            task_description=(
                "Implement a binary search tree (BST) in Python with insert, "
                "search, and inorder traversal."
            ),
            decision_hints=(
                "- Traversal order: which traversal method should be the default.\n"
                "- Duplicate value handling: what insert should do with existing keys.\n"
                "- Implementation style: recursive vs iterative for core operations."
            ),
        ),
        OIFAgentSpec(
            agent_id="a2",
            task_description=(
                "Survey transformer attention mechanisms and key variants from 2017–2022."
            ),
            decision_hints=(
                "- Primary source: which paper is the foundational transformer reference.\n"
                "- Sparse variants: which long-context attention models to cover.\n"
                "- Conflicting claims: how to resolve disagreements between papers."
            ),
        ),
        OIFAgentSpec(
            agent_id="a3",
            task_description=(
                "Design a CSV data cleaning pipeline handling encoding, nulls, "
                "and type mismatches."
            ),
            decision_hints=(
                "- Encoding detection: how to handle unknown or mixed file encodings.\n"
                "- Null strategy: whether to impute or drop rows with missing values.\n"
                "- Type inference: strict schema vs infer-from-data approach."
            ),
        ),
        OIFAgentSpec(
            agent_id="a4",
            task_description=(
                "Debug a Python application that intermittently raises KeyError "
                "in a dictionary-based cache."
            ),
            decision_hints=(
                "- Root cause strategy: systematic bisection vs trace-driven analysis.\n"
                "- Fix approach: defensive get() with default vs explicit key check.\n"
                "- Test coverage: unit test vs integration test for the cache layer."
            ),
        ),
        OIFAgentSpec(
            agent_id="a5",
            task_description=(
                "Write a 1000-word technical blog post explaining transformer "
                "self-attention to a software engineering audience."
            ),
            decision_hints=(
                "- Audience calibration: assume ML background or pure SWE background.\n"
                "- Analogy choice: database joins, attention heads as parallel queries.\n"
                "- Code example: include a NumPy snippet or keep prose-only."
            ),
        ),
    ],
    user_queries=[
        OIFUserQuery(
            delay_s=7.0,
            message="How is the BST implementation progressing? What's the traversal decision?",
            target_agent_id="a1",
            answer_keywords=["inorder", "in-order", "traversal", "recursive", "sorted"],
            judge_context=(
                "Focus-grounded: mentions specific traversal choice (inorder) "
                "and implementation style (recursive).  Registry-only: vague status."
            ),
        ),
        OIFUserQuery(
            delay_s=13.0,
            message="What foundational paper is the transformer survey using as its primary reference?",
            target_agent_id="a2",
            answer_keywords=["vaswani", "2017", "attention is all you need"],
            judge_context=(
                "Focus-grounded: names Vaswani et al. 2017.  "
                "Registry-only: says only that the survey is running."
            ),
        ),
        OIFUserQuery(
            delay_s=19.0,
            message="Has the CSV cleaning agent decided how to handle null values?",
            target_agent_id="a3",
            answer_keywords=["null", "missing", "impute", "drop", "nulls"],
            judge_context=(
                "Focus-grounded: describes the null-handling decision.  "
                "Registry-only: generic update."
            ),
        ),
        OIFUserQuery(
            delay_s=25.0,
            message="What debugging strategy is the cache debugger using?",
            target_agent_id="a4",
            answer_keywords=["bisect", "trace", "get(", "key", "cache", "keyerror"],
            judge_context=(
                "Focus-grounded: describes bisection vs trace-driven choice and "
                "the fix approach (defensive get vs explicit check).  "
                "Registry-only: vague update."
            ),
        ),
        OIFUserQuery(
            delay_s=31.0,
            message="Is the blog post targeting ML practitioners or pure software engineers?",
            target_agent_id="a5",
            answer_keywords=["software engineer", "swe", "swe background", "ml", "audience"],
            judge_context=(
                "Focus-grounded: names the audience calibration decision made (SWE "
                "vs ML).  Registry-only: says only that the blog post is in progress."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OIF_SCENARIOS: dict[str, OIFScenario] = {
    "oif1_n3": OIF1_N3,
    "oif2_n5": OIF2_N5,
}
