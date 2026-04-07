"""Scenario definitions for the DACS real-agent validation experiment.

``ra1_n3`` mirrors the domain topics of synthetic scenario ``s1_n3`` so that
real-agent vs synthetic results are directly comparable in the paper.

Key design constraint: ``decision_hints`` names the *type* of decision without
prescribing the correct answer.  The LLM agent must reason its own way to a
question.  Only ``DecisionRubric.correct_keywords`` and ``judge_context`` (used
exclusively by the offline LLM judge) name the expected answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecisionRubric:
    """Ground-truth rubric for one expected decision point.

    Attributes:
        topic:            Short label used in analysis tables.
        correct_keywords: Case-insensitive substrings; ≥1 must appear in the
                          orchestrator response for the keyword score to be 1.
        judge_context:    Plain-text description of what a *correct* orchestrator
                          response looks like — injected into the LLM judge prompt
                          so the judge can evaluate semantic correctness beyond
                          keyword matching.
    """

    topic: str
    correct_keywords: list[str]
    judge_context: str


@dataclass
class RealAgentSpec:
    """Specification for one real LLM agent in a scenario.

    Attributes:
        agent_id:         Unique identifier used throughout logs ("a1", "a2", …).
        task_description: ≤50-token task written into both the system prompt and
                          the registry entry.
        decision_hints:   Paragraph naming decision *types* the agent should
                          consult the orchestrator on.  No correct answers here.
        rubrics:          Ordered list of expected decision points.  The judge
                          assigns orchestrator responses to rubrics by sequence
                          order within each agent.
    """

    agent_id: str
    task_description: str
    decision_hints: str
    rubrics: list[DecisionRubric] = field(default_factory=list)


@dataclass
class RealAgentScenario:
    """Top-level scenario container."""

    scenario_id: str
    description: str
    agents: list[RealAgentSpec]


# ---------------------------------------------------------------------------
# ra1_n3 — 3 agents, same domains as synthetic s1_n3
# ---------------------------------------------------------------------------
#
# a1: BST implementation  (mirrors s1_n3/a1  CodeWriterAgent)
# a2: Transformer survey  (mirrors s1_n3/a2  ResearchAgent)
# a3: CSV cleaning        (mirrors s1_n3/a3  DataProcessorAgent)
#
# Rubric correct_keywords intentionally overlap with s1_n3 answer_keywords so
# the comparison table in the paper is valid.

RA1_N3 = RealAgentScenario(
    scenario_id="ra1_n3",
    description=(
        "Real-agent validation: 3 LLM agents, same task domains as s1_n3 "
        "(BST / transformer survey / CSV cleaning). "
        "Agents generate their own steering questions autonomously."
    ),
    agents=[
        # ------------------------------------------------------------------
        # a1 — Binary search tree implementation
        # ------------------------------------------------------------------
        RealAgentSpec(
            agent_id="a1",
            task_description=(
                "Implement a binary search tree (BST) in Python with insert, "
                "search, and traversal operations."
            ),
            decision_hints=(
                "- Traversal order: which traversal method (inorder, preorder, "
                "postorder) should be the primary default traversal, and why.\n"
                "- Duplicate value handling: what the insert operation should do "
                "when a value already exists in the tree.\n"
                "- Implementation style: whether recursive or iterative algorithms "
                "are preferable for the core operations."
            ),
            rubrics=[
                DecisionRubric(
                    topic="traversal_order",
                    correct_keywords=["inorder", "in-order", "in order", "sorted order"],
                    judge_context=(
                        "For a BST, inorder traversal (left → root → right) "
                        "visits nodes in ascending sorted order, making it the "
                        "most semantically useful default.  A correct response "
                        "recommends inorder traversal, ideally explaining the "
                        "sorted-order property."
                    ),
                ),
                DecisionRubric(
                    topic="duplicate_handling",
                    correct_keywords=["ignore", "reject", "skip", "discard", "duplicate"],
                    judge_context=(
                        "The standard BST contract treats each key as unique.  "
                        "A correct response recommends ignoring (silently rejecting) "
                        "duplicate inserts, rather than storing duplicates or raising "
                        "an error, though any principled choice with reasoning is acceptable."
                    ),
                ),
                DecisionRubric(
                    topic="implementation_style",
                    correct_keywords=["recursive", "recursion", "recurse"],
                    judge_context=(
                        "Recursive implementations are conventional for BST "
                        "operations because the code mirrors the recursive structure "
                        "of the tree.  A correct response recommends recursive "
                        "implementations (with an optional caveat about stack depth "
                        "for very large trees)."
                    ),
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # a2 — Transformer architecture literature survey
        # ------------------------------------------------------------------
        RealAgentSpec(
            agent_id="a2",
            task_description=(
                "Survey the transformer architecture literature focusing on "
                "attention mechanisms and key variants from 2017–2022."
            ),
            decision_hints=(
                "- Primary source selection: which specific paper should be cited "
                "as the foundational reference for the transformer / self-attention "
                "mechanism.\n"
                "- Sparse attention variant coverage: which specific long-context "
                "or efficient-attention variants to include in the survey.\n"
                "- Citation depth: how to resolve conflicting empirical claims "
                "across papers (e.g. which source to treat as authoritative)."
            ),
            rubrics=[
                DecisionRubric(
                    topic="primary_source",
                    correct_keywords=[
                        "vaswani", "2017", "attention is all you need", "original transformer",
                    ],
                    judge_context=(
                        "Vaswani et al. 2017 ('Attention Is All You Need') is the "
                        "definitive primary source for the transformer.  A correct "
                        "response names Vaswani et al. and/or the year 2017 as the "
                        "foundational reference."
                    ),
                ),
                DecisionRubric(
                    topic="sparse_attention_variants",
                    correct_keywords=[
                        "longformer", "bigbird", "big bird", "reformer", "sparse",
                    ],
                    judge_context=(
                        "The canonical sparse/efficient attention variants from this "
                        "period are Longformer (Beltagy et al. 2020), BigBird (Zaheer "
                        "et al. 2020), and Reformer (Kitaev et al. 2020).  A correct "
                        "response mentions at least one of these by name."
                    ),
                ),
                DecisionRubric(
                    topic="citation_depth",
                    correct_keywords=[
                        "deep", "methodology", "primary", "authoritative", "original",
                        "original paper", "first-hand",
                    ],
                    judge_context=(
                        "When empirical claims conflict, the correct guidance is to "
                        "defer to the original/primary paper for its own method's "
                        "performance, and to use deep methodological scrutiny "
                        "(checking experimental conditions) for cross-paper comparisons.  "
                        "A correct response advises consulting primary sources and/or "
                        "careful methodological comparison."
                    ),
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # a3 — CSV data cleaning pipeline
        # ------------------------------------------------------------------
        RealAgentSpec(
            agent_id="a3",
            task_description=(
                "Design a CSV data cleaning pipeline to handle mixed-encoding "
                "files, null values, and outliers."
            ),
            decision_hints=(
                "- Character encoding strategy: how to handle files that mix "
                "UTF-8 and Latin-1 (ISO-8859-1) encodings in the same dataset.\n"
                "- Null value imputation: which statistical method to use when "
                "filling missing values in numeric columns.\n"
                "- Outlier detection threshold: which method or numerical threshold "
                "to use for identifying and handling outliers in numeric data."
            ),
            rubrics=[
                DecisionRubric(
                    topic="encoding_strategy",
                    correct_keywords=[
                        "coerce", "utf-8", "utf8", "latin", "chardet",
                        "detect", "errors='replace'", "errors='ignore'",
                    ],
                    judge_context=(
                        "For mixed UTF-8/Latin-1 data, the correct approach is to "
                        "detect encoding per-file (e.g. via chardet) or to read with "
                        "UTF-8 and set errors='replace'/'ignore', or to coerce to "
                        "UTF-8.  A correct response recommends explicit encoding "
                        "detection or coercion rather than assuming a single encoding."
                    ),
                ),
                DecisionRubric(
                    topic="null_imputation",
                    correct_keywords=["mode", "fill", "impute", "median", "mean"],
                    judge_context=(
                        "For categorical columns, mode imputation is standard; for "
                        "numeric columns, median is robust to skew.  A correct "
                        "response recommends mode, median, or mean imputation "
                        "(any of these is defensible with brief justification)."
                    ),
                ),
                DecisionRubric(
                    topic="outlier_threshold",
                    correct_keywords=[
                        "clip", "sigma", "iqr", "z-score", "zscore",
                        "3-sigma", "3 sigma", "1.5", "percentile",
                    ],
                    judge_context=(
                        "Standard outlier thresholds are: 3-sigma (z-score > 3), "
                        "IQR-based (1.5× or 3× IQR), or percentile clipping (e.g. "
                        "1st–99th percentile).  A correct response names one of these "
                        "approaches explicitly."
                    ),
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# ra2_n5 — 5 agents, same domains as synthetic s2_n5
# ---------------------------------------------------------------------------
#
# a1–a3: identical task domains to ra1_n3 (BST / transformer survey / CSV)
# a4: graph algorithm (BFS/DFS + cycle detection) — mirrors s2_n5/a4
# a5: RL policy gradient survey    — mirrors s2_n5/a5
#
# max_steering_requests set to 3 per agent → up to 15 steering events per trial.

RA2_N5 = RealAgentScenario(
    scenario_id="ra2_n5",
    description=(
        "Real-agent validation: 5 LLM agents, same task domains as s2_n5 "
        "(BST / transformer survey / CSV cleaning / graph algorithm / RL survey). "
        "Tests whether the DACS N-scaling advantage replicates with real agents."
    ),
    agents=[
        # ------------------------------------------------------------------
        # a1 — Binary search tree (identical to ra1_n3/a1)
        # ------------------------------------------------------------------
        RealAgentSpec(
            agent_id="a1",
            task_description=(
                "Implement a binary search tree (BST) in Python with insert, "
                "search, and traversal operations."
            ),
            decision_hints=(
                "1. Traversal order: which traversal method (inorder, preorder, "
                "postorder) should be the primary default traversal, and why.\n"
                "2. Duplicate value handling: what the insert operation should do "
                "when a value already exists in the tree.\n"
                "3. Implementation style: whether recursive or iterative algorithms "
                "are preferable for the core operations."
            ),
            rubrics=[
                DecisionRubric(
                    topic="traversal_order",
                    correct_keywords=["inorder", "in-order", "in order", "sorted order"],
                    judge_context=(
                        "For a BST, inorder traversal visits nodes in ascending "
                        "sorted order, making it the most useful default. A correct "
                        "response recommends inorder traversal."
                    ),
                ),
                DecisionRubric(
                    topic="duplicate_handling",
                    correct_keywords=["ignore", "reject", "skip", "discard", "duplicate"],
                    judge_context=(
                        "The standard BST contract treats each key as unique. "
                        "A correct response recommends ignoring or silently rejecting "
                        "duplicate inserts."
                    ),
                ),
                DecisionRubric(
                    topic="implementation_style",
                    correct_keywords=["recursive", "recursion", "recurse"],
                    judge_context=(
                        "Recursive implementations are conventional for BST operations "
                        "because the code mirrors the tree's recursive structure. "
                        "A correct response recommends recursive implementations."
                    ),
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # a2 — Transformer architecture survey (identical to ra1_n3/a2)
        # ------------------------------------------------------------------
        RealAgentSpec(
            agent_id="a2",
            task_description=(
                "Survey the transformer architecture literature focusing on "
                "attention mechanisms and key variants from 2017–2022."
            ),
            decision_hints=(
                "1. Primary source selection: which specific paper should be cited "
                "as the foundational reference for the transformer/self-attention "
                "mechanism.\n"
                "2. Sparse attention variant coverage: which specific long-context "
                "or efficient-attention variants to include in the survey.\n"
                "3. Citation depth: how to resolve conflicting empirical claims "
                "across papers (which source to treat as authoritative)."
            ),
            rubrics=[
                DecisionRubric(
                    topic="primary_source",
                    correct_keywords=["vaswani", "2017", "attention is all you need"],
                    judge_context=(
                        "Vaswani et al. 2017 ('Attention Is All You Need') is the "
                        "definitive primary source. A correct response names Vaswani "
                        "et al. and/or 2017 as the foundational reference."
                    ),
                ),
                DecisionRubric(
                    topic="sparse_attention_variants",
                    correct_keywords=[
                        "longformer", "bigbird", "big bird", "reformer", "sparse",
                    ],
                    judge_context=(
                        "The canonical sparse/efficient attention variants are "
                        "Longformer, BigBird, and Reformer. A correct response "
                        "mentions at least one of these by name."
                    ),
                ),
                DecisionRubric(
                    topic="citation_depth",
                    correct_keywords=[
                        "deep", "methodology", "primary", "authoritative",
                        "original", "first-hand",
                    ],
                    judge_context=(
                        "A correct response advises deferring to original/primary "
                        "papers for their own method's results, and using careful "
                        "methodological comparison for cross-paper claims."
                    ),
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # a3 — CSV data cleaning pipeline (identical to ra1_n3/a3)
        # ------------------------------------------------------------------
        RealAgentSpec(
            agent_id="a3",
            task_description=(
                "Design a CSV data cleaning pipeline to handle mixed-encoding "
                "files, null values, and outliers."
            ),
            decision_hints=(
                "1. Character encoding strategy: how to handle files that mix "
                "UTF-8 and Latin-1 (ISO-8859-1) encodings in the same dataset.\n"
                "2. Null value imputation: which statistical method to use when "
                "filling missing values in numeric columns.\n"
                "3. Outlier detection threshold: which method or numerical threshold "
                "to use for identifying and handling outliers."
            ),
            rubrics=[
                DecisionRubric(
                    topic="encoding_strategy",
                    correct_keywords=[
                        "coerce", "utf-8", "utf8", "latin", "chardet",
                        "detect", "errors='replace'", "errors='ignore'",
                    ],
                    judge_context=(
                        "For mixed UTF-8/Latin-1 data, a correct response recommends "
                        "encoding detection (e.g. chardet) or coercion to UTF-8 with "
                        "errors='replace'/'ignore', rather than assuming a single encoding."
                    ),
                ),
                DecisionRubric(
                    topic="null_imputation",
                    correct_keywords=["mode", "fill", "impute", "median", "mean"],
                    judge_context=(
                        "A correct response recommends mode (categorical) or median/"
                        "mean (numeric) imputation with brief justification."
                    ),
                ),
                DecisionRubric(
                    topic="outlier_threshold",
                    correct_keywords=[
                        "clip", "sigma", "iqr", "z-score", "zscore",
                        "3-sigma", "3 sigma", "1.5", "percentile",
                    ],
                    judge_context=(
                        "Standard thresholds: 3-sigma (z-score > 3), IQR-based "
                        "(1.5× or 3× IQR), or percentile clipping. A correct "
                        "response names one explicitly."
                    ),
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # a4 — Graph algorithm: BFS/DFS with cycle detection
        # ------------------------------------------------------------------
        RealAgentSpec(
            agent_id="a4",
            task_description=(
                "Implement graph BFS and DFS with cycle detection in Python "
                "using an adjacency list representation."
            ),
            decision_hints=(
                "1. Graph representation: whether to use an adjacency list or "
                "adjacency matrix as the underlying data structure, and why.\n"
                "2. Visited node tracking: which data structure to use for tracking "
                "visited nodes in BFS and DFS (boolean array vs hash set).\n"
                "3. Cycle detection strategy: which algorithm to use for detecting "
                "cycles in a directed graph (three-color DFS, parent-tracking, "
                "or union-find)."
            ),
            rubrics=[
                DecisionRubric(
                    topic="graph_representation",
                    correct_keywords=[
                        "adjacency list", "list", "sparse", "dict", "defaultdict",
                    ],
                    judge_context=(
                        "For sparse graphs, adjacency list is preferred (O(V+E) space "
                        "vs O(V²) for matrix, and faster iteration). A correct response "
                        "recommends adjacency list, especially for sparse graphs."
                    ),
                ),
                DecisionRubric(
                    topic="visited_tracking",
                    correct_keywords=[
                        "hash set", "set", "dict", "boolean array", "visited set",
                    ],
                    judge_context=(
                        "A hash set (Python set) is preferred when vertex IDs may be "
                        "non-contiguous strings. A correct response recommends a set "
                        "or dict for visited tracking, noting string-ID flexibility."
                    ),
                ),
                DecisionRubric(
                    topic="cycle_detection",
                    correct_keywords=[
                        "three-color", "3-color", "coloring", "white", "gray", "black",
                        "three color", "dfs color",
                    ],
                    judge_context=(
                        "Three-color DFS (white/gray/black node states) is the clean "
                        "standard for directed cycle detection. A correct response "
                        "recommends three-color (or equivalent state-tracking) DFS."
                    ),
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # a5 — Reinforcement learning policy gradient survey
        # ------------------------------------------------------------------
        RealAgentSpec(
            agent_id="a5",
            task_description=(
                "Write a survey on reinforcement learning policy gradient methods, "
                "covering REINFORCE, actor-critic approaches, and modern on-policy "
                "algorithms."
            ),
            decision_hints=(
                "1. Variance reduction framing: whether to present REINFORCE baseline "
                "subtraction or actor-critic methods as the primary variance reduction "
                "technique in the survey.\n"
                "2. Canonical on-policy algorithms: which specific modern on-policy "
                "algorithms to feature as the primary examples (e.g. PPO, TRPO, A3C).\n"
                "3. Evaluation benchmarks: which benchmark environments to use as "
                "primary evaluation platforms in the survey (e.g. Atari, MuJoCo, "
                "continuous control suites)."
            ),
            rubrics=[
                DecisionRubric(
                    topic="variance_reduction",
                    correct_keywords=[
                        "baseline", "actor-critic", "advantage", "critic",
                        "value function", "advantage function",
                    ],
                    judge_context=(
                        "A correct response recommends actor-critic methods (advantage "
                        "function / value baseline) as the principled primary framing, "
                        "mentioning that REINFORCE baseline is a simpler precursor. "
                        "Any response that names advantage estimation or actor-critic "
                        "as the key technique is correct."
                    ),
                ),
                DecisionRubric(
                    topic="canonical_algorithms",
                    correct_keywords=[
                        "ppo", "proximal policy", "trpo", "trust region",
                        "proximal policy optimization",
                    ],
                    judge_context=(
                        "PPO (Proximal Policy Optimization) and/or TRPO (Trust Region "
                        "Policy Optimization) are the canonical on-policy algorithms. "
                        "A correct response names PPO and/or TRPO as primary examples."
                    ),
                ),
                DecisionRubric(
                    topic="evaluation_benchmarks",
                    correct_keywords=[
                        "mujoco", "atari", "continuous control", "both",
                        "gym", "gymnax", "dm control",
                    ],
                    judge_context=(
                        "Both Atari (discrete, pixel-based) and MuJoCo/continuous "
                        "control (continuous action spaces) are standard. A correct "
                        "response recommends both benchmark families or a continuous "
                        "control suite."
                    ),
                ),
            ],
        ),
    ],
)

# Registry of all real-agent scenarios
REAL_SCENARIOS: dict[str, RealAgentScenario] = {
    "ra1_n3": RA1_N3,
    "ra2_n5": RA2_N5,
}
