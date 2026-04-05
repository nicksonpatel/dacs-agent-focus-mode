"""Task suite for DACS experiments.

Each scenario has N agents, each with exactly 3 pre-defined decision points.
The ground-truth answer at each decision point is known in advance.
Steering accuracy is measured as: orchestrator response ⊇ at least one of the
answer_keywords for that decision point.

Structure
---------
SCENARIOS: dict[str, ScenarioSpec]
  Each ScenarioSpec contains a list of AgentSpec objects.
  Each AgentSpec contains:
    - agent_type: which stub class to instantiate
    - task_description: ≤50 tokens (enforced by RegistryManager)
    - decision_points: list of DecisionPoint (one per steering request)

A DecisionPoint maps a question snippet to a set of acceptable answer keywords.
The experiment harness matches these against the actual LLM response text (case-insensitive).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecisionPoint:
    question_fragment: str      # unique substring of the question (for matching)
    answer_keywords: list[str]  # ≥1 keyword must appear in the response to score correct


@dataclass
class AgentSpec:
    agent_id: str
    agent_type: str                              # "code_writer" | "research" | "data_processor" | "generic"
    task_description: str                        # ≤50 tokens
    decision_points: list[DecisionPoint] = field(default_factory=list)
    steps: list[dict] | None = None              # GenericAgent step defs (required when agent_type="generic")


@dataclass
class ScenarioSpec:
    scenario_id: str
    agents: list[AgentSpec]


# ---------------------------------------------------------------------------
# Ground-truth answers per decision point
# (agent stubs ask these exact questions; keywords match correct responses)
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, ScenarioSpec] = {
    # ------------------------------------------------------------------
    # Scenario 1: The canonical 3-agent scenario (N=3)
    # ------------------------------------------------------------------
    "s1_n3": ScenarioSpec(
        scenario_id="s1_n3",
        agents=[
            AgentSpec(
                agent_id="a1",
                agent_type="code_writer",
                task_description="implement binary search tree with insert delete to_list",
                decision_points=[
                    DecisionPoint(
                        question_fragment="iterative or recursive",
                        answer_keywords=["recursive"],
                    ),
                    DecisionPoint(
                        question_fragment="duplicate values",
                        answer_keywords=["ignore"],
                    ),
                    DecisionPoint(
                        question_fragment="traversal order",
                        answer_keywords=["inorder", "in-order", "in order"],
                    ),
                ],
            ),
            AgentSpec(
                agent_id="a2",
                agent_type="research",
                task_description="write survey on transformer attention mechanisms",
                decision_points=[
                    DecisionPoint(
                        question_fragment="conflicting claims on attention complexity",
                        answer_keywords=["vaswani", "2017", "authoritative"],
                    ),
                    DecisionPoint(
                        question_fragment="sparse attention variants",
                        answer_keywords=["longformer", "bigbird", "reformer"],
                    ),
                    DecisionPoint(
                        question_fragment="citation depth",
                        answer_keywords=["deep", "methodology"],
                    ),
                ],
            ),
            AgentSpec(
                agent_id="a3",
                agent_type="data_processor",
                task_description="process and clean sales CSV data for reporting",
                decision_points=[
                    DecisionPoint(
                        question_fragment="mixed UTF-8/Latin-1 encoding",
                        answer_keywords=["coerce", "utf-8", "utf8"],
                    ),
                    DecisionPoint(
                        question_fragment="null values",
                        answer_keywords=["mode", "fill"],
                    ),
                    DecisionPoint(
                        question_fragment="outliers",
                        answer_keywords=["clip", "3-sigma", "3 sigma", "sigma"],
                    ),
                ],
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # Scenario 2: N=5 — diverse domains to stress cross-contamination
    # ------------------------------------------------------------------
    "s2_n5": ScenarioSpec(
        scenario_id="s2_n5",
        agents=[
            # a1–a3: same as s1_n3 (BST, attention survey, CSV pipeline)
            AgentSpec(
                agent_id="a1",
                agent_type="code_writer",
                task_description="implement binary search tree with insert delete to_list",
                decision_points=[
                    DecisionPoint("iterative or recursive", ["recursive"]),
                    DecisionPoint("duplicate values", ["ignore"]),
                    DecisionPoint("traversal order", ["inorder", "in-order", "in order"]),
                ],
            ),
            AgentSpec(
                agent_id="a2",
                agent_type="research",
                task_description="write survey on transformer attention mechanisms",
                decision_points=[
                    DecisionPoint("conflicting claims on attention complexity", ["vaswani", "2017", "authoritative"]),
                    DecisionPoint("sparse attention variants", ["longformer", "bigbird", "reformer"]),
                    DecisionPoint("citation depth", ["deep", "methodology"]),
                ],
            ),
            AgentSpec(
                agent_id="a3",
                agent_type="data_processor",
                task_description="process and clean sales CSV data for reporting",
                decision_points=[
                    DecisionPoint("mixed UTF-8/Latin-1 encoding", ["coerce", "utf-8", "utf8"]),
                    DecisionPoint("null values", ["mode", "fill"]),
                    DecisionPoint("outliers", ["clip", "3-sigma", "3 sigma", "sigma"]),
                ],
            ),
            # a4: graph algorithm implementation (GenericAgent)
            AgentSpec(
                agent_id="a4",
                agent_type="generic",
                task_description="implement graph BFS DFS with cycle detection using adjacency list",
                decision_points=[
                    DecisionPoint(
                        "adjacency list or adjacency matrix",
                        ["adjacency list", "list", "sparse"],
                    ),
                    DecisionPoint(
                        "tracking visited nodes in BFS",
                        ["hash set", "set", "boolean array"],
                    ),
                    DecisionPoint(
                        "cycle detection strategy for directed graph",
                        ["three-color", "3-color", "coloring", "union-find", "dfs coloring"],
                    ),
                ],
                steps=[
                    {"summary": "Designing graph data structure, evaluating representation options", "urgency": "LOW"},
                    {
                        "summary": "Considering adjacency list vs matrix tradeoffs for BFS/DFS",
                        "urgency": "MEDIUM",
                        "question": (
                            "Should this graph use adjacency list or adjacency matrix? "
                            "Adjacency list is O(V+E) space and faster for sparse graphs; "
                            "matrix is O(V²) but O(1) edge lookup. This graph is expected to be sparse."
                        ),
                    },
                    {"summary": "Implementing BFS with queue, writing visited-tracking logic", "urgency": "LOW"},
                    {
                        "summary": "BFS functional but need to decide visited-state data structure for DFS",
                        "urgency": "HIGH",
                        "question": (
                            "For tracking visited nodes in BFS and DFS: use a boolean array indexed by vertex id, "
                            "or a hash set to handle non-contiguous ids? "
                            "Vertices may have string ids in some use cases."
                        ),
                    },
                    {"summary": "DFS implemented, beginning cycle detection phase", "urgency": "LOW"},
                    {
                        "summary": "Need to decide cycle detection strategy for directed graph",
                        "urgency": "MEDIUM",
                        "question": (
                            "For cycle detection strategy for directed graph: "
                            "use three-color DFS (white/gray/black), parent-tracking DFS, or union-find? "
                            "Three-color DFS handles directed cycles cleanly."
                        ),
                    },
                    {"summary": "Cycle detection complete; writing tests", "urgency": "LOW"},
                ],
            ),
            # a5: RL policy gradient survey (GenericAgent)
            AgentSpec(
                agent_id="a5",
                agent_type="generic",
                task_description="write survey on reinforcement learning policy gradient methods",
                decision_points=[
                    DecisionPoint(
                        "REINFORCE baseline vs actor-critic for variance reduction in RL survey",
                        ["baseline", "actor-critic", "advantage"],
                    ),
                    DecisionPoint(
                        "canonical on-policy algorithms to feature",
                        ["ppo", "proximal policy", "trpo"],
                    ),
                    DecisionPoint(
                        "evaluation benchmarks for RL policy gradient survey",
                        ["mujoco", "atari", "continuous control", "both"],
                    ),
                ],
                steps=[
                    {"summary": "Reviewing REINFORCE algorithm and variance reduction literature", "urgency": "LOW"},
                    {
                        "summary": "Conflicting framings found for variance reduction in RL survey",
                        "urgency": "HIGH",
                        "question": (
                            "For variance reduction section of this RL survey: should the survey "
                            "advocate REINFORCE baseline subtraction as the primary technique, or "
                            "frame actor-critic methods as the principled solution? "
                            "REINFORCE baseline vs actor-critic for variance reduction in RL survey."
                        ),
                    },
                    {"summary": "Drafting on-policy vs off-policy taxonomy section", "urgency": "LOW"},
                    {
                        "summary": "Need to choose which canonical on-policy algorithms to feature",
                        "urgency": "MEDIUM",
                        "question": (
                            "Which algorithms to feature as the canonical on-policy algorithms to feature "
                            "in the survey's core section? Options: PPO (proximal policy optimization), "
                            "TRPO (trust region), or A3C (asynchronous advantage actor-critic)? "
                            "PPO has dominated practical adoption."
                        ),
                    },
                    {"summary": "Algorithm coverage complete, planning empirical section", "urgency": "LOW"},
                    {
                        "summary": "Deciding empirical scope for evaluation section",
                        "urgency": "MEDIUM",
                        "question": (
                            "For the evaluation benchmarks for RL policy gradient survey: "
                            "focus on MuJoCo continuous control tasks, Atari discrete action spaces, or both? "
                            "MuJoCo is the standard for policy gradient methods but Atari provides DQN baseline comparisons."
                        ),
                    },
                    {"summary": "Survey draft complete; writing conclusions", "urgency": "LOW"},
                ],
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # Scenario 3: N=10 — maximum diversity, 10 distinct agent domains
    # ------------------------------------------------------------------
    "s3_n10": ScenarioSpec(
        scenario_id="s3_n10",
        agents=[
            # a1–a3: same as s1_n3
            AgentSpec(
                agent_id="a1",
                agent_type="code_writer",
                task_description="implement binary search tree with insert delete to_list",
                decision_points=[
                    DecisionPoint("iterative or recursive", ["recursive"]),
                    DecisionPoint("duplicate values", ["ignore"]),
                    DecisionPoint("traversal order", ["inorder", "in-order", "in order"]),
                ],
            ),
            AgentSpec(
                agent_id="a2",
                agent_type="research",
                task_description="write survey on transformer attention mechanisms",
                decision_points=[
                    DecisionPoint("conflicting claims on attention complexity", ["vaswani", "2017", "authoritative"]),
                    DecisionPoint("sparse attention variants", ["longformer", "bigbird", "reformer"]),
                    DecisionPoint("citation depth", ["deep", "methodology"]),
                ],
            ),
            AgentSpec(
                agent_id="a3",
                agent_type="data_processor",
                task_description="process and clean sales CSV data for reporting",
                decision_points=[
                    DecisionPoint("mixed UTF-8/Latin-1 encoding", ["coerce", "utf-8", "utf8"]),
                    DecisionPoint("null values", ["mode", "fill"]),
                    DecisionPoint("outliers", ["clip", "3-sigma", "3 sigma", "sigma"]),
                ],
            ),
            # a4: graph algorithm (GenericAgent) — same as s2_n5 a4
            AgentSpec(
                agent_id="a4",
                agent_type="generic",
                task_description="implement graph BFS DFS with cycle detection using adjacency list",
                decision_points=[
                    DecisionPoint("adjacency list or adjacency matrix", ["adjacency list", "list", "sparse"]),
                    DecisionPoint("tracking visited nodes in BFS", ["hash set", "set", "boolean array"]),
                    DecisionPoint("cycle detection strategy for directed graph", ["three-color", "3-color", "coloring", "union-find"]),
                ],
                steps=[
                    {"summary": "Evaluating graph representation options for BFS and DFS implementation", "urgency": "LOW"},
                    {
                        "summary": "Choosing between adjacency list and matrix representation",
                        "urgency": "MEDIUM",
                        "question": "Should this graph use adjacency list or adjacency matrix? Adjacency list is O(V+E) space for sparse graphs; matrix is O(V²). Graph is expected to be sparse.",
                    },
                    {"summary": "Implementing BFS queue loop, designing visited tracking", "urgency": "LOW"},
                    {
                        "summary": "Visited data structure decision needed for tracking visited nodes in BFS",
                        "urgency": "HIGH",
                        "question": "For tracking visited nodes in BFS and DFS: boolean array indexed by integer vertex id, or hash set to support non-contiguous string ids?",
                    },
                    {"summary": "BFS complete; implementing DFS for cycle detection", "urgency": "LOW"},
                    {
                        "summary": "Choosing cycle detection strategy for directed graph",
                        "urgency": "MEDIUM",
                        "question": "For cycle detection strategy for directed graph: three-color DFS (white/gray/black), parent-tracking recursive DFS, or union-find?",
                    },
                    {"summary": "Implementation complete; adding test coverage", "urgency": "LOW"},
                ],
            ),
            # a5: federated learning survey (GenericAgent)
            AgentSpec(
                agent_id="a5",
                agent_type="generic",
                task_description="write survey on federated learning privacy and communication efficiency",
                decision_points=[
                    DecisionPoint(
                        "differential privacy vs secure aggregation for federated learning privacy",
                        ["differential privacy", "dp", "differential"],
                    ),
                    DecisionPoint(
                        "communication efficiency methods for federated learning",
                        ["compression", "fedavg", "federated averaging", "gradient compression"],
                    ),
                    DecisionPoint(
                        "non-iid data heterogeneity handling in federated learning",
                        ["personalization", "fine-tuning", "local adaptation", "personalized"],
                    ),
                ],
                steps=[
                    {"summary": "Reviewing federated learning foundations and FedAvg algorithm", "urgency": "LOW"},
                    {
                        "summary": "Conflicting privacy mechanisms found in literature",
                        "urgency": "HIGH",
                        "question": (
                            "For privacy section of this federated learning survey: should it focus on "
                            "differential privacy vs secure aggregation as the primary mechanism? "
                            "DP adds noise to gradients; secure aggregation hides individual updates cryptographically."
                        ),
                    },
                    {"summary": "Privacy section drafted; moving to communication efficiency", "urgency": "LOW"},
                    {
                        "summary": "Multiple communication efficiency methods need to be prioritized",
                        "urgency": "MEDIUM",
                        "question": (
                            "Which communication efficiency methods for federated learning should be featured "
                            "as primary? Options: gradient compression (quantization/sparsification), "
                            "FedAvg local steps, or asynchronous federated optimization?"
                        ),
                    },
                    {"summary": "Communication section written; tackling data heterogeneity", "urgency": "LOW"},
                    {
                        "summary": "Need to cover non-iid data heterogeneity handling in federated learning",
                        "urgency": "MEDIUM",
                        "question": (
                            "For non-iid data heterogeneity handling in federated learning: "
                            "frame the solution as personalization/fine-tuning per client, "
                            "clustered federated learning, or global model regularization?"
                        ),
                    },
                    {"summary": "Survey complete; writing abstract and future directions", "urgency": "LOW"},
                ],
            ),
            # a6: feature engineering pipeline (GenericAgent)
            AgentSpec(
                agent_id="a6",
                agent_type="generic",
                task_description="build feature engineering pipeline for e-commerce churn prediction",
                decision_points=[
                    DecisionPoint(
                        "encoding high cardinality categoricals for churn model",
                        ["target encoding", "target", "embedding", "hash"],
                    ),
                    DecisionPoint(
                        "handling class imbalance in churn dataset",
                        ["smote", "oversample", "class weight", "weighted"],
                    ),
                    DecisionPoint(
                        "feature selection strategy for churn pipeline",
                        ["mutual information", "feature importance", "lasso", "permutation"],
                    ),
                ],
                steps=[
                    {"summary": "Loading raw transaction logs, analyzing categorical feature cardinality", "urgency": "LOW"},
                    {
                        "summary": "Customer ID and product category have high cardinality; encoding strategy needed",
                        "urgency": "MEDIUM",
                        "question": (
                            "For encoding high cardinality categoricals for churn model: "
                            "use target encoding (mean churn rate per category), one-hot with frequency cutoff, "
                            "learned embedding, or feature hashing? Target encoding leaks label info if not done carefully."
                        ),
                    },
                    {"summary": "Encoding applied; computing churn rate — 4.2% positive class", "urgency": "LOW"},
                    {
                        "summary": "Severe class imbalance detected (96/4 split) — strategy needed",
                        "urgency": "HIGH",
                        "question": (
                            "For handling class imbalance in churn dataset with 4% positive rate: "
                            "SMOTE oversampling, random oversampling, class weight adjustment in loss, "
                            "or threshold calibration post-training?"
                        ),
                    },
                    {"summary": "Imbalance handled; running initial model for feature importance", "urgency": "LOW"},
                    {
                        "summary": "360 features generated, need dimensionality reduction",
                        "urgency": "MEDIUM",
                        "question": (
                            "Best feature selection strategy for churn pipeline with 360 candidate features: "
                            "mutual information ranking, tree-based feature importance, LASSO regularization, "
                            "or permutation importance on held-out set?"
                        ),
                    },
                    {"summary": "Feature set reduced to 42 features; pipeline validated", "urgency": "LOW"},
                ],
            ),
            # a7: LRU cache implementation (GenericAgent)
            AgentSpec(
                agent_id="a7",
                agent_type="generic",
                task_description="implement LRU cache with O(1) get and put operations",
                decision_points=[
                    DecisionPoint(
                        "data structure for O(1) access in LRU cache",
                        ["hashmap", "hash map", "dict", "dictionary"],
                    ),
                    DecisionPoint(
                        "eviction order tracking for LRU cache",
                        ["doubly linked", "deque", "double linked"],
                    ),
                    DecisionPoint(
                        "thread safety for LRU cache concurrent access",
                        ["lock", "mutex", "synchronized", "reentrant"],
                    ),
                ],
                steps=[
                    {"summary": "Analyzing LRU cache requirements: O(1) get, O(1) put, capacity eviction", "urgency": "LOW"},
                    {
                        "summary": "Need primary data structure for O(1) access in LRU cache",
                        "urgency": "MEDIUM",
                        "question": (
                            "Which data structure for O(1) access in LRU cache? "
                            "A hashmap/dict keyed by cache key is standard; "
                            "alternatives like trie or balanced BST are O(log n). "
                            "What structure for O(1) cache key lookup?"
                        ),
                    },
                    {"summary": "HashMap implemented for key lookup; designing eviction order tracking", "urgency": "LOW"},
                    {
                        "summary": "Eviction order must be maintained alongside hashmap",
                        "urgency": "HIGH",
                        "question": (
                            "For eviction order tracking for LRU cache: "
                            "doubly linked list (allows O(1) removal from middle), "
                            "deque/ordered dict (Python-native), or sorted structure? "
                            "Doubly linked list + hashmap is the textbook O(1) solution."
                        ),
                    },
                    {"summary": "Doubly linked list integrated; testing get/put correctness", "urgency": "LOW"},
                    {
                        "summary": "Multi-threaded use case mentioned in spec — concurrency handling needed",
                        "urgency": "MEDIUM",
                        "question": (
                            "For thread safety for LRU cache concurrent access: "
                            "add a reentrant lock (mutex) around all operations, "
                            "use a concurrent hashmap, or make it explicitly not thread-safe with documentation?"
                        ),
                    },
                    {"summary": "Thread safety added; all tests passing including concurrent load test", "urgency": "LOW"},
                ],
            ),
            # a8: LLM alignment survey (GenericAgent)
            AgentSpec(
                agent_id="a8",
                agent_type="generic",
                task_description="write survey on large language model alignment and safety techniques",
                decision_points=[
                    DecisionPoint(
                        "RLHF vs constitutional AI as primary alignment framing",
                        ["rlhf", "reinforcement learning from human feedback", "human feedback"],
                    ),
                    DecisionPoint(
                        "evaluation benchmarks for LLM alignment survey",
                        ["truthfulqa", "mmlu", "helpfulness", "harmlessness"],
                    ),
                    DecisionPoint(
                        "safety capability tradeoff framing in alignment survey",
                        ["tradeoff", "tension", "balance", "alignment tax"],
                    ),
                ],
                steps=[
                    {"summary": "Reviewing RLHF, Constitutional AI, and DPO alignment literature", "urgency": "LOW"},
                    {
                        "summary": "Competing alignment frameworks need prioritization for survey structure",
                        "urgency": "HIGH",
                        "question": (
                            "For the primary framing of this LLM alignment survey: "
                            "use RLHF vs constitutional AI as primary alignment framing — "
                            "RLHF (InstructGPT paradigm) as central method with Constitutional AI as variant, "
                            "or treat them as parallel paradigms?"
                        ),
                    },
                    {"summary": "Alignment taxonomy section drafted; moving to evaluation methodology", "urgency": "LOW"},
                    {
                        "summary": "Choosing evaluation benchmarks for the survey's empirical comparison",
                        "urgency": "MEDIUM",
                        "question": (
                            "Which evaluation benchmarks for LLM alignment survey? "
                            "TruthfulQA (truthfulness), MMLU (capability), Helpfulness/Harmlessness/Honesty (HHH), "
                            "or a custom rubric for alignment-specific properties?"
                        ),
                    },
                    {"summary": "Evaluation section written; addressing safety vs capability tension", "urgency": "LOW"},
                    {
                        "summary": "Safety-capability tradeoff is contested — need to frame it carefully",
                        "urgency": "MEDIUM",
                        "question": (
                            "How to frame the safety capability tradeoff framing in alignment survey? "
                            "Present it as an inevitable tradeoff (alignment tax view), "
                            "a false dichotomy that better methods resolve, or an open empirical question?"
                        ),
                    },
                    {"summary": "Survey structure finalized; writing conclusions", "urgency": "LOW"},
                ],
            ),
            # a9: clinical data preprocessing (GenericAgent)
            AgentSpec(
                agent_id="a9",
                agent_type="generic",
                task_description="preprocess and normalize clinical trial data for meta-analysis",
                decision_points=[
                    DecisionPoint(
                        "missing lab values imputation for clinical data",
                        ["multiple imputation", "mice", "impute"],
                    ),
                    DecisionPoint(
                        "outlier detection method for clinical dosage data",
                        ["iqr", "z-score", "domain expert", "clinical range"],
                    ),
                    DecisionPoint(
                        "normalization strategy for multi-site clinical data",
                        ["combat", "standardize", "z-score", "site correction"],
                    ),
                ],
                steps=[
                    {"summary": "Loading 12-site clinical trial data; auditing missingness patterns", "urgency": "LOW"},
                    {
                        "summary": "38% of lab values missing with MCAR and MAR patterns mixed",
                        "urgency": "HIGH",
                        "question": (
                            "For missing lab values imputation for clinical data with 38% missingness: "
                            "multiple imputation (MICE), single imputation with mean/median, "
                            "complete case analysis, or mixed model that handles missingness implicitly?"
                        ),
                    },
                    {"summary": "Imputation strategy applied; checking dosage distribution for outliers", "urgency": "LOW"},
                    {
                        "summary": "Extreme dosage values detected — clinical vs statistical outliers unclear",
                        "urgency": "MEDIUM",
                        "question": (
                            "For outlier detection method for clinical dosage data: "
                            "IQR-based method (flag beyond 1.5×IQR), Z-score threshold, "
                            "or defer to domain expert review for any value outside published clinical range?"
                        ),
                    },
                    {"summary": "Outliers flagged and reviewed; addressing site effect corrections", "urgency": "LOW"},
                    {
                        "summary": "Large site-to-site variance found — harmonization needed",
                        "urgency": "MEDIUM",
                        "question": (
                            "For normalization strategy for multi-site clinical data: "
                            "ComBat (batch effect correction), Z-score standardization per site, "
                            "or mixed-effects model with site as random effect?"
                        ),
                    },
                    {"summary": "Harmonized dataset validated; exporting for meta-analysis", "urgency": "LOW"},
                ],
            ),
            # a10: trie implementation (GenericAgent)
            AgentSpec(
                agent_id="a10",
                agent_type="generic",
                task_description="implement trie for autocomplete with prefix search and word insertion",
                decision_points=[
                    DecisionPoint(
                        "node children representation in trie",
                        ["hashmap", "dict", "dictionary", "hash map"],
                    ),
                    DecisionPoint(
                        "end of word marking in trie node",
                        ["boolean flag", "is_end", "sentinel", "terminal"],
                    ),
                    DecisionPoint(
                        "prefix search return strategy for autocomplete trie",
                        ["dfs", "recursive", "collect all", "recursive dfs"],
                    ),
                ],
                steps=[
                    {"summary": "Designing TrieNode structure and insert algorithm", "urgency": "LOW"},
                    {
                        "summary": "Choosing children representation for trie node",
                        "urgency": "MEDIUM",
                        "question": (
                            "For node children representation in trie: "
                            "fixed-size array of 26 chars (fast but memory-heavy), "
                            "hashmap/dict (memory-efficient, handles Unicode), "
                            "or sorted list (cache-friendly)? Autocomplete may need Unicode support."
                        ),
                    },
                    {"summary": "TrieNode with hashmap children implemented; adding word insertion", "urgency": "LOW"},
                    {
                        "summary": "Need to mark end of words during insertion",
                        "urgency": "HIGH",
                        "question": (
                            "For end of word marking in trie node: "
                            "boolean `is_end` flag on each node, "
                            "sentinel child node with special key ($), "
                            "or word count integer (supports frequency-based autocomplete ranking)?"
                        ),
                    },
                    {"summary": "Word insertion complete; implementing prefix search for autocomplete", "urgency": "LOW"},
                    {
                        "summary": "Need to decide how to collect all words under a prefix node",
                        "urgency": "MEDIUM",
                        "question": (
                            "For prefix search return strategy for autocomplete trie: "
                            "recursive DFS from prefix node collecting all terminal words, "
                            "BFS for breadth-first (shorter completions first), "
                            "or lazy iterator to avoid collecting all words upfront?"
                        ),
                    },
                    {"summary": "Autocomplete trie complete; benchmarking insert and search latencies", "urgency": "LOW"},
                ],
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 2 — Scenario 4: N=3 homogeneous coding agents
    # Three CodeWriter-style agents all in the same domain (algorithms /
    # data-structures) but on different problems.  Tests whether DACS helps
    # even when all agents share the same vocabulary.
    # ------------------------------------------------------------------
    "s4_n3_homogeneous": ScenarioSpec(
        scenario_id="s4_n3_homogeneous",
        agents=[
            # a1: Red-Black Tree
            AgentSpec(
                agent_id="a1",
                agent_type="generic",
                task_description="implement red-black tree with insert rebalance and delete operations",
                decision_points=[
                    DecisionPoint(
                        "null leaf representation in red-black tree",
                        ["sentinel", "nil node", "null sentinel"],
                    ),
                    DecisionPoint(
                        "insert rebalancing cases for red-black tree",
                        ["uncle", "recolor", "rotate"],
                    ),
                    DecisionPoint(
                        "double-black resolution during red-black tree deletion",
                        ["sibling", "double-black", "absorb"],
                    ),
                    DecisionPoint(
                        "color encoding for red-black tree node",
                        ["boolean", "enum", "constant"],
                    ),
                ],
                steps=[
                    {"summary": "Designing RBT node structure with color field", "urgency": "LOW"},
                    {
                        "summary": "Evaluating null leaf representation in red-black tree",
                        "urgency": "MEDIUM",
                        "question": (
                            "For null leaf representation in red-black tree: use a shared sentinel nil node "
                            "(all null leaves point to the same BLACK sentinel object) or use Python None? "
                            "The sentinel simplifies rotation and rebalance code by avoiding null checks."
                        ),
                    },
                    {"summary": "RBT node with sentinel nil defined; implementing insert() with BST path", "urgency": "LOW"},
                    {
                        "summary": "Insert rebalancing cases for red-black tree needed after parent is RED",
                        "urgency": "HIGH",
                        "question": (
                            "For insert rebalancing cases for red-black tree: "
                            "when the new node's parent is RED, the fix depends on the uncle node's color. "
                            "Uncle RED → recolor parent+uncle to BLACK and grandparent to RED, then propagate. "
                            "Uncle BLACK → rotate toward the new node, then recolor. "
                            "Confirm this uncle-based case split is the right approach."
                        ),
                    },
                    {"summary": "Insert rebalancing complete; implementing delete() with transplant helper", "urgency": "LOW"},
                    {
                        "summary": "Delete creates double-black violation — sibling-based case analysis needed",
                        "urgency": "MEDIUM",
                        "question": (
                            "For double-black resolution during red-black tree deletion: "
                            "when the replacement node is double-black, should we first check if sibling is RED "
                            "(rotate parent toward double-black, convert to BLACK sibling case), "
                            "or check if sibling's children are both BLACK first "
                            "(recolor sibling RED, propagate double-black up to parent)? "
                            "Which case should be checked first in the resolution loop?"
                        ),
                    },
                    {
                        "summary": "Delete working; deciding color encoding for red-black tree node",
                        "urgency": "LOW",
                        "question": (
                            "For color encoding for red-black tree node: "
                            "store as Python boolean (True=RED, False=BLACK — minimal and fast) "
                            "or as an Enum (Color.RED, Color.BLACK — self-documenting, explicit)? "
                            "Boolean is simpler; enum is more readable in rotation code."
                        ),
                    },
                    {"summary": "Red-black tree complete; all insert/delete/rotation cases tested", "urgency": "LOW"},
                ],
            ),
            # a2: Hash Table with open addressing
            AgentSpec(
                agent_id="a2",
                agent_type="generic",
                task_description="implement hash table open addressing linear probing dynamic resizing",
                decision_points=[
                    DecisionPoint(
                        "probe sequence for collision resolution in open addressing hash table",
                        ["linear", "linear probing", "linear probe"],
                    ),
                    DecisionPoint(
                        "load factor threshold for hash table rehashing trigger",
                        ["0.7", "0.75", "70", "75"],
                    ),
                    DecisionPoint(
                        "handling deleted slots in open addressing hash table",
                        ["tombstone", "deleted marker", "sentinel marker"],
                    ),
                    DecisionPoint(
                        "hash function for string keys in open addressing hash table",
                        ["polynomial", "djb2", "fnv", "prime multiplier"],
                    ),
                ],
                steps=[
                    {"summary": "Designing internal array with slot states: EMPTY, OCCUPIED, DELETED", "urgency": "LOW"},
                    {
                        "summary": "Choosing collision resolution probe sequence for open addressing hash table",
                        "urgency": "MEDIUM",
                        "question": (
                            "For probe sequence for collision resolution in open addressing hash table: "
                            "linear probing (i+1, i+2 — good cache locality, but primary clustering), "
                            "quadratic probing (i+1², i+2² — less clustering), "
                            "or double hashing (i + k×h₂(key) — best distribution, more complex)? "
                            "Choose the probe sequence for this implementation."
                        ),
                    },
                    {"summary": "Probing implemented; writing put()/get() and basic correctness tests", "urgency": "LOW"},
                    {
                        "summary": "Table fills over time — load factor threshold for rehashing trigger needed",
                        "urgency": "HIGH",
                        "question": (
                            "At what load factor should the hash table trigger a rehash and resize? "
                            "0.5 (frequent but keeps chains short), "
                            "0.75 (Java HashMap default — good average-case performance), "
                            "or 0.9 (space-efficient but more collisions)? "
                            "Choose the load factor threshold for hash table rehashing trigger."
                        ),
                    },
                    {"summary": "Rehashing at 0.75 implemented; testing deletion — found probe chain break", "urgency": "LOW"},
                    {
                        "summary": "Deletion breaks probe chains — need strategy for handling deleted slots",
                        "urgency": "MEDIUM",
                        "question": (
                            "For handling deleted slots in open addressing hash table after deletion: "
                            "place a tombstone/deleted-marker so probe chains stay intact for subsequent lookups, "
                            "or shift all following elements in the cluster left (Robin Hood deletion — compact but complex)? "
                            "Tombstone is simpler and standard; shifting avoids tombstone accumulation."
                        ),
                    },
                    {
                        "summary": "Tombstone deletion working; choosing hash function for string keys",
                        "urgency": "MEDIUM",
                        "question": (
                            "For hash function for string keys in open addressing hash table: "
                            "polynomial rolling hash (sum of char×prime^i), "
                            "DJB2 (5381×33+c — fast and widely used), "
                            "FNV-1a (good avalanche, standard for non-crypto hashing), "
                            "or Python built-in hash()? "
                            "DJB2 and FNV are the standard choices for string hashing."
                        ),
                    },
                    {"summary": "Hash table with string support complete; all benchmarks passing", "urgency": "LOW"},
                ],
            ),
            # a3: Directed weighted graph with topological sort + Dijkstra
            AgentSpec(
                agent_id="a3",
                agent_type="generic",
                task_description="implement directed weighted graph DFS BFS topological sort shortest path",
                decision_points=[
                    DecisionPoint(
                        "topological sort algorithm for directed acyclic graph",
                        ["kahn", "kahn's", "bfs-based", "in-degree"],
                    ),
                    DecisionPoint(
                        "handling disconnected components in topological sort graph",
                        ["all vertices", "outer loop", "every unvisited", "all nodes"],
                    ),
                    DecisionPoint(
                        "shortest path algorithm for this weighted directed graph",
                        ["dijkstra", "dijkstra's"],
                    ),
                    DecisionPoint(
                        "priority queue implementation for Dijkstra's algorithm",
                        ["heap", "heapq", "min-heap", "priority queue"],
                    ),
                ],
                steps=[
                    {"summary": "Adjacency list of (neighbour, weight) tuples defined; BFS and DFS implemented", "urgency": "LOW"},
                    {
                        "summary": "Choosing topological sort algorithm for directed acyclic graph",
                        "urgency": "MEDIUM",
                        "question": (
                            "For topological sort algorithm for directed acyclic graph: "
                            "Kahn's algorithm (BFS with in-degree counts — explicit cycle detection via remaining nodes), "
                            "or DFS post-order (push to stack on DFS return — compact but cycle detection is a separate pass)? "
                            "Kahn's is more debuggable; DFS is more elegant."
                        ),
                    },
                    {"summary": "Kahn's topological sort implemented; testing on sample DAGs", "urgency": "LOW"},
                    {
                        "summary": "Graph has disconnected components — topological sort misses isolated nodes",
                        "urgency": "HIGH",
                        "question": (
                            "Kahn's implementation fails on graphs with disconnected components in topological sort. "
                            "Fix approach: initialise the queue with all vertices that have in-degree zero "
                            "(including isolated nodes), or detect disconnected components first and process each? "
                            "Enqueuing all zero-in-degree nodes at the start handles this naturally."
                        ),
                    },
                    {"summary": "Topological sort handles disconnected graphs; starting shortest path", "urgency": "LOW"},
                    {
                        "summary": "Choosing shortest path algorithm for this weighted directed graph",
                        "urgency": "MEDIUM",
                        "question": (
                            "For shortest path algorithm for this weighted directed graph: "
                            "Dijkstra's (O((V+E)log V), requires non-negative edge weights), "
                            "Bellman-Ford (O(VE), handles negative weights — slower), "
                            "or Floyd-Warshall (O(V³), all-pairs)? "
                            "Edge weights are guaranteed non-negative in this graph."
                        ),
                    },
                    {
                        "summary": "Dijkstra selected; choosing priority queue implementation for Dijkstra's algorithm",
                        "urgency": "MEDIUM",
                        "question": (
                            "For priority queue implementation for Dijkstra's algorithm: "
                            "Python heapq module (binary min-heap, lazy deletion with visited set — standard), "
                            "sortedcontainers SortedList (decreaseKey support but external dependency), "
                            "or a manual d-ary heap? "
                            "heapq with (distance, node) tuples and a visited set is the idiomatic Python approach."
                        ),
                    },
                    {"summary": "Dijkstra with heapq complete; all shortest-path tests passing", "urgency": "LOW"},
                ],
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 2 — Scenario 5: N=5 crossfire — maximally diverse agents
    # Five agents spanning completely different professional domains so that
    # any vocabulary from one agent appearing in another's response is an
    # unambiguous contamination signal.
    # ------------------------------------------------------------------
    "s5_n5_crossfire": ScenarioSpec(
        scenario_id="s5_n5_crossfire",
        agents=[
            # a1: Lock-free concurrent queue in C++
            AgentSpec(
                agent_id="a1",
                agent_type="generic",
                task_description="implement lock-free MPSC queue using C++ atomic CAS operations",
                decision_points=[
                    DecisionPoint(
                        "memory ordering for CAS on head pointer in lock-free queue",
                        ["acquire", "release", "seq_cst", "memory_order"],
                    ),
                    DecisionPoint(
                        "ABA problem mitigation strategy in lock-free queue",
                        ["tagged pointer", "hazard pointer", "version tag", "epoch"],
                    ),
                    DecisionPoint(
                        "sentinel dummy node for lock-free queue initialisation",
                        ["dummy", "sentinel", "dummy node", "sentinel node"],
                    ),
                    DecisionPoint(
                        "dequeue from empty lock-free queue return strategy",
                        ["optional", "std::optional", "null", "false", "empty"],
                    ),
                ],
                steps=[
                    {"summary": "Designing lock-free MPSC queue node structure with atomic next pointer", "urgency": "LOW"},
                    {
                        "summary": "Memory ordering for CAS on head pointer in lock-free queue must be chosen",
                        "urgency": "HIGH",
                        "question": (
                            "For memory ordering for CAS on head pointer in lock-free queue: "
                            "std::memory_order_seq_cst (strongest, safe but may reduce throughput), "
                            "acquire on load + release on successful CAS (correct for producer-consumer), "
                            "or relaxed (too weak, undefined behaviour risk)? "
                            "The head pointer mediates producer-consumer handoff — correct ordering is critical."
                        ),
                    },
                    {"summary": "CAS with acquire/release implemented; testing single producer single consumer", "urgency": "LOW"},
                    {
                        "summary": "ABA problem detected in stress test — mitigation strategy needed",
                        "urgency": "HIGH",
                        "question": (
                            "Under high-frequency enqueue/dequeue the ABA problem corrupts the queue. "
                            "ABA problem mitigation strategy in lock-free queue: "
                            "tagged pointer (pack a version counter in the low bits of the pointer), "
                            "hazard pointers (each thread registers pointers it is reading), "
                            "or epoch-based reclamation (defer frees until all threads advance)? "
                            "Tagged pointers are simplest on 64-bit platforms with spare pointer bits."
                        ),
                    },
                    {"summary": "ABA mitigation applied; implementing initialisation path", "urgency": "LOW"},
                    {
                        "summary": "Empty queue state: sentinel dummy node vs checking head==tail",
                        "urgency": "MEDIUM",
                        "question": (
                            "For lock-free queue initialisation and empty-queue detection: "
                            "allocate a sentinel dummy node so head and tail always point to valid nodes "
                            "(simplifies boundary conditions in enqueue/dequeue), "
                            "or use nullptr sentinels and add null checks throughout? "
                            "The M&S queue paper uses a sentinel dummy node for this reason."
                        ),
                    },
                    {
                        "summary": "Sentinel initialised; deciding dequeue-from-empty return strategy",
                        "urgency": "MEDIUM",
                        "question": (
                            "When dequeue is called on an empty lock-free queue, what should it return? "
                            "std::optional<T> (idiomatic C++17, explicit empty signal), "
                            "return bool with out-parameter T& (C++11 compatible), "
                            "or throw an exception? "
                            "std::optional makes the empty case explicit without exceptions."
                        ),
                    },
                    {"summary": "Lock-free MPSC queue complete; stress tests with 16 threads passing", "urgency": "LOW"},
                ],
            ),
            # a2: Diffusion model training stability survey
            AgentSpec(
                agent_id="a2",
                agent_type="generic",
                task_description="write survey on diffusion model training stability and convergence",
                decision_points=[
                    DecisionPoint(
                        "primary framing for diffusion model survey: score matching or DDPM",
                        ["score matching", "score-based", "ddpm", "denoising"],
                    ),
                    DecisionPoint(
                        "noise schedule analysis for diffusion training stability survey",
                        ["cosine", "linear", "noise schedule"],
                    ),
                    DecisionPoint(
                        "training instability root cause framing in diffusion survey",
                        ["gradient", "exploding", "vanishing", "loss spike"],
                    ),
                    DecisionPoint(
                        "primary evaluation metric for diffusion model quality",
                        ["fid", "frechet", "inception distance"],
                    ),
                ],
                steps=[
                    {"summary": "Reviewing DDPM, score-based SDEs, and DDIM literature for survey taxonomy", "urgency": "LOW"},
                    {
                        "summary": "Primary framing for diffusion model survey needs to be established",
                        "urgency": "HIGH",
                        "question": (
                            "For the primary framing for diffusion model survey on training stability: "
                            "use score matching / score-based SDE framework (Song et al. 2021 — unified view), "
                            "or DDPM probabilistic framing (Ho et al. 2020 — most cited, practical)? "
                            "Score matching unifies DDPMs and SDEs elegantly; DDPM framing is more accessible."
                        ),
                    },
                    {"summary": "Taxonomy section drafted; analysing noise schedules across training runs", "urgency": "LOW"},
                    {
                        "summary": "Noise schedule analysis shows significant effect on training stability",
                        "urgency": "MEDIUM",
                        "question": (
                            "For noise schedule analysis for diffusion training stability survey: "
                            "linear noise schedule (Ho et al. default — simple but poor for high-resolution images), "
                            "cosine noise schedule (Nichol & Dhariwal — smoother signal-to-noise decay, more stable), "
                            "or learned noise schedule (flexible but harder to reproduce)? "
                            "Cosine schedule is the current standard for training stability."
                        ),
                    },
                    {"summary": "Noise schedule section written; analysing root causes of training instability", "urgency": "LOW"},
                    {
                        "summary": "Instability root cause framing in diffusion survey is contested in literature",
                        "urgency": "MEDIUM",
                        "question": (
                            "How to frame training instability root cause framing in diffusion survey? "
                            "Attribute it primarily to gradient explosion/vanishing in the U-Net denoiser, "
                            "to loss spikes from extreme noise level weighting, "
                            "or to numerical precision issues in the variance schedule computation? "
                            "Loss spikes from noise level weighting are the most commonly reported cause."
                        ),
                    },
                    {
                        "summary": "Instability section complete; writing evaluation methodology",
                        "urgency": "LOW",
                        "question": (
                            "What is the primary evaluation metric for diffusion model quality in this survey? "
                            "FID (Fréchet Inception Distance — standard but critic of inception features), "
                            "IS (Inception Score — less reliable), "
                            "CLIP score (semantic alignment, good for text-conditioned models), "
                            "or human evaluation? "
                            "FID is the de-facto standard benchmark metric for generative model quality."
                        ),
                    },
                    {"summary": "Survey complete with evaluation section; writing abstract", "urgency": "LOW"},
                ],
            ),
            # a3: Genomics variant calling ETL pipeline
            AgentSpec(
                agent_id="a3",
                agent_type="generic",
                task_description="build ETL pipeline for genomics VCF variant calling and annotation",
                decision_points=[
                    DecisionPoint(
                        "VCF parsing library for genomics ETL pipeline",
                        ["pysam", "cyvcf2", "pyvcf", "bcftools"],
                    ),
                    DecisionPoint(
                        "variant filtering quality threshold for genomics pipeline",
                        ["qual", "quality score", "30", "phred", "genotype quality"],
                    ),
                    DecisionPoint(
                        "reference genome version for variant annotation",
                        ["grch38", "hg38", "grch37", "reference genome"],
                    ),
                    DecisionPoint(
                        "output format for processed genomics variants",
                        ["parquet", "vcf", "tsv", "hdf5"],
                    ),
                ],
                steps=[
                    {"summary": "Loading multi-sample VCF files from sequencing pipeline; reviewing variant call formats", "urgency": "LOW"},
                    {
                        "summary": "VCF parsing library selection needed for genomics ETL pipeline",
                        "urgency": "MEDIUM",
                        "question": (
                            "For VCF parsing library for genomics ETL pipeline: "
                            "pysam (wraps htslib, C-based, fast random access), "
                            "cyvcf2 (fast Cython VCF parser, full INFO/FORMAT field access), "
                            "PyVCF (pure Python, slower but simpler API), "
                            "or bcftools subprocess (shell out to compiled tool)? "
                            "cyvcf2 is the fastest pure-Python-interfaced option for large VCF files."
                        ),
                    },
                    {"summary": "VCF parser integrated; computing quality distributions for filtering threshold", "urgency": "LOW"},
                    {
                        "summary": "Quality threshold for variant filtering needs to be set",
                        "urgency": "HIGH",
                        "question": (
                            "For variant filtering quality threshold for genomics pipeline: "
                            "what QUAL score cutoff to apply? "
                            "QUAL≥30 (Phred-scaled — 99.9% accuracy, standard GATK recommendation), "
                            "QUAL≥20 (99% accuracy, more permissive), "
                            "or filter by genotype quality (GQ≥20) per sample instead of site QUAL? "
                            "QUAL≥30 with GQ≥20 per sample is the standard pipeline recommendation."
                        ),
                    },
                    {"summary": "Variant filtering applied; connecting to annotation database", "urgency": "LOW"},
                    {
                        "summary": "Reference genome version must be specified for accurate variant annotation",
                        "urgency": "MEDIUM",
                        "question": (
                            "Which reference genome version for variant annotation: "
                            "GRCh38/hg38 (current — required for all new projects, dbSNP 155+ uses it), "
                            "GRCh37/hg19 (legacy — many existing clinical databases still use it), "
                            "or T2T-CHM13 (telomere-to-telomere assembly, cutting-edge but limited annotation)? "
                            "GRCh38 is the required standard for new clinical genomics pipelines."
                        ),
                    },
                    {
                        "summary": "Annotation complete using GRCh38; choosing output format for downstream analysis",
                        "urgency": "LOW",
                        "question": (
                            "For output format for processed genomics variants: "
                            "Apache Parquet (columnar, excellent for downstream pandas/Spark analytics), "
                            "annotated VCF (standard interchange format, interoperable with genomics tools), "
                            "TSV (simple, human-readable), "
                            "or HDF5 (hierarchical, good for paired genotype matrices)? "
                            "Parquet is preferred for analytical pipelines; VCF for interoperability."
                        ),
                    },
                    {"summary": "ETL pipeline outputting Parquet; validation against gold-standard variant calls complete", "urgency": "LOW"},
                ],
            ),
            # a4: C++ memory leak debugging (DebuggerAgent scenario)
            AgentSpec(
                agent_id="a4",
                agent_type="generic",
                task_description="debug and fix memory leak in multi-threaded C++ allocator component",
                decision_points=[
                    DecisionPoint(
                        "memory sanitizer tool to diagnose C++ heap memory leak",
                        ["asan", "address sanitizer", "addresssanitizer"],
                    ),
                    DecisionPoint(
                        "heap profiler for C++ allocation call-stack analysis",
                        ["valgrind", "heaptrack", "massif"],
                    ),
                    DecisionPoint(
                        "minimal reproducer strategy for isolating C++ memory leak",
                        ["minimal reproducer", "minimal", "isolate"],
                    ),
                    DecisionPoint(
                        "fix strategy for C++ memory leak in thread teardown path",
                        ["unique_ptr", "smart pointer", "raii"],
                    ),
                ],
                steps=[
                    {"summary": "Analysing crash reports showing heap growth over runtime; leak confirmed", "urgency": "LOW"},
                    {
                        "summary": "Choosing memory sanitizer tool to diagnose C++ heap memory leak",
                        "urgency": "MEDIUM",
                        "question": (
                            "Which memory sanitizer tool to diagnose C++ heap memory leak first? "
                            "AddressSanitizer (ASan — fast, detects leaks and overflows, GCC/Clang flag -fsanitize=address), "
                            "Valgrind Memcheck (slower, ~10× overhead, but no recompile needed), "
                            "or LeakSanitizer (LSan — lightweight leak-only detection, subset of ASan)? "
                            "ASan is the fastest path to a confirmed leak with backtrace."
                        ),
                    },
                    {"summary": "ASan enabled; confirming heap leak in 5-minute stress run", "urgency": "LOW"},
                    {
                        "summary": "ASan confirms bytes leaked but allocation backtrace is incomplete",
                        "urgency": "HIGH",
                        "question": (
                            "ASan shows leaked bytes but not full allocation call stacks. "
                            "Which heap profiler for C++ allocation call-stack analysis: "
                            "Valgrind Massif (allocation timeline with full call graph), "
                            "heaptrack (lower overhead than Massif, annotated flame graphs), "
                            "or gperftools heap profiler (minimal overhead, production-safe)? "
                            "heaptrack provides the richest call-stack view for complex multi-threaded leaks."
                        ),
                    },
                    {"summary": "heaptrack shows leak originates across 3 modules; starting isolation", "urgency": "LOW"},
                    {
                        "summary": "Minimal reproducer strategy for isolating C++ memory leak across 3 modules",
                        "urgency": "HIGH",
                        "question": (
                            "For minimal reproducer strategy for isolating C++ memory leak: "
                            "write a standalone test that exercises only the suspected ThreadPool module, "
                            "add allocator interception hooks to trace every malloc/free pair, "
                            "or disable modules one by one until the leak disappears? "
                            "A minimal reproducer in a standalone test is fastest and most debuggable."
                        ),
                    },
                    {
                        "summary": "Minimal reproducer confirms leak is in ThreadPool worker teardown path",
                        "urgency": "MEDIUM",
                        "question": (
                            "Fix strategy for C++ memory leak in thread teardown path: "
                            "wrap Worker* raw pointers in std::unique_ptr<Worker> (RAII — automatic cleanup on scope exit), "
                            "add explicit delete calls in all early-exit code paths, "
                            "or use std::shared_ptr<Worker> with a shutdown flag? "
                            "std::unique_ptr (RAII/smart pointer) gives automatic cleanup with no manual tracking."
                        ),
                    },
                    {"summary": "unique_ptr refactor applied; ASan clean on full stress run; adding CI regression", "urgency": "LOW"},
                ],
            ),
            # a5: Clinical trial paper methodology section (LongWriterAgent scenario)
            AgentSpec(
                agent_id="a5",
                agent_type="generic",
                task_description="write methodology section of clinical trial paper for journal submission",
                decision_points=[
                    DecisionPoint(
                        "study design framing for clinical trial methodology",
                        ["randomised", "rct", "controlled", "randomized"],
                    ),
                    DecisionPoint(
                        "power calculation placement in clinical trial paper",
                        ["power", "sample size", "power calculation"],
                    ),
                    DecisionPoint(
                        "primary endpoint framing for clinical trial outcomes section",
                        ["primary endpoint", "primary outcome", "pre-specified"],
                    ),
                    DecisionPoint(
                        "control group description in clinical trial methodology",
                        ["placebo", "standard of care", "inert"],
                    ),
                ],
                steps=[
                    {"summary": "Reviewing CONSORT checklist and target journal formatting requirements", "urgency": "LOW"},
                    {
                        "summary": "Study design framing for clinical trial methodology section must be decided",
                        "urgency": "HIGH",
                        "question": (
                            "For study design framing for clinical trial methodology: "
                            "open with 'This was a randomised controlled trial (RCT) with 1:1 allocation' "
                            "(CONSORT standard — expected by peer reviewers), "
                            "or open with the scientific question and let design description follow? "
                            "CONSORT guidelines require the design to be identified in the first sentence of methods."
                        ),
                    },
                    {"summary": "RCT framing established in section 1; writing statistical analysis subsection", "urgency": "LOW"},
                    {
                        "summary": "Statistical depth decision: power calculation placement in paper",
                        "urgency": "MEDIUM",
                        "question": (
                            "For power calculation placement in clinical trial paper: "
                            "include power calculation and sample size derivation inline in the statistical methods "
                            "(CONSORT 7b requirement — must be reported), "
                            "refer readers to a separate Statistical Analysis Plan document, "
                            "or put it in a supplementary appendix? "
                            "CONSORT explicitly requires sample size and power calculation in the main methods section."
                        ),
                    },
                    {"summary": "Power calculation inline; writing primary and secondary outcomes subsection", "urgency": "LOW"},
                    {
                        "summary": "Primary endpoint framing for clinical trial outcomes section needs to be settled",
                        "urgency": "MEDIUM",
                        "question": (
                            "For primary endpoint framing for clinical trial outcomes section: "
                            "single pre-specified primary endpoint (cleanest — minimises multiple comparison concerns), "
                            "composite endpoint combining multiple events into one outcome, "
                            "or co-primary endpoints with multiplicity adjustment? "
                            "A single pre-specified primary endpoint is the regulatory gold standard."
                        ),
                    },
                    {
                        "summary": "Primary endpoint defined; writing comparator and control group description",
                        "urgency": "LOW",
                        "question": (
                            "How should the control group description in clinical trial methodology be framed? "
                            "As 'placebo control: inert tablet matched in appearance to study drug', "
                            "as 'standard of care comparator' (active treatment), "
                            "or as 'wait-list control' (common in behavioural trials)? "
                            "The trial used an inert placebo — double-blind placebo-controlled is the correct description."
                        ),
                    },
                    {"summary": "Methodology section complete with CONSORT-compliant flow; submitting for co-author review", "urgency": "LOW"},
                ],
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 2 — Scenario 6: N=5 cascade — inter-agent output dependencies
    # Adversarial scenario for DACS: agents' tasks depend on each other's
    # outputs. The flat-context baseline may benefit from seeing all agents'
    # histories simultaneously. Tests whether DACS registry correctly tracks
    # cross-agent context without focus-session bleed.
    # ------------------------------------------------------------------
    "s6_n5_cascade": ScenarioSpec(
        scenario_id="s6_n5_cascade",
        agents=[
            # a1: Architecture planner (upstream dependency for all others)
            AgentSpec(
                agent_id="a1",
                agent_type="generic",
                task_description="plan architecture for a real-time recommendation system",
                decision_points=[
                    DecisionPoint(
                        "system architecture style for recommendation system",
                        ["microservices", "service-oriented", "soa", "micro-service"],
                    ),
                    DecisionPoint(
                        "candidate retrieval strategy for recommendation system",
                        ["ann", "approximate nearest neighbour", "faiss", "vector"],
                    ),
                    DecisionPoint(
                        "serving infrastructure latency target for recommendation system",
                        ["p99", "50ms", "100ms", "latency sla"],
                    ),
                ],
                steps=[
                    {"summary": "Gathering requirements: 50M users, 10M items, <100ms serving, personalised feed", "urgency": "LOW"},
                    {
                        "summary": "Architecture style decision for recommendation system — monolith vs microservices",
                        "urgency": "HIGH",
                        "question": (
                            "For system architecture style for recommendation system at 50M users: "
                            "microservices (independent retrieval, ranking, and feature-serving components — "
                            "each can scale independently), "
                            "monolith (simpler ops, lower latency for co-located components), "
                            "or event-driven lambda architecture (batch + streaming layers)? "
                            "Microservices allow independent scaling of the retrieval and ranking stages."
                        ),
                    },
                    {"summary": "Microservices architecture confirmed; designing candidate retrieval layer", "urgency": "LOW"},
                    {
                        "summary": "Candidate retrieval strategy for recommendation system must be chosen",
                        "urgency": "MEDIUM",
                        "question": (
                            "For candidate retrieval strategy for recommendation system: "
                            "approximate nearest neighbour (ANN) search over item embeddings (FAISS/ScaNN — "
                            "retrieves top-K semantically similar candidates fast), "
                            "collaborative filtering matrix factorisation (item-user score lookup), "
                            "or inverted index with BM25 (keyword-based, fast but less personalised)? "
                            "ANN over learned embeddings is the industry standard for large-scale rec-sys retrieval."
                        ),
                    },
                    {"summary": "ANN retrieval layer designed; defining latency SLAs for serving", "urgency": "LOW"},
                    {
                        "summary": "Serving latency target must be specified to guide all downstream component design",
                        "urgency": "MEDIUM",
                        "question": (
                            "What serving infrastructure latency target for recommendation system should be the SLA? "
                            "P99 < 50ms (aggressive — requires in-memory serving, no cold calls), "
                            "P99 < 100ms (industry standard for recommendation feeds), "
                            "or P95 < 200ms (relaxed — allows some network hops)? "
                            "P99 < 100ms is the standard latency SLA for recommendation serving."
                        ),
                    },
                    {"summary": "Architecture plan complete: microservices, ANN retrieval, P99<100ms SLA", "urgency": "LOW"},
                ],
            ),
            # a2: Retrieval service (depends on a1's ANN decision)
            AgentSpec(
                agent_id="a2",
                agent_type="generic",
                task_description="implement ANN retrieval service for recommendation system",
                decision_points=[
                    DecisionPoint(
                        "ANN index type for recommendation retrieval service",
                        ["hnsw", "ivf", "flat index", "hierarchical"],
                    ),
                    DecisionPoint(
                        "embedding model for retrieval service item representations",
                        ["two-tower", "dual encoder", "item2vec", "embedding model"],
                    ),
                    DecisionPoint(
                        "cache strategy for retrieval service hot items",
                        ["redis", "in-memory cache", "lru cache", "hot item cache"],
                    ),
                ],
                steps=[
                    {"summary": "Retrieval service scaffolded; architecture plan specifies ANN over embeddings", "urgency": "LOW"},
                    {
                        "summary": "ANN index type for recommendation retrieval service must be selected",
                        "urgency": "HIGH",
                        "question": (
                            "The architecture plan established ANN retrieval with FAISS. "
                            "For ANN index type for recommendation retrieval service: "
                            "HNSW (hierarchical navigable small world — best recall, higher memory), "
                            "IVF (inverted file — lower memory, slightly lower recall, good for 10M+ items), "
                            "or flat index (exact search — only feasible for small catalogs)? "
                            "HNSW gives the best recall/latency tradeoff for 10M items at the P99<100ms SLA."
                        ),
                    },
                    {"summary": "HNSW index selected; designing item embedding generation", "urgency": "LOW"},
                    {
                        "summary": "Embedding model for retrieval service item representations must be chosen",
                        "urgency": "MEDIUM",
                        "question": (
                            "For embedding model for retrieval service item representations: "
                            "two-tower / dual encoder model (user tower + item tower — jointly trained, industry standard), "
                            "item2vec (item co-occurrence embeddings — simple, no user tower), "
                            "or fine-tuned BERT on item descriptions (rich semantic embeddings but heavy serving)? "
                            "Two-tower models are the standard for personalised ANN retrieval at scale."
                        ),
                    },
                    {
                        "summary": "Two-tower embedding model chosen; designing caching for hot items",
                        "urgency": "LOW",
                        "question": (
                            "For cache strategy for retrieval service hot items to meet P99<100ms SLA: "
                            "Redis (in-memory key-value store — sub-millisecond, widely used for feature serving), "
                            "application-level LRU cache (co-located, zero network hop), "
                            "or pre-computed embedding cache refreshed every 15 minutes? "
                            "Redis with TTL-based refresh is standard for hot item caching in retrieval services."
                        ),
                    },
                    {"summary": "Retrieval service complete: HNSW index, two-tower embeddings, Redis hot-item cache", "urgency": "LOW"},
                ],
            ),
            # a3: Ranking service (depends on a1's latency SLA; needs a2's retrieval output)
            AgentSpec(
                agent_id="a3",
                agent_type="generic",
                task_description="implement ranking model service for recommendation system",
                decision_points=[
                    DecisionPoint(
                        "ranking model architecture for recommendation ranking service",
                        ["wide and deep", "dcn", "deep cross", "two-stage", "neural"],
                    ),
                    DecisionPoint(
                        "serving framework for recommendation ranking model",
                        ["triton", "torchserve", "tensorflow serving", "onnx"],
                    ),
                    DecisionPoint(
                        "feature freshness strategy for ranking model serving",
                        ["online feature", "real-time feature", "feature store", "streaming"],
                    ),
                ],
                steps=[
                    {"summary": "Ranking service receives top-K candidates from retrieval; must rerank for CTR", "urgency": "LOW"},
                    {
                        "summary": "Ranking model architecture for recommendation ranking service must be decided",
                        "urgency": "HIGH",
                        "question": (
                            "For ranking model architecture for recommendation ranking service "
                            "operating within the P99<100ms SLA: "
                            "Wide & Deep (linear wide component + dense deep component — Google Play standard), "
                            "DCN v2 (deep and cross network — explicit feature interactions), "
                            "or DLRM (Facebook's deep learning recommendation model with embedding tables)? "
                            "Wide and Deep and DCN are the most widely deployed ranking architectures."
                        ),
                    },
                    {"summary": "DCN v2 selected for ranking; choosing serving framework", "urgency": "LOW"},
                    {
                        "summary": "Serving framework for recommendation ranking model must meet P99<100ms constraint",
                        "urgency": "MEDIUM",
                        "question": (
                            "For serving framework for recommendation ranking model at P99<100ms SLA: "
                            "NVIDIA Triton Inference Server (GPU inference, dynamic batching — lowest latency), "
                            "TorchServe (PyTorch native, easier ops), "
                            "TensorFlow Serving (mature, gRPC interface), "
                            "or ONNX Runtime (cross-framework, good CPU latency)? "
                            "Triton with TensorRT optimisation achieves the lowest P99 for neural ranking models."
                        ),
                    },
                    {
                        "summary": "Triton selected; designing feature freshness for online ranking",
                        "urgency": "MEDIUM",
                        "question": (
                            "For feature freshness strategy for ranking model serving: "
                            "online features streamed from Kafka into the feature store (real-time, <1s freshness), "
                            "batch-refreshed features every 15 minutes (simpler ops, tolerable staleness), "
                            "or dual-path (stale batch features with real-time override for critical signals)? "
                            "Real-time feature streaming from the feature store is needed for recency-sensitive signals."
                        ),
                    },
                    {"summary": "Ranking service complete: DCN v2 on Triton, real-time feature streaming", "urgency": "LOW"},
                ],
            ),
            # a4: Feature store (depends on a1's architecture; feeds a3's ranking)
            AgentSpec(
                agent_id="a4",
                agent_type="generic",
                task_description="design and implement feature store for recommendation system",
                decision_points=[
                    DecisionPoint(
                        "feature store technology for recommendation system",
                        ["feast", "tecton", "hopsworks", "redis feature store"],
                    ),
                    DecisionPoint(
                        "online store backend for feature store low-latency serving",
                        ["redis", "cassandra", "dynamodb", "key-value"],
                    ),
                    DecisionPoint(
                        "feature computation scheduling for recommendation feature store",
                        ["spark", "flink", "stream processing", "batch"],
                    ),
                ],
                steps=[
                    {"summary": "Feature store must serve both offline training and online ranking within P99<100ms SLA", "urgency": "LOW"},
                    {
                        "summary": "Feature store technology for recommendation system must be selected",
                        "urgency": "HIGH",
                        "question": (
                            "For feature store technology for recommendation system: "
                            "Feast (open-source, integrates with Redis online store + S3 offline store), "
                            "Tecton (managed, production-grade, higher cost), "
                            "Hopsworks (open-source with HSFS API, strong on Spark integration), "
                            "or build a custom feature store on Redis + Hive? "
                            "Feast is the standard open-source feature store for this architecture."
                        ),
                    },
                    {"summary": "Feast selected; configuring online and offline store backends", "urgency": "LOW"},
                    {
                        "summary": "Online store backend for feature store low-latency serving must be chosen",
                        "urgency": "MEDIUM",
                        "question": (
                            "For online store backend for feature store low-latency serving to meet P99<100ms: "
                            "Redis (sub-millisecond point lookups, in-memory — standard Feast online store), "
                            "Cassandra (wide-column, ~5ms P99, good for very high write throughput), "
                            "DynamoDB (managed, ~5ms P99, AWS-native), "
                            "or Bigtable (Google Cloud, ~5ms P99)? "
                            "Redis is required to meet P99<100ms when the ranking model also runs in the same request."
                        ),
                    },
                    {
                        "summary": "Redis online store configured; choosing batch vs streaming feature computation",
                        "urgency": "MEDIUM",
                        "question": (
                            "For feature computation scheduling for recommendation feature store: "
                            "Apache Spark batch jobs (hourly/daily refresh — simple ops, tolerable staleness), "
                            "Apache Flink stream processing (sub-second freshness — needed for real-time signals), "
                            "or dual-mode (Spark for historical features + Flink for real-time user activity)? "
                            "Flink stream processing is needed to keep engagement features fresh enough for the ranking model."
                        ),
                    },
                    {"summary": "Feature store complete: Feast on Redis online store with Flink streaming computation", "urgency": "LOW"},
                ],
            ),
            # a5: Architecture reviewer (depends on a1-a4's outputs)
            AgentSpec(
                agent_id="a5",
                agent_type="generic",
                task_description="review integrated architecture of recommendation system components",
                decision_points=[
                    DecisionPoint(
                        "most critical integration risk in recommendation system architecture",
                        ["latency", "budget", "p99", "sla", "end-to-end"],
                    ),
                    DecisionPoint(
                        "data consistency concern across recommendation system components",
                        ["embedding", "staleness", "version", "consistency"],
                    ),
                    DecisionPoint(
                        "observability priority for recommendation system review",
                        ["monitoring", "tracing", "latency tracking", "a/b", "experiment"],
                    ),
                ],
                steps=[
                    {"summary": "Reviewing full system: a1 architecture plan, a2 retrieval, a3 ranking, a4 feature store", "urgency": "LOW"},
                    {
                        "summary": "Identifying most critical integration risk across all four components",
                        "urgency": "HIGH",
                        "question": (
                            "The architecture has four components: retrieval (HNSW+Redis), "
                            "ranking (DCN v2 on Triton), feature store (Feast+Redis+Flink), "
                            "and an ANN index — all within P99<100ms SLA. "
                            "What is the most critical integration risk in recommendation system architecture? "
                            "End-to-end latency budget allocation (retrieval+ranking+feature lookup must all fit in 100ms), "
                            "cold-start for new users/items, "
                            "or deployment ordering dependencies between components?"
                        ),
                    },
                    {"summary": "Latency budget allocation flagged as top risk; reviewing data consistency", "urgency": "LOW"},
                    {
                        "summary": "Data consistency concern across recommendation system components must be addressed",
                        "urgency": "MEDIUM",
                        "question": (
                            "With the retrieval service using two-tower embeddings and the feature store using "
                            "Flink-generated features, what is the main data consistency concern across recommendation "
                            "system components? "
                            "Embedding version staleness (retrieval uses v1 embeddings while ranking model expects v2), "
                            "feature staleness between the Flink stream and Redis writes, "
                            "or training-serving skew between offline feature distributions and online feature values?"
                        ),
                    },
                    {
                        "summary": "Embedding versioning flagged; writing observability recommendations",
                        "urgency": "LOW",
                        "question": (
                            "For observability priority for recommendation system review: "
                            "which should be instrumented first? "
                            "P99 latency tracking at each component boundary (retrieval / feature-fetch / ranking), "
                            "A/B experiment framework to measure CTR impact of ranking model changes, "
                            "or distributed tracing end-to-end per recommendation request? "
                            "P99 latency monitoring at each stage is the highest-priority observability need."
                        ),
                    },
                    {"summary": "Architecture review complete; recommendations: latency budget, embedding versioning, P99 monitoring", "urgency": "LOW"},
                ],
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 3 — Scenario 7: N=5 dense D2  (5 agents × 8 decisions = 40 total)
    # RQ4: Does the DACS advantage hold at higher decision density?
    # Agents: async scraper coder, federated-learning researcher,
    #         fraud-detection data engineer, flaky-test debugger,
    #         distributed-cache technical writer.
    # Maximally diverse domains → contamination signal is unambiguous.
    # ------------------------------------------------------------------
    "s7_n5_dense_d2": ScenarioSpec(
        scenario_id="s7_n5_dense_d2",
        agents=[
            # ── a1 : Async web-scraper refactor (CodeWriter-style) ──────────
            AgentSpec(
                agent_id="a1",
                agent_type="generic",
                task_description="refactor Python web scraper to async architecture with rate limiting",
                decision_points=[
                    DecisionPoint(
                        "concurrency model for async web scraper",
                        ["asyncio", "async", "coroutine"],
                    ),
                    DecisionPoint(
                        "error retry policy for HTTP failures in scraper",
                        ["exponential backoff", "backoff", "retry"],
                    ),
                    DecisionPoint(
                        "HTTP session management for scraper connection reuse",
                        ["aiohttp", "session", "connection pool"],
                    ),
                    DecisionPoint(
                        "response cache storage backend for scraper",
                        ["redis", "ttl", "cache"],
                    ),
                    DecisionPoint(
                        "pagination cursor strategy for scraper",
                        ["cursor", "next page token", "offset"],
                    ),
                    DecisionPoint(
                        "rate limiting mechanism for scraper politeness",
                        ["token bucket", "semaphore", "leaky bucket"],
                    ),
                    DecisionPoint(
                        "async scraper test harness approach",
                        ["pytest", "aioresponses", "mock"],
                    ),
                    DecisionPoint(
                        "deployment packaging for async scraper",
                        ["docker", "container", "dockerfile"],
                    ),
                ],
                steps=[
                    {"summary": "Analysing existing sync scraper; planning async migration", "urgency": "LOW"},
                    {
                        "summary": "Need to choose concurrency model for async rewrite",
                        "urgency": "MEDIUM",
                        "question": (
                            "For concurrency model for async web scraper: "
                            "asyncio coroutines with aiohttp (native Python async, best ecosystem support), "
                            "threading with ThreadPoolExecutor (familiar but GIL-limited), "
                            "or multiprocessing (true parallelism but high overhead per URL)? "
                            "The scraper is I/O-bound; asyncio coroutines are the idiomatic choice."
                        ),
                    },
                    {"summary": "Async structure scaffolded; handling transient HTTP errors", "urgency": "LOW"},
                    {
                        "summary": "Need error retry policy for HTTP 429 and 5xx responses",
                        "urgency": "HIGH",
                        "question": (
                            "For error retry policy for HTTP failures in scraper: "
                            "exponential backoff with jitter (standard for distributed HTTP clients), "
                            "fixed-interval retry (simple but risks thundering herd), "
                            "or circuit breaker (appropriate for microservices, not scraping)? "
                            "Exponential backoff prevents server overload on rate-limit errors."
                        ),
                    },
                    {"summary": "Retry logic done; optimising TCP connection reuse across requests", "urgency": "LOW"},
                    {
                        "summary": "Need HTTP session pooling strategy to avoid per-request overhead",
                        "urgency": "MEDIUM",
                        "question": (
                            "For HTTP session management for scraper connection reuse: "
                            "single aiohttp ClientSession shared across all coroutines (recommended — reuses TCP), "
                            "per-domain session pool (more control, more complexity), "
                            "or httpx AsyncClient (alternative library, less ecosystem integration)? "
                            "A single shared aiohttp session is the standard async scraper pattern."
                        ),
                    },
                    {"summary": "Session pool in place; adding response caching to reduce duplicate fetches", "urgency": "LOW"},
                    {
                        "summary": "Need cache storage backend for scraped responses",
                        "urgency": "MEDIUM",
                        "question": (
                            "For response cache storage backend for scraper: "
                            "Redis with TTL expiry (distributed, fast, TTL-native), "
                            "in-process dict (fast but lost on restart, no TTL), "
                            "or SQLite (persistent but adds latency for every cache hit)? "
                            "Redis with TTL is the standard choice for scrapers needing freshness control."
                        ),
                    },
                    {"summary": "Cache added; implementing multi-page traversal", "urgency": "LOW"},
                    {
                        "summary": "Need pagination strategy for API and HTML next-page traversal",
                        "urgency": "MEDIUM",
                        "question": (
                            "For pagination cursor strategy for scraper: "
                            "cursor/next-page-token following (stateless, resilient to insertions), "
                            "offset pagination (simple but brittle on live data), "
                            "or date-range chunking (good for time-series APIs but coupling to schema)? "
                            "Cursor-based pagination is the most robust strategy for live datasets."
                        ),
                    },
                    {"summary": "Pagination implemented; need to respect robots.txt politeness limits", "urgency": "LOW"},
                    {
                        "summary": "Need rate limiting to avoid banning and comply with crawl delay",
                        "urgency": "HIGH",
                        "question": (
                            "For rate limiting mechanism for scraper politeness: "
                            "token bucket algorithm with asyncio.Semaphore (smooth bursty traffic), "
                            "fixed delay between requests (simple but idle under quota), "
                            "or leaky bucket (strictly timed, less burst tolerance)? "
                            "Token bucket with semaphore is preferred for async scrapers needing burst headroom."
                        ),
                    },
                    {"summary": "Rate limiter done; building test coverage", "urgency": "LOW"},
                    {
                        "summary": "Need to decide async test harness approach",
                        "urgency": "MEDIUM",
                        "question": (
                            "For async scraper test harness approach: "
                            "pytest-asyncio with aioresponses to mock HTTP responses (idiomatic async testing), "
                            "responses library with sync adapters (does not support async natively), "
                            "or live integration tests against a local HTTP server (slow, not unit-testable)? "
                            "pytest-asyncio with aioresponses is the standard async scraper testing stack."
                        ),
                    },
                    {"summary": "Tests written and passing; packaging for deployment", "urgency": "LOW"},
                    {
                        "summary": "Need deployment packaging decision for the async scraper service",
                        "urgency": "MEDIUM",
                        "question": (
                            "For deployment packaging for async scraper: "
                            "Docker container with Dockerfile (portable, reproducible, CI-friendly), "
                            "Python package on PyPI (appropriate for libraries, not services), "
                            "or serverless function (cold-start latency is prohibitive for persistent scraping)? "
                            "Docker container is the standard packaging choice for always-on scraper services."
                        ),
                    },
                    {"summary": "Async scraper refactor complete; dockerised and deployed", "urgency": "LOW"},
                ],
            ),

            # ── a2 : Federated-learning literature review (Research-style) ──
            AgentSpec(
                agent_id="a2",
                agent_type="generic",
                task_description="write 20-page federated learning literature review with taxonomy",
                decision_points=[
                    DecisionPoint(
                        "paper inclusion criteria for federated learning review",
                        ["federated", "privacy", "distributed learning"],
                    ),
                    DecisionPoint(
                        "taxonomy structure for federated learning survey",
                        ["horizontal", "vertical", "federated transfer"],
                    ),
                    DecisionPoint(
                        "primary comparison axis for federated learning systems",
                        ["communication efficiency", "communication rounds", "round"],
                    ),
                    DecisionPoint(
                        "privacy comparison axis for federated learning",
                        ["differential privacy", "secure aggregation", "dp"],
                    ),
                    DecisionPoint(
                        "positioning of FedAvg as seminal paper",
                        ["mcmahan", "fedavg", "communication-efficient"],
                    ),
                    DecisionPoint(
                        "gap narrative for federated learning survey",
                        ["non-iid", "heterogeneous", "data heterogeneity"],
                    ),
                    DecisionPoint(
                        "future work framing for federated learning",
                        ["personalization", "personalized federated", "per-client"],
                    ),
                    DecisionPoint(
                        "conclusion emphasis for federated learning review",
                        ["privacy-accuracy tradeoff", "tradeoff", "privacy vs accuracy"],
                    ),
                ],
                steps=[
                    {"summary": "Query planning: searching ACM, IEEE, arxiv for federated learning papers 2016-2025", "urgency": "LOW"},
                    {
                        "summary": "1,200 candidate papers found; need inclusion criteria for review scope",
                        "urgency": "HIGH",
                        "question": (
                            "For paper inclusion criteria for federated learning review: "
                            "include only papers explicitly proposing federated/distributed-on-device methods with "
                            "privacy constraints (tight scope, ~200 papers), "
                            "include all distributed ML papers (too broad, 1,200+ papers), "
                            "or include only papers citing McMahan 2017 FedAvg (citation-tree approach, risks missing independent lineages)? "
                            "Restricting to federated learning with privacy constraints keeps the review tractable."
                        ),
                    },
                    {"summary": "200 papers selected; designing taxonomy structure for classification", "urgency": "LOW"},
                    {
                        "summary": "Need taxonomy structure for organising federated learning literature",
                        "urgency": "MEDIUM",
                        "question": (
                            "For taxonomy structure for federated learning survey: "
                            "horizontal / vertical / federated transfer learning split (matches data partition type — standard FL taxonomy), "
                            "by application domain (medical, financial, mobile — less technically descriptive), "
                            "or by algorithm family (optimisation-based, distillation-based, etc.)? "
                            "Horizontal/vertical/federated-transfer is the accepted FL taxonomy in the literature."
                        ),
                    },
                    {"summary": "Taxonomy finalised; writing Section 3 system comparison", "urgency": "LOW"},
                    {
                        "summary": "Need primary technical axis for comparing FL algorithms in the survey",
                        "urgency": "MEDIUM",
                        "question": (
                            "For primary comparison axis for federated learning systems: "
                            "communication efficiency / number of communication rounds (central bottleneck in FL — McMahan's framing), "
                            "model accuracy on IID data (ignores FL's unique challenge), "
                            "or wall-clock training time (hardware-dependent, hard to reproduce)? "
                            "Communication efficiency and rounds are the standard primary comparison axis in FL surveys."
                        ),
                    },
                    {"summary": "Section 3 written with communication-efficiency axis; writing privacy section", "urgency": "LOW"},
                    {
                        "summary": "Need privacy comparison axis for Section 4 of the survey",
                        "urgency": "MEDIUM",
                        "question": (
                            "For privacy comparison axis for federated learning: "
                            "differential privacy / secure aggregation guarantees (formal privacy budget ε, δ — quantifiable), "
                            "informal privacy (no data leaves device — not formally verifiable), "
                            "or k-anonymity (traditional database metric, not standard in FL)? "
                            "Differential privacy with ε/δ budget is the standard formal privacy axis in FL surveys."
                        ),
                    },
                    {"summary": "Privacy section written; positioning FedAvg in historical narrative", "urgency": "LOW"},
                    {
                        "summary": "Need to determine how to frame FedAvg's role in the survey narrative",
                        "urgency": "MEDIUM",
                        "question": (
                            "For positioning of FedAvg as seminal paper: "
                            "open Section 2 with McMahan et al. FedAvg (2017) as the founding algorithm "
                            "that defined communication-efficient FL (standard survey positioning), "
                            "treat it as one of many parallel works (under-credits its centrality), "
                            "or mention it only in a footnote (inappropriate given its citation count)? "
                            "McMahan et al. FedAvg is universally treated as the foundational FL algorithm."
                        ),
                    },
                    {"summary": "Historical narrative written; identifying open problems", "urgency": "LOW"},
                    {
                        "summary": "Need to frame the primary open gap that motivates future work",
                        "urgency": "HIGH",
                        "question": (
                            "For gap narrative for federated learning survey: "
                            "non-IID / heterogeneous data problem (most-cited open challenge — skewed local distributions degrade global model), "
                            "communication cost reduction (largely addressed by FedAvg and compression), "
                            "or GPU hardware availability at edge (infrastructure, not algorithmic)? "
                            "Data heterogeneity / non-IID is consistently identified as the primary open challenge in FL."
                        ),
                    },
                    {"summary": "Open challenges section written; drafting future directions", "urgency": "LOW"},
                    {
                        "summary": "Need to frame the most promising future research direction",
                        "urgency": "MEDIUM",
                        "question": (
                            "For future work framing for federated learning: "
                            "personalized federated learning (custom per-client models — addresses non-IID directly), "
                            "centralised FL benchmarking (infrastructure, not research frontier), "
                            "or quantum-secure aggregation (speculative, no near-term path)? "
                            "Personalized/per-client FL is the dominant future research direction in current literature."
                        ),
                    },
                    {"summary": "Future work section written; drafting conclusion", "urgency": "LOW"},
                    {
                        "summary": "Need to decide the take-away message to emphasise in the conclusion",
                        "urgency": "MEDIUM",
                        "question": (
                            "For conclusion emphasis for federated learning review: "
                            "the privacy-accuracy tradeoff as the defining tension in FL (captures the core open problem), "
                            "scalability to millions of devices (engineering concern, less novel), "
                            "or the need for standardised benchmarks (process concern, not a scientific contribution)? "
                            "The privacy-accuracy tradeoff is the intellectually central tension to highlight in the conclusion."
                        ),
                    },
                    {"summary": "Literature review complete; 20 pages, 200 references, ready for submission", "urgency": "LOW"},
                ],
            ),

            # ── a3 : Fraud-detection feature pipeline (DataProcessor-style) ─
            AgentSpec(
                agent_id="a3",
                agent_type="generic",
                task_description="build real-time fraud detection feature pipeline with streaming windows",
                decision_points=[
                    DecisionPoint(
                        "streaming window type for fraud feature pipeline",
                        ["tumbling", "sliding", "hopping"],
                    ),
                    DecisionPoint(
                        "categorical feature encoding for fraud transaction data",
                        ["target encoding", "one-hot", "frequency encoding"],
                    ),
                    DecisionPoint(
                        "null value imputation strategy for fraud pipeline",
                        ["median", "impute", "mode"],
                    ),
                    DecisionPoint(
                        "high-cardinality categorical handling in fraud features",
                        ["hashing", "hash trick", "feature hashing"],
                    ),
                    DecisionPoint(
                        "outlier clipping method for transaction amount features",
                        ["iqr", "winsoriz", "percentile clip"],
                    ),
                    DecisionPoint(
                        "class imbalance strategy for fraud detection model",
                        ["smote", "oversample", "class weight"],
                    ),
                    DecisionPoint(
                        "validation split strategy for fraud time-series data",
                        ["temporal", "time-based", "chronological"],
                    ),
                    DecisionPoint(
                        "model deployment artifact format for fraud pipeline",
                        ["onnx", "joblib", "pickle"],
                    ),
                ],
                steps=[
                    {"summary": "Designing streaming fraud feature pipeline architecture with Kafka source", "urgency": "LOW"},
                    {
                        "summary": "Need to select streaming window type for aggregation features",
                        "urgency": "HIGH",
                        "question": (
                            "For streaming window type for fraud feature pipeline: "
                            "tumbling windows (non-overlapping fixed intervals — clean aggregation per period), "
                            "sliding windows (overlapping — more sensitive to recent transactions, higher compute), "
                            "or session windows (gap-based — appropriate for user-session features)? "
                            "Tumbling windows are the standard first choice for fraud rate features aggregated per hour/day."
                        ),
                    },
                    {"summary": "Window aggregation implemented; encoding categorical merchant and card features", "urgency": "LOW"},
                    {
                        "summary": "Need categorical encoding for merchant category and card type features",
                        "urgency": "MEDIUM",
                        "question": (
                            "For categorical feature encoding for fraud transaction data: "
                            "target encoding using fraud rate per category (captures fraud signal directly), "
                            "one-hot encoding (safe but very high dimensionality for merchant IDs), "
                            "or label encoding (creates false ordinal relationships)? "
                            "Target encoding with out-of-fold estimation is preferred for fraud categorical features."
                        ),
                    },
                    {"summary": "Categorical encoding done; handling missing values in merchant data", "urgency": "LOW"},
                    {
                        "summary": "15% of merchant_category values are null; need imputation strategy",
                        "urgency": "MEDIUM",
                        "question": (
                            "For null value imputation strategy for fraud pipeline: "
                            "median imputation for numeric features (robust to outliers from fraudulent amounts), "
                            "mean imputation (sensitive to outlier fraud amounts — not appropriate), "
                            "or drop rows with nulls (loses 15% of training data — too much)? "
                            "Median imputation is the standard for skewed fraud transaction amounts."
                        ),
                    },
                    {"summary": "Nulls handled; addressing high-cardinality features like merchant_id", "urgency": "LOW"},
                    {
                        "summary": "merchant_id has 500K+ unique values; need cardinality reduction strategy",
                        "urgency": "HIGH",
                        "question": (
                            "For high-cardinality categorical handling in fraud features: "
                            "feature hashing / hash trick (fixed-size projection, handles unseen merchants at inference), "
                            "top-N encoding (keeps only top 1000 merchants — misses long-tail fraud), "
                            "or target encoding all 500K values (memory explosion, target leakage risk)? "
                            "Feature hashing is the standard solution for very high-cardinality IDs in ML pipelines."
                        ),
                    },
                    {"summary": "High-cardinality handled; addressing transaction amount outliers", "urgency": "LOW"},
                    {
                        "summary": "Transaction amounts have extreme outliers (max = $2M); need clipping strategy",
                        "urgency": "MEDIUM",
                        "question": (
                            "For outlier clipping method for transaction amount features: "
                            "IQR-based clipping at 1.5× IQR boundaries (robust, data-driven bounds), "
                            "fixed domain cutoff at $10,000 (business-rule based, not statistically derived), "
                            "or log transform (reduces skew but doesn't remove outliers entirely)? "
                            "IQR-based clipping / winsorisation is the standard robust outlier treatment."
                        ),
                    },
                    {"summary": "Outlier treatment done; addressing severe class imbalance (0.1% fraud rate)", "urgency": "LOW"},
                    {
                        "summary": "Dataset is 99.9% non-fraud; need class imbalance handling for model training",
                        "urgency": "HIGH",
                        "question": (
                            "For class imbalance strategy for fraud detection model: "
                            "SMOTE oversampling of minority fraud class (synthetic examples, widely used for fraud), "
                            "random undersampling of majority class (loses non-fraud patterns), "
                            "or class_weight='balanced' in the classifier (no data change, adjusts loss — simplest)? "
                            "SMOTE or class_weight=balanced are both standard; SMOTE is preferred when training data is large enough."
                        ),
                    },
                    {"summary": "Imbalance addressed; setting up model validation framework", "urgency": "LOW"},
                    {
                        "summary": "Need validation split strategy that respects time ordering of transactions",
                        "urgency": "MEDIUM",
                        "question": (
                            "For validation split strategy for fraud time-series data: "
                            "temporal / time-based split (train on older transactions, validate on recent — prevents data leakage), "
                            "random 80/20 split (leaks future fraud patterns into training — invalid for fraud), "
                            "or k-fold cross-validation (violates temporal ordering — overestimates real performance)? "
                            "A temporal/chronological split is mandatory for fraud detection to prevent future data leakage."
                        ),
                    },
                    {"summary": "Validation strategy set; packaging model for deployment", "urgency": "LOW"},
                    {
                        "summary": "Model trained; need to decide deployment artifact format for inference service",
                        "urgency": "MEDIUM",
                        "question": (
                            "For model deployment artifact format for fraud pipeline: "
                            "ONNX (framework-agnostic, optimisable, supported by inference engines), "
                            "pickle (Python-only, version-sensitive, security risk in production), "
                            "or joblib (Python-only, better for sklearn pipelines but not cross-language)? "
                            "ONNX is the preferred production artifact format for fraud models requiring low latency."
                        ),
                    },
                    {"summary": "Fraud detection pipeline complete; ONNX model deployed to inference service", "urgency": "LOW"},
                ],
            ),

            # ── a4 : Flaky test suite debugger ──────────────────────────────
            AgentSpec(
                agent_id="a4",
                agent_type="generic",
                task_description="diagnose and fix test suite with 40 percent flakiness rate",
                decision_points=[
                    DecisionPoint(
                        "flakiness root cause classification approach",
                        ["timing", "order-dependent", "async"],
                    ),
                    DecisionPoint(
                        "flaky test isolation strategy",
                        ["quarantine", "isolat", "separate"],
                    ),
                    DecisionPoint(
                        "service boundary approach for flaky tests",
                        ["mock", "stub", "test double"],
                    ),
                    DecisionPoint(
                        "shared state fix for parallel test flakiness",
                        ["thread-safe", "race condition", "shared state"],
                    ),
                    DecisionPoint(
                        "test ordering dependency detection",
                        ["randomize", "shuffle", "random order"],
                    ),
                    DecisionPoint(
                        "environment leak cleanup in test fixtures",
                        ["teardown", "fixture", "cleanup"],
                    ),
                    DecisionPoint(
                        "flaky test assertion tolerance strategy",
                        ["retry", "rerun", "pytest-rerunfailures"],
                    ),
                    DecisionPoint(
                        "flakiness monitoring and prevention in CI",
                        ["quarantine tag", "flaky marker", "xfail"],
                    ),
                ],
                steps=[
                    {"summary": "Analysing 500 failing test runs to classify flakiness patterns", "urgency": "LOW"},
                    {
                        "summary": "Three observed failure signatures: timing-sensitive, order-dependent, and async callback races",
                        "urgency": "HIGH",
                        "question": (
                            "For flakiness root cause classification approach: "
                            "timing-dependent failures (race conditions, sleep-based waits — most common in async services), "
                            "order-dependent failures (tests rely on state left by previous test — second most common), "
                            "or environment-dependent failures (CI vs local env differences — usually fewer cases)? "
                            "Timing/async race conditions and order dependency are the two dominant flakiness root causes."
                        ),
                    },
                    {"summary": "Root causes classified; need to isolate flaky tests to prevent CI blocking", "urgency": "LOW"},
                    {
                        "summary": "40% of CI runs failing; need strategy to isolate flaky tests immediately",
                        "urgency": "HIGH",
                        "question": (
                            "For flaky test isolation strategy: "
                            "quarantine flaky tests into a separate CI job (unblocks main pipeline while fixes are made), "
                            "delete all failing tests immediately (loses coverage), "
                            "or disable the full test suite until all tests are fixed (stops shipping)? "
                            "Quarantining into a separate CI job is the standard industry approach to flaky tests."
                        ),
                    },
                    {"summary": "Flaky tests quarantined; investigating external service dependencies", "urgency": "LOW"},
                    {
                        "summary": "12 tests call live external APIs causing non-deterministic failures",
                        "urgency": "MEDIUM",
                        "question": (
                            "For service boundary approach for flaky tests: "
                            "replace live API calls with mocks/stubs/test doubles (deterministic, fast, no network), "
                            "use a dedicated test environment with real services (slow, flaky, expensive), "
                            "or skip tests that need external services (loses coverage of integration behaviour)? "
                            "Mocking external service boundaries is the standard fix for network-dependent test flakiness."
                        ),
                    },
                    {"summary": "External services mocked; investigating parallel test runner shared state", "urgency": "LOW"},
                    {
                        "summary": "pytest-xdist parallel runs show intermittent failures not present in serial mode",
                        "urgency": "HIGH",
                        "question": (
                            "For shared state fix for parallel test flakiness: "
                            "audit and eliminate all shared mutable state / race conditions between worker processes, "
                            "disable parallel execution (fixes flakiness but slows CI significantly), "
                            "or add random sleeps between tests (does not fix root cause)? "
                            "Eliminating shared state and race conditions is the correct fix for parallel test flakiness."
                        ),
                    },
                    {"summary": "Shared state fixed; checking for hidden test ordering dependencies", "urgency": "LOW"},
                    {
                        "summary": "Some tests pass in file order but fail randomly; need to detect ordering deps",
                        "urgency": "MEDIUM",
                        "question": (
                            "For test ordering dependency detection: "
                            "randomize test execution order (pytest-randomly) to expose hidden dependencies, "
                            "always run tests in alphabetical order (masks ordering deps), "
                            "or manual code review for global state (slow and misses dynamic dependencies)? "
                            "Randomising test order with pytest-randomly is the standard method to surface ordering dependencies."
                        ),
                    },
                    {"summary": "Ordering dependencies found and fixed; cleaning up fixture teardown", "urgency": "LOW"},
                    {
                        "summary": "Database and temp-file fixtures not always cleaned up between tests",
                        "urgency": "MEDIUM",
                        "question": (
                            "For environment leak cleanup in test fixtures: "
                            "add explicit teardown / yield-based fixtures to guarantee cleanup (pytest best practice), "
                            "rely on garbage collection (non-deterministic, not safe for DB connections), "
                            "or delete test artifacts in setUp only (previous test's teardown gap remains)? "
                            "Yield-based pytest fixtures with teardown are the correct pattern for guaranteed cleanup."
                        ),
                    },
                    {"summary": "Fixtures fixed; handling remaining intermittent assertion failures", "urgency": "LOW"},
                    {
                        "summary": "5 tests still flaky after all fixes; likely genuine timing variance at assertion boundaries",
                        "urgency": "MEDIUM",
                        "question": (
                            "For flaky test assertion tolerance strategy: "
                            "retry/rerun failures up to 3 times with pytest-rerunfailures (accepted practice for genuine timing variance), "
                            "increase assertion timeout thresholds (may mask real slowdowns), "
                            "or mark as xfail (hides the flakiness without fixing or tracking it)? "
                            "pytest-rerunfailures with a 3-retry limit is the accepted practice for unavoidable timing-sensitive tests."
                        ),
                    },
                    {"summary": "Retry policy added; setting up long-term flakiness monitoring in CI", "urgency": "LOW"},
                    {
                        "summary": "Need CI-level strategy to prevent flakiness regression in future",
                        "urgency": "MEDIUM",
                        "question": (
                            "For flakiness monitoring and prevention in CI: "
                            "tag known flaky tests with @pytest.mark.flaky / quarantine marker and track in dedicated CI job, "
                            "fail the build on any new test flakiness (zero-tolerance — too noisy initially), "
                            "or ignore flakiness metrics (reverts back to the original problem)? "
                            "A quarantine-tag + dedicated-job approach is the standard CI flakiness governance pattern."
                        ),
                    },
                    {"summary": "Flakiness eliminated from main CI suite; monitoring in place. Test suite health: 99.2% pass rate.", "urgency": "LOW"},
                ],
            ),

            # ── a5 : Distributed-cache technical design document ─────────────
            AgentSpec(
                agent_id="a5",
                agent_type="generic",
                task_description="write technical design document for distributed in-memory cache system",
                decision_points=[
                    DecisionPoint(
                        "primary audience for distributed cache design document",
                        ["engineer", "technical", "developer"],
                    ),
                    DecisionPoint(
                        "consistency model for distributed cache",
                        ["eventual consistency", "eventual", "strong consistency"],
                    ),
                    DecisionPoint(
                        "eviction policy for distributed cache",
                        ["lru", "least recently used", "lfu"],
                    ),
                    DecisionPoint(
                        "cache partitioning strategy for distributed cache",
                        ["consistent hashing", "hash ring", "rendezvous"],
                    ),
                    DecisionPoint(
                        "replication strategy for distributed cache high availability",
                        ["primary-replica", "replica", "replication factor"],
                    ),
                    DecisionPoint(
                        "failure mode handling for network partition in cache",
                        ["split-brain", "partition tolerance", "fencing"],
                    ),
                    DecisionPoint(
                        "cache invalidation strategy",
                        ["write-through", "write-behind", "invalidation"],
                    ),
                    DecisionPoint(
                        "observability instrumentation for distributed cache",
                        ["hit rate", "cache hit", "eviction rate"],
                    ),
                ],
                steps=[
                    {"summary": "Scoping the distributed cache TDD; listing system requirements", "urgency": "LOW"},
                    {
                        "summary": "Need to define primary audience for the document",
                        "urgency": "MEDIUM",
                        "question": (
                            "For primary audience for distributed cache design document: "
                            "backend engineers / developers who will implement and operate the system (most useful — concrete decisions), "
                            "executive leadership (wrong level — no implementation detail), "
                            "or academic reviewers (wrong style — not an academic paper)? "
                            "Technical design documents target engineers and developers who will build and operate the system."
                        ),
                    },
                    {"summary": "Audience defined; writing Section 2 consistency model", "urgency": "LOW"},
                    {
                        "summary": "Need to choose and document the consistency model for the cache",
                        "urgency": "HIGH",
                        "question": (
                            "For consistency model for distributed cache: "
                            "eventual consistency (high availability, low latency, stale reads acceptable for most cache workloads), "
                            "strong consistency (linearisable reads — high cost, reduced availability, needed for financial data), "
                            "or causal consistency (middle ground — complex to implement for a cache)? "
                            "Eventual consistency is the standard choice for distributed caches where stale reads are acceptable."
                        ),
                    },
                    {"summary": "Consistency model documented; writing eviction policy section", "urgency": "LOW"},
                    {
                        "summary": "Need to specify eviction policy for cache memory management",
                        "urgency": "MEDIUM",
                        "question": (
                            "For eviction policy for distributed cache: "
                            "LRU — Least Recently Used (evicts coldest entries, standard for general-purpose caches), "
                            "LFU — Least Frequently Used (better hit rate for frequency-skewed workloads but higher overhead), "
                            "or FIFO (ignores access patterns — generally inferior to LRU)? "
                            "LRU is the default eviction policy for most distributed caches including Redis and Memcached."
                        ),
                    },
                    {"summary": "Eviction policy documented; writing data partitioning section", "urgency": "LOW"},
                    {
                        "summary": "Need to document key-space partitioning strategy across cache nodes",
                        "urgency": "MEDIUM",
                        "question": (
                            "For cache partitioning strategy for distributed cache: "
                            "consistent hashing / hash ring (minimal key remapping on node add/remove — standard for distributed caches), "
                            "modulo hashing (simple but all keys remap when node count changes), "
                            "or range partitioning (sequential hot-spots — poor for cache workloads)? "
                            "Consistent hashing is the standard partitioning strategy for distributed caches."
                        ),
                    },
                    {"summary": "Partitioning documented; writing high-availability replication section", "urgency": "LOW"},
                    {
                        "summary": "Need to specify replication strategy for cache node failure tolerance",
                        "urgency": "MEDIUM",
                        "question": (
                            "For replication strategy for distributed cache high availability: "
                            "primary-replica replication with configurable replication factor (standard HA in Redis Cluster / Memcached), "
                            "synchronous multi-master (high consistency cost, not standard for caches), "
                            "or no replication (single-node — not HA)? "
                            "Primary-replica replication with replication factor is the standard HA model for distributed caches."
                        ),
                    },
                    {"summary": "Replication documented; writing failure mode analysis", "urgency": "LOW"},
                    {
                        "summary": "Need to document split-brain and partition handling under network failures",
                        "urgency": "HIGH",
                        "question": (
                            "For failure mode handling for network partition in cache: "
                            "split-brain prevention via fencing tokens + quorum writes (prevents inconsistent dual-primary), "
                            "let both partitions accept writes and reconcile on reconnect (high data conflict risk), "
                            "or disable writes during a partition (unacceptable availability regression for a cache)? "
                            "Fencing tokens and split-brain detection are the standard partition-handling mechanisms."
                        ),
                    },
                    {"summary": "Partition handling documented; writing cache invalidation strategy", "urgency": "LOW"},
                    {
                        "summary": "Need to document how cache entries are kept consistent with the source of truth",
                        "urgency": "MEDIUM",
                        "question": (
                            "For cache invalidation strategy: "
                            "write-through (update cache and DB synchronously on every write — strong consistency, low stale risk), "
                            "write-behind / write-back (async DB update — lower write latency, risk of data loss), "
                            "or manual invalidation on DB change (complex application-level logic, easy to miss)? "
                            "Write-through invalidation is the recommended strategy when cache consistency is critical."
                        ),
                    },
                    {"summary": "Invalidation strategy documented; writing observability section", "urgency": "LOW"},
                    {
                        "summary": "Need to specify which cache metrics to instrument for operations",
                        "urgency": "MEDIUM",
                        "question": (
                            "For observability instrumentation for distributed cache: "
                            "cache hit rate and eviction rate as primary metrics (directly measure cache effectiveness and memory pressure), "
                            "CPU utilisation per node (secondary — measures load, not cache quality), "
                            "or network bytes per second (infrastructure metric, not cache-specific)? "
                            "Cache hit rate and eviction rate are the primary observability metrics for a distributed cache."
                        ),
                    },
                    {"summary": "TDD complete: 8 sections, 4,500 words, covering consistency, partitioning, HA, invalidation, and observability", "urgency": "LOW"},
                ],
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # Phase 3 — Scenario 8: N=3 dense D3  (3 agents × 15 decisions = 45 total)
    # RQ4 primary: At D3 density (15 decisions/agent) the flat context has
    # accumulated thousands of tokens of per-agent history — the crossover
    # point where DACS advantage should be clearly non-linear.
    # Agents: ML training loop, iterative hypothesis tester, long-form writer.
    # All use GenericAgent so no additional agent classes are needed.
    # ------------------------------------------------------------------
    "s8_n3_dense_d3": ScenarioSpec(
        scenario_id="s8_n3_dense_d3",
        agents=[
            # ── a1 : ML model training loop (15 decisions) ─────────────────
            AgentSpec(
                agent_id="a1",
                agent_type="generic",
                task_description="iteratively train and tune BERT text classifier on legal document dataset",
                decision_points=[
                    DecisionPoint(
                        "base model architecture for legal text classifier",
                        ["bert", "transformer", "pretrained"],
                    ),
                    DecisionPoint(
                        "tokenizer choice for legal text classification",
                        ["wordpiece", "subword", "bpe"],
                    ),
                    DecisionPoint(
                        "sequence length handling for long legal documents",
                        ["512", "truncat", "sliding window"],
                    ),
                    DecisionPoint(
                        "batch size strategy for GPU memory constraint",
                        ["gradient accumulation", "micro-batch", "accumulation"],
                    ),
                    DecisionPoint(
                        "learning rate schedule for BERT fine-tuning",
                        ["warmup", "cosine", "linear warmup"],
                    ),
                    DecisionPoint(
                        "optimizer for BERT fine-tuning",
                        ["adamw", "weight decay", "adam"],
                    ),
                    DecisionPoint(
                        "dropout strategy to reduce overfitting on legal corpus",
                        ["dropout", "regulariz", "layer dropout"],
                    ),
                    DecisionPoint(
                        "early stopping criterion for training loop",
                        ["validation loss", "val_loss", "patience"],
                    ),
                    DecisionPoint(
                        "class imbalance handling for legal document categories",
                        ["focal loss", "class weight", "weighted"],
                    ),
                    DecisionPoint(
                        "text augmentation for low-resource legal categories",
                        ["back-translation", "synonym", "paraphrase"],
                    ),
                    DecisionPoint(
                        "evaluation metric for legal document classification",
                        ["macro f1", "f1", "weighted f1"],
                    ),
                    DecisionPoint(
                        "model checkpoint save strategy",
                        ["best model", "lowest val", "checkpoint"],
                    ),
                    DecisionPoint(
                        "classification head design for BERT output",
                        ["linear", "classification head", "dense layer"],
                    ),
                    DecisionPoint(
                        "inference optimization for production deployment",
                        ["onnx", "quantiz", "distil"],
                    ),
                    DecisionPoint(
                        "model versioning and registry for production",
                        ["mlflow", "model registry", "versioning"],
                    ),
                ],
                steps=[
                    {"summary": "Task briefing: fine-tune a classifier for 12 legal document categories on 50K labelled examples", "urgency": "LOW"},
                    {
                        "summary": "Need to choose base architecture for the legal classification task",
                        "urgency": "HIGH",
                        "question": (
                            "For base model architecture for legal text classifier: "
                            "BERT or a legal-domain pretrained transformer (LegalBERT) — leverages pretraining on legal corpora, "
                            "training a BiLSTM from scratch (no transfer learning, weaker baseline), "
                            "or bag-of-words TF-IDF + SVM (fast but misses contextual semantics in legal text)? "
                            "A pretrained BERT / transformer model is the standard starting point for legal NLP tasks."
                        ),
                    },
                    {"summary": "BERT selected; configuring tokenization pipeline", "urgency": "LOW"},
                    {
                        "summary": "BERT expects a specific tokenizer; need to confirm choice",
                        "urgency": "MEDIUM",
                        "question": (
                            "For tokenizer choice for legal text classification: "
                            "BERT WordPiece tokenizer matched to the pretrained model (correct — tokenizer must match model vocabulary), "
                            "spaCy tokenizer with whitespace splitting (incompatible with BERT's vocabulary), "
                            "or character-level tokenizer (very long sequences — explodes memory for legal documents)? "
                            "WordPiece / subword tokenizer matching the pretrained BERT model is mandatory."
                        ),
                    },
                    {"summary": "Tokenizer configured; handling long legal documents that exceed 512 tokens", "urgency": "LOW"},
                    {
                        "summary": "Legal documents average 1,800 tokens — exceeds BERT 512 limit",
                        "urgency": "HIGH",
                        "question": (
                            "For sequence length handling for long legal documents: "
                            "truncate to 512 tokens at start-of-document (captures header/title — works well for classification), "
                            "chunk into 512-token sliding windows and aggregate (better recall but 3× compute), "
                            "or use Longformer/BigBird with 4096-token context (higher memory, different model)? "
                            "Truncating to 512 tokens is the standard baseline for legal document classification with BERT."
                        ),
                    },
                    {"summary": "Sequence truncation set; configuring training batch sizes on 16GB GPU", "urgency": "LOW"},
                    {
                        "summary": "Batch size 32 causes OOM on 16GB GPU; need memory management strategy",
                        "urgency": "MEDIUM",
                        "question": (
                            "For batch size strategy for GPU memory constraint: "
                            "gradient accumulation over 8 micro-batches of 4 (effective batch=32, GPU-safe), "
                            "reduce to batch size 4 (wastes GPU bandwidth, poor gradient estimates), "
                            "or use mixed precision only and keep batch 32 (may still OOM for some layers)? "
                            "Gradient accumulation with micro-batches is the standard solution for large-batch BERT training on limited GPU."
                        ),
                    },
                    {"summary": "Gradient accumulation configured; setting up learning rate schedule", "urgency": "LOW"},
                    {
                        "summary": "Need learning rate schedule to stabilise BERT fine-tuning convergence",
                        "urgency": "MEDIUM",
                        "question": (
                            "For learning rate schedule for BERT fine-tuning: "
                            "linear warmup followed by cosine or linear decay (standard BERT fine-tuning recipe), "
                            "constant LR throughout (leads to unstable early training with large BERT weights), "
                            "or cyclical LR (useful for training from scratch, less needed for fine-tuning)? "
                            "Linear warmup with cosine/linear decay is the recommended schedule for all BERT fine-tuning."
                        ),
                    },
                    {"summary": "LR schedule set; choosing optimizer", "urgency": "LOW"},
                    {
                        "summary": "Need to pick optimizer with appropriate weight decay for BERT",
                        "urgency": "MEDIUM",
                        "question": (
                            "For optimizer for BERT fine-tuning: "
                            "AdamW (Adam with decoupled weight decay — the standard for all transformer fine-tuning), "
                            "vanilla Adam (weight decay is coupled — less regularisation), "
                            "or SGD with momentum (requires careful LR tuning, not standard for transformers)? "
                            "AdamW with decoupled weight decay is the universal optimizer for BERT fine-tuning."
                        ),
                    },
                    {"summary": "AdamW configured; mitigating overfitting on 50K legal examples", "urgency": "LOW"},
                    {
                        "summary": "Validation accuracy plateaus at epoch 3 while train continues rising — overfitting signal",
                        "urgency": "HIGH",
                        "question": (
                            "For dropout strategy to reduce overfitting on legal corpus: "
                            "increase dropout rate in classification head and enable hidden-layer dropout (standard regularisation), "
                            "add more training data via augmentation only (helpful but doesn't directly add regularisation), "
                            "or reduce model size to DistilBERT (changes the experiment baseline)? "
                            "Adding dropout regularisation in the classifier head is the standard first response to overfitting in BERT."
                        ),
                    },
                    {"summary": "Dropout tuned; configuring training stopping criteria", "urgency": "LOW"},
                    {
                        "summary": "Training has run 10 epochs; need to define when to stop",
                        "urgency": "MEDIUM",
                        "question": (
                            "For early stopping criterion for training loop: "
                            "stop when validation loss has not improved for N=3 epochs (patience-based — prevents overfitting), "
                            "stop at fixed epoch count regardless of val loss (may overfit or underfit), "
                            "or stop when training loss reaches 0.01 (training loss, not validation — overfitting indicator)? "
                            "Validation loss with patience-based early stopping is the standard training loop termination criterion."
                        ),
                    },
                    {"summary": "Early stopping configured; handling class imbalance (rare category = 0.8% of data)", "urgency": "LOW"},
                    {
                        "summary": "One legal category ('Arbitration Clause') has only 0.8% prevalence — model ignores it",
                        "urgency": "HIGH",
                        "question": (
                            "For class imbalance handling for legal document categories: "
                            "focal loss (down-weights easy majority class examples, focuses on hard minority class), "
                            "class_weight='balanced' (simpler, adjusts loss scaling by inverse frequency), "
                            "or oversample minority class in dataloader (risk of memorising minority examples)? "
                            "Focal loss or class_weight=balanced are the standard approaches; focal loss is preferred for severe long-tail imbalance."
                        ),
                    },
                    {"summary": "Class imbalance addressed; augmenting low-resource legal categories", "urgency": "LOW"},
                    {
                        "summary": "Some categories have <200 training examples; need augmentation to improve recall",
                        "urgency": "MEDIUM",
                        "question": (
                            "For text augmentation for low-resource legal categories: "
                            "back-translation (translate to French/German and back — preserves legal semantics, widely used), "
                            "random word deletion (may remove legally significant terms — risky for legal text), "
                            "or EDA (Easy Data Augmentation — synonym replacement; acceptable if domain vocab preserved)? "
                            "Back-translation is the preferred augmentation for legal text — it preserves meaning better than synonym replacement."
                        ),
                    },
                    {"summary": "Augmentation added; choosing final evaluation metric", "urgency": "LOW"},
                    {
                        "summary": "Need to select evaluation metric for reporting model quality across 12 classes",
                        "urgency": "MEDIUM",
                        "question": (
                            "For evaluation metric for legal document classification: "
                            "macro-averaged F1 (treats all 12 classes equally — appropriate given rare legal categories matter), "
                            "accuracy (dominated by majority classes — hides poor recall on rare categories), "
                            "or micro-averaged F1 (also dominated by majority class — similar problem to accuracy)? "
                            "Macro F1 is the standard metric when all classes (including rare ones) must be correctly classified."
                        ),
                    },
                    {"summary": "Macro F1 set as primary metric; configuring model checkpointing", "urgency": "LOW"},
                    {
                        "summary": "Need checkpoint save strategy to preserve best model across epochs",
                        "urgency": "LOW",
                        "question": (
                            "For model checkpoint save strategy: "
                            "save the model with the best validation loss / highest macro F1 (standard — keeps the generalising checkpoint), "
                            "save every epoch (disk usage grows linearly — wasteful), "
                            "or save only the last epoch (may not be the best model if early stopping kicks in)? "
                            "Saving the best-validation-loss checkpoint is the standard training loop practice."
                        ),
                    },
                    {"summary": "Checkpointing configured; designing classification head", "urgency": "LOW"},
                    {
                        "summary": "Need to design the classification head on top of BERT [CLS] pooled output",
                        "urgency": "MEDIUM",
                        "question": (
                            "For classification head design for BERT output: "
                            "linear layer from hidden_size to num_classes (standard — minimal parameters, works well for fine-tuning), "
                            "two-layer MLP with ReLU activation (adds capacity — may overfit on 50K examples), "
                            "or multi-head attention over token embeddings (expensive, not standard for classification)? "
                            "A single linear classification head on the [CLS] token is the standard BERT fine-tuning architecture."
                        ),
                    },
                    {"summary": "Model training complete; macro F1=0.89 on held-out test set; optimising for inference", "urgency": "LOW"},
                    {
                        "summary": "Model must serve <50ms p99 latency; need inference optimisation",
                        "urgency": "HIGH",
                        "question": (
                            "For inference optimization for production deployment: "
                            "ONNX export with quantisation (framework-agnostic, optimisable, 2–4× speedup), "
                            "TorchScript (PyTorch-only, some speedup, less portable), "
                            "or knowledge distillation to DistilBERT (reduces model size but requires retraining)? "
                            "ONNX with INT8 quantisation is the standard path to production BERT inference latency targets."
                        ),
                    },
                    {"summary": "ONNX export done; registering model for production versioning", "urgency": "LOW"},
                    {
                        "summary": "Need to track model version, parameters, and metrics for reproducibility",
                        "urgency": "MEDIUM",
                        "question": (
                            "For model versioning and registry for production: "
                            "MLflow model registry (tracks experiment runs, parameters, metrics, and artifact versions), "
                            "git-based versioning of model files (diffs not meaningful for binary model files), "
                            "or manual naming convention in S3 (brittle, hard to query by metric)? "
                            "MLflow model registry is the standard solution for tracking ML model versions and experiments."
                        ),
                    },
                    {"summary": "Model registered in MLflow. Legal text classifier pipeline complete. Macro F1=0.89, p99 latency=38ms.", "urgency": "LOW"},
                ],
            ),

            # ── a2 : Iterative hypothesis tester (15 decisions) ─────────────
            AgentSpec(
                agent_id="a2",
                agent_type="generic",
                task_description="run iterative statistical hypothesis testing on clinical trial dataset",
                decision_points=[
                    DecisionPoint(
                        "statistical test for continuous primary outcome",
                        ["t-test", "student", "welch"],
                    ),
                    DecisionPoint(
                        "normality check method before parametric test",
                        ["shapiro", "kolmogorov", "normality test"],
                    ),
                    DecisionPoint(
                        "multiple comparison correction for family of tests",
                        ["bonferroni", "fdr", "benjamini"],
                    ),
                    DecisionPoint(
                        "effect size measure for clinical significance",
                        ["cohen", "d statistic", "effect size"],
                    ),
                    DecisionPoint(
                        "confidence interval width for reporting",
                        ["95%", "95 percent", "confidence interval"],
                    ),
                    DecisionPoint(
                        "outlier handling for clinical measurements",
                        ["winsoriz", "iqr", "robust"],
                    ),
                    DecisionPoint(
                        "missing data imputation for clinical trial",
                        ["multiple imputation", "mice", "impute"],
                    ),
                    DecisionPoint(
                        "sample size justification method",
                        ["power", "statistical power", "power calculation"],
                    ),
                    DecisionPoint(
                        "covariate adjustment strategy",
                        ["confounder", "covariate", "ancova"],
                    ),
                    DecisionPoint(
                        "interaction testing for subgroup effects",
                        ["interaction term", "moderation", "interaction"],
                    ),
                    DecisionPoint(
                        "pre-specified subgroup analysis plan",
                        ["stratif", "subgroup", "pre-specified"],
                    ),
                    DecisionPoint(
                        "sensitivity analysis for robustness",
                        ["sensitivity", "robustness check", "exclusion criteria"],
                    ),
                    DecisionPoint(
                        "reporting standard for clinical trial analysis",
                        ["consort", "strobe", "reporting guideline"],
                    ),
                    DecisionPoint(
                        "meta-analysis eligibility determination",
                        ["heterogeneity", "i-squared", "cochran"],
                    ),
                    DecisionPoint(
                        "conclusion framing for clinical significance",
                        ["clinical significance", "clinically meaningful", "practical significance"],
                    ),
                ],
                steps=[
                    {"summary": "Dataset loaded: 400 patients, treatment vs control, primary outcome = systolic BP reduction at 12 weeks", "urgency": "LOW"},
                    {
                        "summary": "Need to select the primary statistical test for BP reduction comparison",
                        "urgency": "HIGH",
                        "question": (
                            "For statistical test for continuous primary outcome: "
                            "two-sample t-test / Welch's t-test (appropriate for continuous normal outcome, treatment vs control), "
                            "Mann-Whitney U test (non-parametric — appropriate if normality fails), "
                            "or chi-squared test (wrong — for categorical outcomes, not continuous BP)? "
                            "A two-sample Welch's t-test is the appropriate primary test for continuous BP reduction."
                        ),
                    },
                    {"summary": "t-test selected; need to verify normality assumption before proceeding", "urgency": "LOW"},
                    {
                        "summary": "Normality must be confirmed before using parametric t-test",
                        "urgency": "MEDIUM",
                        "question": (
                            "For normality check method before parametric test: "
                            "Shapiro-Wilk test (most powerful normality test for n<2000 — standard in clinical stats), "
                            "visual histogram inspection only (not a formal test — insufficient for reporting), "
                            "or skip normality check (invalid for RCT analysis plan)? "
                            "The Shapiro-Wilk test is the standard normality check for clinical trial datasets."
                        ),
                    },
                    {"summary": "Normality confirmed (p=0.31); proceeding with t-test on primary outcome", "urgency": "LOW"},
                    {
                        "summary": "Testing 8 secondary outcomes simultaneously — need multiple comparison strategy",
                        "urgency": "HIGH",
                        "question": (
                            "For multiple comparison correction for family of tests: "
                            "Benjamini-Hochberg FDR correction (controls false discovery rate — preferred for exploratory secondaries), "
                            "Bonferroni correction (controls FWER — overly conservative for 8 tests, reduces power), "
                            "or no correction (inflates Type I error — not acceptable in clinical reporting)? "
                            "Benjamini-Hochberg FDR is the recommended correction for multiple secondary outcomes in clinical trials."
                        ),
                    },
                    {"summary": "FDR correction applied; primary outcome p=0.003 (significant); quantifying effect magnitude", "urgency": "LOW"},
                    {
                        "summary": "Statistical significance established; need to report clinical effect size",
                        "urgency": "MEDIUM",
                        "question": (
                            "For effect size measure for clinical significance: "
                            "Cohen's d (standardised mean difference — interpretable across studies, meta-analysable), "
                            "p-value only (conveys significance, not magnitude — insufficient for clinical decisions), "
                            "or raw mean difference in mmHg (intuitive but not standardised across studies)? "
                            "Cohen's d is the standard effect size measure for continuous outcomes in clinical trial reporting."
                        ),
                    },
                    {"summary": "Cohen's d = 0.61 (medium effect); reporting interval estimate", "urgency": "LOW"},
                    {
                        "summary": "Need to determine confidence interval width for the primary outcome estimate",
                        "urgency": "LOW",
                        "question": (
                            "For confidence interval width for reporting: "
                            "95% confidence intervals (standard in clinical reporting — required by CONSORT), "
                            "90% CI (less conservative — used in some regulatory submissions but not standard clinical), "
                            "or 99% CI (over-conservative for standard clinical journals)? "
                            "95% confidence intervals are the mandatory standard for clinical trial outcome reporting."
                        ),
                    },
                    {"summary": "95% CIs computed; checking for extreme values in BP measurements", "urgency": "LOW"},
                    {
                        "summary": "Three patients have BP readings >3 SD from mean; need outlier handling decision",
                        "urgency": "MEDIUM",
                        "question": (
                            "For outlier handling for clinical measurements: "
                            "Winsorise at 1st/99th percentile (shrinks extreme values, preserves sample size, robust), "
                            "delete outlier rows (reduces N, may bias results if outliers are treatment responders), "
                            "or report results with and without outliers in a sensitivity analysis? "
                            "Winsorisation is the standard robust outlier treatment for clinical continuous outcomes."
                        ),
                    },
                    {"summary": "Outliers Winsorised; addressing 8% missing outcome data", "urgency": "LOW"},
                    {
                        "summary": "8% of primary outcome measurements missing at 12 weeks; need imputation strategy",
                        "urgency": "HIGH",
                        "question": (
                            "For missing data imputation for clinical trial: "
                            "multiple imputation (MICE / chained equations — gold standard for RCT missing data under MAR assumption), "
                            "last observation carried forward / LOCF (conservative but biased under missing-not-at-random), "
                            "or complete case analysis (drops 8% of patients — reduces power and may bias)? "
                            "Multiple imputation (MICE) is the CONSORT-recommended approach for missing data in RCTs."
                        ),
                    },
                    {"summary": "MICE imputation done; validating the study's statistical power", "urgency": "LOW"},
                    {
                        "summary": "Reviewer likely to ask: was study powered to detect the observed effect?",
                        "urgency": "MEDIUM",
                        "question": (
                            "For sample size justification method: "
                            "report a priori power calculation (target β=0.80, α=0.05 — required by CONSORT), "
                            "report achieved power post-hoc only (post-hoc power is circular and discouraged), "
                            "or omit sample size justification (will be rejected by any peer-reviewed journal)? "
                            "An a priori power calculation with specified α, β, and effect size is required by CONSORT."
                        ),
                    },
                    {"summary": "Power calculation documented; adjusting for known confounders", "urgency": "LOW"},
                    {
                        "summary": "Age and baseline BP differ slightly between arms; need covariate adjustment",
                        "urgency": "MEDIUM",
                        "question": (
                            "For covariate adjustment strategy: "
                            "ANCOVA with age and baseline BP as covariates (standard adjustment for pre-specified confounders in RCTs), "
                            "post-stratification weighting (complex, rarely used in simple two-arm RCTs), "
                            "or no adjustment (ignores known imbalance — reviewers will object)? "
                            "ANCOVA covariate adjustment for pre-specified confounders is the standard RCT analysis approach."
                        ),
                    },
                    {"summary": "ANCOVA done; testing for treatment heterogeneity across subgroups", "urgency": "LOW"},
                    {
                        "summary": "Sponsor wants to know if effect differs by age group (<65 vs ≥65); need interaction test",
                        "urgency": "MEDIUM",
                        "question": (
                            "For interaction testing for subgroup effects: "
                            "formal treatment × age-group interaction term in regression model (correct — test for moderation), "
                            "report subgroup t-tests separately without interaction term (incorrect — inflates Type I error), "
                            "or skip subgroup analysis entirely (misses potential effect modifier)? "
                            "A formal interaction term test is the correct method for subgroup moderation analysis."
                        ),
                    },
                    {"summary": "Interaction test non-significant (p=0.41); proceeding to subgroup reporting plan", "urgency": "LOW"},
                    {
                        "summary": "Need to define which subgroup analyses are pre-specified vs exploratory",
                        "urgency": "MEDIUM",
                        "question": (
                            "For pre-specified subgroup analysis plan: "
                            "clearly stratify pre-specified subgroups (from protocol) vs post-hoc exploratory (required by CONSORT), "
                            "report all subgroups as equal (misleads readers about confirmatory vs exploratory status), "
                            "or skip all subgroup analyses (loses potentially valuable secondary information)? "
                            "Pre-specified vs exploratory stratification is required by CONSORT Sub-group reporting guidelines."
                        ),
                    },
                    {"summary": "Subgroup plan documented; planning sensitivity analyses for reviewer robustness queries", "urgency": "LOW"},
                    {
                        "summary": "Need sensitivity analyses to test robustness of primary result to analysis choices",
                        "urgency": "MEDIUM",
                        "question": (
                            "For sensitivity analysis for robustness: "
                            "re-run primary analysis with different outlier threshold / different imputation model (tests assumption sensitivity), "
                            "re-run on different random seeds only (not a sensitivity analysis of assumptions), "
                            "or omit sensitivity analysis (leaves results open to reviewer criticism on assumption choices)? "
                            "Sensitivity analyses varying outlier treatment and imputation assumptions are standard for clinical trial reporting."
                        ),
                    },
                    {"summary": "Sensitivity analyses complete — results robust; preparing manuscript reporting", "urgency": "LOW"},
                    {
                        "summary": "Need to confirm which reporting guideline to follow for publication",
                        "urgency": "LOW",
                        "question": (
                            "For reporting standard for clinical trial analysis: "
                            "CONSORT statement (Consolidated Standards of Reporting Trials — mandatory for most clinical journals), "
                            "STROBE (for observational studies — this is an RCT, wrong guideline), "
                            "or no formal guideline (rejected by all peer-reviewed clinical journals)? "
                            "CONSORT is the mandatory reporting standard for randomised controlled trials."
                        ),
                    },
                    {"summary": "CONSORT checklist completed; assessing meta-analysis candidacy", "urgency": "LOW"},
                    {
                        "summary": "Trial may be included in a systematic review; need to assess meta-analysis suitability",
                        "urgency": "MEDIUM",
                        "question": (
                            "For meta-analysis eligibility determination: "
                            "assess between-study heterogeneity using Cochran's Q and I² statistic (standard meta-analysis tools), "
                            "assume homogeneity without testing (invalid — studies may use different populations), "
                            "or exclude from meta-analysis without justification (premature, misses evidence synthesis benefit)? "
                            "Cochran's Q and I² are the standard tools for assessing heterogeneity in meta-analysis."
                        ),
                    },
                    {"summary": "Meta-analysis eligibility confirmed (I²=23%, low heterogeneity); writing conclusion", "urgency": "LOW"},
                    {
                        "summary": "Need to frame clinical significance appropriately in the conclusion",
                        "urgency": "HIGH",
                        "question": (
                            "For conclusion framing for clinical significance: "
                            "distinguish statistical significance (p=0.003) from clinical significance "
                            "(a 6.2 mmHg reduction is clinically meaningful based on guideline thresholds — both must be stated), "
                            "report p-value only (does not establish clinical relevance), "
                            "or claim clinical significance without reference to guideline thresholds (unsupported)? "
                            "Clinical significance must be grounded in guideline thresholds and Cohen's d, separate from the p-value."
                        ),
                    },
                    {"summary": "Analysis complete. Primary: BP reduction 6.2 mmHg (p=0.003, d=0.61, 95% CI [4.1, 8.3]). Clinically meaningful per JNC 8 guidelines.", "urgency": "LOW"},
                ],
            ),

            # ── a3 : Long-form technical writer — post-quantum crypto (15 decisions) ─
            AgentSpec(
                agent_id="a3",
                agent_type="generic",
                task_description="write 15-section technical whitepaper on post-quantum cryptography for enterprise",
                decision_points=[
                    DecisionPoint(
                        "target audience framing for post-quantum cryptography whitepaper",
                        ["enterprise", "practitioner", "technical"],
                    ),
                    DecisionPoint(
                        "PQC algorithm coverage selection",
                        ["crystals", "kyber", "dilithium"],
                    ),
                    DecisionPoint(
                        "NIST standardisation depth for PQC whitepaper",
                        ["nist", "nist pqc", "standardiz"],
                    ),
                    DecisionPoint(
                        "migration timeline framing for PQC adoption",
                        ["hybrid", "crypto-agility", "migration"],
                    ),
                    DecisionPoint(
                        "threat model coverage for PQC threat narrative",
                        ["harvest now", "store now decrypt later", "y2q"],
                    ),
                    DecisionPoint(
                        "PQC implementation library recommendation",
                        ["liboqs", "bouncycastle", "library"],
                    ),
                    DecisionPoint(
                        "risk assessment model for PQC enterprise adoption",
                        ["attack surface", "threat model", "risk"],
                    ),
                    DecisionPoint(
                        "TLS integration coverage for PQC deployment",
                        ["tls 1.3", "post-quantum tls", "hybrid kem"],
                    ),
                    DecisionPoint(
                        "certificate and PKI handling for PQC transition",
                        ["x509", "certificate", "pki"],
                    ),
                    DecisionPoint(
                        "hardware security module coverage for PQC key management",
                        ["hsm", "hardware security module", "key management"],
                    ),
                    DecisionPoint(
                        "compliance framework alignment for PQC whitepaper",
                        ["nist sp 800", "fips", "compliance"],
                    ),
                    DecisionPoint(
                        "migration cost estimation approach",
                        ["migration cost", "tco", "total cost"],
                    ),
                    DecisionPoint(
                        "vendor landscape coverage strategy",
                        ["pqshield", "quantinuum", "vendor"],
                    ),
                    DecisionPoint(
                        "appendix design for PQC whitepaper",
                        ["glossary", "appendix", "reference"],
                    ),
                    DecisionPoint(
                        "executive summary emphasis for PQC whitepaper",
                        ["business risk", "ciso", "executive"],
                    ),
                ],
                steps=[
                    {"summary": "Project brief received: 15-section PQC enterprise whitepaper, 25-35 pages", "urgency": "LOW"},
                    {
                        "summary": "Need to define the primary target audience to calibrate technical depth",
                        "urgency": "HIGH",
                        "question": (
                            "For target audience framing for post-quantum cryptography whitepaper: "
                            "enterprise security practitioners / CISOs and architects (technical but not cryptographers — most useful target), "
                            "academic cryptographers (too narrow for a commercial whitepaper), "
                            "or general business audience (too low technical depth for actionable PQC guidance)? "
                            "Enterprise security practitioners are the primary audience for a PQC adoption whitepaper."
                        ),
                    },
                    {"summary": "Audience defined; selecting PQC algorithms to cover in Section 2", "urgency": "LOW"},
                    {
                        "summary": "Need to decide which PQC algorithms to cover — NIST has standardised several",
                        "urgency": "HIGH",
                        "question": (
                            "For PQC algorithm coverage selection: "
                            "CRYSTALS-Kyber (KEM) and CRYSTALS-Dilithium (signatures) — the primary NIST-standardised algorithms (FIPS 203/204), "
                            "all lattice-based, code-based, and hash-based candidates (too broad for a practitioner whitepaper), "
                            "or only hash-based signatures like XMSS (misses the primary KEMs)? "
                            "CRYSTALS-Kyber and CRYSTALS-Dilithium are the NIST-standardised algorithms every enterprise PQC whitepaper must cover."
                        ),
                    },
                    {"summary": "Algorithm selection made; writing Section 3 on NIST standardisation process", "urgency": "LOW"},
                    {
                        "summary": "Need to decide how much depth to give the NIST PQC standardisation process",
                        "urgency": "MEDIUM",
                        "question": (
                            "For NIST standardisation depth for PQC whitepaper: "
                            "cover the NIST PQC standardisation process and final standards (FIPS 203/204/205) in one focused section "
                            "(gives practitioners the compliance anchor they need), "
                            "deep-dive the full 6-year competition (too long for practitioner whitepaper), "
                            "or mention NIST in passing only (insufficient — enterprises need compliance framing)? "
                            "A focused section on NIST PQC standards and FIPS references is essential for enterprise compliance framing."
                        ),
                    },
                    {"summary": "NIST section drafted; writing Section 4 on migration timeline and strategy", "urgency": "LOW"},
                    {
                        "summary": "Need to frame the enterprise migration roadmap — hybrid vs full cutover",
                        "urgency": "HIGH",
                        "question": (
                            "For migration timeline framing for PQC adoption: "
                            "hybrid cryptography approach first (run classical + PQC in parallel — crypto-agility, backward compat), "
                            "immediate full cutover to PQC (breaks backward compatibility, un-deployable for most enterprises), "
                            "or wait until quantum computers exist (harvest-now attacks make this too late)? "
                            "Hybrid cryptography / crypto-agility is the universally recommended migration approach for PQC."
                        ),
                    },
                    {"summary": "Migration framing written; writing Section 5 threat model", "urgency": "LOW"},
                    {
                        "summary": "Need to define the threat narrative — what attack makes PQC urgent now?",
                        "urgency": "HIGH",
                        "question": (
                            "For threat model coverage for PQC threat narrative: "
                            "'harvest now, decrypt later' / 'store now, decrypt later' (adversaries archive encrypted data today to decrypt when quantum computers exist — the urgent threat), "
                            "'quantum computers already exist and can break RSA' (not yet true — overstates current risk), "
                            "or 'long-term data has no quantum risk' (incorrect — government and health data has 10–20 year sensitivity windows)? "
                            "The harvest-now / store-now-decrypt-later threat is the correct framing for why PQC migration is urgent today."
                        ),
                    },
                    {"summary": "Threat narrative written; writing Section 6 implementation guidance", "urgency": "LOW"},
                    {
                        "summary": "Need to recommend specific open-source libraries for PQC implementation",
                        "urgency": "MEDIUM",
                        "question": (
                            "For PQC implementation library recommendation: "
                            "liboqs (Open Quantum Safe — open-source, actively maintained, supports all NIST algorithms in C/Python/Java), "
                            "write PQC algorithms from scratch (impractical, insecure, not recommended), "
                            "or BouncyCastle PQC extension for JVM (valid for Java ecosystem but not cross-platform)? "
                            "liboqs (Open Quantum Safe) is the primary recommended library for cross-platform PQC implementation."
                        ),
                    },
                    {"summary": "Implementation section written; writing Section 7 risk assessment model", "urgency": "LOW"},
                    {
                        "summary": "Need risk framework section for enterprises to self-assess PQC exposure",
                        "urgency": "MEDIUM",
                        "question": (
                            "For risk assessment model for PQC enterprise adoption: "
                            "attack surface + threat model framework (classify assets by quantum sensitivity window and cryptographic exposure), "
                            "financial cost model only (misses technical risk dimensions), "
                            "or qualitative risk matrix without asset mapping (too vague for security teams)? "
                            "An attack-surface and threat-model-based risk framework is the standard for enterprise security assessments."
                        ),
                    },
                    {"summary": "Risk model written; writing Section 8 TLS integration", "urgency": "LOW"},
                    {
                        "summary": "Need to cover how enterprises deploy PQC in their TLS stack",
                        "urgency": "MEDIUM",
                        "question": (
                            "For TLS integration coverage for PQC deployment: "
                            "hybrid KEM in TLS 1.3 (X25519Kyber768 — supported in OQS-OpenSSL, the current deployment path), "
                            "TLS 1.2 with PQC cipher suites (deprecated protocol — not recommended), "
                            "or skip TLS coverage (TLS is the primary enterprise deployment surface for KEM)? "
                            "Hybrid KEM in TLS 1.3 (post-quantum TLS) is the current deployment path for enterprise PQC."
                        ),
                    },
                    {"summary": "TLS section written; writing Section 9 certificate and PKI transition", "urgency": "LOW"},
                    {
                        "summary": "Need to cover how X.509 certificates and PKI infrastructure adapt for PQC",
                        "urgency": "MEDIUM",
                        "question": (
                            "For certificate and PKI handling for PQC transition: "
                            "X.509 certificates with PQC signature algorithms (Dilithium in cert — IETF draft in progress), "
                            "custom certificate format (incompatible with existing PKI infrastructure), "
                            "or reuse RSA certificates unchanged (defeats purpose of PQC migration)? "
                            "X.509 certificates with PQC signatures are the standardised path for PKI transition."
                        ),
                    },
                    {"summary": "PKI section written; writing Section 10 HSM and key management", "urgency": "LOW"},
                    {
                        "summary": "Enterprises need guidance on whether existing HSMs support PQC algorithms",
                        "urgency": "MEDIUM",
                        "question": (
                            "For hardware security module coverage for PQC key management: "
                            "cover HSM PQC support status (major vendors — Thales, nShield — have firmware updates for Kyber/Dilithium), "
                            "recommend replacing all HSMs immediately (impractical, expensive), "
                            "or skip HSM coverage (HSMs are the central key management control for enterprises — must be included)? "
                            "HSM PQC support status and key management migration are essential content for enterprise practitioners."
                        ),
                    },
                    {"summary": "HSM section written; writing Section 11 compliance alignment", "urgency": "LOW"},
                    {
                        "summary": "Enterprises need to understand which compliance frameworks reference PQC requirements",
                        "urgency": "MEDIUM",
                        "question": (
                            "For compliance framework alignment for PQC whitepaper: "
                            "NIST SP 800-208 and upcoming FIPS 203/204/205 (direct compliance references for US federal and regulated industries), "
                            "GDPR only (data protection, not cryptographic algorithm specifications), "
                            "or ISO 27001 (general security management, not algorithm-specific)? "
                            "NIST SP 800 guidelines and FIPS standards are the primary compliance reference for PQC in regulated industries."
                        ),
                    },
                    {"summary": "Compliance section written; writing Section 12 migration cost estimation", "urgency": "LOW"},
                    {
                        "summary": "Enterprises will ask: what does PQC migration actually cost?",
                        "urgency": "MEDIUM",
                        "question": (
                            "For migration cost estimation approach: "
                            "total cost of ownership (TCO) framework: inventory + cryptographic bill of materials + phased migration cost, "
                            "single flat cost estimate (too imprecise — varies enormously by organisation size), "
                            "or no cost guidance (enterprises need business case input — omitting this reduces whitepaper value)? "
                            "A TCO framework with inventory-based migration cost is the appropriate approach for enterprise CFO/CISO audiences."
                        ),
                    },
                    {"summary": "Cost section written; writing Section 13 vendor landscape", "urgency": "LOW"},
                    {
                        "summary": "Need to cover the commercial PQC vendor ecosystem for procurement decisions",
                        "urgency": "MEDIUM",
                        "question": (
                            "For vendor landscape coverage strategy: "
                            "cover major PQC vendors: PQShield, Quantinuum, SandboxAQ, IBM Quantum-safe (gives procurement map), "
                            "endorse a single vendor (commercially inappropriate for an independent whitepaper), "
                            "or omit vendor landscape (misses key practitioner use case: who to buy from)? "
                            "Covering major PQC vendors (PQShield, Quantinuum, SandboxAQ) gives practitioners the procurement landscape they need."
                        ),
                    },
                    {"summary": "Vendor section written; designing appendix", "urgency": "LOW"},
                    {
                        "summary": "Appendix needs to support both quick-reference and deep-dive readers",
                        "urgency": "LOW",
                        "question": (
                            "For appendix design for PQC whitepaper: "
                            "include a glossary of PQC terms and a reference list of standards / further reading (most useful for practitioners), "
                            "include raw algorithm parameter tables only (too technical for most readers), "
                            "or no appendix (reduces utility as a reference document)? "
                            "A glossary and standards reference list is the standard appendix content for practitioner whitepapers."
                        ),
                    },
                    {"summary": "Appendix drafted; writing executive summary", "urgency": "LOW"},
                    {
                        "summary": "Executive summary must motivate action for CISO and board-level stakeholders",
                        "urgency": "HIGH",
                        "question": (
                            "For executive summary emphasis for PQC whitepaper: "
                            "business risk framing — harvest-now threat creates urgency, migration cost is manageable, CISO action items defined "
                            "(drives executive decision-making), "
                            "technical algorithm comparison (wrong level for executive summary), "
                            "or regulatory obligation only (necessary but not sufficient — risk framing more compelling)? "
                            "A business risk / CISO-focused executive summary is the correct framing for board-level PQC adoption decisions."
                        ),
                    },
                    {"summary": "PQC whitepaper complete: 15 sections, 30 pages, glossary, standards references, vendor landscape, executive summary.", "urgency": "LOW"},
                ],
            ),
        ],
    ),
}
