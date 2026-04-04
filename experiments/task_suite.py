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
}
