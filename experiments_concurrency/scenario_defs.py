"""Scenario definitions for the concurrency & interruption experiment.

Two scenarios exercise three simultaneous stressors:
  (1) competing concurrent HIGH-urgency agent steering requests,
  (2) user message interruptions fired mid-trial via a timed injector,
  (3) a heterogeneous mix of agent domains that maximises contamination risk.

Each ConcurrencyScenario wraps a list of AgentSpec objects (reusing
task_suite.AgentSpec/DecisionPoint directly) and a user_injection_schedule
that the UserInjector coroutine consumes.

All agents use agent_type="generic" with explicit steps so every question
is configurable and clearly answerable — required for in-line LLM judge
scoring.

Timing design (cc1_n3)
-----------------------
  t=0.00  All agents start; step-0 (no question) fires immediately.
  t≈0.10  step-1 fires for all three agents simultaneously.
           a2 and a3 emit HIGH-urgency requests → queue collision.
           Orchestrator handles a2 first (HIGH front-of-queue), then
           encounters a3 HIGH → INTERRUPT event logged.
  t≈0.30  User injection fires mid-FOCUS — orchestrator pauses its
           current focus session, responds, then resumes.
  t≈1.20  Second user injection during the mid-to-late steering phase.

Timing design (cc2_n5)
-----------------------
  t=0.00  All 5 agents start.
  t≈0.10  a3 fires HIGH (step1), a4 fires HIGH (step2).
  t≈0.20  User injection 1.
  t≈0.80  User injection 2 (peak queue contention for 5-agent load).
  t≈1.80  User injection 3 (late-trial, registry well-populated).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from experiments.task_suite import AgentSpec, DecisionPoint


# ---------------------------------------------------------------------------
# Extended scenario type
# ---------------------------------------------------------------------------

@dataclass
class ConcurrencyScenario:
    scenario_id: str
    agents: list[AgentSpec]
    # List of (delay_seconds_from_trial_start, message_text)
    user_injection_schedule: list[tuple[float, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# cc1_n3 — 3 agents, 5 DPs each, 2 user injections
#
# Domain mix: database design  ×  API security  ×  ML serving infra
# Competing HIGH urgency: a2 and a3 both fire HIGH at step-1 (~t=0.1s)
# ---------------------------------------------------------------------------

_CC1_A1_STEPS = [
    # step-0: initial work burst, no steering needed yet
    {
        "summary": "Analyzing query patterns and access paths for the e-commerce schema",
        "urgency": "MEDIUM",
        "sleep": 0.1,
    },
    # step-1: first decision
    {
        "summary": "Evaluating primary storage backend options",
        "urgency": "MEDIUM",
        "sleep": 0.0,
        "question": (
            "For the product catalog with full-text search requirements and complex "
            "JOIN-heavy reporting, should we choose PostgreSQL with pg_trgm extension "
            "or MongoDB as the primary storage backend?"
        ),
    },
    {
        "summary": "Designing connection management strategy",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "Should we route database connections through PgBouncer in transaction-pooling "
            "mode, or let each application worker open and manage its own psycopg2 connection?"
        ),
    },
    {
        "summary": "Evaluating partitioning strategy for the orders table",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "The orders table is projected to grow by 50k rows per day. Should we use "
            "range partitioning on created_at or hash partitioning on order_id as the "
            "partition key?"
        ),
    },
    {
        "summary": "Choosing session storage for authentication tokens",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "For storing user session tokens and rate-limit counters, should we use Redis "
            "as a separate in-memory store, or persist them directly in the PostgreSQL "
            "sessions table?"
        ),
    },
    {
        "summary": "Finalising schema migration tooling",
        "urgency": "MEDIUM",
        "sleep": 0.05,
        "question": (
            "Should we run schema migrations at application startup automatically, or as "
            "an explicit pre-deployment step isolated in CI/CD before the new container "
            "image goes live?"
        ),
    },
]

_CC1_A1_DPS = [
    DecisionPoint(
        question_fragment="PostgreSQL with pg_trgm extension or MongoDB",
        answer_keywords=["postgresql", "postgres", "pg_trgm"],
    ),
    DecisionPoint(
        question_fragment="PgBouncer in transaction-pooling mode",
        answer_keywords=["pgbouncer", "pool", "transaction-pooling"],
    ),
    DecisionPoint(
        question_fragment="range partitioning on created_at or hash partitioning",
        answer_keywords=["range", "range partition", "created_at"],
    ),
    DecisionPoint(
        question_fragment="Redis as a separate in-memory store",
        answer_keywords=["redis"],
    ),
    DecisionPoint(
        question_fragment="pre-deployment step isolated in CI/CD",
        answer_keywords=["pre-deployment", "separate", "ci", "cd"],
    ),
]


_CC1_A2_STEPS = [
    {
        "summary": "Running initial OWASP-ZAP scan on authentication endpoints",
        "urgency": "MEDIUM",
        "sleep": 0.1,
    },
    # --- HIGH urgency: JWT token storage vulnerability found ---
    {
        "summary": "Critical: JWT tokens stored in localStorage — XSS risk",
        "urgency": "HIGH",
        "sleep": 0.0,
        "question": (
            "We discovered that JWT access tokens are stored in localStorage, making them "
            "vulnerable to XSS attacks. Should we migrate to HttpOnly cookies for token "
            "storage, or keep localStorage but add strict CSP headers and input sanitisation?"
        ),
    },
    # --- HIGH urgency: CORS misconfiguration found ---
    {
        "summary": "Critical: CORS wildcard origin in production config detected",
        "urgency": "HIGH",
        "sleep": 0.0,
        "question": (
            "A CORS wildcard (Access-Control-Allow-Origin: *) was found active in the "
            "production API config. Should we immediately block all origins and fail "
            "closed, or deploy a strict allowlist of known frontend origins?"
        ),
    },
    {
        "summary": "Evaluating rate limiting strategy for authentication endpoints",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "The login and token-refresh endpoints currently have no rate limiting. "
            "Should we implement a token bucket algorithm or a sliding window counter "
            "to throttle authentication requests per IP?"
        ),
    },
    {
        "summary": "Auditing JWT validation placement",
        "urgency": "MEDIUM",
        "sleep": 0.05,
        "question": (
            "Should JWT signature validation and expiry checks happen in a centralised "
            "API gateway middleware layer, or be delegated to each downstream microservice "
            "independently?"
        ),
    },
]

_CC1_A2_DPS = [
    DecisionPoint(
        question_fragment="HttpOnly cookies for token storage",
        answer_keywords=["httponly", "http-only", "cookie"],
    ),
    DecisionPoint(
        question_fragment="block all origins and fail closed",
        answer_keywords=["allowlist", "allowlisted", "strict", "known"],
    ),
    DecisionPoint(
        question_fragment="token bucket algorithm or a sliding window",
        answer_keywords=["sliding window", "sliding-window"],
    ),
    DecisionPoint(
        question_fragment="centralised API gateway middleware layer",
        answer_keywords=["gateway", "middleware", "centralised", "centralized"],
    ),
]


_CC1_A3_STEPS = [
    {
        "summary": "Benchmarking latency requirements and SLA targets",
        "urgency": "LOW",
        "sleep": 0.1,
    },
    # --- HIGH urgency: model serving SLA breach risk ---
    {
        "summary": "Critical: p99 latency 180ms — far above 50ms SLA",
        "urgency": "HIGH",
        "sleep": 0.0,
        "question": (
            "The 500M-parameter recommendation model has p99 latency of 180ms, "
            "breaching our 50ms SLA. Should we apply TensorRT optimisation and "
            "INT8 quantisation, or switch to ONNX Runtime with dynamic batching "
            "as the primary latency reduction strategy?"
        ),
    },
    {
        "summary": "Designing request batching strategy for model server",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "Should the model server accumulate incoming inference requests in a "
            "micro-batch with a 5ms wait window before forwarding to the GPU, or "
            "forward each request immediately without batching?"
        ),
    },
    {
        "summary": "Evaluating deployment topology for serving layer",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "Should we deploy the model server as a sidecar container co-located "
            "with the API service pod, or as a standalone microservice with its "
            "own Kubernetes deployment and a dedicated gRPC interface?"
        ),
    },
    {
        "summary": "Designing A/B testing strategy for model versions",
        "urgency": "MEDIUM",
        "sleep": 0.05,
        "question": (
            "For safely rolling out a new recommendation model version, should we "
            "use a canary deployment (route 5% of live traffic to v2) or a shadow "
            "deployment (mirror traffic to v2 but discard predictions)?"
        ),
    },
]

_CC1_A3_DPS = [
    DecisionPoint(
        question_fragment="TensorRT optimisation and INT8 quantisation",
        answer_keywords=["tensorrt", "tensor rt", "int8", "quantis"],
    ),
    DecisionPoint(
        question_fragment="micro-batch with a 5ms wait window",
        answer_keywords=["batch", "micro-batch", "5ms"],
    ),
    DecisionPoint(
        question_fragment="standalone microservice with its own Kubernetes",
        answer_keywords=["standalone", "microservice", "grpc"],
    ),
    DecisionPoint(
        question_fragment="canary deployment (route 5% of live traffic",
        answer_keywords=["canary"],
    ),
]


CC1_N3 = ConcurrencyScenario(
    scenario_id="cc1_n3",
    agents=[
        AgentSpec(
            agent_id="a1",
            agent_type="generic",
            task_description="design database schema and query strategy for e-commerce platform",
            decision_points=_CC1_A1_DPS,
            steps=_CC1_A1_STEPS,
        ),
        AgentSpec(
            agent_id="a2",
            agent_type="generic",
            task_description="audit and harden REST API authentication and authorisation layer",
            decision_points=_CC1_A2_DPS,
            steps=_CC1_A2_STEPS,
        ),
        AgentSpec(
            agent_id="a3",
            agent_type="generic",
            task_description="design low-latency ML model serving infrastructure for real-time recommendations",
            decision_points=_CC1_A3_DPS,
            steps=_CC1_A3_STEPS,
        ),
    ],
    user_injection_schedule=[
        (
            0.3,
            "Can you give me a quick status update? Which agents are currently blocked "
            "waiting for a decision?",
        ),
        (
            1.2,
            "What is the most critical open decision right now and which agent owns it?",
        ),
    ],
)


# ---------------------------------------------------------------------------
# cc2_n5 — 5 agents, 4 DPs each, 3 user injections
#
# Domain mix: rate-limiter impl × LLM alignment research × churn features ×
#             memory leak debug × API documentation
# Competing HIGH urgency: a3 (churn, step-1) and a4 (debugger, step-3)
#                         fire during overlapping windows.
# ---------------------------------------------------------------------------

_CC2_A1_STEPS = [
    {
        "summary": "Reviewing Redis data structure options for counter atomicity",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "Should we implement the distributed rate limiter using a Redis fixed-window "
            "counter (INCR/EXPIRE) or a sliding-window log stored in a Redis sorted set "
            "with ZADD/ZRANGEBYSCORE?"
        ),
    },
    {
        "summary": "Designing the atomic counter update script",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "For guaranteeing atomicity of the check-then-increment operation in Redis, "
            "should we use a Lua script executed via EVAL, or a MULTI/EXEC transaction?"
        ),
    },
    {
        "summary": "Deciding counter granularity for rate limit keys",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "Should we key rate limit counters per client IP address only, or per "
            "IP-plus-endpoint combination to allow finer-grained per-route limits?"
        ),
    },
    {
        "summary": "Designing degraded-mode behaviour when Redis is unavailable",
        "urgency": "MEDIUM",
        "sleep": 0.05,
        "question": (
            "When the Redis cluster is unreachable, should the rate limiter fail open "
            "(allow all requests through) or fail closed (block all requests until "
            "Redis recovers)?"
        ),
    },
]

_CC2_A1_DPS = [
    DecisionPoint(
        question_fragment="sliding-window log stored in a Redis sorted set",
        answer_keywords=["sliding window", "sliding-window", "sorted set", "zadd"],
    ),
    DecisionPoint(
        question_fragment="Lua script executed via EVAL",
        answer_keywords=["lua", "eval"],
    ),
    DecisionPoint(
        question_fragment="per IP-plus-endpoint combination",
        answer_keywords=["ip+endpoint", "ip-plus-endpoint", "combination", "per-route"],
    ),
    DecisionPoint(
        question_fragment="fail open (allow all requests through)",
        answer_keywords=["fail open", "allow", "open"],
    ),
]


_CC2_A2_STEPS = [
    {
        "summary": "Scoping the RLHF vs constitutional AI coverage in the survey",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "Should the survey prioritise practical alignment techniques — RLHF, DPO, "
            "and Constitutional AI — or balance them equally with theoretical safety "
            "frameworks from MIRI and the ARC alignment team?"
        ),
    },
    {
        "summary": "Deciding depth of reward model training coverage",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "For the RLHF section, should we cover reward model architecture and training "
            "in technical depth, or focus primarily on the PPO policy optimisation step "
            "and treat the reward model as a black box?"
        ),
    },
    {
        "summary": "Positioning Constitutional AI relative to RLHF",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "Should Constitutional AI be framed in the survey as a superior replacement "
            "for RLHF in reducing harmful outputs, or as a complementary technique that "
            "can be layered on top of RLHF-trained models?"
        ),
    },
    {
        "summary": "Choosing presentation structure for technique comparison",
        "urgency": "LOW",
        "sleep": 0.05,
        "question": (
            "Should the survey present a side-by-side comparison table of alignment "
            "techniques (RLHF, DPO, CAI, RLAIF) along key axes, or present each "
            "technique in its own self-contained section without direct comparison?"
        ),
    },
]

_CC2_A2_DPS = [
    DecisionPoint(
        question_fragment="RLHF, DPO, and Constitutional AI",
        answer_keywords=["practical", "rlhf", "dpo", "constitutional"],
    ),
    DecisionPoint(
        question_fragment="reward model architecture and training in technical depth",
        answer_keywords=["reward model", "depth", "architecture"],
    ),
    DecisionPoint(
        question_fragment="complementary technique that can be layered",
        answer_keywords=["complementary", "layered", "both"],
    ),
    DecisionPoint(
        question_fragment="side-by-side comparison table of alignment techniques",
        answer_keywords=["comparison", "table", "side-by-side"],
    ),
]


_CC2_A3_STEPS = [
    {
        "summary": "Profiling raw feature distributions for churn signal quality",
        "urgency": "MEDIUM",
        "sleep": 0.1,
    },
    # HIGH urgency: dataset issue discovered
    {
        "summary": "Critical: 40% of customers have no purchase in the 30-day window",
        "urgency": "HIGH",
        "sleep": 0.0,
        "question": (
            "For the recency-frequency-monetary features, 40% of customers have zero "
            "activity in the 30-day window. Should we use a 90-day lookback window "
            "instead, or keep 30 days and separately engineer an explicit inactivity "
            "duration feature?"
        ),
    },
    {
        "summary": "Handling customers with no historical purchase data at all",
        "urgency": "MEDIUM",
        "sleep": 0.1,
        "question": (
            "Customers with absolutely no purchase history in the last 180 days — "
            "should we impute their activity features with zeros and include them in "
            "training, or drop them from the training set entirely to avoid label noise?"
        ),
    },
    {
        "summary": "Encoding strategy for high-cardinality product_category feature",
        "urgency": "MEDIUM",
        "sleep": 0.05,
        "question": (
            "The product_category feature has 500 distinct values. Should we use "
            "target encoding (mean-encode with cross-fold smoothing) or collapse "
            "rare categories and apply one-hot encoding?"
        ),
    },
]

_CC2_A3_DPS = [
    DecisionPoint(
        question_fragment="90-day lookback window instead",
        answer_keywords=["90", "90-day", "90 day"],
    ),
    DecisionPoint(
        question_fragment="drop them from the training set entirely",
        answer_keywords=["drop", "exclude", "remove"],
    ),
    DecisionPoint(
        question_fragment="target encoding (mean-encode with cross-fold smoothing)",
        answer_keywords=["target encoding", "target-encoding", "mean encode"],
    ),
]


_CC2_A4_STEPS = [
    {
        "summary": "Setting up profiling harness for memory leak investigation",
        "urgency": "LOW",
        "sleep": 0.1,
        "question": (
            "To start profiling the FastAPI service memory leak, should we use Python's "
            "built-in tracemalloc module for snapshot diffing, or attach a memory_profiler "
            "decorator to the suspected endpoint handler?"
        ),
    },
    {
        "summary": "Reproducing leak under concurrent load",
        "urgency": "LOW",
        "sleep": 0.15,
        "question": (
            "The leak only manifests under concurrent traffic. Should we reproduce it "
            "using Locust with a realistic load profile, or write an asyncio.gather "
            "stress test that fires 200 concurrent requests?"
        ),
    },
    # HIGH urgency: root cause found, need immediate decision
    {
        "summary": "Critical: tracemalloc shows 80% of leaked memory in aiohttp ClientSession objects",
        "urgency": "HIGH",
        "sleep": 0.0,
        "question": (
            "tracemalloc confirms 80% of leaked memory is in unclosed aiohttp ClientSession "
            "objects created per-request. Should we fix this by switching to a shared "
            "singleton ClientSession with connection pool reuse, or by ensuring every "
            "request uses an async context manager to guarantee session closure?"
        ),
    },
    {
        "summary": "Investigating latency regression after connection pooling fix",
        "urgency": "MEDIUM",
        "sleep": 0.05,
        "question": (
            "After introducing the shared session singleton, memory leak is resolved "
            "but p99 latency increased by 40%. Should we tune the aiohttp connector's "
            "limit and limit_per_host pool_size parameters, or investigate whether "
            "the shared session introduces connection contention under async load?"
        ),
    },
]

_CC2_A4_DPS = [
    DecisionPoint(
        question_fragment="tracemalloc module for snapshot diffing",
        answer_keywords=["tracemalloc"],
    ),
    DecisionPoint(
        question_fragment="Locust with a realistic load profile",
        answer_keywords=["locust"],
    ),
    DecisionPoint(
        question_fragment="shared singleton ClientSession with connection pool reuse",
        answer_keywords=["singleton", "shared session", "pool reuse"],
    ),
    DecisionPoint(
        question_fragment="limit_per_host pool_size parameters",
        answer_keywords=["pool_size", "limit", "tune"],
    ),
]


_CC2_A5_STEPS = [
    {
        "summary": "Planning API reference documentation structure",
        "urgency": "LOW",
        "sleep": 0.2,
        "question": (
            "Should we write the API reference by hand in Markdown and then generate "
            "an OpenAPI 3.0 spec from it, or write the OpenAPI spec first and auto-generate "
            "the human-readable reference from it?"
        ),
    },
    {
        "summary": "Designing the developer onboarding guide entry point",
        "urgency": "LOW",
        "sleep": 0.2,
        "question": (
            "Should the onboarding guide open with a working Hello World example that "
            "gets developers to a first successful API call in under 5 minutes, or "
            "start with a conceptual architecture overview before any code?"
        ),
    },
    {
        "summary": "Deciding documentation versioning strategy",
        "urgency": "LOW",
        "sleep": 0.15,
        "question": (
            "Should we version the API documentation separately in a dedicated docs "
            "repository, or keep documentation co-located in the same source repository "
            "as the code for docs-as-code synchronisation?"
        ),
    },
    {
        "summary": "Choosing code example language strategy",
        "urgency": "LOW",
        "sleep": 0.1,
        "question": (
            "For code examples throughout the docs, should we show a single canonical "
            "Python example per endpoint, or provide examples in multiple languages "
            "(Python, TypeScript, cURL) to serve a polyglot developer audience?"
        ),
    },
]

_CC2_A5_DPS = [
    DecisionPoint(
        question_fragment="OpenAPI spec first and auto-generate",
        answer_keywords=["openapi", "spec first", "api spec", "auto-generate"],
    ),
    DecisionPoint(
        question_fragment="working Hello World example",
        answer_keywords=["hello world", "working example", "5 minutes"],
    ),
    DecisionPoint(
        question_fragment="co-located in the same source repository",
        answer_keywords=["same repo", "same repository", "co-located", "docs-as-code"],
    ),
    DecisionPoint(
        question_fragment="multiple languages (Python, TypeScript, cURL)",
        answer_keywords=["multiple", "multiple languages", "polyglot"],
    ),
]


CC2_N5 = ConcurrencyScenario(
    scenario_id="cc2_n5",
    agents=[
        AgentSpec(
            agent_id="a1",
            agent_type="generic",
            task_description="implement distributed rate limiter with Redis and atomic Lua scripting",
            decision_points=_CC2_A1_DPS,
            steps=_CC2_A1_STEPS,
        ),
        AgentSpec(
            agent_id="a2",
            agent_type="generic",
            task_description="write survey on large language model alignment techniques (RLHF, DPO, CAI)",
            decision_points=_CC2_A2_DPS,
            steps=_CC2_A2_STEPS,
        ),
        AgentSpec(
            agent_id="a3",
            agent_type="generic",
            task_description="build feature engineering pipeline for customer churn prediction",
            decision_points=_CC2_A3_DPS,
            steps=_CC2_A3_STEPS,
        ),
        AgentSpec(
            agent_id="a4",
            agent_type="generic",
            task_description="isolate root cause of memory leak in Python FastAPI service under load",
            decision_points=_CC2_A4_DPS,
            steps=_CC2_A4_STEPS,
        ),
        AgentSpec(
            agent_id="a5",
            agent_type="generic",
            task_description="write API reference documentation and developer onboarding guide",
            decision_points=_CC2_A5_DPS,
            steps=_CC2_A5_STEPS,
        ),
    ],
    user_injection_schedule=[
        (
            0.2,
            "What decisions are currently pending? Are any agents blocked?",
        ),
        (
            0.8,
            "Are there any urgent blockers I should know about right now?",
        ),
        (
            1.8,
            "Give me a brief summary of what each agent has accomplished so far.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONCURRENCY_SCENARIOS: dict[str, ConcurrencyScenario] = {
    "cc1_n3": CC1_N3,
    "cc2_n5": CC2_N5,
}
