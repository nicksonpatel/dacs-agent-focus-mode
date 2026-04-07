# DACS — Interface Specifications

**Phase 2 Deliverable 2**  
**Last updated:** April 4, 2026

---

## Design Decisions (Resolved)

| Decision | Choice | Rationale |
|---|---|---|
| LLM provider | **MiniMax-M2.7** (Phases 1–3) / **Claude Haiku 4.5 via OpenRouter** (Phase 4) | Anthropic-compatible endpoint; tiktoken cl100k_base for counting |
| Tokenizer | **tiktoken** (`cl100k_base`) | Deterministic, fast, no GPU, matches encoding used in experiments |
| Concurrency model | **asyncio** | Agents are LLM-call-bound, not CPU-bound; cleaner event loop |
| Agent heartbeat trigger | **Event-based** (every step / status change) | Reproducible; avoids clock-drift across LLM latencies |
| Steering history depth K | **K=10** | Matches FocusContext spec; ablate to K=5 if budget is tight |
| Log format | **JSON lines** (`.jsonl`) | Structured, grep-able, directly parseable by experiment harness |
| Baseline context assembly | **Concatenate all agents** | Simpler; worst-case contamination; fair comparison |
| Token budget T | **T = 204,800** | MiniMax-M2.7 / Claude Haiku 4.5 supported limit; baseline needs this so it doesn't fail mechanically |
| Focus timeout | **60s wall clock or 3 LLM turns**, whichever first | Prevents stuck focus sessions; configurable via `focus_timeout` |

---

## Token Budget Analysis

| Mode | N=3 | N=5 | N=10 |
|---|---|---|---|
| **DACS FOCUS** | F(aᵢ) 4700 + registry 400 + sys 500 = **5,600** | F(aᵢ) 4700 + registry 800 + sys 500 = **6,000** | F(aᵢ) 4700 + registry 1800 + sys 500 = **7,000** |
| **Baseline FLAT** | 3 × 4700 + sys 500 = **14,600** | 5 × 4700 + sys 500 = **24,000** | 10 × 4700 + sys 500 = **47,500** |

F(aᵢ) estimate: 500 (task) + 10×400 (K=10 steering turns) + 200 (current request) = **4,700 tokens**  
Registry entry: 200 tokens max; compressed registry excludes the focus agent.

**All values well within T=204,800.** DACS uses ~7k tokens at N=10; baseline uses ~47.5k. The ~6.8× gap is the central experimental result.

---

## Component A — `RegistryManager` (`src/registry.py`)

### Responsibility

Single source of truth for all agent state. Thread-safe (asyncio lock) read/write for concurrent heartbeats. Enforces ≤200-token budget per entry at write time.

### Enums

```python
from enum import Enum

class AgentStatus(Enum):
    RUNNING            = "RUNNING"
    BLOCKED            = "BLOCKED"
    WAITING_STEERING   = "WAITING_STEERING"
    COMPLETE           = "COMPLETE"
    FAILED             = "FAILED"

class UrgencyLevel(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
```

### Data Schema

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class RegistryEntry:
    agent_id: str
    task_description: str          # ≤50 tokens enforced at write time
    status: AgentStatus
    last_output_summary: str       # ≤100 tokens enforced at write time; truncated if over
    last_updated: str              # ISO 8601 timestamp
    pending_steering_request: bool
    urgency: UrgencyLevel

@dataclass
class RegistryUpdate:
    agent_id: str
    status: AgentStatus
    last_output_summary: str
    urgency: UrgencyLevel
    # task_description and agent_id are immutable after registration
```

### Public Interface

```python
class RegistryManager:
    def __init__(self, context_builder: "ContextBuilder") -> None:
        # context_builder used for token counting enforcement

    def register(self, agent_id: str, task_description: str) -> None:
        # Called once when an agent is created. Sets initial status=RUNNING.
        # Raises ValueError if task_description exceeds 50 tokens.

    def update(self, agent_id: str, update: RegistryUpdate) -> None:
        # Called by agent on every status change or step completion.
        # Truncates last_output_summary to ≤100 tokens if needed (logs warning).
        # Raises KeyError if agent_id not registered.
        # LOG: REGISTRY_UPDATE

    def get_all(self) -> list[RegistryEntry]:
        # Returns current snapshot of all entries.
        # Used by ContextBuilder for both registry and focus contexts.

    def get(self, agent_id: str) -> RegistryEntry:
        # Returns single entry. Raises KeyError if not found.

    def mark_steering_pending(self, agent_id: str) -> None:
        # Sets pending_steering_request=True and status=WAITING_STEERING.
        # Called by Orchestrator when SteeringRequest is dequeued.

    def mark_steering_complete(self, agent_id: str) -> None:
        # Sets pending_steering_request=False and status=RUNNING.
        # Called by Orchestrator after SteeringResponse is delivered.
```

### Invariants

1. `task_description` is never mutated after `register()`.
2. Every `update()` call produces exactly one `REGISTRY_UPDATE` log line.
3. No `RegistryEntry` serialized by `ContextBuilder` may exceed 200 tokens — enforced by assertion in `ContextBuilder`, not just at write time.
4. `pending_steering_request=True` iff and only iff the Orchestrator has dequeued but not yet delivered a response for that agent.

### Log line format (`REGISTRY_UPDATE`)

```json
{"ts": "2026-04-04T12:00:00.123Z", "event": "REGISTRY_UPDATE", "agent_id": "a1", "status": "WAITING_STEERING", "urgency": "HIGH", "summary_tokens": 87}
```

---

## Component B — `SteeringRequest` + Queue (`src/protocols.py`)

### Responsibility

Define the message contract between agents and orchestrator. Manage the priority queue of pending requests.

### Data Schemas

```python
import uuid
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class SteeringRequest:
    agent_id: str
    relevant_context: str    # Specific context O needs — assembled by the agent itself
    question: str            # The decision or clarification needed (≤100 tokens)
    blocking: bool           # True = agent halted waiting for response
    urgency: UrgencyLevel
    timestamp: str           # ISO 8601
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class SteeringResponse:
    request_id: str              # Matches the originating SteeringRequest.request_id
    agent_id: str
    response_text: str           # Orchestrator's steering decision
    context_size_at_time: int    # Token count of context window when decision was made — CRITICAL
    orchestrator_state: str      # "FOCUS" or "FLAT" (baseline)
    timestamp: str               # ISO 8601

@dataclass
class FocusContext:
    agent_id: str
    task_description: str
    steering_history: list[dict]   # Prior (SteeringRequest, SteeringResponse) pairs for this agent
                                   # List of {"request": {...}, "response": {...}} dicts
                                   # Capped at last K=10 pairs
    recent_output: str             # Agent's last K=10 turns of output
    current_request: SteeringRequest
```

### `SteeringRequestQueue` Interface

```python
class SteeringRequestQueue:
    def enqueue(self, request: SteeringRequest) -> None:
        # HIGH urgency → inserted at front (index 0).
        # MEDIUM/LOW → appended in arrival order.
        # LOG: STEERING_REQUEST

    def peek(self) -> SteeringRequest | None:
        # Returns next request without removing.
        # Used by Orchestrator to decide whether to transition.

    def dequeue(self) -> SteeringRequest | None:
        # Removes and returns head of queue.
        # Returns None if queue is empty.

    def has_high_urgency(self) -> bool:
        # Returns True if any HIGH urgency request is at the head of the queue.
        # Checked by Orchestrator after every LLM call completion.

    def size(self) -> int:
        # Returns number of pending requests.
```

### Invariants

1. A `SteeringRequest` with `urgency=HIGH` from agent aⱼ while Orchestrator is in `FOCUS(aᵢ)` (j≠i) MUST trigger an interrupt. `has_high_urgency()` is checked after every LLM call.
2. `request_id` is globally unique (UUID4). Every `SteeringResponse` must carry the matching `request_id`.
3. `context_size_at_time` in `SteeringResponse` is set by `ContextBuilder.count_tokens()` immediately before the LLM call — not after, not estimated.
4. `question` must be ≤100 tokens. Enforced at agent emit time; `SteeringRequestQueue.enqueue()` raises `ValueError` if exceeded.

### Log line formats

```json
{"ts": "2026-04-04T12:00:01.000Z", "event": "STEERING_REQUEST", "request_id": "abc-123", "agent_id": "a2", "urgency": "HIGH", "blocking": true, "question": "Should I include sparse attention variants in the survey?"}
{"ts": "2026-04-04T12:00:02.500Z", "event": "STEERING_RESPONSE", "request_id": "abc-123", "agent_id": "a2", "context_size_at_time": 6834, "orchestrator_state": "FOCUS", "response_tokens": 112}
```

---

## Component C — `ContextBuilder` (`src/context_builder.py`)

### Responsibility

Deterministically assemble the exact token-counted context for every orchestrator LLM call. This is the central experiment variable — every byte produced is logged with its token count.

### Public Interface

```python
import tiktoken

class ContextBudgetError(Exception):
    pass

class ContextBuilder:
    def __init__(self, token_budget: int) -> None:
        # token_budget T: hard cap, enforced before every call.
        # Uses tiktoken cl100k_base encoding (deterministic).
        self._enc = tiktoken.get_encoding("cl100k_base")

    def build_registry_context(self, registry: list[RegistryEntry]) -> str:
        # Serializes all registry entries to prompt format.
        # Format: one entry per line, structured as compact prose or JSON-like block.
        # Guaranteed ≤ N×200 tokens (registry alone always fits within T).
        # LOG: CONTEXT_BUILT (mode=REGISTRY)

    def build_focus_context(
        self,
        focus: FocusContext,
        registry: list[RegistryEntry],
    ) -> str:
        # Assembles F(aᵢ) + compressed registry (excluding aᵢ's entry).
        # Compression priority when over budget:
        #   1. Drop COMPLETE/FAILED entries to 1-line tombstone
        #   2. Truncate LOW urgency entries to task_description only
        #   3. Truncate MEDIUM urgency entries to task_description + status
        #   HIGH urgency + WAITING_STEERING entries are NEVER truncated.
        # Raises ContextBudgetError if F(aᵢ) alone exceeds T.
        # Asserts count_tokens(result) <= token_budget before returning.
        # LOG: CONTEXT_BUILT (mode=FOCUS)

    def build_flat_context(
        self,
        all_focus_contexts: list[FocusContext],
        current_request: SteeringRequest,
    ) -> str:
        # BASELINE ONLY. Concatenates all agent contexts.
        # Same token-count assertion and logging as build_focus_context.
        # LOG: CONTEXT_BUILT (mode=FLAT)

    def count_tokens(self, text: str) -> int:
        # Exposed for use by RegistryManager field enforcement and agent-side token checking.
        return len(self._enc.encode(text))
```

### Serialization Format for `RegistryEntry` (in registry context)

```
[a1] status=RUNNING urgency=LOW
task: implement binary search tree insertion
last_output: defined Node class, insert() stubbed, tests passing for empty tree
```

### Serialization Format for `FocusContext` (in focus context)

```
=== FOCUS: Agent a2 ===
Task: research transformer attention mechanisms for survey paper

Steering history (last 10 exchanges):
[1] Q: Should I include sparse attention variants?
    A: Yes, include Longformer and BigBird. Skip linear attention for now.
...

Recent output (last 10 turns):
...

Current request [HIGH]:
Context: I found a conflicting claim between Vaswani 2017 and a 2024 survey...
Question: Which source should I treat as authoritative for the attention complexity claim?

=== OTHER AGENTS (compressed) ===
[a1] status=RUNNING urgency=LOW | task: implement BST insertion
[a3] status=BLOCKED urgency=MEDIUM | task: process sales CSV, stuck on UTF-8 encoding
```

### Invariants

1. `count_tokens(build_focus_context(...))` ≤ `token_budget` — enforced with `assert` before returning.
2. `count_tokens(build_flat_context(...))` ≤ `token_budget` — same assertion.
3. `ContextBudgetError` is raised if F(aᵢ) alone (without registry) exceeds T. This should never happen given the budget analysis above; if it does, it is a bug in the agent's context assembly, not silent truncation.
4. Tokenizer is initialized once in `__init__`; never re-initialized per call (performance).

### Log line format (`CONTEXT_BUILT`)

```json
{"ts": "2026-04-04T12:00:01.800Z", "event": "CONTEXT_BUILT", "mode": "FOCUS", "agent_id": "a2", "token_count": 6834, "registry_entries": 9, "steering_history_turns": 3}
{"ts": "2026-04-04T12:00:00.900Z", "event": "CONTEXT_BUILT", "mode": "REGISTRY", "agent_id": null, "token_count": 1420, "registry_entries": 5, "steering_history_turns": null}
```

---

## Component D — `Orchestrator` (`src/orchestrator.py`)

### Responsibility

State machine + LLM call dispatch. Routes responses back to correct agents. Handles user messages. Logs all transitions and LLM calls.

### State Enum

```python
from enum import Enum

class OrchestratorState(Enum):
    REGISTRY       = "REGISTRY"
    FOCUS          = "FOCUS"
    USER_INTERACT  = "USER_INTERACT"
```

### Public Interface

```python
class Orchestrator:
    def __init__(
        self,
        registry: RegistryManager,
        queue: SteeringRequestQueue,
        context_builder: ContextBuilder,
        llm_client,                    # OpenAI AsyncOpenAI client
        model: str,                    # e.g. "gpt-4o-mini"
        token_budget: int,
        focus_mode: bool = True,       # False = baseline flat-context mode
        focus_timeout: int = 60,       # seconds; also capped at 3 LLM turns
        log_path: str = "results/run.jsonl",
    ) -> None:

    async def run(self) -> None:
        # Main async event loop:
        # 1. Check SteeringRequestQueue.peek()
        # 2. If request pending AND focus_mode=True: transition to FOCUS(aᵢ)
        # 3. If request pending AND focus_mode=False: build_flat_context(), LLM call, route response
        # 4. If no request: stay in REGISTRY, poll
        # 5. After every LLM call: check has_high_urgency() → interrupt if needed
        # 6. Check focus_timeout; emit SteeringAbandoned if exceeded

    async def handle_user_message(self, message: str) -> str:
        # Transitions to USER_INTERACT.
        # If currently in FOCUS: saves partial state, responds to user, resumes focus.
        # LOG: TRANSITION (to USER_INTERACT and back)

    def _transition(self, to: OrchestratorState, agent_id: str | None, trigger: str) -> None:
        # Internal. Logs the transition and updates self._state.
        # LOG: TRANSITION

    async def _llm_call(self, context: str, system_prompt: str) -> tuple[str, int, int, int]:
        # Internal. Makes one LLM call.
        # Returns (response_text, prompt_tokens, response_tokens, latency_ms).
        # LOG: LLM_CALL
```

### Main Loop Pseudocode

```python
async def run(self) -> None:
    while True:
        request = self.queue.peek()

        if request is not None:
            self.queue.dequeue()
            self.registry.mark_steering_pending(request.agent_id)

            if self.focus_mode:
                focus_ctx = FocusContext(...)  # assembled from registry + request
                context = self.context_builder.build_focus_context(focus_ctx, self.registry.get_all())
                self._transition(OrchestratorState.FOCUS, request.agent_id, "SteeringRequest")
            else:
                context = self.context_builder.build_flat_context(all_focus_contexts, request)
                # No state transition in baseline — stays in FLAT

            response_text, p_tok, r_tok, ms = await self._llm_call(context, SYSTEM_PROMPT)
            steering_response = SteeringResponse(
                request_id=request.request_id,
                agent_id=request.agent_id,
                response_text=response_text,
                context_size_at_time=p_tok,
                orchestrator_state=self._state.value,
                timestamp=now_iso(),
            )
            await self._route_response(steering_response)
            self.registry.mark_steering_complete(request.agent_id)

            if self.focus_mode:
                self._transition(OrchestratorState.REGISTRY, None, "SteeringComplete")

            # Interrupt check
            if self.focus_mode and self.queue.has_high_urgency():
                # transition to FOCUS(aⱼ) handled on next loop iteration
                pass

        else:
            await asyncio.sleep(0.1)  # poll interval
```

### Invariants

1. `self._state` is never simultaneously `FOCUS` for more than one agent.
2. Every `_transition()` call produces exactly one `TRANSITION` log line.
3. Every `_llm_call()` call produces exactly one `LLM_CALL` log line.
4. User messages are never dropped. If in `FOCUS`, they are queued and handled on the next `USER_INTERACT` transition.
5. `SteeringResponse.context_size_at_time` is always the `prompt_tokens` returned by the LLM provider API response — this is the ground-truth measurement, not the pre-call estimate.

### Log line formats

```json
{"ts": "2026-04-04T12:00:01.900Z", "event": "TRANSITION", "from": "REGISTRY", "to": "FOCUS", "agent_id": "a2", "trigger": "SteeringRequest", "request_id": "abc-123"}
{"ts": "2026-04-04T12:00:02.450Z", "event": "LLM_CALL", "state": "FOCUS", "agent_id": "a2", "context_tokens": 6834, "response_tokens": 112, "latency_ms": 1240}
{"ts": "2026-04-04T12:00:02.500Z", "event": "TRANSITION", "from": "FOCUS", "to": "REGISTRY", "agent_id": "a2", "trigger": "SteeringComplete", "request_id": "abc-123"}
{"ts": "2026-04-04T12:00:10.000Z", "event": "INTERRUPT", "interrupted_agent": "a1", "interrupting_agent": "a3", "urgency": "HIGH", "request_id": "xyz-456"}
{"ts": "2026-04-04T12:01:02.000Z", "event": "FOCUS_TIMEOUT", "agent_id": "a1", "elapsed_s": 60.1, "turns": 2}
```

---

## Complete Log Event Reference

All log lines are JSON, written to a single `.jsonl` file per experiment run. File path: `results/{run_id}.jsonl`.

| Event | Producer | Required fields |
|---|---|---|
| `REGISTRY_UPDATE` | `RegistryManager.update()` | `ts`, `agent_id`, `status`, `urgency`, `summary_tokens` |
| `STEERING_REQUEST` | `SteeringRequestQueue.enqueue()` | `ts`, `request_id`, `agent_id`, `urgency`, `blocking`, `question` |
| `CONTEXT_BUILT` | `ContextBuilder` | `ts`, `mode`, `agent_id` (null for REGISTRY), `token_count`, `registry_entries` |
| `TRANSITION` | `Orchestrator._transition()` | `ts`, `from`, `to`, `agent_id`, `trigger`, `request_id` (if applicable) |
| `LLM_CALL` | `Orchestrator._llm_call()` | `ts`, `state`, `agent_id`, `context_tokens`, `response_tokens`, `latency_ms` |
| `STEERING_RESPONSE` | `Orchestrator` (after routing) | `ts`, `request_id`, `agent_id`, `context_size_at_time`, `orchestrator_state`, `response_tokens` |
| `INTERRUPT` | `Orchestrator` | `ts`, `interrupted_agent`, `interrupting_agent`, `urgency`, `request_id` |
| `FOCUS_TIMEOUT` | `Orchestrator` | `ts`, `agent_id`, `elapsed_s`, `turns` |
| `RUN_START` | `run_experiment.py` | `ts`, `run_id`, `condition`, `n_agents`, `focus_mode`, `model`, `token_budget` |
| `RUN_END` | `run_experiment.py` | `ts`, `run_id`, `total_steering_actions`, `total_llm_calls`, `wall_clock_s` |

---

## Pre-Implementation Checklist

- [x] Token budget confirmed: T=204,800 (MiniMax-M2.7 / Claude Haiku 4.5 compatible)
- [x] LLM provider: MiniMax-M2.7 (Phases 1–3), Claude Haiku 4.5 via OpenRouter (Phase 4)
- [x] Tokenizer: tiktoken cl100k_base
- [x] All design decisions resolved (see table at top)
- [x] SteeringRequest/Response schemas finalized
- [x] FocusContext assembly logic specified (task + K=10 steering history + K=10 output turns + current request)
- [x] Log format agreed — example line per event type written above
- [x] Architecture diagram answers all experiment measurement questions
- [ ] Phase 3 can start on Day 14 without further design decisions outstanding
