# DACS — Phase 2: Architecture Design Plan

**Timeline:** Days 7–14   (April 5–12, 2026)  
**Status:** Ready to begin — Phase 1 literature review complete, formal definition resolved  
**Deliverables:** `/docs/architecture.md` + `/docs/interfaces.md`

---

## Context: What Phase 2 Must Accomplish

Phase 3 (implementation) starts on Day 14. Everything built in Phase 2 is the **contract** that implementation must satisfy. Decisions made here directly control:

- Which Python files exist and what functions they expose
- What JSON schemas flow between components
- How the token budget calculation works (determines which context window size T to target)
- What gets logged (the experiment's measurement layer is designed here, not improvised during implementation)

Be precise. Vague specs = debugging time in Phase 3.

---

## Deliverable 1 — Architecture Diagram (`/docs/architecture.md`)

### What it must show

1. **The 4 core components** and their relationships:
   - `RegistryManager` — maintains per-agent state snapshots
   - `SteeringRequestQueue` — receives and prioritizes agent requests
   - `ContextBuilder` — assembles the context window for each orchestrator state
   - `Orchestrator` — state machine + LLM call + response routing

2. **Data flows** with direction and content labels:
   - Agent → SteeringRequestQueue: `SteeringRequest` message
   - Agent → RegistryManager: heartbeat update
   - RegistryManager → ContextBuilder: registry snapshot
   - SteeringRequestQueue → Orchestrator: next request to handle
   - ContextBuilder → Orchestrator: assembled context (token-counted)
   - Orchestrator → Agent: steering response

3. **Orchestrator state machine** (explicit nodes + transitions):
   ```
   REGISTRY ──[SteeringRequest received]──→ FOCUS(aᵢ)
   FOCUS(aᵢ) ──[SteeringComplete]──→ REGISTRY
   FOCUS(aᵢ) ──[HIGH urgency interrupt from aⱼ]──→ FOCUS(aⱼ)
   REGISTRY ──[user message]──→ USER_INTERACT
   USER_INTERACT ──[response sent]──→ REGISTRY
   FOCUS(aᵢ) ──[HIGH urgency user message]──→ USER_INTERACT → FOCUS(aᵢ)
   ```

4. **Baseline orchestrator** (same diagram, focus mode disabled):
   - No FOCUS state — all agent contexts always injected flat into one context
   - Same RegistryManager, same logging — only ContextBuilder behavior differs
   - Show this as a variant/overlay so the paper comparison is visually obvious

### Format

Mermaid diagram (renders on GitHub + can be copy-pasted into paper as TikZ). Include:
- Component boxes with their file path (`src/registry.py`, etc.)
- Labeled arrows with message type
- State machine as a separate subgraph

---

## Deliverable 2 — Interface Specs (`/docs/interfaces.md`)

For each component, define: Python class name, constructor signature, public methods, data schemas, invariants the implementation must uphold.

### Component A: `RegistryManager` (`src/registry.py`)

**Responsibility:** Single source of truth for all agent state. Thread-safe read/write for concurrent agent heartbeats.

**Public interface:**
```python
class RegistryManager:
    def update(self, agent_id: str, update: RegistryUpdate) -> None
        # Called by agent on every status change or step completion
        # update contains: status, last_output_summary, urgency, timestamp
        # Overwrites existing entry for agent_id

    def get_all(self) -> list[RegistryEntry]
        # Returns current snapshot of all entries
        # Used by ContextBuilder for both registry and focus contexts

    def get(self, agent_id: str) -> RegistryEntry
        # Returns single entry — used by Orchestrator after steering to verify state

    def mark_steering_pending(self, agent_id: str) -> None
        # Called when SteeringRequest is received — sets pending_steering_request=True

    def mark_steering_complete(self, agent_id: str) -> None
        # Called after Orchestrator delivers steering response
```

**RegistryEntry schema:**
```python
@dataclass
class RegistryEntry:
    agent_id: str
    task_description: str          # ≤50 tokens enforced at write time
    status: AgentStatus            # RUNNING | BLOCKED | WAITING_STEERING | COMPLETE | FAILED
    last_output_summary: str       # ≤100 tokens enforced at write time
    last_updated: str              # ISO timestamp
    pending_steering_request: bool
    urgency: UrgencyLevel          # LOW | MEDIUM | HIGH
```

**Token budget invariant:** Each serialized `RegistryEntry` must be ≤ 200 tokens. `RegistryManager.update()` must enforce this — truncate `last_output_summary` if needed, never silently accept over-budget entries.

**Logging requirement:** Every `update()` call writes one line to the experiment log: `{timestamp} | REGISTRY_UPDATE | {agent_id} | {status} | {urgency}`.

---

### Component B: `SteeringRequest` + Queue (`src/protocols.py`)

**Responsibility:** Define the message contract between agents and orchestrator. Manage the priority queue of pending requests.

**SteeringRequest schema:**
```python
@dataclass
class SteeringRequest:
    agent_id: str
    relevant_context: str   # The specific context O needs — assembled by the agent itself
    question: str           # The specific decision or clarification needed (≤100 tokens)
    blocking: bool          # True = agent halted waiting for response
    urgency: UrgencyLevel   # HIGH can interrupt active FOCUS session
    timestamp: str          # ISO timestamp of emission
    request_id: str         # UUID — for logging + response matching
```

**SteeringResponse schema:**
```python
@dataclass
class SteeringResponse:
    request_id: str          # Matches the originating SteeringRequest
    agent_id: str
    response_text: str       # Orchestrator's steering decision
    context_size_at_time: int  # Token count of context window when decision was made — CRITICAL for experiment
    orchestrator_state: str  # FOCUS or REGISTRY — which mode was in effect
    timestamp: str
```

**SteeringRequestQueue interface:**
```python
class SteeringRequestQueue:
    def enqueue(self, request: SteeringRequest) -> None
        # HIGH urgency → front of queue (can interrupt)
        # MEDIUM/LOW → ordered by arrival time

    def peek(self) -> SteeringRequest | None
        # Returns next request without removing — used by Orchestrator to decide when to transition

    def dequeue(self) -> SteeringRequest | None
        # Removes and returns head of queue

    def has_high_urgency(self) -> bool
        # Returns True if any HIGH urgency request is waiting
        # Used by Orchestrator during FOCUS to decide whether to interrupt
```

**Invariant:** A `SteeringRequest` with `urgency=HIGH` from agent `aⱼ` while orchestrator is in `FOCUS(aᵢ)` MUST trigger an interrupt if `aⱼ ≠ aᵢ`. The queue must surface this via `has_high_urgency()` checked after every orchestrator LLM call completion.

---

### Component C: `ContextBuilder` (`src/context_builder.py`)

**Responsibility:** Deterministically assemble the exact token-counted context for every orchestrator LLM call. This is the central experiment variable — every byte it produces gets logged.

**Public interface:**
```python
class ContextBuilder:
    def __init__(self, tokenizer, token_budget: int):
        # tokenizer: tiktoken or equivalent — used for hard token counting
        # token_budget T: hard cap enforced before every call

    def build_registry_context(self, registry: list[RegistryEntry]) -> str
        # Output: all registry entries serialized to prompt format
        # Used in: REGISTRY mode, USER_INTERACT mode
        # Guaranteed to fit within T tokens (registry alone is always ≤ N × 200 tokens)

    def build_focus_context(
        self,
        agent_id: str,
        focus_context: FocusContext,
        registry: list[RegistryEntry]
    ) -> str
        # Output: F(aᵢ) + compressed registry (excluding aᵢ's entry)
        # Priority: F(aᵢ) always fits first; registry compressed to fill remaining budget
        # Compression order if registry overflows: drop COMPLETE/FAILED first,
        #   then truncate LOW urgency to task description only,
        #   then truncate MEDIUM urgency to task description only.
        #   HIGH + WAITING_STEERING entries never truncated below task description.
        # Returns assembled prompt string + token count
        # Raises ContextBudgetError if F(aᵢ) alone exceeds T

    def count_tokens(self, text: str) -> int
        # Exposed for use by RegistryManager field enforcement and logging
```

**FocusContext schema:**
```python
@dataclass
class FocusContext:
    agent_id: str
    task_description: str
    steering_history: list[dict]   # Prior SteeringRequest/Response pairs for this agent
    recent_output: str             # Agent's last K=10 turns of output
    current_request: SteeringRequest
```

**Invariant:** `count_tokens(build_focus_context(...))` ≤ `token_budget` — enforced with an assertion before every LLM call. Never rely on the provider to truncate.

**Logging requirement:** Every `build_focus_context()` and `build_registry_context()` call logs: `{timestamp} | CONTEXT_BUILT | {mode} | {agent_id_or_REGISTRY} | {token_count}`.

---

### Component D: `Orchestrator` (`src/orchestrator.py`)

**Responsibility:** State machine + LLM call dispatch. Routes responses back to correct agents. Handles user messages. Logs all transitions.

**State enum:**
```python
class OrchestratorState(Enum):
    REGISTRY = "REGISTRY"
    FOCUS = "FOCUS"
    USER_INTERACT = "USER_INTERACT"
```

**Public interface:**
```python
class Orchestrator:
    def __init__(
        self,
        registry: RegistryManager,
        queue: SteeringRequestQueue,
        context_builder: ContextBuilder,
        llm_client,           # OpenAI or Anthropic client — injected for testability
        focus_mode: bool,     # False = baseline flat-context mode
        token_budget: int
    ):

    def run(self) -> None
        # Main event loop:
        # - Poll queue for SteeringRequests
        # - Transition state as per state machine
        # - Call LLM with assembled context
        # - Route response to correct agent
        # - Check for HIGH urgency interrupts after each LLM call

    def handle_user_message(self, message: str) -> str
        # If in FOCUS: save state, transition to USER_INTERACT, respond, return to prior state
        # If in REGISTRY or USER_INTERACT: respond immediately

    def get_current_state(self) -> tuple[OrchestratorState, str | None]
        # Returns (state, agent_id_if_in_focus)
        # Used by experiment harness to verify state at measurement points
```

**Logging requirement:** Every state transition logs:
```
{timestamp} | TRANSITION | {from_state} | {to_state} | {agent_id} | {trigger}
```
Every LLM call logs:
```
{timestamp} | LLM_CALL | {state} | {agent_id_or_REGISTRY} | {context_tokens} | {response_tokens} | {latency_ms}
```

**Baseline mode (`focus_mode=False`):** `build_focus_context()` is replaced by `build_flat_context()` which injects all agent contexts concatenated. Registry mode does not exist — the orchestrator always holds all agent contexts. This is the only behavioral difference from DACS mode.

---

## Token Budget Calculation

Must be done before writing a line of implementation code.

**Inputs:**
- Target context window: T = 32,768 tokens (conservative; fits within GPT-4o-mini and claude-3-haiku limits)
- Registry: N agents × 200 tokens/entry = max 2,000 tokens for N=10
- Focus context F(aᵢ): task description + up to K=10 prior steering turns + current request
  - Estimated: 500 (task) + 10 × 400 (steering turn avg) + 200 (request) = ~4,700 tokens
- Compressed registry (N-1 entries, worst case): 1,800 tokens
- System prompt overhead: ~500 tokens

**Budget check for FOCUS mode, N=10:**
- F(aᵢ) + compressed registry + system prompt = 4,700 + 1,800 + 500 = **7,000 tokens**
- Well within T=32,768 ✓

**Budget check for baseline FLAT mode, N=10:**
- All 10 agents' full contexts = 10 × 4,700 = 47,000 tokens
- Exceeds T=32,768 for N=10 — the baseline must truncate or use T=128k
- **Decision:** Use T=128,000 for all conditions so baseline doesn't fail mechanically. The experiment measures DACS at 7k tokens vs baseline at ~47k tokens — the gap is the result.

**Action item before Day 14:** Confirm GPT-4o-mini 128k context via API docs. If unavailable, use claude-3-haiku-20240307 (200k context).

---

## Design Decisions to Finalize in Phase 2 (before any code)

| Decision | Options | Recommended |
|---|---|---|
| LLM provider | OpenAI (gpt-4o-mini) vs Anthropic (claude-3-haiku) | **GPT-4o-mini** — cheaper for 60 runs, tiktoken for counting |
| Tokenizer | tiktoken (OpenAI cl100k_base) vs transformers | **tiktoken** — deterministic, fast, no GPU needed |
| Concurrency model | asyncio vs threading vs subprocess per agent | **asyncio** — agents are LLM-call-bound, not CPU-bound; cleaner event loop |
| Agent heartbeat trigger | Time-based (every N seconds) vs event-based (every step) | **Event-based** — cleaner for experiment reproducibility |
| Steering history depth K | Last 5 turns vs last 10 turns | **K=10** — matches FocusContext spec above; ablate if budget is tight |
| Log format | JSON lines vs CSV vs plain text | **JSON lines** — structured, grep-able, directly parseable by experiment harness |
| Baseline context assembly | Concatenate all agents vs interleaved | **Concatenate** — simpler, worst case for contamination |

---

## Phase 2 Task Checklist

### Day 7–9: Architecture Diagram
- [ ] Draft Mermaid diagram for DACS (4 components + data flows + state machine)
- [ ] Add baseline orchestrator overlay
- [ ] Write `/docs/architecture.md` with diagram + 1-paragraph description per component
- [ ] Verify: does the diagram answer "where does each log line come from?"

### Day 9–12: Interface Specs
- [ ] Write `/docs/interfaces.md` with all 4 component specs from this document
- [ ] Token budget calculation confirmed (choose T, verify LLM provider supports it)
- [ ] All design decisions above resolved (fill in the table)
- [ ] SteeringRequest/Response schemas finalized
- [ ] FocusContext assembly logic specified (what exactly goes into F(aᵢ)?)
- [ ] Log format agreed — write one example log line per log point

### Day 12–14: Pre-Implementation Validation
- [ ] Review: can the architecture diagram answer every experiment measurement question?
  - "What was the context size when agent aᵢ received steering?" → CONTEXT_BUILT log
  - "Did the orchestrator reference the wrong agent?" → LLM_CALL log + response text
  - "How long did the user wait?" → TRANSITION log timestamps
- [ ] Review: does the interface spec leave any ambiguity that would become a bug?
- [ ] Confirm Phase 3 can start without any further design decisions outstanding

---

## Key Constraint Reminders

- **Observability is paramount.** Every LLM call must log the exact token count of its context window. Design the logging layer here, not during implementation.
- **Token budget is hard.** `ContextBuilder` must enforce this deterministically via assertion — never assume the provider will truncate cleanly.
- **Baseline must be fair.** The flat-context baseline uses identical code paths; only `focus_mode=False` changes behavior. No other differences.
- **~300 lines total for the harness.** If the architecture requires more, something is over-engineered. The agents are stubs — complexity lives in the orchestrator and context builder only.
