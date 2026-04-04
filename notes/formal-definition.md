# DACS — Formal Definition

*Draft. Refine after reading all 5 papers.*

---

## 1. Entities

- **Orchestrator** `O` — single LLM instance, manages all agents and user interaction
- **Agent set** `A = {a₁, a₂, ..., aₙ}` — N parallel agents doing heterogeneous tasks
- **Context window** `C` — token-limited (T tokens max, e.g. T = 128k)
- **Registry** `R` — set of lightweight state snapshots `{r₁, r₂, ..., rₙ}`, one per agent
- **Focus context** `F(aᵢ)` — full context of agent `aᵢ` required for steering decisions

---

## 2. Registry Entry Schema

Each entry `rᵢ ∈ R`:

```json
{
  "agent_id": "string",
  "task_description": "string (≤50 tokens)",
  "status": "RUNNING | BLOCKED | WAITING_STEERING | COMPLETE | FAILED",
  "last_output_summary": "string (≤100 tokens)",
  "last_updated": "ISO timestamp",
  "pending_steering_request": "boolean",
  "urgency": "LOW | MEDIUM | HIGH"
}
```

**Token budget per entry:** target ≤ 200 tokens  
**Full registry size for N=10:** ≤ 2000 tokens (leaves room for focus context)

---

## 3. Orchestrator States

```
States:
  REGISTRY       — O holds R only. Available for user messages. Monitors all agents.
  FOCUS(aᵢ)      — O holds F(aᵢ) + compressed R. Steers aᵢ. Queues user messages.
  USER_INTERACT  — O responds to user. Holds R only. No active focus session.
```

---

## 4. State Transitions

```
REGISTRY → FOCUS(aᵢ)
  trigger: aᵢ emits SteeringRequest(urgency=ANY)
  action: build_focus_context(aᵢ), log transition + context size

FOCUS(aᵢ) → REGISTRY
  trigger: SteeringComplete or SteeringAbandoned
  action: update rᵢ in R, log transition

REGISTRY → USER_INTERACT
  trigger: user message received
  action: build_registry_context(), respond

USER_INTERACT → REGISTRY
  trigger: response sent
  action: check steering request queue

FOCUS(aᵢ) → FOCUS(aⱼ)  [interrupt]
  trigger: aⱼ emits SteeringRequest(urgency=HIGH) while in FOCUS(aᵢ)
  action: save partial steering state for aᵢ, log interrupt, build_focus_context(aⱼ)

FOCUS(aᵢ) → USER_INTERACT  [urgent user message]
  trigger: user message with urgency=HIGH received while in FOCUS(aᵢ)
  action: save partial steering state for aᵢ, respond to user, return to FOCUS(aᵢ)
```

---

## 5. Context Builder Functions

```
build_focus_context(aᵢ):
  → F(aᵢ) + compressed_registry(exclude=aᵢ)
  → must fit within T tokens
  → if F(aᵢ) + full_registry > T: compress R entries further (truncate summaries)

build_registry_context():
  → R (all entries in full)
  → used in REGISTRY and USER_INTERACT states

compressed_registry(exclude=aᵢ):
  → R without rᵢ (aᵢ's context is in F(aᵢ) already)
  → if needed, truncate lower-urgency entries first
```

---

## 6. SteeringRequest Protocol

Agent `aᵢ` emits when it needs orchestrator input:

```json
{
  "type": "SteeringRequest",
  "agent_id": "aᵢ",
  "relevant_context": "string — the specific context O needs to make this decision",
  "question": "string — the specific decision or clarification needed",
  "blocking": "boolean — is aᵢ halted waiting for this?",
  "urgency": "LOW | MEDIUM | HIGH"
}
```

**Urgency definitions:**
- `HIGH` — agent is blocked and cannot proceed; can interrupt active focus session
- `MEDIUM` — agent is waiting but can continue on a default path
- `LOW` — non-blocking, batching acceptable

---

## 7. Invariants (things that must always hold)

1. O is never simultaneously in FOCUS mode for more than one agent
2. User messages are never dropped — they are queued if O is in FOCUS and answered on next transition to USER_INTERACT or REGISTRY
3. Registry R is always up to date within the last agent heartbeat interval
4. Token budget T is never exceeded — context builder enforces this before any LLM call

---

## 8. Open Questions (resolve after reading papers)

- How exactly does F(aᵢ) get assembled? Full conversation history? Or just recent N turns + task summary?
- What is the right heartbeat interval for registry updates? Every agent step? Every K steps?
- Should registry compression be deterministic (truncate by urgency) or learned?
- Is there a maximum time limit for a focus session? 
- How does the system handle a focus session where O makes a bad steering decision? Recovery?

---

## 9. Comparison to Closest Prior Art

| Dimension | AFM | CodeDelegator | **DACS** |
|---|---|---|---|
| Compression trigger | Every user turn (automatic) | Sub-task boundary (Delegator dispatch) | `SteeringRequest` from specific agent aᵢ |
| Compression scope | Flat single-turn message history | Ephemeral Coder execution traces excluded from Delegator | All agents except aᵢ compressed in registry; aᵢ gets full F(aᵢ) |
| Symmetry | Symmetric — every message scored same way | Symmetric — all Coders get same clean-context treatment | **Asymmetric** — aᵢ gets full context; all others get compressed registry entry |
| Agent-triggered | No | No | **Yes** — aᵢ explicitly emits SteeringRequest |
| N concurrent agents | No — single conversation | No — sequential sub-tasks | **Yes** — handles N simultaneous independent agents |
| Wrong-agent contamination | Not modelled | Not modelled | **Primary failure mode** — measured as first-class metric |

---

## 10. Resolved Open Questions

**Q: How exactly does F(aᵢ) get assembled?**  
Full task description + recent conversation history with orchestrator (last K turns, K configurable, default=10) + all prior steering exchanges for aᵢ + aᵢ's current partial output summary. NOT the raw intermediate computation — that lives in aᵢ's own process. The orchestrator only needs what it said to aᵢ and what aᵢ said back.

**Q: What is the right heartbeat interval for registry updates?**  
Every agent step (i.e., each time an agent completes an atomic action or changes status). Agents push a compact status message to the Registry Manager after each step. Heartbeat data: `{agent_id, status, last_output_summary, urgency}` — all can be updated in ≤200 tokens. The interval is event-driven (status change) not time-driven, to avoid clock drift across LLM call latencies.

**Q: Should registry compression be deterministic or learned?**  
Deterministic for the prototype. Priority order for dropping registry entries when over budget: (1) drop COMPLETE/FAILED entries first (keep only a 1-line tombstone), (2) truncate LOW urgency entries to task description only, (3) truncate MEDIUM urgency entries. HIGH urgency + WAITING_STEERING entries are never truncated. This makes the budget enforcement reproducible and auditable, which is required for the experiment's observability invariant.

**Q: Is there a maximum time limit for a focus session?**  
Yes — configurable `focus_timeout` (default: 60 seconds wall clock or 3 LLM call turns, whichever comes first). On timeout: emit SteeringAbandoned, log the timeout, transition to REGISTRY. aᵢ receives a partial steering message indicating timeout and must decide whether to proceed on default path or re-emit a SteeringRequest.

**Q: How does the system handle a bad steering decision (error recovery)?**  
aᵢ can re-emit a `SteeringRequest` with urgency=HIGH and include the failed outcome in `relevant_context`. O enters FOCUS(aᵢ) again, now with the failure as context. No special recovery protocol needed — the SteeringRequest mechanism itself is the recovery path. For the experiment, steering decisions are fixed-answer (one correct answer per decision point), so "bad decisions" are measurable as the wrong-answer rate metric directly.

