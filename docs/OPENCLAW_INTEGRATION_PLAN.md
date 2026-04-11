# DACS × OpenClaw Integration Plan

**Goal:** Integrate DACS (Dynamic Attentional Context Scoping) into OpenClaw as a first-class context management mechanism for multi-agent orchestration.

**Date:** April 2026  
**Status:** Planning

---

## 1. How DACS Fits in OpenClaw

### The Problem (Context Pollution in OpenClaw Today)

OpenClaw already has strong multi-agent infrastructure (`sessions_spawn`, `sessions_send`, `sessions_history`, `sessions_list`). But when the orchestrator (main session) manages N concurrent sub-agents, it accumulates all their histories in its context window:

```
Orchestrator context (flat, today):
  ├── System prompt + workspace files
  ├── sesssions_history(a1) — full transcript
  ├── sessions_history(a2) — full transcript
  ├── sessions_history(a3) — full transcript
  └── [grows linearly with N agents × steps each]
```

This is exactly the context pollution DACS was designed to eliminate. At N=5+ agents with long tasks, the orchestrator context blows past practical token budgets, degrading steering accuracy by 40–70pp (measured in DACS experiments).

### The Solution (DACS in OpenClaw)

```
Orchestrator context (DACS):
  ├── System prompt + workspace files
  ├── [DACS REGISTRY] compact summaries — ≤200 tokens/agent
  │     a1: "BST impl, 60% done, needs traversal order decision" [HIGH]
  │     a2: "Transformer survey, searching, low urgency" [LOW]
  │     a3: "CSV pipeline, blocked on null imputation choice" [MEDIUM]
  └── [DACS FOCUS: a3] full context for a3 only (when steering a3)
```

---

## 2. The Conceptual Mapping

| DACS Abstraction | OpenClaw Equivalent |
|---|---|
| `Orchestrator` | Main session (or designated orchestrator session) |
| Agent `aᵢ` | Sub-agent session spawned via `sessions_spawn` |
| `Registry (R)` | Compact summaries from `sessions_history` (≤200 tok/entry) |
| `Focus context F(aᵢ)` | Full `sessions_history` for `aᵢ` only |
| `SteeringRequest` | `sessions_send` to orchestrator with `[[STEER: ...]]` marker |
| `SteeringResponse` | Orchestrator reply back via `sessions_send` |
| `REGISTRY mode` | Orchestrator context contains only compact registry |
| `FOCUS(aᵢ) mode` | Orchestrator context holds full `aᵢ` history + compact registry |
| `USER_INTERACT mode` | User message received; orchestrator in registry-only context |
| `INTERRUPT` | HIGH-urgency `sessions_send` preempts an active `FOCUS` session |
| `Context pollution` | Full `sessions_history` of all agents loaded simultaneously |
| `Token budget T` | OpenClaw's context window (enforced in context engine `assemble()`) |
| **OIF — T5 (User-query routing)** | User sends a message referencing an agent's domain → orchestrator enters `FOCUS(aᵢ)` proactively before responding, rather than answering from registry only |
| **OIF — T6 (Lifecycle signals)** | Sub-agent emits `[[DONE: summary]]` or `[[BLOCKED: reason]]` → orchestrator enters `FOCUS(aᵢ)` to review and assign the next task; `[[PROGRESS: update]]` updates registry only |
| **OIF — T7 (Iteration relay)** | While in `FOCUS(aᵢ)`, orchestrator relays a follow-up directive back to `aᵢ` without leaving focus — amortises context-switching cost for multi-turn iteration |
| `trigger_type` | Registry field recording why Focus was entered: `AGENT_STEERING` / `USER_QUERY` / `COMPLETION` / `ITERATION` |
| **OIF mode flag** | `oif_mode: boolean` in `.dacs_state.json` — enables the T5/T6/T7 top-down transitions; can be toggled without disabling core DACS |

---

## 3. Primary Integration Point: Context Engine Plugin

OpenClaw exposes **`api.registerContextEngine(id, factory)`** which owns:
- `ingest()` — runs before each orchestrator turn (use: reload DACS state from disk)
- `assemble({ messages, availableTools, citationsMode })` — builds the context window (use: REGISTRY/FOCUS switching)
- `compact(params)` — summarisation (delegate to OpenClaw runtime via `delegateCompactionToRuntime`)

All four are **confirmed** in [Plugin Internals — Context engine plugins](https://docs.openclaw.ai/plugins/architecture#context-engine-plugins). `buildMemorySystemPromptAddition` and `delegateCompactionToRuntime` are confirmed imports from `openclaw/plugin-sdk/core`.

This is the exact API for DACS. It gives full algorithmic control over what enters the orchestrator's context window — which is the entire point of DACS.

**Important:** `api.runtime.sessions.{list,history,send}` are **not** in the plugin runtime. They are agent-facing tools available inside agent runs, not callable from plugin hooks. Registry updates happen via the `after_tool_call` hook (which intercepts `sessions_history` calls made by the orchestrator agent), and T7 relay uses `api.runtime.subagent.run()` which is the documented plugin path.

Secondary hooks used alongside (all confirmed):
- `message_received` — detect steering requests from sub-agents, trigger FOCUS mode
- `before_prompt_build` — inject DACS mode into the system prompt header
- `after_tool_call` — intercept `sessions_history` results to enforce registry token cap and update registry entries
- `agent_end` — T2 return to REGISTRY, or T7 self-loop if relay pending
- `session_start` — detect user-initiated sessions for T4/T5

---

## 4. Implementation Plan

### Phase 0 — Immediate Proof of Concept: DACS Skill (Week 1)

**What:** A workspace skill (`~/.openclaw/workspace/skills/dacs/SKILL.md`) that prompts the orchestrator to behave like a DACS orchestrator using existing session tools.

**Why:** Validates the concept end-to-end before writing a line of TypeScript. Tests whether OpenClaw's `sessions_send` + `sessions_history` machinery is sufficient to implement DACS-like context isolation at the prompt level.

**What the skill teaches the orchestrator:**
1. Maintain a mental `[DACS REGISTRY]` block — one line per sub-agent (id, task, status, urgency)
2. When a sub-agent sends a `sessions_send` with `[[STEER: question]]`, enter FOCUS mode: read that agent's `sessions_history` fully, answer with only that agent's context in mind
3. Never read all sub-agent histories simultaneously — only one at a time
4. After steering, return to registry-only mode
5. **OIF — User query routing (T5):** When the user asks about a specific agent's work (e.g. "how is the BST implementation going?"), match the query against registry entries. If two or more task-description keywords overlap with the query, call `sessions_history` for that agent and answer from its full context before responding — not just from the registry summary.
6. **OIF — Lifecycle signals (T6):** When a sub-agent emits `[[DONE: summary]]` or `[[BLOCKED: reason]]`, enter FOCUS mode for that agent to review the completed work or blocked state before assigning the next task. Treat `[[PROGRESS: update]]` as a registry-only heartbeat.
7. **OIF — Iteration relay (T7):** When already in FOCUS for agent `aᵢ` and the next action is a follow-up directive to `aᵢ`, relay it without leaving focus (stay in the same `sessions_history` call) to avoid redundant context switches.

**Expected outcome:** A rough baseline for accuracy and context sizes in OpenClaw's native environment. May leak context (prompt-level, not algorithmic) but proves the fit, including OIF routing behaviour.

**Deliverable:** `skills/dacs/SKILL.md`, a 3-agent test scenario using `sessions_spawn`.

---

### Phase 1 — DACS State Store (Week 1–2)

**What:** A persistent DACS state file in the orchestrator workspace.

**Format:** `~/.openclaw/workspace/.dacs_state.json`
```json
{
  "mode": "REGISTRY",
  "focus_agent": null,
  "oif_mode": true,
  "registry": {
    "a1": { "task": "...", "status": "RUNNING", "last_summary": "...", "urgency": "LOW", "trigger_type": "AGENT_STEERING", "tokens": 47 },
    "a2": { "task": "...", "status": "BLOCKED", "last_summary": "...", "urgency": "HIGH", "trigger_type": "COMPLETION", "tokens": 62 }
  },
  "pending_requests": []
}
```

**`oif_mode`:** Boolean flag — when `true`, enables the T5/T6/T7 top-down OIF transitions. Core DACS (T1–T4) is always active regardless.

**`trigger_type`:** Records why the last Focus session for each agent was entered:
- `AGENT_STEERING` — agent emitted a `[[STEER:]]` request (T1 or T3)
- `USER_QUERY` — user message matched the agent's registry entry (T5)
- `COMPLETION` — agent emitted `[[DONE:]]` or `[[BLOCKED:]]` (T6)
- `ITERATION` — orchestrator relayed a directive while already in Focus (T7 self-loop)

**State transitions (complete — agent-triggered T1–T4 + OIF T5–T7):**
```
REGISTRY     ──(SteeringRequest from aᵢ)────────►  FOCUS(aᵢ)          [T1 — agent-triggered]
FOCUS(aᵢ)    ──(response sent)─────────────────►  REGISTRY             [T2]
FOCUS(aᵢ)    ──(HIGH urgency from aⱼ)──────────►  FOCUS(aⱼ)           [T3 — INTERRUPT]
REGISTRY     ──(user message)───────────────────►  USER_INTERACT        [T4]
USER_INTERACT ──(response sent)─────────────────►  REGISTRY             [T4 return]

[OIF — Orchestrator-Initiated Focus]
REGISTRY     ──(user query matches r(aᵢ))────────►  FOCUS(aᵢ)          [T5 — user-query routing]
REGISTRY     ──(aᵢ emits [[DONE:]] / [[BLOCKED:]])►  FOCUS(aᵢ)         [T6 — lifecycle signal]
FOCUS(aᵢ)    ──(directive addressed to aᵢ)───────►  FOCUS(aᵢ)  [self]  [T7 — iteration relay]
```

**TypeScript module:** `src/dacs/state.ts` — read/write DACS state with file locking.

---

### Phase 2 — Registry Manager (Week 2–3)

**What:** Replicates `registry.py` / `RegistryManager` logic in TypeScript.

**Core logic:**
1. After each orchestrator turn, call `sessions_history(aᵢ)` for each active sub-agent
2. Summarize the last 3–5 turns into a ≤200-token registry entry using the compaction model
3. Persist to `.dacs_state.json`

**Token enforcement:** Use OpenClaw's built-in token counter (accessible in context engine `assemble()`) to enforce ≤200 tokens per registry entry. Truncate by extracting last sentence of `last_summary` if over budget.

**Urgency extraction:** Parse `[[STEER: question | urgency: HIGH]]` markers from sub-agent `sessions_send` messages.

**Lifecycle signal parsing (OIF T6):** Also scan `sessions_send` payloads for agent lifecycle markers:
- `[[DONE: summary]]` — agent reached a milestone; set `status: DONE`, `trigger_type: COMPLETION` → T6 fires
- `[[BLOCKED: reason]]` — agent cannot proceed; set `status: BLOCKED`, `trigger_type: COMPLETION` → T6 fires
- `[[PROGRESS: update]]` — routine heartbeat; update `last_summary` and `status` only, remain in REGISTRY

**Module:** `src/dacs/registry.ts` — `RegistryManager.update(agentId, history)`, `RegistryManager.buildCompactSummary()`, `RegistryManager.all()`, `RegistryManager.parseLifecycleSignal(message)`.

---

### Phase 3 — Context Builder (Week 3)

**What:** Replicates `context_builder.py` / `ContextBuilder` logic in TypeScript. This is what makes DACS better than flat-context.

**Two modes:**

**REGISTRY mode output:**
```
[DACS REGISTRY — 3 active agents]
a1 (BST implementation) [RUNNING, LOW]: Implementing insert/delete. Traversal order TBD.
a2 (Transformer survey) [SEARCHING, LOW]: Reviewing attention variants. No blockers.
a3 (CSV pipeline) [BLOCKED, HIGH]: Awaiting null imputation decision.
```
Token cost: ~60–80 tokens for N=3. Scales ~25 tokens/agent (matches DACS paper).

**FOCUS(aᵢ) mode output:**
```
[DACS FOCUS: a3 — CSV Pipeline — BLOCKED]
[Full sessions_history for a3: last 20 messages/tool calls]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Registry for other agents — compact]
a1: Implementing BST insert/delete. [RUNNING, LOW]
a2: Reviewing transformer survey sources. [SEARCHING, LOW]
```
Token cost: ~5,000–8,000 tokens for focus agent + ~40 tokens for each other. Dramatically smaller than loading all histories (3 × 5,000 = 15,000 tokens).

**Token budget enforcement:** Hard-cap at OpenClaw `agents.defaults.contextWindowSize` (or configured value). Truncate oldest focus-agent messages first if over budget.

**Module:** `src/dacs/context-builder.ts` — `buildRegistryContext()`, `buildFocusContext(agentId)`.

---

### Phase 4 — Steering Request Detection (Week 3–4)

**What:** Detect when a sub-agent is asking for orchestrator steering, identify urgency, and trigger the FOCUS mode transition. Also parse agent lifecycle signals that trigger OIF T6.

**Detection:** `message_received` plugin hook on the orchestrator session.

```typescript
api.registerHook("message_received", async ({ message, session }) => {
  // Sub-agents send to orchestrator via sessions_send
  // The payload contains [[STEER: question | urgency: HIGH]] markers
  const steerMatch = message.text?.match(/\[\[STEER:\s*(.+?)(?:\s*\|\s*urgency:\s*(HIGH|MEDIUM|LOW))?\s*\]\]/i);
  if (steerMatch) {
    const question = steerMatch[1].trim();
    const urgency  = (steerMatch[2] ?? "MEDIUM").toUpperCase();
    const agentId  = extractSenderAgentId(message); // from session routing metadata

    await dacsState.transitionToFocus(agentId, urgency, "AGENT_STEERING");
    await dacsRegistry.update(agentId, { urgency, last_question: question });
  }

  // OIF T6 — lifecycle signal from sub-agent
  const lifecycleMatch = message.text?.match(/\[\[(DONE|BLOCKED|PROGRESS):\s*(.+?)\]\]/i);
  if (lifecycleMatch && state.oif_mode) {
    const signal  = lifecycleMatch[1].toUpperCase();   // DONE | BLOCKED | PROGRESS
    const payload = lifecycleMatch[2].trim();
    const agentId = extractSenderAgentId(message);

    await dacsRegistry.parseLifecycleSignal(agentId, signal, payload);

    if (signal === "DONE" || signal === "BLOCKED") {
      // T6: enter Focus to review or unblock — identical context assembly as T1
      await dacsState.transitionToFocus(agentId, "HIGH", "COMPLETION");
    }
    // PROGRESS: registry update only, no focus transition
  }
});
```

**Interrupt handling:** If `urgency === "HIGH"` and current mode is `FOCUS(aⱼ)` where `aⱼ ≠ agentId`, transition immediately: `FOCUS(aⱼ) → FOCUS(agentId)`. The preempted `aⱼ` request is queued.

**Sub-agent convention:** Sub-agents emit steering requests using the same `[[STEER: ...]]` marker already implemented in `agents/llm_agent.py`. On the OpenClaw side, a sub-agent session appended with the DACS skill also uses this marker in its `sessions_send` messages.

---

### Phase 4b — OIF Integration: User-Query Routing and Iteration Relay (Week 4)

**What:** Implement the two remaining OIF transitions — T5 (user-query routing) and T7 (iteration relay) — which are orchestrator-side decisions rather than reactions to sub-agent signals.

#### T5 — User-Query Routing

When the user sends a message that references a specific agent's task domain, the orchestrator enters `FOCUS(aᵢ)` before constructing its response so that the answer is grounded in `aᵢ`'s full context, not just its ≤200-token registry summary.

**Routing algorithm (prefix-stem matcher):**
```typescript
// src/dacs/oif-router.ts
export function matchUserQueryToAgent(
  userMessage: string,
  registry: Record<string, RegistryEntry>,
): string | null {
  // Extract content words longer than 3 characters from the user message.
  const msgStems = stemWords(userMessage).filter(w => w.length > 3);

  let bestAgent: string | null = null;
  let bestCount = 1; // require at least 2 overlapping stems

  for (const [agentId, entry] of Object.entries(registry)) {
    const taskStems = stemWords(entry.task + " " + entry.last_summary);
    const overlap   = msgStems.filter(s => taskStems.includes(s)).length;
    if (overlap > bestCount) { bestCount = overlap; bestAgent = agentId; }
  }
  return bestAgent; // null → no confident match → answer from registry only
}
```

**Integration in the context engine `assemble()` call:**
```typescript
// In DacsContextBuilder.assemble()
if (state.mode === "REGISTRY" && state.oif_mode && isUserMessage) {
  const targetAgent = matchUserQueryToAgent(userMessage, registry.all());
  if (targetAgent) {
    await state.transitionToFocus(targetAgent, "MEDIUM", "USER_QUERY");
    // Fall through: mode is now FOCUS — builder returns buildFocusContext(targetAgent)
  }
}
```

**Empirical results (DACS Phase 5):** T5 achieves **100% M1_T5 routing accuracy** at N∈{3,5} (60 trials, Claude Haiku 4.5) versus 28–40% for DACS without OIF and 28–33% for a flat baseline (p < 0.0001, Mann–Whitney). Focus sessions average 675–742 tokens vs 1,418–2,270 tokens for flat context (2.10–3.06× smaller). The LLM judge (GPT-4o-mini) validates specificity at 96.7%/94.0% hit rate for OIF vs 26.7%/14.0% for baseline.

#### T7 — Iteration Relay (self-loop)

While in `FOCUS(aᵢ)`, if the orchestrator determines the next action is a follow-up directive addressed to `aᵢ` (e.g. from a pipeline dependency or user clarification), it relays the directive without transitioning back to REGISTRY first.

```typescript
// In the agent_end hook — only transition to REGISTRY if no relay pending
api.registerHook("agent_end", async ({ session }) => {
  if (state.mode === "FOCUS") {
    const relay = state.pendingRelay; // set by orchestrator during Focus turn
    if (relay && relay.targetAgentId === state.focusAgent) {
      // T7 self-loop: stay in Focus, send relay to aᵢ.
      // api.runtime.sessions.* is not in the plugin runtime — use api.runtime.subagent.run()
      // which accepts sessionKey + message and is the documented plugin path for steering subagents.
      await api.runtime.subagent.run({
        sessionKey: relay.targetAgentId,
        message: relay.message,
        deliver: false,
      });
      await state.logTransition("FOCUS", "FOCUS", state.focusAgent!, "ITERATION");
      state.pendingRelay = null;
    } else {
      await state.transitionToRegistry();
    }
  }
});
```

**Module:** `src/dacs/oif-router.ts` — `matchUserQueryToAgent()`, `stemWords()`, `buildOifPromptHint()`.

---

### Phase 5 — Context Engine Plugin (Week 4–5)

**What:** The core DACS plugin, registered as a context engine.

```typescript
import { buildMemorySystemPromptAddition, delegateCompactionToRuntime } from "openclaw/plugin-sdk/core";
import { DacsState } from "./dacs/state";
import { DacsRegistry } from "./dacs/registry";
import { DacsContextBuilder } from "./dacs/context-builder";

const plugin: OpenClawPluginDefinition = {
  id: "dacs",
  name: "DACS Context Engine",
  register(api) {
    const state    = new DacsState(api.config);
    const registry = new DacsRegistry(state, api);
    const builder  = new DacsContextBuilder(registry, state, api);

    // 1. Context engine — the core DACS integration
    api.registerContextEngine("dacs", () => ({
      info: { id: "dacs", name: "DACS", ownsCompaction: false },

      async ingest() {
        // Load fresh DACS state from disk. Registry entries are updated lazily via
        // the `after_tool_call` hook when the orchestrator agent calls sessions_history
        // (api.runtime.sessions.* is not in the plugin runtime — sessions_* are
        // agent-facing tools, not plugin helpers). State file is the source of truth.
        await state.reload();
        return { ingested: true };
      },

      async assemble({ messages, availableTools, citationsMode }) {
        const mode = state.mode;
        let systemPromptAddition = "";

        if (mode === "FOCUS") {
          const context = await builder.buildFocusContext(state.focusAgent!);
          systemPromptAddition = context.systemPrompt;
          // Prepend focus context as a system turn (within token budget)
        } else {
          // REGISTRY or USER_INTERACT
          systemPromptAddition = builder.buildRegistryContext();
        }

        return {
          messages,
          estimatedTokens: builder.lastTokenCount,
          systemPromptAddition,
        };
      },

      async compact(params) {
        return await delegateCompactionToRuntime(params);
      },
    }));

    // 2. Steering hook (T1/T3) + OIF T6 lifecycle signals — handled in message_received (Phase 4)
    api.registerHook("message_received", async ({ message }) => {
      const match = detectSteeringRequest(message);
      if (match) await state.transitionToFocus(match.agentId, match.urgency, "AGENT_STEERING");

      // T6 — lifecycle signals
      if (state.oif_mode) {
        const lifecycle = dacsRegistry.parseLifecycleSignal(message);
        if (lifecycle && (lifecycle.signal === "DONE" || lifecycle.signal === "BLOCKED")) {
          await state.transitionToFocus(lifecycle.agentId, "HIGH", "COMPLETION");
        }
      }
    });

    // 3. After response — T2 return to REGISTRY (or T7 self-loop if relay pending)
    api.registerHook("agent_end", async () => {
      if (state.mode === "FOCUS") {
        const relay = state.pendingRelay;
        if (state.oif_mode && relay && relay.targetAgentId === state.focusAgent) {
          // api.runtime.sessions.* not in plugin runtime — use api.runtime.subagent.run() instead
          await api.runtime.subagent.run({
            sessionKey: relay.targetAgentId,
            message: relay.message,
            deliver: false,
          });
          state.pendingRelay = null;
          await state.logTransition("FOCUS", "FOCUS", state.focusAgent!, "ITERATION"); // T7
        } else {
          await state.transitionToRegistry(); // T2
        }
      }
    });

    // 4. User message — T5 OIF routing (if oif_mode) or USER_INTERACT (standard)
    api.registerHook("session_start", async ({ session }) => {
      if (session.isUserInitiated) {
        if (state.oif_mode) {
          // T5 routing happens inside assemble() — mode stays REGISTRY until matched
        }
        await state.transitionToUserInteract();
      }
    });
  },
};
```

**Plugin activation:** In `openclaw.json`:
```json
{
  "plugins": {
    "slots": { "contextEngine": "dacs" },
    "load": { "paths": ["~/.openclaw/plugins/dacs"] }
  }
}
```

---

### Phase 6 — Queue Integration (Week 5)

**What:** Configure OpenClaw's command queue to support DACS urgency-based prioritization.

**Configuration:** Set the orchestrator's queue mode to `steer` so HIGH-urgency sub-agent messages inject immediately into the current orchestrator run, enabling INTERRUPT semantics:

```json
{
  "messages": {
    "queue": {
      "mode": "steer",
      "debounceMs": 200,
      "byChannel": { "webchat": "collect" }
    }
  }
}
```

**Interrupt handling:** The DACS plugin intercepts `steer` queue injection events (via `message_received`), checks urgency, and if HIGH while in `FOCUS(aⱼ)`, emits an INTERRUPT transition before the model call.

**Sub-agent queue modes:** Sub-agents use `maxSpawnDepth: 2` so they can use `sessions_send` back to the orchestrator. Set their queue to `collect` to avoid thrashing.

---

### Phase 7 — Observability (Week 6)

**What:** Make DACS events visible in OpenClaw's native logging and UI.

**Logging:** Emit structured log events via OpenClaw's `logging` hooks:
```
[DACS] TRANSITION: REGISTRY → FOCUS(a3) — urgency=HIGH question="Which encoding for UTF-8 errors?"
[DACS] CONTEXT_BUILT: mode=FOCUS agent=a3 tokens=5847 registry_tokens=142
[DACS] TRANSITION: FOCUS(a3) → REGISTRY — response_sent
[DACS] INTERRUPT: FOCUS(a1) → FOCUS(a2) — HIGH urgency preempt
```

**Chat command:** `/dacs-status` — shows current mode, active focus agent, registry entries with token counts, and contamination risk.

**Gateway HTTP route:**
```typescript
api.registerHttpRoute({
  path: "/dacs/state",
  auth: "gateway",
  handler: async (_req, res) => {
    res.end(JSON.stringify(await state.snapshot()));
    return true;
  },
});
```

**Metrics tracked (matching DACS paper M1–M7):**
- `M1`: Steering accuracy (LLM judge on orchestrator responses)
- `M2`: Contamination rate (other agent IDs in steering responses)
- `M3`: Context tokens at steering time
- `M4`: User message → response latency
- `M5`: Sub-agent `sessions_send` → orchestrator response latency
- `M6`: Registry truncation rate
- `M7`: INTERRUPT rate

---

### Phase 8 — Sub-Agent Skill (Week 6)

**What:** A DACS-compatible skill for sub-agent sessions that teaches them to emit `[[STEER: ...]]` markers and manage urgency correctly.

**File:** `~/.openclaw/workspace/skills/dacs-agent/SKILL.md`

**Teaches the sub-agent to:**
1. Emit `[[STEER: specific question | urgency: HIGH/MEDIUM/LOW]]` when it hits a genuine decision point
2. Continue working (don't block) after emitting — wait for orchestrator guidance in the next message
3. Include relevant local context in the `sessions_send` payload (the `relevant_context` field in DACS terms)
4. Emit `[[DONE: one-sentence summary]]` when a significant milestone is reached (triggers OIF T6 — orchestrator enters Focus to review and assign next task)
5. Emit `[[BLOCKED: reason]]` when it cannot proceed and needs orchestrator intervention (triggers OIF T6)
6. Emit `[[PROGRESS: brief update]]` for routine heartbeats that do not require orchestrator attention (registry update only, no focus transition)

**Signal vocabulary summary:**

| Marker | When to emit | Orchestrator action |
|---|---|---|
| `[[STEER: question \| urgency: X]]` | Needs a decision to continue | Enters Focus(aᵢ) — T1/T3 |
| `[[DONE: summary]]` | Milestone complete | Enters Focus(aᵢ) to review — T6 |
| `[[BLOCKED: reason]]` | Cannot proceed | Enters Focus(aᵢ) to unblock — T6 |
| `[[PROGRESS: update]]` | Routine step complete | Registry update only — no Focus |

This mirrors what `agents/llm_agent.py` does in the Python harness (lifecycle signals added in Phase 5), now in OpenClaw's skill system.

---

### Phase 9 — Validation Experiment (Week 7–8)

**What:** Run a formal DACS vs baseline comparison inside OpenClaw, analogous to `experiments_oif/` (DACS Phase 5) but using native OpenClaw sessions.

**Scenarios:**
- `oc1_n3` — 3 sub-agent sessions (BST / transformer survey / CSV pipeline), each emitting 3 steering requests. Same task domains as `s1_n3`.
- `oc1_n3_oif` — same agents and tasks, with 3 timed user queries mid-trial (one per agent domain), added on top of steering requests. Tests T5 routing in live OpenClaw sessions.

**Conditions (matching `experiments_oif/` three-condition design):**
- `dacs_oif`: DACS context engine active + OIF flag enabled — full T1–T7 state machine
- `dacs_no_oif`: DACS context engine active, OIF disabled — Focus only on explicit SteeringRequests
- `baseline_oc`: Context engine disabled — flat `sessions_history` of all agents for every call

**Harness:** A CLI script that:
1. Spawns 3 sub-agent sessions via `openclaw agent`
2. Runs them in parallel; injects timed user queries via the same `UserInjector` mechanism used in `experiments_oif/`
3. Records context sizes at each `sessions_send` call (via Gateway WS events)
4. Collects orchestrator responses to both steering requests and user queries
5. Runs LLM judge on responses (GPT-4o-mini, same rubric as `experiments_oif/judge.py`)

**Expected results (from DACS Phase 5 paper results):**
| Metric | `dacs_oif` | `dacs_no_oif` | `baseline_oc` |
|---|---|---|---|
| Steering accuracy (M1) | ~90–98% | ~90–98% | ~40–60% |
| User-query routing (M1_T5) | ~100% | ~28–40% | ~28–33% |
| Context at steering time | ~675–742 tokens | ~675–742 tokens | ~1,400–2,300 tokens |
| Context ratio vs baseline | 2.1–3.1× | 2.1–3.1× | 1× |
| Contamination rate (M2) | <4% | <4% | 18–42% |

The T5 routing gap (OIF 100% vs no-OIF 28–40%) is the primary new signal beyond previous phases.

---

## 5. Repository Structure (New Files)

```
packages/dacs/                       # New npm package
  package.json                       # @openclaw/dacs
  openclaw.plugin.json               # Plugin manifest
  src/
    index.ts                         # Plugin entry — register(api)
    dacs/
      state.ts                       # DacsState — mode, focus_agent, oif_mode, trigger_type, transitions
      registry.ts                    # DacsRegistry — per-agent summaries ≤200 tok + lifecycle signal parser
      context-builder.ts             # DacsContextBuilder — REGISTRY/FOCUS assembly + T5 assemble() hook
      steering-detector.ts           # detectSteeringRequest() — [[STEER:...]] parser
      oif-router.ts                  # OIF T5: matchUserQueryToAgent() prefix-stem matcher; T7 relay logic
      token-counter.ts               # Thin wrapper around OpenClaw's counter
    types.ts                         # DacsMode, AgentUrgency, RegistryEntry, TriggerType, OifSignal, etc.
  tests/
    registry.test.ts
    context-builder.test.ts
    steering-detector.test.ts
    oif-router.test.ts               # T5 routing accuracy + T7 relay unit tests

skills/dacs/
  SKILL.md                           # Orchestrator skill: DACS-aware orchestration including OIF 

skills/dacs-agent/
  SKILL.md                           # Sub-agent skill: [[STEER:]], [[DONE:]], [[BLOCKED:]], [[PROGRESS:]]

experiments_openclaw/                # New experimental harness
  __init__.py
  run.py                             # CLI entry: orchestrate via openclaw CLI
  scenario_defs.py                   # oc1_n3, oc1_n3_oif (with timed user queries)
  analyze.py                         # Post-hoc metrics: M1, M1_T5, M2, M3, context ratio
  judge.py                           # LLM judge for orchestrator responses (steering + user queries)
```

---

## 6. What DACS Adds That OpenClaw Currently Lacks

| Capability | OpenClaw Today | With DACS |
|---|---|---|
| Multi-agent context sharing | All sessions_history loaded simultaneously (context pollution) | Isolated per-agent focus windows; compact registry for idle agents |
| Steering quality at scale | Degrades as N grows (measured: -40pp at N=10) | Maintained accuracy across N ∈ {3,5,10} |
| Context cost | O(N × agent_steps) tokens | O(focus_agent_steps + N × 200) tokens |
| Urgency-aware focus | Not applicable | HIGH urgency triggers immediate FOCUS transition |
| Inter-agent contamination | High (18–42% in experiments) | Low (<4% in experiments) |
| User query answering | All agent histories loaded for every user question | OIF T5: enters Focus(aᵢ) when query matches aᵢ's domain — 100% routing accuracy vs 28–33% flat (Phase 5) |
| Agent lifecycle visibility | Orchestrator blind to milestones unless polled | OIF T6: [[DONE:]]/[[BLOCKED:]] signals trigger automatic Focus review; [[PROGRESS:]] updates registry only |
| Multi-turn iteration cost | Context switches on every orchestrator turn | OIF T7: self-loop keeps orchestrator in Focus(aᵢ) for consecutive directives — amortises switch overhead |
| Compaction complement | `/compact` summarises entire session | DACS prevents overload proactively; compaction handles long focus sessions |

---

## 7. What OpenClaw Adds That the Research Harness Lacked

| Capability | DACS Research Harness | OpenClaw Integration |
|---|---|---|
| Real messaging surfaces | Simulated asyncio agents | WhatsApp / Telegram / Slack / Discord / etc. |
| Real tool calls | Scripted responses | Browser, bash, canvas, cron, webhooks |
| Session persistence | In-memory only | JSONL transcripts, daily/idle reset, session history |
| User interaction | Simulated | Real users on real messaging channels |
| Production hardening | Research prototype | Gateway daemon, launchd/systemd, health checks |
| Compaction | Not implemented | OpenClaw's built-in compaction + DACS registry = complementary |

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `api.registerContextEngine` surface stability (marked evolving) | Build on it now; it is the declared direction. Fall back to `before_prompt_build` hook if needed. |
| Sub-agent `sessions_send` payloads may not carry enough context for DACS registry | Supplement with `sessions_history` digest in `ingest()` |
| `[[STEER:]]` marker convention may be unreliable for some models | Keep structured alternative (`{"type": "steer", "question": "...", "urgency": "HIGH"}`) as fallback |
| Token counter mismatch between OpenClaw and tiktoken | Use OpenClaw's native counter in the plugin; validate against tiktoken offline |
| OpenClaw's serialized per-session queue conflicts with DACS parallel FOCUS sessions | DACS plugin manages state transitions; queue `mode: steer` enables real-time injection |
| INTERRUPT semantics require aborting mid-tool-call | Use `steer` queue mode which aborts at next tool boundary — sufficient for research purposes |
| T5 prefix-stem matcher may mis-route ambiguous user queries at large N | Require ≥2 stem overlaps; fall back to registry-only response if no confident match; validated accurate at N∈{3,5} (Phase 5, 100% hit rate) |
| T6 lifecycle signals may be emitted spuriously by weaker models | Guard with confidence threshold: only trigger Focus if signal appears as a standalone suffix, not mid-sentence; `PROGRESS` never triggers Focus |
| `api.runtime.sessions.{list,history,send}` are not in the plugin `api.runtime.*` surface — they are agent-facing tools only | Registry updates happen in the `after_tool_call` hook (intercepts `sessions_history` calls made by the orchestrator agent itself). T7 relay uses `api.runtime.subagent.run({ sessionKey, message, deliver: false })` which IS documented. The `ingest()` method reloads state from disk only. |

---

## 9. Recommended Starting Point

**Immediate (this week):**
1. Create `skills/dacs/SKILL.md` — orchestrator skill (prompt-level DACS + OIF routing instructions)
2. Create `skills/dacs-agent/SKILL.md` — sub-agent skill with `[[STEER:]]`, `[[DONE:]]`, `[[BLOCKED:]]`, `[[PROGRESS:]]` conventions
3. Spawn 3 sub-agents in OpenClaw and test manually: does context isolation improve? Does a user query about agent a1 route into Focus(a1)?

**Next (2–4 weeks):**
1. Build `packages/dacs/src/dacs/state.ts`, `registry.ts`, `context-builder.ts`
2. Build `oif-router.ts` — prefix-stem T5 matcher + T7 relay logic
3. Register the context engine plugin with `oif_mode` flag
4. Wire the `message_received` hook for steering detection (T1/T3) and lifecycle signals (T6)

**Validation (4–6 weeks):**
1. Implement `experiments_openclaw/` harness with `oc1_n3` and `oc1_n3_oif` scenarios
2. Run three-condition comparison: `dacs_oif` / `dacs_no_oif` / `baseline_oc`
3. Compare to published Phase 5 results: expect ~100% T5 routing (OIF), ~28–40% without OIF, ~28–33% flat baseline
