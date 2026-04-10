# Core Concepts

Understanding DACS starts with one question: **what does the orchestrator see?**

In a conventional multi-agent system, the orchestrator prompt looks like this:

```
[Agent A context — 2000 tokens]
[Agent B context — 1800 tokens]
[Agent C context — 2100 tokens]
---
Agent B asks: should I output CSV or JSON?
```

The orchestrator must reason about agent B's question while simultaneously holding A and C's full outputs. The signals blend together. As N grows, accuracy drops sharply.

DACS introduces two asymmetric modes.

---

## The Registry (REGISTRY mode)

In REGISTRY mode — the default — the orchestrator holds a **compact snapshot** of every agent. Each snapshot is ≤ 200 tokens and contains:

| Field | Description |
|---|---|
| `agent_id` | Unique identifier |
| `task_description` | One-sentence task (≤ 50 tokens) |
| `status` | `RUNNING` / `BLOCKED` / `WAITING_STEERING` / `COMPLETE` / `FAILED` |
| `last_output_summary` | Most recent heartbeat from the agent (≤ 100 tokens) |
| `urgency` | `LOW` / `MEDIUM` / `HIGH` |
| `last_updated` | ISO timestamp |

For N=10 agents this is still only ~2000 tokens — regardless of how much work each agent has done.

---

## Focus sessions (FOCUS mode)

When an agent calls `_request_steering()`, a `SteeringRequest` is queued. The orchestrator:

1. **Transitions** `REGISTRY → FOCUS(aᵢ)`
2. **Loads** the full context of agent *aᵢ*:
   - Task description  
   - Steering history (all previous Q&A for this agent)  
   - Recent output (from the request)  
   - The current question
3. **Adds** a compressed registry of all *other* agents (≤ 100 tokens each, stripped to status + summary)
4. Makes a **single LLM call** with this isolated context
5. **Delivers** the response to the agent
6. **Transitions** `FOCUS(aᵢ) → REGISTRY`

The orchestrator **never** sees two agents' full contexts simultaneously.

---

## Context budget invariant

DACS enforces a hard token budget *T* (default 200 000).

The context for a FOCUS(aᵢ) session is:

$$C = F(a_i) + \sum_{j \neq i} \tilde{R}(a_j)$$

Where $F(a_i)$ is the full focus context and $\tilde{R}(a_j)$ is the compressed registry entry. If $|C| > T$, DACS compresses the registry entries further (4-level priority: drop history → shorten summary → drop summary → error).

If $F(a_i)$ alone exceeds the budget, a `ContextBudgetError` is raised immediately.

---

## HIGH-urgency interrupts

If an agent submits a `SteeringRequest` with `urgency=UrgencyLevel.HIGH` while another agent is in a FOCUS session, the orchestrator:

1. Logs an `INTERRUPT` event
2. Delivers the current in-progress response (does not discard it)
3. Processes the HIGH-urgency request next

This ensures critical decisions are never blocked behind lower-priority ones.

---

## The baseline

Set `focus_mode=False` in `DACSRuntime` to run the **flat-context baseline**: all agents' full contexts are concatenated into one prompt on every steering call. This is identical to what most existing multi-agent systems do. Use it to measure DACS's improvement in your own workloads.

---

## Which class does what?

| Class | Responsibility |
|---|---|
| `DACSRuntime` | High-level wiring — creates all components, runs the event loop |
| `RegistryManager` | Stores per-agent state; enforces 200-token-per-entry limit |
| `ContextBuilder` | Assembles prompts; enforces hard token budget via tiktoken |
| `Orchestrator` | REGISTRY/FOCUS/USER_INTERACT state machine |
| `SteeringRequestQueue` | Priority queue — HIGH-urgency requests go to the front |
| `BaseAgent` | Abstract base class — subclass and implement `_execute()` |
| `StepAgent` | Ready-to-use step-driven agent (no subclassing needed) |
| `Logger` | JSONL event log + pluggable sinks |
| `TerminalMonitor` | Rich live event display (`[monitor]` extra) |

---

## Next

- [Tutorial 1: Hello, DACS](tutorials/01_hello_dacs.md) — first working agent
- [Tutorial 2: Three Concurrent Agents](tutorials/02_three_agents.md) — see FOCUS/REGISTRY in action
- [Tutorial 3: Custom Agents](tutorials/03_custom_agents.md) — subclassing `BaseAgent`
