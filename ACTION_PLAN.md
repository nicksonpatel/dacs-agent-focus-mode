# DACS: Dynamic Attentional Context Scoping
## Research Prototype & Paper — 8-Week Action Plan

---

## The Core Idea

An orchestrator managing N parallel agents suffers from **context pollution** — all agent threads compete in the same context window, degrading steering quality. 

**DACS** introduces agent-triggered focus sessions: when an agent needs steering, it raises a request with its own context attached. The orchestrator enters focus mode — isolating that agent's context + lightweight summaries of all others — steers the agent, then exits back to the lightweight registry view. The orchestrator stays responsive to the user at all times.

---

## Key Differentiator from Prior Art

| Mechanism | How context is managed |
|---|---|
| AFM (2511.12712) | Passive compression — all context uniformly tiered by recency/saliency |
| AOI (2512.13956) | Three-layer memory, static architecture |
| **DACS** | **Active, asymmetric, agent-triggered isolated context sessions** |

The novelty: context scoping is **intentional, dynamic, and asymmetric** — not a background compression pass, but a deliberate mode switch triggered by the agent that needs attention.

---

## Phase 1 — Literature & Formal Definition (Days 1–10)

**Goal:** Know exactly where your idea sits relative to prior art before writing a line of code.

### Week 1 Reading List (in order)

1. **[arxiv 2511.12712] AFM — Adaptive Focus Memory**
   - Note: How does it tier context? What triggers FULL vs COMPRESSED vs PLACEHOLDER?
   - Question to answer: Does AFM work across multiple concurrent agent threads, or single-thread?
   - Record: exact mechanism for deciding what gets compressed

2. **[arxiv 2512.13956] AOI — Three-layer memory + dynamic scheduling**
   - Note: The 72.4% compression / 92.8% critical info retention numbers — how measured?
   - Question to answer: Does AOI handle orchestrator-to-agent steering specifically?

3. **[arxiv 2506.12508] AgentOrchestra**
   - Note: What "cross-entity lifecycle and context management" gaps do they identify?
   - This is your strongest related-work ally — they name the problem you're solving

4. **[arxiv 2510.04618] ACE — Evolving Context**
   - Note: How is context treated as a first-class concept?
   - Record any benchmarks they use (you may reuse or extend them)

5. **[arxiv 2602.16873] AdaptOrch — Topology as optimization frontier**
   - Note: How do they frame the shift from model selection to orchestration structure?
   - This is your motivation section

### Deliverables by Day 10

- [ ] **Reading notes file** (`/notes/literature-review.md`) — one page per paper: mechanism, metrics, gaps, what DACS does differently
- [ ] **Formal definition draft** (`/notes/formal-definition.md`) — see template below

### Formal Definition Template

```
DACS Mechanism — Formal Definition

Entities:
  - Orchestrator O
  - Agent set A = {a1, a2, ..., aN}
  - Context window C (size limit T tokens)
  - Registry R: lightweight state snapshot per agent (target: ≤ 200 tokens each)
  - Focus context F(ai): full context of agent ai needed for steering

States:
  - REGISTRY mode: O holds R only — can respond to user, monitor all agents
  - FOCUS(ai) mode: O holds F(ai) + R-compressed — steers agent ai exclusively

Transitions:
  - REGISTRY → FOCUS(ai): triggered by ai emitting SteeringRequest(context, urgency, blocking)
  - FOCUS(ai) → REGISTRY: triggered by SteeringComplete or Interrupt(aj, urgency=HIGH)
  
Interrupt protocol:
  - While in FOCUS(ai), if aj emits SteeringRequest(urgency=HIGH):
    - Save current steering state for ai
    - Emit SteeringPaused(ai)
    - Transition to FOCUS(aj)
    - After handling, resume FOCUS(ai) or return to REGISTRY

Summary schema for R (per agent):
  {
    agent_id: str,
    task: str (≤50 tokens),
    status: RUNNING | BLOCKED | WAITING_STEERING | COMPLETE | FAILED,
    last_output_summary: str (≤100 tokens),
    last_updated: timestamp,
    urgency: LOW | MEDIUM | HIGH
  }
```

---

## Phase 2 — Architecture Design (Days 7–14)

*Overlap with Phase 1 reading — design while reading*

### Core Components to Specify

**A. Registry Manager**
- Maintains per-agent state snapshots
- Updated by agent heartbeats (every N steps or on status change)
- Serialized to JSON, token-counted before injection

**B. Steering Request Protocol**
- Agent emits structured SteeringRequest when it needs orchestrator input
- Contains: relevant_context, blocking (bool), urgency level, specific question
- Orchestrator queues requests; HIGH urgency can interrupt focus mode

**C. Context Builder**
- `build_focus_context(agent_id)` → full agent context + compressed registry
- `build_registry_context()` → all agent summaries only
- Must respect token budget — if F(ai) + registry summaries > T, compress registry further

**D. Orchestrator State Machine**
- Explicit state: REGISTRY | FOCUS(agent_id) | USER_INTERACTION
- Transitions logged with timestamps (for experiment measurement)
- User messages always handled — they queue if orchestrator is in FOCUS, get answered when done

### Deliverable by Day 14
- [ ] Architecture diagram (`/docs/architecture.md`)
- [ ] Component interface specs (`/docs/interfaces.md`)

---

## Phase 3 — Prototype Implementation (Days 14–35)

**Stack:** Minimal custom Python harness (~300 lines) — full context observability, no framework overhead

> **Why not OpenClaw?** OpenClaw is a personal assistant daemon built around a single embedded agent runtime and a five-component architecture (Gateway, Brain, Memory, Skills, Heartbeat). Its context assembly happens inside opaque internals you'd need to fork to intercept. It also injects noise (session management, Markdown memory files, skill loading, heartbeat scheduler) that would contaminate the variables you're trying to measure. A custom harness gives you complete, clean control over exactly what goes into each LLM call — which is the entire point of the experiment.
>
> OpenClaw's role in the paper: motivation and context. "Systems like OpenClaw demonstrate real-world demand for persistent multi-agent orchestration, but their orchestrators suffer from context pollution as concurrent agent count grows..." Integration into production frameworks is future work.

### Week 3 — Core Mechanism (Days 14–21)

- [ ] Implement Registry Manager (`/src/registry.py`)
- [ ] Implement SteeringRequest protocol (`/src/protocols.py`)
- [ ] Implement Context Builder with token budget enforcement (`/src/context_builder.py`)
- [ ] Implement Orchestrator state machine (`/src/orchestrator.py`)

### Week 4 — Agent Stubs & Integration (Days 21–28)

- [ ] Implement 3 stub agent types for experiments:
  - `CodeWriterAgent` — writes code incrementally, occasionally needs design decisions
  - `ResearchAgent` — searches and summarizes, occasionally needs query clarification
  - `DataProcessorAgent` — transforms data, occasionally hits ambiguous format choices
- [ ] Integrate agents with orchestrator via SteeringRequest protocol
- [ ] End-to-end run with 3 agents, manual validation

### Week 5 — Baseline & Hardening (Days 28–35)

- [ ] Implement **baseline orchestrator** — flat context, no focus mode (all agent contexts always injected)
- [ ] Add logging: every steering action, context size at time of steering, wall clock time 
- [ ] Add interrupt mechanism for HIGH urgency requests
- [ ] Run 5 manual trials to validate mechanics work
- [ ] Fix obvious bugs

---

## Phase 4 — Experiment Design & Execution (Days 35–49)

### Experiment: DACS vs Flat-Context Baseline

**Setup:**
- N agents running in parallel (test N = 3, 5, 10)
- Each agent is assigned a task with known correct steering decisions embedded
- Orchestrator must steer each agent at decision points

**Independent Variable:** DACS focus mode vs baseline flat-context

**Dependent Variables (what you measure):**

| Metric | How to measure |
|---|---|
| **Steering accuracy** | Does orchestrator give correct decision at agent's decision point? Score 0/1 per decision |
| **Wrong-agent contamination** | Does orchestrator's response reference context from the wrong agent? NLP similarity check |
| **Context size at steering time** | Log token count of context window when steering decision is made |
| **User responsiveness** | Time from user message to orchestrator response (should be similar to baseline) |
| **Task completion rate** | Did agents complete tasks correctly end-to-end? |

**Task Suite (keep it simple and reproducible):**

- 5 task scenarios, each with 3 pre-defined decision points
- Correct answer at each decision point is known in advance
- Randomize agent assignment per trial
- 10 trials per condition (DACS / baseline) per N value

**Expected result hypothesis:**
- Steering accuracy: DACS > baseline, gap widens as N increases
- Wrong-agent contamination: DACS significantly lower
- Context size: DACS significantly smaller
- User responsiveness: DACS ≈ baseline (no regression)

### Deliverables by Day 49
- [ ] Experiment harness written and validated (`/experiments/`)
- [x] All trials run (N = 3, 5, 10 × 10 trials × 2 conditions = 60 runs) — **COMPLETE**
- [x] Results in CSV + visualizations (`/results/`) — **COMPLETE**

**Phase 1 headline numbers:**
| N | DACS acc | Base acc | Delta | DACS ctx | Base ctx | Ratio |
|---|----------|----------|-------|----------|----------|-------|
| 3 | 96.7% | 60.0% | +36.7 pp | 561 tok | 1,191 tok | 2.12× |
| 5 | 96.7% | 38.7% | +58.0 pp | 633 tok | 1,720 tok | 2.72× |
| 10 | 90.0% | 21.0% | +69.0 pp | 816 tok | 2,883 tok | 3.53× |

Full analysis: `results/PHASE1_RESULTS.md`  
Figures: `results/figures/phase1_overview.png`

---

## Phase 5 — Paper Writing (Days 35–56)

*Writing runs in parallel with experiments — draft while experiments run*

### Paper Structure (target: 8 pages, workshop format)

**1. Introduction** (1 page)
- Problem: orchestrator context pollution in concurrent multi-agent systems
- Gap: prior work compresses passively, doesn't address steering-time context isolation
- Contribution: DACS mechanism + empirical validation

**2. Related Work** (1 page)
- AFM, AOI, AgentOrchestra, ACE, AdaptOrch
- Precise differentiator table (expand the one above)

**3. The DACS Mechanism** (2 pages)
- Formal definition (from Phase 1 deliverable)
- Architecture diagram
- State machine transitions
- Summary schema and token budget design

**4. Experiments** (2 pages)
- Setup, task suite, metrics
- Results tables and graphs
- Analysis: why does DACS help? (context size reduction → fewer cross-agent references)

**5. Discussion & Limitations** (1 page)
- Summary quality is load-bearing — bad summaries hurt
- Preemption latency tradeoffs
- Doesn't address distributed orchestration
- Future: learned summary compression, automatic urgency detection

**6. Conclusion** (0.5 page)

### Writing Timeline
- Days 35–42: Sections 1, 2, 3 (these don't need results)
- Days 42–49: Sections 4, 5, 6 (fill in as results come in)
- Days 49–56: Full revision, proofread, arxiv formatting

---

## Phase 6 — Publication (Days 50–56)

### Submission Targets (in order)

1. **arxiv preprint first** — establish timestamp, get community feedback
   - Category: cs.AI or cs.MA (Multi-Agent Systems)
   - Submit by end of Day 56

2. **Workshop targets** (check deadlines — apply to whichever is open)
   - NeurIPS 2026 Agent/Multi-Agent workshops (typically July deadline)
   - ICLR 2027 workshops
   - AAMAS 2026 (International Conference on Autonomous Agents and Multi-Agent Systems)

3. **If results are strong:** Main track of AAAI 2027 or AAMAS 2026

### arxiv Checklist
- [ ] Paper formatted in LaTeX (use arxiv template)
- [ ] Code released on GitHub with README
- [ ] Reproducibility: experiment harness runnable from one script
- [ ] Abstract clearly states the contribution and key numbers

---

## Week-by-Week Summary

| Week | Focus | Key Deliverable |
|---|---|---|
| 1 | Read 5 papers | Literature notes + formal definition draft |
| 2 | Architecture design + finish reading | Architecture doc + interface specs |
| 3 | Core mechanism implementation | Registry, context builder, state machine |
| 4 | Agents + integration | End-to-end 3-agent run working |
| 5 | Baseline + paper sections 1–3 | Experiment harness ready, draft intro/related/method |
| 6 | Run experiments | All 60 trials complete |
| 7 | Analyze results + write sections 4–6 | Full paper draft |
| 8 | Revision + arxiv submission | Paper live on arxiv |

---

## Critical Path Items

These are the things most likely to cause delays — address them early:

1. **LLM API choice** — the custom harness needs to call an LLM for the orchestrator. Decide in Week 1: OpenAI API, Anthropic API, or a local model (Ollama). The choice affects cost for 60 experiment runs. Anthropic or a local model keeps costs low while preserving full context control.

2. **Summary quality** — the registry summary schema is load-bearing. Test it manually in Week 3 before building experiments around it. A bad summary design will invalidate all your results.

3. **Experiment harness correctness** — the "known correct decision" task suite must be validated manually before automated runs. Run 5 manual trials in Week 4 and check the ground truth yourself.

4. **Timeline risk** — the biggest risk is someone publishing something similar while you're working. Prioritize the arxiv preprint over a polished workshop submission. A preprint in 8 weeks beats a perfect paper in 6 months.

---

## Repository Structure

```
agent-focus-mode/
├── ACTION_PLAN.md          ← this file
├── notes/
│   ├── literature-review.md
│   └── formal-definition.md
├── docs/
│   ├── architecture.md
│   └── interfaces.md
├── src/
│   ├── registry.py
│   ├── protocols.py
│   ├── context_builder.py
│   └── orchestrator.py
├── agents/
│   ├── base_agent.py
│   ├── code_writer_agent.py
│   ├── research_agent.py
│   └── data_processor_agent.py
├── experiments/
│   ├── task_suite.py
│   ├── run_experiment.py
│   └── metrics.py
├── results/
│   └── (auto-generated CSVs and plots)
└── paper/
    ├── main.tex
    └── figures/
```

---

## First Actions (Do These Today)

1. Decide which LLM API the harness will call (OpenAI / Anthropic / local Ollama) — this determines experiment cost
2. Download and skim abstract + conclusion of all 5 papers (full read Week 1)
3. Create the `/notes/` folder and start `literature-review.md`
4. Set up the repo structure above

---

*Start date: April 2026 | Target arxiv submission: early June 2026*
