# DACS — Dynamic Attentional Context Scoping

> **Research prototype & arxiv preprint**  
> Target venues: NeurIPS / ICLR / AAMAS workshop · 8-week timeline from April 2026

---

## What is DACS?

An orchestrator managing *N* parallel agents suffers from **context pollution** — all agent threads compete in the same context window, degrading steering quality as *N* grows.

**DACS** (Dynamic Attentional Context Scoping) introduces **agent-triggered focus sessions**: when an agent needs steering, it raises a `SteeringRequest` with its own context attached. The orchestrator enters *focus mode* — isolating that agent's full context plus lightweight summaries of all others — steers the agent, then exits back to a lightweight registry view. The orchestrator stays responsive to the user at all times.

### How DACS differs from prior work

| Mechanism | How context is managed |
|---|---|
| AFM ([2511.12712](https://arxiv.org/abs/2511.12712)) | Passive compression — all context uniformly tiered by recency / saliency |
| AOI ([2512.13956](https://arxiv.org/abs/2512.13956)) | Three-layer memory, static architecture |
| **DACS** | **Active, asymmetric, agent-triggered isolated context sessions** |

Context scoping in DACS is *intentional, dynamic, and asymmetric* — not a background compression pass, but a deliberate mode switch triggered by the agent that needs attention.

---

## Experimental Results (Phase 1 & 2)

### Phase 1 — Canonical 60-trial experiment (N ∈ {3, 5, 10})

| N  | DACS accuracy | Baseline accuracy | Accuracy gain | Context ratio |
|----|---------------|-------------------|---------------|---------------|
| 3  | **96.7%**     | 60.0%             | +36.7 pp      | 2.12×         |
| 5  | **96.7%**     | 38.7%             | +58.0 pp      | 2.72×         |
| 10 | **90.0%**     | 21.0%             | +69.0 pp      | 3.53×         |

All differences: *p* < 0.0001. DACS context grows at ~+25 tokens per additional agent (sub-linear); baseline grows near-linearly.

### Phase 2 — Agent diversity expansion (3 scenarios)

| Scenario | Description | DACS accuracy | Δ accuracy | Context ratio |
|---|---|---|---|---|
| s4 — homogeneous | 3 × algorithm coders | **90.2%** | +37.7 pp | 2.29× |
| s5 — crossfire | 5 × maximally diverse domains | **96.0%** | +59.0 pp | 2.90× |
| s6 — cascade | 5 agents with inter-agent dependencies | **94.0%** | +37.3 pp | 2.65× |

**Key insight:** The DACS advantage is largest when agent domains are maximally diverse (s5: C++ systems, diffusion models, genomics, memory debugging, clinical writing), confirming that cross-domain semantic bleed is the primary contamination vector in the flat-context baseline.

---

## Core Mechanism

```
DACS States
───────────
REGISTRY mode   — Orchestrator holds only R (≤200-token summaries per agent)
                  Can respond to user, monitor all agents
FOCUS(aᵢ) mode  — Orchestrator holds F(aᵢ) + R-compressed
                  Steers agent aᵢ exclusively

Transitions
───────────
REGISTRY → FOCUS(aᵢ)  : triggered by aᵢ emitting SteeringRequest
FOCUS(aᵢ) → REGISTRY  : triggered by SteeringComplete or HIGH-urgency Interrupt
```

Token budget is enforced deterministically by `src/context_builder.py` before every LLM call — the provider never truncates silently.

---

## Repository Structure

```
src/                     Core mechanism
  ├── orchestrator.py    DACS orchestrator (focus/registry mode switching)
  ├── registry.py        Per-agent state snapshot store (≤200 tokens each)
  ├── context_builder.py Token-counted context assembly & hard cap
  ├── protocols.py       SteeringRequest / SteeringComplete data structures
  ├── logger.py          Full context-window logging for every LLM call
  └── monitor.py         Real-time registry + focus-mode observer

agents/                  Agent implementations
  ├── base_agent.py      Abstract base with SteeringRequest emission
  ├── code_writer_agent.py
  ├── research_agent.py
  ├── data_processor_agent.py
  ├── debugger_agent.py
  ├── generic_agent.py
  └── long_writer_agent.py

experiments/             Experiment harness
  ├── run_experiment.py  Entry point — runs DACS vs baseline trials
  ├── task_suite.py      5 scenarios × 3+ decision points, known-correct answers
  └── metrics.py         Steering accuracy, contamination, context size, responsiveness

results/                 Auto-generated CSVs and JSONL logs — do not edit manually
  ├── PHASE1_RESULTS.md  Phase 1 analysis (N=3/5/10, 60 trials)
  └── PHASE2_RESULTS.md  Phase 2 analysis (agent diversity, 60 trials)

notes/
  ├── literature-review.md   Per-paper notes: AFM, AOI, AgentOrchestra, ACE, AdaptOrch
  └── formal-definition.md   Formal DACS definition (entities, states, transitions)

docs/
  ├── architecture.md        System architecture diagram
  ├── interfaces.md          Component interface specifications
  ├── EXPERIMENT_PLAN.md     Full experiment plan (RQ1–RQ7, all phases)
  └── PHASE_2_PLAN.md        Phase 2 detailed plan

paper/
  ├── main.tex               LaTeX paper (arxiv template)
  └── refs.bib               Bibliography

ACTION_PLAN.md             8-week research timeline
requirements.txt           Python dependencies
```

---

## Setup & Running

**Requirements:** Python 3.11+, an OpenAI- or Anthropic-compatible API key.

```bash
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY=...

# Run the full experiment suite (DACS vs baseline)
python experiments/run_experiment.py

# Plot Phase 1 results
python experiments/plot_phase1.py
```

Results are written to `results/` as JSONL files (one file per trial) and summary Markdown.

---

## Key Design Constraints

- **Observability is paramount.** Every LLM call logs the exact token contents of the context window. This is the central experiment variable.
- **Token budget is deterministic.** `context_builder.py` counts tokens before every call and hard-caps at *T*. The provider never truncates.
- **Fair baseline.** The flat-context baseline orchestrator uses the identical code path as DACS — focus mode only is disabled. No other differences.
- **No external orchestration frameworks.** No LangGraph, CrewAI, etc. The harness is intentionally minimal (~300 lines) to keep full observability over context window contents.

---

## Experiment Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | Canonical 60-trial experiment, N ∈ {3, 5, 10} |
| Phase 2 | ✅ Complete | Agent diversity: homogeneous, crossfire, cascade |
| Phase 3 | 🔜 Planned | Decision density scaling (3 → 50 steering events) |
| Phase 4 | 🔜 Planned | Long-horizon runs (hours of continuous operation) |
| Phase 5 | 🔜 Planned | Adversarial & stress tests (urgency cascades) |
| Phase 6 | 🔜 Planned | Large-scale experiments (N = 20–50) |

---

## Paper

The paper in `paper/main.tex` targets an arxiv preprint and a workshop submission (NeurIPS / ICLR / AAMAS 2026). Closest related work: AFM and AOI — see `notes/literature-review.md` for precise distinctions.

---

## License

Research prototype. Not yet licensed for redistribution.
