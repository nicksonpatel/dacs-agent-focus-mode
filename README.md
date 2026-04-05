# DACS — Dynamic Attentional Context Scoping

> **Research prototype & arxiv preprint**  
> Tag [`phase-3-final`](../../releases/tag/phase-3-final) — all paper results reproducible from this tag.

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

## Experimental Results (3 Phases, 160 Trials)

### Phase 1 — Agent count scaling (N ∈ {3, 5, 10}, 60 trials)

| N  | DACS accuracy | Baseline accuracy | Accuracy gain | Context ratio |
|----|---------------|-------------------|---------------|---------------|
| 3  | **96.7%**     | 60.0%             | +36.7 pp      | 2.12×         |
| 5  | **96.7%**     | 38.7%             | +58.0 pp      | 2.72×         |
| 10 | **90.0%**     | 21.0%             | +69.0 pp      | 3.53×         |

All differences: *p* < 0.0001. DACS context grows at ~+25 tokens per additional agent (sub-linear); baseline grows near-linearly.

### Phase 2 — Agent diversity (3 scenarios, 60 trials)

| Scenario | Description | DACS accuracy | Δ accuracy | Context ratio |
|---|---|---|---|---|
| s4 — homogeneous | 3 × algorithm coders | **90.2%** | +37.7 pp | 2.29× |
| s5 — crossfire | 5 × maximally diverse domains | **96.0%** | +59.0 pp | 2.90× |
| s6 — cascade | 5 agents with inter-agent dependencies | **94.0%** | +37.3 pp | 2.65× |

LLM-as-judge validation on s5 (400 decisions): agreement 98.0%, Cohen's κ = 0.956.

### Phase 3 — Decision density scaling (2 scenarios, 40 trials)

| Scenario | N | D (decisions/agent) | DACS accuracy | Δ accuracy | Context ratio |
|---|---|---|---|---|---|
| s7 — dense | 5 | 8 | **94.0%** | +59.2 pp | 3.24× |
| s8 — ultra-dense | 3 | 15 | **98.4%** | +54.2 pp | 2.39× |

LLM-as-judge validation: κ = 0.933 (s7), κ = 0.886 (s8), mean κ = 0.909. All results in `results/`.

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
src/                     Core mechanism (~300 lines total)
  ├── orchestrator.py    DACS orchestrator (focus/registry mode switching)
  ├── registry.py        Per-agent state snapshot store (≤200 tokens each)
  ├── context_builder.py Token-counted context assembly & hard cap
  ├── protocols.py       SteeringRequest / SteeringComplete data structures
  ├── logger.py          Full context-window logging for every LLM call
  └── monitor.py         Real-time registry + focus-mode observer

agents/                  Agent stub implementations
  ├── base_agent.py      Abstract base with SteeringRequest emission
  ├── code_writer_agent.py
  ├── research_agent.py
  ├── data_processor_agent.py
  ├── debugger_agent.py
  ├── generic_agent.py   Used for Phase 2 & 3 scenarios
  └── long_writer_agent.py

experiments/             Experiment harness
  ├── run_experiment.py  Single entry point — all phases, all scenarios
  ├── task_suite.py      8 scenarios (s1–s8), known-correct answers per decision point
  ├── metrics.py         Steering accuracy, contamination, context size
  ├── llm_judge_phase3.py  LLM-as-judge validation for Phase 3 (s7, s8)
  ├── llm_judge_s8.py    Focused judge pass for s8_n3_dense_d3
  ├── plot_phase1.py     Phase 1 figures
  └── plot_phase2_phase3.py  Phase 2 & 3 figures

results/                 Auto-generated — do not edit manually
  ├── PHASE1_RESULTS.md  Phase 1 analysis (60 trials)
  ├── PHASE2_RESULTS.md  Phase 2 analysis (60 trials)
  ├── PHASE3_RESULTS.md  Phase 3 analysis (40 trials)
  ├── llm_judge_phase3_s7.csv   Judge validation data for s7
  ├── llm_judge_phase3_s8.csv   Judge validation data for s8
  └── *.jsonl            Per-trial full context-window logs

paper/
  ├── draft_v2.tex       Full paper (3 phases, all results)
  ├── main.tex           Phase 1 only version
  ├── refs.bib           Bibliography
  └── figures/           Paper figures (PNG)

notes/
  ├── literature-review.md   Per-paper notes: AFM, AOI, AgentOrchestra, ACE, AdaptOrch
  └── formal-definition.md   Formal DACS definition (entities, states, transitions)

docs/
  ├── architecture.md        System architecture diagram
  └── interfaces.md          Component interface specifications

requirements.txt           Python dependencies
```

---

## Setup & Reproducing Results

**Requirements:** Python 3.11+, API key for an Anthropic-compatible endpoint.

```bash
git clone <repo>
cd agent-focus-mode
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set your API key (the paper used MiniMax-M2.7 via an Anthropic-compatible endpoint)
export MINIMAX_API_KEY=<your-key>
export DACS_MODEL=MiniMax-M2.7   # optional — this is the default
```

### Reproduce Phase 1 (N ∈ {3, 5, 10}, 60 trials)
```bash
python -m experiments.run_experiment --scenario s1_n3 s2_n5 s3_n10 --mode both --trials 10
python -m experiments.plot_phase1
```

### Reproduce Phase 2 (agent diversity, 60 trials)
```bash
python -m experiments.run_experiment --scenario s4_n3_homogeneous s5_n5_crossfire s6_n5_cascade --mode both --trials 10
python -m experiments.plot_phase2_phase3
```

### Reproduce Phase 3 (decision density, 40 trials)
```bash
python -m experiments.run_experiment --scenario s7_n5_dense_d2 s8_n3_dense_d3 --mode both --trials 10 --parallel-trials 4
python -m experiments.llm_judge_phase3  # runs LLM-as-judge validation
python -m experiments.plot_phase2_phase3
```

### Quick smoke test (1 trial per condition, ~10 min)
```bash
python -m experiments.run_experiment --scenario s1_n3 --mode both --trials 1
```

Results are written to `results/summary.csv` (one row per trial) and `results/<run_id>.jsonl` (full context-window log for every LLM call in that trial).

> **Reproducibility note:** All paper results were produced at git tag `phase-3-final`. Check out that tag to guarantee identical code.

---

## Key Design Constraints

- **Observability is paramount.** Every LLM call logs the exact token contents of the context window. This is the central experiment variable.
- **Token budget is deterministic.** `context_builder.py` counts tokens before every call and hard-caps at *T* = 204,800. The provider never truncates.
- **Fair baseline.** The flat-context baseline uses the identical code path as DACS — only `build_focus_context` is replaced by `build_flat_context`. No other differences.
- **No external orchestration frameworks.** No LangGraph, CrewAI, etc. The harness is intentionally minimal (~300 lines) to keep full observability over context window contents.

---

## Experiment Status

| Phase | Status | Scenarios | Trials | RQ answered |
|---|---|---|---|---|
| Phase 1 | ✅ Complete | s1 (N=3), s2 (N=5), s3 (N=10) | 60 | Does DACS accuracy scale with N? |
| Phase 2 | ✅ Complete | s4 (homogeneous), s5 (crossfire), s6 (cascade) | 60 | Does advantage hold across agent diversity? |
| Phase 3 | ✅ Complete | s7 (N=5, D=8), s8 (N=3, D=15) | 40 | Does advantage grow with decision density? |

---

## Paper

The submission paper is `paper/draft_v2.tex` (3 phases, all results, LLM-as-judge validation). Closest related work: AFM and AOI — see `notes/literature-review.md` for precise distinctions.

---

## License

Research prototype. Not yet licensed for redistribution.
