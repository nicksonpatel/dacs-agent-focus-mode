# DACS — Dynamic Attentional Context Scoping

> **Research prototype & arxiv preprint**  
> Active paper: `paper/draft_v3.tex` — Phases 1–4 (200 trials).  
> Tag [`phase-4-final`](../../releases/tag/phase-4-final) — all 4 phases (200 trials) reproducible from this tag.

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

## Experimental Results (4 Phases, 200 Trials)

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

### Phase 4 — Real-agent validation (N ∈ {3, 5}, 40 trials)

Agents are autonomous LLMs (Claude Haiku 4.5 via OpenRouter) that write their own free-form steering questions. The orchestrator also uses Haiku, keeping a single-model setup. Validated with two independent judges.

| Scenario | N | DACS acc (Haiku judge) | Baseline acc | DACS acc (GPT-4o-mini judge) | Baseline acc | Context ratio | *p* |
|---|---|---|---|---|---|---|---|
| ra1\_n3 | 3 | **79.8%** | 62.6% | **85.4%** | 67.7% | 2.08× | 0.0023 |
| ra2\_n5 | 5 | **83.7%** | 63.3% | **89.7%** | 68.2% | 2.85× | 0.0008 |

Accuracy gain consistent with Phase 1 at matched N (+17–21 pp). Real-agent baselines outperform synthetic stubs because real agents phrase questions more precisely; DACS advantage is structurally identical. All results in `results_real_agent_haiku/`.

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
  ├── monitor.py         Real-time registry + focus-mode observer
  └── openrouter_client.py  Anthropic-compatible client for OpenRouter backend

agents/                  Agent implementations
  ├── base_agent.py      Abstract base with SteeringRequest emission
  ├── generic_agent.py   Configurable step-driven agent (Phases 1–3)
  ├── llm_agent.py       Real LLM-driven agent for Phase 4 (emits [[STEER:]] markers)
  ├── code_writer_agent.py
  ├── research_agent.py
  ├── data_processor_agent.py
  ├── debugger_agent.py
  └── long_writer_agent.py

experiments/             Synthetic agent harness (Phases 1–3)
  ├── run_experiment.py  Single entry point — all phases, all scenarios
  ├── task_suite.py      8 scenarios (s1–s8), known-correct answers per decision point
  ├── metrics.py         Steering accuracy, contamination, context size
  ├── llm_judge_phase3.py  LLM-as-judge validation for Phase 3 (s7, s8)
  ├── llm_judge_s8.py    Focused judge pass for s8_n3_dense_d3
  ├── gen_phase4_fig.py  Phase 4 comparison figure (paper/figures/phase4_comparison.png)
  ├── plot_phase1.py     Phase 1 figures
  └── plot_phase2_phase3.py  Phase 2 & 3 figures

experiments_real_agent/  Phase 4 real-agent harness
  ├── run.py             CLI entry point (OpenRouter or MiniMax backend)
  ├── scenario_defs.py   ra1_n3 and ra2_n5 scenario definitions + rubrics
  ├── judge.py           Offline LLM-as-judge (dual-judge: Haiku + GPT-4o-mini)
  ├── analyze.py         Post-hoc analysis of summary_real.csv
  ├── run_both.py        Convenience chain runner: ra1_n3 → ra2_n5 sequentially
  └── _smoke_test.py     Quick 1-trial smoke test

experiments_concurrency/ Concurrency & interruption harness (not in paper)
  ├── run.py             CLI entry point
  ├── harness.py         Trial runner with InlineJudge + UserInjector
  ├── scenario_defs.py   cc1_n3, cc2_n5 scenario definitions
  └── analyze.py         Post-hoc analysis

results/                 Auto-generated — do not edit manually
  ├── PHASE1_RESULTS.md  Phase 1 analysis (60 trials)
  ├── PHASE2_RESULTS.md  Phase 2 analysis (60 trials)
  ├── PHASE3_RESULTS.md  Phase 3 analysis (40 trials)
  ├── llm_judge_phase3_s7.csv   Judge validation data for s7
  ├── llm_judge_phase3_s8.csv   Judge validation data for s8
  └── *.jsonl            Per-trial full context-window logs

results_real_agent_haiku/  Phase 4 judge results (primary)
  ├── summary_real.csv            Aggregated metrics (40 trials)
  ├── judge_results_ra1_n3_*.csv  Dual-judge evaluations for ra1_n3
  └── judge_results_ra2_n5_*.csv  Dual-judge evaluations for ra2_n5

results_concurrency/     Concurrency experiment results
  ├── CONCURRENCY_RESULTS.md  Full analysis (44 trials)
  ├── concurrency_summary.csv Aggregated metrics
  └── *.jsonl                 Per-trial logs

paper/
  ├── draft_v3.tex       Full paper (Phases 1–4, all results) ← active submission
  ├── draft_v2.tex       Phase 1–3 only version (superseded)
  ├── main.tex           Phase 1 only version (superseded)
  ├── refs.bib           Bibliography
  └── figures/           Paper figures (PNG)

notes/
  ├── literature-review.md   Per-paper notes: AFM, AOI, AgentOrchestra, ACE, AdaptOrch
  └── formal-definition.md   Formal DACS definition (entities, states, transitions)

docs/
  ├── architecture.md        System architecture diagram
  ├── interfaces.md          Component interface specifications
  └── real_agent_experiment_architecture.md  Phase 4 design spec

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

# MiniMax (used for Phases 1–3 synthetic experiments)
export MINIMAX_API_KEY=<your-key>
export DACS_MODEL=MiniMax-M2.7   # optional — this is the default

# OpenRouter (used for Phase 4 real-agent experiments, Claude Haiku 4.5)
export OPENROUTER_API_KEY=<your-key>
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

### Reproduce Phase 4 (real-agent validation, 40 trials)
```bash
# Requires OPENROUTER_API_KEY
python -m experiments_real_agent.run --api openrouter --mode both --trials 10 --scenario ra1_n3
python -m experiments_real_agent.run --api openrouter --mode both --trials 10 --scenario ra2_n5

# Dual-judge evaluation (Haiku + GPT-4o-mini) — requires OPENROUTER_API_KEY for both models
python -m experiments_real_agent.judge \
    --models anthropic/claude-haiku-4-5 openai/gpt-4o-mini \
    --results-dir results_real_agent_haiku

# Aggregate metrics
python -m experiments_real_agent.analyze --results-dir results_real_agent_haiku

# Regenerate Phase 4 paper figure
python -m experiments.gen_phase4_fig
```

### Quick smoke test (1 trial per condition, ~10 min)
```bash
python -m experiments.run_experiment --scenario s1_n3 --mode both --trials 1
```

Results are written to `results/summary.csv` (one row per trial) and `results/<run_id>.jsonl` (full context-window log for every LLM call in that trial).

> **Reproducibility note:** All 200 trials (Phases 1–4) reproducible at git tag `phase-4-final`. Phases 1–3 synthetic-only results also available at `phase-3-final`.

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
| Phase 4 | ✅ Complete | ra1\_n3 (N=3), ra2\_n5 (N=5) | 40 | Does advantage hold with real LLM agents? |
| Concurrency | ✅ Complete (supplementary, not in paper) | cc1\_n3, cc2\_n5 | 44 | Stress test: simultaneous HIGH-urgency + user injections |

---

## Paper

The submission paper is `paper/draft_v3.tex` (Phases 1–4, 200 trials, dual-judge validation for Phase 4). Closest related work: AFM ([2511.12712](https://arxiv.org/abs/2511.12712)) and AOI ([2512.13956](https://arxiv.org/abs/2512.13956)) — see `notes/literature-review.md` for precise distinctions.

---

## Supplementary: Concurrency & Interruption Experiment

Not included in the paper submission. 44 trials across two scenarios (cc1\_n3, cc2\_n5) stress-testing DACS under simultaneous HIGH-urgency requests, INTERRUPT preemption events, and mid-trial user message injections. DACS accuracy degrades ≤2 pp under stressors; baseline degrades 8–9 pp. Full analysis in `results_concurrency/CONCURRENCY_RESULTS.md`.

**Structural differences from Phases 1–4:** uses an `InlineJudge` that scores responses live *during* the trial (not post-hoc), a `UserInjector` that fires timed user messages mid-FOCUS session, and a `TrackedQueue` that forwards questions to the judge in real time.

```bash
python -m experiments_concurrency.run --scenario cc1_n3 cc2_n5 --mode both --trials 5
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
