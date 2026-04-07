# DACS — Project Guidelines

## What This Project Is

Research prototype and paper for **Dynamic Attentional Context Scoping (DACS)** — a mechanism where a multi-agent orchestrator enters agent-triggered focus sessions to isolate per-agent context during steering interactions. Target: arxiv preprint + workshop paper (NeurIPS/ICLR/AAMAS).

**Current status:** All experimental phases complete. 204 trials run across 4 experiment phases (Phases 1–3 + Concurrency). Full paper written at `paper/draft_v2.tex`. Ready for arxiv submission and final polish.

Full plan: [ACTION_PLAN.md](../ACTION_PLAN.md)

## Stack

- **Language:** Python (3.11+)
- **LLM calls:** Direct API via Anthropic-compatible endpoint — `AsyncAnthropic` client pointed at `https://api.minimax.io/anthropic`
- **Model used for all experiments:** MiniMax-M2.7 (set via `DACS_MODEL` env var; default in harness)
- **No external orchestration frameworks** (LangGraph, CrewAI, etc.) — the harness is intentionally minimal to keep full observability over context window contents
- **Token counting:** `tiktoken` cl100k_base (deterministic, matches GPT-4o-mini encoding)
- **Paper:** LaTeX (NeurIPS 2024 style) in `paper/` — active file is `draft_v2.tex`
- **Dependencies:** `anthropic`, `tiktoken`, `python-dotenv`, `numpy`, `matplotlib`, `rich`

## Repository Structure

```
src/
  orchestrator.py     DACS orchestrator state machine (REGISTRY / FOCUS / USER_INTERACT)
  registry.py         RegistryManager — per-agent state store (≤200 tokens each entry)
  context_builder.py  Token-counted context assembly + hard cap (central experiment variable)
  protocols.py        SteeringRequest / SteeringResponse / SteeringRequestQueue / enums
  logger.py           JSONL logger — every event written with timestamp; supports sinks
  monitor.py          Rich terminal monitor — live colour-coded event feed via Logger sink

agents/
  base_agent.py           Abstract base — asyncio task, heartbeat + SteeringRequest emission
  generic_agent.py        Configurable step-driven agent (used in Phase 1+ scenarios)
  code_writer_agent.py    Specialist: incremental code writing, design decision steering
  research_agent.py       Specialist: search + summarise, query clarification steering
  data_processor_agent.py Specialist: data transforms, format ambiguity steering
  debugger_agent.py       Specialist: debugging tasks
  long_writer_agent.py    Specialist: long-form writing tasks

experiments/
  run_experiment.py        Single CLI entry point — all Phase 1–3 scenarios
  task_suite.py            8 scenarios (s1–s8), known-correct answers per decision point
  metrics.py               M1–M6 metric computation from JSONL logs
  llm_judge_phase3.py      LLM-as-judge validation for Phase 3 (s7, s8)
  llm_judge_s8.py          Focused judge pass for s8_n3_dense_d3
  plot_phase1.py           Phase 1 figures
  plot_phase2_phase3.py    Phase 2 & 3 figures

experiments_concurrency/
  run.py             CLI entry point — concurrency & interruption experiment
  harness.py         Trial runner: wires logger → registry → TrackedQueue → orchestrator → UserInjector → InlineJudge
  scenario_defs.py   cc1_n3, cc2_n5 scenario definitions with user injection schedules
  rubric_judge.py    Inline async LLM rubric judge (1–10 holistic scoring, fires during trial)
  event_injector.py  UserInjector — fires timed user messages into running trials
  analyze.py         Post-hoc analysis of concurrency_summary.csv
  _smoke_test.py     Smoke test for concurrency harness

results/             Auto-generated — do not edit manually
  PHASE1_RESULTS.md  Phase 1 analysis
  PHASE2_RESULTS.md  Phase 2 analysis
  PHASE3_RESULTS.md  Phase 3 analysis
  *.jsonl            Per-trial full context-window logs (raw ground truth)

results_concurrency/ Auto-generated — do not edit manually
  CONCURRENCY_RESULTS.md  Concurrency & interruption analysis
  concurrency_summary.csv Aggregated metrics
  *.jsonl                 Per-trial logs

notes/
  literature-review.md   Per-paper notes: 9 papers reviewed (AFM, AOI, AgentOrchestra, ACE, AdaptOrch, CodeDelegator, SideQuest, Adaptive Orchestration, Lemon Agent)
  formal-definition.md   Formal DACS definition (entities, states, transitions)

docs/
  architecture.md        Full Mermaid component graph + state machine diagrams
  interfaces.md          Component interface specs (Python signatures, data schemas, invariants)
  EXPERIMENT_PLAN.md     Comprehensive experiment design — RQ1–RQ7, M1–M8, all phases
  PHASE_2_PLAN.md        Architecture design spec (now implemented)

paper/
  draft_v2.tex    Full paper covering all 3 phases + concurrency (active version)
  main.tex        Phase 1-only version (superseded by draft_v2.tex)
  refs.bib        Bibliography
  figures/        Paper figures (PNG)

logs/             Internal experiment logging utilities
  analyze_phase3.py
  check_progress.py
  s7_*.log / s8_*.log   Raw stdout logs from nohup background runs
```

## Key Domain Terminology

Always use these names consistently:

| Term | Meaning |
|---|---|
| `Orchestrator` | The single LLM managing all agents and user interaction |
| `Registry` / `R` | Lightweight per-agent state snapshots (≤200 tokens each) |
| `Focus context` / `F(aᵢ)` | Full context of agent `aᵢ` injected during a focus session |
| `SteeringRequest` | Structured message an agent emits when it needs orchestrator input |
| `SteeringResponse` | Orchestrator's reply, includes `context_size_at_time` (critical metric) |
| `RegistryEntry` | Per-agent snapshot: agent_id, task, status, last_output_summary, urgency |
| `REGISTRY mode` | Orchestrator holds registry summaries only |
| `FOCUS(aᵢ) mode` | Orchestrator holds `F(aᵢ)` + compressed registry, steers `aᵢ` |
| `USER_INTERACT mode` | Orchestrator answers user with registry context; same token footprint as REGISTRY |
| `context pollution` | The problem DACS solves — all agent threads competing in one context window |
| `INTERRUPT` | HIGH-urgency SteeringRequest from `aⱼ` that preempts an active `FOCUS(aᵢ)` session |
| `InlineJudge` | Async LLM rubric judge that fires 1–10 holistic scores during concurrency trials |
| `TrackedQueue` | SteeringRequestQueue subclass used in concurrency harness to forward questions to InlineJudge |

## Core Constraints

- **Observability is paramount.** Every LLM call logs the exact token contents on a `CONTEXT_BUILT` event. This is the central experiment variable. Never abstract away what goes into a prompt.
- **Token budget must be enforced deterministically.** `context_builder.py` counts tokens via tiktoken before every call and hard-caps at `T` (default 204800). Do not rely on the LLM provider to truncate.
- **Baseline must be a fair comparison.** The flat-context baseline orchestrator uses the identical code path as DACS except `focus_mode=False`. That single flag switches `build_focus_context()` to `build_flat_context()`. No other differences.
- **Experiment results are ground truth.** Do not modify files under `results/` or `results_concurrency/` — they are generated by the experiment entry points.
- **Agents run as asyncio tasks** sharing a single event loop with the orchestrator. Thread-safety in `RegistryManager` comes from the single-threaded asyncio model, not explicit locks.
- **`GenericAgent` is the primary agent type for experiments.** Specialist agents (CodeWriter, Research, etc.) exist but most Phase 1–3 scenarios use `GenericAgent` with step-list configurations.

## Experiment Design (completed)

### Phases 1–3 (experiments/ harness, 160 trials)

| Phase | Scenarios | Variable tested | Trials |
|---|---|---|---|
| 1 | s1_n3, s2_n5, s3_n10 | Agent count scaling (N ∈ {3,5,10}) | 60 |
| 2 | s4_homogeneous, s5_crossfire, s6_cascade | Agent diversity + adversarial dependencies | 60 |
| 3 | s7_n5_dense_d2, s8_n3_dense_d3 | Decision density (D=8, D=15) | 40 |

**Key results:** DACS accuracy 90.0–98.4% vs baseline 21.0–60.0% across all 8 scenarios. All p < 0.0001 (Welch's t). Context ratio 2.12×–3.53×. DACS contamination <4% in all but s5/s6.

### Concurrency & Interruption (experiments_concurrency/ harness, 44 trials)

| Scenario | N | Conditions |
|---|---|---|
| cc1_n3 | 3 | dacs_clean, dacs_concurrent, baseline_clean, baseline_concurrent |
| cc2_n5 | 5 | dacs_clean, dacs_concurrent, baseline_clean, baseline_concurrent |

**Stressors:** Competing simultaneous HIGH-urgency requests (INTERRUPT events) + timed user message injections mid-trial.
**Key results:** DACS accuracy under concurrency drops ≤2.1 pp from clean baseline. Baseline drops 8–9 pp. InlineJudge (1–10 rubric) validates quality inline.

### Key metrics (M1–M7)

- **M1 Steering accuracy** — primary: keyword hit rate per decision point, ground-truth from `task_suite.py`
- **M2 Contamination rate** — fraction of steering responses mentioning other agents' identifiers
- **M3 Context size at steering time** — raw tokens from `CONTEXT_BUILT` log event
- **M4 User latency** — ms between USER_REQUEST and USER_RESPONSE (concurrency experiment)
- **M5 Time-to-steer** — ms from STEERING_REQUEST to STEERING_RESPONSE
- **M6 Registry truncation rate** — REGISTRY_TRUNCATION events / REGISTRY_UPDATE events; target <5%
- **M7 Interrupt rate** — INTERRUPT events / FOCUS sessions (DACS-only)

## Log Event Reference

| Event | Source | Key fields |
|---|---|---|
| `RUN_START` | Orchestrator | run_id, condition, scenario, n_agents, focus_mode, model |
| `RUN_END` | Orchestrator | run_id |
| `REGISTRY_UPDATE` | RegistryManager.update() | agent_id, status, urgency |
| `REGISTRY_TRUNCATION` | RegistryManager | agent_id, original_tokens, truncated_tokens |
| `CONTEXT_BUILT` | ContextBuilder | mode (REGISTRY/FOCUS/FLAT), agent_id, token_count |
| `STEERING_REQUEST` | SteeringRequestQueue.enqueue() | agent_id, request_id, urgency, blocking |
| `STEERING_RESPONSE` | Orchestrator | request_id, agent_id, context_size_at_time, orchestrator_state |
| `LLM_CALL` | Orchestrator | in_tokens, out_tokens, latency_ms, state |
| `TRANSITION` | Orchestrator | from_state, to_state, reason |
| `INTERRUPT` | Orchestrator | preempted_agent_id, new_agent_id |
| `FOCUS_TIMEOUT` | Orchestrator | agent_id |
| `JUDGE_SCORE` | InlineJudge (concurrency only) | request_id, agent_id, event_type_judged, score (1–10), reason |
| `INJECTION` | UserInjector (concurrency only) | message, delay_s |

## Paper Framing

- DACS is **not** a context compression algorithm (that's AFM/AOI). It is an **active, asymmetric, agent-triggered context isolation mechanism**.
- Prior work reviewed (9 papers): AFM (2511.12712), AOI (2512.13956), AgentOrchestra (2506.12508), ACE (2510.04618), AdaptOrch (2602.16873), CodeDelegator (2601.14914), SideQuest (2602.22603), Adaptive Orchestration (2601.09742), Lemon Agent (2602.07092).
- **No prior work implements agent-triggered asymmetric REGISTRY/FOCUS mode switching for per-agent context isolation in concurrent multi-agent orchestration** (confirmed by 20 targeted arxiv searches; see `notes/literature-review.md`).
- The closest prior work is AFM (arxiv 2511.12712) and AOI (arxiv 2512.13956) — always be precise about the distinction.
- Active paper file: `paper/draft_v2.tex` (covers all 3 phases + concurrency). `paper/main.tex` is the Phase 1-only superseded version.
- OpenClaw is used as motivation/context in the paper introduction, not as implementation platform.
