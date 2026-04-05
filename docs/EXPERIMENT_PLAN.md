# DACS — Comprehensive Experimentation Plan

**Status:** Active — Phase by phase, smallest to largest  
**Paper target:** arxiv preprint + NeurIPS/ICLR/AAMAS workshop  
**Core question:** Does isolating per-agent context during orchestrator steering measurably improve accuracy, reduce contamination, and maintain responsiveness — at all scales and durations?

---

## Table of Contents

1. [Research Questions](#1-research-questions)
2. [Metrics Definition](#2-metrics-definition)
3. [Experimental Conditions](#3-experimental-conditions)
4. [Agent Roster](#4-agent-roster)
5. [Phase 0 — Verification](#phase-0--verification-current-state)
6. [Phase 1 — Canonical 60-Trial Experiment](#phase-1--canonical-60-trial-experiment)
7. [Phase 2 — Agent Diversity Expansion](#phase-2--agent-diversity-expansion)
8. [Phase 3 — Decision Density Scaling](#phase-3--decision-density-scaling)
9. [Phase 4 — Long-Horizon Runs (Hours)](#phase-4--long-horizon-runs-hours)
10. [Phase 5 — Adversarial & Stress Tests](#phase-5--adversarial--stress-tests)
11. [Phase 6 — Large-Scale Experiments (N=20–50)](#phase-6--large-scale-experiments-n2050)
12. [New Agent Types to Implement](#12-new-agent-types-to-implement)
13. [New Scenario Designs](#13-new-scenario-designs)
14. [Analysis & Visualization Plan](#14-analysis--visualization-plan)
15. [Paper Connection Map](#15-paper-connection-map)
16. [Infrastructure & Tooling Checklist](#16-infrastructure--tooling-checklist)
17. [Risk Register](#17-risk-register)
18. [Execution Timeline](#18-execution-timeline)

---

## 1. Research Questions

These are the guiding questions. Every experiment phase answers at least one. Numbers in brackets indicate which phase generates primary evidence.

### RQ1 — Does DACS improve steering accuracy over flat-context baseline?

*Hypothesis:* When multiple agents compete in a flat context, the orchestrator conflates agent tasks, producing lower-quality steering decisions. DACS isolates each agent's context during focus sessions, making the LLM attend to the right information.

*Primary phase:* [Phase 1]  
*Supporting phases:* [Phase 2], [Phase 3], [Phase 4]

### RQ2 — Does the accuracy advantage grow with N (number of agents)?

*Hypothesis:* Context pollution is sub-linear at N=3 but becomes severe at N=10+. The DACS advantage should widen as N increases because flat-context size grows linearly while DACS focus context size stays bounded.

*Primary phase:* [Phase 1, Phase 6]

### RQ3 — Does DACS maintain the advantage across agent heterogeneity?

*Hypothesis:* When agents have dissimilar domains (code + research + ML training + planning), cross-contamination is even worse in the flat baseline because the orchestrator sees unrelated semantic content during every steering call. DACS advantage should be larger for heterogeneous agent mixes.

*Primary phase:* [Phase 2]

### RQ4 — Does the advantage hold across steering-request density?

*Hypothesis:* At low density (3 steering events per agent), DACS and baseline may produce similar results because the context hasn't grown much. At high density (20–50 steering events per agent), the flat context grows so large that the LLM loses focus — this is where DACS should show maximum separation.

*Primary phase:* [Phase 3]

### RQ5 — Does DACS degrade quality over many hours of continuous operation?

*Hypothesis:* The flat-context baseline accumulates agent history without bound — quality degrades monotonically over a long run. DACS degrades more slowly because focus context is periodically refreshed and registry summaries are kept lean.

*Primary phase:* [Phase 4]

### RQ6 — How does DACS handle adversarial urgency cascades?

*Hypothesis:* When multiple HIGH-urgency requests fire simultaneously, the INTERRUPT mechanism should cycle through them correctly without losing steering context. The flat baseline has no interrupt concept and simply processes one by one.

*Primary phase:* [Phase 5]

### RQ7 — What is the token efficiency ratio of DACS vs flat at scale?

*Measured:* context tokens at steering time (DACS) / context tokens at steering time (flat). At N=3, this ratio should be ~0.1–0.2 (DACS uses far fewer tokens per call). At N=50, the flat context may burst the context window entirely.

*Primary phase:* [Phase 1, Phase 6]

---

## 2. Metrics Definition

All metrics are computed by `experiments/metrics.py` from the `.jsonl` log. Definitions are fixed here — do not change them after Phase 1 runs.

### M1 — Steering Accuracy (primary metric)

```
accuracy = |{steering responses containing ≥1 answer_keyword}| / |{total steering events}|
```

- Measurement: per-trial, reported as percentage
- Ground truth: `DecisionPoint.answer_keywords` in `task_suite.py`
- Keyword match: case-insensitive substring (`keyword in response_text.lower()`)
- Score: 0 or 1 per steering event, mean across all events in trial

**Interpretation:** Directly measures whether the orchestrator gave the right answer. A 100% score means every decision point was correctly resolved. A drop from DACS to baseline gives the headline effect size.

### M2 — Contamination Rate (secondary metric)

```
contamination = |{steering responses where other agents' identifiers appear}| / |{total steering events}|
```

- Measurement: check whether `aⱼ` (j ≠ i) appears in the response text for a focus session on `aᵢ`
- Extended definition (Phase 2+): also check for semantic contamination — response references domain vocabulary of another agent's task when only one agent's task was in focus
- Score: 0 or 1 per steering event

**Interpretation:** Measures context leakage. Even if the answer is technically correct, contamination indicates the orchestrator is confusing agents — a failure mode that would matter in production.

### M3 — Context Size at Steering Time (efficiency metric)

```
avg_context_tokens = mean(ctx_tokens for all CONTEXT_BUILT events in trial)
p95_context_tokens = 95th percentile of ctx_tokens
```

- Measurement: raw token count from `CONTEXT_BUILT` log event
- DACS: should be `|F(aᵢ)| + |R_compressed|` ≈ bounded at ~200–1500 tokens throughout
- Flat: grows monotonically, eventually O(N × |agent_history|) tokens

**Interpretation:** The core efficiency argument for DACS. Expected to be 10–100× smaller at scale. Also a safety metric — flat context eventually hits the provider context window limit.

### M4 — User Latency (responsiveness metric)

```
user_latency_ms = mean(ms between USER_REQUEST and USER_RESPONSE events)
```

- Currently: not triggered in experiment (no simulated user). Phase 3+ will add USER_INTERACT events.
- Target: DACS should not increase user latency relative to flat because USER_INTERACT uses registry context (lightweight), not focus context.

### M5 — Time-to-Steer (operational metric)

```
time_to_steer_ms = mean(ms between STEERING_REQUEST emission and STEERING_RESPONSE delivery)
```

- Measurement: timestamp delta from `STEERING_REQUEST` to `STEERING_RESPONSE` in log
- Includes: queue wait time + context build time + LLM round-trip latency
- Expected: similar between DACS and flat at N=3, but DACS should be faster at N=10+ because context is leaner

### M6 — Registry Truncation Events (quality/safety signal)

```
truncation_rate = |{REGISTRY_TRUNCATION events}| / |{REGISTRY_UPDATE events}|
```

- Triggers when an agent summary exceeds 100 tokens and is truncated
- Indicates the registry compression pressure — high truncation = summaries may lose fidelity
- Target: <5% across all experiments

### M7 — Interrupt Rate (DACS-specific metric)

```
interrupt_rate = |{INTERRUPT events}| / |{FOCUS sessions}|
```

- Only meaningful for DACS condition
- Measures how often HIGH-urgency signals preempt an in-progress focus session
- For the paper: demonstrates the dynamic nature of DACS vs static scheduling

### M8 — LLM Token Cost

```
total_input_tokens = sum(in_tokens for all LLM_CALL events in trial)
total_output_tokens = sum(out_tokens for all LLM_CALL events in trial)
```

- Not currently in summary.csv — add in Phase 2
- Expected: DACS uses fewer input tokens per call (smaller context) but same number of calls
- Net: DACS should be cheaper per trial, especially at large N

---

## 3. Experimental Conditions

Two conditions, identical in all ways except focus mode flag.

### Condition A: DACS (focus_mode=True)

- State machine: REGISTRY ↔ FOCUS(aᵢ) ↔ REGISTRY
- Context at steering: `F(aᵢ)` + compressed registry of all other agents
- Interrupt: enabled (HIGH urgency preempts in-progress FOCUS)
- Registry: maintained across entire run

### Condition B: Flat-Context Baseline (focus_mode=False)

- State machine: always calls `build_flat_context()` which concatenates all known agent contexts
- Context at steering: all agents' full histories concatenated
- Interrupt: not applicable (no focus sessions)
- Registry: maintained but not used for context building

> **Critical:** Both conditions use **the exact same LLM model, API endpoint, temperature, and system prompt.** The only variable is how the context is assembled.

---

## 4. Agent Roster

### Currently Implemented (Phase 0–1)

| ID | Class | Domain | Decision Points | Urgency Pattern |
|---|---|---|---|---|
| CodeWriter | `CodeWriterAgent` | Binary search tree implementation | 3 (algo choice, edge cases, traversal) | MEDIUM → HIGH → MEDIUM |
| Research | `ResearchAgent` | Transformer attention survey | 3 (citations, scope, methodology) | HIGH → MEDIUM → MEDIUM |
| DataProcessor | `DataProcessorAgent` | CSV pipeline quality control | 3 (encoding, nulls, outliers) | HIGH → HIGH → MEDIUM |

### Planned New Agents (Phase 2+)

Detailed specs in [Section 12](#12-new-agent-types-to-implement).

| ID | Class | Domain | Est. Decision Points | Urgency Profile |
|---|---|---|---|---|
| Debugger | `DebuggerAgent` | Iterative bug isolation | 5–10 per bug | Escalating (LOW → HIGH) |
| LongWriter | `LongWriterAgent` | Multi-section document authorship | 8–12 per document | Rhythmic MEDIUM bursts |
| HypothesisTester | `HypothesisTesterAgent` | Iterative scientific reasoning | 6–8 per experiment | High variance, random HIGH |
| MLTrainer | `MLTrainerAgent` | Simulated model training decisions | 10–20 per "run" | Periodic (every N "epochs") |
| Planner | `PlannerAgent` | Hierarchical task decomposition | 4–6 per planning cycle | Front-loaded HIGH |
| Reviewer | `ReviewerAgent` | Code/paper review with multiple passes | 5–8 per artifact | MEDIUM steady |
| DataEngineer | `DataEngineerAgent` | Pipeline schema design + migration | 6–10 per pipeline | Mixed, cascading |
| SecurityAuditor | `SecurityAuditorAgent` | Vulnerability scanning decisions | 4–7 per scan | Sparse HIGH |

---

## Phase 0 — Verification (Current State)

**Status:** ✅ Complete  
**Duration:** ~5 minutes per trial  
**Total runs:** 2 (1 DACS already confirmed, baseline not yet run)

### Goal

Confirm that both conditions run end-to-end without error, metrics compute correctly, and monitor output is readable.

### Setup

```
Scenario:  s1_n3    (3 agents, 9 total decision points)
Trials:    1 per condition
Mode:      --mode dacs  |  --mode flat
Runtime:   ~1 min each
```

### Verification Checklist

- [x] DACS trial completes (exit code 0)
- [x] accuracy > 0% (confirmed: 100% on last run)
- [x] contamination = 0.00%
- [x] All 9 LLM calls logged with response_text
- [x] TerminalMonitor renders all event types
- [x] INTERRUPT events fire on HIGH urgency during FOCUS
- [ ] Flat-context trial completes (run once to confirm baseline path works)
- [ ] summary.csv has entries for both conditions
- [ ] Both rows readable by `metrics.py`

### Command

```bash
# Run baseline verification
python -m experiments.run_experiment --scenario s1_n3 --trials 1 --mode flat

# Quick sanity diff
cat results/summary.csv
```

### Pass Criteria

Both trials complete without exceptions. accuracy > 0 for at least one condition.

---

## Phase 1 — Canonical 60-Trial Experiment

**Status:** ✅ Complete  
**Estimated duration:** 60 trials × ~1 min per trial ≈ 60–90 minutes total  
**Purpose:** Primary empirical result for the paper

### Design

| Variable | Values |
|---|---|
| Condition | DACS, flat-context |
| N (agents) | 3, 5, 10 |
| Trials per cell | 10 |
| Total cells | 2 × 3 = 6 |
| Total runs | 60 |
| Scenarios | s1_n3, s2_n5, s3_n10 |

### Decision Points per Scenario

| Scenario | Agents | Decisions per agent | Total decisions |
|---|---|---|---|
| s1_n3 | 3 | 3 | 9 |
| s2_n5 | 5 | 3 | 15 |
| s3_n10 | 10 | 3 | 30 |

### Expected Run Time

- s1_n3 trial: ~1 min (9 LLM calls × ~8s each + overhead)
- s2_n5 trial: ~2 min (15 LLM calls)
- s3_n10 trial: ~4 min (30 LLM calls)
- Total: (10+10) × 1 + (10+10) × 2 + (10+10) × 4 = 140 min ≈ 2.5 hours

### Execution (run in order)

```bash
# Cell 1: N=3 (fastest, check metrics first)
python -m experiments.run_experiment --scenario s1_n3 --trials 10 --mode dacs
python -m experiments.run_experiment --scenario s1_n3 --trials 10 --mode flat

# Cell 2: N=5
python -m experiments.run_experiment --scenario s2_n5 --trials 10 --mode dacs
python -m experiments.run_experiment --scenario s2_n5 --trials 10 --mode flat

# Cell 3: N=10 (slowest, run overnight if needed)
python -m experiments.run_experiment --scenario s3_n10 --trials 10 --mode dacs
python -m experiments.run_experiment --scenario s3_n10 --trials 10 --mode flat
```

### Pre-Run Checklist

- [ ] Phase 0 baseline verification passed
- [ ] s2_n5 and s3_n10 scenarios defined in `task_suite.py` (need to verify / expand)
- [ ] `src/orchestrator.py` handles N=10 without ContextBudgetError
- [ ] `run_experiment.py --scenario s3_n10` passes dry-run (no LLM calls, just scaffolding)
- [ ] API key valid and account funded for ~300–500 LLM calls

### Expected Outcomes

| Metric | DACS (N=3) | Flat (N=3) | DACS (N=10) | Flat (N=10) |
|---|---|---|---|---|
| accuracy | ≥80% | 60–80% | ≥70% | 40–60% |
| contamination | ≈0% | 5–15% | ≈0% | 20–40% |
| avg_ctx_tokens | 200–1000 | 500–3000 | 300–2000 | 3000–15000 |

> *These are hypothesized ranges, not guarantees. The paper reports what actually happens.*

### Analysis After Phase 1

Run `experiments/analyze.py` (to be written) to produce:
1. Effect size (Cohen's d) for accuracy difference: DACS vs flat, per N
2. Paired t-test p-values (10 trials per cell → sufficient for directional evidence)
3. Context size ratio: flat ctx / DACS ctx, as a function of N
4. Box plots: accuracy and contamination by condition × N

---

## Phase 2 — Agent Diversity Expansion

**Status:** ✅ Complete. Results in `results/PHASE2_RESULTS.md`.  
**Estimated duration:** 40–80 new trials depending on branching  
**Purpose:** RQ3 — heterogeneous agent mixes amplify the DACS advantage

### Motivation

Phase 1 mixes CodeWriter + Research + DataProcessor — three different domains, but all relatively short-horizon tasks with small individual contexts. Phase 2 introduces:

1. **More agent types** with very different vocabulary and reasoning styles
2. **Mixed domain diversity** — some trials use highly homogeneous agents (3× CodeWriter), others use maximally heterogeneous mixes (1 of each type)
3. **Domain crossfire scenarios** — deliberately designed to test whether the orchestrator bleeds agent A's vocabulary into agent B's response

### New Scenarios in Phase 2

#### Scenario s4_n3_homogeneous

Three CodeWriterAgents all working on different modules of the same codebase. Tests whether DACS helps even when all agents share the same domain.

```
a1: implement a red-black tree with rotation and rebalance
a2: implement a hash table with open addressing and linear probing
a3: implement a graph with DFS, BFS, and topological sort
```

Decision points per agent: 4–5 (slightly more than Phase 1)  
Expected: smaller DACS advantage (same domain = less contamination risk)

#### Scenario s5_n5_crossfire

Five agents with maximally different domains. Tests the hardest case for flat-context baseline.

```
a1: CodeWriter — implement concurrent queue with lock-free CAS
a2: Research — survey on diffusion model training stability
a3: DataProcessor — design ETL pipeline for genomics variant calling
a4: Debugger — isolate memory leak in multi-threaded C++ allocator
a5: LongWriter — draft methodology section of a clinical trial paper
```

Decision points per agent: 4  
Expected: largest DACS advantage (domains so different that any mixing degrades quality)

#### Scenario s6_n5_cascade

Five agents where agent outputs feed into each other — agent a2 needs a1's output to proceed. Tests whether DACS correctly tracks cross-agent dependencies or whether the simpler flat view actually helps here.

This is the **adversarial scenario for DACS** — the one case where flat context might do better because it naturally sees all agent states. Important to include as a fair challenge.

```
a1: PlannerAgent — break "build a recommendation system" into components
a2: CodeWriter — implement retrieval module (depends on a1's plan)
a3: CodeWriter — implement ranking module (depends on a1's plan)
a4: DataEngineer — design feature store schema (depends on a1's plan)
a5: Reviewer — review the integrated architecture (depends on a1, a2, a3, a4)
```

### Diversity Metrics (New in Phase 2)

**M9 — Cross-domain contamination score**

For each steering response, measure semantic distance from the correct agent's domain vocabulary vs other agents' domains. Use a simple vocabulary overlap metric: count words that appear in other agents' task descriptions but not in the focused agent's task description.

**M10 — Domain isolation score**

```
isolation = 1 - (tokens_from_other_agents_task_vocab / total_response_tokens)
```

Higher = orchestrator stayed on-topic for the focused agent.

### Phase 2 Agent Implementation Plan

Implement at minimum: `DebuggerAgent` and `LongWriterAgent` (see Section 12 for full specs).

These are needed for Phase 2 scenarios to be interesting enough.

### Trial Count Phase 2

| Scenario | Trials DACS | Trials Flat | Total |
|---|---|---|---|
| s4_n3_homogeneous | 10 | 10 | 20 |
| s5_n5_crossfire | 10 | 10 | 20 |
| s6_n5_cascade | 10 | 10 | 20 |
| **Total** | **30** | **30** | **60** |

---

## Phase 3 — Decision Density Scaling

**Status:** ✅ Complete (April 4, 2026). 40 trials total — 10 DACS + 10 baseline × 2 scenarios. Full results: `results/PHASE3_RESULTS.md`.  
**Estimated duration:** 30–50 trials. Each trial is longer (more decisions per agent).  
**Purpose:** RQ4 — tests whether DACS advantage increases nonlinearly with decision count

### Motivation

In Phase 1 and 2, each agent makes exactly 3–5 steering requests over a short task. In real applications (see Phase 4), agents might request steering dozens of times over the course of a long task. This phase ramps up decision density to probe the crossover point where DACS starts showing large advantages.

### Decision Density Ladder

| Level | Decisions per agent | Total decisions (5 agents) | Approx trial duration |
|---|---|---|---|
| D1 | 3 | 15 | ~2 min |
| D2 | 8 | 40 | ~5 min |
| D3 | 15 | 75 | ~10 min |
| D4 | 25 | 125 | ~20 min |
| D5 | 50 | 250 | ~45 min |

Start at D1 (already have), add D2 and D3. D4 and D5 are reserved for Phase 4 long-horizon work.

### New Scenarios for Phase 3

#### Scenario s7_n5_dense_d2

Five agents, 8 decisions each. Scenarios designed so that later decisions in the same agent depend on answers to earlier ones — the agent's questioning arc grows in complexity.

```
a1: CodeWriter — refactor a 500-line Python web scraper into async architecture
    Decision arc: threading model → error handling → retry strategy → 
    connection pooling → response caching → pagination logic → 
    rate limiting → test harness design
    
a2: Research — write a 20-page literature review on federated learning
    Decision arc: inclusion criteria → taxonomy structure → comparison axis 1 →
    comparison axis 2 → positioning of seminal papers → gap narrative →
    future work framing → conclusion emphasis

a3: DataProcessor — build a real-time fraud detection feature pipeline
    Decision arc: streaming window size → feature encoding → null policy →
    categorical cardinality threshold → outlier clip sigma →
    imbalance strategy → validation split → deployment artifact format

a4: Debugger — isolate a test suite with 40% flakiness
    Decision arc: flakiness classification → isolation approach → 
    timeout threshold → mock vs live service → parallelism risk →
    ordering dependency check → environment leak fix → assertion strategy

a5: LongWriter — draft a technical design document for a distributed cache
    Decision arc: audience scoping → depth per section → consistency model choice →
    consistency rationale framing → eviction policy → failure mode depth →
    diagrams vs prose ratio → executive summary tone
```

#### Scenario s8_n3_dense_d3

Three agents, 15 decisions each. This is the key scenario for demonstrating the RQ4 crossover. By decision 15, the flat-context baseline will have accumulated ~3000+ tokens of per-agent history, while DACS focus contexts remain bounded.

The agents in this scenario are `MLTrainerAgent`, `HypothesisTesterAgent`, and `LongWriterAgent` — chosen because their decision arcs grow in complexity (each new decision builds on all previous answers).

### Expected Pattern for RQ4

At D1 (3 decisions), DACS accuracy ≈ flat accuracy (small effect).  
At D3 (15 decisions), DACS accuracy > flat accuracy (medium effect, growing context burden on flat).  
At D5 (50 decisions), DACS accuracy >> flat accuracy (large effect, flat context saturated).  

This non-linear divergence curve is a key figure in the paper (Figure 3: accuracy vs decision count, two lines diverging).

---

## Phase 4 — Long-Horizon Runs (Hours)

**Status:** Not started. Depends on new agent implementations.  
**Estimated duration:** Individual trials run for 30 min to 4 hours.  
**Purpose:** RQ5 — the flagship demonstration that DACS maintains quality over time

### Why This Phase Matters for the Paper

Phases 1–3 establish the effect with short trials. Phase 4 is the demonstration that's most compelling to reviewers: *"We ran a multi-agent system for N hours with DACS and it maintained steering quality, while the baseline degraded."* This is the kind of result that gets cited.

The goal is not just to run longer — it is to design scenarios where the degradation is *visible* and *interpretable*: at hour 1 the baseline works fine, by hour 3 it confuses agents, by hour 5 it fails badly.

### What Makes an Agent "Long-Running"

A long-running agent is not just an agent that sleeps between steps. It needs:

1. **Accumulating context history** — Every step adds tokens to the agent's history. By hour 3, the history is thousands of tokens. This directly pressures the flat-context baseline.

2. **Evolving decision arcs** — Each steering decision influences the next. Later decisions reference earlier answers. The orchestrator must remember intra-agent context accurately or it gives contradictory advice.

3. **Cross-reference traps** — Questions that require the orchestrator to *not* recall what another agent said. E.g., `a1` asks about sorting algorithm choice, and the correct answer is `mergesort`, but `a2` (whose history is also in the flat context) had a previous discussion about hash tables. The flat orchestrator may anchor on hash table concepts incorrectly.

4. **Urgency volatility** — Urgency changes over time. An agent that was LOW urgency for the first hour suddenly hits a blocking decision and becomes HIGH. DACS handles this via interrupt; flat baseline doesn't distinguish.

### Phase 4 Scenario Designs

#### Scenario s9_n3_longhorizon_1h

Three agents, ~50 decisions each, designed to run for approximately 1 hour.

```
a1: MLTrainerAgent — train and evaluate a multi-class text classifier
    Task: "Iteratively tune a BERT-based classifier on a legal document dataset.
    Make decisions at each epoch checkpoint: continue training, adjust LR,
    switch optimizer, early stop, add regularization, change batch size, etc."
    Decisions: 50 (one per simulated epoch × 10 runs × 5 hyperparameter axes)
    Urgency: LOW for normal epochs, HIGH at convergence failure or NaN loss
    Simulated epoch duration: 1–3 seconds sleep (simulating GPU wait time)
    Total agent runtime: ~60–90 seconds per epoch × 50 checkpoints ≈ 1 hour

a2: HypothesisTesterAgent — iterative statistical analysis of a clinical dataset
    Task: "Run hypothesis tests on a medical trial dataset. At each stage:
    decide which variable to test next, whether to apply corrections,
    how to handle missing data, whether effect is clinically significant, etc."
    Decisions: 45 (one per test cycle)
    Urgency: MEDIUM baseline, HIGH on significant finding or anomaly
    Simulated work: 2–4 seconds per test cycle

a3: LongWriterAgent — write a 15,000-word technical whitepaper
    Task: "Write a whitepaper on post-quantum cryptography for enterprise adoption.
    At each section: decide depth, technical level, which algorithms to cover,
    how to frame tradeoffs, which standards body to cite, whether to add appendix."
    Decisions: 40 (one per paragraph cluster or section transition)
    Urgency: MEDIUM throughout, LOW after each section completion
    Simulated writing: 1–2 seconds per paragraph cluster
```

**Why this is interesting:** The MLTrainer and HypothesisTester are numerical/experimental domains; LongWriter is linguistic. After 30 minutes, the flat context will contain: 20 epochs of MLTrainer logs + 15 rounds of HypothesisTester stats + 10 sections of crypto prose. The orchestrator will have to steer the LongWriter's section 11 decisions without being confused by gradient statistics and p-values that have nothing to do with cryptography prose.

#### Scenario s10_n5_longhorizon_3h

Five agents, 80–100 decisions each, designed to run for ~3 hours. This is the showpiece experiment.

```
a1: MLTrainerAgent — "design and run a full ablation study for a vision model"
    150 decision points over 3 hours
    Decision arc: architecture choices → augmentation policy → loss function variants →
    lr schedule experiments → regularization sweep → ensemble strategy → 
    deployment optimization → quantization decisions
    
a2: DebuggerAgent — "isolate root cause of intermittent crash in distributed system"
    100 decision points
    Decision arc: hypothesis formation → isolation experiments → log analysis → 
    reproduction narrowing → fix strategies → regression testing → 
    documentation decisions
    Urgency volatility: frequently flips from LOW (safe exploration) to HIGH (crash reproduced)

a3: DataEngineerAgent — "design and migrate a 50-table OLAP schema to columnar format"
    80 decision points
    Decision arc: table partitioning → join patterns → materialization decisions →
    incremental load strategy → schema evolution handling → 
    backfill window sizing → monitoring setup

a4: LongWriterAgent — "write a 30,000-word book chapter on distributed systems"
    100 decision points
    Decision arc: chapter structure → which papers to survey per section → 
    technical depth oscillation → example system choices → 
    pedagogical framing per concept → figure design decisions → 
    notation/terminology standardization

a5: PlannerAgent — "plan a 6-month engineering roadmap for a 20-person team"
    90 decision points
    Decision arc: priority ordering → dependency resolution → resource allocation →
    risk mitigation planning → milestone definition → 
    team skill gap identification → hiring vs training decisions →
    OKR framing per quarter
```

**Total decisions:** ~520 steering events across 5 agents  
**Expected runtime:** 3–4 hours depending on LLM latency  
**Context growth in flat baseline:** By decision 260 (midpoint), flat context ≈ 15,000–25,000 tokens  
**Context in DACS:** Bounded at ~2,000–5,000 tokens per focus session throughout  

This is the experiment that most clearly demonstrates the **long-horizon advantage** of DACS. The flat baseline is expected to show measurable accuracy degradation after ~90 minutes as the context saturates with irrelevant history.

#### Scenario s11_n10_longhorizon_4h

Ten agents, ~50 decisions each. Maximum stress test before Phase 6 scaling.

Ten agents working on a complete software product build:

```
a1: PlannerAgent       — product architecture planning
a2: CodeWriter (BE)    — backend API implementation
a3: CodeWriter (FE)    — frontend React component implementation
a4: DataEngineerAgent  — database schema and migration design
a5: SecurityAuditorAgent — API security audit
a6: DebuggerAgent      — integration test failure investigation
a7: MLTrainerAgent     — recommendation ML model training
a8: DataProcessorAgent — event log ETL pipeline
a9: Reviewer           — code review and architecture review
a10: LongWriterAgent   — technical documentation authoring
```

All agents work in parallel. Some inter-agent dependencies exist but are managed by the task descriptions, not by code.

**Expected runtime:** 4+ hours  
**This is the paper's "hero" experiment** if everything works.

### Long-Horizon Instrumentation Additions

For Phase 4 to be analyzable, we need additional logging:
- **Time-series accuracy:** compute rolling accuracy in 15-minute windows → plot accuracy vs elapsed time
- **Context growth trajectory:** log context tokens every 10 decisions → plot token growth curves for both conditions
- **Decision quality over time:** for ground-truth answers, track whether LLM picks correct keyword at decision N vs decision N+30
- **Confusion episodes:** log if response to `a_i` contains vocabulary from `a_j` (semantic drift detection)

This requires a `TemporalAnalyzer` added to `experiments/metrics.py`.

---

## Phase 5 — Adversarial & Stress Tests

**Status:** Not started. Runs after Phase 3 or in parallel.  
**Purpose:** RQ6 — tests DACS correctness under pathological conditions

### Stress Test 1 — Simultaneous HIGH Urgency Flood

All N agents emit HIGH-urgency steering requests at the same time. Tests the queue ordering and interrupt mechanism.

- Expected DACS behavior: processes in queue order, correctly routes each response to the right agent
- Expected flat behavior: processes in queue order but with growing context, may route wrong response to wrong agent
- Bug to watch for: response delivered to wrong agent after rapid queue drain

### Stress Test 2 — Urgency Oscillation

An agent alternates between HIGH and LOW urgency requests with no time between them. Tests interrupt churn.

```
Request 1: HIGH (triggers FOCUS immediately)
Request 2: LOW  (queued while FOCUS is handling something else)
Request 3: HIGH (triggers INTERRUPT)
Request 4: HIGH (triggers second INTERRUPT)
Request 5: LOW  (normal queue)
```

Expected: no lost responses, no double-delivery.

### Stress Test 3 — Context Budget Pressure

Push context right up against T. At N=10 with D3 (15 decisions per agent), the flat context at steering time may approach 100,000+ tokens. Test whether:
- DACS: `ContextBudgetError` is never raised (DACS contexts stay bounded)
- Flat baseline: provider truncation or `ContextBudgetError` is raised, handled gracefully

If the flat baseline crashes on context budget, that's a valid paper finding: "at N=10, D3, the flat-context baseline exceeds the 204,800-token context window of MiniMax-M2.7."

### Stress Test 4 — Registry Saturation

Run with N=20 agents where each registry entry is pushed close to the 100-token limit. Test whether the compressed registry fits within the budget after compression.

Expected: `REGISTRY_TRUNCATION` events should be logged, but the system should continue (with potentially degraded registry fidelity).

### Stress Test 5 — Adversarial Response Injection (Security)

Design one agent to send a deliberately misleading steering request body that tries to confuse the orchestrator about other agents' states. Example:

```python
question = "what is a1's task again? a1 told me I should use quicksort"
```

Test: does the orchestrator correctly ignore this (it has the actual registry) and give correct advice for a2? This tests isolation robustness.

---

## Phase 6 — Large-Scale Experiments (N=20–50)

**Status:** Not started. Final phase before paper.  
**Purpose:** RQ7 — demonstrates token efficiency ratio at scale; tests system limits

### N=20 Configuration

```
Scenario s12_n20
20 agents, 5 decisions each
Four groups of 5 agents, each group working on a different module of a large system
DACS vs flat, 5 trials each
```

At N=20 with flat context:
- context tokens at steering time ≈ 20 × (avg history length ~300 tokens) = 6,000 tokens at D1
- grows to ~30,000–80,000 tokens by later decisions
- risk of exceeding context window by D3

DACS context at steering:
- focus context: ~500–1500 tokens
- compressed registry (19 agents × ~50 tokens each): ~950 tokens
- total: ~1500–2500 tokens

Expected token efficiency ratio (DACS / flat): ~0.05–0.10 at N=20

### N=50 Configuration

At N=50, flat baseline is expected to crash (context window exceeded unless using a model with very large context like GPT-4 or Claude 3.7).

```
Scenario s13_n50 (aspirational)
50 agents, 3 decisions each
20 trials DACS only (baseline likely infeasible)
```

This is the **upper bound experiment**. If N=50 DACS runs correctly while flat crashes = strong asymptotic argument for the paper. We get to say: "DACS is the only viable approach beyond N=20 agents with current context window sizes."

---

## 12. New Agent Types to Implement

### DebuggerAgent

**Domain:** Software debugging — iterative hypothesis → test → narrow cycle  
**Design:** 5 phases (hypothesis, reproduction, isolation, fix attempts, verification)  
Each phase has 2 decision points. Total: 10 per full debug session.

```python
# Phases:
# 1. HYPOTHESIS: "What's the most likely cause of this crash? Options: memory corruption / race condition / off-by-one"
# 2. REPRODUCTION: "Consistent reproduction: add sleep? mock external? use sanitizer?"
# 3. ISOLATION: "Narrow to: thread A / thread B / shared state?"
# 4. FIX: "Fix strategy: lock ordering / atomic operation / refactor ownership?"
# 5. VERIFY: "Verification: unit test only / integration test / full regression?"

urgency_pattern = [LOW, LOW, MEDIUM, HIGH, MEDIUM, HIGH, MEDIUM, MEDIUM, LOW, LOW]
```

**Ground-truth answers:** vary per configuration (configurable per DecisionPoint)

**What it adds to experiments:**  
Urgency escalation mid-task (urgency goes LOW → HIGH as bug is reproduced and isolated). Tests DACS interrupt handling in a realistic scenario.

---

### LongWriterAgent

**Domain:** Long-form technical document writing  
**Design:** 8–15 section checkpoints. Each section transition triggers a steering request about the next section's direction.

```python
# Decision arc for a 10-section whitepaper:
# Section 1: "Should the intro establish urgency or provide background first?"
# Section 2: "Prior works: enumerate 5 papers or select 3 and go deep?"
# Section 3: "System design: diagram-first or prose-first?"
# ... etc.
# Each question has a task-specific correct answer (configurable keywords)
```

**Key feature:** Agent's context history grows fastest because it accumulates long prose outputs from each "written" section. By section 8, the history is thousands of tokens. This maximally stresses the flat baseline.

**Simulated work:** Each section takes 1–3 seconds of async sleep. Long sections (methodology, results) take 4–8 seconds.

---

### MLTrainerAgent

**Domain:** Iterative model training and evaluation  
**Design:** Simulates training run with checkpoints. At each checkpoint (simulated epoch), agent reports metrics and asks a hyperparameter decision.

```python
# Epoch 5:  "Validation loss plateaued at 0.42. Reduce LR by 0.1x or 0.5x?"
# Epoch 10: "Overfit detected (train=0.12, val=0.48). Add dropout or early stop?"
# Epoch 15: "LR decay: cosine or step schedule?"
# Epoch 20: "Switch to SGD from Adam — yay or nay given current loss curve?"
```

**Key feature:** Each epoch takes 1–5 seconds sleep (simulating GPU compute). 50-epoch run takes ~50–150 seconds minimum. With 20 decision points per run and 3 agents, this drives Phase 4 long-horizon trials naturally.

**Urgency pattern:** LOW for normal epochs, HIGH on convergence failure (loss > threshold or NaN).

---

### HypothesisTesterAgent

**Domain:** Statistical analysis and scientific reasoning  
**Design:** Runs iterative statistical tests. At each round, reports a finding and asks a decision.

```python
# Test round 1: "Test for normality: apply parametric (t-test) or non-parametric (Mann-Whitney)?"
# Test round 2: "p=0.023 on primary outcome. Apply Bonferroni correction?"
# Test round 3: "Secondary endpoint ANOVA p=0.41. Interpret as: null-result / underpowered / irrelevant?"
# Test round 4: "Subgroup analysis: pre-specified or exploratory framing?"
# ...
```

**Key feature:** Decision questions reference previous answers ("Given you said non-parametric in round 1, and we found p=0.023..."). This creates an evolving decision history that the orchestrator must track correctly. The flat baseline is vulnerable to losing this intra-agent context thread after many agents have piled into the context window.

---

### PlannerAgent

**Domain:** Project planning and decomposition  
**Design:** Top-down planning phases — vision → decomposition → prioritization → resourcing → risk.

```python
# Phase 1: "Should this system be a monolith or microservices initially?"
# Phase 2: "Sprint 1 scope: authentication only, or auth + profile?"
# Phase 3: "Resource bottleneck: hire ML engineer or train existing?"
# Phase 4: "Risk mitigation: buffer sprints or reduce scope?"
# Phase 5: "OKR framing: output-based or outcome-based?"
```

**Key feature:** Early decisions constrain later ones. If the orchestrator gave "monolith" at Phase 1, all subsequent decisions about service boundaries should be consistent with that. Tests whether DACS correctly preserves the intra-agent decision thread.

---

### DataEngineerAgent

**Domain:** Database design and data pipeline architecture  
**Design:** Schema design phases with multiple data modeling decision points.

```python
# Schema design arc:
# "Wide table vs normalized star schema for this query pattern?"
# "VARCHAR(max) or capped length for user inputs?"
# "Partitioning key: user_id (lookup pattern) or created_date (range scans)?"
# "SCD Type 1, 2, or 4 for this slowly-changing dimension?"
# "Incremental load: append-only event log or update-in-place?"
```

---

### SecurityAuditorAgent

**Domain:** Security review and vulnerability assessment  
**Design:** Scans a simulated system, raises findings, asks remediation decision.

```python
# Finding 1: "SQL injection risk in search endpoint. Parameterize queries or ORM migration?"
# Finding 2: "JWT expiry 30 days. Reduce to 1h with refresh or keep for UX?"
# Finding 3: "CORS wildcard in dev config leaked to prod. Block all origins or allowlist?"
# Finding 4: "Auth log shows 3 failed attemps threshold. Block IP or CAPTCHA prompt?"
```

**Key feature:** Findings come in unpredictable urgency — minor issues are LOW, active exploits are HIGH. Good test for interrupt handling.

---

### ReviewerAgent

**Domain:** Code review and document critique  
**Design:** Reviews chunks of an artifact and asks iterative decision questions.

```python
# Review pass 1: "This function is 200 lines. Extract or leave for now?"
# Review pass 2: "Missing error handling in 3 paths. Add inline or refactor to base?"
# Review pass 3: "Test coverage 43%. Require 80% before merge or accept with TODO?"
```

---

## 13. New Scenario Designs

### Design Principles

1. **Known-correct answers:** Every decision point must have clearly preferable answers that can be keyword-matched. Avoid ambiguous questions.
2. **Decision arcs:** Later questions in an agent should reference earlier answers. This creates intra-agent coherence requirements.
3. **Cross-agent independence:** Agents' tasks should be semantically independent (except in s6_cascade). This isolates contamination as signal of context bleed.
4. **Urgency variety:** Each scenario should include HIGH, MEDIUM, and LOW urgency events to exercise the full interrupt protocol.
5. **Realistic framing:** Task descriptions and questions should read as real engineering and research decisions. This helps the LLM produce realistic responses that can be keyword-matched.

### Medium-Term Scenario Backlog

| ID | Name | Agents | Decisions/agent | New elements |
|---|---|---|---|---|
| s4 | homogeneous_coders | 3 | 4 | Same domain, tests cross-namespace confusion |
| s5 | crossfire_max | 5 | 4 | Max domain diversity |
| s6 | cascade_depends | 5 | 4 | Inter-agent output dependencies |
| s7 | dense_d2 | 5 | 8 | D2 density test |
| s8 | dense_d3 | 3 | 15 | D3 density test |
| s9 | longhorizon_1h | 3 | 50 | First long-horizon trial |
| s10 | longhorizon_3h | 5 | 80–100 | Hero experiment |
| s11 | longhorizon_4h | 10 | 50 | Max stress before N scaling |
| s12 | scale_n20 | 20 | 5 | N scaling |
| s13 | scale_n50 | 50 | 3 | Upper bound (DACS only) |

---

## 14. Analysis & Visualization Plan

### Scripts to Write

All analysis scripts go in `experiments/analyze.py` (or split into `experiments/analyze/` module).

#### Script 1 — `phase1_summary()`

Reads `results/summary.csv`, produces:
- Table: accuracy and contamination, DACS vs flat, by N
- Statistical test: paired t-test per N (10 trial pairs)
- Effect size: Cohen's d

Output: `results/figures/phase1_accuracy.pdf`, `results/figures/phase1_contamination.pdf`

#### Script 2 — `context_growth_curves()`

For each trial (both conditions), reads all `CONTEXT_BUILT` events in order, plots token count vs decision number.

Output: `results/figures/context_growth_n3.pdf`, `_n5.pdf`, `_n10.pdf`

Expected shape:
- DACS: roughly flat line with small oscillations (focus context grows slowly per agent)
- Flat: diagonal line going up-right, eventually hitting capacity

This is **Figure 2** in the paper.

#### Script 3 — `accuracy_vs_decision_count()` (Phase 3 result)

For D1 through D5 conditions, plots accuracy vs decision density.

Expected: two diverging curves. Key finding: "DACS advantage is negligible at D1 but significant at D3+."

This is **Figure 3** in the paper.

#### Script 4 — `temporal_accuracy()` (Phase 4 result)

For long-horizon trials, splits decisions into 15-minute bins and plots accuracy per bin over time.

Expected: DACS flat, flat baseline declining. Annotate bins where contamination events were logged.

This is **Figure 4** in the paper.

#### Script 5 — `token_efficiency_ratio()` (Phase 6 result)

Plots (flat context tokens) / (DACS context tokens) as a function of N.

Expected: ratio grows roughly linearly with N. At N=50, ratio ≈ 20–50×.

This is **Figure 5** in the paper.

### Table for Paper (Table 2)

| Condition | N | Accuracy | Contamination | avg_ctx_tok | p value |
|---|---|---|---|---|---|
| DACS | 3 | | | | — |
| Flat | 3 | | | | 0.XXX |
| DACS | 5 | | | | — |
| Flat | 5 | | | | 0.XXX |
| DACS | 10 | | | | — |
| Flat | 10 | | | | 0.XXX |

---

## 15. Paper Connection Map

| Experiment | Paper Section | Figure/Table |
|---|---|---|
| Phase 1 — 60-trial canonical | §4.1 Main Results | Table 2, Figure 2 |
| Phase 2 — agent diversity | §4.2 Heterogeneity | Figure 3 |
| Phase 3 — decision density | §4.3 Density Scaling | Figure 4 |
| Phase 4 — long horizon | §4.4 Long-Horizon Stability | Figure 5 |
| Phase 5 — stress tests | §4.5 Correctness Under Adversarial Conditions | Prose |
| Phase 6 — N=20..50 | §4.6 Scale | Figure 6, Table 3 |
| Any phase | §2 Related Work | Cited as "motivating experiment" |

### Key Claims the Experiments Support

1. *"DACS improves steering accuracy over flat-context baseline by X% at N=3 and Y% at N=10."* → Phase 1
2. *"The accuracy advantage grows with N."* → Phase 1, Phase 6
3. *"The flat-context baseline exhibits measurable contamination (Z%) while DACS shows near-zero."* → Phase 1
4. *"DACS maintains quality for hours while flat degrades after ~90 minutes."* → Phase 4
5. *"Beyond N=20, flat-context is infeasible due to context window limits; DACS scales to N=50."* → Phase 6

---

## 16. Infrastructure & Tooling Checklist

### Before Phase 1

- [ ] Verify `s2_n5` and `s3_n10` scenarios in `task_suite.py` have correct decision points
- [ ] Add `total_input_tokens` and `total_output_tokens` to `summary.csv` columns
- [ ] Add `time_to_steer_ms` metric computation to `metrics.py`
- [ ] Add `interrupt_rate` metric to `metrics.py`
- [ ] `run_experiment.py` accepts `--all` flag to run all scenarios sequentially
- [ ] `run_experiment.py` writes intermediate checkpoints every 5 trials (in case of crash mid-experiment)
- [ ] Error handling: if one trial fails due to API error, log it and continue to next trial

### Before Phase 2

- [ ] Implement `DebuggerAgent` in `agents/debugger_agent.py`
- [ ] Implement `LongWriterAgent` in `agents/long_writer_agent.py`
- [ ] Add scenarios `s4`–`s6` to `task_suite.py`
- [ ] Add M9 (cross-domain contamination) and M10 (domain isolation) to `metrics.py`
- [ ] Extend `monitor.py` to show domain labels per agent in CONTEXT events

### Before Phase 3

- [ ] Implement `MLTrainerAgent` in `agents/ml_trainer_agent.py`
- [ ] Implement `HypothesisTesterAgent` in `agents/hypothesis_tester_agent.py`
- [ ] Add scenarios `s7`–`s8` with D2 and D3 density
- [ ] `run_experiment.py --analyze` flag triggers analysis scripts after run completes
- [ ] Figure generation scripts in `experiments/analyze.py`

### Before Phase 4

- [ ] Implement `PlannerAgent` in `agents/planner_agent.py`
- [ ] Implement `DataEngineerAgent` in `agents/data_engineer_agent.py`
- [ ] All agents support configurable `sleep_between_steps` parameter for realistic timing
- [ ] Add `TemporalAnalyzer` to `metrics.py` for rolling accuracy windows
- [ ] Add scenarios `s9`–`s11`
- [ ] Monitor shows elapsed wall-clock time per trial
- [ ] `results/` directory separates long-horizon trials from short ones (subdirectory per phase)

### Before Phase 5

- [ ] Implement `SecurityAuditorAgent` in `agents/security_auditor_agent.py`
- [ ] Implement `ReviewerAgent` in `agents/reviewer_agent.py`
- [ ] Stress test runner: a separate script `experiments/stress_tests.py` that runs the 5 adversarial scenarios
- [ ] Semantic contamination detector: vocabulary overlap metric in `metrics.py`

### Before Phase 6

- [ ] N=20 scenario `s12` defined
- [ ] Test that `RegistryManager` handles 50 concurrent agents without asyncio deadlock
- [ ] Test that `SteeringRequestQueue` correctly prioritizes across 50 agents
- [ ] Token budget guard: assert that DACS contexts never exceed 10,000 tokens even at N=50

---

## 17. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| API rate limits during 60-trial experiment | Medium | High — experiment stalls | Add exponential backoff + rate limiter in `orchestrator.py`. Run scenarios in batches with 30s delay between batches. |
| Flat baseline crashes with ContextBudgetError at N=10 | Medium | Low — actually a finding | Catch `ContextBudgetError`, log it as a CONTEXT_OVERFLOW event, skip that trial (annotate as "infeasible"). Report overflow rate in paper. |
| LLM doesn't contain answer keywords in response | Medium | Medium — lowers accuracy below true value | Expand `answer_keywords` lists with synonyms. Keep original keywords as primary, synonyms as secondary. |
| MiniMax M2.7 model access interrupted | Low | Very High — experiment blocked | Document which API calls to switch to (Anthropic Claude Sonnet, GPT-4o) by changing base_url and model name only. Keep switching cost minimal. |
| Long-horizon trial crashes after 2 hours | Medium | High — wasted API cost | Implement checkpointing: serialize registry and queue state to disk every 10 decisions. Resume from checkpoint on re-run. |
| asyncio task ordering non-determinism | High | Medium — trial variance | Acceptable as natural variance. 10 trials per cell covers stochastic effects. Do not fix ordering unless it causes systematic bias. |
| Phase 4 runs too slow (4+ hours per trial) | High | Medium — schedule risk | Add `--fast` flag that uses 0.1s sleep instead of realistic sleep times. Run paper analysis on fast trials, mention wall-clock times separately. |
| MiniMax API billing spike | Medium | Low | Set hard cap on API account. Monitor via API dashboard after each phase run. Budget estimate: 60-trial Phase 1 ≈ 60 × 30 calls × 1000 tokens avg ≈ 1.8M input tokens. |

---

## 18. Execution Timeline

This timeline reflects phase-by-phase ordering. Do not start a phase until the previous one passes its success criteria.

### Week 1 (Now — April 4–11, 2026)

| Day | Task |
|---|---|
| Apr 4 (today) | Run Phase 0 baseline verification. Confirm flat trial works end to end. |
| Apr 5 | Fix any Phase 0 failures. Verify s2_n5 and s3_n10 scenarios are complete. |
| Apr 5–6 | Run Phase 1 N=3 cells (20 trials). Review results. |
| Apr 7 | Run Phase 1 N=5 cells (20 trials). |
| Apr 8 | Run Phase 1 N=10 cells (20 trials, may need overnight). |
| Apr 9 | Run Phase 1 analysis: compute effect sizes, generate Figures 1–2. |
| Apr 10–11 | Write Phase 1 analysis notes. Begin Phase 2 agent implementations (DebuggerAgent, LongWriterAgent). |

### Week 2 (April 12–18, 2026)

| Day | Task |
|---|---|
| Apr 12–13 | Finish DebuggerAgent + LongWriterAgent + new scenarios (s4, s5, s6). |
| Apr 14–15 | Run Phase 2 (60 trials). Analyze. |
| Apr 16 | Implement MLTrainerAgent + HypothesisTesterAgent + dense scenarios (s7, s8). |
| Apr 17–18 | Run Phase 3 D2/D3 density trials. Generate accuracy-vs-density plot. |

### Week 3 (April 19–25, 2026)

| Day | Task |
|---|---|
| Apr 19–20 | Implement PlannerAgent + DataEngineerAgent. Build scenarios s9, s10. |
| Apr 21–22 | Run Phase 4 s9 (1-hour trial, DACS and flat, 3 trials each). Analyze temporal degradation. |
| Apr 23–24 | Run Phase 4 s10 (3-hour trial). This is the hero experiment. |
| Apr 25 | Run Phase 5 stress tests. Check for correctness failures. |

### Week 4 (April 26 – May 2, 2026)

| Day | Task |
|---|---|
| Apr 26–28 | Complete all analysis scripts. Generate all paper figures. |
| Apr 29 – May 2 | Begin paper writing (Sections 1–4). Results are all in hand. |

---

## Status Tracking

| Phase | Status | Trials Done | Key finding |
|---|---|---|---|
| Phase 0 — Verification | ✅ Complete | 2/2 | Both conditions verified, exit 0 |
| Phase 1 — 60 trials | ✅ Complete | 60/60 | DACS 90–96.7% vs baseline 21–60%, all p<0.0001, RQ1+RQ2 confirmed |
| Phase 2 — Diversity | ✅ Complete | 61/60 | DACS 90–96% vs baseline 37–57%, +37–59 pp gap, RQ3 confirmed |
| Phase 3 — Density | ⏳ Not started | 0/~40 | — |
| Phase 4 — Long horizon | ⏳ Not started | 0/~20 | — |
| Phase 5 — Adversarial | ⏳ Not started | 0/~10 | — |
| Phase 6 — Scale N=20..50 | ⏳ Not started | 0/~20 | — |

---

*Last updated: April 4, 2026 — Phase 2 complete*  
*Total planned trials: ~210 experimental runs + long-horizon sessions*  
*Total estimated LLM calls: 5,000–15,000 depending on Phase 4 scope*
