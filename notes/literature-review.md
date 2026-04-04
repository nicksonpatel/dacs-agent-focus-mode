# Literature Review Notes

## Reading Order & Status

### Original 5 Papers (Week 1 reading list)
- [x] [2511.12712] AFM — Adaptive Focus Memory
- [x] [2512.13956] AOI — Three-layer memory + dynamic scheduling
- [x] [2506.12508] AgentOrchestra — lifecycle + context management
- [x] [2510.04618] ACE — Evolving context as first-class concept
- [x] [2602.16873] AdaptOrch — Topology as optimization frontier

### Additional Papers Found via Extended Prior-Art Search (April 4, 2026)
- [x] [2601.14914] CodeDelegator — context pollution via role separation
- [x] [2602.22603] SideQuest — KV cache management single-agent
- [x] [2601.09742] Adaptive Orchestration — DMoE for context pollution + attention decay
- [x] [2602.07092] Lemon Agent — three-tier progressive context compression

### Search Coverage Log (all searched April 4, 2026, all clear of threats)
Arxiv full-text searches conducted — 20 angle-based queries:
- "context pollution multi-agent LLM" → 5 results (CodeDelegator, SideQuest added)
- "context isolation multi-agent LLM orchestrator" → 15 results (none threatening)
- "context switching agent orchestrator LLM steering" → 0 results
- "attention decay context pollution LLM multi-agent" → 1 result (Adaptive Orchestration added)
- "dynamic context management concurrent agents LLM orchestration" → 4 results (Lemon Agent added; others cleared)
- "context scoping LLM agents dynamic" → 14 results (none threatening)
- "selective context injection multi-agent LLM" → 5 results (MEMO/MASteer — different domains, cleared)
- "wrong agent contamination LLM context orchestrator" → 0 results ✓
- "focus session LLM agent orchestration context" → 0 results ✓
- "per-agent context management orchestrator focus mode" → 0 results ✓
- "agent callback orchestrator context management concurrent LLM" → 0 results ✓
- "agent interrupt priority orchestrator context window LLM" → 0 results ✓
- "token budget multi-agent orchestrator context isolation steering" → 0 results ✓
- "cross-agent context contamination orchestrator LLM" → 0 results ✓
- "concurrent agent context window scoping orchestrator LLM mode" → 0 results ✓
- "hierarchical context flat context orchestrator multi-agent LLM" → 0 results ✓
- "orchestrator attention mode agent context concurrent LLM interaction" → 0 results ✓
- "agent registry orchestrator context window multi-agent" → 0 results ✓
- "steering request agent LLM concurrent context" → 0 results ✓
- "surgical context pruning orchestrator agent LLM" → 0 results ✓

**Verdict: No paper found that implements agent-triggered asymmetric REGISTRY/FOCUS mode switching for per-agent context isolation in concurrent multi-agent orchestration. DACS's core mechanism is not anticipated in the literature as of April 4, 2026.**

---

## [2511.12712] AFM — Adaptive Focus Memory

**Link:** https://arxiv.org/abs/2511.12712

**Core mechanism:**
A pluggable context manager operating entirely at the prompt-construction layer for single-user multi-turn dialogue. At each turn, scores every past message with a composite scalar (semantic similarity + recency decay + importance classification), assigns each message a fidelity tier (Full / Compressed / Placeholder), then greedily packs messages chronologically under a fixed token budget `B`. All state is in-process (no retrieval store, no DB, no model changes).

**What triggers FULL vs COMPRESSED vs PLACEHOLDER:**
Per-message score `s_i` is computed piecewise:
- CRITICAL (LLM-classified) → `s_i = 1.0` (force Full regardless of age)
- RELEVANT → `s_i = max(0, sim_i) * (0.4 + 0.4 * w_recency)`
- TRIVIAL → `s_i = max(0, sim_i) * (0.25 * w_recency)`

Recency weight: `w_recency = 0.5^(k/h)`, default half-life `h = 12` turns.

Tier thresholds (OpenAI-backed): `τ_high = 0.45` → Full; `τ_mid = 0.25` → Compressed; else → Placeholder.

During packing, if a Full representation doesn't fit, AFM downgrades to Compressed, then to Placeholder stub. Even stubs are dropped if budget is exhausted.

Two compressors: `HeuristicCompressor` (local, extractive, deterministic budget) and `LLMCompressor` (abstractive via gpt-4o-mini; budget not re-verified post-generation).

**Does it handle multiple concurrent agent threads:**
No. AFM is designed strictly for a single user ↔ assistant conversation. One context window, one history, one current query. No concept of N-concurrent agents or per-agent context isolation.

**Key metrics/numbers:**
- Peanut allergy benchmark (Thailand travel, 30 seeds): AFM **83.3%** pass rate vs **0%** for all baselines (Default, Naive Truncated Replay, Recency Compression) under strict grading
- Tax compliance benchmark (illegal evasion, 30 seeds): all methods 100% — easy because the final request is overt; used as sanity check
- Token reduction: ~⅔ reduction vs full replay in prior informal evals
- AFM latency overhead on graded turn: ~21 s vs ~4–11 s for baselines (pipeline: importance scoring + context selection + optional compression)
- Ablation: removing importance classification collapses allergy pass rate to **0%** — the dominant driver is which messages get Full fidelity, not compression per se

**What DACS does that AFM does NOT:**
- **Multi-agent concurrency:** AFM manages one conversation thread; DACS manages N concurrent agent threads competing for the same orchestrator context window.
- **Agent-triggered isolation:** AFM is query-triggered (fires on every user turn); DACS is agent-triggered via `SteeringRequest` — agents signal when they need orchestrator attention.
- **Asymmetric context switching:** AFM always works on one flat history; DACS has two distinct orchestrator modes — `REGISTRY mode` (holds compressed summaries of all agents) and `FOCUS(aᵢ) mode` (injects the full context of one agent, compresses the rest). The asymmetry is structural, not fidelity-scoring.
- **Context pollution problem:** AFM's problem is constraint drift over a long single conversation. DACS's problem is inter-agent context pollution — multiple agent threads contaminating each other's steering interactions when they share a context window.
- **No steering accuracy metric in AFM:** AFM measures constraint recall; DACS measures steering accuracy (did the orchestrator give the right answer to the right agent?) and wrong-agent contamination rate.

---

## [2512.13956] AOI — Three-layer memory architecture

**Link:** https://arxiv.org/abs/2512.13956

**Core mechanism:**
AOI (AI-Oriented Operations) is a multi-agent framework for autonomous IT operations management in cloud-native microservice environments. Three specialized agents — Observer (coordination + dynamic scheduling), Probe (read-only diagnostics, safety-constrained), Executor (system modification with checkpoint/rollback) — collaborate through a central LLM-based Context Compressor. The framework is formally grounded as a Dec-POMDP over the joint state of the IT infrastructure.

Four innovations:
1. **Dynamic task scheduling** — Observer balances exploration (probing) vs exploitation (remediation) at each timestep via expected-reward maximization over real-time system state.
2. **Three-layer memory architecture** — Layer 1: Raw Context Store (24h retention, unprocessed Probe/Executor outputs). Layer 2: Task Queue Management (structured execution instructions, lifecycle-managed by Observer). Layer 3: Compressed Context Cache (7-day retention, LLM-processed summaries, queried by Observer for strategic decisions).
3. **LLM-based Context Compressor** — Sliding window (50% overlap, 768-token optimum) + semantic summarization. Extracts fault signatures, error codes, anomaly patterns. Two-pass: per-window summary → merge overlaps → secondary compression if needed.
4. **Safety-by-design** — Probe is strictly read-only (whitelist validator, no writes). Executor creates full-system checkpoint before each action; critical failure triggers instant rollback.

**How they achieve 72.4% compression / 92.8% critical info retention:**
Sliding window with 50% overlap (prevents boundary loss) + LLM semantic extraction of operationally critical content (error codes, resource thresholds, causal chains) before summarizing. Overlap ratio ρ > 0.5 gives contraction in the Lyapunov stability proof (Theorem 2), theoretically bounding information loss. Empirically: window size 768 tokens is the sweet spot — 1024+ tokens gives <0.3pp additional IPS gain at higher cost. Information Preservation Bound (Theorem 3): I(C_comp; Y) ≥ (1 − ε_info)·I(C; Y) where ε_info = O(1/(w·ρ)).

**Does it address orchestrator-to-agent steering specifically:**
No. AOI has inter-agent communication but it is role-segregated and unidirectional in purpose: Observer dispatches tasks to Probe/Executor via a prioritized subtask queue; Probe/Executor return results to the Raw Context Store; Context Compressor processes them into the Compressed Cache. There is no concept of an agent *requesting* steering from the orchestrator, nor does the orchestrator context switch to prioritize one agent thread over others. The Observer always has the full (compressed) picture of all activity — there is no per-agent context isolation.

**Benchmarks they use:**
- **AIOpsLab** (synthetic, 1,000 fault-injection scenarios, 50 fault types) — primary. TSR 94.2%, MTTR 22.1 min.
- **Loghub** real-world logs: HDFS, BGL, OpenStack — cross-dataset generalization. TSR 92.1–93.4% across all datasets.
- Baselines: Rule-based Expert System (RES), Traditional AIOps Pipeline (TAP), Single-Agent LLM (SA-LLM), Baseline Multi-Agent System (B-MAS).
- *Potential DACS reuse:* AIOpsLab and Loghub are IT-ops benchmarks, not general multi-agent steering benchmarks. DACS's steering accuracy metric (right answer to right agent) has no counterpart here. Not directly reusable, but AIOpsLab's fault-injection structure is interesting as a domain for future DACS extension.

**What DACS does that AOI does NOT:**
- **Per-agent context isolation during steering.** AOI's Observer always holds a compressed aggregate of all agent activity — it sees everything at once. DACS explicitly separates REGISTRY mode (summary of all agents) from FOCUS(aᵢ) mode (full context of exactly one agent). AOI has no equivalent mode switch.
- **Agent-triggered context requests.** AOI is orchestrator-driven: Observer decomposes and dispatches. Agents execute and report. No agent can signal "I need steering attention now" — that is precisely the `SteeringRequest` mechanism DACS introduces.
- **Context pollution is not AOI's problem.** AOI's bottleneck is information overload from operational data volume (logs, metrics). DACS's bottleneck is inter-agent thread contamination in the orchestrator's context during steering. Different problem statement, different solution shape.
- **Asymmetric context modes.** AOI uses one context path (raw → compressed → cache) for everyone. DACS has two structurally different context states for the orchestrator depending on which agent is being steered.
- **Steering accuracy as a metric.** AOI measures TSR (task success), MTTR, CCR, IPS. DACS measures whether the orchestrator gives the *correct answer to the correct agent* — wrong-agent contamination is a first-class failure mode that AOI does not model.
- **Use in DACS paper:** AOI is useful as a concrete example of *why* multi-agent context management matters — its 5.7pp TSR drop when context compressor is removed shows information management is load-bearing. Cite for: (1) motivating context management in multi-agent systems broadly, (2) contrasting the IT-ops compression problem (volume) with DACS's steering isolation problem (cross-agent contamination).

---

## [2506.12508] AgentOrchestra

**Link:** https://arxiv.org/abs/2506.12508

**Core argument:**
Existing LLM agent protocols (A2A, MCP) leave three protocol-level gaps: (i) lifecycle and context management are fragmented — no unified primitives to manage lifecycles and maintain consistent, versioned execution context across agent-associated components; (ii) self-evolution is not supported at the protocol level — prompts and resources are externally maintained assets with no closed-loop refinement via execution feedback; (iii) environments are not first-class — delegated to application-specific runtimes rather than managed components with explicit boundaries. To address this, the paper introduces the **Tool-Environment-Agent (TEA) protocol**, a unified abstraction treating environments, agents, and tools as first-class, versioned resources with explicit lifecycles. Three context protocols (TCP / ECP / ACP) provide lifecycle management, semantic retrieval via vector embeddings, and versioned component state. Six **protocol transformations** (A2T, T2A, E2T, T2E, A2E, E2A) allow dynamic role reconfiguration. A **Self-Evolution Module** wraps components as optimizable variables (TextGrad + self-reflection), auto-registering improvements as new versions. Built on TEA, **AgentOrchestra** is a hierarchical multi-agent framework: a central planning agent decomposes tasks and delegates to specialized sub-agents (Deep Researcher, Browser Use, Deep Analyzer, Tool Generator, Reporter), each maintaining localized toolsets and context for their domain. The authors explicitly note that flat coordination "tends to accumulate irrelevant context" — hierarchical delegation bounds the orchestrator's context footprint by converting global coordination into localized routing decisions.

**Specific gaps they identify in cross-entity lifecycle and context management:**
Three explicitly stated:
1. **Fragmented lifecycle and context management** — neither A2A nor MCP provides unified primitives to manage component lifecycles or maintain consistent, versioned execution context across the full tool-environment-agent stack. Every application reimplements this glue code.
2. **No protocol-level self-evolution** — both A2A and MCP treat prompts and resources as externally managed static assets; no standard mechanism exists for a closed feedback loop to refine them with traceable versioning.
3. **Environments not first-class** — existing protocols delegate environments to application-specific runtimes, making it hard to switch agents across environments, reuse environments, or isolate parallel runs. Loss of session-critical state during domain transitions (e.g., browser retrieval → local Python analysis) is a documented failure mode on GAIA.

The paper's own framing of the context accumulation problem is DACS's strongest related-work ally: they state flat coordination causes the orchestrator to "tend to accumulate irrelevant context" and justify hierarchical delegation precisely to keep the orchestrator's "decision scope and context footprint bounded."

**What they propose vs what DACS proposes:**
AgentOrchestra solves the context accumulation problem **architecturally** (pre-execution structural choice): by routing each sub-task to a domain-specific sub-agent that exposes only a curated, localized toolset and context, the planning agent never sees all raw context simultaneously. Context scoping is achieved by tree-structured agent hierarchy — the orchestrator interacts with agents, not with all tools/data directly.

DACS solves the context pollution problem **temporally** (runtime, during execution): even after hierarchical decomposition, when N concurrent sub-agents simultaneously need steering from the orchestrator, their contexts still compete in one window. DACS introduces `SteeringRequest`-triggered focus sessions — dynamically switching the orchestrator between REGISTRY mode (compressed summaries of all agents) and FOCUS(aᵢ) mode (full context of one agent, compressed registry) — to isolate per-agent steering interactions. This is orthogonal to AgentOrchestra's approach and addresses the residual pollution that hierarchical routing leaves unsolved at the intra-steering-interaction level.

**Benchmarks used:**
- **GAIA** (301 test / 165 validation questions, 3 difficulty levels): AgentOrchestra achieves **89.04% avg** on test, **82.42% avg** on validation (92.45% L1, 83.72% L2, 57.69% L3) — SOTA. Baselines: AWorld 77.58%, Langfun 76.97%, Manus 73.90%.
- **SimpleQA** (4,326 short-form factuality questions): **95.3% accuracy** — vs o3 49.4%, Perplexity Deep Research 93.9%.
- **HLE — Humanity's Last Exam** (2,500-question expert-level multimodal benchmark): **37.46%** — vs o3 20.3%, Perplexity Deep Research 21.1%.
- Ablation on GAIA test: Planning-only baseline 36.54% → +Researcher+Browser 72.76% → +Analyzer 79.07% (+8.67pp) → +Tool Generator 89.04% (+12.61pp). Tool Generator's 12.61pp gain is the strongest evidence that static toolsets are insufficient.
- Efficiency: simple tasks ~30s / 5k tokens; medium ~3min / 25k tokens; complex ~10min / 100k tokens.
- Self-evolution on AIME25: gpt-4.1 with self-reflection improves from 20.00% → 33.34%.
- *Potential DACS reuse:* GAIA's three difficulty levels and multi-domain task structure are relevant for grounding long-horizon steering scenarios. However, GAIA measures task completion, not steering accuracy or wrong-agent contamination — not directly reusable as DACS's primary metric. Could serve as an application domain for a future DACS extension.

**What DACS does that AgentOrchestra does NOT:**
- **Per-agent context isolation during steering.** AgentOrchestra's planning agent bounds its context footprint via hierarchical routing (localized sub-agents) but still accumulates context across all concurrent agent interactions in one window. The paper acknowledges "frequent inter-agent exchanges can introduce latency and overhead" — the context window accumulation during concurrent steering is exactly DACS's problem. DACS does not just bound the footprint; it *switches* the orchestrator's context to contain only one agent's thread at steering time.
- **Agent-triggered context requests.** AgentOrchestra is planner-driven: the planning agent decomposes and dispatches. Sub-agents execute and return results. No sub-agent can signal "I need steering attention now" — the `SteeringRequest` mechanism DACS introduces is entirely absent. AgentOrchestra's planning agent decides when to invoke agents; DACS's agents decide when they need orchestrator input.
- **REGISTRY/FOCUS mode asymmetry.** TEA's context protocols (TCP/ECP/ACP) manage per-component context for lifecycle and discovery — they are always-on registries. DACS has two structurally different runtime modes for the orchestrator: one where it holds only summaries of all agents, another where it injects one agent's full context. No equivalent mode switch exists in AgentOrchestra.
- **Context pollution between concurrent agents as explicit failure mode.** AgentOrchestra measures task success (GAIA pass@1). DACS measures steering accuracy (did the orchestrator give the correct answer to the correct agent?) and wrong-agent contamination rate — failure modes that AgentOrchestra does not model or measure.
- **Complementarity:** AgentOrchestra's hierarchical routing reduces *how much* context enters the orchestrator. DACS controls *what* context the orchestrator sees *at the moment of steering*. A production system could use both: AgentOrchestra's TEA protocol for lifecycle/tool management, and DACS for runtime steering isolation.
- **Use in DACS paper:** AgentOrchestra is the most direct prior work for citing: (1) independent, empirical validation that context accumulation in flat orchestrators is a real perf problem (their motivation section and ablation both confirm it), (2) contrast — hierarchical routing is a structural partial solution, DACS is a runtime complete solution for the residual problem, (3) their GAIA results show context management drives performance even with state-of-the-art orchestration, grounding DACS's thesis.

---

## [2510.04618] ACE — Agentic Context Engineering (Evolving Context)

**Link:** https://arxiv.org/abs/2510.04618
*(Published ICLR 2026; v3: 29 Mar 2026)*

**Core mechanism:**
ACE (Agentic Context Engineering) is a framework for adapting LLM contexts without weight updates, for both offline settings (e.g., system prompt optimization) and online settings (e.g., test-time agent memory). The core insight is that prior context adaptation methods suffer from two failure modes: (1) **brevity bias** — optimizers converge toward short, generic prompts, discarding domain-specific heuristics; (2) **context collapse** — iterative LLM-based rewriting of the full context compresses accumulated knowledge into much shorter, less informative summaries (empirically: at step 60 a 18,282-token context collapsed to 122 tokens, dropping accuracy from 66.7 → 57.1, below the no-adaptation baseline of 63.7). ACE treats context as an **evolving playbook** — a structured, itemized collection of bullets (each with metadata: unique ID + helpful/harmful counters + content). Three-role architecture: **Generator** (produces reasoning trajectories), **Reflector** (extracts concrete insights from successes/errors via iterative critique), **Curator** (merges insights into compact delta updates). Two key mechanisms: (1) **Incremental delta updates** — instead of rewriting the full context, ACE appends/modifies only the changed bullets, preventing context collapse; (2) **Grow-and-refine** — periodically deduplicates bullets via semantic embeddings and prunes low-utility entries to control playbook size. Works offline (batch optimization over training split) and online (sequential, test-time learning without GT labels).

**How context is treated as first-class:**
ACE argues contexts should be treated as structured, persistent artifacts — not ephemeral prompt strings — with the same lifecycle management afforded to code: versioned bullets with IDs, counters tracking per-bullet utility, de-duplication logic, and rollback capability (grow-and-refine pruning). The key claim is that LLMs are actually *more* effective with long, detailed contexts that include domain heuristics, tool-use rules, and common failure modes — and can distill relevance at inference time — rather than concise summaries that discard that detail. Context is not an implementation detail of the prompt; it is the primary optimization variable for self-improving systems.

**Benchmarks used:**
- **AppWorld** (LLM agent benchmark: multi-turn reasoning, tool/API use, code generation): ReAct + ACE achieves **59.4% avg TGC/SGC** overall, matching IBM CUGA (60.3%, production GPT-4.1 agent) despite using open-source DeepSeek-V3.1; surpasses IBM CUGA on test-challenge split (+8.4% TGC, +0.7% SGC). Average gain over strong baselines: **+10.6%** (vs ICL, GEPA, DC). Without GT labels: +14.8% over base ReAct.
- **FiNER + Formula** (financial XBRL reasoning): avg **+8.6%** over baselines offline; +6.2% online. Base DeepSeek-V3.1: 70.7% → ACE: 78.3% on FiNER.
- **DDXPlus** (medical reasoning, StreamBench): 75.2 → 90.2 (+15.0 accuracy) vs GEPA's minimal +1.2.
- **BIRD-SQL** (text-to-SQL, StreamBench): Base 47.8 → ACE 52.9 (+5.1 avg).
- Cost/latency: 82.3% reduction in adaptation latency vs GEPA (AppWorld); 91.5% reduction vs DC (FiNER). 91.8% of input tokens served from KV cache at eval, reducing billed cost by 82.6%.
- *Potential DACS reuse:* AppWorld's multi-turn agent tasks require accumulated context across episodes — structurally similar to DACS's per-agent context accumulation across a task. However, ACE measures task goal completion for a single agent, not orchestrator steering accuracy across N concurrent agents. Not directly reusable as DACS metric benchmark, but AppWorld is relevant as a demonstration domain.

**What DACS does that ACE does NOT:**
- **Multi-agent concurrency.** ACE operates entirely within a single agent's context window — one generator, one evolving playbook, one task thread. There is no concept of N concurrent agents competing for an orchestrator's attention. DACS is designed precisely for the N-agent case where multiple agents' contexts would collide in the orchestrator window.
- **Agent-triggered context isolation.** ACE's context adaptation fires on every turn/episode automatically (or is pre-computed offline). DACS introduces a fundamentally different trigger: a `SteeringRequest` — an agent explicitly signals when it needs orchestrator input. The orchestrator then *mode-switches* its context, not just updates it.
- **Asymmetric mode switching (REGISTRY vs FOCUS).** ACE maintains one context for one agent — it evolves over time but always describes the same single thread. DACS maintains two structurally distinct orchestrator states: REGISTRY mode (compressed summaries of *all* agents) and FOCUS(aᵢ) mode (full context of *one* agent + compressed registry). This asymmetry has no analogue in ACE.
- **Context pollution between agents.** ACE's problem is *intra-agent* context degradation over time (brevity bias, context collapse within one thread). DACS's problem is *inter-agent* context pollution — multiple concurrent agents contaminating each other's steering interactions when they share an orchestrator context window. Entirely different failure mode.
- **Steering accuracy as metric.** ACE measures task goal completion (TGC/SGC) and domain accuracy. DACS measures whether the orchestrator gives the correct answer to the correct agent — wrong-agent contamination rate is a first-class failure mode that ACE does not model or measure.
- **Complementarity:** ACE could be applied *within* a DACS focus session — once the orchestrator is in FOCUS(aᵢ) mode, ACE-style incremental playbook updates could improve the quality of the steered agent's context over time. The mechanisms are orthogonal: ACE manages single-agent context quality over episodes; DACS manages which agent's context is active in the orchestrator at steering time.
- **Use in DACS paper:** ACE is relevant for: (1) motivating why accumulated agent context matters — ACE shows that detailed, comprehensive context improves task performance by double-digit percentages, which grounds DACS's claim that injecting full F(aᵢ) into the orchestrator is worth the mechanism cost; (2) contrast — ACE addresses single-agent context quality, DACS addresses multi-agent context isolation; cite to establish that context engineering broadly is a load-bearing performance variable; (3) note ACE's observation that LLMs benefit from *more* context detail, not compressed summaries — consistent with DACS's FOCUS mode injecting the full agent context rather than a registry summary.

---

## [2602.16873] AdaptOrch — Topology as optimization frontier

**Link:** https://arxiv.org/abs/2602.16873

**Core argument:**
As frontier LLMs from different providers converge to within ε ≈ 0.03–0.05 of each other on standard benchmarks (MMLU, HumanEval, MATH), the primary optimization lever shifts from *which model* to *how agents are coordinated*. AdaptOrch formalizes this as a **topology selection problem**: given a task, choose among four canonical orchestration topologies (parallel τ_P, sequential τ_S, hierarchical τ_H, hybrid τ_X) based on the structural properties of the task's dependency DAG.

Three contributions:
1. **Performance Convergence Scaling Law** — under ε-convergence, topology variance dominates model-selection variance by Ω(1/ε²). For coding tasks with ω≥3, γ≤0.4, ε≈0.05: Var_τ/Var_M ≥ 20.
2. **Topology Routing Algorithm** (Algorithm 1) — linear-time O(|V|+|E|) routing based on DAG parallelism width ω, coupling density γ, and critical path depth δ. Default thresholds: θ_ω=0.5, θ_γ=0.6, θ_δ=5.
3. **Adaptive Synthesis Protocol** — heuristic consistency scoring (embedding cosine similarity across outputs), with provable termination in ≤5 re-routing iterations; empirically 94% converge in ≤2.

**How they frame the shift from model selection to orchestration structure:**
Explicit "Era 1 → Era 2" framing: model selection dominated when models differed; orchestration topology dominates now that they converge. Practical motivators cited: Claude Code Agent Teams and OpenCode show that parallel agents with isolated context windows each compress multi-hour sequential workflows into minutes — but leave the decomposition itself unsolved. AdaptOrch provides the algorithmic solution to that gap.

Key results (Table 2): AdaptOrch vs. Single Best — SWE-bench Verified +9.8pp (42.8→52.6%), GPQA Diamond +6.9pp (46.2→53.1%), HotpotQA +8.1pp (68.3→76.4%). Token consumption *lower* than MoA-3L and LLM-Blender because topology-aware routing avoids redundant calls (41.8K vs 84.6K tokens/instance on SWE-bench). Router accuracy: 81.2% agreement with oracle topology. Important negative result: Static-Parallel *degrades* GPQA below Single Best — topology mismatch is actively harmful on high-coupling reasoning tasks.

Topology distribution: hybrid τ_X is most common overall (49.7%), parallel τ_P preferred for low-coupling tasks, sequential τ_S dominates high-coupling reasoning (41% on GPQA).

**What DACS adds to this framing:**
- **AdaptOrch optimizes topology *before* execution** — it routes a task DAG to the right structural pattern once. DACS operates *during* execution — it dynamically switches the orchestrator's active context as agents emit `SteeringRequest` signals. These are complementary but different temporal scopes.
- **AdaptOrch does not address intra-orchestrator context pollution.** In the hierarchical executor (τ_H), the lead agent (orchestrator) maintains a global task list with all sub-agent outputs accumulating in one context. AdaptOrch treats this as fine — DACS identifies it as the core problem: when N agent threads compete in that single orchestrator context window, steering accuracy degrades. DACS is the mechanism that *solves* the pollution AdaptOrch's τ_H topology creates at scale.
- **No per-agent context isolation in AdaptOrch.** Parallel agents each have isolated context windows (which AdaptOrch correctly notes as a feature of τ_P), but the orchestrator that synthesizes their outputs has *one flat context*. DACS explicitly manages what that orchestrator context contains at each steering moment.
- **DACS is agent-triggered; AdaptOrch is task-triggered.** AdaptOrch makes one topology decision per task at decomposition time. DACS makes repeated context-switch decisions throughout execution, each triggered by a specific agent's `SteeringRequest`.
- **Use in DACS paper:** AdaptOrch's convergence framing provides the strongest motivation for *why orchestration matters*. Cite for: (1) establishing orchestration as the dominant performance variable, (2) justifying why DACS's intra-orchestrator context management is non-trivial at scale, (3) noting that AdaptOrch leaves the per-steering-interaction context problem unsolved.

---

## [2601.14914] CodeDelegator — Mitigating Context Pollution via Role Separation

**Link:** https://arxiv.org/abs/2601.14914

**Why this was found:** Arxiv search for "context pollution multi-agent LLM" — their title uses the exact term. Read *after* initial 5-paper pass; added to lit review April 2026.

**Core mechanism:**
CodeDelegator addresses context pollution in code-as-action agents by separating planning from implementation via role specialization. A single persistent **Delegator** agent maintains strategic oversight — decomposes tasks, writes specifications, monitors progress — and never executes code. For each sub-task, a **fresh Coder agent** is instantiated with a clean context window containing only its specification and execution environment, shielded from all prior failed attempts and debugging traces. The key coordination primitive is **Ephemeral-Persistent State Separation (EPSS)**: Coder execution state (debug traces, intermediate failures, scratch variables) is flagged as ephemeral and not fed back to the Delegator, while only the final compact result is marked persistent and returned. This prevents debugging traces from accumulating in the Delegator's context.

**What "context pollution" means in this paper:**
A narrower definition than DACS. In CodeDelegator, context pollution = **execution trace accumulation** — specifically, failed code attempts, error messages, and debug output from Coder agents contaminating the Delegator's planning context over time. The Delegator loses strategic clarity as its context fills with irrelevant low-level execution noise from sub-task attempts.

Compare to DACS: in DACS, context pollution = **inter-agent thread contamination** — multiple concurrent agents' full contexts (task state, partial outputs, pending questions from different domains) competing for space in the orchestrator's window at the same time, causing cross-agent steering errors. Entirely different failure mode, even though both papers use the term.

**What CodeDelegator does NOT have (DACS delta):**
- **No N concurrent agents.** CodeDelegator has one Delegator + one Coder at a time (sequential sub-task execution). DACS handles N agents running *simultaneously* with independent lifecycles and steering needs.
- **No agent-triggered isolation.** The Delegator always runs. Coders are spawned fresh by the Delegator's dispatch, not by agents requesting steering attention. The `SteeringRequest` mechanism is entirely absent.
- **No REGISTRY/FOCUS mode asymmetry.** The Delegator is always in the same mode. There is no runtime mode switch where the orchestrator enters a special isolated state to steer one specific agent.
- **Static structural solution, not dynamic.** CodeDelegator solves context pollution by instantiating fresh Coder contexts at task boundary (architectural). DACS solves it by dynamically switching the orchestrator's context during concurrent execution (runtime). CodeDelegator cannot handle the case where multiple concurrent agents simultaneously need steering.
- **No wrong-agent contamination as a failure mode.** CodeDelegator's failure mode is Delegator strategic drift from execution noise. DACS's failure mode is the orchestrator steering Agent A using context from Agent B.

**Use in DACS paper:**
CodeDelegator is strong related work to cite for two reasons:
1. **Validates the problem name.** Their title uses "context pollution" — DACS can adopt this as established terminology with a direct citation.
2. **Sharpens the contrast.** CodeDelegator solves context pollution *statically* (role separation + clean instantiation for single sequential task delegation). DACS solves it *dynamically* for N concurrent heterogeneous agents. The distinction: "CodeDelegator gives each sub-task a clean slate at birth; DACS gives the orchestrator a clean view at steering time, regardless of how many agents are alive simultaneously." These are complementary, not competing — CodeDelegator could be combined with DACS in a production system.

---

## [2602.22603] SideQuest — Model-Driven KV Cache Management

**Link:** https://arxiv.org/abs/2602.22603

**Why this was found:** Same arxiv search; also mentions "context pollution" but at the KV cache layer.

**Core mechanism:**
SideQuest addresses memory explosion in long-horizon single-agent tasks (e.g., deep research requiring multi-hop reasoning over dozens of documents). The LLM context fills rapidly with external retrieval tokens, causing KV cache bloat and inference slowdown. SideQuest frames KV cache compression as an **auxiliary task** running in parallel with the main reasoning task — the Large Reasoning Model (LRM) itself decides which cached token blocks are expendable, using model-driven reasoning rather than heuristics (recency, attention score). The "context pollution" in their framing = retrieval tokens polluting the model's working memory, not inter-agent contamination.

**What DACS does that SideQuest does NOT:**
- SideQuest is single-agent, single-thread, KV cache management at inference time. DACS is multi-agent, multi-thread, orchestrator context management at the prompt-construction layer.
- SideQuest compresses tokens the model already holds. DACS controls which agent's context the orchestrator holds at all.
- No mode switching, no agent-triggered mechanism, no steering accuracy metric.

**Use in DACS paper:**
Brief cite only. Useful for the intro motivation paragraph — even at the single-agent level, context growth is a recognized infrastructure problem (SideQuest: 65% peak token reduction). DACS addresses the structurally harder problem at the multi-agent orchestration layer. Can be cited as: "Context management is load-bearing even for single agents [SideQuest]; for N concurrent agents, DACS addresses the additional inter-thread contamination problem that KV cache techniques leave unsolved."

---

## [2601.09742] Adaptive Orchestration — Scalable Self-Evolving Multi-Agent Systems

**Link:** https://arxiv.org/abs/2601.09742

**Why this matters to DACS:** Uses both "context pollution" and "attention decay" terminology — the only paper found that co-uses these two terms. Must be addressed directly in the DACS paper.

**Core mechanism:**
Addresses the "Generalization-Specialization Dilemma": monolithic LLM agents given extensive toolkits suffer from *context pollution* (too many tools filling the context window) and *attention decay* (critical information gets buried in long contexts, leading to hallucinations). The solution is a Self-Evolving Concierge System using a **Dynamic Mixture of Experts (DMoE)** approach: a "Meta-Cognition Engine" monitors agent performance in real time and dynamically "hires" specialized sub-agent workers when capability gaps are detected. An LRU eviction policy manages resource constraints across the active worker pool. A "Surgical History Pruning" mechanism removes biasing historical outputs from the monolithic agent's context.

**What "context pollution" means in this paper:**
Monolithic single-agent context pollution = the agent's context window filling with tool specifications, prompts, and history across too many domains, weakening task-specific attention. This is a **single-agent, single-context problem** caused by tool overload. The fix is offloading to specialized sub-agents.

**What DACS does that Adaptive Orchestration does NOT:**
- **Different problem:** Adaptive Orchestration solves tool-overload context pollution in a monolithic agent. DACS solves inter-agent thread contamination in an orchestrator that manages N concurrent agents with independent steering needs. These are structurally different: tool overload = too much capability in one context; inter-agent contamination = too many *active task threads* competing in one context.
- **Different mechanism:** Adaptive Orchestration's solution is spawning specialist sub-agents at task time (structural, one-way delegation). DACS's solution is dynamic REGISTRY/FOCUS mode switching triggered by agents requesting steering (runtime, bidirectional). The DMoE approach routes tasks away from the orchestrator; DACS keeps the orchestrator responsive to all agents but selectively isolates one agent's full context during its steering interaction.
- **No agent-triggered context requests:** In Adaptive Orchestration, the Meta-Cognition Engine decides when to spawn workers. Workers don't ask the orchestrator for steering — the orchestrator decides what to delegate. The `SteeringRequest` mechanism is entirely absent.
- **No REGISTRY/FOCUS asymmetry:** Adaptive Orchestration has one context mode (the orchestrator's monolithic context, possibly with specialist workers spawned). There is no explicit runtime state machine with two structurally different context representations.
- **No wrong-agent contamination failure mode:** Their failure mode is hallucination from tool-overloaded context. DACS's failure mode is the orchestrator steering Agent A using Agent B's context.

**Use in DACS paper:**
Two uses: (1) Motivational cite — their "context pollution + attention decay" terminology validates that context overload impairs orchestrator decision quality even in the single-agent case. DACS addresses the structurally more complex case of N concurrent agents. (2) Contrast — dynamic agent spawning (Adaptive Orchestration) solves context pollution *before* interaction, by restructuring who handles the task. DACS solves it *during* interaction, by restructuring what the orchestrator holds in its context window at steering time.

---

## [2602.07092] Lemon Agent — Three-Tier Progressive Context Management

**Link:** https://arxiv.org/abs/2602.07092

**Why this might matter to DACS:** Mentions "three-tier progressive context management strategy" in an orchestrator-worker multi-agent system with parallel execution.

**Core mechanism:**
Lemon Agent is a production multi-agent system built on the AgentCortex framework (Planner-Executor-Memory paradigm). A hierarchical orchestrator allocates one or more workers for parallel sub-task execution. The "three-tier progressive context management strategy" is described as reducing context redundancy and increasing information density *during parallel steps*. Based on the abstract, this appears to be a compression/summarization strategy (inject full context → inject compressed context → inject summary only, depending on position in the task timeline). A self-evolving memory system extracts multi-dimensional insights from historical experiences.

**What DACS does that Lemon Agent does NOT:**
- The "three-tier" strategy is about *compressing a shared context over task time* (fidelity degradation as tasks progress), not about dynamically switching between two structurally different context modes based on which agent is being steered.
- No agent-triggered requests: workers execute and return outputs; they don't signal the orchestrator for steering attention.
- No REGISTRY/FOCUS asymmetry: the orchestrator holds a progressively compressed view of all task history, not a clean two-state mode switch between summary-of-all vs full-context-of-one.
- Wrong-agent contamination is not modelled. Lemon Agent measures GAIA task success, not steering accuracy or cross-agent contamination.

**Use in DACS paper:**
Brief cite only — as a contrast for the "compression" approach (Lemon Agent's three-tier strategy compresses context progressively; DACS mode-switches context completely, giving the orchestrator exact isolation rather than approximate compression). The name is useful in a footnote.

---

## Differentiator Table (fill in after reading all papers)

| Mechanism | Context managed how | N concurrent agents | Dynamic/asymmetric mode switch | Agent-triggered | Wrong-agent contamination modelled |
|---|---|:---:|:---:|:---:|:---:|
| AFM | Per-message fidelity tiers (Full/Compressed/Placeholder) in single-turn history | No | No | No | No |
| AOI | Three-layer memory (Raw / Task Queue / Compressed Cache) via central Context Compressor; Observer holds compressed aggregate of all agents | Yes (fixed roles) | No | No | No |
| AgentOrchestra | Hierarchical per-component context (TCP/ECP/ACP); planning agent bounds footprint via routing but accumulates across concurrent agent interactions | Yes (planner-driven) | No | No | No |
| ACE | Single-agent evolving playbook — incremental delta updates, grow-and-refine deduplication | No | No | No | No |
| AdaptOrch | Task-level DAG topology selection (parallel/sequential/hierarchical/hybrid); no per-agent isolation within orchestrator at steering time | Yes (topology batches) | No | No | No |
| CodeDelegator | Role separation — Delegator (planner) + fresh Coder per sub-task (EPSS); prevents execution trace accumulation in Delegator context | No (sequential) | No | No | No |
| SideQuest | Model-driven KV cache compression as auxiliary task; single-agent retrieval bloat reduction | No | No | No | No |
| Adaptive Orchestration | DMoE spawning of specialist sub-agents to offload tool-overloaded monolithic context; LRU eviction + Surgical History Pruning | No (sequential delegation) | No — structural spawning, not runtime mode switch | No — Meta-Cognition Engine decides, not agents | No |
| Lemon Agent | Three-tier progressive context compression (full → compressed → summary) over task timeline in orchestrator-worker system | Yes (parallel workers) | No — compression degrades uniformly over time, no mode switch | No — orchestrator-driven dispatch | No |
| **DACS** | **Isolated per-agent focus session** — REGISTRY mode (registry summaries only) ↔ FOCUS(aᵢ) mode (full F(aᵢ) + compressed registry) | **Yes (N concurrent)** | **Yes — asymmetric dual modes** | **Yes — SteeringRequest** | **Yes — primary metric** |
