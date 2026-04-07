# DACS Real-Agent Validation Experiment — Architecture

**Addresses reviewer criticism:** *"Synthetic agent harness — real agents don't emit SteeringRequest objects in neat structured formats."*

**Last updated:** April 6, 2026

---

## Overview

The real-agent validation experiment replaces scripted stub agents with `LLMAgent` — an agent that calls the model at every step, generates its own output, and **autonomously decides when it needs orchestrator guidance** by emitting a `[[STEER: ...]]` marker in its response text.

Everything else is identical to the original harness: the same `Orchestrator`, `RegistryManager`, `ContextBuilder`, `SteeringRequestQueue`, and DACS/baseline switching via `focus_mode`. This ensures the only experimental variable is the source of the steering question (hardcoded template vs real LLM output) — not the orchestrator mechanics.

---

## New Component: `LLMAgent`

```
agents/llm_agent.py
```

`LLMAgent` extends `BaseAgent` and replaces the step-list iteration in `GenericAgent` with a real conversation loop.

```mermaid
flowchart TD
    INIT["LLMAgent.__init__()\n────────────────────────\nBuild system prompt:\n• task_description\n• decision_hints (topic types, no answers)\n• [[STEER:]] + [[DONE]] protocol instructions\n• max_steering_requests cap"]

    LOOP["LLMAgent._execute() loop\n(max_steps = 12)"]

    LLM_CALL["LLM call\nclient.messages.create()\n• model = DACS_MODEL (MiniMax-M2.7)\n• system = self._system_prompt\n• messages = conversation history\n• max_tokens = 800"]

    PARSE["Parse response\n(ThinkingBlock-safe: find first block with .text)"]

    HEARTBEAT["_push_update(RUNNING, last 80 chars, LOW)\n→ RegistryManager.update()"]

    CHECK_DONE{"[[DONE]] in text?"}
    CHECK_STEER{"[[STEER: ...]] in text\nand steering_count < max_steer?"}

    STEER["Extract question\n(regex, max 400 chars)\nsteering_count += 1\n\n_request_steering(\n  relevant_context = _recent_output(k=5),\n  question = extracted,\n  urgency = MEDIUM\n)"]

    WAIT["Await orchestrator\nSteeringResponse\n(blocks asyncio task)"]

    INJECT["Append to conversation:\n{role: user,\n content: 'Orchestrator guidance: ...'}\nContinue loop"]

    DONE["Agent COMPLETE\n(base class run() calls\n_push_update COMPLETE)"]

    YIELD["await asyncio.sleep(0)\n(yield to event loop)"]

    INIT --> LOOP
    LOOP --> LLM_CALL --> PARSE --> HEARTBEAT
    HEARTBEAT --> CHECK_DONE
    CHECK_DONE -- yes --> DONE
    CHECK_DONE -- no --> CHECK_STEER
    CHECK_STEER -- yes --> STEER --> WAIT --> INJECT --> LOOP
    CHECK_STEER -- no --> YIELD --> LOOP
```

### Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Steering trigger | `[[STEER: ...]]` regex marker in response text | Robust to partial outputs; no structured JSON required — mirrors real agent unpredictability |
| `decision_hints` in system prompt | Names topic *types*, never correct answers | Forces the LLM to reason to its own question; prevents prompt-steering the answer |
| `max_steering_requests = 3` | Matches number of `DecisionRubric` entries per agent | Prevents runaway steering; gives rubric assignment sequence parity with synthetic agents |
| Conversation format | Append assistant turn, then inject guidance as user turn | Standard Anthropic multi-turn; orchestrator guidance lands in the correct context slot |
| `relevant_context` | `_recent_output(k=5)` — last 5 heartbeat summaries | Gives orchestrator FOCUS context about agent progress without flooding the focus window |
| ThinkingBlock handling | `next(block.text for block in resp.content if hasattr(block, "text"), "")` | MiniMax-M2.7 prepends reasoning blocks before the text response |

---

## Scenario: `ra1_n3`

Mirrors synthetic scenario `s1_n3` for direct comparison. Same domains, same expected decision topics, different evaluation ground truth format.

```
experiments_real_agent/scenario_defs.py
```

```mermaid
graph LR
    subgraph ra1_n3["Scenario ra1_n3  (N=3 agents)"]
        A1["a1 — BST implementation\n────────────────────────\nRubrics:\n• traversal_order\n• duplicate_handling\n• implementation_style"]
        A2["a2 — Transformer survey\n────────────────────────\nRubrics:\n• primary_source\n• sparse_attention_variants\n• citation_depth"]
        A3["a3 — CSV cleaning pipeline\n────────────────────────\nRubrics:\n• encoding_strategy\n• null_imputation\n• outlier_threshold"]
    end
```

### `DecisionRubric` structure

Each agent has 3 `DecisionRubric` entries. Unlike `DecisionPoint` in the synthetic harness (which uses `question_fragment` to label the expected question), `DecisionRubric` is used only by the offline judge — the agent generates the question freely.

```python
@dataclass
class DecisionRubric:
    topic:            str        # label for analysis tables (e.g. "traversal_order")
    correct_keywords: list[str]  # keyword scorer (fallback M1 baseline)
    judge_context:    str        # rubric paragraph for LLM judge prompt
```

### Synthetic vs real-agent scenario comparison

| Property | `s1_n3` (synthetic) | `ra1_n3` (real agent) |
|---|---|---|
| Agent type | `GenericAgent` (scripted steps) | `LLMAgent` (LLM-driven loop) |
| Question source | Hardcoded `question` field in step dict | LLM generates freely via `[[STEER: ...]]` |
| Ground truth binding | `DecisionPoint.question_fragment` labels which question is which | Sequential assignment: 1st response per agent → `rubric[0]` |
| Keyword evaluation | `DecisionPoint.answer_keywords` — can be pre-tuned | `DecisionRubric.correct_keywords` — informed by domain knowledge |
| M1 primary method | Keyword substring match | LLM judge (keyword match used for κ validation only) |
| Steering count | Fixed (exactly one per decision point) | Variable (0 to `max_steering_requests`); measured as *coverage* |

---

## Experiment Runner

```
experiments_real_agent/run.py
```

Wiring is identical to `experiments/run_experiment.py`. The only structural difference is `LLMAgent` takes extra constructor params (`client`, `model`, `decision_hints`).

```mermaid
flowchart TD
    CLI["CLI: python -m experiments_real_agent.run\n--mode dacs|baseline|both\n--trials N\n--scenario ra1_n3\n--model MiniMax-M2.7"]

    SCENARIO["Load RealAgentScenario\nfrom scenario_defs.REAL_SCENARIOS"]

    WIRE["Per-trial wiring\n────────────────────────\nLogger(log_path)\nRegistryManager(logger)\nSteeringRequestQueue(logger)\nContextBuilder(token_budget, logger)\nAsyncAnthropic(api_key, base_url)"]

    ORC["Orchestrator(\n  registry, queue, cb,\n  llm_client=client,\n  model=model,\n  focus_mode=focus_mode,   ← ONLY SWITCH\n  logger\n)"]

    AGENTS["N × LLMAgent(\n  agent_id, task_description,\n  decision_hints,\n  client=client,   ← shared with orchestrator\n  model=model\n)"]

    RUN["asyncio.gather(\n  agent_tasks + orch_task\n)"]

    METRICS["_compute_real_metrics(log_path, scenario)\n────────────────────────\nM2: contamination_rate\nM3: avg_context_tokens\nM3: p95_context_tokens\nn_steering_responses"]

    CSV["results_real_agent/summary_real.csv"]

    CLI --> SCENARIO --> WIRE --> ORC & AGENTS --> RUN --> METRICS --> CSV
```

> **M1 (steering accuracy) is intentionally absent from `run.py` output.** It requires the LLM judge to evaluate each actual question against its rubric. Run `experiments_real_agent/judge.py` after collecting all trial logs.

---

## LLM Judge

```
experiments_real_agent/judge.py
```

Unlike the Phase 1–3 judges (which use `question_fragment` — a known substring of the scripted question), the real-agent judge must handle free-form questions. It:

1. Reads `STEERING_REQUEST` events to get the actual agent-generated question (via the `question` field added to the log event)
2. Pairs each request with its response via `request_id`
3. Assigns responses to rubrics by **sequential order per agent** (same assumption as `metrics.py`)
4. Passes the actual question + orchestrator response + rubric context to the judge LLM

```mermaid
flowchart TD
    LOGS["results_real_agent/*.jsonl"]

    PARSE["collect_decisions(scenario_id)\n────────────────────────\nFor each JSONL file:\n  Build request_map: {request_id → question}\n  Walk STEERING_RESPONSE events\n  Match to DecisionRubric by sequence order\n  Compute keyword_score baseline"]

    JUDGE_LOOP["For each decision not yet in CSV:\n────────────────────────\njudge_prompt = [\n  Agent task domain,\n  Agent's actual question,\n  Orchestrator response[:1200],\n  Rubric context\n]\nclient.messages.create(max_tokens=1024)"]

    PARSE_VERDICT["_parse_verdict(raw)\n────────────────────────\nRegex: <verdict>CORRECT|INCORRECT</verdict>\nFallback: plain-text scan"]

    CSV_OUT["results_real_agent/judge_results.csv\n────────────────────────\nrun_id, condition, agent_id,\nrubric_index, rubric_topic,\nactual_question, keyword_score,\njudge_verdict, judge_reason,\norchestrator_state, response_text"]

    SUMMARY["results_real_agent/judge_summary.md\n────────────────────────\nM1_real accuracy table (DACS vs baseline)\nPer-rubric accuracy\nSteering coverage\nCohen's κ (judge vs keyword)"]

    LOGS --> PARSE --> JUDGE_LOOP --> PARSE_VERDICT --> CSV_OUT --> SUMMARY
```

### Judge prompt structure

```
Agent task domain:        <task_description[:80]>

Agent's question to orchestrator:
<actual_question>         ← free-form LLM-generated text

Orchestrator response:
<response_text[:1200]>

Rubric (what a correct answer looks like):
<judge_context>           ← plain-text explanation from DecisionRubric

Is this response CORRECT or INCORRECT?
```

Verdict format: `<reason>one sentence</reason><verdict>CORRECT</verdict>`

---

## Analysis

```
experiments_real_agent/analyze.py
```

Produces the 4-column comparison table for the paper's *Real-Agent Validation* section.

```mermaid
flowchart LR
    A["results_real_agent/summary_real.csv\n(M2, M3 per trial)"]
    B["results_real_agent/judge_results.csv\n(M1_real per decision)"]
    C["results/summary.csv\n(synthetic s1_n3: M1, M2, M3)"]

    ANALYSIS["analyze.py\n────────────────────────\nGroup judge accuracy by (condition, run_id)\nCompute per-trial M1_real\nLoad M2/M3 from summary_real.csv\nLoad synthetic s1_n3 from summary.csv\nWelch's t on M1_real DACS vs baseline\nSteering coverage per condition"]

    TABLE["Console: 4-column table\n────────────────────────\nDACH (real) | Baseline (real)\n| DACS (s1_n3 syn) | Base (s1_n3 syn)"]

    A & B & C --> ANALYSIS --> TABLE
```

### Steering coverage metric

Unique to the real-agent harness. Synthetic agents always hit exactly their `decision_points` count; real agents may hit zero to `max_steering_requests`.

```
coverage(trial) = steering_responses_matched_to_rubric / expected_rubric_count
expected = sum(len(agent.rubrics) for agent in scenario.agents)  # 9 for ra1_n3
```

Coverage < 100% means an agent resolved some decisions autonomously without asking. This is valid real-world behavior; it's reported separately rather than penalising M1.

---

## Data Flow — From Agent LLM Call to Log Entry

The critical new path (compared to synthetic harness) is the agent LLM call → marker extraction → question logging:

```
LLMAgent._execute()
  │
  │  await client.messages.create(...)
  │    └── ThinkingBlock (skipped), TextBlock (extracted)
  │
  │  text = "...working on BST...should I use [[STEER: For a BST, should I handle duplicates...]]"
  │
  │  _STEER_RE.search(text)  → match.group(1) = "For a BST, should I handle duplicates..."
  │
  │  _push_update(RUNNING, last_80_chars, LOW)
  │    └── RegistryManager.update()  → log: REGISTRY_UPDATE
  │
  │  _request_steering(relevant_context, question, MEDIUM)
  │    ├── SteeringRequest(agent_id="a1", question="For a BST ...", ...)
  │    ├── _push_update(WAITING_STEERING, ...)  → log: REGISTRY_UPDATE
  │    └── queue.enqueue(request)
  │          └── log: STEERING_REQUEST  { request_id, agent_id, urgency, question }
  │                                                                        ↑
  │                                              NEW in this harness — not in synthetic
  │
  │  [blocks on self._response_queue.get()]
  │
  Orchestrator.run() (polling loop)
    └── _handle_steering(request)
          ├── [DACS]     build_focus_context(aᵢ)  → log: CONTEXT_BUILT
          │              _llm_call()               → log: LLM_CALL
          │              log: STEERING_RESPONSE { request_id, agent_id, response_text, ... }
          │              deliver_response(agent_a1)
          └── [Baseline] build_flat_context()      → log: CONTEXT_BUILT
                         _llm_call()               → log: LLM_CALL
                         log: STEERING_RESPONSE
                         deliver_response(agent_a1)

LLMAgent._execute() unblocks
  └── conversation.append({role: user, content: "Orchestrator guidance: ..."})
      Continue loop → next LLM call
```

---

## Log Event Reference (real-agent additions)

All original DACS log events apply. One field was added:

| Event | Field | New? | Notes |
|---|---|---|---|
| `STEERING_REQUEST` | `question` | **New** | Full agent-generated question text. Required by judge.py to reconstruct what the agent actually asked. Previously absent because synthetic agents' questions were known from task_suite.py. |
| All others | — | Unchanged | See `docs/architecture.md` for the full reference. |

The `question` field is populated from `SteeringRequest.question` in `SteeringRequestQueue.enqueue()` (`src/protocols.py`). The addition is backwards-compatible — existing synthetic logs simply lack the field and the original `experiments/metrics.py` never reads it.

---

## Running the Experiment

```bash
# 1. Activate env + API key
set -a && source .env && set +a

# 2. Run trials (sequential, ~4–6 min per trial on MiniMax-M2.7)
python -m experiments_real_agent.run --mode dacs     --trials 10
python -m experiments_real_agent.run --mode baseline --trials 10

# 3. Judge all collected steering responses
python -m experiments_real_agent.judge

# 4. Print comparison table
python -m experiments_real_agent.analyze
```

Results written to `results_real_agent/`:

| File | Contents |
|---|---|
| `<run_id>.jsonl` | Full event log for one trial (same format as `results/*.jsonl`) |
| `summary_real.csv` | Per-trial M2/M3 metrics |
| `judge_results.csv` | Per-decision verdicts with actual question text |
| `judge_summary.md` | Accuracy table + κ + coverage |

---

## Relationship to Original Harness

```mermaid
graph TB
    subgraph Original["Original harness (experiments/)"]
        GA["GenericAgent\nScripted steps\nFixed questions"]
        ORC1["Orchestrator\nfocus_mode=True/False"]
        TS["task_suite.py\nDecisionPoint.answer_keywords"]
        M1_KW["M1: keyword match\nvalidated by LLM judge (κ ≥ 0.88)"]
    end

    subgraph Real["Real-agent harness (experiments_real_agent/)"]
        LA["LLMAgent\nLLM conversation loop\nFree-form [[STEER:...]] questions"]
        ORC2["Orchestrator\nfocus_mode=True/False\n(identical code)"]
        SD["scenario_defs.py\nDecisionRubric.judge_context"]
        M1_JUDGE["M1_real: LLM judge primary\nkeyword match for κ only"]
    end

    GA -. "same BaseAgent\n._request_steering()" .-> LA
    ORC1 -. "identical src/orchestrator.py" .-> ORC2
    TS -. "mirrored domains\n(BST / survey / CSV)" .-> SD
    M1_KW -. "upgraded to judge-primary\nfor free-form questions" .-> M1_JUDGE
```

The real-agent harness is additive — it adds no code to `src/`\* and does not modify any existing experiment file. The only `src/` change is the `question` field in the `STEERING_REQUEST` log event, which is backwards-compatible.

> \*Exception: `agents/llm_agent.py` is a new file under `agents/` — but `agents/` is a collection of agent implementations, not a core framework component.
