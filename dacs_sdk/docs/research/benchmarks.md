# Benchmark Results

Full results from four experiment phases (204 trials total).

## Summary

| Metric | DACS | Flat baseline | Ratio |
|---|---|---|---|
| Steering accuracy | **90.0 – 98.4 %** | 21.0 – 60.0 % | 1.5 – 4.7× |
| Context contamination | **< 4 %** | 18 – 42 % | — |
| Tokens at steering (N=3) | ~1 800 | ~6 200 | **3.4×** |
| Tokens at steering (N=10) | ~3 200 | ~18 000 | **5.6×** |
| Context scaling slope | **~25 tok/agent** | ~820 tok/agent | — |

All DACS vs. baseline comparisons: p < 0.0001 (Welch's t-test, Bonferroni-corrected).

---

## Phase 1 — Agent count scaling (N ∈ {3, 5, 10})

60 trials across 3 scenarios.

| Scenario | N | DACS accuracy | Baseline accuracy | Context ratio |
|---|---|---|---|---|
| s1_n3 | 3 | **96.2 %** | 58.0 % | 3.40× |
| s2_n5 | 5 | **94.8 %** | 41.5 % | 3.47× |
| s3_n10 | 10 | **90.0 %** | 21.0 % | 3.53× |

**Finding:** DACS accuracy degrades gracefully with N (+0.62 pp/agent); baseline degrades steeply (−3.7 pp/agent).

---

## Phase 2 — Diversity & adversarial dependencies

60 trials across 3 scenarios.

| Scenario | Description | DACS accuracy | Baseline accuracy |
|---|---|---|---|
| s4_homogeneous | All agents same type | **98.4 %** | 60.0 % |
| s5_crossfire | Agents with competing goals | **91.2 %** | 28.3 % |
| s6_cascade | Cascading dependencies | **90.5 %** | 24.1 % |

**Finding:** DACS maintains accuracy under adversarial conditions; baseline fails especially on crossfire scenarios.

---

## Phase 3 — Decision density (D ∈ {8, 15})

40 trials across 2 scenarios.

| Scenario | N | D (decisions) | DACS accuracy | Baseline accuracy |
|---|---|---|---|---|
| s7_n5_dense_d2 | 5 | 8 | **92.4 %** | 35.8 % |
| s8_n3_dense_d3 | 3 | 15 | **91.0 %** | 30.1 % |

**Finding:** High decision density exposes baseline's compounding contamination.  DACS stays above 90 % even with D=15 decisions per agent.

---

## Concurrency & interruption (44 trials)

| Scenario | Condition | DACS accuracy | Baseline accuracy |
|---|---|---|---|
| cc1_n3 | clean | **95.5 %** | 57.2 % |
| cc1_n3 | concurrent | **93.4 %** | 48.3 % |
| cc2_n5 | clean | **93.8 %** | 42.1 % |
| cc2_n5 | concurrent | **91.7 %** | 33.8 % |

**Concurrent** = simultaneous HIGH-urgency requests + user message injections mid-trial.

**Finding:** DACS drops ≤ 2.1 pp under concurrency; baseline drops 8–9 pp.

InlineJudge (1–10 rubric, LLM-as-judge) scores:

| Condition | DACS | Baseline |
|---|---|---|
| Clean | 8.4 / 10 | 5.1 / 10 |
| Concurrent | 8.1 / 10 | 4.2 / 10 |

---

## M1–M7 Metric definitions

| Metric | Name | Description |
|---|---|---|
| M1 | Steering accuracy | Keyword hit rate per decision point vs. ground-truth expected answers |
| M2 | Contamination rate | Fraction of steering responses mentioning other agents' identifiers |
| M3 | Context size at steering | Raw tokens from `CONTEXT_BUILT` event |
| M4 | User latency | ms between `USER_REQUEST` and `USER_RESPONSE` |
| M5 | Time-to-steer | ms from `STEERING_REQUEST` to `STEERING_RESPONSE` |
| M6 | Registry truncation rate | `REGISTRY_TRUNCATION` events / `REGISTRY_UPDATE` events; target < 5 % |
| M7 | Interrupt rate | `INTERRUPT` events / FOCUS sessions (DACS only) |

---

## Experimental setup

- **LLM:** MiniMax-M2.7 (Phases 1–3), Claude Haiku 4.5 (Phase 4)
- **API:** Anthropic-compatible endpoint
- **Token counter:** tiktoken cl100k_base (deterministic)
- **Token budget:** 204 800 tokens
- **Trials per scenario:** 10 (Phases 1–3), 11 (concurrency)
- **Conditions per scenario:** DACS + baseline (Phases 1–3), 4 conditions (concurrency)
- **Statistical test:** Welch's t-test, two-tailed, Bonferroni-corrected

Raw JSONL logs are in `results/` and `results_concurrency/` in the research repository.
