# Phase 3 Results — Decision Density Scaling

**Date completed:** April 4, 2026  
**Trials:** 40 (2 scenarios × 10 DACS + 10 baseline each)  
**Research question:** RQ4 — Does the DACS advantage scale with decision density (number of steering requests per agent), not just with agent count?

---

## Summary Table

| Scenario | Condition | Trials | Accuracy | ±SE | Contamination | ±SE | Avg ctx (tok) | ±SE |
|---|---|---|---|---|---|---|---|---|
| s7 (N=5, D=8) | DACS | 10 | **94.0%** | 0.8% | 0.2% | 0.2% | 1,654 | 20 |
| s7 (N=5, D=8) | Baseline | 10 | 34.8% | 1.8% | 49.8% | 3.3% | 5,364 | 98 |
| s8 (N=3, D=15) | DACS | 10 | **98.4%** | 0.5% | 0.9% | 0.9% | 2,755 | 29 |
| s8 (N=3, D=15) | Baseline | 10 | 44.2% | 1.3% | 51.6% | 2.4% | 6,573 | 153 |

*(D = steering decisions per agent per trial)*

---

## Effect Sizes and Statistical Significance

| Scenario | Δ Accuracy | Welch's t | p-value | Context ratio |
|---|---|---|---|---|
| s7 — N=5, dense D=8 | **+59.2 pp** | t = 30.24 | p = 2.1 × 10⁻¹³ | 3.24× |
| s8 — N=3, dense D=15 | **+54.2 pp** | t = 40.30 | p = 9.7 × 10⁻¹⁴ | 2.39× |

Both p-values are well below 0.001. The advantage is consistent and large across both density regimes.

---

## Decision Density Scaling: Cross-Phase Comparison

The central hypothesis for Phase 3 is: as D (steering decisions per agent) grows, the flat-context baseline accumulates more per-agent history in a shared window, causing increasing accuracy degradation. DACS should be immune, since each focus session loads only the target agent's context.

### N=5 trajectory (fixed agent count, increasing D)

| Phase | Scenario | D (decisions/agent) | DACS acc | Baseline acc | Δ acc | Ctx ratio |
|---|---|---|---|---|---|---|
| Phase 1 | s2_n5 | ~3 | 96.7% | 38.7% | +58.0 pp | 2.72× |
| Phase 2 | s5_n5_crossfire | 4 | 96.0% | 37.0% | +59.0 pp | 2.90× |
| Phase 3 | s7_n5_dense_d2 | **8** | 94.0% | 34.8% | +59.2 pp | **3.24×** |

At N=5, as D grows from ~3 to 8, baseline accuracy continues to erode (38.7% → 34.8%) while DACS accuracy stays near-constant (96.7% → 94.0%). The accuracy delta remains stable at ≈+59 pp, but the **context efficiency ratio grows from 2.72× to 3.24×** — DACS becomes increasingly token-efficient relative to the baseline as decision history accumulates.

### N=3 trajectory (fixed agent count, increasing D)

| Phase | Scenario | D (decisions/agent) | DACS acc | Baseline acc | Δ acc | Ctx ratio |
|---|---|---|---|---|---|---|
| Phase 1 | s1_n3 | ~3 | 96.7% | 60.0% | +36.7 pp | 2.12× |
| Phase 2 | s4_n3_homogeneous | 4 | 90.2% | 52.5% | +37.7 pp | 2.29× |
| Phase 3 | s8_n3_dense_d3 | **15** | 98.4% | 44.2% | **+54.2 pp** | 2.39× |

The N=3 trajectory is the more dramatic result. When D scales from ~3 to 15 (5× increase), the baseline accuracy falls from 60.0% to 44.2% while DACS actually *improves* (~96.7% → 98.4%). The delta jumps by +17.5 pp (+36.7 → +54.2). This directly confirms RQ4: **higher decision density is a compounding disadvantage for the flat-context baseline.** At D=15, each agent's full 15-decision history is simultaneously present in the flat context — the LLM must respond to one agent's current decision point while surrounded by all three agents' complete prior interaction histories.

---

## Scenario Descriptions

### s7 — High-density N=5 (`s7_n5_dense_d2`)

- **Agents:** 5 GenericAgent instances — async web scraper refactor, federated learning literature review, fraud detection feature pipeline, flaky test debugger, distributed cache TDD
- **N agents:** 5
- **Decisions per agent:** 8 (40 total per trial)
- **Design purpose:** Tests whether DACS maintains its N=5 advantage when D doubles from the Phase 1/2 baseline. Agents span five maximally distinct domains to keep contamination unambiguous.

**Finding:** DACS holds at 94.0% accuracy with near-zero contamination (0.2%). The baseline's 34.8% accuracy is slightly worse than Phase 1 N=5 (38.7%), confirming that additional decision depth marginally degrades baseline performance. The 3.24× context ratio — the highest recorded for N=5 — reflects that with 8 decisions per agent, the flat window accumulates 5 × 8 = 40 decision interactions, while DACS only loads each agent's 8-decision focus context at steering time.

### s8 — Ultra-high-density N=3 (`s8_n3_dense_d3`)

- **Agents:** 3 GenericAgent instances — BERT legal text classifier training loop (15 DPs), clinical trial hypothesis testing (15 DPs), post-quantum cryptography whitepaper (15 DPs)
- **N agents:** 3
- **Decisions per agent:** 15 (45 total per trial)
- **Design purpose:** Stress-test for D scaling: same N=3 agent count as Phase 1 s1 and Phase 2 s4, but 5× more decisions per agent. Tests whether DACS degrades at very high D.

**Finding:** DACS achieves its highest accuracy in any Phase 3/2/1 experiment: **98.4% ± 0.5%**. All 10 DACS trials scored ≥95.6%, 7/10 scored ≥97.8%. At D=15, DACS appears to *benefit* from richer per-agent context in the focus session — the full 15-decision history is precisely the signal the orchestrator needs to make accurate steering decisions. The baseline, by contrast, must navigate 45 combined interactions and achieves only 44.2% — significantly worse than the Phase 1 N=3 baseline of 60.0% at D≈3. Contamination is 51.6%, similar to other high-N/high-D baselines.

---

## Raw Trial Accuracy

### s7_n5_dense_d2

| Trial | DACS acc | Baseline acc |
|---|---|---|
| t01 | 95.0% | 35.0% |
| t02 | 90.0% | 30.0% |
| t03 | 95.0% | 35.0% |
| t04 | 92.5% | 35.0% |
| t05 | 95.0% | 37.5% |
| t06 | 90.0% | 25.0% |
| t07 | 97.5% | 40.0% |
| t08 | 97.5% | 35.0% |
| t09 | 95.0% | 45.0% |
| t10 | 92.5% | 30.0% |
| **Mean** | **94.0% ± 0.8% SE** | **34.8% ± 1.8% SE** |

### s8_n3_dense_d3

| Trial | DACS acc | Baseline acc |
|---|---|---|
| t01 | 100.0% | 44.4% |
| t02 | 97.8% | 42.2% |
| t03 | 97.8% | 44.4% |
| t04 | 95.6% | 48.9% |
| t05 | 100.0% | 46.7% |
| t06 | 97.8% | 42.2% |
| t07 | 100.0% | 48.9% |
| t08 | 97.8% | 46.7% |
| t09 | 97.8% | 42.2% |
| t10 | 100.0% | 35.6% |
| **Mean** | **98.4% ± 0.5% SE** | **44.2% ± 1.3% SE** |

---

## Key Findings for the Paper

### RQ4 answer: Yes — decision density amplifies the DACS advantage

1. **Baseline accuracy degrades with D.** At N=3, baseline accuracy falls from 60.0% (D≈3) to 52.5% (D=4) to 44.2% (D=15) — a 15.8 pp decline as D quintuples. At N=5, it falls from 38.7% (D≈3) to 34.8% (D=8). Each additional decision interaction from any agent in the flat context competes for the orchestrator's attention.

2. **DACS accuracy is stable across D.** DACS ranges from 90.0%–98.4% across all experiments (Phases 1–3) regardless of D. The focus session mechanism ensures the orchestrator sees only the requesting agent's full history; additional history depth is additive signal, not noise.

3. **Context efficiency ratio grows with D.** The baseline/DACS context token ratio grows from 2.12× (N=3, D≈3) to 2.39× (N=3, D=15), and from 2.72× (N=5, D≈3) to 3.24× (N=5, D=8). As D increases, the flat window must hold all agents' accumulated histories proportionally, while DACS context grows only for the assigned agent.

4. **DACS contamination stays near-zero at high D.** Despite 15 decision interactions per agent, DACS contamination in s8 is 0.9% ± 0.9% — essentially zero. The focus-session isolation mechanism does not degrade under long per-agent histories.

5. **Implication for long-running agentic tasks.** In production multi-agent systems where each agent executes dozens or hundreds of tasks before completing, decision density is a primary context-pollution vector. Phase 3 demonstrates that DACS scales favorably in this regime, making it practically relevant beyond laboratory-scale agent counts.

---

## Cumulative Results: All Three Phases

| Phase | Scenario | N | D | DACS | Baseline | Δ | Ctx ratio |
|---|---|---|---|---|---|---|---|
| 1 | s1_n3 | 3 | ~3 | 96.7% | 60.0% | +36.7 pp | 2.12× |
| 1 | s2_n5 | 5 | ~3 | 96.7% | 38.7% | +58.0 pp | 2.72× |
| 1 | s3_n10 | 10 | ~3 | 90.0% | 21.0% | +69.0 pp | 3.53× |
| 2 | s4_n3_homogeneous | 3 | 4 | 90.2% | 52.5% | +37.7 pp | 2.29× |
| 2 | s5_n5_crossfire | 5 | 4 | 96.0% | 37.0% | +59.0 pp | 2.90× |
| 2 | s6_n5_cascade (adversarial) | 5 | 3 | 94.0% | 56.7% | +37.3 pp | 2.65× |
| **3** | **s7_n5_dense_d2** | **5** | **8** | **94.0%** | **34.8%** | **+59.2 pp** | **3.24×** |
| **3** | **s8_n3_dense_d3** | **3** | **15** | **98.4%** | **44.2%** | **+54.2 pp** | **2.39×** |

Across all 8 experimental scenarios, DACS accuracy ranges 90.0%–98.4%. Baseline accuracy ranges 21.0%–60.0%. The minimum DACS advantage is +36.7 pp (N=3, low D, homogeneous agents) and the maximum is +69.0 pp (N=10, canonical Phase 1). DACS has never lost to the baseline in any scenario.
