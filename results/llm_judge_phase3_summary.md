# LLM-as-Judge Validation — Phase 3 (Decision Density)

## Setup

| Scenario | Coverage | Decisions judged | Sample strategy |
|---|---|---|---|
| s8_n3_dense_d3 | Stratified n=100 | 100 | 50 DACS + 50 baseline, seed=42 |
| s7_n5_dense_d2 | Stratified n=200 | 200 | 100 DACS + 100 baseline, seed=42 |

Judge model: `MiniMax-M2.7` (independent call, no keyword list given)

## Overall Agreement by Scenario

| Scenario | n | Agreement | κ | Keyword acc | Judge acc | FN | FP |
|---|---|---|---|---|---|---|---|
| s8_n3_dense_d3 | 100 | **95.0%** (±2.2%) | **0.886** | 69.0% | 66.0% | 1 | 4 |
| s7_n5_dense_d2 | 200 | **97.0%** (±1.2%) | **0.933** | 67.0% | 66.0% | 2 | 4 |

## Per-Condition Breakdown

| Scenario | Condition | n | Keyword acc | Judge acc | Agreement |
|---|---|---|---|---|---|
| s8_n3_dense_d3 | dacs | 50 | 98.0% | 96.0% | 94.0% |
| s8_n3_dense_d3 | baseline | 50 | 40.0% | 36.0% | 96.0% |
| s7_n5_dense_d2 | dacs | 100 | 96.0% | 96.0% | 98.0% |
| s7_n5_dense_d2 | baseline | 100 | 38.0% | 36.0% | 96.0% |

## Phase 3 Combined

| Total decisions judged | 300 |
|---|---|
| Overall agreement | 96.3% |
| Mean Cohen's κ | 0.909 |

## Interpretation

Mean κ=0.909 (substantial/near-perfect agreement). Keyword matching is a valid proxy for LLM-judged correctness across Phase 3 scenarios. Phase 3 accuracy estimates inherit the same validity established by the Phase 2 judge validation (κ=0.956).

*Full results: `results/llm_judge_phase3_s8.csv`, `results/llm_judge_phase3_s7.csv`*
