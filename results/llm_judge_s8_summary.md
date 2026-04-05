# LLM-as-Judge Validation — s8_n3_dense_d3

## Setup

| | |
|---|---|
| Scenario | s8_n3_dense_d3 (N=3 agents, D=15 decisions each) |
| Total decisions available | 900 (20 trials × 45 decisions each) |
| Sample size | 40 (random sample, seed=42) |
| Judge model | MiniMax-M2.7 (same model, independent call with structured prompt) |
| Keyword metric model | MiniMax-M2.7 |

## Overall Agreement

| Metric | Value |
|---|---|
| Agreement rate | **30.0%** (±7.2%) |
| Cohen's κ | **-0.000** |
| Keyword accuracy | 70.0% |
| Judge accuracy | 0.0% |
| False negatives (kw=0, judge=CORRECT) | 0 (0.0%) |
| False positives (kw=1, judge=INCORRECT) | 28 (70.0%) |

## Per-Condition Breakdown

| Condition | n | Keyword acc | Judge acc | Agreement |
|---|---|---|---|---|
| dacs | 19 | 94.7% | 0.0% | 5.3% |
| baseline | 21 | 47.6% | 0.0% | 52.4% |

## Interpretation

κ=-0.000 — see disagreement analysis below.

*Full per-decision results: `results/llm_judge_s8.csv`*
