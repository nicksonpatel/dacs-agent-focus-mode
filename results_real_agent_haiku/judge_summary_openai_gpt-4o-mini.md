# Real-Agent Validation — Judge Summary

## Steering Accuracy (M1_real)

| Condition | N | Keyword Acc | Judge Acc | Cohen's κ |
|-----------|---|-------------|-----------|-----------|
| baseline | 368 | 84.8% | 64.9% | 0.389 |
| dacs | 384 | 99.2% | 88.5% | -0.015 |

## Per-Rubric Accuracy (Judge)

| Condition | Agent | Rubric | N | Judge Acc |
|-----------|-------|--------|---|-----------|
| baseline | a1 | duplicate_handling | 29 | 72.4% |
| baseline | a1 | implementation_style | 30 | 96.7% |
| baseline | a1 | traversal_order | 29 | 100.0% |
| baseline | a2 | citation_depth | 25 | 32.0% |
| baseline | a2 | primary_source | 29 | 100.0% |
| baseline | a2 | sparse_attention_variants | 30 | 36.7% |
| baseline | a3 | encoding_strategy | 29 | 89.7% |
| baseline | a3 | null_imputation | 28 | 57.1% |
| baseline | a3 | outlier_threshold | 23 | 21.7% |
| baseline | a4 | cycle_detection | 20 | 50.0% |
| baseline | a4 | graph_representation | 19 | 94.7% |
| baseline | a4 | visited_tracking | 20 | 60.0% |
| baseline | a5 | canonical_algorithms | 20 | 20.0% |
| baseline | a5 | evaluation_benchmarks | 18 | 55.6% |
| baseline | a5 | variance_reduction | 19 | 57.9% |
| dacs | a1 | duplicate_handling | 30 | 50.0% |
| dacs | a1 | implementation_style | 30 | 96.7% |
| dacs | a1 | traversal_order | 30 | 100.0% |
| dacs | a2 | citation_depth | 27 | 88.9% |
| dacs | a2 | primary_source | 30 | 100.0% |
| dacs | a2 | sparse_attention_variants | 30 | 86.7% |
| dacs | a3 | encoding_strategy | 30 | 90.0% |
| dacs | a3 | null_imputation | 30 | 83.3% |
| dacs | a3 | outlier_threshold | 27 | 81.5% |
| dacs | a4 | cycle_detection | 20 | 100.0% |
| dacs | a4 | graph_representation | 20 | 100.0% |
| dacs | a4 | visited_tracking | 20 | 100.0% |
| dacs | a5 | canonical_algorithms | 20 | 100.0% |
| dacs | a5 | evaluation_benchmarks | 20 | 95.0% |
| dacs | a5 | variance_reduction | 20 | 65.0% |

## Steering Coverage

Coverage = fraction of rubric slots that received a steering response.

Scenario ra1_n3: mean coverage 208.9%, min 66.7%, max 333.3%

Scenario ra2_n5: mean coverage 125.3%, min 40.0%, max 200.0%

