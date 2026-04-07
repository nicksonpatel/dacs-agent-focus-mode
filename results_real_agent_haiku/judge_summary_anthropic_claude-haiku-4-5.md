# Real-Agent Validation — Judge Summary

## Steering Accuracy (M1_real)

| Condition | N | Keyword Acc | Judge Acc | Cohen's κ |
|-----------|---|-------------|-----------|-----------|
| baseline | 344 | 84.3% | 59.0% | 0.410 |
| dacs | 384 | 99.2% | 77.1% | 0.052 |

## Per-Rubric Accuracy (Judge)

| Condition | Agent | Rubric | N | Judge Acc |
|-----------|-------|--------|---|-----------|
| baseline | a1 | duplicate_handling | 28 | 64.3% |
| baseline | a1 | implementation_style | 28 | 92.9% |
| baseline | a1 | traversal_order | 27 | 100.0% |
| baseline | a2 | citation_depth | 23 | 8.7% |
| baseline | a2 | primary_source | 28 | 100.0% |
| baseline | a2 | sparse_attention_variants | 28 | 50.0% |
| baseline | a3 | encoding_strategy | 27 | 88.9% |
| baseline | a3 | null_imputation | 27 | 25.9% |
| baseline | a3 | outlier_threshold | 23 | 17.4% |
| baseline | a4 | cycle_detection | 18 | 50.0% |
| baseline | a4 | graph_representation | 17 | 100.0% |
| baseline | a4 | visited_tracking | 18 | 55.6% |
| baseline | a5 | canonical_algorithms | 18 | 16.7% |
| baseline | a5 | evaluation_benchmarks | 17 | 23.5% |
| baseline | a5 | variance_reduction | 17 | 58.8% |
| dacs | a1 | duplicate_handling | 30 | 50.0% |
| dacs | a1 | implementation_style | 30 | 76.7% |
| dacs | a1 | traversal_order | 30 | 100.0% |
| dacs | a2 | citation_depth | 27 | 55.6% |
| dacs | a2 | primary_source | 30 | 90.0% |
| dacs | a2 | sparse_attention_variants | 30 | 100.0% |
| dacs | a3 | encoding_strategy | 30 | 80.0% |
| dacs | a3 | null_imputation | 30 | 43.3% |
| dacs | a3 | outlier_threshold | 27 | 44.4% |
| dacs | a4 | cycle_detection | 20 | 100.0% |
| dacs | a4 | graph_representation | 20 | 100.0% |
| dacs | a4 | visited_tracking | 20 | 100.0% |
| dacs | a5 | canonical_algorithms | 20 | 100.0% |
| dacs | a5 | evaluation_benchmarks | 20 | 85.0% |
| dacs | a5 | variance_reduction | 20 | 50.0% |

## Steering Coverage

Coverage = fraction of rubric slots that received a steering response.

Scenario ra1_n3: mean coverage 202.2%, min 66.7%, max 333.3%

Scenario ra2_n5: mean coverage 121.3%, min 40.0%, max 200.0%

