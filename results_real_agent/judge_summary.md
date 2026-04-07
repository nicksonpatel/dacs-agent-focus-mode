# Real-Agent Validation — Judge Summary

## Steering Accuracy (M1_real)

| Condition | N | Keyword Acc | Judge Acc | Cohen's κ |
|-----------|---|-------------|-----------|-----------|
| baseline | 130 | 38.5% | 67.7% | 0.232 |
| dacs | 137 | 94.2% | 97.8% | 0.343 |

## Per-Rubric Accuracy (Judge)

| Condition | Agent | Rubric | N | Judge Acc |
|-----------|-------|--------|---|-----------|
| baseline | a1 | duplicate_handling | 10 | 90.0% |
| baseline | a1 | implementation_style | 7 | 85.7% |
| baseline | a1 | traversal_order | 10 | 100.0% |
| baseline | a2 | citation_depth | 7 | 42.9% |
| baseline | a2 | primary_source | 8 | 75.0% |
| baseline | a2 | sparse_attention_variants | 8 | 25.0% |
| baseline | a3 | encoding_strategy | 10 | 40.0% |
| baseline | a3 | null_imputation | 10 | 50.0% |
| baseline | a3 | outlier_threshold | 4 | 25.0% |
| baseline | a4 | cycle_detection | 10 | 60.0% |
| baseline | a4 | graph_representation | 10 | 90.0% |
| baseline | a4 | visited_tracking | 10 | 80.0% |
| baseline | a5 | canonical_algorithms | 9 | 66.7% |
| baseline | a5 | evaluation_benchmarks | 8 | 75.0% |
| baseline | a5 | variance_reduction | 9 | 77.8% |
| dacs | a1 | duplicate_handling | 9 | 100.0% |
| dacs | a1 | implementation_style | 9 | 100.0% |
| dacs | a1 | traversal_order | 9 | 100.0% |
| dacs | a2 | citation_depth | 8 | 100.0% |
| dacs | a2 | primary_source | 8 | 100.0% |
| dacs | a2 | sparse_attention_variants | 8 | 87.5% |
| dacs | a3 | encoding_strategy | 10 | 100.0% |
| dacs | a3 | null_imputation | 10 | 90.0% |
| dacs | a3 | outlier_threshold | 8 | 100.0% |
| dacs | a4 | cycle_detection | 9 | 100.0% |
| dacs | a4 | graph_representation | 10 | 100.0% |
| dacs | a4 | visited_tracking | 10 | 100.0% |
| dacs | a5 | canonical_algorithms | 10 | 90.0% |
| dacs | a5 | evaluation_benchmarks | 9 | 100.0% |
| dacs | a5 | variance_reduction | 10 | 100.0% |

## Steering Coverage

Coverage = fraction of rubric slots that received a steering response.

Scenario ra1_n3: mean coverage 148.3%, min 122.2%, max 166.7%

Scenario ra2_n5: mean coverage 89.0%, min 73.3%, max 100.0%

