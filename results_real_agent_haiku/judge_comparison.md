# Real-Agent Judge Comparison — Multi-Model

## Accuracy by Condition

| Judge Model | Condition | N | Keyword Acc | Judge Acc | Cohen's κ |
|-------------|-----------|---|-------------|-----------|-----------|
| anthropic/claude-haiku-4-5 | baseline | 344 | 84.3% | 59.0% | 0.410 |
| anthropic/claude-haiku-4-5 | dacs | 378 | 99.2% | 76.7% | 0.051 |
| openai/gpt-4o-mini | baseline | 368 | 84.8% | 64.9% | 0.389 |
| openai/gpt-4o-mini | dacs | 384 | 99.2% | 88.5% | -0.015 |

