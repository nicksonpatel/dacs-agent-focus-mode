# ContextBuilder

Token-counted context assembly with a hard budget cap.

::: dacs._context_builder.ContextBuilder
    options:
      show_source: true
      members:
        - __init__
        - build_focus_context
        - build_registry_context
        - build_flat_context
        - count_tokens

## ContextBudgetError

::: dacs._context_builder.ContextBudgetError

## Token counting

DACS uses `tiktoken` with the `cl100k_base` encoding (matches GPT-4o-mini and is a good approximation for Claude models).  Token counts are computed **before** every LLM call — the provider is never relied upon for truncation.

## Budget enforcement

```
F(aᵢ) fits ? ─No──► ContextBudgetError (fail fast)
     │
    Yes
     │
     ▼
F(aᵢ) + compressed R_{-i} fits ? ─No──► compress R further
                                          until it fits or error
     │
    Yes
     │
     ▼
Build final prompt
```

Compression priority (from least to most aggressive):
1. Drop steering history from compressed entries
2. Shorten summaries to 50 tokens
3. Drop summaries entirely (status + task only)
4. Raise `ContextBudgetError`
