# Tutorial 4: Custom API Endpoints

DACS uses the Anthropic client, which supports any Anthropic-compatible endpoint.  This tutorial covers common configurations.

## Anthropic (default)

```python
from dacs import DACSRuntime

# Uses ANTHROPIC_API_KEY environment variable
runtime = DACSRuntime(model="claude-3-haiku-20240307")

# Or pass the key directly
runtime = DACSRuntime(model="claude-3-haiku-20240307", api_key="sk-ant-...")
```

## OpenRouter

OpenRouter provides access to many Anthropic models (and others) via a compatible API:

```python
runtime = DACSRuntime(
    model="anthropic/claude-3-haiku",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-...",
)
```

Available models: https://openrouter.ai/models?q=anthropic

## MiniMax M2.7

The model used in all DACS benchmark experiments:

```python
import os

runtime = DACSRuntime(
    model="MiniMax-M2.7",
    base_url="https://api.minimax.io/anthropic",
    api_key=os.environ["MINIMAX_API_KEY"],
)
```

## Azure AI Foundry (Claude via Azure)

```python
runtime = DACSRuntime(
    model="claude-3-haiku-20240307",
    base_url="https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT",
    api_key=os.environ["AZURE_OPENAI_KEY"],
)
```

## Vertex AI (via Anthropic SDK)

For Vertex AI, use the `anthropic[vertex]` client directly and pass it to the low-level `Orchestrator`:

```python
import anthropic
from dacs import (
    DACSRuntime, Logger, RegistryManager, ContextBuilder,
    SteeringRequestQueue, Orchestrator
)

vertex_client = anthropic.AnthropicVertex(
    region="us-east5",
    project_id="my-gcp-project",
)

# Use the low-level API to inject the custom client
logger = Logger("run.jsonl")
registry = RegistryManager(logger)
queue = SteeringRequestQueue(logger)
cb = ContextBuilder(200_000, logger)
registry.set_context_builder(cb)

orchestrator = Orchestrator(
    registry=registry,
    queue=queue,
    context_builder=cb,
    llm_client=vertex_client,
    model="claude-3-5-sonnet@20240620",
    token_budget=200_000,
    logger=logger,
)
```

## Adjusting token budget

Different models have different context windows:

```python
# Claude Haiku - 200k context
runtime = DACSRuntime(model="claude-3-haiku-20240307", token_budget=200_000)

# Smaller models with 32k context
runtime = DACSRuntime(model="some-model", token_budget=32_000)
```

!!! tip
    Set `token_budget` to the actual context window of your model.  DACS will
    enforce this deterministically via `tiktoken` (cl100k_base encoding), raising
    `ContextBudgetError` if a focus context would exceed it.

## Using environment variables

Create a `.env` file and load it with `python-dotenv`:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
DACS_MODEL=claude-3-haiku-20240307
DACS_TOKEN_BUDGET=200000
```

```python
from dotenv import load_dotenv
import os

load_dotenv()

runtime = DACSRuntime(
    model=os.environ.get("DACS_MODEL", "claude-3-haiku-20240307"),
    token_budget=int(os.environ.get("DACS_TOKEN_BUDGET", 200_000)),
)
```
