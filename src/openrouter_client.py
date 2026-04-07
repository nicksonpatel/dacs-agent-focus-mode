"""Thin adapter that exposes an Anthropic Messages-API interface over OpenRouter.

The orchestrator and LLMAgent both call ``client.messages.create(...)`` using
the Anthropic messages signature.  This module provides a drop-in replacement
that internally translates those calls to OpenAI-format chat completions sent
to the OpenRouter endpoint.

A shared ``asyncio.Semaphore`` limits global concurrent in-flight calls across
all trials so the OpenRouter rate limit is never exceeded.  The semaphore is
created inside ``OpenRouterClient.__init__`` and is shared whenever you pass
the same client instance to multiple trials (the recommended pattern).

Usage
-----
    from src.openrouter_client import OpenRouterClient
    client = OpenRouterClient(api_key=os.environ["OPENROUTER_API_KEY"])
    # Use exactly like AsyncAnthropic client — messages.create() works the same.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import openai


# ---------------------------------------------------------------------------
# Response dataclasses that mimic the Anthropic SDK response shape
# ---------------------------------------------------------------------------

@dataclass
class _TextBlock:
    type: str
    text: str


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _MessagesResponse:
    content: list[_TextBlock]
    usage: _Usage


# ---------------------------------------------------------------------------
# Namespace that mimics AsyncAnthropic.messages
# ---------------------------------------------------------------------------

class _MessagesNamespace:
    def __init__(
        self,
        openai_client: openai.AsyncOpenAI,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._client    = openai_client
        self._semaphore = semaphore

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
    ) -> _MessagesResponse:
        """Translate Anthropic-style call to OpenAI Chat Completions via OpenRouter.

        Wrapped in the shared semaphore so that at most ``max_concurrent``
        calls are in-flight simultaneously across ALL concurrent trials.
        Retries on 429 / RateLimitError with exponential back-off.
        """
        openai_messages = [{"role": "system", "content": system}] + messages

        async with self._semaphore:
            for attempt in range(8):
                try:
                    resp = await self._client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=openai_messages,  # type: ignore[arg-type]
                    )
                    break
                except openai.RateLimitError:
                    wait = 5 * (2 ** attempt)
                    await asyncio.sleep(wait)
                except openai.APIStatusError as exc:
                    if exc.status_code == 429:
                        wait = 5 * (2 ** attempt)
                        await asyncio.sleep(wait)
                    else:
                        raise
            else:
                raise RuntimeError("OpenRouter rate limit retries exhausted")

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = _Usage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return _MessagesResponse(
            content=[_TextBlock(type="text", text=text)],
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Public client class
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """Drop-in replacement for AsyncAnthropic pointing at OpenRouter.

    Supports the subset of the Anthropic Messages API used by the DACS
    orchestrator and LLMAgent: ``messages.create(model, max_tokens, system,
    messages)``.

    Pass the *same* instance to all parallel trial coroutines — the internal
    semaphore is shared so the global concurrency cap is honoured regardless
    of how many trials run at once.

    Args:
        api_key:        OpenRouter API key (``OPENROUTER_API_KEY`` / ``OR_API_KEY``).
        base_url:       API base (default: ``https://openrouter.ai/api/v1``).
        max_concurrent: Maximum simultaneous in-flight API calls (default: 10).
                        At ~3 s average Haiku latency, 10 concurrent calls
                        saturates the ~3–4 calls/sec OpenRouter limit.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        max_concurrent: int = 10,
    ) -> None:
        self._openai    = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.messages   = _MessagesNamespace(self._openai, self._semaphore)

