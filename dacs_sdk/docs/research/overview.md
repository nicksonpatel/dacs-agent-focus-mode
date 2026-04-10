# Research Overview

DACS was originally developed as a research prototype for studying **context window management in multi-agent LLM orchestration**.  This page summarises the academic framing.

## Problem

Multi-agent LLM orchestration systems (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK) typically use a **flat context**: all agent states are concatenated into one prompt before each orchestrator call.

As agent count *N* grows:

- Context size scales as *O*(*N*) per steering call
- Cross-agent signals bleed together ("context pollution")
- Steering accuracy drops dramatically at N ≥ 5
- Token costs become prohibitive at N ≥ 10

Prior work (AFM, AOI) addresses this with **context compression** — reducing token counts after the fact.  DACS takes a different approach: **proactive asymmetric isolation** before the LLM call is made.

## Mechanism

DACS defines three orchestrator states:

| State | Context contents | When |
|---|---|---|
| `REGISTRY` | Compact snapshot ≤ 200 tok/agent | Default idle mode |
| `FOCUS(aᵢ)` | Full context of *aᵢ* + compressed registry of others | During per-agent steering |
| `USER_INTERACT` | Same as REGISTRY | During user messages |

**Agent-triggered** — the orchestrator never enters FOCUS without an explicit `SteeringRequest` from an agent.

**Asymmetric** — only the requesting agent's full context is loaded; all other agents see token-minimal registry entries.

**Deterministic token budget** — `tiktoken` counts tokens before every LLM call; the budget is enforced, never estimated.

## Key novelty

> **No prior work implements agent-triggered asymmetric REGISTRY/FOCUS mode switching for per-agent context isolation in concurrent multi-agent orchestration.**

This was confirmed after 20 targeted arXiv searches covering 9 papers:

| Paper | arXiv ID | Relation to DACS |
|---|---|---|
| AFM | 2511.12712 | Context compression (post-hoc) — not isolation |
| AOI | 2512.13956 | Attention optimisation — global, not per-agent |
| AgentOrchestra | 2506.12508 | Hierarchical orchestration — no FOCUS mode |
| ACE | 2510.04618 | Capability-based routing — no context isolation |
| AdaptOrch | 2602.16873 | Adaptive workflows — no registry/focus |
| CodeDelegator | 2601.14914 | Code delegation — no isolation mechanism |
| SideQuest | 2602.22603 | Subtask isolation — structural, not context-level |
| Adaptive Orchestration | 2601.09742 | Dynamic routing — no per-agent scoping |
| Lemon Agent | 2602.07092 | Tool-call optimisation — no context mechanism |

## Paper

The full paper is `paper/draft_v3.tex` in the research repository.

- [Benchmark results →](benchmarks.md)
