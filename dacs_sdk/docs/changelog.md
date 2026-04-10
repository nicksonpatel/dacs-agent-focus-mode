# Changelog

All notable changes to `dacs-agent`.

## [0.1.0] — 2025

### Added

Initial release of the DACS framework.

**Core components:**
- `DACSRuntime` — high-level context manager; wires all components
- `BaseAgent` — abstract base class; subclass and implement `_execute()`
- `StepAgent` — ready-to-use step-driven agent (no subclassing)
- `Orchestrator` — REGISTRY / FOCUS / USER_INTERACT state machine
- `RegistryManager` — per-agent state store (≤ 200 tokens/entry, enforced)
- `ContextBuilder` — token-counted context assembly (tiktoken cl100k_base)
- `SteeringRequestQueue` — priority queue (HIGH-urgency requests go to front)
- `Logger` — JSONL event log with pluggable sinks
- `TerminalMonitor` — Rich live terminal display (`[monitor]` extra)

**Protocols:**
- `SteeringRequest` / `SteeringResponse` dataclasses
- `RegistryEntry` / `RegistryUpdate` dataclasses
- `FocusContext` dataclass
- `AgentStatus` / `UrgencyLevel` enums

**Features:**
- DACS mode (`focus_mode=True`) and flat-context baseline (`focus_mode=False`) in the same package
- HIGH-urgency interrupt handling
- FOCUS session timeout with graceful fallback
- Custom API endpoint support (OpenRouter, MiniMax, Azure, Vertex AI)
- `verbose=True` for zero-config live monitoring
- Full JSONL event log with 12 event types

---

The version scheme follows [Semantic Versioning](https://semver.org/).
