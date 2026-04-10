"""
DACS — Dynamic Attentional Context Scoping
==========================================

Agent-triggered focus sessions for isolated per-agent steering
in multi-agent LLM orchestration.

Quick start
-----------
>>> import asyncio
>>> from dacs import DACSRuntime, BaseAgent, UrgencyLevel
>>>
>>> class MyAgent(BaseAgent):
...     async def _execute(self):
...         self._push_update("Started work on the task.")
...         resp = await self._request_steering(
...             context="I have finished the analysis phase.",
...             question="Should I output JSON or CSV format?",
...         )
...         self._push_update(f"Using format: {resp.response_text}")
>>>
>>> async def main():
...     async with DACSRuntime(model="claude-3-haiku-20240307") as runtime:
...         runtime.add_agent(MyAgent(agent_id="a1", task="Analyse and export dataset"))
...         await runtime.run()
>>>
>>> asyncio.run(main())

High-level API
--------------
- :class:`DACSRuntime`  — context manager that wires all components
- :class:`BaseAgent`    — abstract base; subclass and implement ``_execute``
- :class:`StepAgent`    — ready-to-use step-driven agent for quick prototyping
- :data:`UrgencyLevel`  — ``LOW | MEDIUM | HIGH``
- :data:`AgentStatus`   — ``RUNNING | BLOCKED | WAITING_STEERING | COMPLETE | FAILED``

Low-level API (for advanced wiring)
------------------------------------
- :class:`Orchestrator`
- :class:`RegistryManager`
- :class:`ContextBuilder`
- :class:`SteeringRequestQueue`
- :class:`Logger`
- :class:`TerminalMonitor`

Published under the MIT License.
Paper: https://arxiv.org/abs/XXXX.XXXXX
"""

from __future__ import annotations

from dacs._protocols import (
    AgentStatus,
    FocusContext,
    RegistryEntry,
    RegistryUpdate,
    SteeringRequest,
    SteeringResponse,
    SteeringRequestQueue,
    UrgencyLevel,
)
from dacs._logger import Logger
from dacs._registry import RegistryManager
from dacs._context_builder import ContextBuilder, ContextBudgetError
from dacs._orchestrator import Orchestrator, OrchestratorState
from dacs._agent import BaseAgent
from dacs._step_agent import StepAgent
from dacs._runtime import DACSRuntime

try:
    from dacs._monitor import TerminalMonitor
except ImportError:
    TerminalMonitor = None  # type: ignore[assignment,misc]

__version__ = "0.1.0"
__author__ = "Nickson Patel"

__all__ = [
    # High-level
    "DACSRuntime",
    "BaseAgent",
    "StepAgent",
    # Enums
    "AgentStatus",
    "UrgencyLevel",
    "OrchestratorState",
    # Dataclasses
    "SteeringRequest",
    "SteeringResponse",
    "FocusContext",
    "RegistryEntry",
    "RegistryUpdate",
    # Low-level components
    "Orchestrator",
    "RegistryManager",
    "ContextBuilder",
    "SteeringRequestQueue",
    "Logger",
    "TerminalMonitor",
    # Exceptions
    "ContextBudgetError",
    # Metadata
    "__version__",
]
