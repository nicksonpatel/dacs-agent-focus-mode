"""High-level DACSRuntime — wires all DACS components in one place."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

from dacs._agent import BaseAgent
from dacs._context_builder import ContextBuilder
from dacs._logger import Logger
from dacs._orchestrator import Orchestrator
from dacs._protocols import SteeringRequestQueue
from dacs._registry import RegistryManager

if TYPE_CHECKING:
    pass

_DEFAULT_TOKEN_BUDGET = 200_000
_DEFAULT_MODEL = "claude-3-haiku-20240307"


class DACSRuntime:
    """High-level context manager that assembles and runs a DACS system.

    This is the recommended entry point for DACS.  It wires the
    :class:`~dacs.RegistryManager`, :class:`~dacs.ContextBuilder`,
    :class:`~dacs.SteeringRequestQueue`, and :class:`~dacs.Orchestrator`
    together in the correct order and runs all agents as asyncio tasks.

    Parameters
    ----------
    model:
        LLM model identifier (e.g. ``"claude-3-haiku-20240307"``).
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable if not provided.
    base_url:
        Custom API base URL.  Use this to point at an Anthropic-compatible
        endpoint (e.g. Azure, OpenRouter, MiniMax).
    token_budget:
        Hard cap on context window tokens.  Default is 200 000.
    log_path:
        Path to write the JSONL event log.  Set to ``None`` to disable
        file logging.
    focus_mode:
        ``True`` (default) for DACS mode.  ``False`` enables the flat-context
        baseline for comparison.
    focus_timeout:
        Seconds before an idle FOCUS session is abandoned (default 60).
    verbose:
        If ``True``, attaches a :class:`~dacs.TerminalMonitor` to print a
        live event feed to the console.  Requires ``pip install dacs-agent[monitor]``.

    Examples
    --------
    Async context manager (recommended)::

        async with DACSRuntime(model="claude-3-haiku-20240307") as runtime:
            runtime.add_agent(my_agent)
            await runtime.run()

    Manual usage::

        runtime = DACSRuntime(model="claude-3-haiku-20240307")
        await runtime.start()
        runtime.add_agent(my_agent)
        await runtime.run()
        await runtime.stop()
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        log_path: str | None = "dacs_run.jsonl",
        focus_mode: bool = True,
        focus_timeout: int = 60,
        verbose: bool = False,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url
        self._token_budget = token_budget
        self._log_path = log_path
        self._focus_mode = focus_mode
        self._focus_timeout = focus_timeout
        self._verbose = verbose

        self._agents: list[BaseAgent] = []
        self._logger: Logger | None = None
        self._orchestrator: Orchestrator | None = None
        self._registry: RegistryManager | None = None
        self._queue: SteeringRequestQueue | None = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DACSRuntime":
        await self._setup()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._logger:
            self._logger.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _setup(self) -> None:
        """Wire all components. Called automatically by __aenter__."""
        self._logger = Logger(self._log_path)

        if self._verbose:
            try:
                from dacs._monitor import TerminalMonitor

                monitor = TerminalMonitor(token_budget=self._token_budget)
                self._logger.add_sink(monitor.handle)
            except ImportError:
                import warnings

                warnings.warn(
                    "verbose=True requires the 'monitor' extra: "
                    "pip install dacs-agent[monitor]",
                    stacklevel=2,
                )

        self._registry = RegistryManager(self._logger)
        self._queue = SteeringRequestQueue(self._logger)

        cb = ContextBuilder(self._token_budget, self._logger)
        self._registry.set_context_builder(cb)

        client_kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        llm_client = AsyncAnthropic(**client_kwargs)

        self._orchestrator = Orchestrator(
            registry=self._registry,
            queue=self._queue,
            context_builder=cb,
            llm_client=llm_client,
            model=self._model,
            token_budget=self._token_budget,
            focus_mode=self._focus_mode,
            focus_timeout=self._focus_timeout,
            logger=self._logger,
        )

        # Register any agents added before setup
        for agent in self._agents:
            self._registry.register(agent.agent_id, agent.task)
            self._orchestrator.register_agent(agent)

    def add_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the runtime.

        Can be called before or after :meth:`_setup`.

        Parameters
        ----------
        agent:
            An instance of :class:`~dacs.BaseAgent` or :class:`~dacs.StepAgent`.
        """
        agent._registry = self._registry  # type: ignore[attr-defined]
        agent._queue = self._queue  # type: ignore[attr-defined]
        self._agents.append(agent)
        if self._registry and self._orchestrator:
            # Runtime is already set up — register immediately
            self._registry.register(agent.agent_id, agent.task)
            self._orchestrator.register_agent(agent)

    async def run(self) -> None:
        """Run all registered agents and the orchestrator concurrently.

        Returns when all agents have completed.
        """
        if self._orchestrator is None:
            await self._setup()

        assert self._orchestrator is not None

        # Re-wire any agents that were added before _setup()
        for agent in self._agents:
            if agent._registry is None:  # type: ignore[attr-defined]
                agent._registry = self._registry  # type: ignore[attr-defined]
                agent._queue = self._queue  # type: ignore[attr-defined]
            if agent.agent_id not in self._orchestrator._agents:
                self._registry.register(agent.agent_id, agent.task)  # type: ignore[union-attr]
                self._orchestrator.register_agent(agent)

        orchestrator_task = asyncio.create_task(self._orchestrator.run())
        agent_tasks = [asyncio.create_task(agent.run()) for agent in self._agents]

        # Wait for all agents to finish, then stop the orchestrator
        await asyncio.gather(*agent_tasks)
        self._orchestrator.stop()
        await orchestrator_task

    async def ask(self, message: str) -> str:
        """Send a user message to the orchestrator and get a response.

        The orchestrator responds using the current registry context.
        Safe to call during a run.

        Parameters
        ----------
        message:
            Your question or instruction for the orchestrator.

        Returns
        -------
        str
            The orchestrator's response.
        """
        if self._orchestrator is None:
            raise RuntimeError("Runtime not started. Use 'async with DACSRuntime(...) as rt:'")
        return await self._orchestrator.handle_user_message(message)

    @property
    def orchestrator(self) -> Orchestrator | None:
        """Direct access to the underlying :class:`~dacs.Orchestrator`."""
        return self._orchestrator

    @property
    def registry(self) -> RegistryManager | None:
        """Direct access to the underlying :class:`~dacs.RegistryManager`."""
        return self._registry
