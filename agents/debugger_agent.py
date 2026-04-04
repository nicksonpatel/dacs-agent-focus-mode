from __future__ import annotations

from agents.base_agent import BaseAgent
from src.protocols import AgentStatus, UrgencyLevel


class DebuggerAgent(BaseAgent):
    """Stub agent simulating iterative software debugging.

    Phases: hypothesis → reproduction → isolation → fix strategy → verification.
    Two decision points per phase (10 total) with escalating urgency as the
    bug is narrowed down and confirmed reproducible.

    Default ground-truth answers (for s5_n5_crossfire usage):
      1. sanitizer tool         → "asan" / "address sanitizer"
      2. heap profiler          → "valgrind" / "heaptrack"
      3. reproduction strategy  → "minimal reproducer" / "minimise"
      4. isolation target       → "allocation site" / "call stack"
      5. fix strategy           → "unique_ptr" / "smart pointer" / "raii"
      6. regression test type   → "unit test" / "leak check"
    """

    _STEPS: list[tuple[str, bool, str, UrgencyLevel]] = [
        # Phase 1 — Hypothesis
        (
            "Analysing crash reports and heap dumps to form initial hypotheses",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "Multiple potential causes: leak in allocator, use-after-free, or double-free; need tooling",
            True,
            (
                "Which sanitizer tool should we enable first to locate the memory leak? "
                "AddressSanitizer (ASan — fast, detects leaks/overflows/use-after-free), "
                "Valgrind Memcheck (slower but more thorough), "
                "or UndefinedBehaviorSanitizer (UBSan — different class of errors)? "
                "We suspect a heap memory leak specifically."
            ),
            UrgencyLevel.MEDIUM,
        ),
        # Phase 2 — Reproduction
        (
            "Enabled ASan; reproducing crash under test harness",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "ASan confirms heap leak; need to identify allocation hotspot with profiler",
            True,
            (
                "ASan shows leaked bytes but not allocation backtraces clearly. "
                "Which heap profiler to use for detailed allocation site analysis? "
                "Valgrind Massif (allocation timeline), heaptrack (allocation call stacks), "
                "or gperftools heap profiler (low overhead for production-like workloads)?"
            ),
            UrgencyLevel.HIGH,
        ),
        # Phase 3 — Isolation
        (
            "Profiler output shows leaks across 3 modules; narrowing to smallest reproduction",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "Need minimal reproduction strategy to isolate the faulty allocation site",
            True,
            (
                "For reproducing the memory leak in isolation: "
                "build a minimal reproducer that exercises only the suspected module, "
                "add instrumented allocator logging to trace all malloc/free pairs, "
                "or inject fault at the OS paging level? "
                "A minimal reproducer isolates the leak fastest without modifying production code."
            ),
            UrgencyLevel.HIGH,
        ),
        # Phase 4 — Fix strategy
        (
            "Minimal reproducer confirmed: leak is in the ThreadPool worker teardown path",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "Fix strategy needed: raw pointer with manual delete or RAII smart pointer refactor",
            True,
            (
                "The ThreadPool holds raw pointers to Worker objects and teardown misses some on early exit. "
                "Fix strategy: "
                "wrap Worker* in std::unique_ptr<Worker> (RAII, automatic cleanup), "
                "add explicit delete calls in all exit paths, "
                "or use a shared_ptr with a shutdown flag? "
                "Prefer the fix with least manual tracking."
            ),
            UrgencyLevel.HIGH,
        ),
        # Phase 5 — Verification
        (
            "Applying smart pointer refactor across ThreadPool; testing under ASan",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "ASan clean after fix; deciding regression test strategy to prevent recurrence",
            True,
            (
                "For regression test to guard against this memory leak reoccurring: "
                "add a unit test that runs the full teardown path under Valgrind/ASan in CI, "
                "add a leak-check assertion using ASan __lsan_do_leak_check() API in the existing test, "
                "or add a stress test that spawns and destroys 10,000 ThreadPool instances? "
                "A targeted ASan unit test in CI prevents silent regression."
            ),
            UrgencyLevel.MEDIUM,
        ),
    ]

    async def _execute(self) -> None:
        for summary, needs_steering, question, urgency in self._STEPS:
            self._push_update(AgentStatus.RUNNING, summary, urgency)
            if needs_steering:
                response = await self._request_steering(
                    relevant_context=self._recent_output(),
                    question=question,
                    blocking=True,
                    urgency=urgency,
                )
                self._push_update(
                    AgentStatus.RUNNING,
                    f"guidance: {response.response_text[:80]}",
                    UrgencyLevel.LOW,
                )
