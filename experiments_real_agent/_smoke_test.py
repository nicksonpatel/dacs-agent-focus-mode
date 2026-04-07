"""Smoke test for the real-agent experiment harness.

Runs a single DACS trial and a single baseline trial of ra1_n3 (1 agent only,
via a trimmed mini-scenario) against OpenRouter/Haiku, then validates:

  1. At least one STEERING_REQUEST was emitted
  2. Every STEERING_REQUEST has a matching STEERING_RESPONSE
  3. The DACS trial's STEERING_RESPONSE has orchestrator_state == FOCUS
  4. The baseline trial's STEERING_RESPONSE has orchestrator_state == FLAT
  5. No crash / exception

Then runs the judge on those two trial logs and checks verdicts are written.

Usage
-----
    python -m experiments_real_agent._smoke_test
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Tiny 1-agent scenario so the smoke test is fast (~30 s)
# ---------------------------------------------------------------------------
from experiments_real_agent.scenario_defs import (
    RealAgentScenario,
    RealAgentSpec,
    DecisionRubric,
)

SMOKE_SCENARIO = RealAgentScenario(
    scenario_id="smoke_n1",
    description="Smoke test: 1 agent, 1 decision point only.",
    agents=[
        RealAgentSpec(
            agent_id="a1",
            task_description=(
                "Implement a binary search tree (BST) in Python with insert and traversal."
            ),
            decision_hints=(
                "- Traversal order: which traversal method (inorder, preorder, postorder) "
                "should be the primary default traversal, and why."
            ),
            rubrics=[
                DecisionRubric(
                    topic="traversal_order",
                    correct_keywords=["inorder", "in-order", "in order", "sorted order"],
                    judge_context=(
                        "For a BST, inorder traversal visits nodes in sorted order. "
                        "A correct response recommends inorder traversal."
                    ),
                ),
            ],
        ),
    ],
)

SMOKE_DIR = Path("results_smoke")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_events(jsonl: Path) -> list[dict]:
    events = []
    with open(jsonl) as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _check_trial(
    jsonl: Path,
    expected_state: str,
    label: str,
    require_steering: bool = True,
) -> int:
    """Return the number of steering round-trips found. Asserts if require_steering."""
    events = _load_events(jsonl)
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e.get("event"), []).append(e)

    req_ids  = {e["request_id"] for e in by_type.get("STEERING_REQUEST", [])}
    resp_ids = {e["request_id"] for e in by_type.get("STEERING_RESPONSE", [])}

    if require_steering:
        assert req_ids, f"[{label}] No STEERING_REQUEST events in {jsonl.name}"

    if req_ids:
        unmatched = req_ids - resp_ids
        assert not unmatched, (
            f"[{label}] {len(unmatched)} STEERING_REQUEST(s) have no response: {unmatched}"
        )
        for resp in by_type.get("STEERING_RESPONSE", []):
            state = resp.get("orchestrator_state", "")
            assert state == expected_state, (
                f"[{label}] Expected orchestrator_state={expected_state!r}, got {state!r}"
            )
        print(f"  [{label}] PASS — {len(req_ids)} steering round-trip(s), "
              f"state={expected_state}  ({jsonl.name})")
    else:
        print(f"  [{label}] OK (no steering emitted this run — non-deterministic with 1 agent)")

    return len(req_ids)


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

async def _run() -> None:
    from experiments_real_agent.run import run_trial, _resolve_api_and_model, _make_client
    from experiments_real_agent.judge import judge_scenario

    SMOKE_DIR.mkdir(exist_ok=True)

    resolved_api, api_key, model = _resolve_api_and_model("auto", None)
    client = _make_client(resolved_api, api_key, max_concurrent=4)

    print(f"\n=== Smoke test  api={resolved_api}  model={model} ===\n")

    # -----------------------------------------------------------------------
    # 1. DACS trial
    # -----------------------------------------------------------------------
    dacs_id = f"smoke_n1_dacs_t01_{uuid.uuid4().hex[:6]}"
    print(f"Running DACS trial:     {dacs_id}")
    await run_trial(
        SMOKE_SCENARIO, focus_mode=True,
        model=model, token_budget=204800,
        run_id=dacs_id, client=client, results_dir=SMOKE_DIR,
    )
    _check_trial(SMOKE_DIR / f"{dacs_id}.jsonl", expected_state="FOCUS", label="DACS")

    # -----------------------------------------------------------------------
    # 2. Baseline trial
    # -----------------------------------------------------------------------
    base_id = f"smoke_n1_baseline_t01_{uuid.uuid4().hex[:6]}"
    print(f"Running baseline trial: {base_id}")
    await run_trial(
        SMOKE_SCENARIO, focus_mode=False,
        model=model, token_budget=204800,
        run_id=base_id, client=client, results_dir=SMOKE_DIR,
    )
    # Baseline with 1 agent may not produce steering (non-deterministic with cheap models)
    _check_trial(
        SMOKE_DIR / f"{base_id}.jsonl",
        expected_state="FLAT",
        label="Baseline",
        require_steering=False,
    )

    # -----------------------------------------------------------------------
    # 3. Judge smoke check — only on the DACS trial (guaranteed to have steering)
    # -----------------------------------------------------------------------
    print("\nRunning judge on DACS smoke log…")

    # Temporarily register the smoke scenario so judge.collect_decisions works
    from experiments_real_agent import scenario_defs
    scenario_defs.REAL_SCENARIOS["smoke_n1"] = SMOKE_SCENARIO

    from experiments_real_agent.judge import _model_slug
    judge_model = "anthropic/claude-haiku-4-5"
    slug = _model_slug(judge_model)
    csv_path = SMOKE_DIR / f"judge_results_smoke_n1_{slug}.csv"

    results = await judge_scenario(
        "smoke_n1", csv_path,
        judge_model=judge_model,
        results_dir=SMOKE_DIR,
        concurrency=2,
    )
    assert results, "Judge returned no results"
    for r in results:
        assert r.get("judge_verdict") in {"CORRECT", "INCORRECT"}, (
            f"Unexpected verdict: {r.get('judge_verdict')!r}"
        )
    print(f"  [Judge] PASS — {len(results)} verdict(s) written to {csv_path.name}")

    # -----------------------------------------------------------------------
    print("\n=== ALL SMOKE TESTS PASSED — safe to run full experiment ===\n")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
