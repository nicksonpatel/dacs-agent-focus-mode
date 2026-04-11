"""Smoke test for the DACS WebArena benchmark harness.

Runs one DACS trial and one baseline trial of a minimal 1-agent scenario
(agent a2 — map task) and validates:

  1. At least one STEERING_REQUEST was emitted
  2. Every STEERING_REQUEST has a matching STEERING_RESPONSE
  3. The DACS trial's STEERING_RESPONSE has orchestrator_state == FOCUS
  4. The baseline trial's STEERING_RESPONSE has orchestrator_state == FLAT
  5. No crash / exception

Usage
-----
    python -m experiments_webarena._smoke_test
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Tiny 1-agent scenario — agent a2 (map) with just 1 rubric for speed
# ---------------------------------------------------------------------------
from experiments_webarena.scenario_defs import (
    WebArenaScenario,
    WebArenaAgentSpec,
    DecisionRubric,
)

_PAGE_MAP_52 = """\
[Simulated OpenStreetMap routing result]
Start: Carnegie Mellon University, Pittsburgh, PA
End:   Starbucks, 417 South Craig Street, Pittsburgh, PA
Walking route (normal pace ~4 km/h): 7 min  (0.5 km)
Driving route: 2 min  (0.6 km)
"""

SMOKE_SCENARIO = WebArenaScenario(
    scenario_id="smoke_wa_n1",
    description="Smoke test: 1 WebArena agent (map task 52), 1 decision point.",
    agents=[
        WebArenaAgentSpec(
            agent_id="a1",
            webarena_task_id=52,
            webarena_site="map",
            task_description=(
                "Calculate the walking time from Carnegie Mellon University to "
                "the Starbucks on Craig Street, Pittsburgh."
            ),
            simulated_page_context=_PAGE_MAP_52,
            decision_hints=(
                "- Walking speed: whether to use the standard average walking speed "
                "(~4 km/h) or a custom pace when estimating the travel time."
            ),
            rubrics=[
                DecisionRubric(
                    topic="walking_speed",
                    correct_keywords=["average", "standard", "4 km", "normal", "default"],
                    judge_context=(
                        "OpenStreetMap uses a standard average walking pace (~4 km/h). "
                        "A correct response recommends using the standard/average pace."
                    ),
                    webarena_eval="fuzzy_match",
                    reference_answer="7 min",
                ),
            ],
        ),
    ],
)

SMOKE_DIR = Path("results_smoke_wa")


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
    """Return the number of steering round-trips.  Asserts if require_steering."""
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
            f"[{label}] {len(unmatched)} STEERING_REQUEST(s) without response: {unmatched}"
        )
        for resp in by_type.get("STEERING_RESPONSE", []):
            state = resp.get("orchestrator_state", "")
            assert state == expected_state, (
                f"[{label}] Expected orchestrator_state={expected_state!r}, got {state!r}"
            )
        print(
            f"  [{label}] PASS — {len(req_ids)} steering round-trip(s), "
            f"state={expected_state}  ({jsonl.name})"
        )
    else:
        print(f"  [{label}] OK (no steering emitted — non-deterministic with 1 agent)")

    return len(req_ids)


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

async def _run() -> None:
    from experiments_webarena.run import run_trial, _resolve_api_and_model, _make_client

    SMOKE_DIR.mkdir(exist_ok=True)

    resolved_api, api_key, model = _resolve_api_and_model("auto", None)
    client = _make_client(resolved_api, api_key, max_concurrent=4)

    print(f"\n=== WebArena smoke test  api={resolved_api}  model={model} ===\n")

    # ------------------------------------------------------------------
    # 1. DACS trial
    # ------------------------------------------------------------------
    dacs_id = f"smoke_wa_n1_dacs_t01_{uuid.uuid4().hex[:6]}"
    print(f"Running DACS trial:     {dacs_id}")
    await run_trial(
        SMOKE_SCENARIO, focus_mode=True,
        model=model, token_budget=204800,
        run_id=dacs_id, client=client, results_dir=SMOKE_DIR,
    )
    _check_trial(SMOKE_DIR / f"{dacs_id}.jsonl", expected_state="FOCUS", label="DACS")

    # ------------------------------------------------------------------
    # 2. Baseline trial
    # ------------------------------------------------------------------
    base_id = f"smoke_wa_n1_baseline_t01_{uuid.uuid4().hex[:6]}"
    print(f"Running baseline trial: {base_id}")
    await run_trial(
        SMOKE_SCENARIO, focus_mode=False,
        model=model, token_budget=204800,
        run_id=base_id, client=client, results_dir=SMOKE_DIR,
    )
    _check_trial(
        SMOKE_DIR / f"{base_id}.jsonl",
        expected_state="FLAT",
        label="Baseline",
        require_steering=False,  # small model may not always steer in 1-agent case
    )

    print("\n=== WebArena smoke test COMPLETE ===")
    print(f"Logs written to {SMOKE_DIR}/")
    print("Next: run judge to verify M1 scoring:")
    print(
        "  python -m experiments_webarena.judge "
        f"--results-dir {SMOKE_DIR} --scenario smoke_wa_n1"
    )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
