"""Smoke test for the DACS Phase 5 OIF experiment harness.

Runs one trial per condition (dacs_oif, dacs_no_oif, baseline) with a tiny
1-agent + 1-query mini-scenario, then validates:

  1. RUN_START / RUN_END events present
  2. At least one user query was injected (INJECTION event or USER_RESPONSE)
  3. dacs_oif condition emits an OIF_USER_QUERY event
  4. dacs_no_oif / baseline do NOT emit OIF_USER_QUERY
  5. M1_T5 computation does not crash

No LLM judge is run in the smoke test (that requires more time/tokens).

Usage
-----
    python -m experiments_oif._smoke_test
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

from experiments_oif.scenario_defs import OIFAgentSpec, OIFScenario, OIFUserQuery
from experiments_oif.run import run_trial, _resolve_api_and_model, _make_client, _DEFAULT_T

# ---------------------------------------------------------------------------
# Tiny 1-agent, 1-query smoke scenario
# ---------------------------------------------------------------------------

SMOKE_SCENARIO = OIFScenario(
    scenario_id="smoke_oif_n1",
    description="OIF smoke test: 1 agent, 1 user query at t=4s.",
    agents=[
        OIFAgentSpec(
            agent_id="a1",
            task_description=(
                "Implement a binary search tree (BST) in Python with insert and inorder traversal."
            ),
            decision_hints=(
                "- Traversal order: which traversal (inorder, preorder, postorder) "
                "should be the default, and why."
            ),
        ),
    ],
    user_queries=[
        OIFUserQuery(
            delay_s=4.0,
            message=(
                "How is the BST implementation going? "
                "What traversal approach is being used?"
            ),
            target_agent_id="a1",
            answer_keywords=["inorder", "in-order", "traversal", "recursive", "sorted"],
            judge_context=(
                "Focus-grounded: mentions traversal choice (inorder).  "
                "Registry-only: vague status update only."
            ),
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


def _check_structural(jsonl: Path, label: str) -> list[dict]:
    events = _load_events(jsonl)
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e.get("event"), []).append(e)

    assert "RUN_START" in by_type, f"[{label}] Missing RUN_START"
    assert "RUN_END"   in by_type, f"[{label}] Missing RUN_END"

    has_injection = bool(by_type.get("INJECTION") or by_type.get("USER_RESPONSE"))
    assert has_injection, f"[{label}] No INJECTION or USER_RESPONSE event found"

    print(f"  [{label}] structural checks PASS"
          f"  (steering_reqs={len(by_type.get('STEERING_REQUEST', []))}"
          f"  oif_queries={len(by_type.get('OIF_USER_QUERY', []))})")
    return events


def _check_oif_event_present(jsonl: Path, label: str) -> None:
    events = _load_events(jsonl)
    oif_evs = [e for e in events if e.get("event") == "OIF_USER_QUERY"]
    assert oif_evs, f"[{label}] Expected OIF_USER_QUERY event but found none"
    ev = oif_evs[0]
    assert ev.get("trigger") == "T5", f"[{label}] OIF trigger should be T5, got {ev.get('trigger')}"
    print(f"  [{label}] OIF_USER_QUERY present, trigger=T5, agent_id={ev.get('agent_id')}  PASS")


def _check_no_oif_event(jsonl: Path, label: str) -> None:
    events = _load_events(jsonl)
    oif_evs = [e for e in events if e.get("event") == "OIF_USER_QUERY"]
    assert not oif_evs, (
        f"[{label}] Did not expect OIF_USER_QUERY events but found {len(oif_evs)}"
    )
    print(f"  [{label}] No OIF_USER_QUERY events (correct for this condition)  PASS")


def _check_metrics_compute(jsonl: Path, label: str) -> None:
    from experiments_oif.run import _compute_oif_metrics
    metrics = _compute_oif_metrics(str(jsonl), SMOKE_SCENARIO)
    assert "m1_t5_accuracy"     in metrics, f"[{label}] Missing m1_t5_accuracy"
    assert "contamination_rate" in metrics, f"[{label}] Missing contamination_rate"
    print(
        f"  [{label}] metrics OK  "
        f"m1_t5={metrics['m1_t5_accuracy']:.2f}  "
        f"contam={metrics['contamination_rate']:.2f}  "
        f"oif_fired={metrics['n_oif_queries_fired']}"
    )


# ---------------------------------------------------------------------------
# Main smoke runner
# ---------------------------------------------------------------------------

async def _run() -> None:
    SMOKE_DIR.mkdir(exist_ok=True)

    resolved_api, api_key, model = _resolve_api_and_model("auto", None)
    client = _make_client(resolved_api, api_key)

    print(f"\n=== OIF Smoke Test  api={resolved_api}  model={model} ===\n")

    # -----------------------------------------------------------------------
    # 1. dacs_oif — OIF enabled, should emit OIF_USER_QUERY
    # -----------------------------------------------------------------------
    run_id = f"smoke_oif_n1_dacs_oif_t01_{uuid.uuid4().hex[:6]}"
    print(f"[1/3] dacs_oif trial:    {run_id}")
    await run_trial(
        scenario=SMOKE_SCENARIO,
        condition="dacs_oif",
        model=model,
        token_budget=_DEFAULT_T,
        run_id=run_id,
        client=client,
        results_dir=SMOKE_DIR,
    )
    jsonl = SMOKE_DIR / f"{run_id}.jsonl"
    _check_structural(jsonl, "dacs_oif")
    _check_oif_event_present(jsonl, "dacs_oif")
    _check_metrics_compute(jsonl, "dacs_oif")

    # -----------------------------------------------------------------------
    # 2. dacs_no_oif — DACS on, OIF off, should NOT emit OIF_USER_QUERY
    # -----------------------------------------------------------------------
    run_id = f"smoke_oif_n1_dacs_no_oif_t01_{uuid.uuid4().hex[:6]}"
    print(f"\n[2/3] dacs_no_oif trial: {run_id}")
    await run_trial(
        scenario=SMOKE_SCENARIO,
        condition="dacs_no_oif",
        model=model,
        token_budget=_DEFAULT_T,
        run_id=run_id,
        client=client,
        results_dir=SMOKE_DIR,
    )
    jsonl = SMOKE_DIR / f"{run_id}.jsonl"
    _check_structural(jsonl, "dacs_no_oif")
    _check_no_oif_event(jsonl, "dacs_no_oif")
    _check_metrics_compute(jsonl, "dacs_no_oif")

    # -----------------------------------------------------------------------
    # 3. baseline — flat context, no OIF
    # -----------------------------------------------------------------------
    run_id = f"smoke_oif_n1_baseline_t01_{uuid.uuid4().hex[:6]}"
    print(f"\n[3/3] baseline trial:    {run_id}")
    await run_trial(
        scenario=SMOKE_SCENARIO,
        condition="baseline",
        model=model,
        token_budget=_DEFAULT_T,
        run_id=run_id,
        client=client,
        results_dir=SMOKE_DIR,
    )
    jsonl = SMOKE_DIR / f"{run_id}.jsonl"
    _check_structural(jsonl, "baseline")
    _check_no_oif_event(jsonl, "baseline")
    _check_metrics_compute(jsonl, "baseline")

    print("\n=== ALL OIF SMOKE TESTS PASSED — safe to run full experiment ===\n")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
