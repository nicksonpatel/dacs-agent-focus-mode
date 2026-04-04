"""Metrics calculation for DACS experiment results.

Reads a run's .jsonl log file and computes the five primary metrics:
  1. steering_accuracy    — fraction of steering responses that contain ≥1 answer keyword
  2. contamination_rate   — fraction of responses referencing wrong-agent context
  3. avg_context_tokens   — mean token count at steering time
  4. p95_context_tokens   — 95th percentile token count at steering time
  5. user_latency_ms      — mean time from USER_INTERACT entry to exit (ms)

Usage
-----
    from experiments.metrics import compute_metrics
    results = compute_metrics("results/run_abc123.jsonl", scenario_spec)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from experiments.task_suite import ScenarioSpec


def compute_metrics(log_path: str, scenario: ScenarioSpec) -> dict[str, Any]:
    """Parse a run log and return the metrics dict."""
    events = _load_events(log_path)

    steering_responses = [e for e in events if e["event"] == "STEERING_RESPONSE"]
    context_built      = [e for e in events if e["event"] == "CONTEXT_BUILT"]
    transitions        = [e for e in events if e["event"] == "TRANSITION"]

    # Build ground-truth map: agent_id → [DecisionPoint, ...]
    dp_map: dict[str, list] = {
        spec.agent_id: list(spec.decision_points) for spec in scenario.agents
    }
    # Per-agent pointer into the decision_points list (responses arrive in order)
    dp_pointer: dict[str, int] = {aid: 0 for aid in dp_map}

    correct = 0
    total   = 0
    contaminated = 0

    for resp in steering_responses:
        aid   = resp["agent_id"]
        text  = resp.get("response_text", "").lower()
        ptr   = dp_pointer.get(aid, 0)
        dps   = dp_map.get(aid, [])

        if ptr < len(dps):
            dp = dps[ptr]
            total += 1
            if any(kw.lower() in text for kw in dp.answer_keywords):
                correct += 1
            dp_pointer[aid] = ptr + 1

        # Contamination: does the response mention another agent's ID?
        other_ids = [s.agent_id for s in scenario.agents if s.agent_id != aid]
        if any(other_id in text for other_id in other_ids):
            contaminated += 1

    steering_accuracy  = correct / total if total else 0.0
    contamination_rate = contaminated / len(steering_responses) if steering_responses else 0.0

    # Context token sizes at steering time
    focus_tokens = [
        e["token_count"] for e in context_built
        if e["mode"] in ("FOCUS", "FLAT")
    ]
    avg_context   = statistics.mean(focus_tokens)    if focus_tokens else 0
    p95_context   = _percentile(focus_tokens, 95)    if focus_tokens else 0

    # User latency: time between entering and leaving USER_INTERACT
    user_latency_ms = _compute_user_latency(transitions)

    return {
        "steering_accuracy":    round(steering_accuracy, 4),
        "contamination_rate":   round(contamination_rate, 4),
        "avg_context_tokens":   round(avg_context, 1),
        "p95_context_tokens":   round(p95_context, 1),
        "user_latency_ms":      round(user_latency_ms, 1),
        "total_decisions":      total,
        "correct_decisions":    correct,
        "contaminated_responses": contaminated,
        "total_steering":       len(steering_responses),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_events(path: str) -> list[dict]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def _compute_user_latency(transitions: list[dict]) -> float:
    latencies: list[float] = []
    enter_ts: float | None = None
    for t in transitions:
        if t.get("to") == "USER_INTERACT":
            enter_ts = _parse_ts(t["ts"])
        elif t.get("from") == "USER_INTERACT" and enter_ts is not None:
            exit_ts = _parse_ts(t["ts"])
            latencies.append((exit_ts - enter_ts) * 1000)
            enter_ts = None
    return statistics.mean(latencies) if latencies else 0.0


def _parse_ts(iso: str) -> float:
    from datetime import datetime, timezone
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
