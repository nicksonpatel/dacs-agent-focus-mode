"""LLM judge for DACS Phase 5 OIF experiment.

Evaluates T5 (user-query routing) response quality by comparing:
  - OIF-grounded responses (orchestrator entered Focus before answering)
  - Registry-only responses (no OIF — answered from summary only)

The judge rates each response 1–10 on specificity and grounding, and
assigns a binary hit (1/0) based on whether the rubric criteria are met.

Usage
-----
    python -m experiments_oif.judge --results-dir results_oif --scenario oif1_n3
    python -m experiments_oif.judge --results-dir results_oif --all

Environment
-----------
    OPENROUTER_API_KEY  (or OR_API_KEY)   — recommended: GPT-4o-mini via OpenRouter
    JUDGE_MODEL         — override model (default: openai/gpt-4o-mini)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from experiments_oif.scenario_defs import OIF_SCENARIOS, OIFScenario, OIFUserQuery

load_dotenv()

_JUDGE_MODEL   = os.environ.get("JUDGE_MODEL", "openai/gpt-4o-mini")
_RESULTS_DIR   = Path("results_oif")
_JUDGE_OUTFILE = "judge_oif.csv"


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an expert evaluator of multi-agent LLM orchestration systems. "
    "You will be given a user query directed at a running orchestrator and the "
    "orchestrator's response.  Your job is to rate whether the response demonstrates "
    "that the orchestrator accessed the target agent's FULL working context "
    "(Focus-grounded) or only its compact registry summary (registry-only).\n\n"
    "A Focus-grounded response contains specific technical decisions, tool choices, "
    "paper names, algorithm choices, or implementation details that only appear in "
    "the agent's full conversation history — NOT just a vague status update.\n\n"
    "Respond with a JSON object:\n"
    "  {\"score\": <1-10>, \"hit\": <0 or 1>, \"reason\": \"<1-2 sentence explanation>\"}\n\n"
    "score: 1=completely generic/registry-only, 10=highly specific/Focus-grounded\n"
    "hit: 1 if response is clearly Focus-grounded (score≥7 AND contains rubric keywords), 0 otherwise"
)


def _judge_prompt(
    query: OIFUserQuery,
    response_text: str,
) -> str:
    kw_list = ", ".join(f'"{k}"' for k in query.answer_keywords)
    return (
        f"USER QUERY: {query.message}\n\n"
        f"ORCHESTRATOR RESPONSE:\n{response_text}\n\n"
        f"TARGET AGENT: {query.target_agent_id}\n\n"
        f"RUBRIC — what a Focus-grounded answer should contain:\n{query.judge_context}\n\n"
        f"Answer keywords (≥1 should appear if Focus-grounded): {kw_list}\n\n"
        "Rate this response on specificity and grounding."
    )


# ---------------------------------------------------------------------------
# OpenRouter judge client (lightweight, no heavyweight dependency)
# ---------------------------------------------------------------------------

async def _judge_one(
    query: OIFUserQuery,
    response_text: str,
    client,
    model: str,
) -> dict:
    prompt = _judge_prompt(query, response_text)
    resp = await client.messages.create(
        model=model,
        max_tokens=200,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = ""
    for block in resp.content:
        if hasattr(block, "text"):
            raw = block.text
            break
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        return {
            "score":  int(parsed.get("score", 0)),
            "hit":    int(parsed.get("hit",   0)),
            "reason": str(parsed.get("reason", "")),
        }
    except Exception:
        return {"score": 0, "hit": 0, "reason": f"parse_error: {raw[:120]}"}


# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------

def _extract_oif_responses(
    log_path: Path,
    scenario: OIFScenario,
) -> list[tuple[OIFUserQuery, str]]:
    """Return (query_def, response_text) pairs from OIF_USER_QUERY or USER_RESPONSE events."""
    events = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    pairs: list[tuple[OIFUserQuery, str]] = []

    # Match OIF_USER_QUERY events (dacs_oif condition)
    oif_evs = {e["agent_id"]: e for e in events if e.get("event") == "OIF_USER_QUERY"}
    # Match USER_RESPONSE events (no-OIF conditions) — match by query message prefix
    user_resp_evs = [e for e in events if e.get("event") == "USER_RESPONSE"]

    for q in scenario.user_queries:
        response_text = ""
        if q.target_agent_id in oif_evs:
            response_text = oif_evs[q.target_agent_id].get("response_text", "")
        else:
            # Match by message overlap
            for ev in user_resp_evs:
                ev_msg = ev.get("message", "").lower()
                if q.target_agent_id in ev_msg or any(
                    w in ev_msg for w in q.message.lower().split()[:5]
                ):
                    response_text = ev.get("response_text", "")
                    break
        if response_text:
            pairs.append((q, response_text))

    return pairs


# ---------------------------------------------------------------------------
# Main judge runner
# ---------------------------------------------------------------------------

async def judge_log(
    log_path: Path,
    scenario: OIFScenario,
    client,
    model: str,
    condition: str,
    run_id: str,
) -> list[dict]:
    pairs = _extract_oif_responses(log_path, scenario)
    results = []
    for q, resp_text in pairs:
        judged = await _judge_one(q, resp_text, client, model)
        results.append({
            "run_id":           run_id,
            "condition":        condition,
            "scenario":         scenario.scenario_id,
            "target_agent_id":  q.target_agent_id,
            "query_fragment":   q.message[:60],
            "judge_score":      judged["score"],
            "judge_hit":        judged["hit"],
            "judge_reason":     judged["reason"],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="OIF judge for Phase 5")
    parser.add_argument("--results-dir", default="results_oif")
    parser.add_argument("--scenario",    default=None,
                        choices=list(OIF_SCENARIOS))
    parser.add_argument("--all",         action="store_true",
                        help="Judge all JSONL files in results-dir")
    parser.add_argument("--model",       default=None)
    args = parser.parse_args()

    or_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY", "")
    if not or_key:
        raise RuntimeError("Set OPENROUTER_API_KEY to run the OIF judge.")

    try:
        from src.openrouter_client import OpenRouterClient
        client = OpenRouterClient(api_key=or_key, max_concurrent=5)
    except ImportError:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=or_key)

    model = args.model or os.environ.get("JUDGE_MODEL") or _JUDGE_MODEL
    results_dir = Path(args.results_dir)
    all_results: list[dict] = []

    async def _run_all():
        jsonl_files = sorted(results_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL files found in {results_dir}")
            return

        for log_path in jsonl_files:
            # Infer scenario and condition from filename
            stem = log_path.stem
            # stem format: {scenario_id}_{condition}_t{nn}_{hex}
            scenario_id = None
            condition   = "unknown"
            for sid in OIF_SCENARIOS:
                if stem.startswith(sid):
                    scenario_id = sid
                    rest = stem[len(sid) + 1:]
                    condition = rest.rsplit("_t", 1)[0] if "_t" in rest else rest
                    break

            scen_def = None
            if args.scenario:
                scen_def = OIF_SCENARIOS.get(args.scenario)
            elif scenario_id:
                scen_def = OIF_SCENARIOS.get(scenario_id)

            if scen_def is None:
                print(f"Skipping {log_path.name} — could not resolve scenario")
                continue

            run_id = stem
            print(f"Judging {log_path.name}  scenario={scen_def.scenario_id}  cond={condition}")
            judged = await judge_log(log_path, scen_def, client, model, condition, run_id)
            all_results.extend(judged)
            for r in judged:
                print(
                    f"  [{r['target_agent_id']}] score={r['judge_score']}  "
                    f"hit={r['judge_hit']}  {r['judge_reason'][:80]}"
                )

    asyncio.run(_run_all())

    if all_results:
        out_path = results_dir / _JUDGE_OUTFILE
        fieldnames = list(all_results[0].keys())
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nJudge results written to {out_path}")

        # Summary by condition
        from itertools import groupby
        import statistics
        key = lambda r: (r["condition"], r["scenario"])
        print("\nSummary by condition:")
        for (cond, scen), group in groupby(sorted(all_results, key=key), key=key):
            g = list(group)
            mean_score = statistics.mean(r["judge_score"] for r in g)
            hit_rate   = statistics.mean(r["judge_hit"] for r in g)
            print(f"  {cond:20s}  {scen}  mean_score={mean_score:.2f}  hit_rate={hit_rate:.3f}  n={len(g)}")


if __name__ == "__main__":
    main()
