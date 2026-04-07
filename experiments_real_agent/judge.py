"""LLM-as-judge for the DACS real-agent validation experiment.

Unlike the Phase 1–3 judges (which use predefined question fragments), this
judge uses the *actual* question text emitted by each LLMAgent.  It reads
question text from ``STEERING_REQUEST`` log events (the ``question`` field
added by the patched ``SteeringRequestQueue.enqueue()``), pairs each request
with its response via ``request_id``, and evaluates the orchestrator's answer
against the corresponding ``DecisionRubric``.

Rubric assignment: responses are assigned to rubrics in sequence order
per-agent (first STEERING_RESPONSE for agent a1 → rubric[0], etc.),
matching the same assumption used in ``experiments/metrics.py``.

Outputs
-------
    <results-dir>/judge_results_{scenario}_{model_slug}.csv  — per-decision verdicts
    <results-dir>/judge_comparison.md                        — multi-model comparison table

Usage
-----
    set -a && source .env && set +a

    # Single model (default haiku)
    python -m experiments_real_agent.judge --scenario ra1_n3

    # Multi-model comparison on Haiku-run results
    python -m experiments_real_agent.judge \
        --results-dir results_real_agent_haiku \
        --models anthropic/claude-haiku-4-5 anthropic/claude-3-5-sonnet openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

from experiments_real_agent.scenario_defs import REAL_SCENARIOS, RealAgentScenario, DecisionRubric

_RESULTS_DIR = Path("results_real_agent")
_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-haiku-4-5")
_API_KEY     = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY", "")
_BASE_URL    = "https://openrouter.ai/api/v1"

if not _API_KEY:
    raise RuntimeError(
        "Set OPENROUTER_API_KEY (or OR_API_KEY) in .env to run the judge."
    )

CONCURRENCY = 8  # default parallel judge calls per model — can be raised via --concurrency

# Slug used in CSV filenames: replace / and : with _
def _model_slug(model: str) -> str:
    return re.sub(r"[/:]", "_", model)

CSV_FIELDS = [
    "scenario", "run_id", "condition", "agent_id", "rubric_index", "rubric_topic",
    "actual_question", "keyword_score", "judge_verdict", "judge_reason",
    "orchestrator_state", "correct_keywords", "response_text",
]


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _load_events(fpath: Path) -> list[dict]:
    events: list[dict] = []
    with open(fpath) as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _build_request_map(events: list[dict]) -> dict[str, str]:
    """Return {request_id: question} from STEERING_REQUEST events.

    Requires the patched SteeringRequestQueue.enqueue() that logs 'question'.
    Falls back to empty string if the field is absent (older logs).
    """
    return {
        e["request_id"]: e.get("question", "")
        for e in events
        if e.get("event") == "STEERING_REQUEST"
    }


def collect_decisions(
    scenario_id: str,
    files: list[Path] | None = None,
    results_dir: Path = _RESULTS_DIR,
) -> list[dict]:
    """Pair each STEERING_RESPONSE with its question and rubric.

    Assignment strategy: per-agent sequential — agent a1's first response →
    rubric[0], second → rubric[1], etc.  If the agent emitted more responses
    than there are rubrics, the extra responses are labelled with rubric_index
    equal to the rubric count and topic "extra".
    """
    scenario: RealAgentScenario = REAL_SCENARIOS[scenario_id]
    rubric_map: dict[str, list[DecisionRubric]] = {
        spec.agent_id: spec.rubrics for spec in scenario.agents
    }
    task_map: dict[str, str] = {
        spec.agent_id: spec.task_description for spec in scenario.agents
    }

    if files is None:
        files = sorted(results_dir.glob(f"{scenario_id}_*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No JSONL files for {scenario_id} in {results_dir}"
        )

    all_decisions: list[dict] = []

    for fpath in files:
        run_id    = fpath.stem
        condition = "dacs" if "_dacs_" in run_id else "baseline"
        events    = _load_events(fpath)
        req_map   = _build_request_map(events)

        dp_pointer: dict[str, int] = {aid: 0 for aid in rubric_map}

        for event in events:
            if event.get("event") != "STEERING_RESPONSE":
                continue
            aid = event["agent_id"]
            ptr = dp_pointer.get(aid, 0)

            rubrics = rubric_map.get(aid, [])
            if ptr < len(rubrics):
                rubric       = rubrics[ptr]
                rubric_topic = rubric.topic
                rubric_kws   = rubric.correct_keywords
            else:
                rubric_topic = "extra"
                rubric_kws   = []

            request_id    = event.get("request_id", "")
            actual_question = req_map.get(request_id, "")
            response_text   = event.get("response_text", "")
            keyword_score   = (
                1 if any(kw.lower() in response_text.lower() for kw in rubric_kws)
                else 0
            )

            all_decisions.append({
                "scenario":          scenario_id,
                "run_id":            run_id,
                "condition":         condition,
                "agent_id":          aid,
                "rubric_index":      ptr,
                "rubric_topic":      rubric_topic,
                "actual_question":   actual_question,
                "keyword_score":     keyword_score,
                "correct_keywords":  rubric_kws,
                "response_text":     response_text,
                "orchestrator_state": event.get("orchestrator_state", ""),
                "domain_hint":       task_map.get(aid, "")[:80],
                "judge_context":     (
                    rubrics[ptr].judge_context if ptr < len(rubrics) else ""
                ),
            })
            dp_pointer[aid] = ptr + 1

    return all_decisions


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_existing_verdicts(csv_path: Path) -> set[tuple[str, str, int]]:
    """Return set of (run_id, agent_id, rubric_index) already judged."""
    done: set[tuple[str, str, int]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["run_id"], row["agent_id"], int(row["rubric_index"])))
    return done


def _open_csv_writer(csv_path: Path) -> tuple[csv.DictWriter, object]:
    is_new = not csv_path.exists()
    fh     = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if is_new:
        writer.writeheader()
        fh.flush()
    return writer, fh


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an expert technical evaluator assessing whether an AI orchestrator "
    "correctly answered a specialist agent's design question in a multi-agent "
    "workflow.\n\n"
    "You will receive:\n"
    "1. The agent's task domain\n"
    "2. The exact question the agent asked\n"
    "3. The orchestrator's response\n"
    "4. A rubric note explaining what a correct answer looks like\n\n"
    "CORRECT: the response meaningfully and substantively addresses the question "
    "— it gives a technically valid recommendation or makes a defensible choice "
    "consistent with the rubric.\n"
    "INCORRECT: the response fails to address the question, gives a vague or "
    "irrelevant answer, or its recommendation contradicts the rubric.\n\n"
    "Respond in exactly this XML format:\n"
    "<reason>one sentence explanation</reason>\n"
    "<verdict>CORRECT</verdict>\n"
    "or\n"
    "<reason>one sentence explanation</reason>\n"
    "<verdict>INCORRECT</verdict>"
)


def _build_judge_prompt(decision: dict) -> str:
    question_display = (
        decision["actual_question"] if decision["actual_question"]
        else f"[topic: {decision['rubric_topic']}]"
    )
    return (
        f"Agent task domain: {decision['domain_hint']}\n\n"
        f"Agent's question to orchestrator:\n{question_display}\n\n"
        f"Orchestrator response:\n{decision['response_text'][:1200]}\n\n"
        f"Rubric (what a correct answer looks like):\n{decision['judge_context']}\n\n"
        "Is this response CORRECT or INCORRECT?"
    )


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

_VERDICT_RE = re.compile(r"<verdict>\s*(CORRECT|INCORRECT)\s*</verdict>", re.IGNORECASE)
_REASON_RE  = re.compile(r"<reason>(.*?)</reason>", re.IGNORECASE | re.DOTALL)


def _parse_verdict(raw: str) -> tuple[str, str]:
    """Return (verdict, reason) from judge XML output."""
    verdict_m = _VERDICT_RE.search(raw)
    reason_m  = _REASON_RE.search(raw)
    verdict   = verdict_m.group(1).upper() if verdict_m else (
        "CORRECT" if "CORRECT" in raw.upper() and "INCORRECT" not in raw.upper()
        else "INCORRECT"
    )
    reason = reason_m.group(1).strip() if reason_m else raw[:120]
    return verdict, reason


# ---------------------------------------------------------------------------
# Async judge loop
# ---------------------------------------------------------------------------

async def _judge_one(
    client: openai.AsyncOpenAI,
    decision: dict,
    semaphore: asyncio.Semaphore,
    judge_model: str = _JUDGE_MODEL,
) -> dict:
    async with semaphore:
        for attempt in range(8):
            try:
                resp = await client.chat.completions.create(
                    model=judge_model,
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user",   "content": _build_judge_prompt(decision)},
                    ],
                )
                break
            except openai.RateLimitError:
                wait = 15 * (2 ** attempt)
                print(f"  [judge] Rate limit, waiting {wait}s (attempt {attempt+1}/8)…")
                await asyncio.sleep(wait)
            except openai.APIStatusError as exc:
                if exc.status_code == 429:
                    wait = 15 * (2 ** attempt)
                    print(f"  [judge] 429 from OpenRouter, waiting {wait}s (attempt {attempt+1}/8)…")
                    await asyncio.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError("Rate limit retries exhausted")
        raw = resp.choices[0].message.content or ""
        verdict, reason = _parse_verdict(raw)
        return {**decision, "judge_verdict": verdict, "judge_reason": reason}


async def judge_scenario(
    scenario_id: str,
    csv_path: Path,
    judge_model: str = _JUDGE_MODEL,
    results_dir: Path = _RESULTS_DIR,
    concurrency: int = CONCURRENCY,
) -> list[dict]:
    decisions    = collect_decisions(scenario_id, results_dir=results_dir)
    already_done = _load_existing_verdicts(csv_path)

    todo = [
        d for d in decisions
        if (d["run_id"], d["agent_id"], d["rubric_index"]) not in already_done
    ]
    skipped = len(decisions) - len(todo)
    if skipped:
        print(f"  [{scenario_id}] Resuming: {skipped} already judged, {len(todo)} remaining.")

    writer, fh = _open_csv_writer(csv_path)
    client     = openai.AsyncOpenAI(api_key=_API_KEY, base_url=_BASE_URL)
    semaphore  = asyncio.Semaphore(concurrency)

    print(f"  [{scenario_id}] Judging {len(todo)} decisions concurrently (limit={concurrency})…")

    # Run all judge calls concurrently, capped by the semaphore
    results: list[dict] = []
    t0 = time.monotonic()

    async def _run_and_record(decision: dict, index: int) -> None:
        result = await _judge_one(client, decision, semaphore, judge_model=judge_model)
        elapsed = time.monotonic() - t0
        rate    = (index + 1) / elapsed if elapsed > 0 else 0
        kw  = result["keyword_score"]
        vrd = result["judge_verdict"]
        print(
            f"  [{scenario_id}] {index+1}/{len(todo)}  "
            f"kw={kw}  judge={vrd}  rate={rate:.1f}/s  "
            f"run={result['run_id'][-8:]}  agent={result['agent_id']}  "
            f"rubric={result['rubric_topic']}",
            flush=True,
        )
        results.append(result)
        writer.writerow({k: result.get(k, "") for k in CSV_FIELDS})
        fh.flush()  # type: ignore[union-attr]

    await asyncio.gather(*[_run_and_record(d, i) for i, d in enumerate(todo)])

    fh.close()  # type: ignore[union-attr]
    return results


# ---------------------------------------------------------------------------
# Cohen's κ
# ---------------------------------------------------------------------------

def _cohens_kappa(
    verdicts: list[str],
    keyword_scores: list[int],
) -> float:
    """Compute Cohen's κ between LLM judge and keyword scorer."""
    n = len(verdicts)
    if n == 0:
        return float("nan")
    judge_pos   = [1 if v == "CORRECT" else 0 for v in verdicts]
    kw_pos      = keyword_scores
    agree       = sum(j == k for j, k in zip(judge_pos, kw_pos)) / n
    p_kw_pos    = sum(kw_pos) / n
    p_kw_neg    = 1 - p_kw_pos
    p_jdg_pos   = sum(judge_pos) / n
    p_jdg_neg   = 1 - p_jdg_pos
    p_chance    = p_kw_pos * p_jdg_pos + p_kw_neg * p_jdg_neg
    if p_chance >= 1.0:
        return 1.0
    return (agree - p_chance) / (1.0 - p_chance)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _write_summary(all_results: list[dict], summary_path: Path) -> None:
    from collections import defaultdict

    # Accuracy by condition
    by_condition: dict[str, dict[str, list]] = defaultdict(lambda: {"kw": [], "jdg": []})
    for r in all_results:
        cond = r["condition"]
        by_condition[cond]["kw"].append(r["keyword_score"])
        by_condition[cond]["jdg"].append(1 if r["judge_verdict"] == "CORRECT" else 0)

    lines = ["# Real-Agent Validation — Judge Summary\n"]

    lines.append("## Steering Accuracy (M1_real)\n")
    lines.append("| Condition | N | Keyword Acc | Judge Acc | Cohen's κ |")
    lines.append("|-----------|---|-------------|-----------|-----------|")

    for cond in sorted(by_condition.keys()):
        kw  = by_condition[cond]["kw"]
        jdg = by_condition[cond]["jdg"]
        n   = len(kw)
        kappa = _cohens_kappa(
            ["CORRECT" if j else "INCORRECT" for j in jdg], kw
        )
        lines.append(
            f"| {cond} | {n} | {sum(kw)/n:.1%} | {sum(jdg)/n:.1%} | {kappa:.3f} |"
        )
    lines.append("")

    # Rubric-level breakdown
    lines.append("## Per-Rubric Accuracy (Judge)\n")
    lines.append("| Condition | Agent | Rubric | N | Judge Acc |")
    lines.append("|-----------|-------|--------|---|-----------|")
    rubric_groups: dict[tuple, list] = defaultdict(list)
    for r in all_results:
        rubric_groups[(r["condition"], r["agent_id"], r["rubric_topic"])].append(
            1 if r["judge_verdict"] == "CORRECT" else 0
        )
    for key in sorted(rubric_groups.keys()):
        cond, aid, topic = key
        vals = rubric_groups[key]
        lines.append(
            f"| {cond} | {aid} | {topic} | {len(vals)} | {sum(vals)/len(vals):.1%} |"
        )
    lines.append("")

    # Steering coverage
    lines.append("## Steering Coverage\n")
    lines.append(
        "Coverage = fraction of rubric slots that received a steering response.\n"
    )
    from experiments_real_agent.scenario_defs import REAL_SCENARIOS
    for scenario_id, scenario in REAL_SCENARIOS.items():
        expected_per_trial = sum(len(s.rubrics) for s in scenario.agents)
        # actual responses per run
        run_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in all_results:
            if r["rubric_topic"] != "extra":
                run_counts[r["run_id"]][r["agent_id"]] += 1
        coverage_vals: list[float] = []
        for run_id, agent_counts in run_counts.items():
            actual = sum(agent_counts.values())
            coverage_vals.append(actual / expected_per_trial)
        if coverage_vals:
            import statistics
            lines.append(
                f"Scenario {scenario_id}: "
                f"mean coverage {statistics.mean(coverage_vals):.1%}, "
                f"min {min(coverage_vals):.1%}, max {max(coverage_vals):.1%}\n"
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary written to {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main(
    scenario_ids: list[str],
    judge_models: list[str],
    results_dir: Path,
    concurrency: int = CONCURRENCY,
) -> None:
    """Run judging for every (scenario, judge_model) combination.

    Per-model CSVs:  judge_results_{scenario}_{model_slug}.csv
    Comparison report: judge_comparison.md  (multi-model) or judge_summary.md (single)
    """
    # Collect per-model results for the comparison table
    model_results: dict[str, list[dict]] = {}

    for judge_model in judge_models:
        slug = _model_slug(judge_model)
        print(f"\n{'='*60}", flush=True)
        print(f"Judge model: {judge_model}  (concurrency={concurrency})", flush=True)
        print(f"{'='*60}", flush=True)

        all_results_for_model: list[dict] = []
        for scenario_id in scenario_ids:
            csv_path = results_dir / f"judge_results_{scenario_id}_{slug}.csv"
            print(f"\n--- Judging {scenario_id} → {csv_path.name} ---", flush=True)
            results = await judge_scenario(
                scenario_id, csv_path,
                judge_model=judge_model, results_dir=results_dir,
                concurrency=concurrency,
            )
            all_results_for_model.extend(results)

        # Reload CSV to include any previously cached verdicts
        all_results_for_model = []
        for scenario_id in scenario_ids:
            csv_path = results_dir / f"judge_results_{scenario_id}_{slug}.csv"
            if csv_path.exists():
                with open(csv_path, newline="") as fh:
                    for r in csv.DictReader(fh):
                        r["keyword_score"] = int(r.get("keyword_score", 0))
                        all_results_for_model.append(r)

        model_results[judge_model] = all_results_for_model

        # Per-model summary (single model path or final leg of multi-model)
        summary_slug = results_dir / f"judge_summary_{slug}.md"
        _write_summary(all_results_for_model, summary_slug)

    # Multi-model comparison table
    if len(judge_models) > 1:
        _write_comparison(model_results, results_dir / "judge_comparison.md")


def _write_comparison(
    model_results: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """Write a Markdown table comparing judge accuracy across models."""
    from collections import defaultdict

    lines = ["# Real-Agent Judge Comparison — Multi-Model\n"]
    lines.append("## Accuracy by Condition\n")
    lines.append(
        "| Judge Model | Condition | N | Keyword Acc | Judge Acc | Cohen's κ |"
    )
    lines.append(
        "|-------------|-----------|---|-------------|-----------|-----------|"
    )

    for judge_model, results in model_results.items():
        by_condition: dict[str, dict] = defaultdict(lambda: {"kw": [], "jdg": []})
        for r in results:
            cond = r["condition"]
            by_condition[cond]["kw"].append(int(r["keyword_score"]))
            by_condition[cond]["jdg"].append(
                1 if r["judge_verdict"] == "CORRECT" else 0
            )
        for cond in sorted(by_condition):
            kw  = by_condition[cond]["kw"]
            jdg = by_condition[cond]["jdg"]
            n   = len(kw)
            kappa = _cohens_kappa(
                ["CORRECT" if j else "INCORRECT" for j in jdg], kw
            )
            lines.append(
                f"| {judge_model} | {cond} | {n} "
                f"| {sum(kw)/n:.1%} | {sum(jdg)/n:.1%} | {kappa:.3f} |"
            )

    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nComparison table written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge for real-agent results")
    parser.add_argument(
        "--scenario", nargs="+",
        default=list(REAL_SCENARIOS.keys()),
    )
    parser.add_argument(
        "--models", nargs="+",
        default=[_JUDGE_MODEL],
        metavar="MODEL",
        help=(
            "OpenRouter model IDs to use as judge (default: %(default)s). "
            "Pass multiple to generate a comparison table, e.g. "
            "--models anthropic/claude-haiku-4-5 anthropic/claude-3-5-sonnet openai/gpt-4o-mini"
        ),
    )
    parser.add_argument(
        "--results-dir", default=None,
        help=(
            "Directory containing JSONL trial logs to judge "
            "(default: results_real_agent_haiku if it exists, else results_real_agent)"
        ),
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY,
        metavar="N",
        help=f"Max simultaneous judge API calls per model (default: {CONCURRENCY}). "
             "Raise to 12–16 if OpenRouter isn't rate-limiting you.",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        haiku_dir = Path("results_real_agent_haiku")
        results_dir = haiku_dir if haiku_dir.exists() else Path("results_real_agent")

    print(f"Reading results from: {results_dir}", flush=True)
    asyncio.run(_main(args.scenario, args.models, results_dir, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
