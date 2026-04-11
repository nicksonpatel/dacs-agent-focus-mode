"""LLM-as-judge for the DACS WebArena benchmark experiment.

Reads STEERING_RESPONSE events from JSONL trial logs, pairs each response
with its rubric from ``WebArenaAgentSpec.rubrics`` (assigned sequentially per
agent), and evaluates correctness using both keyword matching and an LLM judge.

The judge uses OpenRouter so no MiniMax budget is required.

Outputs
-------
    results_webarena/judge_results_{scenario}_{model_slug}.csv — per-decision rows
    results_webarena/judge_summary_{model_slug}.md              — accuracy summary

Usage
-----
    set -a && source .env && set +a

    # Default (Haiku judge)
    python -m experiments_webarena.judge --scenario wa1_n3

    # Multi-model comparison
    python -m experiments_webarena.judge \
        --models anthropic/claude-haiku-4-5 openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

from experiments_webarena.scenario_defs import WEB_SCENARIOS, WebArenaScenario, DecisionRubric

_RESULTS_DIR = Path("results_webarena")
_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-haiku-4-5")
_API_KEY     = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY", "")
_BASE_URL    = "https://openrouter.ai/api/v1"

if not _API_KEY:
    raise RuntimeError(
        "Set OPENROUTER_API_KEY (or OR_API_KEY) in .env to run the judge."
    )

CONCURRENCY = 8


def _model_slug(model: str) -> str:
    return re.sub(r"[/:]", "_", model)


CSV_FIELDS = [
    "scenario", "run_id", "condition", "agent_id", "rubric_index", "rubric_topic",
    "actual_question", "keyword_score", "judge_verdict", "judge_reason",
    "orchestrator_state", "correct_keywords", "response_text",
    "webarena_task_id", "webarena_site", "webarena_eval",
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
    """Return {request_id: question_text} from STEERING_REQUEST events."""
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
    """Pair each STEERING_RESPONSE with its rubric and question text.

    Rubric assignment: sequential per-agent.  Agent a1's first
    STEERING_RESPONSE → rubric[0], second → rubric[1], etc.
    """
    scenario: WebArenaScenario = WEB_SCENARIOS[scenario_id]

    # Build lookup maps from the scenario spec
    rubric_map: dict[str, list[DecisionRubric]] = {
        spec.agent_id: spec.rubrics for spec in scenario.agents
    }
    task_map: dict[str, str] = {
        spec.agent_id: spec.task_description for spec in scenario.agents
    }
    site_map: dict[str, str] = {
        spec.agent_id: spec.webarena_site for spec in scenario.agents
    }
    task_id_map: dict[str, int] = {
        spec.agent_id: spec.webarena_task_id for spec in scenario.agents
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
                wa_eval      = rubric.webarena_eval
                judge_ctx    = rubric.judge_context
            else:
                rubric_topic = "extra"
                rubric_kws   = []
                wa_eval      = ""
                judge_ctx    = ""

            request_id      = event.get("request_id", "")
            actual_question = req_map.get(request_id, "")
            response_text   = event.get("response_text", "")
            keyword_score   = (
                1 if any(kw.lower() in response_text.lower() for kw in rubric_kws)
                else 0
            )

            all_decisions.append({
                "scenario":           scenario_id,
                "run_id":             run_id,
                "condition":          condition,
                "agent_id":           aid,
                "rubric_index":       ptr,
                "rubric_topic":       rubric_topic,
                "actual_question":    actual_question,
                "keyword_score":      keyword_score,
                "correct_keywords":   rubric_kws,
                "response_text":      response_text,
                "orchestrator_state": event.get("orchestrator_state", ""),
                "domain_hint":        task_map.get(aid, "")[:80],
                "judge_context":      judge_ctx,
                "webarena_task_id":   task_id_map.get(aid, 0),
                "webarena_site":      site_map.get(aid, ""),
                "webarena_eval":      wa_eval,
            })
            dp_pointer[aid] = ptr + 1

    return all_decisions


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_existing_verdicts(csv_path: Path) -> set[tuple[str, str, int]]:
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
    "You are an expert evaluator assessing whether an AI orchestrator correctly "
    "guided a specialist web-task agent in a multi-agent workflow.\n\n"
    "The agent is performing a WebArena information-retrieval task.  It encountered "
    "an ambiguous decision and asked the orchestrator for guidance.  Your job is to "
    "evaluate whether the orchestrator's response correctly resolves the ambiguity.\n\n"
    "You will receive:\n"
    "1. The agent's web task description (including simulated page data)\n"
    "2. The exact question the agent asked\n"
    "3. The orchestrator's response\n"
    "4. A rubric explaining what a correct answer looks like\n\n"
    "CORRECT: the orchestrator's response substantively addresses the decision point "
    "and its recommendation is consistent with the rubric.\n"
    "INCORRECT: the response fails to address the question, is vague, irrelevant, "
    "or contradicts the rubric.\n\n"
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
    site_label = (
        f"WebArena site: {decision['webarena_site']} "
        f"(task_id={decision['webarena_task_id']})"
    )
    return (
        f"Agent web task ({site_label}):\n{decision['domain_hint']}\n\n"
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
                    print(f"  [judge] 429, waiting {wait}s (attempt {attempt+1}/8)…")
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

    print(f"  [{scenario_id}] Judging {len(todo)} decisions (limit={concurrency})…")

    results: list[dict] = []
    t0 = time.monotonic()

    async def _run_and_record(decision: dict, index: int) -> None:
        result  = await _judge_one(client, decision, semaphore, judge_model=judge_model)
        elapsed = time.monotonic() - t0
        rate    = (index + 1) / elapsed if elapsed > 0 else 0
        kw  = result["keyword_score"]
        vrd = result["judge_verdict"]
        print(
            f"  [{scenario_id}] {index+1}/{len(todo)}  "
            f"kw={kw}  judge={vrd}  rate={rate:.1f}/s  "
            f"run={result['run_id'][-8:]}  agent={result['agent_id']}  "
            f"rubric={result['rubric_topic']}  site={result['webarena_site']}",
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

def _cohens_kappa(verdicts: list[str], keyword_scores: list[int]) -> float:
    n = len(verdicts)
    if n == 0:
        return float("nan")
    judge_pos = [1 if v == "CORRECT" else 0 for v in verdicts]
    kw_pos    = keyword_scores
    agree     = sum(j == k for j, k in zip(judge_pos, kw_pos)) / n
    p_kw_pos  = sum(kw_pos) / n
    p_kw_neg  = 1 - p_kw_pos
    p_jdg_pos = sum(judge_pos) / n
    p_jdg_neg = 1 - p_jdg_pos
    p_chance  = p_kw_pos * p_jdg_pos + p_kw_neg * p_jdg_neg
    if p_chance >= 1.0:
        return 1.0
    return (agree - p_chance) / (1.0 - p_chance)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _write_summary(all_results: list[dict], summary_path: Path) -> None:
    by_condition: dict[str, dict] = defaultdict(lambda: {"kw": [], "jdg": []})
    for r in all_results:
        cond = r["condition"]
        by_condition[cond]["kw"].append(int(r["keyword_score"]))
        by_condition[cond]["jdg"].append(1 if r["judge_verdict"] == "CORRECT" else 0)

    lines = ["# WebArena Benchmark — Judge Summary\n"]
    lines.append("## Steering Accuracy (M1_webArena)\n")
    lines.append("| Condition | N | Keyword Acc | Judge Acc | Cohen's κ |")
    lines.append("|-----------|---|-------------|-----------|-----------|")
    for cond in sorted(by_condition.keys()):
        kw  = by_condition[cond]["kw"]
        jdg = by_condition[cond]["jdg"]
        n   = len(kw)
        kappa = _cohens_kappa(["CORRECT" if j else "INCORRECT" for j in jdg], kw)
        lines.append(
            f"| {cond} | {n} | {sum(kw)/n:.1%} | {sum(jdg)/n:.1%} | {kappa:.3f} |"
        )
    lines.append("")

    lines.append("## Per-Site Accuracy (Judge)\n")
    lines.append("| Condition | Site | Agent | Rubric | N | Judge Acc |")
    lines.append("|-----------|------|-------|--------|---|-----------|")
    site_groups: dict[tuple, list] = defaultdict(list)
    for r in all_results:
        site_groups[
            (r["condition"], r.get("webarena_site",""), r["agent_id"], r["rubric_topic"])
        ].append(1 if r["judge_verdict"] == "CORRECT" else 0)
    for key in sorted(site_groups.keys()):
        cond, site, aid, topic = key
        vals = site_groups[key]
        lines.append(
            f"| {cond} | {site} | {aid} | {topic} | {len(vals)} | {sum(vals)/len(vals):.1%} |"
        )
    lines.append("")

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
    for judge_model in judge_models:
        slug = _model_slug(judge_model)
        print(f"\n{'='*60}")
        print(f"Judge model: {judge_model}  (concurrency={concurrency})")
        print(f"{'='*60}")

        all_results: list[dict] = []
        for scenario_id in scenario_ids:
            csv_path = results_dir / f"judge_results_{scenario_id}_{slug}.csv"
            print(f"\n--- Judging {scenario_id} → {csv_path.name} ---")
            await judge_scenario(
                scenario_id, csv_path,
                judge_model=judge_model, results_dir=results_dir,
                concurrency=concurrency,
            )

        # Reload from CSV to include previously cached verdicts
        for scenario_id in scenario_ids:
            csv_path = results_dir / f"judge_results_{scenario_id}_{slug}.csv"
            if csv_path.exists():
                with open(csv_path, newline="") as fh:
                    for r in csv.DictReader(fh):
                        r["keyword_score"] = int(r.get("keyword_score", 0))
                        all_results.append(r)

        summary_path = results_dir / f"judge_summary_{slug}.md"
        _write_summary(all_results, summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM judge for WebArena DACS benchmark results"
    )
    parser.add_argument(
        "--scenario", nargs="+",
        default=list(WEB_SCENARIOS.keys()),
    )
    parser.add_argument(
        "--models", nargs="+",
        default=[_JUDGE_MODEL],
        metavar="MODEL",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory containing JSONL trial logs (default: results_webarena)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY, metavar="N",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _RESULTS_DIR
    print(f"Reading results from: {results_dir}")
    asyncio.run(_main(args.scenario, args.models, results_dir, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
