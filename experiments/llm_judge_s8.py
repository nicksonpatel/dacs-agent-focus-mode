"""LLM-as-judge validation for s8_n3_dense_d3 (Phase 3).

Samples 70 decisions uniformly at random from all 20 s8 trial JSONL files
(10 DACS + 10 baseline × 45 decisions each = 900 total).

For each sampled decision the judge receives:
  - the steering question (question_fragment from the task spec)
  - the orchestrator's response
  - the agent's task domain (for context)

It returns CORRECT or INCORRECT, independent of the keyword list.

Outputs:
  results/llm_judge_s8.csv   — per-decision rows
  results/llm_judge_s8_summary.md  — aggregate stats (agreement, κ)

Usage:
    MINIMAX_API_KEY=<key> python -m experiments.llm_judge_s8
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import statistics
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_N   = 70
SCENARIO   = "s8_n3_dense_d3"
RESULTS_DIR = Path("results")
SEED        = 42

MODEL       = os.environ.get("DACS_MODEL", "MiniMax-M2.7")
API_KEY     = os.environ["MINIMAX_API_KEY"]
BASE_URL    = "https://api.minimax.io/anthropic"

# Concurrency: use 1 to avoid rate limits
CONCURRENCY = 1

# ---------------------------------------------------------------------------
# Task-spec reference (question fragments + keywords + domain label per DP)
# We reconstruct this from task_suite rather than hardcoding
# ---------------------------------------------------------------------------
from experiments.task_suite import SCENARIOS

SPEC = SCENARIOS[SCENARIO]

# Build a lookup: agent_id -> [(question_fragment, answer_keywords, domain_hint), ...]
AGENT_DPS: dict[str, list[tuple[str, list[str], str]]] = {}
AGENT_DOMAIN: dict[str, str] = {}

for agent_spec in SPEC.agents:
    aid = agent_spec.agent_id
    AGENT_DOMAIN[aid] = agent_spec.task_description[:80]
    AGENT_DPS[aid] = [
        (dp.question_fragment, dp.answer_keywords, agent_spec.task_description[:80])
        for dp in agent_spec.decision_points
    ]

# ---------------------------------------------------------------------------
# Step 1: Collect all decisions from all s8 JSONL files
# ---------------------------------------------------------------------------

def collect_all_decisions() -> list[dict]:
    """Parse every s8 JSONL file and return one record per STEERING_RESPONSE."""
    all_decisions = []
    jsonl_files = sorted(RESULTS_DIR.glob(f"{SCENARIO}_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found for {SCENARIO} in {RESULTS_DIR}")

    for fpath in jsonl_files:
        run_id = fpath.stem
        # Determine condition from filename
        condition = "dacs" if "_dacs_" in run_id else "baseline"

        # Load events
        events = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        # Track per-agent dp pointer, same logic as metrics.py
        dp_pointer: dict[str, int] = {aid: 0 for aid in AGENT_DPS}

        for event in events:
            if event.get("event") != "STEERING_RESPONSE":
                continue
            aid = event["agent_id"]
            ptr = dp_pointer.get(aid, 0)
            dps = AGENT_DPS.get(aid, [])

            if ptr >= len(dps):
                continue  # shouldn't happen but guard

            question_fragment, answer_keywords, domain_hint = dps[ptr]
            response_text = event.get("response_text", "")
            keyword_score = 1 if any(kw.lower() in response_text.lower() for kw in answer_keywords) else 0

            all_decisions.append({
                "run_id":            run_id,
                "condition":         condition,
                "agent_id":          aid,
                "dp_index":          ptr,
                "keyword_score":     keyword_score,
                "answer_keywords":   answer_keywords,
                "question_fragment": question_fragment,
                "domain_hint":       domain_hint,
                "response_text":     response_text,
                "orchestrator_state": event.get("orchestrator_state", ""),
            })

            dp_pointer[aid] = ptr + 1

    return all_decisions


# ---------------------------------------------------------------------------
# Step 2: Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert technical evaluator. Your task is to assess whether an AI orchestrator's response correctly addresses a specific steering question within a multi-agent workflow.

You will receive:
1. The agent's task domain
2. The steering question the orchestrator was asked to resolve
3. The orchestrator's response

Respond with exactly one word: CORRECT or INCORRECT.

CORRECT means the response meaningfully and substantively addresses the core decision in the question — it gives a technically valid answer or makes a defensible choice.
INCORRECT means the response fails to address the question, gives an irrelevant answer, or primarily discusses a different topic.

Do not explain your verdict. Output only: CORRECT or INCORRECT"""


def build_judge_prompt(decision: dict) -> str:
    kws = ", ".join(decision["answer_keywords"][:5])
    return f"""Agent domain: {decision['domain_hint']}

Steering question: {decision['question_fragment']}

Orchestrator response:
{decision['response_text'][:1200]}

Is this response CORRECT or INCORRECT?"""


# ---------------------------------------------------------------------------
# Step 3: Async judge calls
# ---------------------------------------------------------------------------

async def judge_one(
    client: anthropic.AsyncAnthropic,
    decision: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Call the LLM judge for a single decision. Returns verdict."""
    async with semaphore:
        prompt = build_judge_prompt(decision)
        for attempt in range(4):
            try:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=128,
                    system=JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = next(
                    (block.text for block in resp.content if block.type == "text"),
                    ""
                ).strip().upper()
                # Accept CORRECT anywhere in response (model may add preamble)
                if "CORRECT" in raw and "INCORRECT" not in raw:
                    verdict = "CORRECT"
                elif "INCORRECT" in raw:
                    verdict = "INCORRECT"
                else:
                    # Fallback: treat ambiguous / empty as INCORRECT
                    print(f"  [WARN] Ambiguous judge response: {repr(raw[:60])}")
                    verdict = "INCORRECT"
                break
            except anthropic.RateLimitError:
                wait = 10 * (2 ** attempt)
                print(f"  [RATE LIMIT] attempt {attempt+1}, waiting {wait}s...")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  [WARN] Judge call failed for {decision['run_id']} a{decision['agent_id']} dp{decision['dp_index']}: {e}")
                verdict = "ERROR"
                break
        else:
            verdict = "ERROR"

        return {**decision, "judge_verdict": verdict}


async def run_judge(sample: list[dict]) -> list[dict]:
    client = anthropic.AsyncAnthropic(api_key=API_KEY, base_url=BASE_URL)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [judge_one(client, d, semaphore) for d in sample]

    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        verdict = result["judge_verdict"]
        kw = result["keyword_score"]
        agree = "✓" if (verdict == "CORRECT") == bool(kw) else "✗"
        print(f"  [{i+1:2d}/{SAMPLE_N}] {result['run_id'][-12:]} a{result['agent_id']} dp{result['dp_index']:2d}  kw={kw} judge={verdict} {agree}")
        results.append(result)

    await client.close()
    return results


# ---------------------------------------------------------------------------
# Step 4: Compute statistics
# ---------------------------------------------------------------------------

def cohen_kappa(results: list[dict]) -> float:
    """Compute Cohen's κ between keyword_score and judge_verdict."""
    valid = [r for r in results if r["judge_verdict"] != "ERROR"]
    n = len(valid)
    if n == 0:
        return 0.0

    kw_pos   = sum(r["keyword_score"] for r in valid) / n
    kw_neg   = 1 - kw_pos
    jdg_pos  = sum(1 for r in valid if r["judge_verdict"] == "CORRECT") / n
    jdg_neg  = 1 - jdg_pos

    p_e = kw_pos * jdg_pos + kw_neg * jdg_neg

    agree = sum(
        1 for r in valid
        if (r["judge_verdict"] == "CORRECT") == bool(r["keyword_score"])
    )
    p_o = agree / n

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def print_and_save_summary(results: list[dict]):
    valid = [r for r in results if r["judge_verdict"] != "ERROR"]
    n = len(valid)

    agreement = sum(1 for r in valid if (r["judge_verdict"] == "CORRECT") == bool(r["keyword_score"]))
    agree_rate = agreement / n if n else 0

    kw_acc    = sum(r["keyword_score"] for r in valid) / n if n else 0
    judge_acc = sum(1 for r in valid if r["judge_verdict"] == "CORRECT") / n if n else 0

    fn = sum(1 for r in valid if r["keyword_score"] == 0 and r["judge_verdict"] == "CORRECT")
    fp = sum(1 for r in valid if r["keyword_score"] == 1 and r["judge_verdict"] == "INCORRECT")
    kappa = cohen_kappa(valid)

    # Per-condition breakdown
    for cond in ("dacs", "baseline"):
        sub = [r for r in valid if r["condition"] == cond]
        if not sub:
            continue
        nc = len(sub)
        ag = sum(1 for r in sub if (r["judge_verdict"] == "CORRECT") == bool(r["keyword_score"]))
        ka = sum(r["keyword_score"] for r in sub) / nc
        ja = sum(1 for r in sub if r["judge_verdict"] == "CORRECT") / nc
        print(f"  {cond:8s}  n={nc}  kw_acc={ka*100:.1f}%  judge_acc={ja*100:.1f}%  agreement={ag/nc*100:.1f}%")

    print(f"\n  Overall n={n}  agreement={agree_rate*100:.1f}%  κ={kappa:.3f}")
    print(f"  False negatives (kw=0, judge=CORRECT): {fn}")
    print(f"  False positives (kw=1, judge=INCORRECT): {fp}")
    print(f"  Keyword accuracy: {kw_acc*100:.1f}%  Judge accuracy: {judge_acc*100:.1f}%")

    # Save CSV
    csv_path = RESULTS_DIR / "llm_judge_s8.csv"
    fieldnames = ["run_id", "condition", "agent_id", "dp_index",
                  "keyword_score", "judge_verdict", "orchestrator_state",
                  "answer_keywords", "question_fragment", "response_text"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(valid, key=lambda x: (x["run_id"], x["agent_id"], x["dp_index"])):
            r2 = dict(r)
            r2["answer_keywords"] = "|".join(r2["answer_keywords"])
            writer.writerow(r2)
    print(f"\n  Saved: {csv_path}")

    # Save summary markdown
    se = (agree_rate * (1 - agree_rate) / n) ** 0.5 if n > 1 else 0
    se_pct = se * 100
    md_path = RESULTS_DIR / "llm_judge_s8_summary.md"
    with open(md_path, "w") as f:
        f.write(f"""# LLM-as-Judge Validation — s8_n3_dense_d3

## Setup

| | |
|---|---|
| Scenario | s8_n3_dense_d3 (N=3 agents, D=15 decisions each) |
| Total decisions available | 900 (20 trials × 45 decisions each) |
| Sample size | {n} (random sample, seed={SEED}) |
| Judge model | {MODEL} (same model, independent call with structured prompt) |
| Keyword metric model | {MODEL} |

## Overall Agreement

| Metric | Value |
|---|---|
| Agreement rate | **{agree_rate*100:.1f}%** (±{se_pct:.1f}%) |
| Cohen's κ | **{kappa:.3f}** |
| Keyword accuracy | {kw_acc*100:.1f}% |
| Judge accuracy | {judge_acc*100:.1f}% |
| False negatives (kw=0, judge=CORRECT) | {fn} ({fn/n*100:.1f}%) |
| False positives (kw=1, judge=INCORRECT) | {fp} ({fp/n*100:.1f}%) |

## Per-Condition Breakdown

| Condition | n | Keyword acc | Judge acc | Agreement |
|---|---|---|---|---|
""")
        for cond in ("dacs", "baseline"):
            sub = [r for r in valid if r["condition"] == cond]
            if not sub:
                continue
            nc = len(sub)
            ag = sum(1 for r in sub if (r["judge_verdict"] == "CORRECT") == bool(r["keyword_score"]))
            ka = sum(r["keyword_score"] for r in sub) / nc
            ja = sum(1 for r in sub if r["judge_verdict"] == "CORRECT") / nc
            f.write(f"| {cond} | {nc} | {ka*100:.1f}% | {ja*100:.1f}% | {ag/nc*100:.1f}% |\n")
        f.write(f"""
## Interpretation

{'Substantial agreement (κ≥0.80) — keyword matching is a valid proxy for LLM-judged correctness.' if kappa >= 0.80 else f'κ={kappa:.3f} — see disagreement analysis below.'}

*Full per-decision results: `results/llm_judge_s8.csv`*
""")
    print(f"  Saved: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print(f"=== LLM Judge — {SCENARIO} (n={SAMPLE_N}) ===\n")

    print("Collecting all decisions from JSONL files...")
    all_decisions = collect_all_decisions()
    print(f"  Total decisions found: {len(all_decisions)}")

    # Stratified sample: equal from dacs and baseline
    dacs_pool     = [d for d in all_decisions if d["condition"] == "dacs"]
    baseline_pool = [d for d in all_decisions if d["condition"] == "baseline"]
    rng = random.Random(SEED)
    sample = (
        rng.sample(dacs_pool,     SAMPLE_N // 2) +
        rng.sample(baseline_pool, SAMPLE_N // 2)
    )
    rng.shuffle(sample)
    print(f"  Sampled: {len(sample)} decisions ({SAMPLE_N//2} DACS + {SAMPLE_N//2} baseline)\n")

    print(f"Running judge ({CONCURRENCY} concurrent calls)...")
    t0 = time.monotonic()
    results = await run_judge(sample)
    elapsed = time.monotonic() - t0
    print(f"\n  Done in {elapsed:.1f}s\n")

    print("=== Results ===")
    print_and_save_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
