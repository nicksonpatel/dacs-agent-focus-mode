"""LLM-as-judge validation for Phase 3 (decision density scaling).

Budget: 1100 API calls total.
  - s8_n3_dense_d3: FULL validation (20 trials × 45 decisions = 900 calls)
  - s7_n5_dense_d2: stratified 200-sample (100 DACS + 100 baseline), seed=42

Incremental / resumable:
  Each verdict is written to CSV immediately after the API call.
  Re-running skips already-judged decisions (matched by run_id+agent_id+dp_index).

Monitor live progress:
    tail -f logs/llm_judge_phase3.log
    # or:
    watch -n 10 'wc -l results/llm_judge_phase3_s8.csv results/llm_judge_phase3_s7.csv'

Usage:
    set -a && source .env && set +a
    python -u -m experiments.llm_judge_phase3 2>&1 | tee -a logs/llm_judge_phase3.log
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
SEED        = 42

MODEL    = os.environ.get("DACS_MODEL", "MiniMax-M2.7")
API_KEY  = os.environ["MINIMAX_API_KEY"]
BASE_URL = "https://api.minimax.io/anthropic"

CONCURRENCY = 1   # serial — avoid rate limits
PROGRESS_EVERY = 50  # print running stats every N verdicts

# Per-scenario config: sample_n=None means judge ALL decisions
SCENARIO_CONFIG = {
    "s8_n3_dense_d3": {"sample_n": 100},   # 100 stratified (50 DACS + 50 baseline)
    "s7_n5_dense_d2": {"sample_n": 200},   # 200 stratified (100 DACS + 100 baseline)
}

CSV_FIELDS = [
    "run_id", "condition", "agent_id", "dp_index",
    "keyword_score", "judge_verdict", "judge_reason", "orchestrator_state",
    "answer_keywords", "question_fragment", "response_text",
]

# ---------------------------------------------------------------------------
# Build agent decision-point lookup from task_suite
# ---------------------------------------------------------------------------
from experiments.task_suite import SCENARIOS


def build_agent_dps(scenario_id: str) -> dict[str, list[tuple[str, list[str], str]]]:
    spec = SCENARIOS[scenario_id]
    result: dict[str, list[tuple[str, list[str], str]]] = {}
    for agent_spec in spec.agents:
        aid = agent_spec.agent_id
        result[aid] = [
            (dp.question_fragment, dp.answer_keywords, agent_spec.task_description[:80])
            for dp in agent_spec.decision_points
        ]
    return result


# ---------------------------------------------------------------------------
# Collect decisions from JSONL files
# ---------------------------------------------------------------------------

def collect_decisions(scenario_id: str) -> list[dict]:
    agent_dps = build_agent_dps(scenario_id)
    all_decisions: list[dict] = []
    jsonl_files = sorted(RESULTS_DIR.glob(f"{scenario_id}_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files for {scenario_id} in {RESULTS_DIR}")

    for fpath in jsonl_files:
        run_id    = fpath.stem
        condition = "dacs" if "_dacs_" in run_id else "baseline"
        events: list[dict] = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        dp_pointer: dict[str, int] = {aid: 0 for aid in agent_dps}
        for event in events:
            if event.get("event") != "STEERING_RESPONSE":
                continue
            aid = event["agent_id"]
            ptr = dp_pointer.get(aid, 0)
            dps = agent_dps.get(aid, [])
            if ptr >= len(dps):
                continue
            question_fragment, answer_keywords, domain_hint = dps[ptr]
            response_text = event.get("response_text", "")
            keyword_score = 1 if any(
                kw.lower() in response_text.lower() for kw in answer_keywords
            ) else 0
            all_decisions.append({
                "scenario":           scenario_id,
                "run_id":             run_id,
                "condition":          condition,
                "agent_id":           aid,
                "dp_index":           ptr,
                "keyword_score":      keyword_score,
                "answer_keywords":    answer_keywords,
                "question_fragment":  question_fragment,
                "domain_hint":        domain_hint,
                "response_text":      response_text,
                "orchestrator_state": event.get("orchestrator_state", ""),
            })
            dp_pointer[aid] = ptr + 1
    return all_decisions


def sample_decisions(decisions: list[dict], sample_n: int | None, rng: random.Random) -> list[dict]:
    if sample_n is None:
        return list(decisions)
    dacs_pool     = [d for d in decisions if d["condition"] == "dacs"]
    baseline_pool = [d for d in decisions if d["condition"] == "baseline"]
    half   = sample_n // 2
    sample = (rng.sample(dacs_pool,     min(half, len(dacs_pool))) +
              rng.sample(baseline_pool, min(half, len(baseline_pool))))
    rng.shuffle(sample)
    return sample


# ---------------------------------------------------------------------------
# Resume: load already-judged decisions from an existing CSV
# ---------------------------------------------------------------------------

def load_existing_verdicts(csv_path: Path) -> set[tuple[str, str, int]]:
    """Return set of (run_id, agent_id, dp_index) already in the CSV."""
    done: set[tuple[str, str, int]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["run_id"], row["agent_id"], int(row["dp_index"])))
    return done


def open_csv_writer(csv_path: Path) -> tuple[csv.DictWriter, object]:
    """Open CSV for append (write header only if new). Returns (writer, file)."""
    is_new = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if is_new:
        writer.writeheader()
        fh.flush()
    return writer, fh


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are an expert technical evaluator. Your task is to assess whether an AI "
    "orchestrator's response correctly addresses a specific steering question within "
    "a multi-agent workflow.\n\n"
    "You will receive:\n"
    "1. The agent's task domain\n"
    "2. The steering question the orchestrator was asked to resolve\n"
    "3. The orchestrator's response\n\n"
    "CORRECT means the response meaningfully and substantively addresses the core "
    "decision in the question — it gives a technically valid answer or makes a "
    "defensible choice.\n"
    "INCORRECT means the response fails to address the question, gives an irrelevant "
    "answer, or primarily discusses a different topic.\n\n"
    "Respond in exactly this XML format:\n"
    "<reason>one sentence explanation</reason>\n"
    "<verdict>CORRECT</verdict>\n"
    "or\n"
    "<reason>one sentence explanation</reason>\n"
    "<verdict>INCORRECT</verdict>"
)


def build_judge_prompt(decision: dict) -> str:
    return (
        f"Agent domain: {decision['domain_hint']}\n\n"
        f"Steering question: {decision['question_fragment']}\n\n"
        f"Orchestrator response:\n{decision['response_text'][:1200]}\n\n"
        f"Is this response CORRECT or INCORRECT?"
    )


# ---------------------------------------------------------------------------
# Running stats (printed every PROGRESS_EVERY calls)
# ---------------------------------------------------------------------------

class RunningStats:
    def __init__(self, total: int, scenario_id: str) -> None:
        self.total      = total
        self.scenario   = scenario_id
        self.done       = 0
        self.agree      = 0
        self.kw_hits    = 0
        self.judge_hits = 0
        self.t0         = time.monotonic()

    def update(self, kw: int, verdict: str) -> None:
        self.done += 1
        self.kw_hits    += kw
        self.judge_hits += 1 if verdict == "CORRECT" else 0
        self.agree      += 1 if (verdict == "CORRECT") == bool(kw) else 0

    def print_progress(self) -> None:
        elapsed = time.monotonic() - self.t0
        rate    = self.done / elapsed if elapsed > 0 else 0
        eta     = (self.total - self.done) / rate if rate > 0 else 0
        agree_r = self.agree / self.done if self.done else 0
        print(
            f"  PROGRESS [{self.scenario}]  {self.done}/{self.total} "
            f"({self.done/self.total*100:.1f}%)  "
            f"agree={agree_r*100:.1f}%  "
            f"rate={rate:.1f}/s  ETA={eta/60:.1f}min",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Per-scenario incremental judge loop
# ---------------------------------------------------------------------------

async def judge_scenario(
    scenario_id: str,
    sample: list[dict],
    csv_path: Path,
    already_done: set[tuple[str, str, int]],
) -> list[dict]:
    """Judge all decisions in sample, skip already-done, write CSV incrementally."""

    todo = [
        d for d in sample
        if (d["run_id"], d["agent_id"], d["dp_index"]) not in already_done
    ]
    skipped = len(sample) - len(todo)
    if skipped:
        print(f"  Resuming: {skipped} already judged, {len(todo)} remaining.", flush=True)

    writer, fh = open_csv_writer(csv_path)
    client     = anthropic.AsyncAnthropic(api_key=API_KEY, base_url=BASE_URL)
    semaphore  = asyncio.Semaphore(CONCURRENCY)
    stats      = RunningStats(len(todo), scenario_id)
    results: list[dict] = []

    async def _judge_one(decision: dict, seq: int) -> dict:
        async with semaphore:
            prompt  = build_judge_prompt(decision)
            verdict = "ERROR"
            judge_reason = ""
            for attempt in range(5):
                try:
                    resp = await client.messages.create(
                        model=MODEL,
                        max_tokens=512,
                        system=JUDGE_SYSTEM,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = "".join(
                        b.text for b in resp.content if b.type == "text"
                    ).strip()
                    # Parse <verdict>...</verdict> tag first (preferred format)
                    m = re.search(r'<verdict>\s*(CORRECT|INCORRECT)\s*</verdict>', raw, re.IGNORECASE)
                    r = re.search(r'<reason>(.*?)</reason>', raw, re.IGNORECASE | re.DOTALL)
                    if r:
                        judge_reason = r.group(1).strip()
                    if m:
                        verdict = m.group(1).upper()
                    elif "CORRECT" in raw.upper() and "INCORRECT" not in raw.upper():
                        verdict = "CORRECT"
                    elif "INCORRECT" in raw.upper():
                        verdict = "INCORRECT"
                    else:
                        print(f"  [WARN] Ambiguous: {repr(raw[:60])}", flush=True)
                        verdict = "INCORRECT"
                    break
                except anthropic.RateLimitError:
                    wait = 15 * (2 ** attempt)
                    print(f"  [RATE LIMIT] attempt {attempt+1}, waiting {wait}s ...", flush=True)
                    await asyncio.sleep(wait)
                except Exception as e:
                    print(
                        f"  [ERR] {decision['run_id']} a{decision['agent_id']} "
                        f"dp{decision['dp_index']}: {e}",
                        flush=True,
                    )
                    break

            kw    = decision["keyword_score"]
            agree = "✓" if (verdict == "CORRECT") == bool(kw) else "✗"
            print(
                f"  [{seq:4d}/{len(todo)}] "
                f"{decision['condition'][:4]:4s} "
                f"a{decision['agent_id']} dp{decision['dp_index']:2d}  "
                f"kw={kw} judge={verdict} {agree}",
                flush=True,
            )

            result = {**decision, "judge_verdict": verdict, "judge_reason": judge_reason}

            # Incremental CSV write — flush immediately
            if verdict != "ERROR":
                row = dict(result)
                row["answer_keywords"] = "|".join(row["answer_keywords"])
                writer.writerow(row)
                fh.flush()

            # Running stats + periodic summary
            stats.update(kw, verdict)
            if stats.done % PROGRESS_EVERY == 0:
                stats.print_progress()

            return result

    # Run serially (CONCURRENCY=1) using as_completed for uniform ordering
    tasks = [_judge_one(d, i + 1) for i, d in enumerate(todo)]
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)

    await client.close()
    fh.close()
    stats.print_progress()  # final progress line
    return results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def load_all_verdicts(csv_path: Path) -> list[dict]:
    """Load all judged rows from CSV (after run completes)."""
    rows: list[dict] = []
    if not csv_path.exists():
        return rows
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row["keyword_score"] = int(row["keyword_score"])
            row["dp_index"]      = int(row["dp_index"])
            rows.append(row)
    return rows


def cohen_kappa(results: list[dict]) -> float:
    valid = [r for r in results if r.get("judge_verdict") not in ("ERROR", "")]
    n = len(valid)
    if n == 0:
        return 0.0
    kw_pos  = sum(int(r["keyword_score"]) for r in valid) / n
    jdg_pos = sum(1 for r in valid if r["judge_verdict"] == "CORRECT") / n
    p_e     = kw_pos * jdg_pos + (1 - kw_pos) * (1 - jdg_pos)
    agree   = sum(
        1 for r in valid
        if (r["judge_verdict"] == "CORRECT") == bool(int(r["keyword_score"]))
    )
    p_o = agree / n
    return 1.0 if p_e == 1.0 else (p_o - p_e) / (1 - p_e)


def scenario_stats(results: list[dict]) -> dict:
    valid = [r for r in results if r.get("judge_verdict") not in ("ERROR", "")]
    n     = len(valid)
    if n == 0:
        return {"n": 0}

    agree = sum(
        1 for r in valid
        if (r["judge_verdict"] == "CORRECT") == bool(int(r["keyword_score"]))
    )
    fn = sum(1 for r in valid if int(r["keyword_score"]) == 0 and r["judge_verdict"] == "CORRECT")
    fp = sum(1 for r in valid if int(r["keyword_score"]) == 1 and r["judge_verdict"] == "INCORRECT")

    per_cond: dict[str, dict] = {}
    for cond in ("dacs", "baseline"):
        sub = [r for r in valid if r["condition"] == cond]
        nc  = len(sub)
        if nc == 0:
            continue
        per_cond[cond] = {
            "n":         nc,
            "kw_acc":    sum(int(r["keyword_score"]) for r in sub) / nc,
            "judge_acc": sum(1 for r in sub if r["judge_verdict"] == "CORRECT") / nc,
            "agreement": sum(
                1 for r in sub
                if (r["judge_verdict"] == "CORRECT") == bool(int(r["keyword_score"]))
            ) / nc,
        }

    return {
        "n":         n,
        "agree_n":   agree,
        "agree_rate": agree / n,
        "kw_acc":    sum(int(r["keyword_score"]) for r in valid) / n,
        "judge_acc": sum(1 for r in valid if r["judge_verdict"] == "CORRECT") / n,
        "kappa":     cohen_kappa(valid),
        "fn":        fn,
        "fp":        fp,
        "per_cond":  per_cond,
    }


def save_summary(stats_by_scenario: dict[str, dict], sample_ns: dict[str, int | None]) -> None:
    path = RESULTS_DIR / "llm_judge_phase3_summary.md"

    lines: list[str] = [
        "# LLM-as-Judge Validation — Phase 3 (Decision Density)\n",
        "## Setup\n",
        "| Scenario | Coverage | Decisions judged | Sample strategy |",
        "|---|---|---|---|",
    ]
    for sid, st in stats_by_scenario.items():
        sn    = sample_ns[sid]
        cov   = "Full" if sn is None else f"Stratified n={sn}"
        strat = "All decisions" if sn is None else f"{sn//2} DACS + {sn//2} baseline, seed={SEED}"
        lines.append(f"| {sid} | {cov} | {st['n']} | {strat} |")

    lines += [
        "",
        f"Judge model: `{MODEL}` (independent call, no keyword list given)\n",
        "## Overall Agreement by Scenario\n",
        "| Scenario | n | Agreement | κ | Keyword acc | Judge acc | FN | FP |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sid, st in stats_by_scenario.items():
        n  = st["n"]
        se = (st["agree_rate"] * (1 - st["agree_rate"]) / n) ** 0.5 if n > 1 else 0
        lines.append(
            f"| {sid} | {n} | **{st['agree_rate']*100:.1f}%** (±{se*100:.1f}%) "
            f"| **{st['kappa']:.3f}** | {st['kw_acc']*100:.1f}% "
            f"| {st['judge_acc']*100:.1f}% | {st['fn']} | {st['fp']} |"
        )

    lines += [
        "",
        "## Per-Condition Breakdown\n",
        "| Scenario | Condition | n | Keyword acc | Judge acc | Agreement |",
        "|---|---|---|---|---|---|",
    ]
    for sid, st in stats_by_scenario.items():
        for cond, cs in st["per_cond"].items():
            lines.append(
                f"| {sid} | {cond} | {cs['n']} "
                f"| {cs['kw_acc']*100:.1f}% | {cs['judge_acc']*100:.1f}% "
                f"| {cs['agreement']*100:.1f}% |"
            )

    kappas        = [st["kappa"] for st in stats_by_scenario.values()]
    avg_kappa     = sum(kappas) / len(kappas)
    all_n         = sum(st["n"] for st in stats_by_scenario.values())
    all_agree     = sum(st["agree_n"] for st in stats_by_scenario.values())
    overall_agree = all_agree / all_n if all_n else 0

    lines += [
        "",
        "## Phase 3 Combined\n",
        f"| Total decisions judged | {all_n} |",
        "|---|---|",
        f"| Overall agreement | {overall_agree*100:.1f}% |",
        f"| Mean Cohen's κ | {avg_kappa:.3f} |",
        "",
        "## Interpretation\n",
    ]
    if avg_kappa >= 0.80:
        lines.append(
            f"Mean κ={avg_kappa:.3f} (substantial/near-perfect agreement). "
            "Keyword matching is a valid proxy for LLM-judged correctness across Phase 3 scenarios. "
            "Phase 3 accuracy estimates inherit the same validity established by the Phase 2 judge validation (κ=0.956)."
        )
    else:
        lines.append(f"Mean κ={avg_kappa:.3f} — see disagreement analysis in per-decision CSVs.")

    lines += [
        "",
        "*Full results: `results/llm_judge_phase3_s8.csv`, `results/llm_judge_phase3_s7.csv`*",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    rng = random.Random(SEED)
    stats_by_scenario: dict[str, dict] = {}
    sample_ns: dict[str, int | None]   = {}

    for scenario_id, cfg in SCENARIO_CONFIG.items():
        sample_n = cfg["sample_n"]
        sample_ns[scenario_id] = sample_n

        csv_path = RESULTS_DIR / f"llm_judge_phase3_{scenario_id[:2]}.csv"
        print(f"\n{'='*60}", flush=True)
        print(
            f"Scenario: {scenario_id}  "
            f"({'ALL' if sample_n is None else sample_n} decisions)  "
            f"CSV: {csv_path}",
            flush=True,
        )
        print(f"{'='*60}", flush=True)

        all_decisions = collect_decisions(scenario_id)
        print(f"  JSONL decisions available: {len(all_decisions)}", flush=True)

        sample       = sample_decisions(all_decisions, sample_n, rng)
        already_done = load_existing_verdicts(csv_path)
        dacs_n  = sum(1 for d in sample if d["condition"] == "dacs")
        base_n  = sum(1 for d in sample if d["condition"] == "baseline")
        todo_n  = sum(
            1 for d in sample
            if (d["run_id"], d["agent_id"], d["dp_index"]) not in already_done
        )
        print(
            f"  Sample: {len(sample)} ({dacs_n} DACS + {base_n} baseline)  "
            f"Already done: {len(already_done)}  Remaining: {todo_n}",
            flush=True,
        )
        print(f"  Monitor:  tail -f {csv_path}", flush=True)
        print(f"  Progress: watch -n 10 'wc -l {csv_path}'", flush=True)
        print("", flush=True)

        t0 = time.monotonic()
        await judge_scenario(scenario_id, sample, csv_path, already_done)
        elapsed = time.monotonic() - t0
        print(f"\n  Finished {scenario_id} in {elapsed/60:.1f}min", flush=True)

        # Load full CSV (includes pre-existing rows) for stats
        all_rows = load_all_verdicts(csv_path)
        stats    = scenario_stats(all_rows)
        stats_by_scenario[scenario_id] = stats

        print(
            f"  Agreement: {stats['agree_rate']*100:.1f}%  κ={stats['kappa']:.3f}  "
            f"(kw_acc={stats['kw_acc']*100:.1f}%  judge_acc={stats['judge_acc']*100:.1f}%)",
            flush=True,
        )
        for cond, cs in stats["per_cond"].items():
            print(
                f"    {cond:8s}  n={cs['n']}  "
                f"kw={cs['kw_acc']*100:.1f}%  judge={cs['judge_acc']*100:.1f}%  "
                f"agree={cs['agreement']*100:.1f}%",
                flush=True,
            )

    print(f"\n{'='*60}", flush=True)
    print("Phase 3 combined summary", flush=True)
    print(f"{'='*60}", flush=True)
    save_summary(stats_by_scenario, sample_ns)


if __name__ == "__main__":
    asyncio.run(main())
