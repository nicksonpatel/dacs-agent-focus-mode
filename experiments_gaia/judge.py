"""GAIA Phase-5 judge: exact-match + LLM-fallback scorer.

Usage
-----
    python -m experiments_gaia.judge                            # all jsonl in results_gaia/
    python -m experiments_gaia.judge --dir results_gaia         # same
    python -m experiments_gaia.judge --dir results_gaia --llm-fallback
    python -m experiments_gaia.judge --file results_gaia/some_run.jsonl

Outputs
-------
    results_gaia/judge_results_gaia.csv   — per-agent verdict
    STDOUT table                          — batch × condition × accuracy
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_JUDGE_MODEL     = "anthropic/claude-haiku-4-5"
_MINIMAX_MODEL   = "MiniMax-M2.7"

# Articles and punctuation to strip when normalising answers
_ARTICLES_RE      = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_RE         = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE    = re.compile(r"\s+")

# Known alias groups (any member matches any other member)
_ALIAS_GROUPS: list[frozenset[str]] = [
    frozenset({"au", "gold", "gold (au)"}),
    frozenset({"dna", "deoxyribonucleic acid"}),
    frozenset({"295 bc", "295 b c", "295 bce", "295 b ce", "295 bc "}),
    frozenset({"tim berners lee", "sir tim berners lee", "timothy john berners lee"}),
    frozenset({"1815", "eighteen fifteen"}),
    frozenset({"steve wozniak", "stephen wozniak", "stephen gary wozniak"}),
    frozenset({"antonio vivaldi", "vivaldi"}),
    frozenset({"christopher nolan", "nolan"}),
    frozenset({"george orwell", "orwell", "eric blair"}),
    frozenset({"leo tolstoy", "tolstoy", "lev tolstoi"}),
    frozenset({"france", "french team"}),
    frozenset({"parasite", "gisaengchung"}),
    frozenset({"python", "python programming language"}),
    frozenset({"pacific ocean", "the pacific", "the pacific ocean"}),
    frozenset({"mount kilimanjaro", "kilimanjaro", "mt kilimanjaro"}),
]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    text = text.lower().strip()
    text = _ARTICLES_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _in_alias_group(pred: str, gold: str) -> bool:
    for group in _ALIAS_GROUPS:
        if pred in group and gold in group:
            return True
    return False


def exact_match(pred: str, gold: str) -> bool:
    """Return True if prediction matches gold (exact or via alias group)."""
    if not pred:
        return False
    np = _normalise(pred)
    ng = _normalise(gold)
    if np == ng:
        return True
    # Containment (e.g., "the answer is 42" vs "42")
    if ng in np or np in ng:
        return True
    return _in_alias_group(np, ng)


# ---------------------------------------------------------------------------
# LLM fallback judge
# ---------------------------------------------------------------------------

_LLM_JUDGE_PROMPT = """\
You are evaluating whether a predicted answer is correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Decide whether the predicted answer is correct. A prediction is correct if:
- It contains the gold answer (possibly with additional words)
- It is a well-known alias or equivalent form of the gold answer
- Spelling variations or punctuation differences don't affect meaning

Respond in exactly this format (no other text):
<verdict>CORRECT</verdict> or <verdict>INCORRECT</verdict>
<reason>One sentence explanation</reason>"""

_VERDICT_RE = re.compile(r"<verdict>(CORRECT|INCORRECT)</verdict>", re.IGNORECASE)


async def _llm_judge_one(
    item: dict,
    client: Any,
    model: str,
    semaphore: asyncio.Semaphore,
) -> bool | None:
    """Returns True/False/None (None = judge error)."""
    prompt = _LLM_JUDGE_PROMPT.format(
        question=item["question"][:400],
        gold=item["gold"],
        pred=item["answer"],
    )
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = ""
                for block in resp.content:
                    if hasattr(block, "text"):
                        text = block.text
                        break
                m = _VERDICT_RE.search(text)
                if m:
                    return m.group(1).upper() == "CORRECT"
            except Exception as exc:
                if attempt == 2:
                    Console().print(f"[red]LLM judge error: {exc}[/red]")
                await asyncio.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _collect_agent_answers(log_path: Path) -> list[dict]:
    """Extract AGENT_ANSWER events from a JSONL run log."""
    run_meta: dict = {}
    answers: list[dict] = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("event") == "RUN_START":
                run_meta = event
            elif event.get("event") == "AGENT_ANSWER":
                answers.append({
                    "run_id":    run_meta.get("run_id", log_path.stem),
                    "batch_id":  run_meta.get("scenario", ""),
                    "condition": run_meta.get("condition", ""),
                    "agent_id":  event["agent_id"],
                    "question":  event.get("question", ""),
                    "gold":      event["gold"],
                    "answer":    event.get("answer", ""),
                    "domain":    event.get("domain", ""),
                })
    return answers


# ---------------------------------------------------------------------------
# Main judge routine
# ---------------------------------------------------------------------------

async def judge_all(
    log_files: list[Path],
    llm_fallback: bool,
    results_dir: Path,
) -> None:
    # Collect all items
    all_items: list[dict] = []
    for logf in log_files:
        all_items.extend(_collect_agent_answers(logf))

    if not all_items:
        Console().print("[yellow]No AGENT_ANSWER events found in logs.[/yellow]")
        return

    Console().print(f"[cyan]Judging {len(all_items)} agent answers...[/cyan]")

    # Exact-match pass
    for item in all_items:
        item["exact_match"] = exact_match(item["answer"], item["gold"])

    # LLM fallback for non-exact-match items
    if llm_fallback:
        or_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY", "")
        minimax_key = os.environ.get("MINIMAX_API_KEY", "")
        if or_key:
            from anthropic import AsyncAnthropic as _AC
            client = _AC(api_key=or_key, base_url=_OPENROUTER_BASE)
            model  = _JUDGE_MODEL
        else:
            client = AsyncAnthropic(
                api_key=minimax_key,
                base_url="https://api.minimax.io/anthropic",
            )
            model = _MINIMAX_MODEL

        sem = asyncio.Semaphore(4)
        borderline = [it for it in all_items if not it["exact_match"] and it.get("answer")]
        Console().print(f"[cyan]{len(borderline)} borderline items → LLM judge fallback[/cyan]")

        verdicts = await asyncio.gather(
            *[_llm_judge_one(it, client, model, sem) for it in borderline],
            return_exceptions=False,
        )
        for it, verdict in zip(borderline, verdicts):
            if verdict is True:
                it["exact_match"] = True
                it["llm_corrected"] = True

    for item in all_items:
        item.setdefault("llm_corrected", False)

    # Write CSV
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "judge_results_gaia.csv"
    fieldnames = [
        "run_id", "batch_id", "condition", "agent_id",
        "domain", "question", "gold", "answer",
        "exact_match", "llm_corrected",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in all_items:
            writer.writerow({k: item.get(k, "") for k in fieldnames})

    Console().print(f"[green]Wrote {len(all_items)} rows → {out_csv}[/green]")

    # Summary table: batch_id × condition → accuracy
    _print_summary(all_items)


def _print_summary(items: list[dict]) -> None:
    """Print accuracy table grouped by batch_id × condition."""
    from collections import defaultdict
    key_fn = lambda it: (it["batch_id"], it["condition"])
    groups: dict[tuple, list[bool]] = defaultdict(list)
    for it in items:
        groups[key_fn(it)].append(bool(it["exact_match"]))

    c = Console()
    t = Table(title="GAIA Phase-5 Accuracy", show_lines=True)
    t.add_column("batch_id")
    t.add_column("condition")
    t.add_column("accuracy")
    t.add_column("n")

    # Overall per-condition
    cond_groups: dict[str, list[bool]] = {}
    for (bid, cond), vals in sorted(groups.items()):
        acc = statistics.mean(vals)
        t.add_row(bid, cond, f"{acc:.1%}", str(len(vals)))
        if cond not in cond_groups:
            cond_groups[cond] = []
        cond_groups[cond].extend(vals)

    t.add_section()
    for cond, vals in sorted(cond_groups.items()):
        t.add_row("OVERALL", cond, f"{statistics.mean(vals):.1%}", str(len(vals)))

    c.print(t)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GAIA Phase-5 judge")
    parser.add_argument("--dir",  default="results_gaia",
                        help="Directory containing result JSONL files")
    parser.add_argument("--file", nargs="*",
                        help="Specific JSONL file(s) to judge (overrides --dir)")
    parser.add_argument("--llm-fallback", action="store_true",
                        help="Use LLM to re-judge borderline non-exact-matches")
    args = parser.parse_args()

    if args.file:
        log_files = [Path(f) for f in args.file if Path(f).suffix == ".jsonl"]
    else:
        d = Path(args.dir)
        log_files = sorted(d.glob("*.jsonl")) if d.exists() else []

    if not log_files:
        Console().print("[red]No JSONL files found.[/red]")
        return

    Console().print(f"[cyan]Judging {len(log_files)} log files...[/cyan]")
    asyncio.run(judge_all(log_files, args.llm_fallback, Path(args.dir)))


if __name__ == "__main__":
    main()
