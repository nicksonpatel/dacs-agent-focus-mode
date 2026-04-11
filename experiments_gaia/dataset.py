"""GAIA Level-1 dataset loader.

Downloads the GAIA Level-1 validation split from Hugging Face, filters to
questions that are suitable for the DACS multi-agent setup (no file attachments,
short deterministic answers), and groups them into N=3-agent batches where each
batch contains maximally diverse question domains.

Usage (one-off, produces scenario_defs.py candidates):
    python -m experiments_gaia.dataset --print-batches

Environment:
    HF_TOKEN  — Hugging Face token with gaia-benchmark/GAIA read access
"""
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GAIAQuestion:
    task_id: str
    question: str
    answer: str
    level: int
    domain_hint: str = ""   # short domain label, assigned during filtering


# ---------------------------------------------------------------------------
# Filtering heuristics
# ---------------------------------------------------------------------------

# Answers longer than this are usually multi-sentence explanations → skip
_MAX_ANSWER_LEN = 60

# Questions referencing files/URLs likely need tools → skip
_FILE_PATTERNS = [
    r"\.(pdf|xlsx|csv|png|jpg|jpeg|mp3|wav|zip)\b",
    r"attached",
    r"the file",
    r"the image",
    r"the table",
    r"the chart",
    r"the graph",
    r"the spreadsheet",
]
_FILE_RE = re.compile("|".join(_FILE_PATTERNS), re.IGNORECASE)

# Numeric-only answers (pure arithmetic) — useful to keep domain diversity
_NUMERIC_RE = re.compile(r"^\d+(\.\d+)?$")

# Rough domain labels — assign the first matching label
_DOMAIN_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("history",     re.compile(r"\b(histor|war|battle|centur|dynasty|empire|president|king|queen|born|died|founded)\b", re.I)),
    ("geography",   re.compile(r"\b(country|capital|city|river|mountain|continent|ocean|island|region|located)\b", re.I)),
    ("science",     re.compile(r"\b(element|atom|molecule|chemical|physics|biology|cell|gene|planet|orbit|speed of light)\b", re.I)),
    ("maths",       re.compile(r"\b(sum|product|calculate|equation|formula|prime|integer|fraction|percent|average|mean)\b", re.I)),
    ("literature",  re.compile(r"\b(novel|author|wrote|book|poem|character|published|play|shakespear)\b", re.I)),
    ("sport",       re.compile(r"\b(sport|olympic|champion|team|player|scored|tournament|match|football|tennis|cricket)\b", re.I)),
    ("technology",  re.compile(r"\b(computer|software|program|internet|algorithm|data|AI|machine learning|invented)\b", re.I)),
    ("music",       re.compile(r"\b(music|song|album|artist|band|compose|orchestra|instrument|opera)\b", re.I)),
    ("film",        re.compile(r"\b(film|movie|director|actor|actress|oscar|cinema|released)\b", re.I)),
    ("other",       re.compile(r".*")),   # fallback
]


def _assign_domain(question: str) -> str:
    for label, pattern in _DOMAIN_PATTERNS:
        if pattern.search(question):
            return label
    return "other"


def _is_suitable(q: GAIAQuestion) -> bool:
    """Return True if this question fits the DACS evaluation setup."""
    # Reject if references files or images
    if _FILE_RE.search(q.question):
        return False
    # Reject empty or trivial answers
    if not q.answer.strip():
        return False
    # Reject very long answers (likely prose explanations)
    if len(q.answer) > _MAX_ANSWER_LEN:
        return False
    return True


# ---------------------------------------------------------------------------
# HF loader
# ---------------------------------------------------------------------------

def load_gaia_level1(split: str = "validation") -> list[GAIAQuestion]:
    """Load GAIA Level-1 questions from Hugging Face datasets.

    Requires:
        pip install datasets huggingface_hub
        HF_TOKEN env var with read access to gaia-benchmark/GAIA

    Returns a list of GAIAQuestion objects filtered to suitable questions.
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Install: pip install datasets huggingface_hub"
        ) from exc

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    if not hf_token:
        raise RuntimeError(
            "Set HF_TOKEN in your .env to access gaia-benchmark/GAIA "
            "(the dataset is gated)."
        )

    data_dir = snapshot_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        token=hf_token,
    )
    ds = load_dataset(data_dir, "2023_level1", split=split)

    questions: list[GAIAQuestion] = []
    for row in ds:
        # The validation split has final_answer; test set answers are private.
        answer = row.get("Final answer", "") or row.get("final_answer", "")
        q = GAIAQuestion(
            task_id=row["task_id"],
            question=row["Question"],
            answer=str(answer).strip(),
            level=int(row.get("Level", 1)),
        )
        q.domain_hint = _assign_domain(q.question)
        if _is_suitable(q):
            questions.append(q)

    return questions


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def build_batches(
    questions: list[GAIAQuestion],
    n_agents: int = 3,
    n_batches: int = 10,
    seed: int = 42,
) -> list[list[GAIAQuestion]]:
    """Group questions into domain-diverse batches of n_agents questions each.

    Strategy:
    1. Sort questions by domain label so domains are spread evenly.
    2. Assign questions to slots in a round-robin across domains.
    3. Fill each batch greedily with the most distinct remaining domains.

    Returns list of n_batches batches, each a list of n_agents GAIAQuestion.
    """
    import random
    rng = random.Random(seed)

    # Group by domain
    domain_buckets: dict[str, list[GAIAQuestion]] = {}
    for q in questions:
        domain_buckets.setdefault(q.domain_hint, []).append(q)

    # Shuffle within each domain for variety
    for bucket in domain_buckets.values():
        rng.shuffle(bucket)

    batches: list[list[GAIAQuestion]] = []
    used_task_ids: set[str] = set()
    domains = list(domain_buckets.keys())

    for _ in range(n_batches):
        batch: list[GAIAQuestion] = []
        selected_domains: set[str] = set()

        # Try to pick n_agents distinct domains
        domain_order = domains.copy()
        rng.shuffle(domain_order)

        for domain in domain_order:
            if len(batch) == n_agents:
                break
            bucket = domain_buckets.get(domain, [])
            available = [q for q in bucket if q.task_id not in used_task_ids]
            if not available:
                continue
            chosen = available[0]
            batch.append(chosen)
            used_task_ids.add(chosen.task_id)
            selected_domains.add(domain)

        # If we couldn't fill from distinct domains, backfill from any domain
        for domain in domain_order:
            if len(batch) == n_agents:
                break
            bucket = domain_buckets.get(domain, [])
            available = [q for q in bucket if q.task_id not in used_task_ids]
            if available:
                chosen = available[0]
                batch.append(chosen)
                used_task_ids.add(chosen.task_id)

        if len(batch) == n_agents:
            batches.append(batch)
        if len(batches) == n_batches:
            break

    return batches


# ---------------------------------------------------------------------------
# CLI helper: print candidate batches for copy-pasting into scenario_defs.py
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GAIA dataset loader utility")
    parser.add_argument("--print-batches", action="store_true",
                        help="Print batch candidates for scenario_defs.py")
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    questions = load_gaia_level1(args.split)
    print(f"Loaded {len(questions)} suitable GAIA Level-1 {args.split} questions")

    domain_counts: dict[str, int] = {}
    for q in questions:
        domain_counts[q.domain_hint] = domain_counts.get(q.domain_hint, 0) + 1
    print("Domains:", dict(sorted(domain_counts.items(), key=lambda x: -x[1])))

    if args.print_batches:
        batches = build_batches(questions, args.n_agents, args.n_batches, args.seed)
        print(f"\nBuilt {len(batches)} batches of {args.n_agents} questions each:\n")
        for i, batch in enumerate(batches):
            print(f"# Batch {i+1}")
            for q in batch:
                print(f"  [{q.domain_hint:12s}]  task_id={q.task_id}")
                print(f"    Q: {q.question[:100]}")
                print(f"    A: {q.answer}")
            print()

        # Dump as JSON for copy-pasting
        out = []
        for i, batch in enumerate(batches):
            out.append({
                "batch_id": f"gaia_b{i+1:02d}_n3",
                "questions": [
                    {
                        "task_id": q.task_id,
                        "question": q.question,
                        "answer": q.answer,
                        "domain": q.domain_hint,
                    }
                    for q in batch
                ],
            })
        print("\n# JSON (copy to scenario_defs.py):")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
