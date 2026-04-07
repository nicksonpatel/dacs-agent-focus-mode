"""Rebuild summary_real.csv from JSONL files for any missing trials."""
import csv
import json
import re
from pathlib import Path

results_dir = Path("results_real_agent")
summary_path = results_dir / "summary_real.csv"

fieldnames = [
    "run_id", "scenario", "condition", "n_agents", "trial",
    "contamination_rate", "avg_context_tokens", "p95_context_tokens",
    "n_steering_responses",
]

existing_ids: set[str] = set()
if summary_path.exists():
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            existing_ids.add(row["run_id"])

print(f"Existing rows: {len(existing_ids)}")

agent_ids = {"a1", "a2", "a3", "a4", "a5"}
new_rows: list[dict] = []

for fpath in sorted(results_dir.glob("ra2_n5_*.jsonl")):
    run_id = fpath.stem
    if run_id in existing_ids:
        continue

    condition = "dacs" if "_dacs_" in run_id else "baseline"
    m = re.search(r"_t(\d+)_", run_id)
    trial = int(m.group(1)) if m else 0

    events: list[dict] = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    steering_resps = [e for e in events if e.get("event") == "STEERING_RESPONSE"]
    contexts = [e.get("token_count", 0) for e in events if e.get("event") == "CONTEXT_BUILT"]

    contam = 0
    for e in steering_resps:
        resp = e.get("response_text", "")
        caid = e.get("agent_id", "")
        others = agent_ids - {caid}
        if any(re.search(r"\b" + o + r"\b", resp) for o in others):
            contam += 1
    contam_rate = contam / len(steering_resps) if steering_resps else 0.0

    avg_ctx = sum(contexts) / len(contexts) if contexts else 0.0
    p95_ctx = sorted(contexts)[int(len(contexts) * 0.95)] if contexts else 0

    row = {
        "run_id": run_id, "scenario": "ra2_n5", "condition": condition,
        "n_agents": 5, "trial": trial, "contamination_rate": contam_rate,
        "avg_context_tokens": avg_ctx, "p95_context_tokens": p95_ctx,
        "n_steering_responses": len(steering_resps),
    }
    new_rows.append(row)
    print(f"  {run_id}: {len(steering_resps)} steering, contam={contam_rate:.2f}")

with open(summary_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    for row in new_rows:
        w.writerow(row)

print(f"\nAdded {len(new_rows)} rows to summary_real.csv")
