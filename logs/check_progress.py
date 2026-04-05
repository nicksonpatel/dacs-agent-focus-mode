"""Phase 3 progress monitor — run standalone."""
import json
import glob
import os
import subprocess
from collections import Counter
from datetime import datetime

files = sorted(glob.glob("results/s7_*.jsonl") + glob.glob("results/s8_*.jsonl"))
print(f"\n=== Phase 3 Progress — {datetime.now().strftime('%H:%M:%S')} ===\n")
print(f"  Active trial files: {len(files)}\n")

done_count = 0
for f in sorted(files):
    try:
        with open(f) as fh:
            events = [json.loads(l) for l in fh]
        c = Counter(e.get("event") for e in events)
        responses = c.get("STEERING_RESPONSE", 0)
        total = 40 if "s7" in f else 45
        pct = responses / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        if c.get("RUN_END"):
            status = "✓ COMPLETE"
            done_count += 1
        else:
            status = f"{responses}/{total} ({pct:.0f}%)"
        name = os.path.basename(f)[:50]
        print(f"  {name:<52} [{bar}] {status}")
    except Exception as e:
        print(f"  {os.path.basename(f)}: error — {e}")

# Summary CSV
result = subprocess.run(
    ["grep", "-c", "s[78]", "results/summary.csv"],
    capture_output=True, text=True
)
completed_rows = result.stdout.strip() or "0"

# Workers
ps = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
workers = [l for l in ps.splitlines() if "run_experiment" in l and "grep" not in l]

print(f"\n  Completed trials in summary.csv : {completed_rows}")
print(f"  Trials with RUN_END in .jsonl   : {done_count}/{len(files)}")
print(f"  Live worker processes            : {len(workers)}")

# Peek at accuracy for any finished trials
if int(completed_rows or 0) > 0:
    print("\n  --- Early accuracy snapshot ---")
    result2 = subprocess.run(
        ["grep", "s[78]", "results/summary.csv"],
        capture_output=True, text=True
    )
    for row in result2.stdout.strip().splitlines()[:20]:
        parts = row.split(",")
        if len(parts) >= 7:
            run_id, scenario, condition = parts[0], parts[1], parts[2]
            accuracy = float(parts[5])
            contam = float(parts[6])
            ctx = float(parts[7])
            print(f"    {condition:8} {scenario}  acc={accuracy:.1%}  contam={contam:.1%}  ctx={ctx:.0f}tok")

print()
