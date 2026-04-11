"""Build ablation_summary.csv with proper condition labels for plotting."""
import csv

rows = []
with open('results/summary.csv') as f:
    for r in csv.DictReader(f):
        rows.append(r)

out_rows = []
fieldnames = list(rows[0].keys())

for scenario in ['s1_n3', 's2_n5']:
    # Full DACS
    dacs = [r for r in rows if r['scenario'] == scenario and r['condition'] == 'dacs'
            and 'no_registry' not in r['run_id']
            and 'random_focus' not in r['run_id']
            and 'flat_ordered' not in r['run_id']][-10:]
    for r in dacs:
        out = dict(r)
        out['condition'] = 'dacs'
        out_rows.append(out)

    # Baseline
    base = [r for r in rows if r['scenario'] == scenario and r['condition'] == 'baseline'][-10:]
    for r in base:
        out = dict(r)
        out['condition'] = 'baseline'
        out_rows.append(out)

    # Ablations
    for ablation in ['no_registry', 'random_focus', 'flat_ordered']:
        abl = [r for r in rows if r['scenario'] == scenario and ablation in r['run_id']][-10:]
        for r in abl:
            out = dict(r)
            out['condition'] = ablation
            out_rows.append(out)

with open('results/ablation_summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"Wrote {len(out_rows)} rows to results/ablation_summary.csv")
# Summary count
from collections import Counter
counts = Counter((r['scenario'], r['condition']) for r in out_rows)
for k, v in sorted(counts.items()):
    print(f"  {k[0]:8s} {k[1]:15s}: {v}")
