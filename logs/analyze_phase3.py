import csv, statistics, math
from collections import defaultdict
from scipy import stats

rows = []
with open("results/summary.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("scenario", "").startswith(("s7_", "s8_")):
            rows.append(row)

print(f"Phase 3 rows: {len(rows)}")

# Group by (scenario_id, condition)
groups = defaultdict(list)
for row in rows:
    key = (row["scenario"], row["condition"])
    groups[key].append(row)

results = {}
for key in sorted(groups):
    vals = groups[key]
    accs = [float(r["steering_accuracy"]) for r in vals]
    contams = [float(r["contamination_rate"]) for r in vals]
    ctxs = [float(r["avg_context_tokens"]) for r in vals]
    n = len(accs)
    mean_acc = statistics.mean(accs)
    se_acc = statistics.stdev(accs) / math.sqrt(n) if n > 1 else 0
    mean_contam = statistics.mean(contams)
    mean_ctx = statistics.mean(ctxs)
    se_contam = statistics.stdev(contams) / math.sqrt(n) if n > 1 else 0
    se_ctx = statistics.stdev(ctxs) / math.sqrt(n) if n > 1 else 0
    results[key] = {
        "accs": accs, "contams": contams, "ctxs": ctxs,
        "mean_acc": mean_acc, "se_acc": se_acc,
        "mean_contam": mean_contam, "se_contam": se_contam,
        "mean_ctx": mean_ctx, "se_ctx": se_ctx, "n": n
    }
    print(f"\n{key[0]} | {key[1]:8s} | n={n}")
    print(f"  accuracy:      {mean_acc*100:.1f}% +/- {se_acc*100:.1f}% SE")
    print(f"  raw accs:      {[round(a*100,1) for a in accs]}")
    print(f"  contamination: {mean_contam*100:.1f}% +/- {se_contam*100:.1f}% SE")
    print(f"  avg_ctx_tokens:{mean_ctx:.0f} +/- {se_ctx:.0f} SE")

# Welch's t-tests: DACS vs baseline per scenario
print("\n--- Welch t-tests (DACS vs Baseline) ---")
for scen in ["s7_n5_dense_d2", "s8_n3_dense_d3"]:
    dacs_accs = results[(scen, "dacs")]["accs"]
    base_accs = results[(scen, "baseline")]["accs"]
    t, p = stats.ttest_ind(dacs_accs, base_accs, equal_var=False)
    d_mean = statistics.mean(dacs_accs)
    b_mean = statistics.mean(base_accs)
    diff = (d_mean - b_mean) * 100
    print(f"\n  {scen}: DACS={d_mean*100:.1f}% vs Baseline={b_mean*100:.1f}%  delta={diff:+.1f}pp")
    print(f"  t={t:.3f}, p={p:.4e}")
    if p < 0.001:
        print(f"  *** p<0.001")
    elif p < 0.01:
        print(f"  ** p<0.01")
    elif p < 0.05:
        print(f"  * p<0.05")

# Context token ratios
print("\n--- Context token ratios (baseline / DACS) ---")
for scen in ["s7_n5_dense_d2", "s8_n3_dense_d3"]:
    dacs_ctx = results[(scen, "dacs")]["mean_ctx"]
    base_ctx = results[(scen, "baseline")]["mean_ctx"]
    ratio = base_ctx / dacs_ctx
    print(f"  {scen}: DACS={dacs_ctx:.0f}tok, Baseline={base_ctx:.0f}tok, ratio={ratio:.2f}x")
