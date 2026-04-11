"""Analyze ablation experiment results and print summary table."""
import csv
import numpy as np
import sys

def main():
    rows = []
    with open('results/summary.csv') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print("=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    print(f"{'Scenario':8s} | {'Condition':15s} | {'N':>3s} | {'Accuracy':>16s} | {'Contamination':>16s}")
    print("-" * 90)

    results = {}

    for scenario in ['s1_n3', 's2_n5']:
        conditions = [
            ('DACS (full)', lambda r, s=scenario: r['scenario'] == s and r['condition'] == 'dacs'
             and 'no_registry' not in r['run_id']
             and 'random_focus' not in r['run_id']
             and 'flat_ordered' not in r['run_id']),
            ('Baseline', lambda r, s=scenario: r['scenario'] == s and r['condition'] == 'baseline'),
            ('no_registry', lambda r, s=scenario: r['scenario'] == s and 'no_registry' in r['run_id']),
            ('random_focus', lambda r, s=scenario: r['scenario'] == s and 'random_focus' in r['run_id']),
            ('flat_ordered', lambda r, s=scenario: r['scenario'] == s and 'flat_ordered' in r['run_id']),
        ]

        for cond_label, cond_filter in conditions:
            matched = [r for r in rows if cond_filter(r)]
            # Take last 10 if more (dedup from initial stalled run)
            matched = matched[-10:]
            if not matched:
                continue
            accs = [float(r['steering_accuracy']) for r in matched]
            conts = [float(r['contamination_rate']) for r in matched]
            n = len(accs)
            acc_mean = np.mean(accs)
            acc_se = np.std(accs, ddof=1) / np.sqrt(n) if n > 1 else 0
            cont_mean = np.mean(conts)
            cont_se = np.std(conts, ddof=1) / np.sqrt(n) if n > 1 else 0
            print(f"{scenario:8s} | {cond_label:15s} | {n:3d} | {acc_mean:.4f} ± {acc_se:.4f} | {cont_mean:.4f} ± {cont_se:.4f}")
            results[(scenario, cond_label)] = {
                'acc_mean': acc_mean, 'acc_se': acc_se,
                'cont_mean': cont_mean, 'cont_se': cont_se, 'n': n
            }
        print()

    # Print LaTeX table rows for paper
    print("\n" + "=" * 90)
    print("LATEX TABLE ROWS (for Table 8 in draft_v4.tex)")
    print("=" * 90)

    ablation_order = ['DACS (full)', 'no_registry', 'flat_ordered', 'random_focus', 'Baseline']
    latex_names = {
        'DACS (full)': r'DACS (full)',
        'no_registry': r'$-$ Registry context',
        'flat_ordered': r'Flat-ordered',
        'random_focus': r'Random focus',
        'Baseline': r'Flat baseline',
    }

    for cond in ablation_order:
        s1 = results.get(('s1_n3', cond))
        s2 = results.get(('s2_n5', cond))
        if s1 and s2:
            # Pooled accuracy
            all_acc = (s1['acc_mean'] + s2['acc_mean']) / 2
            all_cont = (s1['cont_mean'] + s2['cont_mean']) / 2
            name = latex_names.get(cond, cond)
            print(f"{name} & {s1['acc_mean']*100:.1f} \\pm {s1['acc_se']*100:.1f} & "
                  f"{s2['acc_mean']*100:.1f} \\pm {s2['acc_se']*100:.1f} & "
                  f"{s1['cont_mean']*100:.1f} \\pm {s1['cont_se']*100:.1f} & "
                  f"{s2['cont_mean']*100:.1f} \\pm {s2['cont_se']*100:.1f} \\\\")

    # Statistical tests
    print("\n" + "=" * 90)
    print("STATISTICAL TESTS (DACS full vs each ablation)")
    print("=" * 90)
    from scipy import stats

    for scenario in ['s1_n3', 's2_n5']:
        dacs_matched = [r for r in rows if r['scenario'] == scenario and r['condition'] == 'dacs'
                        and 'no_registry' not in r['run_id']
                        and 'random_focus' not in r['run_id']
                        and 'flat_ordered' not in r['run_id']][-10:]
        dacs_accs = [float(r['steering_accuracy']) for r in dacs_matched]

        for ablation in ['no_registry', 'random_focus', 'flat_ordered']:
            abl_matched = [r for r in rows if r['scenario'] == scenario and ablation in r['run_id']][-10:]
            abl_accs = [float(r['steering_accuracy']) for r in abl_matched]
            if dacs_accs and abl_accs:
                t, p = stats.ttest_ind(dacs_accs, abl_accs, equal_var=False)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"{scenario:8s} | DACS vs {ablation:15s} | t={t:7.3f} | p={p:.6f} | {sig}")

    # Baseline comparison too
    print()
    for scenario in ['s1_n3', 's2_n5']:
        dacs_matched = [r for r in rows if r['scenario'] == scenario and r['condition'] == 'dacs'
                        and 'no_registry' not in r['run_id']
                        and 'random_focus' not in r['run_id']
                        and 'flat_ordered' not in r['run_id']][-10:]
        dacs_accs = [float(r['steering_accuracy']) for r in dacs_matched]

        base_matched = [r for r in rows if r['scenario'] == scenario and r['condition'] == 'baseline'][-10:]
        base_accs = [float(r['steering_accuracy']) for r in base_matched]
        if dacs_accs and base_accs:
            t, p = stats.ttest_ind(dacs_accs, base_accs, equal_var=False)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{scenario:8s} | DACS vs Baseline        | t={t:7.3f} | p={p:.6f} | {sig}")

if __name__ == '__main__':
    main()
