"""Analyze N=15, N=20 experiment results and refit scaling law with 5 data points.

Run after experiments complete:
    python -m experiments.analyze_n15_n20

Outputs:
    - Prints per-trial accuracy, contamination, context tokens
    - Refits power law ε(N) = α·N^β for N ∈ {3, 5, 10, 15, 20}
    - Regenerates paper/figures/scaling_law.png with 5-point fit
"""
from __future__ import annotations

import csv
import os
import statistics
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Existing Phase 1 data (N=3, 5, 10) — hardcoded from completed experiments
DACS_ACC_N = {
    3:  [0.9333, 1.0, 1.0, 0.8667, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8667],
    5:  [0.9333, 1.0, 1.0, 1.0, 0.8667, 1.0, 1.0, 0.8667, 1.0, 1.0],
    10: [0.9333, 0.9, 0.9, 0.9, 0.9, 0.9333, 0.9333, 0.9, 0.8333, 0.8667],
}
BASE_ACC_N = {
    3:  [0.6667, 0.6667, 0.6667, 0.6667, 0.3333, 0.6667, 0.6667, 0.6667, 0.6667, 0.3333],
    5:  [0.3333, 0.4667, 0.3333, 0.2667, 0.6, 0.3333, 0.4667, 0.4, 0.4, 0.2667],
    10: [0.2, 0.2667, 0.1667, 0.2333, 0.2667, 0.1, 0.2, 0.2667, 0.1667, 0.2333],
}
DACS_CTX_N = {3: 561, 5: 633, 10: 816}
BASE_CTX_N = {3: 1191, 5: 1720, 10: 2882}


def load_new_results(summary_path: str = "results/summary.csv"):
    """Load N=15 and N=20 results from summary.csv."""
    new_data = {15: {"dacs": [], "baseline": []}, 20: {"dacs": [], "baseline": []}}
    new_ctx = {15: {"dacs": [], "baseline": []}, 20: {"dacs": [], "baseline": []}}
    new_cont = {15: {"dacs": [], "baseline": []}, 20: {"dacs": [], "baseline": []}}

    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row["scenario"]
            n = int(row["n_agents"])
            cond = row["condition"].lower()
            if n not in (15, 20) or cond not in ("dacs", "baseline"):
                continue
            acc = float(row["steering_accuracy"])
            ctx = float(row["avg_context_tokens"])
            cont = float(row["contamination_rate"])
            new_data[n][cond].append(acc)
            new_ctx[n][cond].append(ctx)
            new_cont[n][cond].append(cont)

    return new_data, new_ctx, new_cont


def power_law(x, alpha, beta):
    return alpha * np.power(x, beta)


def fit_power_law(x, y, y_err=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_x = np.log(x)
        log_y = np.log(np.clip(y, 1e-6, None))
        coeffs = np.polyfit(log_x, log_y, 1)
        beta0, alpha0 = coeffs[0], np.exp(coeffs[1])
        try:
            sigma = y_err if y_err is not None else None
            popt, pcov = curve_fit(
                power_law, x, y,
                p0=[alpha0, beta0],
                sigma=sigma,
                absolute_sigma=True if sigma is not None else False,
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError:
            popt = np.array([alpha0, beta0])
            perr = np.array([np.nan, np.nan])

    y_pred = power_law(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"alpha": popt[0], "beta": popt[1], "alpha_se": perr[0], "beta_se": perr[1], "r2": r2}


def main():
    new_data, new_ctx, new_cont = load_new_results()

    # Merge with existing data
    all_dacs_acc = dict(DACS_ACC_N)
    all_base_acc = dict(BASE_ACC_N)
    all_dacs_ctx = dict(DACS_CTX_N)
    all_base_ctx = dict(BASE_CTX_N)

    for n in [15, 20]:
        if new_data[n]["dacs"]:
            all_dacs_acc[n] = new_data[n]["dacs"]
            all_dacs_ctx[n] = statistics.mean(new_ctx[n]["dacs"])
        if new_data[n]["baseline"]:
            all_base_acc[n] = new_data[n]["baseline"]
            all_base_ctx[n] = statistics.mean(new_ctx[n]["baseline"])

    # Print summary
    print("\n" + "=" * 70)
    print("  N-SCALING RESULTS (5 data points)")
    print("=" * 70)
    for n in sorted(all_dacs_acc.keys()):
        d_acc = all_dacs_acc.get(n, [])
        b_acc = all_base_acc.get(n, [])
        if d_acc and b_acc:
            d_mean = statistics.mean(d_acc)
            b_mean = statistics.mean(b_acc)
            d_err = 1 - d_mean
            b_err = 1 - b_mean
            print(f"\n  N={n:2d}:")
            print(f"    DACS     accuracy={d_mean:.3f}  error={d_err:.3f}  trials={len(d_acc)}")
            print(f"    Baseline accuracy={b_mean:.3f}  error={b_err:.3f}  trials={len(b_acc)}")
            if new_cont.get(n, {}).get("dacs"):
                print(f"    DACS contamination:     {statistics.mean(new_cont[n]['dacs']):.3f}")
            if new_cont.get(n, {}).get("baseline"):
                print(f"    Baseline contamination: {statistics.mean(new_cont[n]['baseline']):.3f}")
            if new_ctx.get(n, {}).get("dacs"):
                print(f"    DACS     avg ctx tokens: {statistics.mean(new_ctx[n]['dacs']):.0f}")
            if new_ctx.get(n, {}).get("baseline"):
                print(f"    Baseline avg ctx tokens: {statistics.mean(new_ctx[n]['baseline']):.0f}")

    # Fit power laws with all 5 points
    Ns = sorted(all_dacs_acc.keys())
    N_arr = np.array(Ns, dtype=float)

    dacs_error = np.array([1 - statistics.mean(all_dacs_acc[n]) for n in Ns])
    dacs_se = np.array([statistics.stdev(all_dacs_acc[n]) / len(all_dacs_acc[n]) ** 0.5
                        for n in Ns])
    base_error = np.array([1 - statistics.mean(all_base_acc[n]) for n in Ns])
    base_se = np.array([statistics.stdev(all_base_acc[n]) / len(all_base_acc[n]) ** 0.5
                        for n in Ns])

    dacs_fit = fit_power_law(N_arr, dacs_error, dacs_se)
    base_fit = fit_power_law(N_arr, base_error, base_se)

    print("\n" + "=" * 70)
    print("  POWER LAW FITS: ε(N) = α · N^β")
    print("=" * 70)
    for label, fit in [("Baseline", base_fit), ("DACS", dacs_fit)]:
        print(f"\n  {label}:")
        print(f"    α = {fit['alpha']:.4f} ± {fit['alpha_se']:.4f}")
        print(f"    β = {fit['beta']:.4f} ± {fit['beta_se']:.4f}")
        print(f"    R² = {fit['r2']:.4f}")

    # Generate updated figure
    DACS_COL = "#2563EB"
    BASE_COL = "#DC2626"

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x_fine = np.linspace(2, 22, 200)

    # Baseline
    ax.errorbar(N_arr, base_error * 100, yerr=base_se * 100,
                fmt="s", color=BASE_COL, capsize=4, markersize=8, label="Baseline", zorder=5)
    ax.plot(x_fine, power_law(x_fine, base_fit["alpha"], base_fit["beta"]) * 100,
            "--", color=BASE_COL, alpha=0.6,
            label=f'Fit: $\\varepsilon = {base_fit["alpha"]:.3f} \\cdot N^{{{base_fit["beta"]:.2f}}}$, $R^2$={base_fit["r2"]:.3f}')

    # DACS
    ax.errorbar(N_arr, dacs_error * 100, yerr=dacs_se * 100,
                fmt="o", color=DACS_COL, capsize=4, markersize=8, label="DACS", zorder=5)
    ax.plot(x_fine, power_law(x_fine, dacs_fit["alpha"], dacs_fit["beta"]) * 100,
            "--", color=DACS_COL, alpha=0.6,
            label=f'Fit: $\\varepsilon = {dacs_fit["alpha"]:.3f} \\cdot N^{{{dacs_fit["beta"]:.2f}}}$')

    ax.set_xlabel("Number of agents (N)", fontsize=12)
    ax.set_ylabel("Error rate (%)", fontsize=12)
    ax.set_title("N-scaling: CIP validation with 5 data points ($N \\in \\{3, 5, 10, 15, 20\\}$)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 22)
    ax.set_ylim(0, 100)

    # Mark the new data points
    for n in [15, 20]:
        if all_base_acc.get(n):
            b_e = (1 - statistics.mean(all_base_acc[n])) * 100
            ax.annotate(f'N={n}\n(new)', (n, b_e), textcoords="offset points",
                       xytext=(10, 5), fontsize=8, color=BASE_COL, alpha=0.7)

    output_dir = os.path.join("paper", "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scaling_law_5pt.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
