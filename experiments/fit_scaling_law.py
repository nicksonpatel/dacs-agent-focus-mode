"""Scaling law analysis: fit error(N) and error(D) power laws.

Validates the Context Interference Principle (CIP):
  - Baseline error grows as α·N^β  (positive β ≈ 1 → linear interference)
  - DACS error is near-constant     (β ≈ 0)

Outputs:
  - paper/figures/scaling_law.png   (2-panel: N-scaling + D-scaling)
  - Prints fitted α, β, R² and 95% CI to stdout

Data sources: hardcoded from Phase 1–3 results (10 trials per point).
"""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Raw data from Phase 1-3 (means ± SE, 10 trials each)
# ---------------------------------------------------------------------------

# Phase 1: N-scaling (D ≈ 3 decisions/agent)
N_VALS = np.array([3, 5, 10])

# Per-trial accuracy data for proper error estimation
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

# D-scaling at N=3: s1_n3 (D≈3), s4_homogeneous (D=4), s8_dense_d3 (D=15)
D_VALS_N3 = np.array([3, 4, 15])
DACS_ACC_D_N3 = {
    3:  0.967,   # s1_n3 mean
    4:  0.902,   # s4_homogeneous mean
    15: 0.984,   # s8_dense_d3 mean
}
BASE_ACC_D_N3 = {
    3:  0.600,   # s1_n3 mean
    4:  0.525,   # s4_homogeneous mean
    15: 0.442,   # s8_dense_d3 mean
}
DACS_SE_D_N3 = {3: 0.021, 4: 0.035, 15: 0.005}
BASE_SE_D_N3 = {3: 0.115, 4: 0.017, 15: 0.013}

# D-scaling at N=5: s2_n5 (D≈3), s5_crossfire (D=4), s7_dense_d2 (D=8)
D_VALS_N5 = np.array([3, 4, 8])
DACS_ACC_D_N5 = {
    3: 0.967,   # s2_n5 mean
    4: 0.960,   # s5_crossfire mean
    8: 0.940,   # s7_dense_d2 mean
}
BASE_ACC_D_N5 = {
    3: 0.387,   # s2_n5 mean
    4: 0.370,   # s5_crossfire mean
    8: 0.348,   # s7_dense_d2 mean
}
DACS_SE_D_N5 = {3: 0.057, 4: 0.009, 8: 0.008}
BASE_SE_D_N5 = {3: 0.103, 4: 0.018, 8: 0.018}

# Context sizes (mean tokens) for Pareto data export
DACS_CTX_N = {3: 561, 5: 633, 10: 816}
BASE_CTX_N = {3: 1191, 5: 1720, 10: 2882}


# ---------------------------------------------------------------------------
# Power law model: error = α · x^β
# ---------------------------------------------------------------------------

def power_law(x, alpha, beta):
    return alpha * np.power(x, beta)


def fit_power_law(x, y, y_err=None):
    """Fit error = α·x^β via log-log linear regression + scipy refinement."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Initial guess from log-log OLS
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

    # R²
    y_pred = power_law(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": popt[0],
        "beta": popt[1],
        "alpha_se": perr[0],
        "beta_se": perr[1],
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def compute_all():
    """Compute all scaling law fits and return structured results."""
    results = {}

    # --- N-scaling ---
    dacs_error_n = np.array([1 - np.mean(DACS_ACC_N[n]) for n in N_VALS])
    dacs_se_n = np.array([np.std(DACS_ACC_N[n], ddof=1) / np.sqrt(len(DACS_ACC_N[n])) for n in N_VALS])
    base_error_n = np.array([1 - np.mean(BASE_ACC_N[n]) for n in N_VALS])
    base_se_n = np.array([np.std(BASE_ACC_N[n], ddof=1) / np.sqrt(len(BASE_ACC_N[n])) for n in N_VALS])

    results["n_scaling"] = {
        "N": N_VALS,
        "dacs_error": dacs_error_n,
        "dacs_se": dacs_se_n,
        "base_error": base_error_n,
        "base_se": base_se_n,
        "dacs_fit": fit_power_law(N_VALS, dacs_error_n, dacs_se_n),
        "base_fit": fit_power_law(N_VALS, base_error_n, base_se_n),
    }

    # --- D-scaling at N=3 ---
    dacs_err_d3 = np.array([1 - DACS_ACC_D_N3[d] for d in D_VALS_N3])
    dacs_se_d3 = np.array([DACS_SE_D_N3[d] for d in D_VALS_N3])
    base_err_d3 = np.array([1 - BASE_ACC_D_N3[d] for d in D_VALS_N3])
    base_se_d3 = np.array([BASE_SE_D_N3[d] for d in D_VALS_N3])

    results["d_scaling_n3"] = {
        "D": D_VALS_N3,
        "dacs_error": dacs_err_d3,
        "dacs_se": dacs_se_d3,
        "base_error": base_err_d3,
        "base_se": base_se_d3,
        "dacs_fit": fit_power_law(D_VALS_N3, dacs_err_d3, dacs_se_d3),
        "base_fit": fit_power_law(D_VALS_N3, base_err_d3, base_se_d3),
    }

    # --- D-scaling at N=5 ---
    dacs_err_d5 = np.array([1 - DACS_ACC_D_N5[d] for d in D_VALS_N5])
    dacs_se_d5 = np.array([DACS_SE_D_N5[d] for d in D_VALS_N5])
    base_err_d5 = np.array([1 - BASE_ACC_D_N5[d] for d in D_VALS_N5])
    base_se_d5 = np.array([BASE_SE_D_N5[d] for d in D_VALS_N5])

    results["d_scaling_n5"] = {
        "D": D_VALS_N5,
        "dacs_error": dacs_err_d5,
        "dacs_se": dacs_se_d5,
        "base_error": base_err_d5,
        "base_se": base_se_d5,
        "dacs_fit": fit_power_law(D_VALS_N5, dacs_err_d5, dacs_se_d5),
        "base_fit": fit_power_law(D_VALS_N5, base_err_d5, base_se_d5),
    }

    return results


def print_results(results):
    """Pretty-print all fit results."""
    for key, data in results.items():
        print(f"\n{'='*60}")
        print(f"  {key}")
        print(f"{'='*60}")
        for cond in ["dacs", "base"]:
            fit = data[f"{cond}_fit"]
            label = "DACS" if cond == "dacs" else "Baseline"
            print(f"\n  {label}:")
            print(f"    error(x) = {fit['alpha']:.4f} · x^{fit['beta']:.4f}")
            print(f"    α = {fit['alpha']:.4f} ± {fit['alpha_se']:.4f}")
            print(f"    β = {fit['beta']:.4f} ± {fit['beta_se']:.4f}")
            print(f"    R² = {fit['r2']:.4f}")


def plot_scaling_laws(results):
    """Generate the 2-panel scaling law figure."""
    DACS_COL = "#2563EB"
    BASE_COL = "#DC2626"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel (a): N-scaling ---
    ax = axes[0]
    data = results["n_scaling"]
    N = data["N"]
    x_fine = np.linspace(N.min() * 0.8, N.max() * 1.2, 100)

    # Baseline data + fit
    ax.errorbar(N, data["base_error"] * 100, yerr=data["base_se"] * 100,
                fmt="s", color=BASE_COL, capsize=4, markersize=8, label="Baseline", zorder=5)
    bf = data["base_fit"]
    ax.plot(x_fine, power_law(x_fine, bf["alpha"], bf["beta"]) * 100,
            "--", color=BASE_COL, alpha=0.6,
            label=f'Fit: $\\alpha N^{{\\beta}}$, $\\beta$={bf["beta"]:.2f}, $R^2$={bf["r2"]:.3f}')

    # DACS data + fit
    ax.errorbar(N, data["dacs_error"] * 100, yerr=data["dacs_se"] * 100,
                fmt="o", color=DACS_COL, capsize=4, markersize=8, label="DACS", zorder=5)
    df = data["dacs_fit"]
    ax.plot(x_fine, power_law(x_fine, df["alpha"], df["beta"]) * 100,
            "--", color=DACS_COL, alpha=0.6,
            label=f'Fit: $\\alpha N^{{\\beta}}$, $\\beta$={df["beta"]:.2f}')

    ax.set_xlabel("Number of agents (N)", fontsize=11)
    ax.set_ylabel("Error rate (%)", fontsize=11)
    ax.set_title("(a) Agent count scaling", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 12)
    ax.set_ylim(0, 100)

    # --- Panel (b): D-scaling (both N=3 and N=5) ---
    ax = axes[1]

    # N=3 trajectory
    d3 = results["d_scaling_n3"]
    ax.errorbar(d3["D"], d3["base_error"] * 100, yerr=d3["base_se"] * 100,
                fmt="s", color=BASE_COL, capsize=4, markersize=7, label="Baseline (N=3)", zorder=5)
    ax.errorbar(d3["D"], d3["dacs_error"] * 100, yerr=d3["dacs_se"] * 100,
                fmt="o", color=DACS_COL, capsize=4, markersize=7, label="DACS (N=3)", zorder=5)

    # N=5 trajectory
    d5 = results["d_scaling_n5"]
    ax.errorbar(d5["D"], d5["base_error"] * 100, yerr=d5["base_se"] * 100,
                fmt="^", color=BASE_COL, capsize=4, markersize=7, alpha=0.7,
                label="Baseline (N=5)", zorder=5)
    ax.errorbar(d5["D"], d5["dacs_error"] * 100, yerr=d5["dacs_se"] * 100,
                fmt="D", color=DACS_COL, capsize=4, markersize=7, alpha=0.7,
                label="DACS (N=5)", zorder=5)

    # Fit curves for N=3
    d_fine = np.linspace(2, 16, 100)
    bf3 = d3["base_fit"]
    ax.plot(d_fine, power_law(d_fine, bf3["alpha"], bf3["beta"]) * 100,
            "--", color=BASE_COL, alpha=0.4)
    df3 = d3["dacs_fit"]
    ax.plot(d_fine, power_law(d_fine, df3["alpha"], df3["beta"]) * 100,
            "--", color=DACS_COL, alpha=0.4)

    ax.set_xlabel("Decision density (D)", fontsize=11)
    ax.set_ylabel("Error rate (%)", fontsize=11)
    ax.set_title("(b) Decision density scaling", fontsize=12)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    fig.suptitle("Scaling laws: Context Interference Principle validation", fontsize=13, y=1.02)
    fig.tight_layout()

    output_dir = os.path.join("paper", "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scaling_law.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    results = compute_all()
    print_results(results)
    plot_scaling_laws(results)


if __name__ == "__main__":
    main()
