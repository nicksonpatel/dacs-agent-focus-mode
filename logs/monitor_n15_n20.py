"""Monitor N=15/N=20 experiments. Auto-restarts if stuck or dead."""
import csv
import os
import subprocess
import sys
import time

LOG = "logs/n15_n20_run.log"
CSV_PATH = "results/summary.csv"
SCENARIOS = {"s9_n15", "s10_n20"}
STUCK_THRESHOLD = 120   # seconds without log file modification
CHECK_INTERVAL = 30     # seconds between checks
TARGET_ROWS = 40        # 10 trials x 2 conditions x 2 scenarios


def row_count():
    try:
        count = 0
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("scenario", "") in SCENARIOS:
                    count += 1
        return count
    except Exception:
        return 0


def done_by_key():
    """Return {(scenario, condition): count} for already-completed trials."""
    tally = {}
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sc = row.get("scenario", "")
                cond = row.get("condition", "")
                if sc in SCENARIOS:
                    k = (sc, cond)
                    tally[k] = tally.get(k, 0) + 1
    except Exception:
        pass
    return tally


def last_log_mtime():
    try:
        return os.path.getmtime(LOG)
    except Exception:
        return 0.0


def process_alive():
    r = subprocess.run(
        ["pgrep", "-f", "s9_n15"],
        capture_output=True,
    )
    return r.returncode == 0


def launch():
    p = subprocess.Popen(
        [
            "python", "-m", "experiments.run_experiment",
            "--scenario", "s9_n15", "s10_n20",
            "--trials", "10",
            "--mode", "both",
            "--parallel-trials", "2",
        ],
        stdout=open(LOG, "a"),
        stderr=subprocess.STDOUT,
    )
    print(f"  → Launched PID {p.pid}", flush=True)
    return p.pid


def main():
    rows = row_count()
    print(f"[monitor] start — rows={rows}/{TARGET_ROWS}", flush=True)

    if not process_alive():
        print("[monitor] No process running — launching now.", flush=True)
        launch()
        time.sleep(5)

    last_activity = last_log_mtime()
    cycle = 0

    while True:
        time.sleep(CHECK_INTERVAL)
        cycle += 1
        rows = row_count()
        mtime = last_log_mtime()
        alive = process_alive()
        idle = time.time() - mtime

        print(
            f"[cycle {cycle:3d}] rows={rows:2d}/{TARGET_ROWS}  "
            f"alive={alive}  idle={idle:.0f}s",
            flush=True,
        )

        if rows >= TARGET_ROWS:
            print("[monitor] ✅ ALL 40 TRIALS COMPLETE", flush=True)
            break

        stuck = idle > STUCK_THRESHOLD
        if not alive or stuck:
            reason = "dead" if not alive else f"stuck ({idle:.0f}s idle)"
            print(f"[monitor] ⚠️  Process {reason} — restarting", flush=True)
            subprocess.run(["pkill", "-9", "-f", "s9_n15"], capture_output=True)
            time.sleep(3)
            launch()
            time.sleep(5)
            last_activity = last_log_mtime()
        else:
            last_activity = mtime


if __name__ == "__main__":
    main()
