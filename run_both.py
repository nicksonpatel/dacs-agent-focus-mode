#!/usr/bin/env python3
"""Chain runner: ra1_n3 → ra2_n5 sequentially, both in background."""
import subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).parent

def run(scenario, log):
    log_path = ROOT / "logs" / log
    log_path.parent.mkdir(exist_ok=True)
    print(f"\n>>> Starting {scenario} → {log_path}", flush=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "experiments_real_agent.run",
            "--scenario", scenario,
            "--mode", "both",
            "--trials", "10",
            "--parallel-trials", "4",
            "--max-concurrent", "10",
        ],
        cwd=ROOT,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f">>> {scenario} finished (exit={result.returncode})", flush=True)
    return result.returncode

if __name__ == "__main__":
    rc1 = run("ra1_n3", "ra1_n3_haiku.log")
    if rc1 != 0:
        print("ra1_n3 failed — skipping ra2_n5", flush=True)
        sys.exit(rc1)
    rc2 = run("ra2_n5", "ra2_n5_haiku.log")
    sys.exit(rc2)
