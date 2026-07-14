"""
run_all.py  —  HAMCOR master pipeline script.

Runs every analysis step in the correct order and logs
timing + success/failure for each step.

Usage (from project root, inside venv):
    python run_all.py

Optional: skip slow steps during testing
    python run_all.py --skip-sensitivity
    python run_all.py --skip-fits
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--skip-sensitivity', action='store_true',
                    help='Skip hyperparameter sensitivity scan (slow ~30 min)')
parser.add_argument('--skip-fits', action='store_true',
                    help='Skip real-data fits (slow ~20 min each)')
parser.add_argument('--skip-recovery', action='store_true',
                    help='Skip synthetic recovery run (use existing .npz)')
args = parser.parse_args()

# ── pipeline definition ───────────────────────────────────────────────────────
# Each entry: (label, script, skip_flag)
STEPS = [
    # ── Step 1: Lag measurements ───────────────────────────────────────────
    ("Lag spectrum — 1H 0707-495",       "run_lag_1h0707.py",        False),
    ("Lag spectrum — IRAS 13224",        "compute_lag_iras13224.py", False),
    # MCG-6-30-15 and Mrk 335 lags are computed inside their fit scripts.

    # ── Step 2: Synthetic validation ──────────────────────────────────────
    ("Synthetic recovery (raw runs)",    "run_recovery.py",          args.skip_recovery),
    ("Recovery figure v3",               "make_figure_v3.py",        args.skip_recovery),

    # ── Step 3: Hyperparameter sensitivity ────────────────────────────────
    ("Hyperparameter sensitivity scan",  "run_sensitivity.py",       args.skip_sensitivity),

    # ── Step 4: Real-data fits ─────────────────────────────────────────────
    ("HAMCOR fit — Mrk 335",             "run_mrk335.py",            args.skip_fits),
    ("HAMCOR fit — 1H 0707-495",         "run_1h0707.py",            args.skip_fits),
    ("HAMCOR fit — IRAS 13224",          "run_iras13224.py",         args.skip_fits),
    ("HAMCOR fit — MCG-6-30-15",         "run_mcg6.py",              args.skip_fits),
]

# ── helpers ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def hms(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    return f"{s}s"

# ── run ───────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  HAMCOR — Full Pipeline Run{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

results = []
total_start = time.time()

for label, script, skip in STEPS:
    if skip:
        print(f"{YELLOW}  SKIP  {RESET} {label}")
        results.append((label, "SKIPPED", 0.0))
        continue

    if not Path(script).exists():
        print(f"{RED}  ERROR {RESET} {label}  —  {script} not found")
        results.append((label, "MISSING", 0.0))
        continue

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  ▶  {label}{RESET}")
    print(f"{'─'*60}")

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # stream output live
    )
    elapsed = time.time() - t0

    if proc.returncode == 0:
        status = "OK"
        icon   = f"{GREEN}  OK    {RESET}"
    else:
        status = f"FAILED (exit {proc.returncode})"
        icon   = f"{RED}  FAIL  {RESET}"

    results.append((label, status, elapsed))
    print(f"\n{icon} {label}  [{hms(elapsed)}]")

# ── summary ───────────────────────────────────────────────────────────────────
total = time.time() - total_start
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  SUMMARY{RESET}")
print(f"{BOLD}{'='*60}{RESET}")

all_ok = True
for label, status, elapsed in results:
    if status == "SKIPPED":
        icon = f"{YELLOW}SKIP   {RESET}"
    elif status == "OK":
        icon = f"{GREEN}OK     {RESET}"
    else:
        icon = f"{RED}FAIL   {RESET}"
        all_ok = False
    t_str = f"[{hms(elapsed)}]" if elapsed > 0 else ""
    print(f"  {icon} {label:<45} {t_str}")

print(f"\n  Total time: {hms(total)}")

if all_ok:
    print(f"\n{GREEN}{BOLD}  All steps completed successfully.{RESET}")
    print(f"  Results saved to: results/\n")
else:
    print(f"\n{RED}{BOLD}  One or more steps failed — check output above.{RESET}\n")
    sys.exit(1)
