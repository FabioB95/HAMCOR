#!/usr/bin/env bash
# HAMCOR v2 — full pipeline (reproduces all numbers, tables and figures of the paper).
# Run from the repository root inside WSL2/Linux with SAS and HEASoft initialised:
#   bash run_all.sh            # everything
#   STEP=5 bash run_all.sh     # restart from step 5
#   JOBS=4 bash run_all.sh     # number of parallel workers (default: all cores)
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STEP=${STEP:-0}
JOBS=${JOBS:-$(nproc)}
log() { echo -e "\n\033[1m=== [$1] $2 ===\033[0m"; }

if [ "$STEP" -le 0 ];  then log 0  "unit tests";                         python scripts/s00_tests.py; fi
if [ "$STEP" -le 1 ];  then log 1  "SAS extraction (light curves, spectra)"; python scripts/s01_sas_extract.py all; fi
if [ "$STEP" -le 2 ];  then log 2  "lag-frequency and lag-energy spectra";  python scripts/s02_lagfreq.py all; fi
if [ "$STEP" -le 3 ];  then log 3  "reflection fits (XSPEC + relxill)";     python scripts/s03_fit_spectra.py all; fi
if [ "$STEP" -le 4 ];  then log 4  "synthetic validation";                  python scripts/s05_synthetic.py --jobs "$JOBS"; fi
if [ "$STEP" -le 5 ];  then log 5  "HAMCOR fits (Schwarzschild)";           python scripts/s06_fit_sources.py all --jobs "$JOBS"; fi
if [ "$STEP" -le 6 ];  then log 6  "lamppost comparison";                   python scripts/s07_lamppost.py all; fi
if [ "$STEP" -le 7 ];  then log 7  "lag-energy predictive check";           python scripts/s08_lag_energy.py all; fi
if [ "$STEP" -le 8 ];  then log 8  "tables";                                python scripts/s09_tables.py; fi
if [ "$STEP" -le 9 ];  then log 9  "Kerr transfer (a = 0.998 and 0)";       python scripts/s10_kerr.py all --jobs "$JOBS" --spins 0.998,0.0; fi
if [ "$STEP" -le 10 ]; then log 10 "rest-frame xillver templates (XSPEC)";  python scripts/s11_xillver_templates.py all; fi
if [ "$STEP" -le 11 ]; then log 11 "energy-resolved joint fits";            python scripts/s12_energy_resolved.py all --jobs "$JOBS"; fi
if [ "$STEP" -le 12 ]; then log 12 "reflection-fraction robustness";        python scripts/s13_reflection_robustness.py mrk335_2006; fi
log done "results/ (json, tables/*.tex, figures/*/*.{dat,tex})"
