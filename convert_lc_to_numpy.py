"""
convert_lc_to_numpy.py
───────────────────────
Converts SAS-extracted FITS light curves to .npy format for HAMCOR.

Run AFTER SAS extraction for each epoch:
    python convert_lc_to_numpy.py --year 2009
    python convert_lc_to_numpy.py --year 2015
    python convert_lc_to_numpy.py --year 2018
    python convert_lc_to_numpy.py --year 2019

Expects these FITS files in data/sources/mrk335_YEAR/:
    soft_lc.fits    (0.3-1.0 keV, from evselect)
    hard_lc.fits    (1.5-4.0 keV, from evselect)

Output (in same directory):
    soft_lc.npy
    hard_lc.npy
    time.npy
"""

import argparse, os, sys
import numpy as np

try:
    from astropy.io import fits
except ImportError:
    print("astropy not found: pip install astropy --break-system-packages")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('--year', required=True, type=int)
args   = parser.parse_args()

if args.year == 2006:
    DATA_DIR = 'data/sources/mrk335'
else:
    DATA_DIR = f'data/sources/mrk335_{args.year}'

if not os.path.exists(DATA_DIR):
    print(f"ERROR: {DATA_DIR} does not exist")
    sys.exit(1)

print(f"Converting light curves for Mrk 335 ({args.year})")
print(f"Directory: {DATA_DIR}")

for band, fname in [('soft', 'soft_lc.fits'), ('hard', 'hard_lc.fits')]:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        print(f"  Run evselect in WSL to produce {fname}")
        sys.exit(1)

    with fits.open(path) as hdul:
        # SAS light curves: extension 1 = RATE table
        # Columns: TIME, RATE, ERROR, FRACEXP, BACKV, BACKE
        data = hdul[1].data
        time = data['TIME'].astype(float)
        rate = data['RATE'].astype(float)
        err  = data['ERROR'].astype(float)

        # Replace NaN/negative with 0 (GTI gaps)
        bad  = ~np.isfinite(rate) | (rate < 0)
        rate[bad] = 0.0

    npy_path = os.path.join(DATA_DIR, f'{band}_lc.npy')
    np.save(npy_path, rate)
    print(f"  {band}: {len(rate)} bins, mean={rate[rate>0].mean():.2f} ct/s"
          f"  → {npy_path}")

    if band == 'soft':
        time_path = os.path.join(DATA_DIR, 'time.npy')
        np.save(time_path, time)
        print(f"  time: {(time[-1]-time[0])/1000:.1f} ks → {time_path}")

print(f"\nDone. Now run:")
print(f"  python run_lag_mrk335_epoch.py --obsid OBSID --year {args.year}")
