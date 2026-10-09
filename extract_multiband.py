"""
extract_multiband.py
──────────────────────
Estrae bande energetiche AGGIUNTIVE riusando la calibrazione e il GTI
già prodotti da run_sas_pipeline.py (Step 1-3, non li rifà — solo
Step 4 con più bande).

Bande scelte per matchare il demo di lag_energy_psi.py:
  0.3-0.5, 0.5-0.8, 0.8-1.2, 1.2-2.0 keV (in eV per PI: 300-500 ecc.)
Puoi modificare BANDS se preferisci una scelta diversa.

Prerequisito: aver già girato run_sas_pipeline.py almeno una volta per
l'epoca in questione (serve gti.fits e *PN*ImagingEvts.ds nella cartella
odf/), e aver lanciato 'setsas' nel terminale.
"""

import os
import subprocess
import numpy as np
from astropy.io import fits

# ── stesse 4 epoche di run_sas_pipeline.py ────────────────────────────────────
OBS_CONFIGS = [
    {'obsid': '0600540601', 'year': 2009, 'dir': 'data/sources/mrk335_2009'},
    {'obsid': '0741280201', 'year': 2015, 'dir': 'data/sources/mrk335_2015'},
    {'obsid': '0780500301', 'year': 2018, 'dir': 'data/sources/mrk335_2018'},
    {'obsid': '0831790601', 'year': 2019, 'dir': 'data/sources/mrk335_2019'},
]

# ── bande aggiuntive (eV) — modificabile ──────────────────────────────────────
BANDS = [
    ('b1', 300,  500),
    ('b2', 500,  800),
    ('b3', 800,  1200),
    ('b4', 1200, 2000),
]

def run_sas_cmd(cmd, cwd):
    print(f"  > Running: {cmd[:70]}...")
    result = subprocess.run(cmd, shell=True, executable='/bin/bash',
                             cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ❌ ERROR in command: {cmd}")
        print(result.stderr)
        raise RuntimeError(f"SAS command failed: {cmd}")
    return result.stdout

def extract_bands_for_epoch(obsid, year, base_dir):
    print(f"\n{'='*60}")
    print(f"Extra bands — {year} (ObsID: {obsid})")
    print(f"{'='*60}")

    odf_dir = os.path.join(base_dir, 'odf')
    gti_path = os.path.join(odf_dir, 'gti.fits')
    if not os.path.exists(gti_path):
        print(f"  ⚠️ {gti_path} non trovato — devi prima girare "
              f"run_sas_pipeline.py per questa epoca (Step 1-3).")
        return

    for band_name, pi_min, pi_max in BANDS:
        fname = f'{band_name}_lc.fits'
        cmd = f"""
        evselect table=*PN*ImagingEvts.ds:EVENTS \
          withrateset=yes rateset={fname} \
          expression='GTI(gti.fits,TIME)&&PATTERN<=4&&FLAG==0&&PI>{pi_min}&&PI<{pi_max}' \
          timebinsize=100 maketimecolumn=yes makeratecolumn=yes
        """
        run_sas_cmd(cmd, cwd=odf_dir)

    convert_extra_bands_to_numpy(odf_dir, base_dir, year)

def convert_extra_bands_to_numpy(odf_dir, out_dir, year):
    for band_name, pi_min, pi_max in BANDS:
        fpath = os.path.join(odf_dir, f'{band_name}_lc.fits')
        if not os.path.exists(fpath):
            print(f"  ⚠️ {fpath} non trovato, salto.")
            continue
        with fits.open(fpath) as hdul:
            data = hdul[1].data
            rate = data['RATE']
            np.save(os.path.join(out_dir, f'{band_name}_lc.npy'), rate)
            print(f"  ✅ Salvato {band_name}_lc.npy "
                  f"({pi_min}-{pi_max} eV) per {year}")

if __name__ == "__main__":
    print("🚀 Estrazione bande aggiuntive per lag-energy spectrum (Mrk 335)...")
    print("⚠️  Assicurati di aver lanciato 'setsas' in questo terminale!\n")

    for obs in OBS_CONFIGS:
        extract_bands_for_epoch(obs['obsid'], obs['year'], obs['dir'])

    print("\n" + "=" * 60)
    print("✅ Bande extra estratte per tutte le epoche disponibili")
    print("=" * 60)