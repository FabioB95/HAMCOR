import os
import subprocess
import numpy as np
from astropy.io import fits

# ── Configurazione delle 4 osservazioni ──────────────────────────────────────
# Assicurati che i tarball ODF siano già stati scaricati ed estratti in queste directory.
OBS_CONFIGS = [
    {'obsid': '0600540601', 'year': 2009, 'dir': 'data/sources/mrk335_2009'},
    {'obsid': '0741280201', 'year': 2015, 'dir': 'data/sources/mrk335_2015'},
    {'obsid': '0780500301', 'year': 2018, 'dir': 'data/sources/mrk335_2018'},
    {'obsid': '0831790601', 'year': 2019, 'dir': 'data/sources/mrk335_2019'},
]

def run_sas_cmd(cmd, cwd):
    """Esegue un comando SAS nella directory specificata."""
    print(f"  > Running: {cmd[:60]}...")
    # Esegue in bash, ereditando l'ambiente SAS (assicurati di aver lanciato setsas prima di eseguire lo script!)
    result = subprocess.run(cmd, shell=True, executable='/bin/bash', cwd=cwd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\n  ❌ ERROR in command: {cmd}")
        print(result.stderr)
        raise RuntimeError(f"SAS command failed: {cmd}")
    
    return result.stdout

def process_epoch(obsid, year, base_dir):
    """Esegue la pipeline SAS completa per una singola epoca."""
    print(f"\n{'='*60}")
    print(f"Processing {year} (ObsID: {obsid})")
    print(f"{'='*60}")
    
    odf_dir = os.path.join(base_dir, 'odf')
    if not os.path.exists(odf_dir):
        print(f"  ⚠️ WARNING: {odf_dir} not found. Please extract the ODF tarball first.")
        return

    # ── Step 1: Setup SAS ────────────────────────────────────────────────────
    print("Step 1: Setup SAS (cifbuild, odfingest)")
    # Imposta SAS_ODF sulla cartella odf, esegue cifbuild e odfingest
    setup_cmd = """
    export SAS_ODF=$(pwd)
    cifbuild
    odfingest findinstrumentmodes=no usehousekeeping=no
    export SAS_ODF=$(pwd)/SUM.SAS
    """
    run_sas_cmd(setup_cmd, cwd=odf_dir)
    
    # ── Step 2: Calibrazione ─────────────────────────────────────────────────
    print("Step 2: Calibration (epproc)")
    # epproc processa i dati pn. SAS_ODF deve puntare a SUM.SAS
    run_sas_cmd("epproc", cwd=odf_dir)
    
    # ── Step 3: Filtra flares ────────────────────────────────────────────────
    print("Step 3: Flare filtering")
    evselect_cmd = """
    evselect table=*PN*ImagingEvts.ds:EVENTS \
      expression='PI>10000&&PI<12000&&PATTERN==0' \
      timebinsize=100 rateset=PN_hilo_lc.fits \
      maketimecolumn=yes makeratecolumn=yes
    """
    run_sas_cmd(evselect_cmd, cwd=odf_dir)
    
    tabgtigen_cmd = "tabgtigen table=PN_hilo_lc.fits gtiset=gti.fits expression='RATE<=0.5'"
    run_sas_cmd(tabgtigen_cmd, cwd=odf_dir)
    
    # ── Step 4: Estrai curve di luce soft + hard ─────────────────────────────
    print("Step 4: Extract lightcurves (Soft & Hard bands)")
    
    soft_cmd = """
    evselect table=*PN*ImagingEvts.ds:EVENTS \
      withrateset=yes rateset=soft_lc.fits \
      expression='GTI(gti.fits,TIME)&&PATTERN<=4&&FLAG==0&&PI>150&&PI<1000' \
      timebinsize=100 maketimecolumn=yes makeratecolumn=yes
    """
    run_sas_cmd(soft_cmd, cwd=odf_dir)
    
    hard_cmd = """
    evselect table=*PN*ImagingEvts.ds:EVENTS \
      withrateset=yes rateset=hard_lc.fits \
      expression='GTI(gti.fits,TIME)&&PATTERN<=4&&FLAG==0&&PI>1500&&PI<4000' \
      timebinsize=100 maketimecolumn=yes makeratecolumn=yes
    """
    run_sas_cmd(hard_cmd, cwd=odf_dir)
    
    # ── Step 5: Converti in numpy ────────────────────────────────────────────
    print("Step 5: Convert FITS lightcurves to numpy arrays")
    convert_lc_to_numpy(odf_dir, year)

def convert_lc_to_numpy(odf_dir, year):
    """Legge i file FITS delle curve di luce e salva TIME e RATE in file .npy."""
    for band, fname in [('soft', 'soft_lc.fits'), ('hard', 'hard_lc.fits')]:
        fpath = os.path.join(odf_dir, fname)
        if not os.path.exists(fpath):
            print(f"  ⚠️ WARNING: {fpath} not found. Skipping {band} band.")
            continue
        
        with fits.open(fpath) as hdul:
            # Le curve di luce prodotte da evselect sono nella prima estensione della tabella
            data = hdul[1].data 
            time = data['TIME']
            rate = data['RATE']
            
            # Salva nella directory principale dell'epoca (non in odf/) per pulizia
            out_dir = os.path.dirname(odf_dir)
            np.save(os.path.join(out_dir, f'{band}_lc.npy'), rate)
            np.save(os.path.join(out_dir, f'time.npy'), time)
            print(f"  ✅ Saved {band}_lc.npy and time.npy for {year}")

if __name__ == "__main__":
    print("🚀 Starting SAS Pipeline for Mrk 335 multi-epoch analysis...")
    print("⚠️  Make sure you have run 'setsas' in your terminal before running this script!\n")
    
    for obs in OBS_CONFIGS:
        process_epoch(obs['obsid'], obs['year'], obs['dir'])
        
    print("\n" + "="*60)
    print("✅ ALL 4 EPOCHS PROCESSED SUCCESSFULLY!")
    print("="*60)