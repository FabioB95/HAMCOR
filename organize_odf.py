import os
import shutil

# Mappa delle cartelle da sistemare
OBS_FOLDERS = {
    'data/sources/mrk335_2009': '0600540601',
    'data/sources/mrk335_2015': '0741280201',
    'data/sources/mrk335_2018': '0780500301',
    'data/sources/mrk335_2019': '0831790601',
}

for base_dir, expected_obsid in OBS_FOLDERS.items():
    print(f"\n📁 Sistemazione di {base_dir}...")
    
    if not os.path.exists(base_dir):
        print(f"  ⚠️  {base_dir} non esiste. Creo la cartella...")
        os.makedirs(base_dir)
        continue
    
    # Lista di tutti i file nella cartella
    items = os.listdir(base_dir)
    
    # Cerca se c'è già una cartella 'odf'
    if 'odf' in items:
        print(f"  ✅ La cartella 'odf' esiste già. Verifico il contenuto...")
        odf_items = os.listdir(os.path.join(base_dir, 'odf'))
        if len(odf_items) > 0:
            print(f"  ✅ Trovati {len(odf_items)} file in odf/. Tutto OK!")
            continue
        else:
            print(f"  ⚠️  La cartella odf/ è vuota. La riempio...")
    else:
        print(f"  -> Creo cartella 'odf'...")
        os.makedirs(os.path.join(base_dir, 'odf'))
    
    # Sposta tutti i file e le cartelle (tranne 'odf' stesso) dentro odf/
    odf_path = os.path.join(base_dir, 'odf')
    moved_count = 0
    
    for item in items:
        if item == 'odf':
            continue  # Non spostare la cartella odf dentro se stessa!
        
        src = os.path.join(base_dir, item)
        dst = os.path.join(odf_path, item)
        
        if os.path.isdir(src):
            shutil.move(src, dst)
            print(f"  -> Spostata cartella: {item}/")
        else:
            shutil.move(src, dst)
            print(f"  -> Spostato file: {item}")
        moved_count += 1
    
    print(f"  ✅ Spostati {moved_count} elementi in odf/")

print("\n" + "="*60)
print("✅ STRUTTURA CARTELLE PRONTA PER SAS!")
print("="*60)
print("\nOra puoi procedere con la pipeline SAS in WSL:")
print("1. Apri WSL")
print("2. cd /mnt/c/Users/f.buffoli008/Desktop/QCORONA")
print("3. source /path/to/sas/setsas")
print("4. python run_sas_pipeline.py")