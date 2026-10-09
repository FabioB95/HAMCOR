import os
import tarfile
import shutil

# Mappa dei file tarball e delle cartelle di destinazione
TARBALLS = {
    '0600540601.tar': 'data/sources/mrk335_2009',
    '0741280201.tar': 'data/sources/mrk335_2015',
    '0780500301.tar': 'data/sources/mrk335_2018',
    '0831790601.tar': 'data/sources/mrk335_2019',
}

for tar_name, dest_dir in TARBALLS.items():
    if not os.path.exists(tar_name):
        print(f"⚠️  {tar_name} non trovato. Scaricalo dal sito XSA e mettilo in questa cartella.")
        continue
        
    print(f"\n📦 Estrazione di {tar_name}...")
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Estrai il tarball
    with tarfile.open(tar_name, 'r') as tar:
        tar.extractall(path=dest_dir)
        
    # 2. I tarball XSA di solito creano una sottocartella col nome dell'ObsID.
    #    Lo script SAS si aspetta che 'odf' sia direttamente dentro dest_dir.
    #    Quindi spostiamo i contenuti della sottocartella in dest_dir.
    extracted_dirs = [d for d in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, d))]
    if extracted_dirs:
        extracted_path = os.path.join(dest_dir, extracted_dirs[0])
        if os.path.exists(os.path.join(extracted_path, 'odf')):
            print(f"  -> Riorganizzo i file in {dest_dir} ...")
            for item in os.listdir(extracted_path):
                s = os.path.join(extracted_path, item)
                d = os.path.join(dest_dir, item)
                if os.path.exists(d):
                    if os.path.isdir(d): shutil.rmtree(d)
                    else: os.remove(d)
                shutil.move(s, d)
            os.rmdir(extracted_path) # Rimuovi la cartella ObsID ormai vuota
            
    print(f"✅ {tar_name} estratto e organizzato con successo.")

print("\n" + "="*60)
print("✅ TUTTI I DATI SONO PRONTI!")
print("="*60)
print("Ora apri WSL (Windows Subsystem for Linux) e lancia la pipeline SAS.")