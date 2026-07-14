import os
import tarfile

OBS_FOLDERS = [
    'data/sources/mrk335_2009',
    'data/sources/mrk335_2015',
    'data/sources/mrk335_2018',
    'data/sources/mrk335_2019',
]

for base_dir in OBS_FOLDERS:
    odf_path = os.path.join(base_dir, 'odf')
    print(f"\n📦 Processing {odf_path}...")
    
    # Find the .TAR file inside the odf folder
    tar_files = [f for f in os.listdir(odf_path) if f.lower().endswith('.tar')]
    
    if not tar_files:
        print("  ⚠️ No .TAR file found. Skipping.")
        continue
        
    tar_name = tar_files[0]
    tar_full_path = os.path.join(odf_path, tar_name)
    
    print(f"  -> Extracting {tar_name}...")
    with tarfile.open(tar_full_path, 'r') as tar:
        tar.extractall(path=odf_path)
        
    print(f"  ✅ Extracted successfully!")
    
    # Optional: remove the tar file to save disk space
    os.remove(tar_full_path)
    print(f"  🗑️ Deleted {tar_name} to save space.")

print("\n" + "="*60)
print("✅ RAW ODF FILES EXTRACTED AND READY FOR SAS!")
print("="*60)