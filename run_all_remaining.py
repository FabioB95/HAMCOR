"""
run_all_remaining.py
──────────────────────
Lancia in background tutti gli script rimanenti in parallelo,
con output su file di log separati. Lancialo una volta e torna
quando vuoi — non serve stare a guardare il terminale.
"""
import subprocess
import os

os.makedirs("logs", exist_ok=True)

jobs = [
    ("run_mrk335_centroid.py",   "logs/mrk335_centroid.log"),
    ("run_iras13224_centroid.py","logs/iras13224_centroid.log"),
]

procs = []
for script, logfile in jobs:
    f = open(logfile, "w")
    p = subprocess.Popen(
        ["python", script], stdout=f, stderr=subprocess.STDOUT
    )
    procs.append((script, p, f))
    print(f"Lanciato {script} -> {logfile} (PID {p.pid})")

print("\nTutti i job lanciati in background.")
print("Controlla lo stato con: Get-Content logs\\mrk335_centroid.log -Tail 20 -Wait")
print("(Ctrl+C per smettere di seguire, i processi continuano comunque)")

for script, p, f in procs:
    p.wait()
    f.close()
    print(f"{script} completato.")