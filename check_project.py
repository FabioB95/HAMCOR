#!/usr/bin/env python3
"""
QCORONA Project Structure Analyzer - versione senza venv

Esegue l'analisi del progetto escludendo cartelle di ambienti virtuali
(venv, .venv, env, .env, virtualenv, etc.)
"""

import os
from pathlib import Path

def get_file_status(filepath):
    """Check if file exists and has content."""
    if not os.path.exists(filepath):
        return "❌ MISSING"
    
    size = os.path.getsize(filepath)
    if size == 0:
        return "⚪ EMPTY"
    elif size < 100:
        return f"🟡 SMALL ({size} bytes)"
    else:
        return f"✅ OK ({size} bytes)"

def count_lines(filepath):
    """Count lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except:
        return 0

def should_skip_directory(dirname):
    """Decide quali cartelle ignorare completamente"""
    skip_names = {
        'venv', '.venv', 'env', '.env', 'virtualenv', '__pycache__',
        '.pytest_cache', 'node_modules', '.git', '.idea', '.vscode',
        'dist', 'build', 'eggs', '*.egg-info', 'htmlcov'
    }
    lower_name = dirname.lower()
    return (
        lower_name in skip_names or
        lower_name.startswith('.') or
        lower_name.endswith('.egg-info') or
        lower_name.startswith('venv-') or
        lower_name.startswith('env-')
    )

def main():
    # Expected project structure (lasciata invariata)
    expected_structure = {
        "config.yaml": "Configuration file with hyperparameters",
        "pyproject.toml": "Python package configuration",
        "qcorona/__init__.py": "Package init",
        "qcorona/geometry/__init__.py": "Geometry module init",
        "qcorona/geometry/grid.py": "Grid discretization (CORE)",
        "qcorona/physics/__init__.py": "Physics module init",
        "qcorona/physics/illumination.py": "Disk illumination",
        "qcorona/physics/lags.py": "Reverberation lags",
        "qcorona/physics/compactness.py": "Pair-production limits",
        "qcorona/physics/energy_budget.py": "Energy constraints",
        "qcorona/hamiltonian/__init__.py": "Hamiltonian module init",
        "qcorona/hamiltonian/terms.py": "Individual H terms",
        "qcorona/hamiltonian/total.py": "Full Hamiltonian class",
        "qcorona/optimization/__init__.py": "Optimization module init",
        "qcorona/optimization/classical.py": "Gradient descent optimizer (TODO)",
        "qcorona/data/__init__.py": "Data module init",
        "qcorona/validation/__init__.py": "Validation module init",
        "qcorona/visualization/__init__.py": "Visualization module init",
        "tests/test_grid.py": "Grid tests",
        "tests/test_physics.py": "Physics tests",
        "tests/test_hamiltonian.py": "Hamiltonian tests",
    }
    
    print("=" * 70)
    print("QCORONA PROJECT STRUCTURE ANALYSIS  (senza venv)")
    print("=" * 70)
    print()
    
    if not os.path.exists("qcorona"):
        print("⚠️  ATTENZIONE: cartella 'qcorona' non trovata!")
        print("   Assicurati di eseguire lo script dalla root del progetto.")
        print("   Directory corrente:", os.getcwd())
        print()

    # -------------------------------
    #  Parte 1: controllo file attesi
    # -------------------------------
    print("📁 STATO FILE ATTESI:")
    print("-" * 70)
    
    total_files = ok_files = missing_files = empty_files = 0
    
    for filepath, description in expected_structure.items():
        status = get_file_status(filepath)
        lines = count_lines(filepath) if os.path.exists(filepath) else 0
        
        total_files += 1
        if "OK" in status:
            ok_files += 1
        elif "MISSING" in status:
            missing_files += 1
        elif "EMPTY" in status:
            empty_files += 1
        
        print(f"  {status:25s} {filepath}")
        if lines > 0:
            print(f"  {'':25s}   └── {lines} righe | {description}")
    
    print()
    print("-" * 70)
    print(f"SINTESI: {ok_files}/{total_files} OK, "
          f"{empty_files} vuoti, {missing_files} mancanti")
    print()

    # -------------------------------
    #  Parte 2: albero directory reale (escludendo venv & simili)
    # -------------------------------
    print("📂 STRUTTURA REALE DEL PROGETTO (esclusi ambienti virtuali):")
    print("-" * 70)
    
    for root, dirs, files in os.walk(".", topdown=True):
        # MODIFICA PRINCIPALE: filtriamo le directory da esplorare
        dirs[:] = [d for d in dirs if not should_skip_directory(d)]
        
        level = root.replace(".", "").count(os.sep)
        indent = "  " * level
        print(f"{indent}📁 {os.path.basename(root) or '.'}/")
        
        subindent = "  " * (level + 1)
        for file in sorted(files):
            if file.startswith('.') or file.endswith('.pyc') or file.endswith('.pyo'):
                continue
                
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
            except:
                size = 0
                
            if size == 0:
                icon = "⚪"
            elif file.endswith('.py'):
                icon = "🐍"
            elif file.endswith(('.yaml', '.yml', '.toml', '.json')):
                icon = "⚙️"
            else:
                icon = "📄"
                
            print(f"{subindent}{icon} {file} ({size:,} bytes)")
    
    print()
    print("=" * 70)
    print("PROSSIMI PASSI SUGGERITI:")
    print("=" * 70)
    print("""
1. File mancanti o vuoti → probabilmente devi decomprimere qcorona_step3.zip

2. Per eseguire i test:
   pip install -r requirements-dev.txt    # se esiste
   oppure:
   pip install pytest numpy scipy pyyaml
   python -m pytest tests/ -v

3. Prova veloce:
   python -c "from qcorona.geometry.grid import build_grid; print(build_grid())"
""")

if __name__ == "__main__":
    main()