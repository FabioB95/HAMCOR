#!/usr/bin/env python3
"""validate_kerr_lags.py — a=0 vs Schwarzschild (griglia piccola)."""

import numpy as np
import sys

sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from schwarzschild_lags import compute_lag_matrix_schwarzschild
from kerr_lags import compute_lag_matrix_kerr

GRID = build_grid(
    n_r=3, n_z=2, n_phi=4,
    r_in=5.0, r_out=10.0,
    z_min=1.0, z_max=3.0,
)
DISK = build_disk_grid(n_r=4, n_phi=4)
INCL = 30.0

print(f"Grid cells: {GRID.n_cells}  |  Disk elements: {DISK.n_elements}")
print(f"Pairs: {GRID.n_cells * DISK.n_elements}")

print("\nCalcolo Schwarzschild (riferimento)...")
LM_schw = compute_lag_matrix_schwarzschild(GRID, DISK, observer_inclination=INCL)
print(f"  LM_schw range: [{LM_schw.min():.3f}, {LM_schw.max():.3f}]")
print(f"  LM_schw mean:  {LM_schw.mean():.3f}")

print("\nCalcolo Kerr con a=0 (segni sr/sth + score<0.05)...")
LM_kerr_a0, obs_table = compute_lag_matrix_kerr(
    GRID, DISK, a=0.0, observer_inclination=INCL, verbose=True,
)

n_ok = int(np.sum(~np.isnan(LM_kerr_a0)))
print(f"\nCoppie con lag definito: {n_ok}/{LM_kerr_a0.size}")

diff = np.abs(LM_kerr_a0 - LM_schw)
valid = ~np.isnan(LM_kerr_a0)

print(f"obs_table hits: {len(obs_table)}")
print(f"Coppie valide:  {valid.sum()}/{valid.size}")

if valid.sum() == 0:
    print("Nessuna coppia valida — stop.")
    sys.exit(1)

print(f"\nLM_kerr mean (valide): {np.nanmean(LM_kerr_a0[valid]):.4f}")
print(f"LM_schw mean (valide): {np.nanmean(LM_schw[valid]):.4f}")
print(f"Diff media:   {np.nanmean(diff[valid]):.4f} r_g/c")
print(f"Diff max:     {np.nanmax(diff[valid]):.4f} r_g/c")
print(f"Diff mediana: {np.nanmedian(diff[valid]):.4f} r_g/c")

mean_diff = float(np.nanmean(diff[valid]))
if mean_diff > 50.0:
    print("\nATTENZIONE: diff ancora molto grande. NON a!=0.")
    sys.exit(1)
elif mean_diff > 1.0:
    print(f"\nDiff media = {mean_diff:.2f}: meglio, ma non ancora OK per a!=0.")
    sys.exit(1)

print("\nOK: Kerr a=0 consistente con Schwarzschild.")