"""
debug_shapiro_pair.py
───────────────────────
Debug chirurgico: verifica le mappe di specchio direttamente contro
phi (indipendente dal layout interno di grid/disk), poi scompone
la coppia i=562,k=102 in tutti i suoi pezzi (b_ik, r_i, r_k, le tre
leg) per vedere ESATTAMENTE dove nasce la differenza.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from schwarzschild_lags import (
    _cartesian, _schwarzschild_r, _impact_parameter, _shapiro_delay,
    _observer_hat, B_MIN_RG, SHAPIRO_CAP,
)

def mirror_index_map_corona(grid):
    idx_map = np.zeros(grid.n_cells, dtype=int)
    for i in range(grid.n_cells):
        ir, iz, iphi = grid.get_grid_indices(i)
        idx_map[i] = grid.get_cell_index(ir, iz, grid.n_phi - 1 - iphi)
    return idx_map

def mirror_index_map_disk(disk):
    idx_map = np.zeros(disk.n_elements, dtype=int)
    for k in range(disk.n_elements):
        ir = k // disk.n_phi
        iphi = k % disk.n_phi
        idx_map[k] = ir * disk.n_phi + (disk.n_phi - 1 - iphi)
    return idx_map

GRID = build_grid(n_r=10, n_z=10, n_phi=8, r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
INCL = 45.0
obs_hat = _observer_hat(INCL)

MIRROR_C = mirror_index_map_corona(GRID)
MIRROR_D = mirror_index_map_disk(DISK)

# ── STEP 1: verifica le mappe di specchio direttamente contro phi ────────────
print("=" * 70)
print("STEP 1 — verifica mappe di specchio (indipendente dal layout interno)")
print("=" * 70)
bad_corona = 0
for i in range(GRID.n_cells):
    expected = (-GRID.phi[i]) % (2*np.pi)
    actual   = GRID.phi[MIRROR_C[i]] % (2*np.pi)
    if min(abs(expected-actual), 2*np.pi-abs(expected-actual)) > 1e-6:
        bad_corona += 1
print(f"  Corona: {bad_corona}/{GRID.n_cells} mappature errate")

bad_disk = 0
for k in range(DISK.n_elements):
    expected = (-DISK.phi[k]) % (2*np.pi)
    actual   = DISK.phi[MIRROR_D[k]] % (2*np.pi)
    if min(abs(expected-actual), 2*np.pi-abs(expected-actual)) > 1e-6:
        bad_disk += 1
print(f"  Disco:  {bad_disk}/{DISK.n_elements} mappature errate")

# ── STEP 2: scomponi la coppia incriminata pezzo per pezzo ────────────────────
i, k = 562, 102
si, sk = MIRROR_C[i], MIRROR_D[k]

print("\n" + "=" * 70)
print(f"STEP 2 — scomposizione coppia i={i}, k={k}  (sigma(i)={si}, sigma(k)={sk})")
print("=" * 70)

def decompose(ia, ka, label):
    Ri, zi, phi_i = GRID.R[ia], GRID.z[ia], GRID.phi[ia]
    Rk, phi_k      = DISK.R[ka], DISK.phi[ka]
    x_i = _cartesian(Ri, zi, phi_i)
    x_k = _cartesian(Rk, 0.0, phi_k)
    r_i = _schwarzschild_r(Ri, zi)
    r_k = Rk

    b_ik     = _impact_parameter(x_i, x_k)
    dt_ik_GR = _shapiro_delay(r_i, r_k, b_ik)

    d_ik_flat    = float(np.linalg.norm(x_i - x_k))
    d_k_obs_flat = float(np.dot(x_k, obs_hat))
    d_i_obs_flat = float(np.dot(x_i, obs_hat))

    cos_k = float(np.dot(x_k/(r_k+1e-15), obs_hat)); sin_k = np.sqrt(max(0.,1-cos_k**2))
    b_k_obs = max(r_k*sin_k, B_MIN_RG)
    dt_k_obs_GR = float(np.clip(2.0*np.log(2.0*r_k/b_k_obs),0.0,SHAPIRO_CAP)) if b_k_obs<r_k else 0.0

    cos_i = float(np.dot(x_i/(r_i+1e-15), obs_hat)); sin_i = np.sqrt(max(0.,1-cos_i**2))
    b_i_obs = max(r_i*sin_i, B_MIN_RG)
    dt_i_obs_GR = float(np.clip(2.0*np.log(2.0*r_i/b_i_obs),0.0,SHAPIRO_CAP)) if b_i_obs<r_i else 0.0

    lag_flat = d_ik_flat + d_k_obs_flat - d_i_obs_flat
    lag_GR   = lag_flat + dt_ik_GR + dt_k_obs_GR - dt_i_obs_GR

    print(f"\n  --- {label}: corona idx={ia} (R={Ri:.4f},z={zi:.4f},phi={phi_i:.4f}), "
          f"disco idx={ka} (R={Rk:.4f},phi={phi_k:.4f}) ---")
    print(f"    r_i={r_i:.6f}  r_k={r_k:.6f}  b_ik={b_ik:.6f}")
    print(f"    dt_ik_GR={dt_ik_GR:.4f}  dt_k_obs_GR={dt_k_obs_GR:.4f}  dt_i_obs_GR={dt_i_obs_GR:.4f}")
    print(f"    lag_flat={lag_flat:.4f}  lag_GR(finale, pre-clip)={lag_GR:.4f}")

decompose(i,  k,  "originale:      (i, k)")
decompose(si, k,  "mirror 1: (sigma(i), k)")
decompose(i,  sk, "mirror 2: (i, sigma(k))")
decompose(si, sk, "mirror completo: (sigma(i), sigma(k))")