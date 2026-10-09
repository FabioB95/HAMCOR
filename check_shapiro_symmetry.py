"""
check_shapiro_symmetry.py
──────────────────────────
Isola se la rottura di parita' e' dentro schwarzschild_lags.py stesso,
testando direttamente Delta_t(sigma(i), k) == Delta_t(i, sigma(k))
sulla matrice grezza -- bypassa Hamiltoniana e binning.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from schwarzschild_lags import compute_lag_matrix_schwarzschild
from qcorona.physics.lags import compute_lag_matrix

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

MIRROR_C = mirror_index_map_corona(GRID)
MIRROR_D = mirror_index_map_disk(DISK)

for label, fn in [('flat', compute_lag_matrix), ('schwarzschild', compute_lag_matrix_schwarzschild)]:
    LM = fn(GRID, DISK, observer_inclination=INCL)

    LM_sigma_i = LM[MIRROR_C, :]      # Delta_t(sigma(i), k)
    LM_sigma_k = LM[:, MIRROR_D]      # Delta_t(i, sigma(k))

    diff = np.abs(LM_sigma_i - LM_sigma_k)
    n_bad = (diff > 1e-6).sum()
    print(f"{label:>14}: max diff = {diff.max():.3e}   mean = {diff.mean():.3e}   "
          f"coppie con diff>1e-6: {n_bad}/{diff.size}")

    if n_bad > 0:
        # stampa un esempio concreto per capire dove si rompe
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        i_bad, k_bad = idx
        print(f"    esempio peggiore: i={i_bad}, k={k_bad}")
        print(f"    Delta_t(sigma(i),k)  = {LM_sigma_i[idx]:.6f}")
        print(f"    Delta_t(i,sigma(k))  = {LM_sigma_k[idx]:.6f}")
        Ri, zi, phii = GRID.R[i_bad], GRID.z[i_bad], GRID.phi[i_bad]
        Rk, phik     = DISK.R[k_bad], DISK.phi[k_bad]
        print(f"    cella corona i: R={Ri:.2f}, z={zi:.2f}, phi={phii:.3f}")
        print(f"    elemento disco k: R={Rk:.2f}, phi={phik:.3f}")