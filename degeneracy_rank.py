"""
degeneracy_rank.py
──────────────────
Dimensione della degenerazione fondamentale di H_geo in HAMCOR.

Fisica/matematica:
    psi(t_b) = (M w)_b / (Omega^T w),   M_{b,i} = sum_{k in bin b} Omega_ik

Mappa proiettiva (lineare-frazionaria) di w. Due punti w, w' sul simplesso
danno la stessa psi(t) sse M w e M w' sono proporzionali (stesso raggio in
R^n_bins), non uguali. La normalizzazione Sigma w_i = 1 fissa la scala, quindi
l'insieme dei w che danno una psi_target FISSATA e' un sottospazio affine di
dimensione N - rank(M) (quando M ha rango pieno n_bins), intersecato col
simplesso (w >= 0).

Usage: python degeneracy_rank.py
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.physics.illumination import compute_solid_angles
from qcorona.physics.lags import compute_lag_matrix

# ── stessa griglia di produzione usata nei fit reali (Mrk 335 etc.) ───────────
GRID = build_grid(n_r=10, n_z=10, n_phi=8, r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
N    = GRID.n_cells
SA   = compute_solid_angles(GRID, DISK)
LM   = compute_lag_matrix(GRID, DISK, observer_inclination=45.0)

T_MIN, T_MAX = 0.0, 15.0

def build_M(n_bins, lag_matrix, solid_angles, t_min=T_MIN, t_max=T_MAX):
    """M_{b,i} = sum_{k in bin b} Omega_ik, dato un binning a n_bins."""
    t_bins  = np.linspace(t_min, t_max, n_bins + 1)
    bin_idx = np.digitize(lag_matrix, t_bins) - 1
    valid   = (bin_idx >= 0) & (bin_idx < n_bins)   # scarta coppie fuori range

    N_cells = lag_matrix.shape[0]
    M = np.zeros((n_bins, N_cells))
    for b in range(n_bins):
        mask = valid & (bin_idx == b)
        M[b, :] = np.sum(solid_angles * mask, axis=1)
    return M

def numeric_rank(M, tol_factor=1.0):
    sv  = np.linalg.svd(M, compute_uv=False)
    tol = sv.max() * max(M.shape) * np.finfo(sv.dtype).eps * tol_factor
    return int(np.sum(sv > tol)), sv

# ── caso di produzione: n_bins = 80 (quello usato in tutti i run_*.py) ────────
N_BINS = 80
M = build_M(N_BINS, LM, SA)
rank, sv = numeric_rank(M)

print("=" * 60)
print("DEGENERAZIONE FONDAMENTALE — griglia di produzione")
print("=" * 60)
print(f"  N (celle coronali)      = {N}")
print(f"  n_bins                  = {N_BINS}")
print(f"  rank(M)                 = {rank}")
print(f"  dim(degenerazione)      = N - rank(M) = {N - rank}")
print(f"  frazione non vincolata  = {(N - rank)/N*100:.1f}%")
print(f"\n  Valori singolari (primi 5): {sv[:5]}")
print(f"  Valori singolari (ultimi 5): {sv[-5:]}")

# ── scan: il rango satura con n_bins, o cresce linearmente? ───────────────────
# Se satura ben prima del limite strumentale, la degenerazione non e' solo
# "n_bins finito" ma strutturale alla geometria (risultato piu' forte).
print("\n" + "=" * 60)
print("SCAN: rank(M) al variare di n_bins")
print("=" * 60)
print(f"{'n_bins':>8} {'rank(M)':>10} {'N-rank':>10} {'rank/n_bins':>12}")
for nb in [10, 20, 40, 80, 160, 320, 640, 1280]:
    Mb = build_M(nb, LM, SA)
    r, _ = numeric_rank(Mb)
    print(f"{nb:>8} {r:>10} {N - r:>10} {r/nb:>12.3f}")