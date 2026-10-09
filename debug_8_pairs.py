"""debug_8_pairs.py"""
import sys; sys.path.insert(0, '.')
import numpy as np
from qcorona.geometry.grid import build_grid, build_disk_grid
from schwarzschild_lags import compute_lag_matrix_schwarzschild
from kerr_lags import corona_to_disk_time

GRID = build_grid(n_r=3, n_z=2, n_phi=4, r_in=5.0, r_out=10.0, z_min=1.0, z_max=3.0)
DISK = build_disk_grid(n_r=4, n_phi=4)
INCL = 30.0
LM_schw = compute_lag_matrix_schwarzschild(GRID, DISK, observer_inclination=INCL)

for i in range(GRID.n_cells):
    Ri, zi, phii = float(GRID.R[i]), float(GRID.z[i]), float(GRID.phi[i])
    r_c = np.sqrt(Ri**2+zi**2)
    theta_c = np.arctan2(Ri, zi) if zi > 0 else np.pi/2
    for k in range(DISK.n_elements):
        Rk, phik = float(DISK.R[k]), float(DISK.phi[k])
        result = corona_to_disk_time(r_c, theta_c, phii, Rk, phik, a=0.0, return_score=True)
        if result is None:
            continue
        t_ck, score = result
        if score >= 0.05:
            continue
        print(f"i={i} k={k} r_c={r_c:.3f} theta_c={theta_c:.3f} phi_c={phii:.3f} "
              f"r_d={Rk:.3f} phi_d={phik:.3f} | t_ck={t_ck:.4f} score={score:.5f} "
              f"| LM_schw={LM_schw[i,k]:.4f}")