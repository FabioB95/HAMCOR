"""
diagnose_score_vs_diff_v2.py — confronto corretto: SOLO il leg corona->disco,
non il lag completo a 3 leg. Flat: distanza euclidea. Schwarzschild: distanza
euclidea + correzione Shapiro di quel singolo leg (non l'assemblato finale).
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from kerr_lags import corona_to_disk_time
from schwarzschild_lags import _cartesian, _schwarzschild_r, _impact_parameter, _shapiro_delay

GRID = build_grid(n_r=3, n_z=2, n_phi=4, r_in=5.0, r_out=10.0, z_min=1.0, z_max=3.0)
DISK = build_disk_grid(n_r=4, n_phi=4)

rows = []
for i in range(GRID.n_cells):
    Ri, zi, phii = float(GRID.R[i]), float(GRID.z[i]), float(GRID.phi[i])
    r_c = np.sqrt(Ri**2+zi**2)
    theta_c = np.arctan2(Ri, zi) if zi > 0 else np.pi/2
    x_i = _cartesian(Ri, zi, phii)

    for k in range(DISK.n_elements):
        Rk, phik = float(DISK.R[k]), float(DISK.phi[k])
        x_k = _cartesian(Rk, 0.0, phik)

        # ── riferimento: SOLO leg i->k, flat + Shapiro (non il lag assemblato) ─
        d_ik_flat = float(np.linalg.norm(x_i - x_k))
        b_ik = _impact_parameter(x_i, x_k)
        dt_shapiro_leg = _shapiro_delay(r_c, Rk, b_ik)   # Schwarzschild ma a=0 => stesso
        t_ck_ref_flat  = d_ik_flat
        t_ck_ref_schw  = d_ik_flat + dt_shapiro_leg

        result = corona_to_disk_time(r_c, theta_c, phii, Rk, phik, a=0.0, return_score=True)
        if result is None:
            continue
        t_ck_kerr, score = result

        rows.append((score, t_ck_kerr, t_ck_ref_flat, t_ck_ref_schw))

rows = np.array(rows)
scores, t_kerr, ref_flat, ref_schw = rows[:,0], rows[:,1], rows[:,2], rows[:,3]

order = np.argsort(scores)
print(f"{'score':>8} {'t_kerr':>9} {'ref_flat':>9} {'ref_schw':>9} {'diff_vs_schw':>13}")
for idx in order[:15]:
    print(f"{scores[idx]:8.4f} {t_kerr[idx]:9.3f} {ref_flat[idx]:9.3f} "
          f"{ref_schw[idx]:9.3f} {t_kerr[idx]-ref_schw[idx]:13.3f}")
print("...")
for idx in order[-15:]:
    print(f"{scores[idx]:8.4f} {t_kerr[idx]:9.3f} {ref_flat[idx]:9.3f} "
          f"{ref_schw[idx]:9.3f} {t_kerr[idx]-ref_schw[idx]:13.3f}")

diff_leg = t_kerr - ref_schw
print(f"\n--- Correlazione score vs diff sul singolo leg ---")
for thresh in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
    mask = scores < thresh
    if mask.sum() > 0:
        print(f"score<{thresh}: n={mask.sum():>4}  "
              f"diff media={np.mean(np.abs(diff_leg[mask])):.3f}  "
              f"diff mediana={np.median(np.abs(diff_leg[mask])):.3f}")