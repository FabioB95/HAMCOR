"""
mirror_symmetry_check_v4.py — versione che esercita davvero Schwarzschild
(aggiunge psi_obs per forzare il path compute_H_geo_tf, che rispetta lag_matrix)
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import QCORONAHamiltonian, HamiltonianConfig, ObservationalConstraints, CoronalState
from qcorona.hamiltonian.terms import random_orientations
from qcorona.physics.lags import compute_lag_matrix
from schwarzschild_lags import compute_lag_matrix_schwarzschild

def mirror_index_map(grid):
    idx_map = np.zeros(grid.n_cells, dtype=int)
    for i in range(grid.n_cells):
        ir, iz, iphi = grid.get_grid_indices(i)
        idx_map[i] = grid.get_cell_index(ir, iz, grid.n_phi - 1 - iphi)
    return idx_map

def rho(m):
    m_r = m.copy()
    m_r[:, 1] *= -1.0
    return m_r

N_SEEDS = 5
N_BINS_TF = 40

print(f"{'n_phi':>6} {'incl':>6} {'spacetime':>12} {'H_geo esempio':>14} {'max|diff| su 5 seed':>22}")
print("-" * 66)

for n_phi in [6, 8, 12]:
    for incl in [20.0, 45.0, 70.0]:
        GRID = build_grid(n_r=10, n_z=10, n_phi=n_phi, r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
        DISK = build_disk_grid(n_r=30, n_phi=18)
        MIRROR = mirror_index_map(GRID)

        for spacetime, LM_fn in [('flat', compute_lag_matrix),
                                  ('schwarzschild', compute_lag_matrix_schwarzschild)]:
            LM = LM_fn(GRID, DISK, observer_inclination=incl)

            # psi_obs fittizio -- FORZA il path compute_H_geo_tf (rispetta lag_matrix)
            t_bins = np.linspace(0.0, float(LM.max()) * 1.05, N_BINS_TF + 1)
            t_c = 0.5 * (t_bins[:-1] + t_bins[1:])
            psi_obs = np.exp(-0.5 * ((t_c - t_c.mean()) / (t_c.max()*0.2))**2)
            psi_obs /= psi_obs.sum()

            constraints = ObservationalConstraints(
                delta_t_obs=1.5, R_obs=0.3, L_X=1e43, M_bh=1e7,
                eddington_ratio=0.1, observer_inclination=incl, source_name="mirror_test",
                psi_obs=psi_obs, t_bins_obs=t_bins,
            )
            h_config = HamiltonianConfig(J=0.3, alpha=50.0, beta=10.0, gamma=1.0,
                                          l_crit=1000.0, delta=1.0, f_max=1.0,
                                          use_pair_barrier=False, use_energy_barrier=False)
            H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config, lag_matrix=LM)

            max_diff = 0.0
            h_geo_example = None
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                w0 = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
                m0 = random_orientations(GRID.n_cells, seed=seed + 1000)

                w_mirror = w0[MIRROR]
                m_mirror = rho(m0[MIRROR])

                res      = H_obj.evaluate(CoronalState(emissivity=w0, orientations=m0))
                res_mirr = H_obj.evaluate(CoronalState(emissivity=w_mirror, orientations=m_mirror))

                if seed == 0:
                    h_geo_example = res.H_geo

                for name in ['H_mag', 'H_geo', 'H_rad', 'H_pair', 'H_ener', 'H_total']:
                    diff = abs(getattr(res, name) - getattr(res_mirr, name))
                    max_diff = max(max_diff, diff)

            print(f"{n_phi:>6} {incl:>6.0f} {spacetime:>12} {h_geo_example:>14.4f} {max_diff:>22.2e}")