"""
mcg6_mass_scan.py — sensibilita' del fit alla scelta di massa (r_in=3.0 fisso)
"""
import sys, os, time
import numpy as np
sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import QCORONAHamiltonian, HamiltonianConfig, ObservationalConstraints, CoronalState
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.hamiltonian.terms import vertical_orientations
from qcorona.physics.illumination import compute_solid_angles
from qcorona.physics.lags import compute_transfer_function_full
from schwarzschild_lags import compute_lag_matrix_schwarzschild

T_SUN = 4.9255e-6
LAG_REV_S, ERR_REV_S = 10.5, 10.2   # dalla parte 1 di run_mcg6.py, gia' calcolato

MASS_TABLE = [
    (1.6, "Bentz+2016 RM"),
    (2.0, "Zhou+2010 (Kara+14)"),
    (2.9, "McHardy+05 PSD"),
    (3.0, "vecchio valore usato finora"),
]

GRID = build_grid(n_r=10, n_z=10, n_phi=8, r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
INCL = 30.0
N_STARTS, MAX_ITER, N_BINS_TF = 3, 2000, 80

opt_cfg = OptimizationConfig(max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
                              use_adaptive_lr=True, use_continuation=True,
                              continuation_steps=200, convergence_tol=1e-7, verbose=False)

print(f"{'M[1e6 Msun]':>12} {'label':>22} {'r_g/c':>7} {'lag_obs_rg':>11} {'H_final':>9} {'lag_pred':>9}")
print("-"*76)

for M_e6, label in MASS_TABLE:
    R_G_C = T_SUN * M_e6 * 1e6
    LAG_OBS_RG = LAG_REV_S / R_G_C
    LAG_ERR_RG = max(ERR_REV_S, 10.0) / R_G_C

    LM = compute_lag_matrix_schwarzschild(GRID, DISK, observer_inclination=INCL)
    t_bins = np.linspace(0.0, 15.0, N_BINS_TF + 1)
    t_c    = 0.5 * (t_bins[:-1] + t_bins[1:])
    psi_obs = np.exp(-0.5 * ((t_c - LAG_OBS_RG) / LAG_ERR_RG)**2)
    psi_obs = psi_obs / psi_obs.sum()

    constraints = ObservationalConstraints(
        delta_t_obs=LAG_OBS_RG, R_obs=0.5, L_X=2.0e43, M_bh=M_e6*1e6,
        eddington_ratio=0.1, observer_inclination=INCL,
        source_name=f"MCG6_mass{M_e6}", psi_obs=psi_obs, t_bins_obs=t_bins,
    )
    h_config = HamiltonianConfig(J=0.3, alpha=50.0, beta=10.0, gamma=1.0,
                                  l_crit=1000.0, delta=1.0, f_max=1.0,
                                  use_pair_barrier=False, use_energy_barrier=False)
    H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config, lag_matrix=LM)

    best_H, best_lag = np.inf, None
    for seed in range(N_STARTS):
        rng = np.random.default_rng(seed)
        w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
        s0  = CoronalState(emissivity=w0, orientations=vertical_orientations(GRID.n_cells))
        res = QCORONAOptimizer(H_obj, opt_cfg).optimize(initial_state=s0)
        if res.final_H.H_total < best_H:
            best_H   = res.final_H.H_total
            best_lag = res.final_H.diagnostics['delta_t_pred']

    print(f"{M_e6:>12.1f} {label:>22} {R_G_C:>7.2f} {LAG_OBS_RG:>11.2f} {best_H:>9.4f} {best_lag:>9.2f}")