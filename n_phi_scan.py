"""
n_phi_scan.py — Fix 5: verifica che la risoluzione azimutale non rompa
la degenerazione di parita' (previsione del Teorema, Parte B).

Se il Teorema e' corretto, H_final NON deve migliorare sistematicamente
con n_phi crescente (a differenza di r_in, che aveva un minimo vero).
"""

import sys, time
import numpy as np

sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import (
    QCORONAHamiltonian, HamiltonianConfig,
    ObservationalConstraints, CoronalState,
)
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.hamiltonian.terms import vertical_orientations
from qcorona.physics.illumination import compute_solid_angles
from schwarzschild_lags import compute_lag_matrix_schwarzschild

# ── parametri Mrk 335 (identici a run_mrk335.py) ──────────────────────────────
M_BH       = 1.3e7
R_G_C      = 63.0
LAG_OBS_S  = 91.9
LAG_ERR_S  = 43.4
LAG_OBS_RG = LAG_OBS_S / R_G_C
LAG_ERR_RG = LAG_ERR_S / R_G_C
R_OBS      = 0.3
L_X        = 1.0e43
INCL       = 45.0

N_STARTS  = 5
MAX_ITER  = 2000
N_BINS_TF = 80

N_PHI_LIST = [4, 6, 8, 12, 16, 24, 32]

opt_cfg = OptimizationConfig(
    max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
    use_adaptive_lr=True, use_continuation=True,
    continuation_steps=200, convergence_tol=1e-7, verbose=False,
)

print("=" * 60)
print("Fix 5 — scan n_phi (Mrk 335)")
print(f"  n_phi list = {N_PHI_LIST}")
print("=" * 60)
print(f"{'n_phi':>6} {'N_cells':>8} {'H_final':>10} {'lag_pred':>10} {'chi2':>12}")
print("-" * 52)

results = []
for n_phi in N_PHI_LIST:
    GRID = build_grid(n_r=10, n_z=10, n_phi=n_phi,
                      r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
    DISK = build_disk_grid(n_r=30, n_phi=18)
    SA   = compute_solid_angles(GRID, DISK)
    LM   = compute_lag_matrix_schwarzschild(GRID, DISK, observer_inclination=INCL)

    T_MIN  = max(0.0, float(LM.min()))
    T_MAX  = 15.0
    t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
    t_c    = 0.5 * (t_bins[:-1] + t_bins[1:])
    psi_obs = np.exp(-0.5 * ((t_c - LAG_OBS_RG) / LAG_ERR_RG)**2)
    psi_obs = psi_obs / psi_obs.sum()

    constraints = ObservationalConstraints(
        delta_t_obs=LAG_OBS_RG, R_obs=R_OBS, L_X=L_X, M_bh=M_BH,
        eddington_ratio=0.25, observer_inclination=INCL,
        source_name=f"Mrk335_nphi{n_phi}",
        psi_obs=psi_obs, t_bins_obs=t_bins,
    )
    h_config = HamiltonianConfig(
        J=0.3, alpha=50.0, beta=10.0, gamma=1.0, l_crit=1000.0,
        delta=1.0, f_max=1.0, use_pair_barrier=False, use_energy_barrier=False,
    )
    H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config, lag_matrix=LM)

    best_H, best_lag, best_chi2 = np.inf, None, None
    for seed in range(N_STARTS):
        rng = np.random.default_rng(seed)
        w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
        s0  = CoronalState(emissivity=w0, orientations=vertical_orientations(GRID.n_cells))
        res = QCORONAOptimizer(H_obj, opt_cfg).optimize(initial_state=s0)
        if res.final_H.H_total < best_H:
            best_H   = res.final_H.H_total
            best_lag = res.final_H.diagnostics['delta_t_pred']

    print(f"{n_phi:>6} {GRID.n_cells:>8} {best_H:>10.4f} {best_lag:>10.2f}")
    results.append((n_phi, GRID.n_cells, best_H, best_lag))

print("\nSe H_final resta ~piatto (nessun trend monotono chiaro) al crescere di n_phi,")
print("il Teorema di parita' e' confermato: la risoluzione azimutale non aiuta.")
print("Se invece H_final scende sistematicamente, va rivista l'interpretazione.")