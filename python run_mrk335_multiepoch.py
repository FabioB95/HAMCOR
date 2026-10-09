"""
run_mrk335_multiepoch.py — fit HAMCOR per le 5 epoche Mrk 335, Shapiro definitivo.

Imposta DRY_RUN=True per un test veloce (1 seed, 50 iter, ~10s/epoca) prima
di lanciare il run completo (DRY_RUN=False, 5 seed, 2000 iter).
"""
import sys, os, time
import numpy as np

sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import (
    QCORONAHamiltonian, HamiltonianConfig, ObservationalConstraints, CoronalState,
)
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.hamiltonian.terms import vertical_orientations
from qcorona.physics.illumination import compute_solid_angles
from schwarzschild_lags import compute_lag_matrix_schwarzschild

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DRY_RUN = True   # <-- metti False per il run vero, dopo la verifica

M_BH   = 1.3e7
R_G_C  = 63.0
R_OBS  = 0.3
L_X    = 1.0e43
INCL   = 45.0

# (epoch_label, state, lag_obs_rg, lag_err_rg) — dal Table esistente nel paper
EPOCHS = [
    ("2006", "High",       1.46, 0.69),
    ("2009", "Low",        0.00, 0.10),
    ("2015", "Interm.",    0.31, 0.14),
    ("2018", "Interm.",    0.12, 0.10),
    ("2019", "Recovering", 0.62, 0.12),
]

N_STARTS  = 1 if DRY_RUN else 5
MAX_ITER  = 50 if DRY_RUN else 2000
N_BINS_TF = 80

print("=" * 60)
print(f"Mrk 335 multi-epoch — {'DRY RUN' if DRY_RUN else 'RUN COMPLETO'}")
print("=" * 60)

GRID = build_grid(n_r=10, n_z=10, n_phi=8, r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
LM   = compute_lag_matrix_schwarzschild(GRID, DISK, observer_inclination=INCL)
print(f"Lag matrix range: {LM.min():.2f} -- {LM.max():.2f} r_g/c\n")

opt_cfg = OptimizationConfig(
    max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
    use_adaptive_lr=True, use_continuation=True,
    continuation_steps=min(200, MAX_ITER // 2), convergence_tol=1e-7, verbose=False,
)

results_summary = []

for label, state, lag_obs_rg, lag_err_rg in EPOCHS:
    print(f"--- Epoca {label} ({state}): lag_obs = {lag_obs_rg:.2f} +/- {lag_err_rg:.2f} r_g/c ---")

    T_MIN, T_MAX = max(0.0, float(LM.min())), 15.0
    t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
    t_c = 0.5 * (t_bins[:-1] + t_bins[1:])
    sigma_eff = max(lag_err_rg, 0.05)  # evita sigma=0 per casi tipo 2009
    psi_obs = np.exp(-0.5 * ((t_c - lag_obs_rg) / sigma_eff) ** 2)
    psi_obs /= psi_obs.sum()

    constraints = ObservationalConstraints(
        delta_t_obs=lag_obs_rg, R_obs=R_OBS, L_X=L_X, M_bh=M_BH,
        eddington_ratio=0.25, observer_inclination=INCL,
        source_name=f"Mrk335_{label}", psi_obs=psi_obs, t_bins_obs=t_bins,
    )
    h_config = HamiltonianConfig(
        J=0.3, alpha=50.0, beta=10.0, gamma=1.0, l_crit=1000.0,
        delta=1.0, f_max=1.0, use_pair_barrier=False, use_energy_barrier=False,
    )
    H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config, lag_matrix=LM)

    best_H, best_w, best_diag = np.inf, None, None
    for seed in range(N_STARTS):
        rng = np.random.default_rng(seed)
        w0 = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
        s0 = CoronalState(emissivity=w0, orientations=vertical_orientations(GRID.n_cells))
        t0 = time.time()
        res = QCORONAOptimizer(H_obj, opt_cfg).optimize(initial_state=s0)
        elapsed = time.time() - t0
        if res.final_H.H_total < best_H:
            best_H   = res.final_H.H_total
            best_w   = res.final_state.emissivity
            best_diag = res.final_H.diagnostics
        print(f"  seed {seed}: H={res.final_H.H_total:.4f}  "
              f"lag_pred={res.final_H.diagnostics['delta_t_pred']:.2f}  ({elapsed:.0f}s)")

    R_c = float(np.sum(best_w * GRID.R) / np.sum(best_w))
    z_c = float(np.sum(best_w * GRID.z) / np.sum(best_w))
    print(f"  BEST: H={best_H:.4f}  lag_pred={best_diag['delta_t_pred']:.2f}  "
          f"centroide=({R_c:.2f},{z_c:.2f})\n")

    results_summary.append(dict(
        label=label, state=state, lag_obs=lag_obs_rg, lag_err=lag_err_rg,
        lag_pred=best_diag['delta_t_pred'], H_final=best_H, R_c=R_c, z_c=z_c, w=best_w,
    ))

print("=" * 60)
print("RIEPILOGO")
print("=" * 60)
print(f"{'Epoca':>6} {'Stato':>12} {'lag_obs':>10} {'lag_pred':>10} {'H_final':>9} {'R_c':>6} {'z_c':>6}")
for r in results_summary:
    print(f"{r['label']:>6} {r['state']:>12} "
          f"{r['lag_obs']:>6.2f}±{r['lag_err']:.2f} {r['lag_pred']:>10.2f} "
          f"{r['H_final']:>9.4f} {r['R_c']:>6.2f} {r['z_c']:>6.2f}")

if not DRY_RUN:
    os.makedirs('results', exist_ok=True)
    np.savez('results/mrk335_multiepoch_results.npz',
              **{f"{r['label']}_w": r['w'] for r in results_summary})
    print("\nSalvato: results/mrk335_multiepoch_results.npz")

print(f"\n{'[DRY RUN completato — nessun crash. Pronto per DRY_RUN=False]' if DRY_RUN else '[RUN COMPLETO terminato]'}")