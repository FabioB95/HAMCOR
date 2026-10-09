#!/usr/bin/env python3
"""
run_mcg6_rin_scan.py
Scan r_in per MCG-6-30-15.
Massa fiducial: 1.6e6 M_sun (Bentz+2016) → r_g/c = 7.88 s
Salva summary CSV + una figura per ogni r_in.
"""

import sys, os, time, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import (
    QCORONAHamiltonian, HamiltonianConfig,
    ObservationalConstraints, CoronalState,
)
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.hamiltonian.terms import vertical_orientations
from qcorona.physics.illumination import compute_solid_angles
from qcorona.physics.lags import compute_transfer_function_full
from schwarzschild_lags import compute_lag_matrix_schwarzschild

DATA_DIR = 'data/sources/mcg6'
M_BH  = 1.6e6          # Bentz+2016-scale (NON 3e6)
R_G_C = 7.88           # s (NON 15.0)

if os.path.exists(f'{DATA_DIR}/lag_rev.npy'):
    lag_rev, err_rev, lag_rg, err_rg = np.load(f'{DATA_DIR}/lag_rev.npy')
    LAG_OBS_S = abs(float(lag_rev))
    LAG_ERR_S = max(float(err_rev), 10.0)
else:
    LAG_OBS_S, LAG_ERR_S = 10.5, 10.2

LAG_OBS_RG = LAG_OBS_S / R_G_C   # ~1.33
LAG_ERR_RG = LAG_ERR_S / R_G_C   # ~1.29

R_OBS, L_X, INCL = 0.5, 2.0e43, 30.0
N_STARTS, MAX_ITER, N_BINS_TF = 3, 2000, 80

# Per paper/baseline: 1.5–3.5 basta. Lista ampia solo se documentare il peggioramento.
R_IN_LIST = [1.5, 2.0, 2.5, 3.0, 3.5]

os.makedirs('results/mcg6_rin_scan', exist_ok=True)
rows = []

print("=" * 60)
print("MCG-6 r_in scan (Bentz+2016 mass)")
print(f"  M_bh    = {M_BH:.1e} M_sun")
print(f"  r_g/c   = {R_G_C:.2f} s")
print(f"  lag obs = {LAG_OBS_RG:.2f} +/- {LAG_ERR_RG:.2f} r_g/c")
print(f"  r_in list = {R_IN_LIST}")
print("=" * 60)

for r_in in R_IN_LIST:
    print(f"\n--- r_in = {r_in} ---")
    GRID = build_grid(n_r=10, n_z=10, n_phi=8,
                      r_in=r_in, r_out=15.0, z_min=0.1, z_max=5.0)
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
        eddington_ratio=0.1, observer_inclination=INCL,
        source_name=f"MCG6_rin{r_in}", psi_obs=psi_obs, t_bins_obs=t_bins,
    )
    h_config = HamiltonianConfig(
        J=0.3, alpha=50.0, beta=10.0, gamma=1.0, l_crit=1000.0,
        delta=1.0, f_max=1.0, use_pair_barrier=False, use_energy_barrier=False,
    )
    H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config, lag_matrix=LM)
    opt_cfg = OptimizationConfig(
        max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
        use_adaptive_lr=True, use_continuation=True,
        continuation_steps=200, convergence_tol=1e-7, verbose=False,
    )

    all_results = []
    for seed in range(N_STARTS):
        rng = np.random.default_rng(seed)
        w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
        s0  = CoronalState(emissivity=w0, orientations=vertical_orientations(GRID.n_cells))
        t0  = time.time()
        res = QCORONAOptimizer(H_obj, opt_cfg).optimize(initial_state=s0)
        elapsed = time.time() - t0
        d = res.final_H.diagnostics
        print(f"  seed {seed}: H={res.final_H.H_total:.4f}  "
              f"lag={d['delta_t_pred']:.2f}  ({elapsed:.0f}s)")
        all_results.append(res)

    best = min(all_results, key=lambda r: r.final_H.H_total)
    w_best = best.final_state.emissivity
    d_best = best.final_H.diagnostics
    lag_pred = float(d_best['delta_t_pred'])
    H_final  = float(best.final_H.H_total)

    tf_pred = compute_transfer_function_full(
        GRID, DISK, w_best, SA, observer_inclination=INCL, n_bins=N_BINS_TF,
        t_min=float(t_bins[0]), t_max=float(t_bins[-1]), lag_matrix=LM,
    )
    chi2 = float(np.sum((tf_pred.psi - psi_obs)**2))

    rows.append({
        'r_in': r_in,
        'lag_pred_rgc': lag_pred,
        'H_final': H_final,
        'chi2_psi': chi2,
        'I_pred': float(d_best['I_pred']),
        'LM_min': float(LM.min()),
        'LM_max': float(LM.max()),
    })

    def rz_map(w):
        return w.reshape(GRID.n_r, GRID.n_z, GRID.n_phi).sum(axis=2)

    R_c = np.unique(np.round(GRID.R, 8))[:GRID.n_r]
    z_c = np.unique(np.round(GRID.z, 8))[:GRID.n_z]

    fig = plt.figure(figsize=(7.2, 3.6))
    gs  = gridspec.GridSpec(1, 2, left=0.08, right=0.97, top=0.88,
                            bottom=0.14, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pcolormesh(R_c, z_c, rz_map(w_best).T, cmap='hot_r',
                   shading='auto', rasterized=True)
    ax1.set_xlabel(r'$R\,[r_g]$')
    ax1.set_ylabel(r'$z\,[r_g]$')
    ax1.set_title(f'$r_{{\\rm in}}={r_in}$  lag={lag_pred:.2f}')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(t_c, psi_obs, alpha=0.2, color='#d6604d')
    ax2.plot(t_c, psi_obs, color='#d6604d', lw=1.2)
    ax2.plot(t_c, tf_pred.psi, color='#2166ac', ls='--', lw=1.1)
    ax2.set_xlabel(r'$\Delta t\,[r_g/c]$')
    ax2.set_ylabel(r'$\psi$')
    ax2.set_title(f'H={H_final:.3f}')
    ax2.set_ylim(bottom=0)

    fig.suptitle(
        f'MCG-6 scan  $r_{{\\rm in}}={r_in}$  '
        f'(obs lag {LAG_OBS_RG:.2f}, $M={M_BH:.1e}\\,M_\\odot$)',
        fontsize=10,
    )
    out = f'results/mcg6_rin_scan/fig_mcg6_rin{r_in}.pdf'
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {out}")

csv_path = 'results/mcg6_rin_scan/summary.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved {csv_path}")

print("\nSUMMARY")
print(f"{'r_in':>6} {'lag_pred':>10} {'H_final':>10} {'chi2':>12}")
for r in rows:
    print(f"{r['r_in']:6.1f} {r['lag_pred_rgc']:10.2f} "
          f"{r['H_final']:10.4f} {r['chi2_psi']:12.4e}")