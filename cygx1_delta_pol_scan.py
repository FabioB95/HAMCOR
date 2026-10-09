#!/usr/bin/env python3
"""
run_cygx1_with_polarization.py
────────────────────────────────
Fit Cyg X-1 con lag (Schwarzschild) + polarizzazione IXPE (H_pol).

DELTA_POL = 3.0  → valore fiducial per il paper
  (PA_pred ≈ −21.7°, stabile; figura: results/fig_cygx1_pol_3.0.pdf)

IXPE (Krawczynski+2022): PA = −20.7 deg, PD = 4.01%
"""

import sys, os, time
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
from qcorona.physics.lags import compute_transfer_function_full
from schwarzschild_lags import compute_lag_matrix_schwarzschild
from polarization import compute_projected_moments, calibrate_k

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── IXPE ──────────────────────────────────────────────────────────────────────
PA_OBS_DEG = -20.7
PD_OBS_PCT = 4.01

# ── Cyg X-1 ───────────────────────────────────────────────────────────────────
M_BH       = 14.8
R_G_C      = 7.29e-5
LAG_OBS_S  = 2.1e-3
LAG_ERR_S  = 0.4e-3
LAG_OBS_RG = LAG_OBS_S / R_G_C
LAG_ERR_RG = LAG_ERR_S / R_G_C
R_OBS      = 0.5
L_X        = 2.0e38
INCL       = 27.1

N_STARTS   = 5
MAX_ITER   = 2000
N_BINS_TF  = 80
DELTA_POL  = 3.0    # fiducial paper
W_PA_POL   = 1.0

print("=" * 60)
print("Cyg X-1 — HAMCOR + H_pol (IXPE)")
print("=" * 60)
print(f"  lag obs   = {LAG_OBS_RG:.1f} +/- {LAG_ERR_RG:.1f} r_g/c")
print(f"  PA_obs    = {PA_OBS_DEG:+.1f} deg")
print(f"  PD_obs    = {PD_OBS_PCT:.2f}%")
print(f"  delta_pol = {DELTA_POL}")

GRID = build_grid(n_r=10, n_z=10, n_phi=8,
                  r_in=3.0, r_out=50.0, z_min=0.1, z_max=10.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
LM   = compute_lag_matrix_schwarzschild(GRID, DISK, observer_inclination=INCL)

T_MIN  = max(0.0, float(LM.min()))
T_MAX  = max(LAG_OBS_RG * 3, 80.0)
t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
t_c    = 0.5 * (t_bins[:-1] + t_bins[1:])
psi_obs = np.exp(-0.5 * ((t_c - LAG_OBS_RG) / LAG_ERR_RG)**2)
psi_obs = psi_obs / psi_obs.sum()

opt_cfg = OptimizationConfig(
    max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
    use_adaptive_lr=True, use_continuation=True,
    continuation_steps=200, convergence_tol=1e-7, verbose=False,
)


def run_multi_start(H_obj, label):
    print(f"\n--- {label} ({N_STARTS} seeds) ---")
    all_res = []
    for seed in range(N_STARTS):
        rng = np.random.default_rng(seed)
        w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
        s0  = CoronalState(
            emissivity=w0,
            orientations=vertical_orientations(GRID.n_cells),
        )
        t0  = time.time()
        res = QCORONAOptimizer(H_obj, opt_cfg).optimize(initial_state=s0)
        d   = res.final_H.diagnostics
        pa  = d.get('PA_pred')
        pa_s = f"{pa:.1f}" if pa is not None else "n/a"
        print(f"  seed {seed}: H={res.final_H.H_total:.4f}  "
              f"lag={d['delta_t_pred']:.1f}  PA={pa_s}  "
              f"({time.time()-t0:.0f}s)")
        all_res.append(res)
    best = min(all_res, key=lambda r: r.final_H.H_total)
    return best, all_res


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1: solo lag → k_calib fisso
# ══════════════════════════════════════════════════════════════════════════════
c_lag = ObservationalConstraints(
    delta_t_obs=LAG_OBS_RG, R_obs=R_OBS, L_X=L_X, M_bh=M_BH,
    eddington_ratio=0.1, observer_inclination=INCL,
    source_name="CygX1_lag_only",
    psi_obs=psi_obs, t_bins_obs=t_bins,
)
h_lag = HamiltonianConfig(
    J=0.3, alpha=50.0, beta=10.0, gamma=1.0, l_crit=1000.0,
    delta=1.0, f_max=1.0, use_pair_barrier=False, use_energy_barrier=False,
)
H_lag = QCORONAHamiltonian(GRID, DISK, c_lag, h_lag, lag_matrix=LM)
best_lag, _ = run_multi_start(H_lag, "Fase 1: solo lag")
w_lag = best_lag.final_state.emissivity
d_lag = best_lag.final_H.diagnostics

moments = compute_projected_moments(GRID, w_lag, observer_inclination=INCL)
ecc = moments['eccentricity']
print(f"\n  eccentricity (solo-lag) = {ecc:.4f}")
print(f"  PA_pred (solo-lag)      = {moments['PA_pred_deg']:+.1f} deg")

if ecc < 1e-10:
    print("  ATTENZIONE: e~0 — k_calib non definibile; abort.")
    sys.exit(1)

k_calib = calibrate_k(ecc, PD_OBS_PCT)
print(f"  k_calib (fisso)         = {k_calib:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Fase 2: lag + polarizzazione (delta_pol = 3.0)
# ══════════════════════════════════════════════════════════════════════════════
c_pol = ObservationalConstraints(
    delta_t_obs=LAG_OBS_RG, R_obs=R_OBS, L_X=L_X, M_bh=M_BH,
    eddington_ratio=0.1, observer_inclination=INCL,
    source_name="CygX1_lag_pol_d3",
    psi_obs=psi_obs, t_bins_obs=t_bins,
    PA_obs=PA_OBS_DEG, PD_obs=PD_OBS_PCT, k_calib_pol=k_calib,
)
h_pol = HamiltonianConfig(
    J=0.3, alpha=50.0, beta=10.0, gamma=1.0, l_crit=1000.0,
    delta=1.0, f_max=1.0, use_pair_barrier=False, use_energy_barrier=False,
    delta_pol=DELTA_POL, w_PA_pol=W_PA_POL,
)
H_pol = QCORONAHamiltonian(GRID, DISK, c_pol, h_pol, lag_matrix=LM)
best_pol, all_pol = run_multi_start(H_pol, f"Fase 2: lag + H_pol (delta_pol={DELTA_POL})")
w_pol = best_pol.final_state.emissivity
d_pol = best_pol.final_H.diagnostics

print("\n" + "=" * 60)
print("CONFRONTO")
print("=" * 60)
print(f"  Solo lag:  H={best_lag.final_H.H_total:.4f}  "
      f"lag={d_lag['delta_t_pred']:.1f}  "
      f"PA={moments['PA_pred_deg']:+.1f}")
print(f"  Lag+pol:   H={best_pol.final_H.H_total:.4f}  "
      f"lag={d_pol['delta_t_pred']:.1f}  "
      f"PA_pred={d_pol['PA_pred']}  PD_pred={d_pol['PD_pred']}")
print(f"  IXPE:      PA={PA_OBS_DEG:+.1f}  PD={PD_OBS_PCT:.2f}%")
print(f"  H_pol term = {best_pol.final_H.H_pol:.4f}")
print(f"  delta_pol  = {DELTA_POL}")

# ── figura ────────────────────────────────────────────────────────────────────
def rz_map(w):
    return w.reshape(GRID.n_r, GRID.n_z, GRID.n_phi).sum(axis=2)

R_c = np.unique(np.round(GRID.R, 8))[:GRID.n_r]
z_c = np.unique(np.round(GRID.z, 8))[:GRID.n_z]

tf = compute_transfer_function_full(
    GRID, DISK, w_pol, SA, observer_inclination=INCL, n_bins=N_BINS_TF,
    t_min=float(t_bins[0]), t_max=float(t_bins[-1]), lag_matrix=LM,
)

fig = plt.figure(figsize=(7.2, 4.0))
gs  = gridspec.GridSpec(1, 3, left=0.09, right=0.97, top=0.86, bottom=0.14, wspace=0.45)

ax0 = fig.add_subplot(gs[0, 0])
for res in all_pol:
    H_arr = np.array([h['H_total'] for h in res.history])
    ax0.semilogy(np.abs(H_arr), color='#2166ac',
                 alpha=1.0 if res is best_pol else 0.25,
                 lw=1.3 if res is best_pol else 0.7)
ax0.set_xlabel('Iteration')
ax0.set_ylabel(r'$|\mathcal{H}|$')
ax0.set_title(r'Convergence ($\delta_{\rm pol}=3$)')

ax1 = fig.add_subplot(gs[0, 1])
ax1.pcolormesh(R_c, z_c, rz_map(w_pol).T, cmap='hot_r', shading='auto', rasterized=True)
ax1.set_xlabel(r'$R\,[r_g]$')
ax1.set_ylabel(r'$z\,[r_g]$')
ax1.set_title('Corona (lag+pol)')
ax1.text(
    0.05, 0.95,
    f"lag={d_pol['delta_t_pred']:.1f}\n"
    f"PA={d_pol['PA_pred']:.1f}\n"
    f"PD={d_pol['PD_pred']:.2f}",
    transform=ax1.transAxes, va='top', fontsize=7,
    bbox=dict(boxstyle='round', fc='white', alpha=0.85),
)

ax2 = fig.add_subplot(gs[0, 2])
ax2.fill_between(t_c, psi_obs, alpha=0.2, color='#d6604d')
ax2.plot(t_c, psi_obs, color='#d6604d', lw=1.2, label=r'$\psi_{\rm obs}$')
ax2.plot(t_c, tf.psi, color='#2166ac', ls='--', lw=1.1, label=r'$\psi_{\rm pred}$')
ax2.set_xlabel(r'$\Delta t\,[r_g/c]$')
ax2.set_ylabel(r'$\psi$')
ax2.legend(fontsize=7)
ax2.set_ylim(bottom=0)
ax2.set_title('Transfer function')

fig.suptitle(
    f'Cyg X-1 — lag + H_pol '
    f'(IXPE PA={PA_OBS_DEG}, PD={PD_OBS_PCT}%; '
    f'$\\delta_{{\\rm pol}}={DELTA_POL}$, $k={k_calib:.3f}$)',
    fontsize=9,
)

os.makedirs('results', exist_ok=True)
out_pdf = 'results/fig_cygx1_pol_3.0.pdf'
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {out_pdf}")
print("Done.")