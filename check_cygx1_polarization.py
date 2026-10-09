#!/usr/bin/env python3
"""
check_cygx1_polarization.py
─────────────────────────────
Diagnostico post-hoc: rifà il fit di Cyg X-1 (identico a run_cygx1.py
con correzione Schwarzschild) e calcola PA_pred, PD_pred (Opzione C)
sulla configurazione best-fit, confrontandoli con i dati IXPE
(Krawczynski et al. 2022).

NOTA IMPORTANTE: run_cygx1.py usa INCL=27.1 deg (inclinazione orbitale,
Orosz+2011). La letteratura sulla polarizzazione (Krawczynski+2022, e
il lavoro successivo "Assessing Systematic Uncertainties...") indica
che il fit polarimetrico preferisce inclinazioni più alte (>45-65 deg)
per una corona a sandwich. Se PA_pred/PD_pred non tornano bene a 27.1
deg, questa tensione nota in letteratura è un candidato spiegazione
-- non necessariamente un problema del metodo.
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
from schwarzschild_lags import compute_lag_matrix_schwarzschild   # <-- aggiornato
from polarization import compute_projected_moments, calibrate_k, H_pol

# ── dati IXPE osservati (Krawczynski et al. 2022, Science 378, 650) ──────────
PA_OBS_DEG = -20.7
PD_OBS_PCT = 4.01

# ── stessi identici parametri di run_cygx1.py ─────────────────────────────────
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

print("=" * 60)
print("Diagnostico polarizzazione — Cyg X-1 (Schwarzschild lags)")
print("=" * 60)

GRID = build_grid(n_r=10, n_z=10, n_phi=8, r_in=3.0, r_out=50.0,
                  z_min=0.1, z_max=10.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
LM   = compute_lag_matrix_schwarzschild(GRID, DISK, observer_inclination=INCL)  # <-- aggiornato

T_MIN  = max(0.0, float(LM.min()))
T_MAX  = max(LAG_OBS_RG * 3, 80.0)
t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
t_c    = 0.5 * (t_bins[:-1] + t_bins[1:])
psi_obs = np.exp(-0.5 * ((t_c - LAG_OBS_RG) / LAG_ERR_RG) ** 2)
psi_obs = psi_obs / psi_obs.sum()

constraints = ObservationalConstraints(
    delta_t_obs=LAG_OBS_RG, R_obs=R_OBS, L_X=L_X, M_bh=M_BH,
    eddington_ratio=0.1, observer_inclination=INCL,
    source_name="CygX1_XMM2004", psi_obs=psi_obs, t_bins_obs=t_bins,
)
h_config = HamiltonianConfig(
    J=0.3, alpha=50.0, beta=10.0, gamma=1.0, l_crit=1000.0,
    delta=1.0, f_max=1.0, use_pair_barrier=False, use_energy_barrier=False,
)
H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config, lag_matrix=LM)  # <-- aggiornato

print(f"Optimising ({N_STARTS} cold starts)...")
all_results = []
for seed in range(N_STARTS):
    rng = np.random.default_rng(seed)
    w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
    s0  = CoronalState(emissivity=w0, orientations=vertical_orientations(GRID.n_cells))
    opt = QCORONAOptimizer(
        H_obj,
        OptimizationConfig(
            max_iter=MAX_ITER,
            lr_emissivity=0.01,
            lr_orientation=0.01,
            use_adaptive_lr=True,
            use_continuation=True,
            continuation_steps=200,
            convergence_tol=1e-7,
            verbose=False,
        ),
    )
    res = opt.optimize(initial_state=s0)
    all_results.append(res)

best_idx = int(np.argmin([r.final_H.H_total for r in all_results]))
best     = all_results[best_idx]
w_best   = best.final_state.emissivity
d_best   = best.final_H.diagnostics

print(f"\nBest fit (seed {best_idx}): H_total={best.final_H.H_total:.4f}, "
      f"lag_pred={d_best['delta_t_pred']:.1f} r_g/c "
      f"(obs: {LAG_OBS_RG:.1f}+/-{LAG_ERR_RG:.1f})")

# ── diagnostico polarizzazione (Opzione C) ────────────────────────────────────
moments = compute_projected_moments(GRID, w_best, observer_inclination=INCL)

print("\n" + "=" * 60)
print("POLARIZZAZIONE — confronto con IXPE (Krawczynski+2022)")
print("=" * 60)
print(f"  Eccentricita' proiettata:  {moments['eccentricity']:.4f}")
print(f"  PA_pred (geometria pura, NON calibrato): {moments['PA_pred_deg']:+.1f} deg")
print(f"  PA_obs  (IXPE):                          {PA_OBS_DEG:+.1f} deg")
print(f"  |PA_pred - PA_obs| (mod 180):            "
      f"{abs(((moments['PA_pred_deg']-PA_OBS_DEG+90)%180)-90):.1f} deg")

if moments['eccentricity'] > 1e-10:
    k = calibrate_k(moments['eccentricity'], PD_OBS_PCT)
    PD_pred = k * moments['eccentricity']
    print(f"\n  k calibrato:              {k:.3f}")
    print(f"  PD_pred (calibrato):      {PD_pred:.2f}%  (PD_obs: {PD_OBS_PCT:.2f}%, per costruzione uguali)")
    h_pol_val = H_pol(moments['PA_pred_deg'], PD_pred, PA_OBS_DEG, PD_OBS_PCT)
    print(f"\n  H_pol (post-hoc, w_PA=1): {h_pol_val:.4f}")
else:
    print("\n  ATTENZIONE: eccentricita' ~0 -- geometria quasi assisimmetrica "
          "proiettata sul cielo, PD non calibrabile con questo metodo a INCL=27.1 deg.")

print("=" * 60)