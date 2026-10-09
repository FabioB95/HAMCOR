"""
uncertainty_multiseed.py — Fix 6: caratterizzazione dell'incertezza via
molti piu' multi-start (20-30 invece di 5), come proxy onesto di un vero
posterior (troppo costoso: 800 dimensioni, vedi discussione).

Riporta:
  - dispersione di H_final, lag_pred tra i minimi locali trovati
  - deviazione standard cella-per-cella di w tra i run convergenti
  - "credible band" empirica sulla mappa R-z (percentili 16/50/84)
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

# ── parametri Mrk 335 (stessi di sempre) ──────────────────────────────────────
M_BH       = 1.3e7
R_G_C      = 63.0
LAG_OBS_S  = 91.9
LAG_ERR_S  = 43.4
LAG_OBS_RG = LAG_OBS_S / R_G_C
LAG_ERR_RG = LAG_ERR_S / R_G_C
R_OBS      = 0.3
L_X        = 1.0e43
INCL       = 45.0

N_STARTS_UNCERTAINTY = 30   # <-- molto piu' di 5
MAX_ITER  = 2000
N_BINS_TF = 80

GRID = build_grid(n_r=10, n_z=10, n_phi=8, r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
LM   = compute_lag_matrix_schwarzschild(GRID, DISK, observer_inclination=INCL)

T_MIN, T_MAX = max(0.0, float(LM.min())), 15.0
t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
t_c    = 0.5 * (t_bins[:-1] + t_bins[1:])
psi_obs = np.exp(-0.5 * ((t_c - LAG_OBS_RG) / LAG_ERR_RG)**2)
psi_obs = psi_obs / psi_obs.sum()

constraints = ObservationalConstraints(
    delta_t_obs=LAG_OBS_RG, R_obs=R_OBS, L_X=L_X, M_bh=M_BH,
    eddington_ratio=0.25, observer_inclination=INCL,
    source_name="Mrk335_uncertainty",
    psi_obs=psi_obs, t_bins_obs=t_bins,
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

print("=" * 60)
print(f"Fix 6 — incertezza multi-seed (Mrk 335, {N_STARTS_UNCERTAINTY} start)")
print("=" * 60)

all_w, all_H, all_lag = [], [], []
for seed in range(N_STARTS_UNCERTAINTY):
    rng = np.random.default_rng(seed)
    w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
    s0  = CoronalState(emissivity=w0, orientations=vertical_orientations(GRID.n_cells))
    t0  = time.time()
    res = QCORONAOptimizer(H_obj, opt_cfg).optimize(initial_state=s0)
    d = res.final_H.diagnostics
    print(f"  seed {seed:>2}: H={res.final_H.H_total:.4f}  lag={d['delta_t_pred']:.2f}  "
          f"({time.time()-t0:.0f}s)")
    all_w.append(res.final_state.emissivity)
    all_H.append(res.final_H.H_total)
    all_lag.append(d['delta_t_pred'])

all_w   = np.array(all_w)     # (N_STARTS, N_cells)
all_H   = np.array(all_H)
all_lag = np.array(all_lag)

print("\n" + "=" * 60)
print("STATISTICHE DI DISPERSIONE (proxy di incertezza)")
print("=" * 60)
print(f"  H_final:   media={all_H.mean():.4f}  std={all_H.std():.4f}  "
      f"min={all_H.min():.4f}  max={all_H.max():.4f}")
print(f"  lag_pred:  media={all_lag.mean():.3f}  std={all_lag.std():.3f}  "
      f"min={all_lag.min():.3f}  max={all_lag.max():.3f} r_g/c")

# ── mappa R-z con percentili 16/50/84 ──────────────────────────────────────────
def rz_map(w):
    return w.reshape(GRID.n_r, GRID.n_z, GRID.n_phi).sum(axis=2)

maps = np.array([rz_map(w) for w in all_w])   # (N_STARTS, n_r, n_z)
p16  = np.percentile(maps, 16, axis=0)
p50  = np.percentile(maps, 50, axis=0)
p84  = np.percentile(maps, 84, axis=0)
std_map = maps.std(axis=0)

print(f"\n  Deviazione standard cella-per-cella (mediana su tutte le celle): "
      f"{np.median(std_map):.4f}")
print(f"  Cella piu' incerta (max std): {std_map.max():.4f} "
      f"a indice {np.unravel_index(std_map.argmax(), std_map.shape)}")

np.savez('results/mrk335_uncertainty.npz',
         all_w=all_w, all_H=all_H, all_lag=all_lag,
         p16=p16, p50=p50, p84=p84, std_map=std_map)
print("\nSalvato: results/mrk335_uncertainty.npz")
print("(contiene tutto il necessario per una figura con banda di credibilita' 16-84%)")