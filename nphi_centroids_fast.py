"""
nphi_centroids_fast.py -- fast Mrk 335 (2006) fit at a given n_phi, prints centroid, H, lag_pred.
1 seed, 800 iterations (the 30-seed test shows all seeds reach the same minimum at n_phi=8).

    python nphi_centroids_fast.py 8      # CHECK FIRST: must give H~0.338, lag~2.31, R~7.6, z~0.4
    python nphi_centroids_fast.py 4
    python nphi_centroids_fast.py 6
    python nphi_centroids_fast.py 12
    python nphi_centroids_fast.py 16
(run in separate terminals; each prints one line and saves results/nphi_<n>.npz)
Optional: set NPHI_ITER (default 800), NPHI_SEEDS (default 1).
"""
import sys, os, time
import numpy as np
sys.path.insert(0, '.')
from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import (QCORONAHamiltonian, HamiltonianConfig,
                                       ObservationalConstraints, CoronalState)
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.hamiltonian.terms import vertical_orientations
from qcorona.physics.illumination import compute_solid_angles
from qcorona.physics.lags import compute_lag_matrix

NPHI = int(sys.argv[1]); MAX_ITER = int(os.environ.get('NPHI_ITER', 800))
N_SEEDS = int(os.environ.get('NPHI_SEEDS', 1))
M_BH, R_G_C = 1.3e7, 63.0
LAG_S, ERR_S, R_OBS, L_X, INCL = 91.9, 43.4, 0.3, 1.0e43, 45.0   # Mrk 335, 2006 (as in multi-epoch script)
lag_rg, lag_err = LAG_S / R_G_C, ERR_S / R_G_C

GRID = build_grid(n_r=10, n_z=10, n_phi=NPHI, r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
LM = compute_lag_matrix(GRID, DISK, observer_inclination=INCL)
t_bins = np.linspace(max(0.0, float(LM.min())), 15.0, 81)
t_c = 0.5 * (t_bins[:-1] + t_bins[1:])
psi_obs = np.exp(-0.5 * ((t_c - lag_rg) / lag_err) ** 2); psi_obs /= psi_obs.sum()

cons = ObservationalConstraints(delta_t_obs=lag_rg, R_obs=R_OBS, L_X=L_X, M_bh=M_BH,
        eddington_ratio=L_X / (1.3e38 * M_BH), observer_inclination=INCL,
        source_name=f"Mrk335_nphi{NPHI}", psi_obs=psi_obs, t_bins_obs=t_bins)
hcfg = HamiltonianConfig(J=0.3, alpha=50.0, beta=10.0, gamma=1.0, l_crit=1000.0, delta=1.0,
        f_max=1.0, use_pair_barrier=False, use_energy_barrier=False)
Hobj = QCORONAHamiltonian(GRID, DISK, cons, hcfg)
ocfg = OptimizationConfig(max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
        use_adaptive_lr=True, use_continuation=True, continuation_steps=200,
        convergence_tol=1e-7, verbose=False)

t0 = time.time(); best = None
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    w0 = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
    s0 = CoronalState(emissivity=w0, orientations=vertical_orientations(GRID.n_cells))
    res = QCORONAOptimizer(Hobj, ocfg).optimize(initial_state=s0)
    if best is None or res.final_H.H_total < best.final_H.H_total:
        best = res
w = best.final_state.emissivity
Rc, zc = float(np.dot(w, GRID.R)), float(np.dot(w, GRID.z))
lag = best.final_H.diagnostics['delta_t_pred']
os.makedirs('results', exist_ok=True)
np.savez(f'results/nphi_{NPHI}.npz', n_phi=NPHI, R_c=Rc, z_c=zc, H=best.final_H.H_total, lag_pred=lag, w=w)
print(f"n_phi={NPHI:>2}  N={GRID.n_cells:>4}  H={best.final_H.H_total:.4f}  lag_pred={lag:.2f}  "
      f"R_c={Rc:.2f}  z_c={zc:.2f}   ({time.time()-t0:.0f}s)")
