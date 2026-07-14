"""
save_best_emissivities.py  (updated for production grid + corrected geometries)
────────────────────────────────────────────────────────────────────────────────
Re-runs ONLY the best seed per geometry and patches recovery_results.npz.

Best seeds from the completed run (production grid):
  lamppost  seed=2  H=54.1442  corr=+0.243
  column    seed=4  H=4.3132   corr=+0.498
  ring      seed=2  H=0.0029   corr=+0.118
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
from qcorona.data.synthetic import (
    create_synthetic_geometry, emissivity_correlation, emissivity_overlap,
)
from qcorona.hamiltonian.terms import vertical_orientations

# ── production grid (same as real-data fits) ─────────────────────────────────
GRID = build_grid(n_r=10, n_z=10, n_phi=8,
                  r_in=3.0, r_out=15.0,
                  z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)

# ── corrected geometry parameters (fit within production grid) ────────────────
GEOM_KWARGS = {
    'lamppost': dict(height=2.5,  width=0.8),
    'column':   dict(z_min=0.5,   z_max=4.5, radius=5.0),
    'ring':     dict(R_center=8.0, z_center=2.0, R_width=2.0, z_width=1.0),
}

# ── best seeds from the latest run ───────────────────────────────────────────
BEST_SEEDS = {'lamppost': 2, 'column': 4, 'ring': 2}

MAX_ITER = 800

opt_cfg = OptimizationConfig(
    max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
    use_adaptive_lr=True, use_continuation=True,
    continuation_steps=200, convergence_tol=1e-7, verbose=False,
)

def make_hamiltonian(synthetic):
    R_obs = synthetic.I_true / (1 - synthetic.I_true + 1e-10)
    constraints = ObservationalConstraints(
        delta_t_obs=synthetic.delta_t_true, R_obs=R_obs,
        L_X=1e43, M_bh=1e8, eddington_ratio=0.1,
        observer_inclination=30.0,
        source_name=f"synthetic_{synthetic.name}",
    )
    h_config = HamiltonianConfig(
        J=0.05, alpha=10.0, beta=10.0,
        gamma=1.0, l_crit=1000.0, delta=1.0, f_max=1.0,
        use_pair_barrier=False, use_energy_barrier=False,
    )
    return QCORONAHamiltonian(GRID, DISK, constraints, h_config)

# ── load existing npz ─────────────────────────────────────────────────────────
npz_path = 'results/recovery_results.npz'
if not os.path.exists(npz_path):
    print(f"ERROR: {npz_path} not found. Run run_recovery.py first.")
    sys.exit(1)

existing = dict(np.load(npz_path, allow_pickle=False))
print(f"Loaded {npz_path}  ({len(existing)} keys)")

# ── re-run best seed per geometry ────────────────────────────────────────────
extra = {}
for gtype in ['lamppost', 'column', 'ring']:
    seed = BEST_SEEDS[gtype]
    print(f"\nRe-running {gtype}  (seed={seed})...")
    t0 = time.time()

    syn   = create_synthetic_geometry(GRID, DISK, gtype, **GEOM_KWARGS[gtype])
    H_obj = make_hamiltonian(syn)

    rng = np.random.default_rng(seed)
    w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
    s0  = CoronalState(
        emissivity=w0,
        orientations=vertical_orientations(GRID.n_cells),
    )
    opt = QCORONAOptimizer(H_obj, opt_cfg)
    res = opt.optimize(initial_state=s0)

    w_true = syn.emissivity
    w_rec  = res.final_state.emissivity
    corr   = emissivity_correlation(w_true, w_rec)
    H_hist = np.array([h['H_total'] for h in res.history])

    print(f"  corr={corr:+.3f}  H={res.final_H.H_total:.4f}  "
          f"({time.time()-t0:.0f}s)")

    extra[f'{gtype}_w_true']       = w_true
    extra[f'{gtype}_w_rec']        = w_rec
    extra[f'{gtype}_history_H']    = H_hist
    extra[f'{gtype}_delta_t_true'] = np.array([syn.delta_t_true])
    extra[f'{gtype}_I_true']       = np.array([syn.I_true])

# ── merge and save ────────────────────────────────────────────────────────────
merged = {**existing, **extra}
np.savez(npz_path, **merged)
print(f"\nPatched: {npz_path}  ({len(merged)} keys)")
print("Keys added:", list(extra.keys()))
print("\nDone — now run: python make_figure_v3.py")
