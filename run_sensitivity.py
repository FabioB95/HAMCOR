#!/usr/bin/env python3
"""
run_sensitivity.py  —  Sensitività iperparametri, STESSO setup del recovery.

Griglia, geometrie, α/β/J di riferimento = run_recovery.py.
Niente ψ_obs nel vincolo (come il recovery validato).
Al punto di riferimento devono uscire ρ ≈ 0.24, 0.50, 0.12.

Output:
    results/sensitivity_results.npz
    results/fig_sensitivity.pdf
    results/fig_sensitivity.tex
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
    create_synthetic_geometry,
    emissivity_correlation, emissivity_overlap, centroid_distance,
)
from qcorona.hamiltonian.terms import vertical_orientations
from qcorona.physics.illumination import compute_solid_angles
from qcorona.physics.lags import compute_lag_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'dejavusans',
})

# ── griglia = recovery ────────────────────────────────────────────────────────
print("Building grids (recovery setup)...")
GRID = build_grid(
    n_r=10, n_z=10, n_phi=8,
    r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0,
)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
LM   = compute_lag_matrix(GRID, DISK, observer_inclination=30.0)
print(f"  Corona: {GRID.n_cells} cells  |  Disk: {DISK.n_elements} elements")

# ── geometrie = recovery ──────────────────────────────────────────────────────
GEOM_CONFIGS = {
    'lamppost': dict(height=2.5, width=0.8),
    'column':   dict(z_min=0.5, z_max=4.5, radius=5.0),
    'ring':     dict(R_center=8.0, z_center=2.0, R_width=2.0, z_width=1.0),
}

print("Pre-computing synthetic geometries...")
GEOM_DATA = {}
for gtype, kwargs in GEOM_CONFIGS.items():
    syn = create_synthetic_geometry(GRID, DISK, gtype, **kwargs)
    R_obs = syn.I_true / (1.0 - syn.I_true + 1e-10)
    GEOM_DATA[gtype] = dict(synthetic=syn, R_obs=R_obs)
    print(f"  {gtype}: lag={syn.delta_t_true:.2f} r_g/c, I={syn.I_true:.3f}")

# ── riferimenti = recovery ────────────────────────────────────────────────────
ALPHA_REF = 10.0
BETA_REF  = 10.0
J_REF     = 0.05

ALPHA_VALS = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
BETA_VALS  = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
J_VALS     = np.array([0.005, 0.01, 0.05, 0.1, 0.5, 1.0])

MAX_ITER = 800
N_SEEDS  = 5

OPT_CFG = OptimizationConfig(
    max_iter=MAX_ITER,
    lr_emissivity=0.01,
    lr_orientation=0.01,
    use_adaptive_lr=True,
    use_continuation=True,
    continuation_steps=200,
    convergence_tol=1e-7,
    verbose=False,
)


def make_hamiltonian(gtype, alpha, beta, J):
    """Come recovery: solo lag + illuminazione, niente psi_obs."""
    gd  = GEOM_DATA[gtype]
    syn = gd['synthetic']
    constraints = ObservationalConstraints(
        delta_t_obs=syn.delta_t_true,
        R_obs=gd['R_obs'],
        L_X=1e43,
        M_bh=1e8,
        eddington_ratio=0.1,
        observer_inclination=30.0,
        source_name=f"synthetic_{gtype}",
        # NO psi_obs / t_bins_obs  (allineato al recovery)
    )
    h_config = HamiltonianConfig(
        J=J, alpha=alpha, beta=beta,
        gamma=1.0, l_crit=1000.0,
        delta=1.0, f_max=1.0,
        use_pair_barrier=False,
        use_energy_barrier=False,
    )
    return QCORONAHamiltonian(
        GRID, DISK, constraints, h_config, lag_matrix=LM,
    )


def run_single(gtype, alpha, beta, J):
    H_obj = make_hamiltonian(gtype, alpha, beta, J)
    syn   = GEOM_DATA[gtype]['synthetic']

    best_H = np.inf
    best_w = None
    best_diag = None

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)   # come recovery (non seed+100)
        w0  = rng.exponential(size=GRID.n_cells)
        w0 /= w0.sum()
        s0  = CoronalState(
            emissivity=w0,
            orientations=vertical_orientations(GRID.n_cells),
        )
        res = QCORONAOptimizer(H_obj, OPT_CFG).optimize(initial_state=s0)
        if res.final_H.H_total < best_H:
            best_H = res.final_H.H_total
            best_w = res.final_state.emissivity
            best_diag = res.final_H.diagnostics

    corr      = float(emissivity_correlation(syn.emissivity, best_w))
    overlap   = float(emissivity_overlap(syn.emissivity, best_w))
    cent_err  = float(centroid_distance(GRID, syn.emissivity, best_w))
    lag_err   = abs(best_diag['delta_t_pred'] - syn.delta_t_true) / (
        syn.delta_t_true + 1e-10
    )
    illum_err = abs(best_diag['I_pred'] - syn.I_true) / (syn.I_true + 1e-10)

    return dict(
        corr=corr, overlap=overlap, cent_err=cent_err,
        lag_err=lag_err, illum_err=illum_err, H_final=best_H,
    )


# ── scan alpha ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Scanning alpha: {ALPHA_VALS}  (beta={BETA_REF}, J={J_REF})")
print(f"{'='*55}")

res_alpha = {g: [] for g in GEOM_CONFIGS}
for alpha in ALPHA_VALS:
    print(f"  alpha={alpha:.1f} ... ", end='', flush=True)
    t0 = time.time()
    for gtype in GEOM_CONFIGS:
        r = run_single(gtype, alpha, BETA_REF, J_REF)
        res_alpha[gtype].append(r)
        print(f"\n    {gtype}: ρ={r['corr']:+.3f}", end='', flush=True)
    print(f"  ({time.time()-t0:.0f}s)")

# ── scan beta ─────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Scanning beta: {BETA_VALS}  (alpha={ALPHA_REF}, J={J_REF})")
print(f"{'='*55}")

res_beta = {g: [] for g in GEOM_CONFIGS}
for beta in BETA_VALS:
    print(f"  beta={beta:.1f} ... ", end='', flush=True)
    t0 = time.time()
    for gtype in GEOM_CONFIGS:
        r = run_single(gtype, ALPHA_REF, beta, J_REF)
        res_beta[gtype].append(r)
    print(f"{time.time()-t0:.0f}s")

# ── scan J ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Scanning J: {J_VALS}  (alpha={ALPHA_REF}, beta={BETA_REF})")
print(f"{'='*55}")

res_J = {g: [] for g in GEOM_CONFIGS}
for J in J_VALS:
    print(f"  J={J:.3f} ... ", end='', flush=True)
    t0 = time.time()
    for gtype in GEOM_CONFIGS:
        r = run_single(gtype, ALPHA_REF, BETA_REF, J)
        res_J[gtype].append(r)
    print(f"{time.time()-t0:.0f}s")

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

save_dict = {
    'alpha_vals': ALPHA_VALS,
    'beta_vals':  BETA_VALS,
    'J_vals':     J_VALS,
    'alpha_ref':  np.array([ALPHA_REF]),
    'beta_ref':   np.array([BETA_REF]),
    'J_ref':      np.array([J_REF]),
}
for gtype in GEOM_CONFIGS:
    for metric in ('corr', 'overlap', 'lag_err', 'illum_err', 'H_final'):
        save_dict[f'alpha_{gtype}_{metric}'] = np.array(
            [r[metric] for r in res_alpha[gtype]]
        )
        save_dict[f'beta_{gtype}_{metric}'] = np.array(
            [r[metric] for r in res_beta[gtype]]
        )
        save_dict[f'J_{gtype}_{metric}'] = np.array(
            [r[metric] for r in res_J[gtype]]
        )

np.savez('results/sensitivity_results.npz', **save_dict)
print("\nSaved: results/sensitivity_results.npz")

# ── figura ────────────────────────────────────────────────────────────────────
COLORS  = {'lamppost': '#2166ac', 'column': '#d6604d', 'ring': '#4dac26'}
LABELS  = {'lamppost': 'Lamp-post', 'column': 'Column', 'ring': 'Ring'}
MARKERS = {'lamppost': 'o', 'column': 's', 'ring': '^'}

fig = plt.figure(figsize=(7.2, 6.8))
gs  = gridspec.GridSpec(
    3, 3, figure=fig,
    left=0.10, right=0.97,
    top=0.88, bottom=0.08,
    hspace=0.55, wspace=0.40,
)

scan_params = [
    ('alpha', r'$\alpha$ (lag weight)',   ALPHA_VALS, res_alpha),
    ('beta',  r'$\beta$ (illum. weight)', BETA_VALS,  res_beta),
    ('J',     r'$J$ (mag. coupling)',     J_VALS,     res_J),
]
metrics = [
    ('corr',    r'Correlation $\rho$', True),
    ('overlap', 'Overlap',             True),
    ('lag_err', 'Lag error',           False),
]

for col, (pname, plabel, pvals, res_dict) in enumerate(scan_params):
    for row, (mname, mlabel, _) in enumerate(metrics):
        ax = fig.add_subplot(gs[row, col])
        for gtype in GEOM_CONFIGS:
            vals = np.array([r[mname] for r in res_dict[gtype]])
            ax.plot(
                pvals, vals,
                color=COLORS[gtype], marker=MARKERS[gtype],
                markersize=4, lw=1.1,
            )
        ref_val = {'alpha': ALPHA_REF, 'beta': BETA_REF, 'J': J_REF}[pname]
        ax.axvline(ref_val, color='k', ls='--', lw=0.8, alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(plabel, labelpad=2)
        if col == 0:
            ax.set_ylabel(mlabel, labelpad=2)
        if row == 0:
            ax.set_title(f'Varying {plabel}', pad=4)
        ax.tick_params(labelsize=7, pad=2)
        ax.yaxis.grid(True, lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        if mname in ('corr', 'overlap'):
            ax.set_ylim(0, 1.05)
            # NESSUNA linea threshold

handles = [
    plt.Line2D(
        [0], [0], color=COLORS[g], marker=MARKERS[g],
        markersize=4, lw=1.1, label=LABELS[g],
    )
    for g in GEOM_CONFIGS
]
handles.append(
    plt.Line2D(
        [0], [0], color='k', ls='--', lw=0.8, alpha=0.5,
        label='Reference (recovery)',
    )
)
fig.legend(
    handles=handles, loc='upper center', ncol=4,
    fontsize=7.5, framealpha=0.85,
    bbox_to_anchor=(0.53, 0.97),
)

fig.suptitle(
    'HAMCOR hyperparameter sensitivity '
    r'(same grid/geometries/$\alpha,\beta,J$ as recovery)',
    fontsize=9.5, y=1.0,
)

fig.savefig(
    'results/fig_sensitivity.pdf',
    dpi=300, bbox_inches='tight', facecolor='white',
)
plt.close(fig)
print("Saved: results/fig_sensitivity.pdf")

# ── summary al punto di riferimento ───────────────────────────────────────────
print()
print("=" * 60)
print("SENSITIVITY — reference point (must match recovery ρ)")
print("=" * 60)
print(f"  alpha={ALPHA_REF}, beta={BETA_REF}, J={J_REF}")
print()
alpha_ref_idx = int(np.argmin(np.abs(ALPHA_VALS - ALPHA_REF)))
for gtype in GEOM_CONFIGS:
    r = res_alpha[gtype][alpha_ref_idx]
    print(
        f"  {LABELS[gtype]:<12}: "
        f"corr={r['corr']:+.3f}  overlap={r['overlap']:.3f}  "
        f"lag_err={r['lag_err']*100:.1f}%"
    )
print("  Target recovery: lamppost~0.24, column~0.50, ring~0.12")
print("=" * 60)