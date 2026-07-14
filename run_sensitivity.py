"""
run_sensitivity.py  —  Analisi di sensitività dei iperparametri QCORONA.

Testa come la qualità del recovery (correlazione, overlap, lag_err)
varia al variare di alpha, beta, J su geometrie sintetiche note.

Output:
    results/sensitivity_results.npz
    results/fig_sensitivity.png
    results/fig_sensitivity.tex

Usage:
    python run_sensitivity.py
"""

import sys, os, time
import numpy as np
import itertools

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
from qcorona.physics.lags import compute_lag_matrix, compute_transfer_function_full

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, NullFormatter

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.linewidth':    0.6,
    'lines.linewidth':   1.0,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'mathtext.fontset':  'dejavusans',
})

# ── grids (shared across all runs) ───────────────────────────────────────────
print("Building grids...")
GRID = build_grid(n_r=10, n_z=10, n_phi=8)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
LM   = compute_lag_matrix(GRID, DISK, observer_inclination=30.0)
N_BINS_TF = 80
print(f"  Corona: {GRID.n_cells} cells  |  Disk: {DISK.n_elements} elements")

# ── synthetic geometries to test ─────────────────────────────────────────────
GEOM_CONFIGS = {
    'lamppost': dict(height=5.0, width=2.0),
    'column':   dict(z_min=2.0, z_max=12.0, radius=3.0),
    'ring':     dict(R_center=8.0, z_center=4.0, R_width=3.0, z_width=2.0),
}

# Pre-compute true geometries and psi_obs
print("Pre-computing synthetic geometries...")
GEOM_DATA = {}
for gtype, kwargs in GEOM_CONFIGS.items():
    syn = create_synthetic_geometry(GRID, DISK, gtype, **kwargs)
    R_obs = syn.I_true / (1 - syn.I_true + 1e-10)
    tf_true = compute_transfer_function_full(
        GRID, DISK, syn.emissivity, SA,
        observer_inclination=30.0, n_bins=N_BINS_TF, lag_matrix=LM,
    )
    GEOM_DATA[gtype] = dict(
        synthetic=syn, R_obs=R_obs,
        psi_obs=tf_true.psi, t_bins=tf_true.t_bins,
    )
    print(f"  {gtype}: lag={syn.delta_t_true:.2f} r_g/c, I={syn.I_true:.3f}")

# ── parameter grids ──────────────────────────────────────────────────────────
# Reference values (from validated recovery)
ALPHA_REF = 20.0
BETA_REF  = 10.0
J_REF     = 0.5

# Scan ranges (log-spaced)
ALPHA_VALS = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
BETA_VALS  = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
J_VALS     = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0])

MAX_ITER = 500   # faster for sensitivity scan
N_SEEDS  = 3    # fewer seeds per point

OPT_CFG = OptimizationConfig(
    max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
    use_adaptive_lr=True, use_continuation=True,
    continuation_steps=100, convergence_tol=1e-7, verbose=False,
)


def make_hamiltonian(gtype, alpha, beta, J):
    """Create Hamiltonian for given geometry and hyperparameters."""
    gd = GEOM_DATA[gtype]
    syn = gd['synthetic']
    constraints = ObservationalConstraints(
        delta_t_obs          = syn.delta_t_true,
        R_obs                = gd['R_obs'],
        L_X                  = 1e43,
        M_bh                 = 1e8,
        eddington_ratio      = 0.1,
        observer_inclination = 30.0,
        source_name          = f"synthetic_{gtype}",
        psi_obs              = gd['psi_obs'],
        t_bins_obs           = gd['t_bins'],
    )
    h_config = HamiltonianConfig(
        J=J, alpha=alpha, beta=beta,
        gamma=1.0, l_crit=1000.0,
        delta=1.0, f_max=1.0,
        use_pair_barrier=False,
        use_energy_barrier=False,
    )
    return QCORONAHamiltonian(GRID, DISK, constraints, h_config)


def run_single(gtype, alpha, beta, J):
    """Run optimization and return metrics."""
    H_obj = make_hamiltonian(gtype, alpha, beta, J)
    syn   = GEOM_DATA[gtype]['synthetic']

    best_corr, best_H = -1.0, np.inf
    best_w = None

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed + 100)
        w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
        s0  = CoronalState(
            emissivity=w0,
            orientations=vertical_orientations(GRID.n_cells),
        )
        opt = QCORONAOptimizer(H_obj, OPT_CFG)
        res = opt.optimize(initial_state=s0)
        if res.final_H.H_total < best_H:
            best_H = res.final_H.H_total
            best_w = res.final_state.emissivity
            best_diag = res.final_H.diagnostics

    corr     = float(emissivity_correlation(syn.emissivity, best_w))
    overlap  = float(emissivity_overlap(syn.emissivity, best_w))
    cent_err = float(centroid_distance(GRID, syn.emissivity, best_w))
    lag_err  = abs(best_diag['delta_t_pred'] - syn.delta_t_true) / (syn.delta_t_true + 1e-10)
    illum_err= abs(best_diag['I_pred'] - syn.I_true) / (syn.I_true + 1e-10)

    return dict(
        corr=corr, overlap=overlap, cent_err=cent_err,
        lag_err=lag_err, illum_err=illum_err, H_final=best_H,
    )


# ── scan alpha (beta=ref, J=ref) ─────────────────────────────────────────────
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
    print(f"{time.time()-t0:.0f}s")

# ── scan beta (alpha=ref, J=ref) ─────────────────────────────────────────────
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

# ── scan J (alpha=ref, beta=ref) ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Scanning J: {J_VALS}  (alpha={ALPHA_REF}, beta={BETA_REF})")
print(f"{'='*55}")

res_J = {g: [] for g in GEOM_CONFIGS}
for J in J_VALS:
    print(f"  J={J:.2f} ... ", end='', flush=True)
    t0 = time.time()
    for gtype in GEOM_CONFIGS:
        r = run_single(gtype, ALPHA_REF, BETA_REF, J)
        res_J[gtype].append(r)
    print(f"{time.time()-t0:.0f}s")

# ── save raw data ─────────────────────────────────────────────────────────────
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
            [r[metric] for r in res_alpha[gtype]])
        save_dict[f'beta_{gtype}_{metric}']  = np.array(
            [r[metric] for r in res_beta[gtype]])
        save_dict[f'J_{gtype}_{metric}']     = np.array(
            [r[metric] for r in res_J[gtype]])

np.savez('results/sensitivity_results.npz', **save_dict)
print("\nSaved: results/sensitivity_results.npz")

# ── figure ────────────────────────────────────────────────────────────────────
COLORS = {'lamppost': '#2166ac', 'column': '#d6604d', 'ring': '#4dac26'}
LABELS = {'lamppost': 'Lamp-post', 'column': 'Column', 'ring': 'Ring'}
MARKERS= {'lamppost': 'o', 'column': 's', 'ring': '^'}

fig = plt.figure(figsize=(7.2, 6.5))
gs  = gridspec.GridSpec(
    3, 3, figure=fig,
    left=0.10, right=0.97,
    top=0.93,  bottom=0.09,
    hspace=0.55, wspace=0.40,
)

scan_params = [
    ('alpha', r'$\alpha$ (lag weight)',    ALPHA_VALS, res_alpha),
    ('beta',  r'$\beta$ (illum. weight)',  BETA_VALS,  res_beta),
    ('J',     r'$J$ (mag. coupling)',      J_VALS,     res_J),
]
metrics = [
    ('corr',     r'Correlation $\rho$',   True),
    ('overlap',  'Overlap',               True),
    ('lag_err',  'Lag error',             False),
]

for col, (pname, plabel, pvals, res_dict) in enumerate(scan_params):
    for row, (mname, mlabel, higher_is_better) in enumerate(metrics):
        ax = fig.add_subplot(gs[row, col])

        for gtype in GEOM_CONFIGS:
            vals = np.array([r[mname] for r in res_dict[gtype]])
            ax.plot(pvals, vals,
                    color=COLORS[gtype], marker=MARKERS[gtype],
                    markersize=4, lw=1.1,
                    label=LABELS[gtype] if row == 0 and col == 0 else '')

        # Reference line
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

        # Threshold line for correlation/overlap
        if mname in ('corr', 'overlap'):
            ax.axhline(0.5, color='gray', ls=':', lw=0.7, alpha=0.6)
            ax.set_ylim(0, 1.05)

# Legend
handles = [
    plt.Line2D([0], [0], color=COLORS[g], marker=MARKERS[g],
               markersize=4, lw=1.1, label=LABELS[g])
    for g in GEOM_CONFIGS
]
handles.append(plt.Line2D([0], [0], color='k', ls='--', lw=0.8,
                           alpha=0.5, label='Reference value'))
fig.legend(handles=handles, loc='upper center', ncol=4,
           fontsize=7.5, framealpha=0.85,
           bbox_to_anchor=(0.53, 0.975))

fig.suptitle('HAMCOR hyperparameter sensitivity analysis',
             fontsize=9.5, y=1.01)

png_path = 'results/fig_sensitivity.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {png_path}")

# ── pgfplots LaTeX ────────────────────────────────────────────────────────────
def coords(xs, ys):
    return ' '.join(f'({float(x):.4g},{float(y):.4f})' for x, y in zip(xs, ys))

color_tex = {
    'lamppost': 'blue!70!black',
    'column':   'red!60!black',
    'ring':     'green!55!black',
}
marker_tex = {'lamppost': '*', 'column': 'square*', 'ring': 'triangle*'}

tex = [
    r'% HAMCOR sensitivity analysis — pgfplots source',
    r'% \usepackage{pgfplots}',
    r'% \usepgfplotslibrary{groupplots}',
    r'% \pgfplotsset{compat=1.18}',
    r'',
    r'\begin{tikzpicture}',
    r'\begin{groupplot}[',
    r'  group style={group size=3 by 3,',
    r'               horizontal sep=1.6cm, vertical sep=1.4cm},',
    r'  width=4.2cm, height=3.0cm,',
    r'  xmode=log,',
    r'  tick label style={font=\tiny},',
    r'  label style={font=\scriptsize},',
    r'  ymajorgrids=true, grid style={dotted,thin},',
    r']',
    r'',
]

for col, (pname, plabel, pvals, res_dict) in enumerate(scan_params):
    ref_val = {'alpha': ALPHA_REF, 'beta': BETA_REF, 'J': J_REF}[pname]
    for row, (mname, mlabel, _) in enumerate(metrics):
        ylabel = mlabel if col == 0 else ''
        tex += [
            f'% {plabel} vs {mlabel}',
            r'\nextgroupplot[',
            f'  xlabel={{{plabel}}},',
            f'  ylabel={{{ylabel}}},',
            f'  title={{{plabel}}},',
            r']',
            f'\\addplot[dashed,black,thin] coordinates {{({pvals[0]:.3g},0) ({pvals[-1]:.3g},0)}};',
        ]
        for gtype in GEOM_CONFIGS:
            vals = np.array([r[mname] for r in res_dict[gtype]])
            tex += [
                r'\addplot[thick,' + color_tex[gtype] +
                r',mark=' + marker_tex[gtype] + r',mark size=1pt] coordinates {',
                '  ' + coords(pvals, vals),
                r'};',
            ]
        tex.append('')

tex += [r'\end{groupplot}', r'\end{tikzpicture}']

tex_path = 'results/fig_sensitivity.tex'
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(tex))
print(f"Saved: {tex_path}")

# ── console summary ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SENSITIVITY SUMMARY — reference point performance")
print("=" * 60)
print(f"  alpha={ALPHA_REF}, beta={BETA_REF}, J={J_REF}")
print()
for gtype in GEOM_CONFIGS:
    # Find reference index
    alpha_ref_idx = np.argmin(np.abs(ALPHA_VALS - ALPHA_REF))
    r = res_alpha[gtype][alpha_ref_idx]
    print(f"  {LABELS[gtype]:<12}: "
          f"corr={r['corr']:+.3f}  overlap={r['overlap']:.3f}  "
          f"lag_err={r['lag_err']*100:.1f}%")
print("=" * 60)
