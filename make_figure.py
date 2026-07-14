"""
make_figure.py  —  Genera la figura paper-ready per il recovery test.

Legge results/recovery_results.npz + ricalcola ψ(t) dalle geometrie sintetiche.

Usage:
    python make_figure.py

Output:
    results/fig_recovery_v2.png    (300 dpi, paper-ready)
    results/fig_recovery_v2.tex    (pgfplots, coordinate punto per punto)
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, NullFormatter

sys.path.insert(0, '.')

# ── qcorona imports ──────────────────────────────────────────────────────────
from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.data.synthetic import create_synthetic_geometry
from qcorona.physics.illumination import compute_solid_angles
from qcorona.physics.lags import compute_lag_matrix, compute_transfer_function_full
from qcorona.hamiltonian.total import (
    QCORONAHamiltonian, HamiltonianConfig, ObservationalConstraints, CoronalState,
)
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.data.synthetic import emissivity_correlation, emissivity_overlap
from qcorona.hamiltonian.terms import vertical_orientations

# ── paper style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        300,
    'text.usetex':       False,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'lines.linewidth':   1.0,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'mathtext.fontset':  'dejavuserif',
})

COLORS = {
    'lamppost': '#2166ac',
    'column':   '#d6604d',
    'ring':     '#4dac26',
}
LABELS = {
    'lamppost': 'Lamp-post',
    'column':   'Column',
    'ring':     'Ring',
}
GEOM_KWARGS = {
    'lamppost': dict(height=5.0, width=2.0),
    'column':   dict(z_min=2.0, z_max=12.0, radius=3.0),
    'ring':     dict(R_center=8.0, z_center=4.0, R_width=3.0, z_width=2.0),
}
N_BINS_TF = 80

# ── load scalar metrics ──────────────────────────────────────────────────────
print("Loading recovery_results.npz...")
d = np.load('results/recovery_results.npz')

metrics = {}
for g in ['lamppost', 'column', 'ring']:
    metrics[g] = {
        'corr':      d[f'{g}_corr'],
        'overlap':   d[f'{g}_overlap'],
        'lag_err':   d[f'{g}_lag_err'],
        'illum_err': d[f'{g}_illum_err'],
        'cent_err':  d[f'{g}_cent_err'],
        'H_final':   d[f'{g}_H_final'],
        'n_iter':    d[f'{g}_n_iter'],
    }
    # best run = lowest |H_final|
    best_idx = np.argmin(np.abs(metrics[g]['H_final']))
    metrics[g]['best_idx'] = best_idx

# ── rebuild geometry + ψ(t) on the fly ──────────────────────────────────────
print("Building grids and computing transfer functions...")
GRID = build_grid(n_r=10, n_z=10, n_phi=8)
DISK = build_disk_grid(n_r=30, n_phi=18)
solid_angles = compute_solid_angles(GRID, DISK)
lag_matrix   = compute_lag_matrix(GRID, DISK, observer_inclination=30.0)

geom_data = {}
for gtype in ['lamppost', 'column', 'ring']:
    print(f"  {gtype}...")
    synthetic = create_synthetic_geometry(GRID, DISK, gtype, **GEOM_KWARGS[gtype])

    # ψ_true
    tf_true = compute_transfer_function_full(
        GRID, DISK, synthetic.emissivity, solid_angles,
        observer_inclination=30.0, n_bins=N_BINS_TF, lag_matrix=lag_matrix,
    )

    # Re-run best optimization to get w_recovered
    R_obs = synthetic.I_true / (1 - synthetic.I_true + 1e-10)
    constraints = ObservationalConstraints(
        delta_t_obs=synthetic.delta_t_true, R_obs=R_obs,
        L_X=1e43, M_bh=1e8, eddington_ratio=0.1,
        observer_inclination=30.0,
        source_name=f"synthetic_{gtype}",
        psi_obs=tf_true.psi, t_bins_obs=tf_true.t_bins,
    )
    h_config = HamiltonianConfig(
        J=0.5, alpha=20.0, beta=10.0,
        gamma=1.0, l_crit=1000.0, delta=1.0, f_max=1.0,
        use_pair_barrier=False, use_energy_barrier=False,
    )
    H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config)

    best_seed = int(metrics[gtype]['best_idx'])
    rng = np.random.default_rng(best_seed)
    w_init = rng.exponential(size=GRID.n_cells)
    w_init /= w_init.sum()
    state0 = CoronalState(
        emissivity=w_init,
        orientations=vertical_orientations(GRID.n_cells),
    )
    opt_cfg = OptimizationConfig(
        max_iter=800, lr_emissivity=0.01, lr_orientation=0.01,
        use_adaptive_lr=True, use_continuation=True,
        continuation_steps=200, convergence_tol=1e-7, verbose=False,
    )
    opt = QCORONAOptimizer(H_obj, opt_cfg)
    res = opt.optimize(initial_state=state0)
    w_rec = res.final_state.emissivity

    # ψ_recovered
    tf_rec = compute_transfer_function_full(
        GRID, DISK, w_rec, solid_angles,
        observer_inclination=30.0, n_bins=N_BINS_TF,
        t_min=float(tf_true.t_bins[0]), t_max=float(tf_true.t_bins[-1]),
        lag_matrix=lag_matrix,
    )

    # H history for convergence plot
    history_H = np.array([h['H_total'] for h in res.history])

    geom_data[gtype] = dict(
        synthetic=synthetic,
        w_true=synthetic.emissivity,
        w_rec=w_rec,
        psi_true=tf_true.psi,
        psi_rec=tf_rec.psi,
        t_centers=tf_true.t_centers,
        history_H=history_H,
    )
    corr = emissivity_correlation(synthetic.emissivity, w_rec)
    print(f"    corr={corr:+.3f}  H={res.final_H.H_total:.4f}")

print("Done. Building figure...")

# ── figure layout ─────────────────────────────────────────────────────────────
# 3 rows × 5 cols:
#   col 0: H(iter) convergence
#   col 1: R-z emissivity map (true contour + recovered fill)
#   col 2: ψ(t) true vs recovered  ← NEW
#   col 3: summary bar chart (spans all 3 rows)

fig = plt.figure(figsize=(7.2, 5.8))
gs = gridspec.GridSpec(
    3, 4, figure=fig,
    left=0.08, right=0.97,
    top=0.91,  bottom=0.09,
    hspace=0.55, wspace=0.50,
    width_ratios=[1, 1.1, 1.1, 0.9],
)

def emissivity_rz_map(w, grid):
    nr, nz, nphi = grid.n_r, grid.n_z, grid.n_phi
    return w.reshape(nr, nz, nphi).sum(axis=2)

geom_order = ['lamppost', 'column', 'ring']

for row, gtype in enumerate(geom_order):
    gd    = geom_data[gtype]
    m     = metrics[gtype]
    color = COLORS[gtype]
    label = LABELS[gtype]
    best  = int(m['best_idx'])

    # ── col 0: convergence ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[row, 0])
    H_arr = gd['history_H']
    H_pos = np.where(H_arr > 0, H_arr, np.nan)
    iters = np.arange(len(H_arr))
    ax0.semilogy(iters, np.abs(H_arr), color=color, lw=1.2)
    # mark negative H region with dotted line
    neg_mask = H_arr < 0
    if neg_mask.any():
        ax0.semilogy(iters[neg_mask], np.abs(H_arr[neg_mask]),
                     color=color, lw=1.2, ls=':', alpha=0.7)
    ax0.set_xlabel('Iteration', labelpad=2)
    ax0.set_ylabel(r'$|\mathcal{H}|$', labelpad=2)
    if row == 0:
        ax0.set_title('Convergence', pad=4)
    ax0.text(0.97, 0.95, label, transform=ax0.transAxes,
             ha='right', va='top', fontsize=7.5, color=color, fontstyle='italic')
    ax0.tick_params(axis='both', labelsize=7, pad=2)
    ax0.yaxis.set_minor_locator(LogLocator(subs='all', numticks=10))
    ax0.yaxis.set_minor_formatter(NullFormatter())

    # ── col 1: R-z emissivity map ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[row, 1])
    nr, nz = GRID.n_r, GRID.n_z

    map_true = emissivity_rz_map(gd['w_true'], GRID)
    map_rec  = emissivity_rz_map(gd['w_rec'],  GRID)
    vmax     = max(map_true.max(), map_rec.max())

    R_c = np.unique(np.round(GRID.R, 8))[:nr]
    z_c = np.unique(np.round(GRID.z, 8))[:nz]

    im = ax1.pcolormesh(
        R_c, z_c, map_rec.T,
        cmap='Blues', vmin=0, vmax=vmax,
        shading='auto', rasterized=True,
    )
    ax1.contour(
        R_c, z_c, map_true.T,
        levels=4, colors='k', linewidths=0.6, alpha=0.8,
    )
    ax1.set_xlabel(r'$R\;[r_g]$', labelpad=2)
    ax1.set_ylabel(r'$z\;[r_g]$', labelpad=2)
    if row == 0:
        ax1.set_title('True (contours)\nRecovered (fill)', pad=4)
    ax1.tick_params(axis='both', labelsize=7, pad=2)

    cb = plt.colorbar(im, ax=ax1, pad=0.03, fraction=0.046)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(r'$w_i$', fontsize=7, labelpad=2)

    corr_val = m['corr'][best]
    ax1.text(0.96, 0.05, fr'$\rho={corr_val:+.2f}$',
             transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=7.5, color='k',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    # ── col 2: ψ(t) true vs recovered ───────────────────────────────────
    ax2 = fig.add_subplot(gs[row, 2])
    tc   = gd['t_centers']
    pobs = gd['psi_true']
    prec = gd['psi_rec']

    ax2.fill_between(tc, pobs, alpha=0.20, color=color, label=r'$\psi_\mathrm{true}$')
    ax2.plot(tc, pobs, color=color, lw=1.2, ls='-')
    ax2.plot(tc, prec, color=color, lw=1.0, ls='--', alpha=0.85,
             label=r'$\psi_\mathrm{rec}$')
    ax2.set_xlabel(r'$\Delta t\;[r_g/c]$', labelpad=2)
    ax2.set_ylabel(r'$\psi(t)$', labelpad=2)
    if row == 0:
        ax2.set_title(r'Transfer function $\psi(t)$', pad=4)
        ax2.legend(fontsize=6.5, loc='upper right',
                   framealpha=0.8, handlelength=1.5)
    ax2.tick_params(axis='both', labelsize=7, pad=2)
    ax2.set_ylim(bottom=0)

    # χ² residual annotation
    chi2 = float(np.sum((prec - pobs)**2))
    ax2.text(0.97, 0.93, fr'$\chi^2={chi2:.4f}$',
             transform=ax2.transAxes, ha='right', va='top', fontsize=6.5,
             color='gray')

# ── col 3: summary bar chart (spans all 3 rows) ──────────────────────────────
ax3 = fig.add_subplot(gs[:, 3])

metric_keys  = ['corr', 'overlap']
metric_lbls  = [r'$\rho$', 'Overlap']
x     = np.arange(len(metric_keys))
width = 0.22
offs  = np.array([-1, 0, 1]) * width

for gi, gtype in enumerate(geom_order):
    m     = metrics[gtype]
    color = COLORS[gtype]
    best  = int(m['best_idx'])

    vals = np.array([m[k][best] for k in metric_keys])
    lo   = np.array([m[k].min() for k in metric_keys])
    hi   = np.array([m[k].max() for k in metric_keys])

    ax3.bar(
        x + offs[gi], vals, width,
        color=color, alpha=0.85, label=LABELS[gtype],
        yerr=[vals - lo, hi - vals],
        error_kw=dict(elinewidth=0.8, capsize=2.5, ecolor='#333333'),
        zorder=3,
    )

ax3.axhline(0.5, color='#555555', ls='--', lw=0.8, alpha=0.6, zorder=2,
            label='threshold')
ax3.set_xticks(x)
ax3.set_xticklabels(metric_lbls, fontsize=8.5)
ax3.set_ylim(0, 1.05)
ax3.set_ylabel('Score', labelpad=3)
ax3.set_title('Recovery\nsummary', pad=4)
ax3.legend(fontsize=6.5, loc='upper left', framealpha=0.8,
           handlelength=1.2, borderpad=0.5)
ax3.tick_params(axis='both', labelsize=7, pad=2)
ax3.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
ax3.set_axisbelow(True)

# ── suptitle ─────────────────────────────────────────────────────────────────
fig.suptitle(
    r'QCORONA: cold-start recovery on synthetic geometries',
    fontsize=9.5, y=0.975, fontweight='normal',
)

# ── save PNG ─────────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
png_path = 'results/fig_recovery_v2.png'
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {png_path}")

# ── pgfplots LaTeX ────────────────────────────────────────────────────────────
def coords(xs, ys):
    return ' '.join(f'({x:.4f},{y:.6g})' for x, y in zip(xs, ys))

color_tex = {
    'lamppost': 'blue!70!black',
    'column':   'red!60!black',
    'ring':     'green!55!black',
}

lines = [
    r'% QCORONA recovery figure — pgfplots source',
    r'% Packages: \usepackage{pgfplots,pgfplotscolortbl}',
    r'%           \usepgfplotslibrary{groupplots,fillbetween}',
    r'%           \pgfplotsset{compat=1.18}',
    r'',
    r'\begin{tikzpicture}',
    r'\begin{groupplot}[',
    r'  group style={group size=3 by 3,',
    r'               horizontal sep=1.8cm, vertical sep=1.5cm},',
    r'  width=4.0cm, height=3.0cm,',
    r'  tick label style={font=\tiny},',
    r'  label style={font=\scriptsize},',
    r'  title style={font=\scriptsize\itshape},',
    r'  axis line style={thin},',
    r'  every tick/.style={thin},',
    r']',
    r'',
]

for row, gtype in enumerate(geom_order):
    gd    = geom_data[gtype]
    color = color_tex[gtype]
    lbl   = LABELS[gtype]
    H_arr = gd['history_H']
    tc    = gd['t_centers']
    iters = np.arange(len(H_arr))

    # ── convergence ──────────────────────────────────────────────────────
    lines += [
        f'% -- {lbl}: convergence',
        r'\nextgroupplot[',
        r'  xlabel={Iteration}, ylabel={$|\mathcal{H}|$},',
        '  title={' + lbl + '},',
        r'  ymode=log, ymin=1e-6,',
        r']',
        r'\addplot[thick,' + color + r'] coordinates {',
        '  ' + coords(iters, np.abs(H_arr)),
        r'};',
        r'',
    ]

    # ── R-z map: skip in pgfplots (2D heatmap → use PNG inclusion) ───────
    lines += [
        f'% -- {lbl}: emissivity map (include as PNG in paper)',
        r'\nextgroupplot[',
        r'  hide axis,',
        r'  xmin=0,xmax=1, ymin=0,ymax=1,',
        r']',
        r'\node[anchor=center] at (axis cs:0.5,0.5) {',
        r'  % \includegraphics[width=\linewidth]{fig_rz_' + gtype + r'.pdf}',
        r'};',
        r'',
    ]

    # ── ψ(t) ─────────────────────────────────────────────────────────────
    lines += [
        f'% -- {lbl}: transfer function psi(t)',
        r'\nextgroupplot[',
        r'  xlabel={$\Delta t\;[r_g/c]$}, ylabel={$\psi(t)$},',
        r'  ymin=0,',
        r']',
        r'\addplot[name path=psi_true, thick,' + color + r', fill opacity=0.2]',
        r'  coordinates {' + coords(tc, gd['psi_true']) + r'} \closedcycle;',
        r'\addplot[name path=zero,' + color + r', draw=none]',
        r'  coordinates {' + coords([tc[0], tc[-1]], [0, 0]) + r'};',
        r'\addplot[' + color + r', fill opacity=0.15]',
        r'  fill between[of=psi_true and zero];',
        r'\addplot[dashed, thick,' + color + r', opacity=0.85] coordinates {',
        '  ' + coords(tc, gd['psi_rec']),
        r'};',
        r'',
    ]

lines += [
    r'\end{groupplot}',
    r'',
    r'% Summary bar chart — add separately as a standard pgfplots axis',
    r'% or use the PNG figure directly.',
    r'',
    r'\end{tikzpicture}',
]

tex_path = 'results/fig_recovery_v2.tex'
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"Saved: {tex_path}")

# ── console summary ───────────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"{'Geometry':<12} {'corr(best)':>11} {'overlap':>8} "
      f"{'cent[rg]':>9} {'lag%':>6} {'illum%':>7}")
print("-" * 65)
for gtype in geom_order:
    m    = metrics[gtype]
    best = int(m['best_idx'])
    print(f"{LABELS[gtype]:<12} {m['corr'][best]:>+11.3f} "
          f"{m['overlap'][best]:>8.3f} {m['cent_err'][best]:>9.2f} "
          f"{m['lag_err'][best]*100:>6.1f} {m['illum_err'][best]*100:>7.1f}")
print("=" * 65)
