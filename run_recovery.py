"""
run_recovery.py  —  Cold-start recovery test for all 3 synthetic geometries.

Usage:
    python run_recovery.py

Outputs (in results/):
    recovery_results.npz      raw data
    fig_recovery.png          paper-ready figure (300 dpi)
    fig_recovery.tex          pgfplots LaTeX source
"""

import sys, os, time
import numpy as np

sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import (
    QCORONAHamiltonian, HamiltonianConfig, ObservationalConstraints, CoronalState,
)
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.data.synthetic import (
    create_synthetic_geometry, emissivity_correlation,
    emissivity_overlap, centroid_distance,
)
from qcorona.hamiltonian.terms import vertical_orientations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── paper style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        9,
    'axes.labelsize':   9,
    'axes.titlesize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'figure.dpi':       300,
    'text.usetex':      False,          # set True if TeX installed
    'axes.linewidth':   0.6,
    'xtick.major.width':0.6,
    'ytick.major.width':0.6,
    'lines.linewidth':  1.0,
})

GEOM_STYLES = {
    'lamppost': dict(color='#2166ac', marker='o', label='Lamp-post'),
    'column':   dict(color='#d6604d', marker='s', label='Column'),
    'ring':     dict(color='#4dac26', marker='^', label='Ring'),
}

# ── grids ─────────────────────────────────────────────────────────────────────
print("Building grids...")
GRID = build_grid(n_r=10, n_z=10, n_phi=8,
                  r_in=3.0, r_out=15.0,
                  z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
print(f"  Corona: {GRID.n_cells} cells   Disk: {DISK.n_elements} elements")

GEOM_KWARGS = {
    'lamppost': dict(height=2.5,  width=0.8),   # h=2.5 ben dentro z_max=5
    'column':   dict(z_min=0.5,   z_max=4.5, radius=5.0),  # dentro il grid
    'ring':     dict(R_center=8.0, z_center=2.0, R_width=2.0, z_width=1.0),
}

N_STARTS  = 5      # multi-start per ogni geometria
MAX_ITER  = 800
N_SEEDS   = list(range(N_STARTS))

# ── run ───────────────────────────────────────────────────────────────────────

def make_hamiltonian(synthetic):
    R_obs = synthetic.I_true / (1 - synthetic.I_true + 1e-10)
    constraints = ObservationalConstraints(
        delta_t_obs      = synthetic.delta_t_true,
        R_obs            = R_obs,
        L_X              = 1e43,
        M_bh             = 1e8,
        eddington_ratio  = 0.1,
        observer_inclination = 30.0,
        source_name      = f"synthetic_{synthetic.name}",
    )
    h_config = HamiltonianConfig(
        J=0.05, alpha=10.0, beta=10.0,
        gamma=1.0, l_crit=1000.0,
        delta=1.0, f_max=1.0,
        use_pair_barrier=False,
        use_energy_barrier=False,
    )
    return QCORONAHamiltonian(GRID, DISK, constraints, h_config)


def run_cold_start(geometry_type, seed):
    synthetic = create_synthetic_geometry(GRID, DISK, geometry_type,
                                          **GEOM_KWARGS[geometry_type])
    H_obj = make_hamiltonian(synthetic)

    # cold start: pure random init
    rng = np.random.default_rng(seed)
    w_init = rng.exponential(size=GRID.n_cells)
    w_init /= w_init.sum()
    state0 = CoronalState(
        emissivity   = w_init,
        orientations = vertical_orientations(GRID.n_cells),
    )

    opt_cfg = OptimizationConfig(
        max_iter          = MAX_ITER,
        lr_emissivity     = 0.01,
        lr_orientation    = 0.01,
        use_adaptive_lr   = True,
        use_continuation  = True,
        continuation_steps= 200,
        convergence_tol   = 1e-7,
        verbose           = False,
    )
    opt = QCORONAOptimizer(H_obj, opt_cfg)
    t0  = time.time()
    res = opt.optimize(initial_state=state0)
    elapsed = time.time() - t0

    w_true = synthetic.emissivity
    w_rec  = res.final_state.emissivity

    corr      = emissivity_correlation(w_true, w_rec)
    overlap   = emissivity_overlap(w_true, w_rec)
    cent_err  = centroid_distance(GRID, w_true, w_rec)

    d = res.final_H.diagnostics
    lag_err   = abs(d['delta_t_pred'] - synthetic.delta_t_true) / (synthetic.delta_t_true + 1e-10)
    illum_err = abs(d['I_pred']       - synthetic.I_true)        / (synthetic.I_true + 1e-10)

    return dict(
        geometry     = geometry_type,
        seed         = seed,
        corr         = corr,
        overlap      = overlap,
        cent_err     = cent_err,
        lag_err      = lag_err,
        illum_err    = illum_err,
        H_final      = res.final_H.H_total,
        n_iter       = res.n_iterations,
        converged    = res.converged,
        elapsed      = elapsed,
        history_H    = np.array([h['H_total']    for h in res.history]),
        history_lag  = np.array([h['delta_t_pred'] for h in res.history]),
        history_illum= np.array([h['I_pred']     for h in res.history]),
        delta_t_true = synthetic.delta_t_true,
        I_true       = synthetic.I_true,
        w_true       = synthetic.emissivity,
        w_rec        = res.final_state.emissivity,
    )


all_runs = {}
for gtype in ['lamppost', 'column', 'ring']:
    print(f"\n{'─'*50}")
    print(f"Geometry: {gtype}  ({N_STARTS} cold starts, {MAX_ITER} iter each)")
    runs = []
    for seed in N_SEEDS:
        print(f"  seed={seed} ... ", end='', flush=True)
        r = run_cold_start(gtype, seed)
        runs.append(r)
        print(f"corr={r['corr']:+.3f}  lag_err={r['lag_err']*100:.1f}%  "
              f"H={r['H_final']:.4f}  ({r['elapsed']:.1f}s)")
    all_runs[gtype] = runs

# best run per geometry (lowest H_final)
best = {g: min(runs, key=lambda r: r['H_final']) for g, runs in all_runs.items()}

# ── save raw data ─────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
np.savez('results/recovery_results.npz', **{
    f"{g}_{k}": np.array([r[k] for r in runs])
    for g, runs in all_runs.items()
    for k in ('corr','overlap','cent_err','lag_err','illum_err','H_final','n_iter')
})
print("\nSaved: results/recovery_results.npz")

# ── figure ────────────────────────────────────────────────────────────────────
# Layout: 3 rows × 4 cols
#   col 0: H(iter) convergence
#   col 1: emissivity true vs recovered (R-z projection, 2D heatmap)
#   col 2: lag prediction vs true over iterations
#   col 3: summary bar chart (corr, overlap per geometry)

fig = plt.figure(figsize=(7.0, 5.5))   # fits in a two-column paper
gs  = gridspec.GridSpec(
    3, 4, figure=fig,
    left=0.08, right=0.97,
    top=0.93,  bottom=0.10,
    hspace=0.55, wspace=0.45,
)

geom_labels = {'lamppost': 'Lamp-post', 'column': 'Column', 'ring': 'Ring'}

# ── helper: R-z emissivity map ────────────────────────────────────────────────
def emissivity_rz_map(w, grid, n_r, n_z):
    """Marginalise over phi → (n_r, n_z) map."""
    n_phi = grid.n_phi
    W = w.reshape(n_r, n_z, n_phi)
    return W.sum(axis=2)           # sum over phi

for row, gtype in enumerate(['lamppost', 'column', 'ring']):
    style = GEOM_STYLES[gtype]
    runs  = all_runs[gtype]
    br    = best[gtype]

    # ── col 0: convergence ───────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[row, 0])
    for r in runs:
        iters = np.arange(len(r['history_H']))
        ax0.semilogy(iters, r['history_H'],
                     color=style['color'], alpha=0.25, lw=0.6)
    # best run highlighted
    iters_best = np.arange(len(br['history_H']))
    ax0.semilogy(iters_best, br['history_H'],
                 color=style['color'], lw=1.2, label='best')
    ax0.set_xlabel('Iteration')
    ax0.set_ylabel(r'$\mathcal{H}$')
    if row == 0:
        ax0.set_title('Convergence')
    ax0.text(0.97, 0.95, geom_labels[gtype],
             transform=ax0.transAxes, ha='right', va='top',
             fontsize=7, color=style['color'])
    ax0.tick_params(axis='both', labelsize=7)

    # ── col 1: R-z emissivity map ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[row, 1])
    nr, nz, nphi = GRID.n_r, GRID.n_z, GRID.n_phi

    map_true = emissivity_rz_map(br['w_true'], GRID, nr, nz)
    map_rec  = emissivity_rz_map(br['w_rec'],  GRID, nr, nz)

    # normalise together so colour scale is comparable
    vmax = max(map_true.max(), map_rec.max())

    # R edges and z edges (reconstruct from cell centres via log spacing)
    R_c = np.unique(np.round(GRID.R,  6))[:nr]
    z_c = np.unique(np.round(GRID.z,  6))[:nz]

    # show recovered as filled, true as contour overlay
    im = ax1.pcolormesh(
        R_c, z_c, map_rec.T,
        cmap='Blues', vmin=0, vmax=vmax, shading='auto', rasterized=True,
    )
    ax1.contour(
        R_c, z_c, map_true.T,
        levels=4, colors='k', linewidths=0.5, alpha=0.7,
    )
    ax1.set_xlabel(r'$R\;[r_g]$')
    ax1.set_ylabel(r'$z\;[r_g]$')
    if row == 0:
        ax1.set_title('True (contour) vs\nRecovered (fill)')
    ax1.tick_params(axis='both', labelsize=7)
    cb = plt.colorbar(im, ax=ax1, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    # annotate correlation
    ax1.text(0.97, 0.05,
             fr"$\rho={br['corr']:+.2f}$",
             transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=7, color='k')

    # ── col 2: lag & illumination over iterations ────────────────────────────
    ax2 = fig.add_subplot(gs[row, 2])
    iters = np.arange(len(br['history_lag']))
    ax2.plot(iters, br['history_lag'],
             color=style['color'], lw=1.0, label=r'$\Delta t_\mathrm{pred}$')
    ax2.axhline(br['delta_t_true'], color='k', ls='--', lw=0.8,
                label=r'$\Delta t_\mathrm{true}$')
    ax2_r = ax2.twinx()
    ax2_r.plot(iters, br['history_illum'],
               color=style['color'], ls=':', lw=1.0, alpha=0.7,
               label=r'$I_\mathrm{pred}$')
    ax2_r.axhline(br['I_true'], color='gray', ls=':', lw=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel(r'$\Delta t\;[r_g/c]$', labelpad=1)
    ax2_r.set_ylabel(r'$I$', labelpad=1)
    if row == 0:
        ax2.set_title('Observables')
    ax2.tick_params(axis='both', labelsize=7)
    ax2_r.tick_params(axis='both', labelsize=7)

# ── col 3: summary bar chart (all geometries together, last column) ──────────
ax3 = fig.add_subplot(gs[:, 3])
metrics   = ['corr', 'overlap']
m_labels  = [r'$\rho$', 'Overlap']
geom_list = ['lamppost', 'column', 'ring']

x      = np.arange(len(metrics))
width  = 0.22
offsets= np.array([-1, 0, 1]) * width

for gi, gtype in enumerate(geom_list):
    style = GEOM_STYLES[gtype]
    # best run values
    vals = [best[gtype][m] for m in metrics]
    # all-run spread (min/max)
    lo   = [min(r[m] for r in all_runs[gtype]) for m in metrics]
    hi   = [max(r[m] for r in all_runs[gtype]) for m in metrics]
    yerr_lo = np.array(vals) - np.array(lo)
    yerr_hi = np.array(hi)   - np.array(vals)
    ax3.bar(x + offsets[gi], vals, width,
            color=style['color'], alpha=0.85,
            label=geom_labels[gtype],
            yerr=[yerr_lo, yerr_hi],
            error_kw=dict(elinewidth=0.8, capsize=2, ecolor='k'))

ax3.axhline(0.5, color='k', ls='--', lw=0.7, alpha=0.5, label='threshold')
ax3.set_xticks(x)
ax3.set_xticklabels(m_labels, fontsize=8)
ax3.set_ylim(-0.15, 1.05)
ax3.set_ylabel('Score')
ax3.set_title('Recovery\nsummary')
ax3.legend(fontsize=6.5, loc='lower right', framealpha=0.7)
ax3.tick_params(axis='both', labelsize=7)

fig.suptitle(
    'HAMCOR: cold-start recovery on synthetic geometries',
    fontsize=9, fontweight='normal', y=0.97,
)

pdf_path = 'results/fig_recovery_v3.pdf'
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {pdf_path}")

# ── pgfplots LaTeX ────────────────────────────────────────────────────────────

def fmt_coords(xs, ys):
    return ' '.join(f'({x:.4f},{y:.6g})' for x, y in zip(xs, ys))

lines = [
    r'% ─────────────────────────────────────────────────────────',
    r'% QCORONA  –  synthetic recovery figure (pgfplots source)',
    r'% Generated by run_recovery.py',
    r'% Place inside \begin{figure}...\end{figure}',
    r'% ─────────────────────────────────────────────────────────',
    r'\begin{tikzpicture}',
    r'\begin{groupplot}[',
    r'  group style={group size=3 by 3,',
    r'               horizontal sep=1.6cm, vertical sep=1.4cm},',
    r'  width=4.2cm, height=3.2cm,',
    r'  tick label style={font=\scriptsize},',
    r'  label style={font=\scriptsize},',
    r'  title style={font=\scriptsize},',
    r']',
    r'',
]

color_map = {
    'lamppost': 'blue!70!black',
    'column':   'red!70!black',
    'ring':     'green!60!black',
}

for row, gtype in enumerate(['lamppost', 'column', 'ring']):
    br    = best[gtype]
    color = color_map[gtype]
    label = geom_labels[gtype]
    H_arr = br['history_H']
    lag_arr  = br['history_lag']
    illum_arr= br['history_illum']
    iters = np.arange(len(H_arr))

    # ── convergence panel ──────────────────────────────────────────────────
    lines += [
        r'% ── ' + label + r' · convergence ──',
        r'\nextgroupplot[',
        r'  xlabel={Iteration}, ylabel={$\mathcal{H}$},',
        '  title={' + label + '},',
        r'  ymode=log,',
        r'  xmin=0,',
        '  title style={text={' + color + '}},',
        r']',
        r'\addplot[thick,' + color + r'] coordinates {',
        '  ' + fmt_coords(iters, H_arr),
        r'};',
        r'',
    ]

    # grey traces for other starts
    for r in all_runs[gtype]:
        if r is br:
            continue
        hh = r['history_H']
        ii = np.arange(len(hh))
        lines += [
            r'\addplot[very thin,' + color + r',opacity=0.3] coordinates {',
            '  ' + fmt_coords(ii, hh),
            r'};',
        ]
    lines.append('')

    # ── lag panel ──────────────────────────────────────────────────────────
    lines += [
        r'% ── ' + label + r' · observables ──',
        r'\nextgroupplot[',
        r'  xlabel={Iteration},',
        r'  ylabel={$\Delta t\;[r_g/c]$},',
        r'  axis y line*=left,',
        r']',
        r'\addplot[thick,' + color + r'] coordinates {',
        '  ' + fmt_coords(iters, lag_arr),
        r'};',
        r'\addplot[dashed,black,thin] coordinates {',
        f'  (0,{br["delta_t_true"]:.4f}) ({len(iters)-1},{br["delta_t_true"]:.4f})',
        r'};',
        r'',
    ]

    # ── summary panel (just a node with text for now) ──────────────────────
    lines += [
        r'% ── ' + label + r' · metrics text ──',
        r'\nextgroupplot[',
        r'  hide axis,',
        r'  xmin=0,xmax=1,ymin=0,ymax=1,',
        r']',
        r'\node[anchor=north west, font=\scriptsize, align=left] at (axis cs:0.05,0.95) {',
        f'  \\textbf{{{label}}} (best of {N_STARTS})\\\\',
        f'  $\\rho = {br["corr"]:+.3f}$\\\\',
        f'  Overlap $= {br["overlap"]:.3f}$\\\\',
        f'  Centroid err. $= {br["cent_err"]:.2f}\\,r_g$\\\\',
        f'  Lag err. $= {br["lag_err"]*100:.1f}\\%$\\\\',
        f'  Illum. err. $= {br["illum_err"]*100:.1f}\\%$\\\\',
        f'  $N_{{\\rm iter}} = {br["n_iter"]}$',
        r'};',
        r'',
    ]

lines += [
    r'\end{groupplot}',
    r'\end{tikzpicture}',
    r'',
    r'% ─────────────────────────────────────────────────────────',
    r'% Packages required in preamble:',
    r'%   \usepackage{pgfplots}',
    r'%   \usepgfplotslibrary{groupplots}',
    r'%   \pgfplotsset{compat=1.18}',
    r'% ─────────────────────────────────────────────────────────',
]

tex_path = 'results/fig_recovery.tex'
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"Saved: {tex_path}")

# ── console summary table ──────────────────────────────────────────────────────
print()
print("="*68)
print(f"{'Geometry':<12} {'corr':>7} {'overlap':>8} {'cent[rg]':>10} "
      f"{'lag%':>7} {'illum%':>8} {'H_final':>10}")
print("─"*68)
for gtype in ['lamppost','column','ring']:
    b = best[gtype]
    print(f"{gtype:<12} {b['corr']:>+7.3f} {b['overlap']:>8.3f} "
          f"{b['cent_err']:>10.2f} {b['lag_err']*100:>7.1f} "
          f"{b['illum_err']*100:>8.1f} {b['H_final']:>10.4f}")
print("="*68)
print("Done.")