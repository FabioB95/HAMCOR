"""
make_figure_v3.py  —  Paper-ready recovery figure for HAMCOR.

FIXED vs original:
  - Loads w_true and w_rec directly from recovery_results.npz
    (saved by run_recovery.py) instead of re-running the optimiser.
    This guarantees figure ρ values match Table 1.
  - mathtext.fontset = dejavusans (eliminates cmsy10 warnings)
  - colorbar compact, 3 ticks
  - chi2 in scientific notation
  - psi(t) x-axis zoomed to signal
  - output: results/fig_recovery_v3.pdf

Usage:
    python make_figure_v3.py
    (run AFTER run_recovery.py so recovery_results.npz exists)
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.data.synthetic import (
    create_synthetic_geometry, emissivity_correlation, emissivity_overlap,
)
from qcorona.physics.illumination import compute_solid_angles
from qcorona.physics.lags import compute_lag_matrix, compute_transfer_function_full

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          9,
    'axes.labelsize':     9,
    'axes.titlesize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'figure.dpi':         300,
    'text.usetex':        False,
    'mathtext.fontset':   'dejavusans',
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'lines.linewidth':    1.0,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

COLORS = {'lamppost': '#2166ac', 'column': '#d6604d', 'ring': '#4dac26'}
LABELS = {'lamppost': 'Lamp-post', 'column': 'Column', 'ring': 'Ring'}
GEOM_KWARGS = {
    'lamppost': dict(height=5.0, width=2.0),
    'column':   dict(z_min=2.0, z_max=12.0, radius=3.0),
    'ring':     dict(R_center=8.0, z_center=4.0, R_width=3.0, z_width=2.0),
}
N_BINS_TF  = 80
GEOM_ORDER = ['lamppost', 'column', 'ring']

# ── load ALL data from npz (no re-optimisation) ──────────────────────────────
print("Loading recovery_results.npz...")
npz_path = 'results/recovery_results.npz'
if not os.path.exists(npz_path):
    print(f"ERROR: {npz_path} not found. Run run_recovery.py first.")
    sys.exit(1)

d = np.load(npz_path, allow_pickle=False)

# Diagnostic: print all keys in npz
print(f"  NPZ keys: {sorted(d.files)[:10]} ...")

# Check if emissivity arrays are present
has_arrays = f'lamppost_w_true' in d.files
if not has_arrays:
    print("  WARNING: emissivity arrays not found in npz.")
    print("  Keys present:", [k for k in d.files if 'lamp' in k])
    sys.exit(1)

metrics = {}
for g in GEOM_ORDER:
    m = {k: d[f'{g}_{k}'] for k in
         ('corr', 'overlap', 'lag_err', 'illum_err', 'cent_err',
          'H_final', 'n_iter')}
    m['best_idx'] = int(np.argmin(m['H_final']))
    metrics[g] = m

# ── build grid + disc (same as run_recovery.py) ──────────────────────────────
print("Building grids...")

GRID = build_grid(n_r=10, n_z=10, n_phi=8,
                  r_in=3.0, r_out=15.0,
                  z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
LM   = compute_lag_matrix(GRID, DISK, observer_inclination=30.0)

# ── load emissivity arrays + compute transfer functions ──────────────────────
geom_data = {}
for gtype in GEOM_ORDER:
    print(f"  {gtype}...", end=' ', flush=True)

    # Load saved emissivity arrays
    w_true = d[f'{gtype}_w_true']
    w_rec  = d[f'{gtype}_w_rec']

    # Compute transfer functions
    tf_true = compute_transfer_function_full(
        GRID, DISK, w_true, SA,
        observer_inclination=30.0, n_bins=N_BINS_TF, lag_matrix=LM,
    )
    tf_rec = compute_transfer_function_full(
        GRID, DISK, w_rec, SA,
        observer_inclination=30.0,
        n_bins=N_BINS_TF,
        t_min=float(tf_true.t_bins[0]),
        t_max=float(tf_true.t_bins[-1]),
        lag_matrix=LM,
    )

    # Recompute correlation from saved arrays (must match Table 1)
    corr = emissivity_correlation(w_true, w_rec)
    m    = metrics[gtype]
    b    = m['best_idx']
    print(f"corr={corr:+.3f}  (npz best={m['corr'][b]:+.3f})  H={m['H_final'][b]:.4f}")

    geom_data[gtype] = dict(
        w_true    = w_true,
        w_rec     = w_rec,
        psi_true  = tf_true.psi,
        psi_rec   = tf_rec.psi,
        t_centers = tf_true.t_centers,
        history_H = d[f'{gtype}_history_H'],
        corr      = corr,
    )

# ── figure ────────────────────────────────────────────────────────────────────
print("Building figure...")

def rz_map(w):
    return w.reshape(GRID.n_r, GRID.n_z, GRID.n_phi).sum(axis=2)

fig = plt.figure(figsize=(7.2, 5.8))
gs  = gridspec.GridSpec(
    3, 4, figure=fig,
    left=0.08, right=0.96, top=0.91, bottom=0.09,
    hspace=0.56, wspace=0.52,
    width_ratios=[1, 1.05, 1.1, 0.85],
)

R_c = np.unique(np.round(GRID.R, 8))[:GRID.n_r]
z_c = np.unique(np.round(GRID.z, 8))[:GRID.n_z]

for row, gtype in enumerate(GEOM_ORDER):
    gd  = geom_data[gtype]
    m   = metrics[gtype]
    c   = COLORS[gtype]
    lbl = LABELS[gtype]
    b   = m['best_idx']

    # ── col 0: convergence ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[row, 0])
    H   = gd['history_H']
    itr = np.arange(len(H))
    ax0.semilogy(itr, np.abs(H), color=c, lw=1.2)
    if (H < 0).any():
        ax0.semilogy(itr[H < 0], np.abs(H[H < 0]),
                     color=c, lw=1.2, ls=':', alpha=0.65)
    ax0.set_xlabel('Iteration', labelpad=2)
    ax0.set_ylabel(r'$|\mathcal{H}|$', labelpad=2)
    if row == 0:
        ax0.set_title('Convergence', pad=4)
    ax0.text(0.97, 0.95, lbl, transform=ax0.transAxes,
             ha='right', va='top', fontsize=7.5, color=c, fontstyle='italic')
    ax0.tick_params(labelsize=7, pad=2)
    ax0.yaxis.set_minor_locator(LogLocator(subs='all', numticks=8))
    ax0.yaxis.set_minor_formatter(NullFormatter())

    # ── col 1: R-z map ───────────────────────────────────────────────────
    ax1  = fig.add_subplot(gs[row, 1])
    mt   = rz_map(gd['w_true'])
    mr   = rz_map(gd['w_rec'])
    vmax = max(mt.max(), mr.max())

    im = ax1.pcolormesh(R_c, z_c, mr.T,
                        cmap='Blues', vmin=0, vmax=vmax,
                        shading='auto', rasterized=True)
    ax1.contour(R_c, z_c, mt.T,
                levels=4, colors='k', linewidths=0.6, alpha=0.8)
    ax1.set_xlabel(r'$R\;[r_g]$', labelpad=2)
    ax1.set_ylabel(r'$z\;[r_g]$', labelpad=2)
    if row == 0:
        ax1.set_title('True (contours)\nRecovered (fill)', pad=4)
    ax1.tick_params(labelsize=7, pad=2)

    cb = fig.colorbar(im, ax=ax1, fraction=0.038, pad=0.02, shrink=0.80)
    cb.ax.tick_params(labelsize=5.5, pad=1)
    cb.set_label(r'$w_i$', fontsize=6.5, labelpad=1)
    cb.ax.yaxis.set_major_locator(MaxNLocator(3))

    # ρ annotation uses value computed from saved arrays → matches Table 1
    ax1.text(0.96, 0.05, fr'$\rho={gd["corr"]:+.2f}$',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=7.5,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    # ── col 2: ψ(t) ──────────────────────────────────────────────────────
    ax2  = fig.add_subplot(gs[row, 2])
    tc   = gd['t_centers']
    pobs = gd['psi_true']
    prec = gd['psi_rec']

    ax2.fill_between(tc, pobs, alpha=0.18, color=c)
    ax2.plot(tc, pobs, color=c, lw=1.3, ls='-',
             label=r'$\psi_{\rm true}$')
    ax2.plot(tc, prec, color=c, lw=1.0, ls='--', alpha=0.85,
             label=r'$\psi_{\rm rec}$')
    ax2.set_xlabel(r'$\Delta t\;[r_g/c]$', labelpad=2)
    ax2.set_ylabel(r'$\psi(t)$', labelpad=2)
    if row == 0:
        ax2.set_title(r'Transfer function $\psi(t)$', pad=4)
        ax2.legend(fontsize=6.5, loc='upper right',
                   framealpha=0.85, handlelength=1.4, borderpad=0.4)
    ax2.tick_params(labelsize=7, pad=2)
    ax2.set_ylim(bottom=0)

    # zoom x to signal
    pmax = max(pobs.max(), prec.max())
    sig  = np.where((pobs > 0.001 * pmax) | (prec > 0.001 * pmax))[0]
    if len(sig) > 1:
        ax2.set_xlim(max(0.0, tc[sig[0]] - 3), tc[sig[-1]] + 8)

    chi2     = float(np.sum((prec - pobs) ** 2))
    chi2_str = fr'$\chi^2={chi2:.2e}$'
    ax2.text(0.97, 0.93, chi2_str,
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=6.5, color='gray')

# ── col 3: summary bar chart ─────────────────────────────────────────────────
ax3   = fig.add_subplot(gs[:, 3])
keys  = ['corr', 'overlap']
klbls = [r'$\rho$', 'Overlap']
x     = np.arange(len(keys))
w     = 0.22
offs  = np.array([-1, 0, 1]) * w

for gi, gtype in enumerate(GEOM_ORDER):
    m    = metrics[gtype]
    b    = m['best_idx']
    vals = np.array([m[k][b] for k in keys])
    lo   = np.array([m[k].min() for k in keys])
    hi   = np.array([m[k].max() for k in keys])
    ax3.bar(x + offs[gi], vals, w,
            color=COLORS[gtype], alpha=0.85, label=LABELS[gtype],
            yerr=[vals - lo, hi - vals],
            error_kw=dict(elinewidth=0.8, capsize=2.5, ecolor='#222222'),
            zorder=3)

ax3.axhline(0.5, color='#333333', ls='--', lw=0.9, alpha=0.5, zorder=2)
ax3.text(x[-1] + offs[-1] + w * 0.6, 0.5, 'threshold',
         va='bottom', ha='left', fontsize=5.8, color='#555555')

ax3.set_xticks(x)
ax3.set_xticklabels(klbls, fontsize=8.5)
ax3.set_ylim(0, 1.08)
ax3.set_ylabel('Score', labelpad=3)
ax3.set_title('Recovery\nsummary', pad=4)
ax3.legend(fontsize=6.5, loc='upper left', framealpha=0.85,
           handlelength=1.1, borderpad=0.4, handletextpad=0.5)
ax3.tick_params(labelsize=7, pad=2)
ax3.yaxis.grid(True, lw=0.4, alpha=0.4, zorder=0)
ax3.set_axisbelow(True)

fig.suptitle('HAMCOR: cold-start recovery on synthetic geometries',
             fontsize=9.5, y=0.975)

# ── save ─────────────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
out = 'results/fig_recovery_v3.pdf'
fig.savefig(out, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {out}")

# ── summary: verify ρ values match table ─────────────────────────────────────
print()
print('=' * 55)
print(f"{'Geometry':<12} {'rho (figure)':>14} {'rho (npz best)':>15}")
print('-' * 55)
for g in GEOM_ORDER:
    m = metrics[g]; b = m['best_idx']
    print(f"{LABELS[g]:<12} {geom_data[g]['corr']:>+14.3f} {m['corr'][b]:>+15.3f}")
print('=' * 55)
print("Figure rho values should now match Table 1.")
