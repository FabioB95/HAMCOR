"""
run_mrk335_multiepoch.py
────────────────────────
Apply HAMCOR to 5 XMM-Newton epochs of Mrk 335 at different flux states,
tracking how the coronal geometry evolves with accretion rate.

XMM-Newton observations of Mrk 335:
  Epoch 1: ObsID 0306870101  2006 Jan  HIGH flux  ← already done
  Epoch 2: ObsID 0655200201  2009 Aug  LOW flux   (deep minimum)
  Epoch 3: ObsID 0741280201  2013 Jun  INTER.     
  Epoch 4: ObsID 0741280401  2013 Dec  INTER.     
  Epoch 5: ObsID 0741280501  2013 Dec  INTER.     

Key science question:
  Does the corona contract/expand with flux state?
  Extended disc-corona in high state → compact in low state?

Reference for multi-epoch lags: Kara+2013b, Wilkins+2015

SAS reduction needed for epochs 2-5:
  See instructions at bottom of this file.
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
from qcorona.physics.lags import compute_lag_matrix, compute_transfer_function_full

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'dejavusans',
})

# ── Mrk 335 parameters ────────────────────────────────────────────────────────
M_BH    = 1.3e7    # M_sun
R_G_C   = 63.0     # s per r_g/c

# Multi-epoch lag measurements
# Fill in epochs 2-5 after SAS reduction + lag measurement
# Format: (obsid, year, flux_state, lag_s, lag_err_s, R_obs, L_X, incl)


EPOCHS = [
    {
        'obsid': '0306870101', 'year': 2006, 'state': 'High',
        'lag_s': 91.9, 'lag_err_s': 43.4,
        'R_obs': 0.3, 'L_X': 1.0e43, 'incl': 45.0,
        'data_dir': 'data/sources/mrk335', 'done': True,
    },
    {
        'obsid': '0600540601', 'year': 2009, 'state': 'Low',
        'lag_s': 6.0, 'lag_err_s': 6.0,   # use 1-sigma upper limit
        'R_obs': 2.0, 'L_X': 5.0e41, 'incl': 45.0,
        'data_dir': 'data/sources/mrk335_2009', 'done': True,
    },
    {
        'obsid': '0741280201', 'year': 2015, 'state': 'Intermediate',
        'lag_s': 19.5, 'lag_err_s': 8.7,
        'R_obs': 1.0, 'L_X': 3.0e42, 'incl': 45.0,
        'data_dir': 'data/sources/mrk335_2015', 'done': True,
    },
    {
        'obsid': '0780500301', 'year': 2018, 'state': 'Intermediate',
        'lag_s': 7.7, 'lag_err_s': 6.5,
        'R_obs': 0.8, 'L_X': 5.0e42, 'incl': 45.0,
        'data_dir': 'data/sources/mrk335_2018', 'done': True,
    },
    {
        'obsid': '0831790601', 'year': 2019, 'state': 'Recovering',
        'lag_s': 39.0, 'lag_err_s': 7.7,
        'R_obs': 0.5, 'L_X': 8.0e42, 'incl': 45.0,
        'data_dir': 'data/sources/mrk335_2019', 'done': True,
    },
]

N_STARTS = 5
MAX_ITER = 2000
N_BINS_TF = 80

def run_epoch(epoch):
    """Run HAMCOR on one epoch. Returns result dict or None if not ready."""
    if not epoch['done'] and epoch['lag_s'] is None:
        print(f"  ObsID {epoch['obsid']}: lag not yet measured. Skipping.")
        return None

    lag_rg  = epoch['lag_s']   / R_G_C
    lag_err = epoch['lag_err_s'] / R_G_C

    print(f"\n  ObsID {epoch['obsid']} ({epoch['year']}, {epoch['state']})")
    print(f"  lag = {epoch['lag_s']:.1f} +/- {epoch['lag_err_s']:.1f} s"
          f" = {lag_rg:.2f} +/- {lag_err:.2f} r_g/c")

    GRID = build_grid(n_r=10, n_z=10, n_phi=8,
                      r_in=3.0, r_out=15.0,
                      z_min=0.1, z_max=5.0)
    DISK = build_disk_grid(n_r=30, n_phi=18)
    SA   = compute_solid_angles(GRID, DISK)
    LM   = compute_lag_matrix(GRID, DISK,
                               observer_inclination=epoch['incl'])

    T_MIN  = max(0.0, float(LM.min()))
    T_MAX  = 15.0
    t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
    t_c    = 0.5 * (t_bins[:-1] + t_bins[1:])

    psi_obs = np.exp(-0.5 * ((t_c - lag_rg) / lag_err)**2)
    psi_obs = psi_obs / psi_obs.sum()

    constraints = ObservationalConstraints(
        delta_t_obs=lag_rg, R_obs=epoch['R_obs'],
        L_X=epoch['L_X'], M_bh=M_BH,
        eddington_ratio=epoch['L_X'] / (1.3e38 * M_BH),
        observer_inclination=epoch['incl'],
        source_name=f"Mrk335_{epoch['year']}",
        psi_obs=psi_obs, t_bins_obs=t_bins,
    )
    h_config = HamiltonianConfig(
        J=0.3, alpha=50.0, beta=10.0,
        gamma=1.0, l_crit=1000.0, delta=1.0, f_max=1.0,
        use_pair_barrier=False, use_energy_barrier=False,
    )
    H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config)

    opt_cfg = OptimizationConfig(
        max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
        use_adaptive_lr=True, use_continuation=True,
        continuation_steps=200, convergence_tol=1e-7, verbose=False,
    )

    all_res = []
    for seed in range(N_STARTS):
        rng = np.random.default_rng(seed)
        w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
        s0  = CoronalState(emissivity=w0,
                           orientations=vertical_orientations(GRID.n_cells))
        opt = QCORONAOptimizer(H_obj, opt_cfg)
        res = opt.optimize(initial_state=s0)
        all_res.append(res)

    best_idx = int(np.argmin([r.final_H.H_total for r in all_res]))
    best     = all_res[best_idx]
    w_best   = best.final_state.emissivity
    d_best   = best.final_H.diagnostics

    # Compute emissivity-weighted centroid
    R_cells = GRID.R
    z_cells = GRID.z
    R_centroid = np.dot(w_best, R_cells)
    z_centroid = np.dot(w_best, z_cells)

    print(f"  lag_pred = {d_best['delta_t_pred']:.2f} r_g/c  "
          f"(obs: {lag_rg:.2f} +/- {lag_err:.2f})")
    print(f"  centroid: R={R_centroid:.1f} rg, z={z_centroid:.1f} rg")
    print(f"  H_final  = {best.final_H.H_total:.4f}")

    tf_pred = compute_transfer_function_full(
        GRID, DISK, w_best, SA,
        observer_inclination=epoch['incl'],
        n_bins=N_BINS_TF,
        t_min=float(t_bins[0]), t_max=float(t_bins[-1]),
        lag_matrix=LM,
    )

    return dict(
        epoch=epoch, GRID=GRID,
        w_best=w_best, d_best=d_best,
        lag_rg=lag_rg, lag_err=lag_err,
        lag_pred=d_best['delta_t_pred'],
        R_centroid=R_centroid, z_centroid=z_centroid,
        H_final=best.final_H.H_total,
        t_c=t_c, psi_obs=psi_obs, psi_pred=tf_pred.psi,
        all_results=all_res, best_idx=best_idx,
    )


# ── main ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("HAMCOR — Mrk 335 multi-epoch analysis")
print("=" * 60)

os.makedirs('results', exist_ok=True)
results = []

for epoch in EPOCHS:
    r = run_epoch(epoch)
    if r is not None:
        results.append(r)

if not results:
    print("\nNo epochs ready yet. Complete SAS reduction for epochs 2-5.")
    print("See SAS instructions below.")
    sys.exit(0)

# ── multi-epoch summary figure ────────────────────────────────────────────────
n_done = len(results)
fig = plt.figure(figsize=(7.2, 3.0 * n_done))
gs  = gridspec.GridSpec(n_done, 3, figure=fig,
                        left=0.09, right=0.97, top=0.95, bottom=0.08,
                        hspace=0.6, wspace=0.45)

def rz_map(w, grid):
    return w.reshape(grid.n_r, grid.n_z, grid.n_phi).sum(axis=2)

for row, r in enumerate(results):
    epoch = r['epoch']
    GRID  = r['GRID']
    R_c = np.unique(np.round(GRID.R, 8))[:GRID.n_r]
    z_c = np.unique(np.round(GRID.z, 8))[:GRID.n_z]
    color = {'High': '#d6604d', 'Low': '#2166ac',
             'Intermediate': '#4dac26'}.get(epoch['state'], 'k')

    # Convergence
    ax0 = fig.add_subplot(gs[row, 0])
    for i, res in enumerate(r['all_results']):
        H = np.array([h['H_total'] for h in res.history])
        ax0.semilogy(np.abs(H), color=color,
                     alpha=1.0 if i == r['best_idx'] else 0.2,
                     lw=1.2 if i == r['best_idx'] else 0.5)
    ax0.set_xlabel('Iteration'); ax0.set_ylabel(r'$|\mathcal{H}|$')
    ax0.set_title(f"{epoch['year']} ({epoch['state']})", color=color)
    ax0.tick_params(labelsize=7)

    # Emissivity map
    ax1 = fig.add_subplot(gs[row, 1])
    mmap = rz_map(r['w_best'], GRID)
    im   = ax1.pcolormesh(R_c, z_c, mmap.T,
                           cmap='hot_r', shading='auto', rasterized=True)
    ax1.set_xlabel(r'$R\;[r_g]$'); ax1.set_ylabel(r'$z\;[r_g]$')
    ax1.set_title('Recovered corona')
    fig.colorbar(im, ax=ax1, fraction=0.038, pad=0.02,
                 shrink=0.85).ax.tick_params(labelsize=6)
    ax1.tick_params(labelsize=7)
    ax1.text(0.05, 0.95,
             f'$R_c={r["R_centroid"]:.1f}\\,r_g$\n'
             f'$z_c={r["z_centroid"]:.1f}\\,r_g$',
             transform=ax1.transAxes, va='top', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

    # Transfer function
    ax2 = fig.add_subplot(gs[row, 2])
    ax2.fill_between(r['t_c'], r['psi_obs'], alpha=0.18, color='#d6604d')
    ax2.plot(r['t_c'], r['psi_obs'],  color='#d6604d', lw=1.3,
             label=r'$\psi_{\rm obs}$')
    ax2.plot(r['t_c'], r['psi_pred'], color='#2166ac', lw=1.1, ls='--',
             label=r'$\psi_{\rm pred}$')
    ax2.set_xlabel(r'$\Delta t\;[r_g/c]$')
    ax2.set_ylabel(r'$\psi(t)$')
    ax2.set_title('Transfer function')
    ax2.legend(fontsize=7); ax2.tick_params(labelsize=7)
    ax2.text(0.97, 0.93,
             f'{r["lag_pred"]:.2f} vs {r["lag_rg"]:.2f} $r_g/c$',
             transform=ax2.transAxes, ha='right', va='top', fontsize=7)

fig.suptitle('Mrk~335 — HAMCOR multi-epoch coronal evolution',
             fontsize=10, fontweight='bold')
fig.savefig('results/fig_mrk335_multiepoch.pdf',
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\nSaved: results/fig_mrk335_multiepoch.pdf")

# ── coronal evolution summary plot ────────────────────────────────────────────
if len(results) > 1:
    fig2, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    years   = [r['epoch']['year']   for r in results]
    L_X     = [r['epoch']['L_X']    for r in results]
    R_c     = [r['R_centroid']      for r in results]
    z_c     = [r['z_centroid']      for r in results]
    states  = [r['epoch']['state']  for r in results]
    color_map = {'High': '#d6604d', 'Low': '#2166ac',
             'Intermediate': '#4dac26', 'Recovering': '#9c4dac'}
    colors = [color_map.get(s, '#888888') for s in states]
    

    axes[0].scatter(np.log10(L_X), R_c, c=colors, s=60, zorder=3)
    axes[0].set_xlabel(r'$\log_{10}(L_X\;[\mathrm{erg/s}])$')
    axes[0].set_ylabel(r'$R_{\rm centroid}\;[r_g]$')
    axes[0].set_title('Radial extent vs luminosity')

    axes[1].scatter(np.log10(L_X), z_c, c=colors, s=60, zorder=3)
    axes[1].set_xlabel(r'$\log_{10}(L_X\;[\mathrm{erg/s}])$')
    axes[1].set_ylabel(r'$z_{\rm centroid}\;[r_g]$')
    axes[1].set_title('Height vs luminosity')

    axes[2].scatter(R_c, z_c, c=colors, s=60, zorder=3)
    for r, col in zip(results, colors):
        axes[2].annotate(str(r['epoch']['year']),
                         (r['R_centroid'], r['z_centroid']),
                         fontsize=7, ha='left')
    axes[2].set_xlabel(r'$R_{\rm centroid}\;[r_g]$')
    axes[2].set_ylabel(r'$z_{\rm centroid}\;[r_g]$')
    axes[2].set_title('Coronal geometry track')

    for ax in axes:
        ax.tick_params(labelsize=8)
    fig2.tight_layout()
    fig2.savefig('results/fig_mrk335_evolution.pdf',
                 bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print("Saved: results/fig_mrk335_evolution.pdf")

print("\n" + "=" * 60)
print("Multi-epoch summary:")
print(f"{'Year':<6} {'State':<14} {'L_X':>10} {'R_c':>8} {'z_c':>8} {'lag_pred':>10}")
print("-" * 60)
for r in results:
    ep = r['epoch']
    print(f"{ep['year']:<6} {ep['state']:<14} {ep['L_X']:>10.1e} "
          f"{r['R_centroid']:>8.2f} {r['z_centroid']:>8.2f} "
          f"{r['lag_pred']:>10.2f}")
print("=" * 60)

print("""
╔══════════════════════════════════════════════════════════════╗
║  SAS REDUCTION INSTRUCTIONS FOR EPOCHS 2-5                  ║
╠══════════════════════════════════════════════════════════════╣
║  For each ObsID (0655200201, 0741280201, 0741280401,         ║
║  0741280501):                                                ║
║                                                              ║
║  1. Download from XSA: https://nxsa.esac.esa.int            ║
║     (PPS files or ODF)                                       ║
║                                                              ║
║  2. Set up SAS:                                              ║
║     export SAS_ODF=/path/to/odf                             ║
║     export SAS_CCF=/path/to/ccf                             ║
║     cifbuild && odfingest findinstrumentmodes=no             ║
║                                                              ║
║  3. Run standard pipeline:                                   ║
║     epproc                                                   ║
║                                                              ║
║  4. Filter flares:                                           ║
║     evselect table=*PN*ImagingEvts.ds:EVENTS                 ║
║       expression='PI>10000&&PI<12000&&PATTERN==0'           ║
║       timebinsize=100 rateset=PN_hilo_lc.fits               ║
║     tabgtigen table=PN_hilo_lc.fits                          ║
║       gtiset=gti.fits expression='RATE<=0.5'               ║
║                                                              ║
║  5. Extract light curves (copy run_extract_lc.py pattern)   ║
║     Save to data/sources/mrk335_YYYY/                       ║
║                                                              ║
║  6. Run python run_lag_mrk335.py (adapted for each epoch)   ║
║     to measure lag, then fill EPOCHS above.                  ║
╚══════════════════════════════════════════════════════════════╝
""")
