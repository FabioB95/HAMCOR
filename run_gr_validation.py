"""
run_gr_validation.py
────────────────────
Runs HAMCOR on the Mrk 335 lamppost synthetic geometry with both
flat-spacetime and Schwarzschild lag matrices, quantifies the impact
of GR corrections, and produces Fig. A1 for the paper appendix.

Run AFTER run_recovery.py.

Output:
    results/gr_correction_stats.txt
    results/fig_gr_comparison.pdf
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
from qcorona.physics.lags import compute_lag_matrix as flat_lag_matrix
from qcorona.data.synthetic import create_synthetic_geometry, emissivity_correlation, emissivity_overlap
from schwarzschild_lags import compute_lag_matrix_schwarzschild, compare_flat_vs_schwarzschild

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── grid (same as real data fits) ─────────────────────────────────────────────
GRID = build_grid(n_r=10, n_z=10, n_phi=8,
                  r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
INCL = 30.0

print("Computing lag matrices...")
LM_flat = flat_lag_matrix(GRID, DISK, observer_inclination=INCL)
print("  Flat-spacetime done.")
LM_schw = compute_lag_matrix_schwarzschild(GRID, DISK,
                                            observer_inclination=INCL,
                                            verbose=True)
print("  Schwarzschild done.")

# ── statistics ────────────────────────────────────────────────────────────────
corr_abs  = LM_schw - LM_flat
corr_frac = corr_abs / (LM_flat + 1e-10)

print(f"\n  Flat lag range:  [{LM_flat.min():.3f}, {LM_flat.max():.3f}] r_g/c")
print(f"  Schw lag range:  [{LM_schw.min():.3f}, {LM_schw.max():.3f}] r_g/c")
print(f"  Mean correction: {corr_abs.mean():.3f} r_g/c ({corr_frac.mean()*100:.1f}%)")
print(f"  Cells with >20% correction: {(corr_frac > 0.2).mean()*100:.1f}%")
print(f"  Cells with >40% correction: {(corr_frac > 0.4).mean()*100:.1f}%")

# ── synthetic recovery: flat vs Schwarzschild ─────────────────────────────────
syn = create_synthetic_geometry(GRID, DISK, 'lamppost', height=5.0, width=2.0)

opt_cfg = OptimizationConfig(
    max_iter=1000, lr_emissivity=0.01, lr_orientation=0.01,
    use_adaptive_lr=True, use_continuation=True,
    continuation_steps=200, convergence_tol=1e-7, verbose=False,
)

results = {}
for label, LM in [('flat', LM_flat), ('schwarzschild', LM_schw)]:
    print(f"\nRunning HAMCOR ({label}) on lamppost...")
    R_obs = syn.I_true / (1 - syn.I_true + 1e-10)
    constraints = ObservationalConstraints(
        delta_t_obs=syn.delta_t_true, R_obs=R_obs,
        L_X=1e43, M_bh=1e8, eddington_ratio=0.1,
        observer_inclination=INCL, source_name=f'syn_lamppost_{label}',
    )
    h_config = HamiltonianConfig(
        J=0.05, alpha=10.0, beta=10.0,
        gamma=1.0, l_crit=1000.0, delta=1.0, f_max=1.0,
        use_pair_barrier=False, use_energy_barrier=False,
    )
    H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config)
    # Inject custom lag matrix by patching the internal attribute
    # lag_matrix is a read-only property — set the private backing attribute
    _patched = False
    for _attr in ['_lag_matrix', '_LM', '_lag_mat', '_lag']:
        if hasattr(H_obj, _attr):
            try:
                object.__setattr__(H_obj, _attr, LM)
                print(f"  Patched H_obj.{_attr}")
                _patched = True
                break
            except (AttributeError, TypeError):
                pass
    if not _patched:
        # Try patching via __dict__ directly (bypasses property descriptor)
        for _attr, _val in H_obj.__dict__.items():
            if isinstance(_val, np.ndarray) and _val.shape == LM.shape:
                H_obj.__dict__[_attr] = LM
                print(f"  Patched H_obj.__dict__['{_attr}']")
                _patched = True
                break
    if not _patched:
        print("  All __dict__ items:")
        for _a, _v in H_obj.__dict__.items():
            print(f"    {_a}: {type(_v)} {getattr(_v,'shape','')}")

    best_w, best_H, best_hist = None, np.inf, None
    for seed in range(5):
        rng = np.random.default_rng(seed)
        w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
        s0  = CoronalState(emissivity=w0,
                           orientations=vertical_orientations(GRID.n_cells))
        opt = QCORONAOptimizer(H_obj, opt_cfg)
        res = opt.optimize(initial_state=s0)
        if res.final_H.H_total < best_H:
            best_H   = res.final_H.H_total
            best_w   = res.final_state.emissivity
            best_hist = np.array([h['H_total'] for h in res.history])
            best_diag = res.final_H.diagnostics

    corr  = emissivity_correlation(syn.emissivity, best_w)
    ovlap = emissivity_overlap(syn.emissivity, best_w)
    print(f"  rho={corr:+.3f}  overlap={ovlap:.3f}  "
          f"lag_pred={best_diag['delta_t_pred']:.3f} r_g/c  "
          f"(true={syn.delta_t_true:.3f})")

    results[label] = dict(
        w=best_w, H=best_H, hist=best_hist, diag=best_diag,
        corr=corr, overlap=ovlap,
    )

# ── figure ────────────────────────────────────────────────────────────────────
def rz_map(w):
    return w.reshape(GRID.n_r, GRID.n_z, GRID.n_phi).sum(axis=2)

R_c = np.unique(np.round(GRID.R, 8))[:GRID.n_r]
z_c = np.unique(np.round(GRID.z, 8))[:GRID.n_z]

fig = plt.figure(figsize=(7.2, 4.5))
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        left=0.08, right=0.97, top=0.91, bottom=0.11,
                        hspace=0.5, wspace=0.48,
                        height_ratios=[1.1, 1])

# Row 0: lag matrix comparison (cell-mean lags)
ax_hist = fig.add_subplot(gs[0, 0])
mean_flat = LM_flat.mean(axis=1)
mean_schw = LM_schw.mean(axis=1)
ax_hist.hist(mean_flat, bins=25, alpha=0.7, color='#2166ac',
             label='Flat', density=True)
ax_hist.hist(mean_schw, bins=25, alpha=0.7, color='#d6604d',
             label='Schwarzschild', density=True)
ax_hist.set_xlabel(r'Cell mean lag $[r_g/c]$', labelpad=2)
ax_hist.set_ylabel('Density', labelpad=2)
ax_hist.set_title('Lag distribution', pad=4)
ax_hist.legend(fontsize=7.5)
ax_hist.tick_params(labelsize=7)

ax_frac = fig.add_subplot(gs[0, 1])
im = ax_frac.scatter(
    mean_flat, corr_frac.mean(axis=1) * 100,
    c=np.sqrt(GRID.R**2 + GRID.z**2),
    cmap='viridis_r', s=8, alpha=0.7, rasterized=True,
)
cb = fig.colorbar(im, ax=ax_frac, fraction=0.038, pad=0.02)
cb.set_label(r'$r = \sqrt{R^2+z^2}\;[r_g]$', fontsize=7)
cb.ax.tick_params(labelsize=6)
ax_frac.axhline(20, color='k', ls='--', lw=0.8, alpha=0.5)
ax_frac.axhline(40, color='k', ls=':', lw=0.8, alpha=0.5)
ax_frac.set_xlabel(r'Flat mean lag $[r_g/c]$', labelpad=2)
ax_frac.set_ylabel('Shapiro correction [%]', labelpad=2)
ax_frac.set_title('Fractional GR correction', pad=4)
ax_frac.tick_params(labelsize=7)

ax_rz = fig.add_subplot(gs[0, 2])
corr_rz = corr_frac.mean(axis=1).reshape(GRID.n_r, GRID.n_z, GRID.n_phi).mean(axis=2)
im2 = ax_rz.pcolormesh(R_c, z_c, corr_rz.T * 100,
                        cmap='Reds', shading='auto', vmin=0, vmax=60,
                        rasterized=True)
cb2 = fig.colorbar(im2, ax=ax_rz, fraction=0.038, pad=0.02)
cb2.set_label('Correction [%]', fontsize=7)
cb2.ax.tick_params(labelsize=6)
ax_rz.set_xlabel(r'$R\;[r_g]$', labelpad=2)
ax_rz.set_ylabel(r'$z\;[r_g]$', labelpad=2)
ax_rz.set_title(r'Shapiro delay map', pad=4)
ax_rz.tick_params(labelsize=7)

# Row 1: recovery comparison
for col_i, (label, color) in enumerate([('flat', '#2166ac'),
                                         ('schwarzschild', '#d6604d')]):
    ax = fig.add_subplot(gs[1, col_i])
    res = results[label]
    mt  = rz_map(syn.emissivity)
    mr  = rz_map(res['w'])
    ax.pcolormesh(R_c, z_c, mr.T, cmap='Blues',
                  vmin=0, vmax=max(mt.max(), mr.max()),
                  shading='auto', rasterized=True)
    ax.contour(R_c, z_c, mt.T, levels=4, colors='k', linewidths=0.6)
    lbl = label.replace('schwarzschild', 'Schwarzschild').replace('flat', 'Flat')
    ax.set_title(f'{lbl}  $\\rho={res["corr"]:+.2f}$', pad=4)
    ax.set_xlabel(r'$R\;[r_g]$', labelpad=2)
    ax.set_ylabel(r'$z\;[r_g]$', labelpad=2)
    ax.tick_params(labelsize=7)

# Convergence comparison
ax_conv = fig.add_subplot(gs[1, 2])
for label, color in [('flat', '#2166ac'), ('schwarzschild', '#d6604d')]:
    H = results[label]['hist']
    lbl = label.replace('schwarzschild', 'Schwarzschild').replace('flat', 'Flat')
    ax_conv.semilogy(np.abs(H), color=color, lw=1.3, label=lbl)
ax_conv.set_xlabel('Iteration', labelpad=2)
ax_conv.set_ylabel(r'$|\mathcal{H}|$', labelpad=2)
ax_conv.set_title('Convergence', pad=4)
ax_conv.legend(fontsize=7.5)
ax_conv.tick_params(labelsize=7)

fig.suptitle('HAMCOR: flat spacetime vs Schwarzschild corrections (Mrk 335 lamppost)',
             fontsize=9.5)

os.makedirs('results', exist_ok=True)
fig.savefig('results/fig_gr_comparison.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\nSaved: results/fig_gr_comparison.pdf")

# ── text output ───────────────────────────────────────────────────────────────
with open('results/gr_correction_stats.txt', 'w') as f:
    f.write("HAMCOR Schwarzschild correction statistics\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Grid: r_in=3, r_out=15, z_max=5 rg\n")
    f.write(f"Inclination: {INCL} deg\n\n")
    f.write(f"Flat lag range:  [{LM_flat.min():.3f}, {LM_flat.max():.3f}] r_g/c\n")
    f.write(f"Schw lag range:  [{LM_schw.min():.3f}, {LM_schw.max():.3f}] r_g/c\n")
    f.write(f"Mean correction: {corr_abs.mean():.3f} r_g/c  "
            f"({corr_frac.mean()*100:.1f}%)\n")
    f.write(f"Cells >20% correction: {(corr_frac>0.2).mean()*100:.1f}%\n")
    f.write(f"Cells >40% correction: {(corr_frac>0.4).mean()*100:.1f}%\n\n")
    f.write("Lamppost recovery (best of 5 seeds):\n")
    for label in ['flat', 'schwarzschild']:
        r = results[label]
        f.write(f"  {label:14s}: rho={r['corr']:+.3f}  "
                f"overlap={r['overlap']:.3f}  "
                f"lag_pred={r['diag']['delta_t_pred']:.3f} r_g/c\n")

print("Saved: results/gr_correction_stats.txt")
print("\nDone. Add fig_gr_comparison.pdf to Overleaf figures/ folder.")
print("Replace compute_lag_matrix calls in run_mrk335.py etc. with:")
print("  from schwarzschild_lags import compute_lag_matrix_schwarzschild")
