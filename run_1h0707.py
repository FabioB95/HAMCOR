"""
run_1h0707.py  —  Applica QCORONA a 1H 0707-495 con dati XMM reali.

Lag misurato: -28.3 +/- 5.0 s = 2.83 +/- 0.50 r_g/c
Osservazione: XMM-Newton ObsID 0511580101 (2008, gennaio)
Bande: soft 0.3-1.0 keV vs hard 1.5-4.0 keV
Frequenze: 0.6-3.0 mHz (banda reverberation, Kara+2013)
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

# ── parametri 1H 0707-495 ────────────────────────────────────────────────────
# Massa BH: ~2e6 M_sun (Zoghbi+2010, dalla relazione M-sigma)
# r_g/c = GM/c^3 = 1.48e3 * (M/1e6 Msun) s ≈ 10 s per 2e6 Msun
# Lag misurato: -28.3 +/- 5.0 s = 2.83 +/- 0.50 r_g/c
# Inclinazione: ~50 deg (Dauser+2012)
# Reflection fraction: alta, R~3 (forte soft excess)

M_BH       = 2.0e6    # M_sun
R_G_C      = 10.0     # s per r_g/c
LAG_OBS_S  = 28.3     # s (valore assoluto)
LAG_ERR_S  = 5.0      # s
LAG_OBS_RG = LAG_OBS_S / R_G_C    # 2.83 r_g/c
LAG_ERR_RG = LAG_ERR_S / R_G_C    # 0.50 r_g/c
R_OBS      = 3.0      # alta reflection fraction (Fabian+2009)
L_X        = 5.0e42   # erg/s (NLS1 a bassa massa)
INCL       = 50.0     # gradi

N_STARTS   = 5
MAX_ITER   = 2000
N_BINS_TF  = 80

print("=" * 60)
print("QCORONA — 1H 0707-495 (XMM ObsID 0511580101, 2008)")
print("=" * 60)
print(f"  M_bh    = {M_BH:.1e} M_sun")
print(f"  r_g/c   = {R_G_C:.0f} s")
print(f"  Lag obs = {LAG_OBS_S:.1f} +/- {LAG_ERR_S:.1f} s "
      f"= {LAG_OBS_RG:.2f} +/- {LAG_ERR_RG:.2f} r_g/c")
print(f"  R_obs   = {R_OBS}")
print(f"  L_X     = {L_X:.1e} erg/s")
print(f"  Incl    = {INCL} deg")

# ── grids ─────────────────────────────────────────────────────────────────────
print("\nBuilding grids...")
GRID = build_grid(n_r=10, n_z=10, n_phi=8,
                  r_in=3.0, r_out=15.0,
                  z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
print(f"  Corona: {GRID.n_cells} cells  |  Disk: {DISK.n_elements} elements")

SA = compute_solid_angles(GRID, DISK)
LM = compute_lag_matrix(GRID, DISK, observer_inclination=INCL)
print(f"  Lag matrix range: {LM.min():.2f} -- {LM.max():.2f} r_g/c")
print(f"  Lag obs {LAG_OBS_RG:.2f} in range: {LM.min() < LAG_OBS_RG < LM.max()}")

# ── psi_obs gaussiana centrata sul lag misurato ───────────────────────────────
T_MIN  = max(0.0, float(LM.min()))
T_MAX  = 15.0
t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
t_c    = 0.5 * (t_bins[:-1] + t_bins[1:])

psi_obs = np.exp(-0.5 * ((t_c - LAG_OBS_RG) / LAG_ERR_RG) ** 2)
psi_obs = psi_obs / psi_obs.sum()

print(f"\n  psi_obs: Gaussian at {LAG_OBS_RG:.2f} r_g/c "
      f"(sigma={LAG_ERR_RG:.2f} r_g/c)")
print(f"  t range: [{T_MIN:.1f}, {T_MAX:.1f}] r_g/c, {N_BINS_TF} bins")

# ── constraints ───────────────────────────────────────────────────────────────
constraints = ObservationalConstraints(
    delta_t_obs          = LAG_OBS_RG,
    R_obs                = R_OBS,
    L_X                  = L_X,
    M_bh                 = M_BH,
    eddington_ratio      = 0.5,    # NLS1 tipicamente alta Eddington
    observer_inclination = INCL,
    source_name          = "1H0707_XMM2008",
    psi_obs              = psi_obs,
    t_bins_obs           = t_bins,
)

h_config = HamiltonianConfig(
    J=0.3, alpha=50.0, beta=10.0,
    gamma=1.0, l_crit=1000.0,
    delta=1.0, f_max=1.0,
    use_pair_barrier=False,
    use_energy_barrier=False,
)

H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config)

# ── multi-start optimization ──────────────────────────────────────────────────
print(f"\nOptimizing ({N_STARTS} cold starts, {MAX_ITER} iter each)...")
print(f"{'seed':>5} {'H_final':>10} {'lag_pred':>10} {'I_pred':>8} {'time':>7}")
print("-" * 45)

opt_cfg = OptimizationConfig(
    max_iter           = MAX_ITER,
    lr_emissivity      = 0.01,
    lr_orientation     = 0.01,
    use_adaptive_lr    = True,
    use_continuation   = True,
    continuation_steps = 200,
    convergence_tol    = 1e-7,
    verbose            = False,
)

all_results = []
for seed in range(N_STARTS):
    rng = np.random.default_rng(seed)
    w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
    s0  = CoronalState(
        emissivity   = w0,
        orientations = vertical_orientations(GRID.n_cells),
    )
    opt     = QCORONAOptimizer(H_obj, opt_cfg)
    t0      = time.time()
    res     = opt.optimize(initial_state=s0)
    elapsed = time.time() - t0
    d       = res.final_H.diagnostics
    print(f"{seed:>5} {res.final_H.H_total:>10.4f} "
          f"{d['delta_t_pred']:>10.2f} {d['I_pred']:>8.3f} "
          f"{elapsed:>6.0f}s")
    all_results.append(res)

best_idx = int(np.argmin([r.final_H.H_total for r in all_results]))
best     = all_results[best_idx]
w_best   = best.final_state.emissivity
d_best   = best.final_H.diagnostics

print(f"\nBest result (seed {best_idx}):")
print(f"  H_total    = {best.final_H.H_total:.4f}")
print(f"  lag_pred   = {d_best['delta_t_pred']:.2f} r_g/c  "
      f"(obs: {LAG_OBS_RG:.2f} +/- {LAG_ERR_RG:.2f})")
print(f"  lag_pred_s = {d_best['delta_t_pred']*R_G_C:.1f} s  "
      f"(obs: {LAG_OBS_S:.1f} +/- {LAG_ERR_S:.1f} s)")
print(f"  illum_pred = {d_best['I_pred']:.3f}  "
      f"(obs: {constraints.I_obs:.3f})")

# ── transfer function predetta ────────────────────────────────────────────────
tf_pred = compute_transfer_function_full(
    GRID, DISK, w_best, SA,
    observer_inclination = INCL,
    n_bins               = N_BINS_TF,
    t_min                = float(t_bins[0]),
    t_max                = float(t_bins[-1]),
    lag_matrix           = LM,
)

# ── figura ────────────────────────────────────────────────────────────────────
def rz_map(w):
    return w.reshape(GRID.n_r, GRID.n_z, GRID.n_phi).sum(axis=2)

R_c = np.unique(np.round(GRID.R, 8))[:GRID.n_r]
z_c = np.unique(np.round(GRID.z, 8))[:GRID.n_z]

fig = plt.figure(figsize=(7.2, 4.0))
gs  = gridspec.GridSpec(
    1, 3, figure=fig,
    left=0.09, right=0.97,
    top=0.88,  bottom=0.14,
    wspace=0.48,
)

# col 0: convergenza
ax0 = fig.add_subplot(gs[0, 0])
for i, res in enumerate(all_results):
    H_arr = np.array([h['H_total'] for h in res.history])
    ax0.semilogy(np.arange(len(H_arr)), np.abs(H_arr),
                 color='#2166ac',
                 alpha=1.0 if i == best_idx else 0.25,
                 lw=1.4 if i == best_idx else 0.7)
ax0.set_xlabel('Iteration')
ax0.set_ylabel(r'$|\mathcal{H}|$')
ax0.set_title('Convergence')
ax0.tick_params(labelsize=7)

# col 1: mappa R-z
ax1 = fig.add_subplot(gs[0, 1])
mmap = rz_map(w_best)
im   = ax1.pcolormesh(R_c, z_c, mmap.T,
                       cmap='hot_r', shading='auto', rasterized=True)
ax1.set_xlabel(r'$R\;[r_g]$')
ax1.set_ylabel(r'$z\;[r_g]$')
ax1.set_title('Recovered corona')
cb = fig.colorbar(im, ax=ax1, fraction=0.038, pad=0.02, shrink=0.85)
cb.ax.tick_params(labelsize=6)
cb.set_label(r'$w_i$', fontsize=7)
cb.ax.yaxis.set_major_locator(MaxNLocator(3))
ax1.tick_params(labelsize=7)
ax1.text(0.05, 0.95,
         f'lag = {d_best["delta_t_pred"]:.2f} $r_g/c$\n'
         f'obs: {LAG_OBS_RG:.2f} $r_g/c$',
         transform=ax1.transAxes, va='top', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

# col 2: psi_obs vs psi_pred
ax2 = fig.add_subplot(gs[0, 2])
ax2.fill_between(t_c, psi_obs, alpha=0.18, color='#d6604d')
ax2.plot(t_c, psi_obs,     color='#d6604d', lw=1.3,
         label=r'$\psi_\mathrm{obs}$')
ax2.plot(t_c, tf_pred.psi, color='#2166ac', lw=1.1, ls='--',
         label=r'$\psi_\mathrm{pred}$')
ax2.set_xlabel(r'$\Delta t\;[r_g/c]$')
ax2.set_ylabel(r'$\psi(t)$')
ax2.set_title('Transfer function')
ax2.legend(fontsize=7, framealpha=0.85)
ax2.set_ylim(bottom=0)
sig = np.where(psi_obs > 0.001 * psi_obs.max())[0]
if len(sig) > 1:
    ax2.set_xlim(max(0.0, t_c[sig[0]] - 0.5), t_c[sig[-1]] + 2.0)
ax2.tick_params(labelsize=7)
chi2 = float(np.sum((tf_pred.psi - psi_obs) ** 2))
ax2.text(0.97, 0.93, f'$\\chi^2={chi2:.2e}$',
         transform=ax2.transAxes, ha='right', va='top',
         fontsize=6.5, color='gray')

fig.suptitle(
    '1H 0707$-$495 — HAMCOR fit (XMM ObsID 0511580101, 2008)',
    fontsize=9.5, y=0.98,
)

os.makedirs('results', exist_ok=True)
fig.savefig('results/fig_1h0707.pdf', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\nSaved: results/fig_1h0707.pdf")

# ── pgfplots ──────────────────────────────────────────────────────────────────
def coords(xs, ys):
    return ' '.join(f'({float(x):.4f},{float(y):.6g})'
                    for x, y in zip(xs, ys))

tex = [
    r'% HAMCOR 1H 0707-495 transfer function — pgfplots',
    r'\begin{tikzpicture}',
    r'\begin{axis}[width=6cm,height=4.5cm,',
    r'  xlabel={$\Delta t\;[r_g/c]$},ylabel={$\psi(t)$},ymin=0,',
    r'  title={1H~0707$-$495 transfer function},',
    r'  tick label style={font=\tiny},label style={font=\scriptsize}]',
    r'\addplot[thick,red!60!black,fill=red!15,fill opacity=0.3]',
    r'  coordinates {' + coords(t_c, psi_obs) + r'} \closedcycle;',
    r'\addplot[dashed,thick,blue!70!black] coordinates {',
    '  ' + coords(t_c, tf_pred.psi), r'};',
    r'\legend{$\psi_\mathrm{obs}$,$\psi_\mathrm{pred}$}',
    r'\end{axis}',
    r'\end{tikzpicture}',
]

with open('results/fig_1h0707.tex', 'w', encoding='utf-8') as f:
    f.write('\n'.join(tex))
print("Saved: results/fig_1h0707.tex")

print("\n" + "=" * 60)
print("SUMMARY — 1H 0707-495 HAMCOR fit")
print("=" * 60)
print(f"  Lag obs:    {LAG_OBS_S:.1f} +/- {LAG_ERR_S:.1f} s "
      f"= {LAG_OBS_RG:.2f} +/- {LAG_ERR_RG:.2f} r_g/c")
print(f"  Lag pred:   {d_best['delta_t_pred']*R_G_C:.1f} s "
      f"= {d_best['delta_t_pred']:.2f} r_g/c")
print(f"  Illum obs:  {constraints.I_obs:.3f}")
print(f"  Illum pred: {d_best['I_pred']:.3f}")
print(f"  H_final:    {best.final_H.H_total:.4f}")
print(f"  chi2(psi):  {chi2:.4e}")
print("=" * 60)
